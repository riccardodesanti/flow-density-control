from ..models import DiffusionModel
from ..sampling import Sampler, sample_trajectories_ddpm
from .adjoint_matching import AMTrainerFlow

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from typing import Callable, Optional

class RiskAverseKlTrainer(AMTrainerFlow):
    def __init__(self,
                 config: OmegaConf,
                 model: DiffusionModel,
                 base_model: DiffusionModel,
                 pre_trained_model: DiffusionModel,
                 device: Optional[torch.device] = None,
                 sampler: Optional[Sampler] = None,
                 loss_function: Optional[Callable] = None):
        
        self.alpha_cvar = config.get('alpha_cvar', 0.95)
        self.num_traj_MC = config.get('num_traj_MC', 4)
        self.traj_len = config.get('traj_len', 40)
        alpha_div = config.get('alpha_div', 1.)
        rew_type = config.get('rew_type', 'score-matching')
        self.lmbda = lmbda = config.get('lmbda', 1.)

        if rew_type == 'score-matching':
            grad_reward_fn = lambda x: - lmbda * (self.compute_superquantile_grad(x, base_model, loss_function) - alpha_div * (base_model.score_func(x, torch.tensor(0, device=x.device).float().detach()) - pre_trained_model.score_func(x, torch.tensor(0, device=x.device).float().detach())))
        else:
            raise NotImplementedError
        super().__init__(config.adjoint_matching, model, base_model, grad_reward_fn, None, device=device, sampler=sampler)
    

    def compute_superquantile_grad(self, x, base_model, loss_function):
        alpha_cvar = self.alpha_cvar
        num_traj_MC = self.num_traj_MC
        gamma = 10.0

        # samples based estimation of beta_star
        with torch.no_grad():
            x0 = torch.randn(num_traj_MC, 2, device='cpu')
            trajs1 = sample_trajectories_ddpm(base_model, x0, self.traj_len) 
            trajs1 = trajs1[0]
            samples = trajs1[:, -1, :]
            sample_loss = loss_function(samples)
            beta_star = torch.quantile(sample_loss, self.alpha_cvar)

        # activate gradient tracking for x
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        torch.set_grad_enabled(True)

        # UN-SMOOTHED CVaR
        loss_x = loss_function(x)                               
        mask = (loss_x > beta_star).float()
        grad_L = torch.autograd.grad(loss_x.sum(), x)[0]
        factor = mask / ((1.0 - self.alpha_cvar) * mask.numel())       
        final_gradient = factor.unsqueeze(-1) * grad_L 

        return final_gradient


    def estimate_beta_star_torch(self, samples, loss_function, alpha_cvar, gamma, init_beta=0.0, num_iterations=100):
        beta = torch.tensor([init_beta], dtype=torch.float32, requires_grad=True, device=samples.device)

        def objective_function_torch():
            samples_loss_val = loss_function(samples)
            sp = F.softplus(samples_loss_val - beta, beta=gamma)
            val = beta + (1.0/(1.0 - alpha_cvar)) * torch.mean(sp)
            return val

        optimizer = torch.optim.LBFGS([beta], max_iter=num_iterations, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            loss = objective_function_torch()
            loss.backward()
            return loss

        optimizer.step(closure)

        # eval for debugging 
        est_CVaR = objective_function_torch()
        print("estimated CVaR during opt:", est_CVaR.item())

        return beta.detach()
    

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

