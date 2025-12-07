from ..models import DiffusionModel
from .adjoint_matching import AMTrainerFlow
from ..sampling import Sampler, sample_trajectories_ddpm

from omegaconf import OmegaConf
from typing import Optional
import torch

class OedKlTrainer(AMTrainerFlow):
    def __init__(self,
                 config: OmegaConf,
                 model: DiffusionModel,
                 base_model: DiffusionModel,
                 pre_trained_model: DiffusionModel,
                 device: Optional[torch.device] = None,
                 sampler: Optional[Sampler] = None
                 ):

        rew_type = config.get('rew_type', 'score_matching')
        alpha_div = config.get('alpha_div', 1.)
        self.num_traj_MC = num_traj_MC = config.get('num_traj_MC', 15)
        self.lmbda = lmbda = config.get('lmbda', 1.)
        lambda_reg_ridge = config.get('lambda_reg_ridge', 0.1)

        
        if rew_type == 'score-matching':
            print("Using first variation of OED objective and KL divergence as reward, lambda:", lmbda)
            # linear kernel case \Phi(x) = x
            # oed_grad(x) - alpha_div * (score_base(x) - score_pre(x))
            grad_reward_fn = lambda x: lmbda * self.compute_oed_grad(x, base_model, num_traj_MC, lambda_reg_ridge) - alpha_div * (base_model.score_func(x, torch.tensor(0, device=x.device).float().detach()) - pre_trained_model.score_func(x, torch.tensor(0, device=x.device).float().detach()))
            self.lmbda = lmbda
        else:
            raise NotImplementedError
        
        super().__init__(config.adjoint_matching, model, base_model, grad_reward_fn, None, device=device, sampler=sampler)

    def compute_oed_grad(self, x, base_model, num_traj_MC, lambda_reg_ridge):
        # compute the gradient of the OED objective
        # sample batch of N data points x from current model to estimate expectation of outer product of features
        with torch.no_grad():
            x0 = torch.randn(num_traj_MC, 2, device='cpu')
            trajs1 = sample_trajectories_ddpm(base_model, x0, 100) 
            trajs1 = trajs1[0]
            x_sampled = trajs1[:, -1, :]
            outer_products = x_sampled.unsqueeze(2) * x_sampled.unsqueeze(1)
            MC_expectation = outer_products.mean(dim=0)

        matrix_inverse = torch.linalg.inv(MC_expectation + lambda_reg_ridge * torch.eye(2))
        matrix_inverse_sqrt = matrix_inverse @ matrix_inverse
        matrix_inverse_sqrt = matrix_inverse_sqrt.to(x.device)
        final_gradient = 2 * x @ matrix_inverse_sqrt
        return final_gradient

    def update_reward(self):
        self.grad_reward_fn = lambda x: -self.base_model.score_func(x, torch.tensor(0.0, device=x.device).float().detach())*self.lmbda

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()
