from ..models import DiffusionModel
from .adjoint_matching import AMTrainerFlow
from ..sampling import Sampler
import torch

from omegaconf import OmegaConf
from typing import Callable, Optional

class AndOperatorTrainerTriple(AMTrainerFlow):
    def __init__(self,
                 config: OmegaConf,
                 model: DiffusionModel,
                 base_model: DiffusionModel,
                 pre_trained_model_1: DiffusionModel,
                 pre_trained_model_2: DiffusionModel,
                 grad_reward: Optional[Callable] = None,
                 device: Optional[torch.device] = None,
                 sampler: Optional[Sampler] = None):
    
        rew_type = config.get('rew_type', 'score_matching')
        self.lmbda = lmbda = config.get('lmbda', 1.)
        alpha_div = config.get('alpha_div', [1., 1., 1.])
        
        if rew_type == 'score-matching':
            print("Using first variation of double KL as reward, lambda:", lmbda)
            # grad_reward =  lmbda * (-2*score_base + score_pre1 + score_pre2)
            eps = 0.1
            if grad_reward is not None:
                grad_reward_fn = lambda x: lmbda * (grad_reward(x) - (alpha_div[0] + alpha_div[1]) * self.base_model.score_func(x, torch.tensor(eps, device=x.device).float().detach()) + alpha_div[0] * pre_trained_model_1.score_func(x, torch.tensor(eps, device=x.device).float().detach()) + alpha_div[1] *pre_trained_model_2.score_func(x, torch.tensor(eps, device=x.device).float().detach()))
            else:
                grad_reward_fn = lambda x: lmbda * (- (alpha_div[0] + alpha_div[1]) * self.base_model.score_func(x, torch.tensor(eps, device=x.device).float().detach()) + alpha_div[0] * pre_trained_model_1.score_func(x, torch.tensor(eps, device=x.device).float().detach()) + alpha_div[1] *pre_trained_model_2.score_func(x, torch.tensor(eps, device=x.device).float().detach()))

            # compute gradient of f_k along trajectory
            grad_f_k_trajectory = lambda x, t: lmbda * (- (alpha_div[0] + alpha_div[1]) * self.base_model.score_func(x, torch.tensor(t, device=x.device).float().detach()) + alpha_div[0] * pre_trained_model_1.score_func(x, torch.tensor(t, device=x.device).float().detach()) + alpha_div[1] *pre_trained_model_2.score_func(x, torch.tensor(t, device=x.device).float().detach()))
            self.lmbda = lmbda
        else:
            raise NotImplementedError

        super().__init__(config.adjoint_matching, model, base_model, grad_reward_fn, grad_traj, device=device, sampler=sampler)
    

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())
