from omegaconf import OmegaConf
from genexp.models import FlowModel
from genexp.sampling import Sampler
from genexp.trainers.adjoint_matching import AMTrainerFlow
from typing import Callable, Optional

import torch
import copy


class FDCTrainerFlow(AMTrainerFlow):
    def __init__(self,
                 config: OmegaConf,
                 model: FlowModel,
                 base_model: FlowModel,
                 device: Optional[torch.device] = None,
                 verbose: bool = False,
                 sampler: Optional[Sampler] = None
                ):
            
        self.gamma = config.get('gamma', 1.)
        self.beta = config.get('beta', 0.)
        self.epsilon = torch.tensor(config.epsilon).to(device)
        self.base_base_model = copy.deepcopy(base_model)
        self.combined_score = lambda s, t: base_model.score_func(s, t) - self.beta * self.base_base_model.score_func(s, t)
        grad_reward_fn = lambda x: -self.gamma * self.combined_score(x, 1. - self.epsilon)

        super().__init__(config.adjoint_matching, model, base_model, grad_reward_fn, None, device, verbose, sampler)


    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())
