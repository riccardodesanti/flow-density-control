"""
The point of this trainer is that it computes likelihoods of several models and uses them to compute a reward function.
An example application is the Or-operator or Renyi divergence.
"""

from ..models import DiffusionModel
from .adjoint_matching import AMTrainerFlow
import torch
from ..likelihood import MultiItoDensODE
from tqdm import tqdm
from .adjoint_matching import LeanAdjointSolver, AMDataset, ConcatDataset, sample_trajectories_ddpm
from ..solvers import TorchDiffEqSolver, PFODE, VPSDE, DDIMSolver
from ..trainers.adjoint_matching import sample_trajectories_ddim, sample_trajectories_ddpm
from ..likelihood import prior_likelihood




import logging
logger = logging.getLogger(__name__)


def sample_trajectories_ito(model: DiffusionModel, pre_models: torch.nn.ModuleList,x0, T, avoid_inf=0.0, sample_jumps=True):
    """Sample trajectories from the basqqqqed on probflow ODE, also estimate likelihoods of several models."""
    device = next(model.parameters()).device
    x0 = x0.to(device)
    ito_ode = MultiItoDensODE(model_sampling=model, models=pre_models, sde=model.sde, sign=1)
    ts = torch.linspace(1,0, T, device=device)
    plik = prior_likelihood(x0, 1.0)
    plik = torch.stack([plik] + [plik for _ in range(len(ito_ode.models))])
    x0 = (x0, plik)
    solver = TorchDiffEqSolver(ito_ode)
    res = solver.solve(x0, t=ts, method='euler', atol=1e-5)    
    traj = res[-1]['traj'][0].transpose(1,0) # (n, T, d)
    logp = res[0][1] # (m, n), at [0] we have the logp of the sampling model
    scores = ito_ode.scores
    return traj, logp, scores, ts


def renyi_first_variation_grad(logps, logqs, scorep, scoreq, alpha, estimate_Z=False):
    """
    Computes grad of first variation of the Renyi divergence $D_\alpha( p || q)$.
    

    $$
      \nabla_x \partial D_{\alpha}(p||q) = 
      \frac{\alpha}{\int_\gX} p(z)^{\alpha} q(z)^{1-\alpha} dz
      \Big ( ...  Big)
    $$
    """
    if not estimate_Z:
        Z = 1
    else:
        raise NotImplementedError("Estimate Z not implemented yet")
    
    ps = logps.exp()
    qs = logqs.exp()

    term1 = ((alpha-2)*ps + (1-alpha)*qs).exp() * scorep  
    term2 = ((alpha-1)*ps + (-alpha)*qs).exp() * scoreq

    return Z * (term1 - term2)


class LikelihoodEstTrainer(AdjointMatchingFinetuningTrainer):
    def __init__(self, model: DiffusionModel, 
                 lr, 
                 traj_samples_per_stage, 
                 data_shape, 
                 pre_models,
                 finetune_steps=100, 
                 batch_size=32, 
                 device='cuda',
                 base_model=None,
                 traj_len=100,
                 lmbda=1.0,
                 clip_grad_norm=None,
                 alpha=1.0,
                 estimate_Z=False):
        
        logger.info("Using first variation of double reverse KL as reward, lambda:{}".format(lmbda))
        grad_reward_fn = lambda x: None
        self.lmbda = lmbda
        self.alpha = alpha
        self.estimate_Z = estimate_Z

        super().__init__(model, grad_reward_fn, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, 
                         clip_grad_norm=clip_grad_norm, memsave=False)

        self.fine_model = self.fine_model.to(device)
        self.base_model = self.base_model.to(device)
        self.pre_models = pre_models

    def get_grad_reward_fn(self, logps, scores) -> "torch.Tensor[N, D]":
        """Union reward function."""
        logps = logps.to(self.device)
        scores = scores.to(self.device)


        p_fine = logps[0,:, None].exp()
        p_pre = logps[1,:,None].exp()
        scorep = scores[0]
        scoreq = scores[1]


        logger.debug(f"shapes: {p_fine.shape}, {p_pre.shape}, {scorep.shape}, {scoreq.shape}")

        assert len(logps) == len(scores), "logps and scores must have the same length, but got {} and {}".format(len(logps), len(scores))
        assert len(logps)-1 == len(self.pre_models), "logps and pre_models must have the same length, but got {} and {}".format(len(logps)-1, len(self.pre_models))
        assert len(logps) == 2, "logps must have length 2 for just reny divergence, but got {}".format(len(logps))


  

        rew = renyi_first_variation_grad(p_fine, p_pre, scorep, scoreq, alpha=self.alpha, estimate_Z=self.estimate_Z)


        return lambda _: rew.to('cuda')


    def sample_dataset_stage(self, stages=1, verbose=False):
            """Sample dataset for training based on adjoint ODE and sampled trajectories."""
            
            datasets = []
            # shift model to cuda
            self.fine_model.eval()
            self.base_model.eval()

            dtype = next(self.fine_model.parameters()).dtype
            device = next(self.fine_model.parameters()).device
            logger.info(f"Sampling dataset on {device} with dtype {dtype}")
            with torch.no_grad():
                for _ in tqdm(range(stages), disable=not verbose):
                    x0 = torch.randn(self.traj_samples_per_stage, *self.data_shape).to(self.device).to(dtype)
                    self.fine_model.to(self.device)
                    trajs, logp, scores, ts = sample_trajectories_ito(self.fine_model, self.pre_models, x0=x0, 
                                                              T=self.traj_len, sample_jumps=self.random_jumps) 
                    # create reward function
                    grad_reward_fn = self.get_grad_reward_fn(logp, scores)
                    solver = LeanAdjointSolver(self.base_model, grad_reward_fn)

                    if self.memsave:
                        logger.debug('shifting to cpu')
                        self.fine_model.to('cpu')
                    _, solver_info = solver.solve(trajs.to('cuda'), ts=ts.flip(0).to('cuda'))
                    if self.memsave:
                        logger.debug('shifting to cpu')
                        self.base_model.to('cpu')
                    dataset = AMDataset(solver_info=solver_info)
                    datasets.append(dataset)
            dataset = ConcatDataset(datasets)
            return dataset

    def update_reward(self):
        self.grad_reward_fn = lambda x: -self.base_model.score_func(x, torch.tensor(0.0, device=x.device).float().detach())*self.lmbda

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()
