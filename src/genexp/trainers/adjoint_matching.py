import numpy as np
from omegaconf import OmegaConf

import torch
from torch.utils.data import Dataset, ConcatDataset

from genexp.sampling import EulerMaruyamaSampler, Sampler, Sample
from genexp.models import FlowModel
from typing import List, Callable, Optional


class LeanAdjointSolverFlow:
    """Solver as per adjoint matching paper."""

    def __init__(self, gen_model: FlowModel, grad_reward_fn: Callable, grad_f_k_fn: Optional[Callable] = None, device: Optional[torch.device] = None):
        self.model = gen_model
        self.interpolant_scheduler = gen_model.interpolant_scheduler
        self.grad_reward_fn = grad_reward_fn
        self.grad_f_k_fn = grad_f_k_fn
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def step(self, adj: torch.Tensor, x_t: Sample, t: torch.Tensor, alpha: torch.Tensor, alpha_dot: torch.Tensor, dt: torch.Tensor):
        adj_t = adj.detach() # detach to avoid gradients

        with torch.enable_grad():
            x_t.adjoint = x_t.adjoint.detach().requires_grad_(True)
            v_pred = self.model.velocity_field(x_t.full, t)

            eps_pred = 2 * v_pred - alpha_dot/(alpha + dt) * x_t.adjoint

            g_term = (adj_t * eps_pred).sum()
            v = torch.autograd.grad(g_term, x_t.adjoint, retain_graph=False)[0]

        assert v.shape == x_t.adjoint.shape

        adj_tmh = adj_t + dt * v
        if self.grad_f_k_fn is not None:
            grad_f_k = self.grad_f_k_fn(x_t.full, t + dt).to(adj_t.device)

            assert grad_f_k.shape == x_t.adjoint.shape
            adj_tmh = adj_t + dt * v - dt * grad_f_k
            
        return adj_tmh.detach(), v_pred.detach()

    def solve(self, trajectories: List[Sample], ts: torch.Tensor):
        """Solving loop."""
        T = ts.shape[0]
        assert T == len(trajectories)
        # ts: tensor of shape (num_ts,) (0, dt, 2dt, ..., 1-dt, 1)
        dt = ts[1] - ts[0]
        ts = ts.flip(0) # flip to go from 1 to 0

        alpha_s = self.interpolant_scheduler.alpha_t(ts)
        alpha_dot_s = self.interpolant_scheduler.alpha_t_prime(ts)

        x_1 = trajectories[-1]
        adj = -self.grad_reward_fn(x_1.full)

        trajs_adj = []
        traj_v_pred = []

        for i in range(1, T):
            t = ts[i]
            x_t = trajectories[T - i - 1]
            alpha = alpha_s[i]
            alpha_dot = alpha_dot_s[i]
            adj, v_pred = self.step(adj=adj, x_t=x_t, t=t, alpha=alpha, alpha_dot=alpha_dot, dt=dt)
            trajs_adj.append(adj.detach())
            traj_v_pred.append(v_pred.detach())
        
        res = {
                't': ts.flip(0)[:-1], # (T,)
                'traj_x': trajectories[:-1], # array-like of len T
                'traj_adj': torch.stack(trajs_adj).flip(0), # (T, data_shape)
                'traj_v_pred': torch.stack(traj_v_pred).flip(0), # (T, data_shape)
            }

        assert res['traj_adj'].shape == res['traj_v_pred'].shape
        assert res['traj_adj'].shape[0] == res['t'].shape[0]
        assert len(res['traj_x']) == res['t'].shape[0]
        assert res['t'].shape[0] == ts.shape[0] - 1
        return res


class AMDataset(Dataset):
    def __init__(self, solver_info):
        solver_info = self.detach_all(solver_info)
        self.t = solver_info['t'] # (T,)
        self.traj_x = solver_info['traj_x'] # trajectories (T, batch_shape)
        self.traj_adj = solver_info['traj_adj'] # (T, adjoint_shape)
        self.traj_v_base = solver_info['traj_v_pred'] # (T, adjoint_shape)

        self.T = self.t.size(0) # T = number of time steps
        self.bs = 1 # len(self.traj_g[0].batch_num_nodes())
        
    def __len__(self):
        return self.bs

    def __getitem__(self, idx):
        return {
            'ts': self.t,
            'traj_x': self.traj_x,
            'traj_adj': self.traj_adj,
            'traj_v_base': self.traj_v_base,
        }
    
    def detach_all(self, solver_info):
        for key, value in solver_info.items():
            if isinstance(value, torch.Tensor):
                solver_info[key] = value.detach()
            if isinstance(value, Sample):
                value.detach_all()
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, torch.Tensor):
                        value[i] = v.detach()
                    if isinstance(v, Sample):
                        v.detach_all()
            
        return solver_info


def create_timestep_subset(total_steps, final_percent=0.25, sample_percent=0.25):
    """
    Create a subset of time-steps for efficient computation. (See paper Appendix G2)
    """
    # Calculate the number of steps for each section
    final_steps_count = int(total_steps * final_percent)
    sample_steps_count = int(total_steps * sample_percent)
    
    # Always take the first final_percent steps (assuming highest index is 0)
    final_samples = np.arange(final_steps_count)
    
    # Sample additional steps without replacement from the remaining steps
    # Exclude the steps already in final_samples
    remaining_steps = np.setdiff1d(
        np.arange(total_steps), 
        final_samples
    )
    
    # Sample additional steps
    additional_samples = np.random.choice(
        remaining_steps, 
        size=sample_steps_count, 
        replace=False
    )
    
    # Combine and sort the samples
    combined_samples = np.sort(np.concatenate([final_samples, additional_samples]))
    
    return combined_samples


def adj_matching_loss(v_base, v_fine, adj, sigma):
    """Adjoint matching loss for FM"""
    diff = v_fine - v_base
    new_shape = (diff.shape[0],) + (1,) * (diff.dim() - 1)
    sigma = sigma.reshape(new_shape)
    term_diff = (2 / sigma) * diff
    term_adj = sigma * adj
    term_difference = term_diff - term_adj
    term_difference = torch.sum(torch.square(term_difference), dim=[d for d in range(1, term_difference.dim())])
    loss = torch.mean(term_difference)
    return loss 


class AMTrainerFlow:
    def __init__(self,
            config: OmegaConf,
            model: FlowModel,
            base_model: FlowModel,
            grad_reward_fn: Callable,
            grad_f_k_fn: Optional[Callable] = None,
            device: Optional[torch.device] = None,
            verbose: bool = False,
            sampler: Optional[Sampler] = None
        ):
        # Config
        self.config = config
        self.sampling_config = config.sampling
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.max_nodes = config.get("max_nodes", 210)

        # Clip
        self.clip_grad_norm = config.get("clip_grad_norm", 1e5)
        self.clip_loss = config.get("clip_loss", 0.5)

        # Models
        self.fine_model = model
        self.base_model = base_model
        self.fine_model.to(self.device)
        self.base_model.to(self.device)

        # Reward (Gradient of the reward function(al))
        self.grad_reward_fn = grad_reward_fn
        self.grad_f_k_fn = grad_f_k_fn

        # Setup optimizer
        self.configure_optimizers()

        if sampler is None:
            if 'data_shape' not in self.sampling_config:
                raise ValueError('Either pass an explicit sampler or pass a data_shape in the sampling config (config.sampling)')
            self.sampler = EulerMaruyamaSampler(base_model, self.sampling_config.data_shape)
        else:
            self.sampler = sampler


    def configure_optimizers(self):
        if hasattr(self, 'optimizer'):
            del self.optimizer
        self.optimizer = torch.optim.Adam(self.fine_model.parameters(), lr=self.config.lr)


    def get_model(self):
        return self.fine_model


    def sample_trajectories(self):
        N = self.sampling_config.num_samples
        T = self.sampling_config.num_integration_steps + 1
        self.sampler.model = self.fine_model
        trajectories, ts = self.sampler.sample_trajectories(N=N, T=T)
        ts = ts.to(self.sampler.device)
        sigmas = self.fine_model.interpolant_scheduler.memoryless_sigma_t(ts)
        return trajectories, ts, sigmas


    def generate_dataset(self):
        """Sample dataset for training based on adjoint ODE and sampled trajectories."""
        datasets = []

        # run in eval mode
        self.fine_model.eval()
        self.base_model.eval()

        solver = LeanAdjointSolverFlow(self.base_model, self.grad_reward_fn, self.grad_f_k_fn)

        iterations = self.sampling_config.num_samples // self.config.batch_size
        for i in range(iterations):
            with torch.no_grad():
                trajectories, ts, sigmas = self.sample_trajectories()

            # graph_trajectories is a list of the intermediate graphs
            solver_info = solver.solve(trajectories=trajectories, ts=ts)
            # add sigma_t to solver_info
            solver_info['sigmas'] = sigmas
            dataset = AMDataset(solver_info=solver_info)
            datasets.append(dataset)

        if len(datasets) == 0:
            return None
        dataset = ConcatDataset(datasets)
        return dataset


    def train_step(self, sample):
        """Training step."""

        ts = sample['ts'].to(self.device)
        traj_g = [g.to(self.device) for g in sample['traj_x']]
        traj_adj = sample['traj_adj'].to(self.device)
        traj_v_base = sample['traj_v_base'].to(self.device)

        # Get index for time steps to calculate adjoint matching loss
        idxs = create_timestep_subset(ts.shape[0])

        v_base = []
        v_fine = []
        adj = []
        sigma = self.base_model.interpolant_scheduler.memoryless_sigma_t(ts[idxs])

        for idx in idxs:
            t = ts[idx]
            adj_t = traj_adj[idx]
            v_base_t = traj_v_base[idx]
            g_base_t = traj_g[idx]

            v_fine_t = self.fine_model.velocity_field(g_base_t.full, t)
                        
            v_base.append(v_base_t)
            v_fine.append(v_fine_t)
            adj.append(adj_t)
        
        # stack the tensors
        v_base = torch.stack(v_base, dim=0)
        v_fine = torch.stack(v_fine, dim=0)
        adj = torch.stack(adj, dim=0)
        
        loss = adj_matching_loss(
            v_base=v_base,
            v_fine=v_fine,
            adj=adj,
            sigma=sigma,
        )

        if loss.isnan().any():
            return torch.tensor(float("inf"), device=self.device)
        
        # step optimizer
        self.optimizer.zero_grad()

        # self.fine_model.zero_grad()
        loss.backward(retain_graph=False)

        # loss clapping
        if self.clip_loss > 0.0:
            loss = torch.clamp(loss, min=0.0, max=self.clip_loss)

        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.fine_model.parameters(), self.clip_grad_norm)

        self.optimizer.step()

        return loss


    def finetune(self, dataset, steps=None, debug=False):
        """Finetuning the model."""

        c = 0
        losses = []

        self.fine_model.to(self.device)
        self.fine_model.train()
        
        self.optimizer.zero_grad()

        # iterate over the dataset
        if steps is not None:
            idxs = np.random.permutation(dataset.__len__())[:steps]
        else:
            idxs = np.random.permutation(dataset.__len__())
        
        for idx in idxs:
            sample = dataset[idx]
            loss = self.train_step(sample).item()
            losses.append(loss)
            c+=1 

        del dataset
        return losses if debug else sum(losses) / c