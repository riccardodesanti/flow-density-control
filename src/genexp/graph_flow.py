from genexp.models import FlowModel, InterpolantScheduler
from genexp.sampling import Sampler, Sample
from flowmol import FlowMol
from flowmol.models.interpolant_scheduler import InterpolantScheduler as FlowMolInterpolantScheduler
import torch
import dgl


class GraphInterpolantScheduler(InterpolantScheduler):
    def __init__(self, scheduler: FlowMolInterpolantScheduler):
        super().__init__()
        self.scheduler = scheduler
    
    def beta_t(self, t): # TODO check dimensions etc, and for all other funcs
        return self.scheduler.beta_t(t)[:, 0] # TODO if dim = 2 otw [0]
    
    def beta_t_prime(self, t):
        return self.scheduler.beta_t_prime(t)[:, 0]
    
    def alpha_t(self, t):
        return self.scheduler.alpha_t(t)[:, 0]
    
    def alpha_t_prime(self, t):
        return self.scheduler.alpha_t_prime(t)[:, 0]
    
    def interpolants(self, t):
        return self.alpha_t(t), self.beta_t(t)
    
    def interpolants_prime(self, t):
        return self.alpha_t_prime(t), self.beta_t_prime(t)
    
    def eta_t(self, t):
        at, bt = self.interpolants(t)
        atp, btp = self.interpolants_prime(t)
        kt = atp / at
        return bt * (kt * bt - btp)

    def memoryless_sigma_t(self, t):
        return torch.sqrt(2 * self.eta_t(t))


class GraphFlowModel(FlowModel):
    """
    Wrapper for FlowMol
    """
    def __init__(self, model: FlowMol):
        super().__init__(model, GraphInterpolantScheduler(model.interpolant_scheduler))
    

    def velocity_field(self, x, t, ue_mask=None):
        node_batch_idx = torch.zeros(x.num_nodes(), dtype=torch.long)
        upper_edge_mask = x.edata['ue_mask'] if ue_mask is None else ue_mask

        dst_dict = self.model.vector_field(
            x, 
            t=torch.full((x.batch_size,), t, device=x.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True
        )

        # take integration step for positions
        x_1 = dst_dict['x']
        x_t = x.ndata['x_t']
        alpha, alpha_dot = self.model.interpolant_scheduler.alpha_t(t), self.model.interpolant_scheduler.alpha_t_prime(t)

        v_pred = self.model.vector_field.vector_field(x_t, x_1, alpha, alpha_dot)
        return v_pred


class GraphSample(Sample):
    def __init__(self, graph: dgl.DGLGraph):
        self.obj = graph

    
    @property
    def full(self):
        return self.obj
    

    @full.setter
    def full(self, value):
        self.obj = value


    @property
    def adjoint(self):
        return self.obj.ndata['x_t']
    

    @adjoint.setter
    def adjoint(self, value):
        self.obj = value
    

    def detach_all(self):
        self.obj = self.obj.detach()
    

    def to(self, device: torch.device):
        return GraphSample(self.obj.to(device))
    

class GraphEulerMaruyamaSampler(Sampler):
    def __init__(self, model: GraphFlowModel, sampler_type=None):
        super().__init__(model)
        self.sampler_type = sampler_type
    

    def sample_init_dist(self, N=1, device=None):
        return self.model.model.sample_n_atoms(N)


    def sample_trajectories(self, N=1, T=1000, sample_jumps=False, device=None, n_atoms=None, sampler_type=None):
        """
        Sample N trajectories of length T using memoryless sampling
        """
        atoms_per_mol = self.sample_init_dist(N)
        if n_atoms is not None:
            atoms_per_mol = atoms_per_mol * 0 + n_atoms

        _, graph_trajectories = self.model.model.sample(atoms_per_mol,
                n_timesteps = T,     
                sampler_type = self.sampler_type if sampler_type is None else sampler_type,
                device = self.model.device,
                keep_intermediate_graphs = True,
            )

        trajs = [GraphSample(g) for g in graph_trajectories]

        return trajs, torch.linspace(0, 1, T)