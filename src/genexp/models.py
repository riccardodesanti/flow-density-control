import os
import logging
import abc

import torch

logger = logging.getLogger(__name__)

def adapt(xt, t):
    """Bring input to right shape."""
    if t.ndim == 0:
        t = t.expand(xt.size(0)).view(-1, 1)
    elif t.ndim == 1:
        if t.size(0) == 1:
            t = t.expand(xt.size(0)).view(-1, 1)
        t = t.view(-1, 1)
    return xt, t


class InterpolantScheduler(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def beta_t(self, t):
        raise NotImplementedError()
    
    def beta_t_prime(self, t):
        raise NotImplementedError()
    
    def alpha_t(self, t):
        raise NotImplementedError()
    
    def alpha_t_prime(self, t):
        raise NotImplementedError()
    
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


class FlowModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, interpolant_scheduler: InterpolantScheduler):
        super().__init__()
        self.model = model
        self.interpolant_scheduler = interpolant_scheduler

    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # expand t if needed
        if t.ndim == 0 or len(t) < len(x):
            t = t.expand(x.size(0)).view(-1, 1)

        return self.model(torch.cat([x, t], dim=1))

    
    def velocity_field(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Flow models predict the velocity field by default"""
        return self(x, t)


    def score_func(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        vf = velocity_field(x, t)
        at, adt = self.interpolant_scheduler.interpolants(t)
        bt, bdt = self.interpolant_scheduler.interpolants_prime(t)
        adoat = adt / at
        return (vf - adoat * x) / (bt * (adoat * bt - bdt))


class SDE:
    def sde(self, t):
        """Return the drift and diffusion coefficients of the SDE."""
        raise NotImplementedError()
    
    def get_alpha_sigma(self, t):
        """Return the alpha and sigma coefficients at time t."""
        raise NotImplementedError()
    
    def get_alpha_prime(self, t):
        """Return the time-derivative of alpha at time t."""
        raise NotImplementedError()


def linear_schedule(t, beta_0, beta_1):
        return beta_0 + (beta_1 - beta_0) * t


class VPSDE(SDE):
    def __init__(self, beta_0, beta_1, noise_schedule=linear_schedule, N=1000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.schedule_func = noise_schedule
        self.N = N
        self.discrete_betas = torch.linspace(beta_0 / N, beta_1 / N, N, device=device)
        self.discrete_alphas = (1.0 - self.discrete_betas).cumprod(dim=0).to(device)
        self.sigmas = torch.sqrt(1.0 - self.discrete_alphas)
        self.sqrt_alphas = torch.sqrt(self.discrete_alphas)


    def sde(self, x,t):
        beta_t = self.beta_t(t)
        drift = -0.5 * torch.vmap(lambda x,y: x*y)(beta_t, x)
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def beta_t(self, t):
        return self.schedule_func(t, self.beta_0, self.beta_1)

    def alpha_bar_t(self, t):
        return torch.exp(-t * self.beta_0 - t**2 / 2. * (self.beta_1 - self.beta_0))
    
    def alpha_bar_t_prime(self, t):
        bt = self.beta_t(t)
        abt = self.alpha_bar_t(t)
        return -bt * abt
    
    
    def get_alpha_sigma(self, t):
         # Ensure t is a PyTorch tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)  # Convert to tensor
        if t.ndim == 0:  # Ensure it's a 1D tensor for operations
            t = t.view(1)
        
        # Compute index (rounding and clamping to [0, N - 1])
        idx = torch.clamp(torch.round(t * (self.N - 1)), min=0, max=self.N - 1).long()  # Ensure index is an integer tensor
        
        # Fetch alpha and sigma
        alpha_t = self.discrete_alphas[idx]         # alpha(t), scalar
        sigma_t = self.sigmas[idx]         # sigma(t), scalar
        return alpha_t, sigma_t
    
    
class DiffusionInterpolant(InterpolantScheduler):
    def __init__(self, sde: SDE):
        super().__init__()
        self.sde = sde
    
    def beta_t(self, t):
        return torch.sqrt(1. - self.alpha_t(t)**2)

    def beta_t_prime(self, t):
        return - self.alpha_t(t) * self.alpha_t_prime(t) / self.beta_t(t)
    
    def alpha_t(self, t):
        return torch.sqrt(self.sde.alpha_bar_t(1. - t)) # SDE is forward-time, want reverse time
    
    def alpha_t_prime(self, t):
        abtp = -self.sde.alpha_bar_t_prime(1. - t) # SDE is forward-time, want reverse time
        at = self.alpha_t(t)
        return abtp / (2. * at)


class DiffusionModel(FlowModel):
    def __init__(self, model: torch.nn.Module, sde: SDE):
        super().__init__(model, DiffusionInterpolant(sde))
        self.sde = sde
    
    
    def get_sde(self):
        return self.sde

    
    def velocity_field(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s = self.score_func(x, t)
        at, bt = self.interpolant_scheduler.interpolants(t)
        atp, btp = self.interpolant_scheduler.interpolants_prime(t)
        etat = atp / at

        return etat * x + bt * (etat * bt - btp) * s
        

    def score_func(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Diffusion models predict the error function by default, with time going from 1 (noise) to 0 (data)
        Instead we want time to go from 0 (noise) to 1 (data)
        """
        eps_pred = self.forward(x, 1. - t)
        _, sigma = self.sde.get_alpha_sigma(1. - t)
        return -eps_pred/sigma.to(eps_pred.device).to(eps_pred.dtype)
    

    def sample_init(self, n=1):
        dtype = next(self.model.parameters()).dtype
        x0 = torch.randn(n, *self.data_shape).to(self.device).to(dtype)
        return x0

