import abc
from dataclasses import dataclass
import torchdiffeq
import torch
from .models import DiffusionModel, FlowModel, SDE, VPSDE
from typing import Sequence

class Solver(abc.ABC):
    def solve(self, x0, ts=None, steps=50, store_traj=False, device=None):
        device = x0.device if device is None else device
        dtype = x0.dtype
        self.device = device
        if ts is None:
            ts = torch.linspace(0, 1, steps+1, dtype=dtype, device=device)
        if store_traj:
            traj = [x0]
        else:
            traj = None
        xt = x0
        for t, tph in zip(ts[:-1], ts[1:]):
            xt = self.step(xt=xt, t=t, tph=tph)
            if traj is not None:
                traj.append(xt)
        return xt, {'traj': traj}


    def step(self, *, f, **kwargs):
        raise NotImplementedError()


class ODE:
    def f(self, x, t):
        """Return the derivative of x at time t."""
        raise NotImplemented


class ODESolver(Solver):
    """Simple ODE solver implementation for model predicting velocity."""
    
    def __init__(self, f):
        self._f = f # f is a function that takes x and t and tm1 as input and returns dx/dt

    def solve(self, x0, ts=None, steps=50, store_traj=False):
        if ts is None:
            ts = torch.linspace(1,0, steps+1)
        if store_traj:
            traj = []
        else:
            traj = None
        xt = x0
        for t, tm1 in zip(ts[:-1], ts[1:]):
            xt = self._f(x=xt, t=t, tm1=tm1)
            if traj is not None:
                traj.append(xt.cpu().detach()) # detach to avoid memory leak
        return xt, {'traj': traj}


class TorchDiffEqSolver:
    def __init__(self, ode: ODE):
        self.ode = ode    
    
    def solve(self, x0, **kwargs):
        def f(t, y):
            v = self.ode.f(y, t)
            return v

        traj = torchdiffeq.odeint(
            func=f,
            y0=x0,
            **kwargs
        )
        if isinstance(traj, (tuple,list)):
            final_state = tuple(t[-1] for t in traj)
        else:
            final_state = traj[-1]
        return final_state, {'traj': traj}


class TorchDiffEqRK4Solver(TorchDiffEqSolver):
    """Simple RK4 ODE solver."""
    def solve(self, x0, ts):
        return super().solve(x0, method='rk4', t=ts)


def ddim_step(x, t, tm1, model, 
              noise_func, 
              device='cuda', 
              sig_fn=lambda t,tm1: 0.0,
              avoid_inf=1e-6):
    # model prediction
    x = x.to(device)
    t_in = torch.tensor(t, device=device).expand(x.size(0))[:, None]

    eps_pred = model(x, t_in)

    atm1, btm1 = noise_func(tm1)
    at, bt = noise_func(t)
    if at.ndim != 0:
        at = at.view(-1,1).to(x.device).to(x.dtype)
        atm1 = atm1.view(-1,1).to(x.device).to(x.dtype)
    sqrt_atm1 = atm1.sqrt()
    sqrt_1m_atm1 = torch.sqrt(1.0 - atm1)
    sqrt_at =  at.sqrt()
    sqrt_1m_at = torch.sqrt(1.0 - at)

    sig = sig_fn(t,tm1)
    if torch.is_tensor(sig):
        sig = sig.to(x.device).to(x.dtype)

    assert torch.all(sqrt_atm1 >= sqrt_at)
    
    # print('ddim:', x.shape, sig, sqrt_1m_at.shape)

    e = torch.randn_like(x)
    btm1 = torch.sqrt(torch.clip(1-atm1-sig**2, 0)) # in case of noisy sampling, also clip to avoid negative values


    #print(1-atm1-sig**2)
    #print(sig.isfinite().all(), btm1.isfinite().all(), atm1.isfinite().all(), at.isfinite().all())

    # print(x.device, sqrt_1m_atm1.device, eps_pred.device, e.device)
    x0_pred = (x - sqrt_1m_at*eps_pred)/(sqrt_at + avoid_inf)
    #print(eps_pred.shape, x.shape, x0_pred.shape)
    
    xtm1 = sqrt_atm1 * x0_pred + btm1 * eps_pred + sig*e
    return xtm1


class DDIMSolver(Solver):
    """Simple DDIM solver implementation for model predicting noise."""

    def __init__(self, model: DiffusionModel, avoid_inf=1e-6):
        self.model = model
        self.noise_func = model.get_sde().get_alpha_sigma
        self.avoid_inf = avoid_inf
        self.device = None # set in the solving loop

    def step(self, x, t, tm1, sig_fn):
        return ddim_step(x, t, tm1, self.model, self.noise_func, 
                         sig_fn=sig_fn, device=self.device, avoid_inf=self.avoid_inf)

    def solve(self, x0, ts=None, steps=50, store_traj=False, sig_fn=lambda t,tm1: 0.0):
        device = next(self.model.parameters()).device
        dtype = x0.dtype
        self.device = device
        if ts is None:
            ts = torch.linspace(1,0, steps+1, dtype=dtype, device=device)
        if store_traj:
            traj = [x0.cpu().detach()]
        else:
            traj = None
        xt = x0
        for t, tm1 in zip(ts[:-1], ts[1:]):
            xt = self.step(x=xt, t=t, tm1=tm1,sig_fn=sig_fn)
            if traj is not None:
                traj.append(xt.cpu().detach())
        return xt, {'traj': traj}


class EMDiffusionSolver(Solver):
    """
    Test class: Euler-Maruyama solver for diffusion models
    """

    def __init__(self, model: DiffusionModel, avoid_inf=1e-6):
        self.model = model
        self.noise_func = model.get_sde().get_alpha_sigma
        self.avoid_inf = avoid_inf
        self.device = None # set in the solving loop


    def step(self, xt, t, tph):
        betat = self.model.sde.beta_t(t)
        et = torch.randn_like(xt).to(xt.device)
        score = self.model.score_func(xt, t)
        h = torch.abs(t - tph)
        return xt + h * (0.5 * betat * xt + betat * score) + torch.sqrt(betat * h) * et


class EulerMaruyamaSolver(Solver):
    """
    Euler-Maruyama solver of Flow SDE with arbitrary noise (default is memoryless)
    """

    def __init__(self, model: FlowModel, avoid_inf=1e-6, noise_func=None):
        self.model = model
        self.avoid_inf = avoid_inf
        self.noise_func = self.model.interpolant_scheduler.memoryless_sigma_t if noise_func is None else noise_func


    def step(self, xt, t, tph):
        h = torch.abs(t - tph)
        schedule = self.model.interpolant_scheduler
        alpha_t, beta_t = schedule.interpolants(t)
        alpha_dot_t, beta_dot_t = schedule.interpolants_prime(t)
        kt = alpha_dot_t / alpha_t
        st = self.noise_func(t)
        et = torch.randn_like(xt).to(xt.device)
        score_t = self.model.score_func(xt, t)
        etat = beta_t * (kt * beta_t - beta_dot_t)
        xtph = xt + h * kt * xt + h * (st**2 / 2 + etat) * score_t + torch.sqrt(h) * st * et
        return xtph


class MemorylessFlowSolver(Solver):
    """
    Solver for memoryless sampling with Flow Models (Euler-Maruyama Solver with memoryless noise schedule)
    """

    def __init__(self, model: FlowModel, avoid_inf=1e-6):
        self.model = model
        self.avoid_inf = avoid_inf
        self.device = None # set in the solving loop


    def step(self, xt, t, tph):
        h = torch.abs(t - tph)
        schedule = self.model.interpolant_scheduler
        kt = schedule.alpha_t_prime(t) / schedule.alpha_t(t)
        st = schedule.memoryless_sigma_t(t)
        et = torch.randn_like(xt).to(xt.device)
        vft = self.model.velocity_field(xt, t)
        xtph = xt + h * (2 * vft - kt * xt) + torch.sqrt(h) * st * et
        return xtph

    
def linear_schedule(t, beta_0, beta_1):
    return beta_0 + (beta_1 - beta_0) * t


class PFODE(ODE):
    def __init__(self, model: DiffusionModel, sde: SDE, sign=1):
        self.model = model
        self.sde = sde
        self.sign = sign
    
    def f(self, x, t):
        v = self.sde.pf_ode_vel(x, t, self.model)
        return self.sign*v


class PFODESolver(ODESolver):
    def __init__(self, model, vpsde: VPSDE):
        def f(x, t, tm1):
            return x + vpsde.pf_ode_vel(x, t, model)*(t-tm1)
        super().__init__(f)


def sample_trajectories_ddim(self: DiffusionModel, x0, T, sig_fn=lambda t,tm1: 0.0, avoid_inf=0.0, sample_jumps=True):
    """
    Sample N trajectories of length T from the model.
    """
    if not sample_jumps:
        ts = torch.linspace(1, 0, T).to(x0.device)
    else:
        ts = torch.linspace(1, 0, 1000)
        # sample T ts
        idxs = torch.randperm(998)+1
        idxs = idxs[:T-2]
        ts = torch.cat([ts[0, None], ts[idxs], ts[-1,None]])
        # print(ts)
        # order the ts descending
        ts = torch.sort(ts, descending=True)[0].to(x0.device)
    solver = DDIMSolver(model, avoid_inf=avoid_inf)
    xts, info = solver.solve(x0, ts=ts, sig_fn=sig_fn, store_traj=True)
    traj = torch.stack(info['traj'])
    return traj


def sample_trajectories_ddpm(model: DiffusionModel, x0, T, avoid_inf=0.0, sample_jumps=True):
    """
    Sample N trajectories of length T from the model.
    """
    if not sample_jumps:
        ts = torch.linspace(1, 0, T).to(x0.device)
    else:
        ts = torch.linspace(1, 0, 1000)
        # sample T ts
        idxs = torch.randperm(998)+1
        idxs = idxs[:T-2]
        ts = torch.cat([ts[0, None], ts[idxs], ts[-1,None]])
        # print(ts)
        # order the ts descending
        ts = torch.sort(ts, descending=True)[0].to(x0.device)
    solver = DDIMSolver(model, avoid_inf=avoid_inf)
    def sig_fn_ddpm(t,tm1):
        at, sig = model.sde.get_alpha_sigma(t)
        atm1, _ = model.sde.get_alpha_sigma(tm1)
        sig_t = torch.sqrt((1-atm1)/(1-at+avoid_inf)*(1-at/atm1))
        return sig_t
    xts, info = solver.solve(x0, ts=ts, sig_fn=sig_fn_ddpm, store_traj=True)
    traj = info['traj']
    
    return torch.stack(traj).permute(1,0,2), ts # flip to (B,T,d)


def sample_trajectories_memoryless(model: FlowModel, x0, T, avoid_inf=0.0, sample_jumps=True):
    """
    Sample N trajectories of length T using memoryless sampling
    """
    if not sample_jumps:
        ts = torch.linspace(0, 1, T).to(x0.device)
    else:
        ts = torch.linspace(0, 1, 1000) # TODO remove 1000 magic? use default param? also need T < magic
        # sample T ts
        idxs = torch.randperm(998)+1
        idxs = idxs[:T-2]
        ts = torch.cat([ts[0, None], ts[idxs], ts[-1,None]])

        # order the ts ascending
        ts = torch.sort(ts, descending=False)[0].to(x0.device)
    solver = MemorylessFlowSolver(model, avoid_inf=avoid_inf)

    xts, info = solver.solve(x0, ts=ts, store_traj=True)
    traj = info['traj']
    
    return torch.stack(traj).permute(1,0,2), ts # flip to (B,T,d)


class Sample(object):
    def __init__(self, obj):
        self.obj = obj

    
    @property
    def full(self):
        return self.obj
    

    @full.setter
    def full(self, value):
        self.obj = value


    @property
    def adjoint(self):
        return self.obj
    

    @adjoint.setter
    def adjoint(self, value):
        self.obj = value
    

    def detach_all(self):
        self.obj = self.obj.detach()
    

    def to(self, device: torch.device):
        return Sample(self.obj.to(device))


class Sampler(object):
    def __init__(self, model: FlowModel, data_shape=None, device=None):
        self.model = model
        self.data_shape = data_shape
        self.solver = None
        self.device = device
    
    
    def sample_init_dist(self, N=1, device=None):
        device = self.device if device is None else device
        return torch.randn(N, *self.data_shape, device=device)


    def sample_trajectories(self, N=1, T=1000, sample_jumps=False, device=None) -> tuple[Sequence[Sample], torch.Tensor]:
        """
        Sample N trajectories of length T using memoryless sampling
        """
        x0 = self.sample_init_dist(N, device=device)

        if not sample_jumps:
            ts = torch.linspace(0, 1, T).to(x0.device)
        else:
            ts = torch.linspace(0, 1, 1000)
            # sample T ts
            idxs = torch.randperm(998)+1
            idxs = idxs[:T-2]
            ts = torch.cat([ts[0, None], ts[idxs], ts[-1,None]])

            # order the ts ascending
            ts = torch.sort(ts, descending=False)[0].to(x0.device)

        xts, info = self.solver.solve(x0, ts=ts, store_traj=True, device=x0.device)
        traj = info['traj']
        traj = [Sample(t) for t in traj]
        
        return traj, ts


class EulerMaruyamaSampler(Sampler):
    def __init__(self, model: FlowModel, data_shape, noise_func=None, device=None):
        super().__init__(model, data_shape=data_shape, device=device)
        self.solver = EulerMaruyamaSolver(model, noise_func=noise_func)


class MemorylessSampler(Sampler):
    def __init__(self, model: FlowModel, data_shape, device=None):
        super().__init__(model, data_shape=data_shape, device=device)
        self.solver = MemorylessFlowSolver(model)
    

class DDIMSampler(Sampler):
    def __init__(self, model: DiffusionModel, data_shape, device=None):
        super().__init__(model, device=device)
        self.data_shape = data_shape
        self.solver = DDIMSolver(model)
    

    def sample_trajectories(self, N, T, sample_jumps=True, sig_fn=None):
        """
        Sample N trajectories of length T from the model.
        """
        x0 = self.sample_init_dist(N)
        if not sample_jumps:
            ts = torch.linspace(1, 0, T).to(x0.device)
        else:
            ts = torch.linspace(1, 0, 1000)
            # sample T ts
            idxs = torch.randperm(998)+1
            idxs = idxs[:T-2]
            ts = torch.cat([ts[0, None], ts[idxs], ts[-1,None]])
            # print(ts)
            # order the ts descending
            ts = torch.sort(ts, descending=True)[0].to(x0.device)

        def sig_fn_ddpm(t,tm1):
            avoid_inf = self.solver.avoid_inf
            at, sig = self.model.sde.get_alpha_sigma(t)
            atm1, _ = self.model.sde.get_alpha_sigma(tm1)
            sig_t = torch.sqrt((1-atm1)/(1-at+avoid_inf)*(1-at/atm1))
            return sig_t
        
        sig_fn = sig_fn_ddpm if sig_fn is None else sig_fn

        xts, info = self.solver.solve(x0, ts=ts, sig_fn=sig_fn, store_traj=True)
        traj = info['traj']
        
        return torch.stack(traj).permute(1,0,2), ts # flip to (B,T,d)


class EMDiffusionSampler(Sampler):
    def __init__(self, model: FlowModel, data_shape):
        super().__init__(model, data_shape=data_shape)
        self.solver = EMDiffusionSolver(model)