import os
import logging
import abc

import torch


from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler


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


class DiffusersVPSDE(torch.nn.Module):
    def __init__(self, scheduler: DDIMScheduler, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.beta_0 = scheduler.betas[0]
        self.beta_1 = scheduler.betas[-1]
        self.N = 1000
        self.discrete_betas = torch.nn.Parameter(scheduler.betas, requires_grad=False).detach()
        self.alphas = torch.nn.Parameter((1.0 - self.discrete_betas).cumprod(dim=0).to(device), requires_grad=False).detach()
        self.sigmas = torch.nn.Parameter(torch.sqrt(1.0-self.alphas), requires_grad=False).detach()
        self.sqrt_alphas = torch.nn.Parameter(torch.sqrt(self.alphas), requires_grad=False).detach()
        
    def sde(self, x,t):
        idx = torch.clamp(torch.round(t * 999), min=0, max=999).long() # TODO change 999 to self.N
        beta_t = self.discrete_betas[idx]
        drift = -0.5 * torch.vmap(lambda x,y: x*y)(beta_t, x)
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def beta_t(self, t):
        # discrete estimate
        idx = torch.clamp(torch.round(t * 999), min=0, max=999).long().to(self.discrete_betas.device)
        return self.discrete_betas[idx].to(t.device)
    
    def get_alpha_sigma(self, t):
         # Ensure t is a PyTorch tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)  # Convert to tensor
        if t.ndim == 0:  # Ensure it's a 1D tensor for operations
            t = t.view(1)
        
        # Compute index (rounding and clamping to [0, 999])
        idx = torch.clamp(torch.round(t * 999), min=0, max=999).long()  # Ensure index is an integer tensor
        # Fetch alpha and sigma
        idx = idx.to(self.alphas.device)
        alpha_t = self.alphas[idx]         # alpha(t), scalar
        sigma_t = self.sigmas[idx]         # sigma(t), scalar
        return alpha_t, sigma_t
    
    def pf_ode_vel(self, x, t, model: torch.nn.Module):
        device = x.device
        # Ensure t is a 1D tensor so we can broadcast
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=x.dtype, device=device)
        elif t.ndim == 0:
            t = t.view(1)

        # 1) Compute alpha_t, sigma_t by indexing
        alpha_t, sigma_t = self.get_alpha_sigma(t.item())  # scalar
        # or if you have a batch version, you'd do something vectorized.

        # 2) Evaluate beta(t)
        b_t = self.beta_t(t).to(x.dtype)  # shape (1,) if t is shape (1,)

        # 3) Get eps_pred from the model
        #    Typically: model( concat[x, t], ) => shape (batch, dim)
        #    You might do something like:
        t_expand = t.repeat(x.size(0), 1)  # shape (batch, 1)
        eps_pred = model(x, t_expand.to(x.dtype))  # shape (batch, dim)
        sigma_t = sigma_t.to(eps_pred.dtype)

        # 4) Convert eps_pred => score factor: (+0.5 * beta(t) * eps_pred / sigma_t)
        drift = -0.5 * b_t * x  # shape (batch, dim)
        score_term = 0.5 * b_t * eps_pred / (sigma_t + 1e-12)  # avoid div-by-zero

        dx_dt = drift + score_term

        return dx_dt


class CastModule(torch.nn.Module):
    def __init__(self, module, dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x):
        return self.module(x.to(self.dtype))


class StableDiffusion(DiffusionModel):
    def __init__(self, device='cuda',
                 guidance_scale=None, compile=False, dtype='float32', **pipeline_kwargs):
        dtype = getattr(torch, dtype)
        pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=dtype, **pipeline_kwargs)
        logger.debug(f'pipeline config: {pipe.config}')
        pipe = pipe.to(device)
        # pipe.disable_attention_slicing()
        # pipe.disable_vae_slicing()
        # pipe.disable_vae_tiling()
        if compile:
            pipe.unet = torch.compile(pipe.unet)
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        sde = DiffusersVPSDE(scheduler, device=device)
        self.latent_channels = 4
        self.latent_dim = 64
        self.pipe = pipe
        self.dtype = dtype
        super().__init__(pipe.unet, sde)
        for param in self.sde.parameters():
            # we won't train sde
            param.requires_grad = False
        for param in pipe.text_encoder.parameters():
            param.requires_grad = False
        for param in pipe.vae.parameters():
            param.requires_grad = False

        self.device = device
        self._pemb = None
        self._negpemb = None
        self._guidance_scale = guidance_scale

    def remove_from_pipeline(self, attrs):
        for attr in attrs:
            if hasattr(self.pipe, attr):
                delattr(self.pipe, attr)
            else:
                raise AttributeError(f"Attribute {attr} not found in pipeline")

    def to(self, device):
        self.device = device
        self.pipe = self.pipe.to(device)
        if self._pemb is not None:
            self._pemb = self._pemb.to(device)
            self._negpemb = self._negpemb.to(device)
        self.dtype = next(self.model.parameters()).dtype
        self.sde = self.sde.to(device)
        # if isinstance(self.model.time_proj,CastModule):
        #     self.model.time_proj = CastModule(self.model.time_proj.module, self.dtype)
        # else:
        #     self.model.time_proj = CastModule(self.model.time_proj, self.dtype)

        return super().to(device)

    def encode_prompt(self, prompt):
        """Ideally this is rarely called"""
        res = self.pipe.encode_prompt(
            prompt,
            self.device,
            1, # num images per prompt
            True,
            None, # TODO check if this is truly the negative prompt
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None
        )
        self._pemb, self._negpemb = res[0].detach(), res[1].detach()
        assert self._pemb.shape == self._negpemb.shape
    
    def set_guidance_scale(self, guidance_scale):
        self._guidance_scale = guidance_scale

    def sample_prior(self, N):
        return torch.randn((N, self.latent_channels, self.latent_dim,self.latent_dim), dtype=self.dtype)

    def sample_prior_flat(self, N):
        return self.sample_prior(N).reshape(N, -1)

    def decode(self, x0):
        if x0.ndim == 2:
            B = x0.shape[0]
            x0 = x0.view(B, self.latent_channels, self.latent_dim, self.latent_dim)

        with torch.no_grad():
            return self.pipe.decode_latents(x0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                prompt: list[str]=None, 
                guidance_scale: float=None):
        
        # print('first x', x.shape)

        B,C,H,W = x.shape[0], self.latent_channels, self.latent_dim, self.latent_dim
   
        if prompt is not None:
            self.encode_prompt(prompt)
        if guidance_scale is not None:
            self.set_guidance_scale(guidance_scale)

        # print('x shape 2', x.shape, B, C, H, W)
        x = x.view(B,C,H,W)

        # print('t before', t.shape)
        t = t.flatten()
        
        #print('t shape 1', t.shape)
        assert t.ndim == 1
        # assume t is float for compatibility, convert to long
        t = torch.clamp(torch.round(t * 999), min=0, max=999).long()
        pemb, negpemb = self._pemb, self._negpemb
        
        # print('pemb shape', pemb.shape, negpemb.shape)
        pemb = pemb.repeat(B, 1, 1)
        negpemb = negpemb.repeat(B, 1, 1)

        # print('pemb', pemb.shape, negpemb.shape)
        batch_pemb = torch.cat([pemb, negpemb], 0)
        x = torch.cat([x,x], 0) # because of guidance
        
        if  t.ndim == 0 or len(t) < len(x):
            t = t.expand(len(x)//2).flatten()
            t = torch.cat([t,t], 0)

        # predict the noise residual
        # print('before  model', x.shape, t.shape, batch_pemb.shape)
        # print('x shape', x.shape, x.device, t.shape, t.device, batch_pemb.device, next(self.model.parameters()).device)
        eps = self.model(
            x,
            t,
            encoder_hidden_states=batch_pemb.to(x.dtype), # TODO quick hack
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False
        )[0]

        eps_c, eps_uc = torch.chunk(eps, 2, dim=0)
        logger.debug('c, uc', eps_c.shape, eps_uc.shape)
        
        # guidance
        eps_g =  eps_uc + self._guidance_scale*(eps_c-eps_uc)


        #rint('out shape eps_g', eps_g.shape)
        logger.debug('eps_g', eps_g.shape)
        return eps_g.view(B, -1)

