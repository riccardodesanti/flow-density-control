import torch

import argparse
from omegaconf import OmegaConf
import random

def seed_everything(seed: int):
    """Seed all random generators."""
    # For random:
    random.seed(seed)

    # For numpy:
    np.random.seed(seed)

    # For PyTorch:
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ALM with optional parameter overrides")
    # Settings
    parser.add_argument("--config", type=str, default="configs/example_fdc.yaml",
                        help="Path to config file")
    parser.add_argument("--use_wandb", action='store_true',
                        help="Use wandb, default: false")
    parser.add_argument("--verbose", action='store_true', 
                        help="Verbose output, default: false")
    parser.add_argument("--save_model", action='store_true',
                        help="Save the model, default: false")
    parser.add_argument("--save_samples", action='store_true',
                        help="Create animation of the samples and save the samples, default: false")
    parser.add_argument("--save_plots", action='store_true',
                        help="Save plots of rewards and constraints, default: false")
    parser.add_argument("--plotting_freq", type=int, default=1,
                        help="Plotting frequency")
    # Reward
    parser.add_argument("--reward", type=str, default="dipole",
                        help="Override reward in config")
    # FlowMol arguments
    flowmol_choices = ['qm9_ctmc', 'qm9_gaussian', 'qm9_simplexflow', 'qm9_dirichlet']
    # But just implemented for qm9_ctmc
    parser.add_argument('--flow_model', type=str, choices=flowmol_choices,
                        help='pretrained model to be used')
    # Maximum Entropy Parameters
    parser.add_argument("--reward_lambda", type=float,
                        help="Override reward_lambda in config")
    parser.add_argument("--lmbda", type=str, choices=['const', 'variance', 'cosine'],
                        help="Override lambda_t schedule in config")
    parser.add_argument("--eta", type=float, help="Override eta multiplier for projection")
    parser.add_argument("--epsilon", type=float,
                        help="Override score epsilon in config")
    parser.add_argument("--lr", type=float,
                        help="Override adjoint_matching.lr in config")
    parser.add_argument("--clip_grad_norm",  type=float,
                        help="Override adjoint_matching.clip_grad_norm in config")
    parser.add_argument("--clip_loss",  type=float,
                        help="Override adjoint_matching.clip_loss in config")
    parser.add_argument("--batch_size", type=int,
                        help="Override adjoint_matching.batch_size in config")
    parser.add_argument("--samples_per_update", type=int,
                        help="Override adjoint_matching.num_samples in config")
    parser.add_argument("--num_integration_steps", type=int,
                        help="Override adjoint_matching.num_integration_steps in config")
    parser.add_argument("--finetune_steps", type=int,
                        help="Override adjoint_matching.finetune_steps in config")
    parser.add_argument("--num_iterations", type=int,
                        help="Override number of iterations")
    parser.add_argument("--num_md_iterations", type=int,
                        help="Override outer loop iterations")
    parser.add_argument("--hamdiv_n", type=int,
                        help="Override n_molecules for hamdiv calculation")
    parser.add_argument("--base_model", type=str,
                        help="Override base model")
    parser.add_argument("--seed", type=int,
                        help="Override seed")
    parser.add_argument('--n_metrics', type=int,
                        help='Override n_molecules for metrics calculation')
    parser.add_argument('--gamma_falloff', type=float, help='Override falloff multiplier for gamma')
    parser.add_argument('--gamma_const', type=float, help='Override constant denominator addition for gamma')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--traj', action='store_true', help='compute trajectory rewards for maximum entropy method')
    parser.add_argument('--no_traj', action='store_true', help='override trajectory rewards')
    parser.add_argument('--beta', type=float, help='weight of KL penalty term')
    parser.add_argument('--gamma', type=float, help='stepsize multiplier')
    parser.add_argument('--constraint', type=str, help='constraint function')
    parser.add_argument('--threshold', type=float, help='lower threshold for validity')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes for pytorch lightning')
    parser.add_argument('--gpus', type=int, default=1, help='Number of gpus per node for pytorch lightning')
    parser.add_argument('--sigma', type=float, help='Sigma for vendi computation')
    parser.add_argument('--dpo_finetune_steps', type=int, help='Finetune steps for dpo')
    parser.add_argument('--dpo_num_iterations', type=int, help='number of iterations for dpo')
    parser.add_argument('--dpo_lr', type=float, help='learning rate for dpo')
    parser.add_argument('--dpo_beta', type=float, help='beta for KL for DPO')

    return parser.parse_args()


def update_config_with_args(config, args):

    max_config = config.max_ent if 'max_ent' in config else config
    am_config = max_config.adjoint_matching

    # Reward arguments
    if args.reward is not None:
        config.reward = {}
        config.reward.fn = args.reward
    # FlowMol arguments
    if args.flow_model is not None:
        config.flowmol.model = args.flow_model
    # Adjoint Matching Parameters
    if args.reward_lambda is not None:
        config.reward_lambda = args.reward_lambda
    if args.lmbda is not None:
        max_config.lmbda = args.lmbda
    if args.gamma_falloff is not None:
        max_config.gamma_falloff = args.gamma_falloff
    if args.gamma_const is not None:
        max_config.gamma_const = args.gamma_const
    if args.eta is not None:
        max_config.eta = args.eta
    if args.constraint is not None:
        config.constraint = {}
        config.constraint.fn = args.constraint
    if args.threshold is not None:
        config.constraint.threshold = args.threshold
    if args.num_md_iterations is not None:
        max_config.num_md_iterations = args.num_md_iterations
    if args.epsilon is not None:
        max_config.epsilon = args.epsilon
    if args.beta is not None:
        max_config.beta = args.beta
    if args.gamma is not None:
        max_config.gamma = args.gamma
    if args.lr is not None:
        am_config.lr = args.lr
    if args.clip_grad_norm is not None:
        am_config.clip_grad_norm = args.clip_grad_norm
    if args.clip_loss is not None:
        am_config.clip_loss = args.clip_loss
    if args.batch_size is not None:
        am_config.batch_size = args.batch_size
    if args.samples_per_update is not None:
        am_config.sampling.num_samples = args.samples_per_update
    if args.num_integration_steps is not None:
        am_config.sampling.num_integration_steps = args.num_integration_steps
    if args.finetune_steps is not None:
        am_config.finetune_steps = args.finetune_steps
    if args.num_iterations is not None:
        am_config.num_iterations = args.num_iterations
    if args.num_md_iterations is not None:
        am_config.num_md_iterations = args.num_md_iterations
    if args.hamdiv_n is not None:
        config.hamdiv_n = args.hamdiv_n
    if args.base_model is not None:
        config.base_model = args.base_model
    if args.seed is not None:
        config.seed = args.seed
    if args.n_metrics is not None:
        config.metrics.n_metrics = args.n_metrics
    if args.traj:
        max_config.traj = True
    if args.sigma is not None:
        config.metrics.sigma = args.sigma
    
    if args.dpo_lr is not None:
        config.dpo.lr = args.dpo_lr
    if args.dpo_num_iterations is not None:
        config.dpo.num_iterations = args.dpo_num_iterations
    if args.dpo_finetune_steps is not None:
        config.dpo.finetune_steps = args.dpo_finetune_steps
    if args.dpo_beta is not None:
        config.dpo.beta = args.dpo_beta
    
    if args.no_traj is not None and args.no_traj:
        max_config.traj = False
    
    config.use_wandb = (args.use_wandb is not None and args.use_wandb) or ('use_wandb' in config and config.use_wandb)

    if 'seed' not in config or config.seed == -1:
        config.seed = random.randint(1, 4096)
    return config



def sig_fn_ddpm(diff_model, t,tm1):
    """sigma_t for DDIM so that it becomes DDPM."""
    at, sig = diff_model.sde.get_alpha_sigma(t)
    atm1, _ = diff_model.sde.get_alpha_sigma(tm1)
    sig_t = torch.sqrt((1-atm1)/(1-at+1e-8)*(1-at/atm1))
    return sig_t


def noise_func(t, sqrt_alphas, sigmas):
    t = (t*999).round().long()
    return sqrt_alphas[t], sigmas[t]

def beta_t(t, beta_0, beta_1):
    """
    Continuous beta(t) = beta_0 + t*(beta_1 - beta_0).
    """
    return beta_0 + t*(beta_1 - beta_0)

# diffusion loss
def forward_noise(x, noise_func, t, eps):
    asqrt, bsqrt = noise_func(t)
    return asqrt*x + bsqrt*eps

def mse_loss(model, x, t, noise_func, eps):
    # sample random ts
    xt = forward_noise(x=x, noise_func=noise_func, t=t[:,None], eps=eps)
    # concatenate
    xt_t = torch.cat([xt, t[:,None]], dim=1)
    # pass through model
    eps_ = model(xt_t)
    # compute loss
    loss = torch.mean((eps_ - eps)**2)/2.
    return loss



def train_model(data_loader, model, optimizer, device=None):
    data_itr = iter(data_loader)
    losses = []
    for step in tqdm(range(20_000)):
        try:
            x_batch = next(data_itr).to(device)
            x_batch = x_batch.to(device)
            t_batch = torch.rand(x_batch.size(0), device=device, dtype=torch.float32)
            eps_batch = torch.randn(x_batch.size(0), 2, device=device, dtype=torch.float32)
            loss = mse_loss(model, x_batch, t_batch, noise_func, eps_batch)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        except StopIteration:
            data_itr = iter(data_loader)
            continue

def get_alpha_sigma(t, sqrt_alphas, sigmas):
    """
    Approximate alpha(t), sigma(t) by indexing into precomputed arrays
    sqrt_alphas and sigmas (each of length=1000) at index round(t*999).
    
    t: scalar in [0,1], or a PyTorch tensor in [0,1].
    Returns alpha_t (scalar), sigma_t (scalar).
    """
    # Ensure t is a PyTorch tensor
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)  # Convert to tensor
    if t.ndim == 0:  # Ensure it's a 1D tensor for operations
        t = t.view(1)
    
    # Compute index (rounding and clamping to [0, 999])
    idx = torch.clamp(torch.round(t * 999), min=0, max=999).long()  # Ensure index is an integer tensor
    
    # Fetch alpha and sigma
    alpha_t = sqrt_alphas[idx].item()**2  # alpha(t), scalar
    sigma_t = sigmas[idx].item()          # sigma(t), scalar
    return alpha_t, sigma_t


def pf_ode(t, x, model):
    """Probability flow ODE."""

    device = x.device
    # Ensure t is a 1D tensor so we can broadcast
    if not torch.is_tensor(t):
        t = torch.tensor([t], dtype=x.dtype, device=device)
    elif t.ndim == 0:
        t = t.view(1)

    # 1) Compute alpha_t, sigma_t by indexing
    alpha_t, sigma_t = get_alpha_sigma(t.item())  # scalar
    # or if you have a batch version, you'd do something vectorized.

    # 2) Evaluate beta(t)
    b_t = beta_t(t)  # shape (1,) if t is shape (1,)

    # 3) Get eps_pred from the model
    #    Typically: model( concat[x, t], ) => shape (batch, dim)
    #    You might do something like:
    t_expand = t.repeat(x.size(0), 1)  # shape (batch, 1)
    inp = torch.cat([x, t_expand], dim=1)  # shape (batch, dim+1)
    eps_pred = model(inp)  # shape (batch, dim)

    # 4) Convert eps_pred => score factor: (+0.5 * beta(t) * eps_pred / sigma_t)
    drift = -0.5 * b_t * x  # shape (batch, dim)
    score_term = 0.5 * b_t * eps_pred / (sigma_t + 1e-12)  # avoid div-by-zero
    dx_dt = drift + score_term

    return dx_dt




"""Divergence estimators."""

def skilling_hutchinson_divergence(x, f, eps=None, dim=[1]):
    """Compute the divergence with Skilling-Hutchinson for f(x)."""
    if eps is None:
        eps = torch.randn_like(x)
    with torch.enable_grad():
      out = torch.sum(f * eps)
      grad_x_f = torch.autograd.grad(out, x, retain_graph=True)[0]
    return torch.sum(grad_x_f * eps, dim=dim)    
  

import numpy as np
def discrete_entropy(counts):
    total = counts.sum()
    pxy = counts / total 
    p_nonzero = pxy[pxy > 0]
    H_bits = -np.sum(p_nonzero * np.log2(p_nonzero))
    H_nats = -np.sum(p_nonzero * np.log(p_nonzero))
    return H_nats






"""DDIM solver"""
def ddim_step(x, t, tm1, model, noise_func, device='cuda'):
    # model prediction
    x = x.to(device)

    t_in = torch.tensor(t, device=device).expand(x.size(0))[:, None]

    eps_pred = model(torch.cat([x, t_in], dim=1))

    avoid_inf = 1e-6

    atm1, btm1 = noise_func(tm1)
    at, bt = noise_func(t)

    x0_pred = (x - bt*eps_pred)/(at + avoid_inf)
    #print(eps_pred.shape, x.shape, x0_pred.shape)
    xtm1 = atm1 * x0_pred + btm1 * eps_pred
    return xtm1

def ode_solver(x0, step_func, ts=None, steps=50,store_traj=False):
    if ts is None:
        ts = torch.linspace(1,0, steps+1)
    if store_traj:
        traj = []
    else:
        traj = None
    xt = x0
    for t, tm1 in zip(ts[:-1], ts[1:]):
        xt = step_func(x=xt, t=t, tm1=tm1)
        if traj is not None:
            traj.append(xt.cpu().detach())
    return xt, {'traj': traj}

def cast_to_half(module: torch.nn.Module):
    """
    Recursively casts all float parameters and buffers in `module` to half precision.
    
    Args:
        module (nn.Module): The PyTorch module whose float parameters should be cast to FP16.
    """
    # First, cast parameters in the current module (no recursion here)
    for param in module.parameters(recurse=False):
        if param.dtype == torch.float32:
            param.data = param.data.half()

    # Cast buffers (e.g., running averages in BatchNorm)
    for name, buffer in module.named_buffers(recurse=False):
        if buffer.dtype == torch.float32:
            module.register_buffer(name, buffer.half())

    # Recursively apply to child modules
    for child in module.children():
        cast_to_half(child)


def recursive_to_device(module: torch.nn.Module, device: torch.device) -> None:
    """
    Recursively casts all parameters, their gradients (if any), and buffers
    within the given module and its submodules to the specified device.
    
    Args:
        module (nn.Module): The PyTorch module whose parameters and buffers
                            will be moved.
        device (torch.device): The target device (e.g., torch.device("cuda"),
                               torch.device("cpu"), etc.).
    """
    
    # Move parameters and their gradients to the specified device
    for param in module.parameters(recurse=False):
        # Move the parameter itself
        param.data = param.data.to(device)
        # If there is a gradient, move it as well
        if param.grad is not None:
            param.grad.data = param.grad.data.to(device)

    # Move all registered buffers (e.g., running averages in BatchNorm, etc.) 
    for key in module._buffers:
        buffer = module._buffers[key]
        if buffer is not None:
            module._buffers[key] = buffer.to(device)
    
    # Recursively apply to submodules
    for child in module.children():
        recursive_to_device(child, device)



