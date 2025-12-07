from ..models import DiffusionModel
from .adjoint_matching_trajectory import AdjointMatchingTrajectoryFinetuningTrainer, sample_trajectories_ddpm
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F

class UnionOperatorMixtureTrainer(AdjointMatchingTrajectoryFinetuningTrainer):
    def __init__(self, model: DiffusionModel, 
                 lr,  
                 traj_samples_per_stage, 
                 data_shape, 
                 grad_reward = None,
                 finetune_steps=100, 
                 batch_size=32, 
                 device='cuda',
                 rew_type='score-matching',
                 base_model=None,
                 pre_trained_model_1=None,
                 pre_trained_model_2=None,
                #  pre_trained_model_3=None,
                 alpha_div=[1.0,1.0, 1.0],
                 lmbda=1.0,
                 clip_grad_norm=None,
                 running_cost=False,
                 num_traj_MC=15,
                 traj_len=100,
                 critic_steps=100,
                 critic_lr=1e-5,
                 temp_and_clamp=False):
    
        raise NotImplementedError()
        
        self.lmbda = lmbda
        self.num_traj_MC = num_traj_MC
        self.traj_len = traj_len
        self.saved_grad_reward = grad_reward
        self.critic_steps = critic_steps
        self.critic_lr = critic_lr
        self.device = device
        self.temp_and_clamp = temp_and_clamp

        # cache for critics (single P or mixtures)
        self._critics = {}

        if rew_type == 'score-matching':
            print("Using first variation of KL mixture as reward, lambda:", lmbda)

            # Mixture descriptor: ([P1, P2], [α1, α2]).
            mixture_descriptor = ([pre_trained_model_1, pre_trained_model_2], alpha_div)

            # We want argmax_x  - (α1 KL(P1||ρ) + α2 KL(P2||ρ)).
            # compute_forwardkl_first_variation_grad(mixture) returns (α1+α2) g_mix = α1 g1 + α2 g2,
            # so we subtract it (same as subtracting α1 g1 and α2 g2 separately).
            if grad_reward is not None:
                grad_reward_fn = lambda x: lmbda * (
                    grad_reward(x) - self.compute_forwardkl_first_variation_grad(x, mixture_descriptor)
                )
            else:
                grad_reward_fn = lambda x: lmbda * (
                    - self.compute_forwardkl_first_variation_grad(x, mixture_descriptor)
                )

            grad_f_k_trajectory = None  # only last-layer functional term
        else:
            raise NotImplementedError

        super().__init__(model, grad_reward_fn, grad_f_k_trajectory, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, clip_grad_norm=clip_grad_norm, running_cost=running_cost)

    @torch.no_grad()
    def sample_endpoints_ddpm(self, model, num, traj_len, device):
        z = torch.randn(num, getattr(model, "data_dim", 2), device=device)
        trajs, _ = sample_trajectories_ddpm(model, z, traj_len)
        return trajs[:, -1, :].to(device)   # [num, d]

    @torch.no_grad()
    def sample_endpoints_mixture_ddpm(self, models, weights, num, traj_len, device):
        """
        Draw endpoints from the mixture \bar P = sum_i w_i P_i (weights can be unnormalised).
        """
        w = torch.as_tensor(weights, dtype=torch.float, device=device)
        probs = w / w.sum()
        # assign each of the `num` draws to a component
        comp_idx = torch.multinomial(probs, num_samples=num, replacement=True)  # [num]
        xs = []
        for i, m in enumerate(models):
            c = int((comp_idx == i).sum().item())
            if c > 0:
                xs.append(self.sample_endpoints_ddpm(m, c, traj_len, device))
        if len(xs) == 0:
            d = getattr(models[0], "data_dim", 2)
            return torch.empty(0, d, device=device)
        x = torch.cat(xs, dim=0)
        # shuffle so batches are mixed across components
        perm = torch.randperm(x.size(0), device=device)
        return x[perm]

    def make_critic(self, dim, hidden=128):
        return nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def _critic_key_for_P(self, pre_trained_model):
        """
        Build a persistent-cache key for a single P or a mixture ([P_i], [α_i]).
        """
        # mixture: ( [models], [weights] )
        if (isinstance(pre_trained_model, (list, tuple)) and len(pre_trained_model) == 2
            and isinstance(pre_trained_model[0], (list, tuple)) and isinstance(pre_trained_model[1], (list, tuple))):
            models, alphas = pre_trained_model
            alphas = list(alphas)
            s = float(sum(alphas)) if sum(alphas) != 0 else 1.0
            probs = [float(a)/s for a in alphas]
            return ("mix",
                    tuple(id(m) for m in models),
                    tuple(round(p, 6) for p in probs))
        # single model
        return ("single", id(pre_trained_model))

    def compute_forwardkl_first_variation_grad(self, 
                                               x,                          # [B, d]
                                               pre_trained_model,          # P or mixture: ([P_i],[α_i])
                                               batch_size=256,
                                               exp_clip=20.0,              # (unused with softplus)
                                               grad_smooth_lambda=1e-3, # Uniform: 1e-3, Gaussian: 1e-4
                                               tau=2.0,
                                               T_max=6.0,
                                               device=None):
        """
        Returns the spatial gradient g(x) ≈ ∇_x [δ/δρ KL(P||ρ)] evaluated at ρ = self.fine_model.
        If P is a mixture ([P_i],[α_i]), this returns (sum α_i) * g_mix(x) = sum α_i g_i(x),
        so you can call this ONCE for the whole objective.
        """
        if device is None:
            device = self.device

        x = x.to(device)
        B, d = x.shape

        # Detect mixture vs single P
        is_mixture = (isinstance(pre_trained_model, (list, tuple)) and len(pre_trained_model) == 2
                      and isinstance(pre_trained_model[0], (list, tuple))
                      and isinstance(pre_trained_model[1], (list, tuple)))
        if is_mixture:
            models, alphas = pre_trained_model
            alphas_t = torch.as_tensor(alphas, dtype=torch.float, device=device)
            sum_alpha = float(alphas_t.sum().item()) if alphas_t.sum().item() != 0 else 1.0
        else:
            models, alphas = [pre_trained_model], [1.0]
            sum_alpha = 1.0

        # Persistent critic (+ opt) keyed by the distribution we train against
        key = self._critic_key_for_P(pre_trained_model)
        if key not in self._critics:
            critic = self.make_critic(d).to(device)
            opt = torch.optim.Adam(critic.parameters(), lr=self.critic_lr, betas=(0.5, 0.9))
            self._critics[key] = (critic, opt)
        else:
            critic, opt = self._critics[key]

        # ---- train critic: maximise DV surrogate  E_P[T] - E_ρ[softplus(T)]
        with torch.enable_grad():
            critic.train()
            with torch.no_grad():
                if is_mixture:
                    x_P = self.sample_endpoints_mixture_ddpm(models, alphas, self.num_traj_MC, self.traj_len, device)
                else:
                    x_P = self.sample_endpoints_ddpm(models[0], self.num_traj_MC, self.traj_len, device)
                x_R = self.sample_endpoints_ddpm(self.fine_model, self.num_traj_MC, self.traj_len, device)
            for _ in range(self.critic_steps):
                # minibatches
                idx_P = torch.randint(0, x_P.shape[0], (batch_size,), device=device)
                idx_R = torch.randint(0, x_R.shape[0], (batch_size,), device=device)
                xb_P = x_P[idx_P].detach().requires_grad_(True)
                xb_R = x_R[idx_R].detach().requires_grad_(True)

                opt.zero_grad(set_to_none=True)

                if self.temp_and_clamp:
                    # before loss/regularizer
                    T_P_raw = critic(xb_P).squeeze(-1)
                    T_R_raw = critic(xb_R).squeeze(-1)

                    T_P = T_max * torch.tanh(T_P_raw / T_max)   # soft clamp
                    T_R = T_max * torch.tanh(T_R_raw / T_max)

                    # temperature-softplus surrogate (stable)
                    sp_R = tau * F.softplus(T_R / tau)     # mean over batch below
                    fgan_obj = T_P.mean() - sp_R.mean()
                else:
                    T_P = critic(xb_P).squeeze(-1)
                    T_R = critic(xb_R).squeeze(-1)
                    # DV / f-GAN surrogate for forward KL
                    fgan_obj = T_P.mean() - F.softplus(T_R).mean()
                

                # small smoothness regulariser on ∇_x T over both supports
                gP = autograd.grad(T_P.sum(), xb_P, create_graph=True, retain_graph=True, only_inputs=True)[0]
                gR = autograd.grad(T_R.sum(), xb_R, create_graph=True, retain_graph=True, only_inputs=True)[0]
                reg = 0.5 * grad_smooth_lambda * (gP.pow(2).sum(1).mean() + gR.pow(2).sum(1).mean())

                # --- quick debug ------------------------------------------
                # if _ % 100 == 0:           # every 10 critic steps
                #     print(f"[step {_:3}]  "
                #         f"DV={fgan_obj.item():+.4f}  "
                #         f"T_P={T_P.mean().item():+.2f}  "
                #         f"T_R={T_R.mean().item():+.2f}  "
                #         f"reg={reg.item():.3e}")
                # -----------------------------------------------------------


                loss = -(fgan_obj - reg)  # ascend objective
                loss.backward()
                opt.step()

        # ---- first-variation spatial gradient at query x:
        # using the softplus surrogate: coef = d/dT softplus(T) = sigmoid(T)
        with torch.enable_grad():
            critic.eval()
            x_eval = x.detach().clone().requires_grad_(True)
            if self.temp_and_clamp:
                T_x_raw = critic(x_eval).squeeze(-1)
                T_x = T_max * torch.tanh(T_x_raw / T_max)
            else:
                T_x = critic(x_eval).squeeze(-1)
            gradT = autograd.grad(T_x.sum(), x_eval, create_graph=False, retain_graph=False, only_inputs=True)[0]
            coef  = torch.sigmoid(T_x).unsqueeze(-1)      # approx density-ratio term
            g_eval = - coef * gradT                       # ≈ ∇_x [δ/δρ KL(P||ρ)]

            # SCALE: for a mixture, return (sum α_i) * g_mix = sum α_i g_i
            g_eval = g_eval * sum_alpha

        g_eval = torch.nan_to_num(g_eval, nan=0.0, posinf=0.0, neginf=0.0)
        return g_eval.detach()

    def update_reward(self):
        self.grad_reward_fn = lambda x: self.saved_grad_reward(x) * self.lmbda

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()
