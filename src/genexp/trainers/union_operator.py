from ..models import DiffusionModel
from .adjoint_matching_trajectory import AdjointMatchingTrajectoryFinetuningTrainer, sample_trajectories_ddpm
import torch
from torch import autograd
import torch.nn as nn
from torch.autograd import grad  
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd

class UnionOperatorTrainer(AdjointMatchingTrajectoryFinetuningTrainer):
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
                 alpha_div=[1.0,1.0],
                 lmbda=1.0,
                 clip_grad_norm=None,
                 running_cost=False,
                 num_traj_MC=15,
                 traj_len=100,
                 critic_steps=100,
                 critic_lr=1e-5):
    
        raise NotImplementedError()
        
        self.lmbda = lmbda
        self.num_traj_MC = num_traj_MC
        self.traj_len = traj_len
        self.saved_grad_reward = grad_reward
        self.critic_steps = critic_steps
        self.critic_lr = critic_lr
        
        if rew_type == 'score-matching':
            print("Using first variation of double KL as reward, lambda:", lmbda)
            if grad_reward is not None:
                grad_reward_fn = lambda x: lmbda * (grad_reward(x) - alpha_div[0] * self.compute_forwardkl_first_variation_grad(x, pre_trained_model_1) - alpha_div[1] * self.compute_forwardkl_first_variation_grad(x, pre_trained_model_2))
            else:
                grad_reward_fn = lambda x: lmbda * (-alpha_div[0] * self.compute_forwardkl_first_variation_grad(x, pre_trained_model_1) - alpha_div[1] * self.compute_forwardkl_first_variation_grad(x, pre_trained_model_2))

            grad_f_k_trajectory = None # no gradient of f_k along trajectory (only used for entropy or KL at last layer)
            self.lmbda = lmbda
        else:
            raise NotImplementedError
        super().__init__(model, grad_reward_fn, grad_f_k_trajectory, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, clip_grad_norm=clip_grad_norm, running_cost=running_cost)


    @torch.no_grad()
    def sample_endpoints_ddpm(self, model, num, traj_len, device):
        z = torch.randn(num, model.data_dim, device=device) if hasattr(model, "data_dim") else torch.randn(num, 2, device=device)
        trajs, _ = sample_trajectories_ddpm(model, z, traj_len)
        return trajs[:, -1, :].to(device)   # [num, d]

    def make_critic(self, dim, hidden=128):
        return nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def compute_forwardkl_first_variation_grad(self, 
                                               x,                          # [B, d] points where to evaluate ∇_x δ/δρ KL(P||ρ)
                                               pre_trained_model,        # P (left argument in KL)
                                               batch_size=256,
                                               exp_clip=20.0,              # clamp inside exp(T-1) for stability
                                               grad_smooth_lambda=1e-4,    # tiny ∥∇_x T∥^2 stabilizer
                                               device=None,
                                            ):

        if device is None:
            device = self.device

        x = x.to(device)
        B, d = x.shape

        # # critic network
        # critic = self.make_critic(d).to(device)
        # opt = torch.optim.Adam(critic.parameters(), lr=self.critic_lr, betas=(0.5, 0.9))

        if not hasattr(self, "_critics"):
            self._critics = {}

        key = id(pre_trained_model)                       # unique id per P-model
        if key not in self._critics:                      # create once
            critic = self.make_critic(d).to(device)
            opt = torch.optim.Adam(
                critic.parameters(), lr=self.critic_lr, betas=(0.5, 0.9))
            self._critics[key] = (critic, opt)
        else:                                             # reuse
            critic, opt = self._critics[key]

        # ---- train critic 
        with torch.enable_grad():
            critic.train()
            # sample (no grad through samplers)
            with torch.no_grad():
                x_P = self.sample_endpoints_ddpm(pre_trained_model, self.num_traj_MC, self.traj_len, device)  # P
                x_R = self.sample_endpoints_ddpm(self.fine_model, self.num_traj_MC, self.traj_len, device)  # ρ
            for _ in range(self.critic_steps):
                # minibatches
                idx_P = torch.randint(0, x_P.shape[0], (batch_size,), device=device)
                idx_R = torch.randint(0, x_R.shape[0], (batch_size,), device=device)
                xb_P = x_P[idx_P].detach().requires_grad_(True)  # for smoothness reg
                xb_R = x_R[idx_R].detach().requires_grad_(True)

                opt.zero_grad(set_to_none=True)
                T_P = critic(xb_P).squeeze(-1)
                T_R = critic(xb_R).squeeze(-1)

                # f-GAN (reverse KL): E_P[T] - E_ρ[exp(T-1)]
                fgan_obj = T_P.mean() - F.softplus(T_R).mean()

                # small smoothness regularizer on both supports (keeps ∇_x T stable)
                gP = autograd.grad(T_P.sum(), xb_P, create_graph=True, retain_graph=True, only_inputs=True)[0]
                gR = autograd.grad(T_R.sum(), xb_R, create_graph=True, retain_graph=True, only_inputs=True)[0]
                reg = 0.5 * grad_smooth_lambda * (gP.pow(2).sum(1).mean() + gR.pow(2).sum(1).mean())

                # --- quick debug ------------------------------------------
                # if _ % 300 == 0:           # every 10 critic steps
                #     print(f"[step {_:3}]  "
                #         f"DV={fgan_obj.item():+.4f}  "
                #         f"T_P={T_P.mean().item():+.2f}  "
                #         f"T_R={T_R.mean().item():+.2f}  "
                #         f"reg={reg.item():.3e}")
                # -----------------------------------------------------------

                loss = -(fgan_obj - reg)  # ascend objective
                loss.backward()
                opt.step()

        # ---- gradient of the first variation w.r.t. ρ at the provided x:  -exp(T-1) ∇_x T
        with torch.enable_grad():
            critic.eval()
            x_eval = x.detach().clone().requires_grad_(True)
            T_x   = critic(x_eval).squeeze(-1)
            gradT = autograd.grad(T_x.sum(), x_eval, create_graph=False, retain_graph=False, only_inputs=True)[0]
            coef = torch.sigmoid(T_x).unsqueeze(-1)
            g_eval =  - coef * gradT


        # ---------- NEW: replace NaN / ±∞ by 0 ----------------------
        g_eval = torch.nan_to_num(g_eval, nan=0.0, posinf=0.0, neginf=0.0)
        # ------------------------------------------------------------

        #debug:
        # finite = torch.isfinite(g_eval).all()
        # mean_nrm = g_eval.norm(dim=1).mean().item()
        # print(f"[g_k] finite={finite}  mean‖g‖={mean_nrm:.2e}")

        return g_eval.detach()



    


    def update_reward(self):
        self.grad_reward_fn = lambda x: self.saved_grad_reward(x) * self.lmbda

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()

    