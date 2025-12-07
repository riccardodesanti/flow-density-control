from ..models import DiffusionModel
from .adjoint_matching import AdjointMatchingFinetuningTrainer
import torch
from genexp.trainers.adjoint_matching import AdjointMatchingFinetuningTrainer, sample_trajectories_ddpm
from scipy.optimize import minimize
from torch.autograd import grad  
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd

class Wasserstein1Trainer(AdjointMatchingFinetuningTrainer):
    def __init__(self, model: DiffusionModel,
                 grad_reward, 
                 A_inv, # inverse of the metric tensor
                 lr, 
                 traj_samples_per_stage, 
                 data_shape, 
                 finetune_steps=100, 
                 batch_size=32, 
                 device='cuda',
                 rew_type='score-matching',
                 base_model=None,
                 pre_trained_model=None,
                 alpha_div=1.0,
                 num_traj_MC=15,
                 traj_len=100,
                 lmbda=1.0,
                 clip_grad_norm=None,
                 critic_steps=100,
                 gp_lambda=5.0,
                 critic_lr=1e-5
                 ):
        
        raise NotImplementedError()
        
        self.lmbda = lmbda
        self.num_traj_MC = num_traj_MC
        self.traj_len = traj_len
        self.A_inv = A_inv 
        self.saved_grad_reward = grad_reward
        self.critic_steps = critic_steps
        self.gp_lambda = gp_lambda
        self.critic_lr = critic_lr
        if rew_type == 'score-matching':
            grad_reward_fn = lambda x: lmbda * (grad_reward(x) - alpha_div * self.compute_wasserstein1_grad(x, pre_trained_model))
        else:
            raise NotImplementedError
        super().__init__(model, grad_reward_fn, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, clip_grad_norm=clip_grad_norm)
    

    def compute_wasserstein1_grad(self, x, pre_trained_model):
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True) 
        with torch.enable_grad():
            f_star_x = self.compute_wasserstein1_first_var(x, pre_trained_model, self.critic_steps, self.gp_lambda, self.critic_lr)                       
            w1_grad =  torch.autograd.grad(f_star_x.sum(), x, retain_graph=True)[0]
        return w1_grad



    def compute_wasserstein1_first_var(self, x, pre_trained_model, critic_steps=500, gp_lambda=5.0, critic_lr=1e-5):
        """
        Trains/updates the 1-Lipschitz critic on *its own* fresh minibatch
        drawn from the current policy (self.model) and from p_pre.
        Finally returns f★(x) for the external points x.
        """
        device = x.device

        # draw samples from the current fine-tuned model
        with torch.no_grad():
            z_cur = torch.randn(self.num_traj_MC, 2, device=device)
            traj_cur = sample_trajectories_ddpm(self.fine_model, z_cur, self.traj_len)[0]                                
            x_train = traj_cur[:, -1, :].to(device)        

        # draw samples from the reference model p_pre
        with torch.no_grad():
            z_ref = torch.randn(self.num_traj_MC, 2, device=device)
            traj_ref = sample_trajectories_ddpm(pre_trained_model, z_ref, self.traj_len)[0]
            y_train = traj_ref[:, -1, :].to(device)       

        batch_size = x_train.size(0)           

        # build / reuse critic
        if not hasattr(self, "critic"):
            self.critic = self.make_critic(input_dim=2).to(device)
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=(0.5, 0.9))

        # WGAN-GP inner loop 
        with torch.enable_grad():
            for iter in range(critic_steps):
                self.critic_opt.zero_grad(set_to_none=True)

                f_x  = self.critic(x_train).mean()
                f_y  = self.critic(y_train).mean()
                loss = -(f_x - f_y)             

                # gradient penalty to enforce Lipschitz constraint
                eps    = torch.rand(batch_size, 1, device=device)
                interp = eps * x_train + (1 - eps) * y_train
                interp.requires_grad_(True)
                f_int  = self.critic(interp)
                grad   = autograd.grad(f_int, interp, torch.ones_like(f_int), create_graph=True, retain_graph=True, only_inputs=True)[0]

                # un-weighted gradient penalty (option A)
                # gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()  

                # weighted gradient penalty (option B)
                grad_trans = grad @ self.A_inv
                gp = ((grad_trans.norm(2, dim=1) - 1) ** 2).mean()
                (loss + gp_lambda * gp).backward()
                # print(f"loss: {loss.item()}, gp: {gp.item()}")

                # EVAL debugging block
                # dual_val = (f_x - f_y).item()
                # gp_val = gp.item()
                # if iter % 100 == 0:
                #     print(f"Step {iter}: dual={dual_val:.4f}, gp={gp_val:.4f}") 

                self.critic_opt.step()

        f_star_x = self.critic(x)[:, 0]         
        return f_star_x
    

    # small 1-Lipschitz critic (MLP + spectral-norm)
    # def make_critic(self, input_dim=2, hidden=128):
    #     layers = [
    #         nn.utils.parametrizations.spectral_norm(nn.Linear(input_dim, hidden)),
    #         nn.ReLU(),
    #         nn.utils.parametrizations.spectral_norm(nn.Linear(hidden, 1)),
    #     ]
    #     return nn.Sequential(*layers)

    def make_critic(self, input_dim=2, hidden=128):
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(self.device)
    

    def update_reward(self):
       self.grad_reward_fn = lambda x: self.saved_grad_reward(x) * self.lmbda

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()
