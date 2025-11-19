import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
import numpy as np

# --- Network Models (from your notebook) ---

class ResidualBlock(nn.Module):
    def __init__(self, d: int, use_ln: bool = False, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.ln1 = nn.LayerNorm(d) if use_ln else nn.Identity()
        self.ln2 = nn.LayerNorm(d) if use_ln else nn.Identity()
        self.act = nn.SiLU()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        y = self.act(self.ln1(x))
        y = self.fc1(y)
        y = self.act(self.ln2(y))
        y = self.fc2(y)
        return x + self.alpha * y

class DensityNetwork(nn.Module):
    """ Outputs unnormalized log-density log d_theta(s). """
    def __init__(self, state_dim: int, hidden: int = 256, blocks: int = 5, use_ln: bool = True, box_area: float = 1.0):
        super().__init__()
        self.inp = nn.Linear(state_dim, hidden)
        nn.init.kaiming_normal_(self.inp.weight, nonlinearity='relu')
        nn.init.zeros_(self.inp.bias)

        self.blocks = nn.Sequential(*[ResidualBlock(hidden, use_ln=use_ln, alpha=0.1)
                                      for _ in range(blocks)])

        self.out = nn.Linear(hidden, 1)
        nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        with torch.no_grad():
            self.out.bias.fill_(-math.log(box_area) if box_area > 0.0 else 0.0)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        h = self.blocks(h)
        return self.out(h).squeeze(-1)

    def forward(self, x):
        return self.log_prob(x)

class RFFSampler2D:
    """ Creates the 3-band RFF sampler from your notebook. """
    def __init__(self, 
                 gamma_position=0.15,
                 gamma_velocity=50,
                 lambda_min_pos=0.1,
                 lambda_min_vel=0.004,
                 device=None, 
                 dtype=torch.float32):
        
        self.device = device or torch.device('cpu')
        self.dtype = dtype

        self.sig_pos_base = torch.tensor(2.0 * gamma_position, device=self.device, dtype=self.dtype).sqrt()
        self.sig_vel_base = torch.tensor(2.0 * gamma_velocity, device=self.device, dtype=self.dtype).sqrt()
        self.sig_pos_max = torch.tensor(math.pi / lambda_min_pos, device=self.device, dtype=self.dtype)
        self.sig_vel_max = torch.tensor(math.pi / lambda_min_vel, device=self.device, dtype=self.dtype)
        self.sig_pos_mid = (self.sig_pos_base * self.sig_pos_max).sqrt()
        self.sig_vel_mid = (self.sig_vel_base * self.sig_vel_max).sqrt()
        
        self.omegas = None
        self.biases = None

    @torch.no_grad()
    def sample_batch(self, B: int, device=None, dtype=None):
        device = device or self.device
        dtype = dtype or self.dtype
        B1, B2 = B // 3, B // 3
        B3 = B - B1 - B2

        def band(sig_pos, sig_vel, K):
            op = torch.normal(0.0, sig_pos, (K,), device=device, dtype=dtype)
            ov = torch.normal(0.0, sig_vel, (K,), device=device, dtype=dtype)
            return torch.stack([op, ov], dim=1)

        O1 = band(self.sig_pos_base, self.sig_vel_base, B1)
        O2 = band(self.sig_pos_mid, self.sig_vel_mid, B2)
        O3 = band(self.sig_pos_max, self.sig_vel_max, B3)

        self.omegas = torch.cat([O1, O2, O3], dim=0)
        self.biases = 2 * torch.pi * torch.rand(B, device=device, dtype=dtype)
        return self.omegas, self.biases
    
    @torch.no_grad()
    def features(self, S: torch.Tensor, omegas: torch.Tensor=None, biases: torch.Tensor=None) -> torch.Tensor:
        if omegas is None or biases is None:
            if self.omegas is None or self.biases is None:
                raise ValueError("Call sample_batch(B) first or pass (omegas, biases).")
            omegas, biases = self.omegas, self.biases
        
        proj = S @ omegas.T
        return torch.cos(proj + biases)

# --- Main Estimator Class ---

class DensityEstimator:
    def __init__(self, policy_net, env_helpers, hidden_size, lr, device):
        self.device = device
        
        # Store a *reference* to the SAC policy network
        # This policy_net will be updated by the main SAC algorithm
        self.policy_net = policy_net 
        
        # Store the env helper functions and constants
        self.env_helpers = env_helpers
        self.log_m_s = self.env_helpers.LOG_M_S

        # Initialize Density Net
        self.density_net = DensityNetwork(
            state_dim=2, 
            hidden=hidden_size, 
            box_area=self.env_helpers.BOX_AREA
        ).to(self.device)
        self.density_optim = Adam(self.density_net.parameters(), lr=lr)

        # Initialize RFF Sampler
        self.rff_sampler = RFFSampler2D(device=self.device)

    @torch.no_grad()
    def _policy_probs(self, states: torch.Tensor) -> torch.Tensor:
        """ Gets action probabilities from the *current* SAC policy. """
        logits, probs, _ = self.policy_net.forward(states)
        return probs

    @torch.no_grad()
    def _sample_policy_actions(self, states: torch.Tensor) -> torch.Tensor:
        """ Samples actions from the *current* SAC policy. """
        # We use logits to ensure it's sampling from the most up-to-date policy
        logits, _, _ = self.policy_net.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

    @torch.no_grad()
    def _rff_expect_next_feature(self, states, probs, omega, b):
        """ Computes E_a[f(s^+)] under the *current* SAC policy. """
        N = states.size(0)
        a0 = torch.zeros(N, dtype=torch.long, device=self.device)
        a1 = torch.ones(N,  dtype=torch.long, device=self.device)
        
        sp0 = self.env_helpers.next_state(states, a0)
        sp1 = self.env_helpers.next_state(states, a1)
        
        F_sp0 = self.rff_sampler.features(sp0, omega, b)
        F_sp1 = self.rff_sampler.features(sp1, omega, b)
        
        return probs[:, 0:1] * F_sp0 + probs[:, 1:2] * F_sp1

    def _estimate_R_and_grad(self, N, gamma, omega, b):
        """Return Monte-Carlo estimates of R_f and grad R_f using independent batches."""
        B = omega.size(0)

        # ----- Batch S for \hat R_f (Detached) -----
        S = self.env_helpers.sample_proposal(N, device=self.device)
        S0 = self.env_helpers.sample_start(N, device=self.device)
        pS = self._policy_probs(S) # Use current SAC policy

        with torch.no_grad():
            F_S = self.rff_sampler.features(S, omega, b)
            EF_Sp = self._rff_expect_next_feature(S, pS, omega, b)
            F_S0 = self.rff_sampler.features(S0, omega, b)
        
            log_d_s = self.density_net.log_prob(S)
            w_S = torch.exp(log_d_s - self.log_m_s)
            
            R_all = (w_S.unsqueeze(1) * (F_S - gamma * EF_Sp)).mean(dim=0) \
                    - (1.0 - gamma) * F_S0.mean(dim=0)
            R_all_detached = R_all.detach()

        # ----- Independent batch \tilde S for \hat \nabla_θ R_f (With Grad) -----
        St = self.env_helpers.sample_proposal(N, device=self.device)
        pSt = self._policy_probs(St) # Use current SAC policy

        with torch.no_grad():
            F_St = self.rff_sampler.features(St, omega, b)
            EF_Stp = self._rff_expect_next_feature(St, pSt, omega, b)
            
            A_f = (F_St - gamma * EF_Stp).detach() # (N, B)

        # This part requires grad w.r.t. density_net
        log_d_st = self.density_net.log_prob(St) # (N,)
        wt = torch.exp(log_d_st - self.log_m_s) # (N,)
        
        # This is the "grad_all" part that requires gradients
        # This correctly computes E_m[ A_f * w(s) ]
        # Its gradient is E_m[ A_f * w(s) * grad(log_d) ], which is grad(R_f)
        grad_R_all_with_grad = (A_f * wt.unsqueeze(1)).mean(dim=0) # (B,)

        return R_all_detached, grad_R_all_with_grad

    def update(self, N: int, B: int, gamma: float):
        """Perform one unbiased weak-form residual update step."""
        
        # 1. Sample a fresh batch of test functions
        omega, b = self.rff_sampler.sample_batch(B, device=self.device, dtype=torch.float32)

        # 2. Estimate R (detached) and Grad(R) (with graph)
        R_all_detached, grad_R_all_with_grad = self._estimate_R_and_grad(
            N=N, gamma=gamma, omega=omega, b=b
        )
        
        # 3. Compute the final U-gradient loss following the weak-form residual rule
        # loss = 2.0 * E_f[ R_f * grad(R_f) ]
        loss = 2.0 * torch.mean(R_all_detached * grad_R_all_with_grad)

        # 4. Perform gradient step
        self.density_optim.zero_grad()
        loss.backward()
        self.density_optim.step()

        return loss.item()

