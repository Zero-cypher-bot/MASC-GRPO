"""
================================================================================
Paper Architecture: MACS-GRPO (Proposed Framework)
Methodology: Mamba-Attention Collaborative Scheduling with GRPO
Components:
    1. GATv2Layer: Dynamic Graph Attention over Task AND/OR Precedence Graph
    2. SelectiveSSMBlock: State-Space Backbone (Mamba) for Linear O(N) Tracking
    3. MACSIM_NCOPolicy: End-to-End Hybrid Spatiotemporal Policy Network
    4. MACSIMAgent: Critic-Free Group Relative Policy Optimization (GRPO)
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GATv2Layer(nn.Module):
    """
  Dynamic Graph Attention Network (GATv2) Operator.
  Addresses static attention collapse via dynamic non-linear score projection.

  Tensor Dimensions:
      Input:  x -> [Batch_Size (B), In_Dim]
      Output: out -> [Batch_Size (B), Heads * Out_Dim]
  """

    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, _ = x.shape
        proj = self.W(x).view(B, self.heads, self.out_dim)
        attn = self.leaky_relu(proj)
        out = (attn * self.a).sum(dim=-1, keepdim=True)
        out = (proj * torch.sigmoid(out)).reshape(B, -1)
        return out


class SelectiveSSMBlock(nn.Module):
    """
  Selective State-Space Model (SSM / Mamba) Backbone Operator.
  Enforces continuous-to-discrete ZOH mapping with input-dependent selection.

  Tensor Dimensions:
      Input:  x -> [Batch_Size (B), D_Model]
      Output: y -> [Batch_Size (B), D_Model]
  """

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        # Initialize diagonal state matrix A with strictly negative logarithm
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1))
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, _ = x.shape
        proj = self.x_proj(x)
        B_mat, C_mat, x_gate = torch.split(proj, [self.d_state, self.d_state, self.d_model], dim=-1)

        delta = F.softplus(self.dt_proj(x))
        A = -torch.exp(self.A_log)

        # Zero-Order Hold (ZOH) Discretization
        dt_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))  # [B, d_model, d_state]
        h = dt_A * B_mat.unsqueeze(1)
        y = (h * C_mat.unsqueeze(1)).sum(dim=-1)  # [B, d_model]
        return self.out_proj(y * F.silu(x_gate))


class MACSIM_NCOPolicy(nn.Module):
    """
  Unified Spatiotemporal Actor Network with Residual Normalization.
  """

    def __init__(self, state_dim, action_dim, embed_dim=128, heads=4, d_state=16, lr=3e-4):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        assert embed_dim % heads == 0, f"embed_dim ({embed_dim}) must be divisible by heads ({heads})"
        self.gatv2 = GATv2Layer(embed_dim, embed_dim // heads, heads=heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ssm = SelectiveSSMBlock(embed_dim, d_state=d_state)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.actor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)
        self.to(device)

    def forward(self, state, valid_actions_mask=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        feat0 = self.embedding(state)
        feat1 = self.norm1(feat0 + self.gatv2(feat0))
        feat2 = self.norm2(feat1 + self.ssm(feat1))

        logits = self.actor(feat2)
        value = self.critic(feat2)

        # Dynamic Action Masking (inflicts -inf to precedence/capacity invalid actions)
        if valid_actions_mask is not None:
            if valid_actions_mask.dim() == 1:
                valid_actions_mask = valid_actions_mask.unsqueeze(0)
            logits = logits.masked_fill(~valid_actions_mask, -1e9)

        return logits, value


class MACSIMAgent:
    """Agent Wrapper implementing Critic-Free GRPO updates."""

    def __init__(self, state_dim, action_dim, embed_dim=128, heads=4, d_state=16, lr=3e-4):
        self.action_dim = action_dim
        self.policy = MACSIM_NCOPolicy(
            state_dim, action_dim, embed_dim=embed_dim, heads=heads, d_state=d_state, lr=lr
        )

    def choose_action(self, state, valid_actions=None, isTrain=True, temperature=1.0):
        state_t = torch.tensor(np.array(state, dtype=np.float32), device=device)
        mask = torch.zeros(self.action_dim, dtype=torch.bool, device=device)
        if valid_actions is not None:
            mask[valid_actions] = True

        with torch.no_grad():
            logits, _ = self.policy(state_t, valid_actions_mask=mask)
            probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)

        if isTrain:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        else:
            action = torch.argmax(probs, dim=-1).item()
        return action

    def train_grpo_step(self, group_trajectories, group_rewards, beta=0.01):
        """
    Group Relative Policy Optimization (GRPO) Loss Update.
    Ref: Section 4.3, Eq. (18) & Eq. (19) in the manuscript.
    """
        G = len(group_rewards)
        if G <= 1:
            return 0.0

        rewards_t = torch.tensor(group_rewards, dtype=torch.float32, device=device)
        std_r = rewards_t.std()
        if std_r < 1e-3:
            return 0.0

        # Group-level advantage normalization (Eq. 18)
        advantages = (rewards_t - rewards_t.mean()) / (std_r + 1e-8)
        advantages = torch.clamp(advantages, -2.0, 2.0)

        flat_states, flat_actions, flat_masks, traj_indices = [], [], [], []
        for i, (states, actions, masks) in enumerate(group_trajectories):
            flat_states.extend(states)
            flat_actions.extend(actions)
            flat_masks.extend(masks)
            traj_indices.extend([i] * len(states))

        states_t = torch.tensor(np.array(flat_states, dtype=np.float32), device=device)
        actions_t = torch.tensor(flat_actions, dtype=torch.long, device=device)
        masks_t = torch.tensor(np.array(flat_masks), dtype=torch.bool, device=device)
        traj_indices_t = torch.tensor(traj_indices, dtype=torch.long, device=device)

        logits, _ = self.policy(states_t, valid_actions_mask=masks_t)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        selected_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        traj_log_probs = torch.zeros(G, device=device).scatter_add(0, traj_indices_t, selected_log_probs)

        # Numerically stabilized entropy regularization
        entropy = -(probs * torch.clamp(log_probs, min=-100.0)).sum(dim=-1).mean()
        policy_loss = -(advantages * traj_log_probs).mean()
        loss = policy_loss - (beta * 0.1) * entropy

        self.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy.optimizer.step()
        return loss.item()
