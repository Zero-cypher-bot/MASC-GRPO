"""
================================================================================
Ablation Baseline: Vanilla GRPO
Architecture: Multi-Layer Perceptron (MLP) Backbone with Group Relative Policy Optimization
Purpose: Validates the architectural necessity of GATv2 and Selective SSM.
Ref: Table IV (Ablation Study) in the manuscript.
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VanillaGRPOPolicy(nn.Module):
    """Standard MLP backbone policy."""
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)
        self.to(device)

    def forward(self, state, valid_actions_mask=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.net(state)

        if valid_actions_mask is not None:
            if valid_actions_mask.dim() == 1:
                valid_actions_mask = valid_actions_mask.unsqueeze(0)
            logits = logits.masked_fill(~valid_actions_mask, -1e9)
        return logits


class VanillaGRPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.action_dim = action_dim
        self.policy = VanillaGRPOPolicy(state_dim, action_dim, lr=lr)

    def choose_action(self, state, valid_actions=None, isTrain=True, temperature=1.0):
        state_t = torch.tensor(np.array(state, dtype=np.float32), device=device)
        mask = torch.zeros(self.action_dim, dtype=torch.bool, device=device)
        if valid_actions is not None:
            mask[valid_actions] = True

        with torch.no_grad():
            logits = self.policy(state_t, valid_actions_mask=mask)
            probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)

        if isTrain:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        else:
            action = torch.argmax(probs, dim=-1).item()
        return action

    def train_grpo_step(self, group_trajectories, group_rewards, beta=0.01):
        G = len(group_rewards)
        if G <= 1:
            return 0.0

        rewards_t = torch.tensor(group_rewards, dtype=torch.float32, device=device)
        std_r = rewards_t.std()
        if std_r < 1e-3:
            return 0.0

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

        logits = self.policy(states_t, valid_actions_mask=masks_t)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        selected_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        traj_log_probs = torch.zeros(G, device=device).scatter_add(0, traj_indices_t, selected_log_probs)

        entropy = -(probs * torch.clamp(log_probs, min=-100.0)).sum(dim=-1).mean()
        policy_loss = -(advantages * traj_log_probs).mean()
        loss = policy_loss - (beta * 0.1) * entropy

        self.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy.optimizer.step()
        return loss.item()
