"""
================================================================================
Comparative Baseline: Direct Preference Optimization (Trajectory-DPO)
Methodology: Implicit reward preference alignment between winner/loser rollouts.
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
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


class DPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, beta=0.1):
        self.action_dim = action_dim
        self.beta = beta
        self.policy = DPOPolicy(state_dim, action_dim, lr=lr)

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

    def train_dpo_step(self, group_trajectories, group_rewards):
        G = len(group_rewards)
        if G < 2:
            return 0.0

        best_idx = int(np.argmax(group_rewards))
        worst_idx = int(np.argmin(group_rewards))

        if group_rewards[best_idx] == group_rewards[worst_idx]:
            return 0.0

        win_states, win_actions, win_masks = group_trajectories[best_idx]
        lose_states, lose_actions, lose_masks = group_trajectories[worst_idx]

        # 1. Winner trajectory cumulative log probability
        win_states_t = torch.tensor(np.array(win_states, dtype=np.float32), device=device)
        win_actions_t = torch.tensor(win_actions, dtype=torch.long, device=device)
        win_masks_t = torch.tensor(np.array(win_masks), dtype=torch.bool, device=device)

        win_logits = self.policy(win_states_t, valid_actions_mask=win_masks_t)
        win_log_probs = F.log_softmax(win_logits, dim=-1).gather(1, win_actions_t.unsqueeze(1)).squeeze(1).sum()

        # 2. Loser trajectory cumulative log probability
        lose_states_t = torch.tensor(np.array(lose_states, dtype=np.float32), device=device)
        lose_actions_t = torch.tensor(lose_actions, dtype=torch.long, device=device)
        lose_masks_t = torch.tensor(np.array(lose_masks), dtype=torch.bool, device=device)

        lose_logits = self.policy(lose_states_t, valid_actions_mask=lose_masks_t)
        lose_log_probs = F.log_softmax(lose_logits, dim=-1).gather(1, lose_actions_t.unsqueeze(1)).squeeze(1).sum()

        # 3. DPO implicit cross-entropy loss: -log sigmoid(beta * (log_p_w - log_p_l))
        loss = -F.logsigmoid(self.beta * (win_log_probs - lose_log_probs))

        self.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy.optimizer.step()
        return loss.item()
