"""
================================================================================
Comparative Baseline: Kahneman-Tversky Optimization (KTO)
Methodology: Asymmetric prospect-theoretic loss for single-trajectory feedback.
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class KTOPolicy(nn.Module):
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


class KTOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, beta=0.1):
        self.action_dim = action_dim
        self.beta = beta
        self.policy = KTOPolicy(state_dim, action_dim, lr=lr)

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

    def train_kto_step(self, group_trajectories, group_rewards):
        G = len(group_rewards)
        if G <= 1:
            return 0.0

        mean_r = np.mean(group_rewards)
        total_loss = 0.0

        for i, (states, actions, masks) in enumerate(group_trajectories):
            states_t = torch.tensor(np.array(states, dtype=np.float32), device=device)
            actions_t = torch.tensor(actions, dtype=torch.long, device=device)
            masks_t = torch.tensor(np.array(masks), dtype=torch.bool, device=device)

            logits = self.policy(states_t, valid_actions_mask=masks_t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            traj_log_prob = selected_log_probs.sum()

            # Prospect theory asymmetric weighting: Loss aversion
            if group_rewards[i] >= mean_r:
                loss_i = 1.0 - torch.sigmoid(self.beta * traj_log_prob)
            else:
                loss_i = 1.0 - torch.sigmoid(-self.beta * traj_log_prob)
            total_loss += loss_i

        loss = total_loss / G
        self.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy.optimizer.step()
        return loss.item()
