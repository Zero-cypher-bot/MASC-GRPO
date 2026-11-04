"""
================================================================================
Comparative Baseline: Proximal Policy Optimization (PPO)
Architecture: Actor-Critic with Clipped Importance Sampling
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ActorCriticPPO(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, valid_actions_mask=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.actor(state)
        value = self.critic(state)
        if valid_actions_mask is not None:
            if valid_actions_mask.dim() == 1:
                valid_actions_mask = valid_actions_mask.unsqueeze(0)
            logits = logits.masked_fill(~valid_actions_mask, -1e9)
        return logits, value


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, clip_eps=0.15, ppo_epochs=2):
        self.action_dim = action_dim
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.policy = ActorCriticPPO(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, weight_decay=0.0)

    def choose_action(self, state, valid_actions=None, isTrain=True):
        state_t = torch.tensor(np.array(state, dtype=np.float32), device=device)
        mask = torch.zeros(self.action_dim, dtype=torch.bool, device=device)
        if valid_actions is not None:
            mask[valid_actions] = True

        with torch.no_grad():
            logits, value = self.policy(state_t, valid_actions_mask=mask)
            probs = F.softmax(logits, dim=-1)

        if isTrain:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=device)).item()
        else:
            action = torch.argmax(probs, dim=-1).item()
            log_prob = 0.0
        return action, log_prob, value.item()

    def train_step(self, memory):
        states, actions, old_log_probs, returns, masks = memory
        if len(returns) <= 1:
            return 0.0

        states_t = torch.tensor(np.array(states, dtype=np.float32), device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        masks_t = torch.tensor(np.array(masks), dtype=torch.bool, device=device)

        for _ in range(self.ppo_epochs):
            logits, state_values = self.policy(states_t, valid_actions_mask=masks_t)
            state_values = state_values.squeeze(-1)

            advantages = returns_t - state_values.detach()
            std_a = advantages.std()
            if std_a > 1e-4:
                advantages = (advantages - advantages.mean()) / (std_a + 1e-8)
            advantages = torch.clamp(advantages, -2.0, 2.0)

            log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            ratios = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(state_values, returns_t)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        return loss.item()
