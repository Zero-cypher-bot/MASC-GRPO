"""
================================================================================
Benchmark Pipeline: Multi-Algorithm Training & Convergence Plotting
Description: Trains 6 DRL models across multiple random seeds, logs training
             histories, and exports publication-grade IEEE convergence curves.
Ref: Section 5.3 (Table III & Fig. 4) in the manuscript.
================================================================================
"""

import os
import time
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from env_final import DLBP_FinalEnv
from NCO_GRPO import MACSIMAgent
from NCO_Vanilla_GRPO import VanillaGRPOAgent
from NCO_RLOO import RLOOAgent
from NCO_DPO import DPOAgent
from NCO_KTO import KTOAgent
from NCO_PPO import PPOAgent

# IEEE Publication Figure Standards
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed=42):
    """Hardware deterministic seed setting."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def smooth(data, window_size=20):
    """Causal moving average smoothing."""
    data = np.array(data, dtype=np.float32)
    if len(data) <= window_size:
        return data
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        if i < window_size:
            smoothed[i] = np.mean(data[:i + 1])
        else:
            smoothed[i] = np.mean(data[i - window_size + 1:i + 1])
    return smoothed


def update_cosine_lr(optimizer, episode, max_episodes, initial_lr=3e-4, min_lr=1e-5):
    """Cosine learning rate decay schedule."""
    progress = min(1.0, float(episode) / float(max_episodes))
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    current_lr = min_lr + (initial_lr - min_lr) * cos_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr


def plot_all_results(stats_dict, max_episodes=500, save_name='Algorithm_Training_Convergence_Comparison'):
    """Plot multi-algorithm convergence with variance shading bands."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=300)

    colors = {
        'MACS-GRPO':    '#0052cc',  # Deep Blue (Proposed)
        'GRPO':           '#4192d9',  # Steel Blue (Ablation)
        'RLOO':           '#ff7f0e',  # Vibrant Orange
        'Trajectory-DPO': '#2ca02c',  # Green
        'KTO':            '#9467bd',  # Purple
        'PPO':            '#d62728'   # Crimson Red
    }

    episodes = np.arange(1, max_episodes + 1)
    for algo_name, (mean_curve, std_curve) in stats_dict.items():
        color = colors.get(algo_name, '#333333')
        ax.plot(episodes, mean_curve, label=algo_name, color=color, linewidth=1.8, zorder=3)
        ax.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.12, zorder=2)

    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Total Net Profit ($F$)')
    ax.set_xlim(1, max_episodes)
    ax.grid(True, linestyle='--', alpha=0.4, color='#bbbbbb', zorder=1)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.92, ncol=2)

    plt.tight_layout()
    plt.savefig(f"{save_name}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_name}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"\n[Figure Export] Convergence plot successfully generated: {save_name}.pdf / .png")
    plt.close()


def run_grpo(env, max_episodes, group_size=12):
    agent = MACSIMAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=3e-4)
    rewards_history = []
    global_best_reward = -9999.0

    for episode in range(max_episodes):
        update_cosine_lr(agent.policy.optimizer, episode, max_episodes)
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * episode / max_episodes))
        temperature = 0.15 + 0.85 * cos_decay
        beta = max(0.0005, 0.005 * cos_decay)

        group_trajectories, group_rewards = [], []
        for _ in range(group_size):
            obs = env.reset()
            done = False
            states, actions, masks = [], [], []
            while not done:
                valid_actions = env.get_valid_actions()
                mask = np.zeros(env.action_dim, dtype=bool)
                mask[valid_actions] = True
                action = agent.choose_action(obs, valid_actions=valid_actions, isTrain=True, temperature=temperature)
                states.append(obs)
                actions.append(action)
                masks.append(mask)
                obs, _, done, _ = env.step(action)
            group_trajectories.append((states, actions, masks))
            group_rewards.append(env.total_profit)

        agent.train_grpo_step(group_trajectories, group_rewards, beta=beta)
        rewards_history.append(np.mean(group_rewards))

        if max(group_rewards) > global_best_reward:
            global_best_reward = max(group_rewards)
            torch.save(agent.policy.state_dict(), './checkpoints/grpo_best_model1.pth')
    return rewards_history


def run_vanilla_grpo(env, max_episodes, group_size=12):
    agent = VanillaGRPOAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=3e-4)
    rewards_history = []
    global_best_reward = -9999.0

    for episode in range(max_episodes):
        update_cosine_lr(agent.policy.optimizer, episode, max_episodes)
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * episode / max_episodes))
        temperature = 0.15 + 0.85 * cos_decay
        beta = max(0.0005, 0.005 * cos_decay)

        group_trajectories, group_rewards = [], []
        for _ in range(group_size):
            obs = env.reset()
            done = False
            states, actions, masks = [], [], []
            while not done:
                valid_actions = env.get_valid_actions()
                mask = np.zeros(env.action_dim, dtype=bool)
                mask[valid_actions] = True
                action = agent.choose_action(obs, valid_actions=valid_actions, isTrain=True, temperature=temperature)
                states.append(obs)
                actions.append(action)
                masks.append(mask)
                obs, _, done, _ = env.step(action)
            group_trajectories.append((states, actions, masks))
            group_rewards.append(env.total_profit)

        agent.train_grpo_step(group_trajectories, group_rewards, beta=beta)
        rewards_history.append(np.mean(group_rewards))

        # Peak checkpoint tracking (Fixes vulnerability 3)
        if max(group_rewards) > global_best_reward:
            global_best_reward = max(group_rewards)
            torch.save(agent.policy.state_dict(), './checkpoints/vanilla_grpo_best_model.pth')
    return rewards_history


def run_dpo(env, max_episodes, group_size=8):
    agent = DPOAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=3e-4)
    rewards_history = []
    global_best_reward = -9999.0

    for episode in range(max_episodes):
        update_cosine_lr(agent.policy.optimizer, episode, max_episodes)
        temperature = max(0.2, 1.0 - (episode / (max_episodes * 0.7)))
        group_trajectories, group_rewards = [], []
        for _ in range(group_size):
            obs = env.reset()
            done = False
            states, actions, masks = [], [], []
            while not done:
                valid_actions = env.get_valid_actions()
                mask = np.zeros(env.action_dim, dtype=bool)
                mask[valid_actions] = True
                action = agent.choose_action(obs, valid_actions=valid_actions, isTrain=True, temperature=temperature)
                states.append(obs)
                actions.append(action)
                masks.append(mask)
                obs, _, done, _ = env.step(action)
            group_trajectories.append((states, actions, masks))
            group_rewards.append(env.total_profit)

        agent.train_dpo_step(group_trajectories, group_rewards)
        rewards_history.append(np.mean(group_rewards))

        if max(group_rewards) > global_best_reward:
            global_best_reward = max(group_rewards)
            torch.save(agent.policy.state_dict(), './checkpoints/dpo_best_model.pth')
    return rewards_history


def run_kto(env, max_episodes, group_size=8):
    agent = KTOAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=3e-4)
    rewards_history = []
    global_best_reward = -9999.0

    for episode in range(max_episodes):
        update_cosine_lr(agent.policy.optimizer, episode, max_episodes)
        temperature = max(0.2, 1.0 - (episode / (max_episodes * 0.7)))
        group_trajectories, group_rewards = [], []
        for _ in range(group_size):
            obs = env.reset()
            done = False
            states, actions, masks = [], [], []
            while not done:
                valid_actions = env.get_valid_actions()
                mask = np.zeros(env.action_dim, dtype=bool)
                mask[valid_actions] = True
                action = agent.choose_action(obs, valid_actions=valid_actions, isTrain=True, temperature=temperature)
                states.append(obs)
                actions.append(action)
                masks.append(mask)
                obs, _, done, _ = env.step(action)
            group_trajectories.append((states, actions, masks))
            group_rewards.append(env.total_profit)

        agent.train_kto_step(group_trajectories, group_rewards)
        rewards_history.append(np.mean(group_rewards))

        if max(group_rewards) > global_best_reward:
            global_best_reward = max(group_rewards)
            torch.save(agent.policy.state_dict(), './checkpoints/kto_best_model.pth')
    return rewards_history


def run_rloo(env, max_episodes, group_size=8):
    agent = RLOOAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=3e-4)
    rewards_history = []
    global_best_reward = -9999.0

    for episode in range(max_episodes):
        update_cosine_lr(agent.policy.optimizer, episode, max_episodes)
        temperature = max(0.2, 1.0 - (episode / (max_episodes * 0.7)))
        group_trajectories, group_rewards = [], []
        for _ in range(group_size):
            obs = env.reset()
            done = False
            states, actions, masks = [], [], []
            while not done:
                valid_actions = env.get_valid_actions()
                mask = np.zeros(env.action_dim, dtype=bool)
                mask[valid_actions] = True
                action = agent.choose_action(obs, valid_actions=valid_actions, isTrain=True, temperature=temperature)
                states.append(obs)
                actions.append(action)
                masks.append(mask)
                obs, _, done, _ = env.step(action)
            group_trajectories.append((states, actions, masks))
            group_rewards.append(env.total_profit)

        agent.train_rloo_step(group_trajectories, group_rewards)
        rewards_history.append(np.mean(group_rewards))

        if max(group_rewards) > global_best_reward:
            global_best_reward = max(group_rewards)
            torch.save(agent.policy.state_dict(), './checkpoints/rloo_best_model.pth')
    return rewards_history


def run_ppo(env, max_episodes, group_size=12):
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=3e-4, clip_eps=0.15, ppo_epochs=2)
    rewards_history = []
    global_best_reward = -9999.0

    for episode in range(max_episodes):
        update_cosine_lr(agent.optimizer, episode, max_episodes)
        batch_states, batch_actions, batch_log_probs = [], [], []
        batch_returns, batch_masks = [], []
        group_rewards = []

        for _ in range(group_size):
            obs = env.reset()
            done = False
            traj_states, traj_actions, traj_log_probs, traj_masks = [], [], [], []
            while not done:
                valid_actions = env.get_valid_actions()
                mask = np.zeros(env.action_dim, dtype=bool)
                mask[valid_actions] = True
                action, log_prob, _ = agent.choose_action(obs, valid_actions=valid_actions, isTrain=True)
                traj_states.append(obs)
                traj_actions.append(action)
                traj_log_probs.append(log_prob)
                traj_masks.append(mask)
                obs, _, done, _ = env.step(action)

            T = len(traj_states)
            traj_returns = [float(env.total_profit)] * T
            group_rewards.append(env.total_profit)
            batch_states.extend(traj_states)
            batch_actions.extend(traj_actions)
            batch_log_probs.extend(traj_log_probs)
            batch_masks.extend(traj_masks)
            batch_returns.extend(traj_returns)

        memory = (batch_states, batch_actions, batch_log_probs, batch_returns, batch_masks)
        agent.train_step(memory)
        rewards_history.append(np.mean(group_rewards))

        if max(group_rewards) > global_best_reward:
            global_best_reward = max(group_rewards)
            torch.save(agent.policy.state_dict(), './checkpoints/ppo_best_model.pth')
    return rewards_history


if __name__ == '__main__':
    os.makedirs('./checkpoints/', exist_ok=True)
    os.makedirs('./results_log/', exist_ok=True)

    case_id = 'case10'
    max_episodes = 200
    env = DLBP_FinalEnv(case_id=case_id)
    seeds = [42]

    algorithms = [
        ('MACS-GRPO', run_grpo),
        ('GRPO', run_vanilla_grpo),
        ('RLOO', run_rloo),
        ('Trajectory-DPO', run_dpo),
        ('KTO', run_kto),
        ('PPO', run_ppo)
    ]

    stats_results = {}
    print("=" * 80)
    print(f"[Benchmark Training] Evaluating 6 Algorithms on Case {case_id} across {len(seeds)} Seeds")
    print("=" * 80)

    for name, train_func in algorithms:
        print(f"\n>> Executing Training Pipeline: [{name}] ...")
        all_seed_runs = []
        for s_idx, seed in enumerate(seeds):
            set_seed(seed)
            t_start = time.time()
            raw_history = train_func(env, max_episodes)
            smoothed_run = smooth(raw_history, window_size=25)
            all_seed_runs.append(smoothed_run)
            np.savetxt(f"./results_log/{name}_seed_{seed}.txt", raw_history, fmt='%.2f')
            print(f"   Seed {seed} finished in {(time.time() - t_start) / 60:.2f} min.")

        all_seed_runs = np.array(all_seed_runs)
        mean_curve = np.mean(all_seed_runs, axis=0)
        std_curve = np.std(all_seed_runs, axis=0)
        stats_results[name] = (mean_curve, std_curve)

        np.savetxt(f"./results_log/{name}_mean.txt", mean_curve, fmt='%.2f')
        np.savetxt(f"./results_log/{name}_std.txt", std_curve, fmt='%.2f')

    plot_all_results(stats_results, max_episodes=max_episodes)
