"""
Advantage Actor-Critic（A2C）

同步优势函数演员-评论家。

- Actor：策略网络，输出动作分布
- Critic：价值网络，估计状态价值 V(s)
- 优势：A(s,a) = r + γV(s') - V(s)

用法：
    python algorithms/deep_rl/a2c.py --env CartPole-v1 --episodes 1000
"""

import argparse
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from networks.mlp import MLP


class ActorCritic(nn.Module):
    """共享特征层的 Actor-Critic 网络。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.ReLU())
        self.actor_head = nn.Linear(hidden_size, n_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        feat = self.shared(x)
        logits = self.actor_head(feat)
        value = self.critic_head(feat).squeeze(-1)
        return logits, value


class A2CAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.saved_log_probs: List[torch.Tensor] = []
        self.saved_values: List[torch.Tensor] = []
        self.saved_entropies: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value = self.net(s)
        dist = Categorical(logits=logits)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        self.saved_values.append(value)
        self.saved_entropies.append(dist.entropy())
        return int(action.item())

    def finish_episode(self) -> float:
        G = 0.0
        returns: List[float] = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.FloatTensor(returns).to(self.device)
        values_t = torch.stack(self.saved_values).squeeze()
        log_probs_t = torch.stack(self.saved_log_probs).squeeze()
        entropies_t = torch.stack(self.saved_entropies).squeeze()

        advantages = returns_t - values_t.detach()
        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = nn.functional.mse_loss(values_t, returns_t)
        entropy_loss = -entropies_t.mean()

        loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.saved_log_probs.clear()
        self.saved_values.clear()
        self.saved_entropies.clear()
        self.rewards.clear()
        return loss.item()


def train(args) -> List[float]:
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = A2CAgent(obs_dim=obs_dim, n_actions=n_actions, lr=args.lr, gamma=args.gamma)

    rewards_history: List[float] = []
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.rewards.append(reward)
            total_reward += reward
        agent.finish_episode()
        rewards_history.append(total_reward)
        if ep % 100 == 0:
            print(f"Episode {ep:5d} | Avg(100): {np.mean(rewards_history[-100:]):7.2f}")

    env.close()
    return rewards_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2C")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    train(args)
