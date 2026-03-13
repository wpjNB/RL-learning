"""
Deep Deterministic Policy Gradient（DDPG）

适用于连续动作空间控制任务。

特点：
- Off-policy
- 确定性策略 μ(s)
- 使用经验回放和目标网络

参考：Lillicrap et al., 2016, "Continuous control with deep reinforcement learning"

用法：
    python algorithms/deep_rl/ddpg.py --env Pendulum-v1
"""

import argparse
import copy
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from utils.replay_buffer import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_scale: float, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim), nn.Tanh(),
        )
        self.action_scale = action_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.action_scale


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class DDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_std: float = 0.1,
        buffer_size: int = 100000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.action_scale = action_scale
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.device = torch.device(device)
        self.total_steps = 0

        self.actor = Actor(obs_dim, action_dim, action_scale).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray, noise: bool = True) -> np.ndarray:
        self.total_steps += 1
        if self.total_steps < self.warmup_steps:
            return np.random.uniform(-self.action_scale, self.action_scale, size=(1,))
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(s).cpu().numpy()[0]
        if noise:
            action += np.random.normal(0, self.noise_std, size=action.shape)
        return np.clip(action, -self.action_scale, self.action_scale)

    def _soft_update(self, net: nn.Module, target: nn.Module) -> None:
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        s = torch.FloatTensor(states).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_a = self.actor_target(s_)
            target_q = r + self.gamma * self.critic_target(s_, next_a) * (1 - d)

        current_q = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)


def train(args) -> List[float]:
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = DDPGAgent(obs_dim=obs_dim, action_dim=action_dim, action_scale=action_scale,
                      actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=args.gamma)

    rewards_history: List[float] = []
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
        rewards_history.append(total_reward)
        if ep % 20 == 0:
            print(f"Episode {ep:4d} | Avg(20): {np.mean(rewards_history[-20:]):8.2f}")

    env.close()
    return rewards_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPG")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    train(args)
