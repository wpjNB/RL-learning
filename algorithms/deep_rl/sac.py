"""
Soft Actor-Critic（SAC）

最大熵框架下的 Off-policy 连续控制算法。

特点：
- 最大化奖励 + 策略熵（增强探索）
- 样本效率高
- 连续动作空间表现优秀

参考：Haarnoja et al., 2018, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with a Stochastic Actor"

用法：
    python algorithms/deep_rl/sac.py --env Pendulum-v1
"""

import argparse
import copy
import math
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from utils.replay_buffer import ReplayBuffer

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SACPolicyNet(nn.Module):
    """高斯策略网络，输出均值与对数标准差（重参数化采样）。"""

    def __init__(self, obs_dim: int, action_dim: int, action_scale: float, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)
        self.action_scale = action_scale

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(x)
        mean = self.mean_layer(feat)
        log_std = torch.clamp(self.log_std_layer(feat), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """重参数化采样，返回 (action, log_prob)。"""
        mean, log_std = self(x)
        std = log_std.exp()
        eps = torch.randn_like(std)
        raw = mean + eps * std
        action = torch.tanh(raw) * self.action_scale
        # 计算对数概率（Tanh 校正）
        log_prob = (
            torch.distributions.Normal(mean, std).log_prob(raw)
            - torch.log(self.action_scale * (1 - torch.tanh(raw).pow(2)) + 1e-6)
        ).sum(dim=-1)
        return action, log_prob


class SACCritic(nn.Module):
    """双 Q 网络（Twin Critic）。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super().__init__()

        def _make_q():
            return nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        self.q1 = _make_q()
        self.q2 = _make_q()

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.total_steps = 0

        self.actor = SACPolicyNet(obs_dim, action_dim, action_scale).to(self.device)
        self.critic = SACCritic(obs_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 自动调整温度系数 α
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha

        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.total_steps += 1
        if self.total_steps < self.warmup_steps:
            return np.random.uniform(
                -self.actor.action_scale, self.actor.action_scale, size=(self.action_dim,)
            )
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if deterministic:
                mean, _ = self.actor(s)
                action = torch.tanh(mean) * self.actor.action_scale
            else:
                action, _ = self.actor.sample(s)
        return action.cpu().numpy()[0]

    def _soft_update(self) -> None:
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
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

        # Critic 更新
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(s_)
            q1_next, q2_next = self.critic_target(s_, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi.unsqueeze(1)
            target_q = r + self.gamma * q_next * (1 - d)

        q1, q2 = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 更新
        new_action, log_pi = self.actor.sample(s)
        q1_pi, q2_pi = self.critic(s, new_action)
        actor_loss = (self.alpha * log_pi.unsqueeze(1) - torch.min(q1_pi, q2_pi)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 自动调整 α
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        self._soft_update()


def train(args) -> List[float]:
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim, action_scale=action_scale,
                     actor_lr=args.lr, critic_lr=args.lr, gamma=args.gamma)

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
            print(f"Episode {ep:4d} | Avg(20): {np.mean(rewards_history[-20:]):8.2f} | α: {agent.alpha:.4f}")

    env.close()
    return rewards_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    train(args)
