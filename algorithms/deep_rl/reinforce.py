"""
REINFORCE：蒙特卡洛策略梯度

直接对策略参数进行梯度上升，使用完整轨迹的回报作为基准。

参考：Williams, 1992, "Simple Statistical Gradient-Following Algorithms for
Connectionist Reinforcement Learning"

用法：
    python algorithms/deep_rl/reinforce.py --env CartPole-v1 --episodes 1000
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


class REINFORCEAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes=(128, 128),
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.device = torch.device(device)
        self.policy = MLP(obs_dim, n_actions, list(hidden_sizes)).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy(s)
        dist = Categorical(logits=logits)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return int(action.item())

    def finish_episode(self) -> float:
        """在每回合结束后计算回报并更新策略。"""
        G = 0.0
        returns: List[float] = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_tensor = torch.FloatTensor(returns).to(self.device)
        # 标准化以减小方差
        if len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        loss = -torch.stack(self.log_probs).squeeze() * returns_tensor
        total_loss = loss.sum()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()
        return total_loss.item()


def train(args) -> List[float]:
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = REINFORCEAgent(obs_dim=obs_dim, n_actions=n_actions, lr=args.lr, gamma=args.gamma)

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
    parser = argparse.ArgumentParser(description="REINFORCE")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    train(args)
