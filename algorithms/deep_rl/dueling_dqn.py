"""
Dueling DQN

将 Q(s,a) 分解为状态价值 V(s) 与优势函数 A(s,a)：
    Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))

参考：Wang et al., 2016, "Dueling Network Architectures for Deep Reinforcement Learning"

用法：
    python algorithms/deep_rl/dueling_dqn.py --env CartPole-v1 --episodes 500
"""

import argparse
import copy
import random
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


class DuelingQNet(nn.Module):
    """Dueling 网络结构：共享特征层 + 分支 V 流和 A 流。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        V = self.value_stream(feat)
        A = self.advantage_stream(feat)
        return V + (A - A.mean(dim=1, keepdim=True))


class DuelingDQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 200,
        device: str = "cpu",
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.step_count = 0

        self.q_net = DuelingQNet(obs_dim, n_actions).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray) -> int:
        self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.q_net(s).argmax(dim=1).item())

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        s_ = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = self.q_net(s_).argmax(dim=1, keepdim=True)
            next_q = self.target_net(s_).gather(1, best_a).squeeze(1)
            target = r + self.gamma * next_q * (1 - d)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()


def train(args) -> List[float]:
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DuelingDQNAgent(obs_dim=obs_dim, n_actions=n_actions, lr=args.lr, gamma=args.gamma)

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
        if ep % 50 == 0:
            print(f"Episode {ep:4d} | Avg(50): {np.mean(rewards_history[-50:]):7.2f} | ε: {agent.epsilon:.3f}")

    env.close()
    return rewards_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dueling DQN")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    train(args)
