"""
Proximal Policy Optimization（PPO）

核心思想：限制策略更新幅度（clip 比率），防止性能崩溃。

特点：On-policy，训练稳定，工业界广泛使用。

参考：Schulman et al., 2017, "Proximal Policy Optimization Algorithms"

用法：
    python algorithms/deep_rl/ppo.py --env CartPole-v1
"""

import argparse
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        return self.actor(feat), self.critic(feat).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, x: torch.Tensor, action: torch.Tensor):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)

        self.net = PPOActorCritic(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算广义优势估计（GAE）。"""
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        old_lp = torch.FloatTensor(old_log_probs).to(self.device)
        adv = torch.FloatTensor(advantages).to(self.device)
        ret = torch.FloatTensor(returns).to(self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(s)
        for _ in range(self.n_epochs):
            idxs = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                mb = idxs[start: start + self.batch_size]
                new_lp, entropy, value = self.net.evaluate(s[mb], a[mb])
                ratio = torch.exp(new_lp - old_lp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv[mb]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(value, ret[mb])
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()


def train(args) -> List[float]:
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = PPOAgent(obs_dim=obs_dim, n_actions=n_actions, lr=args.lr, gamma=args.gamma)

    rollout_states: List = []
    rollout_actions: List = []
    rollout_log_probs: List = []
    rollout_rewards: List = []
    rollout_dones: List = []
    rollout_values: List = []

    state, _ = env.reset()
    rewards_history: List[float] = []
    ep_reward = 0.0
    ep = 0

    for total_step in range(1, args.total_steps + 1):
        s = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            action, log_prob, _, value = agent.net.get_action_and_value(s)

        rollout_states.append(state)
        rollout_actions.append(int(action.item()))
        rollout_log_probs.append(float(log_prob.item()))
        rollout_values.append(float(value.item()))

        next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
        done = terminated or truncated
        rollout_rewards.append(reward)
        rollout_dones.append(float(done))
        ep_reward += reward
        state = next_state

        if done:
            rewards_history.append(ep_reward)
            ep += 1
            if ep % 50 == 0:
                print(f"Episode {ep:4d} | Avg(50): {np.mean(rewards_history[-50:]):7.2f}")
            ep_reward = 0.0
            state, _ = env.reset()

        if total_step % args.rollout_steps == 0:
            with torch.no_grad():
                s_ = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                _, last_value = agent.net(s_)
                last_value = float(last_value.item())
            advantages, returns = agent.compute_gae(
                np.array(rollout_rewards),
                np.array(rollout_values),
                np.array(rollout_dones),
                last_value,
            )
            agent.update(
                np.array(rollout_states),
                np.array(rollout_actions),
                np.array(rollout_log_probs),
                advantages,
                returns,
            )
            rollout_states.clear()
            rollout_actions.clear()
            rollout_log_probs.clear()
            rollout_rewards.clear()
            rollout_dones.clear()
            rollout_values.clear()

    env.close()
    return rewards_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--total_steps", type=int, default=200000)
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    train(args)
