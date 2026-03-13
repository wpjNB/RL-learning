"""
经验回放缓冲区（Replay Buffer）

支持：
- 均匀采样（UniformReplayBuffer）：适用于 DQN 等离策略算法
- 优先经验回放（PrioritizedReplayBuffer）：简化版
"""

import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """均匀采样经验回放缓冲区。"""

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward: float, next_state, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        return len(self.buffer) >= 1


class RolloutBuffer:
    """
    On-policy 轨迹缓冲区（用于 PPO / A2C）。
    存储整个 rollout 期间收集的数据，使用后清空。
    """

    def __init__(self) -> None:
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.dones: list = []
        self.log_probs: list = []
        self.values: list = []

    def push(self, state, action, reward: float, done: bool, log_prob, value) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get(self) -> Tuple:
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
        )

    def clear(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def __len__(self) -> int:
        return len(self.states)
