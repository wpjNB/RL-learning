"""
Monte Carlo 控制：蒙特卡洛策略评估与控制（表格型）

使用首次访问（First-Visit）MC 估计 Q(s,a)，配合 ε-贪心改善策略。

用法：
    python algorithms/tabular/monte_carlo.py --env FrozenLake-v1 --episodes 10000
"""

import argparse
from collections import defaultdict
import numpy as np
import gymnasium as gym


def generate_episode(env: gym.Env, Q: np.ndarray, epsilon: float) -> list:
    """使用 ε-贪心策略生成一条完整轨迹。"""
    episode = []
    state, _ = env.reset()
    done = False
    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated
    return episode


def train(env_name: str, episodes: int, gamma: float, epsilon: float) -> np.ndarray:
    """Monte Carlo 首次访问控制，返回 Q 表。"""
    env = gym.make(env_name)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    returns: dict = defaultdict(list)

    rewards_per_episode = []
    for ep in range(1, episodes + 1):
        episode = generate_episode(env, Q, epsilon)
        total_reward = sum(r for _, _, r in episode)
        rewards_per_episode.append(total_reward)

        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[state, action] = np.mean(returns[(state, action)])

        epsilon = max(0.01, epsilon - 1.0 / episodes)

        if ep % 1000 == 0:
            avg = np.mean(rewards_per_episode[-1000:])
            print(f"Episode {ep:6d} | Avg Reward (last 1000): {avg:.3f} | ε: {epsilon:.3f}")

    env.close()
    return Q


def evaluate(env_name: str, Q: np.ndarray, episodes: int = 100) -> float:
    env = gym.make(env_name)
    total = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
    env.close()
    return total / episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Control (tabular)")
    parser.add_argument("--env", type=str, default="FrozenLake-v1")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()

    Q = train(args.env, args.episodes, args.gamma, args.epsilon)
    avg_reward = evaluate(args.env, Q)
    print(f"\n评估结果（贪心策略，100 回合）：平均回报 = {avg_reward:.3f}")
