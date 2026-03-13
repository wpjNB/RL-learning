"""
Q-Learning: 离策略时序差分控制（表格型）

适用环境：离散状态 + 离散动作（如 FrozenLake-v1）

用法：
    python algorithms/tabular/q_learning.py --env FrozenLake-v1 --episodes 5000
"""

import argparse
import numpy as np
import gymnasium as gym


def train(env_name: str, episodes: int, alpha: float, gamma: float, epsilon: float) -> np.ndarray:
    """训练 Q-Learning 智能体，返回 Q 表。"""
    env = gym.make(env_name)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    rewards_per_episode = []
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            # ε-贪心策略
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q 值更新
            td_target = reward + gamma * np.max(Q[next_state]) * (not done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        # 线性衰减 ε
        epsilon = max(0.01, epsilon - 1.0 / episodes)

        if ep % 500 == 0:
            avg = np.mean(rewards_per_episode[-500:])
            print(f"Episode {ep:5d} | Avg Reward (last 500): {avg:.3f} | ε: {epsilon:.3f}")

    env.close()
    return Q


def evaluate(env_name: str, Q: np.ndarray, episodes: int = 100) -> float:
    """使用贪心策略评估 Q 表，返回平均回报。"""
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
    parser = argparse.ArgumentParser(description="Q-Learning (tabular)")
    parser.add_argument("--env", type=str, default="FrozenLake-v1")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.1, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--epsilon", type=float, default=1.0, help="初始探索率")
    args = parser.parse_args()

    Q = train(args.env, args.episodes, args.alpha, args.gamma, args.epsilon)
    avg_reward = evaluate(args.env, Q)
    print(f"\n评估结果（贪心策略，100 回合）：平均回报 = {avg_reward:.3f}")
