"""
SARSA: 在策略时序差分控制（表格型）

与 Q-Learning 的区别：使用实际执行的下一动作更新 Q 值（on-policy）。

用法：
    python algorithms/tabular/sarsa.py --env FrozenLake-v1 --episodes 5000
"""

import argparse
import numpy as np
import gymnasium as gym


def train(env_name: str, episodes: int, alpha: float, gamma: float, epsilon: float) -> np.ndarray:
    """训练 SARSA 智能体，返回 Q 表。"""
    env = gym.make(env_name)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    def choose_action(state: int, eps: float) -> int:
        if np.random.random() < eps:
            return env.action_space.sample()
        return int(np.argmax(Q[state]))

    rewards_per_episode = []
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        action = choose_action(state, epsilon)
        total_reward = 0.0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = choose_action(next_state, epsilon)

            # SARSA 更新：使用实际下一步动作
            td_target = reward + gamma * Q[next_state, next_action] * (not done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state, action = next_state, next_action
            total_reward += reward

        rewards_per_episode.append(total_reward)
        epsilon = max(0.01, epsilon - 1.0 / episodes)

        if ep % 500 == 0:
            avg = np.mean(rewards_per_episode[-500:])
            print(f"Episode {ep:5d} | Avg Reward (last 500): {avg:.3f} | ε: {epsilon:.3f}")

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
    parser = argparse.ArgumentParser(description="SARSA (tabular)")
    parser.add_argument("--env", type=str, default="FrozenLake-v1")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()

    Q = train(args.env, args.episodes, args.alpha, args.gamma, args.epsilon)
    avg_reward = evaluate(args.env, Q)
    print(f"\n评估结果（贪心策略，100 回合）：平均回报 = {avg_reward:.3f}")
