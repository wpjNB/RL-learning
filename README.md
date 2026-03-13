# RL-learning

> 强化学习（Reinforcement Learning）算法学习与实践项目

## 项目简介

本项目旨在系统性地学习和实现经典及前沿强化学习算法，从基础的表格型方法到深度强化学习，涵盖理论推导与代码实践。项目代码以 Python 为主，配合 OpenAI Gymnasium 等主流环境进行实验验证。

---

## 目录

- [算法覆盖](#算法覆盖)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [实验结果](#实验结果)
- [参考资料](#参考资料)

---

## 算法覆盖

### 基础方法（Model-Free）

| 算法 | 类别 | 说明 |
|------|------|------|
| Q-Learning | 值函数方法 | 离策略时序差分控制 |
| SARSA | 值函数方法 | 在策略时序差分控制 |
| Monte Carlo | 值函数方法 | 蒙特卡洛策略评估与控制 |

### 深度强化学习（Deep RL）

| 算法 | 类别 | 说明 |
|------|------|------|
| DQN | 值函数方法 | Deep Q-Network，使用经验回放与目标网络 |
| Double DQN | 值函数方法 | 解决 Q 值高估问题 |
| Dueling DQN | 值函数方法 | 分离状态价值与优势函数 |
| REINFORCE | 策略梯度 | 蒙特卡洛策略梯度 |
| Actor-Critic (A2C) | Actor-Critic | 同步优势函数演员-评论家 |
| PPO | Actor-Critic | 近端策略优化 |
| DDPG | Actor-Critic | 深度确定性策略梯度（连续动作空间） |
| SAC | Actor-Critic | 软演员-评论家（最大熵框架） |

---

## 项目结构

```
RL-learning/
├── README.md                  # 项目文档
├── requirements.txt           # Python 依赖
├── envs/                      # 自定义环境
├── algorithms/                # 算法实现
│   ├── tabular/               # 表格型方法
│   │   ├── q_learning.py
│   │   ├── sarsa.py
│   │   └── monte_carlo.py
│   └── deep_rl/               # 深度强化学习
│       ├── dqn.py
│       ├── double_dqn.py
│       ├── dueling_dqn.py
│       ├── reinforce.py
│       ├── a2c.py
│       ├── ppo.py
│       ├── ddpg.py
│       └── sac.py
├── networks/                  # 神经网络模块
│   ├── mlp.py
│   └── cnn.py
├── utils/                     # 工具函数
│   ├── replay_buffer.py
│   ├── logger.py
│   └── plot.py
└── experiments/               # 实验脚本与结果
    ├── configs/               # 超参数配置
    └── results/               # 训练曲线与模型
```

---

## 环境依赖

- Python >= 3.9
- PyTorch >= 2.0
- gymnasium >= 0.29
- numpy
- matplotlib

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/wpjNB/RL-learning.git
cd RL-learning
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行示例（Q-Learning on FrozenLake）

```bash
python algorithms/tabular/q_learning.py --env FrozenLake-v1 --episodes 5000
```

### 4. 运行深度强化学习示例（DQN on CartPole）

```bash
python algorithms/deep_rl/dqn.py --env CartPole-v1 --episodes 500
```

---

## 实验结果

各算法在标准基准环境上的实验结果将持续更新，训练曲线保存在 `experiments/results/` 目录下。

| 算法 | 环境 | 平均回报（最近100回合） |
|------|------|------------------------|
| Q-Learning | FrozenLake-v1 | - |
| DQN | CartPole-v1 | - |
| PPO | LunarLander-v2 | - |
| SAC | HalfCheetah-v4 | - |

---

## 参考资料

- [Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd Ed.)](https://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Gymnasium 文档](https://gymnasium.farama.org/)
- [DeepMind 强化学习课程（David Silver）](https://www.davidsilver.uk/teaching/)
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)
- [CleanRL —— 单文件深度强化学习实现](https://github.com/vwxyzjn/cleanrl)

---

## 许可证

本项目采用 [MIT License](LICENSE)。