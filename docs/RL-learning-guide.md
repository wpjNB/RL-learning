# 强化学习（RL）学习文档（面向自动驾驶与机器人）

---

## 🎯 学习目标

建立从零基础 → 工程应用 → 科研能力的完整强化学习知识体系，重点面向：

* 🚗 自动驾驶决策与规划
* 🤖 机器人运动控制
* 🧠 AI算法岗位 / 研究方向

---

## 🧱 第一阶段：数学与基础能力（必须）

### 1. 数学基础

重点掌握：

* 线性代数：向量、矩阵运算、特征值
* 概率论：条件概率、期望、方差
* 优化理论：梯度下降、凸优化基础

建议达到：能够理解神经网络反向传播与优化过程。

---

### 2. 编程基础

必备技能：

* Python（核心）
* NumPy（矩阵计算）
* PyTorch（深度学习框架）
* Linux 基础

建议：能够独立实现简单神经网络训练。

---

## 🧠 第二阶段：强化学习理论基础

### 1. 马尔可夫决策过程（MDP）

强化学习问题通常建模为：

* 状态 S
* 动作 A
* 转移概率 P
* 奖励 R
* 折扣因子 γ

目标：最大化长期累计奖励。

---

### 2. 价值函数

* 状态价值函数 V(s)
* 动作价值函数 Q(s,a)

理解"当前决策对未来收益的影响"。

---

### 3. Bellman 方程

强化学习核心递推关系，用于价值估计与策略优化。

$$V(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]$$

$$Q(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q(s',a')$$

---

### 4. 动态规划方法

* Policy Iteration
* Value Iteration

适用于已知环境模型的情况。

---

## 🎮 第三阶段：经典 RL 算法

建议按以下顺序学习：

### 1. Q-Learning

* 无模型（Model-free）
* 离散动作
* 学习 Q 表

理解探索与利用（Exploration vs Exploitation）。

```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
```

---

### 2. Deep Q Network（DQN）

关键技术：

* 神经网络近似 Q 函数
* 经验回放（Replay Buffer）
* 目标网络（Target Network）

适合高维输入（图像等）。

---

### 3. Policy Gradient

直接优化策略函数 π(a|s)。

优点：可处理连续动作空间。

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot G_t \right]$$

---

### 4. Actor-Critic

结构：

* Actor：输出动作
* Critic：评估价值

现代 RL 的基础框架。

---

## 🚀 第四阶段：主流先进算法

### 1. PPO（Proximal Policy Optimization）

特点：

* On-policy
* 训练稳定
* 工业界常用

核心思想：限制策略更新幅度，防止性能崩溃。

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

---

### 2. SAC（Soft Actor-Critic）

特点：

* Off-policy
* 连续控制表现优秀
* 样本效率高

核心思想：最大化奖励 + 策略熵（增强探索）。

$$J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi} \left[ r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

---

## 🤖 第五阶段：机器人与自动驾驶应用

### 1. 控制任务

* 机械臂操作
* 无人机姿态控制
* 移动机器人导航

重点：连续动作空间 RL。

---

### 2. 自动驾驶决策规划

RL 可用于：

* 变道决策
* 跟车策略
* 路口通行
* 交通交互

典型建模：

| 要素 | 描述 |
|------|------|
| State | 周围车辆 + 自车状态 + 地图 |
| Action | 加速度、方向、换道 |
| Reward | 安全、效率、舒适 |

---

### 3. Sim-to-Real 问题

仿真训练 → 真实部署的差距，需要：

* 域随机化（Domain Randomization）
* 鲁棒控制
* 安全约束（Safe RL）

---

## 🌍 第六阶段：科研方向拓展

可深入研究：

* Model-Based RL
* Offline RL
* Multi-Agent RL
* Safe RL
* 模仿学习（Imitation Learning）

---

## 🧪 推荐实验环境

### 自动驾驶仿真

| 环境 | 特点 |
|------|------|
| [CARLA](https://carla.org/) | 高真实感城市驾驶仿真 |
| [SUMO](https://eclipse.dev/sumo/) | 交通流仿真 |
| [LGSVL](https://www.svlsimulator.com/) | 自动驾驶传感器仿真 |

### 机器人仿真

| 环境 | 特点 |
|------|------|
| [MuJoCo](https://mujoco.org/) | 物理精确连续控制 |
| [Isaac Gym](https://developer.nvidia.com/isaac-gym) | GPU 并行机器人仿真 |
| [Gazebo](https://gazebosim.org/) | ROS 集成机器人仿真 |

---

## 🏆 项目建议（强烈推荐）

### 入门项目

* CartPole 平衡控制（DQN / PPO）
* FrozenLake 路径规划（Q-Learning）

### 进阶项目

* 连续控制任务（SAC on HalfCheetah）
* 机械臂抓取（DDPG）

### 高级项目

* 自动驾驶变道决策系统
* 多车交互策略学习

---

## 📈 学习顺序总结

```
1. 数学 + Python + 深度学习基础
        ↓
2. MDP + Q-learning
        ↓
3. DQN
        ↓
4. Policy Gradient / Actor-Critic
        ↓
5. PPO
        ↓
6. SAC
        ↓
7. 机器人 / 自动驾驶应用
        ↓
8. 科研方向拓展
```

---

## 🎯 终极目标能力

完成本路线后，应具备：

- ✅ 能实现主流 RL 算法
- ✅ 能设计奖励函数
- ✅ 能构建仿真训练环境
- ✅ 能将 RL 应用于真实控制问题
- ✅ 具备科研与工程能力

---

## 💼 求职建议

如果用于求职，建议同时准备：

* 深度学习基础（CNN / Transformer）
* 控制理论（PID / MPC）
* C++ 或系统能力
* 实际项目经验（GitHub 展示）

---

## 📚 参考资料

* [Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd Ed.)](https://incompleteideas.net/book/the-book-2nd.html)
* [OpenAI Gymnasium 文档](https://gymnasium.farama.org/)
* [DeepMind 强化学习课程（David Silver）](https://www.davidsilver.uk/teaching/)
* [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)
* [CleanRL —— 单文件深度强化学习实现](https://github.com/vwxyzjn/cleanrl)
* [stable-baselines3 文档](https://stable-baselines3.readthedocs.io/)
