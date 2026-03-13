"""
训练曲线绘图工具
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def smooth(data: List[float], window: int = 10) -> np.ndarray:
    """对数据进行滑动平均平滑。"""
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_rewards(
    rewards: List[float],
    title: str = "Training Reward",
    smooth_window: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """绘制每回合奖励曲线及平滑曲线。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(1, len(rewards) + 1)
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Raw")
    if len(rewards) >= smooth_window:
        smoothed = smooth(rewards, smooth_window)
        offset = len(rewards) - len(smoothed)
        ax.plot(episodes[offset:], smoothed, color="steelblue", linewidth=2, label=f"Smoothed ({smooth_window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"图表已保存至：{save_path}")
    plt.show()


def plot_multiple(
    data_dict: dict,
    title: str = "Comparison",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    smooth_window: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """绘制多个算法的对比曲线。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, values in data_dict.items():
        episodes = np.arange(1, len(values) + 1)
        ax.plot(episodes, values, alpha=0.2)
        if len(values) >= smooth_window:
            smoothed = smooth(values, smooth_window)
            offset = len(values) - len(smoothed)
            ax.plot(episodes[offset:], smoothed, linewidth=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"图表已保存至：{save_path}")
    plt.show()
