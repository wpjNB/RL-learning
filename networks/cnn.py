"""
卷积神经网络（CNN）特征提取模块，适用于图像输入的 RL 任务。
"""

import torch
import torch.nn as nn
from typing import Tuple


class NatureCNN(nn.Module):
    """
    DQN Nature 论文中使用的 CNN 结构。
    输入：(batch, C, H, W) 的灰度或彩色图像帧（像素值已归一化至 [0,1]）。
    """

    def __init__(self, in_channels: int = 4, feature_dim: int = 512) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 计算卷积输出尺寸（假设输入 84×84）
        dummy = torch.zeros(1, in_channels, 84, 84)
        conv_out = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(conv_out, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class SmallCNN(nn.Module):
    """轻量级 CNN，适合小尺寸图像输入。"""

    def __init__(self, in_channels: int = 1, feature_dim: int = 256, input_size: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # 计算卷积输出尺寸
        dummy = torch.zeros(1, in_channels, input_size, input_size)
        conv_out = self.conv(dummy).shape[1]
        self.fc = nn.Linear(conv_out, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))
