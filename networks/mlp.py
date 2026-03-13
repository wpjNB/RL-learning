"""
多层感知机（MLP）网络模块
"""

import torch
import torch.nn as nn
from typing import List, Optional, Type


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int],
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Optional[Type[nn.Module]] = None,
) -> nn.Sequential:
    """构建 MLP 网络。"""
    layers: List[nn.Module] = []
    in_size = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(in_size, h))
        layers.append(activation())
        in_size = h
    layers.append(nn.Linear(in_size, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """通用多层感知机。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int] = (256, 256),
        activation: Type[nn.Module] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.net = build_mlp(input_dim, output_dim, list(hidden_sizes), activation, output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
