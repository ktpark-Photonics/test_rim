"""Lightweight CNN backbones used by policy/value networks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisionBackboneConfig:
    input_channels: int
    feature_dim: int = 512
    model: Literal["conv", "mobilenet"] = "conv"


class ConvBackbone(nn.Module):
    def __init__(self, input_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128 * 20 * 11, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)
        return torch.tanh(self.fc(x))


class VisionBackbone(nn.Module):
    """Factory wrapper selecting between supported backbone architectures."""

    def __init__(self, config: VisionBackboneConfig, input_shape: Tuple[int, int, int]) -> None:
        super().__init__()
        height, width = input_shape[1], input_shape[2]
        if config.model != "conv":  # pragma: no cover - alternative models not yet implemented
            raise NotImplementedError("Only the lightweight conv backbone is implemented in this build")
        if height != 180 or width != 320:
            # adjust linear layer size dynamically
            conv = nn.Sequential(
                nn.Conv2d(config.input_channels, 32, kernel_size=8, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, config.input_channels, height, width)
                conv_out = conv(dummy)
            feature_dim = conv_out.shape[1] * conv_out.shape[2] * conv_out.shape[3]
            self.network = nn.Sequential(conv, nn.Flatten(), nn.Linear(feature_dim, config.feature_dim), nn.Tanh())
        else:
            self.network = ConvBackbone(config.input_channels, config.feature_dim)
        self.output_dim = config.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


__all__ = ["VisionBackbone", "VisionBackboneConfig"]
