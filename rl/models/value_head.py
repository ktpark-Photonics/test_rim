"""State-value head producing V(s) estimates."""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


__all__ = ["ValueHead"]
