"""Policy head producing macro-action logits and parameter distributions."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    param_mean: torch.Tensor
    param_log_std: torch.Tensor


class PolicyHead(nn.Module):
    def __init__(self, feature_dim: int, num_actions: int, param_dim: int) -> None:
        super().__init__()
        self.action_head = nn.Linear(feature_dim, num_actions)
        self.param_head = nn.Linear(feature_dim, param_dim)
        self.param_log_std = nn.Parameter(torch.zeros(param_dim))

    def forward(self, features: torch.Tensor) -> PolicyOutput:
        logits = self.action_head(features)
        mean = torch.tanh(self.param_head(features))
        log_std = self.param_log_std.expand_as(mean)
        return PolicyOutput(logits=logits, param_mean=mean, param_log_std=log_std)


__all__ = ["PolicyHead", "PolicyOutput"]
