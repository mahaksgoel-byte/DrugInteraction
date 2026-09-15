"""Pipeline A — Classical head.
Linear(8->16) -> ReLU -> Dropout(0.2) -> Linear(16->1) -> Sigmoid.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassicalHead(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
