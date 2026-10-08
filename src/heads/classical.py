"""Swap-invariant structure-only pair classifier for the first training pass."""
from __future__ import annotations

import torch
from torch import nn


class SymmetricPairHead(nn.Module):
    """Shared drug encoder with a symmetric pair feature and binary logit."""

    def __init__(self, drug_feature_dim: int = 2048, width: int = 128, dropout: float = .2):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(drug_feature_dim, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, drug_a: torch.Tensor, drug_b: torch.Tensor) -> torch.Tensor:
        a, b = self.phi(drug_a), self.phi(drug_b)
        symmetric = torch.cat((a + b, torch.abs(a - b), a * b), dim=-1)
        return self.classifier(symmetric).squeeze(-1)
