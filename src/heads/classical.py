"""Swap-invariant pair classifier over per-drug feature vectors."""
from __future__ import annotations

import torch
from torch import nn


class SymmetricPairHead(nn.Module):
    """Shared encoder and symmetric pair features yield one interaction logit."""

    def __init__(self, drug_feature_dim: int = 2048, width: int = 128, dropout: float = .2):
        super().__init__()
        # Keep the encoder's parameter names stable so earlier checkpoints can
        # still be loaded, while regularizing its hidden activations in training.
        self.phi = nn.Sequential(nn.Linear(drug_feature_dim, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU())
        self.encoder_dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, drug_a: torch.Tensor, drug_b: torch.Tensor) -> torch.Tensor:
        def encode(x: torch.Tensor) -> torch.Tensor:
            x = self.phi[1](self.phi[0](x))
            x = self.encoder_dropout(x)
            x = self.phi[3](self.phi[2](x))
            return self.encoder_dropout(x)

        a, b = encode(drug_a), encode(drug_b)
        symmetric = torch.cat((a + b, torch.abs(a - b), a * b), dim=-1)
        return self.classifier(symmetric).squeeze(-1)
