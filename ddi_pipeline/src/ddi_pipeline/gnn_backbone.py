"""Stages 6-9 — Self-supervised pretraining, heterogeneous R-GCN encoder,
drug-pair pooling, and bottleneck projection.

This is the shared backbone: Stage 9's output (an 8-dim vector per drug
pair) is the single handoff point fed identically into both the classical
and quantum heads, per the "fork point" note in the architecture diagram.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, RGCNConv, SAGEConv


class HeteroGNNEncoder(nn.Module):
    """Stage 7 — relation-aware encoder, 3 message-passing layers,
    one weight matrix per edge type (implemented via HeteroConv + SAGEConv
    per relation, which is R-GCN's per-relation-weight idea expressed with
    PyG's heterogeneous convolution wrapper).
    """

    def __init__(self, metadata: tuple, in_dims: dict[str, int], hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        node_types, edge_types = metadata
        self.dropout = dropout
        self.lin_in = nn.ModuleDict(
            {nt: nn.Linear(max(in_dims.get(nt, hidden_dims[0]), 1), hidden_dims[0]) for nt in node_types}
        )

        self.convs = nn.ModuleList()
        dims = hidden_dims
        for layer_idx in range(len(dims)):
            in_dim = dims[layer_idx - 1] if layer_idx > 0 else dims[0]
            out_dim = dims[layer_idx]
            conv = HeteroConv(
                {et: SAGEConv((-1, -1), out_dim) for et in edge_types},
                aggr="mean",
            )
            self.convs.append(conv)

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict) -> dict[str, torch.Tensor]:
        h_dict = {nt: F.relu(self.lin_in[nt](x)) for nt, x in x_dict.items()}
        for conv in self.convs:
            out = conv(h_dict, edge_index_dict)
            for nt in h_dict:
                if nt in out:
                    h_dict[nt] = F.relu(out[nt])
                    h_dict[nt] = F.dropout(h_dict[nt], p=self.dropout, training=self.training)
        return h_dict


class SelfSupervisedPretrainer(nn.Module):
    """Stage 6 — masked-feature reconstruction pretraining. Masks ~15% of
    drug node features and trains the encoder + a small decoder to
    reconstruct them, giving the encoder general structural fluency before
    it ever sees a DDI label (improves cold-start performance)."""

    def __init__(self, encoder: HeteroGNNEncoder, drug_in_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(hidden_dim, drug_in_dim)

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict, mask_rate: float) -> torch.Tensor:
        drug_x = x_dict["drug"]
        mask = torch.rand(drug_x.shape[0], device=drug_x.device) < mask_rate
        masked_x = drug_x.clone()
        masked_x[mask] = 0.0

        x_dict_masked = dict(x_dict)
        x_dict_masked["drug"] = masked_x
        h_dict = self.encoder(x_dict_masked, edge_index_dict)
        recon = self.decoder(h_dict["drug"])

        if mask.sum() == 0:
            return torch.tensor(0.0, device=drug_x.device, requires_grad=True)
        loss = F.mse_loss(recon[mask], drug_x[mask])
        return loss


def pretrain_encoder(
    encoder: HeteroGNNEncoder,
    data: HeteroData,
    epochs: int,
    lr: float,
    mask_rate: float,
    final_hidden_dim: int,
    device: str = "cpu",
) -> HeteroGNNEncoder:
    """Stage 6 training loop."""
    pretrainer = SelfSupervisedPretrainer(encoder, data["drug"].x.shape[1], final_hidden_dim)
    pretrainer.to(device)

    optimizer = torch.optim.Adam(pretrainer.parameters(), lr=lr)
    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}

    pretrainer.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = pretrainer(x_dict, edge_index_dict, mask_rate)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"[pretrain] epoch {epoch+1}/{epochs} reconstruction_loss={loss.item():.4f}")
    return pretrainer.encoder


class DrugPairPooling(nn.Module):
    """Stage 8 — pair vector = [h_A || h_B || (h_A - h_B) || (h_A * h_B)]."""

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        return torch.cat([h_a, h_b, h_a - h_b, h_a * h_b], dim=-1)


class BottleneckProjection(nn.Module):
    """Stage 9 — Linear(256 -> bottleneck_dim). Shared handoff point into
    both the classical and quantum heads."""

    def __init__(self, in_dim: int, bottleneck_dim: int = 8):
        super().__init__()
        self.proj = nn.Linear(in_dim, bottleneck_dim)

    def forward(self, pair_vec: torch.Tensor) -> torch.Tensor:
        return self.proj(pair_vec)


class DDIBackbone(nn.Module):
    """Wraps Stages 7-9 into one module: encoder -> pair pooling -> bottleneck."""

    def __init__(self, metadata: tuple, in_dims: dict[str, int], cfg: dict[str, Any]):
        super().__init__()
        gcfg = cfg["gnn"]
        pcfg = cfg["pooling"]
        self.encoder = HeteroGNNEncoder(metadata, in_dims, gcfg["hidden_dims"], dropout=gcfg["dropout"])
        self.pooling = DrugPairPooling()
        pair_dim = gcfg["hidden_dims"][-1] * 4
        self.bottleneck = BottleneckProjection(pair_dim, pcfg["bottleneck_dim"])

    def encode_nodes(self, data: HeteroData, device: str = "cpu") -> dict[str, torch.Tensor]:
        x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
        edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
        return self.encoder(x_dict, edge_index_dict)

    def pair_bottleneck(self, h_drug: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        h_a, h_b = h_drug[idx_a], h_drug[idx_b]
        pair_vec = self.pooling(h_a, h_b)
        return self.bottleneck(pair_vec)
