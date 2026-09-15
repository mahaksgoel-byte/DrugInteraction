"""Shared training loop for both heads, and pair-index sampling from the
DDI split tables produced in preprocessing.py.
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from .classical_head import ClassicalHead
from .gnn_backbone import DDIBackbone
from .quantum_head import QuantumHead


def pairs_to_indices(df: pd.DataFrame, drug_idx: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.tensor([drug_idx[d] for d in df["drug_a_id"]], dtype=torch.long)
    b = torch.tensor([drug_idx[d] for d in df["drug_b_id"]], dtype=torch.long)
    y = torch.tensor(df["interaction"].values, dtype=torch.float32)
    return a, b, y


def train_head(
    backbone: DDIBackbone,
    head: nn.Module,
    data,
    drug_idx: dict[str, int],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str = "cpu",
    early_stopping_patience: int = 8,
) -> dict[str, Any]:
    """Trains `backbone` + `head` jointly with BCE loss and Adam, identical
    recipe for both classical and quantum heads (only the head module and
    its own hyperparameters differ) — this is what keeps the comparison fair.
    """
    backbone.to(device)
    head.to(device)
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.BCELoss()

    train_a, train_b, train_y = pairs_to_indices(train_df, drug_idx)
    val_a, val_b, val_y = pairs_to_indices(val_df, drug_idx)
    train_a, train_b, train_y = train_a.to(device), train_b.to(device), train_y.to(device)
    val_a, val_b, val_y = val_a.to(device), val_b.to(device), val_y.to(device)

    best_val_loss = float("inf")
    patience_left = early_stopping_patience
    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    n = train_a.shape[0]
    for epoch in range(epochs):
        backbone.train()
        head.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            h_drug = backbone.encode_nodes(data, device=device)["drug"]
            bottleneck = backbone.pair_bottleneck(h_drug, train_a[batch_idx], train_b[batch_idx])
            preds = head(bottleneck)
            loss = loss_fn(preds, train_y[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
        epoch_loss /= max(n, 1)

        backbone.eval()
        head.eval()
        with torch.no_grad():
            h_drug = backbone.encode_nodes(data, device=device)["drug"]
            val_bottleneck = backbone.pair_bottleneck(h_drug, val_a, val_b)
            val_preds = head(val_bottleneck)
            val_loss = loss_fn(val_preds, val_y).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        print(f"[train:{head.__class__.__name__}] epoch {epoch+1}/{epochs} "
              f"train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[train:{head.__class__.__name__}] early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time
    return {"history": history, "training_time_sec": training_time, "best_val_loss": best_val_loss}


@torch.no_grad()
def compute_bottleneck_features(
    backbone: DDIBackbone,
    data,
    drug_idx: dict[str, int],
    df: pd.DataFrame,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Freezes the backbone and materializes the Stage 9 bottleneck vector
    for every pair in `df`. Both heads then train on this exact same
    tensor, which is what makes the classical-vs-quantum comparison fair
    (per the diagram's "Fork point" note) rather than each head implicitly
    fine-tuning its own copy of the shared encoder.
    """
    backbone.eval()
    a, b, y = pairs_to_indices(df, drug_idx)
    a, b, y = a.to(device), b.to(device), y.to(device)
    h_drug = backbone.encode_nodes(data, device=device)["drug"]
    bottleneck = backbone.pair_bottleneck(h_drug, a, b)
    return bottleneck, y


def train_head_on_features(
    head: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str = "cpu",
    early_stopping_patience: int = 8,
) -> dict[str, Any]:
    """Trains a head alone against a frozen bottleneck tensor. Used for
    both classical and quantum heads once the shared backbone is frozen,
    so each head sees identical input distributions."""
    head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)

    best_val_loss = float("inf")
    patience_left = early_stopping_patience
    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    n = train_x.shape[0]
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            preds = head(train_x[batch_idx])
            loss = loss_fn(preds, train_y[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
        epoch_loss /= max(n, 1)

        head.eval()
        with torch.no_grad():
            val_preds = head(val_x)
            val_loss = loss_fn(val_preds, val_y).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        print(f"[train:{head.__class__.__name__}] epoch {epoch+1}/{epochs} "
              f"train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[train:{head.__class__.__name__}] early stopping at epoch {epoch+1}")
                break

    return {"history": history, "training_time_sec": time.time() - start_time, "best_val_loss": best_val_loss}


def build_classical_head(cfg: dict[str, Any]) -> ClassicalHead:
    pcfg, ccfg = cfg["pooling"], cfg["classical_head"]
    return ClassicalHead(in_dim=pcfg["bottleneck_dim"], hidden_dim=ccfg["hidden_dim"], dropout=ccfg["dropout"])


def build_quantum_head(cfg: dict[str, Any]) -> QuantumHead:
    qcfg = cfg["quantum_head"]
    return QuantumHead(num_qubits=qcfg["num_qubits"], reps=qcfg["ansatz_reps"])
