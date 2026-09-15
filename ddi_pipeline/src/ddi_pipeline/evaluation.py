"""Stage 10 — Evaluation & comparison.

Runs both heads on identical frozen bottleneck features (regular test set
+ cold-start test set) and reports accuracy, F1, AUROC, AUPRC, parameter
count, and training time side by side — the pipeline's core research result.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


@torch.no_grad()
def evaluate_head_on_features(head: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    head.eval()
    preds = head(x)
    y_np, preds_np = y.cpu().numpy(), preds.cpu().numpy()
    pred_labels = (preds_np >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_np, pred_labels),
        "f1": f1_score(y_np, pred_labels, zero_division=0),
    }
    if len(set(y_np.tolist())) > 1:
        metrics["auroc"] = roc_auc_score(y_np, preds_np)
        metrics["auprc"] = average_precision_score(y_np, preds_np)
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics


def compare_heads(
    backbone_num_params: int,
    heads: dict[str, nn.Module],
    test_features: dict[str, tuple[torch.Tensor, torch.Tensor]],
    cold_start_features: dict[str, tuple[torch.Tensor, torch.Tensor]],
    train_results: dict[str, dict[str, Any]],
) -> "pd.DataFrame":
    """Builds the Stage 10 side-by-side comparison table.

    `test_features`/`cold_start_features` map head name -> (x, y) tensors
    computed from the *same frozen backbone*, so the only thing that
    differs between rows is the head itself.
    """
    import pandas as pd

    rows = []
    for name, head in heads.items():
        test_x, test_y = test_features[name]
        test_metrics = evaluate_head_on_features(head, test_x, test_y)

        if cold_start_features[name][0].shape[0] > 0:
            cold_x, cold_y = cold_start_features[name]
            cold_metrics = evaluate_head_on_features(head, cold_x, cold_y)
        else:
            cold_metrics = {"accuracy": float("nan"), "f1": float("nan"), "auroc": float("nan"), "auprc": float("nan")}

        n_params = backbone_num_params + head.num_parameters()
        rows.append(
            {
                "head": name,
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
                "test_auroc": test_metrics["auroc"],
                "test_auprc": test_metrics["auprc"],
                "cold_start_accuracy": cold_metrics["accuracy"],
                "cold_start_f1": cold_metrics["f1"],
                "cold_start_auroc": cold_metrics["auroc"],
                "cold_start_auprc": cold_metrics["auprc"],
                "num_parameters": n_params,
                "training_time_sec": train_results[name]["training_time_sec"],
            }
        )
    return pd.DataFrame(rows)
