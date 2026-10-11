"""Train and evaluate the symmetric DDInter pair-feature baseline."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, TensorDataset

from src.heads.classical import SymmetricPairHead

ROOT = Path(__file__).resolve().parents[2]


def load_split(path: Path, id_to_row: dict[str, int], fps: np.ndarray) -> TensorDataset:
    frame = pd.read_csv(path, dtype={"drug_id_a": str, "drug_id_b": str})
    missing = (set(frame.drug_id_a) | set(frame.drug_id_b)) - id_to_row.keys()
    if missing:
        raise ValueError(f"Split references {len(missing)} drugs absent from fingerprint cache")
    ia = np.fromiter((id_to_row[x] for x in frame.drug_id_a), dtype=np.int64)
    ib = np.fromiter((id_to_row[x] for x in frame.drug_id_b), dtype=np.int64)
    # Keep endpoint IDs instead of copying two large fingerprints per pair.
    return TensorDataset(torch.from_numpy(ia), torch.from_numpy(ib),
                         torch.tensor(frame.label.to_numpy(dtype=np.float32)))


def evaluate(model: torch.nn.Module, loader: DataLoader, features: torch.Tensor,
             device: torch.device, threshold: float = .5, choose_threshold: bool = False) -> dict:
    model.eval()
    ys, ps, logits_all = [], [], []
    with torch.no_grad():
        for ia, ib, y in loader:
            a, b = features[ia.to(device)], features[ib.to(device)]
            logits = model(a, b)
            p = torch.sigmoid(logits).cpu().numpy()
            logits_all.extend(logits.cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
            ps.extend(p.tolist())
    if len(set(ys)) < 2:
        raise ValueError("AUROC requires both classes in the evaluation split")
    y_true, y_score = np.asarray(ys, dtype=np.int8), np.asarray(ps)
    val_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(logits_all, dtype=torch.float32), torch.tensor(y_true, dtype=torch.float32)))
    if choose_threshold:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        # Youden's J selects an operating threshold from validation data only.
        finite = np.isfinite(thresholds)
        threshold = float(thresholds[finite][int(np.argmax((tpr - fpr)[finite]))])
    predicted = (y_score >= threshold).astype(np.int8)
    # Expected calibration error: average gap between confidence and observed
    # frequency in fixed probability bins. This is descriptive for the sampled
    # benchmark and is not a real-world risk calibration guarantee.
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        in_bin = (y_score >= lower) & ((y_score < upper) if upper < 1.0 else (y_score <= upper))
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(float(y_score[in_bin].mean()) - float(y_true[in_bin].mean()))
    return {"loss": val_loss,
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score)),
            "accuracy_at_0_5": float(accuracy_score(y_true, y_score >= .5)),
            "accuracy": float(accuracy_score(y_true, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "matthews_correlation": float(matthews_corrcoef(y_true, predicted)),
            "brier_score": float(brier_score_loss(y_true, y_score)),
            "expected_calibration_error_10_bins": float(ece),
            "threshold": threshold}


def train(split_dir: Path, feature_file: Path, output_dir: Path, *, seed: int = 0, epochs: int = 100,
          patience: int = 12, batch_size: int = 512, learning_rate: float = 1e-3,
          width: int = 128, dropout: float = .2, weight_decay: float = .01,
          evaluate_test: bool = True, on_epoch: Callable[[dict], None] | None = None) -> dict:
    """Fit with validation-AUPRC checkpointing; evaluate test only after selection."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(feature_file, allow_pickle=False) as data:
        ids = data["drug_ids"].astype(str)
        fingerprints = data["fingerprints"].astype(np.float32)
        fps = data["drug_features"].astype(np.float32) if "drug_features" in data.files else fingerprints
        if "pharmgkb_only" in feature_file.stem:
            feature_type = "ClinPGx only"
        elif "sider_only" in feature_file.stem:
            feature_type = "SIDER only"
        elif "morgan_sider" in feature_file.stem:
            feature_type = "Morgan + SIDER"
        elif fps.shape[1] > fingerprints.shape[1]:
            feature_type = "Morgan + SIDER + ClinPGx"
        else:
            feature_type = "Morgan only"
    id_to_row = {drug_id: i for i, drug_id in enumerate(ids)}
    train_ds = load_split(split_dir / "train.csv.gz", id_to_row, fps)
    val_ds = load_split(split_dir / "val.csv.gz", id_to_row, fps)
    test_ds = load_split(split_dir / "test.csv.gz", id_to_row, fps) if evaluate_test else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds is not None else None
    # A single compact feature table replaces gigabytes of duplicated pair tensors.
    features = torch.from_numpy(fps).float().to(device)
    model = SymmetricPairHead(fps.shape[1], width, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=.5, patience=max(2, patience // 3), min_lr=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_ap, best_epoch, stale, history = -1.0, 0, 0, []
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    epoch_log = (output_dir / "epoch_metrics.jsonl").open("w", encoding="utf-8")
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for ia, ib, y in train_loader:
            ia, ib, y = ia.to(device), ib.to(device), y.to(device)
            a, b = features[ia], features[ib]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(a, b), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, val_loader, features, device, choose_threshold=True)
        scheduler.step(val["auprc"])
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val["loss"],
                  "val_auroc": val["auroc"], "val_auprc": val["auprc"],
                  "val_accuracy": val["accuracy"], "val_accuracy_at_0_5": val["accuracy_at_0_5"],
                  "val_balanced_accuracy": val["balanced_accuracy"],
                  "val_matthews_correlation": val["matthews_correlation"],
                  "val_brier_score": val["brier_score"],
                  "val_expected_calibration_error_10_bins": val["expected_calibration_error_10_bins"],
                  "decision_threshold": val["threshold"],
                  "learning_rate": float(optimizer.param_groups[0]["lr"]),
                  "elapsed_seconds": round(time.time() - started, 2)}
        history.append(record)
        epoch_log.write(json.dumps(record) + "\n")
        epoch_log.flush()
        print(json.dumps(record), flush=True)
        if on_epoch:
            on_epoch(record)
        if val["auprc"] > best_ap + 1e-8:
            best_ap, best_epoch, stale = val["auprc"], epoch, 0
            torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch,
                        "architecture": "shared_mlp_encoder_dropout_symmetric_pair_v2",
                        "drug_feature_dim": int(fps.shape[1]), "width": width, "dropout": dropout,
                        "learning_rate": learning_rate, "weight_decay": weight_decay,
                        "decision_threshold": val["threshold"]}, output_dir / "best.pt")
        else:
            stale += 1
        if patience and stale >= patience:
            break
    epoch_log.close()
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    best_record = next(x for x in history if x["epoch"] == best_epoch)
    result = {"seed": seed, "device": str(device), "split": str(split_dir),
              "feature_file": str(feature_file), "epochs_requested": epochs,
              "feature_type": feature_type, "drug_feature_dim": int(fps.shape[1]),
              "epochs_completed": len(history), "best_epoch": best_epoch,
              "best_validation_auprc": float(best_ap),
              "validation_metrics_at_best": {"auroc": best_record["val_auroc"],
                  "auprc": best_record["val_auprc"], "accuracy": best_record["val_accuracy"],
                  "accuracy_at_0_5": best_record["val_accuracy_at_0_5"],
                  "balanced_accuracy": best_record["val_balanced_accuracy"],
                  "matthews_correlation": best_record["val_matthews_correlation"],
                  "brier_score": best_record["val_brier_score"],
                  "expected_calibration_error_10_bins": best_record["val_expected_calibration_error_10_bins"],
                  "threshold": best_record["decision_threshold"]},
              "architecture": "shared_mlp_encoder_dropout_symmetric_pair_v2",
              "config": {"batch_size": batch_size, "learning_rate": learning_rate,
                         "width": width, "dropout": dropout, "weight_decay": weight_decay,
                         "patience": patience}, "history": history}
    if test_loader is not None:
        result["test_metrics"] = evaluate(model, test_loader, features, device,
                                          threshold=checkpoint["decision_threshold"])
        result["test_checkpoint_epoch"] = best_epoch
    (output_dir / "training_history.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True, help="e.g. data/splits/S3/seed_00")
    parser.add_argument("--features", type=Path, default=ROOT / "cache/features/morgan_r2_2048.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.2)
    parser.add_argument("--weight-decay", type=float, default=.01)
    args = parser.parse_args()
    train(args.split_dir, args.features, args.output_dir, seed=args.seed, epochs=args.epochs,
          patience=args.patience, batch_size=args.batch_size, learning_rate=args.learning_rate,
          width=args.width, dropout=args.dropout, weight_decay=args.weight_decay)


if __name__ == "__main__":
    main()
