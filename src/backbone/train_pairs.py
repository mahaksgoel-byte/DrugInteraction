"""Train the initial structure-only symmetric pair baseline (explicit command only)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
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
    return TensorDataset(torch.from_numpy(fps[ia]).float(), torch.from_numpy(fps[ib]).float(),
                         torch.tensor(frame.label.to_numpy(dtype=np.float32)))


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for a, b, y in loader:
            p = torch.sigmoid(model(a.to(device), b.to(device))).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(p.tolist())
    return {"auroc": float(roc_auc_score(ys, ps)), "auprc": float(average_precision_score(ys, ps))}


def train(split_dir: Path, feature_file: Path, output_dir: Path, *, seed: int = 0, epochs: int = 50,
          batch_size: int = 512, learning_rate: float = 1e-3, width: int = 128, dropout: float = .2) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(feature_file, allow_pickle=False) as data:
        ids, fps = data["drug_ids"].astype(str), data["fingerprints"].astype(np.float32)
    id_to_row = {drug_id: i for i, drug_id in enumerate(ids)}
    train_ds = load_split(split_dir / "train.csv.gz", id_to_row, fps)
    val_ds = load_split(split_dir / "val.csv.gz", id_to_row, fps)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    model = SymmetricPairHead(fps.shape[1], width, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_ap, best_epoch, history = -1.0, 0, []
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for a, b, y in train_loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(a, b), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(record)
        print(json.dumps(record))
        if metrics["auprc"] > best_ap:
            best_ap, best_epoch = metrics["auprc"], epoch
            torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch,
                        "drug_feature_dim": int(fps.shape[1]), "width": width, "dropout": dropout},
                       output_dir / "best.pt")
    (output_dir / "training_history.json").write_text(json.dumps({"seed": seed, "best_epoch": best_epoch,
        "best_validation_auprc": best_ap, "history": history}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True, help="e.g. data/splits/S1/seed_00")
    parser.add_argument("--features", type=Path, default=ROOT / "cache/features/morgan_r2_2048.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.2)
    args = parser.parse_args()
    train(args.split_dir, args.features, args.output_dir, seed=args.seed, epochs=args.epochs,
          batch_size=args.batch_size, learning_rate=args.learning_rate, width=args.width, dropout=args.dropout)


if __name__ == "__main__":
    main()
