"""Small reproducible validation-only grid search for pair-head settings."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from src.backbone.train_pairs import ROOT, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=ROOT / "cache/features/morgan_r2_2048.npz")
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[3e-4, 1e-3])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=.2)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--dropouts", type=float, nargs="+", default=None,
                        help="Optional grid; defaults to the single --dropout value")
    parser.add_argument("--weight-decays", type=float, nargs="+", default=None,
                        help="Optional grid; defaults to the single --weight-decay value")
    args = parser.parse_args()
    args.dropouts = args.dropouts or [args.dropout]
    args.weight_decays = args.weight_decays or [args.weight_decay]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    configurations = itertools.product(args.widths, args.learning_rates, args.dropouts, args.weight_decays)
    configs = list(configurations)
    for index, (width, lr, dropout, weight_decay) in enumerate(configs, start=1):
        name = f"trial_{index:02d}_w{width}_lr{str(lr).replace('.', 'p')}_d{str(dropout).replace('.', 'p')}_wd{str(weight_decay).replace('.', 'p')}"
        print(f"Trial {index}/{len(configs)}: width={width}, learning_rate={lr}, dropout={dropout}, weight_decay={weight_decay}", flush=True)
        result = train(args.split_dir, args.features, args.output_dir / name,
                       seed=args.seed, epochs=args.epochs, patience=args.patience,
                       batch_size=args.batch_size, learning_rate=lr, width=width,
                       dropout=dropout, weight_decay=weight_decay,
                       evaluate_test=False)
        rows.append({"width": width, "learning_rate": lr, "dropout": dropout,
                     "weight_decay": weight_decay, "best_epoch": result["best_epoch"],
                     "validation": result["validation_metrics_at_best"], "trial_dir": name})
    rows.sort(key=lambda row: row["validation"]["auprc"], reverse=True)
    summary = {"selection_metric": "validation_auprc", "test_evaluated": False,
               "best_config": rows[0], "trials": rows}
    (args.output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
