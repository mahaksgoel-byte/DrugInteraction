"""Train-only drug-degree shortcut baseline for DDInter pair splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             matthews_corrcoef, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]


def pair_features(frame: pd.DataFrame, degree: dict[str, int]) -> np.ndarray:
    """Build swap-invariant endpoint popularity features using train-only degrees."""
    da = np.log1p(frame["drug_id_a"].map(degree).fillna(0).to_numpy(dtype=np.float64))
    db = np.log1p(frame["drug_id_b"].map(degree).fillna(0).to_numpy(dtype=np.float64))
    return np.column_stack((da + db, np.abs(da - db), da * db))


def run(split_dir: Path, output_path: Path) -> dict:
    train = pd.read_csv(split_dir / "train.csv.gz", dtype={"drug_id_a": str, "drug_id_b": str})
    validation = pd.read_csv(split_dir / "val.csv.gz", dtype={"drug_id_a": str, "drug_id_b": str})
    test = pd.read_csv(split_dir / "test.csv.gz", dtype={"drug_id_a": str, "drug_id_b": str})

    # Only positive labels from the training partition are allowed to create this
    # shortcut feature. Validation/test interaction degrees are never consulted.
    positive_train = train.loc[train.label == 1]
    degree_series = pd.concat((positive_train.drug_id_a, positive_train.drug_id_b)).value_counts()
    degree = {str(drug_id): int(value) for drug_id, value in degree_series.items()}
    X_train, X_val, X_test = (pair_features(frame, degree) for frame in (train, validation, test))
    y_train = train.label.to_numpy(dtype=np.int8)
    y_val = validation.label.to_numpy(dtype=np.int8)
    y_test = test.label.to_numpy(dtype=np.int8)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
    model.fit(X_train, y_train)
    p_val = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, p_val)
    finite = np.isfinite(thresholds)
    threshold = float(thresholds[finite][np.argmax((tpr - fpr)[finite])])
    p_test = model.predict_proba(X_test)[:, 1]
    pred_test = p_test >= threshold
    result = {
        "name": "train_positive_degree_only_logistic",
        "split": str(split_dir),
        "degree_source": "positive interaction pairs in training split only",
        "feature_definition": ["log1p(degree_a)+log1p(degree_b)", "abs(log1p(degree_a)-log1p(degree_b))", "log1p(degree_a)*log1p(degree_b)"],
        "unique_training_drugs_with_positive_degree": len(degree),
        "validation_selected_threshold": threshold,
        "validation": {
            "accuracy": float(accuracy_score(y_val, p_val >= threshold)),
            "auroc": float(roc_auc_score(y_val, p_val)),
            "auprc": float(average_precision_score(y_val, p_val)),
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, pred_test)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred_test)),
            "matthews_correlation": float(matthews_corrcoef(y_test, pred_test)),
            "auroc": float(roc_auc_score(y_test, p_test)),
            "auprc": float(average_precision_score(y_test, p_test)),
            "fraction_of_test_endpoints_with_nonzero_train_degree": float(np.mean(np.concatenate((
                test.drug_id_a.map(degree).fillna(0).to_numpy() > 0,
                test.drug_id_b.map(degree).fillna(0).to_numpy() > 0)))),
        },
        "interpretation": "Shortcut diagnostic only; it predicts from observed training positive degree, not chemistry or mechanism.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.split_dir, args.output)


if __name__ == "__main__":
    main()
