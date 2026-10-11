"""Controlled classical-vs-Qiskit variational-circuit experiment.

Qiskit Aer/QPU is not required: Qiskit's exact Statevector simulator executes the
four-qubit circuit locally. This is an actual quantum-circuit simulation, not a
quantum-inspired estimator. Pair sampling is shared by both models for fairness.
"""
from __future__ import annotations

import json
import hashlib
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, matthews_corrcoef,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]


def _sample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.reset_index(drop=True)
    return (frame.groupby("label", group_keys=False)
            .apply(lambda group: group.sample(
                n=max(1, round(limit * len(group) / len(frame))),
                random_state=seed), include_groups=True)
            .sample(frac=1, random_state=seed).head(limit).reset_index(drop=True))


def _make_quantum_circuit(x: np.ndarray, weights: np.ndarray) -> float:
    """Build and execute one 4-qubit variational circuit using Qiskit Statevector."""
    qc = QuantumCircuit(4)
    qc.h(range(4))
    for q in range(4):
        qc.ry(float(x[q]) + float(weights[q]), q)
    # ZZ data re-uploading adds pairwise interactions to the angle map.
    for q in range(4):
        next_q = (q + 1) % 4
        qc.rzz(float(x[q] * x[next_q]), q, next_q)
    for q in range(4):
        qc.ry(float(weights[4 + q]), q)
    for q in range(4):
        qc.cx(q, (q + 1) % 4)
    probabilities = Statevector.from_instruction(qc).probabilities()
    # Class 1 is the odd-parity measurement outcome.
    p_positive = float(sum(prob for outcome, prob in enumerate(probabilities)
                           if outcome.bit_count() % 2 == 1))
    return float(np.clip(p_positive, 1e-6, 1 - 1e-6))


def _predict_quantum(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.fromiter((_make_quantum_circuit(row, weights) for row in x),
                       dtype=np.float64, count=len(x))


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    pred = (probability >= threshold).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "auroc": float(roc_auc_score(y, probability)),
            "auprc": float(average_precision_score(y, probability)),
            "matthews_correlation": float(matthews_corrcoef(y, pred)),
            "threshold": float(threshold), "pairs_evaluated": int(len(y))}


def _youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def train_compare(split_dir: Path, feature_file: Path, output_dir: Path, *, seed: int = 0,
                  epochs: int = 8, pair_cap: int = 512, on_epoch: Callable | None = None) -> dict:
    """Fit logistic and a Qiskit VQC against identical sampled pairs and report metrics."""
    rng = np.random.default_rng(seed)
    if epochs < 1 or pair_cap < 64:
        raise ValueError('Use at least one epoch and a pair cap of at least 64.')
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(feature_file, allow_pickle=False) as archive:
        drug_ids = archive["drug_ids"].astype(str)
        drug_features = archive["drug_features"].astype(np.float32)
    id_to_row = {drug_id: i for i, drug_id in enumerate(drug_ids)}
    split_paths = {name: split_dir / f"{name}.csv.gz" for name in ("train", "val", "test")}
    frames = {name: pd.read_csv(path, dtype={"drug_id_a": str, "drug_id_b": str})
              for name, path in split_paths.items()}
    missing = {str(d) for frame in frames.values() for col in ("drug_id_a", "drug_id_b")
               for d in frame[col].unique() if str(d) not in id_to_row}
    if missing:
        raise ValueError(f"Feature cache is missing {len(missing)} split drugs")

    # Cap runtime equally. Identical sampled pairs are used for both models.
    train_frame = _sample(frames["train"], pair_cap, seed)
    val_frame = _sample(frames["val"], max(128, pair_cap // 2), seed + 1)
    test_frame = _sample(frames["test"], max(128, pair_cap // 2), seed + 2)
    seen_train_ids = sorted(set(train_frame.drug_id_a) | set(train_frame.drug_id_b))
    train_rows = np.fromiter((id_to_row[x] for x in seen_train_ids), dtype=np.int64)
    # Fit dimensionality reduction using training-pair endpoints only.
    pca = PCA(n_components=4, random_state=seed)
    pca.fit(drug_features[train_rows])
    drug_z = pca.transform(drug_features)
    def pair_x(frame: pd.DataFrame) -> np.ndarray:
        ia = np.fromiter((id_to_row[x] for x in frame.drug_id_a), dtype=np.int64)
        ib = np.fromiter((id_to_row[x] for x in frame.drug_id_b), dtype=np.int64)
        a, b = drug_z[ia], drug_z[ib]
        # Symmetric pair representation: swapping drugs leaves all values unchanged.
        return np.tanh(np.abs(a - b) + a * b).astype(np.float64)

    x_train = pair_x(train_frame)
    x_val, x_test = pair_x(val_frame), pair_x(test_frame)
    scaler = StandardScaler().fit(x_train)
    x_train, x_val, x_test = (np.clip(scaler.transform(x), -2.5, 2.5) * (math.pi / 2)
                              for x in (x_train, x_val, x_test))
    y_train = train_frame.label.to_numpy(dtype=np.int8)
    y_val = val_frame.label.to_numpy(dtype=np.int8)
    y_test = test_frame.label.to_numpy(dtype=np.int8)
    pair_checksums = {}
    for name, frame in (("train", train_frame), ("validation", val_frame), ("test", test_frame)):
        stable_rows = pd.util.hash_pandas_object(
            frame[["drug_id_a", "drug_id_b", "label"]], index=False).values.tobytes()
        pair_checksums[name] = hashlib.sha256(stable_rows).hexdigest()

    started = time.time()
    # Classical reference uses exactly the same four standardized pair inputs.
    classical = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    classical.fit(x_train, y_train)
    c_val = classical.predict_proba(x_val)[:, 1]
    c_threshold = _youden_threshold(y_val, c_val)

    # SPSA trains eight variational gate angles. Each objective evaluation uses
    # the exact statevector simulator; no hardware or fake quantum backend.
    weights = rng.normal(0, 0.1, size=8)
    best_val_ap, best_weights, best_epoch, stale = -1.0, weights.copy(), 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        ck = 0.15 / (epoch ** 0.101)
        ak = 0.10 / (epoch ** 0.602)
        batch_order = rng.permutation(len(x_train))
        batch_losses = []
        for start in range(0, len(batch_order), 64):
            batch = batch_order[start:start + 64]
            xb, yb = x_train[batch], y_train[batch]
            delta = rng.choice([-1.0, 1.0], size=len(weights))
            p_plus = _predict_quantum(xb, weights + ck * delta)
            p_minus = _predict_quantum(xb, weights - ck * delta)
            loss_plus = -np.mean(yb * np.log(p_plus) + (1 - yb) * np.log(1 - p_plus))
            loss_minus = -np.mean(yb * np.log(p_minus) + (1 - yb) * np.log(1 - p_minus))
            gradient = ((loss_plus - loss_minus) / (2 * ck)) / delta
            weights = weights - ak * gradient
            batch_losses.append((loss_plus + loss_minus) / 2)
        q_val = _predict_quantum(x_val, weights)
        threshold = _youden_threshold(y_val, q_val)
        val_metric = _metrics(y_val, q_val, threshold)
        record = {"epoch": epoch, "train_loss": float(np.mean(batch_losses)),
                  "train_batches": int(len(batch_losses)), "val_auprc": val_metric["auprc"],
                  "val_auroc": val_metric["auroc"], "val_accuracy": val_metric["accuracy"],
                  "val_threshold": threshold, "elapsed_seconds": round(time.time() - started, 2)}
        history.append(record)
        with (output_dir / 'epoch_metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if on_epoch:
            on_epoch(record)
        if val_metric["auprc"] > best_val_ap + 1e-8:
            best_val_ap, best_weights, best_epoch, stale = val_metric["auprc"], weights.copy(), epoch, 0
        else:
            stale += 1
        if stale >= 4:
            break

    q_val = _predict_quantum(x_val, best_weights)
    q_threshold = _youden_threshold(y_val, q_val)
    # Held-out pair labels are touched only after model/checkpoint selection.
    c_test = classical.predict_proba(x_test)[:, 1]
    classical_metrics = _metrics(y_test, c_test, c_threshold)
    q_test = _predict_quantum(x_test, best_weights)
    quantum_metrics = _metrics(y_test, q_test, q_threshold)
    result = {"engine": "Qiskit Statevector", "mode": "local exact statevector simulation",
              "training_method": "SPSA with full capped-sample mini-batch passes per epoch",
              "experiment_version": "qiskit_vqc_pair_features_v2",
              "split": str(split_dir), "feature_file": str(feature_file), "seed": seed,
              "epochs_requested": epochs, "epochs_completed": len(history),
              "best_epoch": best_epoch, "pair_cap": pair_cap,
              "pair_sampling": "stratified; same sampled pairs for both models",
              "train_pairs": len(y_train), "validation_pairs": len(y_val), "test_pairs": len(y_test),
              "pair_sample_sha256": pair_checksums,
              "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
              "classical": classical_metrics, "quantum": quantum_metrics,
              "history": history, "circuit": "4-qubit H + angle RY + ring ZZ + trainable RY + ring CNOT; odd-parity readout",
              "limitations": ["This is simulated quantum computation on a classical CPU, not quantum hardware.",
                              "A capped, stratified sample is used for local runtime; scores are exploratory.",
                              "PCA and scaling are fit using training data only; threshold is selected on validation."]}
    (output_dir / "training_history.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "epoch_metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history), encoding="utf-8")
    np.savez_compressed(output_dir / "model_parameters.npz", quantum_weights=best_weights,
                        classical_coefficients=classical.coef_, classical_intercept=classical.intercept_,
                        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, pca_components=pca.components_,
                        pca_mean=pca.mean_)
    return result
