"""Auditable Qiskit fidelity kernels on a shared frozen feature encoder.

This is an exploratory DDInter experiment, not the planned heterogeneous GNN.
Each circuit is executed by Qiskit Statevector. Classical matrix multiplication
computes exact state overlaps, equivalent to noiseless fidelity measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import qiskit
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, partial_trace, entropy
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from src.heads.classical import SymmetricPairHead

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def circuit(pair, scale=1.0, entangle=True):
    """Two shared four-qubit registers; data re-uploading makes entanglers relevant."""
    qc = QuantumCircuit(8)
    for _ in range(2):
        for q, value in enumerate(np.asarray(pair).ravel()):
            qc.ry(scale * value, q)
            qc.rz(scale * value / 2, q)
        for offset in (0, 4):
            for q in range(4):
                qc.cz(offset + q, offset + (q + 1) % 4)
        if entangle:
            for q in range(4):
                qc.rzz(0.5, q, q + 4)
    return qc


def states(pairs, scale=1.0, entangle=True, on_progress=None):
    result = np.empty((len(pairs), 2, 256), dtype=np.complex128)
    for i, pair in enumerate(pairs):
        for orient, p in enumerate((pair, pair[::-1])):
            result[i, orient] = Statevector.from_instruction(circuit(p, scale, entangle)).data
        if on_progress and (i + 1) % 1000 == 0:
            on_progress(f"Simulated {i + 1}/{len(pairs)} pairs (two orientations each)")
    return result


def fidelity_kernel(left, right, block=256):
    """Normalized Hilbert-Schmidt kernel of swap-averaged pure-state densities.

    Averaging densities (not amplitudes) gives a PSD, swap-invariant kernel.
    Normalization preserves PSD and sets self-similarity to one.
    """
    def norms(s):
        return np.sqrt((1 + np.abs(np.einsum('ij,ij->i', s[:, 0].conj(), s[:, 1])) ** 2) / 2)
    ln, rn = norms(left), norms(right)
    out = np.empty((len(left), len(right)))
    for start in range(0, len(left), block):
        end = min(start + block, len(left))
        part = np.zeros((end - start, len(right)))
        for a in range(2):
            for b in range(2):
                part += np.abs(left[start:end, a].conj() @ right[:, b].T) ** 2 / 4
        out[start:end] = part / (ln[start:end, None] * rn[None, :])
    return out


def symmetric_features(pairs):
    a, b = pairs[:, 0], pairs[:, 1]
    return np.concatenate((a + b, np.abs(a - b), a * b), axis=1)


def metric(y, scores, threshold=None):
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y, scores)
        finite = np.isfinite(thresholds)
        threshold = float(thresholds[finite][np.argmax((tpr - fpr)[finite])])
    pred = scores >= threshold
    return dict(accuracy=float(accuracy_score(y, pred)), auroc=float(roc_auc_score(y, scores)),
                auprc=float(average_precision_score(y, scores)), f1=float(f1_score(y, pred)),
                mcc=float(matthews_corrcoef(y, pred)), threshold=float(threshold), pairs=len(y))


def run(checkpoint_dir: Path, output_dir: Path, cap=1024, on_event=None):
    """Tune each kernel arm with nine validation trials, then score full test once."""
    checkpoint_dir, output_dir = Path(checkpoint_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)  # Never overwrite an experiment.
    template=circuit(np.asarray(list(ParameterVector('x',8)),dtype=object).reshape(2,4))
    (output_dir/'circuit.txt').write_text(str(template.draw('text',fold=100)),encoding='utf-8')
    started = time.time()
    def emit(event):
        row = dict(elapsed_seconds=round(time.time() - started, 2), **event)
        with (output_dir / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if on_event:
            on_event(row)
    origin = json.loads((checkpoint_dir / 'training_history.json').read_text())
    split = Path(origin['split'])
    split = split if split.is_absolute() else ROOT / split
    feature_file = Path(origin['feature_file'])
    feature_file = feature_file if feature_file.is_absolute() else ROOT / feature_file
    ckpt = torch.load(checkpoint_dir / 'best.pt', map_location='cpu', weights_only=False)
    model = SymmetricPairHead(ckpt['drug_feature_dim'], ckpt['width'], ckpt['dropout'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    with np.load(feature_file) as data:
        ids, raw = data['drug_ids'].astype(str), data['drug_features'].astype(np.float32)
    if raw.shape[1] != ckpt['drug_feature_dim']:
        raise ValueError(f'Checkpoint expects {ckpt["drug_feature_dim"]} input features, but its recorded archive now has {raw.shape[1]}. Restore the original archive or train a new encoder; do not silently truncate features.')
    lookup = {key: i for i, key in enumerate(ids)}
    train = pd.read_csv(split / 'train.csv.gz')
    val = pd.read_csv(split / 'val.csv.gz')
    train_ids = sorted(set(train.drug_id_a) | set(train.drug_id_b))
    drug_rows = [lookup[x] for x in train_ids]
    with torch.no_grad():
        embedding = model.phi(torch.from_numpy(raw)).numpy()
    pca = PCA(n_components=4, random_state=origin['seed']).fit(embedding[drug_rows])
    z = pca.transform(embedding)
    scaler = StandardScaler().fit(z[drug_rows])
    z = np.tanh(scaler.transform(z) / 2)  # Avoid the previous saturated raw-feature tanh.
    if cap < len(train):
        selected, _ = train_test_split(np.arange(len(train)), train_size=cap,
                                      stratify=train.label, random_state=origin['seed'])
        train = train.iloc[selected].reset_index(drop=True)
    def pair_inputs(frame):
        return np.stack((z[[lookup[x] for x in frame.drug_id_a]],
                         z[[lookup[x] for x in frame.drug_id_b]]), axis=1)
    xt, xv = pair_inputs(train), pair_inputs(val)
    yt, yv = train.label.to_numpy(), val.label.to_numpy()
    for name, frame in [('train', train), ('val', val)]:
        frame.to_csv(output_dir / f'{name}_pairs.csv.gz', index=False)
    provenance = dict(schema_version=1, split=str(split), split_seed=int(split.name.split('_')[-1]),
                      encoder_seed=origin['seed'], sampling_seed=origin['seed'],
                      checkpoint=str(checkpoint_dir), checkpoint_sha256=sha256(checkpoint_dir / 'best.pt'),
                      feature_file=str(feature_file),
                      features_sha256=sha256(feature_file), training_pairs=len(train), validation_pairs=len(val),
                      representation='Frozen feature MLP encoder; train-drug PCA and scaling',
                      encoder_training_pairs='full original training fold; all arms share this encoder',
                      pca_variance=float(pca.explained_variance_ratio_.sum()),
                      quantum_backend='Qiskit Statevector', qiskit_version=qiskit.__version__,
                      test_evaluation='full fold after validation-only selection',
                      split_hashes={name: sha256(split / f'{name}.csv.gz') for name in ['train','val','test']})
    (output_dir / 'protocol.json').write_text(json.dumps(provenance, indent=2))
    emit(dict(stage='prepared', **provenance))
    classic_scaler = StandardScaler().fit(symmetric_features(xt))
    ct, cv = (classic_scaler.transform(symmetric_features(x)) for x in (xt, xv))
    selected_models, trials = {}, []
    def trial(arm, estimator, train_x, val_x, config):
        estimator.fit(train_x, yt)
        score = estimator.decision_function(val_x)
        metrics = metric(yv, score)
        row = dict(arm=arm, config=config, validation=metrics,
                   solver_iterations=np.asarray(getattr(estimator, 'n_iter_', [])).tolist())
        trials.append(row)
        emit(dict(stage='validation_trial', **row))
        if arm not in selected_models or metrics['auprc'] > selected_models[arm]['validation']['auprc']:
            selected_models[arm] = dict(model=estimator, validation=metrics, config=config)
    for gamma in [0.05, 0.5, 5.0]:
        for c in [0.1, 1.0, 10.0]:
            trial('Classical RBF', SVC(C=c, gamma=gamma), ct, cv, dict(C=c, gamma=gamma))
    for c in [0.1, 1.0, 10.0]:
        trial('Classical logistic', LogisticRegression(C=c, max_iter=2000), ct, cv, dict(C=c))
    best_train_states = None
    for scale in [0.5, 1.0, 2.0]:
        emit(dict(stage='quantum_encoding', scale=scale))
        ts = states(xt, scale)
        vs = states(xv, scale, on_progress=lambda msg: emit(dict(stage='quantum_encoding', message=msg)))
        kt, kv = fidelity_kernel(ts, ts), fidelity_kernel(vs, ts)
        for c in [0.1, 1.0, 10.0]:
            trial('Quantum fidelity', SVC(C=c, kernel='precomputed'), kt, kv, dict(C=c, scale=scale))
        if selected_models['Quantum fidelity']['config']['scale'] == scale:
            best_train_states = ts
    quantum = selected_models['Quantum fidelity']
    scale, c = quantum['config']['scale'], quantum['config']['C']
    emit(dict(stage='ablation', message='Remove cross-register entanglers; keep selected C and scale fixed'))
    abt, abv = states(xt, scale, False), states(xv, scale, False)
    trial('Quantum no cross entanglers', SVC(C=c, kernel='precomputed'), fidelity_kernel(abt, abt),
          fidelity_kernel(abv, abt), dict(C=c, scale=scale, entangle=False, selection='matched ablation'))
    # Save selected fits before opening the test file.
    joblib.dump(dict(pca=pca, scaler=scaler, classic_scaler=classic_scaler,
                     selected_models=selected_models, train_states=best_train_states,
                     ablation_states=abt, checkpoint=str(checkpoint_dir)), output_dir / 'models.joblib')
    (output_dir / 'validation_trials.json').write_text(json.dumps(trials, indent=2))
    test = pd.read_csv(split / 'test.csv.gz')
    x_test, y_test = pair_inputs(test), test.label.to_numpy()
    test.to_csv(output_dir / 'test_pairs.csv.gz', index=False)
    ctest = classic_scaler.transform(symmetric_features(x_test))
    emit(dict(stage='final_test', pairs=len(test), message='Configuration selection finished; scoring the full test fold'))
    qs_test = states(x_test, scale, on_progress=lambda msg: emit(dict(stage='final_test', message=msg)))
    qa_test = states(x_test, scale, False, on_progress=lambda msg: emit(dict(stage='ablation_test', message=msg)))
    matrices = {'Classical RBF':ctest, 'Classical logistic':ctest,
                'Quantum fidelity':fidelity_kernel(qs_test, best_train_states),
                'Quantum no cross entanglers':fidelity_kernel(qa_test, abt)}
    results, predictions = [], test.copy()
    for arm, fit in selected_models.items():
        scores = fit['model'].decision_function(matrices[arm])
        predictions[arm + '_score'] = scores
        predictions[arm + '_prediction'] = (scores >= fit['validation']['threshold']).astype(int)
        results.append(dict(arm=arm, config=fit['config'], validation=fit['validation'],
                            test=metric(y_test, scores, fit['validation']['threshold'])))
    predictions.to_csv(output_dir / 'test_predictions.csv.gz', index=False)
    # Circuit diagnostics use validation inputs, so they cannot tune test performance.
    diagnostic_states = states(xv[:16], scale)
    gram = fidelity_kernel(diagnostic_states, diagnostic_states)
    swap_error = float(np.max(np.abs(gram - fidelity_kernel(diagnostic_states[:, ::-1], diagnostic_states))))
    entropies = [float(entropy(partial_trace(Statevector(s[0]), [4,5,6,7]), base=2)) for s in diagnostic_states]
    diagnostics = dict(swap_max_error=swap_error, minimum_gram_eigenvalue=float(np.linalg.eigvalsh(gram).min()),
                       gram_diagonal_max_error=float(np.abs(np.diag(gram)-1).max()),
                       mean_register_entanglement_bits=float(np.mean(entropies)),
                       diagnostic_validation_pairs=16)
    final = dict(**provenance, test_pairs=len(test), results=results, diagnostics=diagnostics,
                 trials=trials, elapsed_seconds=round(time.time()-started, 2),
                 limitations=['Feature MLP encoder, not the planned heterogeneous GNN.',
                              'Encoder and heads use the same validation fold for selection; results are exploratory.',
                              'Head training is capped; upstream encoder saw all training pairs.',
                              'No quantum speed advantage or hardware execution is claimed.',
                              'Single-cohort pair scores do not establish superiority across unseen drugs.'])
    (output_dir / 'comparison.json').write_text(json.dumps(final, indent=2))
    emit(dict(stage='complete', results=results, diagnostics=diagnostics))
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cap', type=int, default=1024)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.checkpoint_dir, args.output_dir, args.cap)
