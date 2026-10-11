"""Create reproducible DDInter pair splits and uniformly sampled unlabeled pairs.

Unlisted pairs are sampled as *unlabeled negatives* for the benchmark; they are
not claims that the drugs do not interact. Cold-start splits deliberately leave
cross-cohort positive pairs unused to keep the evaluation contract unambiguous.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _positive_edges(pairs: pd.DataFrame) -> list[tuple[str, str]]:
    return [(_pair_key(str(a), str(b))) for a, b in pairs[["drug_id_a", "drug_id_b"]].itertuples(index=False, name=None)]


def _split_edges(edges: list[tuple[str, str]], seed: int) -> dict[str, list[tuple[str, str]]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(edges))
    n_train, n_val = int(len(edges) * .70), int(len(edges) * .10)
    return {"train": [edges[i] for i in order[:n_train]],
            "val": [edges[i] for i in order[n_train:n_train+n_val]],
            "test": [edges[i] for i in order[n_train+n_val:]]}


def _cold_edges(edges: list[tuple[str, str]], seed: int, setting: str) -> tuple[dict[str, list[tuple[str, str]]], dict]:
    rng = np.random.default_rng(seed)
    drugs = sorted({d for edge in edges for d in edge})
    order = list(np.asarray(drugs, dtype=object)[rng.permutation(len(drugs))])
    n_train, n_val = int(.70 * len(order)), int(.10 * len(order))
    train_drugs, val_drugs, test_drugs = set(order[:n_train]), set(order[n_train:n_train+n_val]), set(order[n_train+n_val:])
    folds = {"train": [], "val": [], "test": []}
    dropped = 0
    if setting == "S2":
        folds["train"] = [(a, b) for a, b in edges if a in train_drugs and b in train_drugs]
        visible = {d for edge in folds["train"] for d in edge}
        for a, b in edges:
            if a in train_drugs and b in train_drugs:
                continue
            if (a in val_drugs and b in visible) or (b in val_drugs and a in visible):
                folds["val"].append((a, b))
            elif (a in test_drugs and b in visible) or (b in test_drugs and a in visible):
                folds["test"].append((a, b))
            else:
                dropped += 1
        return folds, {"train_drugs": sorted(train_drugs), "train_visible_drugs": sorted(visible),
                       "val_heldout_drugs": sorted(val_drugs), "test_heldout_drugs": sorted(test_drugs),
                       "positive_pairs_unused_for_cohort_rules": dropped}
    for a, b in edges:
        # S3: both endpoints belong to the same disjoint cohort.
        if a in train_drugs and b in train_drugs:
            folds["train"].append((a, b))
        elif a in val_drugs and b in val_drugs:
            folds["val"].append((a, b))
        elif a in test_drugs and b in test_drugs:
            folds["test"].append((a, b))
        else:
            dropped += 1
    return folds, {"train_drugs": sorted(train_drugs), "val_heldout_drugs": sorted(val_drugs),
                   "test_heldout_drugs": sorted(test_drugs), "positive_pairs_unused_for_cohort_rules": dropped}


def _sample_unlisted(positives: list[tuple[str, str]],
                     drug_universe: list[str], seed: int, setting: str,
                     split_name: str, cohorts: dict, forbidden_mask: np.ndarray) -> list[tuple[str, str]]:
    """Uniformly sample eligible unlabeled pairs without using evaluation labels."""
    if not positives:
        return []
    rng = np.random.default_rng(seed)
    # Candidate endpoints must be drawn from the legal cohort, not inferred
    # from positive test degrees. Using held-out positive degrees to construct
    # test negatives conditions the evaluation set on its answers.
    nodes = np.asarray(sorted(drug_universe), dtype=object)
    node_index = {drug: i for i, drug in enumerate(drug_universe)}
    active_index = np.fromiter((node_index[str(d)] for d in nodes), dtype=np.int64)
    if setting == "S1":
        roles = None
    elif setting == "S2":
        train = set(cohorts.get("train_visible_drugs", cohorts["train_drugs"]))
        held = set(cohorts["val_heldout_drugs"] if split_name == "val" else cohorts["test_heldout_drugs"]) if split_name != "train" else train
        roles = (train, held)
    else:
        group = "train_drugs" if split_name == "train" else ("val_heldout_drugs" if split_name == "val" else "test_heldout_drugs")
        cohort = set(cohorts[group])
        roles = (cohort, cohort)
    left_idx, right_idx = np.triu_indices(len(nodes), k=1)
    a_ids, b_ids = nodes[left_idx], nodes[right_idx]
    keep = np.ones(len(left_idx), dtype=bool)
    if roles is not None:
        left, right = roles
        left_indices = [node_index[d] for d in left]
        right_indices = [node_index[d] for d in right]
        a_codes, b_codes = active_index[left_idx], active_index[right_idx]
        a_left, b_left = np.isin(a_codes, left_indices), np.isin(b_codes, left_indices)
        a_right, b_right = np.isin(a_codes, right_indices), np.isin(b_codes, right_indices)
        keep &= (a_left & b_right) | (b_left & a_right)
    global_codes = active_index[left_idx] * len(drug_universe) + active_index[right_idx]
    keep &= ~forbidden_mask[global_codes]
    eligible = np.flatnonzero(keep)
    if len(eligible) < len(positives):
        raise RuntimeError(f"Only {len(eligible)} eligible unlabeled pairs for {len(positives)} requested")
    selected = rng.choice(eligible, size=len(positives), replace=False)
    return [(_pair_key(str(a_ids[i]), str(b_ids[i]))) for i in selected]


def make_splits(processed_dir: Path, splits_dir: Path, seeds: list[int]) -> None:
    pairs = pd.read_csv(processed_dir / "ddi_positive_pairs.csv", dtype=str)
    drugs = pd.read_csv(processed_dir / "drug_table.csv", dtype=str)
    edges = _positive_edges(pairs)
    all_positive = set(edges)
    drug_universe = sorted(drugs["drug_id"].astype(str).unique())
    if len(all_positive) != len(edges):
        raise ValueError("Positive pair input contains duplicates")
    digest = hashlib.sha256((processed_dir / "ddi_positive_pairs.csv").read_bytes()).hexdigest()
    node_index = {drug: i for i, drug in enumerate(drug_universe)}
    positive_mask = np.zeros(len(drug_universe) ** 2, dtype=bool)
    for a, b in all_positive:
        positive_mask[node_index[a] * len(drug_universe) + node_index[b]] = True
    splits_dir.mkdir(parents=True, exist_ok=True)
    for setting in ("S1", "S2", "S3"):
        for seed in seeds:
            forbidden_mask = positive_mask.copy()
            if setting == "S1":
                folds = _split_edges(edges, seed)
                cohort_meta = {}
            else:
                folds, cohort_meta = _cold_edges(edges, seed, setting)
            out = splits_dir / setting / f"seed_{seed:02d}"
            out.mkdir(parents=True, exist_ok=True)
            meta = {"setting": setting, "seed": seed, "positive_input_sha256": digest,
                    "split_contract": ("positive edge ratios 70/10/20" if setting == "S1" else
                                       "70/10/20 drug cohort assignment; edge counts vary and cross-cohort positives are omitted"),
                    "negative_sampling": "1:1 uniform sampling without replacement within legal fold cohorts; unlabeled pairs only",
                    "positive_pairs_unused_for_cohort_rules": cohort_meta.get("positive_pairs_unused_for_cohort_rules", 0),
                    "cohorts": cohort_meta}
            for split_name, pos in folds.items():
                neg = _sample_unlisted(pos, drug_universe,
                    seed * 101 + ("train", "val", "test").index(split_name), setting, split_name, cohort_meta, forbidden_mask)
                for a, b in neg:
                    forbidden_mask[node_index[a] * len(drug_universe) + node_index[b]] = True
                rows = [(a, b, 1) for a, b in pos] + [(a, b, 0) for a, b in neg]
                frame = pd.DataFrame(rows, columns=["drug_id_a", "drug_id_b", "label"])
                if len(frame):
                    frame = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
                frame.to_csv(out / f"{split_name}.csv.gz", index=False, compression="gzip")
                meta[split_name] = {"positive_pairs": len(pos), "unlabeled_pairs": len(neg), "examples": len(frame)}
            (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"{setting} seed {seed}: " + ", ".join(f"{k}={v['examples']}" for k, v in meta.items() if k in folds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    make_splits(args.processed_dir, args.splits_dir, args.seeds)


if __name__ == "__main__":
    main()
