"""Audit generated DDInter splits for pair, label, cohort and sampling integrity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARTS = ("train", "val", "test")


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def audit_one(split_dir: Path, processed_dir: Path) -> dict:
    meta = json.loads((split_dir / "metadata.json").read_text(encoding="utf-8"))
    setting = str(meta["setting"])
    known_pairs_frame = pd.read_csv(processed_dir / "ddi_positive_pairs.csv", dtype=str)
    known_positives = {pair_key(a, b) for a, b in known_pairs_frame[["drug_id_a", "drug_id_b"]].itertuples(index=False, name=None)}
    positive_path_hash = hashlib.sha256((processed_dir / "ddi_positive_pairs.csv").read_bytes()).hexdigest()
    if meta.get("positive_input_sha256") != positive_path_hash:
        raise ValueError(f"{split_dir}: positive source checksum differs from split manifest")

    frames = {}
    all_pairs: dict[tuple[str, str], str] = {}
    counts = {}
    for part in PARTS:
        frame = pd.read_csv(split_dir / f"{part}.csv.gz", dtype={"drug_id_a": str, "drug_id_b": str})
        if frame.empty or not set(frame.label.unique()).issubset({0, 1}):
            raise ValueError(f"{split_dir}/{part}: empty split or unexpected labels")
        if not (frame.drug_id_a < frame.drug_id_b).all():
            raise ValueError(f"{split_dir}/{part}: pair endpoints are not in canonical order")
        keys = [pair_key(a, b) for a, b in frame[["drug_id_a", "drug_id_b"]].itertuples(index=False, name=None)]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{split_dir}/{part}: duplicate pair within split")
        for key, label in zip(keys, frame.label):
            if key in all_pairs:
                raise ValueError(f"{split_dir}: pair appears in both {all_pairs[key]} and {part}")
            all_pairs[key] = part
            if int(label) == 1 and key not in known_positives:
                raise ValueError(f"{split_dir}/{part}: positive label absent from source positive table")
            if int(label) == 0 and key in known_positives:
                raise ValueError(f"{split_dir}/{part}: known positive incorrectly sampled as unlabeled negative")
        counts[part] = {"pairs": int(len(frame)), "positive": int((frame.label == 1).sum()),
                        "unlabeled": int((frame.label == 0).sum())}
        if counts[part]["positive"] != counts[part]["unlabeled"]:
            raise ValueError(f"{split_dir}/{part}: expected balanced positive/unlabeled sample")
        frames[part] = frame

    cohorts = meta.get("cohorts", {})
    if setting == "S3":
        groups = {"train": set(cohorts["train_drugs"]), "val": set(cohorts["val_heldout_drugs"]),
                  "test": set(cohorts["test_heldout_drugs"])}
        if any(groups[a] & groups[b] for i, a in enumerate(PARTS) for b in PARTS[i + 1:]):
            raise ValueError(f"{split_dir}: S3 drug cohorts overlap")
        for part, frame in frames.items():
            allowed = groups[part]
            if not (frame.drug_id_a.isin(allowed) & frame.drug_id_b.isin(allowed)).all():
                raise ValueError(f"{split_dir}/{part}: S3 pair is outside its assigned drug cohort")
    elif setting == "S2":
        visible = set(cohorts["train_visible_drugs"])
        train_cohort = set(cohorts["train_drugs"])
        held = {"val": set(cohorts["val_heldout_drugs"]), "test": set(cohorts["test_heldout_drugs"])}
        if train_cohort & held["val"] or train_cohort & held["test"] or held["val"] & held["test"]:
            raise ValueError(f"{split_dir}: S2 drug cohorts overlap")
        for part, frame in frames.items():
            a_train, b_train = frame.drug_id_a.isin(visible), frame.drug_id_b.isin(visible)
            if part == "train":
                valid = frame.drug_id_a.isin(train_cohort) & frame.drug_id_b.isin(train_cohort)
            else:
                a_held, b_held = frame.drug_id_a.isin(held[part]), frame.drug_id_b.isin(held[part])
                valid = (a_train & b_held) | (b_train & a_held)
            if not valid.all():
                raise ValueError(f"{split_dir}/{part}: pair violates S2 seen/unseen drug roles")

    return {"setting": setting, "split_seed": int(meta["seed"]), "status": "passed",
            "checks": ["source checksum", "pair uniqueness and canonical order", "known positives never labeled negative",
                       "1:1 unlabeled sampling", "cross-split pair separation", "cold-start drug roles"],
            "counts": counts, "pairs_audited": int(len(all_pairs))}


def audit_all(splits_dir: Path, processed_dir: Path, settings: list[str], seeds: list[int]) -> list[dict]:
    reports = []
    for setting in settings:
        for seed in seeds:
            reports.append(audit_one(splits_dir / setting / f"seed_{seed:02d}", processed_dir))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--settings", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = audit_all(args.splits_dir, args.processed_dir, args.settings, args.seeds)
    result = {"status": "passed", "split_count": len(reports), "reports": reports}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="", flush=True)


if __name__ == "__main__":
    main()
