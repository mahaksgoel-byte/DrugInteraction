"""Stages 2-3, and 10 (part of Preprocessing & Data Quality box) —
ID harmonization, data cleaning, class-balance check, train/val/test split
including a cold-start split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def harmonize_ids(sources: dict[str, pd.DataFrame], min_sources_required: int = 2) -> dict[str, pd.DataFrame]:
    """Stage 2 — unify every drug to a single canonical `drug_id` (here,
    the DrugBank-style id already used by our loaders/synthetic generator
    acts as the canonical join key, matching how DrugBank's own
    cross-reference table is used in production). Drop drugs that appear
    in fewer than `min_sources_required` source tables or have no SMILES.
    """
    chembl = sources["chembl_pubchem"]
    valid_smiles_ids = set(chembl.loc[chembl["smiles"].notna(), "drug_id"])

    presence: dict[str, set] = {
        "chembl_pubchem": set(chembl["drug_id"]),
        "pharmgkb": set(sources["pharmgkb"]["drug_id"]) if len(sources["pharmgkb"]) else set(),
        "sider": set(sources["sider"]["drug_id"]) if len(sources["sider"]) else set(),
    }
    ddi = sources["drugbank_ddi"]
    presence["drugbank_ddi"] = set(ddi["drug_a_id"]) | set(ddi["drug_b_id"])

    all_ids = set().union(*presence.values())
    counts = {d: sum(d in s for s in presence.values()) for d in all_ids}
    keep_ids = {
        d for d in all_ids if counts[d] >= min_sources_required and d in valid_smiles_ids
    }

    dropped = len(all_ids) - len(keep_ids)
    print(f"[preprocessing] ID harmonization: kept {len(keep_ids)}/{len(all_ids)} drugs "
          f"({dropped} dropped for <{min_sources_required} sources or missing SMILES).")

    filtered = dict(sources)
    filtered["chembl_pubchem"] = chembl[chembl["drug_id"].isin(keep_ids)].reset_index(drop=True)
    filtered["pharmgkb"] = sources["pharmgkb"][sources["pharmgkb"]["drug_id"].isin(keep_ids)].reset_index(drop=True)
    filtered["sider"] = sources["sider"][sources["sider"]["drug_id"].isin(keep_ids)].reset_index(drop=True)
    filtered["drugbank_ddi"] = ddi[
        ddi["drug_a_id"].isin(keep_ids) & ddi["drug_b_id"].isin(keep_ids)
    ].reset_index(drop=True)
    filtered["drugbank_targets"] = sources["drugbank_targets"][
        sources["drugbank_targets"]["drug_id"].isin(keep_ids)
    ].reset_index(drop=True)
    filtered["drugbank_enzymes"] = sources["drugbank_enzymes"][
        sources["drugbank_enzymes"]["drug_id"].isin(keep_ids)
    ].reset_index(drop=True)
    filtered["_kept_drug_ids"] = sorted(keep_ids)
    return filtered


def clean_data(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Stage: Data Cleaning — drop incomplete records, deduplicate."""
    cleaned = dict(sources)
    ddi = sources["drugbank_ddi"].dropna(subset=["drug_a_id", "drug_b_id", "interaction"])
    ddi = ddi[ddi["drug_a_id"] != ddi["drug_b_id"]]
    ddi = ddi.drop_duplicates(subset=["drug_a_id", "drug_b_id"])
    cleaned["drugbank_ddi"] = ddi.reset_index(drop=True)

    cleaned["chembl_pubchem"] = sources["chembl_pubchem"].dropna(subset=["drug_id", "smiles"]).drop_duplicates(
        subset=["drug_id"]
    ).reset_index(drop=True)
    cleaned["sider"] = sources["sider"].dropna().drop_duplicates().reset_index(drop=True)
    cleaned["pharmgkb"] = sources["pharmgkb"].drop_duplicates(subset=["drug_id"]).reset_index(drop=True)

    n_before, n_after = len(sources["drugbank_ddi"]), len(cleaned["drugbank_ddi"])
    print(f"[preprocessing] Data cleaning: DDI rows {n_before} -> {n_after} "
          f"({n_before - n_after} dropped as dupes/self-pairs/incomplete).")
    return cleaned


def class_balance_check(ddi: pd.DataFrame) -> dict[str, float]:
    """Stage: Class Balance Check — positive vs negative DDI ratio."""
    pos = int((ddi["interaction"] == 1).sum())
    neg = int((ddi["interaction"] == 0).sum())
    ratio = pos / max(neg, 1)
    print(f"[preprocessing] Class balance: {pos} positive / {neg} negative (ratio={ratio:.2f}).")
    return {"positive": pos, "negative": neg, "ratio": ratio}


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cold_start_test: pd.DataFrame


def train_val_test_split(
    ddi: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    cold_start_fraction: float = 0.1,
    seed: int = 42,
) -> Split:
    """Stage: Train/Val/Test Split — includes a cold-start split.

    Cold-start: a fraction of drugs are entirely withheld from training;
    every pair touching one of those drugs goes to `cold_start_test`,
    testing generalization to unseen drugs. The remaining pairs (both
    drugs "seen") are split normally into train/val/test.
    """
    rng = np.random.default_rng(seed)
    all_drugs = sorted(set(ddi["drug_a_id"]) | set(ddi["drug_b_id"]))
    n_cold = max(1, int(len(all_drugs) * cold_start_fraction))
    cold_drugs = set(rng.choice(all_drugs, size=n_cold, replace=False))

    is_cold = ddi["drug_a_id"].isin(cold_drugs) | ddi["drug_b_id"].isin(cold_drugs)
    cold_start_test = ddi[is_cold].reset_index(drop=True)
    warm = ddi[~is_cold].reset_index(drop=True)

    n = len(warm)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]

    split = Split(
        train=warm.iloc[train_idx].reset_index(drop=True),
        val=warm.iloc[val_idx].reset_index(drop=True),
        test=warm.iloc[test_idx].reset_index(drop=True),
        cold_start_test=cold_start_test,
    )
    print(
        f"[preprocessing] Split: train={len(split.train)} val={len(split.val)} "
        f"test={len(split.test)} cold_start_test={len(split.cold_start_test)} "
        f"({n_cold} drugs held out entirely)."
    )
    return split
