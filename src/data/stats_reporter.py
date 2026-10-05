"""
stats_reporter.py – Dataset statistics report generator.

After cleaning, this module computes and saves the summary statistics required
by plan.md §2.2 step 4:
  • Number of drugs (total, by source)
  • Number of positive DDI pairs (total, by severity level)
  • Drug degree distribution (how many interactions per drug)
  • Targets/enzymes per drug
  • Source coverage (% of drugs with each cross-reference)

The report is saved to:
  • ``data/processed/dataset_statistics.json``  (machine-readable)
  • ``data/processed/dataset_statistics.md``    (human-readable Markdown table)

NOTE: This module only defines the reporting functions.  No data is read
or written until the pipeline runner calls ``generate_stats_report``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_drug_stats(drugs_df: pd.DataFrame) -> dict:
    """Compute summary statistics for the drug table."""
    stats: dict = {
        "total_drugs": int(len(drugs_df)),
        "valid_smiles": int(drugs_df.get("is_valid_smiles", pd.Series(dtype=bool)).sum()),
        "by_source": {},
        "cross_reference_coverage": {},
    }

    if "source" in drugs_df.columns:
        stats["by_source"] = drugs_df["source"].value_counts().to_dict()

    for id_col in ["drugbank_id", "chembl_id", "pubchem_cid", "pharmgkb_id", "ddinter_id"]:
        if id_col in drugs_df.columns:
            n = int(drugs_df[id_col].notna().sum())
            pct = round(100 * n / max(len(drugs_df), 1), 1)
            stats["cross_reference_coverage"][id_col] = {"n": n, "pct": pct}

    return stats


def compute_ddi_stats(pairs_df: pd.DataFrame) -> dict:
    """Compute summary statistics for the DDI pairs table."""
    stats: dict = {
        "total_positive_pairs": int(len(pairs_df)),
        "by_level": {},
        "unique_drugs_involved": 0,
        "degree_distribution": {},
    }

    if "level" in pairs_df.columns:
        stats["by_level"] = pairs_df["level"].value_counts().to_dict()

    if "drug_id_a" in pairs_df.columns and "drug_id_b" in pairs_df.columns:
        all_drugs = pd.concat([pairs_df["drug_id_a"], pairs_df["drug_id_b"]])
        stats["unique_drugs_involved"] = int(all_drugs.nunique())
        degree = all_drugs.value_counts()
        stats["degree_distribution"] = {
            "min":    int(degree.min()),
            "max":    int(degree.max()),
            "mean":   round(float(degree.mean()), 2),
            "median": float(degree.median()),
            "p25":    float(degree.quantile(0.25)),
            "p75":    float(degree.quantile(0.75)),
        }

    return stats


def compute_target_stats(targets_df: pd.DataFrame) -> dict:
    """Compute summary statistics for the drug-target table."""
    stats: dict = {"total_edges": int(len(targets_df))}

    if "drug_id" in targets_df.columns:
        stats["unique_drugs_with_targets"] = int(targets_df["drug_id"].nunique())
        per_drug = targets_df.groupby("drug_id")["uniprot_id"].nunique()
        stats["targets_per_drug"] = {
            "mean": round(float(per_drug.mean()), 2),
            "max":  int(per_drug.max()),
        }

    return stats


def generate_stats_report(
    drugs_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    targets_df: Optional[pd.DataFrame] = None,
    *,
    out_dir: Path,
) -> dict:
    """Compute and save the full dataset statistics report.

    Parameters
    ----------
    drugs_df:
        Cleaned drugs table.
    pairs_df:
        Cleaned DDI pairs table.
    targets_df:
        Cleaned drug-targets table (optional; if None, target stats are skipped).
    out_dir:
        Directory to write ``dataset_statistics.json`` and ``.md``.

    Returns
    -------
    dict
        The full statistics dictionary.
    """
    report = {
        "drugs":   compute_drug_stats(drugs_df),
        "ddi":     compute_ddi_stats(pairs_df),
    }
    if targets_df is not None:
        report["targets"] = compute_target_stats(targets_df)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = out_dir / "dataset_statistics.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("[stats] Saved JSON report → %s", json_path)

    # ── Markdown ─────────────────────────────────────────────────────────────
    md_lines = ["# Dataset Statistics Report\n"]
    md_lines += _dict_to_md_table(report["drugs"],  "Drug Table")
    md_lines += _dict_to_md_table(report["ddi"],    "DDI Pairs Table")
    if "targets" in report:
        md_lines += _dict_to_md_table(report["targets"], "Drug-Target Table")

    md_path = out_dir / "dataset_statistics.md"
    md_path.write_text("\n".join(md_lines))
    logger.info("[stats] Saved Markdown report → %s", md_path)

    return report


def _dict_to_md_table(d: dict, title: str) -> list[str]:
    """Convert a flat/nested dict to a simple Markdown section."""
    lines = [f"\n## {title}\n", "| Key | Value |", "|---|---|"]
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                lines.append(f"| {k}.{k2} | {v2} |")
        else:
            lines.append(f"| {k} | {v} |")
    return lines
