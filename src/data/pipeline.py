"""
pipeline.py – Orchestrates the full data-cleaning and preprocessing pipeline.

This is the single entry point for Step 2 (Data collection and cleaning) and
Step 3 (Splits and negative sampling) of plan.md.

PIPELINE STAGES
───────────────
Stage 0  │ Load raw files + validate schemas  (loaders.py)
Stage 1  │ ID harmonisation                   (harmoniser.py)
Stage 2  │ SMILES canonicalization + props    (cleaner.py)
Stage 3  │ Deduplication + self-pair removal  (cleaner.py + harmoniser.py)
Stage 4  │ Dataset statistics report          (stats_reporter.py)
Stage 5  │ Save processed tables to disk      (Parquet + CSV)

USAGE
─────
  python -m src.data.pipeline --raw-dir data/raw --out-dir data/processed

The pipeline does NOT modify any files under data/raw/.
It writes ONLY to data/processed/ and logs/.

DRY-RUN MODE
────────────
  python -m src.data.pipeline --dry-run

In dry-run mode the pipeline loads and validates every raw file it can find,
logs what it would do, and exits without writing any output.  This is how you
verify your raw data downloads are correct before committing to the full run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage helpers (stubs until raw data is present)
# ---------------------------------------------------------------------------

def stage0_load_raw(raw_dir: Path, dry_run: bool = False) -> dict:
    """Load and validate all raw data sources.

    Returns a dict of source_key → DataFrame for every file that is present.
    Missing optional files are logged as warnings, not errors.
    Missing required files raise FileNotFoundError.
    """
    from .loaders import (
        load_ddinter,
        load_chembl_drugs,
        load_pubchem_mapping,
        load_unichem_mapping,
        load_pharmgkb_drugs,
        load_sider_side_effects,
        load_string_ppi,
        load_uniprot_proteins,
    )

    sources: dict = {}

    # ── Required sources ─────────────────────────────────────────────────────
    logger.info("[stage0] Loading DDInter …")
    sources["ddinter"] = load_ddinter(raw_dir / "ddinter")

    # ── Optional sources (warn if absent) ───────────────────────────────────
    optional_loaders = {
        "chembl_drugs":      (load_chembl_drugs,        raw_dir / "chembl"),
        "pubchem_mapping":   (load_pubchem_mapping,     raw_dir / "pubchem"),
        "unichem_mapping":   (load_unichem_mapping,     raw_dir / "unichem"),
        "pharmgkb_drugs":    (load_pharmgkb_drugs,      raw_dir / "pharmgkb"),
        "sider_se":          (load_sider_side_effects,  raw_dir / "sider"),
        "string_ppi":        (load_string_ppi,          raw_dir / "string"),
        "uniprot_proteins":  (load_uniprot_proteins,    raw_dir / "uniprot"),
    }

    for key, (loader_fn, source_dir) in optional_loaders.items():
        try:
            sources[key] = loader_fn(source_dir)
            logger.info("[stage0] ✓ %s loaded.", key)
        except FileNotFoundError as e:
            logger.warning("[stage0] MISSING %s — skipping.  (%s)", key, e)
        except Exception as e:
            logger.error("[stage0] ERROR loading %s: %s", key, e, exc_info=True)
            raise

    if dry_run:
        logger.info("[stage0] DRY-RUN complete.  %d sources loaded.", len(sources))
    return sources


def stage1_harmonise(sources: dict, dry_run: bool = False) -> tuple[dict, object]:
    """Build unified ID mapping and enrich drug records across sources.

    Returns
    -------
    enriched_sources:
        Updated sources dict with cross-reference columns added.
    mapping_log:
        MappingLog recording all harmonisation statistics.
    """
    from .harmoniser import MappingLog, build_inchikey_index, build_chembl_cid_index

    log = MappingLog()

    inchikey_index: dict = {}
    chembl_cid_index: dict = {}

    if "chembl_drugs" in sources:
        inchikey_index = build_inchikey_index(sources["chembl_drugs"])

    if "pubchem_mapping" in sources:
        chembl_cid_index = build_chembl_cid_index(sources["pubchem_mapping"])

    logger.info("[stage1] Harmonisation indices built.  InChIKey: %d, ChEMBL→CID: %d",
                len(inchikey_index), len(chembl_cid_index))

    if dry_run:
        logger.info("[stage1] DRY-RUN: harmonisation step prepared but not executed.")
        return sources, log

    # Actual harmonisation runs here (to be implemented in a later commit
    # once all raw files are downloaded and enriched with InChIKeys).
    logger.warning(
        "[stage1] Full harmonisation requires all raw files.  "
        "Returning sources unchanged until ChEMBL/PubChem data is present."
    )
    return sources, log


def stage2_clean(sources: dict, dry_run: bool = False) -> tuple[dict, object]:
    """Canonicalize SMILES, compute molecular properties, flag invalid drugs."""
    from .harmoniser import MappingLog
    from .cleaner import (
        canonicalize_smiles_rdkit,
        compute_mol_properties,
        normalize_drug_names,
        normalize_interaction_level,
    )

    log = MappingLog()

    if dry_run:
        logger.info("[stage2] DRY-RUN: cleaning step defined but not executed.")
        return sources, log

    # Apply to DDInter (which provides drug names, not SMILES directly)
    if "ddinter" in sources:
        sources["ddinter"] = normalize_interaction_level(sources["ddinter"])

    # Apply SMILES cleaning to ChEMBL (primary SMILES source)
    if "chembl_drugs" in sources:
        sources["chembl_drugs"] = canonicalize_smiles_rdkit(sources["chembl_drugs"],
                                                            smiles_col="smiles")
        sources["chembl_drugs"] = compute_mol_properties(sources["chembl_drugs"],
                                                         smiles_col="smiles_canon")
        sources["chembl_drugs"] = normalize_drug_names(sources["chembl_drugs"],
                                                        name_col="pref_name")

    return sources, log


def stage3_deduplicate(sources: dict, log: object, dry_run: bool = False) -> dict:
    """Deduplicate drugs and DDI pairs, remove self-pairs."""
    from .harmoniser import deduplicate_pairs, drop_self_pairs, enforce_canonical_pair_order
    from .cleaner import drop_invalid_smiles_rows, deduplicate_drugs_by_inchikey

    if dry_run:
        logger.info("[stage3] DRY-RUN: deduplication step defined but not executed.")
        return sources

    logger.info("[stage3] Deduplication step will run after harmonisation produces "
                "a unified drug table.  Skipping for now.")
    return sources


def stage4_stats(processed: dict, out_dir: Path, dry_run: bool = False) -> None:
    """Generate and save the dataset statistics report."""
    from .stats_reporter import generate_stats_report

    if dry_run:
        logger.info("[stage4] DRY-RUN: stats report step defined but not executed.")
        return

    logger.info("[stage4] Stats report will run after all cleaning steps complete.")


def stage5_save(processed: dict, out_dir: Path, dry_run: bool = False) -> None:
    """Save cleaned DataFrames to Parquet/CSV in data/processed/."""
    if dry_run:
        logger.info("[stage5] DRY-RUN: save step defined but not executed.")
        return

    logger.info("[stage5] Save step will run after all cleaning steps complete.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_dir: Path,
    out_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    """Execute the full data preprocessing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir.parent.parent / "logs" / "pipeline.log"),
        ],
    )

    logger.info("=" * 70)
    logger.info("DDI Preprocessing Pipeline  |  dry_run=%s", dry_run)
    logger.info("  raw_dir : %s", raw_dir)
    logger.info("  out_dir : %s", out_dir)
    logger.info("=" * 70)

    # Stage 0: Load
    sources = stage0_load_raw(raw_dir, dry_run=dry_run)

    # Stage 1: Harmonise
    sources, mapping_log = stage1_harmonise(sources, dry_run=dry_run)

    # Stage 2: Clean
    sources, clean_log = stage2_clean(sources, dry_run=dry_run)

    # Stage 3: Deduplicate
    sources = stage3_deduplicate(sources, clean_log, dry_run=dry_run)

    # Stage 4: Stats
    stage4_stats(sources, out_dir, dry_run=dry_run)

    # Stage 5: Save
    stage5_save(sources, out_dir, dry_run=dry_run)

    logger.info("Pipeline finished.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DDI data preprocessing pipeline (plan.md Steps 2–3)."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to the raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to the output directory (default: data/processed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate schemas and log intentions without writing any output.",
    )
    args = parser.parse_args()
    run_pipeline(args.raw_dir, args.out_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
