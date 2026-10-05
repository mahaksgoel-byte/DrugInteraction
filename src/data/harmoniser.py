"""
harmoniser.py – ID harmonisation across sources.

The goal of this module is to produce a single, unified drug identity table
that maps every drug seen in any source to a canonical (drug_id, chembl_id,
pubchem_cid, drugbank_id) tuple.

WHY THIS STEP:
  Different databases use different primary identifiers:
    • DDInter  → DDInterID  (e.g. "DDInter1")
    • ChEMBL   → CHEMBL ID  (e.g. "CHEMBL25")
    • PubChem  → CID        (e.g. "2244")
    • DrugBank → DB ID      (e.g. "DB00945")
    • PharmGKB → PA ID      (e.g. "PA449057")

  Without harmonisation, the same drug appears as multiple rows across sources,
  causing inflated drug counts, missed features, and broken graph edges.

STRATEGY (in order of reliability):
  1. InChIKey exact match  (structure-derived, source-agnostic)
  2. ChEMBL ID match       (stable, programmatic)
  3. PubChem CID match     (stable, numeric)
  4. DrugBank ID match     (via mapping files)
  5. Name-based fuzzy match (last resort – logged as "low-confidence")

Each step logs how many drugs were mapped and how many were lost.

NOTE: This module only PREPARES the mapping logic (data structures, functions,
      and logging scaffolding).  No actual data is read or written here;
      that happens when the pipeline runner calls these functions with loaded
      DataFrames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .schema_validator import SchemaError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MappingStats:
    """Records the number of drugs before/after each harmonisation step."""
    step: str
    n_before: int
    n_after: int
    notes: str = ""

    @property
    def n_lost(self) -> int:
        return self.n_before - self.n_after

    def log(self) -> None:
        logger.info(
            "[harmonise] %-40s  before=%6d  after=%6d  lost=%6d  %s",
            self.step, self.n_before, self.n_after, self.n_lost,
            f"({self.notes})" if self.notes else "",
        )


@dataclass
class MappingLog:
    """Accumulates MappingStats across all harmonisation steps."""
    entries: list[MappingStats] = field(default_factory=list)

    def record(self, step: str, n_before: int, n_after: int, notes: str = "") -> None:
        stat = MappingStats(step, n_before, n_after, notes)
        stat.log()
        self.entries.append(stat)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "step":     e.step,
                "n_before": e.n_before,
                "n_after":  e.n_after,
                "n_lost":   e.n_lost,
                "notes":    e.notes,
            }
            for e in self.entries
        ])


# ---------------------------------------------------------------------------
# Harmonisation steps (to be called in order by the pipeline runner)
# ---------------------------------------------------------------------------

def build_inchikey_index(chembl_df: pd.DataFrame) -> dict[str, str]:
    """Return {inchi_key → chembl_id} from the ChEMBL drugs table.

    This is the most reliable mapping method: InChIKey is structure-derived
    and source-agnostic.  Any source that provides InChIKey can be linked to
    ChEMBL without needing a pre-built cross-reference table.
    """
    valid = chembl_df.dropna(subset=["inchi_key", "chembl_id"])
    index = dict(zip(valid["inchi_key"].str.strip(), valid["chembl_id"].str.strip()))
    logger.info("[harmonise] InChIKey index: %d entries", len(index))
    return index


def build_chembl_cid_index(pubchem_df: pd.DataFrame) -> dict[str, str]:
    """Return {chembl_id → pubchem_cid} from the PubChem mapping table."""
    valid = pubchem_df.dropna(subset=["chembl_id", "pubchem_cid"])
    index = dict(zip(valid["chembl_id"].str.strip(), valid["pubchem_cid"].str.strip()))
    logger.info("[harmonise] ChEMBL→CID index: %d entries", len(index))
    return index


def map_ddinter_to_chembl(
    ddinter_df: pd.DataFrame,
    inchikey_index: dict[str, str],
    log: MappingLog,
) -> pd.DataFrame:
    """Attempt to map DDInter drugs to ChEMBL IDs via InChIKey.

    DDInter does NOT directly expose InChIKeys or ChEMBL IDs.  The recommended
    strategy (implemented in the download step) is:

      1. Extract unique drug names from DDInter.
      2. Query the ChEMBL API by name to retrieve SMILES + InChIKey.
      3. Join on InChIKey.

    This function receives the result of that query (ddinter_df extended with
    an ``inchi_key`` column from ChEMBL) and performs the join.

    Parameters
    ----------
    ddinter_df:
        DDInter raw DataFrame, expected to already have a ``inchi_key`` column
        added by the download/enrichment step.
    inchikey_index:
        Mapping {inchi_key → chembl_id} from ``build_inchikey_index``.
    log:
        MappingLog to record statistics.
    """
    n_before = ddinter_df["DDInterID_A"].nunique() + ddinter_df["DDInterID_B"].nunique()

    # Placeholder logic — actual implementation fills this in cleaning step
    # For now we just record the intent
    logger.info(
        "[harmonise] map_ddinter_to_chembl: ready to map %d drug mentions.",
        n_before,
    )
    log.record(
        step="ddinter_to_chembl_via_inchikey",
        n_before=n_before,
        n_after=n_before,  # placeholder — updated after actual mapping
        notes="placeholder: actual mapping runs after SMILES enrichment",
    )
    return ddinter_df  # returned unchanged until enrichment step is run


def drop_invalid_smiles(
    drugs_df: pd.DataFrame,
    log: MappingLog,
    *,
    smiles_col: str = "smiles_raw",
) -> pd.DataFrame:
    """Mark drugs with invalid SMILES (RDKit sanitization failed).

    We mark rather than drop here, so the full audit trail is preserved.
    The actual drop happens in cleaner.py after the ``is_valid_smiles`` flag
    is populated by RDKit.

    Parameters
    ----------
    drugs_df:
        Drug table with at least a ``smiles_raw`` column.
    log:
        MappingLog to record statistics.
    smiles_col:
        Name of the column containing raw SMILES strings.
    """
    n_before = len(drugs_df)
    missing_smiles = drugs_df[smiles_col].isna() | (drugs_df[smiles_col].str.strip() == "")
    n_missing = missing_smiles.sum()

    logger.info(
        "[harmonise] %d / %d drugs have no SMILES string (will be flagged).",
        n_missing, n_before,
    )
    log.record(
        step="flag_missing_smiles",
        n_before=n_before,
        n_after=n_before - n_missing,
        notes=f"{n_missing} drugs flagged is_valid_smiles=False due to missing SMILES",
    )
    # Return unchanged; is_valid_smiles flag set in cleaner.py
    return drugs_df


def enforce_canonical_pair_order(
    pairs_df: pd.DataFrame,
    col_a: str = "drug_id_a",
    col_b: str = "drug_id_b",
) -> pd.DataFrame:
    """Ensure drug_id_a < drug_id_b for every pair (sort within each row).

    Canonical ordering eliminates duplicates like (A, B) and (B, A) that refer
    to the same undirected interaction.  We sort within each row so the lower
    ID is always in col_a.

    Parameters
    ----------
    pairs_df:
        DDI pairs DataFrame with at least col_a and col_b columns.
    col_a, col_b:
        Column names for the two drug IDs.
    """
    df = pairs_df.copy()
    mask = df[col_a] > df[col_b]
    df.loc[mask, [col_a, col_b]] = df.loc[mask, [col_b, col_a]].values
    logger.info(
        "[harmonise] canonical pair order: swapped %d rows where %s > %s.",
        mask.sum(), col_a, col_b,
    )
    return df


def deduplicate_pairs(
    pairs_df: pd.DataFrame,
    col_a: str = "drug_id_a",
    col_b: str = "drug_id_b",
    log: Optional[MappingLog] = None,
) -> pd.DataFrame:
    """Remove duplicate (drug_id_a, drug_id_b) rows, keeping the first occurrence.

    After canonical ordering, any (A, B) and (B, A) duplicates are now
    identical rows and will be dropped here.

    Parameters
    ----------
    pairs_df:
        DDI pairs DataFrame (canonical order already applied).
    col_a, col_b:
        Column names for the two drug IDs.
    log:
        Optional MappingLog to record statistics.
    """
    n_before = len(pairs_df)
    df = pairs_df.drop_duplicates(subset=[col_a, col_b], keep="first").reset_index(drop=True)
    n_after = len(df)

    logger.info(
        "[harmonise] deduplicate_pairs: %d → %d (dropped %d duplicates).",
        n_before, n_after, n_before - n_after,
    )
    if log:
        log.record(
            step="deduplicate_pairs",
            n_before=n_before,
            n_after=n_after,
            notes=f"Kept first occurrence of each (drug_id_a, drug_id_b) pair",
        )
    return df


def drop_self_pairs(
    pairs_df: pd.DataFrame,
    col_a: str = "drug_id_a",
    col_b: str = "drug_id_b",
    log: Optional[MappingLog] = None,
) -> pd.DataFrame:
    """Remove rows where drug_id_a == drug_id_b (a drug interacting with itself)."""
    n_before = len(pairs_df)
    mask = pairs_df[col_a] != pairs_df[col_b]
    df = pairs_df[mask].reset_index(drop=True)
    n_dropped = n_before - len(df)

    logger.info("[harmonise] drop_self_pairs: removed %d self-pairs.", n_dropped)
    if log:
        log.record(
            step="drop_self_pairs",
            n_before=n_before,
            n_after=len(df),
            notes="Removed rows with drug_id_a == drug_id_b",
        )
    return df
