"""
cleaner.py – Data cleaning operations (SMILES validation, deduplication, etc.)

CONTRACT:
  • Receives DataFrames that have already been loaded (loaders.py) and had
    schemas validated (schema_validator.py).
  • Returns cleaned DataFrames.  Does NOT write anything to disk.
  • All cleaning decisions are documented with the rationale comment above
    each function.

WHY CLEAN BEFORE BUILDING THE GRAPH?
  Garbage-in → garbage-out.  Molecular graphs built from invalid SMILES
  will crash RDKit and produce nonsensical features.  Duplicate drug records
  inflate node counts and create spurious edges.  Self-pairs are logically
  undefined in a pairwise interaction task.  All these issues are caught and
  resolved here before any downstream step sees the data.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from .harmoniser import MappingLog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SMILES cleaning
# ---------------------------------------------------------------------------

def canonicalize_smiles_rdkit(
    drugs_df: pd.DataFrame,
    *,
    smiles_col: str = "smiles_raw",
    out_col: str = "smiles_canon",
    valid_flag_col: str = "is_valid_smiles",
) -> pd.DataFrame:
    """Canonicalize SMILES using RDKit and flag invalid entries.

    WHY:
      Different databases may represent the same molecule with different but
      chemically equivalent SMILES strings (e.g. atom ordering, aromaticity
      notation).  RDKit canonicalization produces a unique SMILES for each
      molecule so that duplicate drugs can be detected by string equality.

    HOW:
      1. Pass raw SMILES through ``Chem.MolFromSmiles`` (sanitization enabled).
      2. If sanitization fails → ``is_valid_smiles = False``, ``smiles_canon = NaN``.
      3. If it succeeds    → ``is_valid_smiles = True``,  ``smiles_canon`` = RDKit
         canonical SMILES from ``Chem.MolToSmiles``.

    NOTE: RDKit is imported inside the function so that the module can be
    imported and the schema/constants tested even without RDKit installed.

    Parameters
    ----------
    drugs_df:
        Drug table with at least a ``smiles_raw`` column.
    smiles_col:
        Column containing raw SMILES strings.
    out_col:
        Output column for canonical SMILES.
    valid_flag_col:
        Output boolean flag column.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        rdkit_available = True
    except ImportError:
        logger.warning(
            "RDKit not installed.  SMILES canonicalization will be skipped.  "
            "All drugs will be marked is_valid_smiles=None (unknown)."
        )
        rdkit_available = False

    df = drugs_df.copy()

    if not rdkit_available:
        df[out_col] = pd.NA
        df[valid_flag_col] = pd.NA
        return df

    def _process(smi: Optional[str]):
        if pd.isna(smi) or str(smi).strip() == "":
            return pd.NA, False
        try:
            mol = Chem.MolFromSmiles(str(smi).strip(), sanitize=True)
            if mol is None:
                return pd.NA, False
            return Chem.MolToSmiles(mol, canonical=True), True
        except Exception:
            return pd.NA, False

    results = df[smiles_col].apply(_process)
    df[out_col]        = results.apply(lambda x: x[0])
    df[valid_flag_col] = results.apply(lambda x: x[1])

    n_valid   = df[valid_flag_col].sum()
    n_invalid = (~df[valid_flag_col]).sum()
    logger.info(
        "[cleaner] SMILES canonicalization: %d valid, %d invalid (will be dropped later).",
        n_valid, n_invalid,
    )
    return df


def compute_mol_properties(
    drugs_df: pd.DataFrame,
    *,
    smiles_col: str = "smiles_canon",
) -> pd.DataFrame:
    """Compute basic molecular properties from canonical SMILES via RDKit.

    WHY:
      Molecular weight, atom count, and ring count are used in dataset
      statistics reports (plan.md §2.2 step 4) and as sanity-check features.
      Computing them here keeps the data provenance clear.

    Properties computed:
      • ``mol_weight``  – exact molecular weight (Da)
      • ``num_atoms``   – heavy atom count
      • ``num_rings``   – ring count (SSSR)

    Parameters
    ----------
    drugs_df:
        Drug table with ``smiles_canon`` and ``is_valid_smiles`` columns.
    smiles_col:
        Column with canonical SMILES.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        rdkit_available = True
    except ImportError:
        logger.warning("RDKit not installed.  Molecular properties will be NaN.")
        rdkit_available = False

    df = drugs_df.copy()

    if not rdkit_available:
        df["mol_weight"] = float("nan")
        df["num_atoms"]  = pd.NA
        df["num_rings"]  = pd.NA
        return df

    def _props(smi):
        if pd.isna(smi):
            return float("nan"), pd.NA, pd.NA
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return float("nan"), pd.NA, pd.NA
            mw    = Descriptors.ExactMolWt(mol)
            natom = mol.GetNumHeavyAtoms()
            nring = rdMolDescriptors.CalcNumRings(mol)
            return mw, natom, nring
        except Exception:
            return float("nan"), pd.NA, pd.NA

    props = df[smiles_col].apply(_props)
    df["mol_weight"] = props.apply(lambda x: x[0])
    df["num_atoms"]  = props.apply(lambda x: x[1])
    df["num_rings"]  = props.apply(lambda x: x[2])

    logger.info("[cleaner] Molecular properties computed for %d drugs.", len(df))
    return df


def drop_invalid_smiles_rows(
    drugs_df: pd.DataFrame,
    log: Optional[MappingLog] = None,
    *,
    valid_flag_col: str = "is_valid_smiles",
) -> pd.DataFrame:
    """Remove drugs where ``is_valid_smiles`` is False.

    WHY:
      Drugs without valid SMILES cannot have molecular graphs built for them.
      They cannot contribute to the heterogeneous graph and would produce
      NaN features that poison the GNN.  They are removed after canonicalization
      so the removal is logged with full audit detail.

    Parameters
    ----------
    drugs_df:
        Drug table with ``is_valid_smiles`` column.
    log:
        Optional MappingLog to record statistics.
    valid_flag_col:
        Name of the boolean flag column set by ``canonicalize_smiles_rdkit``.
    """
    n_before = len(drugs_df)
    df = drugs_df[drugs_df[valid_flag_col] == True].reset_index(drop=True)
    n_after = len(df)

    logger.info(
        "[cleaner] drop_invalid_smiles: %d → %d (dropped %d).",
        n_before, n_after, n_before - n_after,
    )
    if log:
        log.record(
            step="drop_invalid_smiles",
            n_before=n_before,
            n_after=n_after,
            notes=f"Dropped {n_before - n_after} drugs with invalid/missing SMILES",
        )
    return df


def deduplicate_drugs_by_inchikey(
    drugs_df: pd.DataFrame,
    log: Optional[MappingLog] = None,
    *,
    inchikey_col: str = "inchi_key",
) -> pd.DataFrame:
    """Deduplicate drugs by InChIKey, keeping the row with the most IDs filled.

    WHY:
      After harmonisation, the same molecule may appear from multiple sources
      with different records.  InChIKey is a structure-derived fingerprint that
      is the same regardless of source.  We merge records for the same InChIKey
      into one row, preferring the record that has more cross-reference IDs
      filled in (e.g., a record that has both a ChEMBL ID and a PubChem CID is
      preferred over one that has only a ChEMBL ID).

    Parameters
    ----------
    drugs_df:
        Drug table with ``inchi_key`` column (non-null for valid-SMILES drugs).
    log:
        Optional MappingLog to record statistics.
    inchikey_col:
        Name of the InChIKey column.
    """
    n_before = len(drugs_df)

    # Score each row by how many ID columns are non-null
    id_cols = ["drugbank_id", "chembl_id", "pubchem_cid", "pharmgkb_id", "ddinter_id"]
    id_cols_present = [c for c in id_cols if c in drugs_df.columns]
    df = drugs_df.copy()
    df["_id_score"] = df[id_cols_present].notna().sum(axis=1)

    # Sort so higher-score rows come first, then deduplicate keeping first
    df = (
        df.sort_values("_id_score", ascending=False)
        .drop_duplicates(subset=[inchikey_col], keep="first")
        .drop(columns=["_id_score"])
        .reset_index(drop=True)
    )
    n_after = len(df)

    logger.info(
        "[cleaner] deduplicate_by_inchikey: %d → %d (dropped %d duplicates).",
        n_before, n_after, n_before - n_after,
    )
    if log:
        log.record(
            step="deduplicate_drugs_by_inchikey",
            n_before=n_before,
            n_after=n_after,
            notes="Kept record with most cross-reference IDs per InChIKey",
        )
    return df


def normalize_drug_names(
    drugs_df: pd.DataFrame,
    *,
    name_col: str = "name",
) -> pd.DataFrame:
    """Lowercase, strip whitespace, and normalize internal spaces in drug names.

    WHY:
      Name-based fuzzy matching (the last-resort harmonisation step) is sensitive
      to capitalisation and extra whitespace.  Normalizing names ensures that
      "Aspirin", "ASPIRIN", and " aspirin " all compare as equal.

    Parameters
    ----------
    drugs_df:
        Drug table with a ``name`` column.
    name_col:
        Column containing drug names.
    """
    df = drugs_df.copy()
    df[name_col] = (
        df[name_col]
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    logger.info("[cleaner] Drug names normalized.")
    return df


def normalize_interaction_level(
    pairs_df: pd.DataFrame,
    *,
    level_col: str = "Level",
    out_col: str = "level",
) -> pd.DataFrame:
    """Standardize interaction severity labels to Title Case.

    WHY:
      DDInter uses "Major", "Moderate", "Minor".  A future data source might
      use "MAJOR" or "major".  Normalizing to Title Case ensures consistency.

    Parameters
    ----------
    pairs_df:
        DDI pairs DataFrame with a ``Level`` column.
    level_col:
        Input column name.
    out_col:
        Output column name (canonical "level").
    """
    valid_levels = {"Major", "Moderate", "Minor"}

    df = pairs_df.copy()
    df[out_col] = df[level_col].str.strip().str.capitalize()

    # Map non-standard values to "Unknown"
    non_standard = ~df[out_col].isin(valid_levels)
    if non_standard.any():
        n = non_standard.sum()
        logger.warning(
            "[cleaner] normalize_interaction_level: %d rows with non-standard levels "
            "mapped to 'Unknown'. Values: %s",
            n, df.loc[non_standard, out_col].unique().tolist(),
        )
        df.loc[non_standard, out_col] = "Unknown"

    return df
