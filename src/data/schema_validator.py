"""
schema_validator.py – Pre-processing schema gate.

All raw loaders call ``validate_raw_schema`` BEFORE doing any transformation.
This acts as a canary: if an upstream data provider changes column names or
drops a field, the pipeline fails loudly here instead of silently producing
wrong output later.
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from .constants import SOURCE_SCHEMAS, CANONICAL_COLUMNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_raw_schema(
    df: pd.DataFrame,
    source_key: str,
    *,
    allow_extra_columns: bool = True,
    raise_on_empty: bool = True,
) -> None:
    """Assert that *df* contains at least the expected columns for *source_key*.

    Parameters
    ----------
    df:
        The raw DataFrame to validate.
    source_key:
        Key into ``SOURCE_SCHEMAS`` (e.g. ``"ddinter"``).
    allow_extra_columns:
        If *True* (default), extra columns in *df* are silently ignored.
        If *False*, raises ``SchemaError`` when unexpected columns are present.
    raise_on_empty:
        If *True* (default), raises ``SchemaError`` when *df* has 0 rows.
    """
    if source_key not in SOURCE_SCHEMAS:
        raise SchemaError(
            f"Unknown source key '{source_key}'. "
            f"Known keys: {sorted(SOURCE_SCHEMAS)}"
        )

    expected: list[str] = SOURCE_SCHEMAS[source_key]
    actual: list[str] = df.columns.tolist()

    missing = [c for c in expected if c not in actual]
    extra   = [c for c in actual   if c not in expected]

    if missing:
        raise SchemaError(
            f"[{source_key}] Missing required columns: {missing}\n"
            f"  Expected: {expected}\n"
            f"  Got:      {actual}"
        )

    if extra and not allow_extra_columns:
        raise SchemaError(
            f"[{source_key}] Unexpected extra columns: {extra}"
        )

    if raise_on_empty and len(df) == 0:
        raise SchemaError(f"[{source_key}] DataFrame is empty (0 rows).")

    logger.info(
        "[schema] %s OK — %d rows, %d cols%s",
        source_key,
        len(df),
        len(actual),
        f" (+{len(extra)} extra)" if extra else "",
    )


def validate_canonical_schema(
    df: pd.DataFrame,
    table_key: str,
    *,
    allow_missing_nullable: bool = False,
) -> None:
    """Assert that *df* matches the canonical output schema for *table_key*.

    Parameters
    ----------
    df:
        A processed DataFrame (after cleaning / harmonisation).
    table_key:
        Key into ``CANONICAL_COLUMNS`` (e.g. ``"drugs"`` or ``"ddi_pairs"``).
    allow_missing_nullable:
        If *True*, columns that are expected but absent are only warned, not
        raised.  Useful during incremental construction where some columns are
        not yet populated.
    """
    if table_key not in CANONICAL_COLUMNS:
        raise SchemaError(
            f"Unknown canonical table key '{table_key}'. "
            f"Known: {sorted(CANONICAL_COLUMNS)}"
        )

    expected: list[str] = CANONICAL_COLUMNS[table_key]
    actual:   list[str] = df.columns.tolist()
    missing = [c for c in expected if c not in actual]

    if missing:
        msg = f"[{table_key}] Canonical columns missing: {missing}"
        if allow_missing_nullable:
            logger.warning(msg)
        else:
            raise SchemaError(msg)

    logger.info(
        "[schema] canonical/%s OK — %d rows", table_key, len(df)
    )


def assert_no_duplicate_pairs(
    df: pd.DataFrame,
    col_a: str = "drug_id_a",
    col_b: str = "drug_id_b",
    *,
    label: str = "ddi_pairs",
) -> None:
    """Raise if any (drug_id_a, drug_id_b) pair appears more than once."""
    dupes = df.duplicated(subset=[col_a, col_b], keep=False)
    if dupes.any():
        n = dupes.sum()
        raise SchemaError(
            f"[{label}] Found {n} duplicate pair rows "
            f"(subset=[{col_a!r}, {col_b!r}])."
        )


def assert_no_self_pairs(
    df: pd.DataFrame,
    col_a: str = "drug_id_a",
    col_b: str = "drug_id_b",
    *,
    label: str = "ddi_pairs",
) -> None:
    """Raise if any row has drug_id_a == drug_id_b (self-interaction)."""
    self_pairs = df[col_a] == df[col_b]
    if self_pairs.any():
        n = self_pairs.sum()
        raise SchemaError(
            f"[{label}] Found {n} self-pairs where {col_a} == {col_b}."
        )


def assert_canonical_pair_order(
    df: pd.DataFrame,
    col_a: str = "drug_id_a",
    col_b: str = "drug_id_b",
    *,
    label: str = "ddi_pairs",
) -> None:
    """Raise if any row violates the canonical order drug_id_a < drug_id_b."""
    violations = df[col_a] >= df[col_b]
    if violations.any():
        n = violations.sum()
        raise SchemaError(
            f"[{label}] Found {n} rows where {col_a} >= {col_b}. "
            "Canonical order requires drug_id_a < drug_id_b."
        )


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class SchemaError(ValueError):
    """Raised when a DataFrame does not match the expected schema."""
