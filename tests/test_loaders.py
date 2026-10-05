"""
test_loaders.py – Integration smoke tests for raw data loaders.

These tests check that the loaders for data files that ARE already present
(DDInter code_A and code_B) load correctly and pass schema validation.

Tests for loaders whose raw files have not yet been downloaded are marked
with ``@pytest.mark.skipif`` so they silently skip rather than fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

# Base raw data directory (relative to project root)
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# DDInter – files ARE present
# ---------------------------------------------------------------------------

class TestDDInterLoader:

    def test_load_ddinter_returns_dataframe(self):
        """load_ddinter returns a non-empty DataFrame with expected columns."""
        from src.data.loaders import load_ddinter
        df = load_ddinter(RAW_DIR / "ddinter")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_ddinter_has_required_columns(self):
        from src.data.loaders import load_ddinter
        from src.data.constants import SOURCE_SCHEMAS
        df = load_ddinter(RAW_DIR / "ddinter")
        for col in SOURCE_SCHEMAS["ddinter"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_ddinter_level_values_are_known(self):
        """Severity Level column contains only Major / Moderate / Minor / Unknown."""
        from src.data.loaders import load_ddinter
        df = load_ddinter(RAW_DIR / "ddinter")
        levels = set(df["Level"].dropna().unique())
        known = {"Major", "Moderate", "Minor", "Unknown"}
        unexpected = levels - known
        assert not unexpected, f"Unexpected Level values in real data: {unexpected}"

    def test_ddinter_no_nulls_in_key_columns(self):
        from src.data.loaders import load_ddinter
        df = load_ddinter(RAW_DIR / "ddinter")
        for col in ["DDInterID_A", "DDInterID_B", "Drug_A", "Drug_B"]:
            n_null = df[col].isna().sum()
            assert n_null == 0, f"Column {col} has {n_null} null values"

    def test_ddinter_code_b_has_rows(self):
        """code_B file should have been downloaded and contain rows."""
        path = RAW_DIR / "ddinter" / "ddinter_downloads_code_B.csv"
        assert path.exists(), "ddinter_downloads_code_B.csv not found"
        df = pd.read_csv(path, dtype=str)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Optional sources – skip if file absent
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (RAW_DIR / "chembl" / "chembl_drugs.csv").exists(),
    reason="chembl_drugs.csv not downloaded yet"
)
def test_load_chembl_drugs():
    from src.data.loaders import load_chembl_drugs
    df = load_chembl_drugs(RAW_DIR / "chembl")
    assert isinstance(df, pd.DataFrame) and len(df) > 0


@pytest.mark.skipif(
    not (RAW_DIR / "pharmgkb" / "drugs.tsv").exists(),
    reason="pharmgkb/drugs.tsv not downloaded yet"
)
def test_load_pharmgkb_drugs():
    from src.data.loaders import load_pharmgkb_drugs
    df = load_pharmgkb_drugs(RAW_DIR / "pharmgkb")
    assert isinstance(df, pd.DataFrame) and len(df) > 0


@pytest.mark.skipif(
    not (RAW_DIR / "sider" / "meddra_all_se.tsv").exists(),
    reason="sider/meddra_all_se.tsv not downloaded yet"
)
def test_load_sider():
    from src.data.loaders import load_sider_side_effects
    df = load_sider_side_effects(RAW_DIR / "sider")
    assert isinstance(df, pd.DataFrame) and len(df) > 0


@pytest.mark.skipif(
    not list((RAW_DIR / "string").glob("protein.links*.txt*")),
    reason="STRING PPI file not downloaded yet"
)
def test_load_string_ppi():
    from src.data.loaders import load_string_ppi
    df = load_string_ppi(RAW_DIR / "string")
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    assert (df["combined_score"] >= 700).all(), "Pre-filter on score failed"
