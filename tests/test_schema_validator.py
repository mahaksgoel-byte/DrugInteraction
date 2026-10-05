"""
test_schema_validator.py – Unit tests for schema validation logic.

These tests are designed to run WITHOUT any raw data files.  They verify
that the schema gate catches missing columns, extra-column behaviour, and
empty-frame detection correctly.
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.data.schema_validator import (
    validate_raw_schema,
    validate_canonical_schema,
    assert_no_duplicate_pairs,
    assert_no_self_pairs,
    assert_canonical_pair_order,
    SchemaError,
)
from src.data.constants import SOURCE_SCHEMAS, CANONICAL_COLUMNS


# ---------------------------------------------------------------------------
# validate_raw_schema
# ---------------------------------------------------------------------------

class TestValidateRawSchema:

    def test_passes_with_exact_columns(self):
        """Exact columns → no exception."""
        cols = SOURCE_SCHEMAS["ddinter"]
        df = pd.DataFrame(columns=cols)
        df.loc[0] = ["id_a", "DrugA", "id_b", "DrugB", "Major"]
        # Should not raise
        validate_raw_schema(df, "ddinter")

    def test_passes_with_extra_columns(self):
        """Extra columns are allowed by default."""
        cols = SOURCE_SCHEMAS["ddinter"] + ["extra_col"]
        df = pd.DataFrame(columns=cols)
        df.loc[0] = ["id_a", "DrugA", "id_b", "DrugB", "Major", "extra"]
        validate_raw_schema(df, "ddinter")

    def test_fails_on_missing_required_column(self):
        """Missing column → SchemaError."""
        cols = SOURCE_SCHEMAS["ddinter"][:-1]   # drop "Level"
        df = pd.DataFrame(columns=cols)
        df.loc[0] = ["id_a", "DrugA", "id_b", "DrugB"]
        with pytest.raises(SchemaError, match="Missing required columns"):
            validate_raw_schema(df, "ddinter")

    def test_fails_on_empty_dataframe(self):
        """Empty DataFrame → SchemaError when raise_on_empty=True."""
        cols = SOURCE_SCHEMAS["ddinter"]
        df = pd.DataFrame(columns=cols)
        with pytest.raises(SchemaError, match="empty"):
            validate_raw_schema(df, "ddinter", raise_on_empty=True)

    def test_no_raise_on_empty_when_disabled(self):
        """Empty DataFrame is allowed when raise_on_empty=False."""
        cols = SOURCE_SCHEMAS["ddinter"]
        df = pd.DataFrame(columns=cols)
        # Should not raise
        validate_raw_schema(df, "ddinter", raise_on_empty=False)

    def test_fails_on_unknown_source_key(self):
        with pytest.raises(SchemaError, match="Unknown source key"):
            validate_raw_schema(pd.DataFrame(), "nonexistent_source")

    def test_fails_extra_columns_when_disallowed(self):
        cols = SOURCE_SCHEMAS["ddinter"] + ["bonus"]
        df = pd.DataFrame(columns=cols)
        df.loc[0] = ["id_a", "DrugA", "id_b", "DrugB", "Major", "x"]
        with pytest.raises(SchemaError, match="Unexpected extra columns"):
            validate_raw_schema(df, "ddinter", allow_extra_columns=False)


# ---------------------------------------------------------------------------
# assert_no_duplicate_pairs / assert_no_self_pairs / canonical order
# ---------------------------------------------------------------------------

class TestPairAssertions:

    def _pairs_df(self, rows: list[tuple]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["drug_id_a", "drug_id_b"])

    def test_no_duplicate_pairs_passes(self):
        df = self._pairs_df([(1, 2), (1, 3), (2, 3)])
        assert_no_duplicate_pairs(df)   # should not raise

    def test_duplicate_pairs_detected(self):
        df = self._pairs_df([(1, 2), (1, 2)])
        with pytest.raises(SchemaError, match="duplicate pair rows"):
            assert_no_duplicate_pairs(df)

    def test_no_self_pairs_passes(self):
        df = self._pairs_df([(1, 2), (2, 3)])
        assert_no_self_pairs(df)        # should not raise

    def test_self_pairs_detected(self):
        df = self._pairs_df([(1, 1), (2, 3)])
        with pytest.raises(SchemaError, match="self-pairs"):
            assert_no_self_pairs(df)

    def test_canonical_order_passes(self):
        df = self._pairs_df([(1, 2), (2, 3)])
        assert_canonical_pair_order(df)  # should not raise

    def test_canonical_order_violation_detected(self):
        df = self._pairs_df([(2, 1)])    # 2 > 1 → violates a < b
        with pytest.raises(SchemaError, match="(?i)canonical order"):
            assert_canonical_pair_order(df)


# ---------------------------------------------------------------------------
# All source keys have schemas defined
# ---------------------------------------------------------------------------

class TestSchemaCompleteness:

    def test_all_schema_keys_have_at_least_one_column(self):
        for key, cols in SOURCE_SCHEMAS.items():
            assert len(cols) > 0, f"SOURCE_SCHEMAS[{key!r}] has no columns"

    def test_all_canonical_keys_have_at_least_one_column(self):
        for key, cols in CANONICAL_COLUMNS.items():
            assert len(cols) > 0, f"CANONICAL_COLUMNS[{key!r}] has no columns"
