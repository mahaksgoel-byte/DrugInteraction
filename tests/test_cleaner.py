"""
test_cleaner.py – Unit tests for data cleaning functions.

These tests use small synthetic DataFrames and do NOT require raw data files.
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.data.cleaner import (
    normalize_drug_names,
    normalize_interaction_level,
)
from src.data.harmoniser import (
    enforce_canonical_pair_order,
    deduplicate_pairs,
    drop_self_pairs,
)


class TestNormalizeDrugNames:

    def test_lowercase_and_strip(self):
        df = pd.DataFrame({"name": ["  Aspirin  ", "IBUPROFEN", "Paracetamol"]})
        out = normalize_drug_names(df)
        assert out["name"].tolist() == ["aspirin", "ibuprofen", "paracetamol"]

    def test_multiple_spaces_collapsed(self):
        df = pd.DataFrame({"name": ["drug   a"]})
        out = normalize_drug_names(df)
        assert out["name"].iloc[0] == "drug a"

    def test_nan_becomes_empty_string(self):
        df = pd.DataFrame({"name": [None]})
        out = normalize_drug_names(df)
        assert out["name"].iloc[0] == ""


class TestNormalizeInteractionLevel:

    def test_standard_levels_title_cased(self):
        df = pd.DataFrame({"Level": ["Major", "MODERATE", "minor"]})
        out = normalize_interaction_level(df)
        assert set(out["level"].tolist()) == {"Major", "Moderate", "Minor"}

    def test_unknown_level_mapped(self):
        df = pd.DataFrame({"Level": ["Severe"]})
        out = normalize_interaction_level(df)
        assert out["level"].iloc[0] == "Unknown"

    def test_original_column_preserved(self):
        df = pd.DataFrame({"Level": ["Major"]})
        out = normalize_interaction_level(df)
        assert "Level" in out.columns


class TestCanonicalPairOrder:

    def test_swaps_when_a_greater_than_b(self):
        df = pd.DataFrame({"drug_id_a": [3], "drug_id_b": [1]})
        out = enforce_canonical_pair_order(df)
        assert out.iloc[0]["drug_id_a"] == 1
        assert out.iloc[0]["drug_id_b"] == 3

    def test_no_change_when_already_ordered(self):
        df = pd.DataFrame({"drug_id_a": [1], "drug_id_b": [3]})
        out = enforce_canonical_pair_order(df)
        assert out.iloc[0]["drug_id_a"] == 1
        assert out.iloc[0]["drug_id_b"] == 3


class TestDeduplicatePairs:

    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({
            "drug_id_a": [1, 1, 2],
            "drug_id_b": [2, 2, 3],
            "level": ["Major", "Major", "Minor"],
        })
        out = deduplicate_pairs(df)
        assert len(out) == 2

    def test_keeps_all_when_no_duplicates(self):
        df = pd.DataFrame({"drug_id_a": [1, 2], "drug_id_b": [2, 3]})
        out = deduplicate_pairs(df)
        assert len(out) == 2


class TestDropSelfPairs:

    def test_removes_self_pairs(self):
        df = pd.DataFrame({"drug_id_a": [1, 1, 2], "drug_id_b": [1, 2, 3]})
        out = drop_self_pairs(df)
        assert len(out) == 2
        assert (out["drug_id_a"] != out["drug_id_b"]).all()

    def test_no_change_when_no_self_pairs(self):
        df = pd.DataFrame({"drug_id_a": [1, 2], "drug_id_b": [2, 3]})
        out = drop_self_pairs(df)
        assert len(out) == 2
