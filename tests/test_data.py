"""
Unit tests for data ingestion, schema validation, and target preprocessing.
"""

import pytest
import pandas as pd
import numpy as np

from config.config import TARGET_CLASSES, ALL_FEATURE_COLUMNS, TARGET_COLUMN
from src.data_loader import load_raw_data
from src.validation import normalize_column_names, validate_schema, preprocess_target, ValidationError


@pytest.fixture
def raw_dataset():
    """Load canonical raw dataset."""
    return load_raw_data(auto_download=True)


def test_dataset_loads_successfully(raw_dataset):
    assert raw_dataset is not None
    assert not raw_dataset.empty
    assert len(raw_dataset) >= 400


def test_column_normalization_maps_aliases():
    sample_df = pd.DataFrame({
        "bili": [1.2, 3.4],
        "ast": [120, 150],
        "alk.phos": [1500, 2000],
        "chol": [250, 300],
        "status": [0, 2],
    })
    norm_df = normalize_column_names(sample_df)
    assert "Bilirubin" in norm_df.columns
    assert "SGOT" in norm_df.columns
    assert "Alk_Phos" in norm_df.columns
    assert "Cholesterol" in norm_df.columns
    assert "Status" in norm_df.columns


def test_target_contains_expected_classes(raw_dataset):
    norm_df = normalize_column_names(raw_dataset)
    y, dist_df = preprocess_target(norm_df)
    
    unique_targets = set(y.unique())
    assert unique_targets == {0, 1, 2}
    assert set(dist_df["Class_ID"].unique()) == {0, 1, 2}
    assert dist_df["Count"].sum() == len(raw_dataset)


def test_target_has_no_missing_values(raw_dataset):
    norm_df = normalize_column_names(raw_dataset)
    y, _ = preprocess_target(norm_df)
    assert y.isna().sum() == 0


def test_target_preprocessing_rejects_missing_values():
    bad_df = pd.DataFrame({
        TARGET_COLUMN: [0, np.nan, 2],
    })
    with pytest.raises(ValidationError, match="contains 1 missing values"):
        preprocess_target(bad_df)


def test_target_preprocessing_rejects_unknown_classes():
    bad_df = pd.DataFrame({
        TARGET_COLUMN: [0, 999, 1],
    })
    with pytest.raises(ValidationError, match="unrecognized target values"):
        preprocess_target(bad_df)


def test_schema_validation_detects_missing_required_features():
    incomplete_df = pd.DataFrame({"Age": [50], "Bilirubin": [2.0]})
    with pytest.raises(ValidationError, match="schema validation failed"):
        validate_schema(incomplete_df, require_target=True)
