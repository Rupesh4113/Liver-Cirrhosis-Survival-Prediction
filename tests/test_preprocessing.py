"""
Unit tests for data preprocessing and leakage prevention.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from config.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.preprocessing import build_preprocessor, build_full_pipeline, prepare_data_splits
from src.data_loader import load_raw_data


@pytest.fixture
def raw_dataset():
    return load_raw_data(auto_download=True)


def test_preprocessor_imputes_missing_numeric_values():
    df_sample = pd.DataFrame({
        "Age": [50.0, np.nan, 60.0],
        "Bilirubin": [1.0, 2.5, np.nan],
        "Cholesterol": [np.nan, 300.0, 400.0],
        "Albumin": [3.5, np.nan, 4.0],
        "Copper": [50.0, 100.0, np.nan],
        "Alk_Phos": [np.nan, 1500.0, 2000.0],
        "SGOT": [100.0, np.nan, 120.0],
        "Triglycerides": [np.nan, 150.0, 200.0],
        "Platelets": [200.0, np.nan, 250.0],
        "Prothrombin": [10.5, 11.0, np.nan],
        "Stage": [3.0, 4.0, np.nan],
        "Sex": ["F", "M", "F"],
        "Ascites": [0.0, 1.0, 0.0],
        "Hepatomegaly": [1.0, 0.0, 1.0],
        "Spiders": [0.0, 1.0, 0.0],
        "Edema": [0.0, 0.5, 1.0],
    })

    preprocessor = build_preprocessor(scale_numeric=True)
    transformed = preprocessor.fit_transform(df_sample)
    
    # Assert no NaNs remaining in transformed array
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == 3


def test_preprocessor_handles_unseen_categorical_levels():
    df_train = pd.DataFrame({
        col: [50.0, 60.0] for col in NUMERIC_FEATURES
    })
    df_train["Sex"] = ["F", "M"]
    df_train["Ascites"] = [0.0, 1.0]
    df_train["Hepatomegaly"] = [1.0, 0.0]
    df_train["Spiders"] = [0.0, 1.0]
    df_train["Edema"] = [0.0, 0.5]

    preprocessor = build_preprocessor(scale_numeric=False)
    preprocessor.fit(df_train)

    df_test = df_train.copy()
    # Introduce unseen category
    df_test["Sex"] = ["Unknown", "Other"]

    # Should not crash thanks to handle_unknown='ignore'
    transformed_test = preprocessor.transform(df_test)
    assert not np.isnan(transformed_test).any()


def test_no_data_leakage_in_splits(raw_dataset):
    X_train, X_test, y_train, y_test, _ = prepare_data_splits(raw_dataset, test_size=0.20, random_state=42)

    # Train and test index intersection must be empty
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert len(train_indices.intersection(test_indices)) == 0

    # Ensure train size is ~80% and test is ~20%
    total = len(raw_dataset)
    assert len(X_train) == int(round(total * 0.8))
    assert len(X_test) == total - len(X_train)


def test_leakage_safe_pipeline_imputer_parameters(raw_dataset):
    """
    Ensure the imputer inside a Pipeline computes its statistics only from the training split.
    """
    X_train, X_test, y_train, y_test, _ = prepare_data_splits(raw_dataset, test_size=0.20, random_state=42)

    pipeline = build_full_pipeline(LogisticRegression(max_iter=1000), scale_numeric=True)
    pipeline.fit(X_train, y_train)

    # Imputer statistics should match X_train median, NOT full dataset median
    imputer = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    
    # Calculate manual train median for Bilirubin
    expected_train_median = X_train["Bilirubin"].median()
    # Column index of Bilirubin in extended numeric features is 1 (Age is 0)
    actual_imputer_median = imputer.statistics_[1]

    assert np.isclose(actual_imputer_median, expected_train_median, atol=1e-3)
