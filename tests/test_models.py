"""
Unit tests for the 5 clinical ML model pipelines.
"""

import pytest
import numpy as np
import pandas as pd

from src.models import get_base_models
from src.preprocessing import prepare_data_splits
from src.data_loader import load_raw_data


@pytest.fixture
def split_data():
    raw_df = load_raw_data(auto_download=True)
    return prepare_data_splits(raw_df, test_size=0.20, random_state=42)


def test_base_models_contains_all_five_algorithms():
    models = get_base_models()
    expected = {
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
    }
    assert expected.issubset(set(models.keys()))


@pytest.mark.parametrize("model_name", [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
])
def test_models_fit_and_predict_valid_classes(model_name, split_data):
    X_train, X_test, y_train, y_test, _ = split_data
    models = get_base_models()
    pipeline = models[model_name]

    # Fit on training split
    pipeline.fit(X_train, y_train)

    # Predict on test split
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    
    unique_preds = set(np.unique(preds))
    assert unique_preds.issubset({0, 1, 2})

    # Probability predictions
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_test)
        assert probs.shape == (len(X_test), 3)
        # Probabilities sum to 1
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-3)
