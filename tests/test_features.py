"""
Unit tests for clinical feature engineering transformer.
"""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import ClinicalFeatureEngineer


def test_clinical_feature_engineer_ratios():
    df = pd.DataFrame({
        "Albumin": [3.0, 4.0],
        "Bilirubin": [1.0, 0.0],
        "Copper": [50.0, 100.0],
        "SGOT": [100.0, 80.0],
        "Platelets": [200.0, 150.0],
        "Cholesterol": [np.nan, 250.0],
        "Triglycerides": [120.0, np.nan],
    })

    fe = ClinicalFeatureEngineer(epsilon=1e-5)
    transformed = fe.fit_transform(df)

    assert "ALBI_proxy" in transformed.columns
    assert "log_ALBI_proxy" in transformed.columns
    assert "Copper_Bilirubin_Index" in transformed.columns
    assert "APRI_proxy" in transformed.columns

    # First row ALBI proxy: 3.0 / (1.0 + 1e-5) ~ 3.0
    assert np.isclose(transformed["ALBI_proxy"].iloc[0], 3.0, atol=1e-3)
    # Second row zero-division protection: 4.0 / (0.0 + 1e-5) should be finite
    assert np.isfinite(transformed["ALBI_proxy"].iloc[1])


def test_clinical_feature_engineer_missing_indicators():
    df = pd.DataFrame({
        "Albumin": [3.5, 4.0],
        "Bilirubin": [1.2, 2.0],
        "Cholesterol": [np.nan, 300.0],
        "Copper": [80.0, np.nan],
        "Triglycerides": [np.nan, 140.0],
    })

    fe = ClinicalFeatureEngineer()
    transformed = fe.fit_transform(df)

    assert "Cholesterol_missing" in transformed.columns
    assert "Copper_missing" in transformed.columns
    assert "Triglycerides_missing" in transformed.columns

    assert transformed["Cholesterol_missing"].iloc[0] == 1.0
    assert transformed["Cholesterol_missing"].iloc[1] == 0.0
    assert transformed["Copper_missing"].iloc[0] == 0.0
    assert transformed["Copper_missing"].iloc[1] == 1.0
