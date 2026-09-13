"""
Clinically meaningful feature engineering transformer.
Integrates into sklearn Pipeline to prevent data leakage across CV folds.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leakage-safe clinical biomarker feature engineering.
    
    Engineered features:
    1. ALBI_proxy: Albumin-to-Bilirubin ratio (log-linear proxy of liver synthetic vs excretory function).
    2. Copper_Bilirubin_Index: Multiplicative proxy of cholestatic burden.
    3. Platelet_Albumin_Ratio: Portal hypertension and synthetic function proxy.
    4. Missingness indicators for labile laboratory analytes.
    """

    def __init__(
        self,
        epsilon: float = 1e-5,
        create_missing_indicators: bool = True,
        create_ratios: bool = True
    ):
        self.epsilon = epsilon
        self.create_missing_indicators = create_missing_indicators
        self.create_ratios = create_ratios
        self.feature_names_: List[str] = []

    def fit(self, X, y=None):
        # Feature engineering is purely deterministic per row and requires no fold-level parameters.
        return self

    def transform(self, X):
        """
        Transform features by adding clinically derived biomarker ratios and missing indicators.

        Args:
            X: Input DataFrame or numpy ndarray.

        Returns:
            Transformed DataFrame or array with engineered features appended.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # If numpy array, return as is (caller should pass DataFrame)
            return X

        engineered_cols = {}

        # 1. Missingness indicators for lab tests with frequent clinical missingness
        if self.create_missing_indicators:
            for col in ["Cholesterol", "Copper", "Triglycerides"]:
                if col in df.columns:
                    engineered_cols[f"{col}_missing"] = df[col].isna().astype(float)

        # 2. Albumin-to-Bilirubin proxy (ALBI proxy)
        # In chronic liver disease, declining albumin + rising bilirubin indicates hepatic decompensation
        if self.create_ratios and "Albumin" in df.columns and "Bilirubin" in df.columns:
            # Albumin / (Bilirubin + eps)
            engineered_cols["ALBI_proxy"] = df["Albumin"] / (df["Bilirubin"] + self.epsilon)
            # Log ALBI score proxy: log(Bilirubin + 0.1) - Albumin
            engineered_cols["log_ALBI_proxy"] = np.log(np.maximum(df["Bilirubin"], 0.1)) - (0.5 * df["Albumin"])

        # 3. Cholestasis Burden Proxy: Copper * Bilirubin
        if self.create_ratios and "Copper" in df.columns and "Bilirubin" in df.columns:
            engineered_cols["Copper_Bilirubin_Index"] = df["Copper"] * df["Bilirubin"]

        # 4. AST-to-Platelet Index Proxy (APRI proxy)
        # AST / Platelet ratio is widely used to assess liver fibrosis severity
        if self.create_ratios and "SGOT" in df.columns and "Platelets" in df.columns:
            engineered_cols["APRI_proxy"] = (df["SGOT"] / (df["Platelets"] + self.epsilon)) * 100.0

        if engineered_cols:
            eng_df = pd.DataFrame(engineered_cols, index=df.index)
            result_df = pd.concat([df, eng_df], axis=1)
        else:
            result_df = df

        self.feature_names_ = list(result_df.columns)
        return result_df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return self.feature_names_ if self.feature_names_ else (input_features or [])
