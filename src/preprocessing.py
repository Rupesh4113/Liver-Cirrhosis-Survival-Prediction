"""
Leakage-safe preprocessing pipelines using scikit-learn ColumnTransformer.
Ensures median/mode statistics and scalers are fitted strictly on training data/folds.
"""

import logging
from typing import Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.feature_engineering import ClinicalFeatureEngineer
from src.validation import normalize_column_names, preprocess_target, validate_schema

logger = logging.getLogger(__name__)


def prepare_data_splits(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Standardize schema, extract target, and split into train and holdout test sets.

    Guarantees that train/test split is stratified across target classes and occurs
    prior to any imputation, scaling, or model fitting to prevent data leakage.

    Args:
        df: Raw DataFrame.
        test_size: Proportion for holdout test set (default 0.20).
        random_state: Seed for reproducible split.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, class_distribution_df).
    """
    df_norm = normalize_column_names(df)
    validate_schema(df_norm, require_target=True, allow_missing_features=False)

    y, dist_df = preprocess_target(df_norm, TARGET_COLUMN)
    X = df_norm[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    logger.info(
        "Stratified data split complete: Train samples=%d, Test samples=%d",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test, dist_df


def build_preprocessor(
    scale_numeric: bool = True,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for numeric imputation/scaling and categorical encoding.

    Args:
        scale_numeric: If True, applies StandardScaler to numeric features (essential for Logistic Regression).
        numeric_features: Custom list of numeric features, or None to use default.
        categorical_features: Custom list of categorical features, or None to use default.

    Returns:
        Configured ColumnTransformer instance.
    """
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    # Numeric pipeline
    num_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=num_steps)

    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_full_pipeline(
    classifier,
    scale_numeric: bool = False,
    include_feature_engineering: bool = True,
) -> Pipeline:
    """
    Construct a complete end-to-end scikit-learn Pipeline:
    (Optional Feature Engineering) -> Preprocessor (Impute + Scale/OneHot) -> Classifier.

    This ensures full encapsulation inside cross-validation loops, eliminating data leakage.
    """
    steps = []

    # Dynamic features list if feature engineering is active
    if include_feature_engineering:
        steps.append(("feature_engineer", ClinicalFeatureEngineer()))
        
        # Determine expanded numeric columns after feature engineering
        extended_num_cols = list(NUMERIC_FEATURES) + [
            "Cholesterol_missing",
            "Copper_missing",
            "Triglycerides_missing",
            "ALBI_proxy",
            "log_ALBI_proxy",
            "Copper_Bilirubin_Index",
            "APRI_proxy",
        ]
        preprocessor = build_preprocessor(
            scale_numeric=scale_numeric,
            numeric_features=extended_num_cols,
            categorical_features=CATEGORICAL_FEATURES,
        )
        steps.append(("preprocessor", preprocessor))
    else:
        preprocessor = build_preprocessor(
            scale_numeric=scale_numeric,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
        steps.append(("preprocessor", preprocessor))

    steps.append(("classifier", classifier))

    return Pipeline(steps=steps)
