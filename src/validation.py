"""
Data and Target Validation Layer.
Handles schema normalization, column alias mapping, target class verification,
and clinical physiological range checking.
"""

import logging
from typing import Dict, Tuple, Any, List, Optional
import pandas as pd
import numpy as np

from config.config import (
    ALIAS_MAP,
    ALL_FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    TARGET_CLASSES,
    STATUS_VALUE_MAPPINGS,
    PHYSIOLOGICAL_RANGES,
    VALID_CATEGORICAL_VALUES,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw column aliases to canonical column names without mutating original.

    Args:
        df: Input DataFrame with arbitrary casing or abbreviations.

    Returns:
        DataFrame with canonical column names where aliases match.
    """
    renamed_cols = {}
    for col in df.columns:
        col_clean = str(col).strip()
        # Direct exact match in ALIAS_MAP
        if col_clean in ALIAS_MAP:
            renamed_cols[col] = ALIAS_MAP[col_clean]
        # Case-insensitive match in ALIAS_MAP
        elif col_clean.lower() in ALIAS_MAP:
            renamed_cols[col] = ALIAS_MAP[col_clean.lower()]
        # Already canonical with different casing
        else:
            for canonical in ALL_FEATURE_COLUMNS + [TARGET_COLUMN]:
                if col_clean.lower() == canonical.lower():
                    renamed_cols[col] = canonical
                    break

    df_normalized = df.rename(columns=renamed_cols)
    return df_normalized


def validate_schema(
    df: pd.DataFrame,
    require_target: bool = True,
    allow_missing_features: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate that essential features are present in the DataFrame.

    Args:
        df: Input DataFrame to check.
        require_target: If True, checks for presence of TARGET_COLUMN.
        allow_missing_features: If True, warns rather than fails on missing features.

    Returns:
        Tuple of (is_valid, missing_columns).

    Raises:
        ValidationError: If essential features or target are missing and strict mode is active.
    """
    missing_columns = []
    
    if require_target and TARGET_COLUMN not in df.columns:
        missing_columns.append(TARGET_COLUMN)

    for col in ALL_FEATURE_COLUMNS:
        if col not in df.columns:
            missing_columns.append(col)

    if missing_columns:
        err_msg = (
            f"Dataset schema validation failed. The following expected variables are missing: "
            f"{missing_columns}. Available columns: {list(df.columns)}"
        )
        if not allow_missing_features:
            logger.error(err_msg)
            raise ValidationError(err_msg)
        else:
            logger.warning(err_msg)
            return False, missing_columns

    return True, []


def preprocess_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Extract and standardize target column into canonical integer classes (0=Death, 1=Censored, 2=Transplant).

    Args:
        df: DataFrame containing the target column.
        target_col: Name of the target column.

    Returns:
        Tuple of (encoded_target_series, class_distribution_dataframe).

    Raises:
        ValidationError: If target is missing, contains nulls, or has unrecognized values.
    """
    if target_col not in df.columns:
        raise ValidationError(f"Target column '{target_col}' not found in dataframe.")

    raw_target = df[target_col]
    
    # Validate missing target values
    missing_target_count = int(raw_target.isna().sum())
    if missing_target_count > 0:
        raise ValidationError(
            f"Target variable contains {missing_target_count} missing values. "
            f"Target labels cannot be imputed; rows must be handled or dropped."
        )

    # Apply mapping
    encoded_target = raw_target.map(STATUS_VALUE_MAPPINGS)

    # Check for unmapped values
    if encoded_target.isna().any():
        unmapped_indices = raw_target[encoded_target.isna()].unique()
        raise ValidationError(
            f"Found unrecognized target values {unmapped_indices} that could not be mapped to "
            f"canonical classes {TARGET_CLASSES} using {STATUS_VALUE_MAPPINGS}."
        )

    encoded_target = encoded_target.astype(int)

    # Validate target classes presence
    unique_classes = set(encoded_target.unique())
    expected_classes = set(TARGET_CLASSES.keys())
    if not unique_classes.issubset(expected_classes):
        raise ValidationError(
            f"Encoded target contains unexpected classes: {unique_classes - expected_classes}"
        )

    # Generate dynamic class distribution summary
    counts = encoded_target.value_counts().sort_index()
    total = len(encoded_target)
    dist_data = []
    for cls_idx, cls_name in TARGET_CLASSES.items():
        count = counts.get(cls_idx, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        dist_data.append({
            "Class_ID": cls_idx,
            "Class_Name": cls_name,
            "Count": int(count),
            "Percentage": round(pct, 2),
        })

    dist_df = pd.DataFrame(dist_data)
    logger.info("Target class distribution successfully verified:\n%s", dist_df.to_string(index=False))

    return encoded_target, dist_df


def validate_patient_input(patient_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate a single patient input dictionary against physiological ranges and valid categories.

    Args:
        patient_data: Dict mapping feature names to patient values.

    Returns:
        Tuple of (is_valid, error_message_if_any).
    """
    # Check numeric ranges
    for feature, (min_val, max_val) in PHYSIOLOGICAL_RANGES.items():
        if feature in patient_data:
            val = patient_data[feature]
            if val is not None and not (isinstance(val, (int, float)) and np.isnan(val)):
                try:
                    num_val = float(val)
                except (ValueError, TypeError):
                    return False, f"Feature '{feature}' value '{val}' is not a valid numeric number."
                
                if num_val < min_val or num_val > max_val:
                    return False, (
                        f"Feature '{feature}' value {num_val} is outside physiologically plausible range "
                        f"[{min_val}, {max_val}]."
                    )

    # Check categoricals
    for feature, valid_vals in VALID_CATEGORICAL_VALUES.items():
        if feature in patient_data:
            val = patient_data[feature]
            if val is not None and str(val).strip() != "":
                # Convert string for flexible matching
                matched = any(str(v).lower() == str(val).lower() for v in valid_vals)
                if not matched:
                    return False, (
                        f"Feature '{feature}' value '{val}' is not recognized. "
                        f"Expected one of: {valid_vals}."
                    )

    return True, None
