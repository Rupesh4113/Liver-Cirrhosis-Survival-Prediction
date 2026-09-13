"""
Production-grade inference and prediction engine.
Validates input ranges, checks schema, and generates multiclass predictions
with probability breakdowns and safety disclaimers.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import joblib
import pandas as pd
import numpy as np

from config.config import (
    FINAL_MODEL_PATH,
    MODEL_METADATA_PATH,
    TARGET_CLASSES,
    MEDICAL_DISCLAIMER,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.validation import validate_patient_input, ValidationError

logger = logging.getLogger(__name__)

# Cached model and metadata instances
_LOADED_MODEL = None
_LOADED_METADATA = None


def load_inference_artifacts(
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    force_reload: bool = False,
):
    """
    Load serialized model pipeline and training metadata into memory with caching.
    """
    global _LOADED_MODEL, _LOADED_METADATA

    m_path = Path(model_path) if model_path else FINAL_MODEL_PATH
    meta_path = Path(metadata_path) if metadata_path else MODEL_METADATA_PATH

    if _LOADED_MODEL is None or force_reload:
        if not m_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at {m_path}. Please execute 'python train.py' first."
            )
        _LOADED_MODEL = joblib.load(m_path)
        logger.info("Successfully loaded model artifact from %s", m_path)

    if _LOADED_METADATA is None or force_reload:
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                _LOADED_METADATA = json.load(f)
        else:
            _LOADED_METADATA = {"model_name": "Final_Pipeline", "version": "1.0.0"}

    return _LOADED_MODEL, _LOADED_METADATA


def predict_patient(
    patient_data: Dict[str, Any],
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Validate input biomarkers and predict multiclass survival status for an individual patient.

    Args:
        patient_data: Dictionary mapping feature names to patient values.
        model_path: Optional path to custom serialized model pipeline.

    Returns:
        Dictionary containing:
        - predicted_class: Canonical class index (0, 1, 2)
        - predicted_label: Human-readable outcome ("Death", "Censored", "Transplant")
        - probabilities: Class probabilities dict (e.g. {"Death": 0.72, ...})
        - model_version: Model identifier and training timestamp
        - warning: Mandatory medical research safety disclaimer
    """
    # 1. Physiological validation
    is_valid, err_msg = validate_patient_input(patient_data)
    if not is_valid:
        raise ValidationError(f"Invalid patient clinical inputs: {err_msg}")

    # 2. Format into single-row DataFrame
    expected_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    row_data = {}
    for col in expected_cols:
        val = patient_data.get(col, np.nan)
        row_data[col] = [val]

    df_patient = pd.DataFrame(row_data)

    # 3. Load model artifact
    model, metadata = load_inference_artifacts(model_path=model_path)

    # 4. Generate prediction and probabilities
    pred_class_idx = int(model.predict(df_patient)[0])
    pred_label = TARGET_CLASSES.get(pred_class_idx, "Unknown")

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df_patient)[0]
        for idx, label in TARGET_CLASSES.items():
            probabilities[label] = round(float(probs[idx]), 4) if idx < len(probs) else 0.0
    else:
        for idx, label in TARGET_CLASSES.items():
            probabilities[label] = 1.0 if idx == pred_class_idx else 0.0

    return {
        "predicted_class": pred_class_idx,
        "predicted_label": pred_label,
        "probabilities": probabilities,
        "model_version": metadata.get("model_name", "Primary Biliary Cirrhosis Multiclass Classifier"),
        "training_date": metadata.get("training_date", "N/A"),
        "warning": MEDICAL_DISCLAIMER,
    }


def predict_batch(
    df: pd.DataFrame,
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate predictions and probability distributions for a batch of patient records.
    """
    model, _ = load_inference_artifacts(model_path=model_path)
    preds = model.predict(df)
    probs = model.predict_proba(df) if hasattr(model, "predict_proba") else None

    result_df = df.copy()
    result_df["predicted_class"] = preds
    result_df["predicted_label"] = [TARGET_CLASSES.get(int(p), "Unknown") for p in preds]

    if probs is not None:
        for idx, label in TARGET_CLASSES.items():
            result_df[f"prob_{label}"] = probs[:, idx]

    return result_df
