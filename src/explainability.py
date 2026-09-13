"""
Model explainability module.
Implements Random Forest Gini importance, holdout permutation feature importance,
and SHAP local/global attributions with medical research framing.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import shap

from config.config import RANDOM_STATE, MEDICAL_DISCLAIMER

logger = logging.getLogger(__name__)


def extract_feature_names_from_pipeline(pipeline) -> List[str]:
    """
    Extract transformed feature names from the fitted scikit-learn Pipeline.
    """
    preprocessor = None
    for name, step in pipeline.named_steps.items():
        if name == "preprocessor":
            preprocessor = step
            break

    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            pass

    # Fallback to feature names from transformers
    if preprocessor is not None:
        names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    names.extend(list(transformer.get_feature_names_out(cols)))
                    continue
                except Exception:
                    pass
            names.extend(list(cols))
        if names:
            return names

    return [f"feature_{i}" for i in range(100)]


def calculate_gini_importance(pipeline, top_n: int = 15) -> pd.DataFrame:
    """
    Extract intrinsic feature importances (e.g. MDI / Gini) from tree-based classifier.
    """
    clf = pipeline.named_steps.get("classifier")
    if not hasattr(clf, "feature_importances_"):
        logger.warning("Classifier does not have feature_importances_ attribute.")
        return pd.DataFrame(columns=["Feature", "Gini_Importance"])

    importances = clf.feature_importances_
    feature_names = extract_feature_names_from_pipeline(pipeline)[:len(importances)]

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Gini_Importance": importances,
    }).sort_values(by="Gini_Importance", ascending=False).reset_index(drop=True)

    df_imp["Relative_Percentage"] = (df_imp["Gini_Importance"] / df_imp["Gini_Importance"].sum()) * 100.0
    return df_imp.head(top_n)


def calculate_permutation_importance(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_n: int = 15,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Calculate permutation feature importance on the untouched holdout test set.
    Measures drop in Macro F1 when each column is randomly permuted.
    """
    logger.info("Computing holdout permutation feature importance on %d samples...", len(X_test))
    
    perm_res = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=10,
        random_state=random_state,
        n_jobs=1,
    )

    feature_names = list(X_test.columns)
    df_perm = pd.DataFrame({
        "Feature": feature_names,
        "Permutation_Importance_Mean": perm_res.importances_mean,
        "Permutation_Importance_Std": perm_res.importances_std,
    }).sort_values(by="Permutation_Importance_Mean", ascending=False).reset_index(drop=True)

    return df_perm.head(top_n)


def compute_shap_explanations(
    pipeline,
    X_background: pd.DataFrame,
    X_sample: pd.DataFrame,
) -> Tuple[Optional[Any], Optional[np.ndarray], List[str]]:
    """
    Compute SHAP values using TreeExplainer (if tree ensemble) or KernelExplainer.

    Returns:
        Tuple of (explainer, shap_values, transformed_feature_names).
    """
    try:
        # Transform data through preprocessing steps prior to classifier
        pipe_steps = list(pipeline.named_steps.items())
        classifier = pipe_steps[-1][1]
        
        # Transform background and sample through intermediate steps
        X_bg_trans = X_background.copy()
        X_sample_trans = X_sample.copy()

        for name, step in pipe_steps[:-1]:
            X_bg_trans = step.transform(X_bg_trans)
            X_sample_trans = step.transform(X_sample_trans)

        feature_names = extract_feature_names_from_pipeline(pipeline)[:X_sample_trans.shape[1]]

        # Use TreeExplainer for tree models
        if hasattr(classifier, "estimators_") or "LGBM" in type(classifier).__name__ or "XGB" in type(classifier).__name__:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_sample_trans)
            return explainer, shap_values, feature_names
        else:
            # Fallback to KernelExplainer with small background summary
            bg_summary = shap.sample(X_bg_trans, min(20, len(X_bg_trans)), random_state=RANDOM_STATE)
            explainer = shap.KernelExplainer(classifier.predict_proba, bg_summary)
            shap_values = explainer.shap_values(X_sample_trans)
            return explainer, shap_values, feature_names
    except Exception as exc:
        logger.warning("SHAP computation failed or skipped: %s", exc)
        return None, None, []


def explain_single_patient(
    pipeline,
    patient_df: pd.DataFrame,
    background_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate patient-level model explanation breakdown for a single input profile.
    """
    probabilities = pipeline.predict_proba(patient_df)[0]
    predicted_class = int(np.argmax(probabilities))

    explainer, shap_values, feature_names = compute_shap_explanations(
        pipeline,
        background_df,
        patient_df,
    )

    feature_contributions = []
    if shap_values is not None:
        # For multiclass, shap_values is either a list of arrays (one per class) or a 3D array (samples, features, classes)
        if isinstance(shap_values, list):
            class_shap = shap_values[predicted_class][0]
        elif len(shap_values.shape) == 3:
            class_shap = shap_values[0, :, predicted_class]
        else:
            class_shap = shap_values[0]

        for fname, s_val in zip(feature_names, class_shap):
            feature_contributions.append({
                "Feature": fname,
                "SHAP_Value": float(s_val),
                "Direction": "Increased Risk" if s_val > 0 else "Decreased Risk",
            })
        
        feature_contributions = sorted(feature_contributions, key=lambda x: abs(x["SHAP_Value"]), reverse=True)

    return {
        "predicted_class": predicted_class,
        "probabilities": [float(p) for p in probabilities],
        "feature_contributions": feature_contributions[:10],
        "has_shap": shap_values is not None,
        "disclaimer": (
            "Model explanation — not clinical interpretation. These contributions describe "
            "mathematical model behavior and must not be construed as clinical causality or diagnosis."
        ),
    }
