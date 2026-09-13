"""
Model evaluation module.
Performs Stratified 5-Fold Cross-Validation, independent holdout evaluation,
per-class metric breakdown, confusion matrices, and ROC-AUC One-vs-Rest calculation.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from config.config import RANDOM_STATE, CV_SPLITS, TARGET_CLASSES, PRIMARY_METRIC

logger = logging.getLogger(__name__)


def evaluate_cross_validation(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, float]:
    """
    Perform Stratified K-Fold Cross-Validation strictly on training data.

    Returns:
        Dict containing mean and std of cross-validation metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "macro_precision": "precision_macro",
        "macro_recall": "recall_macro",
    }

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    return {
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "cv_macro_f1_mean": float(np.mean(cv_results["test_macro_f1"])),
        "cv_macro_f1_std": float(np.std(cv_results["test_macro_f1"])),
        "cv_weighted_f1_mean": float(np.mean(cv_results["test_weighted_f1"])),
        "cv_macro_recall_mean": float(np.mean(cv_results["test_macro_recall"])),
        "cv_macro_precision_mean": float(np.mean(cv_results["test_macro_precision"])),
    }


def evaluate_holdout(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluate fitted model on completely independent holdout test set.

    Calculates:
    - Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall
    - Per-class precision, recall, and F1
    - ROC-AUC OvR (One-vs-Rest)
    - Confusion matrix array
    """
    y_pred = model.predict(X_test)

    # Class probabilities if supported
    y_proba = None
    roc_auc_ovr = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc_ovr = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
        except Exception as exc:
            logger.warning("Could not calculate ROC-AUC OvR: %s", exc)

    # Overall metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_test, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_test, y_pred, average=None, zero_division=0)

    class_metrics = {}
    for idx, name in TARGET_CLASSES.items():
        class_metrics[name] = {
            "precision": float(per_class_prec[idx]) if idx < len(per_class_prec) else 0.0,
            "recall": float(per_class_rec[idx]) if idx < len(per_class_rec) else 0.0,
            "f1": float(per_class_f1[idx]) if idx < len(per_class_f1) else 0.0,
        }

    cm = confusion_matrix(y_test, y_pred, labels=list(TARGET_CLASSES.keys()))

    report_str = classification_report(
        y_test,
        y_pred,
        labels=list(TARGET_CLASSES.keys()),
        target_names=list(TARGET_CLASSES.values()),
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "roc_auc_ovr": roc_auc_ovr,
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str,
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist() if y_proba is not None else None,
        "y_true": y_test.tolist(),
    }


def tune_and_evaluate_model(
    name: str,
    pipeline,
    param_grid: Optional[Dict[str, list]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_splits: int = CV_SPLITS,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform cross-validation, optional hyperparameter grid search, and final holdout evaluation.
    """
    best_estimator = pipeline
    best_params = {}
    cv_metrics = {}

    if param_grid:
        logger.info("Tuning hyperparameters with Stratified %d-Fold CV for %s...", n_splits, name)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=skf,
            scoring="f1_macro",
            n_jobs=1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_idx = grid_search.best_index_
        cv_macro_f1_mean = float(grid_search.cv_results_["mean_test_score"][best_idx])
        cv_macro_f1_std = float(grid_search.cv_results_["std_test_score"][best_idx])

        cv_metrics = {
            "cv_macro_f1_mean": cv_macro_f1_mean,
            "cv_macro_f1_std": cv_macro_f1_std,
            "cv_accuracy_mean": float(np.nan),
            "cv_accuracy_std": float(np.nan),
        }
        logger.info("Best parameters for %s: %s (Best CV Macro F1: %.4f +/- %.4f)", name, best_params, cv_macro_f1_mean, cv_macro_f1_std)
    else:
        logger.info("Evaluating %s with Stratified %d-Fold Cross-Validation...", name, n_splits)
        cv_metrics = evaluate_cross_validation(pipeline, X_train, y_train, n_splits=n_splits)
        best_estimator.fit(X_train, y_train)

    # 3. Final evaluation on untouched holdout test set
    holdout_metrics = evaluate_holdout(best_estimator, X_test, y_test)

    summary = {
        "model_name": name,
        "best_params": best_params,
        **cv_metrics,
        **holdout_metrics,
    }

    return best_estimator, summary


def generate_comparison_table(summaries: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a clean DataFrame comparing all models and sort by holdout Macro F1.
    """
    rows = []
    for model_name, s in summaries.items():
        rows.append({
            "Model": model_name,
            "CV Macro F1": round(s.get("cv_macro_f1_mean", 0.0), 4),
            "Holdout Accuracy": round(s.get("accuracy", 0.0), 4),
            "Holdout Macro F1": round(s.get("macro_f1", 0.0), 4),
            "Holdout Weighted F1": round(s.get("weighted_f1", 0.0), 4),
            "Holdout Macro Precision": round(s.get("macro_precision", 0.0), 4),
            "Holdout Macro Recall": round(s.get("macro_recall", 0.0), 4),
            "Holdout ROC-AUC (OvR)": round(s.get("roc_auc_ovr", 0.0), 4) if s.get("roc_auc_ovr") is not None else np.nan,
            "Death Recall": round(s.get("class_metrics", {}).get("Death", {}).get("recall", 0.0), 4),
            "Censored Recall": round(s.get("class_metrics", {}).get("Censored", {}).get("recall", 0.0), 4),
            "Transplant Recall": round(s.get("class_metrics", {}).get("Transplant", {}).get("recall", 0.0), 4),
        })

    df_comp = pd.DataFrame(rows).sort_values(by="Holdout Macro F1", ascending=False).reset_index(drop=True)
    return df_comp
