"""
Main Training Pipeline for Liver Cirrhosis Survival Prediction.

Orchestrates:
1. Dataset ingestion & validation.
2. Leakage-free train/holdout split.
3. 5-algorithm benchmarking with Stratified 5-fold CV & hyperparameter optimization.
4. Model comparison sorted by Macro F1.
5. Best model selection & serialization (final_model.joblib).
6. Holdout permutation, Gini, and SHAP explainability calculations.
7. Artifact generation for reports and Streamlit dashboard.
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import (
    FINAL_MODEL_PATH,
    MODEL_METADATA_PATH,
    ARTIFACTS_DIR,
    RAW_DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    CV_SPLITS,
    TARGET_CLASSES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MEDICAL_DISCLAIMER,
)
from src.data_loader import load_raw_data
from src.preprocessing import prepare_data_splits
from src.models import get_base_models, get_hyperparameter_grids
from src.evaluation import tune_and_evaluate_model, generate_comparison_table
from src.explainability import (
    calculate_gini_importance,
    calculate_permutation_importance,
    compute_shap_explanations,
)
from src.visualization import save_static_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainPipeline")


def run_training_pipeline():
    logger.info("=" * 70)
    logger.info("STARTING LIVER CIRRHOSIS SURVIVAL PREDICTION TRAINING PIPELINE")
    logger.info("Framing: Research & Educational Multiclass Risk Classification")
    logger.info("=" * 70)

    # 1. Load and Validate Data
    logger.info("Step 1: Ingesting dataset...")
    df_raw = load_raw_data(auto_download=True)
    logger.info("Raw dataset shape: %s", df_raw.shape)

    # 2. Stratified Train / Test Split (Strictly before any transformation to prevent data leakage)
    logger.info("Step 2: Executing stratified train/test split (test_size=%.2f, random_state=%d)...", TEST_SIZE, RANDOM_STATE)
    X_train, X_test, y_train, y_test, dist_df = prepare_data_splits(
        df_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    logger.info("Train set: %d samples; Holdout Test set: %d samples", len(X_train), len(X_test))
    logger.info("Class distribution:\n%s", dist_df.to_string(index=False))

    # 3. Instantiate Classifiers and Hyperparameter Search Grids
    logger.info("Step 3: Initializing 5 benchmark classifiers...")
    base_models = get_base_models()
    param_grids = get_hyperparameter_grids()

    # 4. Train, Cross-Validate, and Evaluate Each Model
    logger.info("Step 4: Running Stratified %d-Fold Cross-Validation and Grid Search...", CV_SPLITS)
    trained_estimators = {}
    model_summaries = {}

    for model_name, pipeline in base_models.items():
        logger.info("-" * 50)
        logger.info("Training and tuning: %s", model_name)
        grid = param_grids.get(model_name)
        fitted_model, summary = tune_and_evaluate_model(
            name=model_name,
            pipeline=pipeline,
            param_grid=grid,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_splits=CV_SPLITS,
        )
        trained_estimators[model_name] = fitted_model
        model_summaries[model_name] = summary

        logger.info(
            "%s -> CV Macro F1: %.4f | Holdout Macro F1: %.4f | Holdout Acc: %.4f | Transplant Recall: %.4f",
            model_name,
            summary["cv_macro_f1_mean"],
            summary["macro_f1"],
            summary["accuracy"],
            summary["class_metrics"]["Transplant"]["recall"],
        )

    # 5. Model Comparison & Best Model Selection
    logger.info("=" * 70)
    logger.info("Step 5: Comparative Evaluation & Model Selection")
    comparison_df = generate_comparison_table(model_summaries)
    logger.info("Comparative Algorithm Benchmark Table:\n%s", comparison_df.to_string(index=False))

    # Primary selection criteria: Macro F1 (unweighted average across classes to prioritize minority Transplant class)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline = trained_estimators[best_model_name]
    best_summary = model_summaries[best_model_name]

    logger.info(
        "Selected Best Model: '%s' with Holdout Macro F1 = %.4f (Accuracy = %.4f).",
        best_model_name,
        best_summary["macro_f1"],
        best_summary["accuracy"],
    )
    logger.info("Rationale: Macro F1 treats all three outcome classes equally, ensuring the model does not sacrifice minority Transplant class detection for high majority-class accuracy.")

    # Extract Random Forest Gini importance specifically (since Logistic Regression has linear coefficients)
    rf_pipeline = trained_estimators.get("Random Forest", best_pipeline)
    gini_df = calculate_gini_importance(rf_pipeline, top_n=15)
    perm_df = calculate_permutation_importance(best_pipeline, X_test, y_test, top_n=15, random_state=RANDOM_STATE)

    # Combine importances into unified comparison CSV
    combined_imp = perm_df.merge(
        gini_df.rename(columns={"Feature": "Feature", "Gini_Importance": "Best_Model_Gini_Importance"}),
        on="Feature",
        how="left",
    ).fillna(0.0)

    # 7. Generate Artifacts and Visual Assets
    logger.info("Step 7: Serializing artifacts and visual figures...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save Comparison CSV
    comp_csv_path = ARTIFACTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comp_csv_path, index=False)
    logger.info("Saved model comparison to %s", comp_csv_path)

    # Save Feature Importance CSV
    imp_csv_path = ARTIFACTS_DIR / "feature_importance.csv"
    combined_imp.to_csv(imp_csv_path, index=False)
    logger.info("Saved feature importance to %s", imp_csv_path)

    # Save Confusion Matrix Image
    cm_array = np.array(best_summary["confusion_matrix"])
    cm_img_path = save_static_confusion_matrix(cm_array, list(TARGET_CLASSES.values()))
    logger.info("Saved confusion matrix image to %s", cm_img_path)

    # Save Metrics JSON
    metrics_export = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "best_model": best_model_name,
        "primary_metric": "macro_f1",
        "best_metrics": {
            "cv_macro_f1_mean": best_summary.get("cv_macro_f1_mean"),
            "cv_macro_f1_std": best_summary.get("cv_macro_f1_std"),
            "holdout_accuracy": best_summary.get("accuracy"),
            "holdout_macro_f1": best_summary.get("macro_f1"),
            "holdout_weighted_f1": best_summary.get("weighted_f1"),
            "holdout_macro_precision": best_summary.get("macro_precision"),
            "holdout_macro_recall": best_summary.get("macro_recall"),
            "holdout_roc_auc_ovr": best_summary.get("roc_auc_ovr"),
            "class_metrics": best_summary.get("class_metrics"),
        },
        "all_models": {k: {m: v for m, v in s.items() if m not in ["y_pred", "y_proba", "y_true"]} for k, s in model_summaries.items()},
        "dataset_summary": {
            "total_samples": len(df_raw),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "class_distribution": dist_df.to_dict(orient="records"),
        },
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }

    metrics_json_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    logger.info("Saved evaluation metrics JSON to %s", metrics_json_path)

    # Save EDA Summary JSON
    eda_summary = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "missing_counts": df_raw.isna().sum().to_dict(),
        "summary_statistics": df_raw.describe().to_dict(),
    }
    with open(ARTIFACTS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2, default=str)

    # 8. Serialize Final Best Model and Provenance Metadata
    logger.info("Step 8: Serializing final model to %s...", FINAL_MODEL_PATH)
    joblib.dump(best_pipeline, FINAL_MODEL_PATH)

    metadata = {
        "model_name": best_model_name,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_hyperparameters": best_summary.get("best_params", {}),
        "target_mapping": TARGET_CLASSES,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "split_config": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_splits": CV_SPLITS,
        },
        "evaluation_summary": {
            "holdout_macro_f1": best_summary.get("macro_f1"),
            "holdout_accuracy": best_summary.get("accuracy"),
            "death_recall": best_summary.get("class_metrics", {}).get("Death", {}).get("recall"),
            "censored_recall": best_summary.get("class_metrics", {}).get("Censored", {}).get("recall"),
            "transplant_recall": best_summary.get("class_metrics", {}).get("Transplant", {}).get("recall"),
        },
        "disclaimer": MEDICAL_DISCLAIMER,
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to %s", MODEL_METADATA_PATH)

    # Print Final Summary
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Dataset Loaded: {len(df_raw)} records ({len(X_train)} Train, {len(X_test)} Test)")
    print(f"Best Model Selected: {best_model_name}")
    print(f"Holdout Accuracy:    {best_summary.get('accuracy', 0.0):.4f}")
    print(f"Holdout Macro F1:    {best_summary.get('macro_f1', 0.0):.4f}")
    print(f"Holdout Weighted F1: {best_summary.get('weighted_f1', 0.0):.4f}")
    print(f"Death Class Recall:      {best_summary.get('class_metrics', {}).get('Death', {}).get('recall', 0.0):.4f}")
    print(f"Censored Class Recall:   {best_summary.get('class_metrics', {}).get('Censored', {}).get('recall', 0.0):.4f}")
    print(f"Transplant Class Recall: {best_summary.get('class_metrics', {}).get('Transplant', {}).get('recall', 0.0):.4f}")
    print(f"\nFinal pipeline saved to: {FINAL_MODEL_PATH}")
    print(f"Metadata saved to:       {MODEL_METADATA_PATH}")
    print("=" * 70 + "\n")

    return best_model_name, best_summary


if __name__ == "__main__":
    run_training_pipeline()
