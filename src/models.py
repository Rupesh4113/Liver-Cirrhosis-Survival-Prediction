"""
Model definitions, pipeline builders, and hyperparameter search grids for all 5 clinical classifiers:
1. Multiclass Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. LightGBM
"""

import logging
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

from config.config import RANDOM_STATE
from src.preprocessing import build_full_pipeline

logger = logging.getLogger(__name__)


def get_base_models() -> Dict[str, Any]:
    """
    Instantiate the 5 clinical classifiers with class imbalance handling.

    Returns:
        Dict mapping model name to fully configured scikit-learn Pipeline.
    """
    # 1. Multiclass Logistic Regression (Requires feature scaling, handles imbalance via class_weight)
    lr_clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    lr_pipeline = build_full_pipeline(
        classifier=lr_clf,
        scale_numeric=True,
        include_feature_engineering=True,
    )

    # 2. Random Forest (Balanced class weights, resistant to outliers)
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    rf_pipeline = build_full_pipeline(
        classifier=rf_clf,
        scale_numeric=False,
        include_feature_engineering=True,
    )

    # 3. Gradient Boosting Classifier (sklearn fallback)
    gb_clf = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=RANDOM_STATE,
    )
    gb_pipeline = build_full_pipeline(
        classifier=gb_clf,
        scale_numeric=False,
        include_feature_engineering=True,
    )

    # 4. XGBoost Classifier (Extreme Gradient Boosting for tabular data)
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    xgb_pipeline = build_full_pipeline(
        classifier=xgb_clf,
        scale_numeric=False,
        include_feature_engineering=True,
    )

    # 5. LightGBM Classifier (Balanced multiclass objective)
    lgb_clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgb_pipeline = build_full_pipeline(
        classifier=lgb_clf,
        scale_numeric=False,
        include_feature_engineering=True,
    )

    return {
        "Logistic Regression": lr_pipeline,
        "Random Forest": rf_pipeline,
        "Gradient Boosting": gb_pipeline,
        "XGBoost": xgb_pipeline,
        "LightGBM": lgb_pipeline,
    }


def get_hyperparameter_grids() -> Dict[str, Dict[str, list]]:
    """
    Define computationally reasonable hyperparameter search grids for key models.
    """
    return {
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [4, 6, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
        },
        "LightGBM": {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 5],
            "classifier__num_leaves": [15, 31],
        },
        "XGBoost": {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 5],
        },
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1.0, 5.0],
        },
        "Gradient Boosting": {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 4],
        },
    }
