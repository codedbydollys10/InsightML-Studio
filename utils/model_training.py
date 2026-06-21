"""
InsightML Studio — Model Training
===================================
Trains a full suite of ML models, returns a sorted leaderboard,
and persists the best model + artefacts to disk.
"""

from __future__ import annotations
import logging
import time
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from config import (
    RANDOM_STATE, CV_FOLDS,
    BEST_MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURE_COLS_PATH, METADATA_PATH,
)
from utils.helpers import get_logger, save_json, Timer

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ── Model catalogs ──────────────────────────────────────────────────────────

def _classification_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    return {
        "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree":          DecisionTreeClassifier(random_state=random_state),
        "Random Forest":          RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        "Extra Trees":            ExtraTreesClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        "Gradient Boosting":      GradientBoostingClassifier(random_state=random_state),
        "AdaBoost":               AdaBoostClassifier(random_state=random_state),
        "KNN":                    KNeighborsClassifier(n_jobs=-1),
        "SVM":                    SVC(probability=True, random_state=random_state),
        "Naive Bayes":            GaussianNB(),
        "XGBoost":                xgb.XGBClassifier(
                                      use_label_encoder=False,
                                      eval_metric="logloss",
                                      random_state=random_state,
                                      n_jobs=-1,
                                      verbosity=0,
                                  ),
        "LightGBM":               lgb.LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1),
        "CatBoost":               cb.CatBoostClassifier(random_state=random_state, verbose=0),
        "ANN":                    MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=random_state),
    }


def _regression_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    return {
        "Linear Regression":      LinearRegression(n_jobs=-1),
        "Ridge":                  Ridge(random_state=random_state),
        "Lasso":                  Lasso(random_state=random_state),
        "ElasticNet":             ElasticNet(random_state=random_state),
        "Decision Tree":          DecisionTreeRegressor(random_state=random_state),
        "Random Forest":          RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        "Gradient Boosting":      GradientBoostingRegressor(random_state=random_state),
        "Extra Trees":            ExtraTreesRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        "XGBoost":                xgb.XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0),
        "LightGBM":               lgb.LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1),
        "CatBoost":               cb.CatBoostRegressor(random_state=random_state, verbose=0),
        "ANN":                    MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=random_state),
    }


def get_model_names(problem_type: str) -> list[str]:
    """Return available model names for the current problem type."""
    is_clf = "classification" in problem_type
    models = _classification_models() if is_clf else _regression_models()
    return list(models.keys())


# ── Main training function ──────────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    selected_models: list[str] | None = None,
    progress_callback=None,   # callable(model_name, pct_done)
) -> pd.DataFrame:
    """
    Train all models and return a leaderboard DataFrame.

    Parameters
    ----------
    progress_callback : optional callable(name: str, fraction: float)
        Used to push live progress to the Streamlit UI.

    Returns
    -------
    pd.DataFrame with columns:
        Model | CV Score | Std | Train Time | Primary Metric
    """
    is_clf = "classification" in problem_type
    models = _classification_models() if is_clf else _regression_models()

    if selected_models is not None:
        invalid_models = [name for name in selected_models if name not in models]
        if invalid_models:
            raise ValueError(f"Unknown selected models: {invalid_models}")
        models = {name: models[name] for name in selected_models}

    if not models:
        return pd.DataFrame(
            columns=["Model", "CV Score", "Std", "Train Time", "Metric", "_model_obj"]
        )

    cv_scorer = "roc_auc" if problem_type == "binary_classification" \
        else "f1_weighted" if "classification" in problem_type \
        else "r2"

    cv = (
        StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        if is_clf
        else KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    )

    rows: list[dict] = []
    n = len(models)

    for i, (name, model) in enumerate(models.items()):
        try:
            with Timer() as t:
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring=cv_scorer, n_jobs=-1,
                    error_score="raise",
                )
            # Refit on full training set
            model.fit(X_train, y_train)

            rows.append({
                "Model":        name,
                "CV Score":     round(scores.mean(), 4),
                "Std":          round(scores.std(), 4),
                "Train Time":   round(t.elapsed, 2),
                "Metric":       cv_scorer.upper().replace("_", " "),
                "_model_obj":   model,
            })
            logger.info("✓ %s  cv=%.4f ± %.4f  (%.2fs)", name, scores.mean(), scores.std(), t.elapsed)

        except Exception as exc:
            logger.warning("✗ %s failed: %s", name, exc)
            rows.append({
                "Model":        name,
                "CV Score":     np.nan,
                "Std":          np.nan,
                "Train Time":   np.nan,
                "Metric":       cv_scorer.upper(),
                "_model_obj":   None,
            })

        if progress_callback:
            progress_callback(name, (i + 1) / n)

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("CV Score", ascending=False)
        .reset_index(drop=True)
    )
    return leaderboard


def save_best_model(
    leaderboard: pd.DataFrame,
    preprocessor,
    feature_names: list[str],
    label_enc,
    target_col: str,
    problem_type: str,
    metadata_extra: dict | None = None,
) -> str:
    """Persist the top model + artefacts. Returns model name."""
    best_row = leaderboard.dropna(subset=["CV Score"]).iloc[0]
    best_name = best_row["Model"]
    best_model = best_row["_model_obj"]

    joblib.dump(best_model,    BEST_MODEL_PATH)
    joblib.dump(preprocessor,  SCALER_PATH)
    if label_enc is not None:
        joblib.dump(label_enc, ENCODER_PATH)
    joblib.dump(feature_names, FEATURE_COLS_PATH)

    metadata = {
        "best_model":    best_name,
        "cv_score":      best_row["CV Score"],
        "metric":        best_row["Metric"],
        "target_col":    target_col,
        "problem_type":  problem_type,
        "feature_count": len(feature_names),
        **(metadata_extra or {}),
    }
    save_json(metadata, METADATA_PATH)

    logger.info("Best model saved: %s (%.4f %s)", best_name, best_row["CV Score"], best_row["Metric"])
    return best_name


def load_trained_artefacts():
    """Load saved model + artefacts from disk."""
    model       = joblib.load(BEST_MODEL_PATH)
    preprocessor= joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_COLS_PATH)
    label_enc   = joblib.load(ENCODER_PATH) if ENCODER_PATH.exists() else None
    from utils.helpers import load_json
    meta = load_json(METADATA_PATH) if METADATA_PATH.exists() else {}
    return model, preprocessor, feature_names, label_enc, meta