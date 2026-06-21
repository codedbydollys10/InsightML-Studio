"""
InsightML Studio — Prediction Engine
======================================
Single-record and batch prediction with probability + business recommendation.
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from config import BEST_MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURE_COLS_PATH, METADATA_PATH
from utils.helpers import get_logger, load_json
from utils.feature_engineering import engineer_features

logger = get_logger(__name__)


def load_artefacts() -> dict:
    """Load all saved model artefacts. Returns empty dict if not found."""
    if not BEST_MODEL_PATH.exists():
        return {}
    artefacts = {
        "model":         joblib.load(BEST_MODEL_PATH),
        "preprocessor":  joblib.load(SCALER_PATH),
        "feature_names": joblib.load(FEATURE_COLS_PATH),
        "label_enc":     joblib.load(ENCODER_PATH) if ENCODER_PATH.exists() else None,
        "metadata":      load_json(METADATA_PATH) if METADATA_PATH.exists() else {},
    }
    return artefacts


def predict_single(
    row: dict,
    artefacts: dict,
) -> dict:
    """
    Predict for a single row (dict of raw feature values).

    Returns
    -------
    dict with keys: prediction, probability, confidence, risk_level, recommendation
    """
    model         = artefacts["model"]
    preprocessor  = artefacts["preprocessor"]
    feature_names = artefacts["feature_names"]
    label_enc     = artefacts["label_enc"]
    meta          = artefacts.get("metadata", {})

    df_row = pd.DataFrame([row])
    df_row = engineer_features(df_row, meta.get("target_col", ""), meta.get("problem_type", ""))
    try:
        X = preprocessor.transform(df_row)
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        raise

    full_feature_names = _get_preprocessor_feature_names(preprocessor)
    X_full = pd.DataFrame(X, columns=full_feature_names if X.shape[1] == len(full_feature_names) else None)
    missing = set(feature_names) - set(X_full.columns)
    if missing:
        logger.warning("Missing transformed features at prediction time, filling zeros: %s", missing)
        X_full = X_full.reindex(columns=feature_names, fill_value=0)
    else:
        X_full = X_full[feature_names]
    X_df = X_full

    prediction_raw = model.predict(X_df)[0]

    result: dict = {"prediction_raw": prediction_raw}

    if label_enc is not None:
        result["prediction"] = label_enc.inverse_transform([int(prediction_raw)])[0]
    else:
        result["prediction"] = prediction_raw

    # Probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_df)[0]
        if len(proba) == 2:
            prob = float(proba[1])
        else:
            prob = float(proba.max())
        result["probability"] = round(prob, 4)
        result["confidence"]  = round(max(proba), 4)
        result["risk_level"]  = _risk_level(prob)
        result["recommendation"] = _recommendation(prob, meta.get("problem_type", ""))
    else:
        result["probability"]    = None
        result["confidence"]     = None
        result["risk_level"]     = "Unknown"
        result["recommendation"] = "No probability available."

    return result


def predict_batch(df: pd.DataFrame, artefacts: dict) -> pd.DataFrame:
    """Run prediction on an entire DataFrame, appending prediction columns."""
    model         = artefacts["model"]
    preprocessor  = artefacts["preprocessor"]
    feature_names = artefacts["feature_names"]
    label_enc     = artefacts["label_enc"]
    meta          = artefacts.get("metadata", {})

    df = engineer_features(df.copy(), meta.get("target_col", ""), meta.get("problem_type", ""))
    X = preprocessor.transform(df)
    full_feature_names = _get_preprocessor_feature_names(preprocessor)
    X_full = pd.DataFrame(X, columns=full_feature_names if X.shape[1] == len(full_feature_names) else None)
    missing = set(feature_names) - set(X_full.columns)
    if missing:
        logger.warning("Missing transformed features for batch prediction, filling zeros: %s", missing)
        X_full = X_full.reindex(columns=feature_names, fill_value=0)
    else:
        X_full = X_full[feature_names]
    X_df = X_full

    preds = model.predict(X_df)

    result_df = df.copy().reset_index(drop=True)

    if label_enc is not None:
        result_df["Prediction"] = label_enc.inverse_transform(preds.astype(int))
    else:
        result_df["Prediction"] = preds

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_df)
        if proba.shape[1] == 2:
            result_df["Probability"] = proba[:, 1].round(4)
        else:
            result_df["Probability"] = proba.max(axis=1).round(4)
        result_df["Risk Level"] = result_df["Probability"].apply(_risk_level)

    return result_df


def _get_preprocessor_feature_names(preprocessor) -> list[str]:
    """Recover feature names from a fitted ColumnTransformer."""
    names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if transformer == 'drop' or cols == 'drop':
            continue
        if hasattr(transformer, 'named_steps') and 'ohe' in transformer.named_steps:
            encoder = transformer.named_steps['ohe']
            names.extend(encoder.get_feature_names_out(cols).tolist())
        elif hasattr(transformer, 'named_steps') and 'ord' in transformer.named_steps:
            encoder = transformer.named_steps['ord']
            names.extend(encoder.get_feature_names_out(cols).tolist())
        else:
            names.extend(cols)
    return names


def _risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "🔴 High"
    if prob >= 0.45:
        return "🟡 Medium"
    return "🟢 Low"


def _recommendation(prob: float, problem_type: str) -> str:
    if "classification" not in problem_type:
        return "Review prediction value for action."

    if prob >= 0.75:
        return ("⚠️ High churn probability. Immediate retention intervention recommended: "
                "offer a personalised incentive or dedicated relationship manager.")
    if prob >= 0.45:
        return ("⚡ Moderate churn risk. Monitor closely and consider proactive outreach "
                "or loyalty rewards in the next 30 days.")
    return ("✅ Low churn probability. Customer appears stable. "
            "Continue regular engagement to maintain satisfaction.")