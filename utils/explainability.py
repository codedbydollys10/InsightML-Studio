"""
InsightML Studio — Explainable AI
===================================
SHAP-based explanations (global + local) for any sklearn-compatible model.
"""

from __future__ import annotations
import logging
import warnings

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


def compute_shap_values(
    model,
    X: pd.DataFrame,
    n_background: int = 100,
    n_explain: int = 200,
) -> tuple | None:
    """
    Compute SHAP values for *model* on a sample of *X*.

    Returns (shap_values, explainer, X_sample) or None on failure.
    """
    try:
        import shap

        X_sample = X.sample(min(n_explain, len(X)), random_state=42).reset_index(drop=True)
        background = X.sample(min(n_background, len(X)), random_state=42)

        model_name = type(model).__name__

        # Choose explainer
        if any(k in model_name for k in ("XGB", "LGBM", "LightGBM", "CatBoost")):
            explainer = shap.TreeExplainer(model)
        elif any(k in model_name for k in ("Forest", "Tree", "Boost", "Bag", "Extra")):
            explainer = shap.TreeExplainer(model)
        elif any(k in model_name for k in ("Linear", "Logistic", "Ridge", "Lasso", "ElasticNet")):
            explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                background,
            )

        shap_values = explainer.shap_values(X_sample)

        # For binary classification, keep class-1 values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 3 and shap_arr.shape[-1] == 1:
            shap_values = np.squeeze(shap_arr, axis=-1)
        elif shap_arr.ndim == 3 and shap_arr.shape[-1] == 2:
            shap_values = shap_arr[..., 1]

        logger.info("SHAP values computed for %d samples", len(X_sample))
        return shap_values, explainer, X_sample

    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return None


def shap_summary_df(shap_values: np.ndarray, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Return mean |SHAP| per feature, sorted descending."""
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3 and shap_arr.shape[-1] == 1:
        shap_arr = np.squeeze(shap_arr, axis=-1)
    elif shap_arr.ndim == 3 and shap_arr.shape[-1] == 2:
        shap_arr = shap_arr[..., 1]

    if shap_arr.ndim != 2:
        raise ValueError(f"Expected shap_values to be 2D, got shape {shap_arr.shape}")

    mean_abs = np.abs(shap_arr).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.ravel()

    return (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def local_shap_explanation(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    row_idx: int = 0,
) -> pd.DataFrame:
    """Return SHAP contributions for a single prediction."""
    row_shap = np.asarray(shap_values[row_idx])
    if row_shap.ndim > 1 and row_shap.shape[-1] == 1:
        row_shap = np.squeeze(row_shap, axis=-1)
    elif row_shap.ndim > 1 and row_shap.shape[-1] == 2:
        row_shap = row_shap[..., 1]

    if row_shap.ndim != 1:
        raise ValueError(f"Expected row shap values to be 1D, got shape {row_shap.shape}")

    df = pd.DataFrame({
        "feature": X_sample.columns,
        "shap_value": row_shap,
        "feature_value": X_sample.iloc[row_idx].values,
    })
    df["abs_shap"] = df["shap_value"].abs()
    return df.sort_values("abs_shap", ascending=False).reset_index(drop=True)