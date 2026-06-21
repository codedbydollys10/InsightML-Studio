"""
InsightML Studio — Model Evaluation
=====================================
Computes the full suite of metrics and returns plot-ready data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import learning_curve, StratifiedKFold, KFold
from sklearn.preprocessing import label_binarize
from config import RANDOM_STATE, CV_FOLDS
from utils.helpers import get_logger

logger = get_logger(__name__)


def evaluate_classifier(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Full classification evaluation dict."""
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
        if y_pred_proba.shape[1] == 2:
            y_pred_proba_1d = y_pred_proba[:, 1]
            y_pred = (y_pred_proba_1d >= threshold).astype(int)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_pred_proba_1d = None
    else:
        y_pred = model.predict(X_test)
        y_pred_proba_1d = None

    results: dict = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_pred": y_pred.tolist(),
        "y_test": y_test.tolist(),
    }

    if y_pred_proba_1d is not None:
        try:
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 2:
                y_bin = label_binarize(y_test, classes=np.unique(y_test))
                results["roc_auc"] = round(roc_auc_score(y_bin, y_pred_proba, average="weighted", multi_class="ovr"), 4)
                results["pr_auc"] = round(average_precision_score(y_bin, y_pred_proba, average="weighted"), 4)
            else:
                results["roc_auc"] = round(roc_auc_score(y_test, y_pred_proba_1d), 4)
                results["pr_auc"] = round(average_precision_score(y_test, y_pred_proba_1d), 4)
        except Exception:
            results["roc_auc"] = None
            results["pr_auc"] = None

        if y_pred_proba_1d is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba_1d)
            results["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

            prec, rec, _ = precision_recall_curve(y_test, y_pred_proba_1d)
            results["pr_curve"] = {"precision": prec.tolist(), "recall": rec.tolist()}
            results["y_pred_proba"] = y_pred_proba_1d.tolist()

    # Specificity
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        results["specificity"] = round(tn / (tn + fp) if (tn + fp) else 0.0, 4)
    else:
        tps = np.diag(cm)
        fps = cm.sum(axis=0) - tps
        tns = cm.sum() - (fps + cm.sum(axis=1) - tps)
        spec_per_class = np.divide(tns, tns + fps, out=np.zeros_like(tns, dtype=float), where=(tns + fps) != 0)
        results["specificity"] = round(np.average(spec_per_class, weights=cm.sum(axis=1)), 4)

    return results


def evaluate_regressor(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Full regression evaluation dict."""
    y_pred = model.predict(X_test)

    mae  = float(mean_absolute_error(y_test, y_pred))
    mse  = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_test, y_pred))
    try:
        mape = float(mean_absolute_percentage_error(y_test, y_pred))
    except Exception:
        mape = None

    return {
        "mae":      round(mae, 4),
        "mse":      round(mse, 4),
        "rmse":     round(rmse, 4),
        "r2":       round(r2, 4),
        "mape":     round(mape, 4) if mape is not None else None,
        "y_pred":   y_pred.tolist(),
        "y_test":   y_test.tolist(),
        "residuals": (y_test.values - y_pred).tolist(),
    }


def compute_learning_curve(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    cv: int = CV_FOLDS,
) -> dict:
    """Return train/val learning curve arrays."""
    scoring = "roc_auc" if problem_type == "binary_classification" else \
              "f1_weighted" if "classification" in problem_type else "r2"

    cv_obj = (
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        if "classification" in problem_type
        else KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    )

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=cv_obj,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1,
    )

    return {
        "train_sizes": train_sizes.tolist(),
        "train_mean":  train_scores.mean(axis=1).tolist(),
        "train_std":   train_scores.std(axis=1).tolist(),
        "val_mean":    val_scores.mean(axis=1).tolist(),
        "val_std":     val_scores.std(axis=1).tolist(),
        "scoring":     scoring,
    }


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    """Extract feature importance from tree-based or linear models."""
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if imp is None:
        return None

    imp = np.array(imp).flatten()
    n = min(len(imp), len(feature_names))
    df = pd.DataFrame({
        "feature":    feature_names[:n],
        "importance": imp[:n],
    })
    return df.sort_values("importance", ascending=False).reset_index(drop=True)