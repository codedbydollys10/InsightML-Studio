"""
InsightML Studio — Feature Engineering
=======================================
Domain-aware feature generation and selection utilities.
Works on the *raw* DataFrame before preprocessing.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression

from utils.helpers import get_logger

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame, target_col: str, problem_type: str) -> pd.DataFrame:
    """
    Add domain-aware derived features.
    Runs *before* preprocessing / encoding.
    Only numeric columns are modified here.
    """
    df = df.copy()

    # Banking-domain features (safe — only added if source columns exist)
    df = _banking_features(df)

    # General ratio / log-transform features for skewed numeric columns
    df = _log_transform_skewed(df, exclude=[target_col])

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


def select_features_variance(
    X: pd.DataFrame,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove near-zero-variance features."""
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(X)
    mask = sel.get_support()
    selected = X.columns[mask].tolist()
    dropped = X.columns[~mask].tolist()
    if dropped:
        logger.info("Variance threshold dropped %d features: %s", len(dropped), dropped[:10])
    return X[selected], selected


def get_mutual_info_scores(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    n_top: int = 20,
) -> pd.DataFrame:
    """Return top-N features by mutual information with the target."""
    try:
        if "regression" in problem_type:
            scores = mutual_info_regression(X, y, random_state=42)
        else:
            scores = mutual_info_classif(X, y, random_state=42)

        mi_df = pd.DataFrame({"feature": X.columns, "mi_score": scores})
        return mi_df.sort_values("mi_score", ascending=False).head(n_top).reset_index(drop=True)
    except Exception as exc:
        logger.warning("Mutual info failed: %s", exc)
        return pd.DataFrame(columns=["feature", "mi_score"])


def _banking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add banking-domain derived features when source columns exist."""

    def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
        return a / b.replace(0, np.nan)

    col_map = {c.lower(): c for c in df.columns}

    if "current_balance" in col_map and "previous_month_end_balance" in col_map:
        c = col_map["current_balance"]
        p = col_map["previous_month_end_balance"]
        df["balance_change"] = df[c] - df[p]
        df["balance_change_pct"] = safe_ratio(df["balance_change"], df[p])

    if "current_month_credit" in col_map and "current_month_debit" in col_map:
        cc = col_map["current_month_credit"]
        cd = col_map["current_month_debit"]
        df["credit_debit_ratio"] = safe_ratio(df[cc], df[cd])
        df["net_monthly_flow"] = df[cc] - df[cd]

    if "average_monthly_balance_prevq" in col_map and "average_monthly_balance_prevq2" in col_map:
        q1 = col_map["average_monthly_balance_prevq"]
        q2 = col_map["average_monthly_balance_prevq2"]
        df["qoq_balance_change"] = df[q1] - df[q2]
        df["qoq_balance_change_pct"] = safe_ratio(df["qoq_balance_change"], df[q2])

    if "vintage" in col_map:
        v = col_map["vintage"]
        df["tenure_years"] = df[v] / 365
        df["is_new_customer"] = (df[v] < 365).astype(int)

    return df


def _log_transform_skewed(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    """Log1p-transform numeric columns with skewness > 2.

    When the DataFrame contains only a small number of rows, sample skewness is
    unreliable. In that case, generate log features for all non-negative numeric
    columns to preserve the training feature schema during prediction.
    """
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]
    if len(df) < 3:
        for col in numeric_cols:
            if df[col].min() >= 0 and not col.endswith("_log"):
                df[f"{col}_log"] = np.log1p(df[col])
        return df

    for col in numeric_cols:
        try:
            skew = df[col].dropna().skew()
            if abs(skew) > 2 and df[col].min() >= 0 and not col.endswith("_log"):
                df[f"{col}_log"] = np.log1p(df[col])
        except Exception:
            pass
    return df
