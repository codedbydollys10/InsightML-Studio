"""
InsightML Studio — Preprocessing Pipeline
==========================================
Builds a fully reproducible sklearn Pipeline that handles:
  • ID / datetime column removal
  • Missing-value imputation
  • Encoding (OHE for low-cardinality, target-encoding for high-cardinality)
  • Feature scaling
  • Optional SMOTE oversampling for imbalanced classification
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from utils.helpers import get_logger

logger = get_logger(__name__)

# ── Public API ──────────────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    id_cols: list[str],
    datetime_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    high_cardinality_cols: list[str],
    drop_missing_threshold: float = 0.60,
    scaler_type: str = "standard",          # "standard" | "minmax"
) -> tuple[pd.DataFrame, pd.Series, Pipeline, list[str], LabelEncoder | None]:
    """
    Clean and encode *df*.

    Returns
    -------
    X : pd.DataFrame          — processed feature matrix
    y : pd.Series             — encoded target
    preprocessor : Pipeline   — fitted ColumnTransformer
    feature_names : list[str] — column names after transformation
    label_enc : LabelEncoder | None
    """

    # 1. Drop columns we should not use as features
    drop_cols = set(id_cols) | set(datetime_cols) | {target_col}

    # 2. Drop high-missing columns
    for col in df.columns:
        if col in drop_cols:
            continue
        if df[col].isnull().mean() > drop_missing_threshold:
            logger.info("Dropping high-missing column: %s", col)
            drop_cols.add(col)

    df_clean = df.drop(columns=list(drop_cols), errors="ignore").copy()

    # 3. Separate feature sets
    feat_num = [c for c in numeric_cols if c in df_clean.columns and c != target_col]
    feat_cat_low = [
        c for c in categorical_cols
        if c in df_clean.columns
        and c not in high_cardinality_cols
        and c != target_col
    ]
    feat_cat_high = [
        c for c in high_cardinality_cols
        if c in df_clean.columns and c != target_col
    ]

    logger.info(
        "Features — numeric: %d | cat_ohe: %d | cat_target: %d",
        len(feat_num), len(feat_cat_low), len(feat_cat_high),
    )

    # 4. Build sub-pipelines
    if scaler_type == "robust":
        ScalerCls = RobustScaler
    elif scaler_type == "minmax":
        ScalerCls = MinMaxScaler
    else:
        ScalerCls = StandardScaler

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", ScalerCls()),
    ])

    cat_ohe_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    cat_target_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # We use ordinal encoding as a stand-in; true target-encoding
        # requires the target, so we handle that separately below.
        ("ord", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            max_categories=20,
        )),
    ])

    transformers = []
    if feat_num:
        transformers.append(("num", num_pipeline, feat_num))
    if feat_cat_low:
        transformers.append(("cat_ohe", cat_ohe_pipeline, feat_cat_low))
    if feat_cat_high:
        transformers.append(("cat_high", cat_target_pipeline, feat_cat_high))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # 5. Extract X and y
    X_raw = df_clean
    y_raw = df[target_col].copy()

    # Encode target
    label_enc: LabelEncoder | None = None
    if y_raw.dtype == object or str(y_raw.dtype) == "string":
        label_enc = LabelEncoder()
        y = pd.Series(label_enc.fit_transform(y_raw), name=target_col)
    else:
        y = y_raw.astype(float).reset_index(drop=True)

    y = y.reset_index(drop=True)
    X_raw = X_raw.reset_index(drop=True)

    # 6. Fit-transform X
    X_transformed = preprocessor.fit_transform(X_raw)

    # 7. Recover feature names
    feature_names = _get_feature_names(preprocessor, feat_num, feat_cat_low, feat_cat_high)

    X = pd.DataFrame(X_transformed, columns=feature_names)

    logger.info(
        "Preprocessed shape: %s  | target classes: %s",
        X.shape,
        sorted(y.unique().tolist()),
    )

    return X, y, preprocessor, feature_names, label_enc


def _get_feature_names(
    preprocessor: ColumnTransformer,
    feat_num: list[str],
    feat_cat_low: list[str],
    feat_cat_high: list[str],
) -> list[str]:
    """Reconstruct feature names after ColumnTransformer."""
    names: list[str] = []

    for name, transformer, _ in preprocessor.transformers_:
        if name == "num":
            names.extend(feat_num)
        elif name == "cat_ohe":
            ohe = transformer.named_steps["ohe"]
            ohe_names = ohe.get_feature_names_out(feat_cat_low)
            names.extend(ohe_names.tolist())
        elif name == "cat_high":
            ohe = transformer.named_steps["ord"]
            ohe_names = ohe.get_feature_names_out(feat_cat_high)
            names.extend(ohe_names.tolist())

    return names


# ── SMOTE helper (called from model_training) ────────────────────────────────
def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    sampling_strategy: str | float = "auto",
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE oversampling to handle class imbalance."""
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
        X_res, y_res = sm.fit_resample(X, y)
        logger.info(
            "SMOTE: %d → %d samples", len(y), len(y_res)
        )
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
    except ImportError:
        logger.warning("imbalanced-learn not available; skipping SMOTE")
        return X, y