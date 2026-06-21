"""
InsightML Studio — Data Loader
================================
Loads the raw dataset, performs deep profiling, and stores a
`DataProfile` object that every other module can consume.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import DATA_RAW_DIR, HIGH_CARDINALITY_THRESHOLD, OUTLIER_IQR_FACTOR
from utils.helpers import (
    get_logger,
    df_memory,
    infer_id_columns,
    infer_datetime_columns,
)

logger = get_logger(__name__)


# ── Data Profile ────────────────────────────────────────────────────────────
@dataclass
class DataProfile:
    """Everything we know about the raw dataset in one place."""

    df: pd.DataFrame

    n_rows: int = 0
    n_cols: int = 0

    numeric_cols: list[str]       = field(default_factory=list)
    categorical_cols: list[str]   = field(default_factory=list)
    datetime_cols: list[str]      = field(default_factory=list)
    id_cols: list[str]            = field(default_factory=list)
    high_cardinality_cols: list[str] = field(default_factory=list)
    constant_cols: list[str]      = field(default_factory=list)
    bool_cols: list[str]          = field(default_factory=list)

    missing_counts: pd.Series     = field(default_factory=pd.Series)
    missing_pct: pd.Series        = field(default_factory=pd.Series)
    duplicate_rows: int           = 0

    outlier_counts: dict[str, int] = field(default_factory=dict)

    target_candidates: list[str]  = field(default_factory=list)
    suggested_target: str         = ""
    problem_type: str             = ""   # binary_classification | multiclass | regression

    memory_usage: str             = ""
    column_stats: pd.DataFrame    = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.n_rows, self.n_cols = self.df.shape
        self.memory_usage = df_memory(self.df)


# ── Loader ──────────────────────────────────────────────────────────────────
def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load CSV (or Excel) from *path*."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    logger.info("Loaded %d rows × %d cols from %s", *df.shape, path.name)
    return df


def profile_dataset(df: pd.DataFrame) -> DataProfile:
    """
    Deeply profile *df* and return a :class:`DataProfile`.
    This is the single source of truth for dataset intelligence.
    """
    profile = DataProfile(df=df.copy())

    # ── Column type inference ────────────────────────────────────────────
    id_cols   = infer_id_columns(df)
    dt_cols   = infer_datetime_columns(df)
    num_cols  = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in id_cols
    ]

    def _is_boolean_like(series: pd.Series) -> bool:
        values = set(series.dropna().astype(str).str.strip().str.lower().unique())
        allowed = {"0", "1", "true", "false", "yes", "no"}
        return values <= allowed and len(values) <= 2

    bool_cols = [
        c for c in df.columns
        if c not in id_cols and c not in dt_cols
        and _is_boolean_like(df[c])
    ]

    cat_cols  = [
        c for c in df.select_dtypes(exclude=np.number).columns
        if c not in dt_cols and c not in id_cols
    ]
    hi_card   = [c for c in cat_cols
                 if df[c].nunique() > HIGH_CARDINALITY_THRESHOLD]
    constant  = [c for c in df.columns if df[c].nunique() <= 1]

    profile.id_cols               = id_cols
    profile.datetime_cols         = dt_cols
    profile.numeric_cols          = num_cols
    profile.bool_cols             = bool_cols
    profile.categorical_cols      = cat_cols
    profile.high_cardinality_cols = hi_card
    profile.constant_cols         = constant

    # ── Missing values ───────────────────────────────────────────────────
    profile.missing_counts = df.isnull().sum()
    profile.missing_pct    = (df.isnull().mean() * 100).round(2)

    # ── Duplicates ───────────────────────────────────────────────────────
    profile.duplicate_rows = int(df.duplicated().sum())

    # ── Outliers (IQR method, numeric cols only) ─────────────────────────
    outlier_counts: dict[str, int] = {}
    for col in num_cols:
        if col in bool_cols:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - OUTLIER_IQR_FACTOR * iqr) | \
               (df[col] > q3 + OUTLIER_IQR_FACTOR * iqr)
        cnt = int(mask.sum())
        if cnt:
            outlier_counts[col] = cnt
    profile.outlier_counts = outlier_counts

    # ── Target candidates ────────────────────────────────────────────────
    profile.target_candidates = _guess_targets(df, bool_cols, num_cols, cat_cols)
    if profile.target_candidates:
        profile.suggested_target = profile.target_candidates[0]
        profile.problem_type = _detect_problem_type(
            df, profile.suggested_target
        )

    # ── Column statistics table ───────────────────────────────────────────
    profile.column_stats = _build_column_stats(df, profile)

    logger.info(
        "Profile: %d rows | %d num | %d cat | %d dt | target=%s | task=%s",
        profile.n_rows, len(num_cols), len(cat_cols), len(dt_cols),
        profile.suggested_target, profile.problem_type,
    )
    return profile


# ── Internals ───────────────────────────────────────────────────────────────
def _guess_targets(
    df: pd.DataFrame,
    bool_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
) -> list[str]:
    """
    Heuristically rank target-column candidates.

    Priority:
      1. Columns named 'target', 'label', 'churn', 'default', 'fraud', …
      2. Binary numeric columns (0/1)
      3. Low-cardinality numeric columns
      4. Low-cardinality categorical columns
    """
    known_targets = {
        "target", "label", "churn", "default", "fraud", "survived",
        "outcome", "result", "class", "y", "response", "converted",
        "purchased", "subscribed", "approved",
    }
    candidates: list[str] = []

    # exact name match
    for col in df.columns:
        if col.lower() in known_targets:
            candidates.append(col)

    # binary numerics not already added
    for col in bool_cols:
        if col not in candidates:
            candidates.append(col)

    # low-cardinality numeric (2–20 unique values)
    for col in num_cols:
        if col in candidates or col in bool_cols:
            continue
        nu = df[col].nunique()
        if 2 <= nu <= 20:
            candidates.append(col)

    # low-cardinality categorical
    for col in cat_cols:
        if col in candidates:
            continue
        nu = df[col].nunique()
        if 2 <= nu <= 20:
            candidates.append(col)

    return candidates


def detect_problem_type(df: pd.DataFrame, target: str) -> str:
    """Return the problem type of the selected target."""
    return _detect_problem_type(df, target)


def _detect_problem_type(df: pd.DataFrame, target: str) -> str:
    """Classify the ML task from the target column."""
    series = df[target].dropna()
    n_unique = series.nunique()

    def _is_boolean_like(series: pd.Series) -> bool:
        values = set(series.astype(str).str.strip().str.lower().unique())
        allowed = {"0", "1", "true", "false", "yes", "no"}
        return values <= allowed and len(values) == 2

    if n_unique == 2 or _is_boolean_like(series):
        return "binary_classification"
    if n_unique <= 20 and series.dtype not in (np.float64, np.float32):
        return "multiclass_classification"
    return "regression"


def _build_column_stats(df: pd.DataFrame, profile: DataProfile) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        n_missing = int(profile.missing_counts[col])
        missing_pct = float(profile.missing_pct[col])
        col_type = (
            "ID" if col in profile.id_cols else
            "Datetime" if col in profile.datetime_cols else
            "Boolean" if col in profile.bool_cols else
            "Numeric" if col in profile.numeric_cols else
            "Categorical"
        )
        rows.append({
            "Column": col,
            "Type": col_type,
            "Dtype": dtype,
            "Unique": n_unique,
            "Missing": n_missing,
            "Missing %": missing_pct,
            "Sample": str(df[col].dropna().iloc[0]) if n_missing < len(df) else "—",
        })
    return pd.DataFrame(rows)