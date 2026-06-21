"""
InsightML Studio — General Helpers
====================================
Tiny utility functions used across multiple modules.
"""

from __future__ import annotations
import logging
import time
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Logging setup ───────────────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(level)
    return log


# ── Timing ─────────────────────────────────────────────────────────────────
class Timer:
    """Simple context-manager timer."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.elapsed:.3f}s"


# ── JSON helpers ────────────────────────────────────────────────────────────
def save_json(obj: dict, path: Path) -> None:
    """Serialise *obj* to *path* handling numpy types."""

    class _Encoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, cls=_Encoder))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


# ── DataFrame helpers ────────────────────────────────────────────────────────
def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def df_memory(df: pd.DataFrame) -> str:
    return human_size(df.memory_usage(deep=True).sum())


def infer_id_columns(df: pd.DataFrame) -> list[str]:
    """Heuristically identify ID-like columns (high cardinality, no signal)."""
    candidates = []
    for col in df.columns:
        if col.lower() in {"id", "customer_id", "user_id", "index", "row_id"}:
            candidates.append(col)
        elif df[col].dtype in (np.int64, np.int32) and df[col].nunique() == len(df):
            candidates.append(col)
    return candidates


def infer_datetime_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(df[col].dropna().head(100), infer_datetime_format=True)
            candidates.append(col)
        except Exception:
            pass
    return candidates


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def format_metric(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def percentage(value: float) -> str:
    return f"{value * 100:.2f}%"