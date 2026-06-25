"""
InsightML Studio — Central Configuration
=========================================
All tuneable knobs in one place so nothing is hard-coded elsewhere.
"""

from __future__ import annotations
import os
from pathlib import Path

ROOT_DIR       = Path(__file__).parent
DATA_RAW_DIR   = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR  = ROOT_DIR / "data" / "processed"
MODELS_DIR     = ROOT_DIR / "models"
REPORTS_DIR    = ROOT_DIR / "reports"
ASSETS_DIR     = ROOT_DIR / "assets"

for _d in (DATA_RAW_DIR, DATA_PROC_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Model artefact filenames ────────────────────────────────────────────────
BEST_MODEL_PATH      = MODELS_DIR / "best_model.pkl"
SCALER_PATH          = MODELS_DIR / "scaler.pkl"
ENCODER_PATH         = MODELS_DIR / "encoder.pkl"
FEATURE_COLS_PATH    = MODELS_DIR / "feature_columns.pkl"
METADATA_PATH        = MODELS_DIR / "metadata.json"
PROCESSED_DATA_PATH  = DATA_PROC_DIR / "processed_dataset.csv"

# ── Training ───────────────────────────────────────────────────────────────
RANDOM_STATE   : int   = 42
TEST_SIZE      : float = 0.20
CV_FOLDS       : int   = 5
N_OPTUNA_TRIALS: int   = 30        # per model during HPO
TOP_N_TUNE     : int   = 3         # how many top models to auto-tune

# ── Preprocessing ──────────────────────────────────────────────────────────
HIGH_CARDINALITY_THRESHOLD : int   = 20   # unique values → target-encode
VARIANCE_THRESHOLD         : float = 0.01
MISSING_DROP_THRESHOLD     : float = 0.60  # drop col if >60 % missing
OUTLIER_IQR_FACTOR         : float = 1.5

# ── UI palette (Streamlit CSS injected in app.py) ──────────────────────────
COLOR_PRIMARY   = "#6C63FF"
COLOR_SECONDARY = "#00D4FF"
COLOR_SUCCESS   = "#00C896"
COLOR_WARNING   = "#FFA500"
COLOR_DANGER    = "#FF4B4B"
COLOR_BG        = "#0E1117"
COLOR_CARD      = "#1A1D2E"
COLOR_TEXT      = "#FAFAFA"
COLOR_MUTED     = "#8B8FA8"

# ── Plotly theme shared across all charts ──────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"

PLOTLY_COLORS   = [
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS,
    COLOR_WARNING, COLOR_DANGER,
    "#B48EFF", "#FF6B9D", "#FFD700",
]

# ── App meta ───────────────────────────────────────────────────────────────
APP_TITLE   = "InsightML Studio"
APP_ICON    = "🧠"
APP_VERSION = "1.0.0"
TAGLINE     = "Intelligent Automated Machine Learning & Explainable Analytics Platform"