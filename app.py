from __future__ import annotations
import hashlib
import streamlit as st
import pandas as pd
import importlib
import sys
from pathlib import Path

from config import (
    APP_TITLE, APP_ICON, APP_VERSION, TAGLINE,
    DATA_RAW_DIR,
)
from utils.data_loader import load_dataset, profile_dataset
from utils.ui_theme import inject_css, sidebar_logo, sidebar_footer

PAGE_REGISTRY = {
    "Dataset Explorer":    "pages.Dataset_Explorer",
    "Automated EDA":       "pages.Automated_EDA",
    "Model Training":      "pages.Model_Training",
    "Model Comparison":    "pages.Model_Comparison",
    "Explainable AI":      "pages.Explanable_AI",
    "Prediction":          "pages.Prediction",
    "Batch Prediction":    "pages.Batch_Prediction",
    "Business Insights":   "pages.Business_Insights",
    "Vision AI Analyzer":  "pages.Vision_AI",
    "Downloads":           "pages.Downloads",
}

# Icons paired to each page (displayed in the radio label)
_PAGE_ICONS = {
    "Dataset Explorer":    "📊",
    "Automated EDA":       "🔬",
    "Model Training":      "🚀",
    "Model Comparison":    "📡",
    "Explainable AI":      "🔮",
    "Prediction":          "🎯",
    "Batch Prediction":    "📦",
    "Business Insights":   "💼",
    "Vision AI Analyzer":  "👁",
    "Downloads":           "📥",
}


def _load_module(module_path: str):
    if module_path in sys.modules:
        return importlib.reload(sys.modules[module_path])
    return importlib.import_module(module_path)


def _reload_if_loaded(module_path: str):
    if module_path in sys.modules:
        importlib.reload(sys.modules[module_path])


def _load_uploaded_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file, low_memory=False)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)


def _default_dataset_path() -> Path | None:
    candidates = sorted(DATA_RAW_DIR.glob("*.csv")) + sorted(DATA_RAW_DIR.glob("*.xlsx"))
    return candidates[0] if candidates else None


def _load_data(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    if uploaded_file is not None:
        df = _load_uploaded_data(uploaded_file)
        source = "Uploaded dataset"
        return df, source

    default_path = _default_dataset_path()
    if default_path is not None:
        df = load_dataset(default_path)
        source = f"Sample dataset ({default_path.name})"
        return df, source

    return None, "No dataset available"


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Default UI state ──────────────────────────────────────────────────────
    if "main_nav" not in st.session_state:
        st.session_state["main_nav"] = "Dataset Explorer"

    # ── Inject premium CSS ────────────────────────────────────────────────────
    inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    sidebar = st.sidebar
    sidebar_logo()

    # Navigation — with premium buttons
    sidebar.markdown(
        "<div class='sidebar-section-title'>Navigation</div>",
        unsafe_allow_html=True,
    )
    page_name = st.session_state["main_nav"]
    for page in PAGE_REGISTRY:
        label = f"{_PAGE_ICONS[page]} {page}"
        if page == page_name:
            sidebar.button(label, key=f"nav_{page}", disabled=True, use_container_width=True)
        else:
            if sidebar.button(label, key=f"nav_{page}", use_container_width=True):
                st.session_state["main_nav"] = page
                page_name = page

    # Data source section
    sidebar.markdown(
        "<div class='sidebar-section-title' style='margin-top:1rem;'>Data Source</div>",
        unsafe_allow_html=True,
    )
    uploaded_file = sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])
    if sidebar.button("↩  Reload Default Dataset", use_container_width=True):
        st.session_state.pop("data_frame", None)
        st.session_state.pop("data_profile", None)
        st.session_state.pop("data_source", None)
        st.session_state.pop("trained", None)
        st.experimental_rerun()

    # ── Vision AI short-circuit ───────────────────────────────────────────────
    if page_name == "Vision AI Analyzer":
        sidebar.markdown(
            "<div style='padding:0.5rem 1rem;font-size:0.78rem;color:var(--subtext);'>Computer Vision powered by MediaPipe & Faster R-CNN.</div>",
            unsafe_allow_html=True,
        )
        sidebar_footer()
        _reload_if_loaded("utils.vision")
        _reload_if_loaded("utils.human_analysis")
        module = _load_module(PAGE_REGISTRY[page_name])
        module.render()
        return

    # ── Dataset management ────────────────────────────────────────────────────
    if "data_frame" not in st.session_state:
        df, source = _load_data(uploaded_file)
        st.session_state["data_frame"] = df
        st.session_state["data_source"] = source
        if df is not None:
            st.session_state["data_profile"] = profile_dataset(df)
            st.session_state["data_file_name"] = uploaded_file.name if uploaded_file is not None else None
            st.session_state["data_file_hash"] = hashlib.sha256(uploaded_file.getvalue()).hexdigest() if uploaded_file is not None else None
    elif uploaded_file is not None:
        current_file = st.session_state.get("data_file_name")
        current_hash = st.session_state.get("data_file_hash")
        new_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if st.session_state.get("data_source") != "Uploaded dataset" or uploaded_file.name != current_file or new_hash != current_hash:
            df = _load_uploaded_data(uploaded_file)
            st.session_state["data_frame"] = df
            st.session_state["data_source"] = "Uploaded dataset"
            st.session_state["data_profile"] = profile_dataset(df)
            st.session_state["data_file_name"] = uploaded_file.name
            st.session_state["data_file_hash"] = new_hash
            for key in [
                "trained", "X_test", "y_test", "X_train", "y_train",
                "best_model", "feature_names", "eval_results", "label_enc",
                "shap_values", "shap_X_sample", "leaderboard"
            ]:
                st.session_state.pop(key, None)

    df = st.session_state.get("data_frame")
    profile = st.session_state.get("data_profile")
    source = st.session_state.get("data_source", "No dataset loaded")

    # Sidebar dataset info card
    if df is not None:
        sidebar.markdown(
            f"""<div class='sidebar-card'>
            <div class='sidebar-card-title'>Dataset</div>
            <div class='sidebar-card-body'>
            📄 {source}<br>
            🔢 {df.shape[0]:,} rows &nbsp;×&nbsp; {df.shape[1]} cols
            </div></div>""",
            unsafe_allow_html=True,
        )

    # Sidebar footer
    sidebar_footer()

    # ── No data guard ─────────────────────────────────────────────────────────
    if df is None:
        st.markdown(
            """<div style='text-align:center;padding:4rem 2rem;'>
            <div style='font-size:4rem;margin-bottom:1rem;'>📁</div>
            <h2 style='color:var(--text);font-weight:700;margin-bottom:0.5rem;'>No Dataset Loaded</h2>
            <p style='color:var(--subtext);font-size:1rem;max-width:500px;margin:0 auto 1.5rem;'>
            Upload a CSV or Excel file using the sidebar, or place a file in the
            <code>data/raw</code> folder to get started.
            </p></div>""",
            unsafe_allow_html=True,
        )
        return

    if profile is None:
        profile = profile_dataset(df)
        st.session_state["data_profile"] = profile

    target_col = profile.suggested_target or ""
    problem_type = profile.problem_type or ""

    # ── Route to page ─────────────────────────────────────────────────────────
    if page_name == "Dataset Explorer":
        module = _load_module(PAGE_REGISTRY[page_name])
        module.render(profile, df)
    elif page_name == "Automated EDA":
        module = _load_module(PAGE_REGISTRY[page_name])
        if not target_col or target_col not in df.columns:
            st.warning("No model target candidate was identified. Please verify dataset target column names.")
        else:
            module.render(profile, df, target_col, problem_type)
    elif page_name == "Model Training":
        module = _load_module(PAGE_REGISTRY[page_name])
        module.render(profile, df)
    elif page_name == "Model Comparison":
        module = _load_module(PAGE_REGISTRY[page_name])
        module.render()
    elif page_name == "Explainable AI":
        module = _load_module(PAGE_REGISTRY[page_name])
        module.render()
    elif page_name == "Prediction":
        module = _load_module(PAGE_REGISTRY[page_name])
        module.render(df, profile)
    elif page_name == "Batch Prediction":
        module = _load_module(PAGE_REGISTRY[page_name])
        module.render()
    elif page_name == "Business Insights":
        module = _load_module(PAGE_REGISTRY[page_name])
        if not target_col or target_col not in df.columns:
            st.warning("Target column not available for business insights.")
        else:
            module.render(df, profile, target_col)
    elif page_name == "Downloads":
        module = _load_module(PAGE_REGISTRY[page_name])
        module.render(df)
    elif page_name == "Vision AI Analyzer":
        _reload_if_loaded("utils.vision")
        _reload_if_loaded("utils.human_analysis")
        module = _load_module(PAGE_REGISTRY[page_name])
        module.render()


if __name__ == "__main__":
    main()
