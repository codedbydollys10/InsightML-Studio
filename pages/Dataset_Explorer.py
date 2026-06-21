"""
InsightML Studio — Dataset Explorer Page
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.visualization import fig_missing_values, fig_correlation_heatmap, fig_histogram
from utils.report_generator import to_csv_bytes
from utils.ui_theme import inject_css, page_header, kpi_row, premium_divider
from config import COLOR_PRIMARY, COLOR_SECONDARY


def render(profile, df: pd.DataFrame) -> None:
    inject_css()
    page_header(
        title="Dataset Explorer",
        icon="📊",
        subtitle="Inspect, filter, and understand your raw dataset at a glance.",
        badge="📁 Data Intelligence",
    )

    # ── Top KPI strip ────────────────────────────────────────────────────────
    kpi_row([
        {"label": "Total Rows",        "value": f"{profile.n_rows:,}",                          "icon": "🔢", "color": "purple"},
        {"label": "Columns",           "value": profile.n_cols,                                  "icon": "📋", "color": "blue"},
        {"label": "Numeric Features",  "value": len(profile.numeric_cols),                       "icon": "🔣", "color": "green"},
        {"label": "Categorical",       "value": len(profile.categorical_cols),                   "icon": "🏷️", "color": "orange"},
        {"label": "Missing Values",    "value": f"{int(profile.missing_counts.sum()):,}",        "icon": "⚠️", "color": "red"},
    ])

    premium_divider()

    # ── Dataset preview ──────────────────────────────────────────────────────
    with st.expander("🔍 Dataset Preview", expanded=True):
        n_rows = st.slider("Rows to display", 5, 100, 20)
        col_filter = st.multiselect(
            "Filter columns", df.columns.tolist(), default=df.columns.tolist()[:8]
        )
        st.dataframe(df[col_filter].head(n_rows), use_container_width=True)

    # ── Column statistics ─────────────────────────────────────────────────────
    with st.expander("📋 Column Statistics", expanded=True):
        search = st.text_input("Search columns", placeholder="Type to filter…")
        stats = profile.column_stats.copy()
        if search:
            stats = stats[stats["Column"].str.contains(search, case=False)]
        st.dataframe(
            stats.style.background_gradient(subset=["Missing %"], cmap="Reds"),
            use_container_width=True,
        )

    # ── Interactive filtering ─────────────────────────────────────────────────
    with st.expander("🎛️ Interactive Filter & Explore"):
        col_sel = st.selectbox("Select column to explore", df.columns.tolist())
        col1, col2 = st.columns(2)
        with col1:
            if df[col_sel].dtype in (np.float64, np.int64, np.float32, np.int32):
                mn, mx = float(df[col_sel].min()), float(df[col_sel].max())
                rng = st.slider(f"{col_sel} range", mn, mx, (mn, mx))
                filtered = df[df[col_sel].between(*rng)]
            else:
                vals = df[col_sel].dropna().unique().tolist()
                chosen = st.multiselect(f"Filter {col_sel}", vals, default=vals[:3])
                filtered = df[df[col_sel].isin(chosen)]
            st.caption(f"{len(filtered):,} rows match")
        with col2:
            st.dataframe(filtered.head(50), use_container_width=True)

    # ── Missing values chart ──────────────────────────────────────────────────
    with st.expander("📉 Missing Values Chart", expanded=True):
        st.plotly_chart(fig_missing_values(df), use_container_width=True)

    # ── Descriptive statistics ────────────────────────────────────────────────
    with st.expander("📐 Descriptive Statistics"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    premium_divider()
    st.download_button(
        "⬇️ Download Processed Dataset (CSV)",
        data=to_csv_bytes(df),
        file_name="insightml_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )