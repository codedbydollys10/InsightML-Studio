"""
InsightML Studio — Automated EDA Page
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.visualization import (
    fig_correlation_heatmap, fig_target_distribution, fig_histogram,
    fig_boxplot, fig_violin, fig_scatter, fig_missing_values,
)
from utils.ui_theme import inject_css, page_header, premium_divider, obs_card


def render(profile, df: pd.DataFrame, target_col: str, problem_type: str) -> None:
    inject_css()
    page_header(
        title="Automated EDA",
        icon="🔬",
        subtitle="Automatically explore distributions, correlations, and relationships in your dataset.",
        badge="📈 Exploratory Analysis",
    )

    tabs = st.tabs([
        "📈 Distributions", "🔗 Correlations", "🎯 Target Analysis",
        "📦 Outliers", "🔄 Feature Relationships"
    ])

    num_cols = [c for c in profile.numeric_cols if c in df.columns and c != target_col]
    cat_cols = [c for c in profile.categorical_cols if c in df.columns and c != target_col]

    # ── Distributions ─────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Numeric Distributions")
        if num_cols:
            col_choice = st.selectbox("Select feature", num_cols, key="dist_col")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_histogram(df, col_choice), use_container_width=True)
            with c2:
                st.plotly_chart(fig_boxplot(df, col_choice), use_container_width=True)

            st.markdown("#### All Numeric Summary")
            st.dataframe(df[num_cols].describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)
        else:
            st.info("No numeric columns detected.")

        st.markdown("### Categorical Distributions")
        if cat_cols:
            for cat in cat_cols[:6]:
                vc = df[cat].value_counts().reset_index()
                vc.columns = [cat, "count"]
                import plotly.express as px
                from config import PLOTLY_TEMPLATE
                fig = px.bar(vc.head(20), x=cat, y="count",
                             color="count", color_continuous_scale="purples",
                             title=f"Value Counts — {cat}",
                             template=PLOTLY_TEMPLATE)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

    # ── Correlations ──────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### Correlation Heatmap")
        if len(num_cols) >= 2:
            st.plotly_chart(fig_correlation_heatmap(df[num_cols + [target_col]]), use_container_width=True)

            corr_with_target = df[num_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
            st.markdown("#### Feature Correlation with Target")
            st.dataframe(
                corr_with_target.reset_index().rename(columns={"index": "Feature", 0: "Correlation"})
                .style.background_gradient(cmap="RdYlGn", subset=["Correlation"]),
                use_container_width=True,
            )
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

    # ── Target Analysis ───────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown(f"### Target: `{target_col}`")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_target_distribution(df[target_col], problem_type), use_container_width=True)
        with c2:
            vc = df[target_col].value_counts()
            total = len(df)
            st.markdown("#### Class Balance")
            for cls, cnt in vc.items():
                pct = cnt / total * 100
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>"
                    f"<span style='font-weight:600;color:var(--text);min-width:90px;'>Class {cls}</span>"
                    f"<div style='flex:1;background:var(--border);border-radius:20px;height:10px;overflow:hidden;'>"
                    f"<div style='width:{pct:.1f}%;height:100%;background:linear-gradient(90deg,#7C3AED,#A78BFA);border-radius:20px;box-shadow:0 0 6px rgba(124,58,237,0.5);'></div>"
                    f"</div><span style='color:var(--subtext);font-size:0.82rem;min-width:60px;text-align:right;'>{cnt:,} ({pct:.1f}%)</span></div>",
                    unsafe_allow_html=True,
                )

        if num_cols and "classification" in problem_type:
            st.markdown("### Feature vs Target (Violin)")
            violin_col = st.selectbox("Select feature", num_cols, key="violin_col")
            sample = df[[violin_col, target_col]].dropna().sample(min(3000, len(df)), random_state=42)
            sample[target_col] = sample[target_col].astype(str)
            st.plotly_chart(fig_violin(sample, violin_col, target_col), use_container_width=True)

    # ── Outlier Analysis ──────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### Outlier Summary (IQR Method)")
        if profile.outlier_counts:
            out_df = pd.DataFrame(
                list(profile.outlier_counts.items()),
                columns=["Column", "Outlier Count"],
            ).sort_values("Outlier Count", ascending=False)
            out_df["% of Rows"] = (out_df["Outlier Count"] / len(df) * 100).round(2)
            st.dataframe(out_df.style.background_gradient(subset=["Outlier Count"], cmap="Reds"), use_container_width=True)

            col_out = st.selectbox("Inspect column", out_df["Column"].tolist(), key="out_col")
            st.plotly_chart(fig_boxplot(df, col_out), use_container_width=True)
        else:
            st.success("✅ No significant outliers detected.")

    # ── Feature Relationships ─────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("### Scatter Plot Explorer")
        if len(num_cols) >= 2:
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X axis", num_cols, key="scat_x")
            y_col = c2.selectbox("Y axis", num_cols, index=1, key="scat_y")
            color_opt = c3.selectbox("Color by", ["None", target_col] + cat_cols[:5], key="scat_c")
            color_col = None if color_opt == "None" else color_opt
            sample_df = df.sample(min(3000, len(df)), random_state=42)
            if color_col and sample_df[color_col].dtype in (object, str):
                pass
            elif color_col:
                sample_df[color_col] = sample_df[color_col].astype(str)
            st.plotly_chart(fig_scatter(sample_df, x_col, y_col, color_col), use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns.")

    # ── Auto-observations ─────────────────────────────────────────────────────
    premium_divider()
    st.markdown("### 💡 Automated Observations")
    observations = _generate_observations(profile, df, target_col)
    for obs in observations:
        obs_card(obs)


def _generate_observations(profile, df: pd.DataFrame, target_col: str) -> list[str]:
    obs = []

    if target_col in df.columns:
        vc = df[target_col].value_counts(normalize=True)
        minority_pct = vc.min() * 100
        if minority_pct < 20:
            obs.append(
                f"⚠️ <strong>Class imbalance detected</strong>: the minority class is only {minority_pct:.1f}% of samples. "
                "SMOTE oversampling is recommended."
            )

    high_miss = profile.missing_pct[profile.missing_pct > 20]
    if not high_miss.empty:
        obs.append(f"📭 Columns with >20% missing: <strong>{', '.join(high_miss.index.tolist())}</strong>.")

    if profile.high_cardinality_cols:
        obs.append(
            f"🔢 High-cardinality columns detected (<strong>{', '.join(profile.high_cardinality_cols[:4])}</strong>). "
            "Target encoding will be applied."
        )

    if profile.duplicate_rows > 0:
        obs.append(f"🔁 <strong>{profile.duplicate_rows:,}</strong> duplicate rows found and will be removed.")

    n_out_cols = len(profile.outlier_counts)
    if n_out_cols:
        obs.append(f"📐 Outliers detected in <strong>{n_out_cols} feature(s)</strong>. IQR-based capping recommended.")

    if not obs:
        obs.append("✅ <strong>Dataset looks clean</strong> with no major issues detected.")

    return obs