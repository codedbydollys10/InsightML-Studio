"""
InsightML Studio — Business Insights Page
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.ui_theme import inject_css, page_header, premium_divider, obs_card
from config import PLOTLY_TEMPLATE, PLOTLY_COLORS


def render(df: pd.DataFrame, profile, target_col: str) -> None:
    inject_css()
    page_header(
        title="Business Insights",
        icon="💼",
        subtitle="Automatically generated data-driven insights, risk segments, and actionable recommendations.",
        badge="💡 AI Insights",
    )

    tabs = st.tabs(["📊 Segment Analysis", "🔍 Feature Deep-Dive", "⚠️ Risk Analysis", "💡 Recommendations"])

    num_cols = [c for c in profile.numeric_cols if c in df.columns and c != target_col]
    cat_cols = [c for c in profile.categorical_cols if c in df.columns and c != target_col]

    # ── Segment Analysis ──────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Customer Segment Analysis")

        for cat in cat_cols[:4]:
            seg = df.groupby(cat)[target_col].agg(["mean", "count"]).reset_index()
            seg.columns = [cat, "Churn Rate", "Count"]
            seg["Churn Rate"] = (seg["Churn Rate"] * 100).round(2)
            seg = seg.sort_values("Churn Rate", ascending=False)

            fig = px.bar(
                seg, x=cat, y="Churn Rate",
                color="Churn Rate", color_continuous_scale="RdYlGn_r",
                title=f"Churn Rate by {cat}",
                template=PLOTLY_TEMPLATE,
                text="Churn Rate",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander(f"📋 {cat} — Detail Table"):
                st.dataframe(seg, use_container_width=True)

    # ── Feature Deep-Dive ─────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### Feature vs Churn Deep-Dive")

        for col in num_cols[:8]:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(
                    df, x=col, color=df[target_col].astype(str),
                    barmode="overlay", opacity=0.7,
                    color_discrete_sequence=PLOTLY_COLORS,
                    title=f"{col} Distribution by Churn",
                    template=PLOTLY_TEMPLATE,
                    nbins=30,
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                stats = df.groupby(target_col)[col].agg(["mean", "median", "std"]).reset_index()
                st.dataframe(stats, use_container_width=True)

                if df[target_col].nunique() == 2:
                    churned = df[df[target_col] == 1][col].dropna()
                    retained = df[df[target_col] == 0][col].dropna()
                    diff_pct = (churned.mean() - retained.mean()) / retained.mean() * 100
                    direction = "higher" if diff_pct > 0 else "lower"
                    st.markdown(
                        f"""<div style='background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.3);
                        border-radius:8px;padding:0.65rem 1rem;font-size:0.88rem;color:var(--text);'>
                        Churned customers have <strong>{abs(diff_pct):.1f}% {direction}</strong> average {col}.
                        </div>""",
                        unsafe_allow_html=True,
                    )

    # ── Risk Analysis ─────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### High-Risk Segment Identification")

        if st.session_state.get("trained") and "Probability" in st.session_state.get("eval_results", {}):
            st.info("Train the model and use batch prediction to see probability-based risk segments.")

        risk_cols = []
        if "current_balance" in df.columns:
            df_r = df.copy()
            df_r["low_balance"] = (df_r["current_balance"] < df_r["current_balance"].quantile(0.25)).astype(int)
            risk_cols.append("low_balance")

        if "vintage" in df.columns:
            if "df_r" not in dir():
                df_r = df.copy()
            df_r["new_customer"] = (df_r["vintage"] < 365).astype(int)
            risk_cols.append("new_customer")

        if risk_cols and "df_r" in dir():
            for rf in risk_cols:
                seg = df_r.groupby(rf)[target_col].agg(["mean", "count"]).reset_index()
                seg.columns = [rf, "Churn Rate", "Count"]
                seg["Churn Rate %"] = (seg["Churn Rate"] * 100).round(2)
                st.markdown(f"#### Risk Factor: `{rf}`")
                st.dataframe(seg, use_container_width=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### 💡 Actionable Recommendations")
        recs = _generate_recommendations(df, profile, target_col)
        for i, (title, body) in enumerate(recs, 1):
            with st.expander(f"{i}. {title}", expanded=(i <= 3)):
                st.markdown(body)


def _generate_recommendations(
    df: pd.DataFrame, profile, target_col: str
) -> list[tuple[str, str]]:
    recs = []

    vc = df[target_col].value_counts(normalize=True)
    minority_pct = vc.min() * 100
    if minority_pct < 25:
        recs.append((
            "🎯 Address Class Imbalance",
            f"The minority class represents only {minority_pct:.1f}% of the data. "
            "SMOTE oversampling or class-weight adjustment can significantly improve model recall. "
            "**Action**: Ensure SMOTE is enabled during training.",
        ))

    cat_cols = [c for c in profile.categorical_cols if c in df.columns and c != target_col]
    for cat in cat_cols[:3]:
        seg = df.groupby(cat)[target_col].mean()
        high_churn = seg[seg > seg.mean() + seg.std()]
        if not high_churn.empty:
            segments = ", ".join([f"`{v}`" for v in high_churn.index[:3]])
            recs.append((
                f"⚠️ High-Risk Segment in `{cat}`",
                f"The following {cat} segments show above-average churn: {segments}. "
                f"Maximum churn rate: **{high_churn.max()*100:.1f}%**. "
                "**Action**: Deploy targeted retention campaigns for these segments.",
            ))

    if "current_balance" in df.columns and target_col in df.columns:
        churned_bal = df[df[target_col] == 1]["current_balance"].mean()
        retained_bal = df[df[target_col] == 0]["current_balance"].mean()
        if churned_bal < retained_bal * 0.8:
            recs.append((
                "💰 Low Balance Customers Are at Higher Risk",
                f"Churned customers have an average balance of **{churned_bal:,.0f}** "
                f"vs **{retained_bal:,.0f}** for retained customers — a gap of "
                f"{(retained_bal - churned_bal) / retained_bal * 100:.0f}%. "
                "**Action**: Proactively engage customers with declining balances.",
            ))

    if "vintage" in df.columns and target_col in df.columns:
        new_mask = df["vintage"] < 365
        new_churn = df[new_mask][target_col].mean()
        old_churn = df[~new_mask][target_col].mean()
        if new_churn > old_churn * 1.2:
            recs.append((
                "🆕 New Customers Churn Faster",
                f"Customers within their first year have a churn rate of **{new_churn*100:.1f}%** "
                f"vs **{old_churn*100:.1f}%** for tenured customers. "
                "**Action**: Implement a structured onboarding programme and 90-day check-ins.",
            ))

    total_missing = profile.missing_counts.sum()
    if total_missing > 0:
        recs.append((
            "📭 Incomplete Data Affects Model Quality",
            f"**{int(total_missing):,}** missing values detected across the dataset. "
            "Collecting complete customer profiles will improve prediction accuracy. "
            "**Action**: Review data collection processes for high-missing columns.",
        ))

    if not recs:
        recs.append(("✅ No Critical Issues Found", "The dataset is in good health with no major risk factors."))

    return recs