"""
InsightML Studio — Explainable AI Page
"""

import streamlit as st
import pandas as pd

from utils.explainability import compute_shap_values, shap_summary_df, local_shap_explanation
from utils.evaluation import get_feature_importance
from utils.visualization import fig_feature_importance, fig_shap_bar, fig_shap_waterfall
from utils.ui_theme import inject_css, page_header, premium_divider


def render() -> None:
    inject_css()
    page_header(
        title="Explainable AI",
        icon="🔮",
        subtitle="Understand model decisions using SHAP values and feature importance analysis.",
        badge="🧩 XAI Dashboard",
    )

    if not st.session_state.get("trained"):
        st.markdown(
            """<div style='background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);
            border-radius:12px;padding:1.25rem 1.5rem;'>
            ⚠️ <strong>Please train models first.</strong>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    best_model    = st.session_state["best_model"]
    feature_names = st.session_state["feature_names"]
    X_test        = st.session_state["X_test"]

    tabs = st.tabs(["📊 Feature Importance", "🌊 SHAP Global", "🔍 SHAP Local"])

    # ── Feature Importance ────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Model Feature Importance")
        fi_df = get_feature_importance(best_model, feature_names)
        if fi_df is not None:
            top_n = st.slider("Top N features", 5, min(50, len(fi_df)), 20)
            st.plotly_chart(fig_feature_importance(fi_df, top_n), use_container_width=True)
            st.dataframe(fi_df.head(top_n), use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    # ── SHAP Global ───────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### SHAP Global Feature Importance")
        st.caption("SHAP values explain how much each feature contributes to the prediction.")

        n_samples = st.slider("Samples to explain", 50, min(500, len(X_test)), 150, step=50)

        if st.button("🧮 Compute SHAP Values", type="primary"):
            with st.spinner("Computing SHAP values… (this may take a minute)"):
                shap_result = compute_shap_values(
                    best_model, X_test.sample(min(n_samples, len(X_test)), random_state=42),
                    n_background=50, n_explain=n_samples,
                )

            if shap_result is None:
                st.error("SHAP computation failed for this model type.")
            else:
                shap_values, explainer, X_sample = shap_result
                summary = shap_summary_df(shap_values, X_sample)
                st.session_state["shap_values"] = shap_values
                st.session_state["shap_X_sample"] = X_sample

                top_shap = st.slider("Top features", 5, min(30, len(summary)), 15)
                st.plotly_chart(fig_shap_bar(summary, top_shap), use_container_width=True)
                st.dataframe(summary.head(top_shap), use_container_width=True)

        elif st.session_state.get("shap_values") is not None:
            shap_values = st.session_state["shap_values"]
            X_sample    = st.session_state["shap_X_sample"]
            summary     = shap_summary_df(shap_values, X_sample)
            top_shap    = st.slider("Top features", 5, min(30, len(summary)), 15)
            st.plotly_chart(fig_shap_bar(summary, top_shap), use_container_width=True)

    # ── SHAP Local ────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### Local Prediction Explanation")
        st.caption("Understand why the model made a specific prediction for one sample.")

        if st.session_state.get("shap_values") is None:
            st.markdown(
                """<div style='background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.35);
                border-radius:12px;padding:1rem 1.5rem;'>
                ℹ️ Compute SHAP values first in the <strong>SHAP Global</strong> tab.
                </div>""",
                unsafe_allow_html=True,
            )
            return

        shap_values = st.session_state["shap_values"]
        X_sample    = st.session_state["shap_X_sample"]

        row_idx = st.slider("Select sample index", 0, len(X_sample) - 1, 0)

        row_df = X_sample.iloc[[row_idx]]
        if hasattr(best_model, "predict_proba"):
            prob = best_model.predict_proba(row_df)[0]
            pred_val = prob[1] if len(prob) == 2 else prob.max()
            pred_label = int(best_model.predict(row_df)[0])
            st.metric("Prediction", f"Class {pred_label}", f"Probability: {pred_val:.4f}")
        else:
            pred_val = float(best_model.predict(row_df)[0])
            st.metric("Prediction", f"{pred_val:.4f}")

        local_df = local_shap_explanation(shap_values, X_sample, row_idx)
        top_local = st.slider("Top contributing features", 5, min(20, len(local_df)), 12)
        st.plotly_chart(fig_shap_waterfall(local_df, pred_val, top_local), use_container_width=True)
        st.dataframe(local_df.head(top_local), use_container_width=True)