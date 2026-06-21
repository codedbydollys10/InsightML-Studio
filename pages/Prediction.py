"""
InsightML Studio — Interactive Prediction Page
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np

from utils.prediction import load_artefacts, predict_single
from utils.ui_theme import inject_css, page_header, premium_divider, gauge_bar, result_card
from config import BEST_MODEL_PATH


def render(df: pd.DataFrame, profile) -> None:
    inject_css()
    page_header(
        title="Interactive Prediction",
        icon="🎯",
        subtitle="Enter feature values and get an instant real-time prediction from your trained model.",
        badge="⚡ Live Inference",
    )

    # Load artefacts
    artefacts = load_artefacts()
    if not artefacts:
        st.markdown(
            """<div style='background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);
            border-radius:12px;padding:1.25rem 1.5rem;'>
            ⚠️ <strong>No trained model found.</strong> Please train a model first.
            </div>""",
            unsafe_allow_html=True,
        )
        return

    meta = artefacts.get("metadata", {})
    st.markdown(
        f"""<div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.35);
        border-radius:12px;padding:1rem 1.5rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:12px;'>
        <span style='font-size:1.5rem;'>✅</span>
        <div>
          <div style='font-size:0.72rem;font-weight:700;color:#6EE7B7;text-transform:uppercase;letter-spacing:0.08em;'>Model Loaded</div>
          <div style='color:var(--text);font-weight:600;'>{meta.get("best_model", "Unknown")}</div>
          <div style='color:var(--subtext);font-size:0.82rem;'>CV Score: {meta.get("cv_score", "N/A")}</div>
        </div></div>""",
        unsafe_allow_html=True,
    )

    target_col = meta.get("target_col", "churn")
    problem_type = meta.get("problem_type", "binary_classification")

    # ── Dynamic form ──────────────────────────────────────────────────────────
    st.markdown("### Input Features")
    st.caption("Fill in feature values below to generate a prediction.")

    exclude = set(profile.id_cols) | set(profile.datetime_cols) | {target_col}
    input_cols = [c for c in df.columns if c not in exclude]

    row_input: dict = {}

    cols_per_row = 3
    col_chunks = [input_cols[i:i+cols_per_row] for i in range(0, len(input_cols), cols_per_row)]

    for chunk in col_chunks:
        cols = st.columns(cols_per_row)
        for j, col_name in enumerate(chunk):
            with cols[j]:
                col_data = df[col_name].dropna()
                if col_data.empty:
                    row_input[col_name] = None
                    continue

                if df[col_name].dtype in (np.float64, np.int64, np.float32, np.int32):
                    mn = float(col_data.min())
                    mx = float(col_data.max())
                    med = float(col_data.median())
                    row_input[col_name] = st.number_input(
                        col_name, min_value=mn, max_value=mx, value=med,
                        key=f"pred_{col_name}",
                    )
                else:
                    unique_vals = col_data.unique().tolist()[:30]
                    row_input[col_name] = st.selectbox(
                        col_name, unique_vals, key=f"pred_{col_name}"
                    )

    premium_divider()

    # ── Predict button ─────────────────────────────────────────────────────────
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        try:
            result = predict_single(row_input, artefacts)

            pred = result["prediction"]
            prob = result.get("probability")
            risk = result.get("risk_level", "Unknown")
            rec  = result.get("recommendation", "")

            premium_divider()
            st.markdown("### 🎯 Prediction Result")

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", str(pred))
            if prob is not None:
                c2.metric("Probability", f"{prob:.4f}")
                c3.metric("Risk Level", risk)

            result_card(
                title="💡 Business Recommendation",
                body=rec,
            )

            if prob is not None:
                premium_divider()
                color_cls = "gauge-red" if prob > 0.75 else "gauge-orange" if prob > 0.45 else "gauge-green"
                gauge_bar(prob, label="Churn Probability")

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    # ── Sample predictions from test set ──────────────────────────────────────
    if st.session_state.get("trained"):
        premium_divider()
        st.markdown("### 🎲 Sample Test-Set Predictions")
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        model  = st.session_state["best_model"]

        n_show = st.slider("Samples", 5, 50, 10)
        sample_X = X_test.head(n_show)
        sample_y = y_test.head(n_show).values

        preds = model.predict(sample_X)
        sample_df = pd.DataFrame({
            "Index": range(n_show),
            "Actual": sample_y,
            "Predicted": preds,
            "Correct": sample_y == preds,
        })
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(sample_X)
            sample_df["Probability"] = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)

        def color_correct(val):
            return "color: #10B981" if val else "color: #EF4444"

        st.dataframe(
            sample_df.style.applymap(color_correct, subset=["Correct"]).format({"Probability": "{:.4f}"}),
            use_container_width=True,
        )