"""
InsightML Studio — Batch Prediction Page
"""

import streamlit as st
import pandas as pd

from utils.prediction import load_artefacts, predict_batch
from utils.report_generator import to_csv_bytes, to_excel_bytes
from utils.ui_theme import inject_css, page_header, kpi_row, premium_divider


def render() -> None:
    inject_css()
    page_header(
        title="Batch Prediction",
        icon="📦",
        subtitle="Upload a CSV file and run predictions on thousands of records at once.",
        badge="⚡ Bulk Inference",
    )

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
        f"""<div style='background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.3);
        border-radius:12px;padding:1rem 1.5rem;margin-bottom:1.5rem;display:flex;gap:16px;flex-wrap:wrap;'>
        <div><div style='font-size:0.7rem;font-weight:700;color:#67E8F9;text-transform:uppercase;letter-spacing:0.08em;'>Model</div>
             <div style='color:var(--text);font-weight:600;'>{meta.get("best_model", "Unknown")}</div></div>
        <div><div style='font-size:0.7rem;font-weight:700;color:#67E8F9;text-transform:uppercase;letter-spacing:0.08em;'>Features</div>
             <div style='color:var(--text);font-weight:600;'>{meta.get("feature_count", "N/A")}</div></div>
        <div><div style='font-size:0.7rem;font-weight:700;color:#67E8F9;text-transform:uppercase;letter-spacing:0.08em;'>Task</div>
             <div style='color:var(--text);font-weight:600;'>{meta.get("problem_type", "N/A")}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### Upload CSV for Batch Prediction")
    st.caption("Upload a CSV with the same column structure as the training dataset (target column is optional).")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.markdown(
                f"""<div style='background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);
                border-radius:10px;padding:0.75rem 1.25rem;margin-bottom:1rem;'>
                ✅ Loaded <strong>{len(batch_df):,}</strong> rows × <strong>{batch_df.shape[1]}</strong> columns
                </div>""",
                unsafe_allow_html=True,
            )
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner(f"Running predictions on {len(batch_df):,} rows…"):
                    result_df = predict_batch(batch_df, artefacts)

                st.success(f"✅ Predictions complete for {len(result_df):,} records!")

                # ── Summary stats ─────────────────────────────────────────────
                premium_divider()
                pred_col = "Prediction"
                vc = result_df[pred_col].value_counts()

                metrics = [{"label": "Total Records", "value": f"{len(result_df):,}", "icon": "📊", "color": "purple"}]
                if "Probability" in result_df.columns:
                    metrics.append({"label": "Avg Probability", "value": f"{result_df['Probability'].mean():.4f}", "icon": "📈", "color": "blue"})
                    high_risk = (result_df["Probability"] >= 0.75).sum()
                    metrics.append({"label": "High Risk Records", "value": f"{high_risk:,}", "icon": "⚠️", "color": "red"})
                kpi_row(metrics)

                # Distribution chart
                import plotly.express as px
                from config import PLOTLY_TEMPLATE
                fig = px.histogram(
                    result_df, x="Prediction",
                    color="Prediction", template=PLOTLY_TEMPLATE,
                    title="Prediction Distribution",
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                if "Probability" in result_df.columns:
                    fig2 = px.histogram(
                        result_df, x="Probability", nbins=40,
                        color_discrete_sequence=["#7C3AED"], template=PLOTLY_TEMPLATE,
                        title="Churn Probability Distribution",
                    )
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig2, use_container_width=True)

                # ── Preview results ───────────────────────────────────────────
                premium_divider()
                st.markdown("### Results Preview")
                st.dataframe(result_df.head(50), use_container_width=True)

                # ── Downloads ─────────────────────────────────────────────────
                premium_divider()
                st.markdown("### 📥 Download Results")
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "⬇️ Download CSV",
                        data=to_csv_bytes(result_df),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with c2:
                    st.download_button(
                        "⬇️ Download Excel",
                        data=to_excel_bytes({"Predictions": result_df}),
                        file_name="batch_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

        except Exception as exc:
            st.error(f"Error processing file: {exc}")