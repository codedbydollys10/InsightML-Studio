"""
InsightML Studio — Downloads Page
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from utils.report_generator import to_csv_bytes, to_excel_bytes, generate_model_report_pdf, leaderboard_excel
from utils.ui_theme import inject_css, page_header, premium_divider
from config import PROCESSED_DATA_PATH, REPORTS_DIR


def render(df: pd.DataFrame) -> None:
    inject_css()
    page_header(
        title="Downloads",
        icon="📥",
        subtitle="Download datasets, model artefacts, evaluation reports, and generated PDFs.",
        badge="📦 Export Center",
    )

    tabs = st.tabs(["📊 Datasets", "🤖 Model Reports", "📋 Evaluation Reports"])

    with tabs[0]:
        st.markdown("### Dataset Downloads")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """<div class="download-card">
                <span class="download-card-icon">📄</span>
                <div class="download-card-label">Raw Dataset</div>
                <div class="download-card-sub">Original uploaded data as CSV</div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ Download Raw CSV",
                data=to_csv_bytes(df),
                file_name="bank_customer_churn_raw.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with c2:
            st.markdown(
                """<div class="download-card">
                <span class="download-card-icon">⚙️</span>
                <div class="download-card-label">Processed Dataset</div>
                <div class="download-card-sub">Feature-engineered and preprocessed data</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if PROCESSED_DATA_PATH.exists():
                proc_df = pd.read_csv(PROCESSED_DATA_PATH)
                st.download_button(
                    "⬇️ Download Processed CSV",
                    data=to_csv_bytes(proc_df),
                    file_name="processed_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("Process and train a model first.")

    with tabs[1]:
        st.markdown("### Model Reports")
        leaderboard = st.session_state.get("leaderboard")

        if leaderboard is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    """<div class="download-card">
                    <span class="download-card-icon">📊</span>
                    <div class="download-card-label">Leaderboard (CSV)</div>
                    <div class="download-card-sub">Model rankings and CV scores</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                lb_clean = leaderboard.drop(columns=["_model_obj"], errors="ignore")
                st.download_button(
                    "⬇️ Leaderboard CSV",
                    data=to_csv_bytes(lb_clean),
                    file_name="model_leaderboard.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c2:
                st.markdown(
                    """<div class="download-card">
                    <span class="download-card-icon">📗</span>
                    <div class="download-card-label">Leaderboard (Excel)</div>
                    <div class="download-card-sub">Formatted Excel workbook</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "⬇️ Leaderboard Excel",
                    data=leaderboard_excel(leaderboard),
                    file_name="model_leaderboard.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            premium_divider()
            st.markdown(
                """<div class="download-card">
                <span class="download-card-icon">📑</span>
                <div class="download-card-label">Full PDF Report</div>
                <div class="download-card-sub">Complete model evaluation report with charts and metrics</div>
                </div>""",
                unsafe_allow_html=True,
            )
            from utils.helpers import load_json
            from config import METADATA_PATH
            meta = load_json(METADATA_PATH) if METADATA_PATH.exists() else {}
            eval_results = st.session_state.get("eval_results", {})
            pdf_bytes = generate_model_report_pdf(meta, leaderboard, eval_results)
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="insightml_model_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.markdown(
                """<div style='background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);
                border-radius:12px;padding:1.25rem 1.5rem;'>
                ⚠️ <strong>Train models first</strong> to generate reports.
                </div>""",
                unsafe_allow_html=True,
            )

    with tabs[2]:
        st.markdown("### Evaluation Results")
        eval_results = st.session_state.get("eval_results")
        if eval_results:
            skip = {"confusion_matrix", "classification_report", "roc_curve",
                    "pr_curve", "y_pred", "y_test", "residuals", "y_pred_proba"}
            metrics_df = pd.DataFrame([
                {"Metric": k, "Value": v}
                for k, v in eval_results.items()
                if k not in skip and v is not None
            ])
            st.dataframe(metrics_df, use_container_width=True)
            st.download_button(
                "⬇️ Download Metrics CSV",
                data=to_csv_bytes(metrics_df),
                file_name="evaluation_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if eval_results.get("y_test") and eval_results.get("y_pred"):
                pred_df = pd.DataFrame({
                    "y_true": eval_results["y_test"],
                    "y_pred": eval_results["y_pred"],
                })
                if eval_results.get("y_pred_proba"):
                    pred_df["y_prob"] = eval_results["y_pred_proba"]
                st.download_button(
                    "⬇️ Download Test Predictions CSV",
                    data=to_csv_bytes(pred_df),
                    file_name="test_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.markdown(
                """<div style='background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);
                border-radius:12px;padding:1.25rem 1.5rem;'>
                ⚠️ <strong>Train models first</strong> to download evaluation results.
                </div>""",
                unsafe_allow_html=True,
            )