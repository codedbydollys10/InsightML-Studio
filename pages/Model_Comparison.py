"""
InsightML Studio — Model Comparison Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.visualization import (
    fig_leaderboard_radar, fig_confusion_matrix,
    fig_roc_curve, fig_pr_curve, fig_learning_curve,
)
from utils.evaluation import compute_learning_curve
from utils.ui_theme import inject_css, page_header, kpi_row, premium_divider
from config import PLOTLY_TEMPLATE


def render() -> None:
    inject_css()
    page_header(
        title="Model Comparison",
        icon="📡",
        subtitle="Compare all trained models side-by-side using leaderboards, radar charts, and evaluation curves.",
        badge="⚖️ Performance Analysis",
    )

    if not st.session_state.get("trained"):
        st.markdown(
            """<div style='background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);
            border-radius:12px;padding:1.25rem 1.5rem;'>
            ⚠️ <strong>Please train models first</strong> on the <strong>Model Training</strong> page.
            </div>""",
            unsafe_allow_html=True,
        )
        return

    leaderboard  = st.session_state["leaderboard"]
    eval_results = st.session_state["eval_results"]
    problem_type = st.session_state["problem_type"]
    best_model   = st.session_state["best_model"]
    X_test       = st.session_state["X_test"]
    y_test       = st.session_state["y_test"]
    X_train      = st.session_state["X_train"]
    y_train      = st.session_state["y_train"]

    display = leaderboard.drop(columns=["_model_obj"], errors="ignore").copy()

    tabs = st.tabs(["🏆 Leaderboard", "📡 Radar Chart", "📈 ROC / PR", "📉 Learning Curve", "🔍 Detailed Metrics"])

    with tabs[0]:
        st.markdown("### Model Leaderboard")
        display_sorted = display.dropna(subset=["CV Score"]).sort_values("CV Score", ascending=False).reset_index(drop=True)
        display_sorted.insert(0, "Rank", range(1, len(display_sorted) + 1))

        def highlight_best(row):
            return ["background-color: rgba(124,58,237,0.3); font-weight: bold"
                    if row["Rank"] == 1 else "" for _ in row]

        st.dataframe(
            display_sorted.style.apply(highlight_best, axis=1)
                .format({"CV Score": "{:.4f}", "Std": "{:.4f}"}),
            use_container_width=True,
        )

        fig = px.bar(
            display_sorted.sort_values("CV Score"),
            x="CV Score", y="Model", orientation="h",
            color="CV Score", color_continuous_scale="purples",
            error_x="Std", title="CV Score by Model", template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            display_sorted, x="Train Time", y="CV Score",
            text="Model", color="CV Score", color_continuous_scale="teal",
            title="CV Score vs Training Time", template=PLOTLY_TEMPLATE,
        )
        fig2.update_traces(textposition="top center", marker_size=10)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[1]:
        st.markdown("### Performance Radar — Top Models")
        st.plotly_chart(fig_leaderboard_radar(leaderboard), use_container_width=True)

    with tabs[2]:
        if "classification" in problem_type:
            c1, c2 = st.columns(2)
            if eval_results.get("roc_curve"):
                with c1:
                    roc = eval_results["roc_curve"]
                    auc = eval_results.get("roc_auc", 0.0)
                    st.plotly_chart(fig_roc_curve(roc["fpr"], roc["tpr"], auc), use_container_width=True)
            if eval_results.get("pr_curve"):
                with c2:
                    prc = eval_results["pr_curve"]
                    st.plotly_chart(fig_pr_curve(prc["precision"], prc["recall"]), use_container_width=True)

            if eval_results.get("confusion_matrix"):
                st.markdown("### Confusion Matrix")
                st.plotly_chart(fig_confusion_matrix(eval_results["confusion_matrix"]), use_container_width=True)
        else:
            from utils.visualization import fig_residual_plot
            st.plotly_chart(fig_residual_plot(eval_results["y_test"], eval_results["y_pred"]), use_container_width=True)

    with tabs[3]:
        st.markdown("### Learning Curve — Best Model")
        if st.button("Compute Learning Curve"):
            with st.spinner("Computing…"):
                X_full = pd.concat([X_train, X_test]).reset_index(drop=True)
                y_full = pd.concat([y_train, y_test]).reset_index(drop=True)
                lc = compute_learning_curve(best_model, X_full, y_full, problem_type)
                st.plotly_chart(fig_learning_curve(lc), use_container_width=True)

    with tabs[4]:
        st.markdown("### Full Evaluation Report")
        skip = {"confusion_matrix", "classification_report", "roc_curve",
                "pr_curve", "y_pred", "y_test", "residuals", "y_pred_proba"}
        metrics_df = pd.DataFrame([
            {"Metric": k, "Value": round(v, 4) if isinstance(v, float) else v}
            for k, v in eval_results.items()
            if k not in skip and v is not None
        ])
        st.dataframe(metrics_df, use_container_width=True)

        if eval_results.get("classification_report"):
            st.markdown("#### Classification Report")
            cr_df = pd.DataFrame(eval_results["classification_report"]).T
            st.dataframe(cr_df.style.background_gradient(cmap="Greens"), use_container_width=True)