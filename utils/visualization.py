"""
InsightML Studio — Visualization Library
==========================================
All Plotly figure factories used across the Streamlit pages.
Every function returns a `go.Figure` ready to pass to `st.plotly_chart`.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PLOTLY_TEMPLATE, PLOTLY_COLORS, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_MUTED

# ── Base layout helper ──────────────────────────────────────────────────────
def _base_layout(**kwargs) -> dict:
    return dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        **kwargs,
    )


# ── EDA Charts ───────────────────────────────────────────────────────────────

def fig_missing_values(df: pd.DataFrame) -> go.Figure:
    miss = df.isnull().mean().mul(100).sort_values(ascending=True)
    miss = miss[miss > 0]
    if miss.empty:
        fig = go.Figure()
        fig.add_annotation(text="✅ No missing values found!", showarrow=False, font=dict(size=18))
        fig.update_layout(**_base_layout(height=200))
        return fig

    fig = go.Figure(go.Bar(
        x=miss.values,
        y=miss.index,
        orientation="h",
        marker=dict(
            color=miss.values,
            colorscale=[[0, COLOR_SECONDARY], [1, "#FF4B4B"]],
            showscale=True,
        ),
        text=[f"{v:.1f}%" for v in miss.values],
        textposition="outside",
    ))
    fig.update_layout(
        **_base_layout(height=max(300, len(miss) * 35)),
        xaxis_title="Missing %",
        yaxis_title="",
        title="Missing Values by Column",
    )
    return fig


def fig_target_distribution(y: pd.Series, problem_type: str) -> go.Figure:
    vc = y.value_counts().reset_index()
    vc.columns = ["class", "count"]
    fig = px.bar(
        vc, x="class", y="count",
        color="count", color_continuous_scale=["#6C63FF", "#00D4FF"],
        labels={"class": "Class", "count": "Count"},
        title="Target Distribution",
    )
    fig.update_layout(**_base_layout())
    return fig


def fig_correlation_heatmap(df: pd.DataFrame, max_cols: int = 25) -> go.Figure:
    num = df.select_dtypes(include=np.number).iloc[:, :max_cols]
    corr = num.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig.update_layout(
        **_base_layout(height=max(400, len(corr) * 28)),
        title="Correlation Heatmap",
        xaxis=dict(tickangle=-45),
    )
    return fig


def fig_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(
        df, x=col, nbins=40,
        color_discrete_sequence=[COLOR_PRIMARY],
        title=f"Distribution — {col}",
        marginal="box",
    )
    fig.update_layout(**_base_layout())
    return fig


def fig_boxplot(df: pd.DataFrame, col: str, group_col: Optional[str] = None) -> go.Figure:
    if group_col:
        fig = px.box(df, x=group_col, y=col, color=group_col,
                     color_discrete_sequence=PLOTLY_COLORS, title=f"{col} by {group_col}")
    else:
        fig = px.box(df, y=col, color_discrete_sequence=[COLOR_PRIMARY], title=f"Boxplot — {col}")
    fig.update_layout(**_base_layout())
    return fig


def fig_violin(df: pd.DataFrame, col: str, target_col: Optional[str] = None) -> go.Figure:
    if target_col and target_col in df.columns:
        fig = px.violin(df, x=target_col, y=col, box=True, color=target_col,
                        color_discrete_sequence=PLOTLY_COLORS, title=f"{col} vs {target_col}")
    else:
        fig = px.violin(df, y=col, box=True, color_discrete_sequence=[COLOR_PRIMARY], title=f"Violin — {col}")
    fig.update_layout(**_base_layout())
    return fig


def fig_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None) -> go.Figure:
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x=x_col, y=y_col, color=color_col,
        color_discrete_sequence=PLOTLY_COLORS,
        opacity=0.6, title=f"{x_col} vs {y_col}",
    )
    fig.update_layout(**_base_layout())
    return fig


# ── Model Evaluation Charts ─────────────────────────────────────────────────

def fig_confusion_matrix(cm: list[list[int]], labels: Optional[list] = None) -> go.Figure:
    cm_arr = np.array(cm)
    n = cm_arr.shape[0]
    labels = labels or [str(i) for i in range(n)]

    cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True)
    text = [[f"{cm_arr[i, j]}<br>({cm_norm[i, j]:.1%})" for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=[f"Pred: {l}" for l in labels],
        y=[f"True: {l}" for l in labels],
        colorscale=[[0, "#1A1D2E"], [0.5, "#6C63FF"], [1, "#00D4FF"]],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=14),
        showscale=False,
    ))
    fig.update_layout(**_base_layout(height=350), title="Confusion Matrix")
    return fig


def fig_roc_curve(fpr: list, tpr: list, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC AUC = {auc:.4f}",
        line=dict(color=COLOR_PRIMARY, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color=COLOR_MUTED, dash="dash"),
    ))
    fig.update_layout(
        **_base_layout(height=380),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="ROC Curve",
    )
    return fig


def fig_pr_curve(precision: list, recall: list) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=recall, y=precision, mode="lines",
        line=dict(color=COLOR_SECONDARY, width=2.5),
        fill="tozeroy", fillcolor=f"rgba(0,212,255,0.15)",
    ))
    fig.update_layout(
        **_base_layout(height=380),
        xaxis_title="Recall",
        yaxis_title="Precision",
        title="Precision-Recall Curve",
    )
    return fig


def fig_learning_curve(lc: dict) -> go.Figure:
    sizes = lc["train_sizes"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=lc["train_mean"], mode="lines+markers",
        name="Train", line=dict(color=COLOR_PRIMARY, width=2),
        error_y=dict(type="data", array=lc["train_std"], visible=True),
    ))
    fig.add_trace(go.Scatter(
        x=sizes, y=lc["val_mean"], mode="lines+markers",
        name="Validation", line=dict(color=COLOR_SECONDARY, width=2),
        error_y=dict(type="data", array=lc["val_std"], visible=True),
    ))
    fig.update_layout(
        **_base_layout(height=380),
        xaxis_title="Training Samples",
        yaxis_title=lc.get("scoring", "Score"),
        title="Learning Curve",
    )
    return fig


def fig_feature_importance(fi_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    df = fi_df.head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["importance"],
            colorscale=[[0, "#6C63FF"], [1, "#00D4FF"]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        **_base_layout(height=max(350, top_n * 22)),
        xaxis_title="Importance",
        title=f"Top {top_n} Feature Importances",
    )
    return fig


def fig_shap_bar(shap_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    df = shap_df.head(top_n).sort_values("mean_abs_shap")
    fig = go.Figure(go.Bar(
        x=df["mean_abs_shap"], y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["mean_abs_shap"],
            colorscale=[[0, "#B48EFF"], [1, "#FF6B9D"]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        **_base_layout(height=max(350, top_n * 22)),
        xaxis_title="Mean |SHAP value|",
        title=f"SHAP Feature Importance (top {top_n})",
    )
    return fig


def fig_shap_waterfall(local_df: pd.DataFrame, prediction: float, top_n: int = 12) -> go.Figure:
    df = local_df.head(top_n).sort_values("shap_value")
    colors = [COLOR_SECONDARY if v >= 0 else "#FF4B4B" for v in df["shap_value"]]
    labels = []
    for _, r in df.iterrows():
        value = r["feature_value"]
        if isinstance(value, (int, float)) and not np.isnan(value):
            value_label = f"{value:.3g}"
        else:
            value_label = str(value)
        labels.append(f"{r['feature']}={value_label}")

    fig = go.Figure(go.Bar(
        x=df["shap_value"],
        y=labels,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        **_base_layout(height=max(350, top_n * 28)),
        xaxis_title="SHAP value",
        title=f"Local Explanation — Prediction: {prediction:.4f}",
    )
    return fig


def fig_leaderboard_radar(leaderboard: pd.DataFrame) -> go.Figure:
    top = leaderboard.dropna(subset=["CV Score"]).head(6)
    fig = go.Figure()
    categories = ["CV Score", "Stability"]

    for _, row in top.iterrows():
        stability = 1 - (row["Std"] if not pd.isna(row["Std"]) else 0)
        values = [row["CV Score"], stability]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            name=row["Model"],
            fill="toself",
            opacity=0.7,
        ))
    fig.update_layout(
        **_base_layout(height=420),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Comparison Radar",
    )
    return fig


def fig_residual_plot(y_test: list, y_pred: list) -> go.Figure:
    residuals = np.array(y_test) - np.array(y_pred)
    fig = px.scatter(
        x=y_pred, y=residuals,
        labels={"x": "Predicted", "y": "Residual"},
        title="Residual Plot",
        color_discrete_sequence=[COLOR_PRIMARY],
        opacity=0.5,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_MUTED)
    fig.update_layout(**_base_layout(height=380))
    return fig