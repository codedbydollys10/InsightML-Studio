"""
InsightML Studio — Model Training Page
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils.preprocessing import prepare_data, apply_smote
from utils.feature_engineering import engineer_features, select_features_variance
from utils.model_training import get_model_names, train_all_models, save_best_model
from utils.evaluation import evaluate_classifier, evaluate_regressor, compute_learning_curve
from utils.ui_theme import inject_css, page_header, kpi_row, premium_divider
from config import TEST_SIZE, RANDOM_STATE
import plotly.express as px


def render(profile, df: pd.DataFrame) -> None:
    inject_css()
    page_header(
        title="Model Training",
        icon="🚀",
        subtitle="Configure and launch AutoML training across multiple algorithms with hyperparameter optimization.",
        badge="🤖 AutoML Engine",
    )

    problem_type = profile.problem_type
    model_options = get_model_names(problem_type)

    # ── Config panel ──────────────────────────────────────────────────────────
    with st.expander("⚙️ Training Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        target_col = c1.selectbox(
            "Target Column",
            profile.target_candidates,
            index=0 if profile.target_candidates else 0,
        )
        test_size = c2.slider("Test Set Size", 0.10, 0.40, TEST_SIZE, 0.05)
        scaler = c3.selectbox("Scaler", ["standard", "minmax"])

        c4, c5 = st.columns(2)
        use_smote = c4.checkbox(
            "Apply SMOTE (class imbalance)",
            value=("classification" in problem_type),
        )
        use_feat_eng = c4.checkbox("Feature Engineering", value=True)

        selected_models = st.multiselect(
            "Select models to train",
            options=model_options,
            default=model_options,
        )

        if not selected_models:
            st.warning("Select at least one model to train.")

    # ── Training button ───────────────────────────────────────────────────────
    if st.button("🧠 Start Training", type="primary", use_container_width=True):
        if not selected_models:
            st.warning("Please choose at least one model before starting training.")
        else:
            with st.spinner("Preparing data…"):
                df_work = df.drop_duplicates().copy()

                if use_feat_eng:
                    df_work = engineer_features(df_work, target_col, problem_type)

                X, y, preprocessor, feature_names, label_enc = prepare_data(
                    df=df_work,
                    target_col=target_col,
                    id_cols=profile.id_cols,
                    datetime_cols=profile.datetime_cols,
                    numeric_cols=profile.numeric_cols + [
                        c for c in df_work.columns
                        if c not in df.columns and c != target_col
                    ],
                    categorical_cols=profile.categorical_cols,
                    high_cardinality_cols=profile.high_cardinality_cols,
                    drop_missing_threshold=0.60,
                    scaler_type=scaler,
                )
                X, feature_names = select_features_variance(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=RANDOM_STATE,
                    stratify=y if "classification" in problem_type else None,
                )

                if use_smote and "classification" in problem_type:
                    X_train, y_train = apply_smote(X_train, y_train, RANDOM_STATE)

                st.success(f"✅ Data ready — {X_train.shape[0]:,} train | {X_test.shape[0]:,} test | {X.shape[1]} features")

                # ── Live training progress ────────────────────────────────────
                st.markdown("### Training Progress")
                progress_bar = st.progress(0.0)
                status_text  = st.empty()
                log_area     = st.empty()
                log_lines: list[str] = []

                def progress_cb(name: str, frac: float) -> None:
                    progress_bar.progress(frac)
                    status_text.markdown(f"**Training:** `{name}` ({frac*100:.0f}%)")
                    log_lines.append(f"✓ {name}")
                    log_area.code("\n".join(log_lines[-12:]))

                leaderboard = train_all_models(
                    X_train,
                    y_train,
                    problem_type,
                    selected_models=selected_models,
                    progress_callback=progress_cb,
                )

                progress_bar.progress(1.0)
                status_text.markdown("**Training complete!** 🎉")

                # ── Save best model ───────────────────────────────────────────
                save_best_model(
                    leaderboard=leaderboard,
                    preprocessor=preprocessor,
                    feature_names=feature_names,
                    label_enc=label_enc,
                    target_col=target_col,
                    problem_type=problem_type,
                    metadata_extra={
                        "n_train": len(X_train),
                        "n_test":  len(X_test),
                        "n_features": len(feature_names),
                        "smote_applied": use_smote,
                        "test_size": test_size,
                    },
                )

                from config import PROCESSED_DATA_PATH
                pd.concat([X, y.reset_index(drop=True)], axis=1).to_csv(PROCESSED_DATA_PATH, index=False)

                # ── Evaluate best model on test set ───────────────────────────
                best_model = leaderboard.dropna(subset=["CV Score"]).iloc[0]["_model_obj"]
                if "classification" in problem_type:
                    eval_results = evaluate_classifier(best_model, X_test, y_test)
                else:
                    eval_results = evaluate_regressor(best_model, X_test, y_test)

                # Store in session state for other pages
                st.session_state["leaderboard"]    = leaderboard
                st.session_state["eval_results"]   = eval_results
                st.session_state["X_test"]         = X_test
                st.session_state["y_test"]         = y_test
                st.session_state["X_train"]        = X_train
                st.session_state["y_train"]        = y_train
                st.session_state["best_model"]     = best_model
                st.session_state["feature_names"]  = feature_names
                st.session_state["problem_type"]   = problem_type
                st.session_state["target_col"]     = target_col
                st.session_state["preprocessor"]   = preprocessor
                st.session_state["label_enc"]      = label_enc
                st.session_state["trained"]        = True

                # ── Leaderboard preview ───────────────────────────────────────
                premium_divider()
                st.markdown("### 🏆 Model Leaderboard")
                _render_leaderboard(leaderboard)

                # ── Best model summary ────────────────────────────────────────
                best_row = leaderboard.dropna(subset=["CV Score"]).iloc[0]
                _render_eval_summary(eval_results, best_row, problem_type)

    elif st.session_state.get("trained"):
        st.markdown(
            """<div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.35);
            border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;'>
            ✅ <strong>Model already trained.</strong>
            Navigate to <strong>Model Comparison</strong> or <strong>Explainable AI</strong> pages.
            </div>""",
            unsafe_allow_html=True,
        )
        _render_leaderboard(st.session_state["leaderboard"])


def _render_leaderboard(leaderboard: pd.DataFrame) -> None:
    display = leaderboard.drop(columns=["_model_obj"], errors="ignore").copy()
    display.insert(0, "Rank", range(1, len(display) + 1))

    def highlight_best(row):
        return ["background-color: rgba(124,58,237,0.25); font-weight: bold"
                if row["Rank"] == 1 else "" for _ in row]

    st.dataframe(
        display.style.apply(highlight_best, axis=1).format({"CV Score": "{:.4f}", "Std": "{:.4f}"}),
        use_container_width=True,
    )


def _render_eval_summary(eval_results: dict, best_row: pd.Series, problem_type: str) -> None:
    premium_divider()
    st.markdown(
        f"""<div style='display:flex;align-items:center;gap:12px;margin-bottom:1.25rem;'>
        <span style='font-size:1.5rem;'>🥇</span>
        <div>
          <div style='font-size:0.72rem;font-weight:700;color:var(--subtext);text-transform:uppercase;letter-spacing:0.08em;'>Best Model</div>
          <div style='font-size:1.5rem;font-weight:800;color:var(--text);'>{best_row['Model']}</div>
        </div>
        <span style='margin-left:auto;background:linear-gradient(135deg,#FFD700,#FFA500);
          color:#1A1A1A;border-radius:8px;padding:4px 16px;font-weight:800;font-size:0.8rem;'>
          🏆 BEST MODEL
        </span></div>""",
        unsafe_allow_html=True,
    )

    if "classification" in problem_type:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{eval_results.get('accuracy', 0):.4f}")
        c2.metric("Precision", f"{eval_results.get('precision', 0):.4f}")
        c3.metric("Recall",    f"{eval_results.get('recall', 0):.4f}")
        c4.metric("F1 Score",  f"{eval_results.get('f1', 0):.4f}")
        if eval_results.get("roc_auc"):
            st.metric("ROC AUC", f"{eval_results['roc_auc']:.4f}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE",  f"{eval_results.get('mae', 0):.4f}")
        c2.metric("RMSE", f"{eval_results.get('rmse', 0):.4f}")
        c3.metric("R²",   f"{eval_results.get('r2', 0):.4f}")
        if eval_results.get("mape") is not None:
            c4.metric("MAPE", f"{eval_results['mape']:.4f}")