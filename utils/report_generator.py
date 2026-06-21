"""
InsightML Studio — Report Generator
=====================================
Generates downloadable CSV, Excel, and PDF reports.
"""

from __future__ import annotations
import io
from pathlib import Path
from datetime import datetime

import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Write multiple named sheets to an in-memory Excel workbook."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buf.getvalue()


def generate_model_report_pdf(
    metadata: dict,
    leaderboard: pd.DataFrame,
    eval_results: dict,
) -> bytes:
    """
    Generate a minimal PDF summary report using ReportLab.
    Returns raw PDF bytes.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )
        from reportlab.lib import colors

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("InsightML Studio — Model Report", styles["Title"]))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
        story.append(Spacer(1, 0.5*cm))

        # Metadata section
        story.append(Paragraph("Model Summary", styles["Heading2"]))
        meta_data = [["Key", "Value"]] + [[str(k), str(v)] for k, v in metadata.items()]
        t = Table(meta_data, colWidths=[6*cm, 10*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C63FF")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # Leaderboard
        if not leaderboard.empty:
            story.append(Paragraph("Model Leaderboard", styles["Heading2"]))
            lb_disp = leaderboard.drop(columns=["_model_obj"], errors="ignore").head(10)
            lb_data = [lb_disp.columns.tolist()] + lb_disp.values.tolist()
            t2 = Table(lb_data)
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00D4FF")),
                ("FONTSIZE",   (0, 0), (-1, -1), 8),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]))
            story.append(t2)
            story.append(Spacer(1, 0.5*cm))

        # Evaluation metrics
        if eval_results:
            story.append(Paragraph("Evaluation Metrics", styles["Heading2"]))
            skip_keys = {"confusion_matrix", "classification_report", "roc_curve",
                         "pr_curve", "y_pred", "y_test", "residuals", "y_pred_proba"}
            eval_data = [["Metric", "Value"]] + [
                [k, str(round(v, 4)) if isinstance(v, float) else str(v)]
                for k, v in eval_results.items()
                if k not in skip_keys and v is not None
            ]
            t3 = Table(eval_data, colWidths=[6*cm, 10*cm])
            t3.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C63FF")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTSIZE",   (0, 0), (-1, -1), 9),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]))
            story.append(t3)

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        # Fallback: plain-text PDF alternative
        return b"%PDF-1.4 (ReportLab not available)"


def leaderboard_excel(leaderboard: pd.DataFrame) -> bytes:
    clean = leaderboard.drop(columns=["_model_obj"], errors="ignore")
    return to_excel_bytes({"Leaderboard": clean})