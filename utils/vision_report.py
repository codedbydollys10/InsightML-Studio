import io
import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# ── Colors ───────────────────────────────────────────────────────────────────
BRAND_DARK  = colors.HexColor("#0F172A")
BRAND_BLUE  = colors.HexColor("#3B82F6")
BRAND_CYAN  = colors.HexColor("#06B6D4")
BRAND_LIGHT = colors.HexColor("#F1F5F9")
TEXT_GRAY   = colors.HexColor("#64748B")
GREEN       = colors.HexColor("#10B981")
RED         = colors.HexColor("#EF4444")
AMBER       = colors.HexColor("#F59E0B")


def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _quality_color(label: str):
    return {
        "Excellent": GREEN,
        "Good":      GREEN,
        "Fair":      AMBER,
        "Poor":      RED,
    }.get(label, TEXT_GRAY)


def generate_pdf_report(
    original_image: Image.Image,
    annotated_image: Image.Image | None,
    results: dict,
    filename: str = "vision_ai_report.pdf",
) -> bytes:
    """
    Build a PDF report from the analysis results.

    Parameters
    ----------
    original_image  : PIL Image
    annotated_image : PIL Image or None
    results         : dict from vision.py run_analysis()
    filename        : output filename (used internally)

    Returns
    -------
    bytes of the PDF
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "VisionTitle",
        parent=styles["Title"],
        fontSize=26,
        textColor=BRAND_DARK,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "VisionSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=TEXT_GRAY,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=BRAND_BLUE,
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "VisionBody",
        parent=styles["Normal"],
        fontSize=10,
        textColor=BRAND_DARK,
        spaceAfter=4,
        leading=14,
    )
    kv_key = ParagraphStyle(
        "KVKey",
        parent=styles["Normal"],
        fontSize=9,
        textColor=TEXT_GRAY,
        fontName="Helvetica-Bold",
    )
    kv_val = ParagraphStyle(
        "KVVal",
        parent=styles["Normal"],
        fontSize=9,
        textColor=BRAND_DARK,
    )

    story = []
    page_w = A4[0] - 36 * mm

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("👁 Vision AI Analyzer", title_style))
    story.append(Paragraph(
        f"Analysis Report  ·  {datetime.now().strftime('%B %d, %Y  %H:%M')}",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_BLUE))
    story.append(Spacer(1, 8 * mm))

    # ── Images side by side ───────────────────────────────────────────────────
    img_w = (page_w - 6 * mm) / 2
    img_h = img_w * 0.75

    orig_bytes = _pil_to_bytes(original_image)
    orig_rl = RLImage(io.BytesIO(orig_bytes), width=img_w, height=img_h)

    if annotated_image:
        ann_bytes = _pil_to_bytes(annotated_image)
        ann_rl = RLImage(io.BytesIO(ann_bytes), width=img_w, height=img_h)
        img_table = Table(
            [[orig_rl, ann_rl],
             [Paragraph("Original Image", kv_key),
              Paragraph("Annotated Image", kv_key)]],
            colWidths=[img_w, img_w],
        )
    else:
        img_table = Table(
            [[orig_rl], [Paragraph("Original Image", kv_key)]],
            colWidths=[img_w],
        )

    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 6 * mm))

    # ── Image Information ─────────────────────────────────────────────────────
    img_info = results.get("image_analysis", {})
    if img_info:
        story.append(Paragraph("Image Information", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=BRAND_LIGHT))
        data = [
            ["Resolution", f"{img_info.get('width')} × {img_info.get('height')} px"],
            ["Channels", str(img_info.get("channels"))],
            ["Aspect Ratio", img_info.get("aspect_ratio", "—")],
            ["Approx. File Size", f"{img_info.get('file_size_kb', '—')} KB"],
            ["Brightness", f"{img_info.get('brightness', '—')} / 100"],
            ["Contrast", f"{img_info.get('contrast', '—')} / 100"],
            ["Sharpness", f"{img_info.get('sharpness', '—')} / 100"],
            ["Blur Score", str(img_info.get("blur_score", "—"))],
        ]
        quality = img_info.get("quality", {})
        if quality:
            data.append(["Overall Quality", f"{quality.get('label')} ({quality.get('score')}/100)"])
        _add_kv_table(story, data, page_w, kv_key, kv_val)

    # ── Human Analysis ────────────────────────────────────────────────────────
    human = results.get("human_analysis", {})
    if human:
        story.append(Paragraph("Human Analysis", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=BRAND_LIGHT))
        data = [
            ["Faces Detected", str(human.get("num_faces", 0))],
            ["Hands Detected", str(human.get("num_hands", 0))],
            ["Pose Detected", "Yes" if human.get("pose_detected") else "No"],
        ]
        if human.get("face_confidence"):
            data.append(["Face Confidence", f"{human['face_confidence']:.1%}"])
        if human.get("hand_confidence"):
            data.append(["Hand Confidence", f"{human['hand_confidence']:.1%}"])
        data.append(["Inference Time", f"{human.get('inference_time_ms', '—')} ms"])
        _add_kv_table(story, data, page_w, kv_key, kv_val)

    # ── Object Detection ──────────────────────────────────────────────────────
    obj_det = results.get("object_detection", {})
    if obj_det:
        story.append(Paragraph("Detected Objects", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=BRAND_LIGHT))
        detections = obj_det.get("detections", [])
        if detections:
            header = [
                Paragraph("Object", kv_key),
                Paragraph("Category", kv_key),
                Paragraph("Confidence", kv_key),
                Paragraph("Bounding Box", kv_key),
            ]
            rows = [header]
            for d in detections[:20]:
                bb = d.get("bbox", {})
                bb_str = f"({bb.get('x1')},{bb.get('y1')}) → ({bb.get('x2')},{bb.get('y2')})"
                rows.append([
                    Paragraph(d.get("name", "—"), body_style),
                    Paragraph(d.get("category", "—"), body_style),
                    Paragraph(f"{d.get('confidence', 0):.1%}", body_style),
                    Paragraph(bb_str, body_style),
                ])
            t = Table(rows, colWidths=[page_w * 0.25] * 4)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_LIGHT),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_LIGHT]),
                ("ALIGN", (2, 0), (2, -1), "CENTER"),
            ]))
            story.append(t)
        else:
            story.append(Paragraph("No objects detected.", body_style))
        story.append(Paragraph(
            f"Inference Time: {obj_det.get('inference_time_ms', '—')} ms",
            body_style,
        ))

    # ── Scene Analysis ────────────────────────────────────────────────────────
    scene = results.get("scene_analysis", {})
    if scene:
        story.append(Paragraph("Scene Analysis", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=BRAND_LIGHT))
        story.append(Paragraph(
            f"<b>Predicted Scene:</b> {scene.get('predicted_scene', '—')}",
            body_style,
        ))
        top5 = scene.get("top5", [])
        if top5:
            rows = [[
                Paragraph("Scene", kv_key),
                Paragraph("Confidence", kv_key),
            ]]
            for item in top5:
                rows.append([
                    Paragraph(item["scene"], body_style),
                    Paragraph(f"{item['confidence']:.1%}", body_style),
                ])
            t = Table(rows, colWidths=[page_w * 0.6, page_w * 0.4])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_LIGHT),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ]))
            story.append(t)
        story.append(Paragraph(
            f"Inference Time: {scene.get('inference_time_ms', '—')} ms",
            body_style,
        ))

    # ── Summary ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_BLUE))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Overall Summary", section_style))
    summary_lines = _build_summary(results)
    for line in summary_lines:
        story.append(Paragraph(f"• {line}", body_style))

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        "Generated by InsightML Studio · Vision AI Analyzer",
        ParagraphStyle("Footer", parent=styles["Normal"],
                       fontSize=8, textColor=TEXT_GRAY, alignment=TA_CENTER),
    ))

    doc.build(story)
    return buf.getvalue()


def _add_kv_table(story, data, page_w, kv_key, kv_val):
    """Add a two-column key-value table to the story."""
    rows = [
        [Paragraph(k, kv_key), Paragraph(str(v), kv_val)]
        for k, v in data
    ]
    t = Table(rows, colWidths=[page_w * 0.38, page_w * 0.62])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t)


def _build_summary(results: dict) -> list:
    """Build a human-readable bullet summary from all results."""
    lines = []

    img = results.get("image_analysis", {})
    if img:
        q = img.get("quality", {})
        lines.append(
            f"Image resolution {img.get('width')}×{img.get('height')} px, "
            f"quality rated {q.get('label', '—')} ({q.get('score', '—')}/100)."
        )
        if q.get("issues"):
            lines.append("Quality issues: " + ", ".join(q["issues"]) + ".")

    human = results.get("human_analysis", {})
    if human:
        parts = []
        if human.get("num_faces"):
            parts.append(f"{human['num_faces']} face(s)")
        if human.get("num_hands"):
            parts.append(f"{human['num_hands']} hand(s)")
        if human.get("pose_detected"):
            parts.append("a body pose")
        if parts:
            lines.append("Human analysis detected: " + ", ".join(parts) + ".")
        else:
            lines.append("No humans detected in the selected analysis.")

    obj_det = results.get("object_detection", {})
    if obj_det:
        dets = obj_det.get("detections", [])
        if dets:
            names = list({d["name"] for d in dets})[:8]
            lines.append(f"Object detection found {len(dets)} object(s): {', '.join(names)}.")
        else:
            lines.append("No objects detected in the selected categories.")

    scene = results.get("scene_analysis", {})
    if scene:
        lines.append(f"Scene classified as: {scene.get('predicted_scene', '—')}.")

    return lines or ["Analysis complete."]


def generate_json_report(results: dict, image_meta: dict) -> str:
    """
    Serialize results to a pretty-printed JSON string.

    Parameters
    ----------
    results    : dict from vision.py run_analysis()
    image_meta : dict with original image metadata

    Returns
    -------
    JSON string
    """
    # Make PIL Images serializable
    clean = _make_json_serializable(results)
    report = {
        "report_generated": datetime.now().isoformat(),
        "image_metadata": image_meta,
        "analysis_results": clean,
        "summary": _build_summary(results),
    }
    return json.dumps(report, indent=2, ensure_ascii=False)


def _make_json_serializable(obj):
    """Recursively convert non-serializable types."""
    if isinstance(obj, Image.Image):
        return "<PIL.Image>"
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(i) for i in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
