import io
import sys
import os
import time
import json
import tempfile
import zipfile
from collections import Counter
from typing import Any

...

# Ensure the project root is on the path so utils imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image


def _load_analysis_dependencies():
    from utils.vision import run_analysis
    from utils.vision_report import generate_pdf_report, generate_json_report
    return run_analysis, generate_pdf_report, generate_json_report


def _is_image_filename(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def _extract_images_from_zip(uploaded_zip) -> list[dict]:
    images = []
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(uploaded_zip) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    normalized = info.filename.replace("\\", "/")
                    if normalized.startswith("__MACOSX/") or "/__MACOSX/" in normalized:
                        continue
                    parts = [p for p in normalized.split("/") if p]
                    if any(part.startswith(".") for part in parts):
                        continue
                    if not _is_image_filename(normalized):
                        continue

                    target_path = Path(tmpdir) / normalized
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with archive.open(info) as source, open(target_path, "wb") as target:
                            target.write(source.read())
                        image = Image.open(target_path).convert("RGB")
                        images.append({
                            "filename": os.path.basename(normalized),
                            "image": image,
                        })
                    except Exception:
                        continue
        except zipfile.BadZipFile as exc:
            raise ValueError("Uploaded file is not a valid ZIP archive.") from exc
    return images


def _prepare_batch_uploads(batch_uploads: list[dict]) -> list[dict]:
    """Return all batch uploads without truncating large archives."""
    return batch_uploads


def _generate_dataset_json_report(summary: dict, image_summaries: list[dict], input_filename: str) -> str:
    def _serialize(value):
        if isinstance(value, Counter):
            return dict(value)
        if isinstance(value, dict):
            return {k: _serialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_serialize(v) for v in value]
        return value

    report = {
        "report_generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_file": input_filename,
        "total_images": summary.get("total_images", 0),
        "processed_images": summary.get("processed_images", 0),
        "skipped_files": summary.get("skipped_files", 0),
        "summary": _serialize(summary),
        "images": _serialize(image_summaries),
    }
    return json.dumps(report, indent=2, ensure_ascii=False)


def _summarize_batch_results(batch_results: list[dict], run_human: bool, run_objects: bool, run_scene: bool, run_image_analysis: bool) -> dict:
    totals = {
        "total_images": len(batch_results),
        "processed_images": len(batch_results),
        "skipped_files": 0,
        "total_faces": 0,
        "total_hands": 0,
        "images_with_faces": 0,
        "images_with_hands": 0,
        "objects_by_name": Counter(),
        "objects_by_category": Counter(),
        "scene_counts": Counter(),
        "avg_brightness": None,
        "avg_contrast": None,
        "avg_sharpness": None,
        "avg_blur_score": None,
        "total_inference_ms": 0.0,
    }

    brightness_values = []
    contrast_values = []
    sharpness_values = []
    blur_values = []

    for image_result in batch_results:
        results = image_result["results"]
        totals["total_inference_ms"] += results.get("total_time_ms", 0.0)

        if run_human and results.get("human_analysis"):
            human = results["human_analysis"]
            totals["total_faces"] += human.get("num_faces", 0)
            totals["total_hands"] += human.get("num_hands", 0)
            if human.get("num_faces", 0) > 0:
                totals["images_with_faces"] += 1
            if human.get("num_hands", 0) > 0:
                totals["images_with_hands"] += 1

        if run_objects and results.get("object_detection"):
            for det in results["object_detection"].get("detections", []):
                totals["objects_by_name"][det["name"]] += 1
                totals["objects_by_category"][det["category"]] += 1

        if run_scene and results.get("scene_analysis"):
            predicted = results["scene_analysis"].get("predicted_scene")
            if predicted:
                totals["scene_counts"][predicted] += 1

        if run_image_analysis and results.get("image_analysis"):
            qa = results["image_analysis"]
            if qa.get("brightness") is not None:
                brightness_values.append(qa["brightness"])
            if qa.get("contrast") is not None:
                contrast_values.append(qa["contrast"])
            if qa.get("sharpness") is not None:
                sharpness_values.append(qa["sharpness"])
            if qa.get("blur_score") is not None:
                blur_values.append(qa["blur_score"])

    def _avg(values):
        return round(sum(values) / len(values), 1) if values else None

    totals["avg_brightness"] = _avg(brightness_values)
    totals["avg_contrast"] = _avg(contrast_values)
    totals["avg_sharpness"] = _avg(sharpness_values)
    totals["avg_blur_score"] = _avg(blur_values)
    totals["total_inference_ms"] = round(totals["total_inference_ms"], 1)

    return totals


def _make_image_summary(filename: str, results: dict, run_human: bool, run_objects: bool, run_scene: bool, run_image_analysis: bool) -> dict:
    summary = {"filename": filename}
    if run_human and results.get("human_analysis"):
        human = results["human_analysis"]
        summary.update({
            "num_faces": human.get("num_faces", 0),
            "num_hands": human.get("num_hands", 0),
            "pose_detected": human.get("pose_detected", False),
            "eyes_detected": "Yes" if human.get("eye_detected") else "No",
            "nose_detected": "Yes" if human.get("nose_detected") else "No",
            "lips_detected": "Yes" if human.get("lips_detected") else "No",
            "eyebrows_detected": "Yes" if human.get("eyebrows_detected") else "No",
            "left_ear_detected": "Yes" if human.get("left_ear_detected") else "No",
            "right_ear_detected": "Yes" if human.get("right_ear_detected") else "No",
        })
    if run_objects and results.get("object_detection"):
        dets = results["object_detection"].get("detections", [])
        summary["objects_detected"] = len(dets)
        summary["top_objects"] = [d["name"] for d in dets[:5]]
    if run_scene and results.get("scene_analysis"):
        summary["predicted_scene"] = results["scene_analysis"].get("predicted_scene")
    if run_image_analysis and results.get("image_analysis"):
        qa = results["image_analysis"]
        summary.update({
            "brightness": qa.get("brightness"),
            "contrast": qa.get("contrast"),
            "sharpness": qa.get("sharpness"),
            "blur_score": qa.get("blur_score"),
            "quality_label": qa.get("quality", {}).get("label"),
        })
    return summary


def _render_dataset_summary(summary: dict, top_objects: Counter, top_scenes: Counter) -> None:
    st.markdown("### 📦 Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", summary.get("total_images", 0))
    c2.metric("Faces Found", summary.get("total_faces", 0))
    c3.metric("Hands Found", summary.get("total_hands", 0))
    c4.metric("Avg Inference", f"{summary.get('total_inference_ms', 0) / max(summary.get('total_images', 1), 1):.1f} ms")

    st.markdown("**Top detected objects**")
    if top_objects:
        object_df = pd.DataFrame(
            list(top_objects.most_common(10)),
            columns=["Object", "Count"],
        )
        st.dataframe(object_df, use_container_width=True)
    else:
        st.info("No objects were detected across the dataset.")

    st.markdown("**Top scene labels**")
    if top_scenes:
        scene_df = pd.DataFrame(
            list(top_scenes.most_common(10)),
            columns=["Scene", "Count"],
        )
        st.dataframe(scene_df, use_container_width=True)

    if summary.get("avg_brightness") is not None:
        st.markdown("**Average image quality**")
        qcols = st.columns(4)
        qcols[0].metric("Brightness", summary["avg_brightness"])
        qcols[1].metric("Contrast", summary["avg_contrast"])
        qcols[2].metric("Sharpness", summary["avg_sharpness"])
        qcols[3].metric("Blur Score", summary["avg_blur_score"])


def _select_annotated_image(results: dict) -> Image.Image | None:
    if results.get("object_detection", {}).get("annotated_image") is not None:
        return results["object_detection"]["annotated_image"]
    if results.get("human_analysis", {}).get("annotated_image") is not None:
        return results["human_analysis"]["annotated_image"]
    return None


def _create_annotated_zip(batch_results: list[dict]) -> bytes:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, entry in enumerate(batch_results, start=1):
            annotated = _select_annotated_image(entry["results"])
            if annotated is None:
                continue
            image_bytes = io.BytesIO()
            annotated.save(image_bytes, format="PNG")
            image_bytes.seek(0)
            safe_name = entry["filename"].replace("\\", "/").replace("/", "_").replace(" ", "_")
            archive_name = f"annotated/{index:03d}_{safe_name}_annotated.png"
            archive.writestr(archive_name, image_bytes.read())
    return archive_bytes.getvalue()


def _create_batch_csv_report(image_summaries: list[dict]) -> bytes:
    csv_buffer = io.StringIO()
    pd.DataFrame(image_summaries).to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode("utf-8")


def _run_batch_analysis(
    uploads: list[dict],
    run_human: bool,
    human_options: dict,
    run_objects: bool,
    object_categories: list,
    object_threshold: float,
    run_scene: bool,
    scene_categories: list,
    run_image_analysis: bool,
    run_analysis: Any,
    progress: Any,
    status: Any,
) -> dict:
    batch_results = []
    failed_files = []
    for index, entry in enumerate(uploads, start=1):
        status.text(f"Analyzing {entry['filename']} ({index}/{len(uploads)})...")
        progress.progress(int((index - 1) / len(uploads) * 100))
        try:
            image_results = run_analysis(
                image=entry["image"],
                run_human=run_human,
                human_options=human_options,
                run_objects=run_objects,
                object_categories=object_categories,
                object_threshold=object_threshold,
                run_scene=run_scene,
                scene_categories=scene_categories,
                run_image_analysis=run_image_analysis,
            )
            batch_results.append({
                "filename": entry["filename"],
                "image": entry["image"],
                "results": image_results,
            })
        except Exception as exc:
            failed_files.append(entry["filename"])
            continue

    progress.progress(100)
    status.text("Batch analysis complete.")
    summary = _summarize_batch_results(
        batch_results,
        run_human=run_human,
        run_objects=run_objects,
        run_scene=run_scene,
        run_image_analysis=run_image_analysis,
    )
    summary["total_images"] = len(uploads)
    summary["successful_images"] = len(batch_results)
    summary["failed_images"] = len(failed_files)
    summary["failed_files"] = failed_files

    image_summaries = [
        _make_image_summary(
            item["filename"],
            item["results"],
            run_human=run_human,
            run_objects=run_objects,
            run_scene=run_scene,
            run_image_analysis=run_image_analysis,
        )
        for item in batch_results
    ]
    return {
        "batch_results": batch_results,
        "summary": summary,
        "image_summaries": image_summaries,
        "failed_files": failed_files,
    }


def render():
    # ─────────────────────────────────────────────────────────────────────────────
    # Page config
    # ─────────────────────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Vision AI Analyzer · InsightML Studio",
        page_icon="👁",
        layout="wide",
    )

    # ── Inject global premium CSS first ──────────────────────────────────────
    from utils.ui_theme import inject_css
    inject_css()

    # ─────────────────────────────────────────────────────────────────────────────
    # Additional Vision AI–specific CSS (harmonized with global purple theme)
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* ── Header ── */
    .vai-header {
        background: linear-gradient(135deg, rgba(124,58,237,0.2) 0%, rgba(13,15,26,0.95) 55%, rgba(37,99,235,0.12) 100%);
        border: 1px solid rgba(124,58,237,0.3);
        border-radius: 20px;
        padding: 2rem 2.5rem 1.6rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 30px rgba(124,58,237,0.18);
    }
    .vai-header::before {
        content: "";
        position: absolute;
        top: -50px; right: -50px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(124,58,237,0.3) 0%, transparent 70%);
        border-radius: 50%;
    }
    .vai-header::after {
        content: "";
        position: absolute;
        bottom: -40px; left: 30%;
        width: 150px; height: 150px;
        background: radial-gradient(circle, rgba(6,182,212,0.2) 0%, transparent 70%);
        border-radius: 50%;
    }
    .vai-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A78BFA, #67E8F9, #F0ABFC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0; line-height: 1.1;
        letter-spacing: -0.03em;
    }
    .vai-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    .vai-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(124,58,237,0.15);
        border: 1px solid rgba(124,58,237,0.4);
        color: #A78BFA;
        -webkit-text-fill-color: #A78BFA;
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-top: 0.8rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* ── Cards ── */
    .vai-card {
        background: var(--card, #191D35);
        border: 1px solid var(--border, #2A2F52);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        transition: all 0.25s ease;
    }
    .vai-card:hover {
        border-color: rgba(124,58,237,0.4);
        box-shadow: 0 4px 20px rgba(124,58,237,0.15);
    }
    .vai-card-title {
        color: #A78BFA;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    /* ── Metric pills ── */
    .metric-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
    .metric-pill {
        background: rgba(42,47,82,0.8);
        border: 1px solid rgba(124,58,237,0.25);
        border-radius: 8px;
        padding: 0.3rem 0.75rem;
        font-size: 0.8rem;
        color: #CBD5E1;
    }
    .metric-pill strong { color: #67E8F9; }

    /* ── Detection card ── */
    .det-card {
        background: rgba(13,15,26,0.8);
        border: 1px solid rgba(124,58,237,0.2);
        border-left: 3px solid #7C3AED;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: all 0.25s ease;
    }
    .det-card:hover {
        border-left-color: #A78BFA;
        background: rgba(124,58,237,0.06);
    }
    .det-card-title {
        font-weight: 700;
        color: #E2E8F0;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    .det-card-desc { color: #64748B; font-size: 0.82rem; }

    /* ── Color swatches ── */
    .swatch-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    .swatch {
        width: 48px; height: 48px;
        border-radius: 8px;
        border: 1px solid rgba(124,58,237,0.3);
        display: flex; align-items: flex-end;
        justify-content: center;
        font-size: 0.6rem;
        color: #fff;
        text-shadow: 0 1px 2px #000;
        padding-bottom: 2px;
    }

    /* ── Result badges ── */
    .badge-green  { background:#064E3B; color:#6EE7B7; border:1px solid #059669; border-radius:6px; padding:2px 10px; font-size:0.8rem; }
    .badge-amber  { background:#451A03; color:#FCD34D; border:1px solid #D97706; border-radius:6px; padding:2px 10px; font-size:0.8rem; }
    .badge-red    { background:#450A0A; color:#FCA5A5; border:1px solid #DC2626; border-radius:6px; padding:2px 10px; font-size:0.8rem; }
    .badge-blue   { background:rgba(37,99,235,0.2); color:#93C5FD; border:1px solid rgba(37,99,235,0.5); border-radius:6px; padding:2px 10px; font-size:0.8rem; }
    .badge-purple { background:rgba(124,58,237,0.2); color:#C4B5FD; border:1px solid rgba(124,58,237,0.5); border-radius:6px; padding:2px 10px; font-size:0.8rem; }

    /* ── Section divider ── */
    .section-hr { border: none; border-top: 1px solid rgba(42,47,82,0.8); margin: 1.5rem 0; }

    /* ── Table ── */
    .vai-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .vai-table th { background: rgba(42,47,82,0.8); color: #94A3B8; text-align: left; padding: 6px 12px; }
    .vai-table td { border-bottom: 1px solid rgba(42,47,82,0.6); padding: 6px 12px; color: #CBD5E1; }
    .vai-table tr:hover td { background: rgba(124,58,237,0.06); }
    </style>
    """, unsafe_allow_html=True)


    # ─────────────────────────────────────────────────────────────────────────────
    # Header
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="vai-header">
      <div class="vai-title">👁 Vision AI Analyzer</div>
      <div class="vai-subtitle">
        Analyze uploaded images using advanced Computer Vision and Artificial Intelligence.
      </div>
      <div class="vai-badge">🚀 POWERED BY MEDIAPIPE · FASTER R-CNN · RESNET50</div>
    </div>
    """, unsafe_allow_html=True)


    # ─────────────────────────────────────────────────────────────────────────────
    # Image Upload
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="vai-card">'
        '<div class="vai-card-title">📤 Choose Input</div>'
        '<div style="color:#94A3B8; margin-bottom:0.8rem;">Upload a JPG/PNG image or a ZIP archive containing images.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    upload_mode = st.radio(
        "Upload mode",
        ["Single Image", "ZIP Dataset"],
        index=0,
        horizontal=True,
        key="vision_ai_upload_mode",
    )
    st.markdown(f"**Current mode:** {upload_mode}")
    st.caption("Supported formats: PNG, JPG, JPEG, ZIP")

    uploaded_file = st.file_uploader(
        "Upload image or ZIP dataset",
        type=["png", "jpg", "jpeg", "zip"],
        label_visibility="visible",
        key="vision_ai_data_upload",
    )

    if uploaded_file is None:
        if upload_mode == "Single Image":
            st.info("👆 Upload a JPG/PNG image above to get started.")
        else:
            st.info("👆 Upload a ZIP archive containing JPG/PNG images to analyze a dataset.")
        st.stop()

    batch_mode = upload_mode == "ZIP Dataset"
    batch_uploads = []
    image = None
    width = height = channels = file_bytes = None

    if batch_mode:
        try:
            batch_uploads = _extract_images_from_zip(uploaded_file)
        except Exception as e:
            st.error(f"❌ Invalid ZIP file: {e}")
            st.stop()

        if not batch_uploads:
            st.warning("⚠️ No supported images were found inside the ZIP archive.")
            st.stop()

        st.markdown(f"<div class=\"vai-card\"><div class=\"vai-card-title\">📦 Dataset Loaded</div>\n" \
                    f"<div class=\"det-card-desc\">Found {len(batch_uploads)} image(s) inside the ZIP archive.</div></div>", unsafe_allow_html=True)
        batch_uploads = _prepare_batch_uploads(batch_uploads)
        first_item = batch_uploads[0]
        image = first_item["image"]
        width, height = image.size
        channels = len(image.getbands())
        file_bytes = uploaded_file.size
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"❌ Could not open image: {e}")
            st.stop()

        width, height = image.size
        channels = len(image.getbands())
        file_bytes = uploaded_file.size

    # ─────────────────────────────────────────────────────────────────────────────
    # Image Preview + Metadata
    # ─────────────────────────────────────────────────────────────────────────────
    col_img, col_meta = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="vai-card"><div class="vai-card-title">🖼 Original Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_meta:
        st.markdown(f"""
        <div class="vai-card">
          <div class="vai-card-title">📐 Image Metadata</div>
          <div class="metric-row">
            <div class="metric-pill">Width <strong>{width} px</strong></div>
            <div class="metric-pill">Height <strong>{height} px</strong></div>
            <div class="metric-pill">Channels <strong>{channels}</strong></div>
            <div class="metric-pill">File Size <strong>{file_bytes / 1024:.1f} KB</strong></div>
            <div class="metric-pill">Format <strong>{uploaded_file.type.split("/")[-1].upper()}</strong></div>
            <div class="metric-pill">Mode <strong>{image.mode}</strong></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Mini preview as numpy stats
        arr = np.array(image)
        st.markdown(f"""
        <div class="vai-card">
          <div class="vai-card-title">📊 Quick Stats</div>
          <div class="metric-row">
            <div class="metric-pill">Mean R <strong>{arr[:,:,0].mean():.1f}</strong></div>
            <div class="metric-pill">Mean G <strong>{arr[:,:,1].mean():.1f}</strong></div>
            <div class="metric-pill">Mean B <strong>{arr[:,:,2].mean():.1f}</strong></div>
            <div class="metric-pill">Std <strong>{arr.std():.1f}</strong></div>
            <div class="metric-pill">Min <strong>{arr.min()}</strong></div>
            <div class="metric-pill">Max <strong>{arr.max()}</strong></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown('<hr class="section-hr">', unsafe_allow_html=True)
    st.markdown("## 🔬 Detection Center")
    st.caption("Select what you want to analyze, then click **Analyze Image**.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Module 1 — Human Analysis
    # ─────────────────────────────────────────────────────────────────────────────
    with st.expander("👤 Module 1 · Human Analysis", expanded=False):
        st.markdown("""
        <div class="det-card-desc">
        Detect human features using MediaPipe. Select the body parts and features to analyze.
        </div>
        """, unsafe_allow_html=True)
        enable_human = st.checkbox("Enable Human Analysis", value=False, key="enable_human")

        if enable_human:
            st.markdown("**Select features to detect:**")
            hcol1, hcol2, hcol3 = st.columns(3)
            with hcol1:
                h_face_det  = st.checkbox("Face Detection",      key="h_face_det")
                h_face_mesh = st.checkbox("Face Mesh",           key="h_face_mesh")
                h_eyes      = st.checkbox("Eyes",                key="h_eyes")
                h_nose      = st.checkbox("Nose",                key="h_nose")
            with hcol2:
                h_lips      = st.checkbox("Lips",                key="h_lips")
                h_left_ear  = st.checkbox("Left Ear",            key="h_left_ear")
                h_right_ear = st.checkbox("Right Ear",           key="h_right_ear")
                h_eyebrows  = st.checkbox("Eyebrows",            key="h_eyebrows")
            with hcol3:
                h_hands     = st.checkbox("Hands",               key="h_hands")
                h_fingers   = st.checkbox("Fingers",             key="h_fingers")
                h_pose      = st.checkbox("Pose Detection",      key="h_pose")
                h_skeleton  = st.checkbox("Full Body Skeleton",  key="h_skeleton")

            human_options = {
                "face_detection": h_face_det,
                "face_mesh":      h_face_mesh,
                "eyes":           h_eyes,
                "nose":           h_nose,
                "lips":           h_lips,
                "left_ear":       h_left_ear,
                "right_ear":      h_right_ear,
                "eyebrows":       h_eyebrows,
                "hands":          h_hands,
                "fingers":        h_fingers,
                "pose":           h_pose,
                "skeleton":       h_skeleton,
            }
        else:
            human_options = {}


    # ─────────────────────────────────────────────────────────────────────────────
    # Module 2 — Object Detection
    # ─────────────────────────────────────────────────────────────────────────────
    with st.expander("📦 Module 2 · Object Detection", expanded=False):
        st.markdown("""
        <div class="det-card-desc">
        Detect objects using Faster R-CNN (MobileNetV3 backbone, COCO-trained). 
        Filter by category or detect everything.
        </div>
        """, unsafe_allow_html=True)
        enable_objects = st.checkbox("Enable Object Detection", value=False, key="enable_objects")

        if enable_objects:
            st.markdown("**Select object categories:**")
            ocol1, ocol2 = st.columns(2)
            with ocol1:
                o_humans   = st.checkbox("Humans",        key="o_humans")
                o_animals  = st.checkbox("Animals",       key="o_animals")
                o_vehicles = st.checkbox("Vehicles",      key="o_vehicles")
                o_furniture= st.checkbox("Furniture",     key="o_furniture")
            with ocol2:
                o_electronics = st.checkbox("Electronics",    key="o_electronics")
                o_daily       = st.checkbox("Daily Objects",  key="o_daily")
                o_everything  = st.checkbox("🔍 Detect Everything", key="o_everything")

            object_categories = []
            if o_everything:
                object_categories = ["Detect Everything"]
            else:
                if o_humans:
                    object_categories.append("Humans")
                if o_animals:
                    object_categories.append("Animals")
                if o_vehicles:
                    object_categories.append("Vehicles")
                if o_furniture:
                    object_categories.append("Furniture")
                if o_electronics:
                    object_categories.append("Electronics")
                if o_daily:
                    object_categories.append("Daily Objects")
            object_threshold = st.slider(
                "Object detection confidence threshold",
                min_value=0.20,
                max_value=0.95,
                value=0.45,
                step=0.05,
                key="object_threshold",
            )
        else:
            object_categories = []
            object_threshold = 0.45

    with st.expander("🌄 Module 3 · Scene Classification", expanded=False):
        st.markdown("""
        <div class="det-card-desc">
        Classify the overall scene of the image and map it to broad categories.
        </div>
        """, unsafe_allow_html=True)
        enable_scene = st.checkbox("Enable Scene Classification", value=False, key="enable_scene")
        scene_categories = []
        if enable_scene:
            cols = st.columns(3)
            scene_names = [
                "Indoor", "Outdoor", "Nature", "Urban", "Beach", "Forest",
                "Mountains", "River", "Lake", "Snow", "Desert", "Park",
                "Garden", "Office", "Kitchen", "Bedroom", "Restaurant",
                "Airport", "Railway Station",
            ]
            scene_selections = {}
            for index, name in enumerate(scene_names):
                scene_selections[name] = cols[index % len(cols)].checkbox(name, key=f"s_{name}")
            selected_scenes = [name for name, enabled in scene_selections.items() if enabled]
            scene_categories = selected_scenes or ["Detect Everything"]

    with st.expander("🧠 Module 4 · Image Quality", expanded=False):
        st.markdown("""
        <div class="det-card-desc">
        Analyze brightness, contrast, sharpness and blur for every uploaded image.
        </div>
        """, unsafe_allow_html=True)
        enable_quality = st.checkbox("Enable Image Quality Analysis", value=True, key="enable_quality")

    st.markdown('<hr class="section-hr">', unsafe_allow_html=True)

    analyze_label = "Analyze ZIP Dataset" if batch_mode else "Analyze Image"
    if st.button(analyze_label, key="vision_ai_run"):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        if batch_mode:
            run_analysis, generate_pdf_report, generate_json_report = _load_analysis_dependencies()
            batch_result = _run_batch_analysis(
                batch_uploads,
                enable_human,
                human_options,
                enable_objects,
                object_categories,
                object_threshold,
                enable_scene,
                scene_categories,
                enable_quality,
                run_analysis,
                progress_bar,
                progress_text,
            )

            summary = batch_result["summary"]
            st.success(f"Batch processing complete: {summary['successful_images']} / {summary['total_images']} images processed.")
            if summary.get("failed_images"):
                st.warning(f"{summary['failed_images']} image(s) failed: {', '.join(batch_result['failed_files'])}")

            _render_dataset_summary(summary, summary.get("objects_by_category", Counter()), summary.get("scene_counts", Counter()))

            report_col1, report_col2, report_col3 = st.columns(3)
            csv_bytes = _create_batch_csv_report(batch_result["image_summaries"])
            json_report = _generate_dataset_json_report(summary, batch_result["image_summaries"], getattr(uploaded_file, 'name', 'uploaded.zip'))
            annotated_zip = _create_annotated_zip(batch_result["batch_results"])

            report_col1.download_button("⬇️ Download CSV report", data=csv_bytes, file_name="vision_batch_report.csv", mime="text/csv")
            report_col2.download_button("⬇️ Download JSON report", data=json_report, file_name="vision_batch_report.json", mime="application/json")
            report_col3.download_button("⬇️ Download annotated images ZIP", data=annotated_zip, file_name="vision_batch_annotated_images.zip", mime="application/zip")

            for item in batch_result["batch_results"]:
                with st.expander(item["filename"], expanded=False):
                    cols = st.columns([1, 1])
                    cols[0].image(item["image"], caption="Original", use_column_width=True)
                    annotated = _select_annotated_image(item["results"])
                    if annotated is not None:
                        cols[1].image(annotated, caption="Annotated", use_column_width=True)
                    else:
                        cols[1].markdown("*No annotated image available.*")

                    result = item["results"]
                    if result.get("human_analysis"):
                        human = result["human_analysis"]
                        st.markdown(f"**Human analysis:** {human.get('num_faces', 0)} faces, {human.get('num_hands', 0)} hands.")
                    if result.get("object_detection"):
                        detections = result["object_detection"].get("detections", [])
                        st.markdown(f"**Objects detected:** {len(detections)}")
                        if detections:
                            st.table(pd.DataFrame(detections[:10]))
                    if result.get("scene_analysis"):
                        scene = result["scene_analysis"]
                        st.markdown(f"**Scene:** {scene.get('predicted_scene', '—')}")
                    if result.get("image_analysis"):
                        qa = result["image_analysis"]
                        st.markdown(f"**Quality:** {qa.get('quality', {}).get('label', '—')} | Brightness {qa.get('brightness', '—')} | Contrast {qa.get('contrast', '—')}")

        else:
            run_analysis, generate_pdf_report, generate_json_report = _load_analysis_dependencies()
            try:
                with st.spinner("Running lightweight preview. Heavy vision modules stay off unless you enable them."):
                    results = run_analysis(
                        image=image,
                        run_human=enable_human,
                        human_options=human_options,
                        run_objects=enable_objects,
                        object_categories=object_categories,
                        object_threshold=object_threshold,
                        run_scene=enable_scene,
                        scene_categories=scene_categories,
                        run_image_analysis=enable_quality,
                    )
            except Exception as exc:
                st.error(f"Vision analysis could not be completed: {exc}")
                st.stop()
            annotated = _select_annotated_image(results)
            st.success("Image analysis complete.")
            result_cols = st.columns([1, 1])
            result_cols[0].image(image, caption="Original Image", use_column_width=True)
            if annotated is not None:
                result_cols[1].image(annotated, caption="Annotated Image", use_column_width=True)
            else:
                result_cols[1].markdown("*No annotated image available.*")

            st.markdown("### Analysis Results")
            if results.get("human_analysis"):
                human = results["human_analysis"]
                st.markdown(f"- Faces detected: {human.get('num_faces', 0)}")
                st.markdown(f"- Hands detected: {human.get('num_hands', 0)}")
            if results.get("object_detection"):
                objects = results["object_detection"].get("detections", [])
                st.markdown(f"- Objects detected: {len(objects)}")
                if objects:
                    st.table(pd.DataFrame(objects[:10]))
            if results.get("scene_analysis"):
                st.markdown(f"- Scene prediction: {results['scene_analysis'].get('predicted_scene', '—')}")
            if results.get("image_analysis"):
                qa = results["image_analysis"]
                st.markdown(f"- Quality: {qa.get('quality', {}).get('label', '—')}")
                st.markdown(f"- Brightness: {qa.get('brightness', '—')}")
                st.markdown(f"- Contrast: {qa.get('contrast', '—')}")
                st.markdown(f"- Sharpness: {qa.get('sharpness', '—')}")
                st.markdown(f"- Blur score: {qa.get('blur_score', '—')}")

            pdf_report = generate_pdf_report(image, annotated, results, filename="vision_ai_report.pdf")
            download_cols = st.columns(2)
            download_cols[0].download_button("⬇️ Download PDF report", data=pdf_report, file_name="vision_ai_report.pdf", mime="application/pdf")
            download_cols[1].download_button("⬇️ Download JSON report", data=generate_json_report(results, {
                "filename": getattr(uploaded_file, 'name', 'image'),
                "width": width,
                "height": height,
                "channels": channels,
            }), file_name="vision_ai_report.json", mime="application/json")

  