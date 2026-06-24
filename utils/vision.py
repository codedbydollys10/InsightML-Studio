
import time
from typing import Any

from PIL import Image


def _prepare_image_for_analysis(image: Image.Image, max_side: int = 640) -> Image.Image:
    width, height = image.size
    if max(width, height) <= max_side:
        return image
    scale = max_side / max(width, height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _load_analysis_backend_modules():
    try:
        from utils.object_detection import detect_objects
        from utils.scene_analysis import classify_scene
        from utils.image_analysis import analyze_image
    except Exception as exc:
        raise ImportError(
            "Vision AI dependencies could not be loaded. Ensure torch, torchvision, and OpenCV are installed."
        ) from exc
    return detect_objects, classify_scene, analyze_image


def run_analysis(
    image: Image.Image,
    run_human: bool = False,
    human_options: dict = None,
    run_objects: bool = False,
    object_categories: list = None,
    object_threshold: float = 0.45,
    run_scene: bool = False,
    scene_categories: list = None,
    run_image_analysis: bool = False,
) -> dict:
    """
    Run the selected Vision AI sub-modules on *image*.

    Parameters
    ----------
    image               : PIL Image
    run_human           : whether to run human analysis
    human_options       : dict of bool flags for sub-features
    run_objects         : whether to run object detection
    object_categories   : list of category names
    object_threshold    : detection confidence threshold
    run_scene           : whether to run scene classification
    scene_categories    : list of scene category names
    run_image_analysis  : whether to run image quality analysis

    Returns
    -------
    dict with keys per enabled module, plus "total_time_ms"
    """
    t_total = time.perf_counter()
    results: dict = {}
    image_for_analysis = _prepare_image_for_analysis(image)
    detect_objects, classify_scene, analyze_image = _load_analysis_backend_modules()

    if run_human and human_options:
        try:
            from utils.human_analysis import analyze_humans
        except ImportError as exc:
            raise ImportError(
                "MediaPipe dependency not available. Install 'mediapipe' and restart the app."
            ) from exc
        results["human_analysis"] = analyze_humans(image_for_analysis, human_options)

    if run_objects and object_categories:
        results["object_detection"] = detect_objects(
            image_for_analysis, object_categories, object_threshold
        )

    if run_scene and scene_categories:
        results["scene_analysis"] = classify_scene(image_for_analysis, scene_categories)

    if run_image_analysis:
        results["image_analysis"] = analyze_image(image_for_analysis)

    results["total_time_ms"] = round((time.perf_counter() - t_total) * 1000, 1)
    return results
