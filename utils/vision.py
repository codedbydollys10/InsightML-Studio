
import time
from typing import Any

from PIL import Image

from utils.object_detection import detect_objects
from utils.scene_analysis import classify_scene
from utils.image_analysis import analyze_image


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

    if run_human and human_options:
        try:
            from utils.human_analysis import analyze_humans
        except ImportError as exc:
            raise ImportError(
                "MediaPipe dependency not available. Install 'mediapipe' and restart the app."
            ) from exc
        results["human_analysis"] = analyze_humans(image, human_options)

    if run_objects and object_categories:
        results["object_detection"] = detect_objects(
            image, object_categories, object_threshold
        )

    if run_scene and scene_categories:
        results["scene_analysis"] = classify_scene(image, scene_categories)

    if run_image_analysis:
        results["image_analysis"] = analyze_image(image)

    results["total_time_ms"] = round((time.perf_counter() - t_total) * 1000, 1)
    return results
