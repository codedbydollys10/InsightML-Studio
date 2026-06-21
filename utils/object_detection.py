import time
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF


# COCO class labels (91 classes including background)
COCO_LABELS = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Category mapping: checkbox label -> set of COCO class names
CATEGORY_MAP: dict = {
    "Humans": {"person"},
    "Animals": {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe"},
    "Vehicles": {"bicycle", "car", "motorcycle", "airplane", "bus", "train",
                 "truck", "boat"},
    "Furniture": {"chair", "couch", "potted plant", "bed", "dining table"},
    "Electronics": {"tv", "laptop", "mouse", "remote", "keyboard",
                    "cell phone", "microwave", "oven", "toaster",
                    "refrigerator"},
    "Daily Objects": {"bottle", "wine glass", "cup", "fork", "knife", "spoon",
                      "bowl", "banana", "apple", "sandwich", "orange",
                      "broccoli", "carrot", "hot dog", "pizza", "donut",
                      "cake", "backpack", "umbrella", "handbag", "tie",
                      "suitcase", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"},
}

# Bright distinct colors per category
CATEGORY_COLORS: dict = {
    "Humans":       (255, 80,  80),
    "Animals":      (80,  220, 80),
    "Vehicles":     (80,  140, 255),
    "Furniture":    (255, 200, 0),
    "Electronics":  (220, 80,  255),
    "Daily Objects":(0,   220, 220),
    "Other":        (200, 200, 200),
}

_model = None


def _get_model():
    """Lazy-load Faster R-CNN MobileNetV3 (downloads weights once)."""
    global _model
    if _model is None:
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        _model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights
        )
        _model.eval()
    return _model


def _label_to_category(label: str) -> str:
    """Map a COCO label to one of our user-facing category names."""
    for cat, members in CATEGORY_MAP.items():
        if label in members:
            return cat
    return "Other"


def detect_objects(
    image: Image.Image,
    selected_categories: list,
    confidence_threshold: float = 0.45,
) -> dict:
    """
    Run Faster R-CNN on *image* and return only objects matching
    *selected_categories*.

    Parameters
    ----------
    image                : PIL Image
    selected_categories  : list of category strings (e.g. ["Humans", "Animals"])
                           or ["Detect Everything"]
    confidence_threshold : minimum score to include a detection

    Returns
    -------
    dict with:
        annotated_image  : PIL Image with bounding boxes
        detections       : list of dicts {name, category, confidence, bbox}
        inference_time_ms: float
    """
    t0 = time.perf_counter()

    model = _get_model()
    tensor = TF.to_tensor(image.convert("RGB"))

    with torch.no_grad():
        outputs = model([tensor])[0]

    t1 = time.perf_counter()

    detect_all = "Detect Everything" in selected_categories
    active_cats = set(selected_categories) if not detect_all else set(CATEGORY_MAP.keys()) | {"Other"}

    # Build allowed label set
    allowed_labels: set = set()
    if detect_all:
        allowed_labels = set(COCO_LABELS) - {"__background__", "N/A"}
    else:
        for cat in selected_categories:
            if cat in CATEGORY_MAP:
                allowed_labels |= CATEGORY_MAP[cat]

    detections = []
    img_draw = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_draw)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    for box, score, label_idx in zip(boxes, scores, labels):
        if score < confidence_threshold:
            continue
        label_name = COCO_LABELS[label_idx] if label_idx < len(COCO_LABELS) else "unknown"
        if label_name in ("N/A", "__background__"):
            continue
        if not detect_all and label_name not in allowed_labels:
            continue

        category = _label_to_category(label_name)
        color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["Other"])

        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label_text = f"{label_name} {score:.0%}"
        bbox_text = draw.textbbox((x1, y1 - 18), label_text, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1 - 18), label_text, fill=(255, 255, 255), font=font)

        detections.append({
            "name": label_name,
            "category": category,
            "confidence": round(float(score), 3),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "annotated_image": img_draw,
        "detections": detections,
        "inference_time_ms": round((t1 - t0) * 1000, 1),
    }
