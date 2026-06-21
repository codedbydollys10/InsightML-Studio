import math
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageStat
from sklearn.cluster import KMeans


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to BGR numpy array."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def compute_brightness(gray: np.ndarray) -> float:
    """Mean pixel brightness on a 0-100 scale."""
    return round(float(np.mean(gray)) / 255 * 100, 2)


def compute_contrast(gray: np.ndarray) -> float:
    """Standard deviation of pixel intensities (0-100 scale)."""
    return round(float(np.std(gray)) / 255 * 100, 2)


def compute_blur_score(gray: np.ndarray) -> float:
    """Laplacian variance — higher means sharper image."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return round(float(lap.var()), 2)


def compute_sharpness(gray: np.ndarray) -> float:
    """Normalized sharpness index (0-100)."""
    blur = compute_blur_score(gray)
    # Empirically, a sharp image has variance > 500
    score = min(blur / 500 * 100, 100)
    return round(score, 2)


def compute_dominant_colors(rgb_array: np.ndarray, n_colors: int = 6) -> list:
    """
    Extract dominant colors using K-Means clustering.

    Returns list of {"hex": str, "rgb": (r,g,b), "percentage": float}
    """
    pixels = rgb_array.reshape(-1, 3).astype(np.float32)

    # Subsample for speed
    if len(pixels) > 5000:
        idx = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]

    n_colors = min(n_colors, len(pixels))
    km = KMeans(n_clusters=n_colors, random_state=42, n_init=5, max_iter=100)
    km.fit(pixels)

    counts = np.bincount(km.labels_, minlength=n_colors)
    total = counts.sum()
    centers = km.cluster_centers_.astype(int)

    colors = []
    for center, count in sorted(
        zip(centers, counts), key=lambda x: -x[1]
    ):
        r, g, b = int(center[0]), int(center[1]), int(center[2])
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
        colors.append({
            "hex": hex_color,
            "rgb": (r, g, b),
            "percentage": round(count / total * 100, 1),
        })
    return colors


def compute_histogram(gray: np.ndarray) -> dict:
    """
    Compute per-channel histograms for display.

    Returns {"r": list, "g": list, "b": list, "gray": list} each 256 bins.
    """
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return {
        "gray": hist_gray.flatten().tolist(),
    }


def compute_color_histogram(rgb_array: np.ndarray) -> dict:
    """Return normalized per-channel histograms."""
    r_hist = np.histogram(rgb_array[:, :, 0], bins=64, range=(0, 256))[0]
    g_hist = np.histogram(rgb_array[:, :, 1], bins=64, range=(0, 256))[0]
    b_hist = np.histogram(rgb_array[:, :, 2], bins=64, range=(0, 256))[0]
    total = rgb_array.shape[0] * rgb_array.shape[1]
    return {
        "r": (r_hist / total).tolist(),
        "g": (g_hist / total).tolist(),
        "b": (b_hist / total).tolist(),
        "bins": 64,
    }


def score_image_quality(
    brightness: float,
    contrast: float,
    sharpness: float,
    blur_score: float,
) -> dict:
    """
    Compute an overall quality score and label.

    Returns {"score": int, "label": str, "issues": list}
    """
    issues = []
    score = 100

    if brightness < 20:
        issues.append("Image is too dark")
        score -= 20
    elif brightness > 85:
        issues.append("Image is overexposed")
        score -= 15

    if contrast < 10:
        issues.append("Very low contrast")
        score -= 20
    elif contrast < 20:
        issues.append("Low contrast")
        score -= 10

    if blur_score < 50:
        issues.append("Image appears blurry")
        score -= 25
    elif blur_score < 150:
        issues.append("Slight blur detected")
        score -= 10

    score = max(0, score)

    if score >= 85:
        label = "Excellent"
    elif score >= 70:
        label = "Good"
    elif score >= 50:
        label = "Fair"
    else:
        label = "Poor"

    return {"score": score, "label": label, "issues": issues}


def analyze_image(image: Image.Image) -> dict:
    """
    Compute all image quality metrics for *image*.

    Parameters
    ----------
    image : PIL Image

    Returns
    -------
    dict with all computed metrics
    """
    rgb = image.convert("RGB")
    rgb_array = np.array(rgb)
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    width, height = image.size
    channels = len(image.getbands())
    file_size_kb = (width * height * channels) / 1024  # approx uncompressed

    gcd = math.gcd(width, height)
    aspect_w, aspect_h = width // gcd, height // gcd

    brightness = compute_brightness(gray)
    contrast = compute_contrast(gray)
    blur_score = compute_blur_score(gray)
    sharpness = compute_sharpness(gray)
    dominant_colors = compute_dominant_colors(rgb_array)
    histogram = compute_histogram(gray)
    color_histogram = compute_color_histogram(rgb_array)
    quality = score_image_quality(brightness, contrast, sharpness, blur_score)

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "aspect_ratio": f"{aspect_w}:{aspect_h}",
        "file_size_kb": round(file_size_kb, 1),
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "sharpness": sharpness,
        "dominant_colors": dominant_colors,
        "histogram": histogram,
        "color_histogram": color_histogram,
        "quality": quality,
    }
