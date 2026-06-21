"""
human_analysis.py
-----------------
MediaPipe-powered human analysis: face detection, face mesh, hands, pose.
Returns annotated image and structured metrics.
"""

import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

try:
    from mediapipe.tasks.python.vision import (
        FaceDetector,
        FaceLandmarker,
        HandLandmarker,
        PoseLandmarker,
        FaceLandmarksConnections,
        HandLandmarksConnections,
        PoseLandmarksConnections,
        drawing_utils as mp_drawing,
        drawing_styles as mp_drawing_styles,
    )
    from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat
except ImportError as exc:
    raise ImportError(
        "MediaPipe task API not available. Install a compatible mediapipe version "
        "and ensure the virtual environment interpreter is active. "
        f"Original error: {exc}"
    ) from exc

MODEL_FILE_CANDIDATES = [
    Path.cwd() / "models",
    Path.cwd() / "assets" / "mediapipe_models",
    Path(__file__).resolve().parents[1] / "models",
    Path(__file__).resolve().parents[1] / "assets" / "mediapipe_models",
]

MODEL_FILENAMES = {
    "face_detection": "face_detector.task",
    "face_landmark": "face_landmarker.task",
    "hand_landmark": "hand_landmarker.task",
    "pose_landmark": "pose_landmarker.task",
}

# MediaPipe 478-point face mesh landmark indices per region
LEFT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]
LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

FACE_REGIONS: dict = {
    "eyes": LEFT_EYE_INDICES + RIGHT_EYE_INDICES,
    "nose": [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94],
    "lips": [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
        95, 78, 191, 80, 81, 82, 13, 312, 311, 310,
        415, 308,
    ],
    "eyebrows": LEFT_EYEBROW_INDICES + RIGHT_EYEBROW_INDICES,
    "left_ear": [234, 93, 132, 58, 172, 136, 150, 149, 176, 148],
    "right_ear": [454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
}


def _find_model_path(key: str) -> str:
    """Find a MediaPipe task model file for the requested feature."""
    filename = MODEL_FILENAMES.get(key)
    if not filename:
        raise ValueError(f"Unknown MediaPipe task model key: {key}")
    for base_dir in MODEL_FILE_CANDIDATES:
        model_path = base_dir / filename
        if model_path.exists():
            return str(model_path)
    raise FileNotFoundError(
        f"MediaPipe task model not found: {filename}. "
        f"Place the model in one of: {', '.join(str(p) for p in MODEL_FILE_CANDIDATES)}"
    )


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to BGR numpy array."""
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _pil_to_mp_image(image: Image.Image) -> MpImage:
    """Convert PIL Image to MediaPipe Image."""
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    return MpImage(ImageFormat.SRGB, rgb)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL Image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _draw_region(
    canvas: np.ndarray,
    landmarks: list,
    image_w: int,
    image_h: int,
    indices: list,
    color: tuple = (0, 255, 180),
    radius: int = 4,
) -> None:
    """Draw selected landmark indices on the canvas."""
    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)
        cv2.circle(canvas, (x, y), radius, color, -1)


def _draw_region_outline(
    canvas: np.ndarray,
    landmarks: list,
    image_w: int,
    image_h: int,
    indices: list,
    color: tuple = (0, 0, 255),
    thickness: int = 3,
) -> None:
    """Draw an outlined shape around a region using selected landmark indices."""
    points = []
    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        points.append((int(lm.x * image_w), int(lm.y * image_h)))
    if len(points) >= 3:
        pts_np = np.array(points, dtype=np.int32)
        cv2.polylines(canvas, [pts_np], True, color, thickness)
    elif len(points) == 2:
        cv2.line(canvas, points[0], points[1], color, thickness)
    elif len(points) == 1:
        cv2.circle(canvas, points[0], thickness * 2, color, thickness)


def _draw_eye_overlay(
    canvas: np.ndarray,
    landmarks: list,
    image_w: int,
    image_h: int,
    indices: list,
    label: str,
) -> None:
    """Draw eye landmarks and highlight the eye region."""
    pts = []
    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        pt = (int(lm.x * image_w), int(lm.y * image_h))
        pts.append(pt)
        cv2.circle(canvas, pt, 3, (0, 0, 255), -1)
    if not pts:
        return
    _draw_region_outline(canvas, landmarks, image_w, image_h, indices, color=(0, 0, 255), thickness=2)
    x_coords, y_coords = zip(*pts)
    cv2.putText(
        canvas,
        label,
        (min(x_coords), max(min(y_coords) - 12, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_nose_overlay(
    canvas: np.ndarray,
    landmarks: list,
    image_w: int,
    image_h: int,
    indices: list,
) -> None:
    """Draw a bounding rectangle around the nose and the nose landmarks."""
    pts = []
    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        pt = (int(lm.x * image_w), int(lm.y * image_h))
        pts.append(pt)
        cv2.circle(canvas, pt, 3, (0, 255, 0), -1)
    if not pts:
        return
    x_coords, y_coords = zip(*pts)
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    cv2.rectangle(canvas, (xmin - 6, ymin - 6), (xmax + 6, ymax + 6), (0, 255, 0), 2)
    cv2.putText(canvas, "Nose", (xmin, max(ymin - 14, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)


def _draw_lips_overlay(
    canvas: np.ndarray,
    landmarks: list,
    image_w: int,
    image_h: int,
    indices: list,
) -> None:
    """Draw a contour around the lips."""
    pts = []
    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        pt = (int(lm.x * image_w), int(lm.y * image_h))
        pts.append(pt)
    if len(pts) < 2:
        return
    pts_np = np.array(pts, dtype=np.int32)
    cv2.polylines(canvas, [pts_np], True, (255, 0, 0), 3)
    for pt in pts:
        cv2.circle(canvas, pt, 2, (255, 0, 0), -1)
    x_coords, y_coords = zip(*pts)
    cv2.putText(canvas, "Lips", (min(x_coords), min(y_coords) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA)


def _draw_eyebrows_overlay(
    canvas: np.ndarray,
    landmarks: list,
    image_w: int,
    image_h: int,
    indices: list,
) -> None:
    """Draw contours around eyebrows."""
    pts = []
    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        pt = (int(lm.x * image_w), int(lm.y * image_h))
        pts.append(pt)
    if len(pts) < 2:
        return
    pts_np = np.array(pts, dtype=np.int32)
    cv2.polylines(canvas, [pts_np], False, (0, 255, 255), 3)
    for pt in pts:
        cv2.circle(canvas, pt, 2, (0, 255, 255), -1)
    cv2.putText(canvas, "Eyebrows", (pts[0][0], pts[0][1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)


def _draw_ear_overlay(
    canvas: np.ndarray,
    landmarks: list,
    image_w: int,
    image_h: int,
    indices: list,
    label: str,
) -> None:
    """Highlight the selected ear."""
    pts = []
    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        pt = (int(lm.x * image_w), int(lm.y * image_h))
        pts.append(pt)
    if not pts:
        return
    pts_np = np.array(pts, dtype=np.int32)
    cv2.polylines(canvas, [pts_np], False, (255, 0, 255), 3)
    for pt in pts:
        cv2.circle(canvas, pt, 3, (255, 0, 255), -1)
    x_coords, y_coords = zip(*pts)
    cv2.putText(canvas, label, (min(x_coords), min(y_coords) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2, cv2.LINE_AA)


def _draw_hand_overlay(
    canvas: np.ndarray,
    hand_landmarks: list,
    image_w: int,
    image_h: int,
    draw_fingers: bool,
) -> None:
    """Draw hand landmarks and optionally highlight fingertips."""
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]
    points = []
    for lm in hand_landmarks:
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)
        points.append((x, y))
        cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
    for start, end in connections:
        if start < len(points) and end < len(points):
            cv2.line(canvas, points[start], points[end], (0, 255, 0), 2)
    if points:
        cv2.putText(canvas, "Hand", (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    if draw_fingers:
        tips = [4, 8, 12, 16, 20]
        for tip in tips:
            if tip < len(points):
                cv2.circle(canvas, points[tip], 7, (0, 165, 255), -1)


def _draw_pose_overlay(
    canvas: np.ndarray,
    pose_landmarks: list,
    image_w: int,
    image_h: int,
) -> None:
    """Draw a full pose skeleton."""
    pose_connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
        (25, 27), (26, 28), (27, 31), (28, 32),
        (11, 5), (12, 6), (5, 7), (6, 8),
        (5, 21), (6, 22), (21, 22)
    ]
    points = []
    for lm in pose_landmarks:
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)
        points.append((x, y))
    for start, end in pose_connections:
        if start < len(points) and end < len(points):
            cv2.line(canvas, points[start], points[end], (255, 255, 0), 2)
    for x, y in points:
        cv2.circle(canvas, (x, y), 4, (255, 255, 0), -1)
    if points:
        cv2.putText(canvas, "Pose", (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)


def _region_detected(landmarks: list, indices: list) -> bool:
    return any(idx < len(landmarks) for idx in indices)


def analyze_humans(
    image: Image.Image,
    options: dict,
) -> dict:
    """Run selected human-analysis sub-modules and return annotated results."""
    t0 = time.perf_counter()

    bgr = _pil_to_bgr(image)
    canvas = bgr.copy()
    h, w = canvas.shape[:2]

    num_faces = 0
    num_hands = 0
    pose_detected = False
    face_conf = None
    hand_conf = None
    pose_conf = None

    need_face_detection = options.get("face_detection", False)
    need_face_mesh = options.get("face_mesh", False)
    need_face_shape = any(
        options.get(k, False)
        for k in ("eyes", "nose", "lips", "eyebrows", "left_ear", "right_ear")
    )
    need_face_landmarker = need_face_mesh or need_face_shape
    draw_face_mesh = need_face_mesh and not need_face_shape
    need_hands = options.get("hands", False) or options.get("fingers", False)
    need_pose = options.get("pose", False) or options.get("skeleton", False)

    region_detection = {
        "eyes": False,
        "nose": False,
        "lips": False,
        "eyebrows": False,
        "left_ear": False,
        "right_ear": False,
    }

    mp_image = _pil_to_mp_image(image)

    if need_face_detection:
        face_model_path = _find_model_path("face_detection")
        face_detector = FaceDetector.create_from_model_path(face_model_path)
        try:
            result = face_detector.detect(mp_image)
            if result and getattr(result, "detections", None):
                num_faces = len(result.detections)
                for det in result.detections:
                    if det.categories:
                        score = det.categories[0].score
                        face_conf = max(face_conf or 0.0, score)
                    mp_drawing.draw_detection(canvas, det)
        finally:
            face_detector.close()

    if need_face_landmarker:
        region_detection = {
            "eyes": False,
            "nose": False,
            "lips": False,
            "eyebrows": False,
            "left_ear": False,
            "right_ear": False,
        }
        face_landmark_model_path = _find_model_path("face_landmark")
        face_landmarker = FaceLandmarker.create_from_model_path(face_landmark_model_path)
        try:
            result = face_landmarker.detect(mp_image)
            if result and getattr(result, "face_landmarks", None):
                if not need_face_detection:
                    num_faces = len(result.face_landmarks)
                for face_landmarks in result.face_landmarks:
                    if draw_face_mesh:
                        mp_drawing.draw_landmarks(
                            canvas,
                            face_landmarks,
                            FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                        )

                    left_eye_detected = _region_detected(face_landmarks, LEFT_EYE_INDICES)
                    right_eye_detected = _region_detected(face_landmarks, RIGHT_EYE_INDICES)
                    region_detection["eyes"] = region_detection["eyes"] or left_eye_detected or right_eye_detected
                    if options.get("eyes", False):
                        if left_eye_detected:
                            _draw_eye_overlay(canvas, face_landmarks, w, h, LEFT_EYE_INDICES, "Left Eye")
                        if right_eye_detected:
                            _draw_eye_overlay(canvas, face_landmarks, w, h, RIGHT_EYE_INDICES, "Right Eye")

                    nose_detected = _region_detected(face_landmarks, FACE_REGIONS["nose"])
                    region_detection["nose"] = region_detection["nose"] or nose_detected
                    if options.get("nose", False) and nose_detected:
                        _draw_nose_overlay(canvas, face_landmarks, w, h, FACE_REGIONS["nose"])

                    lips_detected = _region_detected(face_landmarks, FACE_REGIONS["lips"])
                    region_detection["lips"] = region_detection["lips"] or lips_detected
                    if options.get("lips", False) and lips_detected:
                        _draw_lips_overlay(canvas, face_landmarks, w, h, FACE_REGIONS["lips"])

                    eyebrows_detected = _region_detected(face_landmarks, FACE_REGIONS["eyebrows"])
                    region_detection["eyebrows"] = region_detection["eyebrows"] or eyebrows_detected
                    if options.get("eyebrows", False) and eyebrows_detected:
                        _draw_eyebrows_overlay(canvas, face_landmarks, w, h, FACE_REGIONS["eyebrows"])

                    left_ear_detected = _region_detected(face_landmarks, FACE_REGIONS["left_ear"])
                    region_detection["left_ear"] = region_detection["left_ear"] or left_ear_detected
                    if options.get("left_ear", False) and left_ear_detected:
                        _draw_ear_overlay(canvas, face_landmarks, w, h, FACE_REGIONS["left_ear"], "Left Ear")

                    right_ear_detected = _region_detected(face_landmarks, FACE_REGIONS["right_ear"])
                    region_detection["right_ear"] = region_detection["right_ear"] or right_ear_detected
                    if options.get("right_ear", False) and right_ear_detected:
                        _draw_ear_overlay(canvas, face_landmarks, w, h, FACE_REGIONS["right_ear"], "Right Ear")
        finally:
            face_landmarker.close()

    if need_hands:
        hand_model_path = _find_model_path("hand_landmark")
        hand_landmarker = HandLandmarker.create_from_model_path(hand_model_path)
        try:
            result = hand_landmarker.detect(mp_image)
            if result and getattr(result, "hand_landmarks", None):
                num_hands = len(result.hand_landmarks)
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    if result.handedness and i < len(result.handedness):
                        hand_handedness = result.handedness[i]
                        if hand_handedness:
                            score = hand_handedness[0].score
                            hand_conf = max(hand_conf or 0.0, score)
                    if options.get("fingers", False):
                        FINGER_TIPS = [4, 8, 12, 16, 20]
                        FINGER_JOINTS = [2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
                        for idx in FINGER_TIPS + FINGER_JOINTS:
                            if idx >= len(hand_landmarks):
                                continue
                            lm = hand_landmarks[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            color = (0, 255, 100) if idx in FINGER_TIPS else (180, 255, 0)
                            cv2.circle(canvas, (x, y), 5, color, -1)
                    mp_drawing.draw_landmarks(
                        canvas,
                        hand_landmarks,
                        HandLandmarksConnections.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
        finally:
            hand_landmarker.close()

    if need_pose:
        pose_model_path = _find_model_path("pose_landmark")
        pose_landmarker = PoseLandmarker.create_from_model_path(pose_model_path)
        try:
            result = pose_landmarker.detect(mp_image)
            if result and getattr(result, "pose_landmarks", None):
                pose_detected = True
                pose_conf = 0.85
                for pose_landmarks in result.pose_landmarks:
                    if options.get("skeleton", False):
                        mp_drawing.draw_landmarks(
                            canvas,
                            pose_landmarks,
                            PoseLandmarksConnections.POSE_LANDMARKS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                        )
                    else:
                        for lm in pose_landmarks:
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(canvas, (x, y), 6, (255, 100, 0), -1)
        finally:
            pose_landmarker.close()

    t1 = time.perf_counter()

    return {
        "annotated_image": _bgr_to_pil(canvas),
        "num_faces": num_faces,
        "num_hands": num_hands,
        "pose_detected": pose_detected,
        "face_confidence": round(face_conf, 3) if face_conf is not None else None,
        "hand_confidence": round(hand_conf, 3) if hand_conf is not None else None,
        "pose_confidence": round(pose_conf, 3) if pose_conf is not None else None,
        "eye_detected": bool(region_detection.get("eyes")) if need_face_landmarker else False,
        "nose_detected": bool(region_detection.get("nose")) if need_face_landmarker else False,
        "lips_detected": bool(region_detection.get("lips")) if need_face_landmarker else False,
        "eyebrows_detected": bool(region_detection.get("eyebrows")) if need_face_landmarker else False,
        "left_ear_detected": bool(region_detection.get("left_ear")) if need_face_landmarker else False,
        "right_ear_detected": bool(region_detection.get("right_ear")) if need_face_landmarker else False,
        "inference_time_ms": round((t1 - t0) * 1000, 1),
    }
