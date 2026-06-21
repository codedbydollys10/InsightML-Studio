import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image


# Places365 label URL
PLACES365_LABEL_URL = (
    "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
)
PLACES365_WEIGHTS_URL = (
    "http://places2.csail.mit.edu/models_places365/"
    "resnet50_places365.pth.tar"
)

CACHE_DIR = Path(__file__).parent.parent / "models" / "vision"
LABELS_CACHE = CACHE_DIR / "places365_labels.txt"

# Scene category groups — maps Places365-style scene keywords to user categories
SCENE_GROUPS: dict = {
    "Indoor": ["bedroom", "office", "kitchen", "restaurant", "living_room",
               "bathroom", "corridor", "elevator", "auditorium", "classroom",
               "hospital_room", "laundromat", "lobby", "library", "museum",
               "reception", "shopping_mall", "supermarket", "waiting_room"],
    "Outdoor": ["street", "road", "alley", "parking_lot", "plaza", "courtyard",
                "outdoor"],
    "Nature": ["forest", "mountain", "river", "lake", "beach", "valley",
               "waterfall", "canyon", "field", "meadow", "swamp", "bog",
               "tundra", "rainforest", "glacier", "cliff", "volcano"],
    "Urban": ["downtown", "city", "skyscraper", "building", "bridge",
              "stadium", "airport", "highway", "tower", "railroad"],
    "Beach": ["beach", "coast", "shore", "ocean", "sea", "bay", "marina"],
    "Forest": ["forest", "wood", "jungle", "grove", "rainforest", "bamboo"],
    "Mountains": ["mountain", "hill", "cliff", "peak", "ridge", "summit",
                  "valley", "canyon"],
    "River": ["river", "creek", "stream", "canal", "waterfall"],
    "Lake": ["lake", "pond", "reservoir"],
    "Snow": ["snow", "glacier", "tundra", "blizzard", "ski_slope"],
    "Desert": ["desert", "dune", "arid", "badlands"],
    "Park": ["park", "garden", "playground", "botanical", "green"],
    "Garden": ["garden", "greenhouse", "orchard", "vineyard"],
    "Office": ["office", "conference_room", "cubicle", "server_room"],
    "Kitchen": ["kitchen", "pantry", "cafeteria"],
    "Bedroom": ["bedroom", "dorm_room"],
    "Restaurant": ["restaurant", "bar", "pub", "cafe", "cafeteria", "food_court"],
    "Airport": ["airport", "runway", "hangar", "terminal"],
    "Railway Station": ["railroad", "train_station", "subway_station", "platform"],
}

# User-facing category checkboxes order (same as spec)
USER_CATEGORIES = [
    "Indoor", "Outdoor", "Nature", "Urban", "Beach", "Forest",
    "Mountains", "River", "Lake", "Snow", "Desert", "Park", "Garden",
    "Office", "Kitchen", "Bedroom", "Restaurant", "Airport", "Railway Station",
]

_model = None
_labels = None
_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def _load_places365_labels() -> list:
    """Download and cache Places365 category labels."""
    global _labels
    if _labels is not None:
        return _labels

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not LABELS_CACHE.exists():
        try:
            urllib.request.urlretrieve(PLACES365_LABEL_URL, LABELS_CACHE)
        except Exception:
            # Fallback: use a minimal embedded label list
            _labels = [f"scene_{i}" for i in range(365)]
            return _labels

    with open(LABELS_CACHE) as f:
        # Format: "/a/abbey 0"  ->  "abbey"
        raw = [line.strip().split(" ")[0] for line in f.readlines()]
        _labels = [r.split("/")[-1].replace("_", " ") for r in raw]

    return _labels


def _get_model():
    """Lazy-load ResNet50 pretrained on ImageNet as scene proxy."""
    global _model
    if _model is None:
        # We use a ResNet50 ImageNet model as a scene feature extractor
        # and apply softmax over ImageNet classes mapped to scene concepts.
        # For real Places365, weights require a separate download not via
        # torchvision hub, so we use the best available pretrained ResNet50.
        weights = tv_models.ResNet50_Weights.DEFAULT
        _model = tv_models.resnet50(weights=weights)
        _model.eval()
    return _model


# ImageNet classes that loosely map to scene categories for our proxy model
IMAGENET_SCENE_MAP: dict = {
    # index ranges / specific indices -> scene label
    # We'll use top-k predictions and map via text similarity instead
}

# Curated scene label vocabulary for our ImageNet-based proxy
SCENE_VOCAB = {
    "beach": ["seashore", "sandbar", "lakeside", "dock", "pier"],
    "forest": ["woodland", "valley", "cliff", "alp", "volcano"],
    "mountain": ["alp", "cliff", "valley", "volcano", "lakeside"],
    "kitchen": ["kitchen", "restaurant"],
    "office": ["library", "studio"],
    "bedroom": ["bedroom"],
    "street": ["street sign", "traffic light", "crossword puzzle"],
    "river": ["dam", "lakeside", "cliff"],
    "snow": ["ski", "snowmobile", "alp"],
    "desert": ["sandbar", "Arabian camel"],
}


def _match_to_user_category(scene_name: str) -> str:
    """Best-effort mapping of a scene name to user-visible categories."""
    sn = scene_name.lower()
    for cat, keywords in SCENE_GROUPS.items():
        for kw in keywords:
            if kw in sn:
                return cat
    return "Outdoor"  # safe default


def classify_scene(
    image: Image.Image,
    selected_categories: list,
) -> dict:
    """
    Classify the scene in *image* and return top predictions.

    Parameters
    ----------
    image               : PIL Image
    selected_categories : list of category strings or ["Detect Everything"]

    Returns
    -------
    dict with:
        predicted_scene   : str
        matched_category  : str
        top5              : list of {scene, confidence}
        inference_time_ms : float
    """
    t0 = time.perf_counter()

    model = _get_model()
    tensor = _transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    t1 = time.perf_counter()

    # Get ImageNet class names from the model weights
    weights = tv_models.ResNet50_Weights.DEFAULT
    imagenet_classes = weights.meta["categories"]

    top_k = 10
    top_probs, top_indices = torch.topk(probs, top_k)

    top_predictions = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        class_name = imagenet_classes[idx]
        top_predictions.append({
            "scene": class_name,
            "confidence": round(prob, 4),
        })

    # Map ImageNet predictions to scene vocabulary
    # Build a scene confidence aggregator
    detect_all = "Detect Everything" in selected_categories

    scene_scores: dict = {}
    for pred in top_predictions:
        cn = pred["scene"].lower()
        conf = pred["confidence"]
        # Heuristic mapping
        if any(w in cn for w in ["shore", "coast", "beach", "sand", "coral"]):
            scene_scores["Beach"] = scene_scores.get("Beach", 0) + conf
        if any(w in cn for w in ["forest", "wood", "tree", "fern", "jungle"]):
            scene_scores["Forest"] = scene_scores.get("Forest", 0) + conf
        if any(w in cn for w in ["mountain", "alp", "cliff", "valley", "peak"]):
            scene_scores["Mountains"] = scene_scores.get("Mountains", 0) + conf
        if any(w in cn for w in ["lake", "pond", "reservoir"]):
            scene_scores["Lake"] = scene_scores.get("Lake", 0) + conf
        if any(w in cn for w in ["snow", "glacier", "ski", "blizzard"]):
            scene_scores["Snow"] = scene_scores.get("Snow", 0) + conf
        if any(w in cn for w in ["desert", "dune", "camel", "sand"]):
            scene_scores["Desert"] = scene_scores.get("Desert", 0) + conf
        if any(w in cn for w in ["street", "traffic", "car", "bus", "road"]):
            scene_scores["Urban"] = scene_scores.get("Urban", 0) + conf
        if any(w in cn for w in ["kitchen", "oven", "stove", "refrigerator"]):
            scene_scores["Kitchen"] = scene_scores.get("Kitchen", 0) + conf
        if any(w in cn for w in ["book", "library", "desk", "laptop", "monitor"]):
            scene_scores["Office"] = scene_scores.get("Office", 0) + conf
        if any(w in cn for w in ["bed", "pillow", "blanket"]):
            scene_scores["Bedroom"] = scene_scores.get("Bedroom", 0) + conf
        if any(w in cn for w in ["restaurant", "cafe", "dining"]):
            scene_scores["Restaurant"] = scene_scores.get("Restaurant", 0) + conf
        if any(w in cn for w in ["garden", "flower", "park", "grass", "meadow"]):
            scene_scores["Park"] = scene_scores.get("Park", 0) + conf
            scene_scores["Garden"] = scene_scores.get("Garden", 0) + conf
        if any(w in cn for w in ["airplane", "airport", "terminal", "runway"]):
            scene_scores["Airport"] = scene_scores.get("Airport", 0) + conf
        if any(w in cn for w in ["train", "railway", "locomotive", "station"]):
            scene_scores["Railway Station"] = scene_scores.get("Railway Station", 0) + conf
        if any(w in cn for w in ["river", "creek", "stream", "waterfall"]):
            scene_scores["River"] = scene_scores.get("River", 0) + conf

    # Filter by selected categories
    if not detect_all:
        scene_scores = {k: v for k, v in scene_scores.items() if k in selected_categories}

    # If no scene matched, use top ImageNet class as fallback label
    if not scene_scores:
        best_imagenet = top_predictions[0]["scene"]
        matched_cat = "Outdoor"
        for cat in (selected_categories if not detect_all else USER_CATEGORIES):
            kws = SCENE_GROUPS.get(cat, [])
            if any(kw in best_imagenet.lower() for kw in kws):
                matched_cat = cat
                break
        total_conf = sum(p["confidence"] for p in top_predictions[:3])
        scene_scores = {matched_cat: total_conf}

    # Sort and build top-5
    sorted_scenes = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in sorted_scenes) or 1.0
    top5 = [
        {"scene": s, "confidence": round(v / total, 3)}
        for s, v in sorted_scenes[:5]
    ]

    predicted_scene = top5[0]["scene"] if top5 else "Unknown"

    return {
        "predicted_scene": predicted_scene,
        "matched_category": predicted_scene,
        "top5": top5,
        "raw_imagenet_top5": top_predictions[:5],
        "inference_time_ms": round((t1 - t0) * 1000, 1),
    }
