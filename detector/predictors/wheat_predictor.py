"""
Wheat Disease Predictor — independent ONNX inference module.

Loads a wheat-specific ONNX model and class names, runs inference on leaf
images, and returns predictions. This module is completely separate from
the main inference.py and can be extended for other crops later.

Usage (standalone):
    from detector.predictors.wheat_predictor import predict_wheat_disease
    label, confidence, top5 = predict_wheat_disease(image_file)

Usage (in Django views):
    Already integrated via inference.py -> predict_with_wheat_fallback()
"""
import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
from django.conf import settings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_WHEAT_ASSETS_DIR = os.path.join(settings.BASE_DIR, 'detector', 'ml_assets', 'wheat')
_WHEAT_MODEL_PATH = os.path.join(_WHEAT_ASSETS_DIR, 'wheat_model.onnx')
_WHEAT_CLASSES_PATH = os.path.join(_WHEAT_ASSETS_DIR, 'wheat_class_names.json')

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_wheat_session = None
_wheat_class_names = None
_wheat_loaded = False


def load_wheat_assets():
    """Load the wheat ONNX model and class names into memory. Cached after first call."""
    global _wheat_session, _wheat_class_names, _wheat_loaded

    if _wheat_loaded:
        return True

    if not os.path.exists(_WHEAT_MODEL_PATH) or not os.path.exists(_WHEAT_CLASSES_PATH):
        print("WARNING: Wheat model or class names not found in detector/ml_assets/wheat/.")
        print("Wheat-specific predictions will not be available.")
        return False

    try:
        _wheat_session = ort.InferenceSession(_WHEAT_MODEL_PATH)
        with open(_WHEAT_CLASSES_PATH, 'r') as f:
            _wheat_class_names = json.load(f)
        _wheat_loaded = True
        print(f"Wheat model loaded successfully ({len(_wheat_class_names)} classes).")
        return True
    except Exception as e:
        print(f"Error loading wheat ONNX model: {e}")
        return False


# ---------------------------------------------------------------------------
def preprocess_wheat_image(image_file, target_size=224):
    """
    Load an image from a file-like object, resize to a larger canvas,
    then return the raw [0, 255] pixel array in NHWC format.

    The ONNX model already includes preprocess_input in its graph,
    so we feed raw [0, 255] pixel values.
    """
    img = Image.open(image_file).convert('RGB')
    img = img.resize((256, 256), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return arr


def _five_crops(img_np, size=224):
    """Return centre crop + 4 corner crops of a (256,256) image."""
    h, w = img_np.shape[:2]
    d = size
    crops = [
        img_np[(h-d)//2:(h+d)//2, (w-d)//2:(w+d)//2],
        img_np[0:d, 0:d],
        img_np[0:d, w-d:w],
        img_np[h-d:h, 0:d],
        img_np[h-d:h, w-d:w],
    ]
    return crops


def _format_label(raw_class_name):
    """
    Convert a raw folder name to the project's label convention.

    Raw folder name (e.g. "Brown Rust") -> "Wheat___Brown_Rust"

    This matches the format used by the existing model so that
    parse_class_label() in views.py works without modification.
    """
    sanitized = raw_class_name.replace(" ", "_")
    return f"Wheat___{sanitized}"


def predict_wheat_disease(image_file):
    """
    Predict wheat disease from a leaf image.

    Args:
        image_file: A file-like object (e.g. Django UploadedFile or open file)

    Returns:
        (predicted_label: str, confidence: float, top5: list[(label, conf)])

        Label format: "Wheat___Class_Name" (e.g. "Wheat___Brown_Rust")
        This matches the existing model's label convention.
    """
    loaded = load_wheat_assets()
    if not loaded:
        return None, 0.0, []

    try:
        arr = preprocess_wheat_image(image_file, target_size=256)
        input_name = _wheat_session.get_inputs()[0].name

        prob_sum = None
        for crop in _five_crops(arr, size=224):
            tensor = np.expand_dims(crop, axis=0).astype(np.float32)
            logits = _wheat_session.run(None, {input_name: tensor})[0][0]
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)
            prob_sum = probs if prob_sum is None else prob_sum + probs

        avg_probs = prob_sum / 5.0

        top5_indices = np.argsort(avg_probs)[::-1][:5]
        top5 = [
            (_format_label(_wheat_class_names[i]), float(avg_probs[i]))
            for i in top5_indices
        ]

        predicted_label = top5[0][0]
        confidence = top5[0][1]

        return predicted_label, confidence, top5

    except Exception as e:
        print(f"Error during wheat inference: {e}")
        return None, 0.0, []
