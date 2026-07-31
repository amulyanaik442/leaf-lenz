"""
Maize Disease Predictor — independent ONNX inference module.

Loads a maize-specific ONNX model and class names, runs inference on leaf
images, and returns predictions. This module is completely separate from
the main inference.py and the wheat predictor.

Uses temperature scaling (T=0.7) to sharpen softmax outputs for better
calibration on the 4-class problem.

Usage (standalone):
    from detector.predictors.maize_predictor import predict_maize_disease
    label, confidence, top5 = predict_maize_disease(image_file)

Usage (in Django views):
    Already integrated via inference.py -> predict_with_maize_fallback()
"""
import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
from django.conf import settings

_MAIZE_ASSETS_DIR = os.path.join(settings.BASE_DIR, 'detector', 'ml_assets', 'maize')
_MAIZE_MODEL_PATH = os.path.join(_MAIZE_ASSETS_DIR, 'maize_model.onnx')
_MAIZE_CLASSES_PATH = os.path.join(_MAIZE_ASSETS_DIR, 'maize_class_names.json')

_maize_session = None
_maize_class_names = None
_maize_loaded = False

# Temperature scaling factor — <1 sharpens distribution, >1 softens it.
# 0.7 gives a nice boost to the top class for 4-way maize classification.
_MAIZE_TEMPERATURE = 0.7


def load_maize_assets():
    global _maize_session, _maize_class_names, _maize_loaded
    if _maize_loaded:
        return True

    if not os.path.exists(_MAIZE_MODEL_PATH) or not os.path.exists(_MAIZE_CLASSES_PATH):
        print("WARNING: Maize model or class names not found in detector/ml_assets/maize/.")
        return False

    try:
        _maize_session = ort.InferenceSession(_MAIZE_MODEL_PATH)
        with open(_MAIZE_CLASSES_PATH, 'r') as f:
            _maize_class_names = json.load(f)
        _maize_loaded = True
        print(f"Maize model loaded successfully ({len(_maize_class_names)} classes).")
        return True
    except Exception as e:
        print(f"Error loading maize ONNX model: {e}")
        return False


def preprocess_maize_image(image_file, target_size=256):
    """
    Load an image from a file-like object, resize to target_size canvas.

    The ONNX model includes ShuffleNetV2 preprocessing in its graph,
    so we feed raw [0, 255] pixel values.
    """
    img = Image.open(image_file).convert('RGB')
    img = img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return arr


def _five_crops(img_np, size=224):
    """Return centre crop + 4 corner crops."""
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
    """Convert a raw folder name to the project's label convention."""
    sanitized = raw_class_name.replace(" ", "_")
    return f"Maize___{sanitized}"


def predict_maize_disease(image_file):
    """
    Predict maize disease from a leaf image.

    Returns:
        (predicted_label: str, confidence: float, top5: list[(label, conf)])
        Label format: "Maize___Class_Name"
    """
    loaded = load_maize_assets()
    if not loaded:
        return None, 0.0, []

    try:
        arr = preprocess_maize_image(image_file, target_size=256)
        input_name = _maize_session.get_inputs()[0].name

        prob_sum = None
        for crop in _five_crops(arr, size=224):
            tensor = np.expand_dims(crop, axis=0).astype(np.float32)
            logits = _maize_session.run(None, {input_name: tensor})[0][0]
            scaled_logits = logits / _MAIZE_TEMPERATURE
            exp_l = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_l / np.sum(exp_l)
            prob_sum = probs if prob_sum is None else prob_sum + probs

        avg_probs = prob_sum / 5.0

        top5_indices = np.argsort(avg_probs)[::-1][:5]
        top5 = [
            (_format_label(_maize_class_names[i]), float(avg_probs[i]))
            for i in top5_indices
        ]

        predicted_label = top5[0][0]
        confidence = top5[0][1]

        return predicted_label, confidence, top5

    except Exception as e:
        print(f"Error during maize inference: {e}")
        return None, 0.0, []
