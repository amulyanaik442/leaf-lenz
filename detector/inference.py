import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
from django.conf import settings

# Paths to ML assets
ML_ASSETS_DIR = os.path.join(settings.BASE_DIR, 'detector', 'ml_assets')
MODEL_PATH = os.path.join(ML_ASSETS_DIR, 'model.onnx')
CLASSES_PATH = os.path.join(ML_ASSETS_DIR, 'class_names.json')

# Global variables for caching model and classes in memory
_ort_session = None
_class_names = None
_model_loaded = False

def load_inference_assets():
    global _ort_session, _class_names, _model_loaded
    if _model_loaded:
        return True

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        print("WARNING: ONNX model or class names JSON not found in detector/ml_assets/.")
        print("API will run in Mock Inference Mode.")
        return False

    try:
        _ort_session = ort.InferenceSession(MODEL_PATH)
        with open(CLASSES_PATH, 'r') as f:
            _class_names = json.load(f)
        _model_loaded = True
        print("ONNX model and class names loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return False

# Attempt to load assets on import
assets_loaded = load_inference_assets()

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _normalize(arr):
    """Normalize a (H, W, 3) float32 array in [0,1] with ImageNet stats."""
    return (arr - _MEAN) / _STD

def _to_tensor(arr):
    """HWC → CHW → add batch dim → (1, 3, H, W)."""
    return np.expand_dims(arr.transpose(2, 0, 1), axis=0)

def _augmentations(arr):
    """Yield 4 augmented versions: original, hflip, vflip, hvflip."""
    yield arr
    yield arr[:, ::-1, :].copy()
    yield arr[::-1, :, :].copy()
    yield arr[::-1, ::-1, :].copy()

def preprocess_image(image_file):
    """
    Load image from file-like object, resize to 224×224 with BICUBIC,
    convert to float32 in [0,1], and return a (224, 224, 3) numpy array.
    """
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224), Image.BICUBIC)
    return np.array(img, dtype=np.float32) / 255.0

def _run_batch(tensors):
    """
    Run a list of (1, 3, 224, 224) tensors through the ONNX session.
    Returns the averaged softmax probability vector.
    """
    input_name = _ort_session.get_inputs()[0].name
    prob_sum = None
    for t in tensors:
        logits = _ort_session.run(None, {input_name: t})[0][0]
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        prob_sum = probs if prob_sum is None else prob_sum + probs
    return prob_sum / len(tensors)

def predict_leaf_disease(image_file):
    """
    Predicts the leaf type and disease using Test-Time Augmentation (TTA).

    TTA strategy (4 views):
      - 4 augmentations (original, hflip, vflip, hvflip)
      Uses BICUBIC resize to 224×224 (matches training pipeline).
      All softmax probability vectors are averaged before taking argmax.

    Returns:
        predicted_label (str), confidence (float), top5 (list of (label, conf))
    """
    is_loaded = load_inference_assets()

    if not is_loaded:
        return "Tomato___Early_blight", 0.965, [("Tomato___Early_blight", 0.965)]

    try:
        arr = preprocess_image(image_file)  # (224,224,3) bicubic, matching training

        # Build 4 tensors: 4 augmentations (no 5-crop — matches training resize)
        tensors = []
        for aug in _augmentations(arr):
            norm = _normalize(aug)
            tensors.append(_to_tensor(norm).astype(np.float32))

        # Average all 4 probability vectors
        avg_probs = _run_batch(tensors)

        # Top-5 predictions
        top5_indices = np.argsort(avg_probs)[::-1][:5]
        top5 = [(str(_class_names[i]), float(avg_probs[i])) for i in top5_indices]

        predicted_label = top5[0][0]
        confidence = top5[0][1]

        return predicted_label, confidence, top5

    except Exception as e:
        print(f"Error during ONNX inference: {e}")
        return "fallback", 0.0, [("fallback", 0.0)]


# ---------------------------------------------------------------------------
# Crop-specific model integration
# ---------------------------------------------------------------------------
# Add new crop predictors here as they are developed.
# Each entry: (crop_name, predictor_function, confidence_threshold)
# The predictor_function must accept (image_file) and return (label, conf, top5).

WHEAT_CONFIDENCE_THRESHOLD = 0.10
MAIZE_CONFIDENCE_THRESHOLD = 0.50


_wheat_predictor_func = None
_wheat_predictor_loaded = False

def _load_wheat_predictor():
    """Lazy-load the wheat predictor once, then cache it."""
    global _wheat_predictor_func, _wheat_predictor_loaded
    if _wheat_predictor_loaded:
        return _wheat_predictor_func
    try:
        from detector.predictors.wheat_predictor import predict_wheat_disease
        _wheat_predictor_func = predict_wheat_disease
        _wheat_predictor_loaded = True
    except Exception as e:
        print(f"Could not load wheat predictor: {e}")
        _wheat_predictor_func = None
        _wheat_predictor_loaded = True
    return _wheat_predictor_func


_maize_predictor_func = None
_maize_predictor_loaded = False

def _load_maize_predictor():
    """Lazy-load the maize predictor once, then cache it."""
    global _maize_predictor_func, _maize_predictor_loaded
    if _maize_predictor_loaded:
        return _maize_predictor_func
    try:
        from detector.predictors.maize_predictor import predict_maize_disease
        _maize_predictor_func = predict_maize_disease
        _maize_predictor_loaded = True
    except Exception as e:
        print(f"Could not load maize predictor: {e}")
        _maize_predictor_func = None
        _maize_predictor_loaded = True
    return _maize_predictor_func


def predict_with_wheat_fallback(image_file):
    """
    Run crop-specific models first, then fall back to the general model.

    The general model has NO wheat classes (only Corn___ classes for maize).
    Routing a wheat image to the general model always produces wrong results.

    Routing logic:
      1. Try wheat model first (if confidence >= 10%, use it).
         The general model has NO wheat classes, so keep wheat routing aggressive.
      2. Try maize model second (if confidence >= 15%, use it).
         The general model has Corn___ classes as a reasonable fallback.
      3. Fall back to general model for everything else.

    Returns:
        (predicted_label, confidence, top5) -- same format as predict_leaf_disease()
    """
    # 1. Try wheat model first
    wheat_predictor = _load_wheat_predictor()
    if wheat_predictor is not None:
        try:
            if hasattr(image_file, 'seek'):
                image_file.seek(0)

            wheat_label, wheat_conf, wheat_top5 = wheat_predictor(image_file)

            if wheat_label is not None and wheat_conf >= WHEAT_CONFIDENCE_THRESHOLD:
                print(f"[CROP ROUTING] Wheat model: {wheat_label} ({wheat_conf:.4f}) -> USING WHEAT")
                return wheat_label, wheat_conf, wheat_top5
            else:
                print(f"[CROP ROUTING] Wheat model: {wheat_label} ({wheat_conf:.4f}) -- low confidence")
        except Exception as e:
            print(f"[CROP ROUTING] Wheat predictor error: {e}")

    # 2. Try maize model second
    maize_predictor = _load_maize_predictor()
    if maize_predictor is not None:
        try:
            if hasattr(image_file, 'seek'):
                image_file.seek(0)

            maize_label, maize_conf, maize_top5 = maize_predictor(image_file)

            if maize_label is not None and maize_conf >= MAIZE_CONFIDENCE_THRESHOLD:
                print(f"[CROP ROUTING] Maize model: {maize_label} ({maize_conf:.4f}) -> USING MAIZE")
                return maize_label, maize_conf, maize_top5
            else:
                print(f"[CROP ROUTING] Maize model: {maize_label} ({maize_conf:.4f}) -- low confidence, trying general")
        except Exception as e:
            print(f"[CROP ROUTING] Maize predictor error: {e}")

    # 3. Fall back to general model
    main_label, main_conf, main_top5 = predict_leaf_disease(image_file)
    print(f"[CROP ROUTING] General model: {main_label} ({main_conf:.4f}) -> USING GENERAL")
    return main_label, main_conf, main_top5


# ---------------------------------------------------------------------------
# Crop-aware prediction (user selects crop)
# ---------------------------------------------------------------------------
# Maps frontend crop value -> class prefix filter for the general model
CROP_CLASS_PREFIXES = {
    'groundnut': 'Groundnut___',
    'pumpkin': 'Pumpkin___',
    'apple': 'Apple___',
    'bell_pepper': 'Bell_pepper___',
    'blueberry': 'Blueberry___',
    'cassava': 'Cassava___',
    'cherry': 'Cherry___',
    'coffee': 'Coffee___',
    'cotton': 'Cotton___',
    'grape': 'Grape___',

    'mango': 'Mango___',
    'orange': 'Orange___',
    'peach': 'Peach___',
    'potato': 'Potato___',
    'raspberry': 'Raspberry___',
    'rice': 'Rice___',
    'soybean': 'Soybean___',
    'squash': 'Squash___',
    'strawberry': 'Strawberry___',
    'sugarcane': ('Sugercane___', 'Sugarcane___'),
    'tomato': 'Tomato___',
    'watermelon': 'Watermelon___',
}


def _predict_general_filtered(image_file, crop_key):
    """
    Run general model TTA, then zero-out probabilities for classes
    not matching the selected crop. Returns only classes for that crop.
    """
    is_loaded = load_inference_assets()
    if not is_loaded:
        return "fallback", 0.0, [("fallback", 0.0)]

    try:
        arr = preprocess_image(image_file)  # (224,224,3) bicubic

        tensors = []
        for aug in _augmentations(arr):
            norm = _normalize(aug)
            tensors.append(_to_tensor(norm).astype(np.float32))

        avg_probs = _run_batch(tensors)

        prefix = CROP_CLASS_PREFIXES.get(crop_key, '')
        if prefix:
            if isinstance(prefix, tuple):
                match_fn = lambda cn: any(cn.startswith(p) for p in prefix)
            else:
                match_fn = lambda cn: cn.startswith(prefix)
            # Zero out all classes that don't match this crop
            filtered = np.array([
                p if match_fn(_class_names[i]) else 0.0
                for i, p in enumerate(avg_probs)
            ])
            total = filtered.sum()
            if total > 0:
                filtered = filtered / total
            else:
                # No matching classes found — fall back to full probs
                filtered = avg_probs
        else:
            filtered = avg_probs

        top5_indices = np.argsort(filtered)[::-1][:5]
        top5 = [(str(_class_names[i]), float(filtered[i])) for i in top5_indices]
        predicted_label = top5[0][0]
        confidence = top5[0][1]
        return predicted_label, confidence, top5

    except Exception as e:
        print(f"Error during filtered inference: {e}")
        return "fallback", 0.0, [("fallback", 0.0)]


def predict_by_crop(image_file, crop='auto'):
    """
    Predict using the model best suited for the selected crop.

    Returns (label, confidence, top5, crop_mismatch).

    crop='auto'  -> use routing logic (wheat -> maize -> general)
    crop='wheat' -> use wheat specialist model directly
    crop='maize' -> use maize specialist model directly
    crop=other   -> use general model filtered to that crop's classes only
    """
    if crop == 'auto':
        label, conf, top5 = predict_with_wheat_fallback(image_file)
        return label, conf, top5, None

    if crop == 'wheat':
        wheat_predictor = _load_wheat_predictor()
        if wheat_predictor is not None:
            if hasattr(image_file, 'seek'):
                image_file.seek(0)
            label, conf, top5 = wheat_predictor(image_file)
            print(f"[CROP SELECT] Wheat model: {label} ({conf:.4f})")
            return label, conf, top5, None
        label, conf, top5 = predict_with_wheat_fallback(image_file)
        return label, conf, top5, None

    if crop == 'maize':
        maize_predictor = _load_maize_predictor()
        if maize_predictor is not None:
            if hasattr(image_file, 'seek'):
                image_file.seek(0)
            label, conf, top5 = maize_predictor(image_file)
            print(f"[CROP SELECT] Maize model: {label} ({conf:.4f})")
            return label, conf, top5, None
        label, conf, top5 = predict_with_wheat_fallback(image_file)
        return label, conf, top5, None

    # General model with crop-specific filtering
    if hasattr(image_file, 'seek'):
        image_file.seek(0)
    label, conf, top5 = _predict_general_filtered(image_file, crop)
    print(f"[CROP SELECT] General model ({crop}): {label} ({conf:.4f})")
    return label, conf, top5, None


def _detect_crop_from_label(label):
    """Extract crop name from a class label like 'Apple___rust' -> 'apple'."""
    if not label or label == 'fallback':
        return None
    for prefix_key in CROP_CLASS_PREFIXES:
        prefix = CROP_CLASS_PREFIXES[prefix_key]
        if isinstance(prefix, tuple):
            if any(label.startswith(p) for p in prefix):
                return prefix_key
        else:
            if label.startswith(prefix):
                return prefix_key
    # Special cases for specialist models
    if label.startswith('Wheat___'):
        return 'wheat'
    if label.startswith('Maize___'):
        return 'maize'
    return None
