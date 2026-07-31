"""
Export the trained maize disease model to ONNX format.

Usage:
    python ml_model/export_maize_onnx.py

Outputs:
    detector/ml_assets/maize/maize_model.onnx
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "detector", "ml_assets", "maize", "best_maize_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "detector", "ml_assets", "maize", "maize_class_names.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "detector", "ml_assets", "maize")
ONNX_PATH = os.path.join(OUTPUT_DIR, "maize_model.onnx")

IMAGE_SIZE = (224, 224)


def export():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    print(f"\nLoading model from {MODEL_PATH}...")

    class FocalCategoricalCrossentropy(keras.losses.Loss):
        def __init__(self, gamma=2.0, label_smoothing=0.1, **kwargs):
            super().__init__(**kwargs)
            self.gamma = gamma
            self.label_smoothing = label_smoothing
        def call(self, y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            n_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true_smooth = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / n_classes
            cross_entropy = -y_true_smooth * tf.math.log(y_pred)
            weight = y_true_smooth * tf.pow(1.0 - y_pred, self.gamma)
            loss = weight * cross_entropy
            return tf.reduce_sum(loss, axis=-1)

    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
        'FocalCategoricalCrossentropy': FocalCategoricalCrossentropy
    })
    print("Model loaded.")

    try:
        import tf2onnx
        import onnx
    except ImportError:
        print("Installing tf2onnx...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tf2onnx"])
        import tf2onnx
        import onnx

    print(f"\nExporting to ONNX at {ONNX_PATH}...")
    spec = (tf.TensorSpec((None, *IMAGE_SIZE, 3), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=13, output_path=ONNX_PATH,
    )
    print(f"ONNX model saved to {ONNX_PATH}")

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(ONNX_PATH)
        dummy_input = np.random.randn(1, *IMAGE_SIZE, 3).astype(np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: dummy_input})
        output_shape = outputs[0].shape
        print(f"Input: {dummy_input.shape} -> Output: {output_shape}")
        probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
        print(f"Top: {class_names[np.argmax(probs)]} ({np.max(probs)*100:.2f}%)")
        onnx_size = os.path.getsize(ONNX_PATH)
        print(f"ONNX size: {onnx_size / (1024*1024):.2f} MB")
        print("Export + verification complete!")
    except Exception as e:
        print(f"WARNING: Verification failed: {e}")


if __name__ == "__main__":
    export()
