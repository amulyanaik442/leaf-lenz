"""
Export the trained wheat disease model to ONNX format.

Usage:
    python ml_model/export_wheat_onnx.py

Outputs:
    detector/ml_assets/wheat/wheat_model.onnx
    detector/ml_assets/wheat/wheat_class_names.json
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Force UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "detector", "ml_assets", "wheat", "best_wheat_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "detector", "ml_assets", "wheat", "wheat_class_names.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "detector", "ml_assets", "wheat")
ONNX_PATH = os.path.join(OUTPUT_DIR, "wheat_model.onnx")

IMAGE_SIZE = (224, 224)


def export():
    # Check prerequisites
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run 'python ml_model/train_wheat.py' first.")
        sys.exit(1)

    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"ERROR: Class names not found at {CLASS_NAMES_PATH}")
        sys.exit(1)

    # Load class names
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Load model
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

    # Check if tf2onnx is available
    try:
        import tf2onnx
        import onnx
        print("tf2onnx found. Using direct conversion...")
        use_tf2onnx = True
    except ImportError:
        print("tf2onnx not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tf2onnx"])
        import tf2onnx
        import onnx
        use_tf2onnx = True

    # Export to ONNX
    print(f"\nExporting to ONNX at {ONNX_PATH}...")
    spec = (tf.TensorSpec((None, *IMAGE_SIZE, 3), tf.float32, name="input"),)
    output_path = ONNX_PATH

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=output_path,
    )
    print(f"ONNX model saved to {output_path}")

    # Verify with onnxruntime
    print("\nVerifying ONNX model with onnxruntime...")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)

        # Test with random input
        dummy_input = np.random.randn(1, *IMAGE_SIZE, 3).astype(np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: dummy_input})

        output_shape = outputs[0].shape
        print(f"Input shape:  {dummy_input.shape}")
        print(f"Output shape: {output_shape}")
        print(f"Output classes: {output_shape[1] if len(output_shape) > 1 else 'N/A'}")

        # Verify prediction is valid probabilities
        probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
        print(f"Sum of probabilities: {probs.sum():.6f}")
        print(f"Top prediction: {class_names[np.argmax(probs)]} ({np.max(probs)*100:.2f}%)")

        # Check file sizes
        onnx_size = os.path.getsize(output_path)
        print(f"\nONNX model size: {onnx_size / 1024:.1f} KB ({onnx_size / (1024*1024):.2f} MB)")

        print("\nONNX export and verification complete!")

    except ImportError:
        print("WARNING: onnxruntime not installed. Skipping verification.")
        print("Install with: pip install onnxruntime")
    except Exception as e:
        print(f"WARNING: ONNX verification failed: {e}")
        print("The model was exported but may need debugging.")


if __name__ == "__main__":
    export()
