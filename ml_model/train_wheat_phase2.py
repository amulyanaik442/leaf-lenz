"""Phase 2 fine-tuning only - loads best wheat model and fine-tunes."""
import os
import sys
import json
import collections
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHEAT_DATA = os.path.join(os.environ.get("TEMP", os.path.expanduser("~\\AppData\\Local\\Temp")), "wheat_data")
TRAIN_DIR = os.path.join(WHEAT_DATA, "train")
VALID_DIR = os.path.join(WHEAT_DATA, "valid")
OUTPUT_DIR = os.path.join(BASE_DIR, "detector", "ml_assets", "wheat")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_wheat_model.keras")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
FINE_TUNE_LR = 1e-5
FINE_TUNE_EPOCHS = 50
FINE_TUNE_AT = 160
PATIENCE = 12


def build_datasets(class_names):
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels="inferred", label_mode="categorical",
        class_names=class_names, image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, shuffle=True, seed=42,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        VALID_DIR, labels="inferred", label_mode="categorical",
        class_names=class_names, image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, shuffle=False,
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(500).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds


def compute_class_weights(train_ds, num_classes):
    train_labels = []
    for _, labels in train_ds:
        train_labels.extend(np.argmax(labels.numpy(), axis=1))
    train_labels = np.array(train_labels)
    class_counts = collections.Counter(train_labels)
    total = len(train_labels)
    return {i: min(total / (num_classes * count), 3.0)
            for i, count in class_counts.items()}


def main():
    print("TensorFlow version:", tf.__version__)

    # Load class names from dataset directories
    train_classes = sorted([d for d in os.listdir(TRAIN_DIR)
                            if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    num_classes = len(train_classes)
    print(f"Classes ({num_classes}): {train_classes}")

    # Save class names
    with open(os.path.join(OUTPUT_DIR, "wheat_class_names.json"), "w") as f:
        json.dump(train_classes, f, indent=2)

    train_ds, val_ds = build_datasets(train_classes)
    class_weights = compute_class_weights(train_ds, num_classes)
    print(f"Class weights computed.")

    print(f"\nLoading best model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    # Unfreeze base model for fine-tuning
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            base_model = layer
            break
    
    if base_model is None:
        # Try finding MobileNetV2 by name
        for layer in model.layers:
            if 'mobilenet' in layer.name.lower():
                base_model = layer
                break

    if base_model is None:
        print("ERROR: Could not find base model to fine-tune")
        sys.exit(1)

    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_AT]:
        layer.trainable = False

    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"Trainable parameters: {trainable:,}")

    print(f"\nPHASE 2: Fine-tuning last {FINE_TUNE_AT} layers ({FINE_TUNE_EPOCHS} epochs, lr={FINE_TUNE_LR})")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1,
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks, class_weight=class_weights,
    )

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
