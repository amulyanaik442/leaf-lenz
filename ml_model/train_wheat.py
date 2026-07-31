"""
Train wheat disease model — MobileNetV2, balanced dataset.

Usage:
    python ml_model/train_wheat.py
"""
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
# Use temp dir to avoid OneDrive file locking issues
WHEAT_DATA = os.path.join(os.environ.get("TEMP", os.path.expanduser("~\\AppData\\Local\\Temp")), "wheat_data_merged")
TRAIN_DIR = os.path.join(WHEAT_DATA, "train")
VALID_DIR = os.path.join(WHEAT_DATA, "valid")
OUTPUT_DIR = os.path.join(BASE_DIR, "detector", "ml_assets", "wheat")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 8
INITIAL_LR = 0.001
FINE_TUNE_LR = 1e-5
FINE_TUNE_EPOCHS = 15
FINE_TUNE_AT = 160


def verify_dataset():
    if not os.path.exists(TRAIN_DIR):
        print(f"ERROR: Training directory not found at {TRAIN_DIR}")
        sys.exit(1)
    train_classes = sorted([d for d in os.listdir(TRAIN_DIR)
                            if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    print(f"Training classes ({len(train_classes)}): {train_classes}")
    for cls in train_classes:
        n = len([f for f in os.listdir(os.path.join(TRAIN_DIR, cls))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  {cls:25s} {n:5d}")
    return train_classes


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
    train_ds = train_ds.shuffle(200).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds


def build_augmentation_model():
    """Moderate augmentation — enough to regularize, not enough to destroy features."""
    return keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")


def build_model(num_classes, freeze_base=True):
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    augmentation = build_augmentation_model()
    x = augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet",
    )
    base_model.trainable = not freeze_base
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="wheat_disease_model")
    return model, base_model


def train_phase1(model, train_ds, val_ds, epochs, lr, class_weights=None):
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training head ({epochs} epochs, lr={lr})")
    print(f"{'='*60}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "best_wheat_model.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1,
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=callbacks, class_weight=class_weights,
    )
    return history


def train_phase2(model, base_model, train_ds, val_ds, epochs, lr, class_weights=None):
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tuning last {FINE_TUNE_AT} layers ({epochs} epochs, lr={lr})")
    print(f"{'='*60}")

    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_AT]:
        layer.trainable = False

    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"Trainable parameters: {trainable:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "best_wheat_model.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1,
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=callbacks, class_weight=class_weights,
    )
    return history


def main():
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices('GPU'))

    class_names = verify_dataset()
    num_classes = len(class_names)
    print(f"\nNumber of classes: {num_classes}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "wheat_class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    print("\nLoading datasets...")
    train_ds, val_ds = build_datasets(class_names)

    print("\nComputing class weights...")
    train_labels = []
    for _, labels in train_ds:
        train_labels.extend(np.argmax(labels.numpy(), axis=1))
    train_labels = np.array(train_labels)
    class_counts = collections.Counter(train_labels)
    total = len(train_labels)
    class_weights = {i: min(total / (num_classes * count), 3.0)
                     for i, count in class_counts.items()}
    print(f"Class weights: {class_weights}")

    print("\nBuilding MobileNetV2 model...")
    model, base_model = build_model(num_classes, freeze_base=True)
    model.summary()

    train_phase1(model, train_ds, val_ds, MAX_EPOCHS, INITIAL_LR, class_weights)

    best_path = os.path.join(OUTPUT_DIR, "best_wheat_model.keras")
    if os.path.exists(best_path):
        model = keras.models.load_model(best_path)
        print(f"\nLoaded best model from {best_path}")

    train_phase2(model, base_model, train_ds, val_ds,
                 FINE_TUNE_EPOCHS, FINE_TUNE_LR, class_weights)

    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON VALIDATION SET")
    print(f"{'='*60}")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")

    model.save(best_path)
    print(f"\nFinal model saved to {best_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
