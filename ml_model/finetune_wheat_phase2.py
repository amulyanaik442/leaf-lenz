"""Phase 2 fine-tuning only — loads best checkpoint and fine-tunes the base."""
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
BEST_PATH = os.path.join(OUTPUT_DIR, "best_wheat_model.keras")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
FINE_TUNE_LR = 5e-5
FINE_TUNE_EPOCHS = 40
FINE_TUNE_AT = 100
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
    train_ds = train_ds.shuffle(200).prefetch(AUTOTUNE)
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
    if not os.path.exists(BEST_PATH):
        print(f"ERROR: No best model found at {BEST_PATH}")
        sys.exit(1)

    with open(os.path.join(OUTPUT_DIR, "wheat_class_names.json"), "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    train_ds, val_ds = build_datasets(class_names)
    class_weights = compute_class_weights(train_ds, num_classes)

    print(f"\nLoading best checkpoint from {BEST_PATH}...")
    model = keras.models.load_model(BEST_PATH)
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            base_model = layer
            break
    if base_model is None:
        for layer in model.layers:
            if 'mobilenet' in layer.name.lower() or 'base' in layer.name.lower():
                base_model = layer
                break

    if base_model is None:
        print("WARNING: Could not find base model, trying all layers")
        base_model = model.layers[3] if len(model.layers) > 3 else None

    if base_model:
        print(f"Base model: {base_model.name}, total layers: {len(base_model.layers)}")
        base_model.trainable = True
        for layer in base_model.layers[:-FINE_TUNE_AT]:
            layer.trainable = False
        trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        print(f"Trainable parameters: {trainable:,}")
    else:
        print("ERROR: Could not locate base model")
        sys.exit(1)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nCheckpoint val_accuracy before fine-tuning: {val_acc*100:.2f}%")

    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tuning last {FINE_TUNE_AT} layers ({FINE_TUNE_EPOCHS} epochs, lr={FINE_TUNE_LR})")
    print(f"{'='*60}")

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
            BEST_PATH, monitor="val_accuracy", save_best_only=True, verbose=1,
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

    model.save(BEST_PATH)
    print(f"\nModel saved to {BEST_PATH}")
    print("Phase 2 fine-tuning complete.")


if __name__ == "__main__":
    main()
