"""
Prepare the maize dataset into train/valid/test splits.

Usage:
    python ml_model/prepare_maize_dataset.py

Reads raw images from ml_model/maize_data/ and creates:
    ml_model/maize_data/train/   (70% of images per class)
    ml_model/maize_data/valid/   (15% of images per class)
    ml_model/maize_data/test/    (15% of images per class)
"""
import os
import sys
import shutil
import random

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "ml_model", "maize_data")
SPLITS = {"train": 0.70, "valid": 0.15, "test": 0.15}
SEED = 42

# Normalize class names to title case with underscores
CLASS_NAME_MAP = {
    "Common_Rust": "Common Rust",
    "Corn_Common_rust": "Common Rust",
    "Gray_Leaf_Spot": "Gray Leaf Spot",
    "Corn_Gray_leaf_spot": "Gray Leaf Spot",
    "Healthy": "Healthy",
    "Corn_Healthy": "Healthy",
    "Northern_Leaf_Blight": "Northern Leaf Blight",
    "Corn_Northern_leaf_blight": "Northern Leaf Blight",
    "Blight": "Northern Leaf Blight",
}


def main():
    print("=" * 60)
    print("  PREPARING MAIZE DATASET")
    print("=" * 60)

    # Find class directories in raw data
    raw_classes = {}
    for item in os.listdir(RAW_DIR):
        item_path = os.path.join(RAW_DIR, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            images = [f for f in os.listdir(item_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                normalized = CLASS_NAME_MAP.get(item, item)
                if normalized not in raw_classes:
                    raw_classes[normalized] = item_path
                print(f"  Found: {item} -> {normalized} ({len(images)} images)")

    if not raw_classes:
        print("ERROR: No class directories found in maize_data/")
        print(f"Contents of {RAW_DIR}: {os.listdir(RAW_DIR)}")
        sys.exit(1)

    print(f"\nClasses ({len(raw_classes)}): {list(raw_classes.keys())}")

    # Remove old splits if they exist
    for split in SPLITS:
        split_dir = os.path.join(RAW_DIR, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    random.seed(SEED)

    # Create splits
    for cls_name, src_dir in raw_classes.items():
        images = [f for f in os.listdir(src_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLITS["train"])
        n_valid = int(n * SPLITS["valid"])

        splits_data = {
            "train": images[:n_train],
            "valid": images[n_train:n_train + n_valid],
            "test": images[n_train + n_valid:],
        }

        for split, files in splits_data.items():
            split_dir = os.path.join(RAW_DIR, split, cls_name)
            os.makedirs(split_dir, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(split_dir, f))

        print(f"  {cls_name:25s}: {n_train} train, {n_valid} valid, {len(splits_data['test'])} test")

    # Summary
    print(f"\n{'='*60}")
    print("  DATASET SUMMARY")
    print(f"{'='*60}")
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(RAW_DIR, split)
        total = 0
        for cls in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls)
            if os.path.isdir(cls_path):
                n = len([f for f in os.listdir(cls_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                total += n
        print(f"  {split:6s}: {total:5d} images")

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
