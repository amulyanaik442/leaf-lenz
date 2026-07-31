"""
Balance the maize dataset by undersampling all classes to the smallest class count.
Also resplits to 80/10/10 as required.

Current distribution:
  Common Rust:      1306
  Gray Leaf Spot:    574  <-- smallest
  Healthy:          1162
  Northern Leaf Blight: 1146

Target: 574 per class, split 80/10/10.

Usage:
    python ml_model/balance_maize_dataset.py
"""
import os
import sys
import random
import shutil

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIZE_DATA = os.path.join(BASE_DIR, "ml_model", "maize_data")
SEED = 42
SPLITS = {"train": 0.80, "valid": 0.10, "test": 0.10}


def main():
    print("=" * 60)
    print("  BALANCING MAIZE DATASET (UNDERSAMPLE + 80/10/10 SPLIT)")
    print("=" * 60)

    # Step 1: Collect ALL images per class (merge splits)
    raw_classes = {}
    for cls in os.listdir(MAIZE_DATA):
        cls_path = os.path.join(MAIZE_DATA, cls)
        if os.path.isdir(cls_path) and cls not in ("train", "valid", "test"):
            imgs = [f for f in os.listdir(cls_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            raw_classes[cls] = [os.path.join(cls_path, f) for f in imgs]
            print(f"  {cls:30s} {len(imgs):5d} images (raw)")

    if not raw_classes:
        print("ERROR: No raw class directories found.")
        sys.exit(1)

    min_count = min(len(v) for v in raw_classes.values())
    print(f"\nSmallest class: {min_count} images")
    print(f"Target per class: {min_count} (undersample all to this)")

    # Step 2: Undersample each class to min_count
    random.seed(SEED)
    balanced = {}
    for cls, paths in raw_classes.items():
        chosen = random.sample(paths, min_count)
        balanced[cls] = chosen
        print(f"  {cls:30s} {len(paths):5d} -> {len(chosen):5d}")

    # Step 3: Remove old splits
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(MAIZE_DATA, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
            print(f"  Removed {split_dir}")

    # Step 4: Create new 80/10/10 splits
    for cls, paths in balanced.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * SPLITS["train"])
        n_valid = int(n * SPLITS["valid"])

        splits = {
            "train": paths[:n_train],
            "valid": paths[n_train:n_train + n_valid],
            "test": paths[n_train + n_valid:],
        }

        for split, files in splits.items():
            split_dir = os.path.join(MAIZE_DATA, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for f in files:
                fname = os.path.basename(f)
                dest = os.path.join(split_dir, fname)
                if f != dest:
                    shutil.copy2(f, dest)

        print(f"  {cls:30s}: {len(splits['train'])} train, {len(splits['valid'])} valid, {len(splits['test'])} test")

    # Step 5: Verify final distribution
    print(f"\n{'='*60}")
    print("  FINAL DISTRIBUTION (80/10/10)")
    print(f"{'='*60}")
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(MAIZE_DATA, split)
        total = 0
        print(f"\n  {split.upper()}:")
        for cls in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls)
            if os.path.isdir(cls_path):
                n = len([f for f in os.listdir(cls_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                total += n
                print(f"    {cls:30s} {n:5d}")
        print(f"    {'TOTAL':30s} {total:5d}")

    # Final ratio check
    train_dir = os.path.join(MAIZE_DATA, "train")
    counts = {}
    for cls in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len([f for f in os.listdir(cls_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    min_c = min(counts.values())
    max_c = max(counts.values())
    print(f"\n  Ratio (max/min): {max_c}/{min_c} = {max_c/min_c:.2f}x")
    if max_c / min_c <= 1.01:
        print("  PASS: Perfectly balanced.")
    else:
        print("  WARNING: Not perfectly balanced.")

    print("\nBalancing complete.")


if __name__ == "__main__":
    main()
