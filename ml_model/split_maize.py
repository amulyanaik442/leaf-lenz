"""Split flat maize dataset into train/valid/test 80/10/10."""
import os
import sys
import random
import shutil

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIZE_DATA = os.path.join(BASE_DIR, "ml_model", "maize_data")
SEED = 42

def main():
    print("=" * 60)
    print("  SPLITTING MAIZE DATASET (80/10/10)")
    print("=" * 60)

    # Check if already split
    if os.path.exists(os.path.join(MAIZE_DATA, "train")):
        print("Already split. Checking...")
        for split in ["train", "valid", "test"]:
            d = os.path.join(MAIZE_DATA, split)
            if os.path.exists(d):
                total = sum(len([f for f in os.listdir(os.path.join(d, c))
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                           for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))
                print(f"  {split}: {total} images")
        return

    # Collect all images per class from flat directory
    classes = {}
    for item in sorted(os.listdir(MAIZE_DATA)):
        item_path = os.path.join(MAIZE_DATA, item)
        if os.path.isdir(item_path) and item not in ("train", "valid", "test"):
            files = [f for f in os.listdir(item_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            classes[item] = [(os.path.join(item_path, f), f) for f in files]

    print(f"\nClasses ({len(classes)}):")
    total = 0
    for cls, items in sorted(classes.items()):
        print(f"  {cls:30s} {len(items):5d}")
        total += len(items)
    print(f"  {'TOTAL':30s} {total:5d}")

    # Create splits
    random.seed(SEED)
    for cls, items in classes.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.80)
        n_valid = int(n * 0.10)

        splits = {
            "train": items[:n_train],
            "valid": items[n_train:n_train + n_valid],
            "test": items[n_train + n_valid:],
        }

        for split, files in splits.items():
            split_dir = os.path.join(MAIZE_DATA, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for src, fname in files:
                shutil.copy2(src, os.path.join(split_dir, fname))

        print(f"  {cls:30s}: {len(splits['train'])} train, {len(splits['valid'])} valid, {len(splits['test'])} test")

    print("\nSplit complete.")


if __name__ == "__main__":
    main()
