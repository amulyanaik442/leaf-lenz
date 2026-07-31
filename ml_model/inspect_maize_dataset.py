"""
Inspect the maize dataset for class balance and image quality.

Usage:
    python ml_model/inspect_maize_dataset.py
"""
import os
import sys
import random

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIZE_DATA = os.path.join(BASE_DIR, "ml_model", "maize_data")


def inspect():
    print("=" * 60)
    print("  MAIZE DATASET INSPECTION")
    print("=" * 60)

    # 1. Class distribution
    print("\n1. CLASS DISTRIBUTION")
    print("-" * 40)
    class_counts = {}
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(MAIZE_DATA, split)
        if not os.path.exists(split_dir):
            continue
        print(f"\n  {split.upper()}:")
        for cls in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            n = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[cls] = class_counts.get(cls, 0) + n
            print(f"    {cls:30s} {n:5d}")

    print("\n  TOTAL PER CLASS:")
    for cls, count in sorted(class_counts.items()):
        print(f"    {cls:30s} {count:5d}")

    # 2. Check imbalance ratios
    print("\n2. CLASS BALANCE CHECK")
    print("-" * 40)
    counts = list(class_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    min_cls = [k for k, v in class_counts.items() if v == min_count]
    max_cls = [k for k, v in class_counts.items() if v == max_count]
    ratio = max_count / min_count if min_count > 0 else float('inf')

    print(f"  Min class: {min_cls[0]} = {min_count}")
    print(f"  Max class: {max_cls[0]} = {max_count}")
    print(f"  Ratio (max/min): {ratio:.2f}x")

    if ratio > 2.0:
        print(f"  WARNING: Imbalance detected! Max/min ratio {ratio:.2f}x > 2.0x")
        print(f"  Consider augmenting {min_cls[0]} or removing from {max_cls[0]}")
    elif ratio > 1.5:
        print(f"  OK: Slight imbalance ({ratio:.2f}x) but acceptable")
    else:
        print(f"  PASS: Classes are well-balanced ({ratio:.2f}x)")

    # 3. Sample filenames per class
    print("\n3. SAMPLE FILENAMES PER CLASS")
    print("-" * 40)
    for cls in sorted(class_counts.keys()):
        cls_path = os.path.join(MAIZE_DATA, "train", cls)
        if os.path.exists(cls_path):
            files = [f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.seed(42)
            samples = random.sample(files, min(5, len(files)))
            print(f"\n  {cls}:")
            for s in samples:
                print(f"    {s}")

    # 4. Check for duplicate filenames across classes
    print("\n4. DUPLICATE FILENAME CHECK")
    print("-" * 40)
    all_files = {}
    for cls in sorted(class_counts.keys()):
        cls_path = os.path.join(MAIZE_DATA, "train", cls)
        if os.path.exists(cls_path):
            for f in os.listdir(cls_path):
                if f in all_files:
                    print(f"  WARNING: '{f}' found in both {all_files[f]} and {cls}")
                else:
                    all_files[f] = cls
    print("  Check complete.")

    print("\n" + "=" * 60)
    print("  INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    inspect()
