"""Re-split wheat train data into 80/10/10 and balance training set."""
import os
import sys
import random
import shutil

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

WHEAT_DATA = os.path.join(os.environ['TEMP'], 'wheat_data')
SEED = 42
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

def safe_name(fname):
    return fname.replace(" ", "_").replace("(", "").replace(")", "")

def main():
    print("=" * 60)
    print("  RE-SPLITTING WHEAT DATASET (80/10/10 + BALANCE)")
    print("=" * 60)

    # Collect all images from raw train dir
    raw_train_dir = os.path.join(WHEAT_DATA, "train")
    if not os.path.exists(raw_train_dir):
        print("ERROR: No raw train directory found")
        return

    raw_classes = {}
    for cls in sorted(os.listdir(raw_train_dir)):
        cls_path = os.path.join(raw_train_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        raw_classes[cls] = [(os.path.join(cls_path, f), f) for f in files]

    print(f"\nRaw train classes ({len(raw_classes)}):")
    total_raw = 0
    for cls, items in sorted(raw_classes.items()):
        print(f"  {cls:30s} {len(items):5d}")
        total_raw += len(items)
    print(f"  {'TOTAL':30s} {total_raw:5d}")

    # Find smallest class
    min_count = min(len(v) for v in raw_classes.values())
    min_cls = [k for k, v in raw_classes.items() if len(v) == min_count][0]
    print(f"\nSmallest class: {min_cls} = {min_count}")

    # Stage all images
    import tempfile
    stage_dir = tempfile.mkdtemp(prefix="wheat_stage_")
    staged = {}
    for cls, items in raw_classes.items():
        staged[cls] = []
        cls_stage = os.path.join(stage_dir, cls)
        os.makedirs(cls_stage, exist_ok=True)
        for src, fname in items:
            dest = os.path.join(cls_stage, safe_name(fname))
            shutil.copy2(src, dest)
            staged[cls].append(dest)
    print(f"Staged {sum(len(v) for v in staged.values())} images.")

    # Remove old valid/test dirs (keep raw train for now)
    for split in ["valid", "test"]:
        split_dir = os.path.join(WHEAT_DATA, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
            print(f"  Removed {split_dir}")

    # Remove old train dir (staged copy exists)
    shutil.rmtree(raw_train_dir)
    print(f"  Removed {raw_train_dir}")

    # Undersample each class to min_count
    random.seed(SEED)
    balanced = {}
    for cls, paths in staged.items():
        balanced[cls] = random.sample(paths, min_count)

    # Create new splits
    print(f"\nCreating 80/10/10 splits (target: {min_count}/class):")
    for cls, paths in balanced.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * TRAIN_RATIO)
        n_valid = int(n * VALID_RATIO)

        cls_splits = {
            "train": paths[:n_train],
            "valid": paths[n_train:n_train + n_valid],
            "test": paths[n_train + n_valid:],
        }

        for split, files in cls_splits.items():
            split_dir = os.path.join(WHEAT_DATA, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for f in files:
                shutil.copy2(f, os.path.join(split_dir, os.path.basename(f)))

        train_n = len(cls_splits["train"])
        valid_n = len(cls_splits["valid"])
        test_n = len(cls_splits["test"])
        print(f"  {cls:30s}: {train_n} train, {valid_n} valid, {test_n} test")

    # Cleanup staging
    shutil.rmtree(stage_dir, ignore_errors=True)

    # Verify
    print(f"\n{'='*60}")
    print("  FINAL DISTRIBUTION")
    print(f"{'='*60}")
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(WHEAT_DATA, split)
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

    # Ratio check on train
    train_dir = os.path.join(WHEAT_DATA, 'train')
    counts = {}
    for cls in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len([f for f in os.listdir(cls_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    min_c = min(counts.values())
    max_c = max(counts.values())
    ratio = max_c / min_c
    print(f"\n  Train ratio (max/min): {max_c}/{min_c} = {ratio:.2f}x")
    if ratio <= 1.01:
        print("  PASS: Perfectly balanced.")
    else:
        print("  WARNING: Not perfectly balanced.")

    print("\nRe-splitting complete.")


if __name__ == "__main__":
    main()
