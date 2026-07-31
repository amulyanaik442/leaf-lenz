"""Inspect wheat dataset: class balance, ratio analysis, sample file checks."""
import os
import sys
import random

BASE = r"C:\Users\amuly\OneDrive\Desktop\agriculture_ai-leaf_lenz\ml_model\wheat_data"
SPLITS = ["train", "valid", "test"]

all_counts = {}
for split in SPLITS:
    split_path = os.path.join(BASE, split)
    print(f"\n{'='*50}")
    print(f"  {split.upper()} SET")
    print(f"{'='*50}")
    for cls in sorted(os.listdir(split_path)):
        cls_path = os.path.join(split_path, cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            all_counts.setdefault(cls, {})
            all_counts[cls][split] = n
            print(f"  {cls:25s} {n:5d} images")

# Balance analysis
print(f"\n{'='*50}")
print("  BALANCE ANALYSIS (TRAIN SET)")
print(f"{'='*50}")

train_counts = {cls: counts.get("train", 0) for cls, counts in all_counts.items()}
min_cls = min(train_counts, key=train_counts.get)
max_cls = max(train_counts, key=train_counts.get)
min_val = train_counts[min_cls]
max_val = train_counts[max_cls]
ratio = max_val / min_val if min_val > 0 else float("inf")
mean_count = sum(train_counts.values()) / len(train_counts)

print(f"  Min class: {min_cls} = {min_val} images")
print(f"  Max class: {max_cls} = {max_val} images")
print(f"  Ratio (max/min): {ratio:.2f}x  (target: < 2.0x)")
if ratio < 2.0:
    print("  Status: PASS")
else:
    print("  Status: FAIL - NEEDS FIXING")

print(f"\n  Mean per class: {mean_count:.0f} images")
print(f"  Per-class ratio to mean:")
for cls, cnt in sorted(train_counts.items(), key=lambda x: x[1]):
    r = cnt / mean_count
    tag = ""
    if r < 0.5:
        tag = " <-- TOO FEW (< 50% of mean)"
    elif r > 2.0:
        tag = " <-- TOO MANY (> 200% of mean)"
    print(f"    {cls:25s} {cnt:5d}  ({r:.2f}x mean){tag}")

# Check image file integrity
print(f"\n{'='*50}")
print("  IMAGE FILE INTEGRITY CHECK")
print(f"{'='*50}")
bad_files = []
for cls in sorted(os.listdir(os.path.join(BASE, "train"))):
    cls_path = os.path.join(BASE, "train", cls)
    if not os.path.isdir(cls_path):
        continue
    files = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    for f in files:
        fp = os.path.join(cls_path, f)
        sz = os.path.getsize(fp)
        if sz < 100:
            bad_files.append((cls, f, sz, "tiny file"))
        elif sz > 10_000_000:
            bad_files.append((cls, f, sz, "very large file"))

if bad_files:
    print(f"  Found {len(bad_files)} problematic files:")
    for cls, f, sz, reason in bad_files:
        print(f"    {cls}/{f}  ({sz} bytes) - {reason}")
else:
    print("  All files look OK (no tiny or extremely large files)")

# Sample filenames per class for visual inspection reference
print(f"\n{'='*50}")
print("  SAMPLE FILENAMES PER CLASS (for visual inspection)")
print(f"{'='*50}")
for cls in sorted(os.listdir(os.path.join(BASE, "train"))):
    cls_path = os.path.join(BASE, "train", cls)
    if not os.path.isdir(cls_path):
        continue
    files = sorted([f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    samples = random.sample(files, min(3, len(files)))
    print(f"  {cls:25s} -> {', '.join(samples)}")

# Check for duplicate filenames across classes (possible mislabeling)
print(f"\n{'='*50}")
print("  DUPLICATE FILENAME CHECK (cross-class)")
print(f"{'='*50}")
fname_map = {}
for cls in sorted(os.listdir(os.path.join(BASE, "train"))):
    cls_path = os.path.join(BASE, "train", cls)
    if not os.path.isdir(cls_path):
        continue
    for f in os.listdir(cls_path):
        fname_map.setdefault(f, []).append(cls)

dupes = {f: classes for f, classes in fname_map.items() if len(classes) > 1}
if dupes:
    print(f"  Found {len(dupes)} filenames shared across classes:")
    for f, classes in list(dupes.items())[:10]:
        print(f"    {f} -> {classes}")
else:
    print("  No duplicate filenames across classes")
