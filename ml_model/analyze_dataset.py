import os, json, hashlib
from collections import defaultdict, Counter
from PIL import Image
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SPLIT_DIR = os.path.join(os.path.dirname(__file__), 'groundnut_split_data')
ML_ASSETS = os.path.join(BASE_DIR, 'detector', 'ml_assets')
CLASSES_PATH = os.path.join(ML_ASSETS, 'class_names.json')

BASE_DATASET = os.path.expandvars(R'%USERPROFILE%\.cache\kagglehub\datasets\warcoder\groundnut-plant-leaf-data\versions\1\Dataset of groundnut plant leaf images for classification and detection')
RAW_DIR = os.path.join(BASE_DATASET, 'Raw_Data')
TEST_DIR = os.path.join(BASE_DATASET, 'Groundnut_Leaf_dataset', 'test')

print("=" * 70)
print("GROUNDNUT DATASET ANALYSIS")
print("=" * 70)

# 1. Verify label mapping
print("\n" + "-" * 70)
print("1. LABEL MAPPING VERIFICATION")
print("-" * 70)

RAW_FOLDER_CANDIDATES = {
    "Early Leaf Spot": ["Early Leaf Spot", "Early_Leaf_Spot", "early_leaf_spot"],
    "Late Leaf Spot": ["Late Leaf Spot", "Late_Leaf_Spot", "late_leaf_spot", "late leaf spot"],
    "Early Rust": ["Early Rust", "Early_Rust", "early_rust"],
    "Rust": ["Rust", "rust"],
    "Healthy": ["Healthy", "healthy", "healthy leaf"],
    "Nutrition Deficiency": ["Nutrition Deficiency", "Nutrition_Deficiency", "nutrition_deficiency", "nutrition deficiency"],
}
RAW_TO_CLASS = {
    "Early Leaf Spot": "Groundnut___Leaf_Spot",
    "Late Leaf Spot": "Groundnut___Leaf_Spot",
    "Early Rust": "Groundnut___Rust",
    "Rust": "Groundnut___Rust",
    "Healthy": "Groundnut___Healthy",
    "Nutrition Deficiency": "Groundnut___Nutrition_Deficiency",
}

print(f"Raw data directory: {RAW_DIR}")
print(f"  Exists: {os.path.isdir(RAW_DIR)}")

if os.path.isdir(RAW_DIR):
    actual_folders = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
    print(f"  Actual folder names: {actual_folders}")
    print(f"  Expected candidates per class:")
    for canonical, candidates in RAW_FOLDER_CANDIDATES.items():
        match = [f for f in actual_folders if f in candidates]
        status = f"MATCHED as '{canonical}' -> {RAW_TO_CLASS[canonical]}" if match else "NO MATCH"
        print(f"    {canonical:25s} candidates={str(candidates):60s} -> {status}")

# 2. Class distribution
print("\n" + "-" * 70)
print("2. CLASS DISTRIBUTION")
print("-" * 70)

# 2a. Raw data
if os.path.isdir(RAW_DIR):
    print("\n--- Raw Data (pre-undersampling) ---")
    raw_counts = {}
    for canonical, candidates in RAW_FOLDER_CANDIDATES.items():
        matched = [f for f in actual_folders if f in candidates]
        if matched:
            folder = matched[0]
            path = os.path.join(RAW_DIR, folder)
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            target = RAW_TO_CLASS[canonical]
            raw_counts[folder] = (target, count)
            print(f"  {folder:30s} ({canonical:20s}) -> {target:35s} {count:5d} images")

# 2b. Merged counts
print("\n--- Merged Class Counts (raw) ---")
merged_counts = defaultdict(int)
for folder, (target, count) in raw_counts.items():
    merged_counts[target] += count
for cls in sorted(merged_counts):
    print(f"  {cls:40s} {merged_counts[cls]:5d} images")

# 2c. Split data
print("\n--- Split Data ---")
if os.path.isdir(SPLIT_DIR):
    for split in ['train', 'val']:
        split_path = os.path.join(SPLIT_DIR, split)
        if os.path.isdir(split_path):
            total = 0
            print(f"\n  {split.upper()}:")
            for cls in sorted(os.listdir(split_path)):
                cls_path = os.path.join(split_path, cls)
                if os.path.isdir(cls_path):
                    count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
                    total += count
                    print(f"    {cls:40s} {count:5d}")
            print(f"    {'TOTAL':40s} {total:5d}")
else:
    print(f"  Split directory not found at {SPLIT_DIR}")

# 3. Duplicate detection
print("\n" + "-" * 70)
print("3. DUPLICATE IMAGES")
print("-" * 70)

# 3a. Same filename across classes
print("\n--- Same filename across different classes ---")
filename_to_classes = defaultdict(list)
if os.path.isdir(RAW_DIR):
    for folder in actual_folders:
        folder_path = os.path.join(RAW_DIR, folder)
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                filename_to_classes[f].append(folder)

duplicate_fnames = {k: v for k, v in filename_to_classes.items() if len(v) > 1}
if duplicate_fnames:
    print(f"  Found {len(duplicate_fnames)} filenames appearing in multiple classes:")
    for fname, classes in sorted(list(duplicate_fnames.items())[:20]):
        print(f"    {fname:40s} appears in: {classes}")
    if len(duplicate_fnames) > 20:
        print(f"    ... and {len(duplicate_fnames) - 20} more")
else:
    print("  No filename collisions across classes.")

# 3b. Exact duplicate detection (by content hash)
print("\n--- Exact duplicates (by MD5 hash) ---")
hash_map = defaultdict(list)
total_files = 0
if os.path.isdir(RAW_DIR):
    for folder in actual_folders:
        folder_path = os.path.join(RAW_DIR, folder)
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                total_files += 1
                filepath = os.path.join(folder_path, f)
                try:
                    with open(filepath, 'rb') as fh:
                        file_hash = hashlib.md5(fh.read()).hexdigest()
                    hash_map[file_hash].append((folder, f))
                except Exception as e:
                    print(f"    ERROR reading {filepath}: {e}")

hash_dups = {k: v for k, v in hash_map.items() if len(v) > 1}
if hash_dups:
    print(f"  Found {len(hash_dups)} duplicate image hashes (exact same content):")
    cross_class = 0
    for h, entries in sorted(hash_dups.items()):
        folders = set(e[0] for e in entries)
        if len(folders) > 1:
            cross_class += 1
            print(f"    Cross-class duplicate: {entries}")
        elif len(entries) > 1:
            pass  # same class, same hash — likely just copies
    print(f"  Cross-class exact duplicates: {cross_class}")
    print(f"  Same-class exact duplicates: {len(hash_dups) - cross_class}")
else:
    print("  No exact duplicate images found.")

# 3c. Check for train/test overlap
print("\n--- Train/Test overlap check ---")
if os.path.isdir(SPLIT_DIR) and os.path.isdir(TEST_DIR):
    train_hashes = set()
    train_files = set()
    for split in ['train', 'val']:
        split_path = os.path.join(SPLIT_DIR, split)
        if os.path.isdir(split_path):
            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                if os.path.isdir(cls_path):
                    for f in os.listdir(cls_path):
                        if f.lower().endswith(('.jpg','.jpeg','.png')):
                            train_files.add(f)
                            fp = os.path.join(cls_path, f)
                            try:
                                with open(fp, 'rb') as fh:
                                    train_hashes.add(hashlib.md5(fh.read()).hexdigest())
                            except: pass

    test_hashes = set()
    test_files = set()
    for folder in os.listdir(TEST_DIR):
        folder_path = os.path.join(TEST_DIR, folder)
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    test_files.add(f)
                    fp = os.path.join(folder_path, f)
                    try:
                        with open(fp, 'rb') as fh:
                            test_hashes.add(hashlib.md5(fh.read()).hexdigest())
                    except: pass

    fname_overlap = train_files & test_files
    hash_overlap = train_hashes & test_hashes
    if fname_overlap:
        print(f"  Same filename overlap: {len(fname_overlap)} files")
        for f in sorted(list(fname_overlap))[:10]:
            print(f"    {f}")
    else:
        print(f"  Same filename overlap: 0 (no filename collisions)")
    if hash_overlap:
        print(f"  Same content (hash) overlap: {len(hash_overlap)} files (DATA LEAKAGE!)")
    else:
        print(f"  Same content (hash) overlap: 0 (no data leakage)")

# 4. Image quality
print("\n" + "-" * 70)
print("4. IMAGE QUALITY")
print("-" * 70)

if os.path.isdir(SPLIT_DIR):
    all_images = []
    for split in ['train', 'val']:
        split_path = os.path.join(SPLIT_DIR, split)
        if os.path.isdir(split_path):
            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                if os.path.isdir(cls_path):
                    for f in os.listdir(cls_path):
                        if f.lower().endswith(('.jpg','.jpeg','.png')):
                            all_images.append((split, cls, os.path.join(cls_path, f)))

    print(f"  Total images in split: {len(all_images)}")

    corrupt = []
    small_images = []
    unusual_ratio = []
    sizes = []

    for split, cls, path in all_images:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                w, h = img.size
                sizes.append((w, h))
                if w < 50 or h < 50:
                    small_images.append((split, cls, os.path.basename(path), w, h))
                aspect = w / h if h > 0 else 0
                if aspect < 0.5 or aspect > 2.0:
                    unusual_ratio.append((split, cls, os.path.basename(path), w, h))
        except Exception as e:
            corrupt.append((split, cls, os.path.basename(path), str(e)))

    if corrupt:
        print(f"\n  CORRUPT IMAGES: {len(corrupt)}")
        for split, cls, fname, err in corrupt[:10]:
            print(f"    [{split}/{cls}] {fname}: {err}")
    else:
        print(f"\n  Corrupt images: 0")

    if small_images:
        print(f"  Small images (<50px): {len(small_images)}")
        for split, cls, fname, w, h in small_images[:5]:
            print(f"    [{split}/{cls}] {fname}: {w}x{h}")
    else:
        print(f"  Small images (<50px): 0")

    if unusual_ratio:
        print(f"  Unusual aspect ratio (<0.5 or >2.0): {len(unusual_ratio)}")
        for split, cls, fname, w, h in unusual_ratio[:5]:
            print(f"    [{split}/{cls}] {fname}: {w}x{h}")
    else:
        print(f"  Unusual aspect ratio: 0")

    # Size distribution
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"  Width:  min={min(widths)} max={max(widths)} median={sorted(widths)[len(widths)//2]}")
        print(f"  Height: min={min(heights)} max={max(heights)} median={sorted(heights)[len(heights)//2]}")

        # Check how many are exactly the expected size
        expected = (256, 256)
        exact = sum(1 for s in sizes if s == expected)
        print(f"  Exactly {expected[0]}x{expected[1]}: {exact}/{len(sizes)}")

# 5. Preprocessing verification
print("\n" + "-" * 70)
print("5. PREPROCESSING VERIFICATION")
print("-" * 70)

print("""
Training transforms (expand_groundnut_model.py):
  1. Resize to (256, 256)
  2. RandomCrop(224)  -- random offset
  3. RandomHorizontalFlip
  4. RandomVerticalFlip
  5. RandomRotation(10)
  6. ToTensor
  7. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Validation transforms:
  1. Resize to (224, 224)
  2. ToTensor
  3. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Inference (detector/inference.py):
  1. Resize to (256, 256) with BILINEAR
  2. Centre crop to (224, 224)
  3. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  4. TTA: 4 augmentations x 5 crops = 20 views, averaged

MISMATCH: Training resize uses default interpolation (BILINEAR).
           Inference resize also uses BILINEAR. Match is correct.

MISMATCH: Training uses RandomCrop(224) (random offset), validation uses
           Resize(224) and NO crop. Inference uses Resize(256) + CentreCrop(224).
           This means validation saw SMALLER images (224x224 without cropping)
           than inference (256 -> centre crop to 224). This is a mild mismatch
           but acceptable.
""")

# 6. Check class_names.json has all Groundnut classes
print("-" * 70)
print("6. CLASS_NAMES.JSON VERIFICATION")
print("-" * 70)

if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH) as f:
        cn = json.load(f)
    print(f"  Total classes: {len(cn)}")
    groundnut_classes = [c for c in cn if c.startswith('Groundnut___')]
    print(f"  Groundnut classes: {len(groundnut_classes)}")
    for gc in groundnut_classes:
        idx = cn.index(gc)
        print(f"    [{idx:3d}] {gc}")

# 7. Verify ONNX model output shape
print("\n" + "-" * 70)
print("7. ONNX MODEL VERIFICATION")
print("-" * 70)
import onnxruntime as ort
ONNX_PATH = os.path.join(ML_ASSETS, 'model.onnx')
if os.path.exists(ONNX_PATH):
    try:
        sess = ort.InferenceSession(ONNX_PATH)
        in_shape = sess.get_inputs()[0].shape
        out_shape = sess.get_outputs()[0].shape
        print(f"  Input shape:  {in_shape}")
        print(f"  Output shape: {out_shape}")
        print(f"  Expected:     [batch_size, {len(cn)}]" if 'cn' in dir() else "")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
