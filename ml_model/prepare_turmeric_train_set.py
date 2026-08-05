"""
Build the Turmeric leaf training/validation/test working set from the audited
ORIGINAL images of the Mendeley jtttfbx342/1 dataset.

Approved plan: 4 classes (Aphids_Disease, Blotch, Healthy_Leaf, Leaf_Spot),
ORIGINAL (real) images only (no synthetic), balanced by undersample (seed 42).
Mirrors the verified Chilli pipeline exactly.

Steps (seed-fixed for reproducibility):
  1. Enumerate ORIGINAL images from the extracted dataset (Original DataSet).
  2. Deduplicate exact MD5 copies (cross-class conflict -> drop all; within
     class -> keep 1 representative).
  3. Build dHash near-duplicate connected components (<= 8 bits) -> leakage units.
  4. Blur filter (Laplacian variance < 100, mirrors chilli).
  5. Balance: undersample each class to the minority class count (seed 42).
  6. Leakage-free split: assign whole near-dup components to train/valid/test
     (80/10/10) with a greedy class-balanced assignment (seed 42).
  7. Augment the TRAIN portion only: 2 random augmentations per train image,
     then keep 50% of the generated variants (seed 42).

Outputs:
  * dataset/turmeric/Turmeric_Split/{train,valid,test}/Turmeric___<class>/*.jpg
  * dataset/turmeric/turmeric_train_prep_report.json

The extracted archive in temp is left untouched.
"""
import os
import sys
import json
import shutil
import hashlib
from collections import Counter, defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_ROOT = r'C:\Users\amuly\AppData\Local\Temp\opencode\turmeric\extract'
SOURCE_DIR = os.path.join(
    EXTRACT_ROOT, 'Image Dataset for Turmeric Plant Leaf Disease Detection',
    'Original DataSet')
TURMERIC_DIR = os.path.join(BASE_DIR, 'dataset', 'turmeric')
SPLIT_DIR = os.path.join(TURMERIC_DIR, 'Turmeric_Split')
REPORT = os.path.join(TURMERIC_DIR, 'turmeric_train_prep_report.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.WEBP')
NEAR_DUP_DHASH_THRESHOLD = 8
BLUR_LAP_VAR_THRESHOLD = 100.0
ANALYSIS_MAX_DIM = 512
SEED = 42
SPLIT_FRACS = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
AUG_KEEP_RATIO = 0.5
AUGS_PER_IMAGE = 2
JPEG_Q = 95
LOG = []

# Dataset folder name -> Leaf Lenz production class name
CLASS_MAP = {
    'Aphids_Disease': 'Turmeric___aphids_disease',
    'Blotch': 'Turmeric___blotch',
    'Healthy_Leaf': 'Turmeric___healthy',
    'Leaf_Spot': 'Turmeric___leaf_spot',
}


def log(msg):
    LOG.append(str(msg))
    print(msg, flush=True)


def rng_seeded(seed):
    return np.random.RandomState(seed)


def load_original_images():
    """Return list of (cls, abspath, fname) for ORIGINAL images of the 4 classes."""
    files = []
    for cls in CLASS_MAP:
        d = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMAGE_EXTS):
                files.append((cls, os.path.join(d, f), f))
    return files


def load_downscaled_gray(path, max_dim=ANALYSIS_MAX_DIM):
    with Image.open(path) as im:
        w, h = im.size
    mx = max(w, h)
    if mx >= 1024:
        flag = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_REDUCED_GRAYSCALE_8
    elif mx >= 512:
        flag = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_REDUCED_GRAYSCALE_2
    else:
        flag = cv2.IMREAD_GRAYSCALE
    arr = cv2.imread(path, flag)
    if arr is None:
        with Image.open(path) as im:
            arr = np.asarray(im.convert('L'), dtype=np.uint8)
    h, w = arr.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
    return arr


def dhash(img_gray, size=8):
    small = cv2.resize(img_gray, (size + 1, size))
    diff = small[:, 1:] > small[:, :-1]
    return sum(1 << i for i, b in enumerate(diff.ravel()) if b)


def dedup(files):
    md5_map = defaultdict(list)
    for idx, (cls, path, fname) in enumerate(files):
        h = hashlib.md5(open(path, 'rb').read()).hexdigest()
        md5_map[h].append(idx)
    keep_idx = set()
    dropped = []
    for h, members in md5_map.items():
        classes = set(files[i][0] for i in members)
        if len(classes) > 1:
            dropped.append({'md5': h, 'classes': sorted(classes),
                            'files': [(files[i][0], files[i][2]) for i in members],
                            'reason': 'cross-class exact duplicate (label conflict)'})
        else:
            keep_idx.add(min(members))
    kept = [files[i] for i in sorted(keep_idx)]
    return kept, dropped


def build_components(kept):
    hashes = []
    for i, (cls, path, fname) in enumerate(kept):
        gray = load_downscaled_gray(path)
        hashes.append(dhash(gray))
        if (i + 1) % 400 == 0:
            log(f'  ...dhash {i + 1}/{len(kept)}')
    n = len(kept)
    bits = np.unpackbits(np.array(hashes, dtype=np.uint64).view(np.uint8)
                         .reshape(n, 8), axis=1)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        dist = np.count_nonzero(bits[i] != bits, axis=1)
        for j in np.where((dist <= NEAR_DUP_DHASH_THRESHOLD) & (np.arange(n) > i))[0]:
            if kept[i][0] == kept[j][0]:
                union(i, j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values()), hashes


def blur_filter(kept):
    kept_out = []
    removed = []
    for i, (cls, path, fname) in enumerate(kept):
        gray = load_downscaled_gray(path)
        v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if v < BLUR_LAP_VAR_THRESHOLD:
            removed.append({'file': fname, 'class': cls, 'lap_var': round(v, 1)})
        else:
            kept_out.append((cls, path, fname))
        if (i + 1) % 400 == 0:
            log(f'  ...blur {i + 1}/{len(kept)}')
    return kept_out, removed


def balance(kept, rng):
    counts = Counter(cls for cls, _, _ in kept)
    n_min = min(counts.values())
    result = []
    per_class_kept = {}
    for cls in sorted(counts):
        members = [(c, p, f) for c, p, f in kept if c == cls]
        rng.shuffle(members)
        chosen = members[:n_min]
        per_class_kept[cls] = len(chosen)
        result.extend(chosen)
    return result, per_class_kept, n_min


def assign_split(groups, rng):
    keys = list(groups.keys())
    rng.shuffle(keys)
    total_per_class = Counter()
    for k in keys:
        for cls, _p, _f in groups[k]:
            total_per_class[cls] += 1
    needed = {s: {cls: SPLIT_FRACS[s] * tot for cls, tot in total_per_class.items()}
              for s in SPLIT_FRACS}
    counts = {s: Counter() for s in SPLIT_FRACS}
    assignment = {}
    for k in keys:
        cls_counts = Counter(cls for cls, _p, _f in groups[k])
        best_split, best_fill = None, float('inf')
        for s in SPLIT_FRACS:
            new_counts = counts[s] + cls_counts
            fills = [new_counts[cls] / needed[s][cls]
                     for cls in cls_counts if needed[s][cls] > 0]
            if not fills:
                continue
            overshoot = sum(1 for x in fills if x > 1.0)
            fill = float(np.mean(fills)) + 10.0 * overshoot
            if fill < best_fill:
                best_fill, best_split = fill, s
        assignment[k] = best_split
        counts[best_split] += cls_counts
    return assignment, counts


def make_augmenter():
    def aug(c, img, rng):
        h, w = img.shape[:2]
        out = img
        if rng.random() < 0.5:
            out = cv2.flip(out, 1)
        if rng.random() < 0.5:
            out = cv2.flip(out, 0)
        ang = rng.uniform(-10, 10)
        scale = rng.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
        M[0, 2] += rng.uniform(-0.05, 0.05) * w
        M[1, 2] += rng.uniform(-0.05, 0.05) * h
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        alpha = rng.uniform(0.8, 1.2)
        beta = rng.uniform(-20, 20)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
        return out
    return aug


def copy_image(src, split, cls, rng=None, aug=None):
    out_dir = os.path.join(SPLIT_DIR, split, cls)
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.basename(src)
    shutil.copy2(src, os.path.join(out_dir, fname))
    if aug is not None:
        img = cv2.imread(src)
        if img is not None:
            h, w = img.shape[:2]
            s = 1024 / max(h, w)
            if s < 1.0:
                img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                                 interpolation=cv2.INTER_AREA)
        for i in range(AUGS_PER_IMAGE):
            av = aug(None, img, rng)
            stem = os.path.splitext(fname)[0]
            cv2.imwrite(os.path.join(out_dir, f'{stem}__aug{i:02d}.jpg'), av,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])


def main():
    log('=' * 60)
    log('TURMERIC TRAIN SET PREP (original-only, dedup -> blur -> balance -> split -> augment)')
    log('=' * 60)
    rng = rng_seeded(SEED)
    aug = make_augmenter()

    files = load_original_images()
    raw_counts = Counter(cls for cls, _, _ in files)
    log(f'  original images (4 classes): {len(files)}  {dict(raw_counts)}')

    # ---- dedup ----
    kept, dropped = dedup(files)
    log(f'  after exact-MD5 dedup: {len(kept)}')
    log(f'  dropped cross-class conflicts: {len(dropped)}')
    for d in dropped:
        log(f'      {d["reason"]}: {d["files"]}')

    # ---- blur filter ----
    kept, blur_removed = blur_filter(kept)
    log(f'  blur removal (< lap_var {BLUR_LAP_VAR_THRESHOLD}): {len(blur_removed)}')

    # ---- leakage components ----
    components, hashes = build_components(kept)
    log(f'  dHash near-dup leakage components: {len(components)} '
        f'(max group {max(len(g) for g in components)})')
    comp_sizes = Counter(len(g) for g in components)
    log(f'  component size histogram: {dict(sorted(comp_sizes.items()))}')

    # ---- balance ----
    balanced, per_class_kept, n_min = balance(kept, rng)
    log(f'  balance target (min class): {n_min}  per class: {per_class_kept}')
    log(f'  balanced total: {len(balanced)}')

    # ---- leakage-free split ----
    bal_set = set(balanced)
    groups = {}
    for ci, comp in enumerate(components):
        members = [(c, p, f) for i, (c, p, f) in enumerate(kept)
                   if i in comp and (c, p, f) in bal_set]
        if members:
            groups[ci] = members
    assignment, counts = assign_split(groups, rng)
    split_counts = {}
    for s in SPLIT_FRACS:
        split_counts[s] = dict(counts[s])
        log(f'  split {s:5s}: {sum(counts[s].values())}  {dict(counts[s])}')

    # ---- write real images ----
    written = 0
    for ci, members in groups.items():
        split = assignment[ci]
        for cls, path, fname in members:
            out_dir = os.path.join(SPLIT_DIR, split, CLASS_MAP[cls])
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(path, os.path.join(out_dir, fname))
            written += 1
            if written % 50 == 0:
                log(f'  ...copied {written}/{len(balanced)}')
    log(f'  copied {written} real images')
    for s in SPLIT_FRACS:
        for cls in CLASS_MAP:
            d = os.path.join(SPLIT_DIR, s, CLASS_MAP[cls])
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            log(f'  wrote {s:5s}/{CLASS_MAP[cls]:24s} {n}')

    # ---- augment train only ----
    train_items = []
    for ci, members in groups.items():
        if assignment[ci] == 'train':
            train_items.extend(members)
    aug_pool = []
    for cls, path, fname in train_items:
        for i in range(AUGS_PER_IMAGE):
            aug_pool.append((cls, path, fname, i))
    rng.shuffle(aug_pool)
    keep_n = int(round(len(aug_pool) * AUG_KEEP_RATIO))
    gen_per_class = Counter()
    for cls, path, fname, i in aug_pool[:keep_n]:
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            s = 1024 / max(h, w)
            if s < 1.0:
                img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                                 interpolation=cv2.INTER_AREA)
        av = aug(None, img, rng)
        out_dir = os.path.join(SPLIT_DIR, 'train', CLASS_MAP[cls])
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_dir, f'{stem}__aug{i:02d}.jpg'), av,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        gen_per_class[CLASS_MAP[cls]] += 1
    log(f'  augmented train images generated: {keep_n}  {dict(gen_per_class)}')

    # ---- final counts on disk ----
    disk_counts = {}
    for s in SPLIT_FRACS:
        disk_counts[s] = {}
        for cls in CLASS_MAP.values():
            d = os.path.join(SPLIT_DIR, s, cls)
            disk_counts[s][cls] = len(os.listdir(d)) if os.path.isdir(d) else 0
        log(f'  on-disk {s:5s}: {sum(disk_counts[s].values())}  {disk_counts[s]}')

    report = {
        'classes': CLASS_MAP,
        'raw_original_counts': dict(raw_counts),
        'dedup_kept': len(kept),
        'cross_class_conflicts_dropped': dropped,
        'blur_threshold_lap_var': BLUR_LAP_VAR_THRESHOLD,
        'blur_removed': blur_removed,
        'near_dup_component_count': len(components),
        'near_dup_component_size_histogram': dict(sorted(comp_sizes.items())),
        'balance_target': n_min,
        'balanced_per_class': per_class_kept,
        'split_fractions': SPLIT_FRACS,
        'split_counts': split_counts,
        'augments_per_image': AUGS_PER_IMAGE,
        'aug_keep_ratio': AUG_KEEP_RATIO,
        'augmented_generated': dict(gen_per_class),
        'disk_counts': disk_counts,
        'seed': SEED,
        'log': LOG,
    }
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nReport saved: {REPORT}')


if __name__ == '__main__':
    main()
