"""
Build the Bean training/validation/test working set from the audited images of the
Kaggle zunorain/bean-plant-dataset (1,432 images, 4 classes).

Approved plan (audit 2026-08-04):
  * Raw dataset (no augmentation markers).
  * Remove the 174 `resized_resized_*` re-compressed near-duplicate copies from
    LEAFMINNER_LEAF (approved).
  * Exact-MD5 dedup: keep ONE copy per duplicate group; for cross-class label
    conflicts keep the DOWNY_MILDEW_LEAF copy and drop the conflicting
    LEAFMINNER/POWDER copies (approved).
  * No blur filter (approved; lap_var heuristic too aggressive for this dataset).
  * Balance: random undersample each class to the smallest class (seed 42).
  * Stratified 80/10/10 split per class at the FILE level (seed 42).
  * Augment TRAIN only: 2 random augmentations per real image, keep 50% (seed 42).

Outputs:
  * dataset/bean/Bean_Split/{train,valid,test}/bean___<class>/*.jpg
  * dataset/bean/bean_train_prep_report.json

The original Kaggle archive is left untouched.
"""
import os
import sys
import json
import hashlib
import shutil
from collections import Counter, defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(BASE_DIR, 'dataset', 'bean', 'Bean Plant dataset')
PEA_DIR = os.path.join(BASE_DIR, 'dataset', 'bean')
SPLIT_DIR = os.path.join(PEA_DIR, 'Bean_Split')
REPORT = os.path.join(PEA_DIR, 'bean_train_prep_report.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.WEBP')
SEED = 42
SPLIT_FRACS = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
AUG_KEEP_RATIO = 0.5
AUGS_PER_IMAGE = 2
JPEG_Q = 95
LOG = []

# Dataset folder name -> Leaf Lenz production class name (lowercase convention)
CLASS_MAP = {
    'DOWNY_MILDEW_LEAF': 'bean___downy_mildew',
    'FRESH_LEAF': 'bean___healthy',
    'LEAFMINNER_LEAF': 'bean___leafminer',
    'POWDER_MILDEW_LEAF': 'bean___powdery_mildew',
}


def log(msg):
    LOG.append(str(msg))
    print(msg, flush=True)


def rng_seeded(seed):
    return np.random.RandomState(seed)


def load_files():
    """Return list of (cls, abspath, fname)."""
    files = []
    for cls in CLASS_MAP:
        d = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMAGE_EXTS):
                files.append((cls, os.path.join(d, f), f))
    return files


def md5_of(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def deduplicate(files):
    """
    Remove resized_resized_* copies, then exact-MD5 dedup keeping one copy per
    group; for cross-class label conflicts keep the DOWNY copy.
    Returns (kept, dropped_reasons: dict cls -> list of {file, reason}).
    """
    dropped = defaultdict(list)
    kept = []
    for cls, path, fname in files:
        if fname.startswith('resized_resized_'):
            dropped[cls].append({'file': fname, 'reason': 'resized_resized_near_duplicate'})
            continue
        kept.append((cls, path, fname))

    groups = defaultdict(list)
    for item in kept:
        groups[md5_of(item[1])].append(item)

    final = []
    for h, g in groups.items():
        classes = set(x[0] for x in g)
        if len(classes) > 1:
            keep = sorted(g, key=lambda x: 0 if x[0] == 'DOWNY_MILDEW_LEAF' else 1)[0]
            final.append(keep)
            for item in g:
                if item is not keep:
                    dropped[item[0]].append(
                        {'file': item[2], 'reason': f'cross_class_label_conflict_kept_{keep[0]}'})
        else:
            final.append(g[0])
            for item in g[1:]:
                dropped[item[0]].append({'file': item[2], 'reason': 'exact_duplicate'})
    return final, dropped


def balance(files, rng):
    """Random undersample each class to the smallest class count. Returns (balanced, target)."""
    totals = Counter(c for c, _p, _f in files)
    n_min = min(totals.values())
    by_cls = defaultdict(list)
    for item in files:
        by_cls[item[0]].append(item)
    balanced = []
    for cls in sorted(by_cls):
        items = list(by_cls[cls])
        rng.shuffle(items)
        balanced.extend(items[:n_min])
    return balanced, n_min


def assign_split(files, rng):
    """Stratified 80/10/10 file-level split. Returns (split->[items], split_counts)."""
    splits = {s: [] for s in SPLIT_FRACS}
    split_counts = {s: Counter() for s in SPLIT_FRACS}
    by_cls = defaultdict(list)
    for item in files:
        by_cls[item[0]].append(item)
    for cls in sorted(by_cls):
        items = list(by_cls[cls])
        rng.shuffle(items)
        total = len(items)
        n_train = int(round(SPLIT_FRACS['train'] * total))
        n_valid = int(round(SPLIT_FRACS['valid'] * total))
        for s, idx in (('train', slice(0, n_train)),
                       ('valid', slice(n_train, n_train + n_valid)),
                       ('test', slice(n_train + n_valid, None))):
            for item in items[idx]:
                splits[s].append(item)
                split_counts[s][item[0]] += 1
    return splits, split_counts


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


def read_rgb(path):
    img = cv2.imread(path)
    if img is not None:
        h, w = img.shape[:2]
        s = 1024 / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                             interpolation=cv2.INTER_AREA)
    return img


def main():
    log('=' * 60)
    log('PEA TRAIN SET PREP (cleanup -> balance -> split -> augment)')
    log('=' * 60)
    rng = rng_seeded(SEED)
    aug = make_augmenter()

    files = load_files()
    raw_counts = Counter(c for c, _p, _f in files)
    log(f'  raw file counts: {dict(raw_counts)}')

    # ---- cleanup (resized_resized + MD5 dedup + label conflicts) ----
    cleaned, dropped = deduplicate(files)
    cleaned_counts = Counter(c for c, _p, _f in cleaned)
    log(f'  dropped resized_resized / duplicates / conflicts: '
        f'{dict((k, len(v)) for k, v in dropped.items())}')
    log(f'  cleaned file counts: {dict(cleaned_counts)}')

    # ---- balance (random undersample) ----
    balanced, n_min = balance(cleaned, rng)
    log(f'  balance target (smallest class): {n_min}')
    log(f'  balanced counts: {dict(Counter(c for c, _p, _f in balanced))}')

    # ---- stratified file-level split ----
    splits, split_counts = assign_split(balanced, rng)
    for s in SPLIT_FRACS:
        log(f'  split {s:5s}: {len(splits[s])}  {dict(split_counts[s])}')

    # ---- write real images ----
    written = 0
    for s in SPLIT_FRACS:
        for cls, path, fname in splits[s]:
            out_dir = os.path.join(SPLIT_DIR, s, CLASS_MAP[cls])
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(path, os.path.join(out_dir, fname))
            written += 1
    log(f'  copied {written} real images')

    # ---- augment train only ----
    train_items = splits['train']
    aug_pool = []
    for cls, path, fname in train_items:
        for i in range(AUGS_PER_IMAGE):
            aug_pool.append((cls, path, fname, i))
    rng.shuffle(aug_pool)
    keep_n = int(round(len(aug_pool) * AUG_KEEP_RATIO))
    gen_per_class = Counter()
    for cls, path, fname, i in aug_pool[:keep_n]:
        img = read_rgb(path)
        if img is None:
            continue
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
        'source': 'Kaggle zunorain/bean-plant-dataset',
        'download_url': 'https://www.kaggle.com/datasets/zunorain/pea-plant-dataset',
        'raw_file_counts': dict(raw_counts),
        'cleaned_file_counts': dict(cleaned_counts),
        'dropped': dict(dropped),
        'split_strategy': ('file-level stratified random undersample to smallest '
                           'class; 80/10/10 per class, seed 42'),
        'blur_filter': 'disabled (approved: lap_var heuristic too aggressive for this dataset)',
        'balance_target_files': n_min,
        'balanced_file_counts': dict(Counter(c for c, _p, _f in balanced)),
        'split_fractions': SPLIT_FRACS,
        'split_counts': {s: dict(c) for s, c in split_counts.items()},
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
