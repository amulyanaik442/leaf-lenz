"""
Build the Cauliflower training/validation/test working set from the audited
VegNet "Original Dataset" (Mendeley DOI 10.17632/t5sssfgn2v.3, 656 files).

Approved plan (audit 2026-08-04):
  * Original-only (VegNet ships originals and augmentations in separate zips;
    only the Original Dataset was downloaded).
  * Dedup: exact-MD5 + same-photo SSIM>=0.90 clustering (keep ONE highest
    resolution representative per cluster). No cross-class label conflicts were
    found in the audit (0 exact / 0 near cross-class dups).
  * No blur filter (approved; only 1 image below lap_var 30).
  * Balance: random undersample each class to the smallest class (seed 42).
  * Stratified 80/10/10 split per class at the FILE level (seed 42).
  * Augment TRAIN only: 2 random augmentations per real image, keep 50% (seed 42).

Outputs:
  * dataset/cauliflower/Cauliflower_Split/{train,valid,test}/cauliflower___<class>/*.jpg
  * dataset/cauliflower/cauliflower_train_prep_report.json

The original Mendeley archive is left untouched.
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(BASE_DIR, 'dataset', 'cauliflower', 'Original Dataset')
CAULI_DIR = os.path.join(BASE_DIR, 'dataset', 'cauliflower')
SPLIT_DIR = os.path.join(CAULI_DIR, 'Cauliflower_Split')
REPORT = os.path.join(CAULI_DIR, 'cauliflower_train_prep_report.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.WEBP')
SEED = 42
SPLIT_FRACS = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
AUG_KEEP_RATIO = 0.5
AUGS_PER_IMAGE = 2
JPEG_Q = 95
SSIM_THRESHOLD = 0.90
ANALYSIS_MAX_DIM = 256
LOG = []

# Dataset folder name -> Leaf Lenz production class name (lowercase convention)
# NOTE: 'Bacterial spot rot' was removed from the dataset on 2026-08-04.
CLASS_MAP = {
    'No disease': 'cauliflower___healthy',
    'Black Rot': 'cauliflower___black_rot',
    'Downy Mildew': 'cauliflower___downy_mildew',
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


def load_gray_small(path, max_dim=ANALYSIS_MAX_DIM):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    s = max_dim / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return img


def res_of(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    return w * h


def ssim(a, b):
    a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(b, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    mu1, mu2 = a.mean(), b.mean()
    s1, s2 = a.var(), b.var()
    s12 = ((a - mu1) * (b - mu2)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return (2 * mu1 * mu2 + c1) * (2 * s12 + c2) / \
        ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))


def deduplicate(files):
    """
    Exact-MD5 dedup (keep highest-resolution representative per group), then
    within-class same-photo SSIM>=0.90 clustering (keep one per cluster).
    Returns (kept, dropped: dict cls -> list of {file, reason}).
    """
    dropped = defaultdict(list)

    # ---- exact MD5 ----
    groups = defaultdict(list)
    for item in files:
        groups[md5_of(item[1])].append(item)

    exact_kept = []
    for h, g in groups.items():
        classes = set(x[0] for x in g)
        keep = max(g, key=lambda x: res_of(x[1]))
        exact_kept.append(keep)
        if len(classes) > 1:
            for item in g:
                if item is not keep:
                    dropped[item[0]].append(
                        {'file': item[2], 'reason': f'cross_class_label_conflict_kept_{keep[0]}'})
        else:
            for item in g:
                if item is not keep:
                    dropped[item[0]].append({'file': item[2], 'reason': 'exact_duplicate'})

    # ---- within-class same-photo SSIM clustering ----
    by_cls = defaultdict(list)
    for item in exact_kept:
        by_cls[item[0]].append(item)

    ssim_dropped = 0
    final = []
    for cls in sorted(by_cls):
        items = sorted(by_cls[cls], key=lambda x: x[2])
        n = len(items)
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

        smalls = [load_gray_small(item[1]) for item in items]
        for i in range(n):
            for j in range(i + 1, n):
                if ssim(smalls[i], smalls[j]) >= SSIM_THRESHOLD:
                    union(i, j)
        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)
        for idxs in clusters.values():
            keep = max(idxs, key=lambda i: res_of(items[i][1]))
            final.append(items[keep])
            for i in idxs:
                if i != keep:
                    ssim_dropped += 1
                    dropped[cls].append(
                        {'file': items[i][2], 'reason': f'same_photo_ssim_{SSIM_THRESHOLD}'})
    log(f'  SSIM same-photo drops: {ssim_dropped}')
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
    log('CAULIFLOWER TRAIN SET PREP (cleanup -> balance -> split -> augment)')
    log('=' * 60)
    rng = rng_seeded(SEED)
    aug = make_augmenter()

    files = load_files()
    raw_counts = Counter(c for c, _p, _f in files)
    log(f'  raw file counts: {dict(raw_counts)}')

    # ---- cleanup (exact-MD5 + SSIM>=0.90 same-photo dedup) ----
    cleaned, dropped = deduplicate(files)
    cleaned_counts = Counter(c for c, _p, _f in cleaned)
    log(f'  dropped (exact / SSIM / conflicts): '
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
        'source': 'Mendeley VegNet (DOI 10.17632/t5sssfgn2v.3)',
        'download_url': 'https://data.mendeley.com/datasets/t5sssfgn2v/3',
        'raw_file_counts': dict(raw_counts),
        'cleaned_file_counts': dict(cleaned_counts),
        'dropped': dict(dropped),
        'dedup_method': 'exact-MD5 (highest-res representative) + within-class '
                        'same-photo SSIM>=0.90 clustering (highest-res representative)',
        'ssim_threshold': SSIM_THRESHOLD,
        'split_strategy': ('file-level stratified random undersample to smallest '
                           'class; 80/10/10 per class, seed 42'),
        'blur_filter': 'disabled (approved: only 1 image below lap_var 30)',
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
