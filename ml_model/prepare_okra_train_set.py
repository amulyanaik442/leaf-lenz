"""
Build the Okra training/validation/test working set from the audited images of
the Okra DiseaseNet (Mendeley) dataset (1,495 images, 6 classes).

Approved plan (audit 2026-08-05):
  * Raw dataset (original photographs; no augmented images; annotations are
    segmentation-only and orphaned - unused).
  * Exact-MD5 dedup: keep ONE copy per duplicate group (no cross-class label
    conflicts expected for this dataset; if any appear, they are dropped).
  * No blur filter (threshold heuristic; see audit).
  * Balance: random undersample each class to the smallest class (seed 42).
  * Stratified 80/10/10 split per class at the FILE level (seed 42).
  * Augment TRAIN only: 2 random augmentations per real image, keep 50% (seed 42).
  * The authors' own 75/15/10 split is unmappable (see audit) and is ignored;
    a fresh split is created instead.

Outputs:
  * dataset/okra/Okra_Split/{train,valid,test}/okra___<class>/*.jpg
  * dataset/okra/okra_train_prep_report.json

The downloaded raw images in dataset/okra/raw are left untouched.
"""
import os
import sys
import json
import argparse
import hashlib
import shutil
from collections import Counter, defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(BASE_DIR, 'dataset', 'okra', 'raw')
OKRA_DIR = os.path.join(BASE_DIR, 'dataset', 'okra')
SPLIT_DIR = os.path.join(OKRA_DIR, 'Okra_Split')
REPORT = os.path.join(OKRA_DIR, 'okra_train_prep_report.json')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.WEBP')
SEED = 42
SPLIT_FRACS = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
AUG_KEEP_RATIO = 0.5
AUGS_PER_IMAGE = 2
JPEG_Q = 95
LOG = []

# Raw class folder -> Leaf Lenz production class name
CLASS_MAP = {
    'ALS': 'okra___alternaria_leaf_spot',
    'CLS': 'okra___cercospora_leaf_spot',
    'DM': 'okra___downy_mildew',
    'H': 'okra___healthy',
    'LCV': 'okra___leaf_curl_virus',
    'PLS': 'okra___phyllosticta_leaf_spot',
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
    Exact-MD5 dedup keeping one copy per group; cross-class label conflicts
    (if any) drop all but one copy.
    Returns (kept, dropped_reasons: dict cls -> list of {file, reason}).
    """
    dropped = defaultdict(list)
    groups = defaultdict(list)
    for item in files:
        groups[md5_of(item[1])].append(item)

    final = []
    for h, g in groups.items():
        classes = set(x[0] for x in g)
        if len(classes) > 1:
            keep = sorted(g, key=lambda x: x[0])[0]
            final.append(keep)
            for item in g:
                if item is not keep:
                    dropped[item[0]].append(
                        {'file': item[2],
                         'reason': f'cross_class_label_conflict_kept_{keep[0]}'})
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true',
                        help='use ALL cleaned images (skip random undersample)')
    parser.add_argument('--split-dir', default='Okra_Split',
                        help='output split folder name under dataset/okra/ '
                             '(default: Okra_Split)')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--report', default='',
                        help='report filename override (default: okra_train_prep_report.json)')
    args = parser.parse_args()

    global SPLIT_DIR, REPORT
    SPLIT_DIR = os.path.join(OKRA_DIR, args.split_dir)
    REPORT = os.path.join(OKRA_DIR, args.report or
                          (f'okra_train_prep_report_{args.split_dir}.json'
                           if args.split_dir != 'Okra_Split'
                           else 'okra_train_prep_report.json'))
    if args.full:
        SPLIT_DIR = os.path.join(SPLIT_DIR + '_full')

    log('=' * 60)
    log(f'OKRA TRAIN SET PREP ({"full" if args.full else "balanced"} -> split -> augment)')
    log('=' * 60)
    rng = rng_seeded(args.seed)
    aug = make_augmenter()

    files = load_files()
    raw_counts = Counter(c for c, _p, _f in files)
    log(f'  raw file counts: {dict(raw_counts)}')

    # ---- cleanup (MD5 dedup) ----
    cleaned, dropped = deduplicate(files)
    cleaned_counts = Counter(c for c, _p, _f in cleaned)
    log(f'  dropped duplicates / conflicts: '
        f'{dict((k, len(v)) for k, v in dropped.items())}')
    log(f'  cleaned file counts: {dict(cleaned_counts)}')

    # ---- balance (random undersample) or use all cleaned images ----
    if args.full:
        balanced = list(cleaned)
        n_min = None
        log('  balance: NONE (using all cleaned images)')
    else:
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
        'source': 'Okra DiseaseNet (Mendeley Data, DOI 10.17632/nh7zk4hv8z, v1)',
        'download_url': 'https://data.mendeley.com/datasets/nh7zk4hv8z/1',
        'raw_file_counts': dict(raw_counts),
        'cleaned_file_counts': dict(cleaned_counts),
        'dropped': dict(dropped),
        'dedup_policy': ('exact-MD5 dedup, keep one copy per group; '
                         'cross-class conflicts drop all but one'),
        'split_strategy': ('file-level stratified; 80/10/10 per class, seed %d; '
                           'balance = %s' %
                           (args.seed, 'none (all cleaned images, --full)' if args.full
                            else 'random undersample to smallest class')),
        'blur_filter': 'disabled (threshold heuristic; see audit)',
        'balance_target_files': n_min,
        'balanced_file_counts': dict(Counter(c for c, _p, _f in balanced)),
        'split_fractions': SPLIT_FRACS,
        'split_counts': {s: dict(c) for s, c in split_counts.items()},
        'augments_per_image': AUGS_PER_IMAGE,
        'aug_keep_ratio': AUG_KEEP_RATIO,
        'augmented_generated': dict(gen_per_class),
        'disk_counts': disk_counts,
        'seed': args.seed,
        'log': LOG,
    }
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nReport saved: {REPORT}')


if __name__ == '__main__':
    main()
