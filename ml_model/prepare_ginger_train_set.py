"""
Build the Ginger training/validation/test working set from the audited images of
the GitHub wongjay1941/Ginger-Leaf-Dataset (10,910 images, 4 classes).

Approved plan (2026-08-03):
  * Originals and Roboflow augmented copies are MIXED and NOT filename-separable.
  * SOURCE-LEVEL GROUPING: files are grouped by their stripped base name; every
    copy of the same source photo is kept together and placed in the SAME split,
    so no source ever spans train/valid/test (leakage-free by construction).
  * No blur filtering (lap_var heuristic too aggressive for this low-contrast
    dataset; approved).
  * Balance: undersample each class (whole sources) to the smallest class by file
    count (seed 42).
  * Stratified 80/10/10 split per class at the SOURCE level (file-weighted).
  * Augment TRAIN only: 2 random augmentations per real image, keep 50% (seed 42).
  * Cross-class source-name conflicts are dropped (label conflict, mirrors the
    exact-MD5 dedup policy of prior crops).

Outputs:
  * dataset/ginger/Ginger_Split/{train,valid,test}/ginger___<class>/*.jpg
  * dataset/ginger/ginger_train_prep_report.json

The extracted archive in temp is left untouched.
"""
import os
import re
import sys
import json
import shutil
from collections import Counter, defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = r'C:\Users\amuly\AppData\Local\Temp\opencode\ginger\extract\combined'
GINGER_DIR = os.path.join(BASE_DIR, 'dataset', 'ginger')
SPLIT_DIR = os.path.join(GINGER_DIR, 'Ginger_Split')
REPORT = os.path.join(GINGER_DIR, 'ginger_train_prep_report.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.WEBP')
NEAR_DUP_DHASH_THRESHOLD = 8
ANALYSIS_MAX_DIM = 512
SEED = 42
SPLIT_FRACS = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
AUG_KEEP_RATIO = 0.5
AUGS_PER_IMAGE = 2
JPEG_Q = 95
LOG = []

# Roboflow marker:  BASENAME.rf.<32-hex>.<ext>
RF_AUG_RE = re.compile(r'\.rf\.[0-9a-f]{32}\.')

# Dataset folder name -> Leaf Lenz production class name (lowercase convention)
CLASS_MAP = {
    'Damage-Pest': 'ginger___pest_damage',
    'Dehydrated': 'ginger___dehydrated',
    'Healthy': 'ginger___healthy',
    'Leaf-blight': 'ginger___leaf_blight',
}


def log(msg):
    LOG.append(str(msg))
    print(msg, flush=True)


def rng_seeded(seed):
    return np.random.RandomState(seed)


def base_name(fname):
    """Strip the Roboflow .rf.<32-hex>. marker to recover the source photo name."""
    return RF_AUG_RE.sub('.', fname)


def load_source_groups():
    """
    Return (groups, cross_class_conflicts).
    groups: dict cls -> list of (base, [ (cls, abspath, fname), ... ])
    cross_class_conflicts: dict base -> {cls: [files]} for base names seen in >1 class.
    """
    per_cls = defaultdict(list)
    for cls in CLASS_MAP:
        d = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMAGE_EXTS):
                per_cls[cls].append((cls, os.path.join(d, f), f))

    base_to_cls = defaultdict(lambda: defaultdict(list))
    for cls, items in per_cls.items():
        for item in items:
            base_to_cls[base_name(item[2])][cls].append(item)

    groups = defaultdict(list)
    cross_class = {}
    for base, by_cls in base_to_cls.items():
        if len(by_cls) > 1:
            cross_class[base] = {c: files for c, files in by_cls.items()}
        else:
            (cls, files), = list(by_cls.items())
            groups[cls].append((base, files))
    for cls in groups:
        groups[cls].sort()
    return groups, cross_class


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


def balance_sources(groups, rng):
    """
    Undersample whole sources per class so each class's FILE count approaches the
    smallest class file count. Returns (balanced_groups, per_class_targets, n_min).
    """
    totals = {cls: sum(len(files) for _b, files in g) for cls, g in groups.items()}
    n_min = min(totals.values())
    balanced = {}
    targets = {}
    for cls in sorted(groups):
        g = sorted(groups[cls], key=lambda x: len(x[1]))
        total = totals[cls]
        if total <= n_min:
            balanced[cls] = g
            targets[cls] = total
            continue
        rng.shuffle(g)
        chosen = []
        used = 0
        for base, files in g:
            if used + len(files) <= n_min:
                chosen.append((base, files))
                used += len(files)
        # if we undershot badly, allow the smallest remaining source to fill up
        remaining = [x for x in g if x not in chosen]
        if used < n_min and remaining:
            best = min(remaining, key=lambda x: abs(n_min - (used + len(x))))
            chosen.append(best)
            used += len(best[1])
        balanced[cls] = chosen
        targets[cls] = used
    return balanced, targets, n_min


def assign_source_split(balanced, rng):
    """
    File-weighted stratified split of whole SOURCE groups per class.
    Returns (split -> [(cls, path, fname)...], split_counts).
    """
    splits = {s: [] for s in SPLIT_FRACS}
    split_counts = {s: Counter() for s in SPLIT_FRACS}
    for cls, groups in sorted(balanced.items()):
        total = sum(len(files) for _b, files in groups)
        n_train = int(round(SPLIT_FRACS['train'] * total))
        n_valid = int(round(SPLIT_FRACS['valid'] * total))
        rng.shuffle(groups)
        assigned = {'train': [], 'valid': [], 'test': []}
        used = 0
        for base, files in groups:
            n = len(files)
            if used < n_train:
                assigned['train'].append((base, files))
            elif used < n_train + n_valid:
                assigned['valid'].append((base, files))
            else:
                assigned['test'].append((base, files))
            used += n
        for s in SPLIT_FRACS:
            for base, files in assigned[s]:
                for item in files:
                    splits[s].append(item)
                    split_counts[s][item[0]] += 1
    return splits, split_counts


def leakage_report_source_level(splits):
    """dHash similarity of TEST images to TRAIN images from OTHER sources."""
    train_by_cls = defaultdict(list)
    for cls, path, fname in splits['train']:
        train_by_cls[cls].append((fname, path))
    train_hash_by_cls = {}
    for cls, items in train_by_cls.items():
        arr = np.array([dhash(load_downscaled_gray(p)) for _f, p in items], dtype=np.uint64)
        train_hash_by_cls[cls] = (items, arr)
    test_items = splits['test']
    n_test = len(test_items)
    close = 0
    min_dists = []
    for cls, path, fname in test_items:
        base = base_name(fname)
        items, arr = train_hash_by_cls.get(cls, ([], np.array([], dtype=np.uint64)))
        if items:
            mask = np.array([base_name(f) != base for f, _p in items])
            arr = arr[mask]
        if arr.size == 0:
            continue
        hb = np.unpackbits(
            np.array([dhash(load_downscaled_gray(path))], dtype=np.uint64)
            .view(np.uint8).reshape(1, 8), axis=1)
        tb = np.unpackbits(arr.view(np.uint8).reshape(arr.size, 8), axis=1)
        d = np.count_nonzero(tb != hb, axis=1)
        md = int(d.min())
        min_dists.append(md)
        if md <= NEAR_DUP_DHASH_THRESHOLD:
            close += 1
    report = {
        'test_images_checked': n_test,
        'test_images_with_close_train_match_from_other_source_leq_8bits': close,
        'min_dhash_dist_min': min(min_dists) if min_dists else None,
        'min_dhash_dist_median': int(np.median(min_dists)) if min_dists else None,
        'note': ('Split is leakage-free at the SOURCE level: all copies of each source '
                 'photo stay in one split. This report documents near-duplicate '
                 'similarity between DIFFERENT sources that may span splits.'),
    }
    return report


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


def read_downscaled_rgb(path):
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
    log('GINGER TRAIN SET PREP (source-group -> balance -> source-split -> augment)')
    log('=' * 60)
    rng = rng_seeded(SEED)
    aug = make_augmenter()

    # ---- load sources ----
    groups, cross_class = load_source_groups()
    raw_counts = {cls: sum(len(f) for _b, f in g) for cls, g in groups.items()}
    raw_sources = {cls: len(g) for cls, g in groups.items()}
    log(f'  raw file counts: {raw_counts}')
    log(f'  raw source counts: {raw_sources}')
    for base, by_cls in cross_class.items():
        log(f'  CROSS-CLASS SOURCE CONFLICT (dropped): {base} in '
            f'{sorted(by_cls)} ({[len(f) for f in by_cls.values()]})')

    # ---- balance (whole sources) ----
    balanced, targets, n_min = balance_sources(groups, rng)
    log(f'  balance target (smallest class, files): {n_min}')
    for cls in sorted(balanced):
        n = sum(len(f) for _b, f in balanced[cls])
        log(f'  balanced {cls:12s} sources {len(balanced[cls]):5d} files {n}')

    # ---- source-level stratified split ----
    splits, split_counts = assign_source_split(balanced, rng)
    log('  source-level file-weighted stratified 80/10/10 split')
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
        img = read_downscaled_rgb(path)
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

    # ---- leakage doc (source-level) ----
    leakage_report = leakage_report_source_level(splits)
    log(f'  split leakage doc: {leakage_report}')

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
        'source': 'GitHub wongjay1941/Ginger-Leaf-Dataset',
        'raw_file_counts': raw_counts,
        'raw_source_counts': raw_sources,
        'cross_class_source_conflicts_dropped': cross_class,
        'split_strategy': ('source-level grouping: all copies of a source stay in the '
                           'same split; balanced by whole-source undersample to smallest '
                           'class file count; file-weighted stratified 80/10/10 per class'),
        'blur_filter': 'disabled (approved: lap_var heuristic too aggressive for this dataset)',
        'balance_target_files': n_min,
        'balanced_file_counts': {cls: sum(len(f) for _b, f in g) for cls, g in balanced.items()},
        'balanced_source_counts': {cls: len(g) for cls, g in balanced.items()},
        'split_fractions': SPLIT_FRACS,
        'split_counts': {s: dict(c) for s, c in split_counts.items()},
        'split_leakage_report': leakage_report,
        'augments_per_image': AUGS_PER_IMAGE,
        'aug_keep_ratio': AUG_KEEP_RATIO,
        'augmented_generated': dict(gen_per_class),
        'disk_counts': disk_counts,
        'seed': SEED,
        'log': LOG,
    }
    # compute on-disk source counts per split
    disk_sources = {}
    for s in SPLIT_FRACS:
        disk_sources[s] = {}
        for cls in CLASS_MAP.values():
            d = os.path.join(SPLIT_DIR, s, cls)
            if os.path.isdir(d):
                disk_sources[s][cls] = len({base_name(f) for f in os.listdir(d)
                                            if '__aug' not in f})
            else:
                disk_sources[s][cls] = 0
    report['split_source_counts'] = disk_sources

    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nReport saved: {REPORT}')


if __name__ == '__main__':
    main()
