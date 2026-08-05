"""
Build the Cacao leaf training/validation/test working set from the audited crops.

Steps (from the integration plan, seed-fixed for reproducibility):
  1. Read the crop manifest + audit reports.
  2. Blur filter: drop crops with Laplacian variance < 20 (recommended cutoff).
  3. Balance: undersample the majority class to the minority class (seed 42).
  4. Leakage-free split: assign whole SOURCE images to train/valid/test
     (80/10/10) with a greedy class-balanced assignment (seed 42).
  5. Augment the TRAIN portion only: 2 random augmentations per train image,
     then keep 50% of the generated variants (seed 42) -> ~50% of train size.

Outputs:
  * dataset/cacao_leaf/Cacao_Split/{train,valid,test}/{healthy,anthracnose}/*.jpg
  * dataset/cacao_leaf/cacao_train_prep_report.json

Raw_Classification/ is left untouched (raw archive).
"""
import os
import sys
import json
import copy
import shutil
from collections import Counter, defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACAO_DIR = os.path.join(BASE_DIR, 'dataset', 'cacao_leaf')
MANIFEST = os.path.join(CACAO_DIR, 'cacao_leaf_manifest.json')
SPLIT_DIR = os.path.join(CACAO_DIR, 'Cacao_Split')
REPORT = os.path.join(CACAO_DIR, 'cacao_train_prep_report.json')

BLUR_LAP_VAR_THRESHOLD = 20.0
SEED = 42
SPLIT_FRACS = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
AUG_KEEP_RATIO = 0.5
AUGS_PER_IMAGE = 2
JPEG_Q = 95
LOG = []


def log(msg):
    LOG.append(str(msg))
    print(msg, flush=True)


def rng_seeded(seed):
    return np.random.RandomState(seed)


def blur_removal(crops):
    """crops: list of dicts {crop_path, source_image, class}. Returns filtered + report."""
    removed = []
    kept = []
    for c in crops:
        abs_path = os.path.join(BASE_DIR, 'dataset', 'cacao_leaf', c['crop_path'])
        gray = cv2.cvtColor(cv2.imread(abs_path), cv2.COLOR_BGR2GRAY)
        v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        c = dict(c)
        c['lap_var'] = v
        if v < BLUR_LAP_VAR_THRESHOLD:
            removed.append(c)
        else:
            kept.append(c)
    return kept, removed


def balance(crops, rng):
    counts = Counter(c['class'] for c in crops)
    n_min = min(counts.values())
    per_class_kept = {}
    result = []
    for cls in sorted(counts):
        members = [c for c in crops if c['class'] == cls]
        rng.shuffle(members)
        chosen = members[:n_min]
        per_class_kept[cls] = len(chosen)
        result.extend(chosen)
    return result, per_class_kept


def assign_split(groups, rng):
    """Assign whole source-image groups to splits keeping per-class balance.

    Groups are processed in random (seeded) order.  Each group goes to the
    split that is proportionally most empty for the classes it contains,
    using fill = count / (target_frac * class_total), capped by an overshoot
    penalty.  This keeps train ~80% and valid/test ~10% per class while never
    leaking a source image across splits.
    """
    keys = list(groups.keys())
    rng.shuffle(keys)
    total_per_class = Counter()
    for k in keys:
        for _, cls in groups[k]:
            total_per_class[cls] += 1

    needed = {s: {cls: SPLIT_FRACS[s] * tot for cls, tot in total_per_class.items()}
              for s in SPLIT_FRACS}
    counts = {s: Counter() for s in SPLIT_FRACS}
    assignment = {}

    for k in keys:
        cls_counts = Counter(cls for _, cls in groups[k])
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


def save_crops(crops, split, cls, dest_root, rng=None, aug=None):
    """Copy (or augment+write) crops into dest_root/split/cls/."""
    out_dir = os.path.join(dest_root, split, cls)
    os.makedirs(out_dir, exist_ok=True)
    for c in crops:
        src = os.path.join(BASE_DIR, 'dataset', 'cacao_leaf', c['crop_path'])
        fname = os.path.basename(c['crop_path'])
        dst = os.path.join(out_dir, fname)
        shutil.copy2(src, dst)
        if aug is not None:
            img = cv2.imread(src)
            for i in range(AUGS_PER_IMAGE):
                av = aug(c, img, rng)
                stem = os.path.splitext(fname)[0]
                cv2.imwrite(os.path.join(out_dir, f'{stem}__aug{i:02d}.jpg'),
                            av, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])


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


def main():
    log('=' * 60)
    log('CACAO TRAIN SET PREP (blur -> balance -> split -> augment)')
    log('=' * 60)
    rng = rng_seeded(SEED)
    aug = make_augmenter()

    with open(MANIFEST) as f:
        manifest = json.load(f)
    log(f'  manifest crops: {len(manifest)}')

    # ---- blur filter ----
    kept, removed = blur_removal(manifest)
    rem_by_cls = Counter(c['class'] for c in removed)
    log(f'  blur removal (< lap_var {BLUR_LAP_VAR_THRESHOLD}): {len(removed)}'
        f'  {dict(rem_by_cls)}')
    kept_by_cls = Counter(c['class'] for c in kept)
    log(f'  kept after blur: {len(kept)}  {dict(kept_by_cls)}')

    # ---- balance ----
    balanced, per_class_kept = balance(kept, rng)
    log(f'  after balance: {len(balanced)}  {per_class_kept}')

    # ---- leakage-free split ----
    groups = defaultdict(list)
    for c in balanced:
        groups[c['source_image']].append((c['crop_path'], c['class']))
    log(f'  source images: {len(groups)}')
    assignment, counts = assign_split(groups, rng)
    split_counts = {}
    for s in SPLIT_FRACS:
        split_counts[s] = dict(counts[s])
        log(f'  split {s:5s} crops: {sum(counts[s].values())}  {dict(counts[s])}')

    # ---- write real crops ----
    for split in SPLIT_FRACS:
        for cls in sorted(per_class_kept):
            selected = [c for c in balanced
                        if assignment[c['source_image']] == split
                        and c['class'] == cls]
            save_crops(selected, split, cls, SPLIT_DIR)
            log(f'  wrote {split:5s}/{cls:12s} {len(selected)}')

    # ---- augment train only ----
    train_real = [c for c in balanced if assignment[c['source_image']] == 'train']
    gen_per_class = Counter()
    aug_pool = []  # (crop, aug_idx)
    for c in train_real:
        for i in range(AUGS_PER_IMAGE):
            aug_pool.append((c, i))
    rng.shuffle(aug_pool)
    keep_n = int(round(len(aug_pool) * AUG_KEEP_RATIO))
    aug_keep = aug_pool[:keep_n]
    for c, i in aug_keep:
        cls = c['class']
        gen_per_class[cls] += 1
        src = os.path.join(BASE_DIR, 'dataset', 'cacao_leaf', c['crop_path'])
        img = cv2.imread(src)
        av = aug(c, img, rng)
        out_dir = os.path.join(SPLIT_DIR, 'train', cls)
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.basename(c['crop_path'])
        stem = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_dir, f'{stem}__aug{i:02d}.jpg'), av,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    log(f'  augmented train images generated: {keep_n}  {dict(gen_per_class)}')

    # ---- final counts on disk ----
    disk_counts = {}
    for split in SPLIT_FRACS:
        disk_counts[split] = {}
        for cls in sorted(per_class_kept):
            d = os.path.join(SPLIT_DIR, split, cls)
            disk_counts[split][cls] = len(os.listdir(d))
        log(f'  on-disk {split:5s}: {sum(disk_counts[split].values())}  {disk_counts[split]}')

    report = {
        'manifest_crops': len(manifest),
        'blur_threshold_lap_var': BLUR_LAP_VAR_THRESHOLD,
        'blur_removed': len(removed),
        'blur_removed_by_class': dict(rem_by_cls),
        'kept_after_blur': dict(kept_by_cls),
        'balanced_per_class': per_class_kept,
        'split_fractions': SPLIT_FRACS,
        'split_crops': split_counts,
        'augments_per_image': AUGS_PER_IMAGE,
        'aug_keep_ratio': AUG_KEEP_RATIO,
        'augmented_generated': dict(gen_per_class),
        'disk_counts': disk_counts,
        'seed': SEED,
        'blur_removed_files': [c['crop_path'] for c in removed],
        'log': LOG,
    }
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nReport saved: {REPORT}')


if __name__ == '__main__':
    main()
