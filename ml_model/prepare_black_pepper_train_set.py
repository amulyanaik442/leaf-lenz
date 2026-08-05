"""
Build the Black Pepper leaf training/validation/test working set from the audited
images of the Kaggle udi17live dataset (819 images, 3 classes, 273 each).

Approved plan: 3 classes (healthy, leaf_blight, yellow_mottle_virus), real images
only (no synthetic), balanced by undersample (seed 42). Mirrors the verified
Turmeric/Chilli pipeline exactly.

Steps (seed-fixed for reproducibility):
  1. Enumerate images from the extracted dataset.
  2. Deduplicate exact MD5 copies (cross-class conflict -> drop all; within
     class -> keep 1 representative).
  3. Blur filter (Laplacian variance < 100, mirrors chilli/turmeric).
  4. Balance: undersample each class to the minority class count (seed 42).
  5. APPROVED (2026-08-03): plain STRATIFIED 80/10/10 split per class (seed 42),
     NOT leakage-group assignment. This dataset's same-class dHash near-dup
     components are huge (one ~230-266 member component per class), so
     whole-component assignment could not yield a balanced split. Mild
     near-dup similarity may span train/test; a dHash similarity report of
     the split is generated to document the residual leakage risk.
  6. Augment the TRAIN portion only: 2 random augmentations per train image,
     then keep 50% of the generated variants (seed 42).

Outputs:
  * dataset/black_pepper/BlackPepper_Split/{train,valid,test}/BlackPepper___<class>/*.jpg
  * dataset/black_pepper/black_pepper_train_prep_report.json

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
SOURCE_DIR = r'C:\Users\amuly\AppData\Local\Temp\opencode\black_pepper\extract'
PEPPER_DIR = os.path.join(BASE_DIR, 'dataset', 'black_pepper')
SPLIT_DIR = os.path.join(PEPPER_DIR, 'BlackPepper_Split')
REPORT = os.path.join(PEPPER_DIR, 'black_pepper_train_prep_report.json')

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
    'black_pepper_healthy': 'BlackPepper___healthy',
    'black_pepper_leaf_blight': 'BlackPepper___leaf_blight',
    'black_pepper_yellow_mottle_virus': 'BlackPepper___yellow_mottle_virus',
}


def log(msg):
    LOG.append(str(msg))
    print(msg, flush=True)


def rng_seeded(seed):
    return np.random.RandomState(seed)


def load_original_images():
    """Return list of (cls, abspath, fname) for images of the 3 classes."""
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


def assign_split_stratified(balanced, rng):
    """Plain stratified split per class: 80/10/10. Returns (split -> [(cls,path,fname)])."""
    by_class = defaultdict(list)
    for item in balanced:
        by_class[item[0]].append(item)
    splits = {s: [] for s in SPLIT_FRACS}
    per_split_class = {s: Counter() for s in SPLIT_FRACS}
    for cls, members in by_class.items():
        rng.shuffle(members)
        n = len(members)
        n_train = int(round(SPLIT_FRACS['train'] * n))
        n_valid = int(round(SPLIT_FRACS['valid'] * n))
        splits['train'].extend(members[:n_train])
        splits['valid'].extend(members[n_train:n_train + n_valid])
        splits['test'].extend(members[n_train + n_valid:])
    for s in SPLIT_FRACS:
        for cls, _p, _f in splits[s]:
            per_split_class[s][cls] += 1
    return splits, per_split_class


def split_leakage_report(balanced, splits, hashes, kept):
    """dHash similarity between test images and train images (documentation only)."""
    # hashes are aligned with the pre-balance `kept` list; map by unique (cls, path)
    idx = {}
    for i, (cls, path, fname) in enumerate(kept):
        idx.setdefault((cls, path), i)
    train_hashes = {}
    for cls, path, fname in splits['train']:
        train_hashes.setdefault(cls, []).append(hashes[idx[(cls, path)]])
    n_test, close = 0, 0
    min_dists = []
    for cls, path, fname in splits['test']:
        th = np.array(train_hashes.get(cls, []), dtype=np.uint64)
        if th.size == 0:
            continue
        tb = np.unpackbits(th.view(np.uint8).reshape(len(th), 8), axis=1)
        hb = np.unpackbits(np.array([hashes[idx[(cls, path)]]], dtype=np.uint64)
                           .view(np.uint8).reshape(1, 8), axis=1)
        d = np.count_nonzero(tb != hb, axis=1)
        md = int(d.min())
        min_dists.append(md)
        n_test += 1
        if md <= NEAR_DUP_DHASH_THRESHOLD:
            close += 1
    report = {
        'test_images_checked': n_test,
        'test_images_with_a_close_train_match_leq_8bits': close,
        'min_dhash_dist_min': min(min_dists) if min_dists else None,
        'min_dhash_dist_median': int(np.median(min_dists)) if min_dists else None,
        'note': ('Leakage is DOCUMENTED, not eliminated. Stratified split was approved; '
                 'near-duplicate images may span train/test because within-class dHash '
                 'similarity is very high in this dataset.'),
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
    log('BLACK PEPPER TRAIN SET PREP (dedup -> blur -> balance -> split -> augment)')
    log('=' * 60)
    rng = rng_seeded(SEED)
    aug = make_augmenter()

    files = load_original_images()
    raw_counts = Counter(cls for cls, _, _ in files)
    log(f'  original images (3 classes): {len(files)}  {dict(raw_counts)}')

    # ---- dedup ----
    kept, dropped = dedup(files)
    log(f'  after exact-MD5 dedup: {len(kept)}')
    log(f'  dropped cross-class conflicts: {len(dropped)}')
    for d in dropped:
        log(f'      {d["reason"]}: {d["files"]}')

    # ---- blur filter ----
    kept, blur_removed = blur_filter(kept)
    log(f'  blur removal (< lap_var {BLUR_LAP_VAR_THRESHOLD}): {len(blur_removed)}')

    # ---- leakage components (for documentation only) ----
    components, hashes = build_components(kept)
    log(f'  dHash near-dup leakage components: {len(components)} '
        f'(max group {max(len(g) for g in components)})')
    comp_sizes = Counter(len(g) for g in components)
    log(f'  component size histogram: {dict(sorted(comp_sizes.items()))}')

    # ---- balance ----
    balanced, per_class_kept, n_min = balance(kept, rng)
    log(f'  balance target (min class): {n_min}  per class: {per_class_kept}')
    log(f'  balanced total: {len(balanced)}')

    # ---- APPROVED stratified split (not leakage-group assignment) ----
    splits, split_counts = assign_split_stratified(balanced, rng)
    log('  stratified 80/10/10 split (approved; leakage documented not eliminated)')
    for s in SPLIT_FRACS:
        split_counts[s] = dict(split_counts[s])
        log(f'  split {s:5s}: {len(splits[s])}  {dict(split_counts[s])}')

    # ---- write real images ----
    written = 0
    for s in SPLIT_FRACS:
        for cls, path, fname in splits[s]:
            out_dir = os.path.join(SPLIT_DIR, s, CLASS_MAP[cls])
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
            log(f'  wrote {s:5s}/{CLASS_MAP[cls]:30s} {n}')

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

    # ---- leakage documentation for the chosen split ----
    leakage_report = split_leakage_report(balanced, splits, hashes, kept)
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
        'split_type': 'stratified_80_10_10_per_class',
        'split_counts': split_counts,
        'split_leakage_report': leakage_report,
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
