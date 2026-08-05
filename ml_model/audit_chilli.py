"""
Dataset Audit for the Chilli Plant Diseases dataset (ravindubandara3002/chilli-plant-diseases-dataset).

This dataset ships pre-augmented ("Chilli Plant Diseases Dataset(Augmented)"),
so the audit must distinguish REAL images from SYNTHETIC (augmented) copies.
Per the integration plan: keep real images only, no synthetic.

Checks:
  1. Per-split / per-class counts (total, real, augmented-by-name-marker)
  2. Corrupted images (unreadable / truncated JPEG)
  3. Duplicate images (exact MD5) incl. CROSS-SPLIT leakage (train/valid/test overlap)
  4. Perceptual dHash near-duplicates (catches augmented copies w/o name markers)
  5. Blurry images (Laplacian variance)
  6. Image resolution report
  7. Class imbalance (real images only)
  8. Proposed class-name mapping to Leaf Lenz `Chilli___` production names

Usage:
    python ml_model/audit_chilli.py
"""
import os
import sys
import json
import re
import hashlib
from collections import Counter, defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = r'Z:\Chilli Plant Diseases Dataset(Augmented)\Chilli Plant Diseases Dataset'
REPORT = os.path.join(BASE_DIR, 'dataset', 'chilli', 'chilli_audit_report.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.WEBP')
BLUR_LAP_VAR_THRESHOLD = 100.0
NEAR_DUP_DHASH_THRESHOLD = 8
ANALYSIS_MAX_DIM = 512  # downscale before pixel analysis for speed
LOG = []

# Proposed Leaf Lenz production class names
CLASS_MAP = {
    'Chilli__Anthracnos': 'Chilli___anthracnose',
    'Chilli__Damping_Off': 'Chilli___damping_off',
    'Chilli__Leaf_Curl_Virus': 'Chilli___leaf_curl_virus',
    'Chilli__Leaf_Spot': 'Chilli___leaf_spot',
    'Chilli__Veinal_Mottle_Virus': 'Chilli___veinal_mottle_virus',
    'Chilli __Whitefly': 'Chilli___whitefly',
    'Chilli __Yellowish': 'Chilli___yellowish',
    'Chilli___healthy': 'Chilli___healthy',
}

AUG_PREFIX_RE = re.compile(
    r'^(FLIPV|FLIPH|FLIP|ROTATE\d*|ROTATED\d*|BRIGHTNESS\d*|ZOOM\d*|AUG\d*|'
    r'AUGMENTED\d*|RESIZED\d*)[-_]', re.I)
AUG_SUFFIX_RE = re.compile(
    r'_(bright|flipped|rotated|sheared|shifted|zoomed|crop|hflip|vflip|'
    r'rotate\d*|hue|sat|saturation|noise|contrast|scale|resize)_', re.I)


def log(msg):
    LOG.append(str(msg))
    print(msg, flush=True)


def is_augmented_name(fname):
    if AUG_PREFIX_RE.match(fname):
        return True
    if AUG_SUFFIX_RE.search(fname):
        return True
    return False


def load_all_images():
    """Return list of (split, class, abspath, fname)."""
    files = []
    for split in ('train', 'valid', 'test'):
        sroot = os.path.join(SOURCE_DIR, split)
        if not os.path.isdir(sroot):
            continue
        for cls in sorted(os.listdir(sroot)):
            d = os.path.join(sroot, cls)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(IMAGE_EXTS):
                    files.append((split, cls, os.path.join(d, f), f))
    return files


def load_gray_downscaled(path, max_dim=ANALYSIS_MAX_DIM):
    """Decode once to downscaled grayscale numpy array + original PIL size."""
    with Image.open(path) as im:
        orig_size = im.size
        im = im.convert('L')
    arr = np.asarray(im, dtype=np.uint8)
    h, w = arr.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        nw, nh = int(round(w * scale)), int(round(h * scale))
        arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
    return arr, orig_size


def dhash(img_gray, size=8):
    small = cv2.resize(img_gray, (size + 1, size))
    diff = small[:, 1:] > small[:, :-1]
    return sum(1 << i for i, b in enumerate(diff.ravel()) if b)


def audit():
    log('=' * 64)
    log('CHILLI PLANT DISEASES DATASET AUDIT')
    log('=' * 64)

    files = load_all_images()
    log(f'\n[0] Source: {SOURCE_DIR}')
    log(f'    total files loaded: {len(files)}')

    # ---- 1. counts (total / real / augmented) ----
    log('\n[1] Per-split / per-class counts')
    real_files = []
    aug_files = []
    counts = Counter()           # (split, cls)
    real_counts = Counter()      # (split, cls)
    aug_counts = Counter()       # (split, cls)
    real_cls_counts = Counter()  # cls only (across all splits)
    split_counts = Counter()
    for split, cls, path, fname in files:
        counts[(split, cls)] += 1
        split_counts[split] += 1
        if is_augmented_name(fname):
            aug_counts[(split, cls)] += 1
            aug_files.append((split, cls, path, fname))
        else:
            real_counts[(split, cls)] += 1
            real_cls_counts[cls] += 1
            real_files.append((split, cls, path, fname))
    for split in ('train', 'valid', 'test'):
        log(f'  -- {split} (total {split_counts.get(split, 0)}) --')
        for cls in sorted(set(c for s, c in counts if s == split)):
            t = counts.get((split, cls), 0)
            r = real_counts.get((split, cls), 0)
            a = aug_counts.get((split, cls), 0)
            log(f'     {cls:28s} total {t:5d} | real {r:5d} | augmented {a:5d}')

    real_total = len(real_files)
    log(f'\n  real (keepable) images total: {real_total}')
    log(f'  augmented (synthetic) total : {len(aug_files)}')
    nonempty = [c for c in real_cls_counts if real_cls_counts[c] > 0]
    if nonempty:
        rmax = max(real_cls_counts[c] for c in nonempty)
        rmin = min(real_cls_counts[c] for c in nonempty)
        imbalance = {'real_total': real_total,
                     'max_class': rmax, 'min_class': rmin,
                     'max_min_ratio': round(rmax / rmin, 2),
                     'empty_classes': sorted(set(CLASS_MAP) - set(nonempty))}
        log(f'  real-image class imbalance max/min ratio: {imbalance["max_min_ratio"]}')
        log(f'  classes with zero real images: {imbalance["empty_classes"]}')
        for cls in sorted(real_cls_counts):
            log(f'     real {cls:28s} {real_cls_counts[cls]:5d}')
    else:
        imbalance = None

    # ---- 2/6. corrupt + resolution (real images only) ----
    log('\n[2/6] Corrupt detection + resolution report (real images)')
    corrupt = []
    sizes = defaultdict(list)
    gray_cache = {}
    for split, cls, path, fname in real_files:
        try:
            gray, size = load_gray_downscaled(path)
            gray_cache[(split, cls, fname)] = gray
            sizes[cls].append(size)
        except Exception as e:
            corrupt.append({'file': fname, 'class': cls, 'split': split, 'error': str(e)})
    log(f'    corrupt/unreadable: {len(corrupt)}')
    for c in corrupt:
        log(f'      {c}')

    res_report = {}
    for cls in sorted(sizes):
        ws = [s[0] for s in sizes[cls]]
        hs = [s[1] for s in sizes[cls]]
        ws.sort(); hs.sort()
        med = lambda a: a[len(a) // 2]
        res_report[cls] = {
            'count': len(sizes[cls]),
            'min': [min(ws), min(hs)], 'median': [med(ws), med(hs)],
            'max': [max(ws), max(hs)],
            'unique_sizes': sorted(set(sizes[cls]))[:10],
        }
        log(f'    {cls:28s} n {len(sizes[cls]):5d}  min {min(ws)}x{min(hs)}  '
            f'med {med(ws)}x{med(hs)}  max {max(ws)}x{max(hs)}')

    # ---- 3. exact MD5 duplicates + cross-split leakage ----
    log('\n[3] Exact MD5 duplicates (within + cross-split leakage)')
    md5_map = defaultdict(list)
    for split, cls, path, fname in real_files:
        h = hashlib.md5(open(path, 'rb').read()).hexdigest()
        md5_map[h].append((split, cls, fname))
    exact_dup_groups = [g for g in md5_map.values() if len(g) > 1]
    log(f'    exact MD5 duplicate groups: {len(exact_dup_groups)}')
    for g in exact_dup_groups[:10]:
        log(f'      {g}')

    cross_split = []
    for g in md5_map.values():
        splits = set(m[0] for m in g)
        if len(splits) > 1:
            cross_split.append(g)
    log(f'    cross-split leakage groups (same file in >1 split): {len(cross_split)}')
    for g in cross_split[:10]:
        log(f'      {g}')

    # ---- 4. dHash near-duplicates ----
    log('\n[4] Perceptual dHash near-duplicates (real images)')
    hashes = []
    for split, cls, _path, fname in real_files:
        gray = gray_cache[(split, cls, fname)]
        hashes.append((split, cls, fname, dhash(gray)))
    near_groups = []
    seen = set()
    h_arr = np.array([h for _, _, _, h in hashes], dtype=np.uint64)
    bits = np.unpackbits(h_arr.view(np.uint8).reshape(len(hashes), 8), axis=1)
    for i in range(len(hashes)):
        if i in seen:
            continue
        dist = np.count_nonzero(bits[i] != bits, axis=1)
        idx = np.where(dist <= NEAR_DUP_DHASH_THRESHOLD)[0]
        members = [(hashes[j][0], hashes[j][1], hashes[j][2]) for j in idx]
        if len(members) > 1:
            near_groups.append(members)
            seen.update(idx.tolist())
    log(f'    dHash near-duplicate groups (<= {NEAR_DUP_DHASH_THRESHOLD} bits): {len(near_groups)}')
    for g in near_groups[:10]:
        log(f'      {g}')

    # ---- 5. blur ----
    log('\n[5] Blur detection (Laplacian variance < %.0f)' % BLUR_LAP_VAR_THRESHOLD)
    blurry = []
    lap_stats = defaultdict(list)
    for split, cls, _path, fname in real_files:
        gray = gray_cache[(split, cls, fname)]
        v = cv2.Laplacian(gray, cv2.CV_64F).var()
        lap_stats[cls].append(v)
        if v < BLUR_LAP_VAR_THRESHOLD:
            blurry.append({'file': fname, 'class': cls, 'split': split,
                           'lap_var': round(float(v), 1)})
    log(f'    flagged blurry: {len(blurry)}')
    for cls in sorted(lap_stats):
        vs = sorted(lap_stats[cls])
        log(f'    {cls:28s} lap_var min {vs[0]:.1f} med {vs[len(vs)//2]:.1f} max {vs[-1]:.1f}')
    threshold_counts = {}
    for t in (10, 20, 30, 50, 100, 200):
        threshold_counts[t] = sum(1 for vs in lap_stats.values() for v in vs if v < t)
    log(f'    images below lap_var thresholds: {threshold_counts}')

    # ---- 7. report ----
    report = {
        'source': 'kaggle ravindubandara3002/chilli-plant-diseases-dataset (CC0-1.0)',
        'structure': 'Chilli Plant Diseases Dataset(Augmented)/Chilli Plant Diseases Dataset/{train,valid,test}/{class}',
        'n_classes': len(CLASS_MAP),
        'class_map': CLASS_MAP,
        'split_counts': dict(split_counts),
        'per_class_counts': {
            'total': {f'{s}::{c}': counts[(s, c)] for (s, c) in counts},
            'real': {f'{s}::{c}': real_counts[(s, c)] for (s, c) in real_counts},
            'augmented': {f'{s}::{c}': aug_counts[(s, c)] for (s, c) in aug_counts},
        },
        'real_cls_counts': dict(real_cls_counts),
        'real_total': real_total,
        'augmented_total': len(aug_files),
        'class_imbalance_real': imbalance,
        'corrupt_images': corrupt,
        'resolution': res_report,
        'exact_duplicate_groups': [{'members': g, 'n': len(g)} for g in exact_dup_groups],
        'cross_split_leakage_groups': [{'members': g, 'n': len(g)} for g in cross_split],
        'near_duplicate_groups': [{'members': g, 'n': len(g)} for g in near_groups],
        'blurry_threshold_lap_var': BLUR_LAP_VAR_THRESHOLD,
        'blurry_images': blurry,
        'blur_lap_var_threshold_counts': threshold_counts,
        'notes': [
            'Dataset ships pre-augmented: FLIPV/FLIPH/ROTATE*/BRIGHTNESS*/ZOOM* prefixed '
            'files plus _bright_/_flipped_/_rotated_/_sheared_/_shifted_/_zoomed_ suffixed '
            'files are SYNTHETIC copies of the same source image and will be excluded.',
            'Chilli__Veinal_Mottle_Virus contains ONLY augmented copies of a small set of '
            '"Nutrition Deficiency" source images -> ZERO real images for this class. '
            'It must be dropped from training or replaced with real images from elsewhere.',
            'Damping_Off is the rarest real class (~36 images across all splits).',
            'The dataset own train/valid/test splits are leaky: 230 cross-split exact-MD5 '
            'duplicate groups (same image appears in >1 split). A fresh stratified 80/10/10 '
            'split over REAL images only will be used instead.',
            'dHash near-duplicate groups will be kept together when splitting to avoid '
            'train/test leakage from genuine near-duplicates.',
            'Whitefly (insect pest) and Yellowish (symptom) are not fungal diseases; they '
            'are still valid leaf-health classes but disease-card copy should note this.',
        ],
        'leaf_confirmation_limitation': (
            'Cannot visually verify every image is a chilli leaf in this automated audit. '
            'Some images (esp. Damping_Off seedling/soil shots and Whitefly insect shots) '
            'may not be pure leaf close-ups. Manual spot-check is recommended before production.'),
        'log': LOG,
    }
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nAudit report saved: {REPORT}')


if __name__ == '__main__':
    audit()
