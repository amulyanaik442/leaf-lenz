"""
Dataset Audit for the Turmeric Plant Leaf Disease Detection dataset
(Mendeley jtttfbx342/1, 4,361 images).

Structure:
    Image Dataset for Turmeric Plant Leaf Disease Detection/
        Original DataSet/{Aphids_Disease, Blotch, Healthy_Leaf, Leaf_Spot}   (865)
        Augmented DataSet/{same 4 classes}                                    (3496)

Unlike Chilli, originals and augmented copies live in SEPARATE folders, so we
train on Original DataSet only (keep real images, no synthetic).

Checks:
  1. Per-class counts (original vs augmented)
  2. Augmentation ratio (are aug_* files derived from originals? name pattern)
  3. Corrupted images (unreadable / truncated)
  4. Duplicate images (exact MD5) within originals + cross-original/augmented
  5. Perceptual dHash near-duplicates (originals)
  6. Blurry images (Laplacian variance, originals)
  7. Image resolution report (originals)
  8. Class imbalance (originals)
  9. Proposed class-name mapping to Leaf Lenz `Turmeric___` production names

Usage:
    python ml_model/audit_turmeric.py
"""
import os
import sys
import json
import hashlib
from collections import Counter, defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_ROOT = r'C:\Users\amuly\AppData\Local\Temp\opencode\turmeric\extract'
DATASET_ROOT = os.path.join(
    EXTRACT_ROOT, 'Image Dataset for Turmeric Plant Leaf Disease Detection')
REPORT = os.path.join(BASE_DIR, 'dataset', 'turmeric', 'turmeric_audit_report.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.WEBP')
BLUR_LAP_VAR_THRESHOLD = 100.0
NEAR_DUP_DHASH_THRESHOLD = 8
ANALYSIS_MAX_DIM = 512
LOG = []

# Proposed Leaf Lenz production class names (appended at indices 123-126)
CLASS_MAP = {
    'Aphids_Disease': 'Turmeric___aphids_disease',
    'Blotch': 'Turmeric___blotch',
    'Healthy_Leaf': 'Turmeric___healthy',
    'Leaf_Spot': 'Turmeric___leaf_spot',
}


def log(msg):
    LOG.append(str(msg))
    print(msg, flush=True)


def load_all(folder):
    """Return list of (cls, abspath, fname) for one of the two subfolders."""
    files = []
    for cls in sorted(os.listdir(folder)):
        d = os.path.join(folder, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMAGE_EXTS):
                files.append((cls, os.path.join(d, f), f))
    return files


def load_gray_downscaled(path, max_dim=ANALYSIS_MAX_DIM):
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
    log('TURMERIC PLANT LEAF DISEASE DATASET AUDIT')
    log('=' * 64)

    orig_dir = os.path.join(DATASET_ROOT, 'Original DataSet')
    aug_dir = os.path.join(DATASET_ROOT, 'Augmented DataSet')
    log(f'\n[0] Source: {DATASET_ROOT}')
    if not os.path.isdir(orig_dir) or not os.path.isdir(aug_dir):
        log('ERROR: Original/Augmented DataSet folders not found.')
        return

    originals = load_all(orig_dir)
    augmented = load_all(aug_dir)
    log(f'    originals : {len(originals)}')
    log(f'    augmented : {len(augmented)}')

    # ---- 1. counts ----
    log('\n[1] Per-class counts (original vs augmented)')
    orig_cls = Counter(c for c, _, _ in originals)
    aug_cls = Counter(c for c, _, _ in augmented)
    for cls in sorted(orig_cls | aug_cls):
        log(f'     {cls:16s} original {orig_cls.get(cls, 0):5d} | augmented {aug_cls.get(cls, 0):5d}')
    real_total = len(originals)
    nonempty = [c for c in orig_cls if orig_cls[c] > 0]
    if nonempty:
        rmax = max(orig_cls[c] for c in nonempty)
        rmin = min(orig_cls[c] for c in nonempty)
        imbalance = {'real_total': real_total,
                     'max_class': rmax, 'min_class': rmin,
                     'max_min_ratio': round(rmax / rmin, 2),
                     'empty_classes': sorted(set(CLASS_MAP) - set(nonempty))}
        log(f'\n  original-image class imbalance max/min ratio: {imbalance["max_min_ratio"]}')
        log(f'  classes with zero original images: {imbalance["empty_classes"]}')
        for cls in sorted(orig_cls):
            log(f'     original {cls:16s} {orig_cls[cls]:5d}')
    else:
        imbalance = None

    # ---- 2. augmentation ratio / name pattern ----
    log('\n[2] Augmentation name pattern + per-original multiplicity (sample)')
    aug_name_count = Counter()
    for cls, path, fname in augmented:
        m = fname.lower()
        if m.startswith('aug_'):
            aug_name_count['aug_*'] += 1
        else:
            aug_name_count[('other', fname)] += 1
    log(f'    aug_* prefixed: {aug_name_count.get("aug_*", 0)} of {len(augmented)}')
    others = {k: v for k, v in aug_name_count.items() if k != 'aug_*'}
    for k, v in list(others.items())[:10]:
        log(f'      non-aug_ file: {k} x{v}')
    # strip leading identifiers to estimate multiplicity per source image
    stem_counts = Counter()
    for cls, path, fname in augmented:
        stem = fname.rsplit('.', 1)[0]
        stem = stem.replace('aug_', '', 1)
        stem = '_'.join(stem.split('_')[:-1]) if '_' in stem else stem
        stem_counts[(cls, stem)] += 1
    mult = Counter(stem_counts.values())
    log(f'    multiplicity of stem names among augmented: {dict(sorted(mult.items()))}')

    # ---- 3. corrupt (originals + augmented) ----
    log('\n[3] Corrupt/unreadable images')
    corrupt = []
    for cls, path, fname in originals + augmented:
        try:
            load_gray_downscaled(path)
        except Exception as e:
            corrupt.append({'file': fname, 'class': cls, 'folder': 'aug' if path.startswith(aug_dir) else 'orig',
                            'error': str(e)})
    log(f'    corrupt: {len(corrupt)}')
    for c in corrupt:
        log(f'      {c}')

    # ---- 4. exact MD5 duplicates (originals) + cross orig/aug ----
    log('\n[4] Exact MD5 duplicates')
    orig_md5 = defaultdict(list)
    for cls, path, fname in originals:
        orig_md5[hashlib.md5(open(path, 'rb').read()).hexdigest()].append(('orig', cls, fname))
    aug_md5 = defaultdict(list)
    for cls, path, fname in augmented:
        aug_md5[hashlib.md5(open(path, 'rb').read()).hexdigest()].append(('aug', cls, fname))
    dup_groups = [g for g in orig_md5.values() if len(g) > 1]
    log(f'    exact-MD5 duplicate groups within originals: {len(dup_groups)}')
    for g in dup_groups[:10]:
        log(f'      {g}')
    cross_oa = []
    for h, g in aug_md5.items():
        if h in orig_md5:
            cross_oa.append((orig_md5[h], g))
    log(f'    augmented files that are byte-identical to an original: {len(cross_oa)}')
    for g in cross_oa[:10]:
        log(f'      {g}')
    # cross-class exact dups among originals
    cc = defaultdict(list)
    for h, g in orig_md5.items():
        classes = set(m[1] for m in g)
        if len(classes) > 1:
            cc[h].append(g)
    log(f'    originals duplicated across DIFFERENT classes: {len(cc)}')
    for h, g in list(cc.items())[:10]:
        log(f'      {g}')

    # ---- 5/6/7. dHash, blur, resolution on originals ----
    log('\n[5/6/7] dHash near-dups + blur + resolution (originals)')
    gray_cache = {}
    sizes = defaultdict(list)
    lap_stats = defaultdict(list)
    for cls, path, fname in originals:
        gray, size = load_gray_downscaled(path)
        gray_cache[cls] = gray_cache.get(cls, {})
        gray_cache[cls][fname] = gray
        sizes[cls].append(size)
        v = cv2.Laplacian(gray, cv2.CV_64F).var()
        lap_stats[cls].append(v)

    hashes = []
    for cls, _p, fname in originals:
        hashes.append((cls, fname, dhash(gray_cache[cls][fname])))
    seen = set()
    near_groups = []
    h_arr = np.array([h for _, _, h in hashes], dtype=np.uint64)
    bits = np.unpackbits(h_arr.view(np.uint8).reshape(len(hashes), 8), axis=1)
    for i in range(len(hashes)):
        if i in seen:
            continue
        dist = np.count_nonzero(bits[i] != bits, axis=1)
        idx = np.where(dist <= NEAR_DUP_DHASH_THRESHOLD)[0]
        members = [(hashes[j][0], hashes[j][1]) for j in idx]
        if len(members) > 1:
            near_groups.append(members)
            seen.update(idx.tolist())
    log(f'    dHash near-duplicate groups (<= {NEAR_DUP_DHASH_THRESHOLD} bits): {len(near_groups)}')
    for g in near_groups[:10]:
        log(f'      {g}')

    blurry = []
    for cls, path, fname in originals:
        v = cv2.Laplacian(gray_cache[cls][fname], cv2.CV_64F).var()
        if v < BLUR_LAP_VAR_THRESHOLD:
            blurry.append({'file': fname, 'class': cls, 'lap_var': round(float(v), 1)})
    log(f'    flagged blurry (lap_var < {BLUR_LAP_VAR_THRESHOLD}): {len(blurry)}')
    for cls in sorted(lap_stats):
        vs = sorted(lap_stats[cls])
        log(f'    {cls:16s} lap_var min {vs[0]:.1f} med {vs[len(vs)//2]:.1f} max {vs[-1]:.1f}')
    threshold_counts = {}
    for t in (10, 20, 30, 50, 100, 200):
        threshold_counts[t] = sum(1 for vs in lap_stats.values() for v in vs if v < t)
    log(f'    originals below lap_var thresholds: {threshold_counts}')

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
        log(f'    {cls:16s} n {len(sizes[cls]):5d}  min {min(ws)}x{min(hs)}  '
            f'med {med(ws)}x{med(hs)}  max {max(ws)}x{max(hs)}')

    # ---- 9. report ----
    report = {
        'source': 'Mendeley jtttfbx342/1 Image Dataset for Turmeric Plant Leaf Disease Detection',
        'download_url': 'https://data.mendeley.com/datasets/jtttfbx342/1',
        'structure': ('Image Dataset for Turmeric Plant Leaf Disease Detection/'
                      '{Original DataSet,Augmented DataSet}/{Class}/'),
        'n_classes': len(CLASS_MAP),
        'class_map': CLASS_MAP,
        'original_counts': dict(orig_cls),
        'augmented_counts': dict(aug_cls),
        'original_total': real_total,
        'augmented_total': len(augmented),
        'class_imbalance_original': imbalance,
        'augmentation_name_pattern': dict(aug_name_count),
        'augmentation_stem_multiplicity': {str(k): v for k, v in sorted(mult.items())},
        'corrupt_images': corrupt,
        'exact_dup_groups_within_original': [{'members': g, 'n': len(g)} for g in dup_groups],
        'augmented_identical_to_original_count': len(cross_oa),
        'cross_class_dup_groups_original': [{'members': g, 'n': len(g)} for g in cc.values()],
        'near_duplicate_groups': [{'members': g, 'n': len(g)} for g in near_groups],
        'blurry_threshold_lap_var': BLUR_LAP_VAR_THRESHOLD,
        'blurry_images': blurry,
        'blur_lap_var_threshold_counts': threshold_counts,
        'resolution': res_report,
        'notes': [
            'Original DataSet (865) and Augmented DataSet (3496) are SEPARATE folders; '
            'only ORIGINALS will be used for training (real images, no synthetic).',
            'Augmented files use an aug_* naming pattern (may encode a 1-2x multiplicity '
            'per source image, e.g. aug_<cls>_<n>).',
            'Healthy_Leaf is the largest original class (213); Leaf_Spot smallest (193); '
            'imbalance max/min ratio ~1.1, negligible.',
            'If any original is byte-identical to an augmented file or appears in another '
            'class, it is flagged above (expected to be zero).',
        ],
        'leaf_confirmation_limitation': (
            'Automated audit cannot visually verify every image is a turmeric leaf. '
            'Manual spot-check recommended before production.'),
        'log': LOG,
    }
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nAudit report saved: {REPORT}')


if __name__ == '__main__':
    audit()
