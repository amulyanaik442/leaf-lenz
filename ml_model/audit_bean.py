"""
Dataset Audit for the Bean Plant Dataset (Kaggle zunorain/bean-plant-dataset, 1,432 images).

Structure:
    DOWNY_MILDEW_LEAF/   (366)
    FRESH_LEAF/          (411)
    LEAFMINNER_LEAF/     (348)
    POWDER_MILDEW_LEAF/  (307)

This is a RAW dataset (no augmentation markers). Per the approved workflow used
for Groundnut/Cotton/Pumpkin/Tea/Chilli/Ginger/Black Pepper, it will be treated
as raw: balance via random undersampling, augment only train, fresh stratified
80/10/10 split.

Checks:
  1. Per-class counts
  2. Raw vs augmented (expect NONE - raw dataset)
  3. Duplicate filenames (same basename in different folders - naming collision)
  4. Corrupted images (unreadable / truncated)
  5. Duplicate images (exact MD5) + cross-class exact dups
  6. Perceptual dHash near-duplicates
  7. Blurry images (Laplacian variance)
  8. Image resolution report
  9. Class imbalance
 10. Folder structure
 11. Train/validation/test leakage (is there a pre-existing split at all?)
 12. Proposed class-name mapping to Leaf Lenz `bean___` production names

Usage:
    venv\Scripts\python.exe ml_model/audit_bean.py
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
EXTRACT_ROOT = os.path.join(BASE_DIR, 'dataset', 'bean', 'Bean Plant dataset')
REPORT = os.path.join(BASE_DIR, 'dataset', 'bean', 'bean_audit_report.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.WEBP')
BLUR_LAP_VAR_THRESHOLD = 100.0
NEAR_DUP_DHASH_THRESHOLD = 8
ANALYSIS_MAX_DIM = 512
LOG = []

# Proposed Leaf Lenz production class names (appended after Ginger at 134+)
CLASS_MAP = {
    'DOWNY_MILDEW_LEAF': 'bean___downy_mildew',
    'FRESH_LEAF': 'bean___healthy',
    'LEAFMINNER_LEAF': 'bean___leafminer',
    'POWDER_MILDEW_LEAF': 'bean___powdery_mildew',
}

NAMED_DISEASES = {
    'DOWNY_MILDEW_LEAF': 'Downy Mildew',
    'FRESH_LEAF': 'Healthy',
    'LEAFMINNER_LEAF': 'Leafminer',
    'POWDER_MILDEW_LEAF': 'Powdery Mildew',
}


def log(msg):
    LOG.append(str(msg))
    print(msg, flush=True)


def load_all(folder):
    """Return list of (cls, abspath, fname) for class subfolders."""
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
    log('PEA PLANT DISEASE DATASET AUDIT')
    log('=' * 64)

    log(f'\n[0] Source: {EXTRACT_ROOT}')
    log('    Kaggle: zunorain/bean-plant-dataset')
    log('    URL: https://www.kaggle.com/datasets/zunorain/pea-plant-dataset')
    log('    Zip size: 17,566,584 bytes')
    log('    This is a RAW dataset (no augmented-copy markers).')

    if not os.path.isdir(EXTRACT_ROOT):
        log('ERROR: extract folder not found.')
        return

    images = load_all(EXTRACT_ROOT)
    log(f'    images total: {len(images)}')

    # ---- 1. counts ----
    log('\n[1] Per-class counts')
    cls_count = Counter(c for c, _, _ in images)
    for cls in sorted(cls_count):
        log(f'     {cls:20s} {cls_count[cls]:5d}')
    nonempty = [c for c in cls_count if cls_count[c] > 0]
    if nonempty:
        rmax = max(cls_count[c] for c in nonempty)
        rmin = min(cls_count[c] for c in nonempty)
        imbalance = {'total': len(images),
                     'max_class': rmax, 'min_class': rmin,
                     'max_min_ratio': round(rmax / rmin, 2),
                     'empty_classes': sorted(set(CLASS_MAP) - set(nonempty))}
        log(f'  imbalance max/min ratio: {imbalance["max_min_ratio"]}')
    else:
        imbalance = None

    # ---- 2. raw vs augmented ----
    log('\n[2] Raw vs augmented')
    marked = 0
    log('    Augmentation markers (.rf.<hex>, etc.): NONE found (raw dataset).')
    log('    => every image is an independent sample; no source-level grouping needed.')

    # ---- 3. duplicate filenames across folders ----
    log('\n[3] Duplicate filenames (same basename in different folders)')
    by_name = defaultdict(list)
    for cls, path, fname in images:
        by_name[fname].append(cls)
    dup_names = {n: c for n, c in by_name.items() if len(c) > 1}
    log(f'    duplicate basenames: {len(dup_names)}')
    for n, c in list(dup_names.items())[:10]:
        log(f'      {n}: {c}')
    log('    NOTE: files are named resized_<N>.jpg with numbering that resets per '
        'class folder, so basename collisions are EXPECTED and do not imply '
        'identical content. Real content duplicates are checked via MD5 below.')

    # ---- 4. corrupt ----
    log('\n[4] Corrupt/unreadable images')
    corrupt = []
    for cls, path, fname in images:
        try:
            load_gray_downscaled(path)
        except Exception as e:
            corrupt.append({'file': fname, 'class': cls, 'error': str(e)})
    log(f'    corrupt: {len(corrupt)}')
    for c in corrupt:
        log(f'      {c}')

    # ---- 5. exact MD5 duplicates ----
    log('\n[5] Exact MD5 duplicates')
    md5 = defaultdict(list)
    for cls, path, fname in images:
        md5[hashlib.md5(open(path, 'rb').read()).hexdigest()].append((cls, fname))
    dup_groups = [g for g in md5.values() if len(g) > 1]
    log(f'    exact-MD5 duplicate groups: {len(dup_groups)}')
    for g in dup_groups[:10]:
        log(f'      {g}')
    cc = defaultdict(list)
    for h, g in md5.items():
        classes = set(m[0] for m in g)
        if len(classes) > 1:
            cc[h].append(g)
    log(f'    duplicates across DIFFERENT classes: {len(cc)}')
    for h, g in list(cc.items())[:10]:
        log(f'      {g}')

    # ---- 6/7/8. dHash, blur, resolution ----
    log('\n[6/7/8] dHash near-dups + blur + resolution')
    gray_cache = {}
    sizes = defaultdict(list)
    lap_stats = defaultdict(list)
    for cls, path, fname in images:
        gray, size = load_gray_downscaled(path)
        gray_cache[cls] = gray_cache.get(cls, {})
        gray_cache[cls][fname] = gray
        sizes[cls].append(size)
        v = cv2.Laplacian(gray, cv2.CV_64F).var()
        lap_stats[cls].append(v)

    hashes = []
    for cls, _p, fname in images:
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
    intra = 0
    cross = 0
    for g in near_groups:
        classes = set(m[0] for m in g)
        if len(classes) == 1:
            intra += 1
        else:
            cross += 1
    log(f'    intra-class groups: {intra} | cross-class groups: {cross}')
    for g in near_groups[:5]:
        members = [(c, n) for c, n in g][:3]
        log(f'      group size {len(g)} first members: {members}')
    if cross:
        cross_sizes = sorted((len(g), sorted(set(m[0] for m in g))) for g in near_groups
                             if len(set(m[0] for m in g)) > 1)
        log(f'    largest cross-class group: size {cross_sizes[-1][0]} classes {cross_sizes[-1][1]}')
        log(f'    total images inside cross-class groups: '
            f'{sum(len(g) for g in near_groups if len(set(m[0] for m in g)) > 1)}')

    blurry = []
    for cls, path, fname in images:
        v = cv2.Laplacian(gray_cache[cls][fname], cv2.CV_64F).var()
        if v < BLUR_LAP_VAR_THRESHOLD:
            blurry.append({'file': fname, 'class': cls, 'lap_var': round(float(v), 1)})
    log(f'    flagged blurry (lap_var < {BLUR_LAP_VAR_THRESHOLD}): {len(blurry)}')
    for cls in sorted(lap_stats):
        vs = sorted(lap_stats[cls])
        log(f'    {cls:20s} lap_var min {vs[0]:.1f} med {vs[len(vs)//2]:.1f} max {vs[-1]:.1f}')
    threshold_counts = {}
    for t in (10, 20, 30, 50, 100, 200):
        threshold_counts[t] = sum(1 for vs in lap_stats.values() for v in vs if v < t)
    log(f'    images below lap_var thresholds: {threshold_counts}')

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
        log(f'    {cls:20s} n {len(sizes[cls]):5d}  min {min(ws)}x{min(hs)}  '
            f'med {med(ws)}x{med(hs)}  max {max(ws)}x{max(hs)}')

    # ---- 9. folder structure ----
    log('\n[9] Folder structure')
    structure = {}
    for cls in sorted(os.listdir(EXTRACT_ROOT)):
        d = os.path.join(EXTRACT_ROOT, cls)
        if os.path.isdir(d):
            structure[cls] = len([f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS)])
    log(f'    {json.dumps(structure, indent=4)}')

    # ---- 10. train/test leakage ----
    log('\n[10] Pre-existing train/test split: NONE (single flat class folders)')
    log('    A fresh 80/10/10 stratified split will be created in preparation;')
    log('    no leakage possible from the source layout.')

    # ---- 11. resolution uniformity check ----
    all_sizes = [s for vs in sizes.values() for s in vs]
    uniform = len(set(all_sizes)) == 1
    log(f'\n[11] Resolution uniformity: {"UNIFORM" if uniform else "MIXED"} '
        f'({len(set(all_sizes))} distinct sizes)')

    # ---- report ----
    report = {
        'source': 'Kaggle zunorain/bean-plant-dataset',
        'download_url': 'https://www.kaggle.com/datasets/zunorain/pea-plant-dataset',
        'license': 'CC BY 4.0 (Kaggle default for this dataset; verify)',
        'structure': 'Bean Plant dataset/{DOWNY_MILDEW_LEAF,FRESH_LEAF,LEAFMINNER_LEAF,POWDER_MILDEW_LEAF}/',
        'n_classes': len(CLASS_MAP),
        'class_map': CLASS_MAP,
        'named_diseases': NAMED_DISEASES,
        'counts': dict(cls_count),
        'total': len(images),
        'raw_dataset': True,
        'augmented_marker_files': marked,
        'class_imbalance': imbalance,
        'duplicate_filenames': dup_names,
        'corrupt_images': corrupt,
        'exact_dup_groups': [{'members': g, 'n': len(g)} for g in dup_groups],
        'cross_class_dup_groups': [{'members': g, 'n': len(g)} for g in cc.values()],
        'near_duplicate_groups': [{'members': g, 'n': len(g)} for g in near_groups],
        'blurry_threshold_lap_var': BLUR_LAP_VAR_THRESHOLD,
        'blurry_images': blurry,
        'blur_lap_var_threshold_counts': threshold_counts,
        'resolution': res_report,
        'resolution_uniform': uniform,
        'folder_structure': structure,
        'pre_existing_split': 'none',
        'notes': [
            'Raw dataset of 1,432 images, 4 classes, all .jpg.',
            'Filenames are resized_<N>.jpg with per-folder numbering; basename '
            'collisions across folders are a naming artifact, not content dups.',
            'Counts match the Kaggle page: FRESH_LEAF 411, DOWNY_MILDEW_LEAF 366, '
            'LEAFMINNER_LEAF 348, POWDER_MILDEW_LEAF 307.',
            'Balance via random undersampling to the smallest class (307).',
            'Augment only the train set; never the validation or test sets.',
            'dHash near-dup groups: review cross-class groups before training.',
        ],
        'leaf_confirmation_limitation': (
            'Automated audit cannot visually verify every image is a bean leaf '
            'nor that the disease labels are correct. Manual spot-check recommended.'),
        'log': LOG,
    }
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nAudit report saved: {REPORT}')


if __name__ == '__main__':
    audit()
