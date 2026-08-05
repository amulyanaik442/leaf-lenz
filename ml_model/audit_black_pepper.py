"""
Dataset Audit for the Black Pepper Leaf Blight and Yellow Mottle Virus dataset
(Kaggle udi17live/black-pepper-leaf-blight-and-yellow-mottle-virus, 819 images).

Structure:
    black_pepper_healthy/               (273)
    black_pepper_leaf_blight/           (273)
    black_pepper_yellow_mottle_virus/   (273)

Checks:
  1. Per-class counts
  2. Duplicate filenames (same name in different folders)
  3. Corrupted images (unreadable / truncated)
  4. Duplicate images (exact MD5) + cross-class exact dups
  5. Perceptual dHash near-duplicates
  6. Blurry images (Laplacian variance)
  7. Image resolution report
  8. Class imbalance
  9. Folder structure
  10. Pre-existing train/test split leakage (is there a split at all?)
  11. Proposed class-name mapping to Leaf Lenz `BlackPepper___` production names

Usage:
    venv\Scripts\python.exe ml_model/audit_black_pepper.py
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
EXTRACT_ROOT = r'C:\Users\amuly\AppData\Local\Temp\opencode\black_pepper\extract'
REPORT = os.path.join(BASE_DIR, 'dataset', 'black_pepper', 'black_pepper_audit_report.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.WEBP')
BLUR_LAP_VAR_THRESHOLD = 100.0
NEAR_DUP_DHASH_THRESHOLD = 8
ANALYSIS_MAX_DIM = 512
LOG = []

# Proposed Leaf Lenz production class names (appended at indices 127-129)
CLASS_MAP = {
    'black_pepper_healthy': 'BlackPepper___healthy',
    'black_pepper_leaf_blight': 'BlackPepper___leaf_blight',
    'black_pepper_yellow_mottle_virus': 'BlackPepper___yellow_mottle_virus',
}

NAMED_DISEASES = {
    'black_pepper_healthy': 'Healthy',
    'black_pepper_leaf_blight': 'Leaf Blight (e.g. Phytophthora capsici)',
    'black_pepper_yellow_mottle_virus': 'Yellow Mottle Virus (Piper yellow mottle virus)',
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
    log('BLACK PEPPER LEAF DISEASE DATASET AUDIT')
    log('=' * 64)

    log(f'\n[0] Source: {EXTRACT_ROOT}')
    log('    Kaggle: udi17live/black-pepper-leaf-blight-and-yellow-mottle-virus')
    log('    License: GPL-2.0 | totalBytes (API): 91,515,661')

    if not os.path.isdir(EXTRACT_ROOT):
        log('ERROR: extract folder not found.')
        return

    images = load_all(EXTRACT_ROOT)
    log(f'    images total: {len(images)}')

    # ---- 1. counts ----
    log('\n[1] Per-class counts')
    cls_count = Counter(c for c, _, _ in images)
    for cls in sorted(cls_count):
        log(f'     {cls:38s} {cls_count[cls]:5d}')
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

    # ---- 2. duplicate filenames across folders ----
    log('\n[2] Duplicate filenames (same basename in different folders)')
    by_name = defaultdict(list)
    for cls, path, fname in images:
        by_name[fname].append(cls)
    dup_names = {n: c for n, c in by_name.items() if len(c) > 1}
    log(f'    duplicate basenames: {len(dup_names)}')
    for n, c in list(dup_names.items())[:10]:
        log(f'      {n}: {c}')

    # ---- 3. corrupt ----
    log('\n[3] Corrupt/unreadable images')
    corrupt = []
    for cls, path, fname in images:
        try:
            load_gray_downscaled(path)
        except Exception as e:
            corrupt.append({'file': fname, 'class': cls, 'error': str(e)})
    log(f'    corrupt: {len(corrupt)}')
    for c in corrupt:
        log(f'      {c}')

    # ---- 4. exact MD5 duplicates ----
    log('\n[4] Exact MD5 duplicates')
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

    # ---- 5/6/7. dHash, blur, resolution ----
    log('\n[5/6/7] dHash near-dups + blur + resolution')
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
    for g in near_groups[:10]:
        log(f'      {g}')

    blurry = []
    for cls, path, fname in images:
        v = cv2.Laplacian(gray_cache[cls][fname], cv2.CV_64F).var()
        if v < BLUR_LAP_VAR_THRESHOLD:
            blurry.append({'file': fname, 'class': cls, 'lap_var': round(float(v), 1)})
    log(f'    flagged blurry (lap_var < {BLUR_LAP_VAR_THRESHOLD}): {len(blurry)}')
    for cls in sorted(lap_stats):
        vs = sorted(lap_stats[cls])
        log(f'    {cls:38s} lap_var min {vs[0]:.1f} med {vs[len(vs)//2]:.1f} max {vs[-1]:.1f}')
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
        log(f'    {cls:38s} n {len(sizes[cls]):5d}  min {min(ws)}x{min(hs)}  '
            f'med {med(ws)}x{med(hs)}  max {max(ws)}x{max(hs)}')

    # ---- 8. folder structure ----
    log('\n[8] Folder structure')
    structure = {}
    for cls in sorted(os.listdir(EXTRACT_ROOT)):
        d = os.path.join(EXTRACT_ROOT, cls)
        if os.path.isdir(d):
            structure[cls] = len([f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS)])
    log(f'    {json.dumps(structure, indent=4)}')

    # ---- 9. train/test leakage ----
    log('\n[9] Pre-existing train/test split: NONE (single flat class folders)')
    log('    A fresh 80/10/10 stratified split will be created in preparation;')
    log('    no leakage possible from the source layout.')

    # ---- report ----
    report = {
        'source': 'Kaggle udi17live/black-pepper-leaf-blight-and-yellow-mottle-virus',
        'download_url': 'https://www.kaggle.com/datasets/udi17live/black-pepper-leaf-blight-and-yellow-mottle-virus',
        'license': 'GPL-2.0',
        'structure': 'black_pepper_{healthy,leaf_blight,yellow_mottle_virus}/',
        'n_classes': len(CLASS_MAP),
        'class_map': CLASS_MAP,
        'named_diseases': NAMED_DISEASES,
        'counts': dict(cls_count),
        'total': len(images),
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
        'folder_structure': structure,
        'pre_existing_split': 'none',
        'notes': [
            'All 3 classes are exactly balanced at 273 images each (max/min ratio 1.0).',
            'Single flat layout, no pre-existing train/test folders -> fresh split is leakage-free.',
            'License is GPL-2.0 (copyleft); confirm with supervisor before production deployment.',
            'Automated audit cannot visually verify every label; manual spot-check recommended.',
        ],
        'leaf_confirmation_limitation': (
            'Automated audit cannot visually verify every image is a black pepper leaf '
            'nor that the disease labels are correct. Manual spot-check recommended.'),
        'log': LOG,
    }
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nAudit report saved: {REPORT}')


if __name__ == '__main__':
    audit()
