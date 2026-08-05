"""
Dataset Audit for the prepared Cacao leaf classification crops.

Checks (per Step 2 of the integration plan):
  1. Image counts per class
  2. Corrupted images (unreadable / truncated JPEG)
  3. Duplicate images (exact MD5 + perceptual dHash near-duplicates)
  4. Blurry images (Laplacian variance via OpenCV)
  5. Label verification vs the crop manifest
  6. Image resolution report
  7. Class imbalance report

Note on "confirm every image is a leaf": crops come from the Amini cocoa
contamination challenge whose annotation guidelines target cocoa leaves.  We
cannot visually verify every crop here (no image display); the audit instead
reports this limitation and the per-class skip/heuristic stats.

Usage:
    python ml_model/audit_cacao_leaf.py
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
RAW_DIR = os.path.join(BASE_DIR, 'dataset', 'cacao_leaf', 'Raw_Classification')
MANIFEST = os.path.join(BASE_DIR, 'dataset', 'cacao_leaf', 'cacao_leaf_manifest.json')
REPORT = os.path.join(BASE_DIR, 'dataset', 'cacao_leaf', 'cacao_leaf_audit_report.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP')
BLUR_LAP_VAR_THRESHOLD = 100.0  # below this -> flagged blurry
NEAR_DUP_DHASH_THRESHOLD = 8    # <=8 differing bits -> near-duplicate group
LOG = []


def log(msg):
    LOG.append(str(msg))
    print(msg, flush=True)


def load_all_images():
    files = []
    for cls in sorted(os.listdir(RAW_DIR)):
        d = os.path.join(RAW_DIR, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMAGE_EXTS):
                files.append((cls, os.path.join(d, f), f))
    return files


def dhash(img_gray, size=8):
    small = cv2.resize(img_gray, (size + 1, size))
    diff = small[:, 1:] > small[:, :-1]
    return sum(1 << i for i, b in enumerate(diff.ravel()) if b)


def audit():
    log('=' * 60)
    log('CACAO LEAF DATASET AUDIT')
    log('=' * 60)

    # ---- 1. counts ----
    files = load_all_images()
    counts = Counter(cls for cls, _, _ in files)
    log('\n[1] Image counts per class')
    for cls in sorted(counts):
        log(f'    {cls:12s} {counts[cls]}')
    log(f'    total: {len(files)}')
    imbalance = {}
    if counts:
        n_max = max(counts.values())
        n_min = min(counts.values())
        imbalance = {'max_class': n_max, 'min_class': n_min,
                     'max_min_ratio': round(n_max / n_min, 2) if n_min else None}

    # ---- 2 & 6. corrupt + resolution ----
    log('\n[2/6] Corrupt detection + resolution report')
    corrupt = []
    sizes = defaultdict(list)  # cls -> [(w,h)]
    for cls, path, fname in files:
        try:
            with Image.open(path) as im:
                im.load()
                if getattr(im, 'n_frames', 1) > 1:
                    pass
                sizes[cls].append(im.size)
        except Exception as e:
            corrupt.append({'file': fname, 'class': cls, 'error': str(e)})
    log(f'    corrupt/unreadable: {len(corrupt)}')
    for c in corrupt:
        log(f'      {c}')

    res_report = {}
    for cls in sorted(sizes):
        ws = [s[0] for s in sizes[cls]]
        hs = [s[1] for s in sizes[cls]]
        ws.sort(); hs.sort()
        def med(a):
            return a[len(a) // 2]
        res_report[cls] = {
            'count': len(sizes[cls]),
            'min': [min(ws), min(hs)], 'median': [med(ws), med(hs)],
            'max': [max(ws), max(hs)],
            'unique_sizes': sorted(set(sizes[cls]))[:10],
        }
        log(f'    {cls:12s} size min {min(ws)}x{min(hs)}  med {med(ws)}x{med(hs)}  max {max(ws)}x{max(hs)}')

    # ---- 3. duplicates (MD5 + dHash) ----
    log('\n[3] Duplicate detection (exact MD5 + perceptual dHash)')
    md5_map = defaultdict(list)
    hashes = []
    for cls, path, fname in files:
        h = hashlib.md5(open(path, 'rb').read()).hexdigest()
        md5_map[h].append((cls, fname))
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        hashes.append((cls, fname, dhash(gray)))
    exact_dup_groups = [g for g in md5_map.values() if len(g) > 1]
    log(f'    exact MD5 duplicate groups: {len(exact_dup_groups)}')
    for g in exact_dup_groups[:10]:
        log(f'      {g}')

    # near-duplicates within same class (coarse greedy)
    near_groups = []
    seen = set()
    h_arr = np.array([h for _, _, h in hashes], dtype=np.uint64)
    bits = np.unpackbits(h_arr.view(np.uint8).reshape(len(hashes), 8), axis=1)
    for i, (cls, fname, h) in enumerate(hashes):
        if i in seen:
            continue
        dist = np.count_nonzero(bits[i] != bits, axis=1)
        idx = np.where(dist <= NEAR_DUP_DHASH_THRESHOLD)[0]
        members = [(hashes[j][1], hashes[j][0]) for j in idx]
        if len(members) > 1:
            near_groups.append(members)
            seen.update(idx.tolist())
    log(f'    dHash near-duplicate groups (<= {NEAR_DUP_DHASH_THRESHOLD} bits): {len(near_groups)}')
    for g in near_groups[:10]:
        log(f'      {g}')

    # ---- 4. blur ----
    log('\n[4] Blur detection (Laplacian variance < %.0f)' % BLUR_LAP_VAR_THRESHOLD)
    blurry = []
    lap_stats = defaultdict(list)
    for cls, path, fname in files:
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        v = cv2.Laplacian(gray, cv2.CV_64F).var()
        lap_stats[cls].append(v)
        if v < BLUR_LAP_VAR_THRESHOLD:
            blurry.append({'file': fname, 'class': cls, 'lap_var': round(float(v), 1)})
    log(f'    flagged blurry: {len(blurry)}')
    for cls in sorted(lap_stats):
        vs = sorted(lap_stats[cls])
        log(f'    {cls:12s} lap_var min {vs[0]:.1f} med {vs[len(vs)//2]:.1f} max {vs[-1]:.1f}')
    threshold_counts = {}
    for t in (10, 20, 30, 50, 100, 200):
        n = sum(1 for vs in lap_stats.values() for v in vs if v < t)
        threshold_counts[t] = n
    log(f'    images below lap_var thresholds: {threshold_counts}')

    # ---- 5. label verification vs manifest ----
    log('\n[5] Label verification against manifest')
    with open(MANIFEST) as f:
        manifest = json.load(f)
    manifest_count = Counter(m['class'] for m in manifest)
    mismatches = []
    for m in manifest:
        rel = os.path.join(BASE_DIR, 'dataset', 'cacao_leaf', m['crop_path'])
        ok = os.path.isfile(rel)
        if not ok:
            mismatches.append(m['crop_path'])
    log(f'    manifest entries: {len(manifest)} | per class: {dict(manifest_count)}')
    log(f'    manifest->file mismatches (missing files): {len(mismatches)}')
    log(f'    folder counts match manifest: '
        f'{dict(counts) == dict(manifest_count)}')

    # ---- report ----
    report = {
        'source': 'Amini Cocoa Contamination Dataset -> bbox crops (healthy, anthracnose)',
        'image_counts': dict(counts),
        'class_imbalance': imbalance,
        'corrupt_images': corrupt,
        'resolution': res_report,
        'exact_duplicate_groups': [{'members': g, 'n': len(g)} for g in exact_dup_groups],
        'near_duplicate_groups': [{'members': g, 'n': len(g)} for g in near_groups],
        'blurry_threshold_lap_var': BLUR_LAP_VAR_THRESHOLD,
        'blurry_images': blurry,
        'blur_lap_var_threshold_counts': threshold_counts,
        'label_verification': {
            'manifest_entries': len(manifest),
            'per_class_manifest': dict(manifest_count),
            'missing_files': mismatches,
            'folders_match_manifest': dict(counts) == dict(manifest_count),
        },
        'leaf_confirmation_limitation': (
            'Cannot visually confirm every crop is a leaf in this automated audit. '
            'Crops derive from the Amini cocoa contamination challenge whose labels '
            'target cocoa leaves; healthy boxes may occasionally include background '
            'or pods. Manual spot-check is recommended before production.'),
        'log': LOG,
    }
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nAudit report saved: {REPORT}')


if __name__ == '__main__':
    audit()
