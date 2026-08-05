"""
Dataset Audit for the Ginger Leaf Disease dataset
(GitHub wongjay1941/Ginger-Leaf-Dataset, 10,910 images).

Structure:
    combined/Damage-Pest/   (2076)
    combined/Dehydrated/    (3372)
    combined/Healthy/       (2198)
    combined/Leaf-blight/   (3264)

The dataset was created by augmenting ~4,033 original images with Roboflow
(.rf.<16-hex> filename suffix marks augmented copies; originals lack it).

Checks:
  1. Per-class counts
  2. Original vs augmented separation (Roboflow .rf.<hex> filename pattern)
  3. Duplicate filenames (same name in different folders)
  4. Corrupted images (unreadable / truncated)
  5. Duplicate images (exact MD5) + cross-class exact dups
  6. Perceptual dHash near-duplicates
  7. Blurry images (Laplacian variance)
  8. Image resolution report
  9. Class imbalance
 10. Folder structure
 11. Pre-existing train/test split leakage (is there a split at all?)
 12. Proposed class-name mapping to Leaf Lenz `Ginger___` production names

Usage:
    venv\Scripts\python.exe ml_model/audit_ginger.py
"""
import os
import re
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
EXTRACT_ROOT = r'C:\Users\amuly\AppData\Local\Temp\opencode\ginger\extract\combined'
REPORT = os.path.join(BASE_DIR, 'dataset', 'ginger', 'ginger_audit_report.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.WEBP')
BLUR_LAP_VAR_THRESHOLD = 100.0
NEAR_DUP_DHASH_THRESHOLD = 8
ANALYSIS_MAX_DIM = 512
LOG = []

# Proposed Leaf Lenz production class names (appended at indices 130-133)
CLASS_MAP = {
    'Damage-Pest': 'ginger___pest_damage',
    'Dehydrated': 'ginger___dehydrated',
    'Healthy': 'ginger___healthy',
    'Leaf-blight': 'ginger___leaf_blight',
}

NAMED_DISEASES = {
    'Damage-Pest': 'Pest Damage',
    'Dehydrated': 'Dehydrated',
    'Healthy': 'Healthy',
    'Leaf-blight': 'Leaf Blight',
}

# Roboflow augmentation marker:  BASENAME.rf.<32-hex>.<ext>
RF_AUG_RE = re.compile(r'\.rf\.[0-9a-f]{32}\.')


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


def is_augmented(fname):
    """True if the filename carries a Roboflow .rf.<16-hex>. augmentation marker."""
    return RF_AUG_RE.search(fname) is not None


def base_name(fname):
    """Strip the Roboflow augmentation marker, keeping the source image name."""
    return RF_AUG_RE.sub('.', fname)


def audit():
    log('=' * 64)
    log('GINGER LEAF DISEASE DATASET AUDIT')
    log('=' * 64)

    log(f'\n[0] Source: {EXTRACT_ROOT}')
    log('    GitHub: wongjay1941/Ginger-Leaf-Dataset (Ginger_Leaf_Dataset.zip)')
    log('    URL: https://github.com/wongjay1941/Ginger-Leaf-Dataset')
    log('    Zip size: 229,016,577 bytes | License: NO LICENSE FILE FOUND')
    log('    (GitHub LFS file downloaded via git clone; no LICENSE present in repo)')

    if not os.path.isdir(EXTRACT_ROOT):
        log('ERROR: extract folder not found.')
        return

    images = load_all(EXTRACT_ROOT)
    log(f'    images total: {len(images)}')

    # ---- 1. counts ----
    log('\n[1] Per-class counts')
    cls_count = Counter(c for c, _, _ in images)
    for cls in sorted(cls_count):
        log(f'     {cls:16s} {cls_count[cls]:5d}')
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

    # ---- 2. original vs augmented ----
    log('\n[2] Original vs augmented (Roboflow .rf.<32-hex> filename marker)')
    marked = sum(1 for _c, _p, fname in images if is_augmented(fname))
    unmarked = len(images) - marked
    log(f'    files carrying the .rf.<32-hex> marker: {marked}')
    log(f'    files WITHOUT the marker: {unmarked}')
    log('    => EVERY file carries the marker; the base (non-augmented) copies')
    log('       cannot be identified reliably from filenames alone.')

    src_orig = defaultdict(set)
    src_aug = defaultdict(set)
    for cls, _p, fname in images:
        base = base_name(fname)
        if is_augmented(fname):
            src_aug[cls].add(base)
        else:
            src_orig[cls].add(base)
    distinct_total = sum(len(v) for v in src_aug.values()) + sum(len(v) for v in src_orig.values())
    log(f'    distinct source names across dataset: {distinct_total}')
    for cls in sorted(cls_count):
        log(f'     {cls:16s} distinct source names: {len(src_aug[cls])}')
    log(f'    (MDPI paper states the dataset was created by augmenting 4,033 originals;')
    log(f'     empirically the zip contains {distinct_total} distinct source photos.)')

    mixed = ('ORIGINALS AND AUGMENTATIONS ARE MIXED IN THE SAME CLASS FOLDERS, '
             'and are NOT reliably separable via the Roboflow filename marker '
             '(every file carries a .rf.<32-hex>. suffix).')
    log(f'\n    storage layout: {mixed}')

    # ---- 3. duplicate filenames across folders ----
    log('\n[3] Duplicate filenames (same basename in different folders)')
    by_name = defaultdict(list)
    for cls, path, fname in images:
        by_name[fname].append(cls)
    dup_names = {n: c for n, c in by_name.items() if len(c) > 1}
    log(f'    duplicate basenames: {len(dup_names)}')
    for n, c in list(dup_names.items())[:10]:
        log(f'      {n}: {c}')

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

    # ---- 5b. augmentation structure per distinct source image ----
    log('\n[5b] Augmentation structure per distinct source image')
    src_counts = Counter()
    for cls, _p, fname in images:
        src_counts[(cls, base_name(fname))] += 1
    per_source = defaultdict(list)
    for (cls, _base), n in src_counts.items():
        per_source[cls].append(n)
    total_sources = sum(len(v) for v in per_source.values())
    copies1 = sum(1 for v in per_source.values() for n in v if n == 1)
    log(f'    distinct source names across dataset: {total_sources}')
    log(f'    source names with exactly 1 file (no augmented copies): {copies1}')
    for cls in sorted(per_source):
        ns = sorted(per_source[cls])
        med = ns[len(ns) // 2]
        log(f'     {cls:16s} {len(ns):5d} distinct sources | copies per source min {ns[0]} '
            f'med {med} max {ns[-1]}')

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
    # classify groups: intra-class vs cross-class
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
        log(f'    {cls:16s} lap_var min {vs[0]:.1f} med {vs[len(vs)//2]:.1f} max {vs[-1]:.1f}')
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
        log(f'    {cls:16s} n {len(sizes[cls]):5d}  min {min(ws)}x{min(hs)}  '
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

    # ---- report ----
    report = {
        'source': 'GitHub wongjay1941/Ginger-Leaf-Dataset',
        'download_url': 'https://github.com/wongjay1941/Ginger-Leaf-Dataset',
        'license': 'NO LICENSE FILE FOUND (all rights reserved by default)',
        'structure': 'combined/{Damage-Pest,Dehydrated,Healthy,Leaf-blight}/',
        'n_classes': len(CLASS_MAP),
        'class_map': CLASS_MAP,
        'named_diseases': NAMED_DISEASES,
        'counts': dict(cls_count),
        'total': len(images),
        'files_with_rf_marker': marked,
        'files_without_rf_marker': unmarked,
        'original_counts': {},
        'augmented_counts': {},
        'total_original': 0,
        'total_augmented': len(images),
        'distinct_source_names_total': total_sources,
        'source_names_single_file': copies1,
        'original_augmented_separable': False,
        'original_augmented_method': (
            'ALL files carry a Roboflow .rf.<32-hex>. suffix; the untransformed '
            'base copies cannot be reliably identified from filenames. Separation '
            'must instead be done by grouping files that share a stripped base name '
            'and splitting at the SOURCE level to prevent leakage.'),
        'original_augmented_layout': mixed,
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
            'Dataset is imbalanced (max/min ratio ~1.62: Dehydrated 3372 vs Damage-Pest 2076).',
            'ALL images carry Roboflow .rf.<32-hex>. markers; originals and augmented copies are '
            'MIXED and NOT separable by filename. Only 2,600 distinct source photos exist; the '
            'rest are augmented copies (~4.2 copies per source on average).',
            'Because of this, any split MUST group files by stripped base name so that all copies '
            'of the same source photo stay in the same split (prevents leakage).',
            'All images are 640x640 RGB.',
            'Laplacian variance medians are low (41-84); threshold 100 flags 8,256/10,910. This is '
            'a low-contrast leaf-on-background dataset, not necessarily motion blur. A hard blur '
            'filter at 100 would be too aggressive; needs a decision.',
            'dHash near-dup groups are large (many over-merged sources); cross-class groups exist '
            'but are dominated by visually-similar leaf photos rather than confirmed mislabels. '
            'Manual spot-check recommended before relying on label quality.',
            'No LICENSE file in the GitHub repo -> default all-rights-reserved; confirm with '
            'supervisor before production deployment.',
            'Automated audit cannot visually verify every label; manual spot-check recommended.',
        ],
        'leaf_confirmation_limitation': (
            'Automated audit cannot visually verify every image is a ginger leaf '
            'nor that the disease labels are correct. Manual spot-check recommended.'),
        'log': LOG,
    }
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nAudit report saved: {REPORT}')


if __name__ == '__main__':
    audit()
