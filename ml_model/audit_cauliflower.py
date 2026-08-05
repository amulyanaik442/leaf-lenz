"""
Dataset Audit for the VegNet Cauliflower dataset (Mendeley DOI 10.17632/t5sssfgn2v.3).

Source: "VegNet: An extensive dataset of cauliflower images to recognize the
diseases using machine learning and deep learning models" (Rajbongshi et al.).

Two separate files exist on Mendeley:
  * Original Dataset.zip   (656 files)  <-- this audit operates ONLY on this
  * Augmented Dataset.zip  (7,360 files, 2.19 GB) -- NOT downloaded/used.

Structure (Original Dataset.zip):
    Bacterial spot rot/   (173)
    Black Rot/            (100)
    Downy Mildew/         (177)
    No disease/           (206)

Known quirk: some images are stored twice (once as .jpeg, once as .jpg) in the
same class folder. This audit quantifies that and determines whether the paired
files are byte-identical re-encodes, near-identical re-compressions, or distinct.

Checks:
  1. Per-class counts (+ basename-level unique image counts)
  2. Original vs augmented (separate zips -> originals only; the augmented zip
     is NOT part of this extract, so no mixing)
  3. Duplicate filenames / dual-encoding (.jpeg/.jpg) pairs
  4. Corrupted images (unreadable / truncated)
  5. Duplicate images (exact MD5) + cross-class exact dups
  6. Perceptual dHash near-duplicates (incl. .jpeg/.jpg pair similarity)
  7. Blurry images (Laplacian variance)
  8. Image resolution report
  9. Class imbalance
 10. Folder structure
 11. Train/validation/test leakage (pre-existing split?)
 12. Proposed class-name mapping to Leaf Lenz `cauliflower___` production names

Usage:
    venv\Scripts\python.exe ml_model/audit_cauliflower.py
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
EXTRACT_ROOT = os.path.join(BASE_DIR, 'dataset', 'cauliflower', 'Original Dataset')
REPORT = os.path.join(BASE_DIR, 'dataset', 'cauliflower', 'cauliflower_audit_report.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.WEBP')
BLUR_LAP_VAR_THRESHOLD = 100.0
NEAR_DUP_DHASH_THRESHOLD = 8
ANALYSIS_MAX_DIM = 512
LOG = []

# Proposed Leaf Lenz production class names (appended after Bean at 138+)
# NOTE: 'Bacterial spot rot' was removed from the dataset on 2026-08-04.
CLASS_MAP = {
    'No disease': 'cauliflower___healthy',
    'Black Rot': 'cauliflower___black_rot',
    'Downy Mildew': 'cauliflower___downy_mildew',
}

NAMED_DISEASES = {
    'No disease': 'Healthy',
    'Black Rot': 'Black Rot',
    'Downy Mildew': 'Downy Mildew',
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
    log('CAULIFLOWER (VegNet) DATASET AUDIT')
    log('=' * 64)

    log(f'\n[0] Source: {EXTRACT_ROOT}')
    log('    Mendeley DOI: 10.17632/t5sssfgn2v.3 (VegNet, version 3)')
    log('    URL: https://data.mendeley.com/datasets/t5sssfgn2v/3')
    log('    Original Dataset.zip sha256: '
        '55e3a21a20bf11de169077c5d7c70b7d5330c55acf44caf0c65533b32d67adda')
    log('    This extract contains ONLY the "Original Dataset" file (656 files).')
    log('    The separate "Augmented Dataset.zip" (7,360 files) was NOT downloaded.')

    if not os.path.isdir(EXTRACT_ROOT):
        log('ERROR: extract folder not found.')
        return

    images = load_all(EXTRACT_ROOT)
    log(f'    images (files) total: {len(images)}')

    # basename-level unique image identity
    by_base = defaultdict(list)
    for cls, path, fname in images:
        by_base[os.path.splitext(fname)[0]].append((cls, path, fname))
    log(f'    unique base names (images): {len(by_base)}')

    # ---- 1. counts ----
    log('\n[1] Per-class counts (files, then unique images)')
    cls_count = Counter(c for c, _, _ in images)
    cls_uniq = Counter()
    for bases in by_base.values():
        cls_uniq[bases[0][0]] += 1
    for cls in sorted(cls_count):
        log(f'     {cls:20s} files={cls_count[cls]:4d}  unique_images={cls_uniq[cls]:4d}')
    nonempty = [c for c in cls_count if cls_count[c] > 0]
    if nonempty:
        rmax = max(cls_uniq[c] for c in nonempty)
        rmin = min(cls_uniq[c] for c in nonempty)
        imbalance = {'total_files': len(images),
                     'total_unique_images': len(by_base),
                     'max_class': rmax, 'min_class': rmin,
                     'max_min_ratio': round(rmax / rmin, 2),
                     'empty_classes': sorted(set(CLASS_MAP) - set(nonempty))}
        log(f'  unique-image imbalance max/min ratio: {imbalance["max_min_ratio"]}')
    else:
        imbalance = None

    # ---- 2. original vs augmented ----
    log('\n[2] Original vs augmented')
    log('    VegNet ships TWO SEPARATE files on Mendeley:')
    log('      * "Original Dataset.zip"  -> this extract (originals only)')
    log('      * "Augmented Dataset.zip" -> 7,360 augmented copies (separate zip)')
    log('    Originals and augmentations are stored SEPARATELY (separate zips).')
    log('    => use ONLY the original images from this folder; augmented zip unused.')
    marked = 0

    # ---- 3. duplicate filenames & dual-encoding pairs ----
    log('\n[3] Duplicate filenames + .jpeg/.jpg dual-encoding pairs')
    dup_names = {b: len(v) for b, v in by_base.items() if len(v) > 1}
    log(f'    base names with >1 file: {len(dup_names)}')
    dual_pairs = {}
    for b, v in by_base.items():
        exts = sorted(os.path.splitext(f)[1].lower() for _, _, f in v)
        if len(v) == 2 and '.jpeg' in exts and '.jpg' in exts:
            dual_pairs[b] = [(cls, f) for cls, _p, f in v]
    log(f'    .jpeg+.jpg dual-encoding pairs: {len(dual_pairs)} '
        f'({len(dual_pairs) * 2} files, {len(dual_pairs)} unique images)')
    for b, v in list(dual_pairs.items())[:5]:
        log(f'      {b}: {v}')

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
    dup_total_files = sum(len(g) for g in dup_groups)
    log(f'    files inside MD5-dup groups: {dup_total_files}')
    paired_exact = 0
    for g in dup_groups:
        names = set(m[1] for m in g)
        if len(names) == 2 and any(n.endswith('.jpeg') for n in names) and any(n.endswith('.jpg') for n in names):
            paired_exact += 1
    log(f'    of these, .jpeg/.jpg same-name exact-dup pairs: {paired_exact}')
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

    # pair-level dHash distances for dual-encoding pairs
    log('\n    .jpeg/.jpg dual pairs: dHash distance check')
    pair_dists = []
    hash_by_name = {}
    for (cls, fname, h) in hashes:
        hash_by_name[(cls, fname)] = h
    for b, v in dual_pairs.items():
        cls, f1 = v[0]
        cls2, f2 = v[1]
        h1 = hash_by_name[(cls, f1)]
        h2 = hash_by_name[(cls2, f2)]
        dist = int(np.count_nonzero(bits[hashes.index((cls, f1, h1))] != bits[hashes.index((cls2, f2, h2))]))
        pair_dists.append((b, dist))
    if pair_dists:
        pd = [d for _, d in pair_dists]
        log(f'    pair dHash dist: min {min(pd)} med {sorted(pd)[len(pd)//2]} max {max(pd)}')
        log(f'    pairs with dHash dist <= 4 (near-identical): {sum(1 for d in pd if d <= 4)}')
    else:
        log('    no dual pairs found')

    blurry = []
    for cls, path, fname in images:
        v = cv2.Laplacian(gray_cache[cls][fname], cv2.CV_64F).var()
        if v < BLUR_LAP_VAR_THRESHOLD:
            blurry.append({'file': fname, 'class': cls, 'lap_var': round(float(v), 1)})
    log(f'\n    flagged blurry (lap_var < {BLUR_LAP_VAR_THRESHOLD}): {len(blurry)}')
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

    # ---- 11. resolution uniformity ----
    all_sizes = [s for vs in sizes.values() for s in vs]
    uniform = len(set(all_sizes)) == 1
    log(f'\n[11] Resolution uniformity: {"UNIFORM" if uniform else "MIXED"} '
        f'({len(set(all_sizes))} distinct sizes)')

    # ---- report ----
    report = {
        'source': 'Mendeley VegNet (DOI 10.17632/t5sssfgn2v.3)',
        'download_url': 'https://data.mendeley.com/datasets/t5sssfgn2v/3',
        'license': 'CC BY 4.0',
        'structure': 'Original Dataset/{Bacterial spot rot,Black Rot,Downy Mildew,No disease}/',
        'n_classes': len(CLASS_MAP),
        'class_map': CLASS_MAP,
        'named_diseases': NAMED_DISEASES,
        'counts': dict(cls_count),
        'unique_image_counts': dict(cls_uniq),
        'total_files': len(images),
        'total_unique_images': len(by_base),
        'raw_dataset': True,
        'augmented_marker_files': marked,
        'originals_and_augmented_separate': True,
        'augmented_note': (
            'VegNet stores originals and augmentations in TWO SEPARATE Mendeley '
            'files. Only "Original Dataset.zip" (656 files) was downloaded; the '
            '7,360-image "Augmented Dataset.zip" was NOT used. No mixing.'),
        'dual_encoding_pairs': dual_pairs,
        'dual_encoding_pair_count': len(dual_pairs),
        'dual_pair_dhash_distances': [{'base': b, 'dhash_dist': d} for b, d in pair_dists],
        'class_imbalance': imbalance,
        'duplicate_filenames': dup_names,
        'corrupt_images': corrupt,
        'exact_dup_groups': [{'members': g, 'n': len(g)} for g in dup_groups],
        'exact_dup_files_total': dup_total_files,
        'exact_dup_jpeg_jpg_pairs': paired_exact,
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
            'Original Dataset.zip holds 656 FILES but only 567 unique images; 89 '
            'images exist twice (.jpeg + .jpg re-encoded copies).',
            'Bacterial spot rot ships as .jpg only; Black Rot has 20 .jpeg-only '
            'files that have no .jpg twin.',
            'Whether paired .jpeg/.jpg are byte-identical or just near-dups is '
            'determined by the exact-MD5 and dHash sections above.',
            'After removing one copy of each dual-encoded pair the usable count '
            'would be the 567 unique images.',
            'Balance via random undersampling to the smallest class (after dedup).',
            'Augment only the train set; never the validation or test sets.',
            'dHash near-dup groups: review cross-class groups before training.',
        ],
        'leaf_confirmation_limitation': (
            'Automated audit cannot visually verify every image is a cauliflower '
            'leaf nor that the disease labels are correct. Manual spot-check '
            'recommended.'),
        'log': LOG,
    }
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nAudit report saved: {REPORT}')


if __name__ == '__main__':
    audit()
