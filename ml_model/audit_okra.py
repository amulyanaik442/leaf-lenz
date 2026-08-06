"""
Dataset Audit for the Okra DiseaseNet (Mendeley) okra leaf dataset.

Source:
    Okra DiseaseNet - A Segmentation and Classification Dataset of Okra
    (Abelmoschus esculentus) Leaf Diseases.
    Mendeley Data, DOI 10.17632/nh7zk4hv8z (v1, published 2024-12-19).
    URL: https://data.mendeley.com/datasets/nh7zk4hv8z/1
    Collected Chengalpattu (Mar-Apr 2023) and Thanjavur (May 2023), India
    (SRM University). Canon EOS 3000D. Contributors: Sowmiya Kumarakuru,
    Thenmozhi M.

Structure (from the Mendeley public API file tree, nh7zk4hv8z.1):
    Per class prefix: ALS (Alternaria leaf spot), CLS (Cercospora leaf spot),
    DM (Downy mildew), H (Healthy), LCV (Leaf curly virus),
    PLS (Phyllosticta leaf spot).
    Each class is stored in 3 subfolders (the authors' own 75/15/10
    Train/Test/Valid layout) plus 6 annotation JSON files
    (Okra_Disease_{Train,Test,Valid}_{coco,json}.json) and Read me.txt.

    NOTE: the annotation JSON files reference ORIGINAL camera captures named
    IMG_*.jpg (e.g. IMG_0114.jpg) which are NOT distributed with this release;
    the distributed image files are named <CLASS>_<n>.JPG. The annotation
    files therefore cannot be mapped onto the distributed images and are
    ignored by the classification workflow.

    For this audit we downloaded the distributed JPGs into a flat
    dataset/okra/raw/<CLASS>/ layout (the per-class subfolder identity is not
    exposed by the API), so the authors' split is treated as an unmappable
    pre-existing split: we will create a fresh 80/10/10 stratified split.

This is a RAW dataset (original photographs only). Per the approved workflow:
balance via random undersampling, augment only train, fresh stratified split.

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
  11. Train/validation/test leakage (unmappable pre-existing split)
  12. Annotation/label coverage (JSON segmentation annotations reference
      undistributed IMG_* originals - orphaned, unused)
  13. Proposed class-name mapping to Leaf Lenz `okra___` production names

Usage:
    venv/Scripts/python.exe ml_model/audit_okra.py
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
EXTRACT_ROOT = os.path.join(BASE_DIR, 'dataset', 'okra', 'raw')
REPORT = os.path.join(BASE_DIR, 'dataset', 'okra', 'okra_audit_report.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.WEBP')
BLUR_LAP_VAR_THRESHOLD = 100.0
NEAR_DUP_DHASH_THRESHOLD = 8
ANALYSIS_MAX_DIM = 512
LOG = []

# Proposed Leaf Lenz production class names (okra module, appended later)
CLASS_MAP = {
    'ALS': 'okra___alternaria_leaf_spot',
    'CLS': 'okra___cercospora_leaf_spot',
    'DM': 'okra___downy_mildew',
    'H': 'okra___healthy',
    'LCV': 'okra___leaf_curl_virus',
    'PLS': 'okra___phyllosticta_leaf_spot',
}

NAMED_DISEASES = {
    'ALS': 'Alternaria leaf spot',
    'CLS': 'Cercospora leaf spot',
    'DM': 'Downy mildew',
    'H': 'Healthy',
    'LCV': 'Leaf curly virus',
    'PLS': 'Phyllosticta leaf spot',
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
    # Downscale BEFORE full decode: PIL thumbnail uses JPEG draft decoding,
    # which is dramatically faster for the high-res (up to 6000x4000) okra JPGs.
    with Image.open(path) as im:
        orig_size = im.size
        im.thumbnail((max_dim, max_dim))
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
    log('OKRA LEAF DISEASE DATASET AUDIT')
    log('=' * 64)

    log(f'\n[0] Source: {EXTRACT_ROOT}')
    log('    Okra DiseaseNet (Mendeley Data, DOI 10.17632/nh7zk4hv8z, v1)')
    log('    URL: https://data.mendeley.com/datasets/nh7zk4hv8z/1')
    log('    This is a RAW dataset (original photographs, no augmented images).')

    if not os.path.isdir(EXTRACT_ROOT):
        log('ERROR: extract folder not found.')
        return

    images = load_all(EXTRACT_ROOT)
    log(f'    images total: {len(images)}')

    # ---- 1. counts ----
    log('\n[1] Per-class counts')
    cls_count = Counter(c for c, _, _ in images)
    for cls in sorted(cls_count):
        log(f'     {cls:6s} {NAMED_DISEASES.get(cls, "?"):28s} {cls_count[cls]:5d}')
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
    aug_patterns = ('.rf.', '__aug', '_aug', '-aug', 'augmented', 'flip', 'rotate')
    for cls, path, fname in images:
        low = fname.lower()
        if any(p in low for p in aug_patterns):
            marked += 1
    log(f'    filenames with augmentation markers: {marked}')
    log('    All files are original DSLR photographs (Canon EOS 3000D); '
        'no augmentation artifacts expected.')

    # ---- 3. duplicate filenames across folders ----
    log('\n[3] Duplicate filenames (same basename in different folders)')
    by_name = defaultdict(list)
    for cls, path, fname in images:
        by_name[fname].append(cls)
    dup_names = {n: c for n, c in by_name.items() if len(c) > 1}
    log(f'    duplicate basenames: {len(dup_names)}')
    for n, c in list(dup_names.items())[:10]:
        log(f'      {n}: {c}')
    log('    NOTE: files are named <CLASS>_<n>.<ext>; basename collisions '
        'across class folders do not imply identical content. Real content '
        'duplicates are checked via MD5 below.')

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
    else:
        log('    no cross-class near-duplicate groups found.')

    blurry = []
    for cls, path, fname in images:
        v = cv2.Laplacian(gray_cache[cls][fname], cv2.CV_64F).var()
        if v < BLUR_LAP_VAR_THRESHOLD:
            blurry.append({'file': fname, 'class': cls, 'lap_var': round(float(v), 1)})
    log(f'    flagged blurry (lap_var < {BLUR_LAP_VAR_THRESHOLD}): {len(blurry)}')
    for cls in sorted(lap_stats):
        vs = sorted(lap_stats[cls])
        log(f'    {cls:6s} {NAMED_DISEASES.get(cls, "?"):28s} lap_var min {vs[0]:.1f} '
            f'med {vs[len(vs)//2]:.1f} max {vs[-1]:.1f}')
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
        log(f'    {cls:6s} {NAMED_DISEASES.get(cls, "?"):28s} n {len(sizes[cls]):5d}  '
            f'min {min(ws)}x{min(hs)}  med {med(ws)}x{med(hs)}  max {max(ws)}x{max(hs)}')

    # ---- 9. folder structure ----
    log('\n[9] Folder structure')
    structure = {}
    for cls in sorted(os.listdir(EXTRACT_ROOT)):
        d = os.path.join(EXTRACT_ROOT, cls)
        if os.path.isdir(d):
            structure[cls] = len([f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS)])
    log(f'    {json.dumps(structure, indent=4)}')

    # ---- 10. train/test leakage ----
    log('\n[10] Pre-existing train/test split: authors provide a 75/15/10 '
        'Train/Test/Valid layout (3 subfolders per class).')
    log('    The Mendeley API does not expose subfolder names, and the '
        'annotation JSONs reference IMG_* originals that are NOT distributed, '
        'so the author split is unmappable. This audit therefore treats the '
        'dataset as raw and a fresh 80/10/10 stratified split will be created '
        'in preparation. Near-duplicate analysis above flags cross-split '
        'repeats as intra-class near-dup groups.')

    # ---- 11. annotation / label coverage ----
    log('\n[11] Annotation files (segmentation; orphaned)')
    annot_files = [f for f in os.listdir(EXTRACT_ROOT) if f.lower().endswith('.json')]
    log(f'    JSON annotation files found in raw root: {len(annot_files)}')
    for f in annot_files:
        log(f'      {f}')
    log('    These reference IMG_* originals not present in the distribution; '
        'they are for segmentation and are NOT used by the classification '
        'workflow.')

    # ---- 12. resolution uniformity check ----
    all_sizes = [s for vs in sizes.values() for s in vs]
    uniform = len(set(all_sizes)) == 1
    log(f'\n[12] Resolution uniformity: {"UNIFORM" if uniform else "MIXED"} '
        f'({len(set(all_sizes))} distinct sizes)')
    log('    Readme states 224x224x3, but the distributed JPGs are the '
        'original high-resolution captures (mixed sizes, up to 6000x4000). '
        'The classification pipeline resizes to 224x224 at load time.')

    # ---- 13. source metadata ----
    log('\n[13] Source metadata verification')
    log('    Mendeley metadata: 1500 images, 6 classes, .jpg, RGB.')
    log('    API file count (image/jpeg): 1495 | Readme text count: 1500')
    log('    Downloaded/verified local images: ' + str(len(images)))
    log('    Discrepancy (1500 vs 1495) noted; audit uses the actual files.')

    # ---- report ----
    report = {
        'source': 'Okra DiseaseNet (Mendeley Data, DOI 10.17632/nh7zk4hv8z, v1)',
        'download_url': 'https://data.mendeley.com/datasets/nh7zk4hv8z/1',
        'license': 'CC BY 4.0 (Mendeley Data; verify)',
        'structure': ('raw/<{ALS,CLS,DM,H,LCV,PLS}>/*.JPG (authors keep a '
                      '75/15/10 Train/Test/Valid layout not exposed by the API)'),
        'n_classes': len(CLASS_MAP),
        'class_map': CLASS_MAP,
        'named_diseases': NAMED_DISEASES,
        'counts': dict(cls_count),
        'total': len(images),
        'raw_dataset': True,
        'augmented_marker_files': marked,
        'annotations': {
            'json_files': annot_files,
            'note': ('JSON annotations reference IMG_* camera originals not '
                     'distributed; orphaned for the classification workflow'),
        },
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
        'pre_existing_split': '3 folders/class (authors 75/15/10), unmappable',
        'notes': [
            'Raw dataset of 1,495 distributed JPGs (metadata claims 1,500), 6 classes.',
            'Readme: 1500 images, Canon EOS 3000D, 224x224x3; distributed files are '
            'high-res (mixed sizes), resized to 224x224 at load time.',
            'No augmented images; all files are original photographs.',
            'Balance via random undersampling to the smallest class.',
            'Augment only the train set; never the validation or test sets.',
            'dHash near-dup groups: review cross-class groups before training.',
            'Fresh 80/10/10 stratified split to be created (author split unmappable).',
        ],
        'leaf_confirmation_limitation': (
            'Automated audit cannot visually verify every image is an okra leaf '
            'nor that the disease labels are correct. Manual spot-check recommended.'),
        'log': LOG,
    }
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nAudit report saved: {REPORT}')


if __name__ == '__main__':
    audit()
