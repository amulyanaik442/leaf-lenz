"""
Correct the Pumpkin dataset.

Builds `dataset/pumpkin/Corrected_Dataset/` from the existing Kaggle pumpkin
`Original` split plus the user's own powdery mildew web images.

Corrections applied:
  * `Bacterial Leaf Spot`   -> REMOVED  (contains incorrectly labelled images)
  * `Powdery Mildew`        -> RENAMED to `Bacterial Leaf Spot`
                                (those images actually belong to BLS)
  * new `Powdery Mildew`    -> created, filled with valid images copied from
                                the user's desktop Google-Search folder

Robustness:
  * Every image is validated (PIL open + verify); corrupted images are skipped.
  * Exact duplicates removed via MD5; near-duplicates removed via perceptual
    dHash (hamming distance).

Produces a JSON + console report with pre-balancing class counts.

Usage:
    python ml_model/correct_pumpkin_dataset.py
"""
import os
import sys
import json
import hashlib
import shutil

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(BASE_DIR, 'dataset', 'pumpkin', 'Original', 'Original')
DESKTOP_SOURCE = r'C:\Users\amuly\Desktop\powdery mildew - Google Search_files'
OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset', 'pumpkin', 'Corrected_Dataset')
REPORT_PATH = os.path.join(BASE_DIR, 'dataset', 'pumpkin', 'correct_dataset_report.json')

# The 5 canonical class folders (display names preserved from the dataset).
CLASS_NAMES = ['Bacterial Leaf Spot', 'Downy Mildew', 'Healthy Leaf',
               'Mosaic Disease', 'Powdery Mildew']

# Minimum side length (px) for desktop web images. Filters out tiny page icons
# (favicons, chrome assets) that Google caches alongside real photos.
MIN_IMAGE_DIM = 100

# dHash hamming distance threshold. Two images whose dHash differs by <= this
# number of bits are considered duplicates.
DHASH_THRESHOLD = 6

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP')


def is_valid_image(path):
    """Return True if `path` is a decodable image file."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def dhash(path, hash_size=8):
    """Perceptual hash (difference hash) for a single image."""
    try:
        with Image.open(path) as im:
            gray = im.convert('L').resize((hash_size + 1, hash_size),
                                          Image.BILINEAR)
    except Exception:
        return None
    diff = []
    px = gray.load()
    for row in range(hash_size):
        for col in range(hash_size):
            diff.append(1 if px[col, row] > px[col + 1, row] else 0)
    return int(''.join(str(b) for b in diff), 2)


def hamming(a, b):
    return bin(a ^ b).count('1')


def md5(path, chunk=65536):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def collect_desktop_images():
    """
    Collect valid powdery-mildew images from the user's desktop folder.

    Filters out browser chrome (favicon* files and tiny cached icons).
    """
    candidates = []
    for fname in os.listdir(DESKTOP_SOURCE):
        path = os.path.join(DESKTOP_SOURCE, fname)
        if os.path.isdir(path) or fname.startswith('favicon'):
            continue
        if not is_valid_image(path):
            continue
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            continue
        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            continue
        candidates.append(path)
    return candidates


def build_corrected_dataset():
    """Create the corrected, validated, de-duplicated dataset."""
    print('=' * 60)
    print('PUMPKIN DATASET CORRECTION')
    print('=' * 60)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    report = {
        'source_dir': SOURCE_DIR,
        'desktop_source': DESKTOP_SOURCE,
        'output_dir': OUTPUT_DIR,
        'corrections_applied': {
            'Bacterial Leaf Spot': 'REMOVED (mislabeled images)',
            'Powdery Mildew (old)': 'RENAMED to Bacterial Leaf Spot',
            'Powdery Mildew (new)': 'Created from desktop Google-Search images',
        },
        'images_before_balancing': {},
        'corrupted_skipped': {},
        'duplicates_removed': {},
        'empty_folders': [],
    }

    # ---- 1. Remove `Bacterial Leaf Spot` (skip entirely) ----
    bls_dir = os.path.join(SOURCE_DIR, 'Bacterial Leaf Spot')
    if os.path.isdir(bls_dir):
        report['corrupted_skipped']['Bacterial Leaf Spot (removed)'] = len(
            [f for f in os.listdir(bls_dir) if os.path.isfile(os.path.join(bls_dir, f))]
        )
    print('\n[1] Removing Bacterial Leaf Spot (mislabeled) ...')

    # ---- 2. Rename old Powdery Mildew -> Bacterial Leaf Spot ----
    old_pm = os.path.join(SOURCE_DIR, 'Powdery Mildew')
    print('[2] Renaming Powdery Mildew -> Bacterial Leaf Spot ...')

    # ---- 3. Copy relabelled + unchanged classes ----
    mapping = {
        'Downy Mildew': 'Downy Mildew',
        'Healthy Leaf': 'Healthy Leaf',
        'Mosaic Disease': 'Mosaic Disease',
        'Powdery Mildew': 'Bacterial Leaf Spot',  # relabel
    }
    print('[3] Copying classes from Original split ...')
    for src_folder, dest_folder in mapping.items():
        src = os.path.join(SOURCE_DIR, src_folder)
        if not os.path.isdir(src):
            print(f'    WARNING: source {src_folder} not found')
            continue
        dest = os.path.join(OUTPUT_DIR, dest_folder)
        os.makedirs(dest, exist_ok=True)
        copied = 0
        for f in os.listdir(src):
            sp = os.path.join(src, f)
            if not os.path.isfile(sp) or not f.lower().endswith(IMAGE_EXTS):
                continue
            if not is_valid_image(sp):
                report['corrupted_skipped'][src_folder] = \
                    report['corrupted_skipped'].get(src_folder, 0) + 1
                continue
            shutil.copy2(sp, os.path.join(dest, f))
            copied += 1
        print(f'    {src_folder:20s} -> {dest_folder:20s} : {copied} images')

    # ---- 4. Copy desktop powdery mildew images into new Powdery Mildew ----
    print('[4] Copying desktop Powdery Mildew images ...')
    pm_dir = os.path.join(OUTPUT_DIR, 'Powdery Mildew')
    os.makedirs(pm_dir, exist_ok=True)
    desktop_images = collect_desktop_images()
    copied_pm = 0
    for i, path in enumerate(desktop_images):
        if not is_valid_image(path):
            report['corrupted_skipped']['Powdery Mildew (desktop)'] = \
                report['corrupted_skipped'].get('Powdery Mildew (desktop)', 0) + 1
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in IMAGE_EXTS:
            ext = '.jpg'
        dest = os.path.join(pm_dir, f'desktop_powdery_mildew_{i:04d}{ext}')
        shutil.copy2(path, dest)
        copied_pm += 1
    print(f'    Powdery Mildew (desktop)      : {copied_pm} images copied')

    # ---- 5. Global duplicate removal (MD5 then perceptual dHash) ----
    print('\n[5] Removing duplicates (MD5 + perceptual dHash) ...')
    all_files = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(OUTPUT_DIR, cls)
        if os.path.isdir(cls_dir):
            for f in os.listdir(cls_dir):
                all_files.append((os.path.join(cls_dir, f), cls))

    # 5a. Exact duplicates by MD5
    md5_map = {}
    removed_md5 = []
    for path, cls in all_files:
        h = md5(path)
        if h in md5_map:
            removed_md5.append((path, cls, md5_map[h]))
            os.remove(path)
        else:
            md5_map[h] = (path, cls)
    print(f'    MD5 exact duplicates removed: {len(removed_md5)}')

    # 5b. Near duplicates by perceptual dHash
    hashes = []  # (hash, path, cls)
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(OUTPUT_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            p = os.path.join(cls_dir, f)
            if not os.path.isfile(p):
                continue
            h = dhash(p)
            if h is not None:
                hashes.append((h, p, cls))

    removed_dhash = []
    kept = []
    for h, p, cls in hashes:
        dup_of = None
        for kh, kp, kcls in kept:
            if hamming(h, kh) <= DHASH_THRESHOLD:
                dup_of = (kp, kcls)
                break
        if dup_of is not None:
            removed_dhash.append((p, cls, dup_of))
            os.remove(p)
        else:
            kept.append((h, p, cls))
    print(f'    dHash near-duplicates removed: {len(removed_dhash)}')

    report['duplicates_removed']['md5_exact'] = len(removed_md5)
    report['duplicates_removed']['dhash_near'] = len(removed_dhash)
    report['duplicates_removed']['total'] = len(removed_md5) + len(removed_dhash)

    # ---- 6. Final counts + empty-folder check ----
    print('\n[6] Final class distribution (pre-balancing):')
    total = 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(OUTPUT_DIR, cls)
        if not os.path.isdir(cls_dir):
            report['empty_folders'].append(cls)
            report['images_before_balancing'][cls] = 0
            print(f'    {cls:25s}: 0  (EMPTY)')
            continue
        count = len([f for f in os.listdir(cls_dir) if os.path.isfile(
            os.path.join(cls_dir, f))])
        report['images_before_balancing'][cls] = count
        total += count
        print(f'    {cls:25s}: {count}')
    report['total_images_before_balancing'] = total
    report['total_classes'] = len(CLASS_NAMES)

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=4)
    print(f'\nReport saved to {REPORT_PATH}')
    print(f'TOTAL: {total} images across {len(CLASS_NAMES)} classes')
    return report


if __name__ == '__main__':
    build_corrected_dataset()
