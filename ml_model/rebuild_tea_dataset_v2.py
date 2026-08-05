"""
Rebuild the Tea dataset as v2 (approved plan):
  1. Merge 'brown blight' + 'gray light' -> 'blight' (both Pestalotiopsis spp.)
  2. Add independent tealeafBD images (algal leaf, blight = Brown+Gray, healthy)
     to broaden real-world domain coverage.
  3. Balance every class up to TARGET_COUNT=300 using only geometric/colour
     augmentations (same conventions as v1; no AI images).
  4. Stratified 80/10/10 split into `dataset/tea/Tea_Split_v2`.

Sources:
  * dataset/tea/Raw_Dataset (pristine Kaggle download, 8 classes -> 7)
  * ~/.cache/kagglehub/datasets/bmshahriaalam/tealeafbd-tea-leaf-disease-detection

Outputs:
  * dataset/tea/Tea_Dataset_v2   (7 classes, balanced to 300 each)
  * dataset/tea/Tea_Split_v2     (train/valid/test)
  * dataset/tea/tea_dataset_v2_report.json

Usage:
    python ml_model/rebuild_tea_dataset_v2.py
"""
import os
import sys
import json
import hashlib
import random
import shutil

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image, ImageEnhance, ImageFilter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'dataset', 'tea', 'Raw_Dataset')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'tea', 'Tea_Dataset_v2')
SPLIT_DIR = os.path.join(BASE_DIR, 'dataset', 'tea', 'Tea_Split_v2')
REPORT_PATH = os.path.join(BASE_DIR, 'dataset', 'tea', 'tea_dataset_v2_report.json')
TEALEAFBD_DIR = os.path.join(
    os.path.expanduser('~'), '.cache', 'kagglehub', 'datasets',
    'bmshahriaalam', 'tealeafbd-tea-leaf-disease-detection',
    'versions', '1', 'teaLeafBD', 'teaLeafBD')

# Merged 7-class schema (folder names preserved from the original naming)
CLASS_NAMES = ['Anthracnose', 'algal leaf', 'bird eye spot', 'blight',
               'healthy', 'red leaf spot', 'white spot']
RAW_CLASS_MAP = {
    'Anthracnose': 'Anthracnose',
    'algal leaf': 'algal leaf',
    'bird eye spot': 'bird eye spot',
    'brown blight': 'blight',
    'gray light': 'blight',
    'healthy': 'healthy',
    'red leaf spot': 'red leaf spot',
    'white spot': 'white spot',
}
TEALEAFBD_CLASS_MAP = {
    '1. Tea algal leaf spot': 'algal leaf',
    '2. Brown Blight': 'blight',
    '3. Gray Blight': 'blight',
    '7. Healthy leaf': 'healthy',
}
TARGET_COUNT = 300
TRAIN_RATIO = 0.80
SEED = 42
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP')

random.seed(SEED)


def is_valid_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def md5(path, chunk=65536):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dhash(path, hash_size=8):
    try:
        with Image.open(path) as im:
            gray = im.convert('L').resize((hash_size + 1, hash_size),
                                          Image.BILINEAR)
    except Exception:
        return None
    px = gray.load()
    bits = 0
    for r in range(hash_size):
        for c in range(hash_size):
            bits = (bits << 1) | (1 if px[c, r] > px[c + 1, r] else 0)
    return bits


def count_images(cls_dir):
    if not os.path.isdir(cls_dir):
        return 0
    return len([f for f in os.listdir(cls_dir)
                if os.path.isfile(os.path.join(cls_dir, f))
                and f.lower().endswith(IMAGE_EXTS)])


def list_images(cls_dir):
    if not os.path.isdir(cls_dir):
        return []
    return sorted([os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                   if os.path.isfile(os.path.join(cls_dir, f))
                   and f.lower().endswith(IMAGE_EXTS)])


# ---------------------------------------------------------------------------
# Augmentations (identical helpers to v1 / balance_pumpkin_dataset.py)
# ---------------------------------------------------------------------------
def aug_flip_h(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def aug_flip_v(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def aug_rotate(img):
    return img.rotate(random.uniform(-20, 20), resample=Image.BILINEAR,
                      fillcolor=(128, 128, 128))


def aug_brightness(img):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))


def aug_contrast(img):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))


def aug_saturation(img):
    return ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.4))


def aug_sharpness(img):
    return ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 1.5))


def aug_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))


def aug_crop_resize(img):
    w, h = img.size
    crop_pct = random.uniform(0.7, 0.9)
    cw = int(w * crop_pct)
    ch = int(h * crop_pct)
    left = random.randint(0, w - cw)
    top = random.randint(0, h - ch)
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def aug_translate(img):
    w, h = img.size
    dx = random.randint(-int(0.1 * w), int(0.1 * w))
    dy = random.randint(-int(0.1 * h), int(0.1 * h))
    return img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                         fillcolor=(128, 128, 128))


def aug_jitter(img):
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    return img


AUGMENTATIONS = [
    aug_flip_h, aug_flip_v, aug_rotate, aug_brightness, aug_contrast,
    aug_saturation, aug_sharpness, aug_blur, aug_crop_resize, aug_translate,
    aug_jitter,
    lambda img: aug_flip_h(aug_rotate(img)),
    lambda img: aug_brightness(aug_contrast(img)),
    lambda img: aug_crop_resize(aug_flip_h(img)),
    lambda img: aug_jitter(aug_rotate(img)),
    lambda img: aug_blur(aug_brightness(img)),
    lambda img: aug_translate(aug_flip_v(img)),
]


def augment_class(cls_dir, cls_slug, current, target):
    needed = target - current
    if needed <= 0:
        return 0
    files = [f for f in os.listdir(cls_dir)
             if os.path.isfile(os.path.join(cls_dir, f))
             and f.lower().endswith(IMAGE_EXTS)]
    generated = 0
    while generated < needed:
        src = random.choice(files)
        try:
            img = Image.open(os.path.join(cls_dir, src)).convert('RGB')
        except Exception:
            continue
        n_aug = random.randint(2, 3)
        for aug_fn in random.sample(AUGMENTATIONS, n_aug):
            img = aug_fn(img)
        ext = os.path.splitext(src)[1]
        img.save(os.path.join(cls_dir, f'aug_{cls_slug}_{generated:04d}{ext}'))
        generated += 1
    return generated


# ---------------------------------------------------------------------------
def main():
    print('=' * 60)
    print('TEA DATASET V2 REBUILD (merge blight + tealeafBD)')
    print(f'  Target: {TARGET_COUNT} images per class (approved plan)')
    print('=' * 60)

    report = {'raw_per_source': {}, 'dedupe_removed': [], 'corrupt_skipped': [],
              'images_before_balancing': {}, 'images_added_via_augmentation': {},
              'total_after_balancing': 0, 'split': {}, 'split_totals': {}}

    # ---- 1. Collect sources into staging dirs ----
    staging = os.path.join(BASE_DIR, 'dataset', 'tea', '_staging_v2')
    if os.path.exists(staging):
        shutil.rmtree(staging)
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(staging, cls), exist_ok=True)

    # 1a. Raw_Dataset (pristine)
    raw_counts = {}
    for src_cls, dst_cls in RAW_CLASS_MAP.items():
        sdir = os.path.join(RAW_DIR, src_cls)
        n = 0
        for f in os.listdir(sdir):
            sp = os.path.join(sdir, f)
            if not os.path.isfile(sp) or not f.lower().endswith(IMAGE_EXTS):
                continue
            if not is_valid_image(sp):
                report['corrupt_skipped'].append(f'Raw_Dataset/{src_cls}/{f}')
                continue
            shutil.copy2(sp, os.path.join(staging, dst_cls, f))
            n += 1
        raw_counts[f'Raw_Dataset/{src_cls}'] = n
    print('\n[1] Raw_Dataset (merged to 7 classes):')
    for src_cls, dst_cls in RAW_CLASS_MAP.items():
        print(f'    {src_cls:14s} -> {dst_cls:14s} {raw_counts[f"Raw_Dataset/{src_cls}"]}')

    # 1b. tealeafBD
    teab_counts = {}
    for src_cls, dst_cls in TEALEAFBD_CLASS_MAP.items():
        sdir = os.path.join(TEALEAFBD_DIR, src_cls)
        n = 0
        if os.path.isdir(sdir):
            for f in os.listdir(sdir):
                sp = os.path.join(sdir, f)
                if not os.path.isfile(sp) or not f.lower().endswith(IMAGE_EXTS):
                    continue
                if not is_valid_image(sp):
                    report['corrupt_skipped'].append(f'tealeafBD/{src_cls}/{f}')
                    continue
                shutil.copy2(sp, os.path.join(staging, dst_cls, f))
                n += 1
        teab_counts[f'tealeafBD/{src_cls}'] = n
    print('\n    tealeafBD (added):')
    for src_cls, dst_cls in TEALEAFBD_CLASS_MAP.items():
        print(f'    {src_cls:24s} -> {dst_cls:14s} {teab_counts[f"tealeafBD/{src_cls}"]}')

    report['raw_per_source'] = {**raw_counts, **teab_counts}

    # ---- 2. Dedupe (MD5 exact + identical dHash) per class ----
    print('\n[2] Dedupe per class...')
    for cls in CLASS_NAMES:
        cdir = os.path.join(staging, cls)
        hmap = {}
        for f in os.listdir(cdir):
            p = os.path.join(cdir, f)
            if not os.path.isfile(p):
                continue
            h = md5(p)
            if h in hmap:
                os.remove(p)
                report['dedupe_removed'].append(f'{cls}/{f} (MD5 dup of {hmap[h]})')
            else:
                hmap[h] = f
        hset = {}
        for f in os.listdir(cdir):
            p = os.path.join(cdir, f)
            if not os.path.isfile(p):
                continue
            h = dhash(p)
            if h is None:
                continue
            if h in hset:
                os.remove(p)
                report['dedupe_removed'].append(f'{cls}/{f} (near-dup of {hset[h]})')
            else:
                hset[h] = f
    print(f'    Removed: {len(report["dedupe_removed"])}')
    for r in report['dedupe_removed']:
        print(f'      - {r}')

    # ---- 3. Subsample big classes to TARGET_COUNT, then augment small ones ----
    print(f'\n[3] Balancing up to {TARGET_COUNT}/class...')
    rng = random.Random(SEED)
    before = {}
    for cls in CLASS_NAMES:
        cdir = os.path.join(staging, cls)
        files = list_images(cdir)
        before[cls] = len(files)
        if len(files) > TARGET_COUNT:
            rng.shuffle(files)
            for f in files[TARGET_COUNT:]:
                os.remove(f)
        print(f'    {cls:16s} {before[cls]:4d} -> selected {min(len(files), TARGET_COUNT):3d}')
    report['images_before_balancing'] = before

    # Move staging into final Tea_Dataset_v2
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR)
    for cls in CLASS_NAMES:
        shutil.move(os.path.join(staging, cls), os.path.join(DATASET_DIR, cls))

    added = {}
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        current = count_images(cls_dir)
        n = augment_class(cls_dir, cls.replace(' ', '_'), current, TARGET_COUNT)
        added[cls] = n
        print(f'    {cls:16s} {current} -> {current + n}  (+{n})')
    report['images_added_via_augmentation'] = added
    report['total_after_balancing'] = sum(
        count_images(os.path.join(DATASET_DIR, cls)) for cls in CLASS_NAMES)

    shutil.rmtree(staging, ignore_errors=True)

    # ---- 4. Stratified 80/10/10 split ----
    print('\n[4] Stratified 80/10/10 split...')
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    split_summary = {}
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        files = sorted([f for f in os.listdir(cls_dir)
                        if os.path.isfile(os.path.join(cls_dir, f))
                        and f.lower().endswith(IMAGE_EXTS)])
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * 0.10)
        for split_name, subset in [('train', files[:n_train]),
                                   ('valid', files[n_train:n_train + n_val]),
                                   ('test', files[n_train + n_val:])]:
            dest = os.path.join(SPLIT_DIR, split_name, cls)
            os.makedirs(dest, exist_ok=True)
            for f in subset:
                shutil.copy2(os.path.join(cls_dir, f), os.path.join(dest, f))
        split_summary[cls] = {'train': n_train, 'valid': n_val,
                              'test': n - n_train - n_val}
        print(f'    {cls:16s} train={n_train:3d} valid={n_val:3d} test={n - n_train - n_val:3d}')
    report['split'] = split_summary
    report['split_totals'] = {
        s: sum(split_summary[c][s] for c in CLASS_NAMES)
        for s in ('train', 'valid', 'test')
    }

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=4)
    print(f'\nReport saved to {REPORT_PATH}')
    print(f'Tea_Dataset_v2: {report["total_after_balancing"]} images')


if __name__ == '__main__':
    main()
