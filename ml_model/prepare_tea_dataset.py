"""
Prepare the Tea Leaf Disease dataset for integration into the general model.

Pipeline (mirrors `correct_pumpkin_dataset.py` + `balance_pumpkin_dataset.py`):
  1. Copy the pristine Kaggle download to `dataset/tea/Raw_Dataset` (untouched).
  2. Build `dataset/tea/Tea_Dataset` with validation + deduplication
     (MD5 exact duplicates and identical perceptual dHash).
  3. Balance every class up to TARGET_COUNT=150 using only geometric/colour
     augmentations (no AI-generated images) - approved plan.
  4. Stratified 80/10/10 split into `dataset/tea/Tea_Split`.

Class display names preserve the original dataset folder names.

Usage:
    python ml_model/prepare_tea_dataset.py
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
KAGGLE_RAW = r"C:\Users\amuly\.cache\kagglehub\datasets\shashwatwork\identifying-disease-in-tea-leafs\versions\1\tea sickness dataset"
RAW_DIR = os.path.join(BASE_DIR, 'dataset', 'tea', 'Raw_Dataset')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'tea', 'Tea_Dataset')
SPLIT_DIR = os.path.join(BASE_DIR, 'dataset', 'tea', 'Tea_Split')
REPORT_PATH = os.path.join(BASE_DIR, 'dataset', 'tea', 'tea_dataset_report.json')

CLASS_NAMES = ['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight',
               'gray light', 'healthy', 'red leaf spot', 'white spot']
TARGET_COUNT = 150
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


# ---------------------------------------------------------------------------
# Augmentations (identical helpers to balance_pumpkin_dataset.py)
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
    print('TEA DATASET PREPARATION')
    print(f'  Target: {TARGET_COUNT} images per class (approved plan)')
    print('=' * 60)

    # ---- 1. Raw copy ----
    print('\n[1] Copying pristine raw dataset...')
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    shutil.copytree(KAGGLE_RAW, RAW_DIR)
    raw_counts = {cls: count_images(os.path.join(RAW_DIR, cls))
                  for cls in CLASS_NAMES}
    for cls in CLASS_NAMES:
        print(f'    {cls:18s} {raw_counts[cls]}')

    # ---- 2. Build Tea_Dataset (validate + dedupe) ----
    print('\n[2] Building Tea_Dataset (validate + dedupe)...')
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR)
    report = {
        'raw_total': sum(raw_counts.values()),
        'raw_per_class': raw_counts,
        'dedupe_removed': [],
        'corrupt_skipped': [],
        'images_before_balancing': {},
    }
    all_files = []
    for cls in CLASS_NAMES:
        src = os.path.join(RAW_DIR, cls)
        dst = os.path.join(DATASET_DIR, cls)
        os.makedirs(dst, exist_ok=True)
        copied = 0
        for f in os.listdir(src):
            sp = os.path.join(src, f)
            if not os.path.isfile(sp) or not f.lower().endswith(IMAGE_EXTS):
                continue
            if not is_valid_image(sp):
                report['corrupt_skipped'].append(os.path.join(cls, f))
                continue
            shutil.copy2(sp, os.path.join(dst, f))
            all_files.append((os.path.join(dst, f), cls))
            copied += 1
        print(f'    {cls:18s} {copied} copied')

    # MD5 dedupe
    hmap = {}
    for path, cls in all_files:
        h = md5(path)
        if h in hmap:
            other_cls, other_path = hmap[h]
            # Keep the Anthracnose copy of the cross-class image (see analysis)
            keep = (cls == 'Anthracnose' and other_cls != 'Anthracnose')
            if keep:
                os.remove(other_path)
                report['dedupe_removed'].append(
                    f'{os.path.relpath(other_path, DATASET_DIR)} (dup of {os.path.relpath(path, DATASET_DIR)})')
                hmap[h] = (cls, path)
            else:
                os.remove(path)
                report['dedupe_removed'].append(
                    f'{os.path.relpath(path, DATASET_DIR)} (dup of {os.path.relpath(other_path, DATASET_DIR)})')
        else:
            hmap[h] = (cls, path)

    # Identical dHash dedupe (same class only)
    by_hash = {}
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        for f in os.listdir(cls_dir):
            p = os.path.join(cls_dir, f)
            if not os.path.isfile(p):
                continue
            h = dhash(p)
            if h is None:
                continue
            if h in by_hash:
                os.remove(p)
                report['dedupe_removed'].append(
                    f'{os.path.relpath(p, DATASET_DIR)} (near-dup of {os.path.relpath(by_hash[h], DATASET_DIR)})')
            else:
                by_hash[h] = p

    print(f'    Dedupe removed: {len(report["dedupe_removed"])}')
    for r in report['dedupe_removed']:
        print(f'      - {r}')

    before = {}
    for cls in CLASS_NAMES:
        n = count_images(os.path.join(DATASET_DIR, cls))
        before[cls] = n
        print(f'    {cls:18s} {n} (after dedupe)')
    report['images_before_balancing'] = before

    # ---- 3. Balance up to TARGET_COUNT ----
    print(f'\n[3] Balancing up to {TARGET_COUNT}/class...')
    added = {}
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        current = count_images(cls_dir)
        n = augment_class(cls_dir, cls.replace(' ', '_'), current, TARGET_COUNT)
        added[cls] = n
        print(f'    {cls:18s} {current} -> {current + n}  (+{n})')
    report['images_added_via_augmentation'] = added
    report['total_after_balancing'] = sum(
        count_images(os.path.join(DATASET_DIR, cls)) for cls in CLASS_NAMES)

    # ---- 4. Stratified 80/10/10 split ----
    print('\n[4] Stratified 80/10/10 split...')
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    rng = random.Random(SEED)
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
        print(f'    {cls:18s} train={n_train:3d} valid={n_val:3d} test={n - n_train - n_val:3d}')
    report['split'] = split_summary
    report['split_totals'] = {
        s: sum(split_summary[c][s] for c in CLASS_NAMES)
        for s in ('train', 'valid', 'test')
    }

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=4)
    print(f'\nReport saved to {REPORT_PATH}')
    print(f'Tea_Dataset: {report["total_after_balancing"]} images')


if __name__ == '__main__':
    main()
