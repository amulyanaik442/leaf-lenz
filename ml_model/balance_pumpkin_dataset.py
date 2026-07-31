"""
Balance the corrected Pumpkin dataset.

Target: 400 images per class.
  * This is the SAME per-class balance the existing Pumpkin pipeline already
    uses (the Kaggle `Original` split ships 400 images per class, and the
    project's rust-merge step caps classes at 400 - see
    `ml_model/merge_wheat_rust.py`).
  * Classes above the target are randomly undersampled.
  * Classes below the target are augmented (flip, rotate, zoom, brightness /
    contrast / saturation, blur, translation) up to the target.

No AI-generated images are used. Class names and folder structure are
preserved.

Produces a JSON + console post-balancing report.

Usage:
    python ml_model/balance_pumpkin_dataset.py
"""
import os
import sys
import json
import random

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image, ImageEnhance, ImageFilter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'pumpkin', 'Corrected_Dataset')
REPORT_PATH = os.path.join(BASE_DIR, 'dataset', 'pumpkin', 'balance_report.json')

TARGET_COUNT = 400          # existing per-class target in the pumpkin pipeline
SEED = 42

CLASS_NAMES = ['Bacterial Leaf Spot', 'Downy Mildew', 'Healthy Leaf',
               'Mosaic Disease', 'Powdery Mildew']
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP')

random.seed(SEED)

# ---------------------------------------------------------------------------
# Augmentation helpers (same style as ml_model/augment_dataset.py)
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
    """Random zoom (crop + resize back to original size)."""
    w, h = img.size
    crop_pct = random.uniform(0.7, 0.9)
    cw = int(w * crop_pct)
    ch = int(h * crop_pct)
    left = random.randint(0, w - cw)
    top = random.randint(0, h - ch)
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)

def aug_translate(img):
    """Random translation with wrap-around padding."""
    w, h = img.size
    dx = random.randint(-int(0.1 * w), int(0.1 * w))
    dy = random.randint(-int(0.1 * h), int(0.1 * h))
    return img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                         fillcolor=(128, 128, 128))

def aug_jitter(img):
    """Small combined colour jitter."""
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    return img

AUGMENTATIONS = [
    aug_flip_h, aug_flip_v, aug_rotate, aug_brightness, aug_contrast,
    aug_saturation, aug_sharpness, aug_blur, aug_crop_resize, aug_translate,
    aug_jitter,
    # Compound transforms
    lambda img: aug_flip_h(aug_rotate(img)),
    lambda img: aug_brightness(aug_contrast(img)),
    lambda img: aug_crop_resize(aug_flip_h(img)),
    lambda img: aug_jitter(aug_rotate(img)),
    lambda img: aug_blur(aug_brightness(img)),
    lambda img: aug_translate(aug_flip_v(img)),
]


def count_images(cls_dir):
    if not os.path.isdir(cls_dir):
        return 0
    return len([f for f in os.listdir(cls_dir)
                if os.path.isfile(os.path.join(cls_dir, f))
                and f.lower().endswith(IMAGE_EXTS)])


def undersample(cls_dir, target):
    """Randomly remove images until the class has `target` images."""
    files = [f for f in os.listdir(cls_dir)
             if os.path.isfile(os.path.join(cls_dir, f))
             and f.lower().endswith(IMAGE_EXTS)]
    remove = len(files) - target
    if remove <= 0:
        return 0
    to_remove = random.sample(files, remove)
    for f in to_remove:
        os.remove(os.path.join(cls_dir, f))
    return len(to_remove)


def augment_class(cls_dir, current, target, cls_slug):
    """Generate augmented images to bring a class up to the target."""
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
        name = f'aug_{cls_slug}_{generated:04d}{ext}'
        img.save(os.path.join(cls_dir, name))
        generated += 1
    return generated


def main():
    print('=' * 60)
    print('PUMPKIN DATASET BALANCING')
    print(f'  Target: {TARGET_COUNT} images per class (pipeline default)')
    print('=' * 60)

    report = {
        'target_per_class': TARGET_COUNT,
        'rationale': 'Existing per-class balance used by the Pumpkin training '
                     'pipeline (Kaggle Original split = 400/class; project '
                     'cap MAX_PER_CLASS=400 in merge_wheat_rust.py).',
        'before_balancing': {},
        'augmented_added': {},
        'undersampled_removed': {},
        'after_balancing': {},
        'total_after_balancing': 0,
    }

    print('\nPre-balance distribution:')
    total_before = 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        n = count_images(cls_dir)
        total_before += n
        report['before_balancing'][cls] = n
        print(f'  {cls:25s}: {n}')

    print('\nBalancing:')
    total_added = total_removed = 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        n = count_images(cls_dir)
        removed = 0
        added = 0
        if n > TARGET_COUNT:
            removed = undersample(cls_dir, TARGET_COUNT)
        elif n < TARGET_COUNT:
            added = augment_class(cls_dir, n, TARGET_COUNT, cls.lower().replace(' ', '_'))
        report['augmented_added'][cls] = added
        report['undersampled_removed'][cls] = removed
        total_added += added
        total_removed += removed
        status = 'undersampled' if removed else ('augmented' if added else 'unchanged')
        print(f'  {cls:25s}: {n} -> {TARGET_COUNT}  ({status})')

    print('\nPost-balance distribution:')
    total = 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        n = count_images(cls_dir)
        total += n
        report['after_balancing'][cls] = n
        print(f'  {cls:25s}: {n}')

    report['total_images_before_balancing'] = total_before
    report['images_added_through_augmentation'] = total_added
    report['images_removed_through_undersampling'] = total_removed
    report['total_after_balancing'] = total

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=4)
    print(f'\nReport saved to {REPORT_PATH}')
    print(f'  Added via augmentation:   {total_added}')
    print(f'  Removed via undersample:  {total_removed}')
    print(f'  TOTAL after balancing:    {total}')


if __name__ == '__main__':
    main()
