"""
Augment minority classes to balance the wheat dataset.

Target: ~650 images per class (the average of the minority classes).
Only augments classes that are below the target.
Uses diverse augmentations to increase generalization.
"""
import os
import sys
import random
from PIL import Image, ImageEnhance, ImageFilter

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = r"C:\Users\amuly\OneDrive\Desktop\agriculture_ai-leaf_lenz\ml_model\wheat_data\train"
TARGET_COUNT = 650
SEED = 42
random.seed(SEED)

# Augmentation functions
def aug_flip_h(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def aug_flip_v(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def aug_rotate(img, angle=None):
    if angle is None:
        angle = random.uniform(-20, 20)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

def aug_brightness(img, factor=None):
    if factor is None:
        factor = random.uniform(0.6, 1.4)
    return ImageEnhance.Brightness(img).enhance(factor)

def aug_contrast(img, factor=None):
    if factor is None:
        factor = random.uniform(0.6, 1.4)
    return ImageEnhance.Contrast(img).enhance(factor)

def aug_saturation(img, factor=None):
    if factor is None:
        factor = random.uniform(0.6, 1.4)
    return ImageEnhance.Color(img).enhance(factor)

def aug_sharpness(img, factor=None):
    if factor is None:
        factor = random.uniform(0.5, 1.5)
    return ImageEnhance.Sharpness(img).enhance(factor)

def aug_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

def aug_crop_resize(img, size=224):
    w, h = img.size
    crop_pct = random.uniform(0.7, 0.9)
    cw = int(w * crop_pct)
    ch = int(h * crop_pct)
    left = random.randint(0, w - cw)
    top = random.randint(0, h - ch)
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)

def aug_jitter(img):
    """Combined small color jitter."""
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    return img

AUGMENTATIONS = [
    aug_flip_h,
    aug_flip_v,
    aug_rotate,
    aug_brightness,
    aug_contrast,
    aug_saturation,
    aug_sharpness,
    aug_blur,
    aug_crop_resize,
    aug_jitter,
    # Compound augmentations
    lambda img: aug_flip_h(aug_rotate(img)),
    lambda img: aug_brightness(aug_contrast(img)),
    lambda img: aug_crop_resize(aug_flip_h(img)),
    lambda img: aug_jitter(aug_rotate(img)),
    lambda img: aug_blur(aug_brightness(img)),
]


def augment_class(cls_path, cls_name, current_count, target_count):
    """Generate synthetic images for an underrepresented class."""
    needed = target_count - current_count
    if needed <= 0:
        print(f"  {cls_name:25s} {current_count:5d} -> OK (no augmentation needed)")
        return 0

    files = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"  {cls_name:25s} {current_count:5d} -> generating {needed} augmented images...")

    generated = 0
    while generated < needed:
        src_file = random.choice(files)
        src_path = os.path.join(cls_path, src_file)
        try:
            img = Image.open(src_path).convert("RGB")
            # Apply 2-3 random augmentations
            n_aug = random.randint(2, 3)
            augs = random.sample(AUGMENTATIONS, n_aug)
            for aug_fn in augs:
                img = aug_fn(img)

            # Save
            ext = os.path.splitext(src_file)[1]
            new_name = f"aug_{cls_name.lower().replace(' ', '_')}_{generated:04d}{ext}"
            img.save(os.path.join(cls_path, new_name))
            generated += 1
        except Exception as e:
            print(f"    Error augmenting {src_file}: {e}")
            continue

    print(f"    -> Generated {generated} new images. Total: {current_count + generated}")
    return generated


def main():
    print("="*50)
    print("  DATASET AUGMENTATION FOR CLASS BALANCE")
    print("="*50)
    print(f"  Target: {TARGET_COUNT} images per class")
    print(f"  Augmenting classes below target...\n")

    total_augmented = 0
    for cls in sorted(os.listdir(BASE)):
        cls_path = os.path.join(BASE, cls)
        if not os.path.isdir(cls_path):
            continue

        count = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        aug = augment_class(cls_path, cls, count, TARGET_COUNT)
        total_augmented += aug

    print(f"\n{'='*50}")
    print(f"  DONE: Generated {total_augmented} total augmented images")
    print(f"{'='*50}")

    # Verify final counts
    print("\nFinal class distribution:")
    for cls in sorted(os.listdir(BASE)):
        cls_path = os.path.join(BASE, cls)
        if not os.path.isdir(cls_path):
            continue
        count = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        print(f"  {cls:25s} {count:5d}")


if __name__ == "__main__":
    main()
