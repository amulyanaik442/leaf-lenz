"""
Visual inspection of random samples from each maize class.
Generates a grid image showing 5 random samples per class.

Usage:
    python ml_model/visual_inspect_maize.py
"""
import os
import sys
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIZE_DATA = os.path.join(BASE_DIR, "ml_model", "maize_data")
OUTPUT_PATH = os.path.join(BASE_DIR, "ml_model", "maize_visual_grid.png")


def main():
    train_dir = os.path.join(MAIZE_DATA, "train")
    if not os.path.exists(train_dir):
        print("ERROR: train directory not found.")
        sys.exit(1)

    classes = sorted([d for d in os.listdir(train_dir)
                      if os.path.isdir(os.path.join(train_dir, d))])
    print(f"Classes: {classes}")

    samples_per_class = 5
    thumb_size = (224, 224)
    padding = 10
    label_height = 30

    grid_w = samples_per_class * (thumb_size[0] + padding) + padding
    grid_h = len(classes) * (thumb_size[1] + label_height + padding) + padding
    grid = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))

    random.seed(42)

    for row, cls in enumerate(classes):
        cls_path = os.path.join(train_dir, cls)
        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        chosen = random.sample(files, min(samples_per_class, len(files)))

        for col, fname in enumerate(chosen):
            img_path = os.path.join(cls_path, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(thumb_size, Image.BILINEAR)
            except Exception as e:
                print(f"  Could not open {fname}: {e}")
                continue

            x = padding + col * (thumb_size[0] + padding)
            y = padding + row * (thumb_size[1] + label_height + padding)
            grid.paste(img, (x, y))

            draw = ImageDraw.Draw(grid)
            label = f"{cls} ({fname[:20]})"
            draw.text((x, y + thumb_size[1] + 2), label, fill=(0, 0, 0))

    grid.save(OUTPUT_PATH, dpi=(150, 150))
    print(f"\nVisual grid saved to: {OUTPUT_PATH}")
    print("Open the image to verify labels are correct.")


if __name__ == "__main__":
    main()
