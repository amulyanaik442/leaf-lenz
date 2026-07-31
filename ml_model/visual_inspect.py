"""Generate a visual inspection grid: 5 random samples per class."""
import os
import random
from PIL import Image, ImageDraw, ImageFont

BASE = r"C:\Users\amuly\OneDrive\Desktop\agriculture_ai-leaf_lenz\ml_model\wheat_data\train"
OUT = r"C:\Users\amuly\OneDrive\Desktop\agriculture_ai-leaf_lenz\ml_model\dataset_visual_check.png"

classes = sorted([d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d))])
n_classes = len(classes)
samples_per_class = 5
thumb_size = 128
padding = 10
header_height = 30

grid_w = samples_per_class * (thumb_size + padding) + padding
grid_h = n_classes * (thumb_size + padding + header_height) + padding

grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
draw = ImageDraw.Draw(grid)

for row, cls in enumerate(classes):
    cls_path = os.path.join(BASE, cls)
    files = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    samples = random.sample(files, min(samples_per_class, len(files)))

    y_offset = row * (thumb_size + padding + header_height) + padding
    draw.text((padding, y_offset), cls, fill=(0, 0, 0))
    y_offset += header_height

    for col, fname in enumerate(samples):
        img_path = os.path.join(cls_path, fname)
        try:
            img = Image.open(img_path).convert("RGB").resize((thumb_size, thumb_size))
            x_offset = padding + col * (thumb_size + padding)
            grid.paste(img, (x_offset, y_offset))
        except Exception as e:
            draw.text((padding + col * (thumb_size + padding), y_offset), "ERR", fill=(255, 0, 0))

grid.save(OUT)
print(f"Saved visual inspection grid to {OUT}")
print(f"Classes: {n_classes}, Samples per class: {samples_per_class}")
print("Review the image to check for mislabeled or low-quality samples.")
