"""
Build a cleaned Groundnut dataset by:
1. Reclassifying 4 confirmed mislabels (IMG_4306,4310,4311,4312: early_leaf_spot -> Healthy)
2. Excluding top-30 most suspicious mislabels (model confidence > 0.85, disagrees with label)
3. Retaining all other images
"""
import os, json, shutil
import numpy as np
from PIL import Image
import onnxruntime as ort
from collections import defaultdict

BASE_DIR = r'C:\Users\amuly\Desktop\leaf-lenz-main'
ML_ASSETS = os.path.join(BASE_DIR, 'detector', 'ml_assets')

with open(os.path.join(ML_ASSETS, 'class_names.json')) as f:
    CLASS_NAMES = json.load(f)

session = ort.InferenceSession(os.path.join(ML_ASSETS, 'model.onnx'))
input_name = session.get_inputs()[0].name
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def predict(img_path):
    img = Image.open(img_path).convert('RGB').resize((256, 256), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]; d = 224
    crop = arr[(h-d)//2:(h+d)//2, (w-d)//2:(w+d)//2]
    normed = (crop - MEAN) / STD
    tensor = np.expand_dims(normed.transpose(2, 0, 1), axis=0).astype(np.float32)
    logits = session.run(None, {input_name: tensor})[0][0]
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / np.sum(exp_l)
    top1 = np.argmax(probs)
    return CLASS_NAMES[top1], float(probs[top1])

DATA_BASE = os.path.expandvars(r'%USERPROFILE%\.cache\kagglehub\datasets\warcoder\groundnut-plant-leaf-data\versions\1\Dataset of groundnut plant leaf images for classification and detection')

RAW_CLASSES = {
    'Healthy': os.path.join(DATA_BASE, 'Raw_Data', 'healthy leaf'),
    'Early_Leaf_Spot': os.path.join(DATA_BASE, 'Raw_Data', 'early_leaf_spot'),
    'Late_Leaf_Spot': os.path.join(DATA_BASE, 'Raw_Data', 'late leaf spot'),
    'Nutrition_Deficiency': os.path.join(DATA_BASE, 'Raw_Data', 'nutrition deficiency'),
    'Rust': os.path.join(DATA_BASE, 'Raw_Data', 'rust'),
}

LABEL_TO_GN = {
    'Healthy': 'Groundnut___Healthy',
    'Early_Leaf_Spot': 'Groundnut___Leaf_Spot',
    'Late_Leaf_Spot': 'Groundnut___Leaf_Spot',
    'Nutrition_Deficiency': 'Groundnut___Nutrition_Deficiency',
    'Rust': 'Groundnut___Rust',
}

CLEANED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'groundnut_cleaned')

# --- STEP 1: Run model on ALL Raw_Data images to find suspicious ones ---
print("STEP 1: Running model on all Raw_Data images...")
suspicious = []
for cls, folder in RAW_CLASSES.items():
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
        fp = os.path.join(folder, f)
        true_gn = LABEL_TO_GN[cls]
        pred, conf = predict(fp)
        if pred.startswith('Groundnut___') and pred != true_gn and conf > 0.85:
            suspicious.append({'file': f, 'class': cls, 'pred': pred, 'conf': conf, 'path': fp})

suspicious.sort(key=lambda x: -x['conf'])
print(f"  Found {len(suspicious)} highly suspicious (conf>{0.85})")

# Identify the 4 confirmed mislabels
CONFIRMED_RELABEL = {
    'IMG_4306.JPG': ('Early_Leaf_Spot', 'Healthy'),
    'IMG_4310.JPG': ('Early_Leaf_Spot', 'Healthy'),
    'IMG_4311.JPG': ('Early_Leaf_Spot', 'Healthy'),
    'IMG_4312.JPG': ('Early_Leaf_Spot', 'Healthy'),
}

# --- STEP 2: Build cleaned copy ---
print("\nSTEP 2: Building cleaned dataset...")
if os.path.exists(CLEANED_DIR):
    shutil.rmtree(CLEANED_DIR)

# Images to exclude (top 30 most suspicious)
exclude_set = set()
for m in suspicious[:30]:
    exclude_set.add((m['class'], m['file']))
    print(f"  EXCLUDE: {m['class']:25s} {m['file']:30s} pred={m['pred']:35s} conf={m['conf']:.2f}")

# Add relabeled images to exclude (they'll be re-added under new class)
for fn, (from_cls, to_cls) in CONFIRMED_RELABEL.items():
    exclude_set.add((from_cls, fn))
    print(f"  RELABEL: {from_cls:25s} {fn:30s} -> {to_cls}")

# Copy all non-excluded images, plus relabeled ones
copied = {'Healthy': 0, 'Leaf_Spot': 0, 'Nutrition_Deficiency': 0, 'Rust': 0}
relabel_map = {fn: to_cls for fn, (from_cls, to_cls) in CONFIRMED_RELABEL.items()}
relabel_source = {fn: from_cls for fn, (from_cls, to_cls) in CONFIRMED_RELABEL.items()}

for cls, folder in RAW_CLASSES.items():
    target_cls = 'Leaf_Spot' if cls in ('Early_Leaf_Spot', 'Late_Leaf_Spot') else cls
    target_cls = 'Rust' if cls == 'Rust' else target_cls
    
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
        fp = os.path.join(folder, f)
        
        # Check if this file should be relabeled or excluded
        if (cls, f) in exclude_set and f not in relabel_map:
            continue  # Excluded
        if f in relabel_map and relabel_source[f] == cls:
            # Relabel: copy to new class
            new_target = relabel_map[f]
            dest_dir = os.path.join(CLEANED_DIR, new_target)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(fp, os.path.join(dest_dir, f))
            copied[new_target] += 1
            print(f"  RELABEL COPY: {cls}->{new_target}: {f}")
        else:
            # Normal copy
            dest_dir = os.path.join(CLEANED_DIR, target_cls)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(fp, os.path.join(dest_dir, f))
            copied[target_cls] += 1

print(f"\nCleaned dataset summary:")
for cls, n in sorted(copied.items()):
    print(f"  {cls:25s} {n} images")
print(f"  Total: {sum(copied.values())} images")
print(f"\nSaved to: {CLEANED_DIR}")
print(f"\nRun with: python ml_model/expand_groundnut_model.py --train-clean --retrain")
