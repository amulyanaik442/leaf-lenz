import os, json, hashlib
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

TEST_FOLDERS = {
    'Healthy': os.path.join(DATA_BASE, 'Groundnut_Leaf_dataset', 'test', 'healthy_leaf_1'),
    'Early_Leaf_Spot': os.path.join(DATA_BASE, 'Groundnut_Leaf_dataset', 'test', 'early_leaf_spot_1'),
    'Late_Leaf_Spot': os.path.join(DATA_BASE, 'Groundnut_Leaf_dataset', 'test', 'late_leaf_spot_1'),
    'Nutrition_Deficiency': os.path.join(DATA_BASE, 'Groundnut_Leaf_dataset', 'test', 'nutrition_deficiency_1'),
    'Early_Rust': os.path.join(DATA_BASE, 'Groundnut_Leaf_dataset', 'test', 'early_rust_1'),
    'Rust': os.path.join(DATA_BASE, 'Groundnut_Leaf_dataset', 'test', 'rust_1'),
}

TEST_LABEL_MAP = {
    'Healthy': 'Groundnut___Healthy',
    'Early_Leaf_Spot': 'Groundnut___Leaf_Spot',
    'Late_Leaf_Spot': 'Groundnut___Leaf_Spot',
    'Nutrition_Deficiency': 'Groundnut___Nutrition_Deficiency',
    'Early_Rust': 'Groundnut___Rust',
    'Rust': 'Groundnut___Rust',
}

# --- PART 1: Model inference on Raw_Data ---
print("=" * 70)
print("PART 1: MODEL INFERENCE ON RAW_DATA (potential mislabels)")
print("=" * 70)

results = defaultdict(lambda: {'correct': 0, 'total': 0, 'mislabels': []})

for cls, folder in RAW_CLASSES.items():
    print(f"  Processing {cls}...")
    for f in os.listdir(folder):
        if not f.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        fp = os.path.join(folder, f)
        true_gn = LABEL_TO_GN[cls]
        try:
            pred, conf = predict(fp)
        except Exception as e:
            print(f"    ERROR: {fp}: {e}")
            continue

        results[cls]['total'] += 1
        if pred == true_gn:
            results[cls]['correct'] += 1
        elif pred.startswith('Groundnut___'):
            results[cls]['mislabels'].append({
                'file': f, 'pred': pred, 'conf': conf, 'type': 'wrong_gn'
            })
        else:
            results[cls]['mislabels'].append({
                'file': f, 'pred': pred, 'conf': conf, 'type': 'non_gn'
            })

print()
print(f"  {'Class':25s} {'Agree':>15s} {'Wrong GN':>11s} {'Non-GN':>9s} {'High Conf':>10s}")
for cls in RAW_CLASSES:
    d = results[cls]
    correct = d['correct']
    total = d['total']
    wrong_gn = len([m for m in d['mislabels'] if m['type'] == 'wrong_gn'])
    non_gn = len([m for m in d['mislabels'] if m['type'] == 'non_gn'])
    high_conf = len([m for m in d['mislabels'] if m['conf'] > 0.6])
    pct = correct/total*100 if total else 0
    print(f"  {cls:25s} {correct:4d}/{total:4d} ({pct:5.1f}%)  {wrong_gn:6d}     {non_gn:5d}     {high_conf:5d}")

# Show top mislabels
all_mislabels = []
for cls, d in results.items():
    for m in d['mislabels']:
        m['true_cls'] = cls
        all_mislabels.append(m)
all_mislabels.sort(key=lambda x: -x['conf'])

print()
print("=" * 70)
print("TOP 40 MISLABEL CANDIDATES (highest confidence disagreement)")
print("=" * 70)
print(f"  {'Conf':>6s} {'True Class':20s} {'Predicted':35s} {'File':30s}")
for m in all_mislabels[:40]:
    pred_short = m['pred']
    if len(pred_short) > 34:
        pred_short = '...' + pred_short[-31:]
    print(f"  {m['conf']:>5.2f} {m['true_cls']:20s} {pred_short:35s} {m['file']:30s}")

# --- PART 2: Run inference on Test set for comparison ---
print()
print("=" * 70)
print("PART 2: MODEL INFERENCE ON TEST SET (for comparison)")
print("=" * 70)

test_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'mislabels': []})

for cls, folder in TEST_FOLDERS.items():
    print(f"  Processing {cls}...")
    for f in os.listdir(folder):
        if not f.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        fp = os.path.join(folder, f)
        true_gn = TEST_LABEL_MAP[cls]
        try:
            pred, conf = predict(fp)
        except:
            continue
        test_results[cls]['total'] += 1
        if pred == true_gn:
            test_results[cls]['correct'] += 1
        else:
            test_results[cls]['mislabels'].append({
                'file': f, 'pred': pred, 'conf': conf
            })

print()
print(f"  {'Class':25s} {'Agree':>15s}")
for cls in TEST_FOLDERS:
    d = test_results[cls]
    correct = d['correct']
    total = d['total']
    pct = correct/total*100 if total else 0
    print(f"  {cls:25s} {correct:4d}/{total:4d} ({pct:5.1f}%)")

# --- PART 3: Compare train vs test disagreement patterns ---
print()
print("=" * 70)
print("PART 3: TRAIN vs TEST DISAGREEMENT COMPARISON")
print("=" * 70)

raw_disagree = {cls: results[cls]['total'] - results[cls]['correct'] for cls in RAW_CLASSES}
test_disagree = {}
for cls in TEST_FOLDERS:
    key = cls.replace('Early_', '') if cls.startswith('Early_') else cls
    # Map test classes to comparable raw classes
    test_key = cls.replace('Early_Rust', 'Rust').replace('Late_Leaf_Spot', 'Leaf_Spot').replace('Early_Leaf_Spot', 'Leaf_Spot').replace('_1', '')
    test_disagree[cls] = test_results[cls]['total'] - test_results[cls]['correct']

# Merge leaf spot
combined = {
    'Healthy': ('Healthy', 'Healthy'),
    'Leaf_Spot': ('Early_Leaf_Spot', 'Late_Leaf_Spot'),
    'Nutrition_Deficiency': ('Nutrition_Deficiency', 'Nutrition_Deficiency'),
    'Rust': ('Rust', 'Rust'),
}

print(f"  {'Class':25s} {'Train Error':>15s} {'Test Error':>15s} {'Note':>20s}")
for combined_cls, (raw_cls1, raw_cls2) in combined.items():
    if raw_cls2:
        train_err = (results[raw_cls1]['total'] - results[raw_cls1]['correct']) + \
                    (results[raw_cls2]['total'] - results[raw_cls2]['correct'])
        train_total = results[raw_cls1]['total'] + results[raw_cls2]['total']
    else:
        train_err = results[raw_cls1]['total'] - results[raw_cls1]['correct']
        train_total = results[raw_cls1]['total']
    
    # Test: find matching classes
    test_cls = combined_cls if combined_cls != 'Leaf_Spot' else None
    if combined_cls == 'Leaf_Spot':
        test_err = (test_results['Early_Leaf_Spot']['total'] - test_results['Early_Leaf_Spot']['correct']) + \
                   (test_results['Late_Leaf_Spot']['total'] - test_results['Late_Leaf_Spot']['correct'])
        test_total = test_results['Early_Leaf_Spot']['total'] + test_results['Late_Leaf_Spot']['total']
    elif combined_cls == 'Rust':
        test_err = (test_results['Early_Rust']['total'] - test_results['Early_Rust']['correct']) + \
                   (test_results['Rust']['total'] - test_results['Rust']['correct'])
        test_total = test_results['Early_Rust']['total'] + test_results['Rust']['total']
    else:
        test_err = test_results[combined_cls]['total'] - test_results[combined_cls]['correct']
        test_total = test_results[combined_cls]['total']
    
    train_rate = train_err/train_total*100 if train_total else 0
    test_rate = test_err/test_total*100 if test_total else 0
    gap = test_rate - train_rate
    note = ''
    if gap > 10:
        note = 'TEST MUCH WORSE'
    elif gap < -10:
        note = 'TRAIN MUCH WORSE'
    print(f"  {combined_cls:25s} {train_err:4d}/{train_total:4d} ({train_rate:5.1f}%) {test_err:4d}/{test_total:4d} ({test_rate:5.1f}%) {note:>20s}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""  Key findings so far:
  1. No data leakage between Raw_Data and Test set (0 hash matches)
  2. 47 filename collisions in Raw_Data (same name, different content, different classes)
  3. 17 filename overlaps between Raw_Data and Test (different content - no leakage)
  4. Label mismatches: Nutrition_Deficiency images numbered 22-42.jpg appear in
     Late_Leaf_Spot and Healthy test folders (same filename, different class labels)
  5. Suspicious '.jpg' file in train/early_rust_1 is actually a valid image
""")
