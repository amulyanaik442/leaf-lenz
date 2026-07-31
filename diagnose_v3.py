"""
Comprehensive diagnosis of Pumpkin prediction bias.
Uses EXACT inference pipeline (20-view TTA with 256->bilinear->5crops->224).
"""
import json, os, copy
import numpy as np
from PIL import Image
import onnxruntime as ort
import onnx

BASE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(BASE, 'detector', 'ml_assets')
MODEL = os.path.join(ML, 'model.onnx')
CLASSES = os.path.join(ML, 'class_names.json')

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FOLDER_TO_CLASS = {
    'Bacterial Leaf Spot': 'Pumpkin___bacterial_leaf_spot',
    'Downy Mildew': 'Pumpkin___downy_mildew',
    'Healthy Leaf': 'Pumpkin___healthy',
    'Mosaic Disease': 'Pumpkin___mosaic_disease',
    'Powdery Mildew': 'Pumpkin___powdery_mildew',
}
PUMPKIN_CLS = sorted(FOLDER_TO_CLASS.values())

# ── 1. Verify model ──
print("="*70)
print("1. MODEL VERIFICATION")
print("="*70)
print(f"  Model: {MODEL}")
print(f"  Exists: {os.path.exists(MODEL)}, Size: {os.path.getsize(MODEL)/1e6:.1f} MB")

sess = ort.InferenceSession(MODEL)
with open(CLASSES) as f:
    names = json.load(f)
print(f"  Output shape: {sess.get_outputs()[0].shape}")
print(f"  Total classes: {len(names)}")

pumpkin_idx = [i for i, n in enumerate(names) if n.startswith('Pumpkin___')]
print(f"  Pumpkin indices: {pumpkin_idx}")
assert pumpkin_idx == [103,104,105,106,107], "INDEX MISMATCH!"
print(f"  Pumpkin names:")
for i in pumpkin_idx:
    print(f"    [{i}] {names[i]}")

# ── 2. Preprocessing comparison ──
print("\n"+"="*70)
print("2. PREPROCESSING COMPARISON")
print("="*70)
print("""
  TRAINING (validation):
    1. Image.open(path).convert('RGB')
    2. img.resize((224, 224))       # PIL bicubic (default)
    3. ToTensor() -> float32[0,1]   # HWC->CHW
    4. Normalize(mean, std)         # imagenet

  INFERENCE (predict_leaf_disease):
    1. Image.open(file).convert('RGB')
    2. img.resize((256, 256), BILINEAR)  # BILINEAR not bicubic!
    3. np.array / 255.0 -> float32[0,1]  # HWC
    4. 4 augmentations (flip H/V/HV)
    5. 5 crops each -> 20 (224,224,3) crops
    6. _normalize: (arr - mean) / std
    7. _to_tensor: HWC->CHW + expand batch

  CRITICAL DIFFERENCES:
    - Training: bicubic resize 224x224 (stretches to fill)
    - Inference: bilinear resize 256x256, then crop 224x224 (discards 16px border)
    - Training: single view
    - Inference: 20-view softmax averaging (magnifies bias)

  TRAINING (train_aug):
    1. Resize((256,256)) bicubic
    2. RandomCrop(224)
    3. RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(10)
    4. ToTensor + Normalize

  Note: train_aug resizes to 256 then crops to 224 with RANDOM location.
  Inference uses bilinear resize + FIXED centre/corner crops.
  Interpolation mismatch: bicubic vs bilinear.
""")

# ── 3. Compare single-crop vs TTA ──
print("="*70)
print("3. COMPARING INFERENCE PIPELINES ON ORIGINAL SET")
print("="*70)

def pipeline_single_224(path):
    """Exact training validation pipeline: bicubic 224, no crop."""
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))  # bicubic default
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr.transpose(2, 0, 1) - MEAN[:, None, None]) / STD[:, None, None]
    arr = arr.astype(np.float32)[None, :, :, :]
    logits = sess.run(None, {sess.get_inputs()[0].name: arr})[0][0]
    return logits

def pipeline_tta(path):
    """Exact inference pipeline: bilinear 256, 5 crops, 4 augs, softmax avg."""
    img = Image.open(path).convert('RGB')
    img = img.resize((256, 256), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0

    def five_crops(img_np, size=224):
        h, w = img_np.shape[:2]; d = size
        return [img_np[(h-d)//2:(h+d)//2, (w-d)//2:(w+d)//2],
                img_np[0:d,0:d], img_np[0:d,w-d:w],
                img_np[h-d:h,0:d], img_np[h-d:h,w-d:w]]

    def augmentations(a):
        yield a; yield a[:,::-1,:].copy()
        yield a[::-1,:,:].copy(); yield a[::-1,::-1,:].copy()

    def to_tensor(a):
        return np.expand_dims(a.transpose(2,0,1), axis=0)

    prob_sum = None
    for aug in augmentations(arr):
        for crop in five_crops(aug, size=224):
            norm = (crop.astype(np.float32) - MEAN) / STD
            t = to_tensor(norm).astype(np.float32)
            logits = sess.run(None, {sess.get_inputs()[0].name: t})[0][0]
            exp = np.exp(logits - np.max(logits))
            probs = exp / np.sum(exp)
            prob_sum = probs if prob_sum is None else prob_sum + probs
    avg_probs = prob_sum / 20
    return avg_probs

def pipeline_tta_logit_avg(path):
    """Same TTA but averages logits, not probabilities."""
    img = Image.open(path).convert('RGB')
    img = img.resize((256, 256), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0

    def five_crops(img_np, size=224):
        h, w = img_np.shape[:2]; d = size
        return [img_np[(h-d)//2:(h+d)//2, (w-d)//2:(w+d)//2],
                img_np[0:d,0:d], img_np[0:d,w-d:w],
                img_np[h-d:h,0:d], img_np[h-d:h,w-d:w]]

    def augmentations(a):
        yield a; yield a[:,::-1,:].copy()
        yield a[::-1,:,:].copy(); yield a[::-1,::-1,:].copy()

    def to_tensor(a):
        return np.expand_dims(a.transpose(2,0,1), axis=0)

    logit_sum = None
    for aug in augmentations(arr):
        for crop in five_crops(aug, size=224):
            norm = (crop.astype(np.float32) - MEAN) / STD
            t = to_tensor(norm).astype(np.float32)
            logits = sess.run(None, {sess.get_inputs()[0].name: t})[0][0]
            logit_sum = logits if logit_sum is None else logit_sum + logits
    avg_logits = logit_sum / 20
    exp = np.exp(avg_logits - np.max(avg_logits))
    return exp / np.sum(exp)

def run_cm(pipeline_fn, label):
    cm = np.zeros((5,5), dtype=int)
    correct = total = 0
    per_class = {c: {'corr':0, 'total':0, 'fp':0} for c in PUMPKIN_CLS}
    per_class_probs = {c: [] for c in PUMPKIN_CLS}

    for folder, cls in FOLDER_TO_CLASS.items():
        d = os.path.join(BASE, 'dataset', 'pumpkin', 'Original', 'Original', folder)
        true_i = PUMPKIN_CLS.index(cls)
        for fn in os.listdir(d):
            if not fn.lower().endswith(('.png','.jpg','.jpeg')): continue
            path = os.path.join(d, fn)
            out = pipeline_fn(path)
            pumpkin_out = out[pumpkin_idx]
            pred_i = int(np.argmax(pumpkin_out))
            cm[true_i][pred_i] += 1
            total += 1
            per_class[cls]['total'] += 1
            per_class_probs[cls].append(pumpkin_out)
            if pred_i == true_i:
                correct += 1
                per_class[cls]['corr'] += 1

    for j in range(5):
        fp = cm[:,j].sum() - cm[j,j]
        per_class[PUMPKIN_CLS[j]]['fp'] = fp

    return cm, correct, total, per_class, per_class_probs

def print_results(cm, correct, total, per_class, name):
    acc = 100*correct/total
    print(f"\n  --- {name} ---")
    print(f"  Accuracy: {correct}/{total} = {acc:.2f}%")
    short = [n.split('___')[1][:14] for n in PUMPKIN_CLS]
    hdr = ' ' * 16 + ''.join(f'{s:>14s}' for s in short)
    print(f"  {hdr}")
    for i in range(5):
        row = '  ' + short[i].ljust(14) + ''.join(f'{cm[i][j]:6d}' for j in range(5))
        print(f"  {row}")
    print()
    for cls in PUMPKIN_CLS:
        d = per_class[cls]
        rec = 100*d['corr']/d['total']
        prec = 100*d['corr']/(d['corr']+d['fp']) if (d['corr']+d['fp'])>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        print(f"  {cls:40s} rec={rec:.1f}%  prec={prec:.1f}%  f1={f1:.1f}%  FP={d['fp']}")

orig = os.path.join(BASE, 'dataset', 'pumpkin', 'Original', 'Original')
cm_s, corr_s, total_s, pc_s, pprobs_s = run_cm(pipeline_single_224, 'single')
print_results(cm_s, corr_s, total_s, pc_s, 'TRAINING-VAL PIPELINE (single 224 bicubic)')

cm_t, corr_t, total_t, pc_t, pprobs_t = run_cm(pipeline_tta, 'tta')
print_results(cm_t, corr_t, total_t, pc_t, 'INFERENCE TTA (20-view softmax avg)')

cm_l, corr_l, total_l, pc_l, pprobs_l = run_cm(pipeline_tta_logit_avg, 'logit_avg')
print_results(cm_l, corr_l, total_l, pc_l, 'INFERENCE TTA (20-view LOGIT avg)')

# ── 4. Logit/Probability inspection ──
print("\n"+"="*70)
print("4. LOGIT/PROBABILITY INSPECTION")
print("="*70)

# Show some misclassified examples with full logits
print("\n  Misclassification examples (TTA softmax avg):")
count = 0
for folder, cls in FOLDER_TO_CLASS.items():
    d = os.path.join(BASE, 'dataset', 'pumpkin', 'Original', 'Original', folder)
    true_i = PUMPKIN_CLS.index(cls)
    for fn in sorted(os.listdir(d)):
        if count >= 15: break
        if not fn.lower().endswith(('.png','.jpg','.jpeg')): continue
        path = os.path.join(d, fn)
        probs = pipeline_tta(path)  # TTA pipeline
        pumpkin_probs = probs[pumpkin_idx]
        pred_i = int(np.argmax(pumpkin_probs))
        if pred_i != true_i:
            probs_str = ' | '.join(f'{PUMPKIN_CLS[j].split("___")[1]}:{pumpkin_probs[j]:.4f}' for j in range(5))
            print(f"  TRUE={cls.split('___')[1]:20s} PRED={PUMPKIN_CLS[pred_i].split('___')[1]:20s} [{probs_str}]")
            count += 1
    if count >= 15: break

# ── 5. Classifier bias analysis ──
print("\n"+"="*70)
print("5. CLASSIFIER BIAS ANALYSIS")
print("="*70)

model = onnx.load(MODEL)
weight_init = bias_init = None
for init in model.graph.initializer:
    dims = list(init.dims)
    if dims == [108, 1280]: weight_init = init
    if dims == [108]: bias_init = init

if weight_init and bias_init:
    w = np.frombuffer(weight_init.raw_data, dtype=np.float32).reshape([108, 1280])
    b = np.frombuffer(bias_init.raw_data, dtype=np.float32)
    print(f"  Weight shape: {w.shape}, Bias shape: {b.shape}")
    print(f"\n  Pumpkin bias values:")
    for i in pumpkin_idx:
        print(f"    {names[i]:40s} bias={b[i]:+.6f}")
    print(f"\n  Pumpkin bias range: {b[pumpkin_idx].min():+.4f} to {b[pumpkin_idx].max():+.4f}")
    print(f"  Pumpkin bias mean: {b[pumpkin_idx].mean():+.6f}")
    print(f"  Pumpkin weight norms:")
    for i in pumpkin_idx:
        print(f"    {names[i]:40s} norm={np.linalg.norm(w[i]):.4f}")

    # Show what happens with zero features
    print(f"\n  With zero features, bias contribution to softmax:")
    zero_logits = b[pumpkin_idx]
    exp_z = np.exp(zero_logits - np.max(zero_logits))
    zero_probs = exp_z / np.sum(exp_z)
    for i, idx in enumerate(pumpkin_idx):
        print(f"    {names[idx]:40s} bias={b[idx]:+.4f}  ->  softmax={zero_probs[i]:.4f}")

# ── 6. TTA pipeline: compare centre crop vs all crops ──
print("\n"+"="*70)
print("6. TTA DECOMPOSITION: centre crop vs all crops")
print("="*70)

def pipeline_tta_centre_only(path):
    """TTA with only centre crop (5 views: 4 augs x 1 centre crop)."""
    img = Image.open(path).convert('RGB')
    img = img.resize((256, 256), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0

    def centre_crop(img_np, size=224):
        h, w = img_np.shape[:2]; d = size
        return img_np[(h-d)//2:(h+d)//2, (w-d)//2:(w+d)//2]

    def augmentations(a):
        yield a; yield a[:,::-1,:].copy()
        yield a[::-1,:,:].copy(); yield a[::-1,::-1,:].copy()

    def to_tensor(a):
        return np.expand_dims(a.transpose(2,0,1), axis=0)

    prob_sum = None; n = 0
    for aug in augmentations(arr):
        crop = centre_crop(aug, size=224)
        norm = (crop.astype(np.float32) - MEAN) / STD
        t = to_tensor(norm).astype(np.float32)
        logits = sess.run(None, {sess.get_inputs()[0].name: t})[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        prob_sum = probs if prob_sum is None else prob_sum + probs
        n += 1
    return prob_sum / n

cm_co, corr_co, total_co, pc_co, _ = run_cm(pipeline_tta_centre_only, 'centre')
print_results(cm_co, corr_co, total_co, pc_co, 'TTA CENTRE-ONLY (4 augs x 1 crop = 4 views)')

# ── 7. Crop filtering effect ──
print("\n"+"="*70)
print("7. CROP FILTERING EFFECT (simulates _predict_general_filtered)")
print("="*70)

total_agree = 0
total_filtered_agree = 0
for folder, cls in FOLDER_TO_CLASS.items():
    d = os.path.join(BASE, 'dataset', 'pumpkin', 'Original', 'Original', folder)
    true_i = PUMPKIN_CLS.index(cls)
    for fn in os.listdir(d):
        if not fn.lower().endswith(('.png','.jpg','.jpeg')): continue
        path = os.path.join(d, fn)
        probs = pipeline_tta(path)
        # Unfiltered: all 108 classes
        full_pred = int(np.argmax(probs))
        # Filtered: only Pumpkin classes, renormalize
        filtered = probs.copy()
        filtered[:pumpkin_idx[0]] = 0
        filtered[pumpkin_idx[-1]+1:] = 0
        filtered = filtered / filtered.sum()
        pump_pred = int(np.argmax(filtered[pumpkin_idx]))
        # Does allowing non-Pumpkin classes change the result?
        label_full = names[full_pred]
        label_filtered = PUMPKIN_CLS[pump_pred]
        if not label_full.startswith('Pumpkin___'):
            total_agree += 1  # full model says non-Pumpkin
            if label_filtered != cls:
                total_filtered_agree += 1

print(f"  Images where full model predicts non-Pumpkin: {total_agree}/2000")
print(f"  ...and filtered model disagrees with true label: {total_filtered_agree}")

# ── 8. Cached session files ──
print("\n"+"="*70)
print("8. ADDITIONAL CHECKS")
print("="*70)

# Check class_names.json index alignment
print(f"\n  Class_names.json has {len(names)} classes.")
print(f"  Expected Pumpkin indices: 103-107")
print(f"  Actual Pumpkin indices: {pumpkin_idx}")
for i, idx in enumerate(pumpkin_idx):
    expected = PUMPKIN_CLS[i]
    actual = names[idx]
    match = "OK" if expected == actual else "MISMATCH!"
    print(f"    [{idx}] expected={expected:45s} actual={actual:45s} {match}")

# Check model.onnx file hash to ensure it's the latest
print(f"\n  Model file: {MODEL}")
print(f"  Last modified: {os.path.getmtime(MODEL)}")
print(f"  Size: {os.path.getsize(MODEL)/1e6:.1f} MB")
# Check if there are multiple .onnx files
for f in os.listdir(ML):
    if f.endswith('.onnx'):
        fp = os.path.join(ML, f)
        print(f"  Found: {f:40s} {os.path.getsize(fp)/1e6:.1f} MB")

# Clean up temp feature extractor files
for f in os.listdir(ML):
    if f.endswith('_features.onnx'):
        os.remove(os.path.join(ML, f))

print("\n"+"="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
