import os, json, shutil
import numpy as np
from PIL import Image
import onnxruntime as ort
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ML_ASSETS = os.path.join(BASE_DIR, 'detector', 'ml_assets')
MODEL_PATH = os.path.join(ML_ASSETS, 'model.onnx')
CLASSES_PATH = os.path.join(ML_ASSETS, 'class_names.json')

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _normalize(arr):
    return (arr - MEAN) / STD

def _to_tensor(arr):
    return np.expand_dims(arr.transpose(2, 0, 1), axis=0)

def _five_crops(img_np, size=224):
    h, w = img_np.shape[:2]
    d = size
    return [
        img_np[(h-d)//2:(h+d)//2, (w-d)//2:(w+d)//2],
        img_np[0:d,           0:d],
        img_np[0:d,           w-d:w],
        img_np[h-d:h,         0:d],
        img_np[h-d:h,         w-d:w],
    ]

def _augmentations(arr):
    yield arr
    yield arr[:, ::-1, :].copy()
    yield arr[::-1, :, :].copy()
    yield arr[::-1, ::-1, :].copy()

def predict(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((256, 256), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensors = []
    for aug in _augmentations(arr):
        for crop in _five_crops(aug, size=224):
            norm = _normalize(crop)
            tensors.append(_to_tensor(norm).astype(np.float32))
    prob_sum = None
    for t in tensors:
        logits = session.run(None, {input_name: t})[0][0]
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        prob_sum = probs if prob_sum is None else prob_sum + probs
    avg_probs = prob_sum / len(tensors)
    top5_idx = np.argsort(avg_probs)[::-1][:5]
    top5 = [(CLASS_NAMES[i], float(avg_probs[i])) for i in top5_idx]
    return top5[0][0], top5[0][1], top5

GROUNDNUT_CLASSES = ['Groundnut___Healthy', 'Groundnut___Leaf_Spot',
                     'Groundnut___Nutrition_Deficiency', 'Groundnut___Rust']

BASE_DATASET = os.path.expandvars(R'%USERPROFILE%\.cache\kagglehub\datasets\warcoder\groundnut-plant-leaf-data\versions\1\Dataset of groundnut plant leaf images for classification and detection')
TEST_DIR = os.path.join(BASE_DATASET, 'Groundnut_Leaf_dataset', 'test')

OUT = os.path.join(BASE_DIR, 'ml_model', 'evaluation', 'healthy_inspection')
os.makedirs(OUT, exist_ok=True)

# Only test the healthy_leaf_1 folder
folder_name = 'healthy_leaf_1'
true_label = 'Groundnut___Healthy'
folder_path = os.path.join(TEST_DIR, folder_name)

images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])

correct = []
misclassified = []

for img_name in images:
    img_path = os.path.join(folder_path, img_name)
    pred_label, conf, top5 = predict(img_path)
    entry = {
        'img_name': img_name,
        'img_path': img_path,
        'pred_label': pred_label,
        'conf': conf,
        'top5': top5,
    }
    if pred_label == true_label:
        correct.append(entry)
    else:
        misclassified.append(entry)

print(f"\nHealthy class: {len(correct)} correct, {len(misclassified)} misclassified out of {len(images)} total\n")

# Sort misclassified by confidence (highest first — most confidently wrong)
misclassified.sort(key=lambda x: x['conf'], reverse=True)

print("=== TOP 30 MISCLASSIFIED (highest confidence, most confidently wrong) ===\n")
for i, m in enumerate(misclassified[:30], 1):
    print(f"{i:2d}. {m['img_name']}")
    print(f"    Predicted: {m['pred_label']:<45} Confidence: {m['conf']:.4f}")
    print(f"    Top-5:")
    for j, (label, conf) in enumerate(m['top5'], 1):
        arrow = " <--" if label == m['pred_label'] else ""
        print(f"       {j}. {label:<45} {conf:.4f}{arrow}")
    print()

# Also show the correct ones with LOWEST confidence (near-misses)
correct.sort(key=lambda x: x['conf'])
print("=== BOTTOM 10 CORRECT (lowest confidence, barely got it right) ===\n")
for i, c in enumerate(correct[:10], 1):
    print(f"{i:2d}. {c['img_name']}")
    print(f"    Predicted: {c['pred_label']:<45} Confidence: {c['conf']:.4f}")
    print(f"    Top-5:")
    for j, (label, conf) in enumerate(c['top5'], 1):
        arrow = " <--" if label == c['pred_label'] else ""
        print(f"       {j}. {label:<45} {conf:.4f}{arrow}")
    print()

# Copy images to folder for visual inspection
mis_dir = os.path.join(OUT, 'misclassified')
corr_dir = os.path.join(OUT, 'correct_lowconf')
os.makedirs(mis_dir, exist_ok=True)
os.makedirs(corr_dir, exist_ok=True)

for m in misclassified:
    label_short = m['pred_label'].split('___')[-1] if '___' in m['pred_label'] else m['pred_label']
    fname = f"conf{m['conf']:.4f}_{label_short}_{m['img_name']}"
    shutil.copy2(m['img_path'], os.path.join(mis_dir, fname))

for c in correct[:10]:
    fname = f"conf{c['conf']:.4f}_{c['img_name']}"
    shutil.copy2(c['img_path'], os.path.join(corr_dir, fname))

# Create HTML report
html_parts = []
html_parts.append("""
<html><head><title>Healthy Class Inspection</title>
<style>
  body { font-family: monospace; margin: 20px; background: #111; color: #eee; }
  h1, h2 { color: #fff; }
  .gallery { display: flex; flex-wrap: wrap; gap: 16px; }
  .card { background: #222; border: 1px solid #444; border-radius: 8px; padding: 12px; width: 280px; }
  .card img { width: 256px; height: 256px; object-fit: cover; border-radius: 4px; }
  .card .label { font-size: 12px; margin-top: 6px; }
  .true { color: #4caf50; }
  .pred-mis { color: #f44336; }
  .pred-ok { color: #4caf50; }
  .highlight { background: #331; border-color: #664; }
</style></head><body>
<h1>Healthy Class Inspection</h1>
""")

# Summary
html_parts.append(f"<p>Total: {len(images)} | Correct: {len(correct)} | Misclassified: {len(misclassified)}</p>")

# Misclassified images
html_parts.append("<h2>Misclassified Healthy Images</h2><div class='gallery'>")
for m in misclassified[:50]:
    label_short = m['pred_label'].split('___')[-1] if '___' in m['pred_label'] else m['pred_label']
    fname = f"conf{m['conf']:.4f}_{label_short}_{m['img_name']}"
    src = f"misclassified/{fname}"
    top5_html = "<br>".join([f"{j}. {label} ({conf:.3f})" for j, (label, conf) in enumerate(m['top5'], 1)])
    html_parts.append(f"<div class='card highlight'><img src='{src}' loading='lazy'>")
    html_parts.append(f"<div class='label'>True: <span class='true'>Healthy</span></div>")
    html_parts.append(f"<div class='label'>Pred: <span class='pred-mis'>{m['pred_label']}</span> conf={m['conf']:.4f}</div>")
    html_parts.append(f"<div class='label'><small>{top5_html}</small></div></div>")

html_parts.append("</div>")

# Low-confidence correct images
html_parts.append("<h2>Low-Confidence Correct Predictions (near-misses)</h2><div class='gallery'>")
for c in correct[:20]:
    fname = f"conf{c['conf']:.4f}_{c['img_name']}"
    src = f"correct_lowconf/{fname}"
    top5_html = "<br>".join([f"{j}. {label} ({conf:.3f})" for j, (label, conf) in enumerate(c['top5'], 1)])
    html_parts.append(f"<div class='card'><img src='{src}' loading='lazy'>")
    html_parts.append(f"<div class='label'>True: <span class='true'>Healthy</span></div>")
    html_parts.append(f"<div class='label'>Pred: <span class='pred-ok'>Healthy</span> conf={c['conf']:.4f}</div>")
    html_parts.append(f"<div class='label'><small>{top5_html}</small></div></div>")

html_parts.append("</div></body></html>")

html_path = os.path.join(OUT, 'report.html')
with open(html_path, 'w') as f:
    f.write("\n".join(html_parts))

print(f"\nHTML report saved to {html_path}")
print(f"Misclassified images copied to {mis_dir}")
print(f"Low-confidence correct images copied to {corr_dir}")
