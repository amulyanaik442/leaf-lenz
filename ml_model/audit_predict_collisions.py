import os, json
import numpy as np
from PIL import Image
import onnxruntime as ort

ML_ASSETS = r'C:\Users\amuly\Desktop\leaf-lenz-main\detector\ml_assets'
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
    idxs = np.argsort(probs)[::-1][:3]
    return [(CLASS_NAMES[i], float(probs[i])) for i in idxs]

DATA_BASE = r'C:\Users\amuly\.cache\kagglehub\datasets\warcoder\groundnut-plant-leaf-data\versions\1\Dataset of groundnut plant leaf images for classification and detection'

pairs = []
for i in range(4305, 4315):
    pairs.append((f'IMG_{i}.JPG', 'healthy leaf', 'early_leaf_spot'))
for i in range(3483, 3499):
    pairs.append((f'IMG_{i}.JPG', 'early_leaf_spot', 'nutrition deficiency'))
for i in range(5707, 5728):
    pairs.append((f'IMG_{i}.JPG', 'late leaf spot', 'nutrition deficiency'))

header = f"{'Filename':25s} {'Folder A':20s} {'Pred A':30s} {'Conf A':>7s} {'Folder B':20s} {'Pred B':30s} {'Conf B':>7s} {'Agree':>7s}"
print(header)
print('-' * len(header))

mismatch_count = 0
for fn, folder_a, folder_b in pairs:
    fp_a = os.path.join(DATA_BASE, 'Raw_Data', folder_a, fn)
    fp_b = os.path.join(DATA_BASE, 'Raw_Data', folder_b, fn)
    if not os.path.exists(fp_a) or not os.path.exists(fp_b):
        continue
    top3_a = predict(fp_a)
    top3_b = predict(fp_b)
    pred_a, conf_a = top3_a[0]
    pred_b, conf_b = top3_b[0]
    agrees = 'SAME' if pred_a == pred_b else 'DIFF'
    if agrees == 'DIFF':
        mismatch_count += 1
    print(f'{fn:25s} {folder_a:20s} {pred_a:30s} {conf_a:6.2f} {folder_b:20s} {pred_b:30s} {conf_b:6.2f} {agrees:>7s}')

print(f'\nModel disagrees on {mismatch_count}/{len(pairs)} collision pairs')
print()

# Now: for each collision pair, determine which folder's label the model agrees with
# and flag the one it disagrees with as potentially mislabeled
LABEL_TO_GN = {
    'healthy leaf': 'Groundnut___Healthy',
    'early_leaf_spot': 'Groundnut___Leaf_Spot',
    'late leaf spot': 'Groundnut___Leaf_Spot',
    'nutrition deficiency': 'Groundnut___Nutrition_Deficiency',
    'rust': 'Groundnut___Rust',
}

print("Collisions where model disagrees with one folder's label:")
print(f"{'Filename':25s} {'Wrong Folder':20s} {'Model Says':30s} {'Conf':>6s}")
for fn, folder_a, folder_b in pairs:
    fp_a = os.path.join(DATA_BASE, 'Raw_Data', folder_a, fn)
    fp_b = os.path.join(DATA_BASE, 'Raw_Data', folder_b, fn)
    if not os.path.exists(fp_a) or not os.path.exists(fp_b):
        continue
    top3_a = predict(fp_a)
    top3_b = predict(fp_b)
    pred_a, conf_a = top3_a[0]
    pred_b, conf_b = top3_b[0]
    label_a = LABEL_TO_GN[folder_a]
    label_b = LABEL_TO_GN[folder_b]
    a_correct = pred_a == label_a
    b_correct = pred_b == label_b
    if a_correct and not b_correct:
        print(f'{fn:25s} {folder_b:20s} {pred_b:30s} {conf_b:5.2f}')
    elif b_correct and not a_correct:
        print(f'{fn:25s} {folder_a:20s} {pred_a:30s} {conf_a:5.2f}')
    elif not a_correct and not b_correct:
        print(f'{fn:25s} BOTH WRONG: {folder_a}->{pred_a} {folder_b}->{pred_b}')
