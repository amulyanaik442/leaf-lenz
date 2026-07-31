"""Compare per-class accuracy: before correction (bak) vs after (current)."""
import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ASSETS = os.path.join(BASE, 'detector', 'ml_assets')
CLASSES = json.load(open(os.path.join(ML_ASSETS, 'class_names.json')))
PUMPKIN = ['Pumpkin___bacterial_leaf_spot', 'Pumpkin___downy_mildew',
           'Pumpkin___healthy', 'Pumpkin___mosaic_disease', 'Pumpkin___powdery_mildew']
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TEST_DIR = os.path.join(BASE, 'dataset', 'pumpkin', 'Corrected_Split', 'test')
FOLDERS = ['Bacterial Leaf Spot', 'Downy Mildew', 'Healthy Leaf',
           'Mosaic Disease', 'Powdery Mildew']


def preprocess(arr):
    arr = arr.copy()
    if arr.ndim == 3:
        arr = np.stack([arr, arr[:, ::-1], arr[::-1], arr[::-1, ::-1]])
    arr = (arr - MEAN) / STD
    return arr.transpose(0, 3, 1, 2).astype(np.float32)


def predict_probs(sess, img):
    arr = np.array(img.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
    tensors = preprocess(arr)
    inp = sess.get_inputs()[0].name
    probs = []
    for t in tensors:
        logits = sess.run(None, {inp: t[None]})[0][0]
        e = np.exp(logits - logits.max())
        probs.append(e / e.sum())
    return np.mean(probs, axis=0)


def eval_model(path):
    sess = ort.InferenceSession(path)
    cm = np.zeros((5, 5), dtype=int)
    correct = 0
    total = 0
    for i, folder in enumerate(FOLDERS):
        d = os.path.join(TEST_DIR, folder)
        for f in os.listdir(d):
            img = Image.open(os.path.join(d, f)).convert('RGB')
            probs = predict_probs(sess, img)
            pred = int(np.argmax(probs))
            if pred in [103, 104, 105, 106, 107]:
                j = pred - 103
            else:
                j = 5
            if j < 5:
                cm[i, j] += 1
                if j == i:
                    correct += 1
            total += 1
    accs = cm.diagonal() / cm.sum(axis=1).astype(float) * 100
    return accs, cm, 100.0 * correct / total


if __name__ == '__main__':
    results = {}
    for label, path in [('BEFORE (pre-correction)', 'model.onnx.bak_pre_corrected'),
                        ('AFTER (corrected)', 'model.onnx')]:
        accs, cm, overall = eval_model(os.path.join(ML_ASSETS, path))
        results[label] = {'per_class_acc': dict(zip(FOLDERS, accs.round(2))),
                          'overall_acc': round(overall, 2), 'cm': cm.tolist()}
        print(f'\n{label}  (overall {overall:.2f}%)')
        print(f'  {"class":25s} accuracy')
        for c, a in zip(FOLDERS, accs):
            print(f'  {c:25s} {a:.2f}%')
        short = ['BLS', 'DM', 'HL', 'MD', 'PM']
        print('  ' + ' '*14 + ''.join(f'{s:>6s}' for s in short))
        for r in range(5):
            print(f'  {short[r]:14s}' + ''.join(f'{cm[r][c]:6d}' for c in range(5)))
    json.dump(results, open(os.path.join(ML_ASSETS, 'pumpkin_before_after.json'), 'w'),
              indent=4)
    print('\nSaved to detector/ml_assets/pumpkin_before_after.json')
