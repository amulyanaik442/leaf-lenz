"""
Evaluate the Tea candidate vs the production model on an external (held-out)
Google-image test set.

Uses the app's exact preprocessing (bicubic 224 + 4-view TTA) and reports:
  * per-class + overall accuracy for the candidate (116 classes)
  * confusion matrix (png + json)
  * representative misclassified images (copied to an output folder)
  * per-class comparison vs the current production model

Usage:
    python ml_model/evaluate_tea_external.py
"""
import os
import sys
import json
import shutil

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ASSETS = os.path.join(BASE_DIR, 'detector', 'ml_assets')
CANDIDATE = os.path.join(BASE_DIR, 'ml_model', 'tea_retrain_output_v2',
                         'tea_candidate.onnx')
PRODUCTION = os.path.join(ML_ASSETS, 'model.onnx')
CAND_NAMES = os.path.join(BASE_DIR, 'ml_model', 'tea_retrain_output_v2',
                          'candidate_class_names.json')
PROD_NAMES = os.path.join(ML_ASSETS, 'class_names.json')
OUT_DIR = os.path.join(BASE_DIR, 'ml_model', 'tea_retrain_output_v2',
                       'external_eval')
EXTERNAL_DIR = r'C:\Users\amuly\Desktop\tea_external_test'

FOLDER_TO_MODEL = {
    'algal': 'Tea___algal_leaf',
    'anthracnoses': 'Tea___anthracnose',
    'anthracnose': 'Tea___anthracnose',
    'bird eye spot': 'Tea___bird_eye_spot',
    'brown blight': 'Tea___blight',
    'gray light': 'Tea___blight',
    'healthy': 'Tea___healthy',
    'red leaf spot': 'Tea___red_leaf_spot',
    'white spot': 'Tea___white_spot',
}
TEA_NAMES = list(dict.fromkeys(FOLDER_TO_MODEL.values()))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_arr(arr):
    if arr.ndim == 3:
        arr = np.stack([arr, arr[:, ::-1], arr[::-1], arr[::-1, ::-1]])
    arr = (arr - MEAN) / STD
    return arr.transpose(0, 3, 1, 2).astype(np.float32)


def predict_tta(sess, img):
    arr = np.array(img.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
    tensors = preprocess_arr(arr)
    inp = sess.get_inputs()[0].name
    probs = []
    for t in tensors:
        logits = sess.run(None, {inp: t[None]})[0][0]
        e = np.exp(logits - logits.max())
        probs.append(e / e.sum())
    return np.mean(probs, axis=0)


def main():
    if not os.path.isdir(EXTERNAL_DIR):
        print(f'External test dir not found: {EXTERNAL_DIR}')
        print('Please download ~25 Google images per class into the 8 subfolders.')
        return

    with open(CAND_NAMES) as f:
        cand_names = json.load(f)
    with open(PROD_NAMES) as f:
        prod_names = json.load(f)
    tea_idx = [i for i, n in enumerate(cand_names) if n.startswith('Tea___')]
    print(f'Candidate classes: {len(cand_names)} | Tea indices: {tea_idx}')

    cand_sess = ort.InferenceSession(CANDIDATE)
    prod_sess = ort.InferenceSession(PRODUCTION)

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {}
    all_misclassified = []

    print(f'\n{"class":18s} {"n":>4s} {"cand_acc":>8s} {"tea_filt":>10s} {"prod_acc":>8s}')
    for folder, model_cls in FOLDER_TO_MODEL.items():
        d = os.path.join(EXTERNAL_DIR, folder)
        if not os.path.isdir(d):
            print(f'{folder:18s}  MISSING FOLDER')
            continue
        imgs = sorted([f for f in os.listdir(d)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        cand_correct = 0
        filt_correct = 0
        prod_correct = 0
        wrong_candidates = []
        for f in imgs:
            try:
                img = Image.open(os.path.join(d, f)).convert('RGB')
            except Exception:
                continue
            cand_probs = predict_tta(cand_sess, img)
            cand_pred = int(np.argmax(cand_probs))
            # Crop-filtered (app behavior for crop=tea): renormalize over Tea only
            tea_probs = cand_probs[tea_idx]
            tea_probs = tea_probs / tea_probs.sum()
            filt_pred = tea_idx[int(np.argmax(tea_probs))]
            prod_probs = predict_tta(prod_sess, img)
            prod_pred = int(np.argmax(prod_probs))
            target = TEA_NAMES.index(model_cls)
            if cand_pred == tea_idx[target]:
                cand_correct += 1
            if filt_pred == tea_idx[target]:
                filt_correct += 1
            if cand_pred != tea_idx[target]:
                pred_name = cand_names[cand_pred] if cand_pred < len(cand_names) else '?'
                wrong_candidates.append({
                    'file': f, 'true': model_cls, 'pred': pred_name,
                    'conf': round(float(cand_probs[cand_pred]), 3),
                    'true_conf': round(float(cand_probs[tea_idx[target]]), 3),
                })
                all_misclassified.append({'class': folder, 'file': f,
                                          'pred': pred_name})
            if prod_pred == tea_idx[target]:
                prod_correct += 1
        n = len(imgs)
        cand_acc = 100.0 * cand_correct / n if n else 0.0
        filt_acc = 100.0 * filt_correct / n if n else 0.0
        prod_acc = 100.0 * prod_correct / n if n else 0.0
        report[folder] = {'images': n, 'candidate_accuracy': round(cand_acc, 2),
                          'tea_filtered_accuracy': round(filt_acc, 2),
                          'production_accuracy': round(prod_acc, 2),
                          'misclassified': wrong_candidates}
        print(f'{folder:18s} {n:4d} {cand_acc:7.2f}% {filt_acc:10.2f}% {prod_acc:8.2f}%')

    total_n = sum(r['images'] for r in report.values())
    cand_total = sum(r['images'] * r['candidate_accuracy'] / 100
                     for r in report.values())
    filt_total = sum(r['images'] * r['tea_filtered_accuracy'] / 100
                     for r in report.values())
    prod_total = sum(r['images'] * r['production_accuracy'] / 100
                     for r in report.values())
    cand_overall = 100.0 * cand_total / total_n if total_n else 0.0
    filt_overall = 100.0 * filt_total / total_n if total_n else 0.0
    prod_overall = 100.0 * prod_total / total_n if total_n else 0.0
    print('-' * 40)
    print(f'{"OVERALL":18s} {total_n:4d} {cand_overall:7.2f}% {filt_overall:10.2f}% {prod_overall:8.2f}%')
    report['overall'] = {'images': total_n,
                         'candidate_accuracy': round(cand_overall, 2),
                         'tea_filtered_accuracy': round(filt_overall, 2),
                         'production_accuracy': round(prod_overall, 2)}

    # Representative misclassified images (up to 3 per class)
    os.makedirs(os.path.join(OUT_DIR, 'misclassified_samples'), exist_ok=True)
    for folder in FOLDER_TO_MODEL:
        rows = report.get(folder, {}).get('misclassified', [])[:3]
        for r in rows:
            src = os.path.join(EXTERNAL_DIR, folder, r['file'])
            dst = os.path.join(OUT_DIR, 'misclassified_samples',
                               f"{folder}__{r['pred']}__{r['file']}")
            if os.path.exists(src):
                shutil.copy2(src, dst)

    with open(os.path.join(OUT_DIR, 'external_eval_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    print(f'\nFull report: {os.path.join(OUT_DIR, "external_eval_report.json")}')
    print(f'Misclassified samples: {os.path.join(OUT_DIR, "misclassified_samples")}')


if __name__ == '__main__':
    main()
