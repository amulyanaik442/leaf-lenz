"""
Automated label screen for the external Google tea set.
For each external image: compute 1280-d backbone features, then cosine
distance to the 7 training-class centroids (from v2 train features).
Flags images whose nearest centroid is NOT their labeled class (likely
mislabel) or whose distance to their own centroid is an outlier (OOD).

Usage:
    python ml_model/screen_external_tea.py
"""
import os
import sys
import json
import copy
import pickle

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ASSETS = os.path.join(BASE_DIR, 'detector', 'ml_assets')
PRODUCTION_ONNX = os.path.join(ML_ASSETS, 'model.onnx')
FEAT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'tea_features_v2.pkl')
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
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_feature_extractor():
    model = onnx.load(PRODUCTION_ONNX)
    classifier_input = None
    for node in model.graph.node:
        if node.op_type in ('Gemm', 'MatMul'):
            for out in model.graph.output:
                if out.name in node.output:
                    classifier_input = node.input[0]
                    break
        if classifier_input:
            break
    feat_model = copy.deepcopy(model)
    feat_model.graph.output.append(onnx.helper.make_tensor_value_info(
        classifier_input, onnx.TensorProto.FLOAT, [None, 1280]))
    feat_path = PRODUCTION_ONNX.replace('.onnx', '_features_tmp.onnx')
    onnx.save(feat_model, feat_path)
    sess = ort.InferenceSession(feat_path)
    return sess, sess.get_inputs()[0].name, feat_path


def extract_features(sess, inp, img):
    arr = np.array(img.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
    x = (arr - IMG_MEAN) / IMG_STD
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    return sess.run(None, {inp: x})[-1][0]


def l2norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def main():
    with open(FEAT_CACHE, 'rb') as f:
        cache = pickle.load(f)
    train_feats, train_labels = cache['train_feats'], cache['train_labels']

    centroids = {}
    for i, name in enumerate(TEA_NAMES):
        mask = train_labels == i
        if mask.sum():
            c = l2norm(train_feats[mask].mean(axis=0))
            centroids[name] = c
    # own-centroid distance distribution per training class (for thresholds)
    tr_norm = l2norm(train_feats)
    train_dists = {}
    for i, name in enumerate(TEA_NAMES):
        mask = train_labels == i
        d = 1.0 - np.dot(tr_norm[mask], centroids[name])
        train_dists[name] = np.percentile(d, 95)

    sess, inp, tmp = load_feature_extractor()
    flagged = {}
    for folder, model_cls in FOLDER_TO_MODEL.items():
        d = os.path.join(EXTERNAL_DIR, folder)
        if not os.path.isdir(d):
            continue
        rows = []
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            try:
                img = Image.open(os.path.join(d, f)).convert('RGB')
                feat = extract_features(sess, inp, img)
            except Exception as e:
                rows.append({'file': f, 'error': str(e)})
                continue
            feat_n = l2norm(feat)
            dists = {name: float(1.0 - np.dot(feat_n, c))
                     for name, c in centroids.items()}
            nearest = min(dists, key=dists.get)
            own_dist = dists[model_cls]
            threshold = float(train_dists[model_cls])
            is_ood = own_dist > threshold
            is_mislabel = (nearest != model_cls)
            rows.append({'file': f, 'nearest_centroid': nearest,
                         'nearest_dist': round(dists[nearest], 3),
                         'own_class_dist': round(own_dist, 3),
                         'own_95pct_threshold': threshold,
                         'flag': 'OOD' if is_ood and not is_mislabel
                                 else 'MISLABEL?' if is_mislabel
                                 else 'ok'})
        flagged[folder] = rows
        print(f'--- {folder} -> {model_cls} ---')
        for r in rows:
            if r.get('flag', 'ok') != 'ok':
                print(f'  [{r["flag"]}] {r["file"]:40s} nearest={r["nearest_centroid"]:22s} '
                      f'dist={r["nearest_dist"]} own={r["own_class_dist"]} (thr {r["own_95pct_threshold"]})')
    if os.path.exists(tmp):
        os.remove(tmp)

    with open(os.path.join(BASE_DIR, 'ml_model', 'tea_retrain_output_v2',
                           'external_screen_report.json'), 'w') as f:
        json.dump(flagged, f, indent=4)
    print('\nFull screen: ml_model/tea_retrain_output_v2/external_screen_report.json')


if __name__ == '__main__':
    main()
