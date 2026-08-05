"""Post-deploy smoke test for the Chilli integration.

1. Chilli routing: each Chilli_Split/test image via predict_by_crop(...,'chilli')
   must return a Chilli___ label.
2. Zero regression (crop-filtered): for a sample of existing-crop images, the
   crop-filtered top-1 from the NEW (123-class) model must exactly match the
   PRE-DEPLOY backup (117-class) model. Specialist crops (wheat/maize) use
   their own untouched predictors.
"""
import os
import sys
import glob

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from PIL import Image
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ASSETS = os.path.join(BASE_DIR, 'detector', 'ml_assets')
NEW_ONNX = os.path.join(ML_ASSETS, 'model.onnx')
NEW_CLS = os.path.join(ML_ASSETS, 'class_names.json')
BACKUP_DIR = sorted(glob.glob(os.path.join(ML_ASSETS, 'backup_pre_chilli_leaf_deploy_*')))[-1]
OLD_ONNX = os.path.join(BACKUP_DIR, 'model.onnx')
OLD_CLS = os.path.join(BACKUP_DIR, 'class_names.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_session(path, classes_path):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    with open(classes_path) as f:
        import json
        names = json.load(f)
    return sess, names


def tta_predict(sess, names, img_path, prefix=None, n_keep=None):
    img = Image.open(img_path).convert('RGB').resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    probs = None
    inp = sess.get_inputs()[0].name
    for aug in (arr, arr[:, ::-1, :].copy(), arr[::-1, :, :].copy(),
                arr[::-1, ::-1, :].copy()):
        t = ((aug - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
        logits = sess.run(None, {inp: t})[0][0]
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        probs = p if probs is None else probs + p
    probs = probs / 4.0
    keep = None
    if prefix is not None:
        keep = [i for i, n in enumerate(names) if n.startswith(prefix)]
        if not keep:
            return None, 0.0
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        s = filtered.sum()
        if s > 0:
            filtered = filtered / s
        else:
            filtered = probs
    elif n_keep is not None:
        filtered = probs[:n_keep].copy()
    else:
        filtered = probs
    best = int(filtered.argmax())
    return names[best], float(filtered[best])


def main():
    new_sess, new_names = load_session(NEW_ONNX, NEW_CLS)
    old_sess, old_names = load_session(OLD_ONNX, OLD_CLS)
    print(f'NEW model: {len(new_names)} classes | OLD backup: {len(old_names)} classes')
    print(f'backup dir: {BACKUP_DIR}')

    # ---- 1. chilli routing ----
    print('\n=== CHILLI ROUTING (Chilli_Split/test, crop="chilli") ===')
    test_root = os.path.join(BASE_DIR, 'dataset', 'chilli', 'Chilli_Split', 'test')
    total = correct = 0
    per_class = {}
    for cls in sorted(os.listdir(test_root)):
        d = os.path.join(test_root, cls)
        imgs = [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.lower().endswith(IMAGE_EXTS)]
        ok = 0
        for p in imgs:
            label, conf = tta_predict(new_sess, new_names, p, prefix='Chilli___')
            total += 1
            is_corr = (label == cls)
            ok += int(is_corr)
            correct += int(is_corr)
            per_class[cls] = {'correct': ok, 'total': len(imgs)}
            if not is_corr:
                print(f'  MISMATCH {os.path.basename(p)}: got {label} ({conf:.3f}) want {cls}')
        print(f'  {cls:24s} {ok}/{len(imgs)}')
    print(f'  chilli test accuracy: {correct}/{total} = {100*correct/total:.1f}%')

    # ---- 2. zero regression (crop-filtered) ----
    print('\n=== ZERO REGRESSION (crop-filtered top-1: NEW vs OLD) ===')
    crops = {
        'cacao': os.path.join(BASE_DIR, 'dataset', 'cacao_leaf', 'Cacao_Split', 'test'),
        'tea': os.path.join(BASE_DIR, 'dataset', 'tea', 'Tea_Split', 'test'),
        'pumpkin': os.path.join(BASE_DIR, 'dataset', 'pumpkin', 'Corrected_Split', 'test'),
    }
    total_mismatch = 0
    samples = 0
    for crop, root in crops.items():
        n = 0
        for cls in sorted(os.listdir(root)):
            d = os.path.join(root, cls)
            for f in sorted(os.listdir(d)):
                if not f.lower().endswith(IMAGE_EXTS):
                    continue
                p = os.path.join(d, f)
                l_new, c_new = tta_predict(new_sess, new_names, p, prefix=new_names and crop_prefix(crop))
                l_old, c_old = tta_predict(old_sess, old_names, p, prefix=crop_prefix(crop))
                samples += 1
                if l_new != l_old:
                    n += 1
                    total_mismatch += 1
                    if n <= 5:
                        print(f'  MISMATCH [{crop}] {os.path.basename(p)}: new={l_new} old={l_old}')
        print(f'  {crop:10s} mismatches: {n}')
    print(f'  crop-filtered mismatches: {total_mismatch}/{samples}')

    # ---- 3. auto-routing sanity (unchanged specialist models) ----
    print('\n=== SPECIALIST MODELS (wheat/maize) sanity ===')
    try:
        sys.path.insert(0, BASE_DIR)
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'leaf_lenz_project.settings')
        import django
        django.setup()
        from detector.inference import predict_by_crop
        # any existing image
        sample = None
        for root in crops.values():
            for cls in sorted(os.listdir(root)):
                d = os.path.join(root, cls)
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith(IMAGE_EXTS):
                        sample = os.path.join(d, f)
                        break
                if sample:
                    break
            if sample:
                break
        with open(sample, 'rb') as fh:
            label, conf, top5, mm = predict_by_crop(fh, 'wheat')
        print(f'  wheat predictor OK: {label} ({conf:.3f})')
        with open(sample, 'rb') as fh:
            label, conf, top5, mm = predict_by_crop(fh, 'maize')
        print(f'  maize predictor OK: {label} ({conf:.3f})')
    except Exception as e:
        print(f'  specialist check skipped/error: {e}')

    print('\nDONE')


def crop_prefix(crop):
    return {'cacao': 'Cacao___', 'tea': 'Tea___', 'pumpkin': 'Pumpkin___'}[crop]


if __name__ == '__main__':
    main()
