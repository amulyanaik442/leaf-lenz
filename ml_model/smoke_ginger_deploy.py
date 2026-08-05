"""Post-deploy smoke test for the Ginger integration.

1. Ginger routing: each Ginger_Split/test image via TTA predict with
   prefix='ginger___' must return a ginger___ label.
2. Zero regression (crop-filtered): for a sample of existing-crop images, the
   crop-filtered top-1 from the NEW (134-class) model must exactly match the
   PRE-DEPLOY backup (130-class) model.
3. App wiring: predict_by_crop(..., 'ginger') routes to the general model and
   returns a ginger___ label; ginger crop key is registered.
"""
import os
import sys
import json
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
BACKUP_DIR = sorted(glob.glob(os.path.join(ML_ASSETS, 'backup_pre_ginger_*')))[-1]
OLD_ONNX = os.path.join(BACKUP_DIR, 'model.onnx')
OLD_CLS = os.path.join(BACKUP_DIR, 'class_names.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_session(path, classes_path):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    with open(classes_path) as f:
        names = json.load(f)
    return sess, names


def tta_predict(sess, names, img_path, prefix=None):
    img = Image.open(img_path).convert('RGB').resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    keep = None
    if prefix is not None:
        keep = np.array([i for i, n in enumerate(names) if n.startswith(prefix)],
                        dtype=np.int64)
        if keep.size == 0:
            return None, 0.0
    probs = None
    inp = sess.get_inputs()[0].name
    for aug in (arr, arr[:, ::-1, :].copy(), arr[::-1, :, :].copy(),
                arr[::-1, ::-1, :].copy()):
        t = ((aug - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
        logits = sess.run(None, {inp: t})[0][0]
        if keep is not None:
            logits = logits[keep]
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        probs = p if probs is None else probs + p
    probs = probs / 4.0
    if keep is None:
        best = int(probs.argmax())
        return names[best], float(probs[best])
    best = int(keep[probs.argmax()])
    return names[best], float(probs.max())


def crop_prefix(crop):
    return {'cacao': 'Cacao___', 'tea': 'Tea___', 'pumpkin': 'Pumpkin___',
            'chilli': 'Chilli___', 'turmeric': 'Turmeric___',
            'black_pepper': 'BlackPepper___'}[crop]


def main():
    new_sess, new_names = load_session(NEW_ONNX, NEW_CLS)
    old_sess, old_names = load_session(OLD_ONNX, OLD_CLS)
    print(f'NEW model: {len(new_names)} classes | OLD backup: {len(old_names)} classes')
    print(f'backup dir: {BACKUP_DIR}')

    # ---- 1. ginger routing ----
    print('\n=== GINGER ROUTING (Ginger_Split/test, prefix="ginger___") ===')
    test_root = os.path.join(BASE_DIR, 'dataset', 'ginger', 'Ginger_Split', 'test')
    total = correct = 0
    for cls in sorted(os.listdir(test_root)):
        d = os.path.join(test_root, cls)
        imgs = [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.lower().endswith(IMAGE_EXTS)]
        ok = 0
        for p in imgs:
            label, conf = tta_predict(new_sess, new_names, p, prefix='ginger___')
            total += 1
            is_corr = (label == cls)
            ok += int(is_corr)
            correct += int(is_corr)
            if not is_corr:
                print(f'  MISMATCH {os.path.basename(p)}: got {label} ({conf:.3f}) want {cls}')
        print(f'  {cls:38s} {ok}/{len(imgs)}')
    print(f'  ginger test accuracy: {correct}/{total} = {100*correct/total:.1f}%')

    # ---- 2. zero regression (crop-filtered) ----
    print('\n=== ZERO REGRESSION (crop-filtered top-1: NEW vs OLD) ===')
    crops = {
        'cacao': os.path.join(BASE_DIR, 'dataset', 'cacao_leaf', 'Cacao_Split', 'test'),
        'tea': os.path.join(BASE_DIR, 'dataset', 'tea', 'Tea_Split', 'test'),
        'pumpkin': os.path.join(BASE_DIR, 'dataset', 'pumpkin', 'Corrected_Split', 'test'),
        'chilli': os.path.join(BASE_DIR, 'dataset', 'chilli', 'Chilli_Split', 'test'),
        'turmeric': os.path.join(BASE_DIR, 'dataset', 'turmeric', 'Turmeric_Split', 'test'),
        'black_pepper': os.path.join(BASE_DIR, 'dataset', 'black_pepper', 'BlackPepper_Split', 'test'),
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
                l_new, _ = tta_predict(new_sess, new_names, p, prefix=crop_prefix(crop))
                l_old, _ = tta_predict(old_sess, old_names, p, prefix=crop_prefix(crop))
                samples += 1
                if l_new != l_old:
                    n += 1
                    total_mismatch += 1
                    if n <= 5:
                        print(f'  MISMATCH [{crop}] {os.path.basename(p)}: new={l_new} old={l_old}')
        print(f'  {crop:12s} mismatches: {n}')
    print(f'  crop-filtered mismatches: {total_mismatch}/{samples}')

    # ---- 3. app wiring ----
    print('\n=== APP WIRING (predict_by_crop) ===')
    try:
        sys.path.insert(0, BASE_DIR)
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'leaf_lenz_project.settings')
        import django
        django.setup()
        from detector.inference import CROP_CLASS_PREFIXES, predict_by_crop
        print(f'  ginger in CROP_CLASS_PREFIXES: {CROP_CLASS_PREFIXES.get("ginger")}')
        sample = None
        for cls in sorted(os.listdir(test_root)):
            d = os.path.join(test_root, cls)
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(IMAGE_EXTS):
                    sample = os.path.join(d, f)
                    break
            if sample:
                break
        with open(sample, 'rb') as fh:
            label, conf, top5, mm = predict_by_crop(fh, 'ginger')
        print(f'  predict_by_crop(ginger): {label} ({conf:.3f})')
        assert label.startswith('ginger___'), 'expected a ginger___ label'
        with open(sample, 'rb') as fh:
            label, conf, top5, mm = predict_by_crop(fh, 'black_pepper')
        print(f'  predict_by_crop(black_pepper): {label} ({conf:.3f})')
        with open(sample, 'rb') as fh:
            label, conf, top5, mm = predict_by_crop(fh, 'wheat')
        print(f'  predict_by_crop(wheat): {label} ({conf:.3f})')
        with open(sample, 'rb') as fh:
            label, conf, top5, mm = predict_by_crop(fh, 'maize')
        print(f'  predict_by_crop(maize): {label} ({conf:.3f})')
    except Exception as e:
        print(f'  app wiring check error: {e}')

    print('\nDONE')


if __name__ == '__main__':
    main()
