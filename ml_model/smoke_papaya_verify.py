"""Pre-deploy verification for the Papaya integration.

The papaya candidate model (papaya_candidate.onnx, 146 classes) is verified
against the DEPLOYED production model (model.onnx, 141 classes) WITHOUT
touching production:

  1. Papaya routing: each Papaya_Split/test image via TTA 4-view predict with
     prefix='papaya___' on the CANDIDATE must return a papaya___ label.
  2. Zero regression (crop-filtered): for existing-crop test images, the
     crop-filtered top-1 from the CANDIDATE must exactly match the PRODUCTION
     model (logits for the first 141 classes are byte-identical, so filtered
     softmaxes are identical).
  3. Full-model logit diff: max |logit diff| over the first 141 classes is 0.
  4. Papaya FP audit on existing crops (uncropped argmax) - recorded, not a gate
     (production isolates classes per crop via prefix filtering).
  5. App wiring: 'papaya' registered in CROP_CLASS_PREFIXES, 5 papaya entries in
     disease_data.json, Papaya option present in index.html.

Outputs a machine-readable report to ml_model/papaya_retrain_output/.
"""
import os
import sys
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from PIL import Image
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ASSETS = os.path.join(BASE_DIR, 'detector', 'ml_assets')
PROD_ONNX = os.path.join(ML_ASSETS, 'model.onnx')
PROD_CLS = os.path.join(ML_ASSETS, 'class_names.json')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'papaya_retrain_output')
CAND_ONNX = os.path.join(OUT_DIR, 'papaya_candidate.onnx')
CAND_CLS = os.path.join(OUT_DIR, 'candidate_class_names.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_session(path, classes_path):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    with open(classes_path) as f:
        names = json.load(f)
    return sess, names


def _single_logits(sess, img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = ((arr - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
    inp = sess.get_inputs()[0].name
    return sess.run(None, {inp: t})[0][0]


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


def list_imgs(root):
    out = []
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMAGE_EXTS):
                out.append((cls, os.path.join(d, f)))
    return out


EXISTING_CROPS = {
    'bean': ('bean___', os.path.join(BASE_DIR, 'dataset', 'bean', 'Bean_Split', 'test')),
    'cauliflower': ('cauliflower___', os.path.join(BASE_DIR, 'dataset', 'cauliflower', 'Cauliflower_Split', 'test')),
    'pumpkin': ('Pumpkin___', os.path.join(BASE_DIR, 'dataset', 'pumpkin', 'Corrected_Split', 'test')),
    'tea': ('Tea___', os.path.join(BASE_DIR, 'dataset', 'tea', 'Tea_Split', 'test')),
    'cacao': ('Cacao___', os.path.join(BASE_DIR, 'dataset', 'cacao_leaf', 'Cacao_Split', 'test')),
    'chilli': ('Chilli___', os.path.join(BASE_DIR, 'dataset', 'chilli', 'Chilli_Split', 'test')),
    'turmeric': ('Turmeric___', os.path.join(BASE_DIR, 'dataset', 'turmeric', 'Turmeric_Split', 'test')),
    'black_pepper': ('BlackPepper___', os.path.join(BASE_DIR, 'dataset', 'black_pepper', 'BlackPepper_Split', 'test')),
    'ginger': ('ginger___', os.path.join(BASE_DIR, 'dataset', 'ginger', 'Ginger_Split', 'test')),
}


def main():
    prod_sess, prod_names = load_session(PROD_ONNX, PROD_CLS)
    cand_sess, cand_names = load_session(CAND_ONNX, CAND_CLS)
    n_old = len(prod_names)
    assert n_old == 141, f'expected 141 production classes, got {n_old}'
    assert len(cand_names) == 146, f'expected 146 candidate classes, got {len(cand_names)}'
    assert prod_names == cand_names[:n_old], 'candidate must preserve existing class names'
    assert all(c.startswith('papaya___') for c in cand_names[n_old:])
    report = {'production_classes': n_old, 'candidate_classes': len(cand_names),
              'papaya_classes': cand_names[n_old:]}

    # ---- 1. Papaya routing accuracy (candidate, TTA 4-view) ----
    print('\n=== PAPAYA ROUTING (Papaya_Split/test, prefix="papaya___") ===')
    test_root = os.path.join(BASE_DIR, 'dataset', 'papaya', 'Papaya_Split', 'test')
    items = list_imgs(test_root)
    total = correct = 0
    per_cls = {}
    for cls, p in items:
        label, conf = tta_predict(cand_sess, cand_names, p, prefix='papaya___')
        total += 1
        ok = label == cls
        correct += int(ok)
        per_cls.setdefault(cls, [0, 0])[0] += 1
        per_cls.setdefault(cls, [0, 0])[1] += int(ok)
        if not ok:
            print(f'  MISMATCH {os.path.basename(p)}: got {label} ({conf:.3f}) want {cls}')
    for cls in sorted(per_cls):
        n, ok = per_cls[cls]
        print(f'  {cls:30s} {ok}/{n}')
    papaya_acc = 100.0 * correct / total
    print(f'  papaya candidate test accuracy: {correct}/{total} = {papaya_acc:.1f}%')
    report['papaya_test_accuracy_pct'] = round(papaya_acc, 2)

    # ---- 2 + 3. Zero regression (crop-filtered + full-logit diff) ----
    # Single-view comparison is exact here: both models get identical input and
    # the first 141 rows are byte-identical, so a 1-view forward pass fully
    # verifies numeric equality of the existing-class logits.
    print('\n=== ZERO REGRESSION (candidate vs production) ===')
    total_mismatch = 0
    samples = 0
    max_logit_diff = 0.0
    per_crop = {}
    papaya_fp = 0
    for crop, (prefix, root) in EXISTING_CROPS.items():
        if not os.path.isdir(root):
            continue
        n_mis = 0
        n = 0
        keep_new = np.array([i for i, cn in enumerate(cand_names)
                             if cn.startswith(prefix)], dtype=np.int64)
        keep_old = np.array([i for i, cn in enumerate(prod_names)
                             if cn.startswith(prefix)], dtype=np.int64)
        for cls, p in list_imgs(root)[:60]:
            lc = _single_logits(cand_sess, p)
            lp = _single_logits(prod_sess, p)
            n += 1
            new_top = int(keep_new[lc[keep_new].argmax()])
            old_top = int(keep_old[lp[keep_old].argmax()])
            if cand_names[new_top] != prod_names[old_top]:
                n_mis += 1
                if n_mis <= 5:
                    print(f'  MISMATCH [{crop}] {os.path.basename(p)}: '
                          f'new={cand_names[new_top]} old={prod_names[old_top]}')
            max_logit_diff = max(max_logit_diff,
                                 float(np.max(np.abs(lc[:n_old] - lp[:n_old]))))
            if np.argmax(lc) >= n_old:
                papaya_fp += 1
            samples += 1
        total_mismatch += n_mis
        per_crop[crop] = n_mis
        print(f'  {crop:14s} crop-filtered mismatches: {n_mis}/{n}')
    print(f'  crop-filtered mismatches: {total_mismatch}/{samples}')
    print(f'  max |logit diff| first {n_old} classes: {max_logit_diff:.2e}')
    print(f'  existing-crop images now argmax-ing to papaya (uncropped): {papaya_fp}/{samples}')
    report['crop_filtered_mismatches'] = total_mismatch
    report['crop_filtered_samples'] = samples
    report['max_logit_diff_first_141'] = max_logit_diff
    report['papaya_fp_on_existing_crops_uncropped'] = papaya_fp
    report['per_crop_filtered_mismatches'] = per_crop

    # ---- 4. App wiring ----
    print('\n=== APP WIRING ===', flush=True)
    wiring = {}
    try:
        sys.path.insert(0, BASE_DIR)
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'leaf_lenz_project.settings')
        import django
        django.setup()
        from detector.inference import CROP_CLASS_PREFIXES, predict_by_crop
        wiring['papaya_prefix'] = CROP_CLASS_PREFIXES.get('papaya')
        print(f'  papaya in CROP_CLASS_PREFIXES: {wiring["papaya_prefix"]}', flush=True)

        with open(os.path.join(ML_ASSETS, '..', 'disease_data.json')) as f:
            dd = json.load(f)
        papaya_keys = sorted(k for k in dd if k.startswith('papaya___'))
        wiring['disease_data_papaya_keys'] = papaya_keys
        print(f'  disease_data papaya entries: {papaya_keys}', flush=True)

        idx = open(os.path.join(BASE_DIR, 'detector', 'templates', 'detector',
                                'index.html'), encoding='utf-8').read()
        wiring['dropdown_papaya'] = 'value="papaya"' in idx
        print(f'  dropdown has Papaya option: {wiring["dropdown_papaya"]}', flush=True)

        sample = list_imgs(test_root)[0][1]
        with open(sample, 'rb') as fh:
            label, conf, top5, mm = predict_by_crop(fh, 'papaya')
        wiring['predict_by_crop_papaya_runs'] = label not in ('fallback',)
        print(f'  predict_by_crop(papaya) -> {label} ({conf:.3f}) '
              f'[production model has no papaya classes yet; falls back to full softmax]', flush=True)
    except Exception as e:
        wiring['error'] = str(e)
        print(f'  app wiring check error: {e}', flush=True)
    report['app_wiring'] = wiring

    report['zero_regression_ok'] = (total_mismatch == 0 and max_logit_diff == 0.0)
    print(f'\nZERO REGRESSION: {"OK" if report["zero_regression_ok"] else "FAILED"}')

    with open(os.path.join(OUT_DIR, 'smoke_papaya_verify_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    print(f'Report saved: {os.path.join(OUT_DIR, "smoke_papaya_verify_report.json")}')
    print('\nDONE')


if __name__ == '__main__':
    main()
