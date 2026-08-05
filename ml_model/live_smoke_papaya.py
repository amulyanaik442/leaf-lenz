"""Live API smoke test for the deployed Papaya integration.

POSTs every Papaya_Split/test image to the RUNNING server
(http://127.0.0.1:8000/api/predict/) with crop=papaya and verifies:
  - every response has a papaya___ label (routing works end-to-end)
  - predicted class matches the ground-truth folder
Reports accuracy and saves a JSON result.
"""
import os
import sys
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = 'http://127.0.0.1:8000/api/predict/'
TEST_ROOT = os.path.join(BASE_DIR, 'dataset', 'papaya', 'Papaya_Split', 'test')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'papaya_retrain_output', 'live_smoke_papaya.json')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')

results = {'total': 0, 'routed_ok': 0, 'correct': 0, 'by_class': {}, 'errors': []}

for cls in sorted(os.listdir(TEST_ROOT)):
    d = os.path.join(TEST_ROOT, cls)
    if not os.path.isdir(d):
        continue
    per = [0, 0]
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        p = os.path.join(d, f)
        with open(p, 'rb') as fh:
            resp = requests.post(API, files={'image': fh}, data={'crop': 'papaya'}, timeout=120)
        if resp.status_code != 200:
            results['errors'].append({'file': f, 'status': resp.status_code, 'body': resp.text[:300]})
            continue
        data = resp.json()
        pred = data.get('prediction') or {}
        label = pred.get('raw_label') or ''
        conf = pred.get('confidence')
        results['total'] += 1
        per[0] += 1
        routed = label.startswith('papaya___')
        results['routed_ok'] += int(routed)
        ok = label == cls
        results['correct'] += int(ok)
        per[1] += int(ok)
        if not routed or not ok:
            print(f'  {f:40s} -> {label:32s} conf={conf:.3f} want={cls}')
    results['by_class'][cls] = per
    print(f'  {cls:30s} {per[1]}/{per[0]}')

results['accuracy_pct'] = round(100.0 * results['correct'] / results['total'], 2)
results['routing_pct'] = round(100.0 * results['routed_ok'] / results['total'], 2)
print(f'\n  LIVE papaya accuracy: {results["correct"]}/{results["total"]} = {results["accuracy_pct"]}%')
print(f'  LIVE papaya routing:  {results["routed_ok"]}/{results["total"]} = {results["routing_pct"]}%')

with open(OUT, 'w') as f:
    json.dump(results, f, indent=4)
print(f'  saved: {OUT}')
