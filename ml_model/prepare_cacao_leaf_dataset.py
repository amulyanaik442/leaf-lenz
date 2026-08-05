"""
Prepare the Amini Cocoa Contamination dataset (Kaggle) for Leaf Lenz integration.

Converts the YOLO-style object-detection annotations into an image-classification
dataset by cropping every bounding box. Only the two requested LEAF classes are
kept:
    healthy      -> Cacao___healthy
    anthracnose  -> Cacao___anthracnose
(cssvd is deliberately dropped; the user approved healthy + anthracnose only.)

Outputs:
    dataset/cacao_leaf/Raw_Classification/{healthy,anthracnose}/<img>__<i>_<cls>.jpg
    dataset/cacao_leaf/cacao_leaf_manifest.json   (crop -> source image mapping)
    dataset/cacao_leaf/cacao_leaf_prep_report.json (logs + stats)

The manifest preserves the source image per crop so downstream splits can be done
at the IMAGE level (no leakage between train/valid/test).

Usage:
    python ml_model/prepare_cacao_leaf_dataset.py
"""
import os
import sys
import json
import csv
import io
import zipfile
from collections import Counter, defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(BASE_DIR, 'dataset', 'cacao_leaf')
RAW_CLS_DIR = os.path.join(OUT_ROOT, 'Raw_Classification')
MANIFEST_PATH = os.path.join(OUT_ROOT, 'cacao_leaf_manifest.json')
REPORT_PATH = os.path.join(OUT_ROOT, 'cacao_leaf_prep_report.json')

ZIP_PATH = (r'C:\Users\amuly\AppData\Local\Temp\opencode\amini_cocoa'
            r'\amini-cocoa-contamination-dataset.zip')
TRAIN_CSV = (r'C:\Users\amuly\AppData\Local\Temp\opencode\amini_cocoa\Train.csv')

KEEP_CLASSES = ('healthy', 'anthracnose')
MIN_SIDE = 64
SEED = 42
QUALITY = 95

LOG = []


def log(msg):
    LOG.append(str(msg))
    print(msg, flush=True)


def main():
    os.makedirs(RAW_CLS_DIR, exist_ok=True)
    for c in KEEP_CLASSES:
        os.makedirs(os.path.join(RAW_CLS_DIR, c), exist_ok=True)

    # ---- 1. Load annotations ----
    log('[1] Loading Train.csv annotations')
    rows = list(csv.DictReader(open(TRAIN_CSV, encoding='utf-8')))
    log(f'    total annotations: {len(rows)}')

    by_class = Counter(r['class'] for r in rows)
    log(f'    raw class counts: {dict(by_class)}')

    kept = [r for r in rows if r['class'] in KEEP_CLASSES]
    dropped = [r for r in rows if r['class'] not in KEEP_CLASSES]
    log(f'    kept (healthy+anthracnose): {len(kept)} | dropped (cssvd): {len(dropped)}')

    # group by source image
    ann_by_image = defaultdict(list)
    for r in kept:
        ann_by_image[r['Image_ID']].append(r)
    log(f'    unique source images to crop: {len(ann_by_image)}')

    # ---- 2. Crop from zip ----
    log('[2] Cropping bounding boxes from train images (zip streaming)')
    manifest = []
    skipped = {'below_min_side': 0, 'unreadable': 0, 'invalid_box': 0}
    per_class_out = Counter()
    per_class_skipped = Counter()

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for img_id, anns in ann_by_image.items():
            entry = 'dataset/images/train/' + img_id
            try:
                raw = zf.read(entry)
            except KeyError:
                skipped['unreadable'] += 1
                log(f'    WARNING: {entry} not in zip')
                continue
            try:
                im = Image.open(io.BytesIO(raw)).convert('RGB')
            except Exception:
                skipped['unreadable'] += 1
                per_class_skipped['unreadable'] += 1
                continue
            W, H = im.size
            for i, a in enumerate(anns):
                cls = a['class']
                x0 = max(0, int(float(a['xmin'])))
                y0 = max(0, int(float(a['ymin'])))
                x1 = min(W, int(float(a['xmax'])))
                y1 = min(H, int(float(a['ymax'])))
                if x1 <= x0 or y1 <= y0:
                    skipped['invalid_box'] += 1
                    continue
                cw, ch = x1 - x0, y1 - y0
                if min(cw, ch) < MIN_SIDE:
                    skipped['below_min_side'] += 1
                    per_class_skipped[cls] = per_class_skipped.get(cls, 0) + 1
                    continue
                crop = im.crop((x0, y0, x1, y1))
                stem = os.path.splitext(img_id)[0]
                out_name = f'{stem}__{i}_{cls}.jpg'
                out_path = os.path.join(RAW_CLS_DIR, cls, out_name)
                crop.save(out_path, 'JPEG', quality=QUALITY)
                per_class_out[cls] += 1
                manifest.append({
                    'crop_path': os.path.join('Raw_Classification', cls, out_name),
                    'source_image': img_id,
                    'class': cls,
                    'box': [x0, y0, x1, y1],
                    'crop_size': [cw, ch],
                })

    log(f'    crops written: {dict(per_class_out)}')
    log(f'    skipped: {skipped}')
    log(f'    skipped by class: {dict(per_class_skipped)}')

    # ---- 3. Save manifest + report ----
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=1)
    report = {
        'source': 'Amini Cocoa Contamination Dataset (Kaggle) / Zindi Amini Cocoa '
                  'Contamination Challenge, CC BY 4.0',
        'classes_kept': list(KEEP_CLASSES),
        'classes_dropped': ['cssvd'],
        'min_crop_side_px': MIN_SIDE,
        'total_annotations': len(rows),
        'kept_annotations': len(kept),
        'dropped_annotations': len(dropped),
        'unique_source_images': len(ann_by_image),
        'crops_written': dict(per_class_out),
        'skipped': skipped,
        'skipped_by_class': dict(per_class_skipped),
        'manifest': MANIFEST_PATH,
        'log': LOG,
    }
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\nDone. Report: {REPORT_PATH}')
    log(f'Total crops: {sum(per_class_out.values())}')


if __name__ == '__main__':
    main()
