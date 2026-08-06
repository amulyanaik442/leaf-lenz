"""
Train the papaya leaf classifier head with a strict zero-regression approach.

Approach (approved, mirrors bean/cauliflower): freeze all 141 existing
classifier rows, train ONLY the 5 new Papaya rows (indices 141-145) on the
prepared Papaya crops, then splice the trained rows into a candidate ONNX.
Existing rows remain byte-identical.

Pipeline:
   1. Prepared splits: `dataset/papaya/Papaya_Split` (train/valid/test,
      file-level stratified, dedup prefer Curl, balanced, train-augmented,
      80/10/10 - approved).
   2. Extract 1280-d features from the frozen production ONNX backbone.
   3. Train a standalone Linear(1280 -> 5) head.
   4. Append the trained rows to a candidate ONNX (146 classes) + candidate
      class names. Production model and class_names.json are NOT modified.

Verification:
   * zero_regression_check.json  - byte-identical non-papaya rows.
   * inference regression spot check on existing non-papaya crops (logits for
     the first 141 classes must be identical; report any papaya FPs).

Outputs (`ml_model/papaya_retrain_output/`):
   * papaya_candidate.onnx + candidate_class_names.json
   * best_model.pth + checkpoints/epoch_NN.pth
   * training_history.json, training_curves.png
   * confusion_matrix (.png/.npy/.json), classification_report (.txt/.json)
   * final_accuracies.json, zero_regression_check.json

Usage:
    python ml_model/train_papaya.py [--tta] [--seed N]
"""
import os
import sys
import json
import copy
import pickle
import argparse

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import onnx
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ASSETS = os.path.join(BASE_DIR, 'detector', 'ml_assets')
PRODUCTION_ONNX = os.path.join(ML_ASSETS, 'model.onnx')
CLASSES_PATH = os.path.join(ML_ASSETS, 'class_names.json')

SPLIT_DIR = os.path.join(BASE_DIR, 'dataset', 'papaya', 'Papaya_Split')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'papaya_retrain_output')
CKPT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
FEAT_CACHE_TTA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'papaya_features_tta.pkl')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PAPAYA_CLASS_NAMES = [
    'papaya___anthracnose',
    'papaya___bacterial_spot',
    'papaya___curl',
    'papaya___healthy',
    'papaya___ring_spot',
]
N_NEW = len(PAPAYA_CLASS_NAMES)

HEAD_EPOCHS = 60
LR = 1e-3
BATCH_SIZE = 32
SEED = 42
MAX_PATIENCE = 12
TARGET_SIZE, CROP_SIZE = 256, 224
NUM_WORKERS = 0
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP')
FEAT_WINDOW = 16   # images processed per window (bounds peak memory)
FEAT_BATCH = 16    # ONNX batch size during feature extraction

# ---------------------------------------------------------------------------
# Transforms (same as the tea/cacao/chilli/turmeric/black pepper/bean pipelines)
# ---------------------------------------------------------------------------
tform = transforms.Compose([
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_aug = transforms.Compose([
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.RandomCrop(CROP_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Feature extraction from production ONNX backbone
# ---------------------------------------------------------------------------
def create_feature_extractor_session(onnx_path):
    model = onnx.load(onnx_path)
    classifier_input = None
    for node in model.graph.node:
        if node.op_type in ('Gemm', 'MatMul'):
            for out in model.graph.output:
                if out.name in node.output:
                    classifier_input = node.input[0]
                    break
        if classifier_input:
            break
    if classifier_input is None:
        output_name = model.graph.output[0].name
        for node in reversed(model.graph.node):
            if output_name in node.output:
                classifier_input = node.input[0]
                break

    feat_model = copy.deepcopy(model)
    feat_model.graph.output.append(onnx.helper.make_tensor_value_info(
        classifier_input, onnx.TensorProto.FLOAT, [None, 1280]))
    feat_path = onnx_path.replace('.onnx', '_features_tmp.onnx')
    onnx.save(feat_model, feat_path)
    sess = ort.InferenceSession(feat_path)
    print(f'  Feature extractor ready (input {classifier_input})')
    return sess, sess.get_inputs()[0].name, feat_path


# ---------------------------------------------------------------------------
# TTA-aligned feature extraction (matches production 4-view augmentation)
# ---------------------------------------------------------------------------
def _flip_views(t):
    return [t, t[:, :, ::-1].copy(), t[:, ::-1, :].copy(),
            t[:, ::-1, ::-1].copy()]


def extract_tta_split(feat_sess, feat_input, split_paths, tag=''):
    paths = []
    labels = []
    for cls in PAPAYA_CLASS_NAMES:
        d = os.path.join(split_paths, cls)
        if not os.path.isdir(d):
            print(f'  WARNING: {d} not found')
            continue
        p = sorted([os.path.join(d, f) for f in os.listdir(d)
                    if f.lower().endswith(IMAGE_EXTS)])
        paths.extend(p)
        labels.extend([PAPAYA_CLASS_NAMES.index(cls)] * len(p))
        print(f'  {tag} {cls:30s} {len(p)} images', flush=True)
    if not paths:
        return np.zeros((4, 0, 1280), dtype=np.float32), \
            np.zeros((0,), dtype=np.int64)
    kept = []
    skipped = 0
    for p, lab in zip(paths, labels):
        try:
            Image.open(p)
        except Exception as e:
            skipped += 1
            print(f'  WARNING: skip unreadable {p}: {e}', flush=True)
            continue
        kept.append((p, lab))
    if skipped:
        print(f'  WARNING: {tag} skipped {skipped} unreadable images', flush=True)
    n_kept = len(kept)
    views = np.zeros((4, n_kept, 1280), dtype=np.float32)
    labels_out = np.array([lab for _p, lab in kept], dtype=np.int64)
    zero_t = np.zeros((3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    for w in range(0, n_kept, FEAT_WINDOW):
        window = kept[w:w + FEAT_WINDOW]
        m = len(window)
        per_view = [[] for _ in range(4)]
        for p, _lab in window:
            try:
                t = tform(Image.open(p).convert('RGB')).numpy()
            except Exception:
                t = zero_t
            for v, tv in enumerate(_flip_views(t)):
                per_view[v].append(tv)
        for v in range(4):
            arrs = np.stack(per_view[v]).astype(np.float32)
            for i in range(0, m, FEAT_BATCH):
                views[v, w + i:w + i + FEAT_BATCH] = \
                    feat_sess.run(None, {feat_input: arrs[i:i + FEAT_BATCH]})[-1]
            del arrs
        del per_view
    return views, labels_out


def extract_tta_train(feat_sess, feat_input, split_paths, double=True):
    # Pre-allocate the full feature matrix to bound memory (no list + vstack peak).
    per_class = []
    n_total = 0
    for cls in PAPAYA_CLASS_NAMES:
        d = os.path.join(split_paths, cls)
        if not os.path.isdir(d):
            continue
        paths = sorted([os.path.join(d, f) for f in os.listdir(d)
                        if f.lower().endswith(IMAGE_EXTS)])
        vpi = 6 if double else 4
        per_class.append((cls, paths, vpi))
        n_total += len(paths) * vpi
    train_feats = np.zeros((n_total, 1280), dtype=np.float32)
    labels = np.zeros((n_total,), dtype=np.int64)
    cursor = 0
    for cls, paths, vpi in per_class:
        label = PAPAYA_CLASS_NAMES.index(cls)
        skipped = 0
        for w in range(0, len(paths), FEAT_WINDOW):
            win = paths[w:w + FEAT_WINDOW]
            views = []
            for p in win:
                try:
                    t = tform(Image.open(p).convert('RGB')).numpy()
                    v = list(_flip_views(t))
                    if double:
                        aug_im = Image.open(p).convert('RGB')
                        v.extend([train_aug(aug_im).numpy(),
                                  train_aug(aug_im).numpy()])
                except Exception as e:
                    skipped += 1
                    print(f'  WARNING: skip unreadable {p}: {e}', flush=True)
                    continue
                views.extend(v)
            if not views:
                continue
            labels[cursor:cursor + len(views)] = label
            for i in range(0, len(views), FEAT_BATCH):
                arr = np.stack(views[i:i + FEAT_BATCH]).astype(np.float32)
                out = feat_sess.run(None, {feat_input: arr})[-1]
                train_feats[cursor + i:cursor + i + out.shape[0]] = out
            cursor += len(views)
            del views
        if skipped:
            print(f'  WARNING: train {cls} skipped {skipped} unreadable images', flush=True)
        n_views = vpi * len(paths) - skipped * vpi if skipped else vpi * len(paths)
        print(f'  train {cls:30s} {len(paths)} imgs x{vpi} views = {n_views}', flush=True)
    return train_feats[:cursor], labels[:cursor]


def evaluate_tta(head, feat_views, labels, device):
    head.eval()
    probs = None
    with torch.no_grad():
        for v in range(feat_views.shape[0]):
            feats = torch.tensor(feat_views[v], dtype=torch.float32).to(device)
            p = torch.softmax(head(feats), dim=1).cpu().numpy()
            probs = p if probs is None else probs + p
    probs /= feat_views.shape[0]
    preds = probs.argmax(1)
    return 100.0 * float((preds == labels).sum()) / max(1, len(labels)), probs


def confusion_matrix_tta(head, feat_views, labels, device):
    _, probs = evaluate_tta(head, feat_views, labels, device)
    cm = np.zeros((N_NEW, N_NEW), dtype=int)
    for t, p in zip(labels, probs.argmax(1)):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# Training + evaluation helpers
# ---------------------------------------------------------------------------
def evaluate(head, loader, criterion, device):
    head.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = head(feats)
            loss += criterion(outputs, labels).item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return 100.0 * correct / total, loss / max(1, len(loader))


def metrics_from_cm(cm):
    rows = {}
    for i, cls in enumerate(PAPAYA_CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows[cls] = {'precision': round(prec, 4), 'recall': round(rec, 4),
                     'f1': round(f1, 4), 'support': int(tp + fn)}
    acc = cm.diagonal().sum() / cm.sum() if cm.sum() else 0.0
    return rows, acc


def save_artifacts(history, cm, test_acc, train_acc, best_val_acc, device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    short = [c.split('___')[1] for c in PAPAYA_CLASS_NAMES]

    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        epochs = list(range(1, len(history['train_acc']) + 1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, history['train_acc'], label='Train')
        axes[0].plot(epochs, history['val_acc'], label='Valid')
        axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
        axes[0].legend(); axes[0].grid(True)
        axes[1].plot(epochs, history['train_loss'], label='Train')
        axes[1].plot(epochs, history['val_loss'], label='Valid')
        axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
        axes[1].legend(); axes[1].grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(N_NEW), short, rotation=45, ha='right')
        ax.set_yticks(range(N_NEW), short)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Papaya Test Confusion Matrix (acc={100*test_acc:.1f}%)')
        for i in range(N_NEW):
            for j in range(N_NEW):
                ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                        color='white' if cm[i][j] > cm.max() / 2 else 'black')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f'  Warning: could not save plots: {e}')

    np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'), cm)
    with open(os.path.join(OUTPUT_DIR, 'confusion_matrix.json'), 'w') as f:
        json.dump({'classes': PAPAYA_CLASS_NAMES, 'confusion_matrix': cm.tolist()},
                  f, indent=4)

    rows, _ = metrics_from_cm(cm)
    lines = ['papaya Classification Report (internal test set)', '=' * 60]
    lines.append(f'{"class":30s} {"precision":>10s} {"recall":>10s} {"f1":>10s} {"support":>8s}')
    for cls in PAPAYA_CLASS_NAMES:
        r = rows[cls]
        lines.append(f'{cls:30s} {r["precision"]:10.4f} {r["recall"]:10.4f} {r["f1"]:10.4f} {r["support"]:8d}')
    lines.append('-' * 60)
    total_support = sum(r['support'] for r in rows.values())
    lines.append(f'{"accuracy":30s} {100*test_acc:10.2f}% ({int(round(test_acc*total_support))}/{total_support})')
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(OUTPUT_DIR, 'classification_report.json'), 'w') as f:
        json.dump({'rows': rows, 'accuracy': round(float(test_acc), 4)}, f, indent=4)

    with open(os.path.join(OUTPUT_DIR, 'final_accuracies.json'), 'w') as f:
        json.dump({'train_accuracy': round(float(train_acc), 4),
                   'best_val_accuracy': round(float(best_val_acc), 4),
                   'test_accuracy': round(float(test_acc), 4),
                   'device': str(device)}, f, indent=4)

    print(f'\n  Artifacts saved to {OUTPUT_DIR}')


# ---------------------------------------------------------------------------
# Inference zero-regression spot check on existing non-papaya crops
# ---------------------------------------------------------------------------
def collect_existing_crops(max_per_folder=40):
    roots = [
        os.path.join(BASE_DIR, 'dataset', 'bean', 'Bean_Split', 'test'),
        os.path.join(BASE_DIR, 'dataset', 'pumpkin', 'Corrected_Split', 'test'),
        os.path.join(BASE_DIR, 'dataset', 'tea', 'Tea_Split', 'test'),
        os.path.join(BASE_DIR, 'dataset', 'cacao_leaf', 'Cacao_Split', 'test'),
        os.path.join(BASE_DIR, 'dataset', 'chilli', 'Chilli_Split', 'test'),
        os.path.join(BASE_DIR, 'dataset', 'turmeric', 'Turmeric_Split', 'test'),
        os.path.join(BASE_DIR, 'dataset', 'black_pepper', 'BlackPepper_Split', 'test'),
        os.path.join(BASE_DIR, 'dataset', 'ginger', 'Ginger_Split', 'test'),
        os.path.join(BASE_DIR, 'dataset', 'cauliflower', 'Cauliflower_Split', 'test'),
    ]
    paths = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for folder in sorted(os.listdir(root)):
            d = os.path.join(root, folder)
            if not os.path.isdir(d):
                continue
            imgs = sorted([os.path.join(d, f) for f in os.listdir(d)
                           if f.lower().endswith(IMAGE_EXTS)])[:max_per_folder]
            paths.extend(imgs)
    return paths


def inference_zero_regression_check(candidate_path, n_existing=141,
                                    papaya_range=(141, 146)):
    print('\n' + '=' * 60)
    print('INFERENCE ZERO-REGRESSION SPOT CHECK (existing non-papaya crops)')
    print('=' * 60)
    prod = ort.InferenceSession(PRODUCTION_ONNX, providers=['CPUExecutionProvider'])
    cand = ort.InferenceSession(candidate_path, providers=['CPUExecutionProvider'])
    inp = prod.get_inputs()[0].name
    size = (prod.get_inputs()[0].shape[2], prod.get_inputs()[0].shape[3])

    paths = collect_existing_crops()
    if not paths:
        print('  No existing crops found; skipping inference spot check.')
        return None
    print(f'  Existing crops sampled: {len(paths)}')

    protected = [i for i in range(cand.get_outputs()[0].shape[1])
                 if not (papaya_range[0] <= i < papaya_range[1])]
    max_logit_diff = 0.0
    argmax_old_diff = 0
    papaya_fp = 0
    res = transforms.Resize(size)
    to_t = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for p in paths:
        t = normalize(to_t(res(Image.open(p).convert('RGB')))).numpy()
        arr = np.expand_dims(t, 0).astype(np.float32)
        lp = prod.run(None, {inp: arr})[0][0]
        lc = cand.run(None, {inp: arr})[0][0]
        diff = float(np.max(np.abs(lp[protected] - lc[protected])))
        max_logit_diff = max(max_logit_diff, diff)
        if np.argmax(lp[protected]) != np.argmax(lc[protected]):
            argmax_old_diff += 1
        if papaya_range[0] <= np.argmax(lc) < papaya_range[1]:
            papaya_fp += 1

    print(f'  max |logit diff| over protected (non-papaya) classes: {max_logit_diff:.2e}')
    print(f'  crops whose top-1 among protected classes changed: {argmax_old_diff}/{len(paths)}')
    print(f'  crops the candidate now labels papaya (FP risk): {papaya_fp}/{len(paths)}')
    result = {
        'samples': len(paths),
        'max_logit_diff_protected': max_logit_diff,
        'protected_index_count': len(protected),
        'top1_old_class_changed': argmax_old_diff,
        'papaya_false_positives': papaya_fp,
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=HEAD_EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--patience', type=int, default=MAX_PATIENCE)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--tta', action='store_true',
                        help='TTA-aligned training: flip-view train features and '
                             '4-view averaged-softmax val/test eval (matches production)')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--seeds', type=str, default=None,
                        help='comma-separated seeds to train and average '
                             '(linear head weight-space ensemble; overrides --seed)')
    parser.add_argument('--split-dir', default='Papaya_Split',
                        help='split folder under dataset/papaya/ (default: Papaya_Split)')
    parser.add_argument('--prod-model', default='',
                        help='reference frozen ONNX for backbone + zero-regression '
                             '(default: detector/ml_assets/model.onnx)')
    parser.add_argument('--prod-classes', default='',
                        help='reference class_names.json (default: '
                             'detector/ml_assets/class_names.json)')
    parser.add_argument('--no-export', action='store_true')
    args = parser.parse_args()

    global PRODUCTION_ONNX, CLASSES_PATH, SPLIT_DIR, FEAT_CACHE_TTA
    if args.prod_model:
        PRODUCTION_ONNX = args.prod_model
    if args.prod_classes:
        CLASSES_PATH = args.prod_classes
    if args.split_dir != 'Papaya_Split':
        SPLIT_DIR = os.path.join(BASE_DIR, 'dataset', 'papaya', args.split_dir)
    split_tag = os.path.basename(SPLIT_DIR)
    FEAT_CACHE_TTA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f'papaya_features_{split_tag}.pkl')

    seed_list = [int(s) for s in args.seeds.split(',')] if args.seeds else [args.seed]
    ensemble = len(seed_list) > 1

    np.random.seed(seed_list[0])
    torch.manual_seed(seed_list[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('\n' + '=' * 60)
    print('TRAINING CONFIGURATION')
    print('=' * 60)
    print(f'  Backbone:      frozen EfficientNet-B0 (1280-d), from production ONNX')
    print(f'  Head:          Linear(1280 -> {N_NEW}) papaya-only, zero-regression')
    print(f'  Loss:          CrossEntropyLoss(label_smoothing=0.1)')
    print(f'  Optimizer:     Adam lr={LR}')
    print(f'  Scheduler:     ReduceLROnPlateau(mode=max, factor=0.5, patience=5)')
    print(f'  Early stop:    patience={MAX_PATIENCE}, max {HEAD_EPOCHS} epochs')
    print(f'  Batch size:    {BATCH_SIZE}')
    print(f'  Train aug:     Resize(256)->RandomCrop(224), hflip, vflip, rot10')
    print(f'                 + feature doubling (2 views/train image)')
    print(f'  Eval aug:      Resize(224)')
    print(f'  Checkpoints:   every epoch -> {CKPT_DIR}')
    print(f'  New classes:   indices 141-{140 + N_NEW}: {PAPAYA_CLASS_NAMES}')
    print('=' * 60)

    with open(CLASSES_PATH) as f:
        existing_classes = json.load(f)
    n_existing = len(existing_classes)
    first_papaya = next((i for i, c in enumerate(existing_classes)
                         if c.startswith('papaya___')), None)
    if first_papaya is None:
        # Append mode: reference model has no papaya rows yet.
        append_mode = True
        assert n_existing == 141, \
            f'Expected 141 (pre-papaya) classes for append, got {n_existing}'
        candidate_classes = existing_classes + PAPAYA_CLASS_NAMES
        papaya_start = n_existing  # 141
    else:
        # Replace mode: reference model already contains papaya rows.
        append_mode = False
        assert existing_classes[first_papaya:first_papaya + N_NEW] == PAPAYA_CLASS_NAMES, \
            'papaya row order mismatch in reference model'
        papaya_start = first_papaya
        candidate_classes = existing_classes
        print(f'  REPLACE MODE: papaya rows already at {papaya_start}-{papaya_start + N_NEW - 1}; '
              f'these rows will be replaced, all other rows stay byte-identical')
    papaya_end = papaya_start + N_NEW
    print(f'\n  Existing classes: {n_existing} | papaya range: {papaya_start}-{papaya_end - 1} | Candidate total: {len(candidate_classes)}')

    # ---- Features ----
    print('\n' + '=' * 60)
    print('FEATURE EXTRACTION (TTA)')
    print('=' * 60)
    feat_sess, feat_input, tmp_path = create_feature_extractor_session(PRODUCTION_ONNX)
    feat_cache_base = FEAT_CACHE_TTA.replace('.pkl', f'_{args.seed}')

    def load_or_extract(name, extract_fn):
        path = f'{feat_cache_base}_{name}.pkl'
        if os.path.exists(path):
            print(f'  Loading cached {name} features from {path}')
            with open(path, 'rb') as f:
                return pickle.load(f)
        data = extract_fn()
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f'  Cached {name} features to {path}')
        return data

    train_feats, train_labels = load_or_extract(
        'train',
        lambda: extract_tta_train(feat_sess, feat_input,
                                  os.path.join(SPLIT_DIR, 'train')))
    val_views, val_labels = load_or_extract(
        'valid',
        lambda: extract_tta_split(feat_sess, feat_input,
                                  os.path.join(SPLIT_DIR, 'valid'), tag='valid'))
    test_views, test_labels = load_or_extract(
        'test',
        lambda: extract_tta_split(feat_sess, feat_input,
                                  os.path.join(SPLIT_DIR, 'test'), tag='test'))
    print(f'  train: {train_feats.shape[0]}, valid views: {val_views.shape}, test views: {test_views.shape}')
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    del feat_sess
    import gc
    gc.collect()
    print('  Feature extractor session released')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Train (optionally weight-space ensemble across seeds) ----
    print('\n' + '=' * 60)
    print(f'TRAINING ({HEAD_EPOCHS} epochs max) seeds={seed_list} '
          f'({"ensemble" if ensemble else "single"})')
    print('=' * 60)
    train_ds = FeatureDataset(train_feats, train_labels)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    os.makedirs(CKPT_DIR, exist_ok=True)
    avg_state = None
    best_val_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    train_acc = 0.0
    per_seed = []
    for si, seed in enumerate(seed_list):
        np.random.seed(seed)
        torch.manual_seed(seed)
        head = nn.Linear(1280, N_NEW).to(device)
        optimizer = optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5)
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS)
        seed_best = 0.0
        seed_state = None
        patience_counter = 0
        for epoch in range(args.epochs):
            head.train()
            running_loss, correct, total = 0.0, 0, 0
            for feats, labels in loader:
                feats, labels = feats.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = head(feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
            tr_acc = 100.0 * correct / total
            tr_loss = running_loss / max(1, len(loader))

            val_acc, _ = evaluate_tta(head, val_views, val_labels, device)
            scheduler.step(val_acc)

            history['train_acc'].append(round(tr_acc, 4))
            history['val_acc'].append(round(val_acc, 4))
            history['train_loss'].append(round(tr_loss, 4))
            history['val_loss'].append(round(0.0, 4))

            torch.save(head.state_dict(),
                       os.path.join(CKPT_DIR, f'seed{seed}_epoch_{epoch + 1:02d}.pth'))

            if val_acc > seed_best:
                seed_best = val_acc
                seed_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Seed {seed} Epoch [{epoch+1:2d}/{args.epochs}] '
                      f'Train Acc: {tr_acc:.2f}% | Val Acc: {val_acc:.2f}% | '
                      f'LR: {optimizer.param_groups[0]["lr"]:.1e}')

            if patience_counter >= args.patience:
                print(f'  Seed {seed}: early stopping at epoch {epoch + 1}')
                break

        if seed_state is None:
            print(f'  WARNING: seed {seed} produced no best state')
            continue
        per_seed.append({'seed': seed, 'best_val_acc': seed_best,
                         'epochs': epoch + 1})
        if avg_state is None:
            avg_state = {k: v.clone() for k, v in seed_state.items()}
        else:
            for k in avg_state:
                avg_state[k] = avg_state[k] + seed_state[k]
        best_val_acc = max(best_val_acc, seed_best)
        train_acc = tr_acc
        print(f'  Seed {seed}: best val {seed_best:.2f}% '
              f'| running ensemble avg over {len(per_seed)} seed(s)')

    for k in avg_state:
        avg_state[k] = avg_state[k] / len(per_seed)
    head = nn.Linear(1280, N_NEW)
    head.load_state_dict(avg_state)
    head = head.to(device)
    print(f'\nBest validation accuracy '
          f'({"ensemble avg" if ensemble else "single"}): {best_val_acc:.2f}%')

    with torch.no_grad():
        bias_mean = head.bias.mean()
        head.bias.sub_(bias_mean)
    print(f'  Centered papaya biases: mean bias {bias_mean:.6f} subtracted')

    torch.save(avg_state, os.path.join(OUTPUT_DIR, 'best_model.pth'))

    # ---- Test evaluation ----
    print('\n' + '=' * 60)
    print('INTERNAL TEST SET EVALUATION')
    print('=' * 60)
    test_acc, _ = evaluate_tta(head, test_views, test_labels, device)
    cm = confusion_matrix_tta(head, test_views, test_labels, device)
    sv_acc, _ = evaluate_tta(head, test_views[:1], test_labels, device)
    print(f'  Test accuracy (4-view TTA): {test_acc:.2f}%')
    print(f'  Test accuracy (single view): {sv_acc:.2f}%')
    short = [c.split('___')[1][:12] for c in PAPAYA_CLASS_NAMES]
    print('  ' + ' '*15 + ''.join(f'{s:>13s}' for s in short))
    for i in range(N_NEW):
        print(f'  {short[i]:14s}' + ''.join(f'{cm[i][j]:6d}' for j in range(N_NEW)))
    rows, _ = metrics_from_cm(cm)
    for cls in PAPAYA_CLASS_NAMES:
        r = rows[cls]
        print(f'  {cls:30s} prec={r["precision"]:.3f} rec={r["recall"]:.3f} f1={r["f1"]:.3f}')

    save_artifacts(history, cm, test_acc / 100.0, train_acc, best_val_acc, device)
    with open(os.path.join(OUTPUT_DIR, 'final_accuracies.json')) as _f:
        _fa = json.load(_f)
    _fa['seeds'] = seed_list
    _fa['ensemble'] = ensemble
    _fa['split'] = split_tag
    _fa['replace_mode'] = not append_mode
    with open(os.path.join(OUTPUT_DIR, 'final_accuracies.json'), 'w') as _f:
        json.dump(_fa, _f, indent=4)

    # ---- Candidate ONNX (replace or append papaya rows, zero regression) ----
    candidate_path = None
    w_diff = b_diff = None
    ok = None
    inference_result = None
    if not args.no_export:
        print('\n' + '=' * 60)
        print('CANDIDATE ONNX '
              f'({"papaya rows replaced" if not append_mode else "papaya rows appended"}, '
              'other rows frozen)')
        print('=' * 60)
        model = onnx.load(PRODUCTION_ONNX)
        weight_init = bias_init = None
        for init in model.graph.initializer:
            if len(init.dims) == 2 and init.dims[1] == 1280:
                if weight_init is None or init.dims[0] > weight_init.dims[0]:
                    weight_init = init
        if weight_init is None:
            print('ERROR: classifier weights not found')
            return
        classifier_rows = list(weight_init.dims)[0]
        for init in model.graph.initializer:
            if len(init.dims) == 1 and init.dims[0] == classifier_rows:
                bias_init = init
                break

        old_weight = np.frombuffer(weight_init.raw_data, dtype=np.float32) \
            .reshape(list(weight_init.dims))
        old_bias = np.frombuffer(bias_init.raw_data, dtype=np.float32)

        trained_weight = head.weight.detach().cpu().numpy()
        trained_bias = head.bias.detach().cpu().numpy()

        if append_mode:
            assert old_weight.shape[0] == papaya_start, \
                f'Expected {papaya_start} deployed rows, got {old_weight.shape[0]}'
            new_weight = np.vstack([old_weight, trained_weight])
            new_bias = np.concatenate([old_bias, trained_bias])
        else:
            assert old_weight.shape[0] == len(candidate_classes), \
                f'Expected {len(candidate_classes)} deployed rows (replace mode), got {old_weight.shape[0]}'
            new_weight = old_weight.copy()
            new_bias = old_bias.copy()
            new_weight[papaya_start:papaya_end] = trained_weight
            new_bias[papaya_start:papaya_end] = trained_bias

        weight_init.raw_data = new_weight.astype(np.float32).tobytes()
        weight_init.dims[:] = new_weight.shape
        bias_init.raw_data = new_bias.astype(np.float32).tobytes()
        bias_init.dims[0] = new_bias.shape[0]

        for out in model.graph.output:
            if out.type.tensor_type.HasField('shape'):
                dim = out.type.tensor_type.shape.dim
                if len(dim) > 1 and dim[1].dim_value == classifier_rows:
                    dim[1].dim_value = new_weight.shape[0]
                    print(f'  Updated output shape: {dim[1].dim_value}')

        try:
            onnx.checker.check_model(model)
        except Exception as e:
            print(f'  Validation warning: {e}')

        candidate_path = os.path.join(OUTPUT_DIR, 'papaya_candidate.onnx')
        onnx.save(model, candidate_path)
        print(f'  Saved candidate: {candidate_path}')

        with open(os.path.join(OUTPUT_DIR, 'candidate_class_names.json'), 'w') as f:
            json.dump(candidate_classes, f, indent=4)
        print(f'  Saved candidate class names ({len(candidate_classes)} classes)')

        # Zero-regression verification (non-papaya rows byte-identical)
        v1 = onnx.load(PRODUCTION_ONNX)
        v1_w = v1_b = None
        for init in v1.graph.initializer:
            dims = list(init.dims)
            if dims == [classifier_rows, 1280]:
                v1_w = np.frombuffer(init.raw_data, dtype=np.float32).reshape(dims)
            if dims == [classifier_rows]:
                v1_b = np.frombuffer(init.raw_data, dtype=np.float32)
        non_papaya = [i for i in range(new_weight.shape[0])
                      if not (papaya_start <= i < papaya_end)]
        w_diff = float(np.max(np.abs(new_weight[non_papaya] - v1_w[non_papaya])))
        b_diff = float(np.max(np.abs(new_bias[non_papaya] - v1_b[non_papaya])))
        ok = w_diff == 0.0 and b_diff == 0.0
        print(f'  Non-papaya weight max diff: {w_diff:.10f}')
        print(f'  Non-papaya bias max diff:   {b_diff:.10f}')
        print(f'  ZERO REGRESSION: {"OK" if ok else "FAILED"}')

        inference_result = inference_zero_regression_check(
            candidate_path, n_existing=papaya_start, papaya_range=(papaya_start, papaya_end))

        with open(os.path.join(OUTPUT_DIR, 'zero_regression_check.json'), 'w') as f:
            json.dump({'non_papaya_weight_max_diff': w_diff,
                       'non_papaya_bias_max_diff': b_diff,
                       'zero_regression_ok': ok,
                       'papaya_start': papaya_start,
                       'replace_mode': not append_mode,
                       'inference_spot_check': inference_result}, f, indent=4)

    print('\n' + '=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f'  Train acc:    {train_acc:.2f}%')
    print(f'  Best val acc: {best_val_acc:.2f}%')
    print(f'  Test acc:     {test_acc:.2f}%')
    print(f'  Candidate:    {candidate_path}')
    print(f'  NOTE: production model was NOT modified')


if __name__ == '__main__':
    main()
