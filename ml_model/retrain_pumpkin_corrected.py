"""
Retrain the Pumpkin head on the corrected, balanced dataset.

Pipeline (identical to `expand_pumpkin_model_v3.py`):
  * Backbone: frozen EfficientNet-B0 feature extractor built from the current
    production ONNX (1280-d features).
  * Head:     single nn.Linear(1280 -> 5).
  * Loss:     CrossEntropyLoss(label_smoothing=0.1).
  * Optimizer: Adam, lr = 1e-3.
  * Scheduler: ReduceLROnPlateau(mode='max', factor=0.5, patience=5).
  * Early stopping: patience = 10 epochs, max 50 epochs.
  * Bias centering after training (subtract mean bias).
  * The trained Pumpkin classifier rows overwrite the ONNX graph in place.

Split (project convention 80/10/10 -> 320/40/40 per class):
  * Uses the balanced `Corrected_Dataset` (400 images per class).

Outputs (saved to `ml_model/pumpkin_retrain_output/`):
  * trained model (ONNX candidate + production overwrite with backup)
  * best model weights (.pth)
  * training history (.json)
  * accuracy & loss graphs (.png)
  * confusion matrix (.png + .json)
  * classification report (.txt + .json)
  * final train/valid/test accuracies (.json)

Usage:
    python ml_model/retrain_pumpkin_corrected.py
"""
import os
import sys
import json
import random
import shutil
import copy
import pickle
from collections import defaultdict

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
CANDIDATE_ONNX = os.path.join(ML_ASSETS, 'pumpkin_corrected_candidate.onnx')
BACKUP_ONNX = os.path.join(ML_ASSETS, 'model.onnx.bak_pre_corrected')
CLASSES_PATH = os.path.join(ML_ASSETS, 'class_names.json')

BALANCED_DIR = os.path.join(BASE_DIR, 'dataset', 'pumpkin', 'Corrected_Dataset')
SPLIT_DIR = os.path.join(BASE_DIR, 'dataset', 'pumpkin', 'Corrected_Split')
FEAT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'pumpkin_corrected_features.pkl')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'pumpkin_retrain_output')

# ---------------------------------------------------------------------------
# Hyperparameters (identical to expand_pumpkin_model_v3.py)
# ---------------------------------------------------------------------------
HEAD_EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 32
SEED = 42
MAX_PATIENCE = 10
TARGET_SIZE, CROP_SIZE = 256, 224
NUM_WORKERS = 0
TRAIN_RATIO = 0.80            # project 80/10/10 split convention
CLASS_NAMES = ['Bacterial Leaf Spot', 'Downy Mildew', 'Healthy Leaf',
               'Mosaic Disease', 'Powdery Mildew']
PUMPKIN_CLASS_NAMES = sorted([
    'Pumpkin___bacterial_leaf_spot', 'Pumpkin___downy_mildew',
    'Pumpkin___healthy', 'Pumpkin___mosaic_disease', 'Pumpkin___powdery_mildew',
])
FOLDER_TO_MODEL = {
    'Bacterial Leaf Spot': 'Pumpkin___bacterial_leaf_spot',
    'Downy Mildew': 'Pumpkin___downy_mildew',
    'Healthy Leaf': 'Pumpkin___healthy',
    'Mosaic Disease': 'Pumpkin___mosaic_disease',
    'Powdery Mildew': 'Pumpkin___powdery_mildew',
}
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Transforms (same as v3)
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
# Step 1: Split balanced dataset 80/10/10 (stratified, seeded)
# ---------------------------------------------------------------------------
def build_splits():
    print('=' * 60)
    print('STEP 1: Stratified 80/10/10 split of corrected dataset')
    print('=' * 60)
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    rng = random.Random(SEED)
    summary = {}
    for folder in CLASS_NAMES:
        src = os.path.join(BALANCED_DIR, folder)
        files = sorted([f for f in os.listdir(src)
                        if os.path.isfile(os.path.join(src, f))
                        and f.lower().endswith(IMAGE_EXTS)])
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_valid = int(n * 0.10)
        splits = {
            'train': files[:n_train],
            'valid': files[n_train:n_train + n_valid],
            'test': files[n_train + n_valid:],
        }
        for split_name, split_files in splits.items():
            dest = os.path.join(SPLIT_DIR, split_name, folder)
            os.makedirs(dest, exist_ok=True)
            for f in split_files:
                shutil.copy2(os.path.join(src, f), os.path.join(dest, f))
        summary[folder] = {k: len(v) for k, v in splits.items()}
        print(f'  {folder:25s} train={len(splits["train"]):3d} '
              f'valid={len(splits["valid"]):3d} test={len(splits["test"]):3d}')
    return summary


def load_split_paths():
    """Return {split_name: {folder: [image_paths]}}."""
    paths = defaultdict(dict)
    for split_name in ['train', 'valid', 'test']:
        for folder in CLASS_NAMES:
            d = os.path.join(SPLIT_DIR, split_name, folder)
            paths[split_name][folder] = []
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(IMAGE_EXTS):
                        paths[split_name][folder].append(os.path.join(d, f))
    return paths


# ---------------------------------------------------------------------------
# Step 2: Feature extractor from production ONNX
# ---------------------------------------------------------------------------
def create_feature_extractor_session(onnx_path):
    model = onnx.load(onnx_path)
    classifier_input_name = None
    for node in model.graph.node:
        if node.op_type in ('Gemm', 'MatMul'):
            for out in model.graph.output:
                if out.name in node.output:
                    classifier_input_name = node.input[0]
                    break
        if classifier_input_name:
            break
    if classifier_input_name is None:
        output_name = model.graph.output[0].name
        for node in reversed(model.graph.node):
            if output_name in node.output:
                classifier_input_name = node.input[0]
                break

    feat_model = copy.deepcopy(model)
    feat_out = onnx.helper.make_tensor_value_info(
        classifier_input_name, onnx.TensorProto.FLOAT, [None, 1280])
    feat_model.graph.output.append(feat_out)
    feat_path = onnx_path.replace('.onnx', '_features_tmp.onnx')
    onnx.save(feat_model, feat_path)
    sess = ort.InferenceSession(feat_path)
    print(f'  Feature extractor ready (input {classifier_input_name})')
    return sess, sess.get_inputs()[0].name


def extract_split_features(feat_sess, feat_input, paths_by_class, transform,
                           double=False, tag=''):
    """Extract 1280-d features for every image in paths_by_class.

    Args:
        paths_by_class: {folder: [paths]}
        double: if True, extract two augmented views per training image.
    Returns:
        (features np.ndarray Nx1280 float32, labels np.ndarray N int64)
    """
    feats, labels = [], []
    for folder in CLASS_NAMES:
        paths = paths_by_class.get(folder, [])
        if not paths:
            continue
        label = PUMPKIN_CLASS_NAMES.index(FOLDER_TO_MODEL[folder])
        for i in range(0, len(paths), 64):
            chunk = paths[i:i + 64]
            tensors = [transform(Image.open(p).convert('RGB')).numpy()
                       for p in chunk]
            arr = np.stack(tensors).astype(np.float32)
            out = feat_sess.run(None, {feat_input: arr})[-1]
            feats.append(out)
            labels.extend([label] * len(chunk))
            if double:
                tensors2 = [transform(Image.open(p).convert('RGB')).numpy()
                            for p in chunk]
                arr2 = np.stack(tensors2).astype(np.float32)
                out2 = feat_sess.run(None, {feat_input: arr2})[-1]
                feats.append(out2)
                labels.extend([label] * len(chunk))
        print(f'  {tag} {folder:25s} {len(paths)} images', flush=True)
    if not feats:
        return np.zeros((0, 1280), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.vstack(feats), np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Step 3: Training + evaluation helpers
# ---------------------------------------------------------------------------
def evaluate(head, loader, criterion):
    head.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(head.weight.device), labels.to(head.weight.device)
            outputs = head(feats)
            loss += criterion(outputs, labels).item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return 100.0 * correct / total, loss / max(1, len(loader))


def confusion_matrix(head, loader, device):
    head.eval()
    cm = np.zeros((5, 5), dtype=int)
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            _, preds = head(feats).max(1)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1
    return cm


def metrics_from_cm(cm):
    """Per-class precision/recall/F1 from a confusion matrix."""
    rows = {}
    for i, cls in enumerate(PUMPKIN_CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows[cls] = {'precision': round(prec, 4), 'recall': round(rec, 4),
                     'f1': round(f1, 4), 'support': int(tp + fn),
                     'true_positives': int(tp), 'false_positives': int(fp),
                     'false_negatives': int(fn)}
    acc = cm.diagonal().sum() / cm.sum() if cm.sum() else 0.0
    return rows, acc


# ---------------------------------------------------------------------------
# Step 4: Save artifacts
# ---------------------------------------------------------------------------
def save_artifacts(history, cm, final_acc, train_acc, best_val_acc, device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    short = [n.split('___')[1] for n in PUMPKIN_CLASS_NAMES]

    # History
    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # Curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        epochs = list(range(1, len(history['train_acc']) + 1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, history['train_acc'], label='Train')
        axes[0].plot(epochs, history['val_acc'], label='Valid')
        axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy (%)'); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(epochs, history['train_loss'], label='Train')
        axes[1].plot(epochs, history['val_loss'], label='Valid')
        axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f'  Warning: could not save curves: {e}')

    # Confusion matrix plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(5), short, rotation=30, ha='right')
        ax.set_yticks(range(5), short)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Test Confusion Matrix (acc={100*final_acc:.1f}%)')
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                        color='white' if cm[i][j] > cm.max() / 2 else 'black')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f'  Warning: could not save confusion matrix plot: {e}')

    np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'), cm)
    with open(os.path.join(OUTPUT_DIR, 'confusion_matrix.json'), 'w') as f:
        json.dump({'classes': PUMPKIN_CLASS_NAMES,
                   'confusion_matrix': cm.tolist()}, f, indent=4)

    # Classification report
    rows, _ = metrics_from_cm(cm)
    report_txt = []
    report_txt.append('Classification Report (test set)')
    report_txt.append('=' * 60)
    report_txt.append(f'{"class":40s} {"precision":>10s} {"recall":>10s} '
                      f'{"f1":>10s} {"support":>8s}')
    for cls in PUMPKIN_CLASS_NAMES:
        r = rows[cls]
        report_txt.append(f'{cls:40s} {r["precision"]:10.4f} {r["recall"]:10.4f} '
                          f'{r["f1"]:10.4f} {r["support"]:8d}')
    report_txt.append('-' * 60)
    report_txt.append(f'{"accuracy":40s} {100*final_acc:10.2f}% '
                      f'({int(round(final_acc * sum(r["support"] for r in rows.values())))}/'
                      f'{sum(r["support"] for r in rows.values())})')
    report_txt.append('-' * 60)
    report_txt.append('macro avg precision/recall/f1: '
                      f'{np.mean([r["precision"] for r in rows.values()]):.4f} / '
                      f'{np.mean([r["recall"] for r in rows.values()]):.4f} / '
                      f'{np.mean([r["f1"] for r in rows.values()]):.4f}')
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write('\n'.join(report_txt) + '\n')
    with open(os.path.join(OUTPUT_DIR, 'classification_report.json'), 'w') as f:
        json.dump({'rows': rows, 'accuracy': round(float(final_acc), 4)}, f, indent=4)

    # Final accuracies
    final = {
        'train_accuracy': round(float(train_acc), 4),
        'best_val_accuracy': round(float(best_val_acc), 4),
        'test_accuracy': round(float(final_acc), 4),
        'classes': PUMPKIN_CLASS_NAMES,
        'device': str(device),
    }
    with open(os.path.join(OUTPUT_DIR, 'final_accuracies.json'), 'w') as f:
        json.dump(final, f, indent=4)

    print(f'\n  Artifacts saved to {OUTPUT_DIR}')
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        print(f'    {fn}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open(CLASSES_PATH) as f:
        existing_classes = json.load(f)
    pumpkin_idx = [i for i, n in enumerate(existing_classes)
                   if n.startswith('Pumpkin___')]
    print(f'Pumpkin indices in model: {pumpkin_idx}')
    assert len(pumpkin_idx) == 5, f'Expected 5 Pumpkin classes, got {len(pumpkin_idx)}'
    pumpkin_start, pumpkin_end = pumpkin_idx[0], pumpkin_idx[-1] + 1

    build_splits()
    split_paths = load_split_paths()

    print('\n' + '=' * 60)
    print('STEP 2: Feature extraction (batched ONNX)')
    print('=' * 60)
    feat_sess, feat_input = create_feature_extractor_session(PRODUCTION_ONNX)

    if os.path.exists(FEAT_CACHE):
        print(f'Loading cached features from {FEAT_CACHE}')
        with open(FEAT_CACHE, 'rb') as f:
            cache = pickle.load(f)
        train_feats, train_labels = cache['train_feats'], cache['train_labels']
        val_feats, val_labels = cache['val_feats'], cache['val_labels']
        test_feats, test_labels = cache['test_feats'], cache['test_labels']
    else:
        train_feats, train_labels = extract_split_features(
            feat_sess, feat_input, split_paths['train'], train_aug,
            double=True, tag='train')
        val_feats, val_labels = extract_split_features(
            feat_sess, feat_input, split_paths['valid'], tform,
            double=False, tag='valid')
        test_feats, test_labels = extract_split_features(
            feat_sess, feat_input, split_paths['test'], tform,
            double=False, tag='test')
        with open(FEAT_CACHE, 'wb') as f:
            pickle.dump({'train_feats': train_feats, 'train_labels': train_labels,
                         'val_feats': val_feats, 'val_labels': val_labels,
                         'test_feats': test_feats, 'test_labels': test_labels}, f)
        print(f'  Cached features to {FEAT_CACHE}')

    print(f'  train: {train_feats.shape[0]} samples, '
          f'valid: {val_feats.shape[0]}, test: {test_feats.shape[0]}')

    train_ds = FeatureDataset(train_feats, train_labels)
    val_ds = FeatureDataset(val_feats, val_labels)
    test_ds = FeatureDataset(test_feats, test_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    print('\n' + '=' * 60)
    print(f'STEP 3: Train classifier head ({HEAD_EPOCHS} epochs max)')
    print('=' * 60)
    head = nn.Linear(1280, len(PUMPKIN_CLASS_NAMES)).to(device)
    optimizer = optim.Adam(head.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    train_acc = 0.0

    for epoch in range(HEAD_EPOCHS):
        head.train()
        running_loss, correct, total = 0.0, 0, 0
        for feats, labels in train_loader:
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
        train_acc = 100.0 * correct / total
        train_loss = running_loss / max(1, len(train_loader))

        val_acc, val_loss = evaluate(head, val_loader, criterion)
        scheduler.step(val_acc)

        history['train_acc'].append(round(train_acc, 4))
        history['val_acc'].append(round(val_acc, 4))
        history['train_loss'].append(round(train_loss, 4))
        history['val_loss'].append(round(val_loss, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1:2d}/{HEAD_EPOCHS}] Train Acc: {train_acc:.2f}% '
                  f'| Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f} '
                  f'| LR: {optimizer.param_groups[0]["lr"]:.1e}')

        if patience_counter >= MAX_PATIENCE:
            print(f'  Early stopping at epoch {epoch + 1}')
            break

    print(f'\nBest validation accuracy: {best_val_acc:.2f}%')
    head.load_state_dict(best_state)
    head = head.to(device)

    # Bias centering (same as v3)
    with torch.no_grad():
        bias_mean = head.bias.mean()
        head.bias.sub_(bias_mean)
    print(f'  Centered biases: mean bias {bias_mean:.6f} subtracted')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(OUTPUT_DIR, 'best_model.pth'))

    # ---- Test evaluation ----
    print('\n' + '=' * 60)
    print('STEP 4: Test set evaluation')
    print('=' * 60)
    test_acc, _ = evaluate(head, test_loader, criterion)
    cm = confusion_matrix(head, test_loader, device)
    print(f'  Test accuracy: {test_acc:.2f}%')

    short = [n.split('___')[1][:14] for n in PUMPKIN_CLASS_NAMES]
    print('  ' + ' '*16 + ''.join(f'{s:>14s}' for s in short))
    for i in range(5):
        print(f'  {short[i]:14s}' + ''.join(f'{cm[i][j]:6d}' for j in range(5)))

    rows, _ = metrics_from_cm(cm)
    print('\n  Per-class:')
    for cls in PUMPKIN_CLASS_NAMES:
        r = rows[cls]
        print(f'    {cls:45s} prec={r["precision"]:.3f} rec={r["recall"]:.3f} '
              f'f1={r["f1"]:.3f} (tp={r["true_positives"]}, '
              f'fp={r["false_positives"]})')

    save_artifacts(history, cm, test_acc / 100.0, train_acc, best_val_acc, device)

    # ---- Overwrite ONNX ----
    print('\n' + '=' * 60)
    print('STEP 5: Overwrite Pumpkin rows in production ONNX')
    print('=' * 60)
    model = onnx.load(PRODUCTION_ONNX)
    weight_init = bias_init = None
    for init in model.graph.initializer:
        if len(init.dims) == 2 and init.dims[1] == 1280:
            if weight_init is None or init.dims[0] > weight_init.dims[0]:
                weight_init = init
    if weight_init is None:
        print('ERROR: classifier weights not found in ONNX graph')
        return
    classifier_rows = list(weight_init.dims)[0]
    for init in model.graph.initializer:
        if len(init.dims) == 1 and init.dims[0] == classifier_rows:
            bias_init = init
            break

    current_weight = np.frombuffer(weight_init.raw_data, dtype=np.float32) \
        .reshape(list(weight_init.dims))
    current_bias = np.frombuffer(bias_init.raw_data, dtype=np.float32)

    trained_weight = head.weight.detach().cpu().numpy()
    trained_bias = head.bias.detach().cpu().numpy()

    new_weight = current_weight.copy()
    new_bias = current_bias.copy()
    new_weight[pumpkin_start:pumpkin_end] = trained_weight
    new_bias[pumpkin_start:pumpkin_end] = trained_bias

    weight_init.raw_data = new_weight.astype(np.float32).tobytes()
    bias_init.raw_data = new_bias.astype(np.float32).tobytes()

    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f'  Validation warning: {e}')

    onnx.save(model, CANDIDATE_ONNX)
    print(f'  Saved candidate: {CANDIDATE_ONNX}')

    # Zero-regression check
    v1 = onnx.load(PRODUCTION_ONNX)
    v1_w = v1_b = None
    for init in v1.graph.initializer:
        dims = list(init.dims)
        if dims == [classifier_rows, 1280]:
            v1_w = np.frombuffer(init.raw_data, dtype=np.float32).reshape(dims)
        if dims == [classifier_rows]:
            v1_b = np.frombuffer(init.raw_data, dtype=np.float32)
    non_pumpkin = [i for i in range(classifier_rows) if i not in pumpkin_idx]
    w_diff = np.max(np.abs(new_weight[non_pumpkin] - v1_w[non_pumpkin]))
    b_diff = np.max(np.abs(new_bias[non_pumpkin] - v1_b[non_pumpkin]))
    print(f'  Non-Pumpkin weight max diff: {w_diff:.10f}')
    print(f'  Non-Pumpkin bias max diff:   {b_diff:.10f}')
    print('  ZERO REGRESSION OK' if w_diff == 0 and b_diff == 0
          else '  WARNING: non-Pumpkin rows changed!')

    # ---- Deploy to production ----
    if not os.path.exists(BACKUP_ONNX):
        shutil.copy2(PRODUCTION_ONNX, BACKUP_ONNX)
        print(f'  Backed up production model to {BACKUP_ONNX}')
    shutil.copy2(CANDIDATE_ONNX, PRODUCTION_ONNX)
    print(f'  Deployed candidate to production: {PRODUCTION_ONNX}')

    print('\n' + '=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f'  Train acc:  {train_acc:.2f}%')
    print(f'  Best val acc: {best_val_acc:.2f}%')
    print(f'  Test acc:   {test_acc:.2f}%')
    print(f'  Artifacts:  {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
