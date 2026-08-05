"""
Train the Tea classifier head with a strict zero-regression approach.

Approach (approved): freeze all 108 existing classifier rows, train ONLY the 8
new Tea rows (indices 108-115) on Tea data, then splice the trained rows into a
candidate ONNX. Existing crops remain byte-identical.

Pipeline:
  1. Stratified splits already exist: `dataset/tea/Tea_Split` (120/15/15).
  2. Extract 1280-d features from the frozen production ONNX backbone.
  3. Train a standalone Linear(1280 -> 8) head.
  4. Append the trained rows to a candidate ONNX (116 classes) + candidate
     class names.  Production model and class_names.json are NOT modified.

Outputs (`ml_model/tea_retrain_output/`):
  * tea_candidate.onnx + candidate_class_names.json
  * best_model.pth + checkpoints/epoch_NN.pth (every epoch)
  * training_history.json, training_curves.png
  * confusion_matrix (.png/.json), classification_report (.txt/.json)
  * final_accuracies.json, zero_regression_check.json

Usage:
    python ml_model/train_tea.py
"""
import os
import sys
import json
import copy
import pickle
import shutil

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

SPLIT_DIR = os.path.join(BASE_DIR, 'dataset', 'tea', 'Tea_Split')
FEAT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'tea_features.pkl')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'tea_retrain_output')
CKPT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

# ---------------------------------------------------------------------------
# Config (shown before training)
# ---------------------------------------------------------------------------
FOLDER_TO_MODEL = {
    'Anthracnose': 'Tea___anthracnose',
    'algal leaf': 'Tea___algal_leaf',
    'bird eye spot': 'Tea___bird_eye_spot',
    'brown blight': 'Tea___brown_blight',
    'gray light': 'Tea___gray_light',
    'healthy': 'Tea___healthy',
    'red leaf spot': 'Tea___red_leaf_spot',
    'white spot': 'Tea___white_spot',
}
TEA_CLASS_NAMES = list(FOLDER_TO_MODEL.values())

HEAD_EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 32
SEED = 42
MAX_PATIENCE = 10
TARGET_SIZE, CROP_SIZE = 256, 224
NUM_WORKERS = 0
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP')

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Transforms (same as pumpkin retrain pipeline)
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


def extract_split_features(feat_sess, feat_input, split_paths, transform,
                           double=False, tag=''):
    feats, labels = [], []
    for folder, model_cls in FOLDER_TO_MODEL.items():
        d = os.path.join(split_paths, folder)
        if not os.path.isdir(d):
            print(f'  WARNING: {d} not found')
            continue
        paths = sorted([os.path.join(d, f) for f in os.listdir(d)
                        if f.lower().endswith(IMAGE_EXTS)])
        label = TEA_CLASS_NAMES.index(model_cls)
        for i in range(0, len(paths), 64):
            chunk = paths[i:i + 64]
            tensors = [transform(Image.open(p).convert('RGB')).numpy()
                       for p in chunk]
            arr = np.stack(tensors).astype(np.float32)
            feats.append(feat_sess.run(None, {feat_input: arr})[-1])
            labels.extend([label] * len(chunk))
            if double:
                tensors2 = [transform(Image.open(p).convert('RGB')).numpy()
                            for p in chunk]
                arr2 = np.stack(tensors2).astype(np.float32)
                feats.append(feat_sess.run(None, {feat_input: arr2})[-1])
                labels.extend([label] * len(chunk))
        print(f'  {tag} {folder:18s} {len(paths)} images', flush=True)
    if not feats:
        return np.zeros((0, 1280), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.vstack(feats), np.array(labels, dtype=np.int64)


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


def confusion_matrix(head, loader, device):
    head.eval()
    cm = np.zeros((8, 8), dtype=int)
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            _, preds = head(feats).max(1)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1
    return cm


def metrics_from_cm(cm):
    rows = {}
    for i, cls in enumerate(TEA_CLASS_NAMES):
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
    short = [c.split('___')[1] for c in TEA_CLASS_NAMES]

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

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(8), short, rotation=30, ha='right')
        ax.set_yticks(range(8), short)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Tea Test Confusion Matrix (acc={100*test_acc:.1f}%)')
        for i in range(8):
            for j in range(8):
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
        json.dump({'classes': TEA_CLASS_NAMES, 'confusion_matrix': cm.tolist()},
                  f, indent=4)

    rows, _ = metrics_from_cm(cm)
    lines = ['Tea Classification Report (internal test set)', '=' * 60]
    lines.append(f'{"class":28s} {"precision":>10s} {"recall":>10s} {"f1":>10s} {"support":>8s}')
    for cls in TEA_CLASS_NAMES:
        r = rows[cls]
        lines.append(f'{cls:28s} {r["precision"]:10.4f} {r["recall"]:10.4f} {r["f1"]:10.4f} {r["support"]:8d}')
    lines.append('-' * 60)
    lines.append(f'{"accuracy":28s} {100*test_acc:10.2f}% ({int(round(test_acc*sum(r["support"] for r in rows.values())))}/{sum(r["support"] for r in rows.values())})')
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
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ---- Show config ----
    print('\n' + '=' * 60)
    print('TRAINING CONFIGURATION')
    print('=' * 60)
    print(f'  Backbone:      frozen EfficientNet-B0 (1280-d), from production ONNX')
    print(f'  Head:          Linear(1280 -> 8) Tea-only, zero-regression')
    print(f'  Loss:          CrossEntropyLoss(label_smoothing=0.1)')
    print(f'  Optimizer:     Adam lr={LR}')
    print(f'  Scheduler:     ReduceLROnPlateau(mode=max, factor=0.5, patience=5)')
    print(f'  Early stop:    patience={MAX_PATIENCE}, max {HEAD_EPOCHS} epochs')
    print(f'  Batch size:    {BATCH_SIZE}')
    print(f'  Train aug:     Resize(256)->RandomCrop(224), hflip, vflip, rot10')
    print(f'                 + feature doubling (2 views/train image)')
    print(f'  Eval aug:      Resize(224)')
    print(f'  Checkpoints:   every epoch -> {CKPT_DIR}')
    print(f'  New classes:   indices 108-115: {TEA_CLASS_NAMES}')
    print('=' * 60)

    with open(CLASSES_PATH) as f:
        existing_classes = json.load(f)
    n_existing = len(existing_classes)
    assert n_existing == 108, f'Expected 108 existing classes, got {n_existing}'
    assert not any(c.startswith('Tea___') for c in existing_classes), \
        'Tea classes already present!'
    tea_start = n_existing  # 108
    tea_end = tea_start + 8  # 116
    candidate_classes = existing_classes + TEA_CLASS_NAMES
    print(f'\n  Existing classes: {n_existing} | Tea range: {tea_start}-{tea_end - 1} | Candidate total: {len(candidate_classes)}')

    # ---- Features ----
    print('\n' + '=' * 60)
    print('FEATURE EXTRACTION')
    print('=' * 60)
    feat_sess, feat_input, tmp_path = create_feature_extractor_session(PRODUCTION_ONNX)
    if os.path.exists(FEAT_CACHE):
        print(f'Loading cached features from {FEAT_CACHE}')
        with open(FEAT_CACHE, 'rb') as f:
            cache = pickle.load(f)
        train_feats, train_labels = cache['train_feats'], cache['train_labels']
        val_feats, val_labels = cache['val_feats'], cache['val_labels']
        test_feats, test_labels = cache['test_feats'], cache['test_labels']
    else:
        train_feats, train_labels = extract_split_features(
            feat_sess, feat_input, os.path.join(SPLIT_DIR, 'train'), train_aug,
            double=True, tag='train')
        val_feats, val_labels = extract_split_features(
            feat_sess, feat_input, os.path.join(SPLIT_DIR, 'valid'), tform,
            double=False, tag='valid')
        test_feats, test_labels = extract_split_features(
            feat_sess, feat_input, os.path.join(SPLIT_DIR, 'test'), tform,
            double=False, tag='test')
        with open(FEAT_CACHE, 'wb') as f:
            pickle.dump({'train_feats': train_feats, 'train_labels': train_labels,
                         'val_feats': val_feats, 'val_labels': val_labels,
                         'test_feats': test_feats, 'test_labels': test_labels}, f)
        print(f'  Cached features to {FEAT_CACHE}')
    print(f'  train: {train_feats.shape[0]}, valid: {val_feats.shape[0]}, test: {test_feats.shape[0]}')
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    train_ds = FeatureDataset(train_feats, train_labels)
    val_ds = FeatureDataset(val_feats, val_labels)
    test_ds = FeatureDataset(test_feats, test_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    # ---- Train ----
    print('\n' + '=' * 60)
    print(f'TRAINING ({HEAD_EPOCHS} epochs max)')
    print('=' * 60)
    head = nn.Linear(1280, 8).to(device)
    optimizer = optim.Adam(head.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(CKPT_DIR, exist_ok=True)
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

        val_acc, val_loss = evaluate(head, val_loader, criterion, device)
        scheduler.step(val_acc)

        history['train_acc'].append(round(train_acc, 4))
        history['val_acc'].append(round(val_acc, 4))
        history['train_loss'].append(round(train_loss, 4))
        history['val_loss'].append(round(val_loss, 4))

        torch.save(head.state_dict(), os.path.join(CKPT_DIR, f'epoch_{epoch + 1:02d}.pth'))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1:2d}/{HEAD_EPOCHS}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.1e}')

        if patience_counter >= MAX_PATIENCE:
            print(f'  Early stopping at epoch {epoch + 1}')
            break

    print(f'\nBest validation accuracy: {best_val_acc:.2f}%')
    head.load_state_dict(best_state)
    head = head.to(device)

    with torch.no_grad():
        bias_mean = head.bias.mean()
        head.bias.sub_(bias_mean)
    print(f'  Centered Tea biases: mean bias {bias_mean:.6f} subtracted')

    torch.save(best_state, os.path.join(OUTPUT_DIR, 'best_model.pth'))

    # ---- Test evaluation ----
    print('\n' + '=' * 60)
    print('INTERNAL TEST SET EVALUATION')
    print('=' * 60)
    test_acc, _ = evaluate(head, test_loader, criterion, device)
    cm = confusion_matrix(head, test_loader, device)
    print(f'  Test accuracy: {test_acc:.2f}%')
    short = [c.split('___')[1][:12] for c in TEA_CLASS_NAMES]
    print('  ' + ' '*15 + ''.join(f'{s:>13s}' for s in short))
    for i in range(8):
        print(f'  {short[i]:14s}' + ''.join(f'{cm[i][j]:6d}' for j in range(8)))
    rows, _ = metrics_from_cm(cm)
    for cls in TEA_CLASS_NAMES:
        r = rows[cls]
        print(f'  {cls:26s} prec={r["precision"]:.3f} rec={r["recall"]:.3f} f1={r["f1"]:.3f}')

    save_artifacts(history, cm, test_acc / 100.0, train_acc, best_val_acc, device)

    # ---- Candidate ONNX (append rows, zero regression) ----
    print('\n' + '=' * 60)
    print('CANDIDATE ONNX (Tea rows appended, existing rows frozen)')
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

    new_weight = np.vstack([old_weight, trained_weight])
    new_bias = np.concatenate([old_bias, trained_bias])

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

    candidate_path = os.path.join(OUTPUT_DIR, 'tea_candidate.onnx')
    onnx.save(model, candidate_path)
    print(f'  Saved candidate: {candidate_path}')

    with open(os.path.join(OUTPUT_DIR, 'candidate_class_names.json'), 'w') as f:
        json.dump(candidate_classes, f, indent=4)
    print(f'  Saved candidate class names ({len(candidate_classes)} classes)')

    # Zero-regression verification
    v1 = onnx.load(PRODUCTION_ONNX)
    v1_w = v1_b = None
    for init in v1.graph.initializer:
        dims = list(init.dims)
        if dims == [classifier_rows, 1280]:
            v1_w = np.frombuffer(init.raw_data, dtype=np.float32).reshape(dims)
        if dims == [classifier_rows]:
            v1_b = np.frombuffer(init.raw_data, dtype=np.float32)
    non_tea = list(range(0, tea_start))
    w_diff = float(np.max(np.abs(new_weight[non_tea] - v1_w[non_tea])))
    b_diff = float(np.max(np.abs(new_bias[non_tea] - v1_b[non_tea])))
    ok = w_diff == 0.0 and b_diff == 0.0
    print(f'  Non-Tea weight max diff: {w_diff:.10f}')
    print(f'  Non-Tea bias max diff:   {b_diff:.10f}')
    print(f'  ZERO REGRESSION: {"OK" if ok else "FAILED"}')
    with open(os.path.join(OUTPUT_DIR, 'zero_regression_check.json'), 'w') as f:
        json.dump({'non_tea_weight_max_diff': w_diff,
                   'non_tea_bias_max_diff': b_diff, 'zero_regression_ok': ok}, f, indent=4)

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
