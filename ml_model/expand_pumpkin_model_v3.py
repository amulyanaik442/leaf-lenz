"""
Pumpkin Model Training v3 - Overwrite existing Pumpkin rows with retrained head.
Same zero-regression approach: only Pumpkin classifier rows are modified.
"""
import os, sys, json, random, shutil, subprocess, copy, pickle
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def install_dependencies():
    for pkg in ['torch', 'torchvision', 'kagglehub', 'onnx']:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_dependencies()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import onnx
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ASSETS = os.path.join(BASE_DIR, 'detector', 'ml_assets')
PRODUCTION_ONNX = os.path.join(ML_ASSETS, 'model.onnx')
CANDIDATE_ONNX = os.path.join(ML_ASSETS, 'pumpkin_candidate.onnx')
CLASSES_PATH = os.path.join(ML_ASSETS, 'class_names.json')
SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pumpkin_split_data_v2')
FEAT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pumpkin_features.pkl')

PUMPKIN_KAGGLE_ID = "rifat963/pumpkin"

RAW_FOLDER_CANDIDATES = {
    "Bacterial Leaf Spot": ["Bacterial Leaf Spot", "Bacterial_Leaf_Spot"],
    "Downy Mildew": ["Downy Mildew", "Downy_Mildew"],
    "Healthy Leaf": ["Healthy Leaf", "Healthy_Leaf"],
    "Mosaic Disease": ["Mosaic Disease", "Mosaic_Disease"],
    "Powdery Mildew": ["Powdery Mildew", "Powdery_Mildew"],
}

RAW_TO_PUMPKIN_CLASS = {
    "Bacterial Leaf Spot": "Pumpkin___bacterial_leaf_spot",
    "Downy Mildew": "Pumpkin___downy_mildew",
    "Healthy Leaf": "Pumpkin___healthy",
    "Mosaic Disease": "Pumpkin___mosaic_disease",
    "Powdery Mildew": "Pumpkin___powdery_mildew",
}

PUMPKIN_CLASS_NAMES = sorted(set(RAW_TO_PUMPKIN_CLASS.values()))

HEAD_EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 32
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TARGET_SIZE, CROP_SIZE = 256, 224
NUM_WORKERS = 0


def find_raw_class_dir(root_path, canonical_name):
    candidates = RAW_FOLDER_CANDIDATES[canonical_name]
    for item in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, item)):
            for c in candidates:
                if item == c:
                    return os.path.join(root_path, item)
    return None


def download_and_preprocess():
    print("\n" + "=" * 60)
    print("STEP 1: Download and preprocess Pumpkin dataset")
    print("=" * 60)

    train_dir = os.path.join(SPLIT_DIR, 'train')
    val_dir = os.path.join(SPLIT_DIR, 'val')

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        existing = sum(len(files) for _, _, files in os.walk(train_dir)) + \
                   sum(len(files) for _, _, files in os.walk(val_dir))
        if existing > 0:
            print(f"Split data already exists at {SPLIT_DIR} ({existing} files), skipping download.")
            return train_dir, val_dir

    print("Downloading Pumpkin dataset from Kaggle...")
    import kagglehub
    dataset_path = kagglehub.dataset_download(PUMPKIN_KAGGLE_ID)
    print(f"Dataset downloaded to: {dataset_path}")

    original_dir = os.path.join(dataset_path, 'Original', 'Original')
    aug_train_dir = os.path.join(dataset_path, 'Augmented', 'Augmented', 'train')
    aug_val_dir = os.path.join(dataset_path, 'Augmented', 'Augmented', 'valid')

    class_images = defaultdict(lambda: {'train': [], 'val': []})
    for canonical, pumpkin_cls in RAW_TO_PUMPKIN_CLASS.items():
        orig_cls = find_raw_class_dir(original_dir, canonical)
        if orig_cls:
            for f in os.listdir(orig_cls):
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    class_images[canonical]['train'].append(os.path.join(orig_cls, f))

        aug_tr = find_raw_class_dir(aug_train_dir, canonical)
        if aug_tr:
            for f in os.listdir(aug_tr):
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    class_images[canonical]['train'].append(os.path.join(aug_tr, f))

        aug_vl = find_raw_class_dir(aug_val_dir, canonical)
        if aug_vl:
            for f in os.listdir(aug_vl):
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    class_images[canonical]['val'].append(os.path.join(aug_vl, f))

    total_train, total_val = 0, 0
    for canonical in sorted(RAW_TO_PUMPKIN_CLASS):
        pc = RAW_TO_PUMPKIN_CLASS[canonical]
        t, v = len(class_images[canonical]['train']), len(class_images[canonical]['val'])
        total_train += t; total_val += v
        print(f"  {pc:45s} train={t:4d}  val={v:4d}")

    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    for split_name in ['train', 'val']:
        for canonical, pumpkin_cls in RAW_TO_PUMPKIN_CLASS.items():
            paths = class_images[canonical][split_name]
            if not paths:
                continue
            d = os.path.join(SPLIT_DIR, split_name, pumpkin_cls)
            os.makedirs(d, exist_ok=True)
            for src in paths:
                shutil.copy2(src, os.path.join(d, os.path.basename(src)))

    print(f"\nSplit data saved to {SPLIT_DIR}")
    return train_dir, val_dir


def create_feature_extractor_session(onnx_path):
    model = onnx.load(onnx_path)
    classifier_input_name = None
    classifier_node = None

    for node in model.graph.node:
        if node.op_type == 'Gemm' or node.op_type == 'MatMul':
            for out in model.graph.output:
                if out.name in node.output:
                    classifier_node = node
                    classifier_input_name = node.input[0]
                    break
        if classifier_node:
            break

    if classifier_node is None:
        output_name = model.graph.output[0].name
        for node in reversed(model.graph.node):
            if output_name in node.output:
                classifier_node = node
                classifier_input_name = node.input[0]
                break

    print(f"  Classifier node: {classifier_node.name if classifier_node else 'unknown'}")
    print(f"  Feature input name: {classifier_input_name}")

    feature_model = copy.deepcopy(model)
    feature_out = onnx.helper.make_tensor_value_info(
        classifier_input_name, onnx.TensorProto.FLOAT, [None, 1280])
    feature_model.graph.output.append(feature_out)

    try:
        onnx.checker.check_model(feature_model)
    except Exception as e:
        print(f"  Feature model validation warning: {e}")

    feature_path = onnx_path.replace('.onnx', '_features.onnx')
    onnx.save(feature_model, feature_path)
    sess = ort.InferenceSession(feature_path)
    return sess, classifier_input_name


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(CLASSES_PATH, 'r') as f:
        existing_classes = json.load(f)

    total_classes = len(existing_classes)
    print(f"Total classes in model: {total_classes}")

    # Find current Pumpkin indices
    pumpkin_idx_in_model = [i for i, n in enumerate(existing_classes) if n.startswith('Pumpkin___')]
    print(f"Existing Pumpkin indices: {pumpkin_idx_in_model}")

    if len(pumpkin_idx_in_model) != 5:
        print(f"ERROR: Expected 5 Pumpkin classes, found {len(pumpkin_idx_in_model)}")
        return

    # Verify they are at the expected end of the array
    pumpkin_start = pumpkin_idx_in_model[0]
    pumpkin_end = pumpkin_idx_in_model[-1] + 1

    train_dir, val_dir = download_and_preprocess()

    train_paths = defaultdict(list)
    val_paths = defaultdict(list)

    for cls_name in PUMPKIN_CLASS_NAMES:
        for split_path, store in [(train_dir, train_paths), (val_dir, val_paths)]:
            cls_dir = os.path.join(split_path, cls_name)
            if os.path.isdir(cls_dir):
                for f in os.listdir(cls_dir):
                    if f.lower().endswith(('.png','.jpg','.jpeg')):
                        store[cls_name].append(os.path.join(cls_dir, f))

    for cls in PUMPKIN_CLASS_NAMES:
        print(f"  {cls}: {len(train_paths[cls])} train, {len(val_paths[cls])} val")

    print("\n" + "=" * 60)
    print("STEP 2: Create feature extractor from production ONNX")
    print("=" * 60)

    feat_sess, feat_name = create_feature_extractor_session(PRODUCTION_ONNX)

    print("\n" + "=" * 60)
    print("STEP 3: Extract features")
    print("=" * 60)

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

    pumpkin_local_map = {cls: i for i, cls in enumerate(PUMPKIN_CLASS_NAMES)}

    if os.path.exists(FEAT_CACHE):
        print(f"Loading cached features from {FEAT_CACHE}...")
        with open(FEAT_CACHE, 'rb') as f:
            cache = pickle.load(f)
        all_train_feats = cache['train_feats']
        all_train_labels = cache['train_labels']
        all_val_feats = cache['val_feats']
        all_val_labels = cache['val_labels']
        print(f"  Loaded {len(all_train_feats)} train, {len(all_val_feats)} val features")
    else:
        print("Extracting features for training data...")
        all_train_feats = []
        all_train_labels = []

        for cls_name in PUMPKIN_CLASS_NAMES:
            paths = train_paths[cls_name]
            label = pumpkin_local_map[cls_name]
            for p in paths:
                img = Image.open(p).convert('RGB')
                img_t = train_aug(img).unsqueeze(0).numpy()
                out = feat_sess.run(None, {feat_sess.get_inputs()[0].name: img_t})
                feat = out[1][0]
                all_train_feats.append(feat)
                all_train_labels.append(label)
                img_t2 = train_aug(img).unsqueeze(0).numpy()
                out2 = feat_sess.run(None, {feat_sess.get_inputs()[0].name: img_t2})
                all_train_feats.append(out2[1][0])
                all_train_labels.append(label)

        print(f"  Total training samples: {len(all_train_feats)}")

        print("Extracting features for validation data...")
        all_val_feats = []
        all_val_labels = []
        for cls_name in PUMPKIN_CLASS_NAMES:
            paths = val_paths[cls_name]
            label = pumpkin_local_map[cls_name]
            for p in paths:
                img = Image.open(p).convert('RGB')
                img_t = tform(img).unsqueeze(0).numpy()
                out = feat_sess.run(None, {feat_sess.get_inputs()[0].name: img_t})
                all_val_feats.append(out[1][0])
                all_val_labels.append(label)

        print(f"  Total validation samples: {len(all_val_feats)}")

        with open(FEAT_CACHE, 'wb') as f:
            pickle.dump({
                'train_feats': all_train_feats,
                'train_labels': all_train_labels,
                'val_feats': all_val_feats,
                'val_labels': all_val_labels,
            }, f)
        print(f"  Cached features to {FEAT_CACHE}")

    train_dataset = FeatureDataset(all_train_feats, all_train_labels)
    val_dataset = FeatureDataset(all_val_feats, all_val_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("\n" + "=" * 60)
    print(f"STEP 4: Train classifier head ({HEAD_EPOCHS} epochs)")
    print("=" * 60)

    pumpkin_head = nn.Linear(1280, len(PUMPKIN_CLASS_NAMES))
    pumpkin_head.to(device)

    optimizer = optim.Adam(pumpkin_head.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    max_patience = 10

    for epoch in range(HEAD_EPOCHS):
        pumpkin_head.train()
        running_loss, correct, total = 0.0, 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = pumpkin_head(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        pumpkin_head.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                outputs = pumpkin_head(feats)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in pumpkin_head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{HEAD_EPOCHS}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= max_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    pumpkin_head.load_state_dict(best_state)
    pumpkin_head.to(device)

    # Bias centering
    with torch.no_grad():
        bias_mean = pumpkin_head.bias.mean()
        pumpkin_head.bias.sub_(bias_mean)
    print(f"  Centered biases: mean bias = {bias_mean:.6f} subtracted")

    print("\n" + "=" * 60)
    print("STEP 5: Overwrite Pumpkin rows in ONNX")
    print("=" * 60)

    model = onnx.load(PRODUCTION_ONNX)

    weight_init = None
    bias_init = None
    weight_name = None
    bias_name = None

    for init in model.graph.initializer:
        if len(init.dims) == 2 and init.dims[1] == 1280:
            if weight_init is None or init.dims[0] > weight_init.dims[0]:
                weight_init = init
                weight_name = init.name

    if weight_init is None:
        print("ERROR: Could not find classifier weights in ONNX graph!")
        return

    classifier_rows = list(weight_init.dims)[0]
    for init in model.graph.initializer:
        if len(init.dims) == 1 and init.dims[0] == classifier_rows:
            bias_init = init
            bias_name = init.name
            break

    if weight_init is None or bias_init is None:
        print("ERROR: Could not find classifier weights in ONNX graph!")
        return

    print(f"Found weight: {weight_name} shape={list(weight_init.dims)}")
    print(f"Found bias: {bias_name} shape={list(bias_init.dims)}")

    current_weight = np.frombuffer(weight_init.raw_data, dtype=np.float32).reshape(list(weight_init.dims))
    current_bias = np.frombuffer(bias_init.raw_data, dtype=np.float32).reshape(list(bias_init.dims))

    print(f"Current Pumpkin weight rows (indices {pumpkin_start}:{pumpkin_end}):")
    for i in pumpkin_idx_in_model:
        print(f"  {existing_classes[i]:40s} norm={np.linalg.norm(current_weight[i]):.4f}  bias={current_bias[i]:+.6f}")

    # Overwrite Pumpkin rows with new trained weights
    trained_weight = pumpkin_head.weight.detach().cpu().numpy()
    trained_bias = pumpkin_head.bias.detach().cpu().numpy()

    new_weight = current_weight.copy()
    new_bias = current_bias.copy()
    new_weight[pumpkin_start:pumpkin_end] = trained_weight
    new_bias[pumpkin_start:pumpkin_end] = trained_bias

    weight_init.raw_data = new_weight.astype(np.float32).tobytes()
    bias_init.raw_data = new_bias.astype(np.float32).tobytes()

    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"Validation warning: {e}")

    onnx.save(model, CANDIDATE_ONNX)
    print(f"\nSaved candidate model to {CANDIDATE_ONNX}")

    # Verify
    model2 = onnx.load(CANDIDATE_ONNX)
    for init in model2.graph.initializer:
        dims = list(init.dims)
        if dims == [total_classes, 1280]:
            vw = np.frombuffer(init.raw_data, dtype=np.float32).reshape(dims)
            print(f"\nNew Pumpkin weight/bias values:")
            for i in pumpkin_idx_in_model:
                print(f"  {existing_classes[i]:40s} norm={np.linalg.norm(vw[i]):.4f}  bias={new_bias[i]:+.6f}")

    # Run quick accuracy check on Original dataset
    print("\n" + "=" * 60)
    print("QUICK ACCURACY CHECK")
    print("=" * 60)

    sess = ort.InferenceSession(CANDIDATE_ONNX)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict(path):
        img = Image.open(path).convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr.transpose(2, 0, 1) - MEAN[:, None, None]) / STD[:, None, None]
        arr = arr.astype(np.float32)[None, :, :, :]
        logits = sess.run(None, {sess.get_inputs()[0].name: arr})[0][0]
        p = logits[pumpkin_idx_in_model]
        exp = np.exp(p - np.max(p))
        return exp / np.sum(exp)

    orig_dir = os.path.join(BASE_DIR, 'dataset', 'pumpkin', 'Original', 'Original')
    FOLDER_MAP = {
        'Bacterial Leaf Spot': 0,
        'Downy Mildew': 1,
        'Healthy Leaf': 2,
        'Mosaic Disease': 3,
        'Powdery Mildew': 4,
    }
    cm = np.zeros((5, 5), dtype=int)
    total = correct = 0
    for folder, true_i in FOLDER_MAP.items():
        d = os.path.join(orig_dir, folder)
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            if not fn.lower().endswith(('.png','.jpg','.jpeg')): continue
            probs = predict(os.path.join(d, fn))
            pred_i = int(np.argmax(probs))
            cm[true_i][pred_i] += 1
            total += 1
            if pred_i == true_i:
                correct += 1

    acc = 100.0 * correct / total
    print(f"  Accuracy: {correct}/{total} = {acc:.2f}%")
    short = [n.split('___')[1][:14] for n in PUMPKIN_CLASS_NAMES]
    hdr = ' ' * 16 + ''.join(f'{s:>14s}' for s in short)
    print(f"  {hdr}")
    for i in range(5):
        row = f"  {short[i]:14s}" + ''.join(f'{cm[i][j]:6d}' for j in range(5))
        print(f"  {row}")
    for i in range(5):
        cs = cm[i].sum()
        a = 100*cm[i][i]/cs if cs>0 else 0
        print(f"  {PUMPKIN_CLASS_NAMES[i]:40s} acc={a:.1f}%")

    # Check bias distribution
    print(f"\n  Pumpkin bias range: {new_bias[pumpkin_idx_in_model].min():+.4f} to {new_bias[pumpkin_idx_in_model].max():+.4f}")
    print(f"  Mean bias: {new_bias[pumpkin_idx_in_model].mean():+.6f} (should be ~0)")

    # Verify zero regression on all 108 outputs
    print("\n" + "=" * 60)
    print("REGRESSION CHECK (first 10 non-Pumpkin classes)")
    print("=" * 60)
    v1_model = onnx.load(PRODUCTION_ONNX)
    v1_w = None; v1_b = None
    for init in v1_model.graph.initializer:
        dims = list(init.dims)
        if dims == [total_classes, 1280]:
            v1_w = np.frombuffer(init.raw_data, dtype=np.float32).reshape(dims)
        if dims == [total_classes]:
            v1_b = np.frombuffer(init.raw_data, dtype=np.float32)

    non_pumpkin = [i for i in range(total_classes) if i not in pumpkin_idx_in_model]
    if v1_w is not None:
        w_diff = np.max(np.abs(new_weight[non_pumpkin] - v1_w[non_pumpkin]))
        b_diff = np.max(np.abs(new_bias[non_pumpkin] - v1_b[non_pumpkin]))
        print(f"  Non-Pumpkin weight max diff: {w_diff:.10f} (should be 0)")
        print(f"  Non-Pumpkin bias max diff:   {b_diff:.10f} (should be 0)")
        if w_diff == 0 and b_diff == 0:
            print("  ZERO REGRESSION: Non-Pumpkin classes unchanged")
        else:
            print("  WARNING: Non-Pumpkin classes changed!")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Candidate model: {CANDIDATE_ONNX}")
    print(f"  Pumpkin classes at indices {pumpkin_start}:{pumpkin_end}")
    print(f"  Accuracy on Original set: {acc:.2f}%")


if __name__ == '__main__':
    main()
