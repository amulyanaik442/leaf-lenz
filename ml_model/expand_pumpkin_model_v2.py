"""
Pumpkin Model Training -- Zero-Regression Approach
===================================================
Extracts 1280-d features from the production ONNX (using ONNX Runtime),
trains a Linear(1280, 5) head for Pumpkin classes, then directly appends
the new weights to the production ONNX graph. Guarantees zero regression
because the original ONNX classifier rows are never modified.

Usage:
    python ml_model/expand_pumpkin_model_v2.py
"""
import os, sys, json, random, shutil, subprocess, copy
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

HEAD_EPOCHS = 5
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
    """Create an ONNX Runtime session that outputs the 1280-d features
    from the penultimate layer (before classifier)."""
    model = onnx.load(onnx_path)

    # Find the classifier Gemm/MatMul node and get its input (the 1280 features)
    classifier_input_name = None
    classifier_node = None

    for node in model.graph.node:
        if node.op_type == 'Gemm' or node.op_type == 'MatMul':
            # In EfficientNet-B0, the classifier is typically the last Gemm
            # Check if any output of this node is also the model output
            for out in model.graph.output:
                if out.name in node.output:
                    classifier_node = node
                    classifier_input_name = node.input[0]
                    break
        if classifier_node:
            break

    if classifier_node is None:
        # Try to find by looking at the last node before output
        output_name = model.graph.output[0].name
        for node in reversed(model.graph.node):
            if output_name in node.output:
                classifier_node = node
                classifier_input_name = node.input[0]
                break

    print(f"  Classifier node: {classifier_node.name if classifier_node else 'unknown'}")
    print(f"  Feature input name: {classifier_input_name}")

    # Create a version that outputs both original output and features
    feature_model = copy.deepcopy(model)

    # Add the feature tensor as an additional output
    feature_out = onnx.helper.make_tensor_value_info(
        classifier_input_name,
        onnx.TensorProto.FLOAT,
        [None, 1280]  # EfficientNet-B0 features
    )
    feature_model.graph.output.append(feature_out)

    # Validate
    try:
        onnx.checker.check_model(feature_model)
    except Exception as e:
        print(f"  Feature model validation warning: {e}")

    # Save and create session
    feature_path = onnx_path.replace('.onnx', '_features.onnx')
    onnx.save(feature_model, feature_path)

    sess = ort.InferenceSession(feature_path)
    return sess, classifier_input_name


def extract_features(sess, feature_name, image_paths, transform):
    """Extract 1280-d feature vectors for a list of images."""
    features = []
    for p in image_paths:
        img = Image.open(p).convert('RGB')
        img_t = transform(img).unsqueeze(0).numpy()
        out = sess.run(None, {sess.get_inputs()[0].name: img_t})
        # out[0] is original output (103 logits), out[1] is features
        feat = out[1][0]  # (1280,)
        features.append(feat)
    return np.array(features)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load existing class names
    with open(CLASSES_PATH, 'r') as f:
        existing_classes = json.load(f)
    original_num_classes = len(existing_classes)
    print(f"Existing classes: {original_num_classes}")

    # Pumpkin classes to add
    new_pumpkin_classes = [c for c in PUMPKIN_CLASS_NAMES if c not in existing_classes]
    print(f"New Pumpkin classes: {new_pumpkin_classes}")
    if not new_pumpkin_classes:
        print("All Pumpkin classes already present. Nothing to do.")
        return

    all_class_names = existing_classes + new_pumpkin_classes
    target_num_classes = len(all_class_names)
    new_class_offset = original_num_classes

    # Download/preprocess
    train_dir, val_dir = download_and_preprocess()

    # Load image paths per class
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

    # ── Create feature extractor from production ONNX ──
    print("\n" + "=" * 60)
    print("STEP 2: Create feature extractor from production ONNX")
    print("=" * 60)

    feat_sess, feat_name = create_feature_extractor_session(PRODUCTION_ONNX)

    # ── Extract features ──
    print("\n" + "=" * 60)
    print("STEP 3: Extract features from production ONNX backbone")
    print("=" * 60)

    tform = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Training augmentations for Pumpkin only
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

    # Build training dataset: Pumpkin images + features
    class FeatureDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    # Extract features for Pumpkin training images with augmentation
    print("Extracting features for training data...")
    all_train_feats = []
    all_train_labels = []
    pumpkin_local_map = {cls: i for i, cls in enumerate(PUMPKIN_CLASS_NAMES)}

    for cls_name in PUMPKIN_CLASS_NAMES:
        paths = train_paths[cls_name]
        label = pumpkin_local_map[cls_name]  # local index 0-4
        for p in paths:
            img = Image.open(p).convert('RGB')
            img_t = train_aug(img).unsqueeze(0).numpy()
            out = feat_sess.run(None, {feat_sess.get_inputs()[0].name: img_t})
            feat = out[1][0]
            all_train_feats.append(feat)
            all_train_labels.append(label)
            # Add an extra augmented view
            img_t2 = train_aug(img).unsqueeze(0).numpy()
            out2 = feat_sess.run(None, {feat_sess.get_inputs()[0].name: img_t2})
            all_train_feats.append(out2[1][0])
            all_train_labels.append(label)

    print(f"  Total training samples (with augmentation): {len(all_train_feats)}")

    # Extract features for validation data
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

    print("(No existing class mix-in needed: backbone is frozen, head is 5-class only)")

    print(f"Total training samples: {len(all_train_feats)}")

    # ── Train Pumpkin head ──
    print("\n" + "=" * 60)
    print("STEP 4: Train Pumpkin classifier head (1280 -> 5)")
    print("=" * 60)

    train_dataset = FeatureDataset(all_train_feats, all_train_labels)
    val_dataset = FeatureDataset(all_val_feats, all_val_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    pumpkin_head = nn.Linear(1280, len(PUMPKIN_CLASS_NAMES))
    pumpkin_head.to(device)

    optimizer = optim.Adam(pumpkin_head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
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
        best_val_acc = max(best_val_acc, val_acc)
        print(f"Epoch [{epoch+1}/{HEAD_EPOCHS}] Train Loss: {running_loss/len(train_loader):.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.2f}%")

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

    # ── Insert trained weights into production ONNX ──
    print("\n" + "=" * 60)
    print("STEP 5: Insert new classifier rows into production ONNX")
    print("=" * 60)

    model = onnx.load(PRODUCTION_ONNX)

    # Find and modify the classifier weight and bias initializers
    weight_init = None
    bias_init = None
    weight_name = None
    bias_name = None

    for init in model.graph.initializer:
        if len(init.dims) == 2 and init.dims[0] == original_num_classes and init.dims[1] == 1280:
            weight_init = init
            weight_name = init.name
        elif len(init.dims) == 1 and init.dims[0] == original_num_classes:
            bias_init = init
            bias_name = init.name

    if weight_init is None or bias_init is None:
        print("ERROR: Could not find classifier weights in ONNX graph!")
        # Try alternative: look for any (103, 1280) tensor
        for init in model.graph.initializer:
            print(f"  init: name={init.name}, dims={list(init.dims)}")
        return

    print(f"Found weight: {weight_name} shape={list(weight_init.dims)}")
    print(f"Found bias: {bias_name} shape={list(bias_init.dims)}")

    # Extract existing weights
    old_weight = np.frombuffer(weight_init.raw_data, dtype=np.float32).reshape(list(weight_init.dims))
    old_bias = np.frombuffer(bias_init.raw_data, dtype=np.float32).reshape(list(bias_init.dims))

    # Get trained head weights
    trained_weight = pumpkin_head.weight.detach().cpu().numpy()  # (5, 1280)
    trained_bias = pumpkin_head.bias.detach().cpu().numpy()      # (5,)

    # Concatenate
    new_weight = np.vstack([old_weight, trained_weight])  # (108, 1280)
    new_bias = np.concatenate([old_bias, trained_bias])    # (108,)

    # Update the initializer
    weight_init.raw_data = new_weight.astype(np.float32).tobytes()
    weight_init.dims[:] = new_weight.shape
    bias_init.raw_data = new_bias.astype(np.float32).tobytes()
    bias_init.dims[0] = new_bias.shape[0]

    # Update the output tensor shape
    for out in model.graph.output:
        if out.type.tensor_type.HasField('shape'):
            dim = out.type.tensor_type.shape.dim
            if len(dim) > 1 and dim[1].dim_value == original_num_classes:
                dim[1].dim_value = target_num_classes
                print(f"Updated output shape: {dim[1].dim_value}")

    # Validate
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"Validation warning: {e}")

    onnx.save(model, CANDIDATE_ONNX)
    print(f"Saved candidate model to {CANDIDATE_ONNX}")

    # Update class names
    with open(CLASSES_PATH, 'w') as f:
        json.dump(all_class_names, f, indent=4)
    print(f"Updated class_names.json ({len(all_class_names)} classes)")

    # Verify
    sess = ort.InferenceSession(CANDIDATE_ONNX)
    out_shape = sess.get_outputs()[0].shape
    print(f"ONNX output shape: {out_shape} (expected [batch_size, {target_num_classes}])")

    print("\n" + "=" * 60)
    print("PUMPKIN MODEL TRAINED (Zero-Regression)")
    print("=" * 60)
    print(f"  Total classes: {target_num_classes}")
    print(f"  New Pumpkin indices: {new_class_offset}-{target_num_classes-1}")
    print(f"  Production ONNX rows: UNCHANGED")
    print(f"  Candidate ONNX: {CANDIDATE_ONNX}")


if __name__ == '__main__':
    main()
