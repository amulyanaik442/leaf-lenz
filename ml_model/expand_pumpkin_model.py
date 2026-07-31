"""
Expand General Model -- Add Pumpkin classes to the existing EfficientNet-B0 model.

Downloads the Pumpkin dataset, merges Original + Augmented_train for training,
uses Augmented_valid for validation, Augmented_test for testing,
expands the classifier head, trains only the head, and exports to ONNX candidate.

Usage:
    python ml_model/expand_pumpkin_model.py
"""
import os
import sys
import json
import random
import shutil
import subprocess
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
from torchvision import transforms, models
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ASSETS_DIR = os.path.join(BASE_DIR, 'detector', 'ml_assets')
ONNX_PATH = os.path.join(ML_ASSETS_DIR, 'model.onnx')
CANDIDATE_ONNX_PATH = os.path.join(ML_ASSETS_DIR, 'pumpkin_candidate.onnx')
CLASSES_PATH = os.path.join(ML_ASSETS_DIR, 'class_names.json')
SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pumpkin_split_data')

PUMPKIN_KAGGLE_ID = "rifat963/pumpkin"

RAW_FOLDER_CANDIDATES = {
    "Bacterial Leaf Spot": ["Bacterial Leaf Spot", "Bacterial_Leaf_Spot", "bacterial_leaf_spot"],
    "Downy Mildew": ["Downy Mildew", "Downy_Mildew", "downy_mildew"],
    "Healthy Leaf": ["Healthy Leaf", "Healthy_Leaf", "healthy leaf"],
    "Mosaic Disease": ["Mosaic Disease", "Mosaic_Disease", "mosaic_disease"],
    "Powdery Mildew": ["Powdery Mildew", "Powdery_Mildew", "powdery_mildew"],
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


def _parse_flag(name, default):
    for i, arg in enumerate(sys.argv):
        if arg.startswith(f'--{name}='):
            return type(default)(arg.split('=', 1)[1])
        if arg == f'--{name}' and isinstance(default, bool):
            return not default
    return default


SEED = _parse_flag('seed', SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

HEAD_EPOCHS = _parse_flag('epochs', HEAD_EPOCHS)
LR = _parse_flag('lr', LR)
NUM_WORKERS = 0
TARGET_SIZE = 256
CROP_SIZE = 224
EXISTING_SAMPLES_PER_CLASS = 20


class ImageFileDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def count_images_in_dir(d):
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


def find_raw_class_dir(root_path, canonical_name):
    """Find a folder under root_path matching any candidate name for a class."""
    candidates = RAW_FOLDER_CANDIDATES[canonical_name]
    for item in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, item)):
            for c in candidates:
                if item == c:
                    return os.path.join(root_path, item)
    return None


def download_and_preprocess_pumpkin():
    print("\n" + "=" * 60)
    print("STEP 1: Download and preprocess Pumpkin dataset")
    print("=" * 60)

    train_dir = os.path.join(SPLIT_DIR, 'train')
    val_dir = os.path.join(SPLIT_DIR, 'val')
    test_dir = os.path.join(SPLIT_DIR, 'test')

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        existing = sum(len(files) for _, _, files in os.walk(train_dir)) + \
                   sum(len(files) for _, _, files in os.walk(val_dir))
        if existing > 0:
            print(f"Split data already exists at {SPLIT_DIR} ({existing} files), skipping download.")
            return train_dir, val_dir, test_dir

    print("Downloading Pumpkin dataset from Kaggle...")
    import kagglehub
    dataset_path = kagglehub.dataset_download(PUMPKIN_KAGGLE_ID)
    print(f"Dataset downloaded to: {dataset_path}")

    original_dir = os.path.join(dataset_path, 'Original', 'Original')
    aug_train_dir = os.path.join(dataset_path, 'Augmented', 'Augmented', 'train')
    aug_val_dir = os.path.join(dataset_path, 'Augmented', 'Augmented', 'valid')
    aug_test_dir = os.path.join(dataset_path, 'Augmented', 'Augmented', 'test')

    # Collect all images per class from each source
    class_images = defaultdict(lambda: {'train': [], 'val': [], 'test': []})

    for canonical, pumpkin_class in RAW_TO_PUMPKIN_CLASS.items():
        # Original (all go to train)
        orig_class_dir = find_raw_class_dir(original_dir, canonical)
        if orig_class_dir:
            for f in os.listdir(orig_class_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_images[canonical]['train'].append(os.path.join(orig_class_dir, f))

        # Augmented train
        aug_train_class = find_raw_class_dir(aug_train_dir, canonical)
        if aug_train_class:
            for f in os.listdir(aug_train_class):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_images[canonical]['train'].append(os.path.join(aug_train_class, f))

        # Augmented val
        aug_val_class = find_raw_class_dir(aug_val_dir, canonical)
        if aug_val_class:
            for f in os.listdir(aug_val_class):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_images[canonical]['val'].append(os.path.join(aug_val_class, f))

        # Augmented test
        aug_test_class = find_raw_class_dir(aug_test_dir, canonical)
        if aug_test_class:
            for f in os.listdir(aug_test_class):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_images[canonical]['test'].append(os.path.join(aug_test_class, f))

    print("\nPumpkin image counts:")
    total_train, total_val, total_test = 0, 0, 0
    for canonical in sorted(RAW_TO_PUMPKIN_CLASS.keys()):
        pumpkin_class = RAW_TO_PUMPKIN_CLASS[canonical]
        t = len(class_images[canonical]['train'])
        v = len(class_images[canonical]['val'])
        te = len(class_images[canonical]['test'])
        total_train += t
        total_val += v
        total_test += te
        print(f"  {pumpkin_class:45s} train={t:4d}  val={v:4d}  test={te:4d}")
    print(f"  {'TOTAL':45s} train={total_train:4d}  val={total_val:4d}  test={total_test:4d}")

    # Copy to split directory
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    for split_name in ['train', 'val', 'test']:
        for canonical, pumpkin_class in RAW_TO_PUMPKIN_CLASS.items():
            paths = class_images[canonical][split_name]
            if not paths:
                continue
            cls_dir = os.path.join(SPLIT_DIR, split_name, pumpkin_class)
            os.makedirs(cls_dir, exist_ok=True)
            for src_path in paths:
                dest = os.path.join(cls_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dest)

    print(f"\nSplit data saved to {SPLIT_DIR}")
    return train_dir, val_dir, test_dir


def get_onnx_output_classes(onnx_path):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    return sess.get_outputs()[0].shape[1]


def create_model(onnx_num_classes, target_num_classes):
    print("\n" + "=" * 60)
    print(f"STEP 2: Create model ({onnx_num_classes} -> {target_num_classes} classes)")
    print("=" * 60)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(in_features, onnx_num_classes)
    print("Loading ONNX weights...")
    model = load_onnx_weights(ONNX_PATH, model)

    if target_num_classes > onnx_num_classes:
        old_fc = model.classifier[1]
        new_fc = nn.Linear(in_features, target_num_classes)
        with torch.no_grad():
            new_fc.weight[:onnx_num_classes] = old_fc.weight
            new_fc.bias[:onnx_num_classes] = old_fc.bias
            nn.init.xavier_uniform_(new_fc.weight[onnx_num_classes:])
            nn.init.zeros_(new_fc.bias[onnx_num_classes:])
        model.classifier[1] = new_fc
        print(f"Expanded classifier: {onnx_num_classes} -> {target_num_classes} classes")
    else:
        print(f"Classifier already has {target_num_classes} outputs, no expansion needed")

    for param in model.features.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    return model


def load_onnx_weights(onnx_path, pytorch_model):
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx_weights = {}
    for w in onnx_model.graph.initializer:
        try:
            elem_type = w.data_type
            if elem_type == 7:
                arr = np.frombuffer(w.raw_data, dtype=np.int64).reshape(list(w.dims))
                onnx_weights[w.name] = torch.from_numpy(arr).float()
            else:
                arr = np.frombuffer(w.raw_data, dtype=np.float32).reshape(list(w.dims))
                onnx_weights[w.name] = torch.from_numpy(arr)
        except Exception as e:
            print(f"  Skipping ONNX weight '{w.name}': {e}")

    pt_sd = pytorch_model.state_dict()
    loaded = 0
    skipped = 0
    for pt_key in pt_sd.keys():
        if pt_key in onnx_weights:
            if pt_sd[pt_key].shape == onnx_weights[pt_key].shape:
                pt_sd[pt_key] = onnx_weights[pt_key]
                loaded += 1
            else:
                skipped += 1
        else:
            skipped += 1

    pytorch_model.load_state_dict(pt_sd)
    print(f"ONNX weights loaded: {loaded} keys matched, {skipped} skipped")
    return pytorch_model


def load_existing_dataset():
    """Load a representative sample of existing classes for mixed training."""
    print("\n" + "=" * 60)
    print("STEP 3: Load representative samples of existing classes")
    print("=" * 60)

    with open(CLASSES_PATH, 'r') as f:
        existing_classes = json.load(f)
    print(f"Existing classes: {len(existing_classes)}")

    import kagglehub
    try:
        dataset_path = kagglehub.dataset_download("nirmalsankalana/plant-diseases-training-dataset")
        data_dir = os.path.join(dataset_path, "data")
    except Exception as e:
        print(f"WARNING: Could not load existing dataset: {e}")
        return None, None

    extra_datasets = [
        ("aryashah2k/mango-leaf-disease-dataset", "Mango"),
        ("marquis03/plants-classification", "Plant"),
        ("nirmalsankalana/sugarcane-leaf-disease-dataset", "Sugarcane"),
        ("warcoder/potato-leaf-disease-dataset", "Potato"),
        ("arjuntejaswi/plant-village", None),
    ]

    import glob as glob_mod

    class MultiFolderDataset(Dataset):
        def __init__(self, root_dirs_and_prefixes, transform=None):
            self.samples = []
            self.transform = transform
            class_names = set()
            for root_dir, mapping in root_dirs_and_prefixes:
                if not os.path.exists(root_dir):
                    continue
                for folder in os.listdir(root_dir):
                    folder_path = os.path.join(root_dir, folder)
                    if not os.path.isdir(folder_path):
                        continue
                    global_class = self._get_class_name(folder, mapping)
                    if global_class:
                        class_names.add(global_class)
            self.classes = sorted(list(class_names))
            self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
            for root_dir, mapping in root_dirs_and_prefixes:
                if not os.path.exists(root_dir):
                    continue
                for folder in os.listdir(root_dir):
                    folder_path = os.path.join(root_dir, folder)
                    if not os.path.isdir(folder_path):
                        continue
                    global_class = self._get_class_name(folder, mapping)
                    if global_class and global_class in self.class_to_idx:
                        idx = self.class_to_idx[global_class]
                        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                            for img_path in glob_mod.glob(os.path.join(folder_path, ext)):
                                self.samples.append((img_path, idx))

        @staticmethod
        def _get_class_name(folder, mapping):
            if mapping is None:
                return folder
            elif isinstance(mapping, str):
                return f"{mapping}___{folder.replace(' ', '_')}"
            elif isinstance(mapping, dict):
                return mapping.get(folder)
            return folder

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, target = self.samples[idx]
            sample = Image.open(path).convert('RGB')
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target

    class ImagePathDatasetDirect(Dataset):
        def __init__(self, samples, classes, transform=None):
            self.samples = samples
            self.classes = classes
            self.class_to_idx = {name: i for i, name in enumerate(classes)}
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            path, target = self.samples[idx]
            sample = Image.open(path).convert('RGB')
            if self.transform:
                sample = self.transform(sample)
            return sample, target

    val_transform = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    root_dirs_and_prefixes = [(data_dir, None)]
    for kaggle_id, prefix in extra_datasets:
        try:
            d = kagglehub.dataset_download(kaggle_id)
            root_dirs_and_prefixes.append((d, prefix))
            print(f"  Loaded: {kaggle_id}")
        except Exception as e:
            print(f"  WARNING: {kaggle_id} failed: {e}")

    rice_raw = kagglehub.dataset_download("vbookshelf/rice-leaf-diseases")
    rice_dir = os.path.join(rice_raw, "rice_leaf_diseases")
    if not os.path.exists(rice_dir):
        rice_dir = rice_raw
    root_dirs_and_prefixes.append((rice_dir, "Rice_Add"))

    existing_raw = MultiFolderDataset(root_dirs_and_prefixes, transform=val_transform)
    print(f"Loaded {len(existing_raw)} images across {len(existing_raw.classes)} raw classes")

    trusted_set = set(existing_classes)
    filtered_samples = []
    for path, label in existing_raw.samples:
        cls_name = existing_raw.classes[label]
        if cls_name in trusted_set:
            new_label = existing_classes.index(cls_name)
            filtered_samples.append((path, new_label))

    existing = ImagePathDatasetDirect(filtered_samples, existing_classes, transform=val_transform)
    print(f"Filtered to {len(existing)} images across {len(existing.classes)} trusted classes")

    # Stratified sample ~20 per class
    rng = random.Random(SEED + 1)
    by_class = defaultdict(list)
    for path, label in existing.samples:
        cls_name = existing.classes[label]
        by_class[cls_name].append((path, label))

    sampled = []
    for cls_name in existing.classes:
        paths = by_class[cls_name]
        n = min(EXISTING_SAMPLES_PER_CLASS, len(paths))
        sampled.extend(rng.sample(paths, n))

    print(f"Sampled {len(sampled)} representative existing images ({EXISTING_SAMPLES_PER_CLASS}/class)")
    return existing, sampled


def train_head(model, train_loader, val_loader, num_epochs, device, criterion=None, frozen_rows=None):
    print("\n" + "=" * 60)
    print("STEP 4: Train classifier head")
    print("=" * 60)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_count:,} / {total_params:,} ({100*trainable_count/total_params:.2f}%)")

    optimizer = optim.Adam(trainable_params, lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    if criterion is None:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Save frozen row weights to restore after each optimizer step
    if frozen_rows is not None:
        frozen_weight = model.classifier[1].weight[:frozen_rows].data.clone()
        frozen_bias = model.classifier[1].bias[:frozen_rows].data.clone()

    best_val_acc = 0.0
    best_ckpt = os.path.join(SPLIT_DIR, 'best_model.pth')
    no_improve_count = 0
    patience = 3

    for epoch in range(num_epochs):
        model.train()
        model.features.eval()  # Keep BN stats frozen (must be after model.train())
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Restore frozen rows to prevent regression in existing classes
            if frozen_rows is not None:
                with torch.no_grad():
                    model.classifier[1].weight[:frozen_rows] = frozen_weight
                    model.classifier[1].bias[:frozen_rows] = frozen_bias

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.2f}%")

        epoch_ckpt = os.path.join(SPLIT_DIR, f'epoch_{epoch+1:02d}.pth')
        torch.save(model.state_dict(), epoch_ckpt)
        print(f"  -> Saved {epoch_ckpt}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> New best model (val_acc={val_acc:.2f}%)")
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\n  Early stopping triggered after {epoch+1} epochs (no improvement for {patience} consecutive epochs)")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, weights_only=True, map_location=device))
        print("Loaded best checkpoint")

    return model


def export_onnx(model, onnx_path, class_names, device):
    print("\n" + "=" * 60)
    print("STEP 5: Export to ONNX candidate and update class names")
    print("=" * 60)

    model.eval()
    dummy_input = torch.randn(1, 3, CROP_SIZE, CROP_SIZE, device=device)

    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=18,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"ONNX model exported to {onnx_path}")

    with open(CLASSES_PATH, 'w') as f:
        json.dump(class_names, f, indent=4)
    print(f"Class names saved to {CLASSES_PATH} ({len(class_names)} classes)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load existing class names
    with open(CLASSES_PATH, 'r') as f:
        existing_classes = json.load(f)
    print(f"Existing classes in class_names.json: {len(existing_classes)}")

    # Determine ONNX output size
    onnx_num_classes = get_onnx_output_classes(ONNX_PATH)
    print(f"ONNX model has {onnx_num_classes} output classes")

    # Pumpkin classes to add
    new_pumpkin_classes = [c for c in PUMPKIN_CLASS_NAMES if c not in existing_classes]
    print(f"New Pumpkin classes to add: {new_pumpkin_classes}")

    if not new_pumpkin_classes:
        print("All Pumpkin classes already present. Nothing to do.")
        return

    all_class_names = existing_classes + new_pumpkin_classes
    target_num_classes = len(all_class_names)

    # Download, preprocess, and split Pumpkin dataset
    train_dir, val_dir, test_dir = download_and_preprocess_pumpkin()

    # Load samples from split directory
    train_samples = []
    val_samples = []
    cls_to_idx = {c: i for i, c in enumerate(all_class_names)}

    for cls_name in PUMPKIN_CLASS_NAMES:
        cls_idx = cls_to_idx[cls_name]
        for split_name, samples_list in [('train', train_samples), ('val', val_samples)]:
            cls_dir = os.path.join(SPLIT_DIR, split_name, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples_list.append((os.path.join(cls_dir, f), cls_idx))

    rng = random.Random(SEED + 2)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    # Pumpkin is already well-balanced, no capping/oversampling needed
    print(f"\nPumpkin train samples: {len(train_samples)}")
    print(f"Pumpkin val samples: {len(val_samples)}")

    # Load representative existing samples
    existing_dataset, existing_sampled = load_existing_dataset()
    if existing_sampled:
        existing_to_idx = {c: i for i, c in enumerate(existing_classes)}
        remapped = []
        for path, old_label in existing_sampled:
            cls_name = existing_classes[old_label]
            remapped.append((path, existing_to_idx[cls_name]))
        train_samples.extend(remapped)
        rng.shuffle(train_samples)
        print(f"Added {len(remapped)} existing samples to training set")
    else:
        print("Proceeding with Pumpkin-only training (existing dataset unavailable)")

    print(f"Total training samples: {len(train_samples)}")

    # Create datasets
    train_transform = transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFileDataset(train_samples, transform=train_transform)
    val_dataset = ImageFileDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    # Create model
    model = create_model(onnx_num_classes, target_num_classes)
    model = model.to(device)

    print(f"Will freeze first {onnx_num_classes} classifier rows to prevent regression. Training only {target_num_classes - onnx_num_classes} new Pumpkin rows.")

    # Train
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = train_head(model, train_loader, val_loader, HEAD_EPOCHS, device, criterion=criterion, frozen_rows=onnx_num_classes)

    # Export to candidate ONNX (NOT overwriting production model.onnx)
    export_onnx(model, CANDIDATE_ONNX_PATH, all_class_names, device)

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(CANDIDATE_ONNX_PATH)
    out = sess.get_outputs()[0]
    print(f"ONNX output shape: {out.shape} (expected [batch_size, {len(all_class_names)}])")

    print("\n" + "=" * 60)
    print("PUMPKIN MODEL TRAINED")
    print("=" * 60)
    print(f"  Total classes: {len(all_class_names)}")
    print(f"  Candidate ONNX: {CANDIDATE_ONNX_PATH}")
    print(f"  Class names: {CLASSES_PATH}")
    print(f"  Pumpkin classes (indices {onnx_num_classes}-{target_num_classes-1}):")
    for i, cls in enumerate(new_pumpkin_classes):
        print(f"    [{onnx_num_classes + i}] {cls}")


if __name__ == '__main__':
    main()
