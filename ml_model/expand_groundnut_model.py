"""
Expand General Model -- Add Groundnut classes to the existing EfficientNet-B0 model.

Downloads the Groundnut dataset, merges Early Leaf Spot + Late Leaf Spot -> Leaf Spot
and Early Rust + Rust -> Rust (50:50 each), keeps Healthy and Nutrition Deficiency,
keeps ALL images (no undersampling), uses WeightedRandomSampler for balanced batches,
expands the classifier head, trains only the head, and exports to ONNX.

Usage:
    python ml_model/expand_groundnut_model.py
"""
import os
import sys
import json
import random
import glob as glob_mod
import shutil
import subprocess
from collections import defaultdict, Counter

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np
import onnx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ASSETS_DIR = os.path.join(BASE_DIR, 'detector', 'ml_assets')
ONNX_PATH = os.path.join(ML_ASSETS_DIR, 'model.onnx')
CANDIDATE_ONNX_PATH = os.path.join(ML_ASSETS_DIR, 'groundnut_candidate.onnx')
CLASSES_PATH = os.path.join(ML_ASSETS_DIR, 'class_names.json')
SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'groundnut_split_data')

GROUNDNUT_KAGGLE_ID = "warcoder/groundnut-plant-leaf-data"

# Raw folder names in the Kaggle dataset
RAW_EARLY_LEAF_SPOT = "Early Leaf Spot"
RAW_LATE_LEAF_SPOT = "Late Leaf Spot"
RAW_EARLY_RUST = "Early Rust"
RAW_RUST = "Rust"
RAW_HEALTHY = "Healthy"
RAW_NUTRITION_DEFICIENCY = "Nutrition Deficiency"

# Raw folder names mapped to possible Kaggle variations
RAW_FOLDER_CANDIDATES = {
    "Early Leaf Spot": ["Early Leaf Spot", "Early_Leaf_Spot", "early_leaf_spot"],
    "Late Leaf Spot": ["Late Leaf Spot", "Late_Leaf_Spot", "late_leaf_spot", "late leaf spot"],
    "Early Rust": ["Early Rust", "Early_Rust", "early_rust"],
    "Rust": ["Rust", "rust"],
    "Healthy": ["Healthy", "healthy", "healthy leaf"],
    "Nutrition Deficiency": ["Nutrition Deficiency", "Nutrition_Deficiency", "nutrition_deficiency", "nutrition deficiency"],
}

# How raw folders map to merged Groundnut classes
RAW_TO_GROUNDNUT_CLASS = {
    "Early Leaf Spot": "Groundnut___Leaf_Spot",
    "Late Leaf Spot": "Groundnut___Leaf_Spot",
    "Early Rust": "Groundnut___Rust",
    "Rust": "Groundnut___Rust",
    "Healthy": "Groundnut___Healthy",
    "Nutrition Deficiency": "Groundnut___Nutrition_Deficiency",
}

GROUNDNUT_CLASS_NAMES = sorted(set(RAW_TO_GROUNDNUT_CLASS.values()))

HEAD_EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 32
SEED = 42

# Parse CLI flags
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

CAP_SAMPLES = None if _parse_flag('no-cap', False) else 226
HEAD_EPOCHS = _parse_flag('epochs', HEAD_EPOCHS)
LR = _parse_flag('lr', LR)
USING_CLEAN = '--train-clean' in sys.argv
NUM_WORKERS = 0
TARGET_SIZE = 256
CROP_SIZE = 224
TRAIN_RATIO = 0.85
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


def find_groundnut_dir(dataset_path):
    """Walk the downloaded dataset to find the directory with the raw class folders."""
    for root, dirs, _ in os.walk(dataset_path):
        dir_set = set(dirs)
        # Check if any of our expected folder names appear
        for canonical, candidates in RAW_FOLDER_CANDIDATES.items():
            for c in candidates:
                if c in dir_set:
                    return root
    return dataset_path


def download_and_preprocess_groundnut():
    print("\n" + "=" * 60)
    print("STEP 1: Download and preprocess Groundnut dataset")
    print("=" * 60)

    train_dir = os.path.join(SPLIT_DIR, 'train')
    val_dir = os.path.join(SPLIT_DIR, 'val')

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        existing = sum(len(files) for _, _, files in os.walk(train_dir)) + \
                   sum(len(files) for _, _, files in os.walk(val_dir))
        if existing > 0:
            print(f"Split data already exists at {SPLIT_DIR} ({existing} files), skipping download.")
            return train_dir, val_dir

    print("Downloading Groundnut dataset from Kaggle...")
    import kagglehub
    dataset_path = kagglehub.dataset_download(GROUNDNUT_KAGGLE_ID)
    print(f"Dataset downloaded to: {dataset_path}")

    groundnut_dir = find_groundnut_dir(dataset_path)
    print(f"Groundnut data directory: {groundnut_dir}")

    # Discover raw folders (try to match canonical names)
    raw_dirs = {}
    for item in os.listdir(groundnut_dir):
        item_path = os.path.join(groundnut_dir, item)
        if not os.path.isdir(item_path):
            continue
        for canonical, candidates in RAW_FOLDER_CANDIDATES.items():
            if item in candidates:
                raw_dirs[canonical] = item_path
                break

    print(f"Found raw folders: {list(raw_dirs.keys())}")
    for canonical, path in sorted(raw_dirs.items()):
        n = count_images_in_dir(path)
        print(f"  {canonical:30s} {n:5d} images")

    # Merge into Groundnut classes (50:50 contribution for merged classes)
    groundnut_samples = defaultdict(list)
    for canonical, src_path in raw_dirs.items():
        groundnut_class = RAW_TO_GROUNDNUT_CLASS[canonical]
        images = [os.path.join(src_path, f) for f in os.listdir(src_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        groundnut_samples[groundnut_class].extend(images)

    # Print counts (no undersampling — keep all images)
    print("\nGroundnut image counts (all kept):")
    for cls in sorted(groundnut_samples.keys()):
        print(f"  {cls:40s} {len(groundnut_samples[cls]):5d} images")

    # Create train/val split (keep all images per class)
    rng = random.Random(SEED)
    train_samples = defaultdict(list)
    val_samples = defaultdict(list)

    for cls, imgs in sorted(groundnut_samples.items()):
        imgs_sorted = sorted(imgs)
        rng.shuffle(imgs_sorted)
        split_idx = max(1, int(len(imgs_sorted) * TRAIN_RATIO))
        train_samples[cls] = imgs_sorted[:split_idx]
        val_samples[cls] = imgs_sorted[split_idx:]

    print(f"\nTrain/val split (seed={SEED}, ratio={TRAIN_RATIO}):")
    total_train = 0
    total_val = 0
    for cls in sorted(train_samples.keys()):
        t = len(train_samples[cls])
        v = len(val_samples[cls])
        total_train += t
        total_val += v
        print(f"  {cls:40s} train={t:4d}  val={v:4d}")
    print(f"  {'TOTAL':40s} train={total_train:4d}  val={total_val:4d}")

    # Copy files to split directory
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    for split_name, samples_dict in [('train', train_samples), ('val', val_samples)]:
        for cls, paths in samples_dict.items():
            cls_dir = os.path.join(SPLIT_DIR, split_name, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for src_path in paths:
                dest = os.path.join(cls_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dest)

    print(f"\nSplit data saved to {SPLIT_DIR}")
    return train_dir, val_dir


def load_onnx_weights(onnx_path, pytorch_model):
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx_weights = {}
    for w in onnx_model.graph.initializer:
        try:
            # Handle different data types (int64 for num_batches_tracked, float32 for everything else)
            elem_type = w.data_type
            if elem_type == 7:  # int64
                arr = np.frombuffer(w.raw_data, dtype=np.int64).reshape(list(w.dims))
                onnx_weights[w.name] = torch.from_numpy(arr).float()
            else:  # float32 (type 1)
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


def train_head(model, train_loader, val_loader, num_epochs, device, criterion=None):
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

    best_val_acc = 0.0
    best_ckpt = os.path.join(SPLIT_DIR, 'best_model.pth')
    no_improve_count = 0
    patience = 3

    for epoch in range(num_epochs):
        model.train()
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

        # Save checkpoint every epoch
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
    print("STEP 5: Export to ONNX and update class names")
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

    # Check for retrain flag
    retrain = '--retrain' in sys.argv

    # Groundnut classes to add
    new_groundnut_classes = [c for c in GROUNDNUT_CLASS_NAMES if c not in existing_classes]
    print(f"New Groundnut classes to add: {new_groundnut_classes}")

    if not new_groundnut_classes and not retrain:
        print("All Groundnut classes already present. Nothing to do. Use --retrain to force retraining.")
        return

    all_class_names = existing_classes
    if new_groundnut_classes:
        all_class_names = existing_classes + new_groundnut_classes
    target_num_classes = len(all_class_names)

    # Download, preprocess, and split Groundnut dataset
    use_clean = '--train-clean' in sys.argv
    if use_clean:
        CLEAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'groundnut_cleaned')
        CLEAN_SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'groundnut_cleaned_split')
        if not os.path.exists(CLEAN_DIR):
            print(f"Cleaned dataset not found at {CLEAN_DIR}. Run build_cleaned_data.py first.")
            return
        if os.path.exists(CLEAN_SPLIT_DIR):
            shutil.rmtree(CLEAN_SPLIT_DIR)
        # Split 85/15
        rng_split = random.Random(SEED)
        for cls_folder in sorted(os.listdir(CLEAN_DIR)):
            src = os.path.join(CLEAN_DIR, cls_folder)
            if not os.path.isdir(src): continue
            imgs = sorted([f for f in os.listdir(src) if f.lower().endswith(('.png','.jpg','.jpeg'))])
            rng_split.shuffle(imgs)
            split_idx = max(1, int(len(imgs) * TRAIN_RATIO))
            for split_name, subset in [('train', imgs[:split_idx]), ('val', imgs[split_idx:])]:
                dest = os.path.join(CLEAN_SPLIT_DIR, split_name, f'Groundnut___{cls_folder}')
                os.makedirs(dest, exist_ok=True)
                for fn in subset:
                    shutil.copy2(os.path.join(src, fn), os.path.join(dest, fn))
        train_dir = os.path.join(CLEAN_SPLIT_DIR, 'train')
        val_dir = os.path.join(CLEAN_SPLIT_DIR, 'val')
        total_t = sum(len(files) for _,_,files in os.walk(train_dir))
        total_v = sum(len(files) for _,_,files in os.walk(val_dir))
        print(f"Using cleaned dataset: {total_t} train, {total_v} val")
    else:
        train_dir, val_dir = download_and_preprocess_groundnut()

    # Load samples from split directory
    train_samples = []
    val_samples = []
    cls_to_idx = {c: i for i, c in enumerate(all_class_names)}

    for cls_name in GROUNDNUT_CLASS_NAMES:
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

    # Cap or oversample Groundnut classes
    CAP_SIZE = CAP_SAMPLES
    over_sample = _parse_flag('oversample', False)
    if over_sample and not CAP_SIZE:
        # Oversample minority classes to match majority
        cls_groups = {}
        for cls_name in GROUNDNUT_CLASS_NAMES:
            cls_idx = cls_to_idx[cls_name]
            cls_imgs = [(p, l) for p, l in train_samples if l == cls_idx]
            cls_groups[cls_name] = cls_imgs
        max_count = max(len(v) for v in cls_groups.values())
        balanced = []
        for cls_name, cls_imgs in cls_groups.items():
            n = len(cls_imgs)
            if n < max_count:
                repeats = (max_count + n - 1) // n
                cls_imgs = (cls_imgs * repeats)[:max_count]
                print(f"  Oversampling {cls_name} from {n} to {len(cls_imgs)}")
            balanced.extend(cls_imgs)
        train_samples = balanced
        rng.shuffle(train_samples)
        print(f"Groundnut train samples (oversampled): {len(train_samples)}")
    elif CAP_SIZE is not None:
        capped = []
        for cls_name in GROUNDNUT_CLASS_NAMES:
            cls_idx = cls_to_idx[cls_name]
            cls_imgs = [(p, l) for p, l in train_samples if l == cls_idx]
            if len(cls_imgs) > CAP_SIZE:
                print(f"  Capping {cls_name} from {len(cls_imgs)} to {CAP_SIZE}")
                cls_imgs = rng.sample(cls_imgs, CAP_SIZE)
            capped.extend(cls_imgs)
        train_samples = capped
        rng.shuffle(train_samples)
        print(f"Groundnut train samples (capped): {len(train_samples)}")
    else:
        print(f"Groundnut train samples (no cap): {len(train_samples)}")

    # Load representative existing samples
    existing_dataset, existing_sampled = load_existing_dataset()
    if existing_sampled:
        # Remap existing sample labels to the new combined index space
        existing_to_idx = {c: i for i, c in enumerate(existing_classes)}
        remapped = []
        for path, old_label in existing_sampled:
            cls_name = existing_classes[old_label]
            remapped.append((path, existing_to_idx[cls_name]))
        train_samples.extend(remapped)
        rng.shuffle(train_samples)
        print(f"Added {len(remapped)} existing samples to training set")
    else:
        print("Proceeding with Groundnut-only training (existing dataset unavailable)")

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

    # Match inference pipeline: Resize(256) + CenterCrop(224)
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

    # Train
    use_weighted = _parse_flag('weighted', False)
    if use_weighted and not CAP_SAMPLES:
        # Compute class weights for full dataset (no cap)
        cls_counts = defaultdict(int)
        gn_indices = set()
        for cls_name in GROUNDNUT_CLASS_NAMES:
            cls_idx = cls_to_idx[cls_name]
            gn_indices.add(cls_idx)
        for path, label in train_samples:
            if label in gn_indices:
                cls_counts[label] += 1
        if cls_counts:
            total_gn = sum(cls_counts.values())
            weights = torch.ones(len(all_class_names), device=device)
            for cls_idx, count in cls_counts.items():
                weights[cls_idx] = total_gn / (len(cls_counts) * count)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = train_head(model, train_loader, val_loader, HEAD_EPOCHS, device, criterion=criterion)

    # Export to production model.onnx
    export_onnx(model, ONNX_PATH, all_class_names, device)

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH)
    out = sess.get_outputs()[0]
    print(f"ONNX output shape: {out.shape} (expected [batch_size, {len(all_class_names)}])")

    print("\n" + "=" * 60)
    print("GROUNDNUT MODEL RESTORED")
    print("=" * 60)
    print(f"  Total classes: {len(all_class_names)}")
    print(f"  ONNX model: {ONNX_PATH}")
    print(f"  Class names: {CLASSES_PATH}")


if __name__ == '__main__':
    main()
