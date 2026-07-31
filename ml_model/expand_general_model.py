"""
Expand General Model -- Add Cotton classes to the existing EfficientNet-B0 model.

Loads the existing deployed ONNX model weights into PyTorch, expands the
classifier head, trains only the head on mixed data (Option A freeze),
evaluates with per-class metrics, and exports to ONNX.

Usage:
    python ml_model/expand_general_model.py
"""
import os
import sys
import json
import random
import shutil
import subprocess
import glob as glob_mod
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
import onnx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ASSETS_DIR = os.path.join(BASE_DIR, 'detector', 'ml_assets')
ORIGINAL_ONNX_PATH = os.path.join(ML_ASSETS_DIR, 'model.onnx')
ORIGINAL_CLASSES_PATH = os.path.join(ML_ASSETS_DIR, 'class_names.json')
EXPANDED_ONNX_PATH = os.path.join(ML_ASSETS_DIR, 'model.onnx')
EXPANDED_CLASSES_PATH = os.path.join(ML_ASSETS_DIR, 'class_names.json')

COTTON_KAGGLE_ID = "seroshkarim/cotton-leaf-disease-dataset"
COTTON_RAW_NAMES = ['bacterial_blight', 'curl_virus', 'fussarium_wilt', 'healthy']
COTTON_CLASS_NAMES = [
    'Cotton___bacterial_blight',
    'Cotton___curl_virus',
    'Cotton___fussarium_wilt',
    'Cotton___healthy',
]
COTTON_RAW_TO_CLASS = dict(zip(COTTON_RAW_NAMES, COTTON_CLASS_NAMES))

SPLIT_SEED = 42
TRAIN_RATIO = 0.85
IMAGE_SIZE = 224
BATCH_SIZE = 32
HEAD_EPOCHS = 5
HEAD_LR = 1e-3
EXISTING_SAMPLES_PER_CLASS = 20

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class ImagePathDataset(Dataset):
    def __init__(self, samples, num_classes, transform=None):
        self.samples = samples
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


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


def count_images_in_dir(d):
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


def download_cotton_dataset():
    print("\n" + "=" * 60)
    print("STEP 1: Download cotton dataset")
    print("=" * 60)
    import kagglehub
    cotton_raw = kagglehub.dataset_download(COTTON_KAGGLE_ID)
    cotton_dir = cotton_raw
    for candidate in [os.path.join(cotton_raw, "cotton"), cotton_raw]:
        if os.path.isdir(candidate):
            subfolders = [f for f in os.listdir(candidate)
                          if os.path.isdir(os.path.join(candidate, f))]
            if all(rn in subfolders for rn in COTTON_RAW_NAMES):
                cotton_dir = candidate
                break
    print(f"Cotton dataset at: {cotton_dir}")
    total = 0
    for rn in COTTON_RAW_NAMES:
        n = count_images_in_dir(os.path.join(cotton_dir, rn))
        print(f"  {rn:25s} {n:5d} raw images")
        total += n
    print(f"  {'TOTAL':25s} {total:5d}")
    return cotton_dir


def create_cotton_split(cotton_dir, output_base, seed=SPLIT_SEED):
    print("\n" + "=" * 60)
    print(f"STEP 2: Create train/val split (seed={seed})")
    print("=" * 60)
    rng = random.Random(seed)
    train_dir = os.path.join(output_base, 'train')
    val_dir = os.path.join(output_base, 'val')
    file_log = {}

    for raw_name, class_name in COTTON_RAW_TO_CLASS.items():
        src = os.path.join(cotton_dir, raw_name)
        if not os.path.isdir(src):
            print(f"  WARNING: {src} not found, skipping")
            continue
        files = sorted([f for f in os.listdir(src)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        shuffled = list(files)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * TRAIN_RATIO)
        train_files = shuffled[:split_idx]
        val_files = shuffled[split_idx:]

        for split_files, split_dir in [(train_files, train_dir), (val_files, val_dir)]:
            cls_dir = os.path.join(split_dir, class_name)
            os.makedirs(cls_dir, exist_ok=True)
            for f in split_files:
                shutil.copy2(os.path.join(src, f), os.path.join(cls_dir, f))

        file_log[class_name] = {
            'train_files': sorted(train_files),
            'val_files': sorted(val_files),
            'train_count': len(train_files),
            'val_count': len(val_files),
        }
        print(f"  {class_name:40s} train={len(train_files):4d}  val={len(val_files):4d}")

    log_path = os.path.join(output_base, 'split_log.json')
    with open(log_path, 'w') as f:
        json.dump({
            'seed': seed,
            'train_ratio': TRAIN_RATIO,
            'splits': {k: {'train_count': v['train_count'],
                           'val_count': v['val_count'],
                           'train_files': v['train_files'],
                           'val_files': v['val_files']}
                       for k, v in file_log.items()}
        }, f, indent=2)
    print(f"\nSplit log saved to: {log_path}")
    total_train = sum(v['train_count'] for v in file_log.values())
    total_val = sum(v['val_count'] for v in file_log.values())
    print(f"Total cotton: train={total_train}, val={total_val}")
    return train_dir, val_dir, file_log


def load_existing_dataset():
    print("\n" + "=" * 60)
    print("STEP 3: Load existing dataset")
    print("=" * 60)
    import kagglehub

    try:
        dataset_path = kagglehub.dataset_download("nirmalsankalana/plant-diseases-training-dataset")
        data_dir = os.path.join(dataset_path, "data")
    except Exception as e:
        print(f"FATAL: Error downloading main dataset: {e}")
        return None

    extra_datasets = [
        ("aryashah2k/mango-leaf-disease-dataset", "Mango"),
        ("marquis03/plants-classification", "Plant"),
        ("nirmalsankalana/sugarcane-leaf-disease-dataset", "Sugarcane"),
        ("warcoder/potato-leaf-disease-dataset", "Potato"),
        ("arjuntejaswi/plant-village", None),
    ]
    root_dirs_and_prefixes = [(data_dir, None)]
    for kaggle_id, prefix in extra_datasets:
        try:
            d = kagglehub.dataset_download(kaggle_id)
            root_dirs_and_prefixes.append((d, prefix))
            print(f"  Loaded: {kaggle_id}")
        except Exception as e:
            print(f"  WARNING: {kaggle_id} failed: {e}")

    try:
        rice_raw = kagglehub.dataset_download("vbookshelf/rice-leaf-diseases")
        rice_dir = os.path.join(rice_raw, "rice_leaf_diseases")
        if not os.path.exists(rice_dir):
            rice_dir = rice_raw
        root_dirs_and_prefixes.append((rice_dir, "Rice_Add"))
    except Exception as e:
        print(f"  WARNING: rice dataset failed: {e}")

    existing_raw = MultiFolderDataset(root_dirs_and_prefixes, transform=VAL_TRANSFORM)
    print(f"Loaded {len(existing_raw)} images across {len(existing_raw.classes)} raw classes")

    with open(ORIGINAL_CLASSES_PATH, 'r') as f:
        trusted_classes = json.load(f)
    trusted_set = set(trusted_classes)

    filtered_samples = []
    dropped_classes = []
    for path, label in existing_raw.samples:
        cls_name = existing_raw.classes[label]
        if cls_name in trusted_set:
            new_label = trusted_classes.index(cls_name)
            filtered_samples.append((path, new_label))
        else:
            if cls_name not in dropped_classes:
                dropped_classes.append(cls_name)

    if dropped_classes:
        print(f"  DROPPED {len(dropped_classes)} classes not in class_names.json:")
        for c in dropped_classes:
            print(f"    {c}")

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

    existing = ImagePathDatasetDirect(filtered_samples, trusted_classes, transform=VAL_TRANSFORM)
    print(f"Filtered to {len(existing)} images across {len(existing.classes)} trusted classes")
    for cls in existing.classes:
        n = sum(1 for _, l in existing.samples if l == existing.class_to_idx[cls])
        print(f"  {cls:40s} {n:5d}")
    return existing


def load_model_from_onnx(onnx_path):
    print("\n" + "=" * 60)
    print(f"STEP 4: Load model (ImageNet base + ONNX weights)")
    print("=" * 60)
    print("  NOTE: ONNX model was exported with do_constant_folding=True.")
    print("  All BatchNorm layers were fused into Conv weights, making")
    print("  direct weight transfer impossible for Conv/BN layers.")
    print("  Strategy: ImageNet-pretrained base + ONNX SE blocks + classifier.")

    onnx_model = onnx.load(onnx_path)
    onnx_weights = {}
    for init in onnx_model.graph.initializer:
        arr = np.frombuffer(init.raw_data, dtype=np.float32).copy()
        dims = list(init.dims)
        arr = arr.reshape(dims) if dims else arr
        onnx_weights[init.name] = arr
    print(f"  ONNX has {len(onnx_weights)} weight tensors")

    onnx_out = None
    for out in onnx_model.graph.output:
        for d in out.type.tensor_type.shape.dim:
            if d.dim_param == '':
                onnx_out = d.dim_value
                break
    print(f"  ONNX output classes: {onnx_out}")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, onnx_out)

    sd = model.state_dict()
    loaded, skipped = 0, 0
    for name in sd:
        if name in onnx_weights and onnx_weights[name].shape == tuple(sd[name].shape):
            sd[name] = torch.from_numpy(onnx_weights[name])
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(sd, strict=False)
    print(f"  Loaded {loaded} ONNX weights (SE blocks + classifier), {skipped} kept from ImageNet")
    model.eval()
    return model, onnx_out


def evaluate_model(model, dataset, all_class_names, num_model_outputs, device, label=""):
    print(f"\n  Evaluating {label}...")
    model.eval()
    all_preds, all_labels = [], []

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_acc = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"  {label} Overall Accuracy: {overall_acc:.2f}% ({len(all_labels)} samples)")

    per_class = {}
    for idx, cls_name in enumerate(all_class_names):
        if idx >= num_model_outputs:
            break
        mask = all_labels == idx
        if mask.sum() == 0:
            continue
        tp = int(np.sum(all_preds[mask] == idx))
        total = int(mask.sum())
        pred_count = int(np.sum(all_preds == idx))
        recall = tp / total if total > 0 else 0.0
        precision = tp / pred_count if pred_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': total,
        }

    return overall_acc, per_class


def expand_classifier(model, onnx_out, new_class_names, existing_class_names):
    print("\n" + "=" * 60)
    print(f"STEP 5: Expand classifier ({onnx_out} -> {len(new_class_names)})")
    print("=" * 60)

    old_clf = model.classifier[1]
    in_features = old_clf.in_features
    new_clf = nn.Linear(in_features, len(new_class_names))

    with torch.no_grad():
        old_w = old_clf.weight.data
        old_b = old_clf.bias.data

        for new_idx, cls_name in enumerate(new_class_names):
            if cls_name in existing_class_names:
                old_idx = existing_class_names.index(cls_name)
                if old_idx < onnx_out:
                    new_clf.weight.data[new_idx] = old_w[old_idx]
                    new_clf.bias.data[new_idx] = old_b[old_idx]
                    print(f"  Copied: {cls_name} (old[{old_idx}] -> new[{new_idx}])")
                else:
                    nn.init.xavier_uniform_(new_clf.weight.data[new_idx:new_idx+1])
                    new_clf.bias.data[new_idx] = 0.0
                    print(f"  Xavier init (old_idx {old_idx} >= ONNX out): {cls_name}")
            else:
                nn.init.xavier_uniform_(new_clf.weight.data[new_idx:new_idx+1])
                new_clf.bias.data[new_idx] = 0.0
                print(f"  Xavier init (new): {cls_name}")

    model.classifier[1] = new_clf
    print(f"\n  New classifier: Linear({in_features}, {len(new_class_names)})")
    return model


def freeze_and_train(model, cotton_train_dir, existing_dataset, device,
                     existing_class_names, all_class_names):
    print("\n" + "=" * 60)
    print("STEP 6: Freeze base, train classifier head (Option A)")
    print("=" * 60)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")

    all_to_idx = {c: i for i, c in enumerate(all_class_names)}

    cotton_samples = []
    for cls_name in COTTON_CLASS_NAMES:
        cls_dir = os.path.join(cotton_train_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        idx = all_to_idx[cls_name]
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                cotton_samples.append((os.path.join(cls_dir, f), idx))

    existing_by_class = defaultdict(list)
    for path, label in existing_dataset.samples:
        cls_name = existing_class_names[label]
        existing_by_class[cls_name].append(path)

    rng = random.Random(SPLIT_SEED + 1)
    existing_samples = []
    for cls_name in existing_class_names:
        paths = existing_by_class[cls_name]
        n = min(EXISTING_SAMPLES_PER_CLASS, len(paths))
        for p in rng.sample(paths, n):
            existing_samples.append((p, all_to_idx[cls_name]))

    mixed_samples = cotton_samples + existing_samples
    rng.shuffle(mixed_samples)

    print(f"  Cotton train: {len(cotton_samples)} samples")
    print(f"  Existing mix: {len(existing_samples)} samples ({EXISTING_SAMPLES_PER_CLASS}/class x {len(existing_class_names)} classes)")
    print(f"  Total train:  {len(mixed_samples)} samples")

    train_dataset = ImagePathDataset(mixed_samples, len(all_class_names), transform=VAL_TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HEAD_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    ckpt_path = os.path.join(BASE_DIR, 'ml_model', 'expand_best.pth')

    for epoch in range(HEAD_EPOCHS):
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
        avg_loss = running_loss / len(train_loader)
        scheduler.step()

        print(f"  Epoch [{epoch+1}/{HEAD_EPOCHS}]  "
              f"Loss: {avg_loss:.4f}  Acc: {train_acc:.2f}%  "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if train_acc >= best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), ckpt_path)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
        print(f"  Loaded best checkpoint (train acc: {best_acc:.2f}%)")

    return model


def export_to_onnx(model, output_path, device):
    print("\n" + "=" * 60)
    print("STEP 7: Export to ONNX")
    print("=" * 60)
    model.eval()
    model.to(device)
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    torch.onnx.export(
        model, dummy, output_path,
        export_params=True, opset_version=18,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"  Exported to: {output_path}")


def update_class_names(all_class_names):
    print("\n" + "=" * 60)
    print("STEP 8: Update class_names.json")
    print("=" * 60)
    with open(EXPANDED_CLASSES_PATH, 'w') as f:
        json.dump(all_class_names, f, indent=4)
    print(f"  Written {len(all_class_names)} classes")
    print(f"  New: {COTTON_CLASS_NAMES}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(ORIGINAL_CLASSES_PATH, 'r') as f:
        existing_class_names = json.load(f)
    print(f"Original classes in class_names.json: {len(existing_class_names)}")

    all_class_names = sorted(existing_class_names + COTTON_CLASS_NAMES)
    print(f"Expanded classes: {len(all_class_names)} "
          f"(existing={len(existing_class_names)}, new_cotton={len(COTTON_CLASS_NAMES)})")
    print(f"  Cotton classes: {COTTON_CLASS_NAMES}")

    cotton_dir = download_cotton_dataset()

    cotton_split_dir = os.path.join(BASE_DIR, 'ml_model', 'cotton_split_data')
    cotton_train_dir, cotton_val_dir, _ = create_cotton_split(cotton_dir, cotton_split_dir)

    existing_dataset = load_existing_dataset()
    if existing_dataset is None:
        print("FATAL: Could not load existing dataset. Aborting.")
        return

    model, onnx_out = load_model_from_onnx(ORIGINAL_ONNX_PATH)
    model = model.to(device)
    num_existing = len(existing_class_names)
    print(f"  Model loaded: EfficientNet-B0, ONNX outputs={onnx_out}, class_names.json={num_existing}")
    if onnx_out != num_existing:
        print(f"  NOTE: ONNX has {onnx_out} outputs but class_names.json has {num_existing}.")
        print(f"  Using class_names.json ({num_existing}) as source of truth for existing classes.")

    rng_eval = random.Random(SPLIT_SEED + 2)
    existing_by_class = defaultdict(list)
    for path, label in existing_dataset.samples:
        cls_name = existing_class_names[label]
        existing_by_class[cls_name].append(path)

    MAX_EVAL_PER_CLASS = 30
    existing_val_samples = []
    for cls_name in existing_class_names:
        paths = existing_by_class[cls_name]
        n_eval = min(MAX_EVAL_PER_CLASS, len(paths))
        idx = existing_class_names.index(cls_name)
        for p in rng_eval.sample(paths, n_eval):
            existing_val_samples.append((p, idx))
    print(f"  Evaluation samples: {len(existing_val_samples)} ({MAX_EVAL_PER_CLASS}/class)")

    cotton_val_samples = []
    all_to_idx = {c: i for i, c in enumerate(all_class_names)}
    for cls_name in COTTON_CLASS_NAMES:
        cls_dir = os.path.join(cotton_val_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        idx = all_to_idx[cls_name]
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                cotton_val_samples.append((os.path.join(cls_dir, f), idx))

    existing_val_dataset = ImagePathDataset(
        existing_val_samples, len(all_class_names), transform=VAL_TRANSFORM
    )
    combined_val_dataset = ImagePathDataset(
        existing_val_samples + cotton_val_samples, len(all_class_names), transform=VAL_TRANSFORM
    )

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION (original model, existing classes only)")
    print("=" * 60)
    base_acc, base_per_class = evaluate_model(
        model, existing_val_dataset, all_class_names, num_existing, device,
        label="Baseline"
    )

    model = expand_classifier(model, num_existing, all_class_names, existing_class_names)
    model = model.to(device)

    model = freeze_and_train(
        model, cotton_train_dir, existing_dataset, device,
        existing_class_names, all_class_names
    )

    print("\n" + "=" * 60)
    print("POST-EXPANSION EVALUATION (expanded model, all classes)")
    print("=" * 60)
    new_acc, new_per_class = evaluate_model(
        model, combined_val_dataset, all_class_names, len(all_class_names), device,
        label="Expanded"
    )

    print("\n" + "=" * 60)
    print("FORGETTING CHECK: Existing class F1 comparison")
    print("=" * 60)
    print(f"  {'Class':45s} {'Base F1':>8s} {'New F1':>8s} {'Delta':>8s}  Status")
    print("  " + "-" * 82)
    dropped = []
    for cls_name in existing_class_names:
        base_f1 = base_per_class.get(cls_name, {}).get('f1', 0.0) * 100
        new_f1 = new_per_class.get(cls_name, {}).get('f1', 0.0) * 100
        delta = new_f1 - base_f1
        status = "OK" if delta >= -1.0 else ("DROP" if delta >= -5.0 else "SEVERE")
        if delta < -1.0:
            dropped.append((cls_name, delta))
        print(f"  {cls_name:45s} {base_f1:7.2f}% {new_f1:7.2f}% {delta:+7.2f}%  {status}")

    print("\n" + "=" * 60)
    print("NEW COTTON CLASS METRICS")
    print("=" * 60)
    for cls_name in COTTON_CLASS_NAMES:
        m = new_per_class.get(cls_name, {})
        if m:
            print(f"  {cls_name:40s} P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"F1={m['f1']:.4f}  support={m['support']}")
        else:
            print(f"  {cls_name:40s} (no validation samples)")

    if dropped:
        print(f"\nWARNING: {len(dropped)} existing classes dropped >1% F1:")
        for cls, delta in sorted(dropped, key=lambda x: x[1]):
            print(f"  {cls}: {delta:+.2f}%")
    else:
        print("\nNo existing classes dropped >1% F1. Forgetting check PASSED.")

    print(f"\nOverall: baseline={base_acc:.2f}% -> expanded={new_acc:.2f}%")

    export_to_onnx(model, EXPANDED_ONNX_PATH, device)
    update_class_names(all_class_names)

    ckpt = os.path.join(BASE_DIR, 'ml_model', 'expand_best.pth')
    if os.path.exists(ckpt):
        os.remove(ckpt)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Expanded ONNX: {EXPANDED_ONNX_PATH}")
    print(f"  Updated classes: {EXPANDED_CLASSES_PATH} ({len(all_class_names)} classes)")


if __name__ == '__main__':
    main()
