"""Extract Pumpkin features with batched ONNX inference."""
import os, sys, pickle, copy
from collections import defaultdict
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import onnxruntime as ort
import onnx

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ASSETS = os.path.join(BASE, 'detector', 'ml_assets')
SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pumpkin_split_data_v2')
FEAT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pumpkin_features.pkl')
PRODUCTION_ONNX = os.path.join(ML_ASSETS, 'model.onnx')

PUMPKIN_CLASS_NAMES = sorted([
    'Pumpkin___bacterial_leaf_spot', 'Pumpkin___downy_mildew',
    'Pumpkin___healthy', 'Pumpkin___mosaic_disease', 'Pumpkin___powdery_mildew',
])
TARGET_SIZE, CROP_SIZE = 256, 224
BATCH_SIZE = 64

tform = transforms.Compose([
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_aug = transforms.Compose([
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.RandomCrop(CROP_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if not os.path.exists(SPLIT_DIR):
    print("Split data not found. Run expand_pumpkin_model_v3.py to download & split first.")
    sys.exit(1)

# Create feature extractor
model = onnx.load(PRODUCTION_ONNX)
cls_input = None
for node in model.graph.node:
    if node.op_type in ('Gemm', 'MatMul'):
        for out in model.graph.output:
            if out.name in node.output:
                cls_input = node.input[0]; break
    if cls_input: break

feat_model = copy.deepcopy(model)
feat_out = onnx.helper.make_tensor_value_info(cls_input, onnx.TensorProto.FLOAT, [None, 1280])
feat_model.graph.output.append(feat_out)
feat_path = os.path.join(ML_ASSETS, 'model_features.onnx')
onnx.save(feat_model, feat_path)
sess = ort.InferenceSession(feat_path)
input_name = sess.get_inputs()[0].name
print('Feature extractor ready')

def load_paths():
    train_paths, val_paths = defaultdict(list), defaultdict(list)
    for cls_name in PUMPKIN_CLASS_NAMES:
        for sp, store in [('train', train_paths), ('val', val_paths)]:
            d = os.path.join(SPLIT_DIR, sp, cls_name)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.png','.jpg','.jpeg')):
                        store[cls_name].append(os.path.join(d, f))
    return train_paths, val_paths

train_paths, val_paths = load_paths()
pumpkin_local_map = {cls: i for i, cls in enumerate(PUMPKIN_CLASS_NAMES)}

def extract_batched(paths, transform, double=False):
    all_feats, all_labels = [], []
    for cls_name in PUMPKIN_CLASS_NAMES:
        cls_paths = paths[cls_name]
        label = pumpkin_local_map[cls_name]
        batches = []
        for i in range(0, len(cls_paths), BATCH_SIZE):
            batch_paths = cls_paths[i:i+BATCH_SIZE]
            batch = torch.stack([transform(Image.open(p).convert('RGB')) for p in batch_paths]).numpy()
            out = sess.run(None, {input_name: batch})[1]
            all_feats.append(out)
            all_labels.extend([label] * len(batch_paths))
            if double:
                batch2 = torch.stack([transform(Image.open(p).convert('RGB')) for p in batch_paths]).numpy()
                out2 = sess.run(None, {input_name: batch2})[1]
                all_feats.append(out2)
                all_labels.extend([label] * len(batch_paths))
        print(f'  {cls_name}: {len(cls_paths)} done')
    return np.vstack(all_feats), np.array(all_labels)

print('Extracting training features (batched, double aug)...')
all_train_feats, all_train_labels = extract_batched(train_paths, train_aug, double=True)
print(f'Total train: {len(all_train_feats)}')

print('Extracting validation features...')
all_val_feats, all_val_labels = extract_batched(val_paths, tform, double=False)
print(f'Total val: {len(all_val_feats)}')

with open(FEAT_CACHE, 'wb') as f:
    pickle.dump({
        'train_feats': all_train_feats, 'train_labels': all_train_labels,
        'val_feats': all_val_feats, 'val_labels': all_val_labels,
    }, f)
print(f'Cached to {FEAT_CACHE} ({os.path.getsize(FEAT_CACHE)/1e6:.1f} MB)')
