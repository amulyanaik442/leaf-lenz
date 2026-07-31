import os, json, sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch
import torch.nn as nn
from torchvision import models
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ASSETS = os.path.join(BASE_DIR, 'detector', 'ml_assets')
ONNX_PATH = os.path.join(ML_ASSETS, 'model.onnx')
CLASSES_PATH = os.path.join(ML_ASSETS, 'class_names.json')
CKPT_PATH = os.path.join(BASE_DIR, 'ml_model', 'groundnut_split_data', 'best_model.pth')

with open(CLASSES_PATH) as f:
    class_names = json.load(f)
num_classes = len(class_names)
print(f"Classes: {num_classes}")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu', weights_only=True))
model.eval()
print(f"Loaded checkpoint")

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, ONNX_PATH, export_params=True, opset_version=18,
    do_constant_folding=True, input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print(f"Exported: {os.path.getsize(ONNX_PATH)} bytes")

sess = ort.InferenceSession(ONNX_PATH)
print(f"Output shape: {sess.get_outputs()[0].shape}")
print("OK")
