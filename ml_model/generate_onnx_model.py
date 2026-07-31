import os
import sys
import subprocess

def install_dependencies():
    try:
        import torch
    except ImportError:
        print("PyTorch is required to generate the ONNX model.")
        print("Installing torch via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
        print("Torch installed successfully!")

install_dependencies()

import torch
import torch.nn as nn

class LeafLenzTinyModel(nn.Module):
    def __init__(self, num_classes=30):
        super().__init__()
        # A tiny CNN structure to keep ONNX file small, fast and fully functional
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    print("Initializing Model...")
    model = LeafLenzTinyModel(num_classes=30)
    model.eval()

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'detector', 'ml_assets')
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, 'model.onnx')

    print("Exporting model to ONNX format...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully generated and exported model to: {onnx_path}")

if __name__ == '__main__':
    main()
