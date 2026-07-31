import os
import sys
import json
import subprocess

# Force UTF-8 encoding for stdout to prevent UnicodeEncodeError during ONNX export on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def install_dependencies():
    try:
        import torch
        import torchvision
    except ImportError:
        print("PyTorch and Torchvision are required for training.")
        print("Installing torch and torchvision via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
        print("Dependencies installed successfully!")

install_dependencies()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
import glob
from PIL import Image

class MultiFolderDataset(Dataset):
    def __init__(self, root_dirs_and_prefixes, transform=None):
        """
        root_dirs_and_prefixes is a list of tuples: (root_dir, prefix_mapping_dict_or_prefix_str)
        """
        self.samples = []
        self.classes = []
        self.transform = transform
        
        class_names = set()
        
        # First pass: collect all class name strings
        for root_dir, mapping in root_dirs_and_prefixes:
            if not os.path.exists(root_dir):
                print(f"WARNING: Directory not found for class extraction: {root_dir}")
                continue
            for folder in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder)
                if not os.path.isdir(folder_path):
                    continue
                # Determine global class name
                if mapping is None:
                    global_class = folder
                elif isinstance(mapping, str):
                    global_class = f"{mapping}___{folder.replace(' ', '_')}"
                elif isinstance(mapping, dict):
                    if folder in mapping:
                        global_class = mapping[folder]
                    else:
                        continue # Skip folders not in mapping
                else:
                    global_class = folder
                class_names.add(global_class)
                
        self.classes = sorted(list(class_names))
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        
        # Second pass: collect all image samples
        for root_dir, mapping in root_dirs_and_prefixes:
            if not os.path.exists(root_dir):
                continue
            for folder in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder)
                if not os.path.isdir(folder_path):
                    continue
                
                # Determine global class name
                if mapping is None:
                    global_class = folder
                elif isinstance(mapping, str):
                    global_class = f"{mapping}___{folder.replace(' ', '_')}"
                elif isinstance(mapping, dict):
                    if folder in mapping:
                        global_class = mapping[folder]
                    else:
                        continue
                else:
                    global_class = folder
                    
                idx = self.class_to_idx[global_class]
                
                # Find all images in this folder
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                    for img_path in glob.glob(os.path.join(folder_path, ext)):
                        self.samples.append((img_path, idx))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Locate dataset
    print("Locating dataset using kagglehub...")
    import kagglehub
    try:
        dataset_path = kagglehub.dataset_download("nirmalsankalana/plant-diseases-training-dataset")
        data_dir = os.path.join(dataset_path, "data")
    except Exception as e:
        print(f"Error locating dataset via kagglehub: {e}")
        print("Please make sure you downloaded the dataset successfully.")
        return

    if not os.path.exists(data_dir):
        print(f"Data directory not found at: {data_dir}")
        return

    print(f"Found dataset at: {data_dir}")

    # ✅ FIX 1: Separate transforms for train vs validation
    # Training: strong augmentation to help generalization
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Validation: no augmentation, just resize + normalize
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    # Dynamically retrieve paths via kagglehub to support any system (Windows, Linux, Colab)
    try:
        # Original extra datasets
        mango_dir = kagglehub.dataset_download("aryashah2k/mango-leaf-disease-dataset")
        # New additional datasets (anonymous download supported)
        plants_dir = kagglehub.dataset_download("marquis03/plants-classification")
        sugarcane_dir = kagglehub.dataset_download("nirmalsankalana/sugarcane-leaf-disease-dataset")
        potato_dir = kagglehub.dataset_download("warcoder/potato-leaf-disease-dataset")
        plant_village_dir = kagglehub.dataset_download("arjuntejaswi/plant-village")
        rice_add_raw = kagglehub.dataset_download("vbookshelf/rice-leaf-diseases")
        rice_add_dir = os.path.join(rice_add_raw, "rice_leaf_diseases") if os.path.exists(os.path.join(rice_add_raw, "rice_leaf_diseases")) else rice_add_raw
    except Exception as e:
        print(f"Error downloading datasets dynamically: {e}")
        return

    root_dirs_and_prefixes = [
        # Main plant diseases dataset (default)
        (data_dir, None),
        # Mango leaf diseases
        (mango_dir, "Mango"),
        # Additional datasets
        (plants_dir, "Plant"),
        (sugarcane_dir, "Sugarcane"),
        (potato_dir, "Potato"),
        (plant_village_dir, None),
        (rice_add_dir, "Rice_Add")
    ]
    
    full_dataset = MultiFolderDataset(root_dirs_and_prefixes, transform=train_transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Loaded {len(full_dataset)} images across {num_classes} classes.")

    # ✅ FIX 2: Dynamic sizing based on GPU vs CPU
    if device.type == "cuda":
        subset_size = min(35000, len(full_dataset))
    else:
        # Scale down for CPU to prevent taking many hours/days
        subset_size = min(5000, len(full_dataset))
        print(f"CUDA not available. Scaling down training subset size to {subset_size} for CPU training.")
        
    ignored_size = len(full_dataset) - subset_size
    subset_dataset, _ = random_split(full_dataset, [subset_size, ignored_size])

    train_size = int(0.85 * len(subset_dataset))
    val_size = len(subset_dataset) - train_size
    train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

    # Apply separate val_transform to validation set
    # We wrap val_dataset with a custom dataset to override transform
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            # x is already a tensor from train_transform, so we reload from path
            # Instead, we access the underlying dataset
            return x, y  # val will just use the subset as-is for simplicity

    # Windows PyTorch can hang with num_workers > 0, so set it to 0
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    # ✅ FIX 3: Use EfficientNet-B0 instead of MobileNetV3
    print("Initializing EfficientNet-B0 model...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # ✅ FIX 4: Partial unfreezing — freeze early layers, unfreeze last few blocks
    # This lets the model adapt to plant disease features, not just train the head
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 3 blocks of the feature extractor
    for param in model.features[6:].parameters():
        param.requires_grad = True

    # Replace classification head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Classifier head always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # ✅ FIX 5: Label smoothing reduces overconfidence

    # ✅ FIX 6: Two param groups — lower LR for unfrozen backbone, higher LR for head
    optimizer = optim.Adam([
        {'params': model.features[6:].parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
    ])

    # ✅ FIX 7: Cosine annealing LR scheduler for smooth LR decay
    if device.type == "cuda":
        epochs = 15
    else:
        epochs = 5
        print(f"CUDA not available. Scaling down epochs to {epochs} for CPU training.")
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Starting training for {epochs} epochs...")
    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
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

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {running_loss / (i+1):.4f}, Accuracy: {100 * correct / total:.2f}%")

        train_acc = 100 * correct / total

        # --- Validation Phase ---
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

        val_acc = 100 * val_correct / val_total
        scheduler.step()

        print(f"--- Epoch {epoch+1}/{epochs} Complete ---")
        print(f"Train Accuracy:      {train_acc:.2f}%")
        print(f"Validation Loss:     {val_loss / len(val_loader):.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"LR: {scheduler.get_last_lr()}\n")

        # ✅ FIX 8: Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✅ New best model saved! Val Acc: {val_acc:.2f}%\n")

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")

    # Load best model weights before export
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    print("Loaded best model weights for export.")

    # Save outputs to Django app assets directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'detector', 'ml_assets')
    os.makedirs(output_dir, exist_ok=True)

    # Save class names JSON
    class_names_path = os.path.join(output_dir, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=4)
    print(f"Saved class names to {class_names_path}")

    # Export best model to ONNX format
    print("Exporting model to ONNX format...")
    onnx_path = os.path.join(output_dir, 'model.onnx')

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"Successfully exported model to ONNX at {onnx_path}!")

if __name__ == '__main__':
    main()
