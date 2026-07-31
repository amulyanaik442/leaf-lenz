"""
Download the Corn/Maize Leaf Disease dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset

Usage:
    python ml_model/download_maize_dataset.py

Requires:
    - kaggle.json credentials in ~/.kaggle/ or C:/Users/<user>/.kaggle/
    - pip install kagglehub
"""
import os
import sys
import shutil

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "ml_model", "maize_data")

KAGGLE_DATASET = "smaranjitghose/corn-or-maize-leaf-disease-dataset"


def main():
    print("=" * 60)
    print("  DOWNLOADING MAIZE DISEASE DATASET")
    print("=" * 60)
    print(f"  Dataset: {KAGGLE_DATASET}")
    print(f"  Output:  {OUTPUT_DIR}\n")

    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub

    # Download dataset
    print("Downloading from Kaggle...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Downloaded to: {path}")

    # Find the actual data directory (may be nested)
    data_dir = path
    for root, dirs, files in os.walk(path):
        # Look for class directories (Common_Rust, Gray_Leaf_Spot, Healthy, Northern_Leaf_Blight)
        class_dirs = [d for d in dirs if d in [
            "Common_Rust", "Gray_Leaf_Spot", "Healthy", "Northern_Leaf_Blight",
            "Corn_Common_rust", "Corn_Gray_leaf_spot", "Corn_Healthy", "Corn_Northern_leaf_blight",
        ]]
        if len(class_dirs) >= 3:
            data_dir = root
            break

    print(f"\nFound data at: {data_dir}")
    print(f"Contents: {os.listdir(data_dir)}")

    # Copy to our project
    if os.path.exists(OUTPUT_DIR):
        print(f"\nRemoving old data at {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    shutil.copytree(data_dir, OUTPUT_DIR)
    print(f"\nDataset saved to: {OUTPUT_DIR}")

    # Show class counts
    print("\nClass distribution:")
    for cls in sorted(os.listdir(OUTPUT_DIR)):
        cls_path = os.path.join(OUTPUT_DIR, cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  {cls:30s} {n:5d} images")

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
