"""
Download Wheat Plant Diseases dataset from Kaggle.
Requires kaggle.json credentials in ~/.kaggle/kaggle.json
"""
import os
import sys
import shutil
import kagglehub

DATASET_ID = "kushagra3204/wheat-plant-diseases"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "wheat_data")


def download():
    print(f"Downloading dataset '{DATASET_ID}' from Kaggle...")
    try:
        download_path = kagglehub.dataset_download(DATASET_ID)
        print(f"Downloaded to: {download_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure ~/.kaggle/kaggle.json exists with valid credentials.")
        sys.exit(1)

    # The dataset has a 'data' subfolder with train/test/valid
    data_dir = os.path.join(download_path, "data")
    if not os.path.exists(data_dir):
        # Some versions put train/test/valid at the root
        data_dir = download_path

    print(f"Data directory: {data_dir}")

    # Copy to local wheat_data directory for consistent paths
    if os.path.exists(LOCAL_DIR):
        # Remove contents first, then the directory
        for item in os.listdir(LOCAL_DIR):
            item_path = os.path.join(LOCAL_DIR, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        try:
            os.rmdir(LOCAL_DIR)
        except OSError:
            pass
    shutil.copytree(data_dir, LOCAL_DIR, dirs_exist_ok=True)
    print(f"Copied dataset to: {LOCAL_DIR}")

    # Print class summary
    train_dir = os.path.join(LOCAL_DIR, "train")
    if os.path.exists(train_dir):
        classes = sorted(os.listdir(train_dir))
        print(f"\nFound {len(classes)} classes:")
        total = 0
        for cls in classes:
            cls_path = os.path.join(train_dir, cls)
            if os.path.isdir(cls_path):
                count = len([f for f in os.listdir(cls_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {cls}: {count} images")
                total += count
        print(f"\nTotal training images: {total}")

    print("\nDataset ready for training.")


if __name__ == "__main__":
    download()
