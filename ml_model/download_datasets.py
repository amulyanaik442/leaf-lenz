import os
import kagglehub

datasets = {
    "plant_diseases": "nirmalsankalana/plant-diseases-training-dataset",
    "rocole": "diegoagonzalez/rocole-original",
    "mango": "aryashah2k/mango-leaf-disease-dataset",
    "potato": "warcoder/potato-leaf-disease-dataset",
    "plant_village": "arjuntejaswi/plant-village"
}

print("Starting download of datasets using kagglehub...")
downloaded_paths = {}

for name, path in datasets.items():
    print(f"\nDownloading {name} ({path})...")
    try:
        download_path = kagglehub.dataset_download(path)
        print(f"Successfully downloaded {name} to: {download_path}")
        downloaded_paths[name] = download_path
    except Exception as e:
        print(f"Error downloading {name}: {e}")

print("\n--- Summary of downloaded paths ---")
for name, p in downloaded_paths.items():
    print(f"{name}: {p}")
