"""
Rename maize dataset files to remove spaces and parentheses,
which cause OneDrive locking issues on Windows.

Usage:
    python ml_model/rename_maize_files.py
"""
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIZE_DATA = os.path.join(BASE_DIR, "ml_model", "maize_data")

def rename_files(directory):
    count = 0
    for cls in os.listdir(directory):
        cls_path = os.path.join(directory, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            old_path = os.path.join(cls_path, fname)
            if not os.path.isfile(old_path):
                continue
            # Remove spaces and parentheses
            new_fname = fname.replace(" ", "_").replace("(", "").replace(")", "")
            if new_fname != fname:
                new_path = os.path.join(cls_path, new_fname)
                os.rename(old_path, new_path)
                count += 1
    return count

if __name__ == "__main__":
    total = 0
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(MAIZE_DATA, split)
        if os.path.exists(split_dir):
            n = rename_files(split_dir)
            print(f"  {split}: renamed {n} files")
            total += n
    print(f"\nTotal renamed: {total}")
