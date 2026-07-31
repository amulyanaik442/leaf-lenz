"""
Merge wheat dataset: rust->Rust, Leaf Blight+Tan spot->Leaf Spot, Aphid+Mite->Pest
All classes capped at MAX_PER_CLASS images.
"""
import os, shutil, random

WHEAT_DATA = os.path.join(os.environ.get("TEMP", os.path.expanduser("~\\AppData\\Local\\Temp")), "wheat_data")
MERGED_DATA = os.path.join(os.environ.get("TEMP", os.path.expanduser("~\\AppData\\Local\\Temp")), "wheat_data_merged")

RUST_CLASSES = ['Black Rust', 'Brown Rust', 'Yellow Rust']
LEAF_SPOT_CLASSES = ['Leaf Blight', 'Tan spot']
PEST_CLASSES = ['Aphid', 'Mite']
EXTS = ('.png', '.jpg', '.jpeg')
MAX_PER_CLASS = 400


def merge_split(split_name, max_per_class=MAX_PER_CLASS):
    src = os.path.join(WHEAT_DATA, split_name)
    dst = os.path.join(MERGED_DATA, split_name)

    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    random.seed(42)

    merge_groups = {
        'Rust': RUST_CLASSES,
        'Leaf Spot': LEAF_SPOT_CLASSES,
        'Pest': PEST_CLASSES,
    }
    skip = set()
    for target_name, source_classes in merge_groups.items():
        skip.update(source_classes)
        all_imgs = []
        for cls in source_classes:
            cls_path = os.path.join(src, cls)
            if os.path.isdir(cls_path):
                for f in os.listdir(cls_path):
                    if f.lower().endswith(EXTS):
                        all_imgs.append((os.path.join(cls_path, f), cls))
        if len(all_imgs) > max_per_class:
            all_imgs = random.sample(all_imgs, max_per_class)
        dst_dir = os.path.join(dst, target_name)
        os.makedirs(dst_dir, exist_ok=True)
        for src_path, cls in all_imgs:
            prefix = cls.replace(' ', '_').lower()
            fname = os.path.basename(src_path)
            shutil.copy2(src_path, os.path.join(dst_dir, f"{prefix}_{fname}"))

    for cls in sorted(os.listdir(src)):
        cls_path = os.path.join(src, cls)
        if not os.path.isdir(cls_path) or cls in skip:
            continue
        all_imgs = [(os.path.join(cls_path, f), cls) for f in os.listdir(cls_path) if f.lower().endswith(EXTS)]
        if len(all_imgs) > max_per_class:
            all_imgs = random.sample(all_imgs, max_per_class)
        target = os.path.join(dst, cls)
        os.makedirs(target, exist_ok=True)
        for src_path, _ in all_imgs:
            shutil.copy2(src_path, os.path.join(target, os.path.basename(src_path)))

    count = 0
    for d in sorted(os.listdir(dst)):
        dp = os.path.join(dst, d)
        if os.path.isdir(dp):
            n = len([f for f in os.listdir(dp) if f.lower().endswith(EXTS)])
            count += n
            print(f'    {d:28s} {n:5d}')
    return count

random.seed(42)
print(f'Merged dataset (max {MAX_PER_CLASS}/class)...')
print('\nTrain:')
n = merge_split('train')
print(f'    TOTAL: {n}')
print('\nValid:')
n = merge_split('valid', max_per_class=80)
print(f'    TOTAL: {n}')
print('\nTest:')
n = merge_split('test', max_per_class=80)
print(f'    TOTAL: {n}')
