import os, glob, random, shutil
from pathlib import Path
from collections import defaultdict
orig_dir = r"E:\final_pro_alz\archive\OriginalDataset"
aug_dir  = r"E:\final_pro_alz\archive\AugmentedAlzheimerDataset"
out_root = r"E:\final_pro_alz\prepare_data_try1"
train_ratio, val_ratio = 0.8, 0.1
random.seed(42)
classes = sorted([
    d for d in os.listdir(orig_dir)
    if os.path.isdir(os.path.join(orig_dir, d))
])
print("Found classes:", classes)
if os.path.exists(out_root):
    shutil.rmtree(out_root)
for split in ("train", "val", "test"):
    for cls in classes:
        os.makedirs(os.path.join(out_root, split, cls), exist_ok=True)
split_map = defaultdict(list)
for cls in classes:
    files = []
    for ext in ("*.jpg","*.png","*.jpeg","*.bmp"):
        files += glob.glob(os.path.join(orig_dir, cls, ext))
    random.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    split_map["train"] += [(cls, f) for f in files[:n_train]]
    split_map["val"]   += [(cls, f) for f in files[n_train:n_train+n_val]]
    split_map["test"]  += [(cls, f) for f in files[n_train+n_val:]]
for cls in classes:
    aug_files = []
    for ext in ("*.jpg","*.png","*.jpeg","*.bmp"):
        aug_files += glob.glob(os.path.join(aug_dir, cls, ext))
    split_map["train"] += [(cls, f) for f in aug_files]
counts = defaultdict(int)
for split, items in split_map.items():
    for cls, src in items:
        dst = os.path.join(
            out_root, split, cls,
            f"{cls}_{counts[(split, cls)]}{Path(src).suffix}"
        )
        shutil.copy(src, dst)
        counts[(split, cls)] += 1
print("\nDataset distribution:")
for split in ("train", "val", "test"):
    print(f"\n{split.upper()}:")
    for cls in classes:
        path = os.path.join(out_root, split, cls)
        print(f"  {cls}: {len(os.listdir(path))}")
print("\nDataset prepared successfully at:", out_root)