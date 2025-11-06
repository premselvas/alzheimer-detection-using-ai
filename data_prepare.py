import os, glob, random, shutil
from pathlib import Path

src_dirs = [
    r"C:\Users\admin\Desktop\new alzheimer 1\archive\OriginalDataset",
    r"C:\Users\admin\Desktop\new alzheimer 1\archive\AugmentedAlzheimerDataset"
]
out_root = r"C:\Users\admin\Desktop\new alzheimer 1\dataset_combined"
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
random.seed(42)

classes = set()
for d in src_dirs:
    for sub in os.listdir(d):
        if os.path.isdir(os.path.join(d, sub)):
            classes.add(sub)
classes = sorted(classes)
print("Found classes:", classes)

if os.path.exists(out_root):
    shutil.rmtree(out_root)

for split in ("train", "val", "test"):
    for cls in classes:
        os.makedirs(os.path.join(out_root, split, cls), exist_ok=True)

for cls in classes:
    all_files = []
    for d in src_dirs:
        p = os.path.join(d, cls)
        if os.path.isdir(p):
            
            for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.JPEG","*.PNG","*.BMP"):
                all_files += glob.glob(os.path.join(p, ext))
    all_files = sorted(set(all_files))
    random.shuffle(all_files)

    n = len(all_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:]

    def copy_list(lst, split):
        for i, src in enumerate(lst):
            dst = os.path.join(out_root, split, cls, f"{cls}_{i}{Path(src).suffix}")
            shutil.copy2(src, dst)

    copy_list(train_files, "train")
    copy_list(val_files, "val")
    copy_list(test_files, "test")

print("✅ Done. Combined dataset is at:", out_root)

#import os
#root = r"C:\Users\admin\Desktop\new alzheimer 1\dataset_combined"
#for split in ["train","val","test"]:
#    print(split)
#    for c in os.listdir(os.path.join(root, split)):
#        path = os.path.join(root, split, c)
 #       print(" ", c, len(os.listdir(path)))
