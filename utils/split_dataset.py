import os
import shutil
import random
import csv

# ── CONFIG — update these paths if needed ─────────────────────────────────────
LABELS_CSV  = "datasets/raw/labels.csv"
IMAGES_DIR  = "datasets/raw/head_ct/head_ct"
OUTPUT_DIR  = "datasets"
SPLITS      = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED        = 42
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

# Step 1 — Read labels.csv
print("📖 Reading labels.csv...")
hemorrhage    = []   # image ids with hemorrhage (label=1)
no_hemorrhage = []   # image ids without hemorrhage (label=0)

with open(LABELS_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_id = int(row["id"].strip())
        label  = int(row[" hemorrhage"].strip())
        fname  = f"{img_id:03d}.png"   # e.g. 0 → 000.png
        if label == 1:
            hemorrhage.append(fname)
        else:
            no_hemorrhage.append(fname)

print(f"  ✅ Hemorrhage images    : {len(hemorrhage)}")
print(f"  ✅ No hemorrhage images : {len(no_hemorrhage)}")

# Step 2 — Split and copy each class
def split_and_copy(file_list, class_name):
    random.shuffle(file_list)
    total   = len(file_list)
    n_train = int(total * SPLITS["train"])
    n_val   = int(total * SPLITS["val"])

    split_map = {
        "train" : file_list[:n_train],
        "val"   : file_list[n_train : n_train + n_val],
        "test"  : file_list[n_train + n_val:]
    }

    print(f"\n📂 Organizing: {class_name}")
    for split, files in split_map.items():
        dest_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        for fname in files:
            src  = os.path.join(IMAGES_DIR, fname)
            dest = os.path.join(dest_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dest)
            else:
                print(f"  ⚠️  Missing: {src}")

        print(f"  ✅ {split:5} → {len(files)} images  →  {dest_dir}")

split_and_copy(hemorrhage,    "hemorrhage")
split_and_copy(no_hemorrhage, "no_hemorrhage")

# Step 3 — Print final summary
print("\n" + "="*55)
print("🎉 Dataset split complete! Final counts:")
print("="*55)
for split in ["train", "val", "test"]:
    for cls in ["hemorrhage", "no_hemorrhage"]:
        path  = os.path.join(OUTPUT_DIR, split, cls)
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"  {split:5} / {cls:15} → {count:3} images")
print("="*55)
print("\n✅ You're ready to train! Run:")
print("   cd ai_models/ct_scan_model")
print("   python train.py --data_dir ../../datasets --epochs 20")