import os
import sys

dataset_root = sys.argv[1] if len(sys.argv) > 1 else "dataset"

print(f"Dataset root: {os.path.abspath(dataset_root)}\n")

grand_total = 0
for split in ["train", "val", "test"]:
    split_path = os.path.join(dataset_root, split)
    if not os.path.isdir(split_path):
        continue

    print(f"[{split.upper()}]")
    split_total = 0
    for label in ["real", "fake"]:
        label_path = os.path.join(split_path, label)
        count = len([
            f for f in os.listdir(label_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]) if os.path.isdir(label_path) else 0
        print(f"  {label:6s}: {count:,}")
        split_total += count
    print(f"  Subtotal: {split_total:,}\n")
    grand_total += split_total

print(f"Grand Total: {grand_total:,} face crops")
