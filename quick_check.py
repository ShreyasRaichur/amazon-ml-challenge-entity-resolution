import pandas as pd

paths = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv",
]

for path in paths:
    print(f"\n=== Reading {path} ===")
    df = pd.read_csv(path, sep='\t', nrows=5)
    print(df.head().to_string(index=False))
    print(f"columns: {list(df.columns)}")

print("\n=== Basic sizes ===")
for path in paths:
    df = pd.read_csv(path, sep='\t', dtype=str)
    print(f"{path}: {df.shape}")

print("\n=== Ground truth summary ===")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep='\t', dtype=str)
print(gt.head().to_string(index=False))
print(f"rows: {len(gt)}")
