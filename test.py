import pandas as pd

for path in [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv"
]:
    print("Reading:", path)
    df = pd.read_csv(path, sep="\t", nrows=5)
    print(df.head())