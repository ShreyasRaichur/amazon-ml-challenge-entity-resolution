import os
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
REPO_DIR = ROOT / "Amazon_ML_Challange_26-27-main"
REPO_TRAIN_DIR = REPO_DIR / "dataset" / "train"
REPO_TEST_DIR = REPO_DIR / "dataset" / "test"


def resolve_dataset_dir() -> Path:
    if TRAIN_DIR.exists():
        return TRAIN_DIR
    if REPO_TRAIN_DIR.exists():
        return REPO_TRAIN_DIR
    raise FileNotFoundError("No dataset/train directory found. Place the challenge dataset in /dataset/train or in the repo subfolder.")


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f) - 1


def load_tsv_head(path: Path, nrows: int = 5) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, nrows=nrows)


def validate_training_data(train_dir: Path) -> None:
    expected = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv",
    ]
    missing = [p for p in expected if not (train_dir / p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing training files: {missing}")

    s1_head = load_tsv_head(train_dir / "train_source1.tsv")
    s2_head = load_tsv_head(train_dir / "train_source2.tsv")
    s3_head = load_tsv_head(train_dir / "train_source3.tsv")
    gt_head = load_tsv_head(train_dir / "train_ground_truth.tsv")

    required_cols = {"entity_id", "business_name", "business_address", "country"}
    if not required_cols.issubset(set(s1_head.columns)):
        raise ValueError(f"train_source1.tsv is missing required columns: {required_cols - set(s1_head.columns)}")
    if not required_cols.issubset(set(s2_head.columns)):
        raise ValueError(f"train_source2.tsv is missing required columns: {required_cols - set(s2_head.columns)}")
    if not required_cols.issubset(set(s3_head.columns)):
        raise ValueError(f"train_source3.tsv is missing required columns: {required_cols - set(s3_head.columns)}")
    if set(gt_head.columns) != {"source1_entity_id", "matched_entity_ids"}:
        raise ValueError(f"Unexpected ground-truth columns: {list(gt_head.columns)}")

    gt = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str)
    positive_rows = gt[gt["matched_entity_ids"].fillna("").astype(str).str.strip().ne("")]
    print("=== TRAIN DATA SUMMARY ===")
    print(f"Source1 rows: {count_rows(train_dir / 'train_source1.tsv'):,}")
    print(f"Source2 rows: {count_rows(train_dir / 'train_source2.tsv'):,}")
    print(f"Source3 rows: {count_rows(train_dir / 'train_source3.tsv'):,}")
    print(f"Ground truth rows: {len(gt):,}")
    print(f"Entities with matches: {len(positive_rows):,}")
    print(f"Singletons: {len(gt) - len(positive_rows):,}")
    print("Sample ground truth:\n", gt_head.head(5).to_string(index=False))
    print()

    if positive_rows.empty:
        raise ValueError("Ground truth contains no positive matches; validation will collapse to zero precision/recall.")

    # Quick split health check using the actual IDs from the source file head and a seeded sample.
    s1_ids = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", dtype=str, usecols=["entity_id"])
    rng = random.Random(42)
    val_ids = set(rng.sample(s1_ids["entity_id"].tolist(), max(1, int(len(s1_ids) * 0.1))))
    val_positive = gt[gt["source1_entity_id"].isin(val_ids)]
    print(f"Random 10% validation set size: {len(val_ids):,}")
    print(f"Positive entities in validation set: {len(val_positive):,}")
    print("If this is zero, the validation split is invalid for F_0.5 scoring.")
    print()


def validate_test_data(test_dir: Path) -> None:
    expected = [
        "test_source1.tsv",
        "test_source2.tsv",
        "test_source3.tsv",
    ]
    missing = [p for p in expected if not (test_dir / p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing test files: {missing}")

    s1_head = load_tsv_head(test_dir / "test_source1.tsv")
    s2_head = load_tsv_head(test_dir / "test_source2.tsv")
    s3_head = load_tsv_head(test_dir / "test_source3.tsv")
    print("=== TEST DATA SUMMARY ===")
    print(f"Test Source1 rows: {count_rows(test_dir / 'test_source1.tsv'):,}")
    print(f"Test Source2 rows: {count_rows(test_dir / 'test_source2.tsv'):,}")
    print(f"Test Source3 rows: {count_rows(test_dir / 'test_source3.tsv'):,}")
    country_values = set(pd.concat([s1_head[["country"]], s2_head[["country"]], s3_head[["country"]]], ignore_index=True)["country"].tolist())
    print(f"Country values in test data: {sorted(country_values)[:10]}")
    print()


def run_real_pipeline() -> None:
    repo = REPO_DIR if REPO_DIR.exists() else None
    if repo is None:
        raise FileNotFoundError("The challenge repo folder is missing. The real training pipeline is in Amazon_ML_Challange_26-27-main.")

    dataset_dir = REPO_TRAIN_DIR.parent if REPO_TRAIN_DIR.exists() else DATASET_ROOT
    run_cmd = [
        "python",
        str(repo / "run_pipeline.py"),
        "--dev",
    ]
    print("=== REAL PIPELINE COMMAND ===")
    print(" ".join(run_cmd))
    print()
    print("This runs the actual challenge pipeline against the real dataset, not a dummy subset.")
    print("Use the repo's training and predict scripts for the final model run.")


def main() -> None:
    train_dir = resolve_dataset_dir()
    validate_training_data(train_dir)
    if (train_dir.parent / "test").exists():
        validate_test_data(train_dir.parent / "test")
    elif REPO_TEST_DIR.exists():
        validate_test_data(REPO_TEST_DIR)
    run_real_pipeline()


if __name__ == "__main__":
    main()