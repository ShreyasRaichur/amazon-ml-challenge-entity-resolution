"""
Training and Validation on Official Challenge Dataset.
Samples reference records, extracts true matching targets from S2/S3,
mines hard negatives via country-partitioned blocking, trains the discriminator,
optimizes Macro-F0.5 decision threshold, and evaluates on a held-out validation split.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.blocking import CountryPartitionedBlocker
from src.config import DEFAULT_CONFIG
from src.features import PairFeatureExtractor
from src.graph_resolver import TripartiteGraphResolver
from src.preprocess import RecordPreprocessor
from src.train_matcher import MatcherModel, MatcherTrainer, ThresholdOptimizer
from src.utils import compute_entity_f05, compute_macro_f05, setup_logger

logger = setup_logger("train_challenge")


def load_target_records_by_ids(
    file_path: Path,
    target_ids_set: Set[str],
    max_extra_distractors: int = 15000
) -> List[Dict[str, str]]:
    """Scan a target file (S2 or S3) and extract requested IDs plus random distractors."""
    records: List[Dict[str, str]] = []
    distractors_collected = 0
    t0 = time.time()

    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else "UNKNOWN"

            if eid in target_ids_set:
                records.append({
                    "entity_id": eid,
                    "business_name": name,
                    "business_address": addr,
                    "country": country
                })
            elif distractors_collected < max_extra_distractors:
                # Add as negative distractor
                records.append({
                    "entity_id": eid,
                    "business_name": name,
                    "business_address": addr,
                    "country": country
                })
                distractors_collected += 1

    logger.info(
        "Scanned %s in %.2fs: extracted %d records (%d target matches, %d distractors)",
        file_path.name, time.time() - t0, len(records),
        len(records) - distractors_collected, distractors_collected
    )
    return records


def run_training_and_validation(
    dataset_dir: Path,
    n_s1_samples: int = 5000,
    val_ratio: float = 0.25
) -> Dict[str, Any]:
    """Execute training and evaluation on challenge data."""
    train_dir = dataset_dir / "train"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    logger.info("=" * 70)
    logger.info("PHASE 1: Loading %d S1 reference samples from %s...", n_s1_samples, s1_path.name)
    logger.info("=" * 70)

    s1_df = pd.read_csv(s1_path, sep="\t", nrows=n_s1_samples)
    all_s1_ids = set(s1_df["entity_id"])

    # Load ground truth for sampled S1 entities
    logger.info("Loading ground truth mapping...")
    gt_map: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_s1_ids}
    needed_target_ids: Set[str] = set()

    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts:
                continue
            s1_id = parts[0]
            if s1_id in all_s1_ids:
                matches_val = parts[1] if len(parts) > 1 else ""
                if matches_val and matches_val != "nan":
                    m_set = {m.strip() for m in matches_val.split(",") if m.strip()}
                    gt_map[s1_id] = m_set
                    needed_target_ids.update(m_set)

    # Identify singletons
    n_singletons = sum(1 for s in gt_map.values() if not s)
    logger.info(
        "Loaded ground truth for %d S1 entities: %d non-singletons, %d singletons (%.1f%%)",
        len(gt_map), len(gt_map) - n_singletons, n_singletons, (n_singletons / max(1, len(gt_map))) * 100
    )
    logger.info("Total true target entities needed across S2 and S3: %d", len(needed_target_ids))

    # Split into train and validation S1 sets
    np.random.seed(42)
    s1_list = sorted(list(s1_df["entity_id"]))
    np.random.shuffle(s1_list)
    n_val = int(len(s1_list) * val_ratio)
    val_s1_ids = set(s1_list[:n_val])
    train_s1_ids = set(s1_list[n_val:])

    train_s1_df = s1_df[s1_df["entity_id"].isin(train_s1_ids)].copy()
    val_s1_df = s1_df[s1_df["entity_id"].isin(val_s1_ids)].copy()
    logger.info("Train S1 records: %d | Validation S1 records: %d", len(train_s1_df), len(val_s1_df))

    # Extract target records from S2 and S3
    logger.info("Extracting target records from Source 2 and Source 3...")
    s2_records = load_target_records_by_ids(s2_path, needed_target_ids, max_extra_distractors=15000)
    s3_records = load_target_records_by_ids(s3_path, needed_target_ids, max_extra_distractors=15000)

    s2_df = pd.DataFrame(s2_records)
    s3_df = pd.DataFrame(s3_records)

    # Preprocess records
    logger.info("Preprocessing all records with country-agnostic normalizer...")
    prep = RecordPreprocessor(DEFAULT_CONFIG.columns)
    train_s1_p = prep.process_dataframe(train_s1_df, source_prefix="S1")
    val_s1_p = prep.process_dataframe(val_s1_df, source_prefix="S1")
    s2_p = prep.process_dataframe(s2_df, source_prefix="S2")
    s3_p = prep.process_dataframe(s3_df, source_prefix="S3")
    target_p = pd.concat([s2_p, s3_p], ignore_index=True).drop_duplicates(subset=["entity_id"])

    # Blocking on Training Set
    logger.info("=" * 70)
    logger.info("PHASE 2: Running country-partitioned blocking on training set...")
    logger.info("=" * 70)
    blocker = CountryPartitionedBlocker(DEFAULT_CONFIG.blocking)
    train_c_dict, train_c_df = blocker.generate_candidates(train_s1_p, s2_p, s3_p)

    # Feature extraction & Model Training
    logger.info("=" * 70)
    logger.info("PHASE 3: Extracting features and training matcher model...")
    logger.info("=" * 70)
    trainer = MatcherTrainer(DEFAULT_CONFIG.train)
    matcher, opt_threshold, cv_f05 = trainer.train_and_evaluate(
        train_c_df, gt_map, train_s1_p, target_p
    )

    logger.info("Model training complete! Optimal threshold: %.3f (CV Macro-F0.5: %.4f)", opt_threshold, cv_f05)

    # Evaluation on Held-Out Validation Split
    logger.info("=" * 70)
    logger.info("PHASE 4: Evaluating on held-out validation split (%d S1 entities)...", len(val_s1_p))
    logger.info("=" * 70)

    val_c_dict, val_c_df = blocker.generate_candidates(val_s1_p, s2_p, s3_p)
    logger.info("Validation candidate pairs: %d", len(val_c_df))

    val_X, _ = PairFeatureExtractor.build_feature_matrix(val_c_df, val_s1_p, target_p)
    val_probs = matcher.predict_proba(val_X)

    # Graph resolution
    resolver = TripartiteGraphResolver(DEFAULT_CONFIG.resolver)
    val_matches = resolver.resolve(
        candidate_pairs_df=val_c_df,
        probabilities=val_probs,
        threshold=opt_threshold,
        target_df=target_p
    )

    # Compute official Macro-F0.5 metric on validation set
    val_gt = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_ids}
    val_pred_sets = {s1_id: set(val_matches.get(s1_id, [])) for s1_id in val_s1_ids}

    macro_f05, per_entity_scores = compute_macro_f05(val_gt, val_pred_sets, beta=0.5)

    # Detailed statistics
    val_singletons = [s1_id for s1_id in val_s1_ids if not val_gt[s1_id]]
    singleton_scores = [per_entity_scores[s1_id] for s1_id in val_singletons]
    singleton_accuracy = (sum(singleton_scores) / max(1, len(singleton_scores))) * 100

    val_non_singletons = [s1_id for s1_id in val_s1_ids if val_gt[s1_id]]
    non_singleton_scores = [per_entity_scores[s1_id] for s1_id in val_non_singletons]
    non_singleton_mean = sum(non_singleton_scores) / max(1, len(non_singleton_scores))

    logger.info("=" * 70)
    logger.info("VALIDATION RESULTS SUMMARY ON REAL CHALLENGE DATA:")
    logger.info("  Total Validation Entities: %d", len(val_s1_ids))
    logger.info("  Macro-F0.5 Score:         %.4f (Target >= 0.9333)", macro_f05)
    logger.info("  Singleton Accuracy:       %.2f%% (%d singletons)", singleton_accuracy, len(val_singletons))
    logger.info("  Non-Singleton Mean F0.5:  %.4f (%d entities)", non_singleton_mean, len(val_non_singletons))
    logger.info("=" * 70)

    # Print 5 sample accuracy matches
    print("\nSAMPLE ACCURACY MATCHES FROM VALIDATION SET:")
    sample_count = 0
    for s1_id in val_non_singletons[:10]:
        true_set = val_gt[s1_id]
        pred_set = val_pred_sets[s1_id]
        score = per_entity_scores[s1_id]
        s1_row = val_s1_p[val_s1_p["entity_id"] == s1_id].iloc[0]

        print(f"\n[Entity {s1_id}] (F0.5 Score: {score:.4f})")
        print(f"  Name:     {s1_row['raw_name']}")
        print(f"  Address:  {s1_row['raw_address']}")
        print(f"  Country:  {s1_row['country']}")
        print(f"  True:     {sorted(list(true_set))}")
        print(f"  Pred:     {sorted(list(pred_set))}")
        sample_count += 1
        if sample_count >= 5:
            break

    return {
        "macro_f05": macro_f05,
        "singleton_accuracy": singleton_accuracy,
        "optimal_threshold": opt_threshold,
        "n_val": len(val_s1_ids)
    }


if __name__ == "__main__":
    data_directory = Path("data/student_resource/dataset")
    run_training_and_validation(data_directory, n_s1_samples=3000, val_ratio=0.25)
