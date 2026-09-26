#!/usr/bin/env python3
"""
End-to-End Autonomous Pipeline for Amazon ML Challenge: Business Entity Resolution.
Author: Google DeepMind Antigravity AI & Shreyas

Workflow:
1. PHASE 1 (ML Training & Tuning):
   - Ingests ground-truth training records across US and India.
   - Mines hard negatives via country-partitioned blocking.
   - Extracts multi-field similarity features (token Jaccard, normalized edit similarity,
     street number matching, postal code agreement, legal suffix consistency).
   - Trains the ML discriminator (HistGradientBoosting / LightGBM) with K-fold cross-validation.
   - Automatically tunes and solves for the optimal decision threshold that maximizes Macro-F0.5 (beta=0.5).
   - Evaluates on a held-out validation set and saves the tuned model & config artifacts to models/.

2. PHASE 2 (High-Performance C++ Resolution):
   - Loads the tuned optimal decision threshold and rule weights into the C++ direct-access engine.
   - C++ directly accesses test_source1.tsv, test_source2.tsv, and test_source3.tsv from disk.
   - Executes multi-threaded (8 CPU cores) country-partitioned inverted indexing (open set, including France).
   - Generates clean, filtered, properly formatted output TSVs:
     - output/candidate_pairs.tsv
     - output/matching_results.tsv

3. PHASE 3 (Automated Submission Validation):
   - Automatically runs utils/validate_submission.py to guarantee strict format compliance,
     proper headers, singletons formatting, and subset constraints.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cpp_bridge import CPPEngine
from src.train_challenge import run_training_and_validation
from src.utils import setup_logger

logger = setup_logger("run_challenge_pipeline")


def print_banner(text: str) -> None:
    width = 75
    print("\n" + "=" * width)
    print(f" {text}".center(width))
    print("=" * width + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Business Entity Resolution: ML Tuning & Direct C++ Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=4000,
        help="Number of reference S1 training records to sample for ML model training and threshold tuning"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training phase and use previously saved model/threshold if already present in models/"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/student_resource/dataset/train",
        help="Path to training dataset folder"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/student_resource/dataset/test",
        help="Path to test dataset folder"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Target output directory for submission TSV files"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of CPU worker threads for parallel C++ indexing and scoring"
    )

    args = parser.parse_args()

    train_path = (PROJECT_ROOT / args.train_dir).resolve()
    test_path = (PROJECT_ROOT / args.test_dir).resolve()
    output_path = (PROJECT_ROOT / args.output_dir).resolve()
    models_path = (PROJECT_ROOT / "models").resolve()
    models_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    t_total_start = time.time()

    print_banner("AMAZON ML CHALLENGE: AUTONOMOUS RESOLUTION PIPELINE")
    logger.info("Initializing system on %d CPU threads...", args.threads)
    logger.info("Project Root: %s", PROJECT_ROOT)
    logger.info("Test Directory: %s", test_path)
    logger.info("Output Directory: %s", output_path)

    # Compile / check C++ engine
    if not CPPEngine.is_available():
        logger.error("C++ engine shared library failed to load or compile.")
        sys.exit(1)
    logger.info("C++ Core Engine is ready.")

    thresh_config_file = models_path / "threshold_config.json"
    optimal_threshold = 0.55  # Default precision-heavy threshold

    # =========================================================================
    # PHASE 1: ML Model Training & Hyperparameter / Threshold Tuning
    # =========================================================================
    if not args.skip_train or not thresh_config_file.exists():
        print_banner("PHASE 1: TRAINING ML DISCRIMINATOR & TUNING MACRO-F0.5 THRESHOLD")
        dataset_parent = train_path.parent
        t_train_start = time.time()
        
        train_summary = run_training_and_validation(
            dataset_dir=dataset_parent,
            n_s1_samples=args.train_samples,
            val_ratio=0.25
        )

        optimal_threshold = float(train_summary.get("optimal_threshold", 0.55))
        macro_f05 = float(train_summary.get("macro_f05", 0.0))
        singleton_acc = float(train_summary.get("singleton_accuracy", 0.0))

        logger.info(
            "Phase 1 Complete in %.2fs: Tuned Decision Threshold = %.3f | Validation Macro-F0.5 = %.4f | Singleton Accuracy = %.2f%%",
            time.time() - t_train_start, optimal_threshold, macro_f05, singleton_acc
        )
    else:
        logger.info("Loading existing tuned configuration from %s...", thresh_config_file)
        with open(thresh_config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            optimal_threshold = float(cfg.get("optimal_threshold", 0.55))
        logger.info("Using saved optimal threshold: %.3f", optimal_threshold)

    # Save finalized execution config using data-driven tuned threshold and calibrated multi-field penalties
    exec_config = {
        "match_threshold": round(optimal_threshold, 3),
        "min_candidate_score": 0.35,
        "street_num_bonus": 0.15,
        "street_num_penalty": -0.35,
        "postal_bonus": 0.15,
        "postal_penalty": -0.35,
        "addr_bonus": 0.15,
        "addr_penalty": -0.35,
        "max_candidates_per_entity": 10,
        "num_threads": args.threads
    }
    with open(models_path / "threshold_config.json", "w", encoding="utf-8") as f:
        json.dump(exec_config, f, indent=2)

    # =========================================================================
    # PHASE 2: High-Performance C++ Direct Data Ingestion & Resolution
    # =========================================================================
    print_banner("PHASE 2: DIRECT C++ MULTI-THREADED INGESTION & INFERENCE")
    s1_file = test_path / "test_source1.tsv"
    s2_file = test_path / "test_source2.tsv"
    s3_file = test_path / "test_source3.tsv"

    for f in (s1_file, s2_file, s3_file):
        if not f.exists():
            logger.error("Required test file not found: %s", f)
            sys.exit(1)

    t_infer_start = time.time()
    logger.info("Executing C++ resolution pipeline over all test entities...")
    cpp_res = CPPEngine.run_pipeline(
        s1_path=s1_file,
        s2_path=s2_file,
        s3_path=s3_file,
        output_dir=output_path,
        rules=exec_config
    )

    t_infer_elapsed = time.time() - t_infer_start
    logger.info("Phase 2 Complete in %.2fs: %s", t_infer_elapsed, cpp_res)

    cand_file = output_path / "candidate_pairs.tsv"
    match_file = output_path / "matching_results.tsv"

    if not cand_file.exists() or not match_file.exists():
        logger.error("Output files were not generated properly.")
        sys.exit(1)

    # =========================================================================
    # PHASE 3: Official Submission Validation
    # =========================================================================
    print_banner("PHASE 3: SUBMISSION CONSTRAINT VALIDATION")
    val_script = PROJECT_ROOT / "data" / "student_resource" / "utils" / "validate_submission.py"
    
    if val_script.exists():
        logger.info("Auditing generated TSVs against official challenge validator...")
        val_cmd = [
            sys.executable, str(val_script),
            "--matching", str(match_file),
            "--candidate", str(cand_file),
            "--test-dir", str(test_path)
        ]
        val_proc = subprocess.run(val_cmd, capture_output=True, text=True)
        print(val_proc.stdout)
        if val_proc.returncode != 0:
            logger.error("Validation issues detected:\n%s", val_proc.stderr)
            sys.exit(val_proc.returncode)
        else:
            logger.info("VALIDATION PASSED: TSVs are 100% compliant and ready to submit.")
    else:
        logger.warning("Validation script not found at %s. Skipping validation step.", val_script)

    total_time = time.time() - t_total_start
    total_s1 = cpp_res.get('total_s1', 0)
    singletons = cpp_res.get('singletons', 0)
    singleton_pct = (singletons / max(1, total_s1)) * 100.0

    print_banner("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  Total Pipeline Time:        {total_time:.2f} seconds")
    print(f"  Decision Threshold (F0.5):  {exec_config['match_threshold']:.3f}")
    print(f"  Total S1 Records Scored:    {total_s1:,}")
    print(f"  Total Candidates Generated: {cpp_res.get('total_candidates', 0):,}")
    print(f"  Total Matches Resolved:     {cpp_res.get('total_matches', 0):,}")
    print(f"  Singletons (Empty Lists):   {singletons:,} ({singleton_pct:.2f}% | Ground Truth target: ~5.80%)")
    print(f"  Matching Output File:       {match_file} ({match_file.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  Candidate Output File:      {cand_file} ({cand_file.stat().st_size / (1024*1024):.1f} MB)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
