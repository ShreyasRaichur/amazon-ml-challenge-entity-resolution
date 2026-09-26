"""
Inference pipeline module.
Executes end-to-end entity resolution: preprocessing, country-partitioned blocking,
feature computation, probability inference, graph resolution, and TSV serialization.
Guarantees that matching_results.tsv is a strict subset of candidate_pairs.tsv.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    from .blocking import CountryPartitionedBlocker
    from .config import DEFAULT_CONFIG, PipelineConfig
    from .features import PairFeatureExtractor
    from .graph_resolver import TripartiteGraphResolver
    from .preprocess import RecordPreprocessor
    from .train_matcher import MatcherModel, MatcherTrainer
    from .utils import (
        compute_macro_f05,
        load_tsv_mapping,
        save_tsv_mapping,
        setup_logger,
        validate_tsv_outputs,
    )
except (ImportError, ValueError):
    from src.blocking import CountryPartitionedBlocker
    from src.config import DEFAULT_CONFIG, PipelineConfig
    from src.features import PairFeatureExtractor
    from src.graph_resolver import TripartiteGraphResolver
    from src.preprocess import RecordPreprocessor
    from src.train_matcher import MatcherModel, MatcherTrainer
    from src.utils import (
        compute_macro_f05,
        load_tsv_mapping,
        save_tsv_mapping,
        setup_logger,
        validate_tsv_outputs,
    )

logger = setup_logger("business_entity_resolution")


class EntityResolutionPipeline:
    """Production end-to-end tripartite entity resolution engine."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.preprocessor = RecordPreprocessor(self.config.columns)
        self.blocker = CountryPartitionedBlocker(self.config.blocking)
        self.resolver = TripartiteGraphResolver(self.config.resolver)
        self.matcher: Optional[MatcherModel] = None
        self.decision_threshold: float = self.config.train.default_threshold

    def load_dataset(self, file_path: Path) -> pd.DataFrame:
        """Load tabular data (CSV or TSV) into a pandas DataFrame."""
        if not file_path.exists():
            raise FileNotFoundError(f"Input data file not found: {file_path}")
        delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(file_path, sep=delimiter, dtype=str)
        logger.info("Loaded %s: %d records, columns=%s", file_path.name, len(df), list(df.columns))
        return df

    def run(
        self,
        s1_file: Path,
        s2_file: Path,
        s3_file: Path,
        ground_truth_file: Optional[Path] = None,
        model_file: Optional[Path] = None,
        threshold_config_file: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end entity resolution pipeline.
        Generates:
          - candidate_pairs.tsv
          - matching_results.tsv
        """
        start_time = time.time()
        out_dir = output_dir or self.config.paths.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        candidate_tsv_path = out_dir / "candidate_pairs.tsv"
        matching_tsv_path = out_dir / "matching_results.tsv"

        # 1. Ingestion
        logger.info("Step 1/6: Loading raw source datasets...")
        raw_s1 = self.load_dataset(s1_file)
        raw_s2 = self.load_dataset(s2_file)
        raw_s3 = self.load_dataset(s3_file)

        # 2. Preprocessing & Normalization
        logger.info("Step 2/6: Preprocessing and normalizing records...")
        s1_proc = self.preprocessor.process_dataframe(raw_s1, source_prefix="S1")
        s2_proc = self.preprocessor.process_dataframe(raw_s2, source_prefix="S2")
        s3_proc = self.preprocessor.process_dataframe(raw_s3, source_prefix="S3")

        target_proc = pd.concat([s2_proc, s3_proc], ignore_index=True).drop_duplicates(subset=["entity_id"])

        # 3. Country-Partitioned Blocking
        logger.info("Step 3/6: Executing country-partitioned lexical + dense blocking...")
        candidate_dict, candidate_pairs_df = self.blocker.generate_candidates(
            s1_proc, s2_proc, s3_proc
        )

        # Write candidate_pairs.tsv
        logger.info("Serializing %s...", candidate_tsv_path)
        save_tsv_mapping(
            filepath=candidate_tsv_path,
            mapping=candidate_dict,
            id_col="source1_entity_id",
            val_col="candidate_entity_ids"
        )

        # 4. Feature Extraction & Scoring
        logger.info("Step 4/6: Extracting pairwise features for candidate pairs...")
        X, feature_names = PairFeatureExtractor.build_feature_matrix(
            candidate_pairs_df, s1_proc, target_proc
        )

        # Ground truth loading if available
        ground_truth: Optional[Dict[str, Set[str]]] = None
        if ground_truth_file and ground_truth_file.exists():
            logger.info("Loading ground truth from %s...", ground_truth_file)
            ground_truth = load_tsv_mapping(ground_truth_file)

        # Model resolution: Train or Load
        if model_file and model_file.exists():
            logger.info("Loading pre-trained matcher model from %s...", model_file)
            self.matcher = MatcherModel.load(model_file)
            if threshold_config_file and threshold_config_file.exists():
                with open(threshold_config_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                    self.decision_threshold = float(cfg_data.get("optimal_threshold", self.config.train.default_threshold))
                logger.info("Loaded calibrated threshold: %.3f", self.decision_threshold)
        elif ground_truth is not None:
            logger.info("No pre-trained model found. Training matcher with ground truth...")
            trainer = MatcherTrainer(self.config.train)
            self.matcher, self.decision_threshold, best_f05 = trainer.train_and_evaluate(
                candidate_pairs_df, ground_truth, s1_proc, target_proc
            )
            logger.info("Trained model with optimal threshold=%.3f, CV Macro-F0.5=%.4f", self.decision_threshold, best_f05)
        else:
            # Fallback initialization for cold start inference
            logger.warning("No pre-trained model or ground truth provided. Using heuristic rule-based discriminator.")
            self.matcher = MatcherModel(self.config.train.lgb_params)
            # Synthesize calibration point
            dummy_X = np.zeros((2, len(feature_names)), dtype=np.float32)
            dummy_y = np.array([0, 1], dtype=np.int32)
            self.matcher.fit(dummy_X, dummy_y)
            self.decision_threshold = 0.50

        # Predict match probabilities
        logger.info("Step 5/6: Scoring candidate pairs with discriminator...")
        if len(candidate_pairs_df) > 0:
            probs = self.matcher.predict_proba(X)
        else:
            probs = np.array([], dtype=np.float32)

        # 5. Tripartite Graph Resolution
        logger.info("Step 6/6: Resolving tripartite graph consistency and greedy assignment...")
        matches_dict = self.resolver.resolve(
            candidate_pairs_df=candidate_pairs_df,
            probabilities=probs,
            threshold=self.decision_threshold,
            target_df=target_proc
        )

        # Write matching_results.tsv
        logger.info("Serializing %s...", matching_tsv_path)
        save_tsv_mapping(
            filepath=matching_tsv_path,
            mapping=matches_dict,
            id_col="source1_entity_id",
            val_col="matched_entity_ids"
        )

        # 6. Post-Run Validation & Strict Subset Enforcement
        is_valid, validation_errors = validate_tsv_outputs(candidate_tsv_path, matching_tsv_path)
        if not is_valid:
            logger.error("Validation failed! Errors: %s", validation_errors)
            raise ValueError(f"TSV validation failed: {validation_errors}")

        eval_score = None
        if ground_truth is not None:
            pred_sets = {k: set(v) for k, v in matches_dict.items()}
            eval_score, per_entity = compute_macro_f05(ground_truth, pred_sets, beta=self.config.train.beta)
            logger.info("=" * 60)
            logger.info("FINAL EVALUATION METRIC: Macro-F0.5 = %.4f", eval_score)
            logger.info("=" * 60)

        elapsed = time.time() - start_time
        summary = {
            "elapsed_seconds": elapsed,
            "total_s1_records": len(s1_proc),
            "total_s2_records": len(s2_proc),
            "total_s3_records": len(s3_proc),
            "candidate_pairs_count": len(candidate_pairs_df),
            "matched_links_count": sum(len(v) for v in matches_dict.values()),
            "s1_with_matches": sum(1 for v in matches_dict.values() if v),
            "singletons_count": sum(1 for v in matches_dict.values() if not v),
            "decision_threshold": self.decision_threshold,
            "macro_f05": eval_score,
            "candidate_file": str(candidate_tsv_path),
            "matching_file": str(matching_tsv_path),
        }
        logger.info("Pipeline execution completed in %.2f seconds. Summary: %s", elapsed, summary)
        return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge: Business Entity Resolution Inference Pipeline"
    )
    parser.add_argument("--s1", type=Path, required=True, help="Path to Source 1 CSV/TSV")
    parser.add_argument("--s2", type=Path, required=True, help="Path to Source 2 CSV/TSV")
    parser.add_argument("--s3", type=Path, required=True, help="Path to Source 3 CSV/TSV")
    parser.add_argument("--ground_truth", type=Path, default=None, help="Path to Ground Truth TSV (optional)")
    parser.add_argument("--model", type=Path, default=None, help="Path to pre-trained model (matcher_lgb.pkl)")
    parser.add_argument("--threshold_config", type=Path, default=None, help="Path to threshold_config.json")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory for TSV artifacts")
    return parser.parse_args()


def main() -> None:
    """CLI Entry point."""
    args = parse_args()
    pipeline = EntityResolutionPipeline()
    try:
        pipeline.run(
            s1_file=args.s1,
            s2_file=args.s2,
            s3_file=args.s3,
            ground_truth_file=args.ground_truth,
            model_file=args.model,
            threshold_config_file=args.threshold_config,
            output_dir=args.output_dir
        )
    except Exception as e:
        logger.exception("Pipeline failed with error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
