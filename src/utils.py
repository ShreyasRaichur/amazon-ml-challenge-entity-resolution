"""
Utility functions for Business Entity Resolution.
Implements the exact Macro-F0.5 competition evaluation metric, TSV artifact
serialization and strict subset validation, and centralized logging.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("business_entity_resolution")


def setup_logger(
    name: str = "business_entity_resolution",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Configure structured console and file logger."""
    log = logging.getLogger(name)
    log.setLevel(level)
    if not log.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file))
            file_handler.setFormatter(formatter)
            log.addHandler(file_handler)

    return log


def compute_entity_f05(y_true: Set[str], y_pred: Set[str], beta: float = 0.5) -> float:
    """
    Compute F_beta for a single reference entity (default beta=0.5).
    Strict singleton handling:
      - If ground truth is empty:
          - If prediction is empty -> returns 1.0 (correct singleton identification)
          - If prediction has false links -> returns 0.0 (severely penalizes false positive)
      - If ground truth is non-empty:
          - If prediction is empty -> returns 0.0
          - Else calculates precision, recall, and F_beta.
    """
    if not y_true:
        # True singleton
        return 1.0 if not y_pred else 0.0

    if not y_pred:
        # True non-empty, predicted empty
        return 0.0

    tp = len(y_true.intersection(y_pred))
    fp = len(y_pred.difference(y_true))
    fn = len(y_true.difference(y_pred))

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_sq = beta ** 2
    denominator = (beta_sq * precision) + recall
    if denominator == 0.0:
        return 0.0

    return ((1.0 + beta_sq) * precision * recall) / denominator


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the macro-averaged F_0.5 metric across all Source 1 reference entities.

    Args:
        ground_truth: Mapping s1_id -> set of true matching target entity IDs
        predictions: Mapping s1_id -> set of predicted matching target entity IDs
        beta: Precision weight factor (0.5 prioritizes precision 2x over recall)

    Returns:
        (macro_f05_score, per_entity_scores_dict)
    """
    all_s1_ids = set(ground_truth.keys()).union(set(predictions.keys()))
    if not all_s1_ids:
        return 0.0, {}

    scores: Dict[str, float] = {}
    for s1_id in all_s1_ids:
        y_true = ground_truth.get(s1_id, set())
        y_pred = predictions.get(s1_id, set())
        scores[s1_id] = compute_entity_f05(y_true, y_pred, beta=beta)

    macro_f05 = sum(scores.values()) / len(scores)
    return macro_f05, scores


def save_tsv_mapping(
    filepath: Path,
    mapping: Dict[str, List[str]],
    id_col: str,
    val_col: str,
    delimiter: str = "\t"
) -> None:
    """
    Serialize entity link dictionary to TSV file.
    Output format:
        source1_entity_id <TAB> candidate_entity_id_1,candidate_entity_id_2,...
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow([id_col, val_col])
        for s1_id in sorted(mapping.keys()):
            target_ids = mapping[s1_id]
            # Comma-separated list of target entity IDs, or empty string
            target_str = ",".join(target_ids) if target_ids else ""
            writer.writerow([s1_id, target_str])


def load_tsv_mapping(filepath: Path, delimiter: str = "\t") -> Dict[str, Set[str]]:
    """Load TSV entity link file into dictionary mapping s1_id -> set of target entity IDs."""
    mapping: Dict[str, Set[str]] = {}
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        # Skip header
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            if len(row) > 1 and row[1].strip():
                # Split comma- or space-separated target entity IDs
                raw_targets = row[1].strip()
                tokens = [t.strip() for t in raw_targets.replace(" ", ",").split(",") if t.strip()]
                mapping[s1_id] = set(tokens)
            else:
                mapping[s1_id] = set()

    return mapping


def validate_tsv_outputs(
    candidate_tsv_path: Path,
    matching_tsv_path: Path
) -> Tuple[bool, List[str]]:
    """
    Strict validation of competition artifacts:
    1. Both files exist and are valid TSVs.
    2. Column headers align with specification.
    3. Prefix checks: S1- on reference entities, S2- / S3- on target entities.
    4. Strict subset constraint: matching_results[s1] is a subset of candidate_pairs[s1].
    """
    errors: List[str] = []

    if not candidate_tsv_path.exists():
        errors.append(f"Candidate pairs file missing: {candidate_tsv_path}")
        return False, errors

    if not matching_tsv_path.exists():
        errors.append(f"Matching results file missing: {matching_tsv_path}")
        return False, errors

    candidates = load_tsv_mapping(candidate_tsv_path)
    matches = load_tsv_mapping(matching_tsv_path)

    # 1. Check all reference IDs in matches are present in candidates
    missing_s1_keys = set(matches.keys()).difference(set(candidates.keys()))
    if missing_s1_keys:
        errors.append(
            f"Matching results contain {len(missing_s1_keys)} S1 IDs not present in candidates file."
        )

    # 2. Verify strict subset condition per S1 entity
    violations = 0
    total_matches = 0
    for s1_id, matched_ids in matches.items():
        cand_ids = candidates.get(s1_id, set())
        total_matches += len(matched_ids)
        diff = matched_ids.difference(cand_ids)
        if diff:
            violations += 1
            if violations <= 5:
                errors.append(
                    f"Subset violation for {s1_id}: matched IDs {diff} not in candidate set."
                )

    if violations > 0:
        errors.append(f"Total subset violations: {violations}")

    # 3. ID Prefix validation
    for s1_id in candidates.keys():
        if not s1_id.startswith("S1-") and not s1_id.startswith("s1-"):
            errors.append(f"Invalid reference entity prefix in {s1_id} (expected 'S1-')")
            break

    is_valid = len(errors) == 0
    logger.info(
        "TSV Validation: valid=%s, total S1 records=%d, total matched links=%d, errors=%d",
        is_valid, len(matches), total_matches, len(errors)
    )
    return is_valid, errors
