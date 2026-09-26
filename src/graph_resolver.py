"""
Graph resolver module for tripartite consistency and greedy entity clustering.
Enforces cycle consistency across S1-S2-S3 clusters, resolves multi-source
conflicts, and eliminates false positives to maximize Macro-F0.5.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from .config import DEFAULT_CONFIG, ResolverConfig
    from .features import FastStringMetrics
except (ImportError, ValueError):
    from src.config import DEFAULT_CONFIG, ResolverConfig
    from src.features import FastStringMetrics

logger = logging.getLogger(__name__)


class TripartiteGraphResolver:
    """
    Solves the global tripartite entity resolution assignment problem across
    Source 1 (Reference), Source 2 (Target), and Source 3 (Target).
    """

    def __init__(self, config: Optional[ResolverConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG.resolver

    def check_target_mutual_consistency(
        self,
        t1_record: Dict[str, Any],
        t2_record: Dict[str, Any]
    ) -> float:
        """
        Verify mutual consistency between two target records (e.g. S2-x and S3-y)
        linked to the same reference entity.
        Returns a consistency multiplier in [0.0, 1.2].
        """
        # Country conflict check
        c1 = t1_record.get("country", "")
        c2 = t2_record.get("country", "")
        if c1 and c2 and c1 != "UNKNOWN" and c2 != "UNKNOWN" and c1 != c2:
            return 0.1  # Severe cross-country contradiction

        # Street number conflict check
        sn1_str = t1_record.get("street_numbers", "")
        sn2_str = t2_record.get("street_numbers", "")
        sn1 = set(sn1_str.split(",")) if sn1_str else set()
        sn2 = set(sn2_str.split(",")) if sn2_str else set()
        sn1.discard("")
        sn2.discard("")
        if sn1 and sn2 and not sn1.intersection(sn2):
            return 0.3  # Conflicting street numbers

        # Name compatibility
        name1 = t1_record.get("clean_name", "")
        name2 = t2_record.get("clean_name", "")
        root1 = t1_record.get("root_name", "")
        root2 = t2_record.get("root_name", "")
        acro1 = t1_record.get("acronym", "")
        acro2 = t2_record.get("acronym", "")

        ratio = FastStringMetrics.token_set_ratio(name1, name2)
        root_ratio = FastStringMetrics.token_set_ratio(root1, root2)
        is_acronym = bool(
            (acro1 and (acro1 == root2 or acro1 in root2.split())) or
            (acro2 and (acro2 == root1 or acro2 in root1.split()))
        )
        if max(ratio, root_ratio) < 0.30 and not is_acronym:
            return 0.2  # Severe name conflict

        # Mutual support
        if max(ratio, root_ratio) >= 0.75 or is_acronym:
            return 1.15

        return 1.0

    def resolve(
        self,
        candidate_pairs_df: pd.DataFrame,
        probabilities: np.ndarray,
        threshold: float,
        target_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, List[str]]:
        """
        Execute greedy assignment with tripartite consistency validation.

        Args:
            candidate_pairs_df: DataFrame with ['source1_entity_id', 'candidate_entity_id']
            probabilities: Match probability array aligned with candidate_pairs_df
            threshold: Calibrated decision threshold maximizing Macro-F0.5
            target_df: Optional DataFrame of target records for cross-target verification

        Returns:
            Dict mapping s1_id -> list of matched target entity IDs
        """
        if candidate_pairs_df.empty:
            return {}

        target_records: Dict[str, Dict[str, Any]] = {}
        if target_df is not None and not target_df.empty:
            target_records = target_df.set_index("entity_id").to_dict(orient="index")

        # 1. Gather all candidate pairs per S1 entity with probability
        s1_candidates: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        candidate_sets: Dict[str, Set[str]] = defaultdict(set)

        for idx, row in candidate_pairs_df.iterrows():
            s1_id = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"]
            prob = float(probabilities[idx])
            candidate_sets[s1_id].add(cand_id)
            if prob >= threshold:
                s1_candidates[s1_id].append((cand_id, prob))

        # 2. Competitive Target Resolution:
        # A single target entity should not be claimed by multiple conflicting S1 entities.
        # Track target claims: target_id -> list of (s1_id, prob)
        target_claims: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for s1_id, pairs in s1_candidates.items():
            for cand_id, prob in pairs:
                target_claims[cand_id].append((s1_id, prob))

        # Determine winner for contested target entities
        accepted_assignments: Dict[str, Set[str]] = defaultdict(set)
        for cand_id, claims in target_claims.items():
            if len(claims) == 1:
                # Uncontested
                accepted_assignments[claims[0][0]].add(cand_id)
            else:
                # Contested: Winner is S1 entity with highest probability
                claims.sort(key=lambda x: x[1], reverse=True)
                best_s1, best_prob = claims[0]
                runner_up_s1, runner_up_prob = claims[1]
                # If clear winner, assign to best S1
                if best_prob - runner_up_prob >= 0.05:
                    accepted_assignments[best_s1].add(cand_id)
                else:
                    # In case of near-tie, assign only if above conservative high threshold
                    if best_prob >= threshold + 0.08:
                        accepted_assignments[best_s1].add(cand_id)

        # 3. Tripartite Consistency Verification (Cross S2-S3 coherence)
        final_matches: Dict[str, List[str]] = {}
        all_s1_ids = candidate_pairs_df["source1_entity_id"].unique()

        for s1_id in all_s1_ids:
            assigned = list(accepted_assignments.get(s1_id, set()))
            if not assigned:
                final_matches[s1_id] = []
                continue

            # Separate into S2 and S3 candidates
            s2_cands = [t for t in assigned if t.startswith("S2-") or t.startswith("s2-")]
            s3_cands = [t for t in assigned if t.startswith("S3-") or t.startswith("s3-")]

            # If both S2 and S3 candidates are present, check mutual consistency
            valid_cands = set(assigned)
            if s2_cands and s3_cands and target_records:
                for s2_id in s2_cands:
                    r2 = target_records.get(s2_id)
                    if not r2:
                        continue
                    for s3_id in s3_cands:
                        r3 = target_records.get(s3_id)
                        if not r3:
                            continue
                        consistency = self.check_target_mutual_consistency(r2, r3)
                        # Only discard if there is an irreconcilable cross-country contradiction (< 0.15)
                        # and one of the links is distinctly weaker (< 0.65)
                        if consistency < 0.15:
                            p2 = next((p for c, p in s1_candidates[s1_id] if c == s2_id), 0.0)
                            p3 = next((p for c, p in s1_candidates[s1_id] if c == s3_id), 0.0)
                            if p2 >= p3 and p3 < 0.65:
                                valid_cands.discard(s3_id)
                            elif p3 > p2 and p2 < 0.65:
                                valid_cands.discard(s2_id)

            # Ensure strict subset property
            valid_cands = valid_cands.intersection(candidate_sets[s1_id])
            final_matches[s1_id] = sorted(list(valid_cands))

        logger.info(
            "Graph resolution complete. Total S1 entities=%d, Entities with >=1 match=%d, Total matches=%d",
            len(final_matches),
            sum(1 for m in final_matches.values() if m),
            sum(len(m) for m in final_matches.values())
        )

        return final_matches
