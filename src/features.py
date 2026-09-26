"""
Feature engineering module for candidate pair classification.
Extracts high-dimensional pairwise features: RapidFuzz string metrics,
token set intersections, legal corporate suffix alignment, street number
numeric consistency masks, and postal/phone geographic discriminators.
"""

from __future__ import annotations

import difflib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Attempt to import RapidFuzz, falling back to pure-Python implementations
try:
    from rapidfuzz import distance, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    logger.info("RapidFuzz not found. Using high-performance difflib fallback.")


class FastStringMetrics:
    """Calculates string similarity metrics using RapidFuzz or difflib fallback."""

    @staticmethod
    def ratio(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if HAS_RAPIDFUZZ:
            return float(fuzz.ratio(s1, s2)) / 100.0
        return difflib.SequenceMatcher(None, s1, s2).ratio()

    @staticmethod
    def partial_ratio(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if HAS_RAPIDFUZZ:
            return float(fuzz.partial_ratio(s1, s2)) / 100.0
        # Difflib approximation for partial ratio
        short, long_ = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
        if not short:
            return 0.0
        matcher = difflib.SequenceMatcher(None, short, long_)
        blocks = matcher.get_matching_blocks()
        best_ratio = 0.0
        for block in blocks:
            start = max(0, block.b - block.a)
            sub = long_[start:start + len(short)]
            r = difflib.SequenceMatcher(None, short, sub).ratio()
            if r > best_ratio:
                best_ratio = r
        return best_ratio

    @staticmethod
    def token_sort_ratio(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if HAS_RAPIDFUZZ:
            return float(fuzz.token_sort_ratio(s1, s2)) / 100.0
        t1 = " ".join(sorted(s1.split()))
        t2 = " ".join(sorted(s2.split()))
        return difflib.SequenceMatcher(None, t1, t2).ratio()

    @staticmethod
    def token_set_ratio(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if HAS_RAPIDFUZZ:
            return float(fuzz.token_set_ratio(s1, s2)) / 100.0
        tok1 = set(s1.split())
        tok2 = set(s2.split())
        intersection = tok1.intersection(tok2)
        diff1 = tok1.difference(tok2)
        diff2 = tok2.difference(tok1)

        sorted_sect = " ".join(sorted(intersection))
        sorted_1 = (" ".join(sorted(intersection)) + " " + " ".join(sorted(diff1))).strip()
        sorted_2 = (" ".join(sorted(intersection)) + " " + " ".join(sorted(diff2))).strip()

        r1 = difflib.SequenceMatcher(None, sorted_sect, sorted_1).ratio()
        r2 = difflib.SequenceMatcher(None, sorted_sect, sorted_2).ratio()
        r3 = difflib.SequenceMatcher(None, sorted_1, sorted_2).ratio()
        return max(r1, r2, r3)

    @staticmethod
    def jaro_winkler(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if HAS_RAPIDFUZZ:
            return float(distance.JaroWinkler.similarity(s1, s2))
        # Approximation via SequenceMatcher ratio
        return difflib.SequenceMatcher(None, s1, s2).ratio()


class PairFeatureExtractor:
    """
    Extracts rich pairwise feature vectors for discriminator training and inference.
    Specifically engineered to penalize false positives and optimize Macro-F0.5.
    """

    FEATURE_NAMES: List[str] = [
        # Business Name Metrics
        "name_ratio",
        "name_partial_ratio",
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_jaro_winkler",
        "name_exact_match",
        "name_length_diff",
        "name_length_ratio",
        "name_token_jaccard",
        "name_shared_token_count",
        # Root Name Metrics (Legal Suffixes Stripped)
        "root_name_ratio",
        "root_name_token_sort_ratio",
        "root_name_token_set_ratio",
        "root_name_exact_match",
        # Acronym & Abbreviation Metrics
        "acronym_match",
        # Legal Entity Suffix Features
        "legal_type_exact_match",
        "legal_type_mismatch",
        "legal_type_missing",
        # Numeric Street Address Masks (Extreme False Positive Discriminator)
        "street_number_exact_match",
        "street_number_mismatch",
        "has_street_numbers_both",
        # Address & Street Metrics
        "address_ratio",
        "address_token_sort_ratio",
        "address_token_set_ratio",
        "address_token_jaccard",
        # Locality & Region Metrics
        "postal_code_exact_match",
        "postal_code_prefix_match",
        "postal_code_mismatch",
        "city_ratio",
        "city_exact_match",
        "state_ratio",
        "country_exact_match",
        # Phone Verification
        "phone_exact_match",
        "phone_has_both",
        # Missingness Indicators
        "is_missing_address",
        "is_missing_postal",
        "is_missing_city",
        "is_missing_phone",
    ]

    @classmethod
    def extract_pair_features(cls, r1: Dict[str, Any], r2: Dict[str, Any]) -> Dict[str, float]:
        """Compute all pairwise features between reference record r1 and target record r2."""
        feat: Dict[str, float] = {}

        # 1. Business Name Metrics
        n1 = r1.get("clean_name", "")
        n2 = r2.get("clean_name", "")

        feat["name_ratio"] = FastStringMetrics.ratio(n1, n2)
        feat["name_partial_ratio"] = FastStringMetrics.partial_ratio(n1, n2)
        feat["name_token_sort_ratio"] = FastStringMetrics.token_sort_ratio(n1, n2)
        feat["name_token_set_ratio"] = FastStringMetrics.token_set_ratio(n1, n2)
        feat["name_jaro_winkler"] = FastStringMetrics.jaro_winkler(n1, n2)
        feat["name_exact_match"] = 1.0 if (n1 and n1 == n2) else 0.0

        len1, len2 = len(n1), len(n2)
        feat["name_length_diff"] = float(abs(len1 - len2))
        feat["name_length_ratio"] = min(len1, len2) / max(1, max(len1, len2))

        toks1 = set(n1.split())
        toks2 = set(n2.split())
        union_toks = toks1.union(toks2)
        inter_toks = toks1.intersection(toks2)
        feat["name_token_jaccard"] = len(inter_toks) / max(1, len(union_toks))
        feat["name_shared_token_count"] = float(len(inter_toks))

        # 2. Root Name Metrics
        rn1 = r1.get("root_name", "")
        rn2 = r2.get("root_name", "")
        feat["root_name_ratio"] = FastStringMetrics.ratio(rn1, rn2)
        feat["root_name_token_sort_ratio"] = FastStringMetrics.token_sort_ratio(rn1, rn2)
        feat["root_name_token_set_ratio"] = FastStringMetrics.token_set_ratio(rn1, rn2)
        feat["root_name_exact_match"] = 1.0 if (rn1 and rn1 == rn2) else 0.0

        # 3. Acronym Match
        acro1 = r1.get("acronym", "")
        acro2 = r2.get("acronym", "")
        acro_match = False
        if acro1 and (acro1 == rn2 or acro1 in toks2):
            acro_match = True
        if acro2 and (acro2 == rn1 or acro2 in toks1):
            acro_match = True
        feat["acronym_match"] = 1.0 if acro_match else 0.0

        # 4. Legal Entity Suffix Features
        lt1 = r1.get("legal_type", "none")
        lt2 = r2.get("legal_type", "none")
        has_lt1 = lt1 != "none"
        has_lt2 = lt2 != "none"

        feat["legal_type_missing"] = 1.0 if (not has_lt1 or not has_lt2) else 0.0
        feat["legal_type_exact_match"] = 1.0 if (has_lt1 and has_lt2 and lt1 == lt2) else 0.0
        feat["legal_type_mismatch"] = 1.0 if (has_lt1 and has_lt2 and lt1 != lt2) else 0.0

        # 5. Numeric Street Address Masks (Crucial against false positives)
        raw_sn1 = r1.get("street_numbers", "")
        raw_sn2 = r2.get("street_numbers", "")
        sn1 = set(raw_sn1.split(",")) if raw_sn1 else set()
        sn2 = set(raw_sn2.split(",")) if raw_sn2 else set()
        sn1.discard("")
        sn2.discard("")

        has_both_nums = bool(sn1 and sn2)
        feat["has_street_numbers_both"] = 1.0 if has_both_nums else 0.0
        if has_both_nums:
            has_intersect = bool(sn1.intersection(sn2))
            feat["street_number_exact_match"] = 1.0 if has_intersect else 0.0
            # Mismatch: both have street numbers but zero overlap (e.g. 100 vs 102 Main St)
            feat["street_number_mismatch"] = 1.0 if not has_intersect else 0.0
        else:
            feat["street_number_exact_match"] = 0.0
            feat["street_number_mismatch"] = 0.0

        # 6. Address & Street Metrics
        a1 = r1.get("clean_address", "")
        a2 = r2.get("clean_address", "")
        feat["address_ratio"] = FastStringMetrics.ratio(a1, a2)
        feat["address_token_sort_ratio"] = FastStringMetrics.token_sort_ratio(a1, a2)
        feat["address_token_set_ratio"] = FastStringMetrics.token_set_ratio(a1, a2)

        addr_toks1 = set(a1.split())
        addr_toks2 = set(a2.split())
        addr_union = addr_toks1.union(addr_toks2)
        addr_inter = addr_toks1.intersection(addr_toks2)
        feat["address_token_jaccard"] = len(addr_inter) / max(1, len(addr_union))

        # 7. Locality & Region Metrics
        p1 = str(r1.get("postal_code", "")).strip().lower()
        p2 = str(r2.get("postal_code", "")).strip().lower()
        has_p1 = bool(p1)
        has_p2 = bool(p2)

        if has_p1 and has_p2:
            feat["postal_code_exact_match"] = 1.0 if p1 == p2 else 0.0
            feat["postal_code_prefix_match"] = 1.0 if (len(p1) >= 3 and len(p2) >= 3 and p1[:3] == p2[:3]) else 0.0
            feat["postal_code_mismatch"] = 1.0 if p1 != p2 else 0.0
        else:
            feat["postal_code_exact_match"] = 0.0
            feat["postal_code_prefix_match"] = 0.0
            feat["postal_code_mismatch"] = 0.0

        c1 = r1.get("city", "")
        c2 = r2.get("city", "")
        feat["city_ratio"] = FastStringMetrics.ratio(c1, c2)
        feat["city_exact_match"] = 1.0 if (c1 and c1 == c2) else 0.0

        s_st1 = r1.get("state", "")
        s_st2 = r2.get("state", "")
        feat["state_ratio"] = FastStringMetrics.ratio(s_st1, s_st2)

        country1 = r1.get("country", "")
        country2 = r2.get("country", "")
        feat["country_exact_match"] = 1.0 if (country1 and country1 == country2) else 0.0

        # 8. Phone Verification
        ph1 = str(r1.get("phone_digits", "")).strip()
        ph2 = str(r2.get("phone_digits", "")).strip()
        has_ph1 = len(ph1) >= 7
        has_ph2 = len(ph2) >= 7
        feat["phone_has_both"] = 1.0 if (has_ph1 and has_ph2) else 0.0
        feat["phone_exact_match"] = 1.0 if (has_ph1 and has_ph2 and ph1 == ph2) else 0.0

        # 9. Missingness Indicators
        feat["is_missing_address"] = 1.0 if (not a1 or not a2) else 0.0
        feat["is_missing_postal"] = 1.0 if (not has_p1 or not has_p2) else 0.0
        feat["is_missing_city"] = 1.0 if (not c1 or not c2) else 0.0
        feat["is_missing_phone"] = 1.0 if (not has_ph1 or not has_ph2) else 0.0

        return feat

    @classmethod
    def build_feature_matrix(
        cls,
        candidate_pairs_df: pd.DataFrame,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Batch-extract feature matrix X for candidate pairs.
        Returns:
            - X: np.ndarray of shape (N_pairs, N_features)
            - feature_names: List of feature names
        """
        if candidate_pairs_df.empty:
            return np.zeros((0, len(cls.FEATURE_NAMES)), dtype=np.float32), cls.FEATURE_NAMES

        # Index records by entity_id for O(1) lookup
        s1_records = s1_df.set_index("entity_id").to_dict(orient="index")
        target_records = target_df.set_index("entity_id").to_dict(orient="index")

        rows: List[List[float]] = []
        feature_names = cls.FEATURE_NAMES

        for _, row in candidate_pairs_df.iterrows():
            s1_id = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"]

            r1 = s1_records.get(s1_id, {})
            r2 = target_records.get(cand_id, {})

            feat_dict = cls.extract_pair_features(r1, r2)
            rows.append([feat_dict[k] for k in feature_names])

        X = np.asarray(rows, dtype=np.float32)
        return X, feature_names
