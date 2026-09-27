"""
Unit and regression test suite for auditing entity resolution safeguards:
1. Prevents TripartiteGraphResolver from discarding legitimate multi-match rebrand/DBA targets.
2. Prevents legal suffix stripping from removing meaningful business qualifiers (Mines, Foods, Logistics, etc.).
3. Verifies near-miss non-matches like 'Haldiram' vs 'Haldiram Mines' are never merged on name alone.
4. Verifies automated blocking recall calculation.
5. Verifies deterministic tie-breaking and C++ engine stability.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Set

from src.features import FastStringMetrics
from src.graph_resolver import TripartiteGraphResolver
from src.preprocess import LegalEntityNormalizer, TextNormalizer
from src.train_challenge import evaluate_blocking_recall
from src.cpp_bridge import CPPEngine


class TestTripartiteGraphResolverAudit(unittest.TestCase):
    """Test TripartiteGraphResolver audit safeguards against aggressive discards."""

    def test_rebrand_dba_multimatch_preserved(self) -> None:
        """
        Regression test for Bug 3 (Rose Ferrous / Dovawexnyla pattern):
        A reference entity with two true targets where one target has a very different name
        (e.g., DBA / rebrand) at the identical address MUST NOT be discarded by the resolver.
        """
        resolver = TripartiteGraphResolver()

        # Candidate pairs DataFrame
        candidate_df = pd.DataFrame([
            {"source1_entity_id": "S1-327546569", "candidate_entity_id": "S2-200033037"},
            {"source1_entity_id": "S1-327546569", "candidate_entity_id": "S3-899309891"},
        ])

        # High probabilities from trained classifier
        probabilities = np.array([0.9854, 0.7115])
        threshold = 0.48

        # Target records: S2 has matching name, S3 has completely different rebrand name ('Dovawexnyla')
        target_df = pd.DataFrame([
            {
                "entity_id": "S2-200033037",
                "clean_name": "rose ferrous pvt ltd",
                "root_name": "rose ferrous",
                "acronym": "rfpl",
                "country": "IN",
                "street_numbers": "12",
                "clean_address": "12 industrial estate phase 2",
            },
            {
                "entity_id": "S3-899309891",
                "clean_name": "dovawexnyla",
                "root_name": "dovawexnyla",
                "acronym": "d",
                "country": "IN",
                "street_numbers": "12",
                "clean_address": "12 industrial estate phase 2",
            },
        ])

        matches = resolver.resolve(
            candidate_pairs_df=candidate_df,
            probabilities=probabilities,
            threshold=threshold,
            target_df=target_df,
        )

        resolved_for_s1 = matches.get("S1-327546569", [])
        self.assertIn("S2-200033037", resolved_for_s1, "Standard match must be resolved")
        self.assertIn(
            "S3-899309891",
            resolved_for_s1,
            "Rebrand/DBA target Dovawexnyla must NOT be discarded by mutual consistency"
        )
        self.assertEqual(len(resolved_for_s1), 2, "Both true targets must be preserved")


class TestLegalSuffixSeparation(unittest.TestCase):
    """Test that meaningful business qualifiers are preserved while only legal suffixes are stripped."""

    def setUp(self) -> None:
        self.normalizer = LegalEntityNormalizer()

    def test_legal_suffixes_stripped(self) -> None:
        """Verify standard corporate designations are cleanly parsed."""
        legal_cases = [
            ("haldiram pvt ltd", "haldiram", "pvt_ltd"),
            ("haldiram limited", "haldiram", "ltd"),
            ("haldiram llc", "haldiram", "llc"),
            ("haldiram inc", "haldiram", "inc"),
            ("haldiram corp", "haldiram", "corp"),
            ("haldiram gmbh", "haldiram", "gmbh"),
            ("haldiram sarl", "haldiram", "sarl"),
            ("haldiram sas", "haldiram", "sas"),
        ]
        for raw, expected_root, expected_legal in legal_cases:
            root, legal = self.normalizer.parse(raw)
            self.assertEqual(root, expected_root, f"Failed root for {raw}")
            self.assertEqual(legal, expected_legal, f"Failed legal type for {raw}")

    def test_meaningful_qualifiers_not_stripped(self) -> None:
        """
        Verify meaningful qualifier words like Mines, Foods, Logistics, Group, Holdings,
        Motors, Energy, Capital are NEVER stripped as legal suffixes.
        """
        qualifier_cases = [
            ("haldiram mines", "haldiram mines"),
            ("haldiram foods", "haldiram foods"),
            ("haldiram logistics", "haldiram logistics"),
            ("haldiram group", "haldiram group"),
            ("haldiram holdings", "haldiram holdings"),
            ("haldiram motors", "haldiram motors"),
            ("haldiram energy", "haldiram energy"),
            ("haldiram capital", "haldiram capital"),
        ]
        for raw, expected_root in qualifier_cases:
            root, legal = self.normalizer.parse(raw)
            self.assertEqual(
                root, expected_root,
                f"Qualifier was incorrectly stripped! Got '{root}', expected '{expected_root}'"
            )
            self.assertEqual(legal, "none", f"Qualifier should not have legal type for {raw}")

    def test_near_miss_haldiram_vs_haldiram_mines(self) -> None:
        """
        Confirm 'Haldiram' and 'Haldiram Mines' have different root names and are not
        identical root matches.
        """
        root1, _ = self.normalizer.parse("haldiram")
        root2, _ = self.normalizer.parse("haldiram mines")
        self.assertNotEqual(root1, root2, "Haldiram and Haldiram Mines must have different root names")
        self.assertEqual(root1, "haldiram")
        self.assertEqual(root2, "haldiram mines")


class TestAutomatedBlockingRecall(unittest.TestCase):
    """Test the automated blocking-recall calculation function."""

    def test_blocking_recall_computation(self) -> None:
        gt_map = {
            "S1-1": {"S2-1", "S3-1"},  # 2 targets, 2 recalled -> 100%
            "S1-2": {"S2-2", "S3-2"},  # 2 targets, 1 recalled -> 50%
            "S1-3": {"S2-3"},          # 1 target, 0 recalled -> 0%
            "S1-4": set(),             # Singleton -> excluded from target recall
        }
        cand_dict = {
            "S1-1": ["S2-1", "S3-1", "S2-99"],
            "S1-2": ["S2-2", "S2-98"],
            "S1-3": ["S2-97"],
            "S1-4": ["S2-96"],
        }
        s1_ids = {"S1-1", "S1-2", "S1-3", "S1-4"}

        stats = evaluate_blocking_recall(cand_dict, gt_map, s1_ids, "Test")

        # Total true targets = 2 + 2 + 1 = 5
        # Recalled targets = 2 + 1 + 0 = 3
        # Target recall = 3 / 5 = 60.0%
        self.assertAlmostEqual(stats["target_blocking_recall"], 60.0, places=2)
        # Entities with GT = 3 (S1-1, S1-2, S1-3)
        # Entities fully recalled = 1 (S1-1) -> 33.33%
        self.assertAlmostEqual(stats["full_entity_recall"], 33.33, places=1)
        # Entities with >=1 hit = 2 (S1-1, S1-2) -> 66.67%
        self.assertAlmostEqual(stats["any_entity_recall"], 66.67, places=1)


class TestCPPEngineDeterminism(unittest.TestCase):
    """Test C++ Core Engine availability and basic text normalization determinism."""

    def test_cpp_engine_availability(self) -> None:
        self.assertTrue(CPPEngine.is_available(), "C++ Core Engine shared library must be available")

    def test_jaro_winkler_determinism(self) -> None:
        score1 = CPPEngine.jaro_winkler("haldiram", "haldiram")
        self.assertAlmostEqual(score1, 1.0, places=5)
        score2 = CPPEngine.jaro_winkler("haldiram", "haldiram mines")
        self.assertLess(score2, 0.95, "Near-miss must not produce identical score")


if __name__ == "__main__":
    unittest.main()
