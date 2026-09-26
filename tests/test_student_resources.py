"""
Unit test verifying BusinessEntityResolutionAgent with student resources data files.
"""

import unittest
from pathlib import Path

from src.agent import BusinessEntityResolutionAgent


class TestStudentResourcesAgent(unittest.TestCase):
    """Test suite verifying resource discovery, ingestion, and inspection on student_resource dataset."""

    def setUp(self) -> None:
        self.base_dir = Path(__file__).resolve().parent.parent
        self.train_dir = self.base_dir / "data" / "student_resource" / "dataset" / "train"
        self.test_dir = self.base_dir / "data" / "student_resource" / "dataset" / "test"

    def test_student_resources_discovery_and_inspection(self) -> None:
        """Verify agent ingesting and inspecting student_resource train data."""
        agent = BusinessEntityResolutionAgent()
        
        # Test directory auto-discovery
        ingest_summary = agent.ingest_resources(
            s1=self.train_dir / "train_source1.tsv",
            s2=self.train_dir / "train_source2.tsv",
            s3=self.train_dir / "train_source3.tsv",
            ground_truth=self.train_dir / "train_ground_truth.tsv"
        )

        self.assertEqual(ingest_summary["status"], "ingested_successfully")
        self.assertGreater(ingest_summary["s1"]["record_count"], 1_000_000)
        self.assertGreater(ingest_summary["s2"]["record_count"], 1_000_000)
        self.assertGreater(ingest_summary["s3"]["record_count"], 1_000_000)

        # Inspect data resources
        report = agent.inspect_resources()
        self.assertIn("overall_health_score", report)
        self.assertGreaterEqual(report["overall_health_score"], 70.0)


if __name__ == "__main__":
    unittest.main()
