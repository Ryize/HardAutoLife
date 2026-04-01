import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from metrics import evaluate_entities, extract_reference_groups, update_statistics_file


class MetricsTestCase(unittest.TestCase):
    def test_extract_reference_groups_supports_vehicles_format(self):
        payload = {
            "vehicles": [
                {"vehicle_id": "veh_1", "track_ids": ["trk_1", "trk_2"]},
                {"vehicle_id": "veh_2", "track_ids": ["trk_3"]},
            ]
        }

        groups = extract_reference_groups(payload)

        self.assertEqual(2, len(groups))
        self.assertEqual(["trk_1", "trk_2"], groups[0]["track_ids"])

    def test_extract_reference_groups_supports_entities_format(self):
        payload = {
            "entities": [
                {"entity_id": "entity_1", "track_ids": ["trk_1", "trk_2"]},
            ]
        }

        groups = extract_reference_groups(payload)

        self.assertEqual(1, len(groups))
        self.assertEqual(["trk_1", "trk_2"], groups[0]["track_ids"])

    def test_evaluate_entities_returns_perfect_scores_for_identical_grouping(self):
        predicted = [
            {"entity_id": "entity_1", "track_ids": ["trk_1", "trk_2"]},
            {"entity_id": "entity_2", "track_ids": ["trk_3"]},
        ]
        reference = [
            {"vehicle_id": "veh_1", "track_ids": ["trk_1", "trk_2"]},
            {"vehicle_id": "veh_2", "track_ids": ["trk_3"]},
        ]
        track_time_lookup = {
            "trk_1": {
                "start": datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 1, 1, 10, 1, tzinfo=timezone.utc),
            },
            "trk_2": {
                "start": datetime(2026, 1, 1, 10, 2, tzinfo=timezone.utc),
                "end": datetime(2026, 1, 1, 10, 3, tzinfo=timezone.utc),
            },
            "trk_3": {
                "start": datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 1, 1, 11, 1, tzinfo=timezone.utc),
            },
        }

        metrics = evaluate_entities(predicted, reference, track_time_lookup=track_time_lookup)

        self.assertEqual(1.0, metrics["pairwise_precision"])
        self.assertEqual(1.0, metrics["pairwise_recall"])
        self.assertEqual(1.0, metrics["pairwise_f1"])
        self.assertEqual(1.0, metrics["rand_accuracy"])
        self.assertEqual(1.0, metrics["exact_group_match"])
        self.assertEqual(1.0, metrics["cluster_purity_score"])
        self.assertEqual(1.0, metrics["entity_continuity_score"])
        self.assertEqual(1.0, metrics["temporal_coherence_score"])

    def test_evaluate_entities_penalizes_merge_and_split_errors(self):
        predicted = [
            {"entity_id": "entity_1", "track_ids": ["trk_1", "trk_2", "trk_3"]},
            {"entity_id": "entity_2", "track_ids": ["trk_4"]},
        ]
        reference = [
            {"vehicle_id": "veh_1", "track_ids": ["trk_1", "trk_2"]},
            {"vehicle_id": "veh_2", "track_ids": ["trk_3", "trk_4"]},
        ]
        track_time_lookup = {
            "trk_1": {
                "start": datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 1, 1, 10, 1, tzinfo=timezone.utc),
            },
            "trk_2": {
                "start": datetime(2026, 1, 1, 10, 2, tzinfo=timezone.utc),
                "end": datetime(2026, 1, 1, 10, 3, tzinfo=timezone.utc),
            },
            "trk_3": {
                "start": datetime(2026, 1, 1, 10, 4, tzinfo=timezone.utc),
                "end": datetime(2026, 1, 1, 10, 5, tzinfo=timezone.utc),
            },
            "trk_4": {
                "start": datetime(2026, 1, 1, 10, 6, tzinfo=timezone.utc),
                "end": datetime(2026, 1, 1, 10, 7, tzinfo=timezone.utc),
            },
        }

        metrics = evaluate_entities(predicted, reference, track_time_lookup=track_time_lookup)

        self.assertEqual(1, metrics["pairwise_true_positive"])
        self.assertEqual(2, metrics["pairwise_false_positive"])
        self.assertEqual(1, metrics["pairwise_false_negative"])
        self.assertAlmostEqual(1.0 / 3.0, metrics["pairwise_precision"])
        self.assertAlmostEqual(0.5, metrics["pairwise_recall"])
        self.assertLess(metrics["pairwise_f1"], 1.0)
        self.assertLess(metrics["cluster_purity_score"], 1.0)
        self.assertLess(metrics["exact_group_match"], 1.0)

    def test_update_statistics_file_creates_and_appends_history(self):
        metrics = {
            "pairwise_f1": 0.75,
            "request_processing_time_seconds": 1.23,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            stats_path = Path(tmp_dir) / "statistics.json"

            update_statistics_file(
                stats_path=stats_path,
                input_path=Path("examples/500.json"),
                reference_path=Path("result/result.json"),
                batch_id="batch_1",
                metrics=metrics,
            )
            update_statistics_file(
                stats_path=stats_path,
                input_path=Path("examples/500.json"),
                reference_path=Path("result/result.json"),
                batch_id="batch_2",
                metrics=metrics,
            )

            payload = json.loads(stats_path.read_text(encoding="utf-8"))

        self.assertEqual(2, payload["runs_count"])
        self.assertEqual(2, len(payload["history"]))
        self.assertEqual("batch_2", payload["last_run"]["batch_id"])
        self.assertEqual(0.75, payload["last_run"]["metrics"]["pairwise_f1"])


if __name__ == "__main__":
    unittest.main()
