import importlib.util
import json
from pathlib import Path
import sys
import unittest


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts"))
spec = importlib.util.spec_from_file_location("priority_abba_analysis", root / "scripts/analyze_sampling_prepass_priority_abba.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class PriorityAbbaTests(unittest.TestCase):
    def test_v13_names_default_priority_without_renaming_qualification(self):
        version = json.loads((root / "benchmarks/versions/v13.json").read_text())
        self.assertEqual(version["name"], analysis.VERSION_NAMES["default"])
        self.assertEqual(version["qualification_profile"], "full_mlp_overlap_graph")
        self.assertEqual(version["actual_renderer_variant_id"], "full_mlp_overlap_graph_pb0")
        config = version["sampling_prepass_configuration"]
        self.assertEqual(config["stream_priorities"], {"control": 0, "sampling": 0, "prepass": 0})
        self.assertFalse(config["SM_partition"])
        self.assertNotEqual(analysis.VERSION_NAMES["default"], analysis.VERSION_NAMES["high"])

    def test_outer_abba_compares_two_both_times_not_standalone_sum(self):
        rows = [{"arm": "A", "both_ms": 6.2}, {"arm": "B", "both_ms": 6.0},
                {"arm": "B", "both_ms": 6.1}, {"arm": "A", "both_ms": 6.3}]
        summary = analysis.paired_summary(rows)
        self.assertAlmostEqual(summary["default_both_ms"], 6.25)
        self.assertAlmostEqual(summary["high_both_ms"], 6.05)
        self.assertAlmostEqual(summary["saved_ms"], .2)
        self.assertAlmostEqual(summary["time_saved_percent"], 3.2)

    def test_incomplete_or_changed_order_rejected(self):
        for labels in (("A", "B"), ("A", "B", "A", "B"), ("B", "A", "A", "B")):
            with self.assertRaises(ValueError):
                analysis.paired_summary([{"arm": arm, "both_ms": 1} for arm in labels])


if __name__ == "__main__":
    unittest.main()
