import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "scripts/benchmark_sampling_prepass_graph.py"
spec = importlib.util.spec_from_file_location("sampling_prepass_bench", path)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class SummaryTests(unittest.TestCase):
    def test_isolated_sum_not_concurrent_span_sum(self):
        rows = [{"arm": "A", "sampling_ms": 1, "prepass_ms": 5},
                {"arm": "B", "both_ms": 5.5}, {"arm": "B", "both_ms": 5.5},
                {"arm": "A", "sampling_ms": 1, "prepass_ms": 5}]
        result = bench.summarize_abba(rows)["median"]
        self.assertEqual(result["standalone_sum_ms"], 6)
        self.assertEqual(result["saved_ms"], .5)
        self.assertAlmostEqual(result["both_over_standalone_sum"], 5.5 / 6)

    def test_reject_incomplete_block(self):
        with self.assertRaises(ValueError):
            bench.summarize_abba([{"arm": "A"}, {"arm": "B"}])


if __name__ == "__main__":
    unittest.main()
