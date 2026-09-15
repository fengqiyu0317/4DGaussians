"""CPU contracts for Phase-4 sequence/order quality validation."""

import ast
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_tacker_modes.py"
SPEC = importlib.util.spec_from_file_location("validate_tacker_modes", MODULE_PATH)
VALIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE)


class SequenceContractTests(unittest.TestCase):
    def test_prefill_drain_and_steady_state_counts(self):
        for frames in (1, 2, 50, 500):
            expected = VALIDATE._expected_tacker_execution_counts(frames)
            steady = frames - 1
            self.assertEqual(expected["input_frames"], frames)
            self.assertEqual(expected["full_deformation"], 1)
            self.assertEqual(expected["prefix"], steady)
            self.assertEqual(expected["mixed_launches"], steady)
            self.assertEqual(expected["suffix"], steady)
            self.assertEqual(expected["solo_raster"], 1)
            self.assertEqual(expected["outputs"], frames)
            self.assertEqual(
                expected["selected_head_evaluations_per_head"], frames
            )

    def test_report_seals_order_counts_and_per_view_deltas(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for field in (
            '"pipeline_execution_counts"',
            '"output_order_passed"',
            '"execution_counts_passed"',
            '"expected_pipeline_execution_counts"',
            '"per_view_deltas"',
            'record["view_index"] = view_index',
        ):
            self.assertIn(field, source)
        self.assertIn(
            'errors.append("{} outputs are not in legacy input order"', source
        )

    def test_python37_grammar(self):
        ast.parse(
            MODULE_PATH.read_text(encoding="utf-8"),
            str(MODULE_PATH),
            feature_version=7,
        )


if __name__ == "__main__":
    unittest.main()
