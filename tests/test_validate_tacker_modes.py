"""CPU contracts for Phase-4 sequence/order quality validation."""

import ast
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_tacker_modes.py"
SPEC = importlib.util.spec_from_file_location("validate_tacker_modes", MODULE_PATH)
VALIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE)


class SequenceContractTests(unittest.TestCase):
    def test_scene_name_normalisation_supports_all_dynerf_scenes(self):
        self.assertEqual(
            VALIDATE._normalise_scene_name("Coffee-Martini"),
            "coffee_martini",
        )
        self.assertEqual(
            VALIDATE._normalise_scene_name(" flame_salmon_1 "),
            "flame_salmon_1",
        )
        with self.assertRaises(ValueError):
            VALIDATE._normalise_scene_name("  ")

    def test_prefill_drain_and_steady_state_counts(self):
        with mock.patch.dict(
            os.environ,
            {"TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12": "0"},
        ):
            for frames in (1, 2, 50, 500):
                expected = VALIDATE._expected_tacker_execution_counts(frames)
                steady = frames - 1
                self.assertEqual(expected["input_frames"], frames)
                self.assertEqual(expected["full_deformation"], 1)
                self.assertEqual(expected["prefix"], steady)
                self.assertEqual(expected["mixed_launches"], steady)
                self.assertEqual(expected["raster_prepasses"], 0)
                self.assertEqual(expected["suffix"], steady)
                self.assertEqual(expected["solo_raster"], 1)
                self.assertEqual(expected["outputs"], frames)
                self.assertEqual(
                    expected["selected_head_evaluations_per_head"], frames
                )

    def test_v12_counts_one_raster_prepass_per_steady_frame(self):
        with mock.patch.dict(
            os.environ,
            {"TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12": "1"},
        ):
            expected = VALIDATE._expected_tacker_execution_counts(50)
        self.assertEqual(expected["raster_prepasses"], 49)

    def test_green_prepare_handoff_counts_exclude_prefill_and_drain(self):
        for enabled in ("0", "1"):
            with mock.patch.dict(os.environ, {
                "TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12": "1",
                "TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF": enabled,
            }, clear=True):
                expected = VALIDATE._expected_tacker_execution_counts(50)
                self.assertEqual(expected.get("raster_prepare_handoffs", 0),
                                 49 if enabled == "1" else 0)
                self.assertNotIn("raster_prepare_handoffs",
                                 VALIDATE._expected_tacker_execution_counts(1))

    def test_report_seals_order_counts_and_per_view_deltas(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for field in (
            '"pipeline_execution_counts"',
            '"output_order_passed"',
            '"execution_counts_passed"',
            '"expected_pipeline_execution_counts"',
            '"per_view_deltas"',
            'record["view_index"] = view_index',
            '"--expected-gaussian-count"',
            '"--expected-image-width"',
            '"--expected-image-height"',
        ):
            self.assertIn(field, source)
        self.assertIn(
            'errors.append("{} outputs are not in legacy input order"', source
        )

    def test_raster_hexplane_experiment_counts_preserve_strict_equality(self):
        for mode in ("fused", "separate", "batched", "block_fused"):
            with mock.patch.dict(os.environ, {
                "TACKER_EXPERIMENTAL_RASTER_HEXPLANE": mode,
                "TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12": "1",
            }):
                expected = VALIDATE._expected_tacker_execution_counts(10)
                self.assertEqual(expected["raster_prepasses"], 9)
                self.assertEqual(expected["raster_hexplane_fused_launches"],
                                 9 if mode in ("fused", "block_fused") else 0)
                self.assertEqual(expected["raster_hexplane_sampling_graph_replays"],
                                 9 if mode in ("separate", "batched") else 0)
                self.assertNotIn("raster_hexplane_fused_launches", VALIDATE._expected_tacker_execution_counts(1))

    def test_python37_grammar(self):
        ast.parse(
            MODULE_PATH.read_text(encoding="utf-8"),
            str(MODULE_PATH),
            feature_version=7,
        )

    def test_partial_sampling_graph_counts_are_exact_and_opt_in(self):
        for mode in ("fused", "separate"):
            for planes in (1, 3, 6, 9, 12):
                with mock.patch.dict(os.environ, {
                    "TACKER_EXPERIMENTAL_RASTER_HEXPLANE": mode,
                    "TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12": "1",
                    "TACKER_RASTER_HEXPLANE_MIXED_PLANES": str(planes),
                }, clear=True):
                    expected = VALIDATE._expected_tacker_execution_counts(50)
                    self.assertEqual(expected.get("raster_hexplane_early_sampling_graph_replays", 0),
                                     49 if planes < 12 else 0)
                    self.assertNotIn("raster_hexplane_early_sampling_graph_replays",
                                     VALIDATE._expected_tacker_execution_counts(1))


if __name__ == "__main__":
    unittest.main()
