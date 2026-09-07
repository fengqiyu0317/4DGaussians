"""CPU-only contracts for the fixed Tacker leaf/Raster profiler."""

import ast
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "profile_tacker_leaves.py"
SPEC = importlib.util.spec_from_file_location("profile_tacker_leaves", MODULE_PATH)
PROFILER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILER)


class _Args:
    pass


def _output_args(root):
    args = _Args()
    args.device_output = str(root / "device.json")
    args.raster_output = str(root / "raster.json")
    args.leaf_output = str(root / "leaf.json")
    args.report = str(root / "report.json")
    args.split = "test"
    return args


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary(samples):
    return PROFILER.summarize_samples(samples)


class StatisticsContractTest(unittest.TestCase):
    def test_summary_preserves_samples_and_interpolates_p95(self):
        summary = _summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["samples_ms"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["sample_count"], 4)
        self.assertEqual(summary["p50_ms"], 2.5)
        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertAlmostEqual(summary["p95_ms"], 3.85)
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 4.0)

    def test_single_sample_percentiles_are_defined(self):
        summary = _summary([0.125])
        self.assertEqual(summary["p50_ms"], 0.125)
        self.assertEqual(summary["p95_ms"], 0.125)

    def test_nan_infinity_zero_and_empty_fail_closed(self):
        for samples in (
            [],
            [float("nan")],
            [float("inf")],
            [float("-inf")],
            [0.0],
            [-0.1],
        ):
            with self.subTest(samples=samples):
                with self.assertRaises(ValueError):
                    PROFILER.summarize_samples(samples)


class SchemaContractTest(unittest.TestCase):
    def setUp(self):
        self.workload = {
            "scene": "flame_steak",
            "iteration": 14000,
            "resolution": [1352, 1014],
            "gaussian_count": 111525,
        }
        self.device = {
            "name": "NVIDIA RTX A6000",
            "compute_capability": [8, 6],
            "cuda_arch": "sm_86",
        }
        self.numerics = {"passed": True, "per_view_pair": [{}, {}]}

    def test_admission_fields_share_the_same_full_call_p50(self):
        device, raster, leaf = PROFILER.build_measurement_documents(
            self.workload,
            self.device,
            {"rasterizer": {}, "head": {}},
            self.numerics,
            _summary([8.0, 8.2]),
            _summary([8.4, 8.6]),
            _summary([0.7, 0.9]),
            sample_view_indices=[0, 1],
        )

        self.assertEqual(device["kind"], "4dgaussians_tacker_device")
        self.assertEqual(raster["kind"], "4dgaussians_tacker_raster_profile")
        self.assertEqual(leaf["kind"], "4dgaussians_tacker_leaf_profile")
        self.assertEqual(
            raster["measurements"]["solo_raster_p50_ms"],
            leaf["measurements"]["solo_raster_p50_ms"],
        )
        self.assertEqual(
            raster["measurements"]["mixed_raster_p50_ms"],
            leaf["measurements"]["mixed_p50_ms"],
        )
        self.assertEqual(leaf["measurements"]["solo_head_p50_ms"], 0.8)
        self.assertFalse(raster["measurement_semantics"]["kernel_only"])
        self.assertIn(
            "complete public forward_with_head call",
            leaf["measurement_semantics"]["mixed_full"],
        )
        self.assertEqual(raster["sample_view_indices"], [0, 1])
        self.assertEqual(
            raster["measurement_config"], {"persistent_blocks": 0}
        )
        self.assertEqual(
            leaf["measurement_config"], raster["measurement_config"]
        )

    def test_optional_gptb_is_diagnostic_not_an_admission_replacement(self):
        _, _, leaf = PROFILER.build_measurement_documents(
            self.workload,
            self.device,
            {},
            self.numerics,
            _summary([8.0]),
            _summary([8.5]),
            _summary([0.8]),
            gptb_head=_summary([0.6]),
        )
        self.assertEqual(leaf["measurements"]["solo_head_p50_ms"], 0.8)
        self.assertEqual(
            leaf["measurements"]["gptb_head_p50_ms_diagnostic"], 0.6
        )

    def test_nonfinite_p50_cannot_enter_output_schema(self):
        bad = {"p50_ms": float("nan")}
        with self.assertRaises(ValueError):
            PROFILER.build_measurement_documents(
                self.workload,
                self.device,
                {},
                self.numerics,
                bad,
                _summary([8.5]),
                _summary([0.8]),
            )


class AtomicOutputContractTest(unittest.TestCase):
    def test_failed_report_does_not_touch_measurement_targets(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = _output_args(root)
            sentinels = {
                "device": {"old": "device"},
                "raster": {"old": "raster"},
                "leaf": {"old": "leaf"},
            }
            for key, value in sentinels.items():
                Path(getattr(args, "{}_output".format(key))).write_text(
                    json.dumps(value), encoding="utf-8"
                )
            failed = {
                "schema_version": 1,
                "kind": "4dgaussians_tacker_leaf_profile_report",
                "passed": False,
                "errors": ["numerical mismatch"],
            }

            written = PROFILER.write_profile_outputs(
                None, None, None, failed, args
            )

            self.assertFalse(written["measurement_outputs_written"])
            for key, value in sentinels.items():
                self.assertEqual(
                    _read_json(getattr(args, "{}_output".format(key))), value
                )
            self.assertFalse(_read_json(args.report)["passed"])
            self.assertEqual(list(root.glob("*.tmp")), [])
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_success_writes_all_machine_readable_documents(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = _output_args(root)
            report = {"passed": True, "errors": []}
            device = {"device": {"name": "NVIDIA RTX A6000"}}
            raster = {"measurements": {"solo_raster_p50_ms": 8.0}}
            leaf = {"measurements": {"solo_head_p50_ms": 0.8}}

            written = PROFILER.write_profile_outputs(
                device, raster, leaf, report, args
            )

            self.assertTrue(written["measurement_outputs_written"])
            self.assertEqual(_read_json(args.device_output), device)
            self.assertEqual(_read_json(args.raster_output), raster)
            self.assertEqual(_read_json(args.leaf_output), leaf)
            self.assertTrue(_read_json(args.report)["passed"])

    def test_atomic_writer_rejects_nan_without_replacing_old_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            target = Path(temporary_dir) / "value.json"
            target.write_text('{"sentinel": true}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                PROFILER.atomic_write_json(target, {"bad": float("nan")})
            self.assertEqual(_read_json(target), {"sentinel": True})
            self.assertFalse(
                any(path.suffix == ".tmp" for path in target.parent.iterdir())
            )


class SourceAndCliContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODULE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(
            cls.source, str(MODULE_PATH), feature_version=7
        )

    def test_module_import_is_torch_free(self):
        top_level_imports = []
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_imports.append(node.module)
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("diff_gaussian_rasterization", top_level_imports)
        self.assertNotIn("gaussian_renderer", top_level_imports)
        self.assertNotIn("scene", top_level_imports)

    def test_cli_has_fixed_workload_and_four_required_outputs(self):
        parser, _, _, _ = PROFILER._build_parser()
        required = {
            action.dest: action.required for action in parser._actions
        }
        for destination in (
            "device_output",
            "raster_output",
            "leaf_output",
            "report",
        ):
            self.assertTrue(required[destination])
        defaults = parser.parse_args(
            [
                "--device-output",
                "device.json",
                "--raster-output",
                "raster.json",
                "--leaf-output",
                "leaf.json",
                "--report",
                "report.json",
            ]
        )
        self.assertEqual(defaults.iteration, 14000)
        self.assertEqual(defaults.scene_name, "flame_steak")
        self.assertGreaterEqual(defaults.views, 2)
        self.assertEqual(defaults.persistent_blocks, 0)

    def test_source_profiles_real_split_head_and_public_mixed_calls(self):
        for required in (
            "prepare_render_context(",
            "deform_for_render(",
            "prepare_pos_head_task(",
            ".forward_with_head(",
            "head_linear_solo(",
            "torch.cuda.Event(enable_timing=True)",
            "torch.cuda.current_stream()",
            "torch.no_grad()",
        ):
            self.assertIn(required, self.source)
        self.assertIn('gpu_name != EXPECTED_GPU_NAME', self.source)
        self.assertIn('capability != EXPECTED_COMPUTE_CAPABILITY', self.source)
        self.assertIn('"mixed_threads": 384', self.source)
        self.assertIn('"raster_named_barrier_id": 1', self.source)
        self.assertNotIn("ctypes.CDLL", self.source)
        self.assertIn("head_backend.tacker_capabilities()", self.source)

    def test_known_remote_dataset_basename_is_accepted(self):
        self.assertIn(
            "flame_steak_4dgs_min", PROFILER.EXPECTED_SOURCE_BASENAMES
        )

    def test_no_shell_or_subprocess_and_python37_grammar(self):
        self.assertNotIn("subprocess", self.source)
        self.assertNotIn("os.system", self.source)
        self.assertNotIn("shell=True", self.source)
        # setUpClass already parses explicitly with Python 3.7 grammar.
        self.assertIsInstance(self.tree, ast.Module)

    def test_no_per_iteration_device_synchronize_in_timer(self):
        timer = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_time_cuda_calls"
        )
        timer_source = ast.get_source_segment(self.source, timer)
        self.assertNotIn("cuda.synchronize", timer_source)
        self.assertIn("ends[-1].synchronize()", timer_source)


if __name__ == "__main__":
    unittest.main()
