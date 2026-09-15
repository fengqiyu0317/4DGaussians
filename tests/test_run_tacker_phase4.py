"""CPU-only contract tests for the fail-closed Phase-4 coordinator."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib.util
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "run_tacker_phase4.py"
SPEC = importlib.util.spec_from_file_location("run_tacker_phase4", MODULE_PATH)
PHASE4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHASE4)
BOOTSTRAP_PATH = PROJECT_ROOT / "scripts" / "run_profile_render_sealed.py"
BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "run_profile_render_sealed_contract", BOOTSTRAP_PATH
)
BOOTSTRAP = importlib.util.module_from_spec(BOOTSTRAP_SPEC)
BOOTSTRAP_SPEC.loader.exec_module(BOOTSTRAP)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
    return path


def _inline_workload(name, key, iteration, width, height, mix, resolution):
    return json.dumps(
        {
            "schema_version": 1,
            "kind": PHASE4.WORKLOAD_KIND,
            "name": name,
            "model_path": "/abs/model/{}".format(name),
            "source_path": "/abs/source/{}".format(name),
            "config": "/abs/config/{}.py".format(name),
            "iteration": iteration,
            "split": "test",
            "image_width": width,
            "image_height": height,
            "gaussian_count": 90000 + iteration,
            "raster_deformation_mix": mix,
            "workload_key": key,
            "profile_args": ["--resolution", str(resolution)],
        },
        sort_keys=True,
    )


def _cli_argv(output_dir="/tmp/phase4-output", plan_output=None):
    values = [
        "--phase31-run-root", "/abs/sealed-phase31",
        "--output-dir", str(output_dir),
        "--tacker-root", "/abs/tacker-runtime",
        "--template-profile", "/abs/disabled-profile.json",
        "--current-tacker-profile", "/abs/current-profile.json",
        "--model-path", "/abs/model/primary",
        "--source-path", "/abs/source/primary",
        "--configs", "/abs/config/primary.py",
        "--generalization-workload",
        _inline_workload(
            "native-checkpoint", "native-key", 3000, 1352, 1014,
            "raster_heavy", 1,
        ),
        "--generalization-workload",
        _inline_workload(
            "scaled-checkpoint", "scaled-key", 14000, 338, 254,
            "deformation_heavy", 4,
        ),
        "--dry-run",
    ]
    if plan_output is not None:
        values.extend(["--plan-output", str(plan_output)])
    return values


def _quality_report(frames=2):
    per_view = [
        {"batch_index": index, "view_index": index}
        for index in range(frames)
    ]
    return {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_quality_validation",
        "passed": True,
        "workload": {
            "scene": "primary",
            "iteration": 14000,
            "split": "test",
            "frames": frames,
            "view_indices": list(range(frames)),
            "resolution": [1352, 1014],
            "gaussian_count": 111525,
            "model_path": "/abs/model/primary",
            "source_path": "/abs/source/primary",
        },
        "device": {
            "name": "NVIDIA RTX A6000",
            "index": 0,
            "compute_capability": [8, 6],
            "cuda_arch": "sm_86",
        },
        "thresholds": {
            "psnr_drop_db_max": 0.05,
            "ssim_drop_max": 1e-4,
            "lpips_increase_max": 1e-4,
        },
        "qualification": {
            "enabled": False,
            "admission_claimed": True,
            "profile_override": None,
        },
        "tacker_profile": None,
        "modes": {
            "serial": {
                "actual_execution_mode": "serial",
                "fallback_reason": None,
                "per_view": copy.deepcopy(per_view),
            },
            "tacker": {
                "actual_execution_mode": "tacker",
                "fallback_reason": None,
                "per_view": copy.deepcopy(per_view),
            },
        },
        "deltas": {
            "tacker": {
                "psnr_drop_db": 0.01,
                "ssim_drop": 0.00001,
                "lpips_increase": 0.00001,
            }
        },
        "gates": [{"mode": "tacker", "passed": True}],
    }


class DryRunAndCliTests(unittest.TestCase):
    def test_bootstrap_binding_path_names_the_compiled_application_source(self):
        record, _raw = BOOTSTRAP._read_stable_source(
            PROJECT_ROOT / "profile_render.py"
        )
        self.assertEqual(record["path"], PHASE4._script("profile_render.py"))
        validator_source = inspect.getsource(
            PHASE4._verify_profile_render_provenance
        )
        self.assertIn(
            'expected_bootstrap = str(_script("profile_render.py"))',
            validator_source,
        )
        self.assertIn('"source.namespace.utils"', validator_source)

    def test_dry_run_is_deterministic_and_invokes_zero_subprocesses(self):
        stdout_one = io.StringIO()
        stdout_two = io.StringIO()
        with mock.patch.object(
            PHASE4.subprocess, "run", side_effect=AssertionError("subprocess.run called")
        ), mock.patch.object(
            PHASE4.subprocess,
            "check_output",
            side_effect=AssertionError("subprocess.check_output called"),
        ):
            with redirect_stdout(stdout_one):
                self.assertEqual(PHASE4.main(_cli_argv()), 0)
            with redirect_stdout(stdout_two):
                self.assertEqual(PHASE4.main(_cli_argv()), 0)

        first = json.loads(stdout_one.getvalue())
        second = json.loads(stdout_two.getvalue())
        self.assertEqual(first, second)
        self.assertTrue(first["passed"])
        self.assertTrue(first["dry_run"])
        self.assertEqual(first["subprocesses_invoked"], 0)
        self.assertEqual(
            [item["name"] for item in first["stages"]], list(PHASE4.STAGES)
        )
        expected = copy.deepcopy(first)
        observed_hash = expected.pop("plan_sha256")
        self.assertEqual(
            observed_hash,
            PHASE4.sha256_json(expected, "tacker-phase4-dry-run-v1"),
        )

    def test_dry_run_plan_output_is_no_clobber(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan_path = Path(temporary) / "plan.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(PHASE4.main(_cli_argv(plan_output=plan_path)), 0)
            before = plan_path.read_bytes()
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                self.assertEqual(PHASE4.main(_cli_argv(plan_output=plan_path)), 2)
            self.assertEqual(plan_path.read_bytes(), before)
            self.assertIn("refusing to overwrite", stderr.getvalue())

    def test_cli_rejects_noncanonical_pin_wrong_gpu_and_short_long_run(self):
        cases = (
            (["--phase31-matrix-sha256", "0" * 64], "canonical pinned"),
            (["--gpu", "1"], "logical device 0"),
            (["--long-frames", "50"], "greater than 50"),
            (["--resume"], "mutually exclusive"),
        )
        for suffix, message in cases:
            with self.subTest(suffix=suffix):
                args = PHASE4._parser().parse_args(_cli_argv() + list(suffix))
                with self.assertRaisesRegex(PHASE4.Phase4Error, message):
                    PHASE4._validate_cli_args(args, require_files=False)

    def test_cli_requires_exactly_two_generalization_specs(self):
        values = _cli_argv()
        marker_positions = [
            index
            for index, value in enumerate(values)
            if value == "--generalization-workload"
        ]
        del values[marker_positions[-1]:marker_positions[-1] + 2]
        args = PHASE4._parser().parse_args(values)
        with self.assertRaisesRegex(PHASE4.Phase4Error, "exactly two"):
            PHASE4._validate_cli_args(args, require_files=False)

    def test_safe_command_rejects_phase31_generation_entry_points(self):
        with self.assertRaisesRegex(PHASE4.Phase4Error, "forbids candidate"):
            PHASE4._validate_safe_command(
                [sys.executable, "/repo/scripts/run_tacker_phase31.py"]
            )


class JournalTests(unittest.TestCase):
    def test_new_run_never_appends_to_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase4"
            identity = {"sha256": "a" * 64, "payload": {"run": 1}}
            journal = PHASE4.Journal(output, identity, False)
            journal.close()
            with self.assertRaisesRegex(PHASE4.Phase4Error, "refusing to append"):
                PHASE4.Journal(output, identity, False)

    def test_resume_reuses_hash_bound_stage_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase4"
            identity = {"sha256": "b" * 64, "payload": {"run": 2}}
            journal = PHASE4.Journal(output, identity, False)

            def action(directory):
                artifact = Path(directory) / "artifact.json"
                _write_json(artifact, {"passed": True})
                return {"value": 7}, [artifact]

            self.assertEqual(
                journal.run("preflight", {"device": "A6000"}, action),
                {"value": 7},
            )
            artifact_path = Path(journal.state["stages"][0]["artifacts"][0]["path"])
            journal.close()

            resumed = PHASE4.Journal(output, identity, True)
            forbidden_action = mock.Mock(side_effect=AssertionError("stage reran"))
            self.assertEqual(
                resumed.run(
                    "preflight", {"device": "A6000"}, forbidden_action
                ),
                {"value": 7},
            )
            forbidden_action.assert_not_called()
            resumed.close()

            artifact_path.write_text('{"passed": false}\n', encoding="utf-8")
            tampered = PHASE4.Journal(output, identity, True)
            try:
                with self.assertRaisesRegex(PHASE4.Phase4Error, "artifact changed"):
                    tampered.run(
                        "preflight", {"device": "A6000"}, forbidden_action
                    )
            finally:
                tampered.close()

    def test_resume_rejects_stage_input_hash_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase4"
            identity = {"sha256": "c" * 64, "payload": {"run": 3}}
            journal = PHASE4.Journal(output, identity, False)

            def action(directory):
                artifact = Path(directory) / "artifact"
                artifact.write_text("stable", encoding="utf-8")
                return {"ok": True}, [artifact]

            journal.run("preflight", {"input": 1}, action)
            journal.close()

            resumed = PHASE4.Journal(output, identity, True)
            try:
                with self.assertRaisesRegex(PHASE4.Phase4Error, "inputs changed"):
                    resumed.run("preflight", {"input": 2}, action)
            finally:
                resumed.close()

    def test_resume_rejects_checkpoint_result_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase4"
            identity = {"sha256": "d" * 64, "payload": {"run": 4}}
            journal = PHASE4.Journal(output, identity, False)

            def action(directory):
                artifact = Path(directory) / "artifact.json"
                _write_json(artifact, {"passed": True})
                return {"execution_mode": "tacker"}, [artifact]

            journal.run("preflight", {"device": "A6000"}, action)
            state_path = journal.path
            journal.close()
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["stages"][0]["result"]["execution_mode"] = "serial"
            _write_json(state_path, state)
            resumed = PHASE4.Journal(output, identity, True)
            try:
                with self.assertRaisesRegex(
                    PHASE4.Phase4Error, "checkpoint result changed"
                ):
                    resumed.run("preflight", {"device": "A6000"}, action)
            finally:
                resumed.close()

class GateValidationTests(unittest.TestCase):
    @staticmethod
    def _quality_workload():
        return {
            "name": "primary",
            "iteration": 14000,
            "image_width": 1352,
            "image_height": 1014,
            "gaussian_count": 111525,
            "model_path": "/abs/model/primary",
            "source_path": "/abs/source/primary",
        }

    def test_quality_accepts_aligned_legacy_reference_and_tacker(self):
        report = _quality_report(frames=2)
        self.assertIs(
            PHASE4.validate_quality_report(
                report, self._quality_workload(), ("serial", "tacker"),
                expect_tacker=True, frames=2,
            ),
            report,
        )

    def test_quality_rejects_fallback_threshold_and_view_order(self):
        mutations = []
        fallback = _quality_report(frames=2)
        fallback["modes"]["tacker"]["actual_execution_mode"] = "serial"
        fallback["modes"]["tacker"]["fallback_reason"] = "stale profile"
        mutations.append((fallback, "fell back"))
        threshold = _quality_report(frames=2)
        threshold["deltas"]["tacker"]["psnr_drop_db"] = 0.051
        mutations.append((threshold, "exceeds"))
        reordered = _quality_report(frames=2)
        reordered["modes"]["serial"]["per_view"][1]["batch_index"] = 0
        mutations.append((reordered, "ordering"))
        for document, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(PHASE4.Phase4Error, message):
                    PHASE4.validate_quality_report(
                        document,
                        self._quality_workload(),
                        ("serial", "tacker"),
                        expect_tacker=True,
                        frames=2,
                    )

    def test_leaf_bundle_binds_candidate_matrix_resources_and_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {
                "schema_version": 2,
                "kind": "4dgaussians_tacker_leaf_profile_report",
                "passed": True,
                "numerics": {"passed": True},
                "variant_id": "candidate-c3",
                "profile_binding": {
                    "candidate_sha256": "a" * 64,
                    "candidate_matrix_sha256": PHASE4.EXPECTED_PHASE31_MATRIX,
                    "profile_file_sha256": "b" * 64,
                },
                "parameters": {
                    "qualification_profile": True,
                    "used_as_deployment": False,
                    "candidate_profile_deployment_enabled": False,
                    "candidate_profile": "/sealed/c3.json",
                },
                "execution_source_snapshot": {
                    "verified_unchanged_after_measurement": True
                },
                "resources": {
                    "registers_per_thread": 64,
                    "static_shared_bytes": 0,
                    "active_blocks_per_multiprocessor": 2,
                },
                "measurements": {
                    "solo_raster_p50_ms": 4.0,
                    "mixed_raster_p50_ms": 10.0,
                    "solo_head_p50_ms": 1.0,
                    "multi_head_solo_p50_ms": 3.0,
                    "mixed_p50_ms": 8.0,
                },
            }
            paths = {
                "report": _write_json(root / "report.json", report),
                "device": _write_json(root / "device.json", {"passed": True}),
                "raster": _write_json(root / "raster.json", {"passed": True}),
                "leaf": _write_json(root / "leaf.json", {"passed": True}),
            }
            bundle = PHASE4.validate_leaf_bundle(
                paths,
                candidate_name="candidate-c3",
                candidate={
                    "name": "candidate-c3",
                    "candidate_sha256": "a" * 64,
                    "profile": {
                        "path": "/sealed/c3.json",
                        "sha256": "b" * 64,
                    },
                },
            )
            self.assertEqual(bundle["resources"]["registers_per_thread"], 64)
            self.assertAlmostEqual(
                bundle["diagnostics"]["raster_slowdown_fraction"], 1.5
            )
            self.assertFalse(bundle["diagnostics"]["qos_gate"])

            legacy = copy.deepcopy(report)
            legacy["schema_version"] = 1
            del legacy["measurements"]["multi_head_solo_p50_ms"]
            _write_json(paths["report"], legacy)
            current = PHASE4.validate_leaf_bundle(paths)
            self.assertEqual(
                current["measurements"]["multi_head_solo_p50_ms"], 1.0
            )
            self.assertEqual(
                current["diagnostics"]["multi_head_solo_source"],
                "legacy_solo_head_p50_ms_alias",
            )
            finalist_without_multi = copy.deepcopy(report)
            del finalist_without_multi["measurements"][
                "multi_head_solo_p50_ms"
            ]
            _write_json(paths["report"], finalist_without_multi)
            with self.assertRaisesRegex(
                PHASE4.Phase4Error, "multi_head_solo_p50_ms"
            ):
                PHASE4.validate_leaf_bundle(
                    paths, candidate_name="candidate-c3"
                )

            legacy_finalist = copy.deepcopy(report)
            legacy_finalist["variant_id"] = "candidate-c0"
            legacy_finalist["profile_binding"]["candidate_sha256"] = "c" * 64
            legacy_finalist["profile_binding"]["profile_file_sha256"] = "d" * 64
            legacy_finalist["parameters"]["candidate_profile"] = "/sealed/c0.json"
            del legacy_finalist["measurements"]["multi_head_solo_p50_ms"]
            _write_json(paths["report"], legacy_finalist)
            c0 = PHASE4.validate_leaf_bundle(
                paths,
                candidate_name="candidate-c0",
                candidate={
                    "name": "candidate-c0",
                    "candidate_sha256": "c" * 64,
                    "abi_family": "legacy_pos_l1_v1",
                    "profile": {
                        "path": "/sealed/c0.json",
                        "sha256": "d" * 64,
                    },
                },
            )
            self.assertEqual(c0["measurements"]["multi_head_solo_p50_ms"], 1.0)
            self.assertEqual(
                c0["diagnostics"]["multi_head_solo_source"],
                "legacy_abi1_solo_head_p50_ms_alias",
            )

            bad = copy.deepcopy(report)
            bad["profile_binding"]["candidate_matrix_sha256"] = "0" * 64
            _write_json(paths["report"], bad)
            with self.assertRaisesRegex(PHASE4.Phase4Error, "sealed Phase-3.1"):
                PHASE4.validate_leaf_bundle(
                    paths, candidate_name="candidate-c3"
                )

    def test_leaf_failure_schema_matches_legacy_or_finalist_producer(self):
        base = {
            "kind": "4dgaussians_tacker_leaf_profile_report",
            "passed": False,
            "errors": ["numerics failed"],
        }
        legacy = dict(base, schema_version=1)
        finalist = dict(base, schema_version=2)
        self.assertIs(
            PHASE4._validate_leaf_failure_report(
                legacy, "current_tacker", None
            ),
            legacy,
        )
        self.assertIs(
            PHASE4._validate_leaf_failure_report(
                finalist, "candidate-c3", {"name": "candidate-c3"}
            ),
            finalist,
        )
        with self.assertRaisesRegex(PHASE4.Phase4Error, "structured reason"):
            PHASE4._validate_leaf_failure_report(
                legacy, "candidate-c3", {"name": "candidate-c3"}
            )

    def _formal_fixture(self, root):
        finalists = [{"name": "candidate-c3"}]
        sealed = {"finalists": finalists}
        workload = {"name": "primary", "iteration": 14000}
        candidates = [
            {
                "name": "serial", "execution_mode": "serial",
                "profile_path": None, "qualification_mode": False,
            },
            {
                "name": "two_stream", "execution_mode": "two_stream",
                "profile_path": None, "qualification_mode": False,
            },
            {
                "name": "current_tacker", "execution_mode": "tacker",
                "profile_path": "/profiles/current.json",
                "qualification_mode": False,
            },
            {
                "name": "candidate-c3", "execution_mode": "tacker",
                "profile_path": "/profiles/c3.json",
                "qualification_mode": True,
            },
        ]
        metadata = _write_json(Path(root) / "child.json", {"passed": True})
        digest = PHASE4.sha256_file(metadata)
        runs = []
        for round_index in range(PHASE4.FORMAL_TRIALS):
            for candidate in candidates:
                runs.append(
                    {
                        "candidate_name": candidate["name"],
                        "error": None,
                        "round_index": round_index,
                        "metadata_path": str(metadata),
                        "metadata_sha256": digest,
                    }
                )
        document = {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_fps_benchmark",
            "passed": True,
            "candidates": candidates,
            "contract": {
                "profile_frames": PHASE4.FORMAL_FRAMES,
                "warmup_frames": PHASE4.FORMAL_WARMUP,
                "view_indices": list(range(PHASE4.FORMAL_FRAMES)),
                "workload_name": workload["name"],
                "iteration": workload["iteration"],
            },
            "resume_identity": {
                "payload": {
                    "schedule_strategy": PHASE4.FORMAL_SCHEDULE,
                    "schedule_seed": PHASE4.FORMAL_SEED,
                    "trials": PHASE4.FORMAL_TRIALS,
                }
            },
            "completed_execution_count": len(runs),
            "runs": runs,
            "summaries": {
                candidate["name"]: {"trial_count": PHASE4.FORMAL_TRIALS}
                for candidate in candidates
            },
        }
        return document, sealed, workload

    def test_formal_requires_exact_sealed_order_and_ten_by_fifty_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            document, sealed, workload = self._formal_fixture(temporary)
            with mock.patch.object(
                PHASE4,
                "validate_render_metadata",
                return_value={"rasterizer_binary_sha256": "e" * 64},
            ) as validate:
                result = PHASE4.validate_formal_benchmark(
                    document, sealed, workload
                )
            self.assertEqual(validate.call_count, 40)
            self.assertEqual(
                result["candidate_names"],
                ["serial", "two_stream", "current_tacker", "candidate-c3"],
            )
            self.assertEqual(result["loaded_rasterizer_sha256"], "e" * 64)

            reordered = copy.deepcopy(document)
            reordered["candidates"][0], reordered["candidates"][1] = (
                reordered["candidates"][1], reordered["candidates"][0]
            )
            with self.assertRaisesRegex(PHASE4.Phase4Error, "set/order"):
                PHASE4.validate_formal_benchmark(reordered, sealed, workload)

            incomplete = copy.deepcopy(document)
            incomplete["completed_execution_count"] -= 1
            with self.assertRaisesRegex(PHASE4.Phase4Error, "every 10x50"):
                PHASE4.validate_formal_benchmark(incomplete, sealed, workload)

    def test_admission_accepts_baseline_without_manufacturing_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist.json"
            document = {
                "schema_version": 2,
                "kind": "4dgaussians_tacker_admission_report",
                "passed": True,
                "selected_variant_id": "serial",
                "candidates": [
                    {
                        "variant_id": "serial",
                        "benchmark_candidate_name": "serial",
                        "execution_mode": "serial",
                    }
                ],
                "selection": {"deployment_winner_variant_id": "serial"},
                "deployment": {"enabled": False, "valid": False},
                "profile_sha256": None,
                "provenance": {"validated_abi": {}},
            }
            result = PHASE4._validate_admission(
                document, output, {"deployment_winner": "serial"}
            )
            self.assertEqual(result["execution_mode"], "serial")
            self.assertIsNone(result["enabled_profile"])

            _write_json(output, {"unexpected": True})
            with self.assertRaisesRegex(PHASE4.Phase4Error, "must not manufacture"):
                PHASE4._validate_admission(
                    document, output, {"deployment_winner": "serial"}
                )

    def test_admission_tacker_requires_enabled_profile_and_full_abi_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "enabled.json"
            profile = {
                "schema_version": 2,
                "workload_key": "primary-key",
                "selection_objective": "whole_run_median_throughput_fps",
                "selected_variant_id": "candidate-c3",
                "manifest": {"variant_id": "candidate-c3"},
                "manifest_sha256": "1" * 64,
                "correctness_thresholds": {},
                "candidates": [],
                "selection": {},
                "deployment": {"enabled": True, "valid": True},
                "provenance": {},
            }
            profile["profile_sha256"] = PHASE4._schema2_profile_sha256(profile)
            _write_json(output, profile)
            document = {
                "schema_version": 2,
                "kind": "4dgaussians_tacker_admission_report",
                "passed": True,
                "selected_variant_id": "candidate-c3",
                "candidates": [
                    {
                        "variant_id": "candidate-c3",
                        "benchmark_candidate_name": "candidate-c3",
                        "execution_mode": "tacker",
                    }
                ],
                "selection": {
                    "deployment_winner_variant_id": "candidate-c3"
                },
                "deployment": {"enabled": True, "valid": True},
                "profile_sha256": profile["profile_sha256"],
                "provenance": {
                    "validated_abi": {
                        "mixed_abi_version": 3,
                        "mixed_abi_manifest_sha256": "2" * 64,
                        "head_abi_version": 2,
                        "head_abi_manifest_sha256": "3" * 64,
                    }
                },
            }
            result = PHASE4._validate_admission(
                document, output, {"deployment_winner": "candidate-c3"}
            )
            self.assertEqual(result["execution_mode"], "tacker")
            self.assertEqual(
                result["enabled_profile"]["sha256"], PHASE4.sha256_file(output)
            )

            missing_abi = copy.deepcopy(document)
            del missing_abi["provenance"]["validated_abi"]["head_abi_version"]
            with self.assertRaisesRegex(PHASE4.Phase4Error, "validated ABI"):
                PHASE4._validate_admission(
                    missing_abi, output, {"deployment_winner": "candidate-c3"}
                )


class CommandAndBaselineWinnerTests(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            python_executable="/opt/python",
            gpu=0,
            leaf_views=2,
            leaf_warmup=5,
            leaf_repetitions=50,
            timeout_seconds=30,
            template_profile="/profiles/template.json",
            tacker_root="/runtime",
            cmake="cmake",
            ctest="ctest",
            nvcc="/usr/local/cuda-12.4/bin/nvcc",
        )

    def _workload(self):
        return {
            "name": "primary",
            "model_path": "/model",
            "source_path": "/source",
            "config": "/config.py",
            "iteration": 14000,
            "profile_args": ["--resolution", "4"],
        }

    def test_in_place_extensions_are_sealed_inputs_not_journal_publications(self):
        build_source = inspect.getsource(PHASE4._build_and_cuda_stage)
        self.assertNotIn('artifacts.append(artifact["path"])', build_source)
        self.assertIn('"extension_binaries": extension_binaries', build_source)

        run_source = inspect.getsource(PHASE4.run_phase4)
        build_assignment = run_source.index('build = execute(')
        immediate_verification = run_source.index(
            "_verify_build_artifacts_unchanged(build)", build_assignment
        )
        seal_assignment = run_source.index('sealed = execute(', build_assignment)
        self.assertLess(immediate_verification, seal_assignment)

    def test_phase31_replay_uses_the_identity_sealed_source_root(self):
        project = Path("/sealed/phase31-project")
        report = {
            "identity": {
                "payload": {
                    "scripts": {
                        "autotune": {
                            "path": str(project / "scripts/tacker_autotune.py")
                        },
                        "benchmark": {
                            "path": str(project / "scripts/benchmark_tacker_fps.py")
                        },
                        "top3": {
                            "path": str(project / "scripts/profile_tacker_top3.py")
                        },
                    }
                }
            }
        }
        self.assertEqual(PHASE4._phase31_replay_project_root(report), project)

        mismatched = copy.deepcopy(report)
        mismatched["identity"]["payload"]["scripts"]["top3"]["path"] = (
            "/different/project/scripts/profile_tacker_top3.py"
        )
        with self.assertRaisesRegex(PHASE4.Phase4Error, "one sealed project root"):
            PHASE4._phase31_replay_project_root(mismatched)

        stage_source = inspect.getsource(PHASE4._verify_phase31_stage)
        self.assertIn('replay_env["TACKER_PHASE31_PROJECT_ROOT"]', stage_source)
        self.assertIn('"replay_project_root": str(replay_project_root)', stage_source)

    def test_selection_metadata_consumes_finite_resource_measurements(self):
        sealed = {"finalists": [{"name": "candidate-c3"}]}
        resources = {
            "records": {
                "current_tacker": {
                    "valid": True,
                    "resources": {
                        "registers_per_thread": 48,
                        "static_shared_bytes": 0,
                    },
                },
                "candidate-c3": {
                    "valid": True,
                    "resources": {
                        "registers_per_thread": 64,
                        "static_shared_bytes": 1024,
                    },
                },
            }
        }
        metadata = PHASE4._selection_metadata_from_resources(sealed, resources)
        self.assertEqual(metadata["current_tacker"]["registers_per_thread"], 48.0)
        self.assertEqual(metadata["candidate-c3"]["shared_memory_bytes"], 1024.0)
        self.assertEqual(metadata["serial"], {"abi_complexity": 0.0})

    def test_render_and_quality_commands_preserve_owned_protocol_flags(self):
        args = self._args()
        workload = self._workload()
        serial = PHASE4._render_command(
            args, workload, "/out.json", "serial", 50, 10, 10
        )
        self.assertEqual(serial.count("--trials"), 1)
        self.assertNotIn("--tacker-profile", serial)
        self.assertEqual(serial[serial.index("--frames") + 1], "50")
        self.assertEqual(serial[-2:], ["--resolution", "4"])

        tacker = PHASE4._render_command(
            args, workload, "/out.json", "tacker", 50, 10, 10,
            profile="/profiles/enabled.json",
        )
        self.assertEqual(
            tacker[tacker.index("--tacker-profile") + 1],
            "/profiles/enabled.json",
        )
        qualification = PHASE4._render_command(
            args, workload, "/out.json", "tacker", 2, 1, 0,
            profile="/profiles/disabled.json", qualification=True,
        )
        self.assertIn("--qualification-mode", qualification)
        self.assertNotIn("--tacker-profile", qualification)
        self.assertEqual(
            qualification[qualification.index("--qualification-profile") + 1],
            "/profiles/disabled.json",
        )
        with self.assertRaisesRegex(PHASE4.Phase4Error, "requires Tacker"):
            PHASE4._render_command(
                args, workload, "/out.json", "serial", 2, 1, 0,
                qualification=True,
            )
        with self.assertRaisesRegex(PHASE4.Phase4Error, "explicit profile"):
            PHASE4._render_command(
                args, workload, "/out.json", "tacker", 50, 10, 10
            )

        quality = PHASE4._quality_command(
            args,
            workload,
            "/quality.json",
            ("serial", "tacker"),
            qualification_profile="/profiles/sealed.json",
        )
        modes = quality[quality.index("--modes") + 1:quality.index("--gpu")]
        self.assertEqual(modes, ["serial", "tacker"])
        self.assertIn("--qualification-mode", quality)
        self.assertNotIn("--tacker-profile", quality)

    def test_build_command_contract_includes_cpu_cuda_and_sm86_suites(self):
        commands, runtime_build = PHASE4._build_commands(
            self._args(), Path("/tmp/phase4-attempt")
        )
        by_name = {name: argv for name, argv, _cwd, _env in commands}
        self.assertEqual(runtime_build, Path("/tmp/phase4-attempt/runtime-build"))
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", by_name["runtime-configure"])
        self.assertIn("tests.test_run_tacker_phase4", by_name["phase4-cpu-contracts"])
        self.assertEqual(
            by_name["head-cpu-contracts"][2:4], ["unittest", "discover"]
        )
        self.assertIn("tests.test_head_linear_v2_cuda", by_name["head-cuda-tests"])
        self.assertIn("tests.test_tacker_mixed_cuda", by_name["raster-cuda-tests"])
        for _name, argv, _cwd, _env in commands:
            PHASE4._validate_safe_command(argv)

    def test_leaf_candidate_command_is_bound_to_sealed_profile_and_matrix(self):
        outputs = {
            key: "/out/{}.json".format(key)
            for key in ("device", "raster", "leaf", "report")
        }
        argv = PHASE4._leaf_command(
            self._args(),
            self._workload(),
            outputs,
            candidate={"profile": {"path": "/sealed/c3.json"}},
            matrix_path="/sealed/matrix.json",
        )
        self.assertEqual(
            argv[argv.index("--candidate-profile") + 1], "/sealed/c3.json"
        )
        self.assertEqual(
            argv[argv.index("--candidate-matrix") + 1], "/sealed/matrix.json"
        )
        self.assertNotIn("--persistent-blocks", argv)

    def test_selection_stage_allows_baseline_winner_and_emits_no_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            formal_path = _write_json(
                directory / "formal.json", {"deployment_winner": "serial"}
            )
            admission_document = {
                "schema_version": 2,
                "kind": "4dgaussians_tacker_admission_report",
                "passed": True,
                "selected_variant_id": "serial",
                "candidates": [
                    {
                        "variant_id": "serial",
                        "benchmark_candidate_name": "serial",
                        "execution_mode": "serial",
                    }
                ],
                "selection": {"deployment_winner_variant_id": "serial"},
                "deployment": {"enabled": False, "valid": False},
                "profile_sha256": None,
                "provenance": {"validated_abi": {}},
            }
            captured = {}

            def fake_run(argv, log, **_kwargs):
                captured["argv"] = list(argv)
                report = Path(argv[argv.index("--report") + 1])
                _write_json(report, admission_document)
                return {"argv": list(argv), "returncode": 0}

            resources = {
                "records": {
                    "current_tacker": {
                        "valid": True,
                        "outputs": {
                            key: {"path": "/measurements/{}.json".format(key)}
                            for key in ("device", "raster", "leaf")
                        }
                    }
                }
            }
            quality = {
                "baseline_quality": {"path": "/quality/baseline.json"},
                "correctness": {"path": "/quality/correctness.json"},
            }
            formal = {"report": {"path": str(formal_path)}}
            sealed = {
                "current_tacker_profile": {"path": "/profiles/current.json"},
                "finalists": [
                    {
                        "name": "candidate-c3",
                        "profile": {"path": "/sealed/c3.json"},
                    }
                ],
            }
            with mock.patch.object(PHASE4, "run_command", side_effect=fake_run):
                result, artifacts = PHASE4._selection_admission_stage(
                    self._args(), sealed, resources, quality, formal, directory
                )

            deployment = result["deployment"]
            self.assertEqual(deployment["execution_mode"], "serial")
            self.assertEqual(
                deployment["profile_action"],
                "publish_baseline_selection_without_profile",
            )
            self.assertIsNone(deployment["enabled_profile"])
            self.assertFalse((directory / "enabled-profile.json").exists())
            self.assertEqual(captured["argv"].count("--mixed-abi-json"), 4)
            self.assertEqual(captured["argv"].count("--candidate-profile"), 2)
            self.assertNotIn(directory / "enabled-profile.json", artifacts)


if __name__ == "__main__":
    unittest.main()
