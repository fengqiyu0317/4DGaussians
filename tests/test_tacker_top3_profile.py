"""CPU contracts for ranking-bound top-3 Nsight orchestration."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "profile_tacker_top3.py"
SPEC = importlib.util.spec_from_file_location("profile_tacker_top3", MODULE_PATH)
TOP3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOP3)


def _summary(mode, metadata, frames=50, fps=100.0, launches=500):
    return {
        "frame_count": frames,
        "metadata": metadata,
        "render_loop": {"fps": fps, "ms_per_frame": 1000.0 / fps},
        "frame_nvtx": {
            "median_ms": 10.0,
            "max_ms": 12.0,
            "stddev_ms": 0.5,
        },
        "main_stages": {
            "renderer/deformation": {"gpu_projected_ms_per_frame": 4.0},
            "renderer/rasterization": {"gpu_projected_ms_per_frame": 5.0},
        },
        "kernels": {
            "ms_per_frame": 8.0,
            "launches": launches,
            "launches_per_frame": launches / float(frames),
            "categories": {
                "raster_render": {"ms_per_frame": 3.0},
                "gemm": {"ms_per_frame": 2.0},
            },
        },
        "cuda_api": {
            "stream_synchronize": {
                "calls": 1,
                "calls_per_frame": 0.02,
                "total_ms": 0.1,
            },
            "kernel_launch": {
                "calls": launches,
                "calls_per_frame": launches / float(frames),
                "total_ms": 1.0,
            },
        },
    }


class Top3Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profile_script = self.root / "profile_nsight.sh"
        self.profile_script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.config = self.root / "config.py"
        self.config.write_text("x = 1\n", encoding="utf-8")
        self.model = self.root / "model"
        self.source = self.root / "source"
        self.model.mkdir()
        self.source.mkdir()
        self.current_profile = self.root / "current.json"
        self.future_profile = self.root / "future.json"
        self.current_profile.write_text(
            '{"schema_version":1,"profile":"current"}\n', encoding="utf-8"
        )
        self.future_profile.write_text(
            '{"schema_version":2,"profile":"future"}\n', encoding="utf-8"
        )
        self.report = {
            "schema_version": 1,
            "kind": TOP3.FPS_REPORT_KIND,
            "passed": True,
            "eligible_ranking": [
                "future",
                "current_tacker",
                "two_stream",
                "serial",
            ],
            "candidates": [
                {
                    "name": "serial",
                    "execution_mode": "serial",
                    "profile_path": None,
                    "profile_file_sha256": None,
                    "qualification_mode": False,
                },
                {
                    "name": "two_stream",
                    "execution_mode": "two_stream",
                    "profile_path": None,
                    "profile_file_sha256": None,
                    "qualification_mode": False,
                },
                {
                    "name": "current_tacker",
                    "execution_mode": "tacker",
                    "profile_path": str(self.current_profile.resolve()),
                    "profile_file_sha256": TOP3.sha256_file(self.current_profile),
                    "qualification_mode": False,
                },
                {
                    "name": "future",
                    "execution_mode": "tacker",
                    "profile_path": str(self.future_profile.resolve()),
                    "profile_file_sha256": TOP3.sha256_file(self.future_profile),
                    "qualification_mode": True,
                },
            ],
            "summaries": {
                "future": {"median_throughput_fps": 110.0},
                "current_tacker": {"median_throughput_fps": 100.0},
                "two_stream": {"median_throughput_fps": 95.0},
                "serial": {"median_throughput_fps": 90.0},
            },
        }
        self.fps_path = self.root / "fps.json"
        source_files = {
            "profile_render.py": "1" * 64,
            "configs": TOP3.sha256_file(self.config),
            "gaussian_renderer/__init__.py": "2" * 64,
            "gaussian_renderer/tacker_pipeline.py": "3" * 64,
            "diff_gaussian_rasterization/__init__.py": "4" * 64,
            "diff_gaussian_rasterization._C": "5" * 64,
        }
        self.source_files = source_files
        self.stable_provenance = {
            "environment": {
                "gpu_name": "Synthetic NVIDIA RTX A6000",
                "cuda_runtime": "12.4",
                "pytorch_version": "2.4.1",
            },
            "repository_commit": "a" * 40,
            "repository_commit_source": "test",
            "repository_dirty": False,
            "submodules": [],
            "source_files": source_files,
        }
        self.report["contract"] = {
            "model_path": str(self.model.resolve()),
            "source_path": str(self.source.resolve()),
            "iteration": 14000,
            "workload_name": "flame_steak",
        }
        self.report["stable_provenance"] = self.stable_provenance
        self.report["runs"] = []
        for candidate in self.report["candidates"]:
            is_tacker = candidate["execution_mode"] == "tacker"
            is_legacy = candidate["name"] == "current_tacker"
            runtime_hash = "6" * 64 if is_tacker else None
            self.report["runs"].append(
                {
                    "candidate_name": candidate["name"],
                    "passed": True,
                    "metrics": {
                        "profile_manifest_sha256": runtime_hash,
                        "profile_selection_sha256": (
                            None if is_legacy else runtime_hash
                        ),
                        "selected_variant_id": (
                            candidate["name"] if is_tacker else None
                        ),
                        "selected_candidate_abi_sha256": (
                            None if is_legacy else runtime_hash
                        ),
                        "persistent_blocks": 7000 if is_tacker else None,
                    },
                }
            )
        self.fps_path.write_text(json.dumps(self.report), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _run_arguments(self, output_name="nsight"):
        output = self.root / output_name
        return {
            "fps_report_path": self.fps_path,
            "output_dir": output,
            "report_path": output / "report.json",
            "profile_script": self.profile_script,
            "model_path": self.model,
            "config_path": self.config,
            "source_path": self.source,
            "gpu": 1,
            "frames": 50,
            "iteration": 14000,
            "workload_name": "flame_steak",
        }

    def _metadata_for_command(self, command):
        mode = command[9]
        profile_path = command[10] or None
        qualification = command[11] == "1"
        candidate = next(
            item
            for item in TOP3.select_top_candidates(self.report)
            if item["execution_mode"] == mode
            and item["profile_path"] == profile_path
        )
        active_hash = candidate["profile_file_sha256"]
        is_tacker = mode == "tacker"
        tacker_hash = active_hash if is_tacker and not qualification else None
        qualification_hash = active_hash if qualification else None
        runtime_hash = "6" * 64 if is_tacker else None
        optional_runtime_hash = (
            runtime_hash
            if candidate.get("profile_schema_version") == 2
            else None
        )
        return {
            "schema_version": 1,
            "kind": TOP3.CHILD_KIND,
            "passed": True,
            "model_path": str(self.model.resolve()),
            "source_path": str(self.source.resolve()),
            "iteration": 14000,
            "split": "test",
            "warmup_frames": 10,
            "profile_frames": 50,
            "execution_mode": mode,
            "actual_execution_mode": mode,
            "qualification_mode_requested": qualification,
            "qualification_mode_executed": qualification,
            "workload_name": "flame_steak" if mode == "tacker" else None,
            "tacker_profile": (
                profile_path if mode == "tacker" and not qualification else None
            ),
            "qualification_profile": profile_path if qualification else None,
            "active_profile_sha256": active_hash,
            "tacker_profile_sha256": tacker_hash,
            "qualification_profile_sha256": qualification_hash,
            "profile_manifest_sha256": runtime_hash,
            "profile_selection_sha256": optional_runtime_hash,
            "selected_candidate_abi_sha256": optional_runtime_hash,
            "selected_variant_id": candidate["name"] if is_tacker else None,
            "persistent_blocks": 7000 if is_tacker else None,
            "two_stream_fallback_reason": None,
            "tacker_fallback_reason": None,
            "profile_hashes": {
                "active_profile_sha256": active_hash,
                "tacker_profile_sha256": tacker_hash,
                "qualification_profile_sha256": qualification_hash,
                "profile_manifest_sha256": runtime_hash,
                "profile_selection_sha256": optional_runtime_hash,
                "selected_candidate_abi_sha256": optional_runtime_hash,
            },
            "source_files": dict(self.source_files),
            "repository": {
                "commit": "a" * 40,
                "commit_source": "test",
                "dirty": False,
                "submodules": [],
                "source_files": dict(self.source_files),
            },
            "gpu_name": "Synthetic NVIDIA RTX A6000",
            "cuda_runtime": "12.4",
            "pytorch_version": "2.4.1",
        }

    def _write_outputs(self, command, fps=100.0):
        mode = command[9]
        output_dir = Path(command[4])
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._metadata_for_command(command)
        (output_dir / "{}_profile_metadata.json".format(mode)).write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (output_dir / "4dgs_render_{}.nsys-rep".format(mode)).write_bytes(
            b"synthetic nsys"
        )
        (output_dir / "4dgs_render_{}_stats.csv".format(mode)).write_text(
            "synthetic stats\n", encoding="utf-8"
        )
        (output_dir / "{}_summary.json".format(mode)).write_text(
            json.dumps(_summary(mode, metadata, fps=fps)), encoding="utf-8"
        )


class SelectionTests(Top3Fixture):
    def test_selects_formal_top_three_including_a_ranked_baseline(self):
        selected = TOP3.select_top_candidates(self.report)
        self.assertEqual(
            [item["name"] for item in selected],
            ["future", "current_tacker", "two_stream"],
        )
        self.assertTrue(selected[0]["qualification_mode"])
        self.assertEqual(selected[0]["profile_schema_version"], 2)
        self.assertEqual(selected[1]["profile_schema_version"], 1)
        self.assertIsNone(selected[2]["profile_path"])

    def test_changed_candidate_profile_fails_closed(self):
        self.future_profile.write_text('{"changed":true}\n', encoding="utf-8")
        with self.assertRaisesRegex(TOP3.Top3ContractError, "hash changed"):
            TOP3.select_top_candidates(self.report)

    def test_command_is_an_argv_list_and_preserves_metacharacters(self):
        selected = TOP3.select_top_candidates(self.report)[0]
        odd_source = self.root / "source;still-one-argument"
        command = TOP3.build_nsight_command(
            self.profile_script,
            selected,
            self.root / "out",
            self.model,
            self.config,
            odd_source,
            1,
            50,
            14000,
            "flame_steak",
        )
        self.assertIsInstance(command, list)
        self.assertEqual(
            command[:2], ["/bin/bash", str(self.profile_script.resolve())]
        )
        self.assertIn(str(odd_source.resolve()), command)
        self.assertEqual(command[-3:], [str(self.future_profile.resolve()), "1", "flame_steak"])

    def test_diagnostics_require_exact_frames_and_a_critical_path_stage(self):
        candidate = TOP3.select_top_candidates(self.report)[0]
        summary = _summary("tacker", {}, frames=49)
        with self.assertRaisesRegex(TOP3.Top3ContractError, "frame_count"):
            TOP3.extract_nsight_diagnostics(
                summary, candidate, expected={"frames": 50}
            )
        summary = _summary("tacker", {})
        summary["main_stages"] = {}
        with self.assertRaisesRegex(TOP3.Top3ContractError, "critical-path"):
            TOP3.extract_nsight_diagnostics(summary, candidate)


class ExecutionTests(Top3Fixture):
    def test_profiles_top_three_and_emits_comparison_diagnostics(self):
        calls = []

        def runner(command, **kwargs):
            self.assertFalse(kwargs["shell"])
            calls.append(list(command))
            self._write_outputs(command, fps=100.0 - len(calls))
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        result = TOP3.run_top3_profile(runner=runner, **self._run_arguments())
        self.assertTrue(result["passed"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            result["profiles"][0]["diagnostics"]["critical_path_proxy"]["stage"],
            "renderer/rasterization",
        )
        self.assertEqual(
            result["profiles"][0]["diagnostics"]["straggler"]["max_to_median_ratio"],
            1.2,
        )
        self.assertNotIn("relative_to_rank1", result["profiles"][2]["diagnostics"])
        self.assertEqual(len(result["rank1_comparisons"]), 3)

    def test_hard_kill_in_first_child_resumes_from_empty_checkpoint(self):
        arguments = self._run_arguments("first-child-hard-kill")
        partial_paths = []

        def interrupted_runner(command, **kwargs):
            output_dir = Path(command[4])
            output_dir.mkdir(parents=True, exist_ok=True)
            partial = output_dir / "partial-first-child.txt"
            partial.write_text("incomplete\n", encoding="utf-8")
            partial_paths.append(partial)
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            TOP3.run_top3_profile(runner=interrupted_runner, **arguments)

        checkpoint_path = Path(arguments["output_dir"]) / "top3.checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["report"]["profiles"], [])
        self.assertTrue(partial_paths[0].is_file())

        resumed_calls = []

        def resumed_runner(command, **kwargs):
            resumed_calls.append(list(command))
            self._write_outputs(command)
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        arguments["resume"] = True
        result = TOP3.run_top3_profile(runner=resumed_runner, **arguments)
        self.assertTrue(result["passed"])
        self.assertEqual(result["resume"]["recovered_count"], 0)
        self.assertEqual(len(resumed_calls), 3)
        self.assertFalse(partial_paths[0].exists())

    def test_failed_candidate_can_resume_without_rerunning_completed_prefix(self):
        first_calls = []

        def first_runner(command, **kwargs):
            mode = command[9]
            first_calls.append(mode)
            if len(first_calls) == 2:
                return subprocess.CompletedProcess(command, 8, stdout="", stderr="fail")
            self._write_outputs(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        arguments = self._run_arguments("resume")
        first = TOP3.run_top3_profile(runner=first_runner, **arguments)
        self.assertFalse(first["passed"])
        self.assertEqual(len(first_calls), 2)
        resumed_calls = []

        def second_runner(command, **kwargs):
            mode = command[9]
            resumed_calls.append(mode)
            self._write_outputs(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        arguments["resume"] = True
        second = TOP3.run_top3_profile(runner=second_runner, **arguments)
        self.assertTrue(second["passed"])
        self.assertEqual(second["resume"]["recovered_count"], 1)
        self.assertEqual(len(resumed_calls), 2)

    def test_resume_rejects_changed_summary_before_launch(self):
        calls = []

        def failing_runner(command, **kwargs):
            mode = command[9]
            calls.append(mode)
            if len(calls) == 2:
                return subprocess.CompletedProcess(command, 4, stdout="", stderr="")
            self._write_outputs(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        arguments = self._run_arguments("tamper")
        first = TOP3.run_top3_profile(runner=failing_runner, **arguments)
        Path(first["profiles"][0]["summary_path"]).write_text("{}\n", encoding="utf-8")
        # Use a tiny sentinel callable to make an accidental launch unmistakable.
        launched = []

        def sentinel(*args, **kwargs):
            launched.append(True)
            raise AssertionError("runner should not be called")

        arguments["resume"] = True
        with self.assertRaisesRegex(TOP3.Top3ContractError, "summary changed"):
            TOP3.run_top3_profile(runner=sentinel, **arguments)
        self.assertEqual(launched, [])

    def test_failed_attempt_outputs_are_removed_before_resume(self):
        calls = []

        def first_runner(command, **kwargs):
            calls.append(command[9])
            self._write_outputs(command)
            return subprocess.CompletedProcess(command, 9, stdout="", stderr="fail")

        arguments = self._run_arguments("stale-attempt")
        first = TOP3.run_top3_profile(runner=first_runner, **arguments)
        self.assertFalse(first["passed"])

        def writes_nothing(command, **kwargs):
            # A zero exit must not make the stale summary from attempt 1 valid.
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        arguments["resume"] = True
        second = TOP3.run_top3_profile(runner=writes_nothing, **arguments)
        self.assertFalse(second["passed"])
        self.assertIn("cannot load Nsight summary", second["errors"][0])

    def test_resume_identity_binds_iteration_checkpoint_bytes(self):
        checkpoint = (
            self.model
            / "point_cloud"
            / "iteration_14000"
            / "point_cloud.ply"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"point-cloud-v1")

        def failing_runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 4, stdout="", stderr="fail")

        arguments = self._run_arguments("workload-drift")
        TOP3.run_top3_profile(runner=failing_runner, **arguments)
        checkpoint.write_bytes(b"point-cloud-v2")
        arguments["resume"] = True
        with self.assertRaisesRegex(TOP3.Top3ContractError, "identity does not match"):
            TOP3.run_top3_profile(runner=lambda *args, **kwargs: None, **arguments)

    def test_resume_rebuilds_and_rejects_tampered_record_command(self):
        def failing_runner(command, **kwargs):
            self._write_outputs(command)
            return subprocess.CompletedProcess(command, 7, stdout="", stderr="fail")

        arguments = self._run_arguments("record-contract")
        TOP3.run_top3_profile(runner=failing_runner, **arguments)
        checkpoint_path = Path(arguments["output_dir"]) / "top3.checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["report"]["profiles"][0]["command"].append("--injected")
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        Path(arguments["report_path"]).write_text(
            json.dumps(checkpoint["report"]), encoding="utf-8"
        )
        arguments["resume"] = True
        launched = []

        def runner(*args, **kwargs):
            launched.append(True)
            raise AssertionError("must reject before launch")

        with self.assertRaisesRegex(TOP3.Top3ContractError, "command changed"):
            TOP3.run_top3_profile(runner=runner, **arguments)
        self.assertEqual(launched, [])

    def test_summary_provenance_mismatch_fails_closed(self):
        def runner(command, **kwargs):
            self._write_outputs(command)
            mode = command[9]
            output = Path(command[4])
            metadata_path = output / "{}_profile_metadata.json".format(mode)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["repository"]["commit"] = "b" * 40
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            summary_path = output / "{}_summary.json".format(mode)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["metadata"] = metadata
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        result = TOP3.run_top3_profile(
            runner=runner, **self._run_arguments("bad-provenance")
        )
        self.assertFalse(result["passed"])
        self.assertIn("provenance differs", result["errors"][0])

    def test_resume_rejects_tampered_raw_nsight_artifact(self):
        calls = []

        def runner(command, **kwargs):
            calls.append(command[9])
            if len(calls) == 2:
                return subprocess.CompletedProcess(command, 5, stdout="", stderr="")
            self._write_outputs(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        arguments = self._run_arguments("raw-tamper")
        first = TOP3.run_top3_profile(runner=runner, **arguments)
        raw_path = Path(first["profiles"][0]["output_dir"]) / "4dgs_render_tacker.nsys-rep"
        raw_path.write_bytes(b"tampered")
        arguments["resume"] = True
        with self.assertRaisesRegex(TOP3.Top3ContractError, "raw artifacts changed"):
            TOP3.run_top3_profile(runner=lambda *args, **kwargs: None, **arguments)

    def test_fresh_publish_does_not_clobber_report_created_during_run(self):
        arguments = self._run_arguments("publish-race")

        def runner(command, **kwargs):
            self._write_outputs(command)
            Path(arguments["report_path"]).write_text(
                '{"owner":"other"}\n', encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with self.assertRaisesRegex(TOP3.Top3ContractError, "created during profiling"):
            TOP3.run_top3_profile(runner=runner, **arguments)
        self.assertEqual(
            json.loads(Path(arguments["report_path"]).read_text(encoding="utf-8")),
            {"owner": "other"},
        )

    def test_resume_publish_uses_compare_and_swap(self):
        arguments = self._run_arguments("resume-publish-race")
        TOP3.run_top3_profile(
            runner=lambda command, **kwargs: subprocess.CompletedProcess(
                command, 3, stdout="", stderr="fail"
            ),
            **arguments
        )
        changed = {"owner": "other"}

        def runner(command, **kwargs):
            self._write_outputs(command)
            Path(arguments["report_path"]).write_text(
                json.dumps(changed), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        arguments["resume"] = True
        with self.assertRaisesRegex(TOP3.Top3ContractError, "compare-and-swap"):
            TOP3.run_top3_profile(runner=runner, **arguments)
        self.assertEqual(
            json.loads(Path(arguments["report_path"]).read_text(encoding="utf-8")),
            changed,
        )


if __name__ == "__main__":
    unittest.main()
