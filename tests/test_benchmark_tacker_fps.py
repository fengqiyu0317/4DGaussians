"""CPU-only tests for the interleaved whole-run FPS benchmark driver."""

import ast
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "benchmark_tacker_fps.py"
SPEC = importlib.util.spec_from_file_location("benchmark_tacker_fps", MODULE_PATH)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def _percentile(values, probability):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def synthetic_metadata(candidate, contract, fps, fallback_reason=None):
    frame_count = contract["profile_frames"]
    elapsed_seconds = frame_count / float(fps)
    total_render_ms = elapsed_seconds * 1000.0
    # Unequal intervals make p50/p95/max mixups observable in tests.
    completion = [float(index + 1) for index in range(frame_count)]
    cumulative = []
    running = 0.0
    for value in completion:
        running += value
        cumulative.append(running)
    p50 = float(BENCHMARK.statistics.median(completion))
    p95 = float(_percentile(completion, 0.95))
    maximum = float(max(completion))
    cuda_total = cumulative[-1]
    cuda_mean = sum(completion) / frame_count
    trial = {
        "trial_index": 1,
        "elapsed_seconds": elapsed_seconds,
        "total_render_ms": total_render_ms,
        "throughput_fps": float(fps),
        "mean_frame_ms": total_render_ms / frame_count,
        "cuda_event_total_render_ms": cuda_total,
        "cuda_event_mean_frame_ms": cuda_mean,
        "p50_frame_ms": p50,
        "p95_frame_ms": p95,
        "max_frame_ms": maximum,
        "frame_completion_ms": completion,
        "cumulative_frame_completion_ms": cumulative,
        "fallback_reason": fallback_reason,
    }
    qualification_mode = bool(candidate.get("qualification_mode", False))
    active_profile_hash = (
        candidate["profile_file_sha256"]
        if candidate["execution_mode"] == "tacker"
        else None
    )
    is_tacker = candidate["execution_mode"] == "tacker"
    profile_selection_sha256 = "2" * 64 if is_tacker else None
    selected_variant_id = (
        (
            "legacy_pos_l1"
            if candidate["name"] == "current_tacker"
            else candidate["name"]
        )
        if is_tacker
        else None
    )
    selected_candidate_abi_sha256 = "3" * 64 if is_tacker else None
    persistent_blocks = 7000 if is_tacker else None
    source_files = {
        "profile_render.py": "b" * 64,
        "configs": "c" * 64,
        "gaussian_renderer/__init__.py": "d" * 64,
        "gaussian_renderer/tacker_pipeline.py": "e" * 64,
        "diff_gaussian_rasterization/__init__.py": "f" * 64,
        "diff_gaussian_rasterization._C": "1" * 64,
    }
    document = {
        "schema_version": 1,
        "kind": BENCHMARK.CHILD_KIND,
        "passed": True,
        "model_path": contract["model_path"],
        "source_path": contract["source_path"],
        "iteration": contract["iteration"],
        "split": contract["split"],
        "warmup_frames": contract["warmup_frames"],
        "profile_frames": frame_count,
        "view_indices": list(contract["view_indices"]),
        "execution_mode": candidate["execution_mode"],
        "actual_execution_mode": candidate["execution_mode"],
        "two_stream_fallback_reason": None,
        "tacker_fallback_reason": None,
        "qualification_mode_requested": qualification_mode,
        "qualification_mode_executed": qualification_mode,
        "workload_name": contract["workload_name"],
        "tacker_profile": (
            candidate["profile_path"]
            if candidate["execution_mode"] == "tacker" and not qualification_mode
            else None
        ),
        "qualification_profile": (
            candidate["profile_path"] if qualification_mode else None
        ),
        "profile_manifest_sha256": (
            "a" * 64 if candidate["execution_mode"] == "tacker" else None
        ),
        "profile_selection_sha256": profile_selection_sha256,
        "selected_variant_id": selected_variant_id,
        "selected_candidate_abi_sha256": selected_candidate_abi_sha256,
        "persistent_blocks": persistent_blocks,
        "active_profile_sha256": active_profile_hash,
        "tacker_profile_sha256": (
            candidate["profile_file_sha256"]
            if candidate["execution_mode"] == "tacker" and not qualification_mode
            else None
        ),
        "qualification_profile_sha256": (
            candidate["profile_file_sha256"] if qualification_mode else None
        ),
        "trial_count": 1,
        "trials": [trial],
        "aggregate_method": "median_across_whole_sequence_trials",
        "primary_metric": BENCHMARK.SELECTION_OBJECTIVE,
        "primary_metric_higher_is_better": True,
        "representative_trial_index": 1,
        "elapsed_seconds": elapsed_seconds,
        "total_render_ms": total_render_ms,
        "throughput_fps": float(fps),
        "mean_frame_ms": total_render_ms / frame_count,
        "cuda_event_total_render_ms": cuda_total,
        "cuda_event_mean_frame_ms": cuda_mean,
        "p50_frame_ms": p50,
        "p95_frame_ms": p95,
        "max_frame_ms": maximum,
        "frame_completion_ms": completion,
        "cumulative_frame_completion_ms": cumulative,
        "median_elapsed_seconds": elapsed_seconds,
        "median_total_render_ms": total_render_ms,
        "median_throughput_fps": float(fps),
        "median_mean_frame_ms": total_render_ms / frame_count,
        "median_cuda_event_total_render_ms": cuda_total,
        "median_cuda_event_mean_frame_ms": cuda_mean,
        "median_p50_frame_ms": p50,
        "median_p95_frame_ms": p95,
        "median_max_frame_ms": maximum,
        "timing_method": BENCHMARK.TIMING_METHOD,
        "frame_timing_method": BENCHMARK.FRAME_TIMING_METHOD,
        "io_in_timed_region": False,
        "timing_contract": dict(
            BENCHMARK.TIMING_CONTRACT,
            frames_per_trial=frame_count,
            trial_count=1,
        ),
        "gaussian_count": contract["gaussian_count"],
        "image_width": contract["image_width"],
        "image_height": contract["image_height"],
        "gpu_name": "Synthetic NVIDIA RTX A6000",
        "cuda_runtime": "12.4",
        "pytorch_version": "2.4.1",
        "profile_render_sha256": "b" * 64,
        "source_files": dict(source_files),
        "repository_dirty": False,
        # A failed optional NVML probe is deliberately legal.  The physical
        # execution contract above remains authoritative.
        "environment": {
            "gpu": {"name": "Synthetic NVIDIA RTX A6000"},
            "cuda_runtime": "12.4",
            "pytorch_version": "2.4.1",
            "nvidia_smi": None,
        },
        "repository": {
            "commit": "a" * 40,
            "commit_source": "test",
            "dirty": False,
            "submodules": [],
            "source_files": dict(source_files),
        },
        "profile_hashes": {
            "active_profile_sha256": (
                active_profile_hash
            ),
            "tacker_profile_sha256": (
                candidate["profile_file_sha256"]
                if candidate["execution_mode"] == "tacker"
                and not qualification_mode
                else None
            ),
            "qualification_profile_sha256": (
                candidate["profile_file_sha256"] if qualification_mode else None
            ),
            "profile_manifest_sha256": (
                "a" * 64 if candidate["execution_mode"] == "tacker" else None
            ),
            "profile_selection_sha256": profile_selection_sha256,
            "selected_candidate_abi_sha256": selected_candidate_abi_sha256,
        },
        "metadata_collection_errors": ["nvidia-smi: synthetic unavailable"],
    }
    if candidate["execution_mode"] == "two_stream":
        document["two_stream_fallback_reason"] = fallback_reason
    if candidate["execution_mode"] == "tacker":
        document["tacker_fallback_reason"] = fallback_reason
    return document


def synthetic_summaries(fps_trials_by_name):
    return {
        name: {
            "throughput_fps_trials": list(values),
            "median_throughput_fps": float(BENCHMARK.statistics.median(values)),
        }
        for name, values in fps_trials_by_name.items()
    }


def valid_qualifications(names):
    return {name: {"valid": True} for name in names}


class BenchmarkFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_path = self.root / "model"
        self.source_path = self.root / "source"
        self.model_path.mkdir()
        self.source_path.mkdir()
        self.profile_render = self.root / "profile_render.py"
        self.profile_render.write_text("# synthetic child\n", encoding="utf-8")
        self.config_path = self.root / "flame_steak.py"
        self.config_path.write_text("ModelHiddenParams = {}\n", encoding="utf-8")
        self.current_profile = self.root / "current.json"
        self.future_profile = self.root / "future.json"
        self.disabled_v2_profile = self.root / "disabled-v2.json"
        self.current_profile.write_text('{"profile":"current"}\n', encoding="utf-8")
        self.future_profile.write_text('{"profile":"future"}\n', encoding="utf-8")
        self.disabled_v2_profile.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "deployment": {"enabled": False, "valid": False},
                }
            ),
            encoding="utf-8",
        )
        self.contract = BENCHMARK.make_contract(
            self.model_path,
            self.source_path,
            "flame_steak",
            14000,
            "test",
            10,
            4,
            1352,
            1014,
            111525,
            view_indices=[0, 1, 2, 3],
        )
        self.candidates = BENCHMARK.make_candidates(
            self.current_profile,
            ["future={}".format(self.future_profile)],
        )
        self.by_name = {item["name"]: item for item in self.candidates}
        self.correctness_path = self.root / "correctness.json"
        self.correctness_path.write_text(
            json.dumps(valid_qualifications(list(self.by_name))),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _option(command, name):
        return command[command.index(name) + 1]

    def _candidate_for_command(self, command):
        execution_mode = self._option(command, "--execution-mode")
        if execution_mode == "serial":
            return self.by_name["serial"]
        if execution_mode == "two_stream":
            return self.by_name["two_stream"]
        profile_option = (
            "--qualification-profile"
            if "--qualification-mode" in command
            else "--tacker-profile"
        )
        profile_path = self._option(command, profile_option)
        for candidate in self.candidates:
            if candidate["profile_path"] == profile_path:
                return candidate
        raise AssertionError("unknown profile in synthetic command")

    def _main_arguments(self, output_path):
        return [
            "--output",
            str(output_path),
            "--profile-render",
            str(self.profile_render),
            "--current-tacker-profile",
            str(self.current_profile),
            "--candidate",
            "future={}".format(self.future_profile),
            "--correctness-json",
            str(self.correctness_path),
            "--model-path",
            str(self.model_path),
            "--source-path",
            str(self.source_path),
            "--configs",
            str(self.config_path),
            "--workload-name",
            "flame_steak",
            "--iteration",
            "14000",
            "--frames",
            "4",
            "--expected-image-width",
            "1352",
            "--expected-image-height",
            "1014",
            "--expected-gaussian-count",
            "111525",
            "--trials",
            "2",
            "--bootstrap-resamples",
            "100",
        ]


class ScheduleTests(unittest.TestCase):
    def test_abba_is_deterministic_balanced_and_reversed_in_pairs(self):
        names = ["serial", "two_stream", "current_tacker", "future"]
        first = BENCHMARK.build_schedule(names, 6, "abba", seed=17)
        second = BENCHMARK.build_schedule(names, 6, "abba", seed=17)
        self.assertEqual(first, second)
        rounds = [
            [
                item["candidate_name"]
                for item in first
                if item["round_index"] == round_index
            ]
            for round_index in range(6)
        ]
        for order in rounds:
            self.assertEqual(set(order), set(names))
        self.assertEqual(rounds[1], list(reversed(rounds[0])))
        self.assertEqual(rounds[3], list(reversed(rounds[2])))
        self.assertEqual(rounds[5], list(reversed(rounds[4])))
        self.assertNotEqual(rounds[0], rounds[2])

    def test_round_robin_rotates_one_position_each_round(self):
        names = ["a", "b", "c"]
        schedule = BENCHMARK.build_schedule(names, 3, "round_robin", seed=3)
        rounds = [
            [
                item["candidate_name"]
                for item in schedule
                if item["round_index"] == index
            ]
            for index in range(3)
        ]
        self.assertEqual(rounds[1], rounds[0][1:] + rounds[0][:1])
        self.assertEqual(rounds[2], rounds[1][1:] + rounds[1][:1])


class StatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_preserves_ratios(self):
        candidate = [110.0, 90.0, 120.0, 100.0]
        reference = [100.0, 100.0, 100.0, 100.0]
        first = BENCHMARK.paired_bootstrap_ratio(
            candidate, reference, resamples=500, seed=41, label="x-vs-y"
        )
        second = BENCHMARK.paired_bootstrap_ratio(
            candidate, reference, resamples=500, seed=41, label="x-vs-y"
        )
        self.assertEqual(first, second)
        self.assertEqual(first["paired_fps_ratios"], [1.1, 0.9, 1.2, 1.0])
        self.assertAlmostEqual(first["median_paired_fps_ratio"], 1.05)
        self.assertAlmostEqual(first["median_fps_ratio"], 1.05)
        interval = first["paired_bootstrap_95_ci"]
        self.assertLessEqual(interval["lower"], first["median_fps_ratio"])
        self.assertGreaterEqual(interval["upper"], first["median_fps_ratio"])
        self.assertEqual(interval["resampling_unit"], "paired_round")

    def test_aggregate_ranks_median_fps_not_latency_diagnostics(self):
        names = ["serial", "two_stream", "current_tacker", "future"]
        values = {
            "serial": [80.0, 81.0, 79.0],
            "two_stream": [90.0, 89.0, 91.0],
            "current_tacker": [95.0, 96.0, 94.0],
            "future": [101.0, 99.0, 100.0],
        }
        runs = []
        for name in names:
            for round_index, fps in enumerate(values[name]):
                runs.append(
                    {
                        "candidate_name": name,
                        "round_index": round_index,
                        "passed": True,
                        "metadata_path": "/synthetic/{}-{}.json".format(
                            name, round_index
                        ),
                        "metrics": {
                            "throughput_fps": fps,
                            "elapsed_seconds": 50.0 / fps,
                            "total_render_ms": 50000.0 / fps,
                            "profile_manifest_sha256": None,
                        },
                    }
                )
        summaries, ranking, comparisons = BENCHMARK.aggregate_runs(
            runs, names, 3, bootstrap_resamples=200, seed=9
        )
        self.assertEqual(ranking, ["future", "current_tacker", "two_stream", "serial"])
        self.assertEqual(summaries["future"]["median_throughput_fps"], 100.0)
        future_vs_current = next(
            item
            for item in comparisons
            if item["candidate"] == "future"
            and item["reference"] == "current_tacker"
        )
        self.assertEqual(len(future_vs_current["paired_fps_ratios"]), 3)
        self.assertGreater(future_vs_current["median_fps_ratio"], 1.0)
        selection = BENCHMARK.select_candidates(
            summaries,
            valid_qualifications(names),
            candidate_names=names,
            bootstrap_resamples=200,
            seed=9,
        )
        future_evaluation = next(
            item
            for item in selection["promotion"]["candidate_evaluations"]
            if item["candidate"] == "future"
        )
        self.assertEqual(future_evaluation["comparison"], future_vs_current)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.names = ["serial", "two_stream", "current_tacker", "future"]

    def test_oversized_json_integer_is_rejected_as_nonfinite(self):
        summaries = synthetic_summaries(
            {
                "serial": [90.0] * 3,
                "two_stream": [95.0] * 3,
                "current_tacker": [100.0] * 3,
                "future": [101.0] * 3,
            }
        )
        summaries["future"]["throughput_fps_trials"][0] = 10**309
        with self.assertRaisesRegex(
            BENCHMARK.BenchmarkContractError, "finite"
        ):
            BENCHMARK.select_candidates(
                summaries,
                valid_qualifications(self.names),
                candidate_names=self.names,
                bootstrap_resamples=100,
                seed=0,
            )

    def test_correctness_invalid_candidate_cannot_enter_effective_ranking(self):
        summaries = synthetic_summaries(
            {
                "serial": [120.0, 120.0, 120.0],
                "two_stream": [90.0, 90.0, 90.0],
                "current_tacker": [100.0, 100.0, 100.0],
                "future": [200.0, 200.0, 200.0],
            }
        )
        qualifications = valid_qualifications(self.names)
        qualifications["future"] = {
            "valid": False,
            "reason": "numerical mismatch",
        }

        selection = BENCHMARK.select_candidates(
            summaries,
            qualifications,
            candidate_names=self.names,
            bootstrap_resamples=100,
            seed=2,
        )

        self.assertEqual(selection["experimental_winner"], "serial")
        self.assertNotIn("future", selection["eligible_ranking"])
        self.assertEqual(
            selection["excluded_candidates"],
            [{"name": "future", "reason": "correctness_invalid"}],
        )

    def test_invalid_incumbent_is_replaced_without_ratio_or_bootstrap_gate(self):
        summaries = synthetic_summaries(
            {
                "serial": [90.0, 90.0, 90.0],
                "two_stream": [100.0, 100.0, 100.0],
                # An invalid incumbent intentionally has no performance summary.
                "future": [100.1, 100.1, 100.1],
            }
        )
        qualifications = valid_qualifications(self.names)
        qualifications["current_tacker"] = {
            "valid": False,
            "reason": "correctness regression",
        }
        selection = BENCHMARK.select_candidates(
            summaries,
            qualifications,
            candidate_selection_metadata={"future": {"abi_complexity": 0}},
            candidate_names=self.names,
            bootstrap_resamples=100,
            seed=2,
        )

        self.assertEqual(selection["experimental_winner"], "future")
        self.assertEqual(selection["deployment_winner"], "future")
        self.assertEqual(selection["promotion"]["reason"], "incumbent_invalid")
        self.assertEqual(
            selection["promotion"]["decision"], "replace_invalid_incumbent"
        )
        self.assertIsNone(selection["promotion"]["comparison"])
        self.assertNotIn(
            "median_fps_ratio_min", selection["promotion"]["requirements"]
        )
        self.assertIn(
            "current_tacker",
            [item["name"] for item in selection["excluded_candidates"]],
        )

    def test_builtin_abi_defaults_prefer_simpler_baselines_when_equivalent(self):
        summaries = synthetic_summaries(
            {
                "serial": [100.0] * 3,
                "two_stream": [100.0] * 3,
                "current_tacker": [100.0] * 3,
                "future": [100.0] * 3,
            }
        )
        selection = BENCHMARK.select_candidates(
            summaries,
            valid_qualifications(self.names),
            candidate_names=self.names,
            bootstrap_resamples=100,
            seed=3,
        )
        self.assertEqual(
            selection["equivalence"]["candidates"],
            ["serial", "two_stream", "current_tacker", "future"],
        )
        self.assertEqual(
            selection["equivalence"]["preferred_candidate"], "serial"
        )
        self.assertEqual(
            selection["candidate_selection_metadata"]["serial"]
            ["abi_complexity"],
            0.0,
        )
        self.assertEqual(
            selection["candidate_selection_metadata"]["two_stream"]
            ["abi_complexity"],
            1.0,
        )
        self.assertEqual(
            selection["candidate_selection_metadata"]["future"]
            ["abi_complexity"],
            2.0,
        )

    def test_raster_and_leaf_diagnostics_never_veto_e2e_winner(self):
        summaries = synthetic_summaries(
            {
                "serial": [80.0, 80.0, 80.0],
                "two_stream": [90.0, 90.0, 90.0],
                "current_tacker": [100.0, 100.0, 100.0],
                "future": [110.0, 110.0, 110.0],
            }
        )
        qualifications = valid_qualifications(self.names)
        qualifications["future"].update(
            {
                "diagnostics": {
                    "raster_slowdown_pct": 47.0,
                    "mixed_p50_ms": 9.0,
                    "solo_leaf_sum_p50_ms": 4.0,
                    "leaf_savings_ms": -5.0,
                }
            }
        )

        selection = BENCHMARK.select_candidates(
            summaries,
            qualifications,
            candidate_names=self.names,
            bootstrap_resamples=100,
            seed=3,
        )

        self.assertEqual(selection["experimental_winner"], "future")
        self.assertEqual(selection["deployment_winner"], "future")
        self.assertTrue(selection["promotion"]["promoted"])
        self.assertEqual(
            selection["correctness_qualifications"]["future"]["diagnostics"]
            ["raster_slowdown_pct"],
            47.0,
        )

    def test_promotion_requires_ratio_and_strict_confidence_lower_bound(self):
        ratio_too_small = synthetic_summaries(
            {
                "serial": [80.0] * 5,
                "two_stream": [95.0] * 5,
                "current_tacker": [100.0] * 5,
                "future": [100.9] * 5,
            }
        )
        qualifications = valid_qualifications(self.names)
        metadata = {
            "future": {
                "abi_complexity": 1,
                "peak_memory_bytes": 1,
                "registers_per_thread": 1,
                "shared_memory_bytes": 1,
            }
        }
        selection = BENCHMARK.select_candidates(
            ratio_too_small,
            qualifications,
            candidate_selection_metadata=metadata,
            candidate_names=self.names,
            bootstrap_resamples=200,
            seed=4,
        )
        self.assertEqual(selection["experimental_winner"], "future")
        self.assertEqual(selection["deployment_winner"], "current_tacker")

        self.assertFalse(
            selection["promotion"]["criteria"]["median_fps_ratio"]["passed"]
        )

        uncertain = synthetic_summaries(
            {
                "serial": [80.0] * 5,
                "two_stream": [95.0] * 5,
                "current_tacker": [100.0] * 5,
                "future": [90.0, 102.0, 102.0, 102.0, 110.0],
            }
        )
        selection = BENCHMARK.select_candidates(
            uncertain,
            qualifications,
            candidate_names=self.names,
            bootstrap_resamples=1000,
            seed=5,
        )
        self.assertGreaterEqual(
            selection["promotion"]["comparison"]["median_fps_ratio"], 1.01
        )
        self.assertFalse(
            selection["promotion"]["criteria"]
            ["paired_bootstrap_95_ci_lower"]["passed"]
        )
        self.assertEqual(selection["deployment_winner"], "current_tacker")

    def test_valid_incumbent_below_two_stream_uses_the_baseline_floor(self):
        summaries = synthetic_summaries(
            {
                "serial": [80.0] * 5,
                "two_stream": [100.5] * 5,
                "current_tacker": [100.0] * 5,
                # Best FPS, but less than the required 1% incumbent gain.
                "future": [100.9] * 5,
            }
        )
        selection = BENCHMARK.select_candidates(
            summaries,
            valid_qualifications(self.names),
            candidate_names=self.names,
            bootstrap_resamples=200,
            seed=4,
        )

        self.assertEqual(selection["experimental_winner"], "future")
        self.assertEqual(selection["deployment_winner"], "two_stream")
        self.assertEqual(
            selection["promotion"]["decision"], "selected_two_stream_floor"
        )
        self.assertFalse(selection["promotion"]["promoted"])
        self.assertFalse(
            selection["promotion"]["criteria"]
            ["incumbent_two_stream_fps_ratio"]["passed"]
        )

    def test_equivalence_preference_is_stable_but_argmax_remains_exact(self):
        summaries = synthetic_summaries(
            {
                "serial": [80.0] * 3,
                "two_stream": [90.0] * 3,
                "current_tacker": [99.0] * 3,
                "future": [100.0] * 3,
                "fast_complex": [100.4] * 3,
            }
        )
        names = self.names + ["fast_complex"]
        metadata = {
            "future": {
                "abi_complexity": 1,
                "peak_memory_bytes": 20,
                "registers_per_thread": 32,
                "shared_memory_bytes": 64,
            },
            "fast_complex": {
                "abi_complexity": 2,
                "peak_memory_bytes": 10,
                "registers_per_thread": 16,
                "shared_memory_bytes": 32,
            },
        }
        selection = BENCHMARK.select_candidates(
            summaries,
            valid_qualifications(names),
            candidate_selection_metadata=metadata,
            candidate_names=names,
            bootstrap_resamples=100,
            seed=6,
        )
        self.assertEqual(selection["experimental_winner"], "fast_complex")
        self.assertEqual(selection["performance_argmax"], "fast_complex")
        self.assertEqual(
            selection["equivalence"]["preferred_candidate"], "future"
        )
        self.assertEqual(selection["deployment_winner"], "future")

    def test_deployment_never_uses_equivalent_candidate_below_two_stream(self):
        summaries = synthetic_summaries(
            {
                "serial": [80.0] * 3,
                "two_stream": [100.4] * 3,
                "current_tacker": [98.5] * 3,
                "future": [100.0] * 3,
            }
        )
        metadata = {
            "future": {"abi_complexity": 1},
            "two_stream": {"abi_complexity": 2},
        }
        selection = BENCHMARK.select_candidates(
            summaries,
            valid_qualifications(self.names),
            candidate_selection_metadata=metadata,
            candidate_names=self.names,
            bootstrap_resamples=100,
            seed=6,
        )
        self.assertEqual(selection["experimental_winner"], "two_stream")
        self.assertEqual(
            selection["equivalence"]["preferred_candidate"], "future"
        )
        future_evaluation = next(
            item
            for item in selection["promotion"]["candidate_evaluations"]
            if item["candidate"] == "future"
        )
        self.assertFalse(
            future_evaluation["criteria"]["baseline_fps_ratios"]["passed"]
        )
        self.assertEqual(selection["deployment_winner"], "two_stream")

    def test_missing_tie_break_metadata_falls_back_to_name(self):
        summaries = synthetic_summaries(
            {
                "serial": [80.0] * 3,
                "two_stream": [90.0] * 3,
                "current_tacker": [99.0] * 3,
                "zeta": [100.0] * 3,
                "alpha": [100.0] * 3,
            }
        )
        names = ["serial", "two_stream", "current_tacker", "zeta", "alpha"]
        selection = BENCHMARK.select_candidates(
            summaries,
            valid_qualifications(names),
            candidate_names=names,
            bootstrap_resamples=100,
            seed=7,
        )
        self.assertEqual(selection["experimental_winner"], "alpha")
        self.assertEqual(
            selection["equivalence"]["preferred_candidate"], "alpha"
        )

    def test_explicit_qualification_map_must_cover_candidates_exactly(self):
        with self.assertRaisesRegex(
            BENCHMARK.BenchmarkContractError, "cover candidates exactly"
        ):
            BENCHMARK.normalize_correctness_qualifications(
                self.names,
                {
                    "serial": {"valid": True},
                    "two_stream": {"valid": True},
                    "current_tacker": {"valid": True},
                },
            )

    def test_serial_and_two_stream_must_remain_valid_safety_baselines(self):
        summaries = synthetic_summaries(
            {
                "serial": [80.0] * 3,
                "two_stream": [90.0] * 3,
                "current_tacker": [100.0] * 3,
                "future": [110.0] * 3,
            }
        )
        for baseline in ("serial", "two_stream"):
            qualifications = valid_qualifications(self.names)
            qualifications[baseline] = {"valid": False}
            with self.subTest(baseline=baseline):
                with self.assertRaisesRegex(
                    BENCHMARK.BenchmarkContractError,
                    "serial and two_stream baselines",
                ):
                    BENCHMARK.select_candidates(
                        summaries,
                        qualifications,
                        candidate_names=self.names,
                        bootstrap_resamples=100,
                    )


class ContractTests(BenchmarkFixture):
    def test_valid_metadata_accepts_unavailable_optional_nvidia_smi(self):
        metadata = synthetic_metadata(
            self.by_name["current_tacker"], self.contract, 100.0
        )
        metrics = BENCHMARK.validate_child_metadata(
            metadata, self.by_name["current_tacker"], self.contract
        )
        self.assertEqual(metrics["throughput_fps"], 100.0)
        for key in (
            "actual_execution_mode",
            "two_stream_fallback_reason",
            "tacker_fallback_reason",
            "qualification_mode_requested",
            "qualification_mode_executed",
            "profile_selection_sha256",
            "selected_variant_id",
            "selected_candidate_abi_sha256",
            "persistent_blocks",
        ):
            self.assertEqual(metrics[key], metadata.get(key))

    def test_profile_render_source_hash_is_bound_to_repository_metadata(self):
        metadata = synthetic_metadata(
            self.by_name["serial"], self.contract, 80.0
        )
        metadata["repository"]["source_files"]["profile_render.py"] = "c" * 64
        with self.assertRaises(BENCHMARK.BenchmarkContractError):
            BENCHMARK.validate_child_metadata(
                metadata, self.by_name["serial"], self.contract
            )

    def test_every_runtime_source_hash_is_required(self):
        for source_name in BENCHMARK.REQUIRED_PROVENANCE_SOURCE_FILES:
            with self.subTest(source_name=source_name):
                metadata = synthetic_metadata(
                    self.by_name["serial"], self.contract, 80.0
                )
                metadata["source_files"][source_name] = "not-a-sha256"
                metadata["repository"]["source_files"][source_name] = (
                    "not-a-sha256"
                )
                with self.assertRaisesRegex(
                    BENCHMARK.BenchmarkContractError, "lowercase SHA-256"
                ):
                    BENCHMARK.validate_child_metadata(
                        metadata, self.by_name["serial"], self.contract
                    )

    def test_view_workload_and_timing_drift_fail_closed(self):
        cases = []
        wrong_views = synthetic_metadata(
            self.by_name["serial"], self.contract, 80.0
        )
        wrong_views["view_indices"] = [1, 2, 3, 4]
        cases.append(wrong_views)

        wrong_workload = synthetic_metadata(
            self.by_name["serial"], self.contract, 80.0
        )
        wrong_workload["gaussian_count"] += 1
        cases.append(wrong_workload)

        wrong_timing = synthetic_metadata(
            self.by_name["serial"], self.contract, 80.0
        )
        wrong_timing["timing_contract"]["unit"] = "single_frame"
        cases.append(wrong_timing)

        wrong_fps = synthetic_metadata(
            self.by_name["serial"], self.contract, 80.0
        )
        wrong_fps["trials"][0]["throughput_fps"] = 999.0
        cases.append(wrong_fps)

        for metadata in cases:
            with self.subTest(metadata=metadata):
                with self.assertRaises(BENCHMARK.BenchmarkContractError):
                    BENCHMARK.validate_child_metadata(
                        metadata, self.by_name["serial"], self.contract
                    )

    def test_actual_mode_or_fallback_is_rejected(self):
        fallback = synthetic_metadata(
            self.by_name["current_tacker"],
            self.contract,
            100.0,
            fallback_reason="unsupported ABI",
        )
        fallback["actual_execution_mode"] = "two_stream"
        with self.assertRaises(BENCHMARK.BenchmarkContractError):
            BENCHMARK.validate_child_metadata(
                fallback, self.by_name["current_tacker"], self.contract
            )

    def test_nullable_execution_and_profile_evidence_must_be_explicit(self):
        cases = (
            ("serial", "two_stream_fallback_reason"),
            ("serial", "tacker_profile_sha256"),
            ("current_tacker", "tacker_fallback_reason"),
            ("current_tacker", "profile_selection_sha256"),
        )
        for candidate_name, field in cases:
            with self.subTest(candidate=candidate_name, field=field):
                candidate = self.by_name[candidate_name]
                metadata = synthetic_metadata(candidate, self.contract, 100.0)
                metadata.pop(field)
                with self.assertRaisesRegex(
                    BENCHMARK.BenchmarkContractError,
                    "{} is required".format(field),
                ):
                    BENCHMARK.validate_child_metadata(
                        metadata, candidate, self.contract
                    )

        metadata = synthetic_metadata(
            self.by_name["serial"], self.contract, 100.0
        )
        metadata["profile_hashes"].pop("profile_selection_sha256")
        with self.assertRaisesRegex(
            BENCHMARK.BenchmarkContractError, "profile_hashes fields changed"
        ):
            BENCHMARK.validate_child_metadata(
                metadata, self.by_name["serial"], self.contract
            )

        metadata = synthetic_metadata(
            self.by_name["serial"], self.contract, 100.0
        )
        metadata["repository"].pop("submodules")
        with self.assertRaisesRegex(
            BENCHMARK.BenchmarkContractError,
            "repository.submodules must be an explicit array",
        ):
            BENCHMARK.validate_child_metadata(
                metadata, self.by_name["serial"], self.contract
            )

    def test_tacker_profile_content_hash_is_bound_to_candidate(self):
        metadata = synthetic_metadata(
            self.by_name["future"], self.contract, 100.0
        )
        metadata["tacker_profile_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            BENCHMARK.BenchmarkContractError, "candidate file"
        ):
            BENCHMARK.validate_child_metadata(
                metadata, self.by_name["future"], self.contract
            )

    def test_disabled_v2_candidate_uses_verified_qualification_override(self):
        candidates = BENCHMARK.make_candidates(
            self.current_profile,
            ["unreleased={}".format(self.disabled_v2_profile)],
        )
        candidate = next(
            item for item in candidates if item["name"] == "unreleased"
        )
        self.assertTrue(candidate["qualification_mode"])
        command = BENCHMARK.build_child_command(
            "python",
            self.profile_render,
            candidate,
            self.contract,
            self.root / "qualification.metadata.json",
        )
        self.assertIn("--qualification-mode", command)
        self.assertEqual(
            self._option(command, "--qualification-profile"),
            str(self.disabled_v2_profile.resolve()),
        )
        self.assertNotIn("--tacker-profile", command)

        metadata = synthetic_metadata(candidate, self.contract, 105.0)
        metrics = BENCHMARK.validate_child_metadata(
            metadata, candidate, self.contract
        )
        self.assertEqual(metrics["throughput_fps"], 105.0)

        wrong_execution = synthetic_metadata(candidate, self.contract, 105.0)
        wrong_execution["qualification_mode_executed"] = False
        with self.assertRaisesRegex(
            BENCHMARK.BenchmarkContractError, "qualification_mode_executed"
        ):
            BENCHMARK.validate_child_metadata(
                wrong_execution, candidate, self.contract
            )

        wrong_hash = synthetic_metadata(candidate, self.contract, 105.0)
        wrong_hash["qualification_profile_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            BENCHMARK.BenchmarkContractError, "candidate file"
        ):
            BENCHMARK.validate_child_metadata(wrong_hash, candidate, self.contract)

    def test_disabled_v2_profile_cannot_be_the_current_incumbent(self):
        with self.assertRaisesRegex(
            BENCHMARK.BenchmarkContractError, "must be deployment-enabled"
        ):
            BENCHMARK.make_candidates(self.disabled_v2_profile, [])

    def test_correctness_json_accepts_direct_and_wrapped_candidate_mappings(self):
        qualifications = valid_qualifications(
            ["serial", "two_stream", "current_tacker", "future"]
        )
        direct_path = self.root / "correctness-direct.json"
        wrapped_path = self.root / "correctness-wrapped.json"
        direct_path.write_text(json.dumps(qualifications), encoding="utf-8")
        wrapped_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "kind": "synthetic_correctness_report",
                    "candidates": qualifications,
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(
            BENCHMARK.load_correctness_qualifications(direct_path),
            qualifications,
        )
        self.assertEqual(
            BENCHMARK.load_correctness_qualifications(wrapped_path),
            qualifications,
        )

    def test_selection_input_digest_comes_from_the_parsed_byte_snapshot(self):
        path = self.root / "snapshot-correctness.json"
        original = json.dumps(
            valid_qualifications(
                ["serial", "two_stream", "current_tacker", "future"]
            ),
            sort_keys=True,
        ).encode("utf-8")
        path.write_bytes(original)

        loaded, digest = BENCHMARK.load_correctness_qualifications(
            path, include_sha256=True
        )
        path.write_text('{"changed": {"valid": false}}', encoding="utf-8")

        self.assertIn("current_tacker", loaded)
        self.assertEqual(digest, BENCHMARK.hashlib.sha256(original).hexdigest())

    def test_loaded_correctness_still_requires_strict_boolean_valid(self):
        path = self.root / "wrong-valid-type.json"
        qualifications = valid_qualifications(
            ["serial", "two_stream", "current_tacker", "future"]
        )
        qualifications["future"]["valid"] = 1
        path.write_text(json.dumps(qualifications), encoding="utf-8")
        loaded = BENCHMARK.load_correctness_qualifications(path)
        with self.assertRaisesRegex(
            BENCHMARK.BenchmarkContractError, "valid must be a boolean"
        ):
            BENCHMARK.normalize_correctness_qualifications(
                list(qualifications), loaded
            )

    def test_selection_metadata_json_requires_an_object(self):
        path = self.root / "selection-list.json"
        path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(
            BENCHMARK.BenchmarkContractError, "must be a JSON object"
        ):
            BENCHMARK.load_candidate_selection_metadata(path)

    def test_driver_owned_passthrough_options_are_rejected(self):
        with self.assertRaises(BENCHMARK.BenchmarkContractError):
            BENCHMARK.validate_profile_args(["--frames=2"])
        self.assertEqual(
            BENCHMARK.validate_profile_args(["--resolution", "1"]),
            ["--resolution", "1"],
        )


class ExecutionTests(BenchmarkFixture):
    def test_cli_rejects_unknown_or_null_selection_metadata_before_running(self):
        cases = (
            ("unknown", {"serial": {"typo_memory": 1}}, "unknown fields"),
            (
                "null",
                {"serial": {"peak_memory_bytes": None}},
                "must be finite",
            ),
        )
        for name, metadata, error_text in cases:
            with self.subTest(name=name):
                metadata_path = self.root / "{}-selection.json".format(name)
                metadata_path.write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                output_path = self.root / "{}-report.json".format(name)
                result = BENCHMARK.main(
                    self._main_arguments(output_path)
                    + ["--selection-metadata-json", str(metadata_path)]
                )
                self.assertEqual(result, 1)
                report = json.loads(output_path.read_text(encoding="utf-8"))
                self.assertFalse(report["passed"])
                self.assertEqual(report["runs"], [])
                self.assertIn(error_text, report["errors"][0])

    def test_main_refuses_to_overwrite_an_existing_output(self):
        original = self.current_profile.read_bytes()
        result = BENCHMARK.main(
            [
                "--output",
                str(self.current_profile),
                "--current-tacker-profile",
                str(self.current_profile),
                "--correctness-json",
                str(self.correctness_path),
                "--model-path",
                str(self.model_path),
                "--source-path",
                str(self.source_path),
                "--configs",
                str(self.config_path),
                "--workload-name",
                "flame_steak",
                "--iteration",
                "14000",
                "--expected-image-width",
                "1352",
                "--expected-image-height",
                "1014",
                "--expected-gaussian-count",
                "111525",
            ]
        )
        self.assertEqual(result, 1)
        self.assertEqual(self.current_profile.read_bytes(), original)

    def test_main_does_not_clobber_output_created_during_the_run(self):
        output_path = self.root / "raced-report.json"
        raced_bytes = b'{"owner":"other process"}\n'

        def create_raced_output(*args, **kwargs):
            output_path.write_bytes(raced_bytes)
            return {
                "passed": True,
                "artifacts": {},
                "experimental_winner": "current_tacker",
                "deployment_winner": "current_tacker",
                "summaries": {
                    "current_tacker": {"median_throughput_fps": 100.0}
                },
                "errors": [],
            }

        with mock.patch.object(
            BENCHMARK,
            "run_interleaved_benchmark",
            side_effect=create_raced_output,
        ):
            result = BENCHMARK.main(self._main_arguments(output_path))

        self.assertEqual(result, 1)
        self.assertEqual(output_path.read_bytes(), raced_bytes)

    def test_main_loads_cli_selection_inputs_and_passes_them_to_runner(self):
        names = [item["name"] for item in self.candidates]
        qualifications = valid_qualifications(names)
        qualifications["future"] = {
            "valid": False,
            "reason": "synthetic CLI qualification",
        }
        selection_metadata = {
            "future": {
                "abi_complexity": 2,
                "peak_memory_bytes": 4096,
            }
        }
        correctness_path = self.root / "cli-correctness.json"
        selection_path = self.root / "cli-selection.json"
        correctness_path.write_text(
            json.dumps({"candidates": qualifications}), encoding="utf-8"
        )
        selection_path.write_text(
            json.dumps(selection_metadata), encoding="utf-8"
        )
        output_path = self.root / "cli-report.json"
        synthetic_report = {
            "passed": True,
            "artifacts": {},
            "experimental_winner": "current_tacker",
            "deployment_winner": "current_tacker",
            "summaries": {
                "current_tacker": {"median_throughput_fps": 100.0}
            },
            "errors": [],
        }

        with mock.patch.object(
            BENCHMARK,
            "run_interleaved_benchmark",
            return_value=synthetic_report,
        ) as run_benchmark:
            result = BENCHMARK.main(
                self._main_arguments(output_path)
                + [
                    "--correctness-json",
                    str(correctness_path),
                    "--selection-metadata-json",
                    str(selection_path),
                ]
            )

        self.assertEqual(result, 0)
        call = run_benchmark.call_args
        self.assertEqual(call.kwargs["candidate_qualifications"], qualifications)
        self.assertEqual(
            call.kwargs["candidate_selection_metadata"], selection_metadata
        )
        persisted = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(
            persisted["selection_inputs"]["correctness_json"]["path"],
            str(correctness_path.resolve()),
        )
        self.assertEqual(
            persisted["selection_inputs"]["selection_metadata_json"]["sha256"],
            BENCHMARK.sha256_file(selection_path),
        )

    def test_parser_requires_correctness_but_keeps_metadata_optional(self):
        args = BENCHMARK._parser().parse_args(
            self._main_arguments(self.root / "unused-report.json")
        )
        self.assertEqual(args.correctness_json, str(self.correctness_path))
        self.assertIsNone(args.selection_metadata_json)

    def test_mocked_subprocess_runs_unique_metadata_and_aggregates(self):
        fps_by_name = {
            "serial": 80.0,
            "two_stream": 90.0,
            "current_tacker": 100.0,
            "future": 110.0,
        }
        seen_commands = []
        seen_metadata = []

        def runner(command, **kwargs):
            self.assertIs(kwargs["shell"], False)
            self.assertEqual(self._option(command, "--trials"), "1")
            candidate = self._candidate_for_command(command)
            metadata_path = Path(self._option(command, "--metadata"))
            seen_commands.append(list(command))
            seen_metadata.append(metadata_path)
            metadata_path.write_text(
                json.dumps(
                    synthetic_metadata(
                        candidate, self.contract, fps_by_name[candidate["name"]]
                    )
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                command, 0, stdout="synthetic stdout", stderr=""
            )

        session = self.root / "successful-session"
        report = BENCHMARK.run_interleaved_benchmark(
            self.candidates,
            self.contract,
            trials=2,
            strategy="abba",
            seed=7,
            bootstrap_resamples=100,
            session_dir=session,
            profile_render_path=self.profile_render,
            project_root=self.root,
            runner=runner,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["experimental_winner"], "future")
        self.assertEqual(len(seen_commands), 8)
        self.assertEqual(len(set(seen_metadata)), 8)
        self.assertEqual(report["completed_execution_count"], 8)
        self.assertEqual(report["expected_execution_count"], 8)
        self.assertTrue(all(item["passed"] for item in report["runs"]))
        self.assertEqual(
            {
                name: metadata["abi_complexity"]
                for name, metadata in report[
                    "candidate_selection_metadata"
                ].items()
            },
            {
                "serial": 0.0,
                "two_stream": 1.0,
                "current_tacker": 2.0,
                "future": 2.0,
            },
        )
        for item in report["runs"]:
            self.assertTrue(Path(item["stdout_path"]).is_file())
            self.assertTrue(Path(item["stderr_path"]).is_file())
            self.assertTrue(Path(item["metadata_path"]).is_file())

    def test_repository_dirty_or_loaded_binary_drift_fails_the_run(self):
        for drift in ("repository_dirty", "raster_binary"):
            with self.subTest(drift=drift):
                call_count = [0]

                def runner(command, **kwargs):
                    candidate = self._candidate_for_command(command)
                    metadata = synthetic_metadata(
                        candidate, self.contract, 100.0
                    )
                    if call_count[0] > 0:
                        if drift == "repository_dirty":
                            metadata["repository_dirty"] = True
                            metadata["repository"]["dirty"] = True
                        else:
                            key = "diff_gaussian_rasterization._C"
                            metadata["source_files"][key] = "2" * 64
                            metadata["repository"]["source_files"][key] = (
                                "2" * 64
                            )
                    call_count[0] += 1
                    Path(self._option(command, "--metadata")).write_text(
                        json.dumps(metadata), encoding="utf-8"
                    )
                    return subprocess.CompletedProcess(
                        command, 0, stdout="", stderr=""
                    )

                report = BENCHMARK.run_interleaved_benchmark(
                    self.candidates,
                    self.contract,
                    trials=2,
                    strategy="abba",
                    seed=10,
                    bootstrap_resamples=100,
                    session_dir=self.root / "{}-session".format(drift),
                    profile_render_path=self.profile_render,
                    project_root=self.root,
                    runner=runner,
                )

                self.assertFalse(report["passed"])
                self.assertIn("provenance changed", report["errors"][0])

    def test_invalid_candidate_is_not_benchmarked_and_selection_is_reported(self):
        qualifications = valid_qualifications(
            [item["name"] for item in self.candidates]
        )
        qualifications["future"] = {
            "valid": False,
            "reason": "synthetic correctness failure",
            "diagnostics": {"raster_slowdown_pct": 80.0},
        }
        executed = []

        def runner(command, **kwargs):
            candidate = self._candidate_for_command(command)
            executed.append(candidate["name"])
            metadata_path = Path(self._option(command, "--metadata"))
            fps = {
                "serial": 80.0,
                "two_stream": 90.0,
                "current_tacker": 100.0,
            }[candidate["name"]]
            metadata_path.write_text(
                json.dumps(synthetic_metadata(candidate, self.contract, fps)),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        report = BENCHMARK.run_interleaved_benchmark(
            self.candidates,
            self.contract,
            trials=2,
            strategy="abba",
            seed=8,
            bootstrap_resamples=100,
            session_dir=self.root / "qualification-filter-session",
            profile_render_path=self.profile_render,
            project_root=self.root,
            runner=runner,
            candidate_qualifications=qualifications,
        )

        self.assertTrue(report["passed"])
        self.assertNotIn("future", executed)
        self.assertEqual(len(executed), 6)
        self.assertEqual(report["eligible_ranking"][0], "current_tacker")
        self.assertEqual(report["experimental_winner"], "current_tacker")
        self.assertEqual(report["deployment_winner"], "current_tacker")
        self.assertEqual(
            report["excluded_candidates"],
            [{"name": "future", "reason": "correctness_invalid"}],
        )
        self.assertEqual(report["selection"]["performance_argmax"], "current_tacker")

    def test_invalid_current_incumbent_is_skipped_and_replaced(self):
        qualifications = valid_qualifications(
            [item["name"] for item in self.candidates]
        )
        qualifications["current_tacker"] = {
            "valid": False,
            "reason": "synthetic incumbent failure",
        }
        executed = []

        def runner(command, **kwargs):
            candidate = self._candidate_for_command(command)
            executed.append(candidate["name"])
            metadata_path = Path(self._option(command, "--metadata"))
            fps = {"serial": 80.0, "two_stream": 90.0, "future": 91.0}[
                candidate["name"]
            ]
            metadata_path.write_text(
                json.dumps(synthetic_metadata(candidate, self.contract, fps)),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        report = BENCHMARK.run_interleaved_benchmark(
            self.candidates,
            self.contract,
            trials=2,
            strategy="abba",
            seed=9,
            bootstrap_resamples=100,
            session_dir=self.root / "invalid-incumbent-session",
            profile_render_path=self.profile_render,
            project_root=self.root,
            runner=runner,
            candidate_qualifications=qualifications,
        )

        self.assertTrue(report["passed"])
        self.assertNotIn("current_tacker", executed)
        self.assertEqual(len(executed), 6)
        self.assertNotIn("current_tacker", report["summaries"])
        self.assertNotIn("current_tacker", report["eligible_ranking"])
        self.assertEqual(report["experimental_winner"], "future")
        self.assertEqual(report["deployment_winner"], "future")
        self.assertEqual(report["promotion"]["reason_code"], "incumbent_invalid")
        self.assertIsNone(report["promotion"]["comparison"])

    def test_child_failure_stops_and_preserves_auditable_paths(self):
        call_count = [0]

        def runner(command, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                return subprocess.CompletedProcess(
                    command, 23, stdout="partial", stderr="synthetic failure"
                )
            candidate = self._candidate_for_command(command)
            metadata_path = Path(self._option(command, "--metadata"))
            metadata_path.write_text(
                json.dumps(synthetic_metadata(candidate, self.contract, 100.0)),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        report = BENCHMARK.run_interleaved_benchmark(
            self.candidates,
            self.contract,
            trials=2,
            strategy="round_robin",
            seed=0,
            bootstrap_resamples=100,
            session_dir=self.root / "failed-session",
            profile_render_path=self.profile_render,
            project_root=self.root,
            runner=runner,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(len(report["runs"]), 2)
        failed = report["runs"][-1]
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["returncode"], 23)
        self.assertIn("status 23", failed["error"])
        self.assertTrue(Path(failed["stdout_path"]).is_file())
        self.assertTrue(Path(failed["stderr_path"]).is_file())
        self.assertIn("synthetic failure", Path(failed["stderr_path"]).read_text())
        self.assertIn("metadata_path", failed)
        self.assertEqual(report["summaries"], {})
        self.assertEqual(report["ranking"], [])

    def test_contract_failure_from_child_metadata_is_fail_closed(self):
        def runner(command, **kwargs):
            candidate = self._candidate_for_command(command)
            metadata = synthetic_metadata(candidate, self.contract, 100.0)
            metadata["warmup_frames"] = 0
            Path(self._option(command, "--metadata")).write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        report = BENCHMARK.run_interleaved_benchmark(
            self.candidates,
            self.contract,
            trials=2,
            strategy="abba",
            seed=0,
            bootstrap_resamples=100,
            session_dir=self.root / "contract-failure-session",
            profile_render_path=self.profile_render,
            project_root=self.root,
            runner=runner,
        )
        self.assertFalse(report["passed"])
        self.assertIn("warmup_frames mismatch", report["errors"][0])
        self.assertTrue(Path(report["runs"][0]["metadata_path"]).is_file())


class SourceContractTests(unittest.TestCase):
    def test_python37_grammar_and_non_shell_subprocess_contract(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        ast.parse(source, str(MODULE_PATH), feature_version=7)
        self.assertIn("shell=False", source)
        self.assertNotIn("shell=True", source)

    def test_atomic_report_rejects_nan_without_replacing_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            BENCHMARK.atomic_write_json(target, {"passed": True})
            with self.assertRaises(ValueError):
                BENCHMARK.atomic_write_json(target, {"fps": float("nan")})
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")), {"passed": True}
            )

    def test_atomic_no_clobber_publish_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            target.write_text('{"owner":"first"}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                BENCHMARK.atomic_write_json_no_clobber(
                    target, {"owner": "second"}
                )
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"owner": "first"},
            )


if __name__ == "__main__":
    unittest.main()
