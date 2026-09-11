"""CPU-only tests for Phase-1 correctness qualification/FPS selection."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "benchmark_tacker_admission.py"
SPEC = importlib.util.spec_from_file_location("benchmark_tacker_admission", MODULE_PATH)
ADMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADMISSION)

MODEL_PATH = "/data/model/flame_steak"
SOURCE_PATH = "/data/source/flame_steak"
VIEW_INDICES = list(range(50))


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _workload():
    return {
        "scene": "flame_steak",
        "iteration": 14000,
        "resolution": [1352, 1014],
        "gaussian_count": 111525,
        "split": "test",
        "model_path": MODEL_PATH,
        "source_path": SOURCE_PATH,
        "current_view_indices": [0, 1],
        "next_view_indices": [1, 2],
    }


def _leaf_document(kind, measurements, persistent_blocks):
    return {
        "schema_version": 1,
        "kind": kind,
        "passed": True,
        "workload": _workload(),
        "device": {"name": "NVIDIA RTX A6000"},
        "measurements": measurements,
        "measurement_config": {"persistent_blocks": persistent_blocks},
        "numerics": {"passed": True},
    }


def _summary(trials):
    values = [float(value) for value in trials]
    median = statistics.median(values)
    return {
        "trial_count": len(values),
        "round_indices": list(range(len(values))),
        "throughput_fps_trials": values,
        "median_throughput_fps": median,
        "median_total_render_ms": 4000.0 / median,
        "min_throughput_fps": min(values),
        "max_throughput_fps": max(values),
    }


def _comparison(name, reference, summaries, lower=None):
    candidate = summaries[name]
    incumbent = summaries[reference]
    ratios = [
        candidate_fps / reference_fps
        for candidate_fps, reference_fps in zip(
            candidate["throughput_fps_trials"],
            incumbent["throughput_fps_trials"],
        )
    ]
    ratio = (
        candidate["median_throughput_fps"]
        / incumbent["median_throughput_fps"]
    )
    resamples = ADMISSION.FORMAL_BOOTSTRAP_RESAMPLES
    lower, upper = ADMISSION._paired_bootstrap_interval(
        candidate["throughput_fps_trials"],
        incumbent["throughput_fps_trials"],
        resamples,
        0,
        "{}-vs-{}".format(name, reference),
    )
    return {
        "candidate": name,
        "reference": reference,
        "round_indices": list(candidate["round_indices"]),
        "paired_fps_ratios": ratios,
        "median_paired_fps_ratio": statistics.median(ratios),
        "median_fps_ratio": ratio,
        "paired_bootstrap_95_ci": {
            "lower": float(lower),
            "upper": upper,
            "confidence": 0.95,
            "resamples": resamples,
            "seed": 0,
            "statistic": "median(candidate_fps)/median(reference_fps)",
            "resampling_unit": "paired_round",
            "percentile_method": "linear_type_7",
        },
    }


def _seal_benchmark_selection(report):
    """Populate the exact deterministic selector output and top-level mirrors."""

    names = [candidate["name"] for candidate in report["candidates"]]
    selection = ADMISSION._select_fps_candidates(
        report["summaries"],
        candidate_qualifications=report["correctness_qualifications"],
        candidate_selection_metadata=report["candidate_selection_metadata"],
        incumbent_name="current_tacker",
        candidate_names=names,
        bootstrap_resamples=ADMISSION.FORMAL_BOOTSTRAP_RESAMPLES,
        seed=ADMISSION.FORMAL_BOOTSTRAP_SEED,
        promotion_min_ratio=ADMISSION.PROMOTION_MIN_FPS_RATIO,
        equivalence_fraction=ADMISSION.EQUIVALENCE_FRACTION,
    )
    report["selection"] = copy.deepcopy(selection)
    report["eligible_ranking"] = list(selection["eligible_ranking"])
    report["experimental_winner"] = selection["experimental_winner"]
    report["deployment_winner"] = selection["deployment_winner"]
    report["promotion"] = copy.deepcopy(selection["promotion"])
    return report


def _set_benchmark_source_identity(
    report,
    candidate_name,
    *,
    manifest_hash,
    selection_hash,
    selected_variant_id,
    candidate_abi_hash,
    persistent_blocks,
    deployment_enabled,
):
    """Make every synthetic trial reflect one source profile identity."""

    candidate = next(
        item for item in report["candidates"] if item["name"] == candidate_name
    )
    qualification_mode = not deployment_enabled
    candidate["qualification_mode"] = qualification_mode
    for run in report["runs"]:
        if run["candidate_name"] != candidate_name:
            continue
        run["qualification_mode"] = qualification_mode
        metrics = run["metrics"]
        metrics["qualification_mode_requested"] = qualification_mode
        metrics["qualification_mode_executed"] = qualification_mode
        metrics["profile_manifest_sha256"] = manifest_hash
        metrics["profile_selection_sha256"] = selection_hash
        metrics["selected_variant_id"] = selected_variant_id
        metrics["selected_candidate_abi_sha256"] = candidate_abi_hash
        metrics["persistent_blocks"] = persistent_blocks
        hashes = metrics["provenance"]["profile_hashes"]
        hashes["profile_manifest_sha256"] = manifest_hash
        hashes["profile_selection_sha256"] = selection_hash
        hashes["selected_candidate_abi_sha256"] = candidate_abi_hash
        hashes["tacker_profile_sha256"] = (
            None if qualification_mode else candidate["profile_file_sha256"]
        )
        hashes["qualification_profile_sha256"] = (
            candidate["profile_file_sha256"] if qualification_mode else None
        )


def _bind_benchmark_profile_sha256(
    report, candidate_name, digest, source_profile=None
):
    """Keep a synthetic candidate row and all of its measured runs coherent."""

    candidate = next(
        item for item in report["candidates"] if item["name"] == candidate_name
    )
    candidate["profile_file_sha256"] = digest
    for run in report["runs"]:
        if run["candidate_name"] != candidate_name:
            continue
        hashes = run["metrics"]["provenance"]["profile_hashes"]
        hashes["active_profile_sha256"] = digest
        active_key = (
            "qualification_profile_sha256"
            if candidate["qualification_mode"]
            else "tacker_profile_sha256"
        )
        hashes[active_key] = digest
    if source_profile is None:
        return
    schema_version = source_profile["schema_version"]
    if schema_version == 1:
        selected_variant_id = "legacy_pos_l1"
        candidate_abi_hash = None
        selection_hash = None
        persistent_blocks = source_profile["manifest"]["persistent_blocks"]
        deployment_enabled = source_profile["admission"]["enabled"]
    else:
        selected_variant_id = source_profile["selected_variant_id"]
        selected = _candidate(source_profile, selected_variant_id)
        candidate_abi_hash = selected["abi_manifest_sha256"]
        selection_hash = source_profile["profile_sha256"]
        persistent_blocks = selected["persistent_blocks"]
        deployment_enabled = source_profile["deployment"]["enabled"]
    _set_benchmark_source_identity(
        report,
        candidate_name,
        manifest_hash=source_profile["manifest_sha256"],
        selection_hash=selection_hash,
        selected_variant_id=selected_variant_id,
        candidate_abi_hash=candidate_abi_hash,
        persistent_blocks=persistent_blocks,
        deployment_enabled=deployment_enabled,
    )


def _annotate_synthetic_source_profile(
    descriptor, *, deployment_enabled, manifest_hash="1" * 64,
    selection_hash="2" * 64
):
    descriptor.update(
        {
            "source_profile_schema_version": 2,
            "source_profile_manifest_sha256": manifest_hash,
            "source_profile_selection_sha256": selection_hash,
            "source_profile_deployment_enabled": deployment_enabled,
        }
    )
    return descriptor


def _benchmark(
    fps_by_name,
    modes=None,
    lower_by_name=None,
    qualifications=None,
    selection_metadata=None,
    strategy="abba",
):
    # Deployment evidence uses the frozen Phase-0 protocol: at least ten
    # paired whole-sequence trials, each covering all 50 views.  Individual
    # test cases may specify a shorter repeating pattern for readability.
    fps_by_name = {
        name: (
            list(values)
            if len(values) >= 10
            else [values[index % len(values)] for index in range(10)]
        )
        for name, values in fps_by_name.items()
    }
    modes = dict(modes or {})
    modes.setdefault("serial", "serial")
    modes.setdefault("two_stream", "two_stream")
    modes.setdefault("current_tacker", "tacker")
    lower_by_name = dict(lower_by_name or {})
    names = list(fps_by_name)
    if qualifications is None:
        qualifications = {name: {"valid": True} for name in names}
    candidates = []
    for name in names:
        mode = modes.get(name, "tacker")
        profile_sha256 = None
        profile_path = None
        if mode == "tacker":
            profile_sha256 = (
                "c" * 64
                if name == "current_tacker"
                else hashlib.sha256(name.encode("utf-8")).hexdigest()
            )
            profile_path = "/profiles/{}.json".format(name)
        candidates.append(
            {
                "name": name,
                "execution_mode": mode,
                "profile_path": profile_path,
                "profile_file_sha256": profile_sha256,
                "qualification_mode": (
                    mode == "tacker" and name != "current_tacker"
                ),
            }
        )
    candidate_by_name = {candidate["name"]: candidate for candidate in candidates}
    eligible_names = [name for name in names if qualifications[name]["valid"]]
    summaries = {
        name: _summary(fps_by_name[name]) for name in eligible_names
    }
    ranking = sorted(
        eligible_names,
        key=lambda name: (-summaries[name]["median_throughput_fps"], name),
    )
    comparisons = [
        _comparison(
            name,
            reference,
            summaries,
            lower=lower_by_name.get(name),
        )
        for reference in ("two_stream", "current_tacker")
        if reference in summaries
        for name in eligible_names
        if name != reference
    ]
    stable_environment = {
        "gpu_name": "NVIDIA RTX A6000",
        "cuda_runtime": "12.4",
        "pytorch_version": "synthetic",
    }
    source_files = {
        "profile_render.py": "b" * 64,
        "configs": "c" * 64,
        "gaussian_renderer/__init__.py": "d" * 64,
        "gaussian_renderer/tacker_pipeline.py": "e" * 64,
        "diff_gaussian_rasterization/__init__.py": "f" * 64,
        "diff_gaussian_rasterization._C": "0" * 64,
    }
    stable_provenance = {
        "environment": copy.deepcopy(stable_environment),
        "repository_commit": "a" * 40,
        "repository_commit_source": "test",
        "repository_dirty": False,
        "source_files": copy.deepcopy(source_files),
        "submodules": [],
    }
    runs = []
    trial_count = len(next(iter(summaries.values()))["throughput_fps_trials"])
    executions, base_order = ADMISSION._expected_benchmark_executions(
        eligible_names, trial_count, strategy, ADMISSION.FORMAL_BOOTSTRAP_SEED
    )
    for execution in executions:
        name = execution["candidate_name"]
        round_index = execution["round_index"]
        fps = summaries[name]["throughput_fps_trials"][round_index]
        candidate = candidate_by_name[name]
        if candidate["execution_mode"] == "tacker":
            active_profile_sha256 = candidate["profile_file_sha256"]
            profile_manifest_sha256 = "1" * 64
            profile_selection_sha256 = "2" * 64
            selected_candidate_abi_sha256 = (
                ADMISSION.EXPECTED_MIXED_ABI_SHA256
            )
            selected_variant_id = (
                "legacy_pos_l1" if name == "current_tacker" else name
            )
            persistent_blocks = 7000
            tacker_profile_sha256 = (
                None
                if candidate["qualification_mode"]
                else active_profile_sha256
            )
            qualification_profile_sha256 = (
                active_profile_sha256
                if candidate["qualification_mode"]
                else None
            )
        else:
            active_profile_sha256 = None
            profile_manifest_sha256 = None
            profile_selection_sha256 = None
            selected_candidate_abi_sha256 = None
            selected_variant_id = None
            persistent_blocks = None
            tacker_profile_sha256 = None
            qualification_profile_sha256 = None
        profile_hashes = {
            "active_profile_sha256": active_profile_sha256,
            "tacker_profile_sha256": tacker_profile_sha256,
            "qualification_profile_sha256": qualification_profile_sha256,
            "profile_manifest_sha256": profile_manifest_sha256,
            "profile_selection_sha256": profile_selection_sha256,
            "selected_candidate_abi_sha256": selected_candidate_abi_sha256,
        }
        run = dict(execution)
        run.update(
            {
                "passed": True,
                "requested_execution_mode": candidate["execution_mode"],
                "profile_path": candidate["profile_path"],
                "qualification_mode": candidate["qualification_mode"],
                "returncode": 0,
                "error": None,
                "metadata_sha256": "9" * 64,
                "metrics": {
                    "throughput_fps": fps,
                    "p50_frame_ms": 1000.0 / fps,
                    "p95_frame_ms": 1100.0 / fps,
                    "max_frame_ms": 1200.0 / fps,
                    "cuda_event_total_render_ms": 4000.0 / fps,
                    "actual_execution_mode": candidate["execution_mode"],
                    "two_stream_fallback_reason": None,
                    "tacker_fallback_reason": None,
                    "qualification_mode_requested": candidate[
                        "qualification_mode"
                    ],
                    "qualification_mode_executed": candidate[
                        "qualification_mode"
                    ],
                    "profile_manifest_sha256": profile_manifest_sha256,
                    "profile_selection_sha256": profile_selection_sha256,
                    "selected_variant_id": selected_variant_id,
                    "selected_candidate_abi_sha256": (
                        selected_candidate_abi_sha256
                    ),
                    "persistent_blocks": persistent_blocks,
                    "stable_environment": copy.deepcopy(stable_environment),
                    "stable_provenance": copy.deepcopy(stable_provenance),
                    "provenance": {
                        "environment": copy.deepcopy(stable_environment),
                        "repository": {
                            "commit": stable_provenance["repository_commit"],
                            "commit_source": stable_provenance[
                                "repository_commit_source"
                            ],
                            "dirty": False,
                            "submodules": [],
                            "source_files": copy.deepcopy(source_files),
                        },
                        "profile_hashes": profile_hashes,
                        "metadata_collection_errors": [],
                    },
                },
            }
        )
        runs.append(run)
    report = {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_fps_benchmark",
        "passed": True,
        "selection_objective": "median_throughput_fps",
        "generated_at_utc": "2026-09-10T00:00:00+00:00",
        "contract": {
            "workload_name": "flame_steak",
            "iteration": 14000,
            "image_width": 1352,
            "image_height": 1014,
            "gaussian_count": 111525,
            "split": "test",
            "warmup_frames": 10,
            "profile_frames": len(VIEW_INDICES),
            "view_indices": list(VIEW_INDICES),
            "model_path": MODEL_PATH,
            "source_path": SOURCE_PATH,
            "timing_method": "perf_counter_with_cuda_synchronize",
            "frame_timing_method": "cuda_event_consumer_completion_intervals",
            "io_in_timed_region": False,
            "throughput_definition": "profile_frames / elapsed_seconds",
        },
        "candidates": candidates,
        "summaries": summaries,
        "ranking": ranking,
        "experimental_winner": ranking[0],
        "paired_comparisons": comparisons,
        "schedule": {
            "strategy": strategy,
            "seed": ADMISSION.FORMAL_BOOTSTRAP_SEED,
            "trials_per_candidate": trial_count,
            "base_order": base_order,
            "executions": executions,
        },
        "bootstrap": {
            "confidence": 0.95,
            "resamples": ADMISSION.FORMAL_BOOTSTRAP_RESAMPLES,
            "seed": ADMISSION.FORMAL_BOOTSTRAP_SEED,
            "resampling_unit": "paired_round",
        },
        "runs": runs,
        "stable_environment": stable_environment,
        "stable_provenance": stable_provenance,
        "errors": [],
        "phase0_exit_condition": {
            "required_trials_per_candidate": 10,
            "required_frames_per_trial": 50,
            "met": True,
        },
    }
    report["correctness_qualifications"] = copy.deepcopy(qualifications)
    report["eligible_candidates"] = list(eligible_names)
    report["excluded_candidates"] = [
        {"name": name, "reason": "correctness_invalid"}
        for name in names
        if not qualifications[name]["valid"]
    ]
    supplied_metadata = copy.deepcopy(selection_metadata or {})
    effective_metadata = {}
    for name in names:
        entry = dict(supplied_metadata.get(name, {}))
        if "abi_complexity" not in entry:
            entry["abi_complexity"] = {
                "serial": 0.0,
                "two_stream": 1.0,
                "tacker": 2.0,
            }[modes.get(name, "tacker")]
        effective_metadata[name] = entry
    report["candidate_selection_metadata"] = effective_metadata
    return _seal_benchmark_selection(report)


def passing_inputs(fps_by_name=None, modes=None, lower_by_name=None):
    template = _read_json(
        PROJECT_ROOT / "tacker_profiles" / "raster_head_sm86.json"
    )
    current_descriptor = copy.deepcopy(
        next(
            candidate
            for candidate in template["candidates"]
            if candidate["variant_id"] == "legacy_pos_l1"
        )
    )
    current_descriptor["source_profile_file_sha256"] = "c" * 64
    _annotate_synthetic_source_profile(
        current_descriptor, deployment_enabled=True
    )
    persistent_blocks = template["manifest"]["persistent_blocks"]
    if fps_by_name is None:
        fps_by_name = {
            "serial": [95.0, 95.1, 94.9, 95.0],
            "two_stream": [100.0, 100.1, 99.9, 100.0],
            "current_tacker": [103.0, 103.1, 102.9, 103.0],
        }
    return {
        "device": {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_device",
            "passed": True,
            "workload": _workload(),
            "device": {
                "name": "NVIDIA RTX A6000",
                "compute_capability": [8, 6],
                "cuda_arch": "sm_86",
            },
            "extensions": {
                "rasterizer": {
                    "capabilities": dict(ADMISSION.EXPECTED_RASTER_CAPABILITIES),
                    "cuda_global_symbols": [ADMISSION.EXPECTED_MIXED_SYMBOL],
                },
                "head": {
                    "capabilities": dict(ADMISSION.EXPECTED_HEAD_CAPABILITIES),
                    "cuda_global_symbols": [
                        ADMISSION.EXPECTED_HEAD_SOLO_SYMBOL,
                        ADMISSION.EXPECTED_HEAD_GPTB_SYMBOL,
                    ],
                },
            },
        },
        "quality": {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_quality_validation",
            "passed": True,
            "workload": dict(
                _workload(),
                frames=len(VIEW_INDICES),
                view_indices=list(VIEW_INDICES),
            ),
            "device": {"name": "NVIDIA RTX A6000"},
            "modes": {
                "serial": {"actual_mode": "serial"},
                "two_stream": {"actual_mode": "two_stream"},
                "tacker": {"actual_mode": "tacker"},
            },
            "deltas": {
                "tacker": {
                    "psnr_drop_db": 0.01,
                    "ssim_drop": 0.00001,
                    "lpips_increase": 0.00001,
                }
            },
        },
        # 25% Raster slowdown and a mixed leaf slower than the solo sum are
        # intentional: both are diagnostics in Phase 1.
        "raster": _leaf_document(
            "4dgaussians_tacker_raster_profile",
            {
                "solo_raster_p50_ms": 8.0,
                "mixed_raster_p50_ms": 10.0,
            },
            persistent_blocks,
        ),
        "leaf": _leaf_document(
            "4dgaussians_tacker_leaf_profile",
            {
                "mixed_p50_ms": 9.0,
                "solo_raster_p50_ms": 8.0,
                "solo_head_p50_ms": 0.8,
            },
            persistent_blocks,
        ),
        "fps_benchmark": _benchmark(
            fps_by_name,
            modes=modes,
            lower_by_name=lower_by_name,
        ),
        "candidate_descriptors": {"current_tacker": current_descriptor},
        "mixed_abi": _read_json(
            PROJECT_ROOT
            / "submodules"
            / "depth-diff-gaussian-rasterization"
            / "abi"
            / "tacker_mixed_render_head_v1.json"
        ),
        "head_abi": _read_json(
            PROJECT_ROOT / "tacker_ext" / "abi" / "head_linear_v1.json"
        ),
        "template": template,
    }


def _legacy_enabled_profile():
    manifest = {
        "rasterizer_commit": ADMISSION.EXPECTED_RASTERIZER_COMMIT,
        "pair_key": ADMISSION.EXPECTED_PAIR_KEY,
        "cuda_arch": ADMISSION.EXPECTED_CUDA_ARCH,
        "compute_capability": list(ADMISSION.EXPECTED_COMPUTE_CAPABILITY),
        "gpu_name": ADMISSION.EXPECTED_GPU_NAME,
        "workload": ADMISSION.EXPECTED_SCENE,
        "iteration": ADMISSION.EXPECTED_ITERATION,
        "gaussian_count": ADMISSION.EXPECTED_GAUSSIANS,
        "resolution": list(ADMISSION.EXPECTED_RESOLUTION),
        "physical_cta_threads": 384,
        "raster_thread_range_inclusive": [0, 255],
        "head_thread_range_inclusive": [256, 383],
        "raster_named_barrier_id": 1,
        "head_named_barrier_ids": [],
        "head_input_dtype": "float16",
        "head_weight_dtype": "float16",
        "head_bias_dtype": "float32",
        "head_accumulation_dtype": "float32",
        "head_output_dtype": "float32",
        "persistent_blocks": 7000,
    }
    return {
        "schema_version": 1,
        "manifest": manifest,
        "manifest_sha256": ADMISSION.manifest_sha256(manifest),
        "thresholds": dict(ADMISSION.LEGACY_PROFILE_THRESHOLDS),
        "admission": {"enabled": True, "valid": True},
        "measurements": {
            "raster_slowdown_pct": 25.0,
            "mixed_p50_ms": 9.0,
            "solo_raster_p50_ms": 8.0,
            "solo_head_p50_ms": 0.8,
            "tacker_end_to_end_p50_ms": 12.0,
            "two_stream_end_to_end_p50_ms": 11.0,
            "psnr_drop_db": 0.01,
            "ssim_drop": 0.00001,
            "lpips_increase": 0.00001,
        },
    }


def _candidate(report, variant_id):
    return next(
        item for item in report["candidates"] if item["variant_id"] == variant_id
    )


class AdmissionV2Tests(unittest.TestCase):
    def test_slow_raster_and_slow_leaf_are_diagnostics_not_gates(self):
        report, profile = ADMISSION.evaluate_admission(passing_inputs())

        self.assertTrue(report["passed"])
        self.assertIsNotNone(profile)
        self.assertEqual(profile["schema_version"], 2)
        self.assertEqual(report["profile_sha256"], profile["profile_sha256"])
        self.assertEqual(profile["selected_variant_id"], "legacy_pos_l1")
        self.assertEqual(profile["deployment"], {"enabled": True, "valid": True})
        self.assertNotIn("thresholds", profile)
        self.assertNotIn("pair_key", profile["manifest"])
        current = _candidate(profile, "legacy_pos_l1")
        self.assertEqual(current["tile_shape"], [16, 16])
        self.assertIn("resources", current)
        self.assertEqual(current["diagnostics"]["raster_slowdown_pct"], 25.0)
        self.assertFalse(
            current["diagnostics"]["mixed_leaf_faster_than_solo_sum"]
        )
        self.assertEqual(
            set(profile["correctness_thresholds"]),
            {
                "psnr_drop_db_max",
                "ssim_drop_max",
                "lpips_increase_max",
            },
        )

        from tests.test_tacker_pipeline import _load_module

        runtime = _load_module()
        self.assertIsNone(runtime.tacker_profile_admission_reason(profile))

    def test_fastest_correctness_invalid_candidate_is_excluded(self):
        inputs = self._future_inputs(
            bootstrap_lower=1.001,
            future_fps=120.0,
        )
        inputs["candidate_correctness"] = {
            "future": {"valid": False, "passed": False}
        }

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertTrue(report["passed"])
        self.assertEqual(
            report["selection"]["experimental_winner_variant_id"],
            "legacy_pos_l1",
        )
        self.assertIn("future", report["selection"]["ineligible_variant_ids"])
        self.assertFalse(_candidate(report, "future")["correctness"]["valid"])
        self.assertIsNotNone(profile)

    def test_prefiltered_benchmark_may_omit_invalid_candidate_trials(self):
        inputs = self._future_inputs(
            bootstrap_lower=1.001,
            future_fps=120.0,
        )
        qualifications = {
            "serial": {"valid": True},
            "two_stream": {"valid": True},
            "current_tacker": {"valid": True},
            "future": {"valid": False, "diagnostics": {"reason": "numeric"}},
        }
        inputs["fps_benchmark"] = _benchmark(
            {
                "serial": [95.0] * 4,
                "two_stream": [100.0] * 4,
                "current_tacker": [103.0] * 4,
                "future": [120.0] * 4,
            },
            modes={"future": "tacker"},
            qualifications=qualifications,
        )
        inputs["candidate_correctness"] = {
            "future": {"valid": False, "passed": False}
        }

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertTrue(report["passed"])
        self.assertIsNone(_candidate(report, "future")["performance"])
        self.assertIsNotNone(profile)

    def test_only_phase0_may_fallback_to_legacy_current_descriptor(self):
        frozen_phase0 = _read_json(
            PROJECT_ROOT
            / "tacker_profiles"
            / "baselines"
            / "a6000_flame_steak_phase0_20260910"
            / "baseline-report.json"
        )
        validated_phase0 = ADMISSION._validate_fps_benchmark(frozen_phase0)
        self.assertTrue(validated_phase0["phase0_compatibility"])
        self.assertEqual(
            ADMISSION._source_digest(frozen_phase0),
            ADMISSION.FROZEN_PHASE0_FPS_REPORT_SHA256,
        )

        downgraded = copy.deepcopy(passing_inputs()["fps_benchmark"])
        for key in (
            "correctness_qualifications",
            "eligible_candidates",
            "excluded_candidates",
            "candidate_selection_metadata",
            "selection",
            "eligible_ranking",
            "deployment_winner",
            "promotion",
        ):
            downgraded.pop(key, None)
        with self.assertRaisesRegex(
            ADMISSION.AdmissionInputError, "only the frozen Phase-0"
        ):
            ADMISSION._validate_fps_benchmark(downgraded)

        fps = {
            "serial": [95.0] * 10,
            "two_stream": [100.0] * 10,
            "current_tacker": [103.0] * 10,
        }
        qualifications = {name: {"valid": True} for name in fps}
        prefiltered = passing_inputs(fps_by_name=fps)
        prefiltered["fps_benchmark"] = _benchmark(
            fps, qualifications=qualifications
        )
        prefiltered["candidate_descriptors"].pop("current_tacker")
        report, profile = ADMISSION.evaluate_admission(prefiltered)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("requires a SHA-bound current_tacker", report["errors"][0])

        mismatched = passing_inputs(fps_by_name=fps)
        mismatched["fps_benchmark"] = _benchmark(
            fps, qualifications=qualifications
        )
        mismatched["candidate_descriptors"]["current_tacker"][
            "source_profile_file_sha256"
        ] = "d" * 64
        report, profile = ADMISSION.evaluate_admission(mismatched)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("exact profile_file_sha256", report["errors"][0])

    def test_prefiltered_benchmark_requires_measured_valid_baselines(self):
        fps = {
            "serial": [95.0] * 4,
            "two_stream": [100.0] * 4,
            "current_tacker": [103.0] * 4,
        }
        for invalid_baseline in ("serial", "two_stream"):
            with self.subTest(invalid_baseline=invalid_baseline):
                qualifications = {
                    name: {"valid": name != invalid_baseline} for name in fps
                }
                inputs = passing_inputs(fps_by_name=fps)
                benchmark = _benchmark(fps)
                benchmark["correctness_qualifications"] = qualifications
                benchmark["eligible_candidates"] = [
                    name for name in fps if name != invalid_baseline
                ]
                benchmark["excluded_candidates"] = [
                    {
                        "name": invalid_baseline,
                        "reason": "correctness_invalid",
                    }
                ]
                inputs["fps_benchmark"] = benchmark

                report, profile = ADMISSION.evaluate_admission(inputs)

                self.assertFalse(report["passed"])
                self.assertIsNone(profile)
                self.assertIn(
                    "correctness-valid measured {} baseline".format(
                        invalid_baseline
                    ),
                    report["errors"][0],
                )

        physical_mismatch = passing_inputs()
        physical_mismatch["quality"]["modes"]["two_stream"][
            "actual_mode"
        ] = "serial"
        report, profile = ADMISSION.evaluate_admission(physical_mismatch)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn(
            "two_stream baseline failed admission correctness",
            report["errors"][0],
        )

    def test_benchmark_requires_frozen_ten_by_fifty_protocol(self):
        missing_exit = passing_inputs()
        missing_exit["fps_benchmark"].pop("phase0_exit_condition")
        report, profile = ADMISSION.evaluate_admission(missing_exit)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("phase0_exit_condition", report["errors"][0])

        short = passing_inputs()
        summary = short["fps_benchmark"]["summaries"]["serial"]
        summary["trial_count"] = 1
        summary["round_indices"] = summary["round_indices"][:1]
        summary["throughput_fps_trials"] = summary[
            "throughput_fps_trials"
        ][:1]
        report, profile = ADMISSION.evaluate_admission(short)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("at least 10 whole-run trials", report["errors"][0])

        short_frames = passing_inputs()
        contract = short_frames["fps_benchmark"]["contract"]
        contract["profile_frames"] = 49
        contract["view_indices"] = contract["view_indices"][:49]
        report, profile = ADMISSION.evaluate_admission(short_frames)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("exactly 50 profile frames", report["errors"][0])

        bad_warmup = passing_inputs()
        bad_warmup["fps_benchmark"]["contract"]["warmup_frames"] = 9
        report, profile = ADMISSION.evaluate_admission(bad_warmup)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("warmup_frames must be 10", report["errors"][0])

        wrong_views = passing_inputs()
        wrong_views["fps_benchmark"]["contract"]["view_indices"] = list(
            range(1, 51)
        )
        report, profile = ADMISSION.evaluate_admission(wrong_views)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("fixed test views 0-49", report["errors"][0])

    def test_benchmark_schedule_and_run_order_are_sealed(self):
        round_robin = passing_inputs()
        round_robin["fps_benchmark"] = _benchmark(
            {
                "serial": [95.0] * 10,
                "two_stream": [100.0] * 10,
                "current_tacker": [103.0] * 10,
            },
            strategy="round_robin",
        )
        report, profile = ADMISSION.evaluate_admission(round_robin)
        self.assertTrue(report["passed"])
        self.assertIsNotNone(profile)

        cases = []
        bad_seed = passing_inputs()
        bad_seed["fps_benchmark"]["schedule"]["seed"] = 1
        cases.append(("seed", bad_seed, "schedule.seed"))

        bad_abba = passing_inputs()
        executions = bad_abba["fps_benchmark"]["schedule"]["executions"]
        executions[0], executions[1] = executions[1], executions[0]
        cases.append(("abba", bad_abba, "ordering contract"))

        bad_runs = passing_inputs()
        bad_runs["fps_benchmark"]["runs"][0]["round_index"] = 1
        cases.append(("runs", bad_runs, "runs disagree"))

        for name, inputs, error_text in cases:
            with self.subTest(name=name):
                report, profile = ADMISSION.evaluate_admission(inputs)
                self.assertFalse(report["passed"])
                self.assertIsNone(profile)
                self.assertIn(error_text, report["errors"][0])

    def test_prefiltered_selector_output_is_required_and_fully_sealed(self):
        missing = passing_inputs()
        missing["fps_benchmark"].pop("selection")
        report, profile = ADMISSION.evaluate_admission(missing)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("prefiltered FPS benchmark selection", report["errors"][0])

        tampered_selection = passing_inputs()
        tampered_selection["fps_benchmark"]["selection"]["equivalence"][
            "preferred_candidate"
        ] = "serial"
        report, profile = ADMISSION.evaluate_admission(tampered_selection)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("recomputed selector output", report["errors"][0])

        tampered_mirror = passing_inputs()
        tampered_mirror["fps_benchmark"]["promotion"]["decision"] = (
            "forged"
        )
        report, profile = ADMISSION.evaluate_admission(tampered_mirror)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("promotion disagrees", report["errors"][0])

        reverse_only = self._future_inputs(
            bootstrap_lower=1.001, future_fps=102.0
        )
        benchmark = reverse_only["fps_benchmark"]
        direct_index = next(
            index
            for index, comparison in enumerate(benchmark["paired_comparisons"])
            if comparison["candidate"] == "future"
            and comparison["reference"] == "current_tacker"
        )
        benchmark["paired_comparisons"][direct_index] = _comparison(
            "current_tacker", "future", benchmark["summaries"]
        )
        report, profile = ADMISSION.evaluate_admission(reverse_only)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("complete ordered direct-comparison", report["errors"][0])

    def test_run_profile_identity_and_provenance_are_bound_to_candidate(self):
        rebound = self._future_inputs(
            bootstrap_lower=1.001, future_fps=102.0
        )
        rebound_row = next(
            row
            for row in rebound["fps_benchmark"]["candidates"]
            if row["name"] == "future"
        )
        rebound_row["profile_file_sha256"] = "d" * 64
        rebound["candidate_descriptors"]["future"][
            "source_profile_file_sha256"
        ] = "d" * 64
        report, profile = ADMISSION.evaluate_admission(rebound)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("measured profile bytes", report["errors"][0])

        wrong_slot = self._future_inputs(
            bootstrap_lower=1.001, future_fps=102.0
        )
        future_run = next(
            run
            for run in wrong_slot["fps_benchmark"]["runs"]
            if run["candidate_name"] == "future"
        )
        hashes = future_run["metrics"]["provenance"]["profile_hashes"]
        hashes["qualification_profile_sha256"] = hashes[
            "tacker_profile_sha256"
        ]
        hashes["tacker_profile_sha256"] = None
        report, profile = ADMISSION.evaluate_admission(wrong_slot)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("measured profile bytes", report["errors"][0])

        baseline_hash = passing_inputs()
        serial_run = next(
            run
            for run in baseline_hash["fps_benchmark"]["runs"]
            if run["candidate_name"] == "serial"
        )
        serial_run["metrics"]["provenance"]["profile_hashes"][
            "active_profile_sha256"
        ] = "e" * 64
        report, profile = ADMISSION.evaluate_admission(baseline_hash)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("baseline run", report["errors"][0])

        provenance_drift = passing_inputs()
        provenance_drift["fps_benchmark"]["runs"][0]["metrics"][
            "stable_provenance"
        ]["repository_dirty"] = True
        report, profile = ADMISSION.evaluate_admission(provenance_drift)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("stable_provenance disagrees", report["errors"][0])

        descriptor_drift = self._future_inputs(
            bootstrap_lower=1.001, future_fps=102.0
        )
        descriptor_drift["candidate_descriptors"]["future"][
            "source_profile_file_sha256"
        ] = "f" * 64
        report, profile = ADMISSION.evaluate_admission(descriptor_drift)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("exact profile_file_sha256", report["errors"][0])

    def test_every_tacker_trial_identity_is_bound_to_source_profile(self):
        tampered_values = {
            "profile_manifest_sha256": "a" * 64,
            "profile_selection_sha256": "b" * 64,
            "selected_variant_id": "forged_variant",
            "selected_candidate_abi_sha256": "d" * 64,
            "persistent_blocks": 7001,
        }
        hash_fields = {
            "profile_manifest_sha256",
            "profile_selection_sha256",
            "selected_candidate_abi_sha256",
        }
        for candidate_name in ("current_tacker", "future"):
            for field, forged_value in tampered_values.items():
                with self.subTest(candidate=candidate_name, field=field):
                    inputs = (
                        passing_inputs()
                        if candidate_name == "current_tacker"
                        else self._future_inputs(
                            bootstrap_lower=1.001, future_fps=102.0
                        )
                    )
                    candidate_runs = [
                        run
                        for run in inputs["fps_benchmark"]["runs"]
                        if run["candidate_name"] == candidate_name
                    ]
                    # Corrupt the final matching trial so an implementation
                    # that samples only the first trial remains vulnerable.
                    target = candidate_runs[-1]
                    target["metrics"][field] = forged_value
                    if field in hash_fields:
                        target["metrics"]["provenance"]["profile_hashes"][
                            field
                        ] = forged_value

                    report, profile = ADMISSION.evaluate_admission(inputs)

                    self.assertFalse(report["passed"])
                    self.assertIsNone(profile)
                    self.assertIn(field, report["errors"][0])

    def test_qualification_mode_is_derived_from_source_deployment(self):
        inputs = self._future_inputs(
            bootstrap_lower=1.001, future_fps=102.0
        )
        candidate = next(
            row
            for row in inputs["fps_benchmark"]["candidates"]
            if row["name"] == "future"
        )
        self.assertTrue(candidate["qualification_mode"])
        candidate["qualification_mode"] = False
        for run in inputs["fps_benchmark"]["runs"]:
            if run["candidate_name"] != "future":
                continue
            run["qualification_mode"] = False
            metrics = run["metrics"]
            metrics["qualification_mode_requested"] = False
            metrics["qualification_mode_executed"] = False
            hashes = metrics["provenance"]["profile_hashes"]
            hashes["tacker_profile_sha256"] = hashes[
                "active_profile_sha256"
            ]
            hashes["qualification_profile_sha256"] = None

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("source profile deployment state", report["errors"][0])

    def test_formal_nullable_fields_must_be_explicit(self):
        cases = []

        missing_row = passing_inputs()
        next(
            row
            for row in missing_row["fps_benchmark"]["candidates"]
            if row["name"] == "serial"
        ).pop("profile_path")
        cases.append(("candidate-row", missing_row, "required profile fields"))

        missing_outer = passing_inputs()
        missing_outer["fps_benchmark"]["runs"][0].pop("error")
        cases.append(("outer-run", missing_outer, "missing required fields"))

        missing_fallback = passing_inputs()
        missing_fallback["fps_benchmark"]["runs"][0]["metrics"].pop(
            "two_stream_fallback_reason"
        )
        cases.append(
            ("fallback", missing_fallback, "metrics is missing required fields")
        )

        missing_identity = passing_inputs()
        missing_identity["fps_benchmark"]["runs"][0]["metrics"].pop(
            "profile_selection_sha256"
        )
        cases.append(
            ("identity", missing_identity, "metrics is missing required fields")
        )

        missing_top_submodules = passing_inputs()
        missing_top_submodules["fps_benchmark"]["stable_provenance"].pop(
            "submodules"
        )
        for run in missing_top_submodules["fps_benchmark"]["runs"]:
            run["metrics"]["stable_provenance"].pop("submodules")
            run["metrics"]["provenance"]["repository"].pop("submodules")
        cases.append(
            (
                "top-submodules",
                missing_top_submodules,
                "stable_provenance.submodules",
            )
        )

        missing_nested_submodules = passing_inputs()
        for run in missing_nested_submodules["fps_benchmark"]["runs"]:
            run["metrics"]["provenance"]["repository"].pop("submodules")
        cases.append(
            (
                "nested-submodules",
                missing_nested_submodules,
                "repository.submodules",
            )
        )

        for name, inputs, error_text in cases:
            with self.subTest(name=name):
                report, profile = ADMISSION.evaluate_admission(inputs)
                self.assertFalse(report["passed"])
                self.assertIsNone(profile)
                self.assertIn(error_text, report["errors"][0])

    def test_formal_bootstrap_contract_prevents_one_sample_false_promotion(self):
        current = [100.0] * 10
        future = [80.0] * 4 + [102.0] * 6
        one_sample_lower, _ = ADMISSION._paired_bootstrap_interval(
            future, current, 1, 3, "future-vs-current_tacker"
        )
        formal_lower, _ = ADMISSION._paired_bootstrap_interval(
            future,
            current,
            ADMISSION.FORMAL_BOOTSTRAP_RESAMPLES,
            ADMISSION.FORMAL_BOOTSTRAP_SEED,
            "future-vs-current_tacker",
        )
        self.assertGreater(one_sample_lower, 1.0)
        self.assertLessEqual(formal_lower, 1.0)

        report, profile = ADMISSION.evaluate_admission(
            self._future_inputs(bootstrap_lower=1.0)
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_variant_id"], "legacy_pos_l1")
        self.assertIsNotNone(profile)

        for location in ("top", "comparison"):
            with self.subTest(location=location):
                inputs = self._future_inputs(bootstrap_lower=1.001)
                if location == "top":
                    inputs["fps_benchmark"]["bootstrap"]["resamples"] = 1
                else:
                    comparison = next(
                        item
                        for item in inputs["fps_benchmark"]["paired_comparisons"]
                        if item["candidate"] == "future"
                        and item["reference"] == "current_tacker"
                    )
                    comparison["paired_bootstrap_95_ci"]["resamples"] = 1
                rejected, rejected_profile = ADMISSION.evaluate_admission(inputs)
                self.assertFalse(rejected["passed"])
                self.assertIsNone(rejected_profile)
                self.assertIn("10000", rejected["errors"][0])

    def test_benchmark_requires_stable_environment_and_source_hashes(self):
        for mutation, error_text in (
            (
                lambda benchmark: benchmark.update({"stable_environment": None}),
                "stable_environment",
            ),
            (
                lambda benchmark: benchmark["stable_provenance"].update(
                    {"source_files": {}}
                ),
                "source files require",
            ),
        ):
            with self.subTest(error_text=error_text):
                inputs = passing_inputs()
                mutation(inputs["fps_benchmark"])
                report, profile = ADMISSION.evaluate_admission(inputs)
                self.assertFalse(report["passed"])
                self.assertIsNone(profile)
                self.assertIn(error_text, report["errors"][0])

        fps = {
            "serial": [95.0] * 10,
            "two_stream": [100.0] * 10,
            "current_tacker": [103.0] * 10,
        }
        qualifications = {name: {"valid": True} for name in fps}
        for mutation, error_text in (
            (
                lambda provenance: provenance.pop("repository_dirty"),
                "repository_dirty",
            ),
            (
                lambda provenance: provenance["source_files"].pop(
                    "gaussian_renderer/tacker_pipeline.py"
                ),
                "missing source hashes",
            ),
        ):
            with self.subTest(prefiltered_error=error_text):
                inputs = passing_inputs(fps_by_name=fps)
                inputs["fps_benchmark"] = _benchmark(
                    fps, qualifications=qualifications
                )
                mutation(inputs["fps_benchmark"]["stable_provenance"])
                report, profile = ADMISSION.evaluate_admission(inputs)
                self.assertFalse(report["passed"])
                self.assertIsNone(profile)
                self.assertIn(error_text, report["errors"][0])

    def test_additional_benchmark_candidates_must_use_tacker_mode(self):
        inputs = passing_inputs(
            fps_by_name={
                "serial": [95.0] * 10,
                "two_stream": [100.0] * 10,
                "current_tacker": [103.0] * 10,
                "future": [120.0] * 10,
            },
            modes={"future": "serial"},
        )

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("additional candidate future must execute tacker", report["errors"][0])

    def test_baseline_winner_passes_report_without_writing_tacker_profile(self):
        inputs = passing_inputs(
            fps_by_name={
                "serial": [95.0] * 4,
                "two_stream": [104.0] * 4,
                "current_tacker": [100.0] * 4,
            },
            lower_by_name={"two_stream": 1.02},
        )

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_variant_id"], "two_stream")
        self.assertEqual(report["deployment"], {"enabled": False, "valid": False})
        self.assertIsNone(profile)

    def test_two_stream_floor_overrides_retained_slower_incumbent(self):
        inputs = self._future_inputs(
            bootstrap_lower=0.999, future_fps=100.9
        )
        inputs["fps_benchmark"] = _benchmark(
            {
                "serial": [90.0] * 10,
                "two_stream": [100.5] * 10,
                "current_tacker": [100.0] * 10,
                "future": [100.9] * 10,
            }
        )

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_variant_id"], "two_stream")
        self.assertEqual(
            report["selection"]["promotion"]["decision"],
            "selected_two_stream_floor",
        )
        self.assertEqual(report["deployment"], {"enabled": False, "valid": False})
        self.assertIsNone(profile)

    def _future_inputs(self, bootstrap_lower, future_fps=102.0):
        future_trials = [future_fps] * 4
        if bootstrap_lower <= 1.0 and future_fps == 102.0:
            # Median ratio is 1.02, but paired round variation makes the
            # deterministic 95% lower bound fall below 1.0.
            future_trials = [80.0] * 4 + [102.0] * 6
        inputs = passing_inputs(
            fps_by_name={
                "serial": [95.0] * 4,
                "two_stream": [99.0] * 4,
                "current_tacker": [100.0] * 4,
                "future": future_trials,
            },
            lower_by_name={"future": bootstrap_lower},
        )
        descriptor = copy.deepcopy(
            _candidate(inputs["template"], "legacy_pos_l1")
        )
        descriptor["variant_id"] = "future"
        descriptor["source_profile_file_sha256"] = next(
            candidate["profile_file_sha256"]
            for candidate in inputs["fps_benchmark"]["candidates"]
            if candidate["name"] == "future"
        )
        _annotate_synthetic_source_profile(
            descriptor, deployment_enabled=False
        )
        inputs["candidate_descriptors"]["future"] = descriptor
        inputs["quality"]["modes"]["future"] = {"actual_mode": "tacker"}
        inputs["quality"]["deltas"]["future"] = {
            "psnr_drop_db": 0.01,
            "ssim_drop": 0.00001,
            "lpips_increase": 0.00001,
            "numerics": {"passed": True},
        }
        return inputs

    def test_incumbent_promotion_succeeds_at_one_percent_with_ci_above_one(self):
        report, profile = ADMISSION.evaluate_admission(
            self._future_inputs(bootstrap_lower=1.001, future_fps=101.0)
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_variant_id"], "future")
        self.assertTrue(report["selection"]["promotion"]["passed"])
        self.assertAlmostEqual(
            report["selection"]["promotion"]["median_fps_ratio"], 1.01
        )
        self.assertIsNotNone(profile)
        self.assertEqual(profile["profile_sha256"], ADMISSION.profile_sha256(profile))

    def test_incumbent_is_retained_when_bootstrap_lower_does_not_exceed_one(self):
        report, profile = ADMISSION.evaluate_admission(
            self._future_inputs(bootstrap_lower=1.0)
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_variant_id"], "legacy_pos_l1")
        self.assertFalse(report["selection"]["promotion"]["passed"])
        self.assertEqual(
            report["selection"]["promotion"]["decision"],
            "retained_incumbent",
        )
        self.assertIsNotNone(profile)

    def test_equivalent_candidates_use_stable_resource_preference(self):
        inputs = self._future_inputs(bootstrap_lower=0.999, future_fps=100.4)
        inputs["fps_benchmark"]["candidate_selection_metadata"].update(
            {
                "future": {"abi_complexity": 1},
                "current_tacker": {"abi_complexity": 2},
            }
        )
        _seal_benchmark_selection(inputs["fps_benchmark"])

        report, _ = ADMISSION.evaluate_admission(inputs)

        equivalence = report["selection"]["equivalence"]
        self.assertEqual(equivalence["preferred_variant_id"], "future")
        self.assertEqual(
            equivalence["candidate_variant_ids_in_preference_order"][:2],
            ["future", "legacy_pos_l1"],
        )
        self.assertEqual(
            report["selection"]["experimental_winner_variant_id"], "future"
        )
        self.assertEqual(report["selected_variant_id"], "legacy_pos_l1")

    def _oppositely_named_tie_inputs(self, descriptor_resources=None):
        fps = {
            "serial": [95.0] * 10,
            "two_stream": [99.0] * 10,
            "current_tacker": [100.0] * 10,
            "alpha": [102.0] * 10,
            "zeta": [102.0] * 10,
        }
        modes = {"alpha": "tacker", "zeta": "tacker"}
        qualifications = {name: {"valid": True} for name in fps}
        inputs = passing_inputs(fps_by_name=fps, modes=modes)
        inputs["fps_benchmark"] = _benchmark(
            fps,
            modes=modes,
            qualifications=qualifications,
        )
        for name, variant_id in (
            ("alpha", "zz_variant"),
            ("zeta", "aa_variant"),
        ):
            descriptor = copy.deepcopy(
                _candidate(inputs["template"], "legacy_pos_l1")
            )
            descriptor["variant_id"] = variant_id
            descriptor["source_profile_file_sha256"] = next(
                candidate["profile_file_sha256"]
                for candidate in inputs["fps_benchmark"]["candidates"]
                if candidate["name"] == name
            )
            _annotate_synthetic_source_profile(
                descriptor, deployment_enabled=False
            )
            _set_benchmark_source_identity(
                inputs["fps_benchmark"],
                name,
                manifest_hash=descriptor[
                    "source_profile_manifest_sha256"
                ],
                selection_hash=descriptor[
                    "source_profile_selection_sha256"
                ],
                selected_variant_id=variant_id,
                candidate_abi_hash=descriptor["abi_manifest_sha256"],
                persistent_blocks=descriptor["persistent_blocks"],
                deployment_enabled=False,
            )
            if descriptor_resources is not None:
                descriptor["resources"] = copy.deepcopy(
                    descriptor_resources[name]
                )
            inputs["candidate_descriptors"][name] = descriptor
            inputs["quality"]["modes"][name] = {"actual_mode": "tacker"}
            inputs["quality"]["deltas"][name] = {
                "psnr_drop_db": 0.01,
                "ssim_drop": 0.00001,
                "lpips_increase": 0.00001,
                "numerics": {"passed": True},
            }
        return inputs

    def test_benchmark_name_breaks_exact_fps_and_equivalence_ties(self):
        report, profile = ADMISSION.evaluate_admission(
            self._oppositely_named_tie_inputs()
        )

        self.assertTrue(report["passed"])
        self.assertEqual(
            report["selection"]["global_median_fps_ranking"][:2],
            ["zz_variant", "aa_variant"],
        )
        self.assertEqual(
            report["selection"]["equivalence"][
                "candidate_variant_ids_in_preference_order"
            ][:2],
            ["zz_variant", "aa_variant"],
        )
        self.assertEqual(report["selected_variant_id"], "zz_variant")
        self.assertIsNotNone(profile)
        self.assertIsNotNone(
            ADMISSION._validate_source_profile_runtime_contract(
                profile, "generated profile"
            )
        )

    def test_prefiltered_descriptor_resources_do_not_change_selector_tie(self):
        inputs = self._oppositely_named_tie_inputs(
            {
                "alpha": {"peak_memory_bytes": 1000},
                "zeta": {"peak_memory_bytes": 1},
            }
        )

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_variant_id"], "zz_variant")
        self.assertNotIn(
            "peak_memory_bytes",
            _candidate(report, "zz_variant")["selection_metadata"],
        )
        self.assertIsNotNone(profile)

    def test_equivalence_fraction_is_inclusive_only_at_half_percent(self):
        included = self._future_inputs(
            bootstrap_lower=0.999, future_fps=99.5
        )
        included["fps_benchmark"]["candidate_selection_metadata"].update(
            {
                "future": {"abi_complexity": 1},
                "current_tacker": {"abi_complexity": 2},
            }
        )
        _seal_benchmark_selection(included["fps_benchmark"])
        report, _ = ADMISSION.evaluate_admission(included)
        self.assertIn(
            "future",
            report["selection"]["equivalence"][
                "candidate_variant_ids_in_preference_order"
            ],
        )

        excluded = self._future_inputs(
            bootstrap_lower=0.999, future_fps=99.49
        )
        excluded["fps_benchmark"]["candidate_selection_metadata"] = copy.deepcopy(
            included["fps_benchmark"]["candidate_selection_metadata"]
        )
        _seal_benchmark_selection(excluded["fps_benchmark"])
        report, _ = ADMISSION.evaluate_admission(excluded)
        self.assertNotIn(
            "future",
            report["selection"]["equivalence"][
                "candidate_variant_ids_in_preference_order"
            ],
        )

    def test_invalid_incumbent_equivalence_never_drops_below_two_stream(self):
        inputs = passing_inputs(
            fps_by_name={
                "serial": [99.8] * 4,
                "two_stream": [100.0] * 4,
                "current_tacker": [90.0] * 4,
            }
        )
        inputs["quality"]["deltas"]["tacker"]["psnr_drop_db"] = 0.1
        inputs["fps_benchmark"]["candidate_selection_metadata"].update(
            {
                "serial": {"abi_complexity": 1},
                "two_stream": {"abi_complexity": 2},
            }
        )
        _seal_benchmark_selection(inputs["fps_benchmark"])

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertTrue(report["passed"])
        self.assertEqual(
            report["selection"]["equivalence"]["preferred_variant_id"],
            "serial",
        )
        self.assertEqual(report["selected_variant_id"], "two_stream")
        self.assertIsNone(profile)

    def test_prefiltered_invalid_incumbent_needs_no_trials_or_bootstrap(self):
        inputs = passing_inputs(
            fps_by_name={
                "serial": [99.8] * 4,
                "two_stream": [100.0] * 4,
                "current_tacker": [120.0] * 4,
            }
        )
        qualifications = {
            "serial": {"valid": True},
            "two_stream": {"valid": True},
            "current_tacker": {"valid": False},
        }
        inputs["fps_benchmark"] = _benchmark(
            {
                "serial": [99.8] * 4,
                "two_stream": [100.0] * 4,
                "current_tacker": [120.0] * 4,
            },
            qualifications=qualifications,
            selection_metadata={
                "serial": {"abi_complexity": 0},
                "two_stream": {"abi_complexity": 1},
                "current_tacker": {"abi_complexity": 2},
            },
        )
        inputs["quality"]["deltas"]["tacker"]["psnr_drop_db"] = 0.1

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertTrue(report["passed"])
        self.assertIsNone(_candidate(report, "legacy_pos_l1")["performance"])
        self.assertEqual(report["selected_variant_id"], "two_stream")
        self.assertEqual(
            report["selection"]["promotion"]["decision"],
            "selected_best_valid_candidate_incumbent_invalid",
        )
        self.assertIsNone(profile)

    def test_profile_hash_is_stable_and_seals_trials_and_winner(self):
        inputs = passing_inputs()
        _, first = ADMISSION.evaluate_admission(inputs)
        _, second = ADMISSION.evaluate_admission(copy.deepcopy(inputs))
        self.assertEqual(first["profile_sha256"], second["profile_sha256"])

        tampered_trial = copy.deepcopy(first)
        _candidate(tampered_trial, "legacy_pos_l1")["performance"][
            "throughput_fps_trials"
        ][0] += 1.0
        self.assertNotEqual(
            tampered_trial["profile_sha256"],
            ADMISSION.profile_sha256(tampered_trial),
        )

        tampered_winner = copy.deepcopy(first)
        tampered_winner["selected_variant_id"] = "two_stream"
        self.assertNotEqual(
            tampered_winner["profile_sha256"],
            ADMISSION.profile_sha256(tampered_winner),
        )

        timestamp_only = copy.deepcopy(first)
        timestamp_only["provenance"]["generated_at_utc"] = "later"
        self.assertEqual(
            timestamp_only["profile_sha256"],
            ADMISSION.profile_sha256(timestamp_only),
        )

    def test_whole_run_summary_or_ranking_tamper_fails_closed(self):
        oversized = passing_inputs()
        oversized["fps_benchmark"]["summaries"]["current_tacker"][
            "throughput_fps_trials"
        ][0] = 10**309
        report, profile = ADMISSION.evaluate_admission(oversized)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("finite", report["errors"][0])

        inputs = passing_inputs()
        inputs["fps_benchmark"]["summaries"]["current_tacker"][
            "median_throughput_fps"
        ] += 1.0
        report, profile = ADMISSION.evaluate_admission(inputs)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("median disagrees", report["errors"][0])

        inputs = passing_inputs()
        inputs["fps_benchmark"]["ranking"].reverse()
        report, profile = ADMISSION.evaluate_admission(inputs)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("ranking", report["errors"][0])

        inputs = passing_inputs()
        inputs["fps_benchmark"]["paired_comparisons"][0][
            "paired_bootstrap_95_ci"
        ]["lower"] += 0.01
        report, profile = ADMISSION.evaluate_admission(inputs)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("bootstrap lower", report["errors"][0])

    def test_quality_failure_and_tampered_runtime_descriptor_fail_closed(self):
        inputs = passing_inputs()
        inputs["quality"]["passed"] = False
        inputs["quality"]["errors"] = [
            "tacker exceeded at least one quality threshold"
        ]
        report, profile = ADMISSION.evaluate_admission(inputs)
        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_variant_id"], "two_stream")
        self.assertIsNone(profile)

        inputs = passing_inputs()
        legacy = _candidate(inputs["template"], "legacy_pos_l1")
        legacy["cuda_symbol"] = "bogus"
        inputs["template"]["profile_sha256"] = ADMISSION.profile_sha256(
            inputs["template"]
        )
        report, profile = ADMISSION.evaluate_admission(inputs)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("cuda_symbol", report["errors"][0])

    def test_additional_tacker_requires_kernel_numerics(self):
        for numerics in (None, {"passed": False}):
            with self.subTest(numerics=numerics):
                inputs = self._future_inputs(
                    bootstrap_lower=1.001, future_fps=120.0
                )
                evidence = inputs["quality"]["deltas"]["future"]
                if numerics is None:
                    evidence.pop("numerics")
                else:
                    evidence["numerics"] = numerics

                report, profile = ADMISSION.evaluate_admission(inputs)

                self.assertTrue(report["passed"])
                self.assertEqual(report["selected_variant_id"], "legacy_pos_l1")
                self.assertIsNotNone(profile)
                future = _candidate(report, "future")
                self.assertFalse(future["correctness"]["valid"])
                self.assertTrue(
                    any(
                        "numerical validation" in reason
                        for reason in future["correctness"]["reasons"]
                    )
                )

    def test_empty_variant_and_invalid_resources_fail_before_profile_write(self):
        for field, value, error_text in (
            ("variant_id", "", "variant_id must be a non-empty string"),
            ("resources", {"registers_per_thread": -1.0}, "resources"),
            ("resources", {"registers_per_thread": True}, "resources"),
        ):
            with self.subTest(field=field, value=value):
                inputs = self._future_inputs(bootstrap_lower=1.001)
                inputs["candidate_descriptors"]["future"][field] = value

                report, profile = ADMISSION.evaluate_admission(inputs)

                self.assertFalse(report["passed"])
                self.assertIsNone(profile)
                self.assertIn(error_text, report["errors"][0])

    def test_cli_loads_additional_candidate_profile_and_checks_file_hash(self):
        source_inputs = self._future_inputs(bootstrap_lower=1.001)
        _, source_profile = ADMISSION.evaluate_admission(source_inputs)
        self.assertEqual(source_profile["selected_variant_id"], "future")
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _, current_source_profile = ADMISSION.evaluate_admission(
                passing_inputs()
            )
            current_path = root / "current-source.json"
            current_path.write_text(
                json.dumps(
                    current_source_profile, sort_keys=True, allow_nan=False
                )
                + "\n",
                encoding="utf-8",
            )
            candidate_path = root / "future-source.json"
            candidate_path.write_text(
                json.dumps(source_profile, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            candidate_sha256 = ADMISSION._sha256_file(candidate_path)

            cli_inputs = self._future_inputs(bootstrap_lower=1.001)
            _bind_benchmark_profile_sha256(
                cli_inputs["fps_benchmark"],
                "future",
                candidate_sha256,
                source_profile,
            )
            _bind_benchmark_profile_sha256(
                cli_inputs["fps_benchmark"],
                "current_tacker",
                ADMISSION._sha256_file(current_path),
                current_source_profile,
            )
            flags = {
                "--device-json": "device",
                "--quality-json": "quality",
                "--raster-json": "raster",
                "--leaf-json": "leaf",
                "--fps-benchmark-json": "fps_benchmark",
                "--mixed-abi-json": "mixed_abi",
                "--head-abi-json": "head_abi",
                "--template-profile": "template",
            }
            argv = []
            for flag, key in flags.items():
                path = root / "{}.json".format(key)
                path.write_text(
                    json.dumps(cli_inputs[key], sort_keys=True, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
                argv.extend([flag, str(path)])
            report_path = root / "report.json"
            enabled_path = root / "enabled.json"
            argv.extend(
                [
                    "--candidate-profile",
                    "current_tacker={}".format(current_path),
                    "--candidate-profile",
                    "future={}".format(candidate_path),
                    "--report",
                    str(report_path),
                    "--enabled-profile",
                    str(enabled_path),
                ]
            )

            self.assertEqual(ADMISSION.main(argv), 0)
            written = _read_json(enabled_path)
            self.assertEqual(written["selected_variant_id"], "future")
            self.assertEqual(
                _candidate(written, "future")["source_profile_file_sha256"],
                candidate_sha256,
            )

            bad_benchmark = copy.deepcopy(cli_inputs["fps_benchmark"])
            next(
                item
                for item in bad_benchmark["candidates"]
                if item["name"] == "future"
            )["profile_file_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                ADMISSION.AdmissionInputError, "file SHA-256"
            ):
                ADMISSION._descriptor_from_candidate_profile(
                    bad_benchmark, "future", candidate_path
                )

    def test_v2_current_profile_becomes_the_next_incumbent(self):
        source_inputs = self._future_inputs(
            bootstrap_lower=1.001, future_fps=101.0
        )
        source_inputs["candidate_descriptors"]["future"][
            "persistent_blocks"
        ] = 6000
        _set_benchmark_source_identity(
            source_inputs["fps_benchmark"],
            "future",
            manifest_hash=source_inputs["candidate_descriptors"]["future"][
                "source_profile_manifest_sha256"
            ],
            selection_hash=source_inputs["candidate_descriptors"]["future"][
                "source_profile_selection_sha256"
            ],
            selected_variant_id="future",
            candidate_abi_hash=ADMISSION.EXPECTED_MIXED_ABI_SHA256,
            persistent_blocks=6000,
            deployment_enabled=False,
        )
        source_report, source_profile = ADMISSION.evaluate_admission(source_inputs)
        self.assertTrue(source_report["passed"])
        self.assertEqual(source_profile["selected_variant_id"], "future")
        self.assertEqual(source_profile["manifest"]["persistent_blocks"], 6000)

        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_path = Path(temporary_dir) / "current-v2.json"
            profile_path.write_text(
                json.dumps(source_profile, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            file_sha256 = ADMISSION._sha256_file(profile_path)
            fps = {
                "serial": [95.0] * 10,
                "two_stream": [99.0] * 10,
                "current_tacker": [100.0] * 10,
                "next": [101.0] * 10,
            }
            benchmark = _benchmark(fps)
            _bind_benchmark_profile_sha256(
                benchmark, "current_tacker", file_sha256, source_profile
            )
            current_descriptor, _ = ADMISSION._descriptor_from_candidate_profile(
                benchmark, "current_tacker", profile_path
            )
            current_descriptor["source_profile_file_sha256"] = file_sha256
            self.assertEqual(current_descriptor["variant_id"], "future")
            self.assertEqual(current_descriptor["persistent_blocks"], 6000)

            inputs = passing_inputs(fps_by_name=fps)
            inputs["fps_benchmark"] = benchmark
            for profiler_name in ("raster", "leaf"):
                inputs[profiler_name]["measurement_config"][
                    "persistent_blocks"
                ] = 6000
            next_descriptor = copy.deepcopy(
                _candidate(inputs["template"], "legacy_pos_l1")
            )
            next_descriptor["variant_id"] = "next"
            next_descriptor["persistent_blocks"] = 5000
            next_descriptor["source_profile_file_sha256"] = next(
                row["profile_file_sha256"]
                for row in benchmark["candidates"]
                if row["name"] == "next"
            )
            _annotate_synthetic_source_profile(
                next_descriptor, deployment_enabled=False
            )
            _set_benchmark_source_identity(
                benchmark,
                "next",
                manifest_hash=next_descriptor[
                    "source_profile_manifest_sha256"
                ],
                selection_hash=next_descriptor[
                    "source_profile_selection_sha256"
                ],
                selected_variant_id="next",
                candidate_abi_hash=next_descriptor["abi_manifest_sha256"],
                persistent_blocks=5000,
                deployment_enabled=False,
            )
            inputs["candidate_descriptors"] = {
                "current_tacker": current_descriptor,
                "next": next_descriptor,
            }
            inputs["quality"]["modes"]["next"] = {"actual_mode": "tacker"}
            inputs["quality"]["deltas"]["next"] = {
                "psnr_drop_db": 0.01,
                "ssim_drop": 0.00001,
                "lpips_increase": 0.00001,
                "numerics": {"passed": True},
            }

            report, profile = ADMISSION.evaluate_admission(inputs)

            self.assertTrue(report["passed"])
            self.assertEqual(report["selection"]["incumbent_variant_id"], "future")
            self.assertEqual(report["selected_variant_id"], "next")
            self.assertAlmostEqual(
                report["selection"]["promotion"]["median_fps_ratio"], 1.01
            )
            self.assertEqual(profile["manifest"]["persistent_blocks"], 5000)
            self.assertEqual(
                _candidate(profile, "future")["performance"][
                    "throughput_fps_trials"
                ],
                benchmark["summaries"]["current_tacker"][
                    "throughput_fps_trials"
                ],
            )

            from tests.test_tacker_pipeline import _load_module

            runtime = _load_module()
            self.assertIsNone(runtime.tacker_profile_admission_reason(profile))

    def test_cli_migrates_sha_bound_enabled_v1_current_profile(self):
        fps = {
            "serial": [95.0] * 10,
            "two_stream": [100.0] * 10,
            "current_tacker": [103.0] * 10,
        }
        inputs = passing_inputs(fps_by_name=fps)
        inputs["fps_benchmark"] = _benchmark(
            fps,
            qualifications={name: {"valid": True} for name in fps},
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            legacy_path = root / "legacy-current.json"
            legacy_path.write_text(
                json.dumps(
                    _legacy_enabled_profile(), sort_keys=True, allow_nan=False
                )
                + "\n",
                encoding="utf-8",
            )
            _bind_benchmark_profile_sha256(
                inputs["fps_benchmark"],
                "current_tacker",
                ADMISSION._sha256_file(legacy_path),
                _legacy_enabled_profile(),
            )
            flags = {
                "--device-json": "device",
                "--quality-json": "quality",
                "--raster-json": "raster",
                "--leaf-json": "leaf",
                "--fps-benchmark-json": "fps_benchmark",
                "--mixed-abi-json": "mixed_abi",
                "--head-abi-json": "head_abi",
                "--template-profile": "template",
            }
            argv = []
            for flag, key in flags.items():
                path = root / "{}.json".format(key)
                path.write_text(
                    json.dumps(inputs[key], sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                argv.extend([flag, str(path)])
            report_path = root / "report.json"
            enabled_path = root / "enabled.json"
            argv.extend(
                [
                    "--candidate-profile",
                    "current_tacker={}".format(legacy_path),
                    "--report",
                    str(report_path),
                    "--enabled-profile",
                    str(enabled_path),
                ]
            )

            self.assertEqual(ADMISSION.main(argv), 0)
            profile = _read_json(enabled_path)
            self.assertEqual(profile["selected_variant_id"], "legacy_pos_l1")
            self.assertEqual(
                _candidate(profile, "legacy_pos_l1")[
                    "source_profile_schema_version"
                ],
                1,
            )

            disabled_path = root / "disabled-v1.json"
            disabled = _legacy_enabled_profile()
            disabled["admission"] = {"enabled": False, "valid": False}
            disabled_path.write_text(
                json.dumps(disabled, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            benchmark = copy.deepcopy(inputs["fps_benchmark"])
            next(
                row
                for row in benchmark["candidates"]
                if row["name"] == "current_tacker"
            )["profile_file_sha256"] = ADMISSION._sha256_file(disabled_path)
            with self.assertRaisesRegex(
                ADMISSION.AdmissionInputError, "enabled, valid schema-v1"
            ):
                ADMISSION._descriptor_from_candidate_profile(
                    benchmark, "current_tacker", disabled_path
                )

    def test_candidate_profile_loader_rejects_nonfinite_anywhere(self):
        _, source_profile = ADMISSION.evaluate_admission(
            self._future_inputs(bootstrap_lower=1.001)
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            cases = []
            sealed = copy.deepcopy(source_profile)
            sealed["manifest"]["persistent_blocks"] = math.nan
            cases.append(("sealed", sealed))
            unsealed = copy.deepcopy(source_profile)
            unsealed["debug_only"] = math.inf
            cases.append(("unsealed", unsealed))

            for name, document in cases:
                with self.subTest(name=name):
                    path = root / "{}.json".format(name)
                    path.write_text(
                        json.dumps(document, sort_keys=True, allow_nan=True),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ADMISSION.AdmissionInputError, "cannot load candidate profile"
                    ):
                        ADMISSION._load_hashed_json(path, "candidate profile")

    def test_candidate_profile_rejects_runtime_invalid_source(self):
        source_inputs = self._future_inputs(
            bootstrap_lower=1.001, future_fps=101.0
        )
        _, source_profile = ADMISSION.evaluate_admission(source_inputs)
        cases = []

        persistent_mismatch = copy.deepcopy(source_profile)
        persistent_mismatch["manifest"]["persistent_blocks"] += 1
        persistent_mismatch["manifest_sha256"] = ADMISSION.manifest_sha256(
            persistent_mismatch["manifest"]
        )
        persistent_mismatch["profile_sha256"] = ADMISSION.profile_sha256(
            persistent_mismatch
        )
        cases.append(("persistent", persistent_mismatch, "persistent_blocks"))

        ranking_tamper = copy.deepcopy(source_profile)
        ranking_tamper["selection"]["global_median_fps_ranking"].reverse()
        ranking_tamper["profile_sha256"] = ADMISSION.profile_sha256(
            ranking_tamper
        )
        cases.append(("ranking", ranking_tamper, "ranking"))

        forged_retention = copy.deepcopy(source_profile)
        forged_retention["selected_variant_id"] = "legacy_pos_l1"
        forged_retention["selection"][
            "deployment_winner_variant_id"
        ] = "legacy_pos_l1"
        forged_retention["selection"]["promotion"].update(
            {
                "challenger_variant_id": "future",
                "passed": False,
                "decision": "retained_incumbent",
            }
        )
        forged_retention["profile_sha256"] = ADMISSION.profile_sha256(
            forged_retention
        )
        cases.append(
            (
                "forged-retention",
                forged_retention,
                "promotion decision disagrees",
            )
        )

        forged_ci = copy.deepcopy(source_profile)
        promotion = forged_ci["selection"]["promotion"]
        evaluation = promotion["candidate_evaluations"][0]
        forged_lower = 1.5
        evaluation["paired_comparison"]["paired_bootstrap_95_ci"][
            "lower"
        ] = forged_lower
        evaluation["criteria"]["paired_bootstrap_95_ci_lower"].update(
            {"observed": forged_lower, "passed": True}
        )
        promotion["paired_comparison"] = copy.deepcopy(
            evaluation["paired_comparison"]
        )
        promotion["criteria"] = copy.deepcopy(evaluation["criteria"])
        promotion["paired_bootstrap_95_ci_lower"] = forged_lower
        forged_ci["profile_sha256"] = ADMISSION.profile_sha256(forged_ci)
        cases.append(("forged-ci", forged_ci, "bootstrap lower disagrees"))

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for name, document, error_text in cases:
                with self.subTest(name=name):
                    path = root / "{}.json".format(name)
                    path.write_text(
                        json.dumps(document, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                    benchmark = copy.deepcopy(source_inputs["fps_benchmark"])
                    next(
                        row
                        for row in benchmark["candidates"]
                        if row["name"] == "future"
                    )["profile_file_sha256"] = ADMISSION._sha256_file(path)
                    with self.assertRaisesRegex(
                        ADMISSION.AdmissionInputError, error_text
                    ):
                        ADMISSION._descriptor_from_candidate_profile(
                            benchmark, "future", path
                        )

    def test_invalid_extra_tacker_still_requires_runtime_descriptor(self):
        fps = {
            "serial": [95.0] * 4,
            "two_stream": [100.0] * 4,
            "current_tacker": [103.0] * 4,
            "future": [120.0] * 4,
        }
        qualifications = {
            "serial": {"valid": True},
            "two_stream": {"valid": True},
            "current_tacker": {"valid": True},
            "future": {"valid": False},
        }
        inputs = self._future_inputs(bootstrap_lower=1.001)
        descriptor = inputs["candidate_descriptors"].pop("future")
        inputs["fps_benchmark"] = _benchmark(
            fps,
            modes={"future": "tacker"},
            qualifications=qualifications,
        )

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn(
            "future has no sealed Tacker runtime descriptor", report["errors"][0]
        )

        inputs["candidate_descriptors"]["future"] = descriptor
        report, profile = ADMISSION.evaluate_admission(inputs)
        self.assertTrue(report["passed"])
        self.assertIsNotNone(profile)
        self.assertFalse(_candidate(profile, "future")["correctness"]["valid"])

        from tests.test_tacker_pipeline import _load_module

        runtime = _load_module()
        self.assertIsNone(runtime.tacker_profile_admission_reason(profile))

    def test_legacy_single_run_api_is_read_only(self):
        inputs = passing_inputs()
        inputs.pop("fps_benchmark")
        report, profile = ADMISSION.evaluate_admission(inputs)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("read-only", report["errors"][0])


class OutputContractTests(unittest.TestCase):
    def test_outputs_are_atomic_finite_and_template_cannot_be_clobbered(self):
        report, profile = ADMISSION.evaluate_admission(passing_inputs())
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            report_path = root / "report.json"
            profile_path = root / "profile.json"
            template_path = root / "template.json"
            template_bytes = b'{"template":true}\n'
            template_path.write_bytes(template_bytes)

            written = ADMISSION.write_admission_outputs(
                report,
                profile,
                report_path,
                profile_path,
                template_path=template_path,
            )
            self.assertTrue(written["enabled_profile_written"])
            self.assertEqual(
                _read_json(profile_path)["profile_sha256"],
                profile["profile_sha256"],
            )

            with self.assertRaisesRegex(ValueError, "template"):
                ADMISSION.write_admission_outputs(
                    report,
                    profile,
                    report_path,
                    template_path,
                    template_path=template_path,
                )
            self.assertEqual(template_path.read_bytes(), template_bytes)

            report_path.write_text('{"sentinel":true}\n', encoding="utf-8")
            bad_report = copy.deepcopy(report)
            bad_report["bad"] = math.nan
            with self.assertRaises(ValueError):
                ADMISSION.write_admission_outputs(
                    bad_report,
                    profile,
                    report_path,
                    profile_path,
                    template_path=template_path,
                )
            self.assertEqual(_read_json(report_path), {"sentinel": True})

    def test_baseline_winner_does_not_touch_existing_profile_target(self):
        inputs = passing_inputs(
            fps_by_name={
                "serial": [95.0] * 4,
                "two_stream": [104.0] * 4,
                "current_tacker": [100.0] * 4,
            },
            lower_by_name={"two_stream": 1.02},
        )
        report, profile = ADMISSION.evaluate_admission(inputs)
        self.assertIsNone(profile)
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            report_path = root / "report.json"
            profile_path = root / "existing.json"
            sentinel = {"keep": True}
            profile_path.write_text(json.dumps(sentinel), encoding="utf-8")

            written = ADMISSION.write_admission_outputs(
                report, profile, report_path, profile_path
            )

            self.assertTrue(written["passed"])
            self.assertFalse(written["enabled_profile_written"])
            self.assertEqual(_read_json(profile_path), sentinel)


class Phase2DescriptorContractTests(unittest.TestCase):
    @staticmethod
    def _descriptor(heads, worker_groups):
        modules = ADMISSION.HEAD_MODULES
        first = lambda name: "deformation.{}[1].linear_128x128".format(modules[name])
        suffix = lambda name: [
            "deformation.{}[2]".format(modules[name]),
            "deformation.{}[3]".format(modules[name]),
        ]
        deltas = {
            "pos": "deformation.pos_delta",
            "scales": "deformation.scales_delta",
            "rotations": "deformation.rotations_delta",
            "opacity": "deformation.opacity_delta",
            "shs": "deformation.shs_delta",
        }
        subgroups = []
        for index in range(worker_groups):
            begin = 256 + 128 * index
            subgroups.append(
                {
                    "name": "head_worker_{}".format(index),
                    "thread_range_inclusive": [begin, begin + 127],
                    "threads": 128,
                    "named_barrier_ids": [2],
                }
            )
        return {
            "variant_id": "phase2_{}".format("_".join(heads)),
            "execution_mode": "tacker",
            "partition": {
                "kind": "first_linear_heads",
                "selected_heads": list(heads),
                "worker_groups": worker_groups,
            },
            "cuda_symbol": "tacker_mix_render_heads_v2",
            "abi_manifest_sha256": ADMISSION.EXPECTED_MIXED_MULTI_ABI_SHA256,
            "head_abi_manifest_sha256": ADMISSION.EXPECTED_HEAD_MULTI_ABI_SHA256,
            "physical_cta_threads": 256 + 128 * worker_groups,
            "raster_threads": 256,
            "raster_thread_range_inclusive": [0, 255],
            "raster_named_barrier_id": 1,
            "persistent_blocks": 80,
            "tile_shape": [16, 16],
            "fused_nodes": ["raster.render_leaf"] + [first(name) for name in heads],
            "parallel_nodes": [
                "deformation.{}".format(modules[name])
                for name in ADMISSION.HEAD_ORDER if name not in heads
            ],
            "suffix_nodes": sum((suffix(name) for name in heads), [])
            + ["deformation.apply_residuals"],
            "skipped_python_nodes": [first(name) for name in heads],
            "required_outputs": ["raster.color", "raster.depth", "raster.radii"]
            + [deltas[name] for name in ADMISSION.HEAD_ORDER],
            "stream_lifetimes": [
                "head_inputs:deform_prefix->mixed_done",
                "head_parameters:cache_ready->mixed_done",
                "head_outputs:mixed_done->suffix_ready",
                "render_state:suffix_ready->raster_done",
            ],
            "backend_named_barriers": [
                {
                    "id": 2,
                    "participants": 128 * worker_groups,
                    "purpose": "head_descriptor_broadcast",
                }
            ],
            "backend_subgroups": subgroups,
            "tensor_contract": {
                "input_dtype": "float16",
                "weight_dtype": "float16",
                "bias_dtype": "float32",
                "accumulation_dtype": "float32",
                "output_dtype": "float32",
                "features": 128,
                "max_heads": 5,
            },
            "capability_requirements": {
                "cuda_arch": "sm_86",
                "compute_capability": [8, 6],
                "mixed_render_heads_abi": 2,
            },
            "resources": {
                "registers_per_thread": 32,
                "static_shared_memory_bytes": 0,
                "max_threads_per_block": 1024,
                "active_blocks_per_sm": 1,
            },
        }

    def test_all_c1_and_one_c2_descriptors_pass_generic_admission_contract(self):
        for head in ADMISSION.HEAD_ORDER:
            descriptor = self._descriptor([head], 1)
            self.assertIs(
                ADMISSION._validate_tacker_descriptor(descriptor, "candidate"),
                descriptor,
            )
        pair = self._descriptor(["pos", "scales"], 2)
        self.assertIs(
            ADMISSION._validate_tacker_descriptor(pair, "candidate"), pair
        )

    def test_v2_descriptor_requires_resource_and_dependency_evidence(self):
        for field in (
            "resources",
            "stream_lifetimes",
            "skipped_python_nodes",
            "head_abi_manifest_sha256",
        ):
            descriptor = self._descriptor(["pos", "scales"], 2)
            descriptor.pop(field)
            with self.assertRaises(ADMISSION.AdmissionInputError):
                ADMISSION._validate_tacker_descriptor(descriptor, "candidate")

    def test_v2_descriptor_resources_match_runtime_numeric_rules(self):
        for name, value in (
            ("launch_supported", True),
            ("local_memory_bytes", -1),
        ):
            descriptor = self._descriptor(["pos"], 1)
            descriptor["resources"][name] = value
            with self.subTest(name=name), self.assertRaisesRegex(
                ADMISSION.AdmissionInputError,
                "finite non-negative number",
            ):
                ADMISSION._validate_tacker_descriptor(descriptor, "candidate")

        descriptor = self._descriptor(["pos"], 1)
        descriptor["resources"]["local_memory_bytes"] = None
        self.assertIs(
            ADMISSION._validate_tacker_descriptor(descriptor, "candidate"),
            descriptor,
        )

    def test_v2_manifests_are_validated_and_v1_evidence_cannot_label_v2(self):
        mixed_v2 = _read_json(
            PROJECT_ROOT
            / "submodules"
            / "depth-diff-gaussian-rasterization"
            / "abi"
            / "tacker_mixed_render_heads_v2.json"
        )
        head_v2 = _read_json(
            PROJECT_ROOT / "tacker_ext" / "abi" / "head_linear_v2.json"
        )
        self.assertEqual(
            ADMISSION._validate_mixed_abi(mixed_v2), frozenset((1, 2))
        )
        self.assertEqual(
            ADMISSION._validate_head_abi(head_v2), frozenset((1, 2))
        )

        candidate = self._descriptor(["pos", "scales"], 2)
        candidate["correctness"] = {"valid": True}
        candidate["performance"] = {"median_throughput_fps": 1.0}
        with self.assertRaisesRegex(
            ADMISSION.AdmissionInputError, "mixed ABI v2 evidence"
        ):
            ADMISSION._validate_candidate_abi_evidence(
                [candidate], frozenset((1,)), frozenset((1,))
            )
        ADMISSION._validate_candidate_abi_evidence(
            [candidate], frozenset((1, 2)), frozenset((1, 2))
        )

        tampered = copy.deepcopy(mixed_v2)
        tampered["tacker_ext_dependency"]["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ADMISSION.AdmissionInputError, "exact head ABI"
        ):
            ADMISSION._validate_mixed_abi(tampered)

    def test_v2_manifest_semantic_mutations_fail_closed(self):
        mixed_v2 = _read_json(
            PROJECT_ROOT
            / "submodules"
            / "depth-diff-gaussian-rasterization"
            / "abi"
            / "tacker_mixed_render_heads_v2.json"
        )
        head_v2 = _read_json(
            PROJECT_ROOT / "tacker_ext" / "abi" / "head_linear_v2.json"
        )

        tampered_mixed = copy.deepcopy(mixed_v2)
        tampered_mixed["tensor_contract"]["output_dtype"] = "float16"
        with self.assertRaisesRegex(
            ADMISSION.AdmissionInputError, "semantic digest"
        ):
            ADMISSION._validate_mixed_abi(tampered_mixed)

        tampered_head = copy.deepcopy(head_v2)
        tampered_head["limits"]["max_head_tasks"] = 4
        with self.assertRaisesRegex(
            ADMISSION.AdmissionInputError, "semantic digest"
        ):
            ADMISSION._validate_head_abi(tampered_head)


if __name__ == "__main__":
    unittest.main()
