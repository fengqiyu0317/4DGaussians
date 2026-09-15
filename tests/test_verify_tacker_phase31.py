"""Fixture-driven CPU tests for the independent Phase-3.1 postflight."""

import importlib.util
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "verify_tacker_phase31.py"
SPEC = importlib.util.spec_from_file_location("verify_tacker_phase31", MODULE_PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def load_fixture_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUTOTUNE = load_fixture_module("phase31_verify_fixture_autotune", VERIFY.AUTOTUNE_PATH)
BENCHMARK = load_fixture_module("phase31_verify_fixture_benchmark", VERIFY.BENCHMARK_PATH)
TOP3 = load_fixture_module("phase31_verify_fixture_top3", VERIFY.TOP3_PATH)
SUMMARIZER = load_fixture_module(
    "phase31_verify_fixture_nsight_summarizer",
    ROOT / "scripts" / "summarize_nsight_stats.py",
)
BENCHMARK_FIXTURES = load_fixture_module(
    "phase31_verify_fixture_benchmark_helpers",
    ROOT / "tests" / "test_benchmark_tacker_fps.py",
)


def formal_wrapper(matrix, ranking, top_k):
    generated = AUTOTUNE.build_formal_candidate_set(matrix, ranking, top_k)
    payload = {
        "schema_version": 1,
        "kind": VERIFY.FORMAL_SET_KIND,
        "matrix_sha256": matrix["matrix_sha256"],
        "screening_ranking_sha256": ranking["ranking_sha256"],
        "baseline_candidates": list(VERIFY.BASELINES),
        "generated_candidate_set": generated,
    }
    document = dict(payload)
    document["formal_set_sha256"] = VERIFY.sha256_json(
        payload, "tacker-phase31-formal-set-v1"
    )
    return document


def synthetic_nsight_stats_csv():
    lines = [
        "Processing [/tmp/nvtx_sum.py]...",
        "Range,Total Time (ns),Instances",
        "profile/render_loop,500000000,1",
    ]
    lines.extend(
        "profile/frame_{},10000000,1".format(index) for index in range(50)
    )
    lines.extend(
        [
            "renderer/setup,50000000,50",
            "renderer/deformation,150000000,50",
            "renderer/activation,50000000,50",
            "renderer/rasterization,150000000,50",
            "",
            "Processing [/tmp/nvtx_gpu_proj_sum.py]...",
            "Range,Total Proj Time (ns),Total GPU Ops",
            "profile/render_loop,450000000,500",
            "renderer/setup,40000000,50",
            "renderer/deformation,140000000,150",
            "renderer/activation,40000000,50",
            "renderer/rasterization,140000000,250",
            "",
            "Processing [/tmp/cuda_api_sum.py]...",
            "Name,Total Time (ns),Num Calls",
            "cudaStreamSynchronize,100000,1",
            "cudaLaunchKernel,1000000,500",
            "",
            "Processing [/tmp/cuda_gpu_kern_sum.py]...",
            "Name,Total Time (ns),Instances",
            "renderCUDA,300000000,250",
            "sgemm_kernel,100000000,250",
            "",
            "Processing [/tmp/cuda_gpu_mem_time_sum.py]...",
            "Name,Total Time (ns)",
            "memcpy,1000000",
            "",
        ]
    )
    return "\n".join(lines)


class CompletedRunFixture(object):
    def __init__(self, root):
        self.root = Path(root)
        self.run = self.root / "run"
        self.inputs = self.root / "inputs"
        self.run.mkdir()
        self.inputs.mkdir()
        self.stages = []
        self._build_identity()
        self._build_matrices_and_screening()
        self._build_profiles()
        self._build_formal_and_selection()
        self._build_report_and_state()

    def write_json(self, path, value):
        VERIFY.atomic_write_json(path, value)
        return Path(path)

    def facts(self, path):
        return VERIFY._artifact_facts(path)

    def add_stage(self, name, result, paths):
        self.stages.append(
            {
                "name": name,
                "status": "succeeded",
                "result": result,
                "artifacts": [self.facts(path) for path in paths],
            }
        )

    def _build_identity(self):
        self.seed_file = self.inputs / "seed.txt"
        self.seed_file.write_text("identity input\n", encoding="utf-8")
        self.model = self.inputs / "model"
        self.source = self.inputs / "source"
        iteration = self.model / "point_cloud" / "iteration_14000"
        iteration.mkdir(parents=True)
        self.source.mkdir()
        workload_paths = {
            "cfg_args": self.model / "cfg_args",
            "point_cloud.ply": iteration / "point_cloud.ply",
            "deformation.pth": iteration / "deformation.pth",
            "deformation_table.pth": iteration / "deformation_table.pth",
            "poses_bounds.npy": self.source / "poses_bounds.npy",
        }
        for name, path in workload_paths.items():
            path.write_bytes((name + "\n").encode("ascii"))
        self.config = self.inputs / "config.py"
        self.config.write_text("value = 1\n", encoding="utf-8")
        self.template = self.inputs / "template.json"
        self.current = self.inputs / "current.json"
        self.write_json(
            self.template,
            {"schema_version": 2, "deployment": {"enabled": False, "valid": False}},
        )
        self.write_json(
            self.current,
            {"schema_version": 1, "admission": {"enabled": True, "valid": True}},
        )
        python = Path(sys.executable).resolve()
        seed = self.facts(self.seed_file)
        profile_render = ROOT / "profile_render.py"
        pipeline = ROOT / "gaussian_renderer" / "tacker_pipeline.py"
        raster_binding = (
            ROOT
            / "submodules"
            / "depth-diff-gaussian-rasterization"
            / "diff_gaussian_rasterization"
            / "__init__.py"
        )
        payload = {
            "paths": {
                "model": str(self.model.resolve()),
                "source": str(self.source.resolve()),
                "python": str(python),
                "nvidia_smi": str(python),
            },
            "files": {
                "config": self.facts(self.config),
                "template_profile": self.facts(self.template),
                "current_tacker_profile": self.facts(self.current),
                "python_executable": self.facts(python),
            },
            "scripts": {
                "runner31": self.facts(ROOT / "scripts" / "run_tacker_phase31.py"),
                "autotune": self.facts(VERIFY.AUTOTUNE_PATH),
                "benchmark": self.facts(VERIFY.BENCHMARK_PATH),
                "top3": self.facts(VERIFY.TOP3_PATH),
                "nsight": self.facts(ROOT / "scripts" / "profile_nsight.sh"),
            },
            "runtime_sources": {
                "profile_render": self.facts(profile_render),
                "tacker_pipeline": self.facts(pipeline),
                "raster_python_binding": self.facts(raster_binding),
            },
            "manifests": {"abi": dict(seed)},
            "workload_files": {
                name: self.facts(path) for name, path in workload_paths.items()
            },
            "configuration_chain": [self.facts(self.config)],
            "environment": {
                "CUDA_VISIBLE_DEVICES": "1",
                "FOURDGS_SOURCE_COMMIT": "a" * 40,
            },
            "workload": {
                "name": "flame_steak",
                "iteration": 14000,
                "split": "test",
                "image_width": 1352,
                "image_height": 1014,
                "gaussian_count": 111525,
                "head_rows": 111525,
                "logical_gpu": 0,
                "physical_gpu": 1,
            },
            "search": {
                "current_persistent_blocks": 7000,
                "extra_persistent_blocks": [],
                "packed_persistent_blocks": [],
                "whole_head_persistent_blocks": [],
                "c3_top_k": 1,
                "c4_top_k_per_family": 1,
                "formal_top_k": 2,
                "screen_batch_size": 30,
                "screen_frames": 10,
                "screen_warmup": 2,
                "screen_trials": 2,
                "seed": 0,
                "leaf_views": 2,
                "leaf_warmup": 5,
                "leaf_repetitions": 50,
                "timeout_seconds": None,
            },
        }
        self.identity = {
            "sha256": VERIFY.sha256_json(payload, "tacker-phase31-identity-v1"),
            "payload": payload,
        }

    def _profile(self, candidate, matrix_sha256):
        return {
            "schema_version": 2,
            "selected_variant_id": candidate["variant_id"],
            "deployment": {"enabled": False, "valid": False},
            "provenance": {
                "matrix_sha256": matrix_sha256,
                "candidate_sha256": candidate["candidate_sha256"],
            },
        }

    def _records(self, candidates, report_facts, score_base):
        return [
            {
                "candidate_sha256": candidate["candidate_sha256"],
                "status": "succeeded",
                "score": score_base - index * 0.001,
                "error": None,
                "artifact_sha256": report_facts["sha256"],
                "attempt": 1,
            }
            for index, candidate in enumerate(candidates)
        ]

    def _screen_group(self, label, candidates, matrix):
        directory = self.run / "screen" / label
        profiles = directory / "profiles"
        profiles.mkdir(parents=True)
        score_base = {"base-all": 1000.0, "c3-all": 2000.0, "c4-all": 3000.0}[label]
        summaries = {
            candidate["variant_id"]: {
                "median_throughput_fps": score_base - index * 0.001
            }
            for index, candidate in enumerate(candidates)
        }
        report_path = self.write_json(
            directory / "benchmark.json",
            {
                "passed": True,
                "contract": {"profile_frames": 10, "warmup_frames": 2},
                "schedule": {
                    "strategy": "round_robin",
                    "trials_per_candidate": 2,
                },
                "candidates": [
                    {"name": name}
                    for name in list(VERIFY.BASELINES)
                    + [candidate["variant_id"] for candidate in candidates]
                ],
                "summaries": summaries,
            },
        )
        report_facts = self.facts(report_path)
        records = self._records(candidates, report_facts, score_base)
        bindings = {}
        paths = [report_path]
        for candidate in candidates:
            profile_path = self.write_json(
                profiles / "{}.json".format(candidate["variant_id"]),
                self._profile(candidate, matrix["matrix_sha256"]),
            )
            paths.append(profile_path)
            bindings[candidate["candidate_sha256"]] = {
                "source_matrix_sha256": matrix["matrix_sha256"],
                "profile": self.facts(profile_path),
                "report": report_facts,
                "screening_batch": label,
            }
        result_path = self.write_json(
            directory / "screening-batch.json",
            {
                "schema_version": 1,
                "kind": "tacker_phase31_screening_batch",
                "source_matrix_sha256": matrix["matrix_sha256"],
                "family": candidates[0]["search_family"],
                "head_count": len(candidates[0]["selected_heads"]),
                "candidate_count": len(candidates),
                "protocol": {
                    "frames": 10,
                    "warmup": 2,
                    "trials": 2,
                    "schedule": "round_robin",
                    "seed": 0,
                },
                "records": records,
                "measurement_source_bindings": bindings,
                "benchmark_report": report_facts,
                "benchmark_passed": True,
            },
        )
        paths.append(result_path)
        result = {
            "path": str(result_path.resolve()),
            "records": records,
            "measurement_source_bindings": bindings,
        }
        self.add_stage("screen-{}".format(label), result, paths)
        return records, bindings

    def _matrix_stage(self, name, matrix):
        path = self.write_json(self.run / "{}.json".format(name), matrix)
        result = {
            "matrix_path": str(path.resolve()),
            "matrix": matrix,
            "matrix_sha256": matrix["matrix_sha256"],
            "candidate_count": len(matrix["candidates"]),
        }
        self.add_stage(name, result, [path])
        return path

    def _build_matrices_and_screening(self):
        resources = self.write_json(
            self.run / "resources.json",
            {
                "schema_version": 1,
                "kind": "tacker_phase31_a6000_resource_query",
                "passed": True,
                "device": {
                    "index": 0,
                    "name": "NVIDIA RTX A6000",
                    "compute_capability": [8, 6],
                    "sm_count": 84,
                },
                "raster_capabilities": {
                    "resource_query_family_aware": True,
                    "supported_mixed_abis": [1, 2, 3, 4],
                    "supported_backend_families": [
                        "first_linear_heads_v2",
                        "packed_first_linear_v3",
                        "whole_heads_v4",
                    ],
                },
                "families": {},
                "manifests": {
                    "abi": {"sha256": self.identity["payload"]["manifests"]["abi"]["sha256"]}
                },
            },
        )
        smi = self.write_json(
            self.run / "nvidia-smi.json",
            {"physical_gpu": 1, "stdout": "NVIDIA RTX A6000"},
        )
        preflight = {
            "resources": str(resources.resolve()),
            "device": VERIFY.load_json(resources)["device"],
            "nvidia_smi": str(smi.resolve()),
        }
        self.add_stage("preflight", preflight, [resources, smi])
        raster_tiles = ((1352 + 15) // 16) * ((1014 + 15) // 16)
        backend_blocks = ((111525 + 15) // 16) * 2
        blocks = AUTOTUNE.derive_persistent_blocks(
            84,
            raster_tiles,
            backend_blocks,
            current_persistent_blocks=7000,
            extra_values=[],
        )
        self.base = AUTOTUNE.build_phase31_base_matrix(
            blocks,
            sm_count=84,
            raster_tile_count=raster_tiles,
            backend_logical_blocks=backend_blocks,
            whole_head_logical_blocks=111525,
            packed_persistent_blocks=blocks,
            whole_head_persistent_blocks=blocks,
        )
        self.base_path = self._matrix_stage("base-matrix", self.base)
        self.base_records, self.base_bindings = self._screen_group(
            "base-all", self.base["candidates"], self.base
        )
        self.c3 = AUTOTUNE.extend_phase31_with_c3(
            self.base, self.base_records, 1
        )
        self.c3_path = self._matrix_stage("c3-matrix", self.c3)
        base_digests = {
            candidate["candidate_sha256"] for candidate in self.base["candidates"]
        }
        c3_new = [
            candidate
            for candidate in self.c3["candidates"]
            if candidate["candidate_sha256"] not in base_digests
        ]
        self.c3_records, self.c3_bindings = self._screen_group(
            "c3-all", c3_new, self.c3
        )
        self.c4 = AUTOTUNE.extend_phase31_with_c4(
            self.c3, self.base_records + self.c3_records, 1
        )
        self.c4_path = self._matrix_stage("c4-matrix", self.c4)
        c3_digests = {
            candidate["candidate_sha256"] for candidate in self.c3["candidates"]
        }
        c4_new = [
            candidate
            for candidate in self.c4["candidates"]
            if candidate["candidate_sha256"] not in c3_digests
        ]
        self.c4_records, self.c4_bindings = self._screen_group(
            "c4-all", c4_new, self.c4
        )
        self.records = self.base_records + self.c3_records + self.c4_records
        self.bindings = {}
        for values in (self.base_bindings, self.c3_bindings, self.c4_bindings):
            self.bindings.update(values)
        portable = {
            "protocol": {
                "frames": 10,
                "warmup": 2,
                "trials": 2,
                "schedule": "round_robin",
                "seed": 0,
            },
            "measurement_source_bindings": VERIFY._portable_bindings(self.bindings),
        }
        screening_input_sha256 = VERIFY.sha256_json(
            portable, "tacker-phase31-full-screening-input-v1"
        )
        measurement = {
            "portable_hash_input": portable,
            "screening_input_sha256": screening_input_sha256,
            "actual_measurement_source_bindings": self.bindings,
        }
        measurement_path = self.write_json(self.run / "measurement-sources.json", measurement)
        self.ranking = AUTOTUNE.build_screening_ranking(
            self.c4,
            self.records,
            screening_input_sha256=screening_input_sha256,
        )
        ranking_path = self.write_json(self.run / "screening-ranking.json", self.ranking)
        ranking_result = {
            "ranking": self.ranking,
            "ranking_path": str(ranking_path.resolve()),
            "ranking_sha256": self.ranking["ranking_sha256"],
            "screening_input_sha256": screening_input_sha256,
            "measurement_sources": str(measurement_path.resolve()),
        }
        self.add_stage(
            "screening-ranking", ranking_result, [measurement_path, ranking_path]
        )

    def _build_profiles(self):
        self.final_profiles = {
            candidate["variant_id"]: self._profile(
                candidate, self.c4["matrix_sha256"]
            )
            for candidate in self.c4["candidates"]
        }
        profile_dir = self.run / "final-profiles"
        manifest = AUTOTUNE.publish_qualification_profiles(
            self.c4, self.final_profiles, profile_dir
        )
        manifest_path = profile_dir / "qualification_profiles.json"
        result = {
            "label": "final",
            "source_matrix_sha256": self.c4["matrix_sha256"],
            "matrix_path": str(self.c4_path.resolve()),
            "profiles_dir": str(profile_dir.resolve()),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": VERIFY.sha256_file(manifest_path),
            "candidate_count": len(self.c4["candidates"]),
        }
        paths = [manifest_path] + [
            profile_dir / "{}.json".format(candidate["variant_id"])
            for candidate in self.c4["candidates"]
        ]
        self.add_stage("profiles-final", result, paths)
        self.profile_dir = profile_dir
        self.profile_manifest = manifest

    def _dummy_stage(self, name, result):
        path = self.write_json(self.run / "{}-evidence.json".format(name), {"ok": True})
        self.add_stage(name, result, [path])

    def _quality_per_view(self):
        return [
            {
                "batch_index": index,
                "psnr_db": 40.0,
                "ssim": 0.99,
                "lpips": 0.01,
            }
            for index in range(50)
        ]

    def _quality_workload(self):
        workload = self.identity["payload"]["workload"]
        return {
            "scene": workload["name"],
            "iteration": workload["iteration"],
            "split": workload["split"],
            "frames": 50,
            "view_indices": list(range(50)),
            "resolution": [workload["image_width"], workload["image_height"]],
            "gaussian_count": workload["gaussian_count"],
            "model_path": self.identity["payload"]["paths"]["model"],
            "source_path": self.identity["payload"]["paths"]["source"],
        }

    def _build_qualification_and_baseline(self, formal_set):
        candidates = [item["candidate"] for item in formal_set["candidates"]]
        profile_by_digest = {
            item["candidate_sha256"]: self.facts(item["path"])
            for item in self.profile_manifest["profiles"]
        }
        attempted = []
        evidence = {}
        for candidate in candidates:
            digest = candidate["candidate_sha256"]
            profile = profile_by_digest[digest]
            directory = self.run / "qualification" / digest[:20]
            directory.mkdir(parents=True)
            leaf = {
                "schema_version": 2,
                "kind": "4dgaussians_tacker_leaf_profile_report",
                "passed": True,
                "workload": {
                    "scene": "flame_steak",
                    "iteration": 14000,
                    "split": "test",
                    "resolution": [1352, 1014],
                    "gaussian_count": 111525,
                },
                "variant_id": candidate["variant_id"],
                "profile_binding": {
                    "candidate_sha256": digest,
                    "candidate_matrix_sha256": self.c4["matrix_sha256"],
                    "candidate_matrix_file_sha256": VERIFY.sha256_file(self.c4_path),
                    "profile_file_sha256": profile["sha256"],
                },
                "parameters": {
                    "candidate_profile": profile["path"],
                    "qualification_profile": True,
                    "used_as_deployment": False,
                },
                "measurement_outputs_written": True,
            }
            leaf_path = self.write_json(directory / "leaf-report.json", leaf)
            quality = {
                "schema_version": 1,
                "kind": "4dgaussians_tacker_quality_validation",
                "passed": True,
                "workload": self._quality_workload(),
                "qualification": {
                    "enabled": True,
                    "admission_claimed": False,
                    "profile_override": profile["path"],
                },
                "modes": {
                    "serial": {"per_view": self._quality_per_view()},
                    "tacker": {
                        "actual_mode": "tacker",
                        "fallback_reason": None,
                        "qualification_executed": True,
                        "per_view": self._quality_per_view(),
                    },
                },
                "gates": [{"mode": "tacker", "passed": True}],
            }
            quality_path = self.write_json(directory / "quality.json", quality)
            result = {
                "schema_version": 1,
                "kind": "tacker_phase31_candidate_qualification",
                "matrix_sha256": self.c4["matrix_sha256"],
                "candidate_sha256": digest,
                "variant_id": candidate["variant_id"],
                "search_family": candidate["search_family"],
                "profile": profile,
                "leaf_report": self.facts(leaf_path),
                "quality_report": self.facts(quality_path),
                "leaf_passed": True,
                "quality_50_view_passed": True,
                "quality_actual_tacker_without_fallback": True,
                "quality_unique_requested_views": True,
                "quality_exact_per_view_records": True,
                "valid": True,
                "errors": [],
            }
            result_path = self.write_json(directory / "qualification.json", result)
            stage_result = {"result": result, "path": str(result_path.resolve())}
            self.add_stage(
                "qualify-{}".format(digest[:20]),
                stage_result,
                [result_path, leaf_path, quality_path],
            )
            attempted.append(
                {"candidate": candidate, "reason": "formal_set", "result": result}
            )
            evidence[digest] = self.facts(result_path)
        family_coverage = {
            family: [
                item["candidate_sha256"]
                for item in candidates
                if item["search_family"] == family
            ]
            for family in VERIFY.SEARCH_FAMILIES
        }
        hash_payload = {
            "schema_version": 1,
            "kind": "tacker_phase31_qualification_plan",
            "matrix_sha256": self.c4["matrix_sha256"],
            "screening_ranking_sha256": self.ranking["ranking_sha256"],
            "formal_candidate_set_sha256": self.formal_set["formal_set_sha256"],
            "autotune_candidate_set_sha256": formal_set["candidate_set_sha256"],
            "attempted": [
                {
                    "candidate_sha256": item["candidate"]["candidate_sha256"],
                    "search_family": item["candidate"]["search_family"],
                    "reason": item["reason"],
                    "valid": True,
                    "leaf_passed": True,
                    "quality_50_view_passed": True,
                    "qualification_evidence": VERIFY._portable_artifact(
                        evidence[item["candidate"]["candidate_sha256"]]
                    ),
                }
                for item in attempted
            ],
            "same_family_backfill": [],
            "valid_finalist_sha256": [
                item["candidate_sha256"] for item in candidates
            ],
            "valid_family_coverage": family_coverage,
        }
        plan = {
            "schema_version": 1,
            "kind": "tacker_phase31_qualification_plan",
            "matrix_sha256": self.c4["matrix_sha256"],
            "screening_ranking_sha256": self.ranking["ranking_sha256"],
            "formal_candidate_set_sha256": self.formal_set["formal_set_sha256"],
            "autotune_candidate_set_sha256": formal_set["candidate_set_sha256"],
            "attempted": attempted,
            "same_family_backfill": [],
            "valid_finalists": candidates,
            "valid_family_coverage": family_coverage,
            "portable_hash_input": hash_payload,
            "qualification_plan_sha256": VERIFY.sha256_json(
                hash_payload, "tacker-phase31-qualification-plan-v1"
            ),
        }
        plan_path = self.write_json(self.run / "qualification-plan.json", plan)
        self.qualification_result = {
            "document": plan,
            "path": str(plan_path.resolve()),
        }
        self.add_stage("qualification-plan", self.qualification_result, [plan_path])

        modes = {}
        gates = []
        for mode in ("serial", "two_stream", "tacker"):
            modes[mode] = {
                "requested_mode": mode,
                "actual_mode": mode,
                "fallback_reason": None,
                "qualification_requested": False,
                "qualification_executed": False,
                "per_view": self._quality_per_view(),
            }
            if mode != "serial":
                gates.append(
                    {
                        "mode": mode,
                        "actual_mode": mode,
                        "actual_mode_passed": True,
                        "qualification_passed": True,
                        "quality_passed": True,
                        "passed": True,
                    }
                )
        baseline_report = {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_quality_validation",
            "passed": True,
            "workload": self._quality_workload(),
            "modes": modes,
            "gates": gates,
            "errors": [],
            "qualification": {
                "enabled": False,
                "admission_claimed": True,
                "profile_override": None,
            },
            "tacker_profile": str(self.current.resolve()),
        }
        baseline_path = self.write_json(
            self.run / "baseline-quality.json", baseline_report
        )
        correctness = {
            "serial": {
                "valid": True,
                "actual_mode": "serial",
                "fallback_reason": None,
                "quality_report_sha256": VERIFY.sha256_file(baseline_path),
            },
            "two_stream": {
                "valid": True,
                "actual_mode": "two_stream",
                "fallback_reason": None,
                "quality_report_sha256": VERIFY.sha256_file(baseline_path),
            },
            "current_tacker": {
                "valid": True,
                "actual_mode": "tacker",
                "fallback_reason": None,
                "quality_report_sha256": VERIFY.sha256_file(baseline_path),
            },
        }
        correctness_path = self.write_json(
            self.run / "baseline-correctness.json", correctness
        )
        self.baseline_result = {
            "quality_report": str(baseline_path.resolve()),
            "correctness_path": str(correctness_path.resolve()),
            "correctness": correctness,
            "all_valid": True,
        }
        self.add_stage(
            "baseline-quality",
            self.baseline_result,
            [baseline_path, correctness_path],
        )

    def _build_formal_and_selection(self):
        self.formal_set = formal_wrapper(self.c4, self.ranking, 2)
        formal_set_path = self.write_json(
            self.run / "formal-candidate-set.json", self.formal_set
        )
        generated = self.formal_set["generated_candidate_set"]
        formal_set_result = {
            "formal_set": generated,
            "path": str(formal_set_path.resolve()),
            "candidate_set_sha256": generated["candidate_set_sha256"],
            "formal_set_sha256": self.formal_set["formal_set_sha256"],
        }
        self.add_stage("formal-candidate-set", formal_set_result, [formal_set_path])
        self._build_qualification_and_baseline(generated)

        formal_candidates = [item["candidate"] for item in generated["candidates"]]
        names = list(VERIFY.BASELINES) + [
            candidate["variant_id"] for candidate in formal_candidates
        ]
        qualifications = dict(self.baseline_result["correctness"])
        for candidate in formal_candidates:
            qualifications[candidate["variant_id"]] = {
                "valid": True,
                "candidate_sha256": candidate["candidate_sha256"],
                "qualification_plan_sha256": self.qualification_result["document"][
                    "qualification_plan_sha256"
                ],
            }
        profile_by_name = {
            item["variant_id"]: self.facts(item["path"])
            for item in self.profile_manifest["profiles"]
        }
        candidate_entries = [
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
                "profile_path": str(self.current.resolve()),
                "profile_file_sha256": VERIFY.sha256_file(self.current),
                "qualification_mode": False,
            },
        ]
        candidate_entries.extend(
            {
                "name": candidate["variant_id"],
                "execution_mode": "tacker",
                "profile_path": profile_by_name[candidate["variant_id"]]["path"],
                "profile_file_sha256": profile_by_name[candidate["variant_id"]]["sha256"],
                "qualification_mode": True,
            }
            for candidate in formal_candidates
        )
        modes = {
            item["name"]: item["execution_mode"] for item in candidate_entries
        }
        metadata = BENCHMARK.selection_metadata_with_defaults(
            names, {}, execution_modes=modes
        )
        contract = BENCHMARK.make_contract(
            self.model,
            self.source,
            "flame_steak",
            14000,
            "test",
            10,
            50,
            1352,
            1014,
            111525,
            view_indices=list(range(50)),
        )
        schedule = BENCHMARK.build_schedule(names, 10, "abba", seed=0)
        formal_correctness = self.write_json(
            self.run / "formal-correctness.json", qualifications
        )
        selection_inputs = {
            "correctness_json": {
                "path": str(formal_correctness.resolve()),
                "sha256": VERIFY.sha256_file(formal_correctness),
            }
        }
        resume_identity = BENCHMARK.benchmark_resume_identity(
            candidate_entries,
            contract,
            qualifications,
            metadata,
            schedule,
            "abba",
            0,
            10,
            100,
            "current_tacker",
            BENCHMARK.DEFAULT_PROMOTION_MIN_RATIO,
            BENCHMARK.DEFAULT_EQUIVALENCE_FRACTION,
            ROOT / "profile_render.py",
            sys.executable,
            ROOT,
            configs=self.config,
            profile_args=[],
            selection_inputs=selection_inputs,
        )
        required_sources = resume_identity["payload"]["required_provenance_sources"]
        source_files = {}
        for source_name, source_record in required_sources.items():
            if source_name == "diff_gaussian_rasterization._C":
                source_files[source_name] = source_record[0]["sha256"]
            else:
                source_files[source_name] = source_record["sha256"]
        formal_run_root = (
            self.run / "sessions" / "formal" / "runs" / "phase31-formal"
        )
        formal_run_root.mkdir(parents=True)
        runs = []
        validated_runs = []
        raw_paths = []
        by_name = {item["name"]: item for item in candidate_entries}
        base_fps = {
            "serial": 80.0,
            "two_stream": 85.0,
            "current_tacker": 100.0,
        }
        for schedule_item in schedule:
            candidate = by_name[schedule_item["candidate_name"]]
            name = candidate["name"]
            fps = base_fps.get(name, 90.0 - names.index(name) * 0.01)
            record = BENCHMARK._make_run_record(
                schedule_item,
                candidate,
                formal_run_root,
                ROOT / "profile_render.py",
                Path(sys.executable).resolve(),
                contract,
                str(self.config.resolve()),
                [],
            )
            child = BENCHMARK_FIXTURES.synthetic_metadata(
                candidate, contract, fps
            )
            child["source_files"] = dict(source_files)
            child["profile_render_sha256"] = source_files["profile_render.py"]
            child["repository"]["source_files"] = dict(source_files)
            child["repository"]["commit"] = "a" * 40
            child["repository"]["commit_source"] = "test"
            child["gpu_name"] = "NVIDIA RTX A6000"
            child["environment"]["gpu"]["name"] = "NVIDIA RTX A6000"
            if name not in VERIFY.BASELINES:
                child["pipeline_execution_counts"] = dict(
                    VERIFY.EXPECTED_FORMAL_EXECUTION_COUNTS
                )
            metadata_path = self.write_json(record["metadata_path"], child)
            Path(record["stdout_path"]).write_text("ok\n", encoding="utf-8")
            Path(record["stderr_path"]).write_text("", encoding="utf-8")
            raw_paths.extend(
                [metadata_path, Path(record["stdout_path"]), Path(record["stderr_path"])]
            )
            metrics = BENCHMARK.validate_child_metadata(
                child, candidate, contract
            )
            record.update(
                {
                    "passed": True,
                    "returncode": 0,
                    "error": None,
                    "stdout_sha256": VERIFY.sha256_file(record["stdout_path"]),
                    "stderr_sha256": VERIFY.sha256_file(record["stderr_path"]),
                    "metadata_sha256": VERIFY.sha256_file(metadata_path),
                    "metrics": metrics,
                }
            )
            runs.append(record)
            if name not in VERIFY.BASELINES:
                validated_runs.append(
                    {
                        "candidate_name": name,
                        "round_index": schedule_item["round_index"],
                        "metadata": self.facts(metadata_path),
                        "pipeline_execution_counts": dict(
                            VERIFY.EXPECTED_FORMAL_EXECUTION_COUNTS
                        ),
                    }
                )
        summaries, ranking, paired_comparisons = BENCHMARK.aggregate_runs(
            runs, names, 10, 100, 0
        )
        finalist_order = {
            candidate["variant_id"]: index
            for index, candidate in enumerate(formal_candidates)
        }
        validated_runs.sort(
            key=lambda item: (
                finalist_order[item["candidate_name"]], item["round_index"]
            )
        )
        selection = BENCHMARK.select_candidates(
            summaries,
            candidate_qualifications=qualifications,
            candidate_selection_metadata=metadata,
            candidate_names=names,
            bootstrap_resamples=100,
            seed=0,
        )
        stable_environment = runs[0]["metrics"]["stable_environment"]
        stable_provenance = runs[0]["metrics"]["stable_provenance"]
        checkpoint_path = formal_run_root / BENCHMARK.CHECKPOINT_FILE_NAME
        self.write_json(
            checkpoint_path,
            {
                "schema_version": 1,
                "kind": BENCHMARK.CHECKPOINT_KIND,
                "resume_identity": resume_identity,
            },
        )
        raw_paths.append(checkpoint_path)
        formal_report = {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_fps_benchmark",
            "passed": True,
            "selection_objective": BENCHMARK.SELECTION_OBJECTIVE,
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "phase0_exit_condition": {
                "required_trials_per_candidate": 10,
                "required_frames_per_trial": 50,
                "met": True,
            },
            "contract": contract,
            "candidates": candidate_entries,
            "eligible_candidates": names,
            "excluded_candidates": [],
            "summaries": summaries,
            "ranking": ranking,
            "runs": runs,
            "paired_comparisons": paired_comparisons,
            "correctness_qualifications": qualifications,
            "candidate_selection_metadata": metadata,
            "selection_inputs": selection_inputs,
            "bootstrap": {
                "confidence": 0.95,
                "resamples": 100,
                "seed": 0,
                "resampling_unit": "paired_round",
            },
            "schedule": {
                "seed": 0,
                "strategy": "abba",
                "trials_per_candidate": 10,
                "base_order": BENCHMARK._stable_order(names, 0),
                "executions": schedule,
            },
            "artifacts": {
                "session_dir": str(formal_run_root),
                "driver_sha256": VERIFY.sha256_file(VERIFY.BENCHMARK_PATH),
                "profile_render_sha256": VERIFY.sha256_file(
                    ROOT / "profile_render.py"
                ),
                "checkpoint_path": str(checkpoint_path),
                "report_path": str((self.run / "formal-fps.json").resolve()),
            },
            "resume_identity": resume_identity,
            "resume": {"enabled": False, "recovered_execution_count": 0},
            "expected_execution_count": len(schedule),
            "completed_execution_count": len(schedule),
            "stable_environment": stable_environment,
            "stable_provenance": stable_provenance,
            "selection": selection,
            "eligible_ranking": selection["eligible_ranking"],
            "experimental_winner": selection["experimental_winner"],
            "deployment_winner": selection["deployment_winner"],
            "promotion": selection["promotion"],
            "errors": [],
        }
        formal_path = self.write_json(self.run / "formal-fps.json", formal_report)
        integrity = {
            "schema_version": 1,
            "kind": "tacker_phase31_formal_execution_integrity",
            "protocol": {
                "frames_per_sequence": 50,
                "trials_per_candidate": 10,
                "schedule": "abba",
            },
            "expected_pipeline_execution_counts": dict(
                VERIFY.EXPECTED_FORMAL_EXECUTION_COUNTS
            ),
            "finalist_names": [
                candidate["variant_id"] for candidate in formal_candidates
            ],
            "validated_runs": validated_runs,
            "claim_scope": (
                "Each finalist child sequence executed Tacker without fallback "
                "and reported exactly one prefill, 49 mixed steps, one drain, "
                "50 outputs, and 50 selected-head evaluations per head."
            ),
        }
        integrity_path = self.write_json(
            self.run / "formal-execution-integrity.json", integrity
        )
        formal_result = {
            "fps_report": str(formal_path.resolve()),
            "correctness": str(formal_correctness.resolve()),
            "deployment_winner": selection["deployment_winner"],
            "formal_candidate_names": [candidate["variant_id"] for candidate in formal_candidates],
            "c3_real_50_view_candidates": [
                candidate["variant_id"]
                for candidate in formal_candidates
                if candidate["search_family"] == "c3"
            ],
            "c4_real_50_view_candidates": [
                candidate["variant_id"]
                for candidate in formal_candidates
                if candidate["search_family"] == "c4"
            ],
            "execution_integrity": {
                "no_fallback_and_exact_scheduler_counts_validated": True,
                "artifact": self.facts(integrity_path),
                "scope": "finalist_tacker_children_10x50",
            },
        }
        self.add_stage(
            "formal-benchmark",
            formal_result,
            [formal_path, formal_correctness, integrity_path] + raw_paths,
        )

        reused = self.facts(self.current)
        selection_payload = {
            "schema_version": 1,
            "kind": VERIFY.SELECTION_KIND,
            "matrix_sha256": self.c4["matrix_sha256"],
            "formal_report_sha256": VERIFY.sha256_file(formal_path),
            "deployment_winner": selection["deployment_winner"],
            "winner_is_new_challenger": False,
            "disabled_winner_qualification_profile": None,
            "reused_incumbent_profile": reused,
            "baseline_winner_has_no_synthetic_profile": True,
        }
        portable = dict(selection_payload)
        portable["reused_incumbent_profile"] = VERIFY._portable_artifact(reused)
        selection_document = dict(selection_payload)
        selection_document["portable_hash_input"] = portable
        selection_document["selection_sha256"] = VERIFY.sha256_json(
            portable, "tacker-phase31-selection-v1"
        )
        selection_path = self.write_json(self.run / "selection.json", selection_document)
        selection_result = {
            "document": selection_document,
            "path": str(selection_path.resolve()),
        }
        self.add_stage("selection", selection_result, [selection_path])
        selected = TOP3.select_top_candidates(formal_report, limit=3)
        top3_identity = TOP3._identity(
            formal_path,
            VERIFY.sha256_file(formal_path),
            selected,
            ROOT / "scripts" / "profile_nsight.sh",
            self.model,
            self.config,
            self.source,
            1,
            50,
            14000,
            "flame_steak",
        )
        expected_summary = TOP3._expected_summary_contract(
            formal_report, top3_identity, selected
        )
        output_root = self.run / "sessions" / "nsight" / "profiles"
        output_root.mkdir(parents=True)
        nsight_profiles = []
        nsight_artifacts = []
        for candidate in selected:
            record = TOP3._make_profile_record(
                candidate,
                output_root,
                ROOT / "scripts" / "profile_nsight.sh",
                self.model,
                self.config,
                self.source,
                1,
                50,
                14000,
                "flame_steak",
            )
            directory = Path(record["output_dir"])
            directory.mkdir(parents=True)
            runtime = expected_summary["candidate_runtime_identities"][
                candidate["name"]
            ]
            is_tacker = candidate["execution_mode"] == "tacker"
            qualification = bool(candidate["qualification_mode"])
            active_hash = candidate["profile_file_sha256"]
            tacker_hash = active_hash if is_tacker and not qualification else None
            qualification_hash = active_hash if qualification else None
            metadata_document = {
                "schema_version": 1,
                "kind": TOP3.CHILD_KIND,
                "passed": True,
                "model_path": str(self.model.resolve()),
                "source_path": str(self.source.resolve()),
                "iteration": 14000,
                "split": "test",
                "warmup_frames": 10,
                "profile_frames": 50,
                "execution_mode": candidate["execution_mode"],
                "actual_execution_mode": candidate["execution_mode"],
                "qualification_mode_requested": qualification,
                "qualification_mode_executed": qualification,
                "workload_name": "flame_steak" if is_tacker else None,
                "tacker_profile": (
                    candidate["profile_path"]
                    if is_tacker and not qualification
                    else None
                ),
                "qualification_profile": (
                    candidate["profile_path"] if qualification else None
                ),
                "active_profile_sha256": active_hash,
                "tacker_profile_sha256": tacker_hash,
                "qualification_profile_sha256": qualification_hash,
                "profile_manifest_sha256": runtime["profile_manifest_sha256"],
                "profile_selection_sha256": runtime["profile_selection_sha256"],
                "selected_variant_id": runtime["selected_variant_id"],
                "selected_candidate_abi_sha256": runtime[
                    "selected_candidate_abi_sha256"
                ],
                "persistent_blocks": runtime["persistent_blocks"],
                "two_stream_fallback_reason": None,
                "tacker_fallback_reason": None,
                "profile_hashes": {
                    "active_profile_sha256": active_hash,
                    "tacker_profile_sha256": tacker_hash,
                    "qualification_profile_sha256": qualification_hash,
                    "profile_manifest_sha256": runtime[
                        "profile_manifest_sha256"
                    ],
                    "profile_selection_sha256": runtime[
                        "profile_selection_sha256"
                    ],
                    "selected_candidate_abi_sha256": runtime[
                        "selected_candidate_abi_sha256"
                    ],
                },
                "source_files": dict(source_files),
                "repository": {
                    "commit": "a" * 40,
                    "commit_source": "test",
                    "dirty": False,
                    "submodules": [],
                    "source_files": dict(source_files),
                },
                "gpu_name": "NVIDIA RTX A6000",
                "cuda_runtime": "12.4",
                "pytorch_version": "2.4.1",
            }
            mode = candidate["execution_mode"]
            metadata_path = self.write_json(
                directory / "{}_profile_metadata.json".format(mode),
                metadata_document,
            )
            (directory / "4dgs_render_{}.nsys-rep".format(mode)).write_bytes(
                b"synthetic nsys"
            )
            stats_path = directory / "4dgs_render_{}_stats.csv".format(mode)
            stats_path.write_text(synthetic_nsight_stats_csv(), encoding="utf-8")
            summary_document = SUMMARIZER.summarize(stats_path, metadata_path)
            summary_path = self.write_json(record["summary_path"], summary_document)
            stdout_path = Path(record["stdout_path"])
            stderr_path = Path(record["stderr_path"])
            stdout_path.write_text("ok\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            diagnostics = TOP3.extract_nsight_diagnostics(
                summary_document, candidate, expected_summary
            )
            raw_artifacts = TOP3._validate_raw_artifacts(directory, candidate)
            record.update(
                {
                    "passed": True,
                    "returncode": 0,
                    "error": None,
                    "stdout_sha256": VERIFY.sha256_file(stdout_path),
                    "stderr_sha256": VERIFY.sha256_file(stderr_path),
                    "summary_sha256": VERIFY.sha256_file(summary_path),
                    "profile_metadata_sha256": VERIFY.sha256_file(metadata_path),
                    "raw_artifacts": raw_artifacts,
                    "diagnostics": diagnostics,
                }
            )
            nsight_profiles.append(record)
            nsight_artifacts.extend(
                path for path in directory.rglob("*") if path.is_file()
            )
        baseline_diagnostics = nsight_profiles[0]["diagnostics"]
        rank1_comparisons = []
        for record in nsight_profiles:
            item = record["diagnostics"]
            rank1_comparisons.append(
                {
                    "candidate": record["candidate"]["name"],
                    "rank1_candidate": nsight_profiles[0]["candidate"]["name"],
                    "nsight_fps_ratio": (
                        item["nsight_render_fps"]
                        / baseline_diagnostics["nsight_render_fps"]
                    ),
                    "kernel_launches_per_frame_delta": (
                        item["kernel_launches"]["per_frame"]
                        - baseline_diagnostics["kernel_launches"]["per_frame"]
                    ),
                }
            )
        nsight_report = {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_top3_nsight",
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "passed": True,
            "identity": top3_identity,
            "fps_report_sha256": VERIFY.sha256_file(formal_path),
            "selection_policy": (
                "first N candidates in formal eligible_ranking, including baselines"
            ),
            "requested_limit": 3,
            "selected_candidates": selected,
            "resume": {"enabled": False, "recovered_count": 0},
            "profiles": nsight_profiles,
            "rank1_comparisons": rank1_comparisons,
            "errors": [],
        }
        checkpoint_path = self.write_json(
            output_root / "top3.checkpoint.json",
            {
                "schema_version": 1,
                "kind": TOP3.CHECKPOINT_KIND,
                "identity": top3_identity,
                "report": nsight_report,
            },
        )
        nsight_artifacts.append(checkpoint_path)
        nsight_path = self.write_json(
            self.run / "sessions" / "nsight" / "top3-nsight.json",
            nsight_report,
        )
        self.nsight_result = {"report": str(nsight_path.resolve())}
        self.add_stage(
            "top3-nsight", self.nsight_result, [nsight_path] + nsight_artifacts
        )

        self.formal_result = formal_result
        self.selection_result = selection_result
        self.formal_set_result = formal_set_result

    def _build_report_and_state(self):
        ranking_result = next(
            stage["result"] for stage in self.stages if stage["name"] == "screening-ranking"
        )
        report = {
            "schema_version": 1,
            "kind": VERIFY.RUN_REPORT_KIND,
            "passed": True,
            "identity": self.identity,
            "matrix": {
                "path": str(self.c4_path.resolve()),
                "matrix_sha256": self.c4["matrix_sha256"],
                "candidate_count": len(self.c4["candidates"]),
                "exhaustive_c2_count": 450,
            },
            "screening": {
                "ranking": ranking_result["ranking_path"],
                "ranking_sha256": ranking_result["ranking_sha256"],
                "measurement_sources": ranking_result["measurement_sources"],
            },
            "formal_candidate_set": {
                "path": self.formal_set_result["path"],
                "candidate_set_sha256": self.formal_set_result["candidate_set_sha256"],
                "formal_set_sha256": self.formal_set_result["formal_set_sha256"],
            },
            "qualification": self.qualification_result,
            "baseline": self.baseline_result,
            "formal": self.formal_result,
            "selection": self.selection_result,
            "nsight": self.nsight_result,
            "checkpoint": str((self.run / "phase31-state.json").resolve()),
        }
        self.report_path = self.write_json(self.run / "phase31-report.json", report)
        state = {
            "schema_version": 1,
            "kind": VERIFY.STATE_KIND,
            "status": "succeeded",
            "identity": self.identity,
            "stages": self.stages,
            "report": self.facts(self.report_path),
        }
        self.state_path = self.write_json(self.run / "phase31-state.json", state)

    def materialize(self, matrix_path, resources, template, output_dir):
        matrix = AUTOTUNE.load_json_file(matrix_path)
        AUTOTUNE.publish_qualification_profiles(
            matrix, self.final_profiles, output_dir
        )


class VerifyPhase31Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = CompletedRunFixture(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_completed_fixture_replays_every_hash_chain(self):
        replay_inputs = []

        def materialize(matrix_path, resources, template, output_dir):
            replay_inputs.append(
                (Path(matrix_path), Path(resources), Path(template), Path(output_dir))
            )
            self.fixture.materialize(matrix_path, resources, template, output_dir)

        report = VERIFY.verify_phase31(
            self.fixture.run,
            python_executable=sys.executable,
            profile_materializer=materialize,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["checks"]["matrices"]["c2_candidate_count"], 450)
        self.assertEqual(
            report["checks"]["qualification_profiles"]["profile_hash_mismatch_count"],
            0,
        )
        self.assertEqual(
            report["checks"]["selection"]["deployment_winner"],
            "current_tacker",
        )
        self.assertGreater(
            report["checks"]["formal_execution"][
                "validated_50_view_sequence_count"
            ],
            0,
        )
        self.assertGreaterEqual(
            report["checks"]["formal_execution"][
                "c3_real_50_view_candidate_count"
            ],
            1,
        )
        self.assertGreaterEqual(
            report["checks"]["formal_execution"][
                "c4_real_50_view_candidate_count"
            ],
            1,
        )
        self.assertEqual(len(replay_inputs), 1)
        original_inputs = {
            self.fixture.c4_path.resolve(),
            self.fixture.template.resolve(),
            Path(
                next(
                    stage["result"]["resources"]
                    for stage in self.fixture.stages
                    if stage["name"] == "preflight"
                )
            ),
        }
        self.assertTrue(
            all(path not in original_inputs for path in replay_inputs[0][:3])
        )
        self.assertTrue(all(not path.exists() for path in replay_inputs[0]))

        formal = VERIFY.load_json(self.fixture.formal_result["fps_report"])
        wrong_gpu = deepcopy(formal["stable_provenance"])
        wrong_gpu["environment"]["gpu_name"] = "NVIDIA H100"
        with self.assertRaisesRegex(
            VERIFY.VerificationError, "stable provenance is not the A6000"
        ):
            VERIFY._verify_stable_provenance(
                wrong_gpu,
                self.fixture.identity,
                formal["resume_identity"]["payload"][
                    "required_provenance_sources"
                ],
            )

    def test_mutated_succeeded_stage_artifact_is_rejected(self):
        VERIFY.atomic_write_json(self.fixture.base_path, {"tampered": True})
        with self.assertRaisesRegex(VERIFY.VerificationError, "artifact bytes changed"):
            VERIFY.verify_phase31(
                self.fixture.run, profile_materializer=self.fixture.materialize
            )

    def test_profile_replay_mismatch_is_rejected(self):
        def mismatching(matrix_path, resources, template, output_dir):
            matrix = AUTOTUNE.load_json_file(matrix_path)
            profiles = dict(self.fixture.final_profiles)
            first = matrix["candidates"][0]
            changed = dict(profiles[first["variant_id"]])
            changed["note"] = "changed"
            profiles[first["variant_id"]] = changed
            AUTOTUNE.publish_qualification_profiles(matrix, profiles, output_dir)

        with self.assertRaisesRegex(VERIFY.VerificationError, "profile hashes differ"):
            VERIFY.verify_phase31(
                self.fixture.run, profile_materializer=mismatching
            )

    def test_path_escape_in_stage_ledger_is_rejected(self):
        state = VERIFY.load_json(self.fixture.state_path)
        state["stages"][0]["artifacts"][0] = self.fixture.facts(
            self.fixture.seed_file
        )
        VERIFY.atomic_write_json(self.fixture.state_path, state)
        with self.assertRaisesRegex(VERIFY.VerificationError, "escapes"):
            VERIFY.verify_phase31(
                self.fixture.run, profile_materializer=self.fixture.materialize
            )

    def test_failed_screening_record_must_bind_nonempty_report(self):
        state = VERIFY.load_json(self.fixture.state_path)
        stage = next(
            item for item in state["stages"] if item["name"] == "screen-base-all"
        )
        digest = next(iter(stage["result"]["measurement_source_bindings"]))
        binding = stage["result"]["measurement_source_bindings"][digest]
        record = next(
            item
            for item in stage["result"]["records"]
            if item["candidate_sha256"] == digest
        )
        record.update(
            {
                "status": "failed",
                "score": None,
                "error": "sealed failure",
                "artifact_sha256": binding["profile"]["sha256"],
            }
        )
        batch_path = Path(stage["result"]["path"])
        batch = VERIFY.load_json(batch_path)
        batch["records"] = stage["result"]["records"]
        VERIFY.atomic_write_json(batch_path, batch)
        for index, artifact in enumerate(stage["artifacts"]):
            if artifact["path"] == str(batch_path.resolve()):
                stage["artifacts"][index] = self.fixture.facts(batch_path)
        VERIFY.atomic_write_json(self.fixture.state_path, state)
        with self.assertRaisesRegex(
            VERIFY.VerificationError, "terminal record evidence hash"
        ):
            VERIFY.verify_phase31(
                self.fixture.run, profile_materializer=self.fixture.materialize
            )

    def test_replay_code_must_match_identity_sealed_modules(self):
        identity = json.loads(json.dumps(self.fixture.identity))
        identity["payload"]["scripts"]["autotune"] = self.fixture.facts(
            self.fixture.seed_file
        )
        identity["sha256"] = VERIFY.sha256_json(
            identity["payload"], "tacker-phase31-identity-v1"
        )
        with self.assertRaisesRegex(
            VERIFY.VerificationError, "identity-sealed source"
        ):
            VERIFY._identity_paths(identity)

    def test_identity_input_symlink_replacement_is_rejected(self):
        replacement = self.fixture.inputs / "replacement.txt"
        replacement.write_text("identity input\n", encoding="utf-8")
        self.fixture.seed_file.unlink()
        os.symlink(str(replacement), str(self.fixture.seed_file))
        with self.assertRaisesRegex(VERIFY.VerificationError, "symbolic link"):
            VERIFY._identity_paths(self.fixture.identity)

    def test_resealed_formal_metadata_semantic_tamper_is_rejected(self):
        formal_stage_name = "formal-benchmark"
        state = VERIFY.load_json(self.fixture.state_path)
        stage = next(
            item for item in state["stages"] if item["name"] == formal_stage_name
        )
        formal_path = Path(stage["result"]["fps_report"])
        formal = VERIFY.load_json(formal_path)
        metadata_path = Path(formal["runs"][0]["metadata_path"])
        metadata = VERIFY.load_json(metadata_path)
        metadata["actual_execution_mode"] = "serial"
        VERIFY.atomic_write_json(metadata_path, metadata)
        formal["runs"][0]["metadata_sha256"] = VERIFY.sha256_file(metadata_path)
        VERIFY.atomic_write_json(formal_path, formal)
        replacements = {
            str(metadata_path.resolve()): self.fixture.facts(metadata_path),
            str(formal_path.resolve()): self.fixture.facts(formal_path),
        }
        stage["artifacts"] = [
            replacements.get(item["path"], item) for item in stage["artifacts"]
        ]
        VERIFY.atomic_write_json(self.fixture.state_path, state)
        with self.assertRaisesRegex(
            VERIFY.VerificationError, "metadata validation failed"
        ):
            VERIFY.verify_phase31(
                self.fixture.run, profile_materializer=self.fixture.materialize
            )

    def test_resealed_formal_summary_cannot_override_raw_run_aggregate(self):
        state = VERIFY.load_json(self.fixture.state_path)
        stage = next(
            item for item in state["stages"] if item["name"] == "formal-benchmark"
        )
        formal_path = Path(stage["result"]["fps_report"])
        formal = VERIFY.load_json(formal_path)
        name = formal["eligible_ranking"][0]
        formal["summaries"][name]["median_throughput_fps"] += 1.0
        VERIFY.atomic_write_json(formal_path, formal)
        stage["artifacts"] = [
            self.fixture.facts(formal_path)
            if item["path"] == str(formal_path)
            else item
            for item in stage["artifacts"]
        ]
        VERIFY.atomic_write_json(self.fixture.state_path, state)
        with self.assertRaisesRegex(
            VERIFY.VerificationError, "aggregate/ranking/CI replay differs"
        ):
            VERIFY.verify_phase31(
                self.fixture.run, profile_materializer=self.fixture.materialize
            )

    def test_resealed_nsight_diagnostics_cannot_override_raw_summary(self):
        state = VERIFY.load_json(self.fixture.state_path)
        stage = next(
            item for item in state["stages"] if item["name"] == "top3-nsight"
        )
        report_path = Path(stage["result"]["report"])
        report = VERIFY.load_json(report_path)
        report["profiles"][0]["diagnostics"]["nsight_render_fps"] += 1.0
        VERIFY.atomic_write_json(report_path, report)
        stage["artifacts"] = [
            self.fixture.facts(report_path)
            if item["path"] == str(report_path)
            else item
            for item in stage["artifacts"]
        ]
        VERIFY.atomic_write_json(self.fixture.state_path, state)
        with self.assertRaisesRegex(
            VERIFY.VerificationError, "diagnostics replay differs"
        ):
            VERIFY.verify_phase31(
                self.fixture.run, profile_materializer=self.fixture.materialize
            )

    def test_resealed_nsight_stats_cannot_override_sealed_summary(self):
        state = VERIFY.load_json(self.fixture.state_path)
        stage = next(
            item for item in state["stages"] if item["name"] == "top3-nsight"
        )
        report_path = Path(stage["result"]["report"])
        report = VERIFY.load_json(report_path)
        profile = report["profiles"][0]
        mode = profile["candidate"]["execution_mode"]
        stats_path = Path(profile["output_dir"]) / (
            "4dgs_render_{}_stats.csv".format(mode)
        )
        stats = stats_path.read_text(encoding="utf-8")
        self.assertIn("profile/render_loop,500000000,1", stats)
        stats_path.write_text(
            stats.replace(
                "profile/render_loop,500000000,1",
                "profile/render_loop,510000000,1",
                1,
            ),
            encoding="utf-8",
        )
        stats_facts = self.fixture.facts(stats_path)
        profile["raw_artifacts"][stats_path.name] = {
            "sha256": stats_facts["sha256"],
            "size_bytes": stats_facts["size_bytes"],
        }
        VERIFY.atomic_write_json(report_path, report)
        checkpoint_path = (
            Path(profile["output_dir"]).parent / "top3.checkpoint.json"
        )
        checkpoint = VERIFY.load_json(checkpoint_path)
        checkpoint["report"] = report
        VERIFY.atomic_write_json(checkpoint_path, checkpoint)
        replacements = {
            str(stats_path.resolve()): self.fixture.facts(stats_path),
            str(report_path.resolve()): self.fixture.facts(report_path),
            str(checkpoint_path.resolve()): self.fixture.facts(checkpoint_path),
        }
        stage["artifacts"] = [
            replacements.get(item["path"], item) for item in stage["artifacts"]
        ]
        VERIFY.atomic_write_json(self.fixture.state_path, state)
        with self.assertRaisesRegex(
            VERIFY.VerificationError, "summary reconstruction differs"
        ):
            VERIFY.verify_phase31(
                self.fixture.run, profile_materializer=self.fixture.materialize
            )

    def test_formal_set_hash_covers_baselines(self):
        original = self.fixture.formal_set
        payload = dict(original)
        observed = payload.pop("formal_set_sha256")
        self.assertEqual(
            observed,
            VERIFY.sha256_json(payload, "tacker-phase31-formal-set-v1"),
        )
        payload["baseline_candidates"] = ["serial"]
        self.assertNotEqual(
            observed,
            VERIFY.sha256_json(payload, "tacker-phase31-formal-set-v1"),
        )

    def test_cli_writes_failed_report_atomically_outside_run_root(self):
        VERIFY.atomic_write_json(self.fixture.base_path, {"tampered": True})
        output = self.fixture.root / "postflight.json"
        returncode = VERIFY.main(
            [
                "--run-root",
                str(self.fixture.run),
                "--output",
                str(output),
                "--python-executable",
                sys.executable,
            ]
        )
        self.assertEqual(returncode, 1)
        document = VERIFY.load_json(output)
        self.assertFalse(document["passed"])
        self.assertEqual(document["kind"], VERIFY.VERIFY_REPORT_KIND)

    def test_default_profile_replay_uses_safe_argv(self):
        calls = []
        preflight_resources = Path(
            next(
                stage["result"]["resources"]
                for stage in self.fixture.stages
                if stage["name"] == "preflight"
            )
        )

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            matrix_path = Path(argv[argv.index("--matrix") + 1])
            resources_path = Path(argv[argv.index("--resources") + 1])
            template_path = Path(argv[argv.index("--template") + 1])
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            self.assertNotEqual(Path(argv[1]), VERIFY.AUTOTUNE_PATH)
            self.assertEqual(
                Path(argv[1]).read_bytes(), VERIFY.AUTOTUNE_PATH.read_bytes()
            )
            self.assertNotEqual(matrix_path, self.fixture.c4_path)
            self.assertEqual(matrix_path.read_bytes(), self.fixture.c4_path.read_bytes())
            self.assertNotEqual(resources_path, preflight_resources)
            self.assertEqual(
                resources_path.read_bytes(), preflight_resources.read_bytes()
            )
            self.assertNotEqual(template_path, self.fixture.template)
            self.assertEqual(template_path.read_bytes(), self.fixture.template.read_bytes())
            self.fixture.materialize(
                matrix_path, resources_path, template_path, output_dir
            )
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        report = VERIFY.verify_phase31(
            self.fixture.run,
            python_executable=sys.executable,
            runner=runner,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0][0], list)
        self.assertIs(calls[0][1]["shell"], False)


class ExternalVerifierInvocationTests(unittest.TestCase):
    def test_run_root_symlink_is_rejected_lexically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real-run"
            real.mkdir()
            alias = root / "run-link"
            os.symlink(str(real), str(alias))
            with self.assertRaisesRegex(VERIFY.VerificationError, "symbolic link"):
                VERIFY.verify_phase31(alias)

    def test_same_fd_snapshot_detects_atomic_path_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "state.json"
            replacement = root / "replacement.json"
            VERIFY.atomic_write_json(target, {"value": 1})
            VERIFY.atomic_write_json(replacement, {"value": 1})
            replaced = [False]

            def hook(path):
                if path == target and not replaced[0]:
                    replaced[0] = True
                    os.replace(str(replacement), str(target))

            store = VERIFY.SnapshotStore(after_read_hook=hook)
            with self.assertRaisesRegex(
                VERIFY.VerificationError, "changed while being snapshotted"
            ):
                store.json(target, "checkpoint")

    def test_subprocess_input_revalidation_detects_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "python"
            replacement = root / "replacement"
            target.write_bytes(b"sealed executable")
            replacement.write_bytes(b"replacement executable")
            store = VERIFY.SnapshotStore()
            store.snapshot(target, "Python executable")
            os.replace(str(replacement), str(target))
            with self.assertRaisesRegex(
                VERIFY.VerificationError, "changed across replay"
            ):
                store.revalidate(target, "Python executable")

    def test_external_script_uses_explicit_clean_project_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "verify_tacker_phase31.py"
            shutil.copy2(str(MODULE_PATH), str(copied))
            environment = os.environ.copy()
            environment[VERIFY.PROJECT_ROOT_ENV] = str(ROOT)
            completed = subprocess.run(
                [sys.executable, str(copied), "--help"],
                cwd=temporary,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(VERIFY.PROJECT_ROOT_ENV, completed.stdout)


if __name__ == "__main__":
    unittest.main()
