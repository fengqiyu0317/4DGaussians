"""CPU contracts for the Phase-3 experiment coordinator."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_tacker_phase3.py"
SPEC = importlib.util.spec_from_file_location("run_tacker_phase3", MODULE_PATH)
PHASE3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHASE3)


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def valid_resource_query():
    def raster(abi_version, worker_groups, physical_threads, active_blocks):
        return {
            "abi_version": abi_version,
            "worker_groups": worker_groups,
            "physical_threads": physical_threads,
            "device_ordinal": 0,
            "compute_capability_major": 8,
            "compute_capability_minor": 6,
            "multiprocessor_count": 84,
            "device_max_threads_per_block": 1024,
            "device_max_threads_per_multiprocessor": 1536,
            "kernel_max_threads_per_block": 1024,
            "registers_per_thread": 68,
            "static_shared_bytes": 7376,
            "local_bytes_per_thread": 0,
            "max_dynamic_shared_bytes": 0,
            "active_blocks_per_multiprocessor": active_blocks,
            "occupancy": active_blocks * physical_threads / 1536.0,
            "launch_supported": True,
        }

    head_facts = {
        "registers_per_thread": 48,
        "static_shared_memory_bytes": 0,
        "local_memory_bytes": 200,
        "max_threads_per_block": 640,
        "ptx_version": 86,
        "binary_version": 86,
        "worker_group_threads": [128, 256, 384, 512, 640],
        "active_blocks_per_sm": [10, 5, 3, 2, 2],
    }
    return {
        "schema_version": 1,
        "kind": "tacker_phase3_a6000_resource_query",
        "passed": True,
        "device": {
            "index": 0,
            "name": "NVIDIA RTX A6000",
            "compute_capability": [8, 6],
            "sm_count": 84,
            "cuda_runtime": "12.4",
            "pytorch_version": "2.4.1",
        },
        "families": {
            "legacy_pos_l1_v1": {
                "worker_groups": {"1": raster(1, 1, 384, 4)}
            },
            "first_linear_heads_v2": {
                "worker_groups": {
                    str(worker_groups): raster(
                        2,
                        worker_groups,
                        256 + 128 * worker_groups,
                        2 if worker_groups == 1 else 1,
                    )
                    for worker_groups in range(1, 6)
                }
            },
        },
        "head_v2": {
            "capabilities": {
                "abi_version": 2,
                "sm_target": "sm_86",
                "resource_query": "tacker_resources_v2",
                "supported_worker_groups": [1, 2, 3, 4, 5],
                "global_kernel_symbols": {
                    "multi_solo": PHASE3.HEAD_MULTI_SOLO_SYMBOL,
                    "multi_gptb": PHASE3.HEAD_MULTI_GPTB_SYMBOL,
                },
            },
            "resources": {
                "abi_version": 2,
                "kernels": {
                    symbol: dict(head_facts)
                    for symbol in PHASE3.HEAD_RESOURCE_SYMBOLS
                },
            },
        },
    }


class Phase3Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.source = self.root / "source"
        iteration = self.model / "point_cloud" / "iteration_14000"
        iteration.mkdir(parents=True)
        self.source.mkdir()
        for path in (
            self.model / "cfg_args",
            iteration / "point_cloud.ply",
            iteration / "deformation.pth",
            iteration / "deformation_table.pth",
            self.source / "poses_bounds.npy",
        ):
            path.write_bytes((path.name + "\n").encode("utf-8"))
        self.config = self.root / "config.py"
        self.config.write_text("x = 1\n", encoding="utf-8")
        self.current = self.root / "current.json"
        self.current.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest": {"persistent_blocks": 7000},
                    "admission": {"enabled": True, "valid": True},
                }
            ),
            encoding="utf-8",
        )
        self.template = self.root / "template.json"
        self.template.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "manifest": {"persistent_blocks": 7000},
                    "deployment": {"enabled": False, "valid": False},
                }
            ),
            encoding="utf-8",
        )
        self.suite = self.root / "suite.json"
        suite_digest = "a" * 64
        self.suite.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "4dgaussians_tacker_phase3_cuda_validation",
                    "passed": True,
                    "abi": {
                        "loaded_raster_binary_capabilities_verified": True,
                        "head_v1_manifest_sha256": suite_digest,
                        "head_v2_manifest_sha256": suite_digest,
                        "raster_v1_manifest_sha256": suite_digest,
                        "raster_v2_manifest_sha256": suite_digest,
                    },
                    "cuda_tests": {
                        "head": {"passed": 16, "total": 16},
                        "raster_v2_and_legacy": {"passed": 12, "total": 12},
                    },
                    "full_test_suites": {
                        "head": {"passed": 60, "total": 60},
                        "raster_v2_and_legacy": {"passed": 34, "total": 34},
                    },
                    "runtime_resources": {
                        "queries": [
                            {"worker_groups": value, "launch_supported": True}
                            for value in range(1, 6)
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        self.nsight = self.root / "profile_nsight.sh"
        self.nsight.write_text("#!/bin/sh\n", encoding="utf-8")
        self.args = types.SimpleNamespace(
            model_path=str(self.model),
            source_path=str(self.source),
            config=str(self.config),
            current_tacker_profile=str(self.current),
            template_profile=str(self.template),
            cuda_suite_report=str(self.suite),
            output_dir=str(self.root / "output"),
            python_executable=os.sys.executable,
            nvidia_smi="nvidia-smi",
            profile_nsight_script=str(self.nsight),
            gpu=0,
            workload_name="flame_steak",
            iteration=14000,
            split="test",
            image_width=1352,
            image_height=1014,
            gaussian_count=111525,
            head_rows=111525,
            persistent_block=[],
            beam_width=3,
            top_k=2,
            screen_frames=5,
            screen_warmup=1,
            screen_trials=2,
            leaf_views=2,
            leaf_warmup=1,
            leaf_repetitions=2,
            seed=0,
            timeout_seconds=None,
            resume=False,
            stop_after=None,
            dry_run_plan=False,
        )
        self.visible = mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"})
        self.visible.start()

    def tearDown(self):
        self.visible.stop()
        self.temporary.cleanup()


class StaticContractTest(Phase3Fixture):
    def test_python37_ast_and_no_shell_command_string(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), feature_version=(3, 7))
        self.assertIsInstance(tree, ast.Module)
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return completed()

        PHASE3.invoke(runner, ["tool", "$(touch nope)", "; rm -rf nope"])
        self.assertEqual(calls[0][0][1], "$(touch nope)")
        self.assertIsInstance(calls[0][0], list)
        self.assertIs(calls[0][1]["shell"], False)

    def test_physical_and_logical_gpu_are_not_conflated(self):
        identity = PHASE3._identity(self.args)
        workload = identity["payload"]["workload"]
        self.assertEqual(workload["logical_gpu"], 0)
        self.assertEqual(workload["physical_gpu"], 1)
        directory = self.root / "leaf"
        directory.mkdir()
        leaf = PHASE3.build_leaf_command(
            self.args, self.template, self.root / "matrix.json", directory
        )
        self.assertEqual(leaf[leaf.index("--gpu") + 1], "0")

    def test_visible_device_must_be_one_numeric_ordinal(self):
        for value in ("0,1", "GPU-deadbeef", ""):
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}):
                with self.assertRaisesRegex(PHASE3.Phase3Error, "exactly one"):
                    PHASE3.physical_gpu_from_environment()

    def test_current_and_template_are_separate_roles(self):
        self.args.template_profile = self.args.current_tacker_profile
        with self.assertRaisesRegex(PHASE3.Phase3Error, "must be distinct"):
            PHASE3._identity(self.args)

    def test_identity_binds_raw_workload_files(self):
        first = PHASE3._identity(self.args)
        point_cloud = self.model / "point_cloud" / "iteration_14000" / "point_cloud.ply"
        point_cloud.write_bytes(b"changed\n")
        second = PHASE3._identity(self.args)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertNotEqual(
            first["payload"]["workload_files"]["point_cloud.ply"]["sha256"],
            second["payload"]["workload_files"]["point_cloud.ply"]["sha256"],
        )

    def test_identity_recursively_binds_literal_base_configs(self):
        base = self.root / "base.py"
        base.write_text("value = 1\n", encoding="utf-8")
        self.config.write_text("_base_ = './base.py'\n", encoding="utf-8")
        first = PHASE3._identity(self.args)
        self.assertEqual(len(first["payload"]["configuration_chain"]), 2)
        base.write_text("value = 2\n", encoding="utf-8")
        second = PHASE3._identity(self.args)
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_phase2_cuda_suite_schema_is_accepted_strictly(self):
        digest = "a" * 64
        report = {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_phase2_targeted_validation",
            "status": {"phase2_exit_condition_complete": True},
            "abi": {
                "loaded_raster_binary_capabilities_verified": True,
                "head_v1_manifest_sha256": digest,
                "head_v2_manifest_sha256": digest,
                "raster_v1_manifest_sha256": digest,
                "raster_v2_manifest_sha256": digest,
            },
            "cuda_tests": {
                "head": {"passed": 16, "total": 16},
                "raster_v2_and_legacy": {"passed": 12, "total": 12},
            },
            "remote_full_test_suites": {
                "head": {"passed": 60, "total": 60},
                "raster_v2_and_legacy": {"passed": 34, "total": 34},
            },
            "runtime_resources": {
                "queries": [
                    {"worker_groups": value, "launch_supported": True}
                    for value in range(1, 6)
                ]
            },
        }
        self.assertTrue(PHASE3._cuda_suite_passed(report))
        report["cuda_tests"]["head"]["passed"] = 15
        self.assertFalse(PHASE3._cuda_suite_passed(report))
        self.assertFalse(PHASE3._cuda_suite_passed({"passed": True}))

    def test_phase3_cuda_suite_cannot_launder_tiny_or_renamed_suites(self):
        report = json.loads(self.suite.read_text(encoding="utf-8"))
        self.assertTrue(PHASE3._cuda_suite_passed(report))
        report["cuda_tests"]["head"] = {"passed": 1, "total": 1}
        self.assertFalse(PHASE3._cuda_suite_passed(report))
        report = json.loads(self.suite.read_text(encoding="utf-8"))
        report["full_test_suites"]["renamed"] = report[
            "full_test_suites"
        ].pop("raster_v2_and_legacy")
        self.assertFalse(PHASE3._cuda_suite_passed(report))

    def test_resource_query_requires_head_sm86_resources_for_every_wg(self):
        document = valid_resource_query()
        self.assertEqual(PHASE3._validate_resource_query(document, 0)["sm_count"], 84)

        cases = []
        wrong_ptx = valid_resource_query()
        wrong_ptx["head_v2"]["resources"]["kernels"][
            PHASE3.HEAD_MULTI_SOLO_SYMBOL
        ]["ptx_version"] = 80
        cases.append((wrong_ptx, "PTX/binary sm_86"))
        missing_local = valid_resource_query()
        del missing_local["head_v2"]["resources"]["kernels"][
            PHASE3.HEAD_MULTI_GPTB_SYMBOL
        ]["local_memory_bytes"]
        cases.append((missing_local, "local_memory_bytes"))
        zero_wg5 = valid_resource_query()
        zero_wg5["head_v2"]["resources"]["kernels"][
            PHASE3.HEAD_MULTI_GPTB_SYMBOL
        ]["active_blocks_per_sm"][4] = 0
        cases.append((zero_wg5, "WG1-5 occupancy"))
        missing_raster = valid_resource_query()
        del missing_raster["families"]["first_linear_heads_v2"][
            "worker_groups"
        ]["5"]
        cases.append((missing_raster, "Raster first_linear_heads_v2 WG5"))
        for invalid, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(PHASE3.Phase3Error, error):
                    PHASE3._validate_resource_query(invalid, 0)

    def test_identity_rejects_missing_or_symlinked_workload_files(self):
        point_cloud = self.model / "point_cloud" / "iteration_14000" / "point_cloud.ply"
        point_cloud.unlink()
        with self.assertRaisesRegex(PHASE3.Phase3Error, "required workload file"):
            PHASE3._identity(self.args)
        target = self.root / "point-cloud-target"
        target.write_bytes(b"payload")
        point_cloud.symlink_to(target)
        with self.assertRaisesRegex(PHASE3.Phase3Error, "required workload file"):
            PHASE3._identity(self.args)


class JournalTest(Phase3Fixture):
    def test_resume_clears_stopped_status_before_running_next_stage(self):
        output = self.root / "resume-status"
        identity = {"sha256": "a" * 64, "payload": {}}
        journal = PHASE3.Journal(output, identity, False)
        journal.state["status"] = "stopped_after_prepare"
        journal.save()
        journal.close()

        observed = []

        def action(directory):
            observed.append(PHASE3.load_json(output / "phase3-state.json")["status"])
            path = directory / "artifact.json"
            PHASE3.atomic_write_json(path, {"value": 1})
            return ({"path": str(path)}, [path])

        resumed = PHASE3.Journal(output, identity, True)
        resumed.run("next", {"x": 1}, action)
        resumed.close()
        self.assertEqual(observed, ["running"])

    def test_resume_skips_intact_stage_and_rejects_tampering(self):
        output = self.root / "journal"
        identity = {"sha256": "a" * 64, "payload": {}}
        calls = []

        def action(directory):
            calls.append(str(directory))
            path = directory / "artifact.json"
            PHASE3.atomic_write_json(path, {"value": 1})
            return ({"path": str(path)}, [path])

        journal = PHASE3.Journal(output, identity, False)
        result = journal.run("one", {"x": 1}, action)
        journal.close()
        resumed = PHASE3.Journal(output, identity, True)
        self.assertEqual(resumed.run("one", {"x": 1}, action), result)
        resumed.close()
        self.assertEqual(len(calls), 1)
        Path(result["path"]).write_text('{"value":2}\n', encoding="utf-8")
        resumed = PHASE3.Journal(output, identity, True)
        with self.assertRaisesRegex(PHASE3.Phase3Error, "artifact changed"):
            resumed.run("one", {"x": 1}, action)
        resumed.close()


class CommandStageTest(Phase3Fixture):
    def _journal(self, name):
        identity = {
            "sha256": "b" * 64,
            "payload": {
                "workload": {
                    "name": "flame_steak",
                    "logical_gpu": 0,
                    "physical_gpu": 1,
                },
                "search": {},
                "configuration_chain": [],
            },
        }
        return PHASE3.Journal(self.root / name, identity, False)

    def test_prepare_flow_builds_complete_base_matrix_with_separate_template(self):
        self.args.output_dir = str(self.root / "prepare-output")
        self.args.stop_after = "prepare"
        commands = []
        autotune = PHASE3._load_autotune_module()

        def runner(argv, **kwargs):
            commands.append(list(argv))
            if argv[0] == "nvidia-smi":
                return completed(stdout="1, NVIDIA RTX A6000, 8.6\n")
            if "_resource-query" in argv:
                PHASE3.atomic_write_json(
                    Path(argv[argv.index("--output") + 1]),
                    valid_resource_query(),
                )
                return completed()
            if "matrix" in argv:
                matrix = autotune.build_base_matrix(
                    [0, 84, 168, 336, 5440, 7000, 13942],
                    sm_count=84,
                    raster_tile_count=5440,
                    backend_logical_blocks=13942,
                )
                PHASE3.atomic_write_json(Path(argv[argv.index("--output") + 1]), matrix)
                return completed()
            if "profiles" in argv:
                matrix = PHASE3.load_json(argv[argv.index("--matrix") + 1])
                output = Path(argv[argv.index("--output-dir") + 1])
                output.mkdir(parents=True)
                for candidate in matrix["candidates"]:
                    PHASE3.atomic_write_json(
                        output / "{}.json".format(candidate["variant_id"]),
                        {
                            "selected_variant_id": candidate["variant_id"],
                            "deployment": {"enabled": False, "valid": False},
                            "provenance": {
                                "candidate_sha256": candidate["candidate_sha256"],
                                "matrix_sha256": matrix["matrix_sha256"],
                            },
                        },
                    )
                PHASE3.atomic_write_json(
                    output / "qualification_profiles.json",
                    {"matrix_sha256": matrix["matrix_sha256"]},
                )
                return completed(stdout="{}")
            return completed(returncode=2, stderr="unexpected command")

        report = PHASE3.run_phase3(self.args, runner=runner)
        self.assertEqual(report["stopped_after"], "prepare")
        matrix = report["result"]["matrix"]["matrix"]
        self.assertTrue(any(item["search_level"] == "c0" for item in matrix["candidates"]))
        self.assertTrue(any(item["search_level"] == "c1" for item in matrix["candidates"]))
        self.assertTrue(any(item["search_level"] == "c2" for item in matrix["candidates"]))
        self.assertIn("--id=1", commands[0])
        resource = next(item for item in commands if "_resource-query" in item)
        self.assertEqual(resource[resource.index("--gpu") + 1], "0")
        profile_command = next(item for item in commands if "profiles" in item)
        self.assertEqual(
            Path(profile_command[profile_command.index("--template") + 1]),
            self.template.resolve(),
        )

    def test_formal_requires_real_10x50_protocol_field(self):
        journal = self._journal("formal")
        baseline = self.root / "baseline.json"
        baseline.write_text("{}\n", encoding="utf-8")
        commands = []

        def runner(argv, **kwargs):
            commands.append(list(argv))
            if "formal-plan" in argv:
                plan_path = Path(argv[argv.index("--output") + 1])
                correctness = Path(argv[argv.index("--correctness-output") + 1])
                PHASE3.atomic_write_json(
                    plan_path,
                    {
                        "kind": "tacker_autotune_formal_benchmark_plan",
                        "matrix_sha256": "3" * 64,
                        "candidates": [
                            {"variant_id": "x"},
                            {"variant_id": "y"},
                        ],
                        "benchmark_argv_fragment": ["--candidate", "x=/p"],
                    },
                )
                PHASE3.atomic_write_json(correctness, {})
            elif "benchmark_tacker_fps.py" in argv[1]:
                output = Path(argv[argv.index("--output") + 1])
                run_id = argv[argv.index("--run-id") + 1]
                runs = Path(argv[argv.index("--runs-dir") + 1]) / run_id
                runs.mkdir(parents=True)
                PHASE3.atomic_write_json(runs / "benchmark.checkpoint.json", {"ok": True})
                PHASE3.atomic_write_json(
                    output,
                    {
                        "kind": "4dgaussians_tacker_fps_benchmark",
                        "passed": True,
                        "phase0_exit_condition": {"met": True},
                        "contract": {"profile_frames": 50, "warmup_frames": 10},
                        "schedule": {"strategy": "abba", "trials_per_candidate": 10},
                        "candidates": [
                            {"name": "serial"},
                            {"name": "two_stream"},
                            {"name": "current_tacker"},
                            {"name": "x"},
                            {"name": "y"},
                        ],
                        "paired_comparisons": [
                            {"candidate": "x", "reference": "two_stream"},
                            {"candidate": "x", "reference": "current_tacker"},
                            {"candidate": "y", "reference": "two_stream"},
                            {"candidate": "y", "reference": "current_tacker"},
                        ],
                        "experimental_winner": "x",
                        "deployment_winner": "current_tacker",
                        "promotion": {
                            "reason": "synthetic incumbent retained",
                            "reason_code": "incumbent_preferred",
                            "criteria": {},
                        },
                    },
                )
            return completed()

        result = PHASE3._formal(
            journal,
            self.args,
            runner,
            {
                "db": str(self.root / "db.sqlite"),
                "screening_input_sha256": "1" * 64,
            },
            {"correctness_input_sha256": "2" * 64},
            {"correctness_path": str(baseline)},
            {"matrix_sha256": "3" * 64},
            self.root,
        )
        journal.close()
        self.assertTrue(Path(result["fps_report"]).is_file())
        formal_plan = next(item for item in commands if "formal-plan" in item)
        top_k_index = formal_plan.index("--top-k")
        self.assertEqual(formal_plan[top_k_index + 1], "2")
        self.assertEqual(formal_plan[top_k_index + 2], "--profiles-dir")
        benchmark = next(item for item in commands if "benchmark_tacker_fps.py" in item[1])
        self.assertEqual(benchmark[benchmark.index("--frames") + 1], "50")
        self.assertEqual(benchmark[benchmark.index("--trials") + 1], "10")
        self.assertEqual(benchmark[benchmark.index("--schedule") + 1], "abba")

    def test_correctness_failure_releases_exact_db_claim_once(self):
        journal = self._journal("correctness")
        candidate = {
            "candidate_sha256": "4" * 64,
            "variant_id": "candidate",
            "selected_heads": ["pos"],
        }
        profiles = self.root / "profiles"
        profiles.mkdir()
        (profiles / "candidate.json").write_text("{}\n", encoding="utf-8")
        matrix_path = self.root / "matrix.json"
        matrix_path.write_text("{}\n", encoding="utf-8")
        commands = []

        def runner(argv, **kwargs):
            commands.append(list(argv))
            if "db-claim" in argv:
                return completed(
                    stdout=json.dumps(
                        {
                            "action": "run",
                            "claim_token": "5" * 32,
                            "record": {"input_sha256": "6" * 64},
                        }
                    )
                )
            if "db-fail" in argv:
                return completed(stdout="{}")
            if "profile_tacker_leaves.py" in argv[1]:
                return completed(returncode=2, stderr="synthetic leaf failure")
            return completed(stdout="{}")

        with self.assertRaisesRegex(PHASE3.Phase3Error, "status 2"):
            PHASE3._candidate_correctness(
                journal,
                self.args,
                runner,
                self.root / "db.sqlite",
                matrix_path,
                {"matrix_sha256": "7" * 64},
                profiles,
                [{"candidate": candidate}],
            )
        journal.close()
        failures = [item for item in commands if "db-fail" in item]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][failures[0].index("--claim-token") + 1], "5" * 32)

    def test_interrupted_screen_reuses_child_checkpoint(self):
        output_root = self.root / "screen-resume"
        identity = {
            "sha256": "8" * 64,
            "payload": {
                "workload": {},
                "search": {},
            },
        }
        candidate = {
            "candidate_sha256": "9" * 64,
            "variant_id": "c2_pos_scales_wg1_pb84",
            "selected_heads": ["pos", "scales"],
            "search_level": "c2",
        }
        matrix = {
            "matrix_sha256": "a" * 64,
            "candidates": [candidate],
        }
        profiles = self.root / "screen-profiles"
        profiles.mkdir()
        profile = profiles / "c2_pos_scales_wg1_pb84.json"
        profile.write_text(
            json.dumps(
                {
                    "selected_variant_id": candidate["variant_id"],
                    "deployment": {"enabled": False, "valid": False},
                    "provenance": {
                        "candidate_sha256": candidate["candidate_sha256"],
                        "matrix_sha256": matrix["matrix_sha256"],
                    },
                }
            ),
            encoding="utf-8",
        )
        benchmark_commands = []

        def runner(argv, **kwargs):
            benchmark_commands.append(list(argv))
            report = Path(argv[argv.index("--output") + 1])
            runs_dir = Path(argv[argv.index("--runs-dir") + 1])
            run_id = argv[argv.index("--run-id") + 1]
            child = runs_dir / run_id
            child.mkdir(parents=True, exist_ok=True)
            PHASE3.atomic_write_json(child / "benchmark.checkpoint.json", {"prefix": 1})
            if len(benchmark_commands) == 1:
                PHASE3.atomic_write_json(report, {"passed": False})
                raise RuntimeError("coordinator interrupted")
            PHASE3.atomic_write_json(
                report,
                {
                    "passed": True,
                    "contract": {"profile_frames": 5, "warmup_frames": 1},
                    "schedule": {
                        "strategy": "round_robin",
                        "trials_per_candidate": 2,
                    },
                    "candidates": [
                        {"name": "serial"},
                        {"name": "two_stream"},
                        {"name": "current_tacker"},
                        {"name": candidate["variant_id"]},
                    ],
                    "summaries": {
                        candidate["variant_id"]: {"median_throughput_fps": 91.0}
                    },
                },
            )
            return completed()

        journal = PHASE3.Journal(output_root, identity, False)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            PHASE3._screen_level(
                journal, self.args, runner, matrix, profiles, 2, "h2"
            )
        journal.close()
        self.args.resume = True
        resumed = PHASE3.Journal(output_root, identity, True)
        result = PHASE3._screen_level(
            resumed, self.args, runner, matrix, profiles, 2, "h2"
        )
        resumed.close()
        self.assertTrue(Path(result["scores_path"]).is_file())
        self.assertEqual(len(benchmark_commands), 2)
        self.assertNotIn("--resume", benchmark_commands[0])
        self.assertIn("--resume", benchmark_commands[1])
        for option in ("--runs-dir", "--run-id", "--output"):
            left = benchmark_commands[0][benchmark_commands[0].index(option) + 1]
            right = benchmark_commands[1][benchmark_commands[1].index(option) + 1]
            self.assertEqual(left, right)

    def test_beam_stage_delegates_expansion_to_autotune(self):
        autotune = PHASE3._load_autotune_module()
        matrix = autotune.build_base_matrix(
            [84], sm_count=84, raster_tile_count=5440, backend_logical_blocks=13942
        )
        matrix_path = self.root / "beam-base.json"
        PHASE3.atomic_write_json(matrix_path, matrix)
        scores = {
            item["candidate_sha256"]: {
                "candidate_sha256": item["candidate_sha256"],
                "score": float(index + 1),
                "status": "succeeded",
            }
            for index, item in enumerate(matrix["candidates"])
            if len(item["selected_heads"]) == 2
        }
        scores_path = self.root / "beam-scores.json"
        PHASE3.atomic_write_json(scores_path, scores)
        commands = []

        def runner(argv, **kwargs):
            commands.append(list(argv))
            output = Path(argv[argv.index("--output") + 1])
            expanded = autotune.extend_matrix_with_beam(
                matrix, scores, self.args.beam_width, 3
            )
            PHASE3.atomic_write_json(output, expanded)
            return completed()

        journal = self._journal("beam")
        result = PHASE3._beam(
            journal,
            self.args,
            runner,
            matrix_path,
            matrix,
            scores_path,
            3,
        )
        journal.close()
        self.assertTrue(
            any(len(item["selected_heads"]) == 3 for item in result["matrix"]["candidates"])
        )
        self.assertIn("beam", commands[0])
        self.assertEqual(commands[0][commands[0].index("--target-head-count") + 1], "3")

    def test_screening_db_completion_uses_evidence_and_result_once(self):
        journal = self._journal("screening-db")
        source_report = self.root / "source-screen.json"
        source_profile = self.root / "source-profile.json"
        manifest = self.root / "profiles-manifest.json"
        matrix_path = self.root / "screen-matrix.json"
        for path in (source_report, source_profile, manifest, matrix_path):
            path.write_text("{}\n", encoding="utf-8")
        candidate = {
            "candidate_sha256": "d" * 64,
            "variant_id": "candidate",
            "selected_heads": ["pos"],
        }
        profile_facts = PHASE3.artifact(source_profile)
        report_facts = PHASE3.artifact(source_report)
        ranking = {
            "scores": {
                candidate["candidate_sha256"]: {
                    "candidate_sha256": candidate["candidate_sha256"],
                    "score": 90.0,
                    "status": "succeeded",
                }
            },
            "source_reports": {candidate["candidate_sha256"]: str(source_report)},
            "source_bindings": {
                candidate["candidate_sha256"]: {
                    "source_matrix_sha256": "c" * 64,
                    "profile": profile_facts,
                    "report": report_facts,
                }
            },
        }
        commands = []

        def runner(argv, **kwargs):
            commands.append(list(argv))
            if "db-claim" in argv:
                return completed(
                    stdout=json.dumps(
                        {
                            "action": "run",
                            "claim_token": "b" * 32,
                            "record": {"input_sha256": "a" * 64},
                        }
                    )
                )
            return completed(stdout="{}")

        result = PHASE3._record_screening_db(
            journal,
            self.args,
            runner,
            self.root / "screen.sqlite",
            matrix_path,
            {"matrix_sha256": "e" * 64, "candidates": [candidate]},
            ranking,
            manifest,
        )
        journal.close()
        completions = [item for item in commands if "db-complete" in item]
        self.assertEqual(len(completions), 1)
        command = completions[0]
        self.assertEqual(command.count("--artifact"), 1)
        self.assertEqual(command.count("--result"), 1)
        self.assertEqual(command[command.index("--claim-token") + 1], "b" * 32)
        self.assertEqual(result["screening_input_sha256"], "a" * 64)

    def _correctness_candidates(self):
        profiles = self.root / "backfill-profiles"
        profiles.mkdir(exist_ok=True)
        candidates = []
        ranked = []
        for index in range(3):
            candidate = {
                "candidate_sha256": str(index + 1) * 64,
                "variant_id": "candidate{}".format(index),
                "selected_heads": ["pos"],
            }
            (profiles / "{}.json".format(candidate["variant_id"])).write_text(
                "{}\n", encoding="utf-8"
            )
            candidates.append(candidate)
            ranked.append({"candidate": candidate, "score": 100.0 - index})
        matrix_path = self.root / "backfill-matrix.json"
        matrix_path.write_text("{}\n", encoding="utf-8")
        return profiles, ranked, matrix_path

    def test_invalid_top_candidate_is_backfilled_to_target_k(self):
        profiles, ranked, matrix_path = self._correctness_candidates()
        journal = self._journal("backfill")
        completed_candidates = []

        def runner(argv, **kwargs):
            if "db-claim" in argv:
                digest = argv[argv.index("--candidate-sha256") + 1]
                return completed(
                    stdout=json.dumps(
                        {
                            "action": "run",
                            "claim_token": digest[:32],
                            "record": {"input_sha256": "f" * 64},
                        }
                    )
                )
            if "profile_tacker_leaves.py" in argv[1]:
                profile_path = str(Path(argv[argv.index("--candidate-profile") + 1]).resolve())
                matrix_requested = str(Path(argv[argv.index("--candidate-matrix") + 1]).resolve())
                variant_id = Path(profile_path).stem
                candidate = next(
                    item["candidate"] for item in ranked if item["candidate"]["variant_id"] == variant_id
                )
                PHASE3.atomic_write_json(
                    Path(argv[argv.index("--report") + 1]),
                    {
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
                        "variant_id": variant_id,
                        "profile_binding": {
                            "candidate_sha256": candidate["candidate_sha256"],
                            "candidate_matrix_sha256": "e" * 64,
                            "candidate_matrix_file_sha256": PHASE3.sha256_file(matrix_path),
                            "profile_file_sha256": PHASE3.sha256_file(profile_path),
                        },
                        "parameters": {
                            "candidate_profile": profile_path,
                            "qualification_profile": True,
                            "used_as_deployment": False,
                        },
                        "measurement_outputs_written": True,
                        "candidate_matrix_requested": matrix_requested,
                    },
                )
                for option in ("--device-output", "--raster-output", "--leaf-output"):
                    PHASE3.atomic_write_json(Path(argv[argv.index(option) + 1]), {"ok": True})
                return completed()
            if "validate_tacker_modes.py" in argv[1]:
                profile = Path(argv[argv.index("--qualification-profile") + 1])
                passed = profile.stem != "candidate0"
                PHASE3.atomic_write_json(
                    Path(argv[argv.index("--output") + 1]),
                    {
                        "schema_version": 1,
                        "kind": "4dgaussians_tacker_quality_validation",
                        "passed": passed,
                        "workload": {
                            "scene": "flame_steak",
                            "iteration": 14000,
                            "resolution": [1352, 1014],
                            "gaussian_count": 111525,
                        },
                        "qualification": {
                            "enabled": True,
                            "admission_claimed": False,
                            "profile_override": str(profile.resolve()),
                        },
                        "modes": (
                            {
                                "serial": {},
                                "tacker": {
                                    "actual_mode": "tacker",
                                    "qualification_executed": True,
                                },
                            }
                            if passed
                            else {}
                        ),
                        "gates": ([{"mode": "tacker", "passed": True}] if passed else []),
                    },
                )
                return completed(returncode=0 if passed else 1)
            if "db-complete" in argv:
                completed_candidates.append(argv[argv.index("--candidate-sha256") + 1])
                return completed(stdout="{}")
            return completed(stdout="{}")

        result = PHASE3._candidate_correctness(
            journal,
            self.args,
            runner,
            self.root / "backfill.sqlite",
            matrix_path,
            {"matrix_sha256": "e" * 64},
            profiles,
            ranked,
        )
        journal.close()
        self.assertEqual(len(result["records"]), 3)
        self.assertFalse(result["records"][0]["result"]["valid"])
        self.assertTrue(result["records"][1]["result"]["valid"])
        self.assertTrue(result["records"][2]["result"]["valid"])
        self.assertEqual(len(completed_candidates), 3)

    def test_resumed_db_results_count_invalid_then_valid_without_rerun(self):
        profiles, ranked, matrix_path = self._correctness_candidates()
        journal = self._journal("backfill-skip")
        heavy_commands = []

        def runner(argv, **kwargs):
            if "db-claim" in argv:
                digest = argv[argv.index("--candidate-sha256") + 1]
                valid = digest != "1" * 64
                return completed(
                    stdout=json.dumps(
                        {
                            "action": "skip",
                            "claim_token": None,
                            "record": {
                                "input_sha256": "f" * 64,
                                "result": {"valid": valid},
                            },
                        }
                    )
                )
            heavy_commands.append(list(argv))
            return completed()

        result = PHASE3._candidate_correctness(
            journal,
            self.args,
            runner,
            self.root / "backfill-skip.sqlite",
            matrix_path,
            {"matrix_sha256": "e" * 64},
            profiles,
            ranked,
        )
        journal.close()
        self.assertEqual(len(result["records"]), 3)
        self.assertTrue(all(item["resumed_from_db"] for item in result["records"]))
        self.assertEqual(heavy_commands, [])


if __name__ == "__main__":
    unittest.main()
