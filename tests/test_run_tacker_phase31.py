"""CPU-only contracts for the independent Phase-3.1 coordinator."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_tacker_phase31.py"
SPEC = importlib.util.spec_from_file_location("run_tacker_phase31", MODULE_PATH)
PHASE31 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHASE31)
AUTOTUNE = PHASE31.AUTOTUNE


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = load_module(
    "phase31_benchmark_name_contract", ROOT / "scripts" / "benchmark_tacker_fps.py"
)
TOP3 = load_module(
    "phase31_top3_name_contract", ROOT / "scripts" / "profile_tacker_top3.py"
)


def raster_resource(abi, family, worker_groups):
    physical_threads = 384 if abi == 1 else 256 + 128 * worker_groups
    return {
        "abi_version": abi,
        "backend_abi_version": abi,
        "backend_family": family,
        "family": family,
        "worker_groups": worker_groups,
        "physical_threads": physical_threads,
        "device_ordinal": 0,
        "compute_capability_major": 8,
        "compute_capability_minor": 6,
        "multiprocessor_count": 84,
        "device_max_threads_per_block": 1024,
        "device_max_threads_per_multiprocessor": 1536,
        "kernel_max_threads_per_block": 1024,
        "registers_per_thread": 64,
        "static_shared_bytes": 512 if abi == 4 else 0,
        "local_bytes_per_thread": 0,
        "max_dynamic_shared_bytes": 0,
        "active_blocks_per_multiprocessor": 1,
        "occupancy": physical_threads / 1536.0,
        "launch_supported": True,
    }


def valid_resource_query():
    raster_manifests, head_manifest = PHASE31._manifest_paths()
    capabilities = {
        "resource_query_family_aware": True,
        "supported_mixed_abis": [1, 2, 3, 4],
        "supported_backend_families": [
            "first_linear_heads_v2",
            "packed_first_linear_v3",
            "whole_heads_v4",
        ],
    }
    families = {}
    manifests = {}
    for family, contract in PHASE31.RASTER_FAMILY_CONTRACTS.items():
        capabilities[contract["capability_abi"]] = contract["abi_version"]
        capabilities[contract["capability_enabled"]] = True
        capabilities[contract["capability_symbol"]] = contract["symbol"]
        digest = PHASE31.sha256_file(raster_manifests[family])
        capabilities[contract["capability_manifest"]] = digest
        head_capability = contract.get("capability_head_manifest")
        if head_capability is not None:
            capabilities[head_capability] = PHASE31.sha256_file(head_manifest)
        resource_family = contract["resource_family"]
        families[family] = {
            "worker_groups": {
                str(worker_groups): raster_resource(
                    contract["abi_version"], resource_family, worker_groups
                )
                for worker_groups in contract["worker_groups"]
            }
        }
        manifests[family] = {"sha256": digest}
    manifests["head_linear_v2"] = {
        "sha256": PHASE31.sha256_file(head_manifest)
    }
    kernels = {}
    for name, symbol in PHASE31.HEAD_SYMBOLS.items():
        thread_grid = (
            [128]
            if name == "whole_head_gptb"
            else [128, 256, 384, 512, 640]
        )
        kernels[symbol] = {
            "registers_per_thread": 48,
            "static_shared_memory_bytes": 0,
            "local_memory_bytes": 0,
            "max_threads_per_block": 1024,
            "ptx_version": 86,
            "binary_version": 86,
            "worker_group_threads": thread_grid,
            "active_blocks_per_sm": [2] * len(thread_grid),
        }
    return {
        "schema_version": 1,
        "kind": PHASE31.RESOURCE_KIND,
        "passed": True,
        "device": {
            "index": 0,
            "name": "NVIDIA RTX A6000",
            "compute_capability": [8, 6],
            "sm_count": 84,
        },
        "raster_capabilities": capabilities,
        "families": families,
        "head_v2": {
            "capabilities": {
                "abi_version": 2,
                "sm_target": "sm_86",
                "global_kernel_symbols": dict(PHASE31.HEAD_SYMBOLS),
            },
            "resources": {"abi_version": 2, "kernels": kernels},
        },
        "manifests": manifests,
    }


def valid_cuda_suite(root, abi_digest="a" * 64):
    root = Path(root)
    suites = {}
    for index, (name, scope) in enumerate(
        PHASE31.REQUIRED_CUDA_SUITE_SCOPES.items()
    ):
        log = root / "{}-{}.log".format(index, name)
        log.write_text("{} passed\n".format(name), encoding="utf-8")
        suites[name] = {
            "passed": 12,
            "total": 12,
            "scope": list(scope),
            "log_artifact": PHASE31.artifact(log),
        }
    ptxas_log = root / "ptxas.log"
    ptxas_log.write_text("ptxas evidence\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_phase31_cuda_validation",
        "passed": True,
        "abi": {
            key: abi_digest
            for key in (
                "head_v2_manifest_sha256",
                "raster_v1_manifest_sha256",
                "raster_v2_manifest_sha256",
                "raster_v3_manifest_sha256",
                "raster_v4_manifest_sha256",
            )
        },
        "test_suites": suites,
        "ptxas": {
            "passed": True,
            "sm_target": "sm_86",
            "artifact": PHASE31.artifact(ptxas_log),
            "kernels": {
                symbol: {
                    "accepted": True,
                    "registers_per_thread": 64,
                    "static_shared_bytes": 0,
                    "local_bytes_per_thread": 0,
                }
                for symbol in PHASE31.REQUIRED_PTXAS_SYMBOLS
            },
        },
    }


def base_matrix():
    blocks = AUTOTUNE.derive_persistent_blocks(
        84,
        5525,
        13942,
        current_persistent_blocks=7000,
        extra_values=[],
    )
    return AUTOTUNE.build_phase31_base_matrix(
        blocks,
        sm_count=84,
        raster_tile_count=5525,
        backend_logical_blocks=13942,
        whole_head_logical_blocks=111525,
    )


def successful_records(matrix, family=None, start=1000.0):
    records = []
    for index, candidate in enumerate(matrix["candidates"]):
        if family is not None and candidate["search_family"] != family:
            continue
        records.append(
            {
                "candidate_sha256": candidate["candidate_sha256"],
                "status": "succeeded",
                "score": start - index * 0.001,
            }
        )
    return records


def quality_per_view():
    return [
        {"batch_index": index, "psnr_db": 40.0, "ssim": 0.99, "lpips": 0.01}
        for index in range(50)
    ]


def baseline_quality_report(args, invalid_mode=None):
    modes = {}
    gates = []
    for mode in ("serial", "two_stream", "tacker"):
        invalid = mode == invalid_mode
        actual = "serial_fallback" if invalid else mode
        modes[mode] = {
            "requested_mode": mode,
            "actual_mode": actual,
            "fallback_reason": "forced fallback" if invalid else None,
            "qualification_requested": False,
            "qualification_executed": False,
            "per_view": quality_per_view(),
        }
        if mode != "serial":
            gates.append(
                {
                    "mode": mode,
                    "actual_mode": actual,
                    "actual_mode_passed": not invalid,
                    "qualification_passed": True,
                    "quality_passed": True,
                    "passed": not invalid,
                }
            )
    passed = invalid_mode is None
    return {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_quality_validation",
        "passed": passed,
        "workload": {
            "scene": "flame_steak",
            "iteration": 14000,
            "split": "test",
            "frames": 50,
            "view_indices": list(range(50)),
            "resolution": [1352, 1014],
            "gaussian_count": 111525,
            "model_path": str(Path(args.model_path).resolve()),
            "source_path": str(Path(args.source_path).resolve()),
        },
        "modes": modes,
        "gates": gates,
        "errors": [] if passed else ["{} invalid".format(invalid_mode)],
        "qualification": {
            "enabled": False,
            "admission_claimed": passed,
            "profile_override": None,
        },
        "tacker_profile": str(Path(args.current_tacker_profile).resolve()),
    }


class ResourceContractTests(unittest.TestCase):
    def test_validates_all_raster_families_and_head_symbols(self):
        device = PHASE31._validate_resource_query(valid_resource_query(), 0)
        self.assertEqual(device["sm_count"], 84)

    def test_rejects_wrong_c3_symbol(self):
        document = valid_resource_query()
        document["raster_capabilities"]["mixed_packed_symbol"] = "wrong"
        with self.assertRaisesRegex(PHASE31.Phase31Error, "packed"):
            PHASE31._validate_resource_query(document, 0)

    def test_rejects_nonlaunchable_c4_resource(self):
        document = valid_resource_query()
        document["families"]["whole_heads_v4"]["worker_groups"]["1"][
            "launch_supported"
        ] = False
        with self.assertRaisesRegex(PHASE31.Phase31Error, "launchable"):
            PHASE31._validate_resource_query(document, 0)

    def test_legacy_query_uses_ext_accepted_resource_family(self):
        self.assertEqual(
            PHASE31.RASTER_FAMILY_CONTRACTS["legacy_pos_l1_v1"]["resource_family"],
            "first_linear_head_v1",
        )
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('contract["resource_family"]', source)

    def test_rejects_backend_specific_head_manifest_mismatch(self):
        document = valid_resource_query()
        document["raster_capabilities"][
            "mixed_whole_head_manifest_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(PHASE31.Phase31Error, "head dependency"):
            PHASE31._validate_resource_query(document, 0)

    def test_cuda_suite_requires_phase31_and_all_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = valid_cuda_suite(temporary)
            self.assertIs(PHASE31._validate_cuda_suite_report(report), report)
            del report["abi"]["raster_v4_manifest_sha256"]
            with self.assertRaises(PHASE31.Phase31Error):
                PHASE31._validate_cuda_suite_report(report)

    def test_cuda_suite_requires_fixed_scopes_and_live_distinct_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = valid_cuda_suite(temporary)
            del report["test_suites"]["runtime_c3_c4_fallback"]
            with self.assertRaisesRegex(PHASE31.Phase31Error, "required test suites"):
                PHASE31._validate_cuda_suite_report(report)

            report = valid_cuda_suite(temporary)
            report["test_suites"]["raster_abi1_4_cuda"]["scope"] = [
                "raster_abi4"
            ]
            with self.assertRaisesRegex(PHASE31.Phase31Error, "scope"):
                PHASE31._validate_cuda_suite_report(report)

            report = valid_cuda_suite(temporary)
            log_path = Path(
                report["test_suites"]["head_extension_cuda"]["log_artifact"][
                    "path"
                ]
            )
            log_path.write_text("mutated\n", encoding="utf-8")
            with self.assertRaises(PHASE31.Phase31Error):
                PHASE31._validate_cuda_suite_report(report)


class IdentityHardeningTests(unittest.TestCase):
    def test_identity_seals_timeout_executable_and_production_runtime_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            source = root / "source"
            iteration = model / "point_cloud" / "iteration_14000"
            iteration.mkdir(parents=True)
            source.mkdir()
            for path in (
                model / "cfg_args",
                iteration / "point_cloud.ply",
                iteration / "deformation.pth",
                iteration / "deformation_table.pth",
                source / "poses_bounds.npy",
            ):
                path.write_bytes(b"fixed-workload\n")
            config = root / "config.py"
            config.write_text("# fixed config\n", encoding="utf-8")
            current = root / "current.json"
            PHASE31.atomic_write_json(
                current,
                {
                    "schema_version": 2,
                    "deployment": {"enabled": True, "valid": True},
                    "manifest": {"persistent_blocks": 7000},
                },
            )
            template = root / "template.json"
            PHASE31.atomic_write_json(
                template,
                {
                    "schema_version": 2,
                    "deployment": {"enabled": False, "valid": False},
                },
            )
            nvidia_smi = root / "nvidia-smi"
            nvidia_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            nvidia_smi.chmod(0o755)
            suite = valid_cuda_suite(root)
            raster_manifests, head_manifest = PHASE31._manifest_paths()
            suite["abi"] = {
                "head_v2_manifest_sha256": PHASE31.sha256_file(head_manifest),
                "raster_v1_manifest_sha256": PHASE31.sha256_file(
                    raster_manifests["legacy_pos_l1_v1"]
                ),
                "raster_v2_manifest_sha256": PHASE31.sha256_file(
                    raster_manifests["first_linear_heads_v2"]
                ),
                "raster_v3_manifest_sha256": PHASE31.sha256_file(
                    raster_manifests["packed_first_linear_v3"]
                ),
                "raster_v4_manifest_sha256": PHASE31.sha256_file(
                    raster_manifests["whole_heads_v4"]
                ),
            }
            suite_path = root / "cuda-suite.json"
            PHASE31.atomic_write_json(suite_path, suite)
            args = types.SimpleNamespace(
                model_path=str(model),
                source_path=str(source),
                config=str(config),
                current_tacker_profile=str(current),
                template_profile=str(template),
                cuda_suite_report=str(suite_path),
                python_executable=os.path.realpath(os.sys.executable),
                nvidia_smi=str(nvidia_smi),
                profile_nsight_script=str(ROOT / "scripts" / "profile_nsight.sh"),
                workload_name="flame_steak",
                iteration=14000,
                split="test",
                image_width=1352,
                image_height=1014,
                gaussian_count=111525,
                head_rows=111525,
                gpu=0,
                persistent_block=[],
                packed_persistent_block=[],
                whole_head_persistent_block=[],
                c3_top_k=4,
                c4_top_k_per_family=2,
                top_k=5,
                screen_batch_size=30,
                screen_frames=10,
                screen_warmup=2,
                screen_trials=2,
                seed=0,
                leaf_views=2,
                leaf_warmup=5,
                leaf_repetitions=50,
                timeout_seconds=123.0,
            )
            with mock.patch.object(
                PHASE31, "physical_gpu_from_environment", return_value=1
            ), mock.patch.object(PHASE31, "_configuration_chain", return_value=[]):
                first = PHASE31._identity(args)
                args.timeout_seconds = 124.0
                timeout_changed = PHASE31._identity(args)
                args.timeout_seconds = 123.0
                nvidia_smi.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
                executable_changed = PHASE31._identity(args)
                real_artifact = PHASE31.artifact

                def changed_runtime_artifact(path):
                    facts = real_artifact(path)
                    if Path(path).resolve() == (
                        ROOT / "gaussian_renderer" / "tacker_pipeline.py"
                    ).resolve():
                        facts = dict(facts)
                        facts["sha256"] = "f" * 64
                    return facts

                with mock.patch.object(
                    PHASE31, "artifact", side_effect=changed_runtime_artifact
                ):
                    runtime_changed = PHASE31._identity(args)

            payload = first["payload"]
            self.assertEqual(payload["search"]["timeout_seconds"], 123.0)
            self.assertEqual(
                payload["files"]["nvidia_smi_executable"]["path"],
                str(nvidia_smi.resolve()),
            )
            self.assertNotEqual(
                payload["files"]["nvidia_smi_executable"]["sha256"],
                PHASE31.artifact(nvidia_smi)["sha256"],
            )
            for name in (
                "profile_render",
                "tacker_pipeline",
                "raster_python_binding",
                "raster_extension_binding",
                "raster_cuda_dispatch",
                "raster_mixed_kernels",
            ):
                self.assertIn(name, payload["runtime_sources"])
            self.assertNotEqual(first["sha256"], timeout_changed["sha256"])
            self.assertNotEqual(first["sha256"], executable_changed["sha256"])
            self.assertNotEqual(
                executable_changed["sha256"], runtime_changed["sha256"]
            )


class MatrixAndSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = base_matrix()

    def test_exhaustive_c2_is_exactly_450(self):
        PHASE31._validate_matrix(self.base, require_c2_450=True)
        counts = {
            family: sum(
                candidate["search_family"] == family
                for candidate in self.base["candidates"]
            )
            for family in PHASE31.SEARCH_FAMILIES
        }
        self.assertEqual(counts, {"c0": 6, "c1": 30, "c2": 450, "c3": 0, "c4": 0})

    def test_batches_never_mix_family_or_head_count(self):
        batches = PHASE31.candidate_batches(
            self.base, ("c0", "c1", "c2"), 17
        )
        flattened = []
        for batch in batches:
            self.assertLessEqual(len(batch["candidates"]), 17)
            self.assertEqual(
                {item["search_family"] for item in batch["candidates"]},
                {batch["family"]},
            )
            self.assertEqual(
                {len(item["selected_heads"]) for item in batch["candidates"]},
                {batch["head_count"]},
            )
            flattened.extend(item["candidate_sha256"] for item in batch["candidates"])
        self.assertEqual(len(flattened), 486)
        self.assertEqual(len(set(flattened)), 486)

    def test_c3_then_c4_generation_covers_packed_and_whole_shapes(self):
        base_records = successful_records(self.base)
        c3 = AUTOTUNE.extend_phase31_with_c3(
            self.base, base_records, top_k=4
        )
        c3_records = base_records + successful_records(c3, family="c3", start=2000.0)
        c4 = AUTOTUNE.extend_phase31_with_c4(
            c3, c3_records, top_k_per_family=2
        )
        PHASE31._validate_matrix(c4, require_c2_450=True)
        c3_candidates = [item for item in c4["candidates"] if item["search_family"] == "c3"]
        c4_candidates = [item for item in c4["candidates"] if item["search_family"] == "c4"]
        self.assertTrue(c3_candidates)
        self.assertTrue(any(len(item["selected_heads"]) == 1 for item in c4_candidates))
        self.assertTrue(any(len(item["selected_heads"]) > 1 for item in c4_candidates))

    def test_ranking_is_terminal_and_hash_reproducible(self):
        records = successful_records(self.base)
        digest = "b" * 64
        first = AUTOTUNE.build_screening_ranking(
            self.base, records, screening_input_sha256=digest
        )
        second = AUTOTUNE.build_screening_ranking(
            self.base, records, screening_input_sha256=digest
        )
        self.assertEqual(first, second)
        self.assertEqual(first["terminal_count"], 486)
        self.assertEqual(first["ranking_sha256"], second["ranking_sha256"])

    def test_formal_set_includes_every_generated_family(self):
        base_records = successful_records(self.base)
        c3 = AUTOTUNE.extend_phase31_with_c3(self.base, base_records, 2)
        records = base_records + successful_records(c3, "c3", 2000.0)
        c4 = AUTOTUNE.extend_phase31_with_c4(c3, records, 1)
        records += successful_records(c4, "c4", 3000.0)
        ranking = AUTOTUNE.build_screening_ranking(c4, records)
        formal = AUTOTUNE.build_formal_candidate_set(c4, ranking, 3)
        selected_families = {
            item["search_family"] for item in formal["candidates"]
        }
        self.assertEqual(selected_families, set(PHASE31.SEARCH_FAMILIES))
        self.assertEqual(formal["families_without_screening_success"], [])

    def test_same_family_backfill_never_crosses_family(self):
        records = successful_records(self.base)
        ranking = AUTOTUNE.build_screening_ranking(self.base, records)
        formal = AUTOTUNE.build_formal_candidate_set(self.base, ranking, 2)
        qualification = {
            item["candidate_sha256"]: {"valid": False}
            for item in formal["candidates"]
        }
        plans = PHASE31.plan_family_backfill(
            self.base, ranking, formal, qualification
        )
        self.assertTrue(plans)
        for plan in plans:
            self.assertTrue(plan["alternatives"])
            self.assertTrue(
                all(
                    item["search_family"] == plan["search_family"]
                    for item in plan["alternatives"]
                )
            )


class JournalAndArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_journal_uses_phase31_names_and_exact_resume_identity(self):
        output = self.root / "new-run"
        identity = {"sha256": "c" * 64, "payload": {"x": 1}}
        journal = PHASE31.Journal(output, identity, False)
        try:
            def action(directory):
                path = directory / "evidence.json"
                PHASE31.atomic_write_json(path, {"value": 1})
                return {"ok": True}, [path]

            self.assertEqual(journal.run("stage", {"x": 1}, action), {"ok": True})
        finally:
            journal.close()
        self.assertTrue((output / "phase31-state.json").is_file())
        self.assertFalse((output / "phase3-state.json").exists())
        resumed = PHASE31.Journal(output, identity, True)
        try:
            self.assertEqual(resumed.run("stage", {"x": 1}, mock.Mock()), {"ok": True})
        finally:
            resumed.close()
        with self.assertRaises(PHASE31.Phase31Error):
            PHASE31.Journal(output, {"sha256": "d" * 64}, True)

    def test_resume_detects_mutated_stage_artifact(self):
        output = self.root / "mutated"
        identity = {"sha256": "e" * 64}
        journal = PHASE31.Journal(output, identity, False)
        evidence = None
        try:
            def action(directory):
                nonlocal evidence
                evidence = directory / "evidence.json"
                PHASE31.atomic_write_json(evidence, {"value": 1})
                return {}, [evidence]

            journal.run("stage", {}, action)
        finally:
            journal.close()
        PHASE31.atomic_write_json(evidence, {"value": 2})
        resumed = PHASE31.Journal(output, identity, True)
        try:
            with self.assertRaisesRegex(PHASE31.Phase31Error, "artifact changed"):
                resumed.run("stage", {}, mock.Mock())
        finally:
            resumed.close()

    def test_refuses_sealed_phase3_output_tree(self):
        target = PHASE31.SEALED_PHASE3_ROOT / "phase31"
        with self.assertRaisesRegex(PHASE31.Phase31Error, "sealed"):
            PHASE31._assert_independent_output(target, False)

    def test_winner_profile_written_only_for_new_challenger(self):
        output = self.root / "selection-run"
        journal = PHASE31.Journal(output, {"sha256": "f" * 64}, False)
        try:
            profiles = self.root / "profiles"
            profiles.mkdir()
            candidate = {
                "variant_id": "c3_candidate",
                "candidate_sha256": "1" * 64,
            }
            matrix = {"matrix_sha256": "2" * 64, "candidates": [candidate]}
            PHASE31.atomic_write_json(
                profiles / "c3_candidate.json",
                {
                    "deployment": {"enabled": False, "valid": False},
                    "provenance": {
                        "matrix_sha256": "2" * 64,
                        "candidate_sha256": "1" * 64,
                    },
                },
            )
            fps = self.root / "fps.json"
            PHASE31.atomic_write_json(fps, {"passed": True})
            args = types.SimpleNamespace(current_tacker_profile=str(fps))
            challenger = PHASE31._winner_selection(
                journal,
                args,
                matrix,
                {"profiles_dir": str(profiles)},
                {"fps_report": str(fps), "deployment_winner": "c3_candidate"},
            )
            self.assertTrue(
                challenger["document"]["winner_is_new_challenger"]
            )
            self.assertIsNotNone(
                challenger["document"]["disabled_winner_qualification_profile"]
            )
        finally:
            journal.close()

    def test_baseline_winner_reuses_or_omits_profile_without_empty_file(self):
        output = self.root / "baseline-selection"
        journal = PHASE31.Journal(output, {"sha256": "a" * 64}, False)
        try:
            fps = self.root / "formal.json"
            current = self.root / "current.json"
            PHASE31.atomic_write_json(fps, {"passed": True})
            PHASE31.atomic_write_json(current, {"enabled": True})
            result = PHASE31._winner_selection(
                journal,
                types.SimpleNamespace(current_tacker_profile=str(current)),
                {"matrix_sha256": "3" * 64, "candidates": []},
                {"profiles_dir": str(self.root)},
                {"fps_report": str(fps), "deployment_winner": "current_tacker"},
            )["document"]
            self.assertFalse(result["winner_is_new_challenger"])
            self.assertIsNone(result["disabled_winner_qualification_profile"])
            self.assertEqual(result["reused_incumbent_profile"]["path"], str(current.resolve()))
            self.assertFalse(any(output.rglob("winner-qualification-profile.json")))
        finally:
            journal.close()


class ProtocolTests(unittest.TestCase):
    def test_longest_c3_id_passes_shared_128_character_name_contract(self):
        candidate = AUTOTUNE.make_phase31_candidate(
            AUTOTUNE.PACKED_FIRST_LINEAR_ABI_FAMILY,
            AUTOTUNE.HEAD_ORDER,
            5,
            111525,
        )
        name = candidate["variant_id"]
        self.assertGreater(len(name), 64)
        for pattern in (
            PHASE31.NAME_RE,
            PHASE31.PHASE3.NAME_RE,
            BENCHMARK.NAME_PATTERN,
            TOP3.NAME_PATTERN,
        ):
            self.assertIsNotNone(pattern.fullmatch(name))
            self.assertIsNone(pattern.fullmatch("x" * 129))

    def test_failed_multi_candidate_batch_is_retried_one_candidate_at_a_time(self):
        candidates = [
            {
                "candidate_sha256": character * 64,
                "variant_id": "candidate_{}".format(character),
                "search_family": "c2",
                "selected_heads": ["pos", "scales"],
            }
            for character in ("1", "2", "3")
        ]
        batch = {
            "family": "c2",
            "head_count": 2,
            "batch_index": 0,
            "candidates": candidates,
        }
        calls = []

        def fake_screen(*call_args):
            retry_batch = call_args[-2]
            label = call_args[-1]
            calls.append((label, len(retry_batch["candidates"])))
            records = []
            bindings = {}
            for candidate in retry_batch["candidates"]:
                succeeded = len(retry_batch["candidates"]) == 1
                records.append(
                    {
                        "candidate_sha256": candidate["candidate_sha256"],
                        "status": "succeeded" if succeeded else "failed",
                        "score": 1.0 if succeeded else None,
                    }
                )
                bindings[candidate["candidate_sha256"]] = {"label": label}
            return {
                "records": records,
                "measurement_source_bindings": bindings,
                "path": "/tmp/{}.json".format(label),
            }

        args = types.SimpleNamespace(screen_batch_size=30)
        with mock.patch.object(PHASE31, "candidate_batches", return_value=[batch]), mock.patch.object(
            PHASE31, "_screen_batch", side_effect=fake_screen
        ):
            result = PHASE31._screen_families(
                mock.Mock(), args, mock.Mock(), {}, {}, ("c2",), "base"
            )
        self.assertEqual(calls[0][1], 3)
        self.assertEqual([count for _, count in calls[1:]], [1, 1, 1])
        self.assertTrue(all(item["status"] == "succeeded" for item in result["records"]))

    def test_quality_requires_50_unique_views_and_no_fallback(self):
        report = {
            "workload": {"frames": 50, "view_indices": list(range(50))},
            "modes": {
                "tacker": {
                    "per_view": quality_per_view(),
                    "actual_mode": "tacker",
                    "fallback_reason": None,
                    "qualification_executed": True,
                }
            },
        }
        with mock.patch.object(PHASE31.PHASE3, "_validate_quality_report", return_value=True):
            self.assertTrue(
                PHASE31._validate_quality_report(report, mock.Mock(), "/x", 0)
            )
            report["workload"]["view_indices"][-1] = 0
            with self.assertRaisesRegex(PHASE31.Phase31Error, "unique"):
                PHASE31._validate_quality_report(report, mock.Mock(), "/x", 0)

    def test_baseline_quality_strictly_binds_fixed_workload_and_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = types.SimpleNamespace(
                model_path=str(root / "model"),
                source_path=str(root / "source"),
                current_tacker_profile=str(root / "current.json"),
            )
            report = baseline_quality_report(args)
            status = PHASE31._validate_baseline_quality_report(report, args, 0)
            self.assertTrue(all(item["valid"] for item in status.values()))

            changed = json.loads(json.dumps(report))
            changed["workload"]["split"] = "train"
            with self.assertRaisesRegex(PHASE31.Phase31Error, "contract"):
                PHASE31._validate_baseline_quality_report(changed, args, 0)

            changed = json.loads(json.dumps(report))
            changed["tacker_profile"] = str(root / "other.json")
            with self.assertRaisesRegex(PHASE31.Phase31Error, "contract"):
                PHASE31._validate_baseline_quality_report(changed, args, 0)

            changed = json.loads(json.dumps(report))
            changed["modes"]["tacker"]["per_view"][49]["batch_index"] = 0
            with self.assertRaisesRegex(PHASE31.Phase31Error, "0..49"):
                PHASE31._validate_baseline_quality_report(changed, args, 0)

    def test_baseline_quality_preserves_explicit_invalid_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = types.SimpleNamespace(
                model_path=str(root / "model"),
                source_path=str(root / "source"),
                current_tacker_profile=str(root / "current.json"),
            )
            report = baseline_quality_report(args, invalid_mode="two_stream")
            status = PHASE31._validate_baseline_quality_report(report, args, 1)
            self.assertTrue(status["serial"]["valid"])
            self.assertFalse(status["two_stream"]["valid"])
            self.assertTrue(status["current_tacker"]["valid"])
            with self.assertRaisesRegex(PHASE31.Phase31Error, "contract"):
                PHASE31._validate_baseline_quality_report(report, args, 0)

    def test_formal_execution_counts_bind_all_finalist_round_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            finalists = ["c3_x", "c4_y"]
            runs = []
            for name in finalists:
                profile = root / "{}.json".format(name)
                profile.write_text("{}\n", encoding="utf-8")
                for round_index in range(10):
                    metadata_path = root / "{}-{}.metadata.json".format(
                        name, round_index
                    )
                    PHASE31.atomic_write_json(
                        metadata_path,
                        {
                            "schema_version": 1,
                            "kind": "4dgaussians_tacker_render_profile",
                            "passed": True,
                            "execution_mode": "tacker",
                            "actual_execution_mode": "tacker",
                            "profile_frames": 50,
                            "view_indices": list(range(50)),
                            "two_stream_fallback_reason": None,
                            "tacker_fallback_reason": None,
                            "qualification_mode_requested": True,
                            "qualification_mode_executed": True,
                            "tacker_profile": None,
                            "qualification_profile": str(profile.resolve()),
                            "selected_variant_id": name,
                            "pipeline_execution_counts": dict(
                                PHASE31.EXPECTED_FORMAL_EXECUTION_COUNTS
                            ),
                        },
                    )
                    runs.append(
                        {
                            "candidate_name": name,
                            "round_index": round_index,
                            "passed": True,
                            "returncode": 0,
                            "profile_path": str(profile.resolve()),
                            "metadata_path": str(metadata_path.resolve()),
                            "metadata_sha256": PHASE31.sha256_file(metadata_path),
                        }
                    )
            report = {"runs": runs}
            evidence = PHASE31._validate_formal_execution_counts(
                report, finalists, root
            )
            self.assertEqual(len(evidence), 20)
            self.assertTrue(
                all(item["pipeline_execution_counts"]["outputs"] == 50 for item in evidence)
            )

            broken = Path(runs[0]["metadata_path"])
            document = PHASE31.load_json(broken)
            document["pipeline_execution_counts"]["mixed_launches"] = 48
            PHASE31.atomic_write_json(broken, document)
            runs[0]["metadata_sha256"] = PHASE31.sha256_file(broken)
            with self.assertRaisesRegex(PHASE31.Phase31Error, "execution counts"):
                PHASE31._validate_formal_execution_counts(report, finalists, root)

    def test_formal_report_requires_abba_paired_ci_and_reason(self):
        names = ["c3_x", "c4_y"]
        comparisons = []
        for name in names:
            for reference in ("two_stream", "current_tacker"):
                comparisons.append(
                    {
                        "candidate": name,
                        "reference": reference,
                        "paired_bootstrap_95_ci": {"lower": 1.0, "upper": 1.1},
                    }
                )
        report = {
            "kind": "4dgaussians_tacker_fps_benchmark",
            "passed": True,
            "phase0_exit_condition": {"met": True},
            "contract": {"profile_frames": 50, "warmup_frames": 10},
            "schedule": {"strategy": "abba", "trials_per_candidate": 10},
            "candidates": [
                {"name": name}
                for name in ("serial", "two_stream", "current_tacker") + tuple(names)
            ],
            "paired_comparisons": comparisons,
            "deployment_winner": "c3_x",
            "promotion": {"reason": "faster", "reason_code": "promoted", "criteria": {}},
        }
        self.assertEqual(
            PHASE31._validate_formal_report(report, mock.Mock(), names), "c3_x"
        )
        del report["paired_comparisons"][0]["paired_bootstrap_95_ci"]
        with self.assertRaisesRegex(PHASE31.Phase31Error, "paired CI"):
            PHASE31._validate_formal_report(report, mock.Mock(), names)

    def test_invalid_baseline_is_explicit_and_does_not_require_pair(self):
        report = {
            "kind": "4dgaussians_tacker_fps_benchmark",
            "passed": True,
            "phase0_exit_condition": {"met": True},
            "contract": {"profile_frames": 50, "warmup_frames": 10},
            "schedule": {"strategy": "abba", "trials_per_candidate": 10},
            "candidates": [
                {"name": name}
                for name in ("serial", "two_stream", "current_tacker", "c3_x")
            ],
            "correctness_qualifications": {
                "serial": {"valid": True},
                "two_stream": {"valid": True},
                "current_tacker": {"valid": False},
                "c3_x": {"valid": True},
            },
            "excluded_candidates": [
                {"name": "current_tacker", "reason": "correctness_invalid"}
            ],
            "paired_comparisons": [
                {
                    "candidate": "c3_x",
                    "reference": "two_stream",
                    "paired_bootstrap_95_ci": {"lower": 1.0, "upper": 1.2},
                }
            ],
            "deployment_winner": "c3_x",
            "promotion": {
                "reason": "incumbent invalid",
                "reason_code": "replacement",
                "criteria": {},
            },
        }
        self.assertEqual(
            PHASE31._validate_formal_report(report, mock.Mock(), ["c3_x"]),
            "c3_x",
        )
        report["excluded_candidates"] = []
        with self.assertRaisesRegex(PHASE31.Phase31Error, "explicitly"):
            PHASE31._validate_formal_report(report, mock.Mock(), ["c3_x"])

    def test_dry_run_lists_all_staged_families_and_executes_nothing(self):
        args = types.SimpleNamespace()
        plan = PHASE31.dry_run_plan(args, {"sha256": "9" * 64})
        self.assertEqual(plan["kind"], PHASE31.DRY_RUN_KIND)
        self.assertFalse(plan["executes_commands"])
        self.assertEqual(plan["expected_exhaustive_c2_candidates"], 450)
        text = " ".join(plan["stages"])
        for family in ("C0", "C1", "C2", "C3", "C4"):
            self.assertIn(family, text)

    def test_parser_exposes_resume_dry_run_and_all_stop_points(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"--dry-run-plan"', source)
        self.assertIn('"--resume"', source)
        self.assertIn('"--stop-after"', source)
        self.assertEqual(
            PHASE31.STOP_POINTS,
            ("preflight", "base", "c3", "c4", "screen", "quality", "formal", "nsight"),
        )

    def test_runner_never_uses_shell_true(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("shell=True", source)
        self.assertIn("PHASE3.build_benchmark_command", source)

    def test_formal_integrity_claim_is_scoped_to_exact_child_counts(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn(
            '"no_fallback_or_duplicate_validated_by_benchmark": True', source
        )
        self.assertIn(
            '"no_fallback_and_exact_scheduler_counts_validated": True', source
        )


if __name__ == "__main__":
    unittest.main()
