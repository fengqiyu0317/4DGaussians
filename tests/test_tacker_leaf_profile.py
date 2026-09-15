"""CPU-only contracts for the fixed Tacker leaf/Raster profiler."""

import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "profile_tacker_leaves.py"
SPEC = importlib.util.spec_from_file_location("profile_tacker_leaves", MODULE_PATH)
PROFILER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILER)


class _Args:
    pass


class _ShapeOnly:
    def __init__(self, shape):
        self.shape = shape


class _PackedOutput(_ShapeOnly):
    def __init__(self, shape):
        super().__init__(shape)
        self.indices = []

    def __getitem__(self, index):
        self.indices.append(index)
        return "packed-head-{}".format(index)


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


def _disabled_v2_profile(heads=("pos", "scales"), worker_groups=2):
    from tests.test_tacker_pipeline import _load_module

    module = _load_module()
    profile = module.load_tacker_profile()
    persistent_blocks = 80
    resources = {
        "block_threads": 256 + 128 * worker_groups,
        "registers_per_thread": 32,
        "static_shared_memory_bytes": 0,
        "max_threads_per_block": 1024,
        "active_blocks_per_sm": 1,
        "occupancy": 0.5,
    }
    candidate = module.first_linear_candidate_contract(
        "{}_l1_w{}".format("_".join(heads), worker_groups),
        heads,
        worker_groups=worker_groups,
        persistent_blocks=persistent_blocks,
        resources=resources,
    )
    profile["candidates"].append(candidate)
    profile["selected_variant_id"] = candidate["variant_id"]
    profile["manifest"]["persistent_blocks"] = persistent_blocks
    profile["manifest_sha256"] = module.manifest_sha256(profile["manifest"])
    profile["selection"] = None
    profile["deployment"] = {"enabled": False, "valid": False}
    profile["provenance"].update(
        {"candidate_sha256": "4" * 64, "matrix_sha256": "5" * 64}
    )
    profile["profile_sha256"] = module.profile_sha256(profile)
    return module, profile, candidate


def _matrix_qualification_profile(
    abi_family, heads=("pos", "scales"), worker_groups=2
):
    from scripts import tacker_autotune
    from tests.test_tacker_pipeline import _load_module

    module = _load_module()
    matrix = tacker_autotune.build_base_matrix([80])
    candidate = next(
        item
        for item in matrix["candidates"]
        if item["abi_family"] == abi_family
        and (
            abi_family == tacker_autotune.LEGACY_ABI_FAMILY
            or (
                item["selected_heads"] == list(heads)
                and item["worker_groups"] == worker_groups
            )
        )
    )
    abi_version = (
        1 if abi_family == tacker_autotune.LEGACY_ABI_FAMILY else 2
    )
    resources = module._test_resource_requirements(
        abi_version, candidate["worker_groups"]
    )
    profile = tacker_autotune.build_disabled_qualification_profile(
        matrix, candidate, resources, module=module
    )
    return module, tacker_autotune, matrix, profile, candidate


def _variant_metadata():
    return {
        "variant_id": "pos_scales_l1_w2",
        "selected_heads": ["pos", "scales"],
        "worker_groups": 2,
        "persistent_blocks": 80,
        "abi_version": 2,
        "cuda_symbol": "tacker_mix_render_heads_v2",
        "physical_cta_threads": 512,
        "candidate_sha256": "a" * 64,
        "profile_file_sha256": "b" * 64,
        "profile_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
        "profile_path": "/tmp/candidate.json",
        "profile_schema_version": 2,
        "deployment_enabled": False,
        "qualification_profile": True,
        "used_as_deployment": False,
        "resources": {
            "profile_candidate": {
                "registers_per_thread": 32,
                "active_blocks_per_sm": 1,
            },
            "rasterizer_runtime": {
                "physical_threads": 512,
                "occupancy": 0.5,
            },
            "head_runtime": {
                "abi_version": 2,
                "kernels": {},
            },
        },
        "abi": {
            "version": 2,
            "mixed_symbol": "tacker_mix_render_heads_v2",
            "mixed_manifest": {"file_sha256": "e" * 64},
            "head_manifest": {"file_sha256": "f" * 64},
        },
    }


def _legacy_variant_metadata():
    value = _variant_metadata()
    value.update(
        {
            "variant_id": "c0_legacy_pos_l1_pb80",
            "selected_heads": ["pos"],
            "worker_groups": 1,
            "abi_version": 1,
            "cuda_symbol": "tacker_mix_render_head_v1",
            "physical_cta_threads": 384,
        }
    )
    value["abi"] = {
        "version": 1,
        "mixed_symbol": "tacker_mix_render_head_v1",
        "mixed_manifest": {"file_sha256": "e" * 64},
        "head_manifest": {"file_sha256": "f" * 64},
    }
    value["resources"]["rasterizer_runtime"]["physical_threads"] = 384
    value["resources"]["head_runtime"] = {
        "abi_version": 1,
        "physical_threads": 128,
    }
    return value


def _head_numerics():
    return {
        "pos": {
            "passed": True,
            "thresholds": {"atol": 0.002, "rtol": 0.002},
            "per_view_pair": [{"mixed_max_abs": 0.0001}],
        },
        "scales": {
            "passed": True,
            "thresholds": {"atol": 0.002, "rtol": 0.002},
            "per_view_pair": [{"mixed_max_abs": 0.0002}],
        },
    }


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

    def test_variant_documents_bind_profile_resources_and_each_head(self):
        provenance = {
            "workload_source": {"dataset_source_path": "/data/flame_steak"},
            "configuration_source": {
                "path": "/tmp/config.py",
                "file_sha256": "1" * 64,
            },
            "extensions": {
                "rasterizer": {"file_sha256": "2" * 64},
                "head": {"file_sha256": "3" * 64},
            },
        }
        device, raster, leaf = PROFILER.build_measurement_documents(
            self.workload,
            self.device,
            {},
            {
                "passed": True,
                "per_selected_head": _head_numerics(),
            },
            _summary([8.0, 8.2]),
            _summary([8.4, 8.6]),
            _summary([1.1, 1.3]),
            gptb_head=_summary([0.9, 1.0]),
            sample_view_indices=[0, 1],
            persistent_blocks=80,
            variant=_variant_metadata(),
            head_numerics=_head_numerics(),
            provenance=provenance,
            head_sample_view_indices=[1, 2],
            mixed_sample_view_pairs=[
                {"current_view_index": 0, "next_view_index": 1},
                {"current_view_index": 1, "next_view_index": 2},
            ],
        )

        for document in (device, raster, leaf):
            self.assertEqual(document["schema_version"], 2)
            self.assertEqual(document["variant_id"], "pos_scales_l1_w2")
            self.assertEqual(
                document["profile_binding"]["candidate_sha256"], "a" * 64
            )
            self.assertEqual(
                document["resources"]["rasterizer_runtime"][
                    "physical_threads"
                ],
                512,
            )
            self.assertEqual(document["abi"]["version"], 2)
            self.assertEqual(document["provenance"], provenance)
            self.assertFalse(document["variant"]["used_as_deployment"])
        self.assertEqual(
            raster["measurement_config"]["selected_heads"],
            ["pos", "scales"],
        )
        self.assertIn("mixed_full_raster_heads", raster["timings"])
        self.assertAlmostEqual(
            leaf["measurements"]["multi_head_solo_p50_ms"], 1.2
        )
        self.assertEqual(
            leaf["measurements"][
                "multi_head_gptb_p50_ms_diagnostic"
            ],
            0.95,
        )
        self.assertEqual(
            set(leaf["selected_head_numerics"]), {"pos", "scales"}
        )
        self.assertAlmostEqual(
            leaf["selected_head_numerics"]["scales"]["per_view_pair"][0][
                "mixed_max_abs"
            ],
            0.0002,
        )
        self.assertEqual(leaf["sample_view_indices"], [0, 1])
        self.assertEqual(leaf["head_sample_view_indices"], [1, 2])
        self.assertEqual(
            leaf["mixed_sample_view_pairs"][0],
            {"current_view_index": 0, "next_view_index": 1},
        )

    def test_c0_variant_keeps_v1_physical_timing_semantics(self):
        pos_numerics = {"pos": _head_numerics()["pos"]}
        _, raster, leaf = PROFILER.build_measurement_documents(
            self.workload,
            self.device,
            {},
            {"passed": True, "per_selected_head": pos_numerics},
            _summary([8.0]),
            _summary([8.5]),
            _summary([0.8]),
            gptb_head=_summary([0.6]),
            sample_view_indices=[0],
            persistent_blocks=80,
            variant=_legacy_variant_metadata(),
            head_numerics=pos_numerics,
            provenance={"source": "test"},
            head_sample_view_indices=[1],
            mixed_sample_view_pairs=[
                {"current_view_index": 0, "next_view_index": 1}
            ],
        )
        self.assertEqual(raster["schema_version"], 2)
        self.assertIn("forward_with_head call", raster["measurement_semantics"]["mixed_full"])
        self.assertNotIn("mixed_full_raster_heads", raster["timings"])
        self.assertNotIn("multi_head_solo_p50_ms", leaf["measurements"])
        self.assertEqual(set(leaf["selected_head_numerics"]), {"pos"})
        self.assertEqual(leaf["head_sample_view_indices"], [1])

    def test_phase31_variant_documents_bind_packed_and_whole_backends(self):
        contracts = (
            (
                3,
                "tacker_mix_render_packed_heads_v3",
                "packed_first_linear",
                "packed_first_linear_v3",
                "forward_with_packed_heads",
            ),
            (
                4,
                "tacker_mix_render_whole_heads_v4",
                "whole_head",
                "whole_heads_v4",
                "forward_with_whole_heads",
            ),
        )
        for abi_version, symbol, backend, family, method in contracts:
            with self.subTest(backend=backend):
                variant = _variant_metadata()
                variant.update(
                    {
                        "abi_version": abi_version,
                        "cuda_symbol": symbol,
                        "backend": backend,
                        "family": family,
                    }
                )
                variant["abi"]["version"] = abi_version
                _, raster, leaf = PROFILER.build_measurement_documents(
                    self.workload,
                    self.device,
                    {},
                    {"passed": True, "per_selected_head": _head_numerics()},
                    _summary([8.0]),
                    _summary([8.5]),
                    _summary([1.2]),
                    persistent_blocks=80,
                    variant=variant,
                    head_numerics=_head_numerics(),
                    provenance={"source": "test"},
                )
                self.assertEqual(raster["variant"]["backend"], backend)
                self.assertEqual(leaf["variant"]["family"], family)
                self.assertIn(
                    method, raster["measurement_semantics"]["mixed_full"]
                )

    def test_phase31_variant_metadata_rejects_backend_abi_mismatch(self):
        variant = _variant_metadata()
        variant.update(
            {
                "abi_version": 3,
                "cuda_symbol": "tacker_mix_render_packed_heads_v3",
                "backend": "whole_head",
                "family": "packed_first_linear_v3",
            }
        )
        variant["abi"]["version"] = 3
        with self.assertRaisesRegex(ValueError, "backend"):
            PROFILER.build_measurement_documents(
                self.workload,
                self.device,
                {},
                {"passed": True},
                _summary([8.0]),
                _summary([8.5]),
                _summary([1.2]),
                persistent_blocks=80,
                variant=variant,
                head_numerics=_head_numerics(),
                provenance={"source": "test"},
            )

    def test_variant_documents_fail_closed_on_hash_resource_or_head_drift(self):
        base = _variant_metadata()
        cases = []
        bad_hash = copy.deepcopy(base)
        bad_hash["candidate_sha256"] = "not-a-hash"
        cases.append(bad_hash)
        missing_resources = copy.deepcopy(base)
        missing_resources["resources"] = None
        cases.append(missing_resources)
        claims_deployment = copy.deepcopy(base)
        claims_deployment["used_as_deployment"] = True
        cases.append(claims_deployment)
        nonfinite = copy.deepcopy(base)
        nonfinite["resources"]["rasterizer_runtime"]["occupancy"] = math.nan
        cases.append(nonfinite)

        for variant in cases:
            with self.subTest(variant=variant):
                with self.assertRaises(ValueError):
                    PROFILER.build_measurement_documents(
                        self.workload,
                        self.device,
                        {},
                        {"passed": True},
                        _summary([8.0]),
                        _summary([8.5]),
                        _summary([1.2]),
                        persistent_blocks=80,
                        variant=variant,
                        head_numerics=_head_numerics(),
                        provenance={"source": "test"},
                    )

        with self.assertRaisesRegex(ValueError, "exactly the selected"):
            PROFILER.build_measurement_documents(
                self.workload,
                self.device,
                {},
                {"passed": True},
                _summary([8.0]),
                _summary([8.5]),
                _summary([1.2]),
                persistent_blocks=80,
                variant=base,
                head_numerics={"pos": _head_numerics()["pos"]},
                provenance={"source": "test"},
            )


class CandidateProfileContractTest(unittest.TestCase):
    def test_disabled_v2_profile_resolves_unique_multi_head_variant(self):
        module, profile, candidate = _disabled_v2_profile()
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "candidate.json"
            encoded = (
                json.dumps(profile, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")
            path.write_bytes(encoded)

            descriptor = PROFILER.load_candidate_profile_descriptor(
                path, tacker_api=module
            )

        self.assertEqual(descriptor["variant_id"], candidate["variant_id"])
        self.assertEqual(descriptor["selected_heads"], ("pos", "scales"))
        self.assertEqual(descriptor["worker_groups"], 2)
        self.assertEqual(descriptor["persistent_blocks"], 80)
        self.assertEqual(
            descriptor["profile_file_sha256"], hashlib.sha256(encoded).hexdigest()
        )
        self.assertEqual(
            descriptor["candidate_descriptor_sha256"],
            PROFILER._canonical_sha256(candidate),
        )
        self.assertEqual(
            descriptor["candidate_sha256"],
            descriptor["candidate_descriptor_sha256"],
        )
        self.assertEqual(
            descriptor["candidate_identity_kind"],
            "canonical_selected_descriptor",
        )
        self.assertEqual(
            descriptor["source_candidate_sha256_claim"], "4" * 64
        )
        self.assertFalse(
            descriptor["source_candidate_sha256_claim_verified"]
        )
        self.assertIsNone(descriptor["candidate_matrix_sha256"])
        self.assertEqual(descriptor["source_matrix_sha256_claim"], "5" * 64)
        self.assertFalse(descriptor["source_matrix_sha256_claim_verified"])
        self.assertEqual(descriptor["profile_sha256"], profile["profile_sha256"])
        self.assertTrue(descriptor["qualification_profile"])
        self.assertFalse(descriptor["deployment_enabled"])
        self.assertFalse(descriptor["used_as_deployment"])

    def test_matrix_proves_exact_multi_head_member_and_claims(self):
        (
            module,
            autotune,
            matrix,
            profile,
            candidate,
        ) = _matrix_qualification_profile("first_linear_heads_v2")
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            profile_path = root / "profile.json"
            matrix_path = root / "matrix.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            descriptor = PROFILER.load_candidate_profile_descriptor(
                profile_path,
                tacker_api=module,
                candidate_matrix_path=matrix_path,
                autotune_api=autotune,
            )
        self.assertEqual(descriptor["candidate_sha256"], candidate["candidate_sha256"])
        self.assertEqual(descriptor["candidate_matrix_sha256"], matrix["matrix_sha256"])
        self.assertEqual(descriptor["candidate_identity_kind"], "validated_matrix_member")
        self.assertTrue(descriptor["source_candidate_sha256_claim_verified"])
        self.assertTrue(descriptor["source_matrix_sha256_claim_verified"])
        self.assertFalse(descriptor["legacy_physical_abi"])

    def test_matrix_proves_schema_v2_c0_while_preserving_v1_abi(self):
        from scripts import tacker_autotune

        module, autotune, matrix, profile, candidate = (
            _matrix_qualification_profile(tacker_autotune.LEGACY_ABI_FAMILY)
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            profile_path = root / "profile.json"
            matrix_path = root / "matrix.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            descriptor = PROFILER.load_candidate_profile_descriptor(
                profile_path,
                tacker_api=module,
                candidate_matrix_path=matrix_path,
                autotune_api=autotune,
            )
        self.assertTrue(descriptor["legacy_physical_abi"])
        self.assertEqual(descriptor["selected_heads"], ("pos",))
        self.assertEqual(descriptor["worker_groups"], 1)
        self.assertEqual(descriptor["variant"].abi_version, 1)
        self.assertEqual(descriptor["candidate_sha256"], candidate["candidate_sha256"])

    def test_matrix_or_profile_identity_drift_fails_closed(self):
        module, autotune, matrix, profile, _candidate = (
            _matrix_qualification_profile("first_linear_heads_v2")
        )
        tampered_matrix = copy.deepcopy(matrix)
        tampered_matrix["candidates"][0]["worker_groups"] = 99
        bad_claim = copy.deepcopy(profile)
        bad_claim["provenance"]["candidate_sha256"] = "0" * 64
        bad_claim["profile_sha256"] = module.profile_sha256(bad_claim)
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            matrix_path = root / "matrix.json"
            profile_path = root / "profile.json"
            for candidate_profile, candidate_matrix in (
                (profile, tampered_matrix),
                (bad_claim, matrix),
            ):
                profile_path.write_text(
                    json.dumps(candidate_profile), encoding="utf-8"
                )
                matrix_path.write_text(
                    json.dumps(candidate_matrix), encoding="utf-8"
                )
                with self.assertRaises(PROFILER.ProfileContractError):
                    PROFILER.load_candidate_profile_descriptor(
                        profile_path,
                        tacker_api=module,
                        candidate_matrix_path=matrix_path,
                        autotune_api=autotune,
                    )

    def test_candidate_profile_hashes_bind_exact_file_and_candidate(self):
        module, profile, _candidate = _disabled_v2_profile()
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            compact = root / "compact.json"
            pretty = root / "pretty.json"
            compact.write_text(
                json.dumps(profile, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            pretty.write_text(
                json.dumps(profile, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            first = PROFILER.load_candidate_profile_descriptor(
                compact, tacker_api=module
            )
            second = PROFILER.load_candidate_profile_descriptor(
                pretty, tacker_api=module
            )

        self.assertNotEqual(
            first["profile_file_sha256"], second["profile_file_sha256"]
        )
        self.assertEqual(first["profile_sha256"], second["profile_sha256"])
        self.assertEqual(first["candidate_sha256"], second["candidate_sha256"])
        self.assertEqual(
            first["candidate_descriptor_sha256"],
            second["candidate_descriptor_sha256"],
        )

    def test_invalid_or_nonfinite_candidate_profiles_fail_closed(self):
        module, profile, _candidate = _disabled_v2_profile()
        corrupt_hash = copy.deepcopy(profile)
        corrupt_hash["profile_sha256"] = "0" * 64
        baseline = copy.deepcopy(profile)
        baseline["selected_variant_id"] = "serial"
        baseline["profile_sha256"] = module.profile_sha256(baseline)

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            cases = {
                "corrupt": json.dumps(corrupt_hash),
                "baseline": json.dumps(baseline),
                "nonfinite": json.dumps(profile)[:-1] + ', "bad": NaN}',
                "array": "[]",
            }
            for name, payload in cases.items():
                with self.subTest(name=name):
                    path = root / "{}.json".format(name)
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(PROFILER.ProfileContractError):
                        PROFILER.load_candidate_profile_descriptor(
                            path, tacker_api=module
                        )


class ResourceAndInputBindingContractTest(unittest.TestCase):
    def test_phase31_head_resources_select_backend_specific_symbols(self):
        contracts = (
            (
                "packed_first_linear",
                PROFILER.EXPECTED_HEAD_PACKED_GPTB_SYMBOL,
                [128, 256, 384, 512, 640],
                [6, 3, 2, 1, 1],
                256,
            ),
            (
                "whole_head",
                PROFILER.EXPECTED_HEAD_WHOLE_GPTB_SYMBOL,
                [128],
                [6],
                128,
            ),
        )
        for backend, symbol, threads, active, selected_threads in contracts:
            with self.subTest(backend=backend):
                raw = {
                    "abi_version": 2,
                    "kernels": {
                        symbol: {
                            "registers_per_thread": 40,
                            "static_shared_memory_bytes": 0,
                            "local_memory_bytes": 0,
                            "max_threads_per_block": 1024,
                            "ptx_version": 86,
                            "binary_version": 86,
                            "worker_group_threads": threads,
                            "active_blocks_per_sm": active,
                        }
                    },
                }
                resources = PROFILER._normalise_v2_head_runtime_resources(
                    raw,
                    2,
                    {"device_max_threads_per_multiprocessor": 1536},
                    backend=backend,
                )
                self.assertEqual(resources["backend"], backend)
                self.assertEqual(
                    resources["selected_physical_threads"], selected_threads
                )
                self.assertEqual(set(resources["kernels"]), {symbol})

    def test_v2_head_resources_select_exact_worker_group_occupancy(self):
        kernels = {}
        for symbol in (
            PROFILER.EXPECTED_HEAD_MULTI_SOLO_SYMBOL,
            PROFILER.EXPECTED_HEAD_MULTI_GPTB_SYMBOL,
        ):
            kernels[symbol] = {
                "registers_per_thread": 40,
                "static_shared_memory_bytes": 0,
                "local_memory_bytes": 8,
                "max_threads_per_block": 1024,
                "ptx_version": 86,
                "binary_version": 86,
                "worker_group_threads": [128, 256, 384, 512, 640],
                "active_blocks_per_sm": [6, 3, 2, 1, 1],
            }
        resources = PROFILER._normalise_v2_head_runtime_resources(
            {"abi_version": 2, "kernels": kernels},
            2,
            {"device_max_threads_per_multiprocessor": 1536},
        )
        self.assertEqual(resources["selected_physical_threads"], 256)
        for facts in resources["kernels"].values():
            self.assertEqual(facts["registers_per_thread"], 40)
            self.assertEqual(facts["static_shared_memory_bytes"], 0)
            self.assertEqual(facts["local_memory_bytes"], 8)
            self.assertEqual(facts["max_threads_per_block"], 1024)
            self.assertEqual(facts["ptx_version"], 86)
            self.assertEqual(facts["binary_version"], 86)
            self.assertEqual(facts["active_blocks_per_sm"], 3)
            self.assertEqual(facts["physical_threads"], 256)
            self.assertAlmostEqual(facts["occupancy"], 0.5)

    def test_v2_head_resources_fail_closed_on_missing_or_zero_facts(self):
        base = {
            "abi_version": 2,
            "kernels": {
                symbol: {
                    "registers_per_thread": 40,
                    "static_shared_memory_bytes": 0,
                    "local_memory_bytes": 8,
                    "max_threads_per_block": 1024,
                    "ptx_version": 86,
                    "binary_version": 86,
                    "worker_group_threads": [128, 256, 384, 512, 640],
                    "active_blocks_per_sm": [6, 3, 2, 1, 1],
                }
                for symbol in (
                    PROFILER.EXPECTED_HEAD_MULTI_SOLO_SYMBOL,
                    PROFILER.EXPECTED_HEAD_MULTI_GPTB_SYMBOL,
                )
            },
        }
        cases = []
        missing_regs = copy.deepcopy(base)
        del missing_regs["kernels"][
            PROFILER.EXPECTED_HEAD_MULTI_SOLO_SYMBOL
        ]["registers_per_thread"]
        cases.append(missing_regs)
        for field in ("local_memory_bytes", "ptx_version", "binary_version"):
            missing = copy.deepcopy(base)
            del missing["kernels"][
                PROFILER.EXPECTED_HEAD_MULTI_SOLO_SYMBOL
            ][field]
            cases.append(missing)
        zero_ptx = copy.deepcopy(base)
        zero_ptx["kernels"][
            PROFILER.EXPECTED_HEAD_MULTI_SOLO_SYMBOL
        ]["ptx_version"] = 0
        cases.append(zero_ptx)
        wrong_ptx_arch = copy.deepcopy(base)
        wrong_ptx_arch["kernels"][
            PROFILER.EXPECTED_HEAD_MULTI_SOLO_SYMBOL
        ]["ptx_version"] = 80
        cases.append(wrong_ptx_arch)
        wrong_binary_arch = copy.deepcopy(base)
        wrong_binary_arch["kernels"][
            PROFILER.EXPECTED_HEAD_MULTI_GPTB_SYMBOL
        ]["binary_version"] = 90
        cases.append(wrong_binary_arch)
        zero_active = copy.deepcopy(base)
        zero_active["kernels"][
            PROFILER.EXPECTED_HEAD_MULTI_GPTB_SYMBOL
        ]["active_blocks_per_sm"][1] = 0
        cases.append(zero_active)
        too_few_threads = copy.deepcopy(base)
        too_few_threads["kernels"][
            PROFILER.EXPECTED_HEAD_MULTI_SOLO_SYMBOL
        ]["max_threads_per_block"] = 128
        cases.append(too_few_threads)
        missing_thread_slot = copy.deepcopy(base)
        missing_thread_slot["kernels"][
            PROFILER.EXPECTED_HEAD_MULTI_SOLO_SYMBOL
        ]["worker_group_threads"] = [128, 256, 384, 512]
        cases.append(missing_thread_slot)
        missing_active_slot = copy.deepcopy(base)
        missing_active_slot["kernels"][
            PROFILER.EXPECTED_HEAD_MULTI_GPTB_SYMBOL
        ]["active_blocks_per_sm"] = [6, 3, 2, 1]
        cases.append(missing_active_slot)
        invalid_unselected_occupancy = copy.deepcopy(base)
        invalid_unselected_occupancy["kernels"][
            PROFILER.EXPECTED_HEAD_MULTI_SOLO_SYMBOL
        ]["active_blocks_per_sm"][4] = 4
        cases.append(invalid_unselected_occupancy)
        for resources in cases:
            with self.subTest(resources=resources):
                with self.assertRaises(PROFILER.ProfileContractError):
                    PROFILER._normalise_v2_head_runtime_resources(
                        resources,
                        2,
                        {"device_max_threads_per_multiprocessor": 1536},
                    )

    def test_v2_raster_resources_bind_sealed_candidate_and_query_snapshot(self):
        variant = _Args()
        variant.worker_groups = 2
        variant.physical_cta_threads = 512
        raw = {
            "abi_version": 2,
            "worker_groups": 2,
            "physical_threads": 512,
            "device_ordinal": 0,
            "compute_capability_major": 8,
            "compute_capability_minor": 6,
            "multiprocessor_count": 84,
            "device_max_threads_per_block": 1024,
            "device_max_threads_per_multiprocessor": 1536,
            "warp_size": 32,
            "kernel_max_threads_per_block": 1024,
            "registers_per_thread": 32,
            "static_shared_bytes": 0,
            "local_bytes_per_thread": 0,
            "max_dynamic_shared_bytes": 0,
            "active_blocks_per_multiprocessor": 1,
            "active_warps_per_multiprocessor": 16,
            "max_warps_per_multiprocessor": 48,
            "occupancy": 0.5,
            "launch_supported": True,
        }
        first = PROFILER._normalise_raster_runtime_resources(raw, 2, 2)
        sealed = {
            key: value
            for key, value in first.items()
            if key != "launch_supported" and PROFILER._is_finite_number(value)
        }
        candidate = {"resources": sealed}
        stable = PROFILER._normalise_v2_raster_runtime_resources(
            raw, candidate, variant
        )
        self.assertEqual(stable, first)

        wrong_arch = copy.deepcopy(raw)
        wrong_arch["compute_capability_minor"] = 0
        with self.assertRaisesRegex(
            PROFILER.ProfileContractError, "compute_capability_minor"
        ):
            PROFILER._normalise_v2_raster_runtime_resources(
                wrong_arch, candidate, variant
            )

        changed = copy.deepcopy(raw)
        changed["registers_per_thread"] = 40
        with self.assertRaisesRegex(
            PROFILER.ProfileContractError, "sealed candidate"
        ):
            PROFILER._normalise_v2_raster_runtime_resources(
                changed, candidate, variant
            )

    def test_v2_raster_second_query_drift_is_rejected(self):
        variant = _Args()
        variant.worker_groups = 2
        variant.physical_cta_threads = 512
        raw = {
            "abi_version": 2,
            "worker_groups": 2,
            "physical_threads": 512,
            "compute_capability_major": 8,
            "compute_capability_minor": 6,
            "device_max_threads_per_block": 1024,
            "device_max_threads_per_multiprocessor": 1536,
            "kernel_max_threads_per_block": 1024,
            "registers_per_thread": 32,
            "static_shared_bytes": 0,
            "active_blocks_per_multiprocessor": 1,
            "occupancy": 0.5,
            "launch_supported": True,
        }
        first = PROFILER._normalise_raster_runtime_resources(raw, 2, 2)
        candidate = {
            "resources": {
                key: value
                for key, value in first.items()
                if key != "launch_supported"
                and PROFILER._is_finite_number(value)
            }
        }
        support_snapshot = PROFILER._normalise_v2_raster_runtime_resources(
            raw, candidate, variant
        )
        second = copy.deepcopy(raw)
        second["diagnostic_generation"] = "second-query"
        second = PROFILER._normalise_v2_raster_runtime_resources(
            second, candidate, variant
        )
        with self.assertRaisesRegex(
            PROFILER.ProfileContractError, "support gate"
        ):
            PROFILER._require_v2_raster_snapshot_match(
                second, support_snapshot
            )

    def test_checkpoint_and_config_chain_are_hashed_and_reverified(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model = root / "model"
            checkpoint = model / "point_cloud" / "iteration_14000"
            checkpoint.mkdir(parents=True)
            (checkpoint / "point_cloud.ply").write_bytes(b"ply")
            (checkpoint / "deformation.pth").write_bytes(b"weights")
            (model / "cfg_args").write_text("Namespace()\n", encoding="utf-8")
            configs = root / "configs"
            configs.mkdir()
            (configs / "base.py").write_text("value = 1\n", encoding="utf-8")
            (configs / "second.py").write_text("other = 2\n", encoding="utf-8")
            active = configs / "active.py"
            active.write_text(
                "_base_ = ['./base.py', './second.py']\n", encoding="utf-8"
            )
            source = root / "flame_steak"
            source.mkdir()
            (source / "poses_bounds.npy").write_bytes(b"poses")
            (source / "points3D_downsample2.ply").write_bytes(b"points")
            for camera_index, image_count in ((0, 3), (1, 1)):
                video = source / "cam{:02d}.mp4".format(camera_index)
                video.write_bytes(b"video")
                image_dir = source / "cam{:02d}".format(camera_index) / "images"
                image_dir.mkdir(parents=True)
                for image_index in range(image_count):
                    (image_dir / "{:04d}.png".format(image_index)).write_bytes(
                        b"img"
                    )
            args = _Args()
            args.configs = str(active)
            args.split = "test"
            args.view_start = 0
            args.view_stride = 1
            args.views = 2
            dataset = _Args()
            dataset.model_path = str(model)
            dataset.source_path = str(source)

            snapshot = PROFILER._capture_workload_input_snapshot(args, dataset)
            roles = [item["role"] for item in snapshot["files"]]
            self.assertIn("checkpoint.point_cloud", roles)
            self.assertIn("checkpoint.deformation", roles)
            self.assertIn("model.cfg_args", roles)
            self.assertIn("dataset.camera_time_poses", roles)
            self.assertIn("dataset.aabb_point_cloud", roles)
            self.assertGreaterEqual(
                len([role for role in roles if role.startswith("dataset.used_image")]),
                3,
            )
            PROFILER._verify_input_snapshot(snapshot)
            self.assertTrue(snapshot["verified_unchanged_after_measurement"])

            selected_image = next(
                item
                for item in snapshot["files"]
                if item["role"].startswith("dataset.used_image")
            )
            Path(selected_image["path"]).write_bytes(b"IMG")
            with self.assertRaisesRegex(
                PROFILER.ProfileContractError, "changed during profiling"
            ):
                PROFILER._verify_input_snapshot(snapshot)
            Path(selected_image["path"]).write_bytes(b"img")
            PROFILER._verify_input_snapshot(snapshot)

            (checkpoint / "deformation_table.pth").write_bytes(b"appeared")
            with self.assertRaisesRegex(
                PROFILER.ProfileContractError, "changed during profiling"
            ):
                PROFILER._verify_input_snapshot(snapshot)

    def test_execution_snapshot_precedes_config_load_and_supports_base_lists(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = root / "first.py"
            second = root / "second.py"
            theme = root / "theme.py"
            first.write_text("first = 1\n", encoding="utf-8")
            second.write_text("second = 2\n", encoding="utf-8")
            theme.write_text(
                "_base_ = ['first.py', 'second.py']\nvalue = 3\n",
                encoding="utf-8",
            )
            model = root / "model"
            model.mkdir()
            (model / "cfg_args").write_text(
                "Namespace(source_path='dataset')\n", encoding="utf-8"
            )
            args = _Args()
            args.configs = str(theme)
            args.model_path = str(model)
            snapshot = PROFILER._capture_execution_source_snapshot(args)
            self.assertIn(
                "execution.model_cfg_args",
                [item["role"] for item in snapshot["files"]],
            )
            config_paths = [
                Path(item["path"]).name
                for item in snapshot["files"]
                if item["role"].startswith("configuration[")
            ]
            self.assertEqual(config_paths, ["theme.py", "first.py", "second.py"])
            second.write_text("second = 9\n", encoding="utf-8")
            with self.assertRaisesRegex(
                PROFILER.ProfileContractError, "changed during profiling"
            ):
                PROFILER._verify_input_snapshot(snapshot)


class Phase31StandaloneDispatchContractTest(unittest.TestCase):
    def _bundle(self, backend):
        variant = _Args()
        variant.backend = backend
        variant.selected_heads = ("pos", "scales")
        variant.worker_groups = 2
        variant.persistent_blocks = 80
        task = _Args()
        if backend == "packed_first_linear":
            task.shared_head_input = "shared"
            task.packed_head_weights = ("pw0", "pw1")
            task.packed_head_biases = ("pb0", "pb1")
        else:
            task.whole_head_inputs = ("wi0", "wi1")
            task.whole_first_weights = ("fw0", "fw1")
            task.whole_first_biases = ("fb0", "fb1")
            task.whole_tail_weights = ("tw0", "tw1")
            task.whole_tail_biases = ("tb0", "tb1")
        return {"variant": variant, "task": task}

    def test_packed_operands_share_input_and_dispatch_once(self):
        bundle = self._bundle("packed_first_linear")
        operands = PROFILER._variant_head_operands(bundle)
        self.assertEqual([item["input"] for item in operands], ["shared"] * 2)

        class Api:
            def __init__(self):
                self.calls = []

            def head_linear_packed_gptb(self, *args):
                self.calls.append(args)
                return ("out0", "out1")

        api = Api()
        self.assertEqual(
            PROFILER._variant_standalone_outputs(bundle, api),
            ("out0", "out1"),
        )
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(api.calls[0][-2:], (2, 80))

    def test_whole_head_dispatch_preserves_each_tail(self):
        bundle = self._bundle("whole_head")

        class Api:
            def __init__(self):
                self.calls = []

            def whole_head_gptb(self, *args):
                self.calls.append(args)
                return "out{}".format(len(self.calls) - 1)

        api = Api()
        self.assertEqual(
            PROFILER._variant_standalone_outputs(bundle, api),
            ("out0", "out1"),
        )
        self.assertEqual(
            api.calls,
            [
                ("wi0", "fw0", "fb0", "tw0", "tb0", 80),
                ("wi1", "fw1", "fb1", "tw1", "tb1", 80),
            ],
        )

    def _packed_mixed_bundle(self, heads, rows):
        variant = _Args()
        variant.backend = "packed_first_linear"
        variant.selected_heads = tuple(heads)
        task = _Args()
        task.shared_head_input = _ShapeOnly((rows, 128))
        context = _Args()
        context.means3D = _ShapeOnly((rows, 3))
        context.means2D = _ShapeOnly((rows, 3))
        context.screenspace_points = _ShapeOnly((rows, 3))

        class Partition:
            @staticmethod
            def result_from_mixed(_context, mixed_outputs):
                return "raster-result", mixed_outputs

        return {
            "variant": variant,
            "task": task,
            "context": context,
            "partition": Partition(),
        }

    def test_packed_mixed_tensor_slices_h2_and_h5_by_selected_head(self):
        for heads in (
            ("pos", "scales"),
            ("pos", "scales", "rotations", "opacity", "shs"),
        ):
            with self.subTest(head_count=len(heads)):
                bundle = self._packed_mixed_bundle(heads, rows=17)
                packed = _PackedOutput((len(heads), 17, 128))
                raster, by_head = PROFILER._variant_mixed_outputs(
                    bundle, packed
                )
                self.assertEqual(raster, "raster-result")
                self.assertEqual(
                    by_head,
                    {
                        head: "packed-head-{}".format(index)
                        for index, head in enumerate(heads)
                    },
                )
                self.assertEqual(packed.indices, list(range(len(heads))))

    def test_packed_mixed_tensor_rejects_malformed_shapes(self):
        bundle = self._packed_mixed_bundle(("pos", "scales"), rows=17)
        malformed = (
            _ShapeOnly((2, 17)),
            _ShapeOnly((1, 17, 128)),
            _ShapeOnly((2, 18, 128)),
            _ShapeOnly((2, 17, 64)),
            _ShapeOnly((2, -1, 128)),
            _ShapeOnly(("2", 17, 128)),
            ("head-0", "head-1"),
        )
        for packed in malformed:
            with self.subTest(shape=getattr(packed, "shape", None)):
                with self.assertRaises(PROFILER.ProfileContractError):
                    PROFILER._variant_mixed_outputs(bundle, packed)

    def test_packed_mixed_tensor_rejects_disagreeing_context_rows(self):
        bundle = self._packed_mixed_bundle(("pos", "scales"), rows=17)
        bundle["context"].means2D = _ShapeOnly((18, 3))
        with self.assertRaisesRegex(
            PROFILER.ProfileContractError, "row counts disagree"
        ):
            PROFILER._variant_mixed_outputs(
                bundle, _PackedOutput((2, 17, 128))
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
            published_report = _read_json(args.report)
            self.assertTrue(published_report["passed"])
            for label, path in (
                ("device", args.device_output),
                ("raster", args.raster_output),
                ("leaf", args.leaf_output),
            ):
                self.assertEqual(
                    published_report["measurement_output_sha256"][label],
                    hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                )

    def test_passing_run_refuses_any_existing_target_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = _output_args(root)
            Path(args.raster_output).write_text("sentinel", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                PROFILER.write_profile_outputs(
                    {"device": {}},
                    {"measurements": {}},
                    {"measurements": {}},
                    {"passed": True},
                    args,
                )
            self.assertEqual(
                Path(args.raster_output).read_text(encoding="utf-8"),
                "sentinel",
            )
            self.assertFalse(Path(args.device_output).exists())
            self.assertFalse(Path(args.leaf_output).exists())
            self.assertFalse(Path(args.report).exists())

    def test_injected_publication_failure_rolls_back_this_run(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = _output_args(root)
            original = PROFILER._publish_json_no_clobber

            def injected(path, value):
                if Path(path).name == "leaf.json":
                    raise OSError("injected leaf publication failure")
                return original(path, value)

            with mock.patch.object(
                PROFILER, "_publish_json_no_clobber", side_effect=injected
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    PROFILER.write_profile_outputs(
                        {"device": {}},
                        {"measurements": {}},
                        {"measurements": {}},
                        {"passed": True},
                        args,
                    )
            for path in (
                args.device_output,
                args.raster_output,
                args.leaf_output,
                args.report,
            ):
                self.assertFalse(Path(path).exists())

    def test_report_temp_unlink_failure_recovers_as_complete_publication(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = _output_args(root)
            original_unlink = PROFILER.os.unlink
            injected = {"done": False}

            def flaky_unlink(path):
                if (
                    not injected["done"]
                    and Path(path).name.startswith(".report.json-")
                ):
                    injected["done"] = True
                    raise OSError("injected report staging unlink failure")
                return original_unlink(path)

            with mock.patch.object(PROFILER.os, "unlink", side_effect=flaky_unlink):
                written = PROFILER.write_profile_outputs(
                    {"device": {}},
                    {"measurements": {}},
                    {"measurements": {}},
                    {"passed": True, "errors": []},
                    args,
                )
            self.assertTrue(injected["done"])
            self.assertTrue(written["measurement_outputs_written"])
            published_report = _read_json(args.report)
            self.assertTrue(published_report["passed"])
            for label, path in (
                ("device", args.device_output),
                ("raster", args.raster_output),
                ("leaf", args.leaf_output),
            ):
                self.assertTrue(Path(path).is_file())
                self.assertEqual(
                    published_report["measurement_output_sha256"][label],
                    hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                )
            self.assertFalse(any(root.glob(".*.tmp")))

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
        # Use only the ast.parse API available in Python 3.7.  A separate
        # validation command may ask a newer interpreter to enforce its grammar.
        cls.tree = ast.parse(cls.source, str(MODULE_PATH))

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
        self.assertIsNone(defaults.candidate_profile)
        self.assertIsNone(defaults.candidate_matrix)
        self.assertTrue(defaults.mixed_multi_abi.endswith("heads_v2.json"))
        self.assertTrue(
            defaults.mixed_packed_abi.endswith("packed_heads_v3.json")
        )
        self.assertTrue(
            defaults.mixed_whole_abi.endswith("whole_heads_v4.json")
        )
        self.assertTrue(defaults.head_multi_abi.endswith("head_linear_v2.json"))

    def test_source_profiles_real_split_head_and_public_mixed_calls(self):
        for required in (
            "prepare_render_context(",
            "deform_for_render(",
            "prepare_pos_head_task(",
            "fusion_partition.prepare(",
            ".forward_with_head(",
            "forward_with_heads",
            "head_linear_solo(",
            "head_linear_multi_solo(",
            "head_linear_multi_gptb(",
            "head_linear_packed_gptb(",
            "whole_head_gptb(",
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
        self.assertIn('"--candidate-profile"', self.source)
        self.assertIn("qualification_mode=True", self.source)

    def test_known_remote_dataset_basename_is_accepted(self):
        self.assertIn(
            "flame_steak_4dgs_min", PROFILER.EXPECTED_SOURCE_BASENAMES
        )

    def test_cli_snapshots_configs_before_their_actual_loaders(self):
        main_start = self.source.index("def main():")
        main_source = self.source[main_start:]
        snapshot_at = main_source.index(
            "execution_source_snapshot = _capture_execution_source_snapshot(args)"
        )
        cfg_args_at = main_source.index("args = get_combined_args(parser)")
        explicit_config_at = main_source.index("load_config(args.configs)")
        workload_at = main_source.index(
            "workload_input_snapshot = _capture_workload_input_snapshot("
        )
        cuda_at = main_source.index("safe_state(args.quiet)")
        self.assertLess(snapshot_at, cfg_args_at)
        self.assertLess(snapshot_at, explicit_config_at)
        self.assertLess(workload_at, cuda_at)

    def test_no_shell_or_subprocess_and_parseable_source(self):
        self.assertNotIn("subprocess", self.source)
        self.assertNotIn("os.system", self.source)
        self.assertNotIn("shell=True", self.source)
        self.assertIsInstance(self.tree, ast.Module)

    def test_no_per_iteration_device_synchronize_in_timer(self):
        timer = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_time_cuda_calls"
        )
        top_level_lines = [
            node.lineno
            for node in self.tree.body
            if hasattr(node, "lineno") and node.lineno > timer.lineno
        ]
        end_line = min(top_level_lines) - 1 if top_level_lines else len(
            self.source.splitlines()
        )
        timer_source = "\n".join(
            self.source.splitlines()[timer.lineno - 1 : end_line]
        )
        self.assertNotIn("cuda.synchronize", timer_source)
        self.assertIn("ends[-1].synchronize()", timer_source)


if __name__ == "__main__":
    unittest.main()
