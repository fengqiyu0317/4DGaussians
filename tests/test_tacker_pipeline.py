"""CPU-only contracts for the physical 4DGS Tacker pipeline."""

from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class FakeTensor:
    def __init__(self, value=0.0, shape=(2, 1), dtype="float32", name=None):
        self.value = value
        self.shape = shape
        self.dtype = dtype
        self.name = name or str(value)
        self.device = "cuda:0"
        self.is_cuda = True
        self.recorded_streams = []

    def detach(self):
        return self

    def to(self, dtype=None, **_kwargs):
        return FakeTensor(self.value, self.shape, dtype or self.dtype, self.name)

    def contiguous(self):
        return self

    def record_stream(self, stream):
        self.recorded_streams.append(stream.name)

    def unsqueeze(self, _dimension):
        return FakeTensor(self.value, self.shape + (1,), self.dtype, self.name)

    def flatten(self, _dimension):
        return self

    def sin(self):
        return self

    def cos(self):
        return self

    def reshape(self, shape):
        return FakeTensor(self.value, tuple(shape), self.dtype, self.name)

    def __getitem__(self, key):
        del key
        return FakeTensor(self.value, self.shape, self.dtype, self.name)

    def __mul__(self, other):
        value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(self.value * value, self.shape, self.dtype)

    def __add__(self, other):
        value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(self.value + value, self.shape, self.dtype)

    def __gt__(self, other):
        value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(float(self.value > value), self.shape, "bool")


class FakeStream:
    def __init__(self, name, handle, log, device="cuda:0"):
        self.name = name
        self.cuda_stream = handle
        self.device = device
        self.log = log

    def wait_stream(self, stream):
        self.log.append(("wait_stream", self.name, stream.name))

    def wait_event(self, event):
        self.log.append(("wait_event", self.name, event.name))

    def synchronize(self):
        self.log.append(("synchronize", self.name))


class FakeEvent:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def record(self, stream):
        self.log.append(("event", self.name, stream.name))

    def synchronize(self):
        self.log.append(("synchronize_event", self.name))


class FakeCuda:
    def __init__(
        self,
        log,
        available=False,
        capability=(8, 6),
        device_name="NVIDIA RTX A6000",
    ):
        self.log = log
        self.available = available
        self.capability = capability
        self.device_name = device_name
        self.current = FakeStream("consumer", 1, log)
        self.stream_count = 0
        self.event_count = 0

    def is_available(self):
        return self.available

    def get_device_capability(self, _device=None):
        return self.capability

    def get_device_name(self, _device=None):
        return self.device_name

    def current_stream(self, _device=None):
        return self.current

    def Stream(self, device=None):
        name = "private{}".format(self.stream_count)
        result = FakeStream(name, 100 + self.stream_count, self.log, device)
        self.stream_count += 1
        self.log.append(("create_stream", name))
        return result

    def Event(self, blocking=False):
        del blocking
        name = "event{}".format(self.event_count)
        self.event_count += 1
        return FakeEvent(name, self.log)

    @contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous


class GaussianRenderState:
    def __init__(self, means3D, scales, rotations, opacities, shs):
        self.means3D = means3D
        self.scales = scales
        self.rotations = rotations
        self.opacities = opacities
        self.shs = shs

    def record_stream(self, stream):
        for value in vars(self).values():
            if hasattr(value, "record_stream"):
                value.record_stream(stream)


class RenderResult:
    def __init__(self, render, viewspace_points, visibility_filter, radii, depth):
        self.render = render
        self.viewspace_points = viewspace_points
        self.visibility_filter = visibility_filter
        self.radii = radii
        self.depth = depth

    def record_stream(self, stream):
        for value in vars(self).values():
            if hasattr(value, "record_stream"):
                value.record_stream(stream)

    def as_dict(self):
        return {
            "render": self.render,
            "viewspace_points": self.viewspace_points,
            "visibility_filter": self.visibility_filter,
            "radii": self.radii,
            "depth": self.depth,
        }


class DummyTwoStreamRenderer:
    def __init__(self, *_args, **_kwargs):
        self.last_fallback_reason = None
        self.synchronize_calls = 0

    def render_sequence(self, views):
        for view in views:
            yield {"two_stream": view}

    def synchronize(self):
        self.synchronize_calls += 1


class GaussianRasterizer:
    def forward_with_head(self, **_kwargs):
        raise AssertionError("the CPU contract must install a physical stub")

    def forward_with_heads(self, **_kwargs):
        raise AssertionError("the CPU contract must install a physical v2 stub")


def _load_module(available=False, grad_enabled=False, capability=(8, 6)):
    log = []
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = FakeTensor
    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    torch_stub.cuda = FakeCuda(log, available=available, capability=capability)
    torch_stub.is_grad_enabled = lambda: grad_enabled
    torch_stub.ones_like = lambda value: FakeTensor(1.0, value.shape, value.dtype)
    torch_stub.zeros_like = lambda value: FakeTensor(0.0, value.shape, value.dtype)
    torch_stub.cat = lambda values, _dimension: values[0]

    capabilities = {
        "stream_aware": True,
        "mixed_render_head_abi": 1,
        "mixed_render_head": True,
        "mixed_symbol": "tacker_mix_render_head_v1",
        "mixed_manifest_sha256": (
            "231c90c429321b2673b88ecd09efb40b6aedda7a23f3e061a2bcedec06d44426"
        ),
        "head_manifest_sha256": (
            "24570aa6e67e8b9b10fa94524fec4dc03a4eb3fdc3bf822af34c2c52ce4937ac"
        ),
        "sm_target": "sm_86",
        "mixed_threads": 384,
        "raster_threads": 256,
        "head_threads": 128,
        "head_thread_base": 256,
        "head_features": 128,
        "raster_named_barrier_id": 1,
        "rasterizer_commit": "e49506654e8e11ed8a62d22bcb693e943fdecacf",
        "mixed_render_heads_abi": 2,
        "mixed_render_heads": True,
        "mixed_multi_symbol": "tacker_mix_render_heads_v2",
        "mixed_multi_manifest_sha256": (
            "310b15957c5920773bb03a61a37c5771f6d4570393061ece4e1805fd20989056"
        ),
        "head_multi_manifest_sha256": (
            "9d6a1558acd6b642b975bcabe22abcbe3fd7242e4c9e0d635636ef4d2eb5da7f"
        ),
        "max_head_tasks": 5,
        "max_mixed_heads": 5,
        "min_worker_groups": 1,
        "max_worker_groups": 5,
        "worker_group_threads": 128,
        "head_descriptor_named_barrier_id": 2,
        "supported_worker_groups": [1, 2, 3, 4, 5],
        "mixed_threads_by_worker_groups": {
            worker_groups: 256 + 128 * worker_groups
            for worker_groups in range(1, 6)
        },
        "resource_query": "tacker_resource_requirements",
    }
    diff_stub = types.ModuleType("diff_gaussian_rasterization")
    diff_stub.GaussianRasterizer = GaussianRasterizer
    diff_stub.tacker_capabilities = lambda: dict(capabilities)

    def resource_requirements(abi_version=2, worker_groups=1):
        physical_threads = (
            384 if abi_version == 1 else 256 + 128 * worker_groups
        )
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
            "warp_size": 32,
            "kernel_max_threads_per_block": 1024,
            "registers_per_thread": 32,
            "static_shared_bytes": 0,
            "local_bytes_per_thread": 0,
            "max_dynamic_shared_bytes": 0,
            "active_blocks_per_multiprocessor": 1,
            "active_warps_per_multiprocessor": physical_threads // 32,
            "max_warps_per_multiprocessor": 48,
            "occupancy": 0.5,
            "launch_supported": True,
        }

    diff_stub.tacker_resource_requirements = resource_requirements
    diff_stub.tacker_variant_resources = lambda worker_groups: {
        **resource_requirements(2, worker_groups),
        "block_threads": 256 + 128 * worker_groups,
        "registers_per_thread": 32,
        "static_shared_memory_bytes": 0,
        "max_threads_per_block": 1024,
        "active_blocks_per_sm": 1,
    }

    renderer_stub = types.ModuleType("gaussian_renderer")
    renderer_stub.GaussianRasterizer = GaussianRasterizer
    renderer_stub.GaussianRenderState = GaussianRenderState
    renderer_stub.RenderResult = RenderResult
    renderer_stub.TwoStreamRenderer = DummyTwoStreamRenderer
    renderer_stub._record_context_stream = lambda _context, _stream: None
    renderer_stub.deform_for_render = lambda *_args, **_kwargs: None
    renderer_stub.prepare_render_context = lambda *_args, **_kwargs: None
    renderer_stub.rasterize_state = lambda *_args, **_kwargs: None

    name = "tacker_pipeline_cpu_contract"
    path = ROOT / "gaussian_renderer" / "tacker_pipeline.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "torch": torch_stub,
        "diff_gaussian_rasterization": diff_stub,
        "gaussian_renderer": renderer_stub,
        name: module,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    module._test_log = log
    module._test_torch = torch_stub
    module._test_capabilities = capabilities
    module._test_resource_requirements = resource_requirements
    return module


def _valid_profile(module):
    with module.DEFAULT_PROFILE_PATH.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    trials = {
        "serial": [80.0] * 10,
        "two_stream": [90.0] * 10,
        "legacy_pos_l1": [100.0] * 10,
    }
    for candidate in profile["candidates"]:
        variant_id = candidate["variant_id"]
        candidate["benchmark_candidate_name"] = (
            "current_tacker" if variant_id == "legacy_pos_l1" else variant_id
        )
        candidate["correctness"] = {
            "valid": True,
            "actual_execution_mode": candidate["execution_mode"],
            "fallback_reason": None,
        }
        if candidate["execution_mode"] == "tacker":
            candidate["resources"] = {
                "block_threads": 384,
                "registers_per_thread": 32,
                "static_shared_memory_bytes": 0,
                "max_threads_per_block": 1024,
                "active_blocks_per_sm": 1,
                "occupancy": 0.5,
            }
            candidate["correctness"].update(
                {
                    "psnr_drop_db": 0.05,
                    "ssim_drop": 0.0001,
                    "lpips_increase": 0.0001,
                }
            )
            # Explicitly diagnostic: neither value may veto this fastest,
            # correctness-valid candidate.
            candidate["diagnostics"] = {
                "raster_slowdown_pct": 25.0,
                "mixed_p50_ms": 6.0,
                "solo_raster_p50_ms": 4.0,
                "solo_head_p50_ms": 1.0,
            }
        candidate["performance"] = {
            "trial_count": len(trials[variant_id]),
            "round_indices": list(range(len(trials[variant_id]))),
            "throughput_fps_trials": trials[variant_id],
            "median_throughput_fps": trials[variant_id][0],
        }
    profile["selection"] = {
        "eligible_variant_ids": ["legacy_pos_l1", "two_stream", "serial"],
        "ineligible_variant_ids": [],
        "global_median_fps_ranking": [
            "legacy_pos_l1",
            "two_stream",
            "serial",
        ],
        "experimental_winner_variant_id": "legacy_pos_l1",
        "deployment_winner_variant_id": "legacy_pos_l1",
        "incumbent_variant_id": "legacy_pos_l1",
        "equivalence": {
            "fraction": 0.005,
            "candidate_variant_ids_in_preference_order": ["legacy_pos_l1"],
            "preferred_variant_id": "legacy_pos_l1",
        },
        "promotion": {
            "challenger_variant_id": "legacy_pos_l1",
            "incumbent_variant_id": "legacy_pos_l1",
            "minimum_median_fps_ratio": 1.01,
            "minimum_bootstrap_lower_exclusive": 1.0,
            "required": False,
            "passed": True,
            "decision": "incumbent_is_global_winner",
            "candidate_evaluations": [],
        },
    }
    profile["deployment"] = {"enabled": True, "valid": True}
    profile["provenance"] = {
        "template": False,
        "input_sha256": {"synthetic": "0" * 64},
    }
    profile["profile_sha256"] = module.profile_sha256(profile)
    return profile


def _paired_comparison(
    module,
    challenger,
    incumbent,
    resamples=10000,
    seed=0,
):
    challenger_performance = challenger["performance"]
    incumbent_performance = incumbent["performance"]
    challenger_trials = challenger_performance["throughput_fps_trials"]
    incumbent_trials = incumbent_performance["throughput_fps_trials"]
    label = "{}-vs-{}".format(
        challenger["benchmark_candidate_name"],
        incumbent["benchmark_candidate_name"],
    )
    lower, upper = module._paired_bootstrap_interval(
        challenger_trials,
        incumbent_trials,
        resamples,
        seed,
        label,
    )
    ratios = [
        challenger_fps / incumbent_fps
        for challenger_fps, incumbent_fps in zip(
            challenger_trials, incumbent_trials
        )
    ]
    return {
        "candidate": challenger["benchmark_candidate_name"],
        "reference": incumbent["benchmark_candidate_name"],
        "round_indices": list(challenger_performance["round_indices"]),
        "paired_fps_ratios": ratios,
        "median_paired_fps_ratio": module.statistics.median(ratios),
        "median_fps_ratio": (
            challenger_performance["median_throughput_fps"]
            / incumbent_performance["median_throughput_fps"]
        ),
        "paired_bootstrap_95_ci": {
            "lower": lower,
            "upper": upper,
            "confidence": 0.95,
            "resamples": resamples,
            "seed": seed,
            "statistic": "median(candidate_fps)/median(reference_fps)",
            "resampling_unit": "paired_round",
            "percentile_method": "linear_type_7",
        },
    }


def _promotion_evaluation(module, challenger, incumbent, two_stream):
    comparison = _paired_comparison(module, challenger, incumbent)
    ratio = comparison["median_fps_ratio"]
    lower = comparison["paired_bootstrap_95_ci"]["lower"]
    challenger_fps = challenger["performance"]["median_throughput_fps"]
    floor_ratios = {
        "two_stream": (
            challenger_fps
            / two_stream["performance"]["median_throughput_fps"]
        ),
        incumbent["variant_id"]: (
            challenger_fps
            / incumbent["performance"]["median_throughput_fps"]
        ),
    }
    ratio_passed = ratio >= 1.01
    ci_passed = lower > 1.0
    floor_passed = all(value >= 1.0 for value in floor_ratios.values())
    return {
        "candidate_variant_id": challenger["variant_id"],
        "passed": ratio_passed and ci_passed and floor_passed,
        "paired_comparison": comparison,
        "criteria": {
            "median_fps_ratio": {
                "observed": ratio,
                "required_min": 1.01,
                "passed": ratio_passed,
            },
            "paired_bootstrap_95_ci_lower": {
                "observed": lower,
                "required_strictly_greater_than": 1.0,
                "passed": ci_passed,
            },
            "baseline_fps_ratios": {
                "observed": floor_ratios,
                "required_min": 1.0,
                "passed": floor_passed,
            },
        },
    }


def _profile_with_challenger(module, challenger_trials):
    """Build admission-shaped evidence for one current-ABI challenger."""

    profile = _valid_profile(module)
    incumbent = profile["candidates"][2]
    two_stream = profile["candidates"][1]
    challenger = deepcopy(incumbent)
    challenger["variant_id"] = "future"
    challenger["benchmark_candidate_name"] = "future"
    challenger["persistent_blocks"] = 80
    challenger["performance"] = {
        "trial_count": len(challenger_trials),
        "round_indices": list(range(len(challenger_trials))),
        "throughput_fps_trials": list(challenger_trials),
        "median_throughput_fps": module.statistics.median(challenger_trials),
    }
    profile["candidates"].append(challenger)
    ranking = sorted(
        profile["candidates"],
        key=lambda candidate: (
            -candidate["performance"]["median_throughput_fps"],
            candidate["benchmark_candidate_name"],
        ),
    )
    top_fps = ranking[0]["performance"]["median_throughput_fps"]
    equivalent = [
        candidate
        for candidate in ranking
        if (
            top_fps - candidate["performance"]["median_throughput_fps"]
        )
        / top_fps
        <= 0.005
    ]
    equivalent.sort(key=lambda candidate: candidate["benchmark_candidate_name"])
    equivalent_ids = [candidate["variant_id"] for candidate in equivalent]
    evaluation = _promotion_evaluation(
        module, challenger, incumbent, two_stream
    )
    selected = challenger if evaluation["passed"] else incumbent
    profile["selected_variant_id"] = selected["variant_id"]
    profile["manifest"]["persistent_blocks"] = selected["persistent_blocks"]
    profile["manifest_sha256"] = module.manifest_sha256(profile["manifest"])
    promotion = {
        "challenger_variant_id": challenger["variant_id"],
        "incumbent_variant_id": incumbent["variant_id"],
        "minimum_median_fps_ratio": 1.01,
        "minimum_bootstrap_lower_exclusive": 1.0,
        "required": True,
        "passed": evaluation["passed"],
        "decision": (
            "promoted_equivalent_challenger"
            if evaluation["passed"]
            else "retained_incumbent"
        ),
        "candidate_evaluations": [evaluation],
        "paired_comparison": evaluation["paired_comparison"],
        "median_fps_ratio": evaluation["criteria"]["median_fps_ratio"][
            "observed"
        ],
        "paired_bootstrap_95_ci_lower": evaluation["criteria"]
        ["paired_bootstrap_95_ci_lower"]["observed"],
        "criteria": evaluation["criteria"],
    }
    profile["selection"] = {
        "eligible_variant_ids": [candidate["variant_id"] for candidate in ranking],
        "ineligible_variant_ids": [],
        "global_median_fps_ranking": [
            candidate["variant_id"] for candidate in ranking
        ],
        "experimental_winner_variant_id": ranking[0]["variant_id"],
        "deployment_winner_variant_id": selected["variant_id"],
        "incumbent_variant_id": incumbent["variant_id"],
        "equivalence": {
            "fraction": 0.005,
            "candidate_variant_ids_in_preference_order": equivalent_ids,
            "preferred_variant_id": equivalent_ids[0],
        },
        "promotion": promotion,
    }
    profile["profile_sha256"] = module.profile_sha256(profile)
    return profile


def _valid_legacy_profile(module):
    manifest = {
        "rasterizer_commit": module.RASTERIZER_COMMIT,
        "pair_key": module.PAIR_KEY,
        "cuda_arch": "sm_86",
        "compute_capability": [8, 6],
        "gpu_name": "NVIDIA RTX A6000",
        "workload": "flame_steak",
        "iteration": 14000,
        "gaussian_count": 111525,
        "resolution": [1352, 1014],
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
        "manifest_sha256": module.manifest_sha256(manifest),
        "thresholds": {
            "raster_slowdown_pct_max": 5.0,
            "mixed_p50_strictly_less_than_solo_sum": True,
            "end_to_end_ratio_max": 1.0,
            "psnr_drop_db_max": 0.05,
            "ssim_drop_max": 0.0001,
            "lpips_increase_max": 0.0001,
        },
        "admission": {"enabled": True, "valid": True},
        "measurements": {
            "raster_slowdown_pct": 50.0,
            "mixed_p50_ms": 7.0,
            "solo_raster_p50_ms": 4.0,
            "solo_head_p50_ms": 1.0,
            "tacker_end_to_end_p50_ms": 12.0,
            "two_stream_end_to_end_p50_ms": 10.0,
            "psnr_drop_db": 0.05,
            "ssim_drop": 0.0001,
            "lpips_increase": 0.0001,
        },
    }


class ReLU:
    def __init__(self, log=None, label="relu"):
        self.log = log
        self.label = label

    def __call__(self, value):
        if self.log is not None:
            self.log.append((self.label, value.value))
        return value


class Linear:
    def __init__(self, in_features, out_features, value_fn=None, log=None, label="linear"):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = FakeTensor(0, (out_features, in_features), "float32")
        self.bias = FakeTensor(0, (out_features,), "float32")
        self.value_fn = value_fn or (lambda value: value)
        self.log = log
        self.label = label
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        if self.log is not None:
            self.log.append((self.label, value.value))
        return FakeTensor(self.value_fn(value.value), value.shape, "float32")


class Sequential:
    def __init__(self, *modules):
        self.modules = list(modules)

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Sequential(*self.modules[item])
        return self.modules[item]

    def __call__(self, value):
        for module in self.modules:
            value = module(value)
        return value


def _exact_head(output_width):
    return Sequential(
        ReLU(),
        Linear(128, 128),
        ReLU(),
        Linear(128, output_width),
    )


def _exact_model(training=False):
    args = types.SimpleNamespace(
        no_dx=False,
        no_ds=False,
        no_dr=False,
        no_do=False,
        no_dshs=False,
        apply_rotation=False,
        static_mlp=False,
        empty_voxel=False,
    )
    network = types.SimpleNamespace(
        W=128,
        D=0,
        args=args,
        pos_deform=_exact_head(3),
        scales_deform=_exact_head(3),
        rotations_deform=_exact_head(4),
        opacity_deform=_exact_head(1),
        shs_deform=_exact_head(48),
    )
    deformation = types.SimpleNamespace(
        training=training,
        deformation_net=network,
        pos_poc=FakeTensor(),
        rotation_scaling_poc=FakeTensor(),
    )
    return types.SimpleNamespace(
        _deformation=deformation,
        get_xyz=FakeTensor(shape=(111525, 3)),
        scaling_activation=lambda value: value,
        rotation_activation=lambda value: value,
        opacity_activation=lambda value: value,
    )


class ProfileContractTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def test_repository_profile_is_hashed_and_fail_closed(self):
        profile = self.module.load_tacker_profile()
        self.assertEqual(
            self.module.manifest_sha256(profile["manifest"]),
            profile["manifest_sha256"],
        )
        self.assertEqual(
            self.module.tacker_profile_admission_reason(profile),
            "Tacker profile deployment is disabled",
        )

    def test_hash_mismatch_and_weakened_correctness_gate_are_rejected(self):
        profile = _valid_profile(self.module)
        profile["manifest"]["persistent_blocks"] = 3
        with self.assertRaisesRegex(self.module.TackerProfileError, "SHA-256"):
            self.module.validate_tacker_profile(profile)

        profile = _valid_profile(self.module)
        profile["correctness_thresholds"]["psnr_drop_db_max"] = 0.051
        profile["profile_sha256"] = self.module.profile_sha256(profile)
        with self.assertRaisesRegex(self.module.TackerProfileError, "weakens"):
            self.module.validate_tacker_profile(profile)

    def test_v2_requires_tile_resource_and_equivalence_contracts(self):
        for field, message in (
            ("tile_shape", "tile_shape"),
            ("resources", "resources"),
        ):
            profile = _valid_profile(self.module)
            del profile["candidates"][2][field]
            profile["profile_sha256"] = self.module.profile_sha256(profile)
            with self.assertRaisesRegex(self.module.TackerProfileError, message):
                self.module.validate_tacker_profile(profile)

        profile = _valid_profile(self.module)
        del profile["selection"]["equivalence"]
        profile["profile_sha256"] = self.module.profile_sha256(profile)
        with self.assertRaisesRegex(
            self.module.TackerProfileError, "selection.equivalence"
        ):
            self.module.validate_tacker_profile(profile)

        profile = _valid_profile(self.module)
        profile["note"] = float("nan")
        with self.assertRaisesRegex(
            self.module.TackerProfileError, "non-canonical data"
        ):
            self.module.validate_tacker_profile(profile)

    def test_oversized_json_integer_fails_closed_without_overflowing(self):
        profile = _valid_profile(self.module)
        profile["candidates"][2]["performance"]["throughput_fps_trials"][
            0
        ] = 10**309
        profile["profile_sha256"] = self.module.profile_sha256(profile)

        reason = self.module.tacker_profile_admission_reason(profile)

        self.assertIsInstance(reason, str)
        self.assertIn("finite", reason)

    def test_qos_diagnostics_are_not_runtime_gates_but_quality_is(self):
        profile = _valid_profile(self.module)
        self.assertIsNone(self.module.tacker_profile_admission_reason(profile))

        selected = next(
            item
            for item in profile["candidates"]
            if item["variant_id"] == "legacy_pos_l1"
        )
        selected["diagnostics"].update(
            {
                "raster_slowdown_pct": 500.0,
                "mixed_p50_ms": 50.0,
                "solo_raster_p50_ms": 1.0,
                "solo_head_p50_ms": 1.0,
            }
        )
        profile["profile_sha256"] = self.module.profile_sha256(profile)
        self.assertIsNone(self.module.tacker_profile_admission_reason(profile))

        for key, value in (
            ("psnr_drop_db", 0.051),
            ("ssim_drop", 0.00011),
            ("lpips_increase", 0.00011),
        ):
            failing = deepcopy(profile)
            failing_selected = next(
                item
                for item in failing["candidates"]
                if item["variant_id"] == "legacy_pos_l1"
            )
            failing_selected["correctness"][key] = value
            failing["profile_sha256"] = self.module.profile_sha256(failing)
            self.assertIn("correctness threshold", self.module.tacker_profile_admission_reason(failing))

    def test_schema_v1_is_read_only_compatible_without_old_performance_vetoes(self):
        profile = _valid_legacy_profile(self.module)

        self.assertIs(self.module.validate_tacker_profile(profile), profile)
        self.assertIsNone(self.module.tacker_profile_admission_reason(profile))
        self.assertEqual(
            self.module.selected_tacker_candidate(profile)["variant_id"],
            "legacy_pos_l1",
        )

    def test_selection_and_measurement_tampering_breaks_profile_hash(self):
        profile = _valid_profile(self.module)
        profile["provenance"]["input_sha256"]["synthetic"] = "1" * 64
        with self.assertRaisesRegex(self.module.TackerProfileError, "selection SHA-256"):
            self.module.validate_tacker_profile(profile)

        profile = _valid_profile(self.module)
        profile["selected_variant_id"] = "two_stream"
        profile["selection"]["deployment_winner_variant_id"] = "two_stream"
        profile["profile_sha256"] = self.module.profile_sha256(profile)
        with self.assertRaisesRegex(
            self.module.TackerProfileError,
            "cannot select a baseline|experimental FPS argmax",
        ):
            self.module.validate_tacker_profile(profile)

        profile = _valid_profile(self.module)
        profile["candidates"][2]["performance"]["throughput_fps_trials"][0] = 999.0
        with self.assertRaisesRegex(
            self.module.TackerProfileError,
            "median_throughput_fps|selection SHA-256",
        ):
            self.module.validate_tacker_profile(profile)

        profile = _valid_profile(self.module)
        profile["candidates"][2]["performance"] = {
            "trial_count": 10,
            "round_indices": list(range(10)),
            "throughput_fps_trials": [85.0] * 10,
            "median_throughput_fps": 85.0,
        }
        profile["profile_sha256"] = self.module.profile_sha256(profile)
        with self.assertRaisesRegex(
            self.module.TackerProfileError,
            "ranking disagrees",
        ):
            self.module.validate_tacker_profile(profile)

    def test_v2_selects_one_of_multiple_current_abi_candidates(self):
        profile = _valid_profile(self.module)
        alternative = deepcopy(profile["candidates"][2])
        alternative["variant_id"] = "pos_l1_pb80"
        alternative["benchmark_candidate_name"] = "future"
        alternative["persistent_blocks"] = 80
        alternative["performance"] = {
            "trial_count": 10,
            "round_indices": list(range(10)),
            "throughput_fps_trials": [101.0] * 10,
            "median_throughput_fps": 101.0,
        }
        profile["candidates"].append(alternative)
        profile["selected_variant_id"] = alternative["variant_id"]
        profile["manifest"]["persistent_blocks"] = 80
        profile["manifest_sha256"] = self.module.manifest_sha256(
            profile["manifest"]
        )
        profile["selection"]["experimental_winner_variant_id"] = alternative[
            "variant_id"
        ]
        profile["selection"]["eligible_variant_ids"] = [
            "pos_l1_pb80",
            "legacy_pos_l1",
            "two_stream",
            "serial",
        ]
        profile["selection"]["global_median_fps_ranking"] = list(
            profile["selection"]["eligible_variant_ids"]
        )
        profile["selection"]["deployment_winner_variant_id"] = alternative[
            "variant_id"
        ]
        profile["selection"]["equivalence"] = {
            "fraction": 0.005,
            "candidate_variant_ids_in_preference_order": [
                alternative["variant_id"]
            ],
            "preferred_variant_id": alternative["variant_id"],
        }
        incumbent = profile["candidates"][2]
        two_stream = profile["candidates"][1]
        evaluation = _promotion_evaluation(
            self.module, alternative, incumbent, two_stream
        )
        profile["selection"]["promotion"] = {
            "challenger_variant_id": alternative["variant_id"],
            "incumbent_variant_id": incumbent["variant_id"],
            "minimum_median_fps_ratio": 1.01,
            "minimum_bootstrap_lower_exclusive": 1.0,
            "required": True,
            "passed": True,
            "decision": "promoted_equivalent_challenger",
            "candidate_evaluations": [evaluation],
            "paired_comparison": evaluation["paired_comparison"],
            "median_fps_ratio": evaluation["criteria"]["median_fps_ratio"][
                "observed"
            ],
            "paired_bootstrap_95_ci_lower": evaluation["criteria"]
            ["paired_bootstrap_95_ci_lower"]["observed"],
            "criteria": evaluation["criteria"],
        }
        profile["profile_sha256"] = self.module.profile_sha256(profile)

        self.assertIs(self.module.validate_tacker_profile(profile), profile)
        self.assertEqual(
            self.module.selected_tacker_candidate(profile)["persistent_blocks"],
            80,
        )

    def test_runtime_uses_benchmark_name_for_exact_and_equivalence_ties(self):
        profile = _valid_profile(self.module)
        incumbent = profile["candidates"][2]
        two_stream = profile["candidates"][1]
        challengers = []
        for benchmark_name, variant_id, persistent_blocks in (
            ("alpha", "zz_variant", 80),
            ("zeta", "aa_variant", 96),
        ):
            challenger = deepcopy(incumbent)
            challenger["benchmark_candidate_name"] = benchmark_name
            challenger["variant_id"] = variant_id
            challenger["persistent_blocks"] = persistent_blocks
            challenger["performance"] = {
                "trial_count": 10,
                "round_indices": list(range(10)),
                "throughput_fps_trials": [102.0] * 10,
                "median_throughput_fps": 102.0,
            }
            profile["candidates"].append(challenger)
            challengers.append(challenger)

        preferred = challengers[0]
        evaluations = [
            _promotion_evaluation(
                self.module, challenger, incumbent, two_stream
            )
            for challenger in challengers
        ]
        selected_evaluation = evaluations[0]
        profile["selected_variant_id"] = preferred["variant_id"]
        profile["manifest"]["persistent_blocks"] = preferred[
            "persistent_blocks"
        ]
        profile["manifest_sha256"] = self.module.manifest_sha256(
            profile["manifest"]
        )
        profile["selection"] = {
            "eligible_variant_ids": [
                "zz_variant",
                "aa_variant",
                "legacy_pos_l1",
                "two_stream",
                "serial",
            ],
            "ineligible_variant_ids": [],
            "global_median_fps_ranking": [
                "zz_variant",
                "aa_variant",
                "legacy_pos_l1",
                "two_stream",
                "serial",
            ],
            "experimental_winner_variant_id": "zz_variant",
            "deployment_winner_variant_id": "zz_variant",
            "incumbent_variant_id": "legacy_pos_l1",
            "equivalence": {
                "fraction": 0.005,
                "candidate_variant_ids_in_preference_order": [
                    "zz_variant",
                    "aa_variant",
                ],
                "preferred_variant_id": "zz_variant",
            },
            "promotion": {
                "challenger_variant_id": "zz_variant",
                "incumbent_variant_id": "legacy_pos_l1",
                "minimum_median_fps_ratio": 1.01,
                "minimum_bootstrap_lower_exclusive": 1.0,
                "required": True,
                "passed": True,
                "decision": "promoted_equivalent_challenger",
                "candidate_evaluations": evaluations,
                "paired_comparison": selected_evaluation[
                    "paired_comparison"
                ],
                "median_fps_ratio": selected_evaluation["criteria"][
                    "median_fps_ratio"
                ]["observed"],
                "paired_bootstrap_95_ci_lower": selected_evaluation[
                    "criteria"
                ]["paired_bootstrap_95_ci_lower"]["observed"],
                "criteria": selected_evaluation["criteria"],
            },
        }
        profile["profile_sha256"] = self.module.profile_sha256(profile)

        self.assertIs(self.module.validate_tacker_profile(profile), profile)
        self.assertEqual(
            self.module.selected_tacker_candidate(profile)[
                "benchmark_candidate_name"
            ],
            "alpha",
        )

    def test_runtime_recomputes_promotion_and_rejects_forged_incumbent_retention(self):
        profile = _profile_with_challenger(self.module, [102.0] * 10)
        incumbent = profile["candidates"][2]
        evaluation = profile["selection"]["promotion"][
            "candidate_evaluations"
        ][0]
        self.assertTrue(evaluation["passed"])

        # Forge a self-consistent-looking retained-incumbent decision while
        # leaving the raw trials and paired evidence untouched.  Runtime must
        # derive the actual passing challenger instead of trusting `passed`.
        profile["selected_variant_id"] = incumbent["variant_id"]
        profile["selection"]["deployment_winner_variant_id"] = incumbent[
            "variant_id"
        ]
        profile["selection"]["promotion"]["passed"] = False
        profile["selection"]["promotion"]["decision"] = "retained_incumbent"
        profile["manifest"]["persistent_blocks"] = incumbent[
            "persistent_blocks"
        ]
        profile["manifest_sha256"] = self.module.manifest_sha256(
            profile["manifest"]
        )
        profile["profile_sha256"] = self.module.profile_sha256(profile)

        with self.assertRaisesRegex(
            self.module.TackerProfileError,
            "deployment winner disagrees with recomputed incumbent promotion",
        ):
            self.module.validate_tacker_profile(profile)

    def test_runtime_recomputes_bootstrap_and_rejects_forged_ci(self):
        trials = [80.0] * 4 + [102.0] * 6
        one_sample_lower, _ = self.module._paired_bootstrap_interval(
            trials,
            [100.0] * 10,
            1,
            3,
            "future-vs-current_tacker",
        )
        self.assertGreater(one_sample_lower, 1.0)
        profile = _profile_with_challenger(self.module, trials)
        promotion = profile["selection"]["promotion"]
        evaluation = promotion["candidate_evaluations"][0]
        self.assertGreaterEqual(
            evaluation["criteria"]["median_fps_ratio"]["observed"], 1.01
        )
        self.assertLessEqual(
            evaluation["criteria"]["paired_bootstrap_95_ci_lower"]["observed"],
            1.0,
        )

        # Forge every recorded CI-derived decision field, including the top
        # summary.  The original trials still determine a failing lower bound.
        fake_lower = 1.001
        comparison = evaluation["paired_comparison"]
        comparison["paired_bootstrap_95_ci"]["lower"] = fake_lower
        evaluation["criteria"]["paired_bootstrap_95_ci_lower"].update(
            {"observed": fake_lower, "passed": True}
        )
        evaluation["passed"] = True
        challenger = profile["candidates"][-1]
        profile["selected_variant_id"] = challenger["variant_id"]
        profile["selection"]["deployment_winner_variant_id"] = challenger[
            "variant_id"
        ]
        promotion.update(
            {
                "passed": True,
                "decision": "promoted_equivalent_challenger",
                "paired_bootstrap_95_ci_lower": fake_lower,
            }
        )
        profile["manifest"]["persistent_blocks"] = challenger[
            "persistent_blocks"
        ]
        profile["manifest_sha256"] = self.module.manifest_sha256(
            profile["manifest"]
        )
        profile["profile_sha256"] = self.module.profile_sha256(profile)

        with self.assertRaisesRegex(
            self.module.TackerProfileError,
            "paired bootstrap lower.*whole-run trials",
        ):
            self.module.validate_tacker_profile(profile)

    def test_runtime_rejects_nonformal_bootstrap_configuration(self):
        profile = _profile_with_challenger(self.module, [102.0] * 10)
        interval = profile["selection"]["promotion"][
            "candidate_evaluations"
        ][0]["paired_comparison"]["paired_bootstrap_95_ci"]
        interval["resamples"] = 1
        profile["profile_sha256"] = self.module.profile_sha256(profile)

        with self.assertRaisesRegex(
            self.module.TackerProfileError, "10000 resamples and seed 0"
        ):
            self.module.validate_tacker_profile(profile)

    def test_valid_incumbent_retention_cannot_waive_two_stream_floor(self):
        profile = _valid_profile(self.module)
        incumbent = profile["candidates"][2]
        two_stream = profile["candidates"][1]
        two_stream["performance"] = {
            "trial_count": 10,
            "round_indices": list(range(10)),
            "throughput_fps_trials": [100.5] * 10,
            "median_throughput_fps": 100.5,
        }
        evaluation = _promotion_evaluation(
            self.module, two_stream, incumbent, two_stream
        )
        self.assertFalse(evaluation["passed"])
        profile["selection"].update(
            {
                "eligible_variant_ids": [
                    "two_stream",
                    "legacy_pos_l1",
                    "serial",
                ],
                "global_median_fps_ranking": [
                    "two_stream",
                    "legacy_pos_l1",
                    "serial",
                ],
                "experimental_winner_variant_id": "two_stream",
                "deployment_winner_variant_id": "legacy_pos_l1",
                "equivalence": {
                    "fraction": 0.005,
                    "candidate_variant_ids_in_preference_order": [
                        "legacy_pos_l1",
                        "two_stream",
                    ],
                    "preferred_variant_id": "legacy_pos_l1",
                },
                "promotion": {
                    "challenger_variant_id": "two_stream",
                    "incumbent_variant_id": "legacy_pos_l1",
                    "minimum_median_fps_ratio": 1.01,
                    "minimum_bootstrap_lower_exclusive": 1.0,
                    "required": False,
                    "passed": False,
                    "decision": "retained_incumbent",
                    "candidate_evaluations": [evaluation],
                    "paired_comparison": evaluation["paired_comparison"],
                    "median_fps_ratio": evaluation["criteria"]
                    ["median_fps_ratio"]["observed"],
                    "paired_bootstrap_95_ci_lower": evaluation["criteria"]
                    ["paired_bootstrap_95_ci_lower"]["observed"],
                    "criteria": evaluation["criteria"],
                },
            }
        )
        profile["profile_sha256"] = self.module.profile_sha256(profile)

        with self.assertRaisesRegex(
            self.module.TackerProfileError, "slower than two_stream"
        ):
            self.module.validate_tacker_profile(profile)

    def test_v2_incumbent_variant_is_not_hard_coded_to_legacy_id(self):
        profile = _valid_profile(self.module)
        incumbent = profile["candidates"][2]
        incumbent["variant_id"] = "pos_l1_pb7000_v2"
        profile["selected_variant_id"] = incumbent["variant_id"]
        profile["selection"].update(
            {
                "eligible_variant_ids": [
                    incumbent["variant_id"],
                    "two_stream",
                    "serial",
                ],
                "global_median_fps_ranking": [
                    incumbent["variant_id"],
                    "two_stream",
                    "serial",
                ],
                "experimental_winner_variant_id": incumbent["variant_id"],
                "deployment_winner_variant_id": incumbent["variant_id"],
                "incumbent_variant_id": incumbent["variant_id"],
                "equivalence": {
                    "fraction": 0.005,
                    "candidate_variant_ids_in_preference_order": [
                        incumbent["variant_id"]
                    ],
                    "preferred_variant_id": incumbent["variant_id"],
                },
                "promotion": {
                    "challenger_variant_id": incumbent["variant_id"],
                    "incumbent_variant_id": incumbent["variant_id"],
                    "minimum_median_fps_ratio": 1.01,
                    "minimum_bootstrap_lower_exclusive": 1.0,
                    "required": False,
                    "passed": True,
                    "decision": "incumbent_is_global_winner",
                    "candidate_evaluations": [],
                },
            }
        )
        profile["profile_sha256"] = self.module.profile_sha256(profile)

        self.assertIs(self.module.validate_tacker_profile(profile), profile)

    def test_invalid_incumbent_does_not_waive_two_stream_fps_floor(self):
        profile = _valid_profile(self.module)
        incumbent = profile["candidates"][2]
        incumbent["correctness"] = {"valid": False}
        incumbent["performance"] = None

        challenger = deepcopy(incumbent)
        challenger["variant_id"] = "replacement"
        challenger["benchmark_candidate_name"] = "replacement"
        challenger["correctness"] = {
            "valid": True,
            "actual_execution_mode": "tacker",
            "fallback_reason": None,
            "psnr_drop_db": 0.0,
            "ssim_drop": 0.0,
            "lpips_increase": 0.0,
        }
        challenger["performance"] = {
            "trial_count": 10,
            "round_indices": list(range(10)),
            "throughput_fps_trials": [89.6] * 10,
            "median_throughput_fps": 89.6,
        }
        challenger["selection_metadata"] = {"abi_complexity": 0.0}
        profile["candidates"][1]["selection_metadata"] = {
            "abi_complexity": 1.0
        }
        profile["candidates"].append(challenger)
        profile["selected_variant_id"] = "replacement"
        profile["selection"] = {
            "eligible_variant_ids": ["two_stream", "replacement", "serial"],
            "ineligible_variant_ids": ["legacy_pos_l1"],
            "global_median_fps_ranking": [
                "two_stream",
                "replacement",
                "serial",
            ],
            "experimental_winner_variant_id": "two_stream",
            "deployment_winner_variant_id": "replacement",
            "incumbent_variant_id": "legacy_pos_l1",
            "equivalence": {
                "fraction": 0.005,
                "candidate_variant_ids_in_preference_order": [
                    "replacement",
                    "two_stream",
                ],
                "preferred_variant_id": "replacement",
            },
            "promotion": {
                "challenger_variant_id": "replacement",
                "incumbent_variant_id": "legacy_pos_l1",
                "minimum_median_fps_ratio": 1.01,
                "minimum_bootstrap_lower_exclusive": 1.0,
                "required": True,
                "passed": True,
                "decision": "selected_best_valid_candidate_incumbent_invalid",
                "candidate_evaluations": [],
            },
        }
        profile["profile_sha256"] = self.module.profile_sha256(profile)

        with self.assertRaisesRegex(
            self.module.TackerProfileError, "slower than two_stream"
        ):
            self.module.validate_tacker_profile(profile)

        # The invalid-incumbent path deliberately waives the 1%/bootstrap
        # stability gates, but accepts the same preferred replacement once it
        # clears the physical two-stream fallback's measured throughput.
        challenger["performance"].update(
            {
                "throughput_fps_trials": [90.1] * 10,
                "median_throughput_fps": 90.1,
            }
        )
        profile["selection"].update(
            {
                "eligible_variant_ids": ["replacement", "two_stream", "serial"],
                "global_median_fps_ranking": [
                    "replacement",
                    "two_stream",
                    "serial",
                ],
                "experimental_winner_variant_id": "replacement",
            }
        )
        profile["profile_sha256"] = self.module.profile_sha256(profile)
        self.assertIs(self.module.validate_tacker_profile(profile), profile)


class SupportGateContractTest(unittest.TestCase):
    def test_exact_contract_admits_and_specific_changes_explain_fallback(self):
        module = _load_module(available=True, grad_enabled=False)
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        profile = _valid_profile(module)

        self.assertIsNone(
            module.tacker_support_reason(
                pc,
                pipe,
                profile,
                stage="fine",
                cam_type="dynerf",
                workload_name="flame_steak",
                iteration=14000,
            )
        )

        self.assertIn(
            "workload name",
            module.tacker_support_reason(
                pc,
                pipe,
                profile,
                stage="fine",
                cam_type="dynerf",
                workload_name=None,
                iteration=14000,
            ),
        )

        pc._deformation.deformation_net.args.no_dx = True
        self.assertIn(
            "no_dx",
            module.tacker_support_reason(
                pc,
                pipe,
                profile,
                stage="fine",
                cam_type="dynerf",
                workload_name="flame_steak",
                iteration=14000,
            ),
        )
        pc._deformation.deformation_net.args.no_dx = False
        module._test_torch.cuda.capability = (8, 0)
        self.assertIn(
            "8.6",
            module.tacker_support_reason(
                pc,
                pipe,
                profile,
                stage="fine",
                cam_type="dynerf",
                workload_name="flame_steak",
                iteration=14000,
            ),
        )
        module._test_torch.cuda.capability = (8, 6)
        module._test_torch.cuda.device_name = "NVIDIA GeForce RTX 3090"
        self.assertIn(
            "RTX A6000",
            module.tacker_support_reason(
                pc,
                pipe,
                profile,
                stage="fine",
                cam_type="dynerf",
                workload_name="flame_steak",
                iteration=14000,
            ),
        )

    def test_camera_resolution_is_bound_to_profile(self):
        module = _load_module()
        profile = _valid_profile(module)
        matching = types.SimpleNamespace(image_width=1352, image_height=1014)
        mismatch = types.SimpleNamespace(image_width=800, image_height=800)
        self.assertIsNone(module._view_profile_reason(matching, profile))
        self.assertIn("resolution", module._view_profile_reason(mismatch, profile))

    def test_legacy_capability_identity_and_resources_fail_closed(self):
        module = _load_module(available=True, grad_enabled=False)
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        profile = _valid_profile(module)
        common = {
            "stage": "fine",
            "cam_type": "dynerf",
            "workload_name": "flame_steak",
            "iteration": 14000,
        }

        selected = module.selected_tacker_candidate(profile)
        selected["resources"]["optional_build_diagnostic"] = None
        profile["profile_sha256"] = module.profile_sha256(profile)
        self.assertIsNone(
            module.tacker_support_reason(pc, pipe, profile, **common)
        )
        for field, replacement in (
            ("mixed_symbol", "wrong_symbol"),
            ("mixed_manifest_sha256", "f" * 64),
            ("head_manifest_sha256", "f" * 64),
        ):
            capabilities = dict(module._test_capabilities)
            capabilities[field] = replacement
            with self.subTest(capability=field), mock.patch.object(
                module, "_query_rasterizer_capabilities", return_value=capabilities
            ):
                self.assertIn(
                    field,
                    module.tacker_support_reason(pc, pipe, profile, **common),
                )

        resource_cases = []
        changed = module._test_resource_requirements(1, 1)
        changed["registers_per_thread"] = 33
        resource_cases.append((changed, "registers_per_thread changed"))
        zero = module._test_resource_requirements(1, 1)
        zero["active_blocks_per_multiprocessor"] = 0
        zero["occupancy"] = 0.0
        resource_cases.append((zero, "zero runtime occupancy"))
        too_large = module._test_resource_requirements(1, 1)
        too_large["kernel_max_threads_per_block"] = 383
        resource_cases.append((too_large, "compiled maximum thread count"))
        unsupported = module._test_resource_requirements(1, 1)
        unsupported["launch_supported"] = False
        resource_cases.append((unsupported, "not launch-supported"))
        for resources, message in resource_cases:
            with self.subTest(resource=message), mock.patch.object(
                module,
                "_query_rasterizer_variant_resources",
                return_value=resources,
            ):
                self.assertIn(
                    message,
                    module.tacker_support_reason(pc, pipe, profile, **common),
                )

    def test_qualification_skips_only_measured_admission(self):
        module = _load_module(available=True, grad_enabled=False)
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        candidate = module.load_tacker_profile()
        common = {
            "stage": "fine",
            "cam_type": "dynerf",
            "workload_name": "flame_steak",
            "iteration": 14000,
        }

        self.assertEqual(
            module.tacker_support_reason(pc, pipe, candidate, **common),
            "Tacker profile deployment is disabled",
        )
        selected = module.selected_tacker_candidate(candidate)
        selected["resources"] = {
            "block_threads": 384,
            "registers_per_thread": 32,
            "static_shared_memory_bytes": 0,
            "max_threads_per_block": 1024,
            "active_blocks_per_sm": 1,
            "occupancy": 0.5,
        }
        for index, profile_candidate in enumerate(candidate["candidates"]):
            if profile_candidate["variant_id"] == selected["variant_id"]:
                candidate["candidates"][index] = selected
                break
        candidate["profile_sha256"] = module.profile_sha256(candidate)
        self.assertIsNone(
            module.tacker_support_reason(
                pc,
                pipe,
                candidate,
                qualification_mode=True,
                **common
            )
        )

        bad = deepcopy(candidate)
        bad["manifest"]["pair_key"] = "wrong"
        self.assertIn(
            "qualification profile",
            module.tacker_support_reason(
                pc,
                pipe,
                bad,
                qualification_mode=True,
                **common
            ),
        )


class SplitHeadContractTest(unittest.TestCase):
    def test_selected_linear_is_not_repeated_and_physical_output_enters_suffix(self):
        module = _load_module(available=True, grad_enabled=False)
        log = []
        selected = Linear(
            128,
            128,
            value_fn=lambda _value: (_ for _ in ()).throw(
                AssertionError("selected Linear was executed by Python")
            ),
            log=log,
            label="selected",
        )
        suffix = Linear(
            128,
            3,
            value_fn=lambda value: value + 30.0,
            log=log,
            label="pos_suffix",
        )
        pos_head = Sequential(
            ReLU(log, "pos_prefix_relu"),
            selected,
            ReLU(log, "pos_suffix_relu"),
            suffix,
        )

        class OtherHead:
            def __init__(self, label, value):
                self.label = label
                self.value = value

            def __call__(self, _hidden):
                log.append((self.label, self.value))
                return FakeTensor(self.value, (2, 1), "float32")

        class StaticMask:
            def __call__(self, _hidden):
                return FakeTensor(0.5, (2, 1), "float32")

        args = types.SimpleNamespace(
            static_mlp=True,
            empty_voxel=False,
            no_ds=False,
            no_dr=False,
            no_do=False,
            no_dshs=False,
            apply_rotation=False,
        )
        network = types.SimpleNamespace(
            args=args,
            pos_deform=pos_head,
            scales_deform=OtherHead("scale_head", 4.0),
            rotations_deform=OtherHead("rotation_head", 5.0),
            opacity_deform=OtherHead("opacity_head", 7.0),
            shs_deform=OtherHead("shs_head", 8.0),
            static_mlp=StaticMask(),
            query_time=lambda *_args: FakeTensor(10.0, (2, 128), "float32"),
        )
        deformation = types.SimpleNamespace(
            deformation_net=network,
            pos_poc=FakeTensor(),
            rotation_scaling_poc=FakeTensor(),
        )
        pc = types.SimpleNamespace(
            _deformation=deformation,
            scaling_activation=lambda value: value,
            rotation_activation=lambda value: value,
            opacity_activation=lambda value: value,
        )
        context = types.SimpleNamespace(
            means3D=FakeTensor(2.0, (2, 3)),
            scales=FakeTensor(3.0, (2, 3)),
            rotations=FakeTensor(4.0, (2, 4)),
            opacity=FakeTensor(6.0, (2, 1)),
            shs=FakeTensor(8.0, (2, 16, 3)),
            timestamp=FakeTensor(0.25, (2, 1)),
        )
        deform_stream = module._test_torch.cuda.Stream(device="cuda:0")
        prefix_ready = FakeEvent("prefix_ready", log)
        weight = FakeTensor(1.0, (128, 128), "float16")
        bias = FakeTensor(0.0, (128,), "float32")

        with mock.patch.object(module, "_poc_fre", side_effect=lambda value, _poc: value):
            task = module.prepare_pos_head_task(
                context,
                pc,
                deform_stream,
                head_weight=weight,
                head_bias=bias,
                prefix_ready=prefix_ready,
            )
        self.assertEqual(selected.calls, 0)
        prefix_index = log.index(("event", "prefix_ready", deform_stream.name))
        self.assertLess(prefix_index, log.index(("scale_head", 4.0)))

        physical_output = FakeTensor(70.0, (2, 128), "float32")
        state = module.finish_pos_head_task(task, pc, physical_output)

        self.assertEqual(selected.calls, 0)
        self.assertIn(("pos_suffix", 70.0), log)
        self.assertEqual(state.means3D.value, 101.0)  # 2*0.5 + (70+30)
        self.assertEqual(state.scales.value, 5.5)  # 3*0.5 + 4
        self.assertEqual(state.rotations.value, 9.0)  # rotation has no mask
        self.assertEqual(state.opacities.value, 10.0)  # 6*0.5 + 7
        self.assertEqual(state.shs.value, 12.0)  # 8*0.5 + 8

    def test_physical_binding_adapter_uses_the_fixed_keyword_abi(self):
        module = _load_module()
        captured = {}

        class Rasterizer:
            def forward_with_head(self, **kwargs):
                captured.update(kwargs)
                return "image", "radii", "depth", "head"

        context = types.SimpleNamespace(
            rasterizer=Rasterizer(),
            means2D="means2d",
            colors_precomp=None,
            cov3D_precomp=None,
        )
        state = GaussianRenderState(
            means3D="means3d",
            scales="scales",
            rotations="rotations",
            opacities="opacity",
            shs="shs",
        )
        task = types.SimpleNamespace(
            head_input="head_input",
            head_weight="head_weight",
            head_bias="head_bias",
        )

        outputs = module._forward_with_head(context, state, task, 17)

        self.assertEqual(outputs, ("image", "radii", "depth", "head"))
        self.assertEqual(
            captured,
            {
                "means3D": "means3d",
                "means2D": "means2d",
                "opacities": "opacity",
                "head_input": "head_input",
                "head_weight": "head_weight",
                "head_bias": "head_bias",
                "shs": "shs",
                "colors_precomp": None,
                "scales": "scales",
                "rotations": "rotations",
                "cov3D_precomp": None,
                "persistent_blocks": 17,
            },
        )


class PipelineOrderingContractTest(unittest.TestCase):
    def test_two_frame_submission_is_d0_prefix1_mixed_suffix1_r1(self):
        module = _load_module(available=True, grad_enabled=False)
        operations = []
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )

        class Context:
            def __init__(self, view):
                self.viewpoint_camera = view
                self.screenspace_points = FakeTensor(name="screen" + view)
                self.rasterizer = object()

        class Task:
            def __init__(self, context, prefix_ready):
                self.context = context
                self.prefix_ready = prefix_ready
                self.head_input = FakeTensor(name="head_input" + context.viewpoint_camera)
                self.head_weight = FakeTensor(shape=(128, 128))
                self.head_bias = FakeTensor(shape=(128,))
                self.head_output = None

            def record_stream(self, _stream):
                pass

        def prepare(view, *_args, **_kwargs):
            return Context(view)

        def full_deform(context, _pc, stage="fine"):
            del stage
            operations.append("D" + context.viewpoint_camera)
            tensor = FakeTensor(name="state" + context.viewpoint_camera)
            return GaussianRenderState(tensor, tensor, tensor, tensor, tensor)

        def prefix(context, _pc, _stream, **kwargs):
            operations.append("prefix" + context.viewpoint_camera)
            kwargs["prefix_ready"].record(module._test_torch.cuda.current)
            return Task(context, kwargs["prefix_ready"])

        def mixed(context, _state, task, _persistent):
            operations.append(
                "R{}+H{}".format(
                    context.viewpoint_camera,
                    task.context.viewpoint_camera,
                )
            )
            tensor = FakeTensor(name="mixed" + context.viewpoint_camera)
            return tensor, FakeTensor(1), tensor, FakeTensor(9, (2, 128), "float32")

        def suffix(task, _pc, head_output):
            operations.append("suffix" + task.context.viewpoint_camera)
            self_value = head_output
            return GaussianRenderState(
                self_value, self_value, self_value, self_value, self_value
            )

        def solo(context, _state):
            operations.append("R" + context.viewpoint_camera)
            tensor = FakeTensor(name="solo" + context.viewpoint_camera)
            return RenderResult(tensor, tensor, FakeTensor(1), FakeTensor(1), tensor)

        with mock.patch.object(module, "tacker_support_reason", return_value=None), mock.patch.object(
            module, "_view_profile_reason", return_value=None
        ), mock.patch.object(
            module, "_cache_pos_head_parameters", return_value=(FakeTensor(), FakeTensor())
        ), mock.patch.object(
            module, "prepare_render_context", side_effect=prepare
        ), mock.patch.object(
            module, "deform_for_render", side_effect=full_deform
        ), mock.patch.object(
            module, "prepare_pos_head_task", side_effect=prefix
        ), mock.patch.object(
            module, "_forward_with_head", side_effect=mixed
        ), mock.patch.object(
            module, "finish_pos_head_task", side_effect=suffix
        ), mock.patch.object(
            module, "rasterize_state", side_effect=solo
        ):
            renderer = module.TackerRenderer(
                pc,
                pipe,
                bg_color=FakeTensor(),
                cam_type="dynerf",
                profile_override=module.load_tacker_profile(),
                workload_name="flame_steak",
                iteration=14000,
                qualification_mode=True,
            )
            outputs = list(renderer.render_sequence(iter(("0", "1"))))

        self.assertEqual(len(outputs), 2)
        self.assertEqual(
            operations,
            ["D0", "prefix1", "R0+H1", "suffix1", "R1"],
        )
        self.assertTrue(renderer.last_used_tacker)
        self.assertIsNone(renderer.last_fallback_reason)
        self.assertTrue(renderer.last_qualification_mode)
        self.assertEqual(renderer.actual_execution_mode, "tacker")

        renderer.synchronize()
        synchronized = [
            entry[1]
            for entry in module._test_log
            if entry[0] == "synchronize"
        ]
        self.assertEqual(synchronized, ["private0", "private1", "consumer"])

        with mock.patch.object(
            module,
            "_view_profile_reason",
            return_value="camera resolution does not match the admitted profile",
        ):
            with self.assertRaisesRegex(RuntimeError, "changed mid-sequence"):
                renderer._require_profile_view(object())

    def test_admission_failure_delegates_and_reports_reason(self):
        module = _load_module(available=False, grad_enabled=False)
        delegated = []

        class Fallback:
            def __init__(self, *_args, **_kwargs):
                self.last_fallback_reason = "CUDA is unavailable"
                self.synchronize_calls = 0

            def render_sequence(self, views):
                for view in views:
                    delegated.append(view)
                    yield {"fallback": view}

            def synchronize(self):
                self.synchronize_calls += 1

        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        reason = "measured profile did not pass admission"
        with mock.patch.object(module, "TwoStreamRenderer", Fallback), mock.patch.object(
            module, "tacker_support_reason", return_value=reason
        ), mock.patch.object(
            module, "_cache_pos_head_parameters", return_value=(FakeTensor(), FakeTensor())
        ):
            renderer = module.TackerRenderer(
                pc,
                pipe,
                bg_color=FakeTensor(),
                cam_type="dynerf",
            )
            outputs = list(renderer.render_sequence(iter(("0", "1"))))

        self.assertEqual(outputs, [{"fallback": "0"}, {"fallback": "1"}])
        self.assertEqual(delegated, ["0", "1"])
        self.assertEqual(renderer.last_fallback_reason, reason)
        self.assertFalse(renderer.last_used_tacker)
        self.assertEqual(module._test_torch.cuda.stream_count, 0)
        self.assertEqual(renderer.fallback_backend_reason, "CUDA is unavailable")
        self.assertEqual(renderer.actual_execution_mode, "serial")
        renderer.synchronize()
        self.assertEqual(renderer._fallback_renderer.synchronize_calls, 1)

    def test_qualification_mode_requires_explicit_override(self):
        module = _load_module(available=False, grad_enabled=False)
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        with mock.patch.object(
            module, "_cache_pos_head_parameters", return_value=(FakeTensor(), FakeTensor())
        ):
            renderer = module.TackerRenderer(
                pc,
                pipe,
                bg_color=FakeTensor(),
                cam_type="dynerf",
                workload_name="flame_steak",
                iteration=14000,
                qualification_mode=True,
            )

        self.assertIn("explicit profile_override", renderer.fallback_reason)
        self.assertFalse(renderer.last_qualification_mode)
        self.assertEqual(renderer.actual_execution_mode, "not_run")
        renderer.synchronize()
        self.assertIn(("synchronize_event", "event0"), module._test_log)


if __name__ == "__main__":
    unittest.main()
