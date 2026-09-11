"""Inference-only Raster(t) + deformation-head(t+1) Tacker pipeline.

The production path is deliberately fail closed.  It is enabled only by a
hashed schema-v2 profile whose selected candidate passed correctness
qualification and offline whole-run FPS selection, plus an exact
model/rasterizer contract.  Read-only schema-v1 profiles remain runnable
during migration, but their historical Raster-QoS diagnostics are not
reinterpreted as runtime gates.
When any part of that contract is missing, :class:`TackerRenderer` delegates
the complete sequence to ``TwoStreamRenderer`` and exposes the reason.

Schema-v1 and the deployed Phase-1 candidate still use the original positional
head ABI.  Schema-v2 candidates may instead select the first ``Linear(128,
128)`` from any non-empty subset of the five deformation heads.  A partition
object owns the prefix, physical mixed launch, parallel work, suffix, and
stream-lifetime contract, so selected Linear nodes are never executed a second
time by PyTorch.
"""

from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

import diff_gaussian_rasterization as _rasterizer_module

try:
    from . import (
        GaussianRasterizer,
        GaussianRenderState,
        RenderResult,
        TwoStreamRenderer,
        _record_context_stream,
        deform_for_render,
        prepare_render_context,
        rasterize_state,
    )
except (ImportError, ValueError):  # Supports direct import in CPU contract tests.
    from gaussian_renderer import (  # type: ignore
        GaussianRasterizer,
        GaussianRenderState,
        RenderResult,
        TwoStreamRenderer,
        _record_context_stream,
        deform_for_render,
        prepare_render_context,
        rasterize_state,
    )


PROFILE_SCHEMA_VERSION = 2
LEGACY_PROFILE_SCHEMA_VERSION = 1
RASTERIZER_COMMIT = "e49506654e8e11ed8a62d22bcb693e943fdecacf"
PAIR_KEY = "raster.render_leaf+deformation.pos_deform[1].linear_128x128"
LEGACY_VARIANT_ID = "legacy_pos_l1"
SELECTION_OBJECTIVE = "median_throughput_fps"
WORKLOAD_KEY = "flame_steak:14000:111525:1352x1014:sm_86"
EQUIVALENCE_FRACTION = 0.005
PROMOTION_MIN_FPS_RATIO = 1.01
PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE = 1.0
FORMAL_BOOTSTRAP_RESAMPLES = 10000
FORMAL_BOOTSTRAP_SEED = 0
MIN_SELECTION_TRIALS = 10
MIXED_ABI_SHA256 = (
    "231c90c429321b2673b88ecd09efb40b6aedda7a23f3e061a2bcedec06d44426"
)
HEAD_ABI_SHA256 = (
    "24570aa6e67e8b9b10fa94524fec4dc03a4eb3fdc3bf822af34c2c52ce4937ac"
)
# Updated after hashing ``abi/tacker_mixed_render_heads_v2.json``.  Keeping the
# digest in the application contract makes a source/profile mismatch fail
# closed before the first physical launch.
MIXED_MULTI_ABI_SHA256 = (
    "310b15957c5920773bb03a61a37c5771f6d4570393061ece4e1805fd20989056"
)
HEAD_MULTI_ABI_SHA256 = (
    "9d6a1558acd6b642b975bcabe22abcbe3fd7242e4c9e0d635636ef4d2eb5da7f"
)
MIXED_MULTI_ABI_VERSION = 2
FIRST_LINEAR_PARTITION_KIND = "first_linear_heads"
HEAD_ORDER = ("pos", "scales", "rotations", "opacity", "shs")
HEAD_MODULES = {
    "pos": "pos_deform",
    "scales": "scales_deform",
    "rotations": "rotations_deform",
    "opacity": "opacity_deform",
    "shs": "shs_deform",
}
HEAD_DISABLE_FLAGS = {
    "pos": "no_dx",
    "scales": "no_ds",
    "rotations": "no_dr",
    "opacity": "no_do",
    "shs": "no_dshs",
}
HEAD_DELTA_OUTPUTS = {
    "pos": "deformation.pos_delta",
    "scales": "deformation.scales_delta",
    "rotations": "deformation.rotations_delta",
    "opacity": "deformation.opacity_delta",
    "shs": "deformation.shs_delta",
}
DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tacker_profiles"
    / "raster_head_sm86.json"
)

_CORRECTNESS_LIMITS = {
    "psnr_drop_db_max": 0.05,
    "ssim_drop_max": 0.0001,
    "lpips_increase_max": 0.0001,
}

# Schema-v1 profiles remain readable during migration.  Their old performance
# limits are checked for structural compatibility only; Phase 1 deliberately
# does not use them as runtime admission gates.
_LEGACY_LOCKED_LIMITS = {
    "raster_slowdown_pct_max": 5.0,
    "end_to_end_ratio_max": 1.0,
    **_CORRECTNESS_LIMITS,
}

_MEASUREMENT_KEYS = (
    "raster_slowdown_pct",
    "mixed_p50_ms",
    "solo_raster_p50_ms",
    "solo_head_p50_ms",
    "tacker_end_to_end_p50_ms",
    "two_stream_end_to_end_p50_ms",
    "psnr_drop_db",
    "ssim_drop",
    "lpips_increase",
)


class TackerProfileError(ValueError):
    """Raised when an admission profile is malformed or has a bad hash."""


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(manifest):
    """Return the stable SHA-256 used by the profile's sealed manifest."""

    return hashlib.sha256(_canonical_json(manifest)).hexdigest()


_PROFILE_SEALED_KEYS = (
    "schema_version",
    "workload_key",
    "selection_objective",
    "selected_variant_id",
    "manifest",
    "manifest_sha256",
    "correctness_thresholds",
    "candidates",
    "selection",
    "deployment",
    "provenance",
)


def profile_sha256(profile):
    """Hash every schema-v2 field that can change dispatch or selection.

    Human notes and generation timestamps are intentionally outside this
    payload.  Candidate trials, correctness results, input hashes, the chosen
    variant, and the sealed workload/ABI manifest are all covered.
    """

    if not isinstance(profile, dict):
        raise TackerProfileError("profile must be a JSON object")
    payload = {key: profile.get(key) for key in _PROFILE_SEALED_KEYS}
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        provenance.pop("generated_at_utc", None)
        payload["provenance"] = provenance
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _is_finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _require_exact(mapping, key, expected, section):
    if key not in mapping:
        raise TackerProfileError("{}.{} is required".format(section, key))
    if mapping[key] != expected:
        raise TackerProfileError(
            "{}.{} must be {!r}".format(section, key, expected)
        )


def _validate_legacy_tacker_profile(profile):
    """Validate the read-only schema-v1 profile contract."""

    if not isinstance(profile, dict):
        raise TackerProfileError("profile must be a JSON object")
    _require_exact(
        profile,
        "schema_version",
        LEGACY_PROFILE_SCHEMA_VERSION,
        "profile",
    )

    manifest = profile.get("manifest")
    if not isinstance(manifest, dict):
        raise TackerProfileError("profile.manifest must be an object")
    digest = profile.get("manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise TackerProfileError("profile.manifest_sha256 must be a SHA-256 hex string")
    expected_digest = manifest_sha256(manifest)
    if digest.lower() != expected_digest:
        raise TackerProfileError("profile manifest SHA-256 mismatch")

    _require_exact(manifest, "rasterizer_commit", RASTERIZER_COMMIT, "manifest")
    _require_exact(manifest, "pair_key", PAIR_KEY, "manifest")
    _require_exact(manifest, "cuda_arch", "sm_86", "manifest")
    _require_exact(manifest, "compute_capability", [8, 6], "manifest")
    _require_exact(manifest, "gpu_name", "NVIDIA RTX A6000", "manifest")
    _require_exact(manifest, "workload", "flame_steak", "manifest")
    _require_exact(manifest, "iteration", 14000, "manifest")
    _require_exact(manifest, "gaussian_count", 111525, "manifest")
    _require_exact(manifest, "resolution", [1352, 1014], "manifest")
    _require_exact(manifest, "physical_cta_threads", 384, "manifest")
    _require_exact(manifest, "raster_thread_range_inclusive", [0, 255], "manifest")
    _require_exact(manifest, "head_thread_range_inclusive", [256, 383], "manifest")
    _require_exact(manifest, "raster_named_barrier_id", 1, "manifest")
    _require_exact(manifest, "head_named_barrier_ids", [], "manifest")
    _require_exact(manifest, "head_input_dtype", "float16", "manifest")
    _require_exact(manifest, "head_weight_dtype", "float16", "manifest")
    _require_exact(manifest, "head_bias_dtype", "float32", "manifest")
    _require_exact(manifest, "head_accumulation_dtype", "float32", "manifest")
    _require_exact(manifest, "head_output_dtype", "float32", "manifest")

    persistent_blocks = manifest.get("persistent_blocks")
    if (
        not isinstance(persistent_blocks, int)
        or isinstance(persistent_blocks, bool)
        or persistent_blocks < 0
    ):
        raise TackerProfileError("manifest.persistent_blocks must be an int >= 0")

    thresholds = profile.get("thresholds")
    if not isinstance(thresholds, dict):
        raise TackerProfileError("profile.thresholds must be an object")
    for key, locked_limit in _LEGACY_LOCKED_LIMITS.items():
        value = thresholds.get(key)
        if not _is_finite_number(value):
            raise TackerProfileError("thresholds.{} must be finite".format(key))
        if float(value) > locked_limit:
            raise TackerProfileError(
                "thresholds.{} weakens the locked limit {}".format(
                    key, locked_limit
                )
            )
    _require_exact(
        thresholds,
        "mixed_p50_strictly_less_than_solo_sum",
        True,
        "thresholds",
    )

    admission = profile.get("admission")
    if not isinstance(admission, dict):
        raise TackerProfileError("profile.admission must be an object")
    for key in ("enabled", "valid"):
        if type(admission.get(key)) is not bool:
            raise TackerProfileError("admission.{} must be boolean".format(key))

    measurements = profile.get("measurements")
    if measurements is not None and not isinstance(measurements, dict):
        raise TackerProfileError("profile.measurements must be null or an object")
    if admission["enabled"] or admission["valid"]:
        if not isinstance(measurements, dict):
            raise TackerProfileError(
                "enabled/valid profiles require measured admission data"
            )
        for key in _MEASUREMENT_KEYS:
            if not _is_finite_number(measurements.get(key)):
                raise TackerProfileError(
                    "measurements.{} must be finite".format(key)
                )
        for key in (
            "mixed_p50_ms",
            "solo_raster_p50_ms",
            "solo_head_p50_ms",
            "tacker_end_to_end_p50_ms",
            "two_stream_end_to_end_p50_ms",
        ):
            if float(measurements[key]) <= 0.0:
                raise TackerProfileError(
                    "measurements.{} must be greater than zero".format(key)
                )
    return profile


def _require_sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TackerProfileError("{} must be a lowercase SHA-256".format(label))


def _validate_v2_manifest(manifest, workload_key):
    if not isinstance(manifest, dict):
        raise TackerProfileError("profile.manifest must be an object")
    expected = {
        "workload_key": workload_key,
        "rasterizer_commit": RASTERIZER_COMMIT,
        "cuda_arch": "sm_86",
        "compute_capability": [8, 6],
        "gpu_name": "NVIDIA RTX A6000",
        "workload": "flame_steak",
        "iteration": 14000,
        "gaussian_count": 111525,
        "resolution": [1352, 1014],
    }
    for key, value in expected.items():
        _require_exact(manifest, key, value, "manifest")
    persistent_blocks = manifest.get("persistent_blocks")
    if (
        type(persistent_blocks) is not int
        or persistent_blocks < 0
    ):
        raise TackerProfileError("manifest.persistent_blocks must be an int >= 0")


def _first_linear_node(head_name):
    return "deformation.{}[1].linear_128x128".format(HEAD_MODULES[head_name])


def _full_head_node(head_name):
    return "deformation.{}".format(HEAD_MODULES[head_name])


def _suffix_head_nodes(head_name):
    module_name = HEAD_MODULES[head_name]
    return [
        "deformation.{}[2]".format(module_name),
        "deformation.{}[3]".format(module_name),
    ]


def _canonical_head_names(value, section):
    if not isinstance(value, list) or not value:
        raise TackerProfileError("{}.selected_heads must be a non-empty array".format(section))
    if any(not isinstance(name, str) or name not in HEAD_ORDER for name in value):
        raise TackerProfileError(
            "{}.selected_heads contains an unsupported deformation head".format(section)
        )
    if len(set(value)) != len(value):
        raise TackerProfileError("{}.selected_heads contains duplicates".format(section))
    canonical = [name for name in HEAD_ORDER if name in value]
    if value != canonical:
        raise TackerProfileError(
            "{}.selected_heads must use canonical head order".format(section)
        )
    return tuple(canonical)


def _validate_candidate_resources(candidate, section, require_measured=False):
    if "resources" not in candidate:
        raise TackerProfileError("{}.resources is required".format(section))
    resources = candidate["resources"]
    if resources is None:
        if require_measured:
            raise TackerProfileError(
                "{}.resources must be measured for a correctness-valid v2 ABI candidate"
                .format(section)
            )
        return
    if not isinstance(resources, dict):
        raise TackerProfileError("{}.resources must be null or an object".format(section))
    for name, value in resources.items():
        if value is not None and (
            not _is_finite_number(value) or float(value) < 0.0
        ):
            raise TackerProfileError(
                "{}.resources.{} must be null or a finite non-negative number"
                .format(section, name)
            )
    if require_measured:
        for name in (
            "registers_per_thread",
            "static_shared_memory_bytes",
            "max_threads_per_block",
            "active_blocks_per_sm",
        ):
            if not _is_finite_number(resources.get(name)):
                raise TackerProfileError(
                    "{}.resources.{} is required for runtime filtering".format(
                        section, name
                    )
                )
        if int(resources["active_blocks_per_sm"]) < 1:
            raise TackerProfileError(
                "{}.resources.active_blocks_per_sm must be >= 1".format(section)
            )


def _validate_pos_l1_v2_candidate(candidate):
    """Validate a current-ABI pos-L1 variant (including block-count scans)."""

    section = "candidate {}".format(candidate.get("variant_id", "<unknown>"))
    expected = {
        "execution_mode": "tacker",
        "pair_key": PAIR_KEY,
        "cuda_symbol": "tacker_mix_render_head_v1",
        "abi_manifest_sha256": MIXED_ABI_SHA256,
        "physical_cta_threads": 384,
        "raster_threads": 256,
        "raster_thread_range_inclusive": [0, 255],
        "raster_named_barrier_id": 1,
        "persistent_blocks": candidate.get("persistent_blocks"),
    }
    for key, value in expected.items():
        _require_exact(candidate, key, value, section)
    persistent_blocks = candidate.get("persistent_blocks")
    if type(persistent_blocks) is not int or persistent_blocks < 0:
        raise TackerProfileError(
            "{}.persistent_blocks must be an int >= 0".format(section)
        )
    fused_nodes = candidate.get("fused_nodes")
    if fused_nodes != [
        "raster.render_leaf",
        "deformation.pos_deform[1].linear_128x128",
    ]:
        raise TackerProfileError("{}.fused_nodes changed".format(section))
    task_graph = {
        "parallel_nodes": [
            "deformation.scales_deform",
            "deformation.rotations_deform",
            "deformation.opacity_deform",
            "deformation.shs_deform",
        ],
        "suffix_nodes": [
            "deformation.pos_deform[2]",
            "deformation.pos_deform[3]",
            "deformation.apply_residuals",
        ],
        "required_outputs": [
            "raster.color",
            "raster.depth",
            "raster.radii",
            "deformation.pos_delta",
        ],
    }
    for key, value in task_graph.items():
        _require_exact(candidate, key, value, section)
    subgroups = candidate.get("backend_subgroups")
    if subgroups != [
        {
            "name": "pos_deform_l1",
            "thread_range_inclusive": [256, 383],
            "threads": 128,
            "named_barrier_ids": [],
        }
    ]:
        raise TackerProfileError("{}.backend_subgroups changed".format(section))
    tensor_contract = candidate.get("tensor_contract")
    if not isinstance(tensor_contract, dict):
        raise TackerProfileError("{}.tensor_contract must be an object".format(section))
    expected_dtypes = {
        "input_dtype": "float16",
        "weight_dtype": "float16",
        "bias_dtype": "float32",
        "accumulation_dtype": "float32",
        "output_dtype": "float32",
    }
    for key, value in expected_dtypes.items():
        _require_exact(tensor_contract, key, value, "{}.tensor_contract".format(section))
    capability = candidate.get("capability_requirements")
    if not isinstance(capability, dict):
        raise TackerProfileError(
            "{}.capability_requirements must be an object".format(section)
        )
    _require_exact(capability, "cuda_arch", "sm_86", "{}.capability_requirements".format(section))
    _require_exact(capability, "compute_capability", [8, 6], "{}.capability_requirements".format(section))
    _require_exact(candidate, "tile_shape", [16, 16], section)
    _validate_candidate_resources(candidate, section)


def _validate_first_linear_v2_candidate(candidate):
    """Validate the generic C1/C2 first-linear mixed ABI contract."""

    section = "candidate {}".format(candidate.get("variant_id", "<unknown>"))
    partition = candidate.get("partition")
    if not isinstance(partition, dict):
        raise TackerProfileError("{}.partition must be an object".format(section))
    _require_exact(
        partition, "kind", FIRST_LINEAR_PARTITION_KIND, "{}.partition".format(section)
    )
    selected_heads = _canonical_head_names(partition.get("selected_heads"), "{}.partition".format(section))
    worker_groups = partition.get("worker_groups")
    if (
        type(worker_groups) is not int
        or worker_groups < 1
        or worker_groups > len(selected_heads)
        or worker_groups > len(HEAD_ORDER)
    ):
        raise TackerProfileError(
            "{}.partition.worker_groups must be in [1, selected head count]"
            .format(section)
        )

    expected = {
        "execution_mode": "tacker",
        "cuda_symbol": "tacker_mix_render_heads_v2",
        "abi_manifest_sha256": MIXED_MULTI_ABI_SHA256,
        "head_abi_manifest_sha256": HEAD_MULTI_ABI_SHA256,
        "physical_cta_threads": 256 + 128 * worker_groups,
        "raster_threads": 256,
        "raster_thread_range_inclusive": [0, 255],
        "raster_named_barrier_id": 1,
        "persistent_blocks": candidate.get("persistent_blocks"),
        "fused_nodes": ["raster.render_leaf"]
        + [_first_linear_node(name) for name in selected_heads],
        "parallel_nodes": [
            _full_head_node(name) for name in HEAD_ORDER if name not in selected_heads
        ],
        "suffix_nodes": sum(
            (_suffix_head_nodes(name) for name in selected_heads), []
        )
        + ["deformation.apply_residuals"],
        "skipped_python_nodes": [
            _first_linear_node(name) for name in selected_heads
        ],
        "required_outputs": [
            "raster.color",
            "raster.depth",
            "raster.radii",
        ]
        + [HEAD_DELTA_OUTPUTS[name] for name in HEAD_ORDER],
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
        "tile_shape": [16, 16],
    }
    for key, value in expected.items():
        _require_exact(candidate, key, value, section)

    persistent_blocks = candidate.get("persistent_blocks")
    if type(persistent_blocks) is not int or persistent_blocks < 0:
        raise TackerProfileError(
            "{}.persistent_blocks must be an int >= 0".format(section)
        )
    expected_subgroups = []
    for worker_index in range(worker_groups):
        begin = 256 + 128 * worker_index
        expected_subgroups.append(
            {
                "name": "head_worker_{}".format(worker_index),
                "thread_range_inclusive": [begin, begin + 127],
                "threads": 128,
                "named_barrier_ids": [2],
            }
        )
    _require_exact(candidate, "backend_subgroups", expected_subgroups, section)

    tensor_contract = candidate.get("tensor_contract")
    if not isinstance(tensor_contract, dict):
        raise TackerProfileError("{}.tensor_contract must be an object".format(section))
    expected_dtypes = {
        "input_dtype": "float16",
        "weight_dtype": "float16",
        "bias_dtype": "float32",
        "accumulation_dtype": "float32",
        "output_dtype": "float32",
        "features": 128,
        "max_heads": 5,
    }
    for key, value in expected_dtypes.items():
        _require_exact(
            tensor_contract, key, value, "{}.tensor_contract".format(section)
        )

    capability = candidate.get("capability_requirements")
    if not isinstance(capability, dict):
        raise TackerProfileError(
            "{}.capability_requirements must be an object".format(section)
        )
    capability_section = "{}.capability_requirements".format(section)
    _require_exact(capability, "cuda_arch", "sm_86", capability_section)
    _require_exact(capability, "compute_capability", [8, 6], capability_section)
    _require_exact(
        capability, "mixed_render_heads_abi", MIXED_MULTI_ABI_VERSION, capability_section
    )
    _validate_candidate_resources(
        candidate,
        section,
        require_measured=bool(candidate.get("correctness", {}).get("valid")),
    )


def first_linear_candidate_contract(
    variant_id,
    selected_heads,
    worker_groups=1,
    persistent_blocks=0,
    resources=None,
):
    """Build the deterministic structural half of a C1/C2 candidate.

    Offline qualification fills correctness, performance, diagnostics, and
    measured resources before deployment.  Keeping contract generation here
    prevents the tuner and runtime validator from drifting apart.
    """

    if not isinstance(variant_id, str) or not variant_id:
        raise ValueError("variant_id must be a non-empty string")
    names = _canonical_head_names(list(selected_heads), "candidate.partition")
    if (
        type(worker_groups) is not int
        or worker_groups < 1
        or worker_groups > len(names)
    ):
        raise ValueError("worker_groups must be in [1, selected head count]")
    if type(persistent_blocks) is not int or persistent_blocks < 0:
        raise ValueError("persistent_blocks must be an int >= 0")
    subgroups = []
    for worker_index in range(worker_groups):
        begin = 256 + 128 * worker_index
        subgroups.append(
            {
                "name": "head_worker_{}".format(worker_index),
                "thread_range_inclusive": [begin, begin + 127],
                "threads": 128,
                "named_barrier_ids": [2],
            }
        )
    return {
        "variant_id": variant_id,
        "execution_mode": "tacker",
        "partition": {
            "kind": FIRST_LINEAR_PARTITION_KIND,
            "selected_heads": list(names),
            "worker_groups": worker_groups,
        },
        "fused_nodes": ["raster.render_leaf"]
        + [_first_linear_node(name) for name in names],
        "parallel_nodes": [
            _full_head_node(name) for name in HEAD_ORDER if name not in names
        ],
        "suffix_nodes": sum((_suffix_head_nodes(name) for name in names), [])
        + ["deformation.apply_residuals"],
        "skipped_python_nodes": [_first_linear_node(name) for name in names],
        "required_outputs": [
            "raster.color",
            "raster.depth",
            "raster.radii",
        ]
        + [HEAD_DELTA_OUTPUTS[name] for name in HEAD_ORDER],
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
        "cuda_symbol": "tacker_mix_render_heads_v2",
        "abi_manifest_sha256": MIXED_MULTI_ABI_SHA256,
        "head_abi_manifest_sha256": HEAD_MULTI_ABI_SHA256,
        "tensor_contract": {
            "input_dtype": "float16",
            "weight_dtype": "float16",
            "bias_dtype": "float32",
            "accumulation_dtype": "float32",
            "output_dtype": "float32",
            "features": 128,
            "max_heads": 5,
        },
        "physical_cta_threads": 256 + 128 * worker_groups,
        "raster_threads": 256,
        "raster_thread_range_inclusive": [0, 255],
        "raster_named_barrier_id": 1,
        "backend_subgroups": subgroups,
        "persistent_blocks": persistent_blocks,
        "tile_shape": [16, 16],
        "resources": deepcopy(resources),
        "capability_requirements": {
            "cuda_arch": "sm_86",
            "compute_capability": [8, 6],
            "mixed_render_heads_abi": MIXED_MULTI_ABI_VERSION,
        },
        "correctness": {"valid": False},
        "performance": None,
        "diagnostics": None,
    }


def _validate_v2_candidate(candidate, correctness_limits):
    if not isinstance(candidate, dict):
        raise TackerProfileError("profile.candidates entries must be objects")
    variant_id = candidate.get("variant_id")
    if not isinstance(variant_id, str) or not variant_id:
        raise TackerProfileError("candidate.variant_id must be a non-empty string")
    execution_mode = candidate.get("execution_mode")
    if execution_mode not in ("serial", "two_stream", "tacker"):
        raise TackerProfileError(
            "candidate {} has an invalid execution_mode".format(variant_id)
        )
    correctness = candidate.get("correctness")
    if not isinstance(correctness, dict) or type(correctness.get("valid")) is not bool:
        raise TackerProfileError(
            "candidate {} correctness.valid must be boolean".format(variant_id)
        )
    if correctness["valid"]:
        if correctness.get("actual_execution_mode") != execution_mode:
            raise TackerProfileError(
                "candidate {} did not execute its declared physical mode".format(
                    variant_id
                )
            )
        if correctness.get("fallback_reason") is not None:
            raise TackerProfileError(
                "candidate {} correctness recorded a fallback".format(variant_id)
            )
        if execution_mode == "tacker":
            quality_keys = {
                "psnr_drop_db": "psnr_drop_db_max",
                "ssim_drop": "ssim_drop_max",
                "lpips_increase": "lpips_increase_max",
            }
            for measurement, limit_name in quality_keys.items():
                value = correctness.get(measurement)
                if not _is_finite_number(value):
                    raise TackerProfileError(
                        "candidate {} correctness.{} must be finite".format(
                            variant_id, measurement
                        )
                    )
                if float(value) > correctness_limits[limit_name]:
                    raise TackerProfileError(
                        "candidate {} exceeds correctness threshold {}".format(
                            variant_id, limit_name
                        )
                    )

    performance = candidate.get("performance")
    if performance is not None:
        if not isinstance(performance, dict):
            raise TackerProfileError(
                "candidate {} performance must be null or an object".format(
                    variant_id
                )
            )
        trials = performance.get("throughput_fps_trials")
        if not isinstance(trials, list) or not trials:
            raise TackerProfileError(
                "candidate {} requires whole-run FPS trials".format(variant_id)
            )
        if any(not _is_finite_number(value) or float(value) <= 0.0 for value in trials):
            raise TackerProfileError(
                "candidate {} FPS trials must be finite and > 0".format(variant_id)
            )
        median_fps = performance.get("median_throughput_fps")
        expected_median = statistics.median(float(value) for value in trials)
        if (
            not _is_finite_number(median_fps)
            or not math.isclose(
                float(median_fps), expected_median, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise TackerProfileError(
                "candidate {} median_throughput_fps disagrees with trials".format(
                    variant_id
                )
            )
    diagnostics = candidate.get("diagnostics")
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise TackerProfileError(
            "candidate {} diagnostics must be null or an object".format(variant_id)
        )
    if execution_mode == "tacker":
        if candidate.get("partition") is None:
            _validate_pos_l1_v2_candidate(candidate)
        else:
            _validate_first_linear_v2_candidate(candidate)
    return candidate


def _same_finite_number(actual, expected, label):
    if not _is_finite_number(actual) or not math.isclose(
        float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise TackerProfileError("{} disagrees with whole-run trials".format(label))
    return float(actual)


def _percentile(sorted_values, probability):
    if not sorted_values:
        raise TackerProfileError("cannot compute an empty bootstrap percentile")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


class _DeterministicGenerator(object):
    """Version-independent SHA-256 sampler shared with Phase-1 admission."""

    def __init__(self, seed, label):
        self._key = _canonical_json({"seed": seed, "label": label})
        self._counter = 0

    def index(self, stop):
        material = self._key + self._counter.to_bytes(16, byteorder="big")
        self._counter += 1
        return int.from_bytes(hashlib.sha256(material).digest(), "big") % stop


def _paired_bootstrap_interval(
    candidate_values, reference_values, resamples, seed, label
):
    """Recompute the paired-round ratio-of-medians bootstrap interval."""

    if (
        not candidate_values
        or len(candidate_values) != len(reference_values)
        or type(resamples) is not int
        or resamples <= 0
        or type(seed) is not int
    ):
        raise TackerProfileError("paired bootstrap metadata is invalid")
    generator = _DeterministicGenerator(seed, label)
    count = len(candidate_values)
    values = []
    for _ in range(resamples):
        indices = [generator.index(count) for _ in range(count)]
        values.append(
            statistics.median(candidate_values[index] for index in indices)
            / statistics.median(reference_values[index] for index in indices)
        )
    values.sort()
    return _percentile(values, 0.025), _percentile(values, 0.975)


def _performance_by_round(candidate):
    """Return a sealed round -> FPS mapping for one eligible candidate."""

    variant_id = candidate["variant_id"]
    performance = candidate["performance"]
    trials = [float(value) for value in performance["throughput_fps_trials"]]
    trial_count = performance.get("trial_count")
    if type(trial_count) is not int or trial_count != len(trials):
        raise TackerProfileError(
            "candidate {} trial_count disagrees with FPS trials".format(variant_id)
        )
    if trial_count < MIN_SELECTION_TRIALS:
        raise TackerProfileError(
            "candidate {} requires at least {} whole-run trials".format(
                variant_id, MIN_SELECTION_TRIALS
            )
        )
    rounds = performance.get("round_indices")
    if (
        not isinstance(rounds, list)
        or len(rounds) != trial_count
        or any(type(value) is not int or value < 0 for value in rounds)
        or len(set(rounds)) != len(rounds)
    ):
        raise TackerProfileError(
            "candidate {} requires one unique round index per FPS trial".format(
                variant_id
            )
        )
    return list(rounds), dict(zip(rounds, trials))


def _validate_paired_promotion_evidence(comparison, challenger, incumbent):
    """Validate and independently recompute one challenger comparison."""

    if not isinstance(comparison, dict):
        raise TackerProfileError("promotion paired_comparison must be an object")
    challenger_name = challenger.get("benchmark_candidate_name")
    incumbent_name = incumbent.get("benchmark_candidate_name")
    if not isinstance(challenger_name, str) or not challenger_name:
        raise TackerProfileError(
            "candidate {} benchmark_candidate_name is required".format(
                challenger["variant_id"]
            )
        )
    if not isinstance(incumbent_name, str) or not incumbent_name:
        raise TackerProfileError(
            "candidate {} benchmark_candidate_name is required".format(
                incumbent["variant_id"]
            )
        )
    if (
        comparison.get("candidate") != challenger_name
        or comparison.get("reference") != incumbent_name
    ):
        raise TackerProfileError(
            "promotion paired comparison names the wrong candidate/reference"
        )

    challenger_rounds, challenger_by_round = _performance_by_round(challenger)
    incumbent_rounds, incumbent_by_round = _performance_by_round(incumbent)
    rounds = comparison.get("round_indices")
    if rounds != challenger_rounds or rounds != incumbent_rounds:
        raise TackerProfileError(
            "promotion paired comparison must cover every shared FPS round"
        )
    challenger_values = [challenger_by_round[index] for index in rounds]
    incumbent_values = [incumbent_by_round[index] for index in rounds]
    expected_ratios = [
        challenger_value / incumbent_value
        for challenger_value, incumbent_value in zip(
            challenger_values, incumbent_values
        )
    ]
    ratios = comparison.get("paired_fps_ratios")
    if (
        not isinstance(ratios, list)
        or len(ratios) != len(expected_ratios)
        or any(not _is_finite_number(value) or float(value) <= 0.0 for value in ratios)
        or any(
            not math.isclose(
                float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
            )
            for actual, expected in zip(ratios, expected_ratios)
        )
    ):
        raise TackerProfileError(
            "promotion paired FPS ratios disagree with whole-run trials"
        )
    expected_ratio = (
        float(challenger["performance"]["median_throughput_fps"])
        / float(incumbent["performance"]["median_throughput_fps"])
    )
    ratio = _same_finite_number(
        comparison.get("median_fps_ratio"),
        expected_ratio,
        "promotion median_fps_ratio",
    )
    _same_finite_number(
        comparison.get("median_paired_fps_ratio"),
        statistics.median(expected_ratios),
        "promotion median_paired_fps_ratio",
    )

    interval = comparison.get("paired_bootstrap_95_ci")
    if not isinstance(interval, dict):
        raise TackerProfileError(
            "promotion paired_bootstrap_95_ci must be an object"
        )
    lower = interval.get("lower")
    upper = interval.get("upper")
    if (
        not _is_finite_number(lower)
        or float(lower) <= 0.0
        or not _is_finite_number(upper)
        or float(upper) <= 0.0
        or float(lower) > float(upper)
    ):
        raise TackerProfileError("promotion paired bootstrap interval is invalid")
    if not _is_finite_number(interval.get("confidence")) or not math.isclose(
        float(interval["confidence"]), 0.95, rel_tol=0.0, abs_tol=1e-15
    ):
        raise TackerProfileError("promotion paired bootstrap confidence must be 0.95")
    for key, expected in (
        ("statistic", "median(candidate_fps)/median(reference_fps)"),
        ("resampling_unit", "paired_round"),
        ("percentile_method", "linear_type_7"),
    ):
        if interval.get(key) != expected:
            raise TackerProfileError(
                "promotion paired bootstrap {} changed".format(key)
            )
    resamples = interval.get("resamples")
    seed = interval.get("seed")
    if (
        resamples != FORMAL_BOOTSTRAP_RESAMPLES
        or seed != FORMAL_BOOTSTRAP_SEED
    ):
        raise TackerProfileError(
            "promotion paired bootstrap must use 10000 resamples and seed 0"
        )
    if comparison.get("derived_by_inverting_reported_comparison") is True:
        reverse_lower, reverse_upper = _paired_bootstrap_interval(
            incumbent_values,
            challenger_values,
            resamples,
            seed,
            "{}-vs-{}".format(incumbent_name, challenger_name),
        )
        expected_lower = 1.0 / reverse_upper
        expected_upper = 1.0 / reverse_lower
    else:
        expected_lower, expected_upper = _paired_bootstrap_interval(
            challenger_values,
            incumbent_values,
            resamples,
            seed,
            "{}-vs-{}".format(challenger_name, incumbent_name),
        )
    lower = _same_finite_number(
        lower, expected_lower, "promotion paired bootstrap lower"
    )
    _same_finite_number(
        upper, expected_upper, "promotion paired bootstrap upper"
    )
    return ratio, lower


def _validate_promotion_criterion(criterion, observed, required_key, required):
    if not isinstance(criterion, dict):
        raise TackerProfileError("promotion criterion must be an object")
    _same_finite_number(
        criterion.get("observed"), observed, "promotion criterion observed value"
    )
    _same_finite_number(
        criterion.get(required_key), required, "promotion criterion requirement"
    )


def _validate_promotion_evaluation(
    evaluation, challenger, incumbent, two_stream
):
    if not isinstance(evaluation, dict):
        raise TackerProfileError("promotion candidate evaluation must be an object")
    if evaluation.get("candidate_variant_id") != challenger["variant_id"]:
        raise TackerProfileError("promotion candidate evaluation changed order")
    if type(evaluation.get("passed")) is not bool:
        raise TackerProfileError("promotion candidate evaluation passed must be boolean")
    ratio, ci_lower = _validate_paired_promotion_evidence(
        evaluation.get("paired_comparison"), challenger, incumbent
    )
    challenger_fps = float(
        challenger["performance"]["median_throughput_fps"]
    )
    incumbent_fps = float(incumbent["performance"]["median_throughput_fps"])
    two_stream_fps = float(two_stream["performance"]["median_throughput_fps"])
    ratio_passed = ratio >= PROMOTION_MIN_FPS_RATIO
    ci_passed = ci_lower > PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE
    floor_ratios = {
        "two_stream": challenger_fps / two_stream_fps,
        incumbent["variant_id"]: challenger_fps / incumbent_fps,
    }
    floor_passed = all(value >= 1.0 for value in floor_ratios.values())
    expected_passed = ratio_passed and ci_passed and floor_passed
    if evaluation["passed"] != expected_passed:
        raise TackerProfileError(
            "promotion candidate pass result disagrees with recomputed gates"
        )

    criteria = evaluation.get("criteria")
    if not isinstance(criteria, dict):
        raise TackerProfileError("promotion evaluation criteria must be an object")
    ratio_criterion = criteria.get("median_fps_ratio")
    _validate_promotion_criterion(
        ratio_criterion, ratio, "required_min", PROMOTION_MIN_FPS_RATIO
    )
    if ratio_criterion.get("passed") is not ratio_passed:
        raise TackerProfileError("promotion ratio criterion result changed")
    ci_criterion = criteria.get("paired_bootstrap_95_ci_lower")
    _validate_promotion_criterion(
        ci_criterion,
        ci_lower,
        "required_strictly_greater_than",
        PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE,
    )
    if ci_criterion.get("passed") is not ci_passed:
        raise TackerProfileError("promotion bootstrap criterion result changed")
    floor_criterion = criteria.get("baseline_fps_ratios")
    if not isinstance(floor_criterion, dict):
        raise TackerProfileError(
            "promotion baseline_fps_ratios criterion must be an object"
        )
    observed_floor = floor_criterion.get("observed")
    if not isinstance(observed_floor, dict) or set(observed_floor) != set(floor_ratios):
        raise TackerProfileError("promotion baseline FPS ratio set changed")
    for baseline_id, expected in floor_ratios.items():
        _same_finite_number(
            observed_floor.get(baseline_id),
            expected,
            "promotion baseline {} FPS ratio".format(baseline_id),
        )
    _same_finite_number(
        floor_criterion.get("required_min"),
        1.0,
        "promotion baseline FPS requirement",
    )
    if floor_criterion.get("passed") is not floor_passed:
        raise TackerProfileError("promotion baseline FPS criterion result changed")
    return expected_passed, ratio, ci_lower


def _validate_v2_selection(selection, candidates, selected_variant_id):
    """Recompute ranking, equivalence and every incumbent-promotion gate."""

    by_id = {candidate["variant_id"]: candidate for candidate in candidates}
    eligible = [
        candidate
        for candidate in candidates
        if candidate["correctness"]["valid"]
    ]
    if not eligible:
        raise TackerProfileError("selection has no correctness-valid candidate")
    if any(candidate.get("performance") is None for candidate in eligible):
        raise TackerProfileError(
            "every correctness-valid candidate requires whole-run performance"
        )

    benchmark_names = []
    shared_rounds = None
    for candidate in eligible:
        benchmark_name = candidate.get("benchmark_candidate_name")
        if not isinstance(benchmark_name, str) or not benchmark_name:
            raise TackerProfileError(
                "candidate {} benchmark_candidate_name is required".format(
                    candidate["variant_id"]
                )
            )
        benchmark_names.append(benchmark_name)
        rounds, _ = _performance_by_round(candidate)
        if shared_rounds is None:
            shared_rounds = rounds
        elif rounds != shared_rounds:
            raise TackerProfileError(
                "correctness-valid candidates must share the same paired rounds"
            )
    if len(benchmark_names) != len(set(benchmark_names)):
        raise TackerProfileError(
            "correctness-valid candidates require unique benchmark_candidate_name values"
        )

    ranking = sorted(
        eligible,
        key=lambda candidate: (
            -float(candidate["performance"]["median_throughput_fps"]),
            candidate["benchmark_candidate_name"],
        ),
    )
    ranking_ids = [candidate["variant_id"] for candidate in ranking]
    if selection.get("global_median_fps_ranking") != ranking_ids:
        raise TackerProfileError(
            "selection ranking disagrees with correctness-valid median FPS"
        )
    if selection.get("eligible_variant_ids") != ranking_ids:
        raise TackerProfileError(
            "selection eligible_variant_ids disagrees with ranking"
        )
    ineligible = selection.get("ineligible_variant_ids")
    expected_ineligible = {
        candidate["variant_id"]
        for candidate in candidates
        if not candidate["correctness"]["valid"]
    }
    if (
        not isinstance(ineligible, list)
        or len(ineligible) != len(set(ineligible))
        or set(ineligible) != expected_ineligible
    ):
        raise TackerProfileError(
            "selection ineligible_variant_ids disagrees with correctness"
        )
    experimental_id = selection.get("experimental_winner_variant_id")
    if experimental_id != ranking_ids[0]:
        raise TackerProfileError(
            "experimental winner is not the global median FPS argmax"
        )
    top_fps = float(ranking[0]["performance"]["median_throughput_fps"])
    equivalent_ids = [
        candidate["variant_id"]
        for candidate in ranking
        if (
            top_fps
            - float(candidate["performance"]["median_throughput_fps"])
        )
        / top_fps
        <= EQUIVALENCE_FRACTION
    ]
    equivalence = selection.get("equivalence")
    if not isinstance(equivalence, dict):
        raise TackerProfileError("selection.equivalence must be an object")
    fraction = equivalence.get("fraction")
    if not _is_finite_number(fraction) or not math.isclose(
        float(fraction), EQUIVALENCE_FRACTION, rel_tol=0.0, abs_tol=1e-15
    ):
        raise TackerProfileError("selection equivalence fraction must be 0.005")

    def preference_key(candidate):
        metadata = candidate.get("selection_metadata", {})
        if not isinstance(metadata, dict):
            raise TackerProfileError(
                "candidate selection_metadata must be an object"
            )
        key = []
        for name in (
            "abi_complexity",
            "peak_memory_bytes",
            "registers_per_thread",
            "shared_memory_bytes",
        ):
            value = metadata.get(name)
            if value is not None and (
                not _is_finite_number(value) or float(value) < 0.0
            ):
                raise TackerProfileError(
                    "candidate selection metadata must be finite and non-negative"
                )
            key.append((1, 0.0) if value is None else (0, float(value)))
        key.append(candidate["benchmark_candidate_name"])
        return tuple(key)

    preferred_equivalent = sorted(
        (by_id[variant_id] for variant_id in equivalent_ids),
        key=preference_key,
    )
    preferred_ids = [candidate["variant_id"] for candidate in preferred_equivalent]
    recorded_equivalent = equivalence.get(
        "candidate_variant_ids_in_preference_order",
        equivalence.get("candidates", equivalence.get("equivalent_variant_ids")),
    )
    if recorded_equivalent != preferred_ids:
        raise TackerProfileError(
            "selection equivalence preference disagrees with candidate resources"
        )
    preferred_id = equivalence.get(
        "preferred_variant_id", equivalence.get("preferred_candidate")
    )
    if preferred_id != preferred_ids[0]:
        raise TackerProfileError("selection preferred equivalent candidate changed")
    if selection.get("deployment_winner_variant_id") != selected_variant_id:
        raise TackerProfileError(
            "selected_variant_id disagrees with deployment winner"
        )

    incumbent_id = selection.get("incumbent_variant_id")
    if (
        not isinstance(incumbent_id, str)
        or incumbent_id not in by_id
        or by_id[incumbent_id]["execution_mode"] != "tacker"
    ):
        raise TackerProfileError(
            "selection incumbent_variant_id must identify a Tacker candidate"
        )
    promotion = selection.get("promotion")
    if not isinstance(promotion, dict) or type(promotion.get("passed")) is not bool:
        raise TackerProfileError("selection.promotion.passed must be boolean")
    _same_finite_number(
        promotion.get("minimum_median_fps_ratio"),
        PROMOTION_MIN_FPS_RATIO,
        "selection promotion minimum_median_fps_ratio",
    )
    _same_finite_number(
        promotion.get("minimum_bootstrap_lower_exclusive"),
        PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE,
        "selection promotion minimum_bootstrap_lower_exclusive",
    )
    if promotion.get("incumbent_variant_id") != incumbent_id:
        raise TackerProfileError(
            "selection promotion incumbent disagrees with selection"
        )

    incumbent = by_id[incumbent_id]
    selected = by_id[selected_variant_id]
    two_stream = by_id["two_stream"]
    selected_fps = float(selected["performance"]["median_throughput_fps"])
    two_stream_fps = float(
        two_stream["performance"]["median_throughput_fps"]
    )
    # This floor applies before every early-return case, including a retained
    # incumbent and an invalid-incumbent replacement.
    if selected_fps < two_stream_fps:
        raise TackerProfileError("deployment winner is slower than two_stream")

    if not incumbent["correctness"]["valid"]:
        replacements = [
            variant_id
            for variant_id in preferred_ids
            if float(by_id[variant_id]["performance"]["median_throughput_fps"])
            >= two_stream_fps
        ]
        if not replacements or selected_variant_id != replacements[0]:
            raise TackerProfileError(
                "invalid incumbent replacement disagrees with the two_stream floor"
            )
        if promotion.get("candidate_evaluations") != []:
            raise TackerProfileError(
                "invalid incumbent replacement must not claim bootstrap evaluations"
            )
        if (
            promotion.get("challenger_variant_id") != selected_variant_id
            or promotion.get("decision")
            != "selected_best_valid_candidate_incumbent_invalid"
            or promotion["passed"] is not True
            or promotion.get("required") is not True
        ):
            raise TackerProfileError(
                "invalid incumbent replacement decision is not sealed"
            )
        return

    incumbent_rounds, _ = _performance_by_round(incumbent)
    if incumbent_rounds != shared_rounds:
        raise TackerProfileError("incumbent does not share the paired FPS rounds")
    evaluation_ids = [
        variant_id for variant_id in preferred_ids if variant_id != incumbent_id
    ]
    evaluations = promotion.get("candidate_evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != len(evaluation_ids):
        raise TackerProfileError(
            "promotion candidate evaluations must cover every equivalent challenger"
        )
    recomputed = {}
    passing_ids = []
    for variant_id, evaluation in zip(evaluation_ids, evaluations):
        challenger = by_id[variant_id]
        passed, ratio, ci_lower = _validate_promotion_evaluation(
            evaluation, challenger, incumbent, two_stream
        )
        recomputed[variant_id] = {
            "evaluation": evaluation,
            "ratio": ratio,
            "ci_lower": ci_lower,
        }
        if passed:
            passing_ids.append(variant_id)

    summary_id = None
    if passing_ids:
        expected_selected_id = passing_ids[0]
        expected_decision = "promoted_equivalent_challenger"
        expected_passed = True
        expected_required = True
        summary_id = expected_selected_id
    elif experimental_id == incumbent_id:
        expected_selected_id = incumbent_id
        expected_decision = "incumbent_is_global_winner"
        expected_passed = True
        expected_required = False
    else:
        expected_selected_id = incumbent_id
        expected_decision = "retained_incumbent"
        expected_passed = False
        expected_required = preferred_id != incumbent_id
        if evaluation_ids:
            summary_id = (
                preferred_id if preferred_id in recomputed else evaluation_ids[0]
            )

    if selected_variant_id != expected_selected_id:
        raise TackerProfileError(
            "deployment winner disagrees with recomputed incumbent promotion"
        )
    if (
        promotion["passed"] is not expected_passed
        or promotion.get("decision") != expected_decision
        or promotion.get("required") is not expected_required
    ):
        raise TackerProfileError(
            "selection promotion decision disagrees with recomputed gates"
        )
    expected_challenger_id = (
        summary_id
        if summary_id is not None
        else incumbent_id
    )
    if promotion.get("challenger_variant_id") != expected_challenger_id:
        raise TackerProfileError(
            "selection promotion challenger disagrees with recomputed gates"
        )

    if summary_id is None:
        for key in (
            "paired_comparison",
            "median_fps_ratio",
            "paired_bootstrap_95_ci_lower",
            "criteria",
        ):
            if key in promotion:
                raise TackerProfileError(
                    "incumbent-winner promotion contains unexpected {}".format(key)
                )
        return

    summary = recomputed[summary_id]
    evaluation = summary["evaluation"]
    if promotion.get("paired_comparison") != evaluation["paired_comparison"]:
        raise TackerProfileError(
            "selection promotion comparison disagrees with candidate evaluation"
        )
    if promotion.get("criteria") != evaluation["criteria"]:
        raise TackerProfileError(
            "selection promotion criteria disagree with candidate evaluation"
        )
    _same_finite_number(
        promotion.get("median_fps_ratio"),
        summary["ratio"],
        "selection promotion median_fps_ratio",
    )
    _same_finite_number(
        promotion.get("paired_bootstrap_95_ci_lower"),
        summary["ci_lower"],
        "selection promotion paired_bootstrap_95_ci_lower",
    )


def _validate_v2_tacker_profile(profile):
    _require_exact(profile, "schema_version", PROFILE_SCHEMA_VERSION, "profile")
    _require_exact(profile, "workload_key", WORKLOAD_KEY, "profile")
    _require_exact(
        profile, "selection_objective", SELECTION_OBJECTIVE, "profile"
    )
    manifest = profile.get("manifest")
    _validate_v2_manifest(manifest, profile["workload_key"])
    digest = profile.get("manifest_sha256")
    _require_sha256(digest, "profile.manifest_sha256")
    if digest != manifest_sha256(manifest):
        raise TackerProfileError("profile manifest SHA-256 mismatch")

    correctness_limits = profile.get("correctness_thresholds")
    if not isinstance(correctness_limits, dict):
        raise TackerProfileError("profile.correctness_thresholds must be an object")
    for key, locked_limit in _CORRECTNESS_LIMITS.items():
        value = correctness_limits.get(key)
        if not _is_finite_number(value):
            raise TackerProfileError(
                "correctness_thresholds.{} must be finite".format(key)
            )
        if float(value) > locked_limit:
            raise TackerProfileError(
                "correctness_thresholds.{} weakens the locked limit {}".format(
                    key, locked_limit
                )
            )

    candidates = profile.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise TackerProfileError("profile.candidates must be a non-empty array")
    by_id = {}
    for candidate in candidates:
        _validate_v2_candidate(candidate, correctness_limits)
        variant_id = candidate["variant_id"]
        if variant_id in by_id:
            raise TackerProfileError(
                "profile.candidates has duplicate variant_id {!r}".format(variant_id)
            )
        by_id[variant_id] = candidate
    for baseline_id in ("serial", "two_stream"):
        if baseline_id not in by_id:
            raise TackerProfileError(
                "profile.candidates must include the {} baseline".format(baseline_id)
            )
        if by_id[baseline_id]["execution_mode"] != baseline_id:
            raise TackerProfileError(
                "candidate {} must execute the {} baseline".format(
                    baseline_id, baseline_id
                )
            )

    selected_variant_id = profile.get("selected_variant_id")
    if not isinstance(selected_variant_id, str) or selected_variant_id not in by_id:
        raise TackerProfileError(
            "profile.selected_variant_id must identify exactly one candidate"
        )
    selected = by_id[selected_variant_id]
    if selected["execution_mode"] == "tacker":
        if selected["persistent_blocks"] != manifest["persistent_blocks"]:
            raise TackerProfileError(
                "selected candidate persistent_blocks disagrees with manifest"
            )

    deployment = profile.get("deployment")
    if not isinstance(deployment, dict):
        raise TackerProfileError("profile.deployment must be an object")
    for key in ("enabled", "valid"):
        if type(deployment.get(key)) is not bool:
            raise TackerProfileError("deployment.{} must be boolean".format(key))
    if deployment["enabled"] != deployment["valid"]:
        raise TackerProfileError("deployment.enabled and valid must agree")
    selection = profile.get("selection")
    if selection is not None and not isinstance(selection, dict):
        raise TackerProfileError("profile.selection must be null or an object")
    if deployment["enabled"]:
        if selected["execution_mode"] != "tacker":
            raise TackerProfileError(
                "deployable Tacker profiles cannot select a baseline mode"
            )
        for baseline_id in ("serial", "two_stream"):
            baseline = by_id[baseline_id]
            if (
                not baseline["correctness"]["valid"]
                or baseline.get("performance") is None
            ):
                raise TackerProfileError(
                    "deployed profile requires valid measured {} baseline".format(
                        baseline_id
                    )
                )
        if not selected["correctness"]["valid"]:
            raise TackerProfileError(
                "deployed selected candidate is not correctness-valid"
            )
        if selected.get("performance") is None:
            raise TackerProfileError(
                "deployed selected candidate has no whole-run performance"
            )
        if not isinstance(selection, dict):
            raise TackerProfileError("deployed profile requires selection evidence")
        _validate_v2_selection(selection, candidates, selected_variant_id)

    provenance = profile.get("provenance")
    if not isinstance(provenance, dict):
        raise TackerProfileError("profile.provenance must be an object")
    sealed_digest = profile.get("profile_sha256")
    _require_sha256(sealed_digest, "profile.profile_sha256")
    if sealed_digest != profile_sha256(profile):
        raise TackerProfileError("profile selection SHA-256 mismatch")
    return profile


def validate_tacker_profile(profile):
    """Validate a schema-v2 profile or a read-only schema-v1 legacy profile."""

    if not isinstance(profile, dict):
        raise TackerProfileError("profile must be a JSON object")
    version = profile.get("schema_version")
    try:
        # Python's decoder accepts non-standard NaN/Infinity tokens.  Validate
        # the complete document, including unsealed human-readable fields, so
        # every accepted profile is finite JSON rather than only hashable in
        # the dispatch-relevant subset.
        _canonical_json(profile)
        if version == LEGACY_PROFILE_SCHEMA_VERSION:
            return _validate_legacy_tacker_profile(profile)
        if version == PROFILE_SCHEMA_VERSION:
            return _validate_v2_tacker_profile(profile)
    except (TypeError, ValueError) as error:
        if isinstance(error, TackerProfileError):
            raise
        raise TackerProfileError("profile contains non-canonical data: {}".format(error))
    raise TackerProfileError(
        "profile.schema_version must be {} or read-only legacy {}".format(
            PROFILE_SCHEMA_VERSION, LEGACY_PROFILE_SCHEMA_VERSION
        )
    )


def selected_tacker_candidate(profile):
    """Return the selected physical candidate in either supported schema."""

    if profile.get("schema_version") == LEGACY_PROFILE_SCHEMA_VERSION:
        candidate = dict(profile["manifest"])
        candidate.update(
            {
                "variant_id": LEGACY_VARIANT_ID,
                "execution_mode": "tacker",
            }
        )
        return candidate
    selected_variant_id = profile["selected_variant_id"]
    for candidate in profile["candidates"]:
        if candidate["variant_id"] == selected_variant_id:
            return candidate
    raise TackerProfileError("selected candidate disappeared after validation")


def load_tacker_profile(profile_path=None, profile_override=None):
    """Load and structurally validate one profile.

    ``profile_override`` is an explicit injection point for tests and for a
    caller that has just produced a measured profile.  It does not bypass the
    manifest/profile hashes or correctness qualification.
    """

    if profile_path is not None and profile_override is not None:
        raise TackerProfileError(
            "profile_path and profile_override are mutually exclusive"
        )
    if profile_override is not None:
        profile = deepcopy(profile_override)
    else:
        path = DEFAULT_PROFILE_PATH if profile_path is None else Path(profile_path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                profile = json.load(handle)
        except (OSError, ValueError) as error:
            raise TackerProfileError(
                "cannot load Tacker profile {}: {}".format(path, error)
            )
    return validate_tacker_profile(profile)


def tacker_profile_admission_reason(profile):
    """Return ``None`` only for a qualified, selected Tacker deployment.

    Performance selection is already sealed into schema v2.  Runtime does not
    reinterpret Raster slowdown, mixed-leaf savings, or per-frame latency.
    Schema-v1 profiles keep only their correctness and enabled-state checks.
    """

    try:
        validate_tacker_profile(profile)
    except TackerProfileError as error:
        return "invalid Tacker profile: {}".format(error)

    if profile["schema_version"] == LEGACY_PROFILE_SCHEMA_VERSION:
        admission = profile["admission"]
        if not admission["enabled"]:
            return "Tacker profile is disabled"
        if not admission["valid"]:
            return "Tacker profile is not marked valid"
        thresholds = profile["thresholds"]
        measured = profile["measurements"]
        if measured["psnr_drop_db"] > thresholds["psnr_drop_db_max"]:
            return "PSNR drop exceeds the profile threshold"
        if measured["ssim_drop"] > thresholds["ssim_drop_max"]:
            return "SSIM drop exceeds the profile threshold"
        if measured["lpips_increase"] > thresholds["lpips_increase_max"]:
            return "LPIPS increase exceeds the profile threshold"
        return None

    deployment = profile["deployment"]
    if not deployment["enabled"]:
        return "Tacker profile deployment is disabled"
    if not deployment["valid"]:
        return "Tacker profile deployment is not marked valid"
    selected = selected_tacker_candidate(profile)
    if selected["execution_mode"] != "tacker":
        return "selected deployment winner is the {} baseline".format(
            selected["execution_mode"]
        )
    if not selected["correctness"]["valid"]:
        return "selected Tacker candidate is not correctness-valid"
    return None


def _capability_value(capabilities, name, default=None):
    if isinstance(capabilities, dict):
        return capabilities.get(name, default)
    return getattr(capabilities, name, default)


def _capability_matches(actual, expected):
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    return actual == expected


def _query_rasterizer_capabilities():
    provider = getattr(_rasterizer_module, "tacker_capabilities", None)
    if provider is None:
        backend = getattr(_rasterizer_module, "_C", None)
        provider = getattr(backend, "tacker_capabilities", None)
    if provider is None:
        raise RuntimeError("rasterizer has no Tacker capability query")
    return provider() if callable(provider) else provider


def _runtime_resource_mapping(value):
    if isinstance(value, dict):
        resources = dict(value)
    else:
        try:
            resources = dict(value)
        except Exception:
            raise RuntimeError("rasterizer resource query returned a non-object")

    aliases = {
        "block_threads": ("block_threads", "physical_threads"),
        "static_shared_memory_bytes": (
            "static_shared_memory_bytes",
            "static_shared_bytes",
        ),
        "max_threads_per_block": (
            "max_threads_per_block",
            "kernel_max_threads_per_block",
        ),
        "active_blocks_per_sm": (
            "active_blocks_per_sm",
            "active_blocks_per_multiprocessor",
        ),
    }
    for target, sources in aliases.items():
        for source in sources:
            if source in resources:
                resources[target] = resources[source]
                break
    return resources


def _query_rasterizer_variant_resources(worker_groups, abi_version=2):
    """Return one normalized active-device resource report.

    The compiled extension's raw query supports both the legacy ABI and the
    multi-head ABI.  Prefer it so C0 is subject to the same launch/occupancy
    checks as C1/C2.  The normalized v2 wrapper remains a compatibility path
    for Python packages that expose the public helper but not ``_C``.
    """

    provider = getattr(_rasterizer_module, "tacker_resource_requirements", None)
    if provider is None:
        backend = getattr(_rasterizer_module, "_C", None)
        provider = getattr(backend, "tacker_resource_requirements", None)
    if callable(provider):
        return _runtime_resource_mapping(provider(abi_version, worker_groups))

    if abi_version == MIXED_MULTI_ABI_VERSION:
        provider = getattr(_rasterizer_module, "tacker_variant_resources", None)
        if provider is None:
            backend = getattr(_rasterizer_module, "_C", None)
            provider = getattr(backend, "tacker_variant_resources", None)
        if callable(provider):
            return _runtime_resource_mapping(provider(worker_groups))
    raise RuntimeError(
        "rasterizer has no ABI-v{} per-variant resource query".format(
            abi_version
        )
    )


def _module_name(module):
    return type(module).__name__


def _linear_shape_reason(module, in_features, out_features, label):
    if _module_name(module) != "Linear":
        return "{} must be Linear".format(label)
    if getattr(module, "in_features", None) != in_features:
        return "{} has the wrong input width".format(label)
    if getattr(module, "out_features", None) != out_features:
        return "{} has the wrong output width".format(label)
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    if weight is None or tuple(getattr(weight, "shape", ())) != (
        out_features,
        in_features,
    ):
        return "{} weight shape is not [{}, {}]".format(
            label, out_features, in_features
        )
    if bias is None or tuple(getattr(bias, "shape", ())) != (out_features,):
        return "{} bias shape is not [{}]".format(label, out_features)
    return None


def _head_structure_reason(network):
    heads = (
        ("pos_deform", 3),
        ("scales_deform", 3),
        ("rotations_deform", 4),
        ("opacity_deform", 1),
        ("shs_deform", 48),
    )
    for name, output_width in heads:
        head = getattr(network, name, None)
        if head is None or _module_name(head) != "Sequential":
            return "{} must be an exact Sequential head".format(name)
        try:
            modules = list(head)
        except TypeError:
            return "{} is not iterable".format(name)
        if len(modules) != 4:
            return "{} must contain exactly four modules".format(name)
        if _module_name(modules[0]) != "ReLU" or _module_name(modules[2]) != "ReLU":
            return "{} must be ReLU/Linear/ReLU/Linear".format(name)
        reason = _linear_shape_reason(modules[1], 128, 128, "{}[1]".format(name))
        if reason is not None:
            return reason
        reason = _linear_shape_reason(
            modules[3], 128, output_width, "{}[3]".format(name)
        )
        if reason is not None:
            return reason
    return None


def _rasterizer_capability_contract_reason(
    capabilities, selected_candidate, variant, manifest
):
    """Compare the sealed candidate with facts exported by the binary."""

    if _capability_value(capabilities, "stream_aware", None) is not True:
        return "rasterizer extension is not stream-aware"
    expected_arch = manifest["cuda_arch"]
    if _capability_value(capabilities, "sm_target", None) != expected_arch:
        return "rasterizer mixed ABI was not built for {}".format(expected_arch)
    if (
        _capability_value(capabilities, "rasterizer_commit", None)
        != manifest["rasterizer_commit"]
    ):
        return "rasterizer compiled commit does not match the sealed profile"

    sealed_raster_range = selected_candidate["raster_thread_range_inclusive"]
    raster_threads = selected_candidate.get(
        "raster_threads", sealed_raster_range[1] - sealed_raster_range[0] + 1
    )
    if _capability_value(capabilities, "raster_threads", None) != raster_threads:
        return "rasterizer capability raster_threads does not match {}".format(
            raster_threads
        )
    expected_raster_range = [0, raster_threads - 1]
    if sealed_raster_range != expected_raster_range:
        return "sealed raster thread range disagrees with compiled thread count"
    if (
        _capability_value(capabilities, "raster_named_barrier_id", None)
        != selected_candidate["raster_named_barrier_id"]
    ):
        return "rasterizer named barrier contract changed"

    expected_mixed_hash = selected_candidate.get(
        "abi_manifest_sha256", MIXED_ABI_SHA256
    )
    if variant.legacy_pos_l1:
        expected = {
            "mixed_render_head_abi": 1,
            "mixed_render_head": True,
            "mixed_symbol": variant.cuda_symbol,
            "mixed_threads": variant.physical_cta_threads,
            "head_threads": 128,
            "head_thread_base": 256,
            "head_features": 128,
            "mixed_manifest_sha256": expected_mixed_hash,
            "head_manifest_sha256": HEAD_ABI_SHA256,
        }
        for key, value in expected.items():
            if not _capability_matches(
                _capability_value(capabilities, key, None), value
            ):
                return "rasterizer capability {} does not match {}".format(
                    key, value
                )
        expected_head_range = [
            expected["head_thread_base"],
            expected["head_thread_base"] + expected["head_threads"] - 1,
        ]
        sealed_head_range = selected_candidate.get("head_thread_range_inclusive")
        if sealed_head_range is None:
            subgroups = selected_candidate.get("backend_subgroups", ())
            if len(subgroups) == 1:
                sealed_head_range = subgroups[0].get("thread_range_inclusive")
        if sealed_head_range != expected_head_range:
            return "sealed head thread range disagrees with compiled capabilities"
        if not callable(getattr(GaussianRasterizer, "forward_with_head", None)):
            return "GaussianRasterizer.forward_with_head is unavailable"
        return None

    tensor_contract = selected_candidate["tensor_contract"]
    max_heads = tensor_contract["max_heads"]
    worker_threads = selected_candidate["backend_subgroups"][0]["threads"]
    barriers = selected_candidate["backend_named_barriers"]
    barrier = barriers[0]
    expected = {
        "mixed_render_heads_abi": MIXED_MULTI_ABI_VERSION,
        "mixed_render_heads": True,
        "mixed_multi_symbol": variant.cuda_symbol,
        "mixed_multi_manifest_sha256": expected_mixed_hash,
        "head_multi_manifest_sha256": selected_candidate[
            "head_abi_manifest_sha256"
        ],
        "head_features": tensor_contract["features"],
        "max_head_tasks": max_heads,
        "max_mixed_heads": max_heads,
        "min_worker_groups": 1,
        "max_worker_groups": max_heads,
        "worker_group_threads": worker_threads,
        "head_descriptor_named_barrier_id": barrier["id"],
        "resource_query": "tacker_resource_requirements",
    }
    for key, value in expected.items():
        if not _capability_matches(
            _capability_value(capabilities, key, None), value
        ):
            return "rasterizer capability {} does not match {}".format(key, value)

    supported = _capability_value(capabilities, "supported_worker_groups", None)
    try:
        supported = tuple(supported)
    except TypeError:
        return "rasterizer supported worker-group contract is invalid"
    expected_supported = tuple(range(1, max_heads + 1))
    if supported != expected_supported:
        return "rasterizer supported worker-group contract changed"
    if variant.worker_groups not in supported:
        return "rasterizer does not support the selected worker-group count"

    threads_by_groups = _capability_value(
        capabilities, "mixed_threads_by_worker_groups", None
    )
    if not isinstance(threads_by_groups, dict):
        try:
            threads_by_groups = dict(threads_by_groups)
        except Exception:
            return "rasterizer mixed thread-count contract is invalid"
    for worker_groups in expected_supported:
        expected_threads = raster_threads + worker_groups * worker_threads
        if not _capability_matches(
            threads_by_groups.get(worker_groups), expected_threads
        ):
            return "rasterizer mixed thread-count contract changed"
    if threads_by_groups[variant.worker_groups] != variant.physical_cta_threads:
        return "selected mixed CTA thread count changed"

    expected_participants = worker_threads * variant.worker_groups
    if barrier["participants"] != expected_participants:
        return "selected descriptor barrier participant count changed"
    expected_head_begin = raster_threads
    for worker_index, subgroup in enumerate(
        selected_candidate["backend_subgroups"]
    ):
        begin = expected_head_begin + worker_index * worker_threads
        if subgroup["thread_range_inclusive"] != [begin, begin + worker_threads - 1]:
            return "sealed head worker range disagrees with compiled capabilities"
        if subgroup["named_barrier_ids"] != [barrier["id"]]:
            return "sealed head worker barrier contract changed"
    if not callable(getattr(GaussianRasterizer, "forward_with_heads", None)):
        return "GaussianRasterizer.forward_with_heads is unavailable"
    return None


def _rasterizer_resource_contract_reason(
    runtime_resources,
    selected_candidate,
    variant,
    manifest,
    require_measured,
):
    """Validate launch feasibility and compare sealed measured resources."""

    try:
        runtime_resources = _runtime_resource_mapping(runtime_resources)
    except RuntimeError as error:
        return str(error)
    expected_threads = variant.physical_cta_threads
    expected_worker_groups = variant.worker_groups
    expected_capability = manifest["compute_capability"]
    exact_runtime = {
        "abi_version": variant.abi_version,
        "worker_groups": expected_worker_groups,
        "block_threads": expected_threads,
        "compute_capability_major": expected_capability[0],
        "compute_capability_minor": expected_capability[1],
    }
    for key, value in exact_runtime.items():
        actual = runtime_resources.get(key)
        if type(value) is int:
            matches = type(actual) is int and actual == value
        else:
            matches = actual == value
        if not matches:
            return "rasterizer runtime resource {} does not match {}".format(
                key, value
            )
    if runtime_resources.get("launch_supported") is not True:
        return "selected mixed CTA is not launch-supported"

    max_threads = runtime_resources.get("max_threads_per_block")
    if type(max_threads) is not int or expected_threads > max_threads:
        return "selected mixed CTA exceeds the compiled maximum thread count"
    device_max_threads = runtime_resources.get("device_max_threads_per_block")
    if type(device_max_threads) is not int or expected_threads > device_max_threads:
        return "selected mixed CTA exceeds the device maximum thread count"
    active_blocks = runtime_resources.get("active_blocks_per_sm")
    if type(active_blocks) is not int or active_blocks < 1:
        return "selected mixed CTA has zero runtime occupancy"
    occupancy = runtime_resources.get("occupancy")
    if not _is_finite_number(occupancy) or float(occupancy) <= 0.0:
        return "selected mixed CTA has zero runtime occupancy"

    measured_resources = selected_candidate.get("resources")
    if require_measured and not isinstance(measured_resources, dict):
        return "selected Tacker candidate has no sealed measured resources"
    if isinstance(measured_resources, dict):
        for key, expected in measured_resources.items():
            if expected is None:
                continue
            if key not in runtime_resources:
                return "rasterizer resource {} is unavailable at runtime".format(
                    key
                )
            if runtime_resources[key] != expected:
                return "rasterizer resource {} changed from profile".format(key)
    return None


def tacker_support_reason(
    pc,
    pipe,
    profile,
    stage="fine",
    cam_type=None,
    override_color=None,
    workload_name=None,
    iteration=None,
    qualification_mode=False,
):
    """Return the first reason the physical two-task path cannot be admitted."""

    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if torch.is_grad_enabled():
        return "autograd is enabled"
    if stage != "fine":
        return "only the exact fine inference stage is supported"
    if cam_type == "PanopticSports":
        return "PanopticSports is not supported"
    if cam_type != "dynerf":
        return "the first Tacker profile is restricted to dynerf"
    if bool(getattr(pipe, "debug", False)):
        return "rasterizer debug mode is unsupported"
    if bool(getattr(pipe, "compute_cov3D_python", False)):
        return "compute_cov3D_python requires fallback"
    if bool(getattr(pipe, "convert_SHs_python", False)):
        return "convert_SHs_python requires fallback"
    if override_color is not None:
        return "override_color requires fallback"

    if qualification_mode:
        try:
            validate_tacker_profile(profile)
        except TackerProfileError as error:
            return "invalid Tacker qualification profile: {}".format(error)
    else:
        profile_reason = tacker_profile_admission_reason(profile)
        if profile_reason is not None:
            return profile_reason
    selected_candidate = selected_tacker_candidate(profile)
    if selected_candidate.get("execution_mode") != "tacker":
        return "selected profile candidate is not a physical Tacker variant"
    manifest = profile["manifest"]
    if workload_name != manifest["workload"]:
        return "workload name does not match the admitted profile"
    if iteration != manifest["iteration"]:
        return "checkpoint iteration does not match the admitted profile"

    deformation = getattr(pc, "_deformation", None)
    if deformation is None:
        return "model has no deformation network"
    if bool(getattr(deformation, "training", True)):
        return "deformation network is not in eval mode"
    network = getattr(deformation, "deformation_net", None)
    if network is None:
        return "model has no inner deformation network"
    if getattr(network, "W", None) != 128:
        return "deformation width W must be 128"
    if getattr(network, "D", None) != 0:
        return "defor_depth must be 0"
    args = getattr(network, "args", None)
    if args is None:
        return "deformation arguments are unavailable"
    try:
        variant = fusion_variant_from_candidate(selected_candidate)
    except TackerProfileError as error:
        return "selected fusion partition is invalid: {}".format(error)
    for head_name in variant.selected_heads:
        disable_flag = HEAD_DISABLE_FLAGS[head_name]
        if bool(getattr(args, disable_flag, False)):
            return "{} must be false for selected {} head".format(
                disable_flag, head_name
            )
    structure_reason = _head_structure_reason(network)
    if structure_reason is not None:
        return structure_reason

    xyz_shape = tuple(getattr(getattr(pc, "get_xyz", None), "shape", ()))
    expected_count = manifest["gaussian_count"]
    if not xyz_shape or xyz_shape[0] != expected_count:
        return "Gaussian count does not match the admitted profile"

    try:
        capabilities = _query_rasterizer_capabilities()
    except Exception as error:
        return "rasterizer capability query failed: {}".format(error)
    capability_reason = _rasterizer_capability_contract_reason(
        capabilities, selected_candidate, variant, manifest
    )
    if capability_reason is not None:
        return capability_reason
    try:
        runtime_resources = _query_rasterizer_variant_resources(
            variant.worker_groups, variant.abi_version
        )
    except Exception as error:
        return "rasterizer variant resource query failed: {}".format(error)
    resource_reason = _rasterizer_resource_contract_reason(
        runtime_resources,
        selected_candidate,
        variant,
        manifest,
        require_measured=profile["schema_version"] == PROFILE_SCHEMA_VERSION,
    )
    if resource_reason is not None:
        return resource_reason

    device = getattr(getattr(pc, "get_xyz", None), "device", None)
    try:
        compute_capability = tuple(torch.cuda.get_device_capability(device))
    except Exception as error:
        return "cannot query GPU compute capability: {}".format(error)
    if compute_capability != (8, 6):
        return "the first Tacker path requires GPU capability 8.6"
    try:
        gpu_name = " ".join(str(torch.cuda.get_device_name(device)).split())
    except Exception as error:
        return "cannot query GPU name: {}".format(error)
    if gpu_name != manifest["gpu_name"]:
        return "GPU name does not match the admitted NVIDIA RTX A6000 profile"

    return None


def _poc_fre(input_data, poc_buf):
    """Exact local copy of ``scene.deformation.poc_fre``."""

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    return torch.cat([input_data, input_data_sin, input_data_cos], -1)


def _record_tensor_stream(value, stream):
    if bool(getattr(value, "is_cuda", False)) and callable(
        getattr(value, "record_stream", None)
    ):
        value.record_stream(stream)


def _view_profile_reason(view, profile):
    """Validate the first camera against the profile-bound raster workload."""

    manifest = profile["manifest"]
    expected_width, expected_height = manifest["resolution"]
    width = getattr(view, "image_width", None)
    height = getattr(view, "image_height", None)
    if width != expected_width or height != expected_height:
        return "camera resolution does not match the admitted profile"
    return None


@dataclass(frozen=True)
class FusionVariant:
    """Validated physical partition selected by an offline profile."""

    variant_id: str
    selected_heads: tuple
    worker_groups: int
    persistent_blocks: int
    abi_version: int
    cuda_symbol: str
    physical_cta_threads: int
    legacy_pos_l1: bool = False


def fusion_variant_from_candidate(candidate):
    """Resolve one validated candidate into a runtime-only descriptor."""

    if not isinstance(candidate, dict) or candidate.get("execution_mode") != "tacker":
        raise TackerProfileError("selected candidate is not a physical Tacker variant")
    partition = candidate.get("partition")
    if partition is None:
        return FusionVariant(
            variant_id=candidate.get("variant_id", LEGACY_VARIANT_ID),
            selected_heads=("pos",),
            worker_groups=1,
            persistent_blocks=candidate.get("persistent_blocks", 0),
            abi_version=1,
            cuda_symbol="tacker_mix_render_head_v1",
            physical_cta_threads=384,
            legacy_pos_l1=True,
        )
    selected_heads = _canonical_head_names(
        partition.get("selected_heads"), "candidate.partition"
    )
    return FusionVariant(
        variant_id=candidate["variant_id"],
        selected_heads=selected_heads,
        worker_groups=partition["worker_groups"],
        persistent_blocks=candidate["persistent_blocks"],
        abi_version=MIXED_MULTI_ABI_VERSION,
        cuda_symbol=candidate["cuda_symbol"],
        physical_cta_threads=candidate["physical_cta_threads"],
        legacy_pos_l1=False,
    )


def _record_stream_tree(value, stream):
    if isinstance(value, dict):
        for item in value.values():
            _record_stream_tree(item, stream)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _record_stream_tree(item, stream)
        return
    _record_tensor_stream(value, stream)


@dataclass
class FusionTask:
    """Candidate-owned prefix, parallel values, and physical output storage."""

    context: object
    variant: object
    point_emb: object
    scales_emb: object
    rotations_emb: object
    opacity_emb: object
    shs_emb: object
    hidden: object
    mask: object
    head_inputs: object
    head_weights: object
    head_biases: object
    parallel_outputs: object
    prefix_ready: object
    skipped_python_nodes: object
    executed_python_nodes: object
    mixed_outputs: object = None
    # Compatibility fields for the read-only v1 profile and its tests.
    head_input: object = None
    head_weight: object = None
    head_bias: object = None
    head_output: object = None
    scale_delta: object = None
    rotation_delta: object = None
    opacity_delta: object = None
    shs_delta: object = None

    def record_stream(self, stream):
        for field in fields(self):
            _record_stream_tree(getattr(self, field.name), stream)

    def assert_selected_nodes_not_executed_by_python(self):
        duplicate = sorted(
            set(self.skipped_python_nodes).intersection(self.executed_python_nodes)
        )
        if duplicate:
            raise RuntimeError(
                "selected fusion nodes were executed twice: {}".format(
                    ", ".join(duplicate)
                )
            )


# Source compatibility for callers that imported the Phase-1 task name.
PosHeadTask = FusionTask


def _cache_head_parameters(pc, head_names):
    network = pc._deformation.deformation_net
    weights = []
    biases = []
    for head_name in head_names:
        selected = getattr(network, HEAD_MODULES[head_name])[1]
        weights.append(
            selected.weight.detach().to(dtype=torch.float16).contiguous()
        )
        biases.append(
            selected.bias.detach().to(dtype=torch.float32).contiguous()
        )
    return tuple(weights), tuple(biases)


def _cache_pos_head_parameters(pc):
    weights, biases = _cache_head_parameters(pc, ("pos",))
    return weights[0], biases[0]


def prepare_fusion_task(
    context,
    pc,
    deform_stream,
    variant,
    head_weights=None,
    head_biases=None,
    prefix_ready=None,
):
    """Enqueue a candidate's real prefix and all independent Python heads.

    The event is recorded after every selected input has been materialized and
    before any non-selected head is submitted.  This is the exact dependency
    edge consumed by the physical Raster stream.
    """

    if not isinstance(variant, FusionVariant):
        variant = fusion_variant_from_candidate(variant)
    if prefix_ready is None:
        prefix_ready = torch.cuda.Event(blocking=False)
    if head_weights is None or head_biases is None:
        head_weights, head_biases = _cache_head_parameters(
            pc, variant.selected_heads
        )
    if len(head_weights) != len(variant.selected_heads) or len(head_biases) != len(
        variant.selected_heads
    ):
        raise ValueError("cached head parameter count does not match selected heads")

    with torch.cuda.stream(deform_stream):
        deformation = pc._deformation
        network = deformation.deformation_net
        point_emb = _poc_fre(context.means3D, deformation.pos_poc)
        scales_emb = _poc_fre(context.scales, deformation.rotation_scaling_poc)
        rotations_emb = _poc_fre(
            context.rotations, deformation.rotation_scaling_poc
        )
        hidden = network.query_time(
            point_emb,
            scales_emb,
            rotations_emb,
            None,
            context.timestamp,
        )

        args = network.args
        if bool(getattr(args, "static_mlp", False)):
            mask = network.static_mlp(hidden)
        elif bool(getattr(args, "empty_voxel", False)):
            mask = network.empty_voxel(point_emb[:, :3])
        else:
            mask = torch.ones_like(context.opacity[:, 0]).unsqueeze(-1)

        head_inputs = []
        for head_name in variant.selected_heads:
            head = getattr(network, HEAD_MODULES[head_name])
            # head[1] is intentionally skipped here and in finish_fusion_task.
            head_inputs.append(
                head[0](hidden).to(dtype=torch.float16).contiguous()
            )
        head_inputs = tuple(head_inputs)
        prefix_ready.record(deform_stream)

        parallel_outputs = {}
        executed_python_nodes = []
        for head_name in HEAD_ORDER:
            if head_name in variant.selected_heads:
                continue
            if bool(getattr(args, HEAD_DISABLE_FLAGS[head_name], False)):
                parallel_outputs[head_name] = None
                continue
            parallel_outputs[head_name] = getattr(
                network, HEAD_MODULES[head_name]
            )(hidden)
            executed_python_nodes.append(_full_head_node(head_name))

    task = FusionTask(
        context=context,
        variant=variant,
        point_emb=point_emb,
        scales_emb=scales_emb,
        rotations_emb=rotations_emb,
        opacity_emb=context.opacity,
        shs_emb=context.shs,
        hidden=hidden,
        mask=mask,
        head_inputs=head_inputs,
        head_weights=tuple(head_weights),
        head_biases=tuple(head_biases),
        parallel_outputs=parallel_outputs,
        prefix_ready=prefix_ready,
        skipped_python_nodes=tuple(
            _first_linear_node(name) for name in variant.selected_heads
        ),
        executed_python_nodes=executed_python_nodes,
    )
    if len(head_inputs) == 1:
        task.head_input = head_inputs[0]
        task.head_weight = head_weights[0]
        task.head_bias = head_biases[0]
    task.scale_delta = parallel_outputs.get("scales")
    task.rotation_delta = parallel_outputs.get("rotations")
    task.opacity_delta = parallel_outputs.get("opacity")
    task.shs_delta = parallel_outputs.get("shs")
    task.assert_selected_nodes_not_executed_by_python()
    return task


def prepare_pos_head_task(
    context,
    pc,
    deform_stream,
    head_weight=None,
    head_bias=None,
    prefix_ready=None,
):
    """Compatibility wrapper for the Phase-1 positional-head partition."""

    if head_weight is None or head_bias is None:
        head_weight, head_bias = _cache_pos_head_parameters(pc)
    variant = FusionVariant(
        variant_id=LEGACY_VARIANT_ID,
        selected_heads=("pos",),
        worker_groups=1,
        persistent_blocks=0,
        abi_version=1,
        cuda_symbol="tacker_mix_render_head_v1",
        physical_cta_threads=384,
        legacy_pos_l1=True,
    )
    return prepare_fusion_task(
        context,
        pc,
        deform_stream,
        variant,
        head_weights=(head_weight,),
        head_biases=(head_bias,),
        prefix_ready=prefix_ready,
    )


def _batch_quaternion_multiply(left, right):
    from utils.graphics_utils import batch_quaternion_multiply

    return batch_quaternion_multiply(left, right)


def _normalise_head_outputs(task, head_outputs):
    if isinstance(head_outputs, dict):
        if set(head_outputs) != set(task.variant.selected_heads):
            raise ValueError("mixed head output names do not match selected heads")
        return {name: head_outputs[name] for name in task.variant.selected_heads}
    if len(task.variant.selected_heads) == 1 and not isinstance(
        head_outputs, (list, tuple)
    ):
        values = (head_outputs,)
    else:
        if not isinstance(head_outputs, (list, tuple)):
            raise TypeError("mixed head outputs must be a sequence")
        values = tuple(head_outputs)
    if len(values) != len(task.variant.selected_heads):
        raise ValueError("mixed head output count does not match selected heads")
    return dict(zip(task.variant.selected_heads, values))


def _validate_head_output(task, head_name, value):
    dtype = getattr(value, "dtype", None)
    if dtype is not None and str(dtype) not in (
        "float32",
        "float",
        "torch.float32",
    ):
        raise TypeError("mixed {} head output must be FP32".format(head_name))
    shape = tuple(getattr(value, "shape", ()))
    if shape and (len(shape) != 2 or shape[1] != 128):
        raise ValueError(
            "mixed {} head output must have shape [N, 128]".format(head_name)
        )
    hidden_shape = tuple(getattr(task.hidden, "shape", ()))
    if shape and hidden_shape and shape[0] != hidden_shape[0]:
        raise ValueError("mixed head output row count changed")


def finish_fusion_task(task, pc, head_outputs):
    """Consume all physical outputs and construct one unique render state."""

    outputs = _normalise_head_outputs(task, head_outputs)
    network = pc._deformation.deformation_net
    args = network.args
    deltas = dict(task.parallel_outputs)
    for head_name in task.variant.selected_heads:
        value = outputs[head_name]
        _validate_head_output(task, head_name, value)
        head = getattr(network, HEAD_MODULES[head_name])
        deltas[head_name] = head[2:](value)
        task.executed_python_nodes.extend(_suffix_head_nodes(head_name))

    task.mixed_outputs = outputs
    if len(outputs) == 1:
        task.head_output = outputs[task.variant.selected_heads[0]]
    task.scale_delta = deltas.get("scales")
    task.rotation_delta = deltas.get("rotations")
    task.opacity_delta = deltas.get("opacity")
    task.shs_delta = deltas.get("shs")
    task.assert_selected_nodes_not_executed_by_python()

    if bool(getattr(args, "no_dx", False)):
        pts = task.point_emb[:, :3]
    else:
        pts = torch.zeros_like(task.point_emb[:, :3])
        pts = task.point_emb[:, :3] * task.mask + deltas["pos"]

    if bool(getattr(args, "no_ds", False)):
        scales = task.scales_emb[:, :3]
    else:
        scales = torch.zeros_like(task.scales_emb[:, :3])
        scales = task.scales_emb[:, :3] * task.mask + deltas["scales"]

    if bool(getattr(args, "no_dr", False)):
        rotations = task.rotations_emb[:, :4]
    else:
        rotations = torch.zeros_like(task.rotations_emb[:, :4])
        if bool(getattr(args, "apply_rotation", False)):
            rotations = _batch_quaternion_multiply(
                task.rotations_emb, deltas["rotations"]
            )
        else:
            rotations = task.rotations_emb[:, :4] + deltas["rotations"]

    if bool(getattr(args, "no_do", False)):
        opacity = task.opacity_emb[:, :1]
    else:
        opacity = torch.zeros_like(task.opacity_emb[:, :1])
        opacity = task.opacity_emb[:, :1] * task.mask + deltas["opacity"]

    if bool(getattr(args, "no_dshs", False)):
        shs = task.shs_emb
    else:
        dshs = deltas["shs"].reshape([task.shs_emb.shape[0], 16, 3])
        shs = torch.zeros_like(task.shs_emb)
        shs = task.shs_emb * task.mask.unsqueeze(-1) + dshs

    return GaussianRenderState(
        means3D=pts,
        scales=pc.scaling_activation(scales),
        rotations=pc.rotation_activation(rotations),
        opacities=pc.opacity_activation(opacity),
        shs=shs,
    )


def finish_pos_head_task(task, pc, head_output):
    """Compatibility wrapper for the Phase-1 positional-head suffix."""

    return finish_fusion_task(task, pc, head_output)


def _forward_with_head(context, state, task, persistent_blocks):
    """Small adapter around the fixed physical binding API."""

    return context.rasterizer.forward_with_head(
        means3D=state.means3D,
        means2D=context.means2D,
        opacities=state.opacities,
        head_input=task.head_input,
        head_weight=task.head_weight,
        head_bias=task.head_bias,
        shs=state.shs,
        colors_precomp=context.colors_precomp,
        scales=state.scales,
        rotations=state.rotations,
        cov3D_precomp=context.cov3D_precomp,
        persistent_blocks=persistent_blocks,
    )


def _forward_with_heads(context, state, task, persistent_blocks):
    """Adapter around the schema-v2 sequence-based mixed binding."""

    method = getattr(context.rasterizer, "forward_with_heads", None)
    if not callable(method):
        raise RuntimeError("GaussianRasterizer.forward_with_heads is unavailable")
    return method(
        means3D=state.means3D,
        means2D=context.means2D,
        opacities=state.opacities,
        head_inputs=task.head_inputs,
        head_weights=task.head_weights,
        head_biases=task.head_biases,
        shs=state.shs,
        colors_precomp=context.colors_precomp,
        scales=state.scales,
        rotations=state.rotations,
        cov3D_precomp=context.cov3D_precomp,
        worker_groups=task.variant.worker_groups,
        persistent_blocks=persistent_blocks,
    )


def _result_from_mixed(context, mixed_outputs):
    image, radii, depth, head_output = mixed_outputs
    result = RenderResult(
        render=image,
        viewspace_points=context.screenspace_points,
        visibility_filter=radii > 0,
        radii=radii,
        depth=depth,
    )
    return result, head_output


def _result_from_mixed_heads(context, mixed_outputs, expected_head_count):
    if not isinstance(mixed_outputs, (list, tuple)) or len(mixed_outputs) < 4:
        raise RuntimeError("mixed v2 binding returned an invalid result")
    image, radii, depth = mixed_outputs[:3]
    if len(mixed_outputs) == 4 and isinstance(mixed_outputs[3], (list, tuple)):
        head_outputs = tuple(mixed_outputs[3])
    else:
        head_outputs = tuple(mixed_outputs[3:])
    if len(head_outputs) != expected_head_count:
        raise RuntimeError("mixed v2 binding returned the wrong head count")
    result = RenderResult(
        render=image,
        viewspace_points=context.screenspace_points,
        visibility_filter=radii > 0,
        radii=radii,
        depth=depth,
    )
    return result, head_outputs


class FusionPartition:
    """Candidate interface used by the two-slot scheduler."""

    def __init__(self, variant):
        self.variant = variant

    def cache_parameters(self, pc):
        return _cache_head_parameters(pc, self.variant.selected_heads)

    def prepare(self, context, pc, deform_stream, cached, prefix_ready):
        weights, biases = cached
        return prepare_fusion_task(
            context,
            pc,
            deform_stream,
            self.variant,
            head_weights=weights,
            head_biases=biases,
            prefix_ready=prefix_ready,
        )

    def launch_mixed(self, context, state, task):
        return _forward_with_heads(
            context, state, task, self.variant.persistent_blocks
        )

    def result_from_mixed(self, context, mixed_outputs):
        return _result_from_mixed_heads(
            context, mixed_outputs, len(self.variant.selected_heads)
        )

    def finish(self, task, pc, head_outputs):
        return finish_fusion_task(task, pc, head_outputs)


class LegacyPosFusionPartition(FusionPartition):
    """Adapter preserving the sealed v1 ABI without leaking it into slots."""

    def cache_parameters(self, pc):
        weight, bias = _cache_pos_head_parameters(pc)
        return (weight,), (bias,)

    def prepare(self, context, pc, deform_stream, cached, prefix_ready):
        weights, biases = cached
        return prepare_pos_head_task(
            context,
            pc,
            deform_stream,
            head_weight=weights[0],
            head_bias=biases[0],
            prefix_ready=prefix_ready,
        )

    def launch_mixed(self, context, state, task):
        return _forward_with_head(
            context, state, task, self.variant.persistent_blocks
        )

    def result_from_mixed(self, context, mixed_outputs):
        return _result_from_mixed(context, mixed_outputs)

    def finish(self, task, pc, head_output):
        return finish_pos_head_task(task, pc, head_output)


def resolve_fusion_partition(candidate):
    variant = fusion_variant_from_candidate(candidate)
    if variant.legacy_pos_l1:
        return LegacyPosFusionPartition(variant)
    return FusionPartition(variant)


@dataclass
class _TackerSlot:
    ready: object
    prefix_ready: object
    mixed_done: object
    raster_done: object
    context: object = None
    state: object = None
    task: object = None
    mixed_outputs: object = None
    has_raster_done: bool = False


class TackerRenderer:
    """Two-slot physical pipeline for ``R(t) + head(t+1)``.

    Unsupported or unmeasured configurations run through a real
    ``TwoStreamRenderer`` instance.  ``last_fallback_reason`` distinguishes
    that case from an admitted physical Tacker execution.
    """

    def __init__(
        self,
        pc,
        pipe,
        bg_color,
        scaling_modifier=1.0,
        stage="fine",
        cam_type=None,
        profile_path=None,
        profile_override=None,
        workload_name=None,
        iteration=None,
        qualification_mode=False,
    ):
        self.pc = pc
        self.pipe = pipe
        self.bg_color = bg_color
        self.scaling_modifier = scaling_modifier
        self.stage = stage
        self.cam_type = cam_type
        self.workload_name = workload_name
        self.iteration = iteration
        self.qualification_mode = qualification_mode
        self.device = getattr(pc.get_xyz, "device", None)

        self._profile_error = None
        self.profile = None
        if type(qualification_mode) is not bool:
            self._profile_error = "qualification_mode must be boolean"
        elif qualification_mode and (
            profile_override is None or profile_path is not None
        ):
            self._profile_error = (
                "qualification_mode requires an explicit profile_override and "
                "forbids profile_path/default-profile admission"
            )
        try:
            if self._profile_error is None:
                self.profile = load_tacker_profile(
                    profile_path=profile_path,
                    profile_override=profile_override,
                )
        except TackerProfileError as error:
            self.profile = None
            self._profile_error = str(error)

        # Resolve the selected partition and convert its immutable parameters
        # once, outside the render loop and all timing ranges.
        self.partition = None
        self.cached_head_parameters = None
        self.head_weight = None
        self.head_bias = None
        self._cache_ready = None
        self._cache_error = None
        try:
            if self.profile is None:
                # Preserve the Phase-1 prepare/synchronize contract even when
                # profile parsing fails; this work is constructor-time and is
                # never admitted into a measured render sequence.
                self.head_weight, self.head_bias = _cache_pos_head_parameters(pc)
                self.cached_head_parameters = (
                    (self.head_weight,),
                    (self.head_bias,),
                )
                cache_values = (self.head_weight,)
            else:
                candidate = selected_tacker_candidate(self.profile)
                if candidate.get("execution_mode") == "tacker":
                    self.partition = resolve_fusion_partition(candidate)
                    self.cached_head_parameters = self.partition.cache_parameters(pc)
                    weights, biases = self.cached_head_parameters
                    if len(weights) == 1:
                        self.head_weight = weights[0]
                        self.head_bias = biases[0]
                    cache_values = weights
                else:
                    cache_values = ()
            if any(bool(getattr(value, "is_cuda", False)) for value in cache_values):
                self._cache_ready = torch.cuda.Event(blocking=False)
                self._cache_ready.record(torch.cuda.current_stream(self.device))
        except Exception as error:
            self._cache_error = str(error)

        self.persistent_blocks = 0
        if self.partition is not None:
            self.persistent_blocks = self.partition.variant.persistent_blocks

        self.raster_stream = None
        self.deform_stream = None
        self.slots = None
        self._active = False
        self._last_fallback_reason = None
        self._last_used_tacker = False
        self._last_qualification_mode = False
        self._last_caller_stream = None
        self._last_execution_counts = None
        self._fallback_renderer = TwoStreamRenderer(
            pc,
            pipe,
            bg_color,
            scaling_modifier=scaling_modifier,
            stage=stage,
            cam_type=cam_type,
        )

    @property
    def fallback_reason(self):
        if self._profile_error is not None:
            return "invalid Tacker profile: {}".format(self._profile_error)
        reason = tacker_support_reason(
            self.pc,
            self.pipe,
            self.profile,
            stage=self.stage,
            cam_type=self.cam_type,
            override_color=None,
            workload_name=self.workload_name,
            iteration=self.iteration,
            qualification_mode=self.qualification_mode,
        )
        if reason is None and self._cache_error is not None:
            return "cannot cache selected-head parameters: {}".format(
                self._cache_error
            )
        return reason

    @property
    def last_fallback_reason(self):
        return self._last_fallback_reason

    @property
    def last_used_tacker(self):
        return self._last_used_tacker

    @property
    def last_qualification_mode(self):
        """Whether the last physical run bypassed measured admission explicitly."""

        return self._last_qualification_mode

    @property
    def last_execution_counts(self):
        """Return auditable scheduler counts for the latest sequence."""

        return deepcopy(self._last_execution_counts)

    def _assert_execution_counts(self):
        counts = self._last_execution_counts
        if counts is None or not self._last_used_tacker:
            return
        frame_count = counts["input_frames"]
        mixed_count = max(frame_count - 1, 0)
        expected = {
            "full_deformation": 1,
            "prefix": mixed_count,
            "mixed_launches": mixed_count,
            "suffix": mixed_count,
            "solo_raster": 1,
            "outputs": frame_count,
            "selected_head_evaluations_per_head": frame_count,
        }
        for name, value in expected.items():
            if counts.get(name) != value:
                raise RuntimeError(
                    "Tacker sequence execution count {} changed: {} != {}"
                    .format(name, counts.get(name), value)
                )

    @property
    def fallback_backend_reason(self):
        """Why the delegated two-stream renderer itself selected serial."""

        if self._last_used_tacker:
            return None
        return getattr(self._fallback_renderer, "last_fallback_reason", None)

    @property
    def actual_execution_mode(self):
        """Return ``not_run``, ``tacker``, ``two_stream``, or ``serial``."""

        if self._last_fallback_reason is None and not self._last_used_tacker:
            return "not_run"
        if self._last_used_tacker:
            return "tacker"
        if self.fallback_backend_reason is not None:
            return "serial"
        return "two_stream"

    def _ensure_cuda_resources(self):
        if self.raster_stream is not None:
            return
        self.raster_stream = torch.cuda.Stream(device=self.device)
        self.deform_stream = torch.cuda.Stream(device=self.device)
        self.slots = [
            _TackerSlot(
                ready=torch.cuda.Event(blocking=False),
                prefix_ready=torch.cuda.Event(blocking=False),
                mixed_done=torch.cuda.Event(blocking=False),
                raster_done=torch.cuda.Event(blocking=False),
            )
            for _ in range(2)
        ]
        if self._cache_ready is not None:
            self.raster_stream.wait_event(self._cache_ready)
            self.deform_stream.wait_event(self._cache_ready)

    def prepare(self):
        """Finish one-time parameter/resource setup before timed rendering."""

        reason = self.fallback_reason
        if reason is None:
            self._ensure_cuda_resources()
        else:
            prepare_fallback = getattr(self._fallback_renderer, "prepare", None)
            if callable(prepare_fallback):
                prepare_fallback()
        if self._cache_ready is not None:
            self._cache_ready.synchronize()
        return reason

    @staticmethod
    def _stream_identity(stream):
        return (
            getattr(stream, "device", None),
            getattr(stream, "cuda_stream", id(stream)),
        )

    def _require_caller_stream(self, caller_stream):
        current = torch.cuda.current_stream(self.device)
        if self._stream_identity(current) != self._stream_identity(caller_stream):
            raise RuntimeError(
                "TackerRenderer.render_sequence() must be consumed from the "
                "same CUDA stream that started it"
            )

    def _require_profile_view(self, view):
        reason = _view_profile_reason(view, self.profile)
        if reason is not None:
            # Work already queued for earlier fixed-workload frames must finish
            # before reporting the heterogeneous-view contract violation.
            self.synchronize()
            raise RuntimeError("Tacker workload changed mid-sequence: {}".format(reason))

    def _prepare_context(self, view):
        return prepare_render_context(
            view,
            self.pc,
            self.pipe,
            self.bg_color,
            scaling_modifier=self.scaling_modifier,
            override_color=None,
            cam_type=self.cam_type,
        )

    def _wait_slot_reuse(self, slot):
        if slot.has_raster_done:
            self.deform_stream.wait_event(slot.raster_done)

    def _enqueue_full_deformation(self, view, slot):
        with torch.cuda.stream(self.deform_stream):
            self._wait_slot_reuse(slot)
            slot.context = self._prepare_context(view)
            slot.state = deform_for_render(slot.context, self.pc, stage=self.stage)
            slot.task = None
            slot.mixed_outputs = None
            _record_context_stream(slot.context, self.deform_stream)
            slot.state.record_stream(self.deform_stream)
            slot.ready.record(self.deform_stream)

    def _enqueue_prefix(self, view, slot):
        with torch.cuda.stream(self.deform_stream):
            self._wait_slot_reuse(slot)
            slot.context = self._prepare_context(view)
            slot.state = None
            slot.mixed_outputs = None
            slot.task = self.partition.prepare(
                slot.context,
                self.pc,
                self.deform_stream,
                self.cached_head_parameters,
                slot.prefix_ready,
            )
            _record_context_stream(slot.context, self.deform_stream)
            slot.task.record_stream(self.deform_stream)

    def _enqueue_suffix(self, slot):
        with torch.cuda.stream(self.deform_stream):
            self.deform_stream.wait_event(slot.mixed_done)
            _record_stream_tree(slot.mixed_outputs, self.deform_stream)
            slot.state = self.partition.finish(
                slot.task,
                self.pc,
                slot.mixed_outputs,
            )
            slot.state.record_stream(self.deform_stream)
            slot.ready.record(self.deform_stream)

    def _enqueue_mixed(self, current_slot, next_slot):
        with torch.cuda.stream(self.raster_stream):
            self.raster_stream.wait_event(current_slot.ready)
            self.raster_stream.wait_event(next_slot.prefix_ready)
            current_slot.state.record_stream(self.raster_stream)
            _record_context_stream(current_slot.context, self.raster_stream)
            next_slot.task.record_stream(self.raster_stream)

            mixed = self.partition.launch_mixed(
                current_slot.context,
                current_slot.state,
                next_slot.task,
            )
            result, head_outputs = self.partition.result_from_mixed(
                current_slot.context, mixed
            )
            next_slot.mixed_outputs = head_outputs
            next_slot.task.mixed_outputs = head_outputs
            if len(self.partition.variant.selected_heads) == 1:
                next_slot.task.head_output = head_outputs
            _record_stream_tree(head_outputs, self.raster_stream)
            result.record_stream(self.raster_stream)

            next_slot.mixed_done.record(self.raster_stream)
            current_slot.raster_done.record(self.raster_stream)
            current_slot.has_raster_done = True
        return result

    def _enqueue_solo_raster(self, slot):
        with torch.cuda.stream(self.raster_stream):
            self.raster_stream.wait_event(slot.ready)
            slot.state.record_stream(self.raster_stream)
            _record_context_stream(slot.context, self.raster_stream)
            result = rasterize_state(slot.context, slot.state)
            result.record_stream(self.raster_stream)
            slot.raster_done.record(self.raster_stream)
            slot.has_raster_done = True
        return result

    def render_sequence(self, views):
        """Yield input-ordered frames using prefill, mixed steady state, drain."""

        if self._active:
            raise RuntimeError(
                "TackerRenderer does not support interleaved or reentrant sequences"
            )
        self._active = True
        try:
            iterator = iter(views)
            try:
                current_view = next(iterator)
            except StopIteration:
                self._last_execution_counts = {
                    "input_frames": 0,
                    "full_deformation": 0,
                    "prefix": 0,
                    "mixed_launches": 0,
                    "suffix": 0,
                    "solo_raster": 0,
                    "outputs": 0,
                    "selected_head_evaluations_per_head": 0,
                }
                return

            self._last_execution_counts = {
                "input_frames": 1,
                "full_deformation": 0,
                "prefix": 0,
                "mixed_launches": 0,
                "suffix": 0,
                "solo_raster": 0,
                "outputs": 0,
                "selected_head_evaluations_per_head": 0,
            }

            fallback_reason = self.fallback_reason
            if fallback_reason is None:
                fallback_reason = _view_profile_reason(current_view, self.profile)
            self._last_fallback_reason = fallback_reason
            self._last_used_tacker = fallback_reason is None
            self._last_qualification_mode = bool(
                fallback_reason is None and self.qualification_mode
            )
            if fallback_reason is not None:
                for output in self._fallback_renderer.render_sequence(
                    _prepend(current_view, iterator)
                ):
                    yield output
                return

            self._ensure_cuda_resources()
            caller_stream = torch.cuda.current_stream(self.device)
            self._last_caller_stream = caller_stream
            self.raster_stream.wait_stream(caller_stream)
            self.deform_stream.wait_stream(caller_stream)

            # D(0) is the only full deformation prefill.  Every steady-state
            # D(t+1) below is split around the physical selected head.
            self._enqueue_full_deformation(current_view, self.slots[0])
            self._last_execution_counts["full_deformation"] += 1
            self._last_execution_counts[
                "selected_head_evaluations_per_head"
            ] += 1
            try:
                next_view = next(iterator)
                self._last_execution_counts["input_frames"] += 1
                self._require_profile_view(next_view)
                has_next = True
            except StopIteration:
                next_view = None
                has_next = False

            frame_index = 0
            while True:
                self._require_caller_stream(caller_stream)
                current_slot = self.slots[frame_index % 2]

                if has_next:
                    next_slot = self.slots[(frame_index + 1) % 2]
                    self._enqueue_prefix(next_view, next_slot)
                    self._last_execution_counts["prefix"] += 1
                    try:
                        result = self._enqueue_mixed(current_slot, next_slot)
                        self._last_execution_counts["mixed_launches"] += 1
                        self._last_execution_counts[
                            "selected_head_evaluations_per_head"
                        ] += 1
                    except Exception as error:
                        # A synchronous launch/configuration failure on the
                        # first mixed leaf is replayable because no frame has
                        # reached the caller yet.  Isolate queued private-stream
                        # work before delegating the complete sequence.
                        if self.raster_stream is not None:
                            self.raster_stream.synchronize()
                            self.deform_stream.synchronize()
                        if frame_index != 0:
                            raise RuntimeError(
                                "Tacker mixed launch failed after output was "
                                "published; sequence cannot be replayed safely: {}"
                                .format(error)
                            )
                        self._last_fallback_reason = (
                            "Tacker mixed launch failed before first output: {}"
                            .format(error)
                        )
                        self._last_used_tacker = False
                        self._last_qualification_mode = False
                        replay = _prepend(
                            current_view, _prepend(next_view, iterator)
                        )
                        for output in self._fallback_renderer.render_sequence(replay):
                            yield output
                        return
                    self._enqueue_suffix(next_slot)
                    self._last_execution_counts["suffix"] += 1
                else:
                    result = self._enqueue_solo_raster(current_slot)
                    self._last_execution_counts["solo_raster"] += 1

                self._require_caller_stream(caller_stream)
                caller_stream.wait_event(current_slot.raster_done)
                result.record_stream(caller_stream)
                self._last_execution_counts["outputs"] += 1
                if not has_next:
                    self._assert_execution_counts()
                yield result.as_dict()

                self._require_caller_stream(caller_stream)
                if not has_next:
                    return
                current_view = next_view
                frame_index += 1
                try:
                    next_view = next(iterator)
                    self._last_execution_counts["input_frames"] += 1
                    self._require_profile_view(next_view)
                    has_next = True
                except StopIteration:
                    next_view = None
                    has_next = False
        finally:
            self._active = False

    def synchronize(self):
        """Wait for the backend used by the most recent sequence."""

        if self.actual_execution_mode == "not_run":
            # Parameter conversion is intentionally constructor-time work. An
            # explicit pre-warmup synchronize keeps it outside timed regions.
            if self._cache_ready is not None:
                self._cache_ready.synchronize()
            return
        if self._last_used_tacker:
            if self.raster_stream is not None:
                self.raster_stream.synchronize()
                self.deform_stream.synchronize()
            if self._last_caller_stream is not None:
                self._last_caller_stream.synchronize()
            return
        synchronize_fallback = getattr(self._fallback_renderer, "synchronize", None)
        if callable(synchronize_fallback):
            synchronize_fallback()


def _prepend(first, iterator):
    """Python-3.7-compatible one-element chain without materialising views."""

    yield first
    for item in iterator:
        yield item


__all__ = [
    "DEFAULT_PROFILE_PATH",
    "LEGACY_PROFILE_SCHEMA_VERSION",
    "LEGACY_VARIANT_ID",
    "FIRST_LINEAR_PARTITION_KIND",
    "FusionPartition",
    "FusionTask",
    "FusionVariant",
    "HEAD_MULTI_ABI_SHA256",
    "HEAD_ORDER",
    "MIXED_MULTI_ABI_VERSION",
    "PAIR_KEY",
    "PROFILE_SCHEMA_VERSION",
    "SELECTION_OBJECTIVE",
    "WORKLOAD_KEY",
    "PosHeadTask",
    "TackerProfileError",
    "TackerRenderer",
    "finish_fusion_task",
    "finish_pos_head_task",
    "first_linear_candidate_contract",
    "fusion_variant_from_candidate",
    "load_tacker_profile",
    "manifest_sha256",
    "profile_sha256",
    "prepare_fusion_task",
    "prepare_pos_head_task",
    "resolve_fusion_partition",
    "tacker_profile_admission_reason",
    "tacker_support_reason",
    "selected_tacker_candidate",
    "validate_tacker_profile",
]
