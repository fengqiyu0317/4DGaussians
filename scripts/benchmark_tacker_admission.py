#!/usr/bin/env python3
"""Fail-closed admission report/profile generator for the Tacker renderer.

This script does not run a GPU benchmark.  It first qualifies correctness,
then consumes an interleaved whole-run FPS report to rank only valid
candidates and apply conservative incumbent promotion.  Raster/leaf latency
and per-frame completion intervals are preserved solely as diagnostics.  A
baseline deployment produces a passing report but no enabled Tacker profile.

The module deliberately imports only the Python standard library so its
contract and boundary behaviour can be tested on CPU-only hosts.
"""

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

try:
    # Direct script execution places ``scripts/`` on sys.path.
    from benchmark_tacker_fps import (
        BenchmarkContractError as FPSSelectionError,
        select_candidates as _select_fps_candidates,
    )
except ImportError:  # CPU contract tests import this file from the repo root.
    from scripts.benchmark_tacker_fps import (
        BenchmarkContractError as FPSSelectionError,
        select_candidates as _select_fps_candidates,
    )


# ``SCHEMA_VERSION`` is retained for callers that import the old, single-
# candidate input contract.  New selector reports and deployable profiles use
# ``PROFILE_SCHEMA_VERSION``.  Keeping the names separate is intentional: a
# Phase-0 FPS report is schema v1 input to a schema-v2 selection result.
SCHEMA_VERSION = 1
PROFILE_SCHEMA_VERSION = 2
FPS_BENCHMARK_SCHEMA_VERSION = 1
FPS_BENCHMARK_KIND = "4dgaussians_tacker_fps_benchmark"
SELECTION_OBJECTIVE = "median_throughput_fps"
LEGACY_VARIANT_ID = "legacy_pos_l1"
INCUMBENT_BENCHMARK_NAME = "current_tacker"
WORKLOAD_KEY = "flame_steak:14000:111525:1352x1014:sm_86"
PROMOTION_MIN_FPS_RATIO = 1.01
PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE = 1.0
FORMAL_BOOTSTRAP_RESAMPLES = 10000
FORMAL_BOOTSTRAP_SEED = 0
EQUIVALENCE_FRACTION = 0.005
# The only report allowed to omit Phase-1 correctness/selection evidence is
# the immutable Phase-0 artifact archived in this repository.  Keying the
# compatibility path to its canonical document digest prevents a new report
# from deleting Phase-1 fields and silently acquiring legacy defaults.
FROZEN_PHASE0_FPS_REPORT_SHA256 = (
    "ae90457636387b43b659338cae01d9117efd1f39fcfede5986a75607b7c6e34d"
)
EXPECTED_SCENE = "flame_steak"
EXPECTED_ITERATION = 14000
EXPECTED_RESOLUTION = [1352, 1014]
EXPECTED_GAUSSIANS = 111525
EXPECTED_COMPUTE_CAPABILITY = [8, 6]
EXPECTED_CUDA_ARCH = "sm_86"
EXPECTED_GPU_NAME = "NVIDIA RTX A6000"
EXPECTED_RASTERIZER_COMMIT = "e49506654e8e11ed8a62d22bcb693e943fdecacf"
EXPECTED_PAIR_KEY = (
    "raster.render_leaf+deformation.pos_deform[1].linear_128x128"
)
EXPECTED_MIXED_SYMBOL = "tacker_mix_render_head_v1"
EXPECTED_HEAD_SOLO_SYMBOL = "tacker_head_linear_solo_v1"
EXPECTED_HEAD_GPTB_SYMBOL = "tacker_head_linear_gptb_v1"
EXPECTED_MIXED_ABI_SHA256 = (
    "231c90c429321b2673b88ecd09efb40b6aedda7a23f3e061a2bcedec06d44426"
)

EXPECTED_RASTER_CAPABILITIES = {
    "stream_aware": True,
    "mixed_render_head_abi": 1,
    "mixed_render_head": True,
    "mixed_symbol": EXPECTED_MIXED_SYMBOL,
    "mixed_threads": 384,
    "raster_threads": 256,
    "head_threads": 128,
    "head_thread_base": 256,
    "raster_named_barrier_id": 1,
    "head_features": 128,
    "rasterizer_commit": EXPECTED_RASTERIZER_COMMIT,
    "sm_target": EXPECTED_CUDA_ARCH,
}

EXPECTED_HEAD_CAPABILITIES = {
    "abi_version": 1,
    "sm_target": EXPECTED_CUDA_ARCH,
    "head_features": 128,
    "block_threads": 128,
    "input_dtype": "float16",
    "weight_dtype": "float16",
    "bias_dtype": "float32",
    "accumulation_dtype": "float32",
    "output_dtype": "float32",
    "solo_symbol": EXPECTED_HEAD_SOLO_SYMBOL,
    "gptb_symbol": EXPECTED_HEAD_GPTB_SYMBOL,
}

LEGACY_PROFILE_THRESHOLDS = {
    "raster_slowdown_pct_max": 5.0,
    "mixed_p50_strictly_less_than_solo_sum": True,
    "end_to_end_ratio_max": 1.0,
    "psnr_drop_db_max": 0.05,
    "ssim_drop_max": 0.0001,
    "lpips_increase_max": 0.0001,
}

# Schema v2 has correctness thresholds only.  Raster slowdown, leaf timing,
# completion-interval latency and memory/resource observations remain useful
# diagnostics, but are deliberately absent from this mapping so they cannot
# accidentally become admission gates again.
CORRECTNESS_THRESHOLDS = {
    "psnr_drop_db_max": 0.05,
    "ssim_drop_max": 0.0001,
    "lpips_increase_max": 0.0001,
}

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


class AdmissionInputError(ValueError):
    """Raised when a measurement or ABI input is incomplete or inconsistent."""


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(manifest):
    return hashlib.sha256(_canonical_json(manifest)).hexdigest()


def profile_sha256(profile):
    """Return the stable schema-v2 selection/profile digest.

    The timestamp is operational metadata and therefore excluded.  Everything
    that can affect qualification, ranking, promotion or runtime dispatch is
    sealed, including raw FPS trials and the selected variant.  This helper is
    deliberately public and byte-for-byte aligned with the runtime loader.
    """

    if not isinstance(profile, dict):
        raise AdmissionInputError("profile must be a JSON object")
    payload = {key: profile.get(key) for key in _PROFILE_SEALED_KEYS}
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        provenance.pop("generated_at_utc", None)
        payload["provenance"] = provenance
    try:
        return hashlib.sha256(_canonical_json(payload)).hexdigest()
    except (TypeError, ValueError) as error:
        raise AdmissionInputError(
            "profile contains non-canonical data: {}".format(error)
        )


def _is_finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _evidence_values_equal(actual, expected):
    """Compare sealed evidence structurally with tolerance for float replay."""

    if _is_finite_number(actual) and _is_finite_number(expected):
        return math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _evidence_values_equal(actual[key], expected[key])
            for key in expected
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _evidence_values_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def _is_lower_hex(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(mapping, keys, label, positive=False, nonnegative=False):
    """Return the first present finite number from ``keys``."""

    for key in keys:
        if key in mapping:
            value = mapping[key]
            if not _is_finite_number(value):
                raise AdmissionInputError("{} must be a finite number".format(label))
            value = float(value)
            if positive and value <= 0.0:
                raise AdmissionInputError("{} must be > 0".format(label))
            if nonnegative and value < 0.0:
                raise AdmissionInputError("{} must be >= 0".format(label))
            return value
    raise AdmissionInputError("{} is required".format(label))


def _mapping(value, label):
    if not isinstance(value, dict):
        raise AdmissionInputError("{} must be a JSON object".format(label))
    return value


def _nested_measurements(document):
    document = _mapping(document, "measurement document")
    measurements = document.get("measurements", document)
    return _mapping(measurements, "measurements")


def _normalise_scene(value):
    if not isinstance(value, str):
        return value
    return value.strip().replace("-", "_").lower()


def _workload(document, label):
    """Extract and validate the fixed workload from one input document."""

    document = _mapping(document, label)
    nested = document.get("workload")
    source = nested if isinstance(nested, dict) else document
    scene = source.get(
        "scene",
        source.get("dataset", source.get("workload_name")),
    )
    if scene is None and isinstance(nested, str):
        scene = nested
    if scene is None and isinstance(document.get("source_path"), str):
        scene = Path(document["source_path"].rstrip("/\\")).name
    iteration = source.get("iteration", document.get("iteration"))
    if _normalise_scene(scene) != EXPECTED_SCENE:
        raise AdmissionInputError(
            "{} scene/dataset must be {}".format(label, EXPECTED_SCENE)
        )
    if type(iteration) is not int or iteration != EXPECTED_ITERATION:
        raise AdmissionInputError(
            "{} iteration must be {}".format(label, EXPECTED_ITERATION)
        )

    resolution = source.get("resolution", document.get("resolution"))
    if resolution is None:
        width = source.get("image_width", document.get("image_width"))
        height = source.get("image_height", document.get("image_height"))
        if width is not None or height is not None:
            resolution = [width, height]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(type(value) is not int for value in resolution)
        or resolution != EXPECTED_RESOLUTION
    ):
        raise AdmissionInputError(
            "{} resolution is required and must be {}".format(
                label, EXPECTED_RESOLUTION
            )
        )

    gaussian_count = source.get(
        "gaussian_count", document.get("gaussian_count")
    )
    if type(gaussian_count) is not int or gaussian_count != EXPECTED_GAUSSIANS:
        raise AdmissionInputError(
            "{} gaussian_count is required and must be {}".format(
                label, EXPECTED_GAUSSIANS
            )
        )


def _require_document_contract(document, label, kind, require_numerics=False):
    document = _mapping(document, label)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise AdmissionInputError("{} schema_version must be 1".format(label))
    if document.get("kind") != kind:
        raise AdmissionInputError(
            "{} kind must be {}".format(label, kind)
        )
    if document.get("passed") is not True:
        raise AdmissionInputError("{} must be explicitly marked passed".format(label))
    _workload(document, label)
    if require_numerics:
        numerics = _mapping(document.get("numerics"), "{} numerics".format(label))
        if numerics.get("passed") is not True:
            raise AdmissionInputError(
                "{} numerics must be explicitly marked passed".format(label)
            )
    return document


def _measurement_persistent_blocks(document, label):
    config = _mapping(
        document.get("measurement_config"),
        "{} measurement_config".format(label),
    )
    value = config.get("persistent_blocks")
    if type(value) is not int or value < 0:
        raise AdmissionInputError(
            "{} measurement_config.persistent_blocks must be an int >= 0".format(
                label
            )
        )
    return value


def _leaf_workload_contract(document, label):
    workload = _mapping(document.get("workload"), "{} workload".format(label))
    split = workload.get("split")
    model_path = workload.get("model_path")
    source_path = workload.get("source_path")
    current_indices = workload.get("current_view_indices")
    next_indices = workload.get("next_view_indices")
    if split not in ("train", "test", "video"):
        raise AdmissionInputError("{} workload split is required".format(label))
    if not isinstance(model_path, str) or not model_path:
        raise AdmissionInputError(
            "{} workload model_path is required".format(label)
        )
    if not isinstance(source_path, str) or not source_path:
        raise AdmissionInputError(
            "{} workload source_path is required".format(label)
        )
    if (
        not isinstance(current_indices, list)
        or not isinstance(next_indices, list)
        or not current_indices
        or len(current_indices) != len(next_indices)
        or any(type(index) is not int or index < 0 for index in current_indices)
        or any(type(index) is not int or index < 0 for index in next_indices)
        or any(
            next_index <= current_index
            for current_index, next_index in zip(current_indices, next_indices)
        )
    ):
        raise AdmissionInputError(
            "{} workload must contain ordered non-negative view pairs".format(label)
        )
    return {
        "split": split,
        "model_path": model_path,
        "source_path": source_path,
        "current_view_indices": list(current_indices),
        "next_view_indices": list(next_indices),
    }


def _normalise_capability(value):
    if value == EXPECTED_COMPUTE_CAPABILITY:
        return EXPECTED_COMPUTE_CAPABILITY
    if isinstance(value, tuple) and list(value) == EXPECTED_COMPUTE_CAPABILITY:
        return EXPECTED_COMPUTE_CAPABILITY
    if isinstance(value, str):
        compact = value.strip().lower().replace("compute_", "")
        compact = compact.replace("capability", "").replace(" ", "")
        if compact in ("sm_86", "8.6", "86"):
            return EXPECTED_COMPUTE_CAPABILITY
    if value == 86:
        return EXPECTED_COMPUTE_CAPABILITY
    return None


def _device(document):
    document = _require_document_contract(
        document,
        "device input",
        "4dgaussians_tacker_device",
    )
    extensions = _mapping(document.get("extensions"), "device extensions")
    rasterizer = _mapping(
        extensions.get("rasterizer"), "device rasterizer extension"
    )
    head_extension = _mapping(
        extensions.get("head"), "device head extension"
    )
    raster_capabilities = _mapping(
        rasterizer.get("capabilities"), "device rasterizer capabilities"
    )
    head_capabilities = _mapping(
        head_extension.get("capabilities"), "device head capabilities"
    )
    for key, expected in EXPECTED_RASTER_CAPABILITIES.items():
        if raster_capabilities.get(key) != expected:
            raise AdmissionInputError(
                "device rasterizer capability {} must be {!r}".format(
                    key, expected
                )
            )
    for key, expected in EXPECTED_HEAD_CAPABILITIES.items():
        if head_capabilities.get(key) != expected:
            raise AdmissionInputError(
                "device head capability {} must be {!r}".format(key, expected)
            )
    if rasterizer.get("cuda_global_symbols") != [EXPECTED_MIXED_SYMBOL]:
        raise AdmissionInputError("device rasterizer CUDA symbol provenance changed")
    if head_extension.get("cuda_global_symbols") != [
        EXPECTED_HEAD_SOLO_SYMBOL,
        EXPECTED_HEAD_GPTB_SYMBOL,
    ]:
        raise AdmissionInputError("device head CUDA symbol provenance changed")
    nested = document.get("device")
    source = nested if isinstance(nested, dict) else document
    name = source.get("name", source.get("gpu_name", document.get("gpu_name")))
    capability = source.get(
        "compute_capability",
        source.get("cuda_arch", source.get("sm_target")),
    )
    if not isinstance(name, str) or name.strip() != EXPECTED_GPU_NAME:
        raise AdmissionInputError(
            "device must be an NVIDIA RTX A6000 (got {!r})".format(name)
        )
    if _normalise_capability(capability) is None:
        raise AdmissionInputError("device compute capability must be sm_86 / 8.6")
    return {
        "name": EXPECTED_GPU_NAME,
        "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
        "cuda_arch": EXPECTED_CUDA_ARCH,
    }


def _optional_device_name(document, expected_name, label):
    nested = document.get("device") if isinstance(document, dict) else None
    source = nested if isinstance(nested, dict) else document
    if not isinstance(source, dict):
        return
    name = source.get("name", source.get("gpu_name", document.get("gpu_name")))
    if name is not None and name != expected_name:
        raise AdmissionInputError(
            "{} GPU name does not match device input".format(label)
        )


def _validate_mixed_abi(abi):
    abi = _mapping(abi, "mixed ABI")
    required = {
        "abi_version": 1,
        "rasterizer_upstream_commit": EXPECTED_RASTERIZER_COMMIT,
        "cuda_arch": EXPECTED_CUDA_ARCH,
        "global_kernel_symbol": EXPECTED_MIXED_SYMBOL,
        "python_binding": "rasterize_gaussians_with_head",
        "python_method": "GaussianRasterizer.forward_with_head",
        "capability_query": "tacker_capabilities",
    }
    for key, expected in required.items():
        if abi.get(key) != expected:
            raise AdmissionInputError(
                "mixed ABI {} must be {!r}".format(key, expected)
            )
    launch = _mapping(abi.get("physical_launch"), "mixed ABI physical_launch")
    if launch.get("threads") != 384:
        raise AdmissionInputError("mixed ABI must launch 384 physical threads")
    subgroups = _mapping(abi.get("subgroups"), "mixed ABI subgroups")
    raster = _mapping(subgroups.get("raster"), "mixed ABI raster subgroup")
    head = _mapping(subgroups.get("head"), "mixed ABI head subgroup")
    expected_raster = {
        "thread_range": [0, 255],
        "threads": 256,
        "named_barrier_id": 1,
        "named_barrier_participants": 256,
    }
    expected_head = {
        "thread_range": [256, 383],
        "threads": 128,
        "named_barrier_ids": [],
        "input_dtype": "float16",
        "weight_dtype": "float16",
        "bias_dtype": "float32",
        "accumulation_dtype": "float32",
        "output_dtype": "float32",
    }
    for key, expected in expected_raster.items():
        if raster.get(key) != expected:
            raise AdmissionInputError(
                "mixed ABI raster.{} must be {!r}".format(key, expected)
            )
    for key, expected in expected_head.items():
        if head.get(key) != expected:
            raise AdmissionInputError(
                "mixed ABI head.{} must be {!r}".format(key, expected)
            )


def _validate_head_abi(abi):
    abi = _mapping(abi, "head ABI")
    if abi.get("abi_version") != 1:
        raise AdmissionInputError("head ABI version must be 1")
    if abi.get("cuda_arch") != EXPECTED_CUDA_ARCH:
        raise AdmissionInputError("head ABI cuda_arch must be sm_86")
    if abi.get("capability_query") != "tacker_capabilities":
        raise AdmissionInputError(
            "head ABI capability_query must be tacker_capabilities"
        )
    launch = _mapping(abi.get("logical_launch"), "head ABI logical_launch")
    if launch.get("block_threads") != 128:
        raise AdmissionInputError("head ABI block_threads must be 128")
    symbols = _mapping(
        abi.get("global_kernel_symbols"), "head ABI global_kernel_symbols"
    )
    solo = _mapping(symbols.get("solo"), "head ABI solo symbol")
    gptb = _mapping(symbols.get("gptb"), "head ABI GPTB symbol")
    if solo.get("symbol") != EXPECTED_HEAD_SOLO_SYMBOL:
        raise AdmissionInputError("head ABI solo symbol is not the v1 symbol")
    if gptb.get("symbol") != EXPECTED_HEAD_GPTB_SYMBOL:
        raise AdmissionInputError("head ABI GPTB symbol is not the v1 symbol")
    tensors = _mapping(abi.get("tensors"), "head ABI tensors")
    expected_dtypes = {
        "input": "float16",
        "weight": "float16",
        "bias": "float32",
        "output": "float32",
    }
    for name, dtype in expected_dtypes.items():
        tensor = _mapping(tensors.get(name), "head ABI tensor {}".format(name))
        if tensor.get("dtype") != dtype:
            raise AdmissionInputError(
                "head ABI {} dtype must be {}".format(name, dtype)
            )


def _validate_template(template):
    template = _mapping(template, "profile template")
    if template.get("schema_version") != SCHEMA_VERSION:
        raise AdmissionInputError("profile template schema_version must be 1")
    manifest = _mapping(template.get("manifest"), "profile template manifest")
    digest = template.get("manifest_sha256")
    if digest != manifest_sha256(manifest):
        raise AdmissionInputError("profile template manifest SHA-256 mismatch")
    expected = {
        "rasterizer_commit": EXPECTED_RASTERIZER_COMMIT,
        "pair_key": EXPECTED_PAIR_KEY,
        "cuda_arch": EXPECTED_CUDA_ARCH,
        "compute_capability": EXPECTED_COMPUTE_CAPABILITY,
        "gpu_name": EXPECTED_GPU_NAME,
        "workload": EXPECTED_SCENE,
        "iteration": EXPECTED_ITERATION,
        "gaussian_count": EXPECTED_GAUSSIANS,
        "resolution": EXPECTED_RESOLUTION,
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
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise AdmissionInputError(
                "profile template manifest.{} must be {!r}".format(key, value)
            )
    persistent_blocks = manifest.get("persistent_blocks")
    if (
        type(persistent_blocks) is not int
        or persistent_blocks < 0
    ):
        raise AdmissionInputError(
            "profile template persistent_blocks must be an int >= 0"
        )
    thresholds = _mapping(template.get("thresholds"), "profile thresholds")
    for key, locked in LEGACY_PROFILE_THRESHOLDS.items():
        value = thresholds.get(key)
        if isinstance(locked, bool):
            if value is not locked:
                raise AdmissionInputError(
                    "profile threshold {} must be true".format(key)
                )
        elif not _is_finite_number(value) or float(value) > locked:
            raise AdmissionInputError(
                "profile threshold {} may not exceed {}".format(key, locked)
            )
    return manifest, dict(thresholds)


def _quality_measurements(document):
    document = _require_document_contract(
        document,
        "quality input",
        "4dgaussians_tacker_quality_validation",
    )
    measurements = _nested_measurements(document)
    if all(
        key in measurements
        for key in ("psnr_drop_db", "ssim_drop", "lpips_increase")
    ):
        source = measurements
    else:
        deltas = document.get("deltas")
        if not isinstance(deltas, dict) or not isinstance(deltas.get("tacker"), dict):
            raise AdmissionInputError(
                "quality input requires measurements or deltas.tacker"
            )
        source = deltas["tacker"]
    values = {
        "psnr_drop_db": _finite(
            source, ("psnr_drop_db",), "quality psnr_drop_db"
        ),
        "ssim_drop": _finite(source, ("ssim_drop",), "quality ssim_drop"),
        "lpips_increase": _finite(
            source, ("lpips_increase",), "quality lpips_increase"
        ),
    }
    modes = _mapping(document.get("modes"), "quality modes")
    for expected_mode in ("serial", "two_stream", "tacker"):
        mode = _mapping(
            modes.get(expected_mode),
            "quality mode {}".format(expected_mode),
        )
        if mode.get("actual_mode") != expected_mode:
            raise AdmissionInputError(
                "quality input did not execute physical {} mode".format(
                    expected_mode
                )
            )
    tacker_mode = modes["tacker"]
    qualification = document.get("qualification")
    qualification_mode = False
    if qualification is not None:
        qualification = _mapping(qualification, "quality qualification")
        if type(qualification.get("enabled")) is not bool:
            raise AdmissionInputError(
                "quality qualification.enabled must be boolean"
            )
        qualification_mode = qualification["enabled"]
        if qualification_mode and qualification.get("admission_claimed") is not False:
            raise AdmissionInputError(
                "qualification quality data may not claim prior admission"
            )
        if qualification_mode and (
            tacker_mode is None
            or tacker_mode.get("qualification_executed") is not True
        ):
            raise AdmissionInputError(
                "qualification quality data must record physical qualification execution"
            )
    workload = _mapping(document.get("workload"), "quality workload")
    split = workload.get("split")
    frames = workload.get("frames")
    view_indices = workload.get("view_indices")
    model_path = workload.get("model_path")
    source_path = workload.get("source_path")
    if split not in ("train", "test", "video"):
        raise AdmissionInputError("quality workload split is required")
    if type(frames) is not int or frames <= 0:
        raise AdmissionInputError("quality workload frames must be a positive int")
    if (
        not isinstance(view_indices, list)
        or len(view_indices) != frames
        or any(type(index) is not int or index < 0 for index in view_indices)
    ):
        raise AdmissionInputError(
            "quality workload view_indices must contain one index per frame"
        )
    if not isinstance(model_path, str) or not model_path:
        raise AdmissionInputError("quality workload model_path is required")
    if not isinstance(source_path, str) or not source_path:
        raise AdmissionInputError("quality workload source_path is required")
    benchmark_contract = {
        "split": split,
        "profile_frames": frames,
        "view_indices": list(view_indices),
        "model_path": model_path,
        "source_path": source_path,
    }
    return values, qualification_mode, benchmark_contract


def _raster_measurements(document):
    document = _require_document_contract(
        document,
        "raster input",
        "4dgaussians_tacker_raster_profile",
        require_numerics=True,
    )
    source = _nested_measurements(document)
    solo = _finite(
        source,
        ("solo_raster_p50_ms", "baseline_raster_p50_ms"),
        "raster solo_raster_p50_ms",
        positive=True,
    )
    mixed_raster = _finite(
        source,
        ("mixed_raster_p50_ms", "raster_in_mixed_p50_ms"),
        "raster mixed_raster_p50_ms",
        positive=True,
    )
    # Normalise harmless binary representation noise so a mathematically exact
    # 5% boundary is written as 5.0 and is accepted identically by the runtime.
    slowdown = round((mixed_raster / solo - 1.0) * 100.0, 12)
    return {
        "solo_raster_p50_ms": solo,
        "mixed_raster_p50_ms": mixed_raster,
        "raster_slowdown_pct": slowdown,
        "persistent_blocks": _measurement_persistent_blocks(
            document, "raster input"
        ),
        "workload_contract": _leaf_workload_contract(
            document, "raster input"
        ),
    }


def _leaf_measurements(document):
    document = _require_document_contract(
        document,
        "leaf input",
        "4dgaussians_tacker_leaf_profile",
        require_numerics=True,
    )
    source = _nested_measurements(document)
    return {
        "mixed_p50_ms": _finite(
            source, ("mixed_p50_ms",), "leaf mixed_p50_ms", positive=True
        ),
        "solo_raster_p50_ms": _finite(
            source,
            ("solo_raster_p50_ms",),
            "leaf solo_raster_p50_ms",
            positive=True,
        ),
        "solo_head_p50_ms": _finite(
            source,
            ("solo_head_p50_ms",),
            "leaf solo_head_p50_ms",
            positive=True,
        ),
        "persistent_blocks": _measurement_persistent_blocks(
            document, "leaf input"
        ),
        "workload_contract": _leaf_workload_contract(document, "leaf input"),
    }


def _end_to_end(document, expected_mode, label):
    document = _require_document_contract(
        document,
        label,
        "4dgaussians_tacker_render_profile",
    )
    actual_mode = document.get("actual_execution_mode")
    if actual_mode is None:
        actual_mode = document.get("actual_mode", document.get("execution_mode"))
    if actual_mode != expected_mode:
        raise AdmissionInputError(
            "{} actual execution mode must be {}".format(label, expected_mode)
        )
    source = _nested_measurements(document)
    keys = (
        "{}_end_to_end_p50_ms".format(expected_mode),
        "end_to_end_p50_ms",
        "p50_frame_ms",
        "mean_frame_ms",
    )
    value = _finite(source, keys, "{} frame time".format(label), positive=True)
    statistic = "mean" if "mean_frame_ms" in source and not any(
        key in source for key in keys[:-1]
    ) else "p50"
    split = document.get("split")
    warmup_frames = document.get("warmup_frames")
    profile_frames = document.get("profile_frames")
    view_indices = document.get("view_indices")
    model_path = document.get("model_path")
    source_path = document.get("source_path")
    if split not in ("train", "test", "video"):
        raise AdmissionInputError("{} split is required".format(label))
    if type(warmup_frames) is not int or warmup_frames < 0:
        raise AdmissionInputError(
            "{} warmup_frames must be an int >= 0".format(label)
        )
    if type(profile_frames) is not int or profile_frames <= 0:
        raise AdmissionInputError(
            "{} profile_frames must be a positive int".format(label)
        )
    if (
        not isinstance(view_indices, list)
        or len(view_indices) != profile_frames
        or any(type(index) is not int or index < 0 for index in view_indices)
    ):
        raise AdmissionInputError(
            "{} view_indices must contain one index per frame".format(label)
        )
    if not isinstance(model_path, str) or not model_path:
        raise AdmissionInputError("{} model_path is required".format(label))
    if not isinstance(source_path, str) or not source_path:
        raise AdmissionInputError("{} source_path is required".format(label))
    if document.get("timing_method") != "perf_counter_with_cuda_synchronize":
        raise AdmissionInputError("{} timing_method changed".format(label))
    if (
        document.get("frame_timing_method")
        != "cuda_event_consumer_completion_intervals"
    ):
        raise AdmissionInputError("{} frame_timing_method changed".format(label))
    if document.get("io_in_timed_region") is not False:
        raise AdmissionInputError("{} timed region must exclude I/O".format(label))
    benchmark_contract = {
        "split": split,
        "warmup_frames": warmup_frames,
        "profile_frames": profile_frames,
        "view_indices": list(view_indices),
        "model_path": model_path,
        "source_path": source_path,
    }
    return value, statistic, benchmark_contract


def _source_digest(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _evaluate_legacy_admission(inputs):
    """Reject legacy single-run admission without reapplying old QoS gates.

    Schema-v1 profiles and measurement documents remain readable by their
    dedicated helpers, but Phase 1 requires a whole-run FPS benchmark before a
    new deployment decision can be emitted.
    """

    digests = {}
    for name, value in inputs.items():
        try:
            digests[name] = _source_digest(value)
        except (TypeError, ValueError):
            digests[name] = None
    report = _v2_base_report(digests)
    report["errors"].append(
        "legacy single-run admission is read-only; provide fps_benchmark input "
        "for schema-v2 whole-run FPS selection"
    )
    return report, None


def _v2_base_report(input_digests=None):
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "kind": "4dgaussians_tacker_admission_report",
        "generated_at_utc": _utc_now(),
        "passed": False,
        "workload_key": WORKLOAD_KEY,
        "selection_objective": SELECTION_OBJECTIVE,
        "selected_variant_id": None,
        "manifest": None,
        "manifest_sha256": None,
        "correctness_thresholds": dict(CORRECTNESS_THRESHOLDS),
        "candidates": [],
        "selection": None,
        "deployment": {"enabled": False, "valid": False},
        "provenance": {"input_sha256": input_digests or {}},
        "errors": [],
        "enabled_profile_written": False,
    }


def _require_equal(mapping, key, expected, label):
    if mapping.get(key) != expected:
        raise AdmissionInputError(
            "{}.{} must be {!r}".format(label, key, expected)
        )


def _finite_list(value, label, positive=False):
    if not isinstance(value, list) or not value:
        raise AdmissionInputError("{} must be a non-empty array".format(label))
    result = []
    for index, item in enumerate(value):
        if not _is_finite_number(item):
            raise AdmissionInputError(
                "{}[{}] must be a finite number".format(label, index)
            )
        item = float(item)
        if positive and item <= 0.0:
            raise AdmissionInputError(
                "{}[{}] must be > 0".format(label, index)
            )
        result.append(item)
    return result


def _percentile(sorted_values, probability):
    if not sorted_values:
        raise AdmissionInputError("cannot compute percentile of an empty array")
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
    """Version-independent SHA-256 sampler shared with the FPS driver."""

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
    if (
        not candidate_values
        or len(candidate_values) != len(reference_values)
        or type(resamples) is not int
        or resamples <= 0
        or type(seed) is not int
    ):
        raise AdmissionInputError("paired bootstrap metadata is invalid")
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


def _benchmark_workload_contract(document):
    contract = _mapping(document.get("contract"), "FPS benchmark contract")
    expected = {
        "iteration": EXPECTED_ITERATION,
        "image_width": EXPECTED_RESOLUTION[0],
        "image_height": EXPECTED_RESOLUTION[1],
        "gaussian_count": EXPECTED_GAUSSIANS,
        "timing_method": "perf_counter_with_cuda_synchronize",
        "frame_timing_method": "cuda_event_consumer_completion_intervals",
        "io_in_timed_region": False,
        "throughput_definition": "profile_frames / elapsed_seconds",
    }
    for key, value in expected.items():
        _require_equal(contract, key, value, "FPS benchmark contract")
    if _normalise_scene(contract.get("workload_name")) != EXPECTED_SCENE:
        raise AdmissionInputError(
            "FPS benchmark contract.workload_name must be {}".format(
                EXPECTED_SCENE
            )
        )
    split = contract.get("split")
    if split != "test":
        raise AdmissionInputError("FPS benchmark contract.split must be test")
    warmup_frames = contract.get("warmup_frames")
    profile_frames = contract.get("profile_frames")
    if type(warmup_frames) is not int or warmup_frames != 10:
        raise AdmissionInputError(
            "FPS benchmark contract.warmup_frames must be 10"
        )
    if type(profile_frames) is not int or profile_frames != 50:
        raise AdmissionInputError(
            "FPS benchmark requires exactly 50 profile frames per trial"
        )
    view_indices = contract.get("view_indices")
    if (
        not isinstance(view_indices, list)
        or len(view_indices) != profile_frames
        or any(type(index) is not int or index < 0 for index in view_indices)
    ):
        raise AdmissionInputError(
            "FPS benchmark contract.view_indices must contain one index per frame"
        )
    if view_indices != list(range(50)):
        raise AdmissionInputError(
            "FPS benchmark contract.view_indices must be the fixed test views 0-49"
        )
    model_path = contract.get("model_path")
    source_path = contract.get("source_path")
    if not isinstance(model_path, str) or not model_path:
        raise AdmissionInputError("FPS benchmark contract.model_path is required")
    if not isinstance(source_path, str) or not source_path:
        raise AdmissionInputError("FPS benchmark contract.source_path is required")
    return {
        "split": split,
        "warmup_frames": warmup_frames,
        "profile_frames": profile_frames,
        "view_indices": list(view_indices),
        "model_path": model_path,
        "source_path": source_path,
    }


def _validate_paired_comparison(comparison, summaries, candidate_names):
    comparison = _mapping(comparison, "paired comparison")
    candidate_name = comparison.get("candidate")
    reference_name = comparison.get("reference")
    if candidate_name not in candidate_names or reference_name not in candidate_names:
        raise AdmissionInputError(
            "paired comparison candidate/reference must name benchmark candidates"
        )
    if candidate_name == reference_name:
        raise AdmissionInputError("paired comparison may not compare a candidate to itself")
    candidate_summary = summaries[candidate_name]
    reference_summary = summaries[reference_name]
    expected_ratio = (
        candidate_summary["median_throughput_fps"]
        / reference_summary["median_throughput_fps"]
    )
    measured_ratio = _finite(
        comparison,
        ("median_fps_ratio",),
        "paired comparison median_fps_ratio",
        positive=True,
    )
    if not math.isclose(
        measured_ratio, expected_ratio, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AdmissionInputError(
            "paired comparison median_fps_ratio disagrees with candidate summaries"
        )

    rounds = comparison.get("round_indices")
    if (
        not isinstance(rounds, list)
        or not rounds
        or any(type(value) is not int or value < 0 for value in rounds)
        or len(set(rounds)) != len(rounds)
    ):
        raise AdmissionInputError(
            "paired comparison round_indices must be unique non-negative integers"
        )
    candidate_by_round = dict(
        zip(
            candidate_summary["round_indices"],
            candidate_summary["throughput_fps_trials"],
        )
    )
    reference_by_round = dict(
        zip(
            reference_summary["round_indices"],
            reference_summary["throughput_fps_trials"],
        )
    )
    if any(
        round_index not in candidate_by_round
        or round_index not in reference_by_round
        for round_index in rounds
    ):
        raise AdmissionInputError(
            "paired comparison refers to a round missing from its summaries"
        )
    ratios = _finite_list(
        comparison.get("paired_fps_ratios"),
        "paired comparison paired_fps_ratios",
        positive=True,
    )
    if len(ratios) != len(rounds):
        raise AdmissionInputError(
            "paired comparison must contain one FPS ratio per round"
        )
    expected_ratios = [
        candidate_by_round[index] / reference_by_round[index] for index in rounds
    ]
    if any(
        not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        for actual, expected in zip(ratios, expected_ratios)
    ):
        raise AdmissionInputError(
            "paired comparison FPS ratios disagree with whole-run trials"
        )
    paired_median = _finite(
        comparison,
        ("median_paired_fps_ratio",),
        "paired comparison median_paired_fps_ratio",
        positive=True,
    )
    if not math.isclose(
        paired_median,
        statistics.median(ratios),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AdmissionInputError(
            "paired comparison median_paired_fps_ratio disagrees with paired trials"
        )
    interval = _mapping(
        comparison.get("paired_bootstrap_95_ci"),
        "paired comparison paired_bootstrap_95_ci",
    )
    lower = _finite(
        interval, ("lower",), "paired bootstrap lower", positive=True
    )
    upper = _finite(
        interval, ("upper",), "paired bootstrap upper", positive=True
    )
    if lower > upper:
        raise AdmissionInputError("paired bootstrap lower may not exceed upper")
    if "confidence" in interval and (
        not _is_finite_number(interval["confidence"])
        or not math.isclose(float(interval["confidence"]), 0.95)
    ):
        raise AdmissionInputError("paired bootstrap confidence must be 0.95")
    for key, expected in (
        ("statistic", "median(candidate_fps)/median(reference_fps)"),
        ("resampling_unit", "paired_round"),
        ("percentile_method", "linear_type_7"),
    ):
        if interval.get(key) != expected:
            raise AdmissionInputError(
                "paired bootstrap {} must be {!r}".format(key, expected)
            )
    resamples = interval.get("resamples")
    seed = interval.get("seed")
    if (
        resamples != FORMAL_BOOTSTRAP_RESAMPLES
        or seed != FORMAL_BOOTSTRAP_SEED
    ):
        raise AdmissionInputError(
            "paired bootstrap must use the formal 10000 resamples and seed 0"
        )
    candidate_values = [candidate_by_round[index] for index in rounds]
    reference_values = [reference_by_round[index] for index in rounds]
    expected_lower, expected_upper = _paired_bootstrap_interval(
        candidate_values,
        reference_values,
        resamples,
        seed,
        "{}-vs-{}".format(candidate_name, reference_name),
    )
    if not math.isclose(lower, expected_lower, rel_tol=1e-12, abs_tol=1e-12):
        raise AdmissionInputError(
            "paired bootstrap lower disagrees with whole-run trials"
        )
    if not math.isclose(upper, expected_upper, rel_tol=1e-12, abs_tol=1e-12):
        raise AdmissionInputError(
            "paired bootstrap upper disagrees with whole-run trials"
        )
    normalized = {
        "candidate": candidate_name,
        "reference": reference_name,
        "median_fps_ratio": measured_ratio,
        "median_paired_fps_ratio": paired_median,
        "round_indices": list(rounds),
        "paired_fps_ratios": ratios,
        "paired_bootstrap_95_ci": dict(interval),
    }
    return normalized


def _stable_benchmark_order(candidate_names, seed):
    """Reproduce the FPS driver's cross-process stable base order."""

    def key(item):
        index, name = item
        material = "{}\0{}\0{}".format(seed, index, name).encode("utf-8")
        return hashlib.sha256(material).hexdigest(), index

    return [
        name
        for _, name in sorted(enumerate(candidate_names), key=key)
    ]


def _expected_benchmark_executions(candidate_names, trials, strategy, seed):
    """Return the exact schedule contract emitted by the FPS driver."""

    names = list(candidate_names)
    base = _stable_benchmark_order(names, seed)
    executions = []
    run_index = 0
    for round_index in range(trials):
        if strategy == "round_robin":
            rotation = round_index % len(base)
            order = base[rotation:] + base[:rotation]
        else:
            pair_index = round_index // 2
            rotation = pair_index % len(base)
            forward = base[rotation:] + base[:rotation]
            order = forward if round_index % 2 == 0 else list(reversed(forward))
        for position, candidate_name in enumerate(order):
            executions.append(
                {
                    "run_index": run_index,
                    "round_index": round_index,
                    "position_in_round": position,
                    "candidate_name": candidate_name,
                }
            )
            run_index += 1
    return executions, base


def _validate_fps_benchmark(document):
    document = _mapping(document, "FPS benchmark")
    _require_equal(
        document,
        "schema_version",
        FPS_BENCHMARK_SCHEMA_VERSION,
        "FPS benchmark",
    )
    _require_equal(document, "kind", FPS_BENCHMARK_KIND, "FPS benchmark")
    _require_equal(document, "passed", True, "FPS benchmark")
    _require_equal(
        document, "selection_objective", SELECTION_OBJECTIVE, "FPS benchmark"
    )
    if document.get("errors") not in (None, []):
        raise AdmissionInputError("FPS benchmark contains errors")
    exit_condition = _mapping(
        document.get("phase0_exit_condition"),
        "FPS benchmark phase0_exit_condition",
    )
    _require_equal(
        exit_condition,
        "required_trials_per_candidate",
        10,
        "FPS benchmark phase0_exit_condition",
    )
    _require_equal(
        exit_condition,
        "required_frames_per_trial",
        50,
        "FPS benchmark phase0_exit_condition",
    )
    if exit_condition.get("met") is not True:
        raise AdmissionInputError(
            "FPS benchmark does not meet the required trial/frame exit condition"
        )
    contract = _benchmark_workload_contract(document)
    if contract["profile_frames"] != 50:
        raise AdmissionInputError(
            "FPS benchmark requires exactly 50 profile frames per trial"
        )

    bootstrap = _mapping(
        document.get("bootstrap"), "FPS benchmark bootstrap"
    )
    if (
        not _is_finite_number(bootstrap.get("confidence"))
        or not math.isclose(
            float(bootstrap["confidence"]), 0.95, rel_tol=0.0, abs_tol=1e-15
        )
        or bootstrap.get("resamples") != FORMAL_BOOTSTRAP_RESAMPLES
        or bootstrap.get("seed") != FORMAL_BOOTSTRAP_SEED
        or bootstrap.get("resampling_unit") != "paired_round"
    ):
        raise AdmissionInputError(
            "FPS benchmark bootstrap must use confidence 0.95, 10000 "
            "resamples, seed 0, and paired_round resampling"
        )

    rows = document.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise AdmissionInputError("FPS benchmark.candidates must be a non-empty array")
    candidates = []
    by_name = {}
    for index, row in enumerate(rows):
        row = _mapping(row, "FPS benchmark candidate {}".format(index))
        name = row.get("name")
        mode = row.get("execution_mode")
        if not isinstance(name, str) or not name:
            raise AdmissionInputError("FPS benchmark candidate.name is required")
        if name in by_name:
            raise AdmissionInputError(
                "FPS benchmark has duplicate candidate {!r}".format(name)
            )
        if mode not in ("serial", "two_stream", "tacker"):
            raise AdmissionInputError(
                "FPS benchmark candidate {} has an invalid execution_mode".format(
                    name
                )
            )
        expected_reserved_mode = {
            "serial": "serial",
            "two_stream": "two_stream",
            INCUMBENT_BENCHMARK_NAME: "tacker",
        }.get(name)
        if expected_reserved_mode is not None and mode != expected_reserved_mode:
            raise AdmissionInputError(
                "FPS benchmark candidate {} must execute {}".format(
                    name, expected_reserved_mode
                )
            )
        if expected_reserved_mode is None and mode != "tacker":
            raise AdmissionInputError(
                "FPS benchmark additional candidate {} must execute tacker".format(
                    name
                )
            )
        normalized = dict(row)
        normalized["name"] = name
        normalized["execution_mode"] = mode
        candidates.append(normalized)
        by_name[name] = normalized
    for required_name in ("serial", "two_stream", INCUMBENT_BENCHMARK_NAME):
        if required_name not in by_name:
            raise AdmissionInputError(
                "FPS benchmark must include {}".format(required_name)
            )

    raw_qualifications = document.get("correctness_qualifications")
    phase0_compatibility = raw_qualifications is None
    if phase0_compatibility:
        if _source_digest(document) != FROZEN_PHASE0_FPS_REPORT_SHA256:
            raise AdmissionInputError(
                "only the frozen Phase-0 FPS report may omit Phase-1 "
                "correctness and selection evidence"
            )
        qualifications = {
            name: {"valid": True, "source": "phase0_compatibility_default"}
            for name in by_name
        }
    else:
        for candidate in candidates:
            name = candidate["name"]
            mode = candidate["execution_mode"]
            required_profile_fields = {
                "profile_path",
                "profile_file_sha256",
                "qualification_mode",
            }
            missing_profile_fields = sorted(
                required_profile_fields.difference(candidate)
            )
            if missing_profile_fields:
                raise AdmissionInputError(
                    "prefiltered FPS benchmark candidate {} is missing "
                    "required profile fields: {!r}".format(
                        name, missing_profile_fields
                    )
                )
            profile_path = candidate.get("profile_path")
            profile_file_sha256 = candidate.get("profile_file_sha256")
            qualification_mode = candidate.get("qualification_mode")
            if type(qualification_mode) is not bool:
                raise AdmissionInputError(
                    "prefiltered FPS benchmark candidate {} requires boolean "
                    "qualification_mode".format(name)
                )
            if mode == "tacker":
                if not isinstance(profile_path, str) or not profile_path:
                    raise AdmissionInputError(
                        "prefiltered FPS benchmark Tacker candidate {} requires "
                        "profile_path".format(name)
                    )
                if not _is_lower_hex(profile_file_sha256, 64):
                    raise AdmissionInputError(
                        "prefiltered FPS benchmark Tacker candidate {} requires "
                        "profile_file_sha256".format(name)
                    )
                if (
                    name == INCUMBENT_BENCHMARK_NAME
                    and qualification_mode
                ):
                    raise AdmissionInputError(
                        "prefiltered FPS benchmark current_tacker may not use "
                        "qualification mode"
                    )
            elif (
                profile_path is not None
                or profile_file_sha256 is not None
                or qualification_mode
            ):
                raise AdmissionInputError(
                    "prefiltered FPS benchmark baseline {} may not carry a "
                    "Tacker profile".format(name)
                )
        raw_qualifications = _mapping(
            raw_qualifications, "FPS benchmark correctness_qualifications"
        )
        if set(raw_qualifications) != set(by_name):
            raise AdmissionInputError(
                "FPS benchmark correctness_qualifications must cover candidates exactly"
            )
        qualifications = {}
        for name in by_name:
            entry = _mapping(
                raw_qualifications[name],
                "FPS benchmark correctness qualification {}".format(name),
            )
            if type(entry.get("valid")) is not bool:
                raise AdmissionInputError(
                    "FPS benchmark correctness qualification {}.valid must be boolean".format(
                        name
                    )
                )
            qualifications[name] = dict(entry)
        eligible_names = [
            name for name in by_name if qualifications[name]["valid"]
        ]
        if document.get("eligible_candidates") != eligible_names:
            raise AdmissionInputError(
                "FPS benchmark eligible_candidates disagrees with correctness qualifications"
            )
        excluded = document.get("excluded_candidates")
        if not isinstance(excluded, list):
            raise AdmissionInputError(
                "FPS benchmark excluded_candidates must be an array"
            )
        excluded_names = []
        for item in excluded:
            item = _mapping(item, "FPS benchmark excluded candidate")
            name = item.get("name")
            if name not in by_name or item.get("reason") != "correctness_invalid":
                raise AdmissionInputError(
                    "FPS benchmark excluded_candidates entry is invalid"
                )
            excluded_names.append(name)
        expected_excluded = [
            name for name in by_name if not qualifications[name]["valid"]
        ]
        if excluded_names != expected_excluded:
            raise AdmissionInputError(
                "FPS benchmark excluded_candidates disagrees with correctness qualifications"
            )
    eligible_names = [name for name in by_name if qualifications[name]["valid"]]
    if not eligible_names:
        raise AdmissionInputError(
            "FPS benchmark has no correctness-valid measured candidate"
        )
    # A deployable profile always carries both safety baselines.  The runtime
    # validator intentionally requires them to remain correctness-valid and
    # measured even when an experimental Tacker candidate wins.  Rejecting the
    # benchmark here keeps admission and runtime acceptance on the same side of
    # that contract.
    for baseline_name in ("serial", "two_stream"):
        if not qualifications[baseline_name]["valid"]:
            raise AdmissionInputError(
                "FPS benchmark requires correctness-valid measured {} baseline".format(
                    baseline_name
                )
            )
    raw_selection_metadata = document.get("candidate_selection_metadata", {})
    raw_selection_metadata = _mapping(
        raw_selection_metadata, "FPS benchmark candidate_selection_metadata"
    )
    if set(raw_selection_metadata) - set(by_name):
        raise AdmissionInputError(
            "FPS benchmark candidate_selection_metadata contains unknown candidates"
        )
    if not phase0_compatibility and set(raw_selection_metadata) != set(by_name):
        raise AdmissionInputError(
            "prefiltered FPS benchmark candidate_selection_metadata must "
            "cover candidates exactly"
        )
    selection_metadata = {}
    for name, raw_metadata in raw_selection_metadata.items():
        raw_metadata = _mapping(
            raw_metadata,
            "FPS benchmark candidate selection metadata {}".format(name),
        )
        allowed_metadata_keys = {
            "abi_complexity",
            "peak_memory_bytes",
            "registers_per_thread",
            "shared_memory_bytes",
        }
        unknown_metadata_keys = sorted(set(raw_metadata) - allowed_metadata_keys)
        if unknown_metadata_keys:
            raise AdmissionInputError(
                "FPS benchmark candidate selection metadata {} has unknown "
                "fields: {!r}".format(name, unknown_metadata_keys)
            )
        normalized_metadata = {}
        for key in (
            "abi_complexity",
            "peak_memory_bytes",
            "registers_per_thread",
            "shared_memory_bytes",
        ):
            value = raw_metadata.get(key)
            if value is None:
                if key in raw_metadata:
                    raise AdmissionInputError(
                        "FPS benchmark candidate selection metadata "
                        "{}.{} must be finite and >= 0".format(name, key)
                    )
                continue
            if not _is_finite_number(value) or float(value) < 0.0:
                raise AdmissionInputError(
                    "FPS benchmark candidate selection metadata "
                    "{}.{} must be finite and >= 0".format(
                        name, key
                    )
                )
            normalized_metadata[key] = float(value)
        selection_metadata[name] = normalized_metadata
        if not phase0_compatibility and "abi_complexity" not in normalized_metadata:
            raise AdmissionInputError(
                "prefiltered FPS benchmark effective selection metadata for "
                "{} must include abi_complexity".format(name)
            )

    raw_summaries = _mapping(document.get("summaries"), "FPS benchmark summaries")
    if set(raw_summaries) != set(eligible_names):
        raise AdmissionInputError(
            "FPS benchmark summaries must match correctness-valid candidates exactly"
        )
    summaries = {}
    for name in eligible_names:
        summary = _mapping(
            raw_summaries[name], "FPS benchmark summary {}".format(name)
        )
        trials = _finite_list(
            summary.get("throughput_fps_trials"),
            "FPS benchmark summary {} throughput_fps_trials".format(name),
            positive=True,
        )
        trial_count = summary.get("trial_count")
        if type(trial_count) is not int or trial_count != len(trials):
            raise AdmissionInputError(
                "FPS benchmark summary {} trial_count disagrees with trials".format(
                    name
                )
            )
        if trial_count < 10:
            raise AdmissionInputError(
                "FPS benchmark summary {} requires at least 10 whole-run trials".format(
                    name
                )
            )
        rounds = summary.get("round_indices")
        if (
            not isinstance(rounds, list)
            or len(rounds) != trial_count
            or any(type(value) is not int or value < 0 for value in rounds)
            or len(set(rounds)) != len(rounds)
        ):
            raise AdmissionInputError(
                "FPS benchmark summary {} requires one unique round per trial".format(
                    name
                )
            )
        median_fps = _finite(
            summary,
            ("median_throughput_fps",),
            "FPS benchmark summary {} median_throughput_fps".format(name),
            positive=True,
        )
        if not math.isclose(
            median_fps,
            statistics.median(trials),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AdmissionInputError(
                "FPS benchmark summary {} median disagrees with trials".format(name)
            )
        normalized = {
            "trial_count": trial_count,
            "round_indices": list(rounds),
            "throughput_fps_trials": trials,
            "median_throughput_fps": median_fps,
        }
        for key in (
            "median_total_render_ms",
            "min_throughput_fps",
            "max_throughput_fps",
        ):
            if key in summary:
                normalized[key] = _finite(
                    summary,
                    (key,),
                    "FPS benchmark summary {} {}".format(name, key),
                    positive=True,
                )
        summaries[name] = normalized

    expected_rounds = summaries[eligible_names[0]]["round_indices"]
    for name in eligible_names[1:]:
        if summaries[name]["round_indices"] != expected_rounds:
            raise AdmissionInputError(
                "FPS benchmark candidates must share the same paired rounds"
            )

    schedule = _mapping(document.get("schedule"), "FPS benchmark schedule")
    strategy = schedule.get("strategy")
    schedule_seed = schedule.get("seed")
    trials_per_candidate = schedule.get("trials_per_candidate")
    if strategy not in ("abba", "round_robin"):
        raise AdmissionInputError(
            "FPS benchmark schedule.strategy must be abba or round_robin"
        )
    if schedule_seed != FORMAL_BOOTSTRAP_SEED:
        raise AdmissionInputError(
            "FPS benchmark schedule.seed must match formal bootstrap seed 0"
        )
    if (
        type(trials_per_candidate) is not int
        or trials_per_candidate < 10
    ):
        raise AdmissionInputError(
            "FPS benchmark schedule.trials_per_candidate must be an int >= 10"
        )
    if any(
        summary["trial_count"] != trials_per_candidate
        or summary["round_indices"] != list(range(trials_per_candidate))
        for summary in summaries.values()
    ):
        raise AdmissionInputError(
            "FPS benchmark summaries must cover every scheduled round exactly once"
        )
    expected_executions, expected_base_order = _expected_benchmark_executions(
        eligible_names, trials_per_candidate, strategy, schedule_seed
    )
    if schedule.get("base_order") != expected_base_order:
        raise AdmissionInputError(
            "FPS benchmark schedule.base_order disagrees with candidate order and seed"
        )
    if schedule.get("executions") != expected_executions:
        raise AdmissionInputError(
            "FPS benchmark schedule.executions violates its {} ordering contract"
            .format(strategy)
        )
    runs = document.get("runs")
    if not isinstance(runs, list) or len(runs) != len(expected_executions):
        raise AdmissionInputError(
            "FPS benchmark runs must correspond one-for-one with schedule executions"
        )
    identity_keys = (
        "run_index",
        "round_index",
        "position_in_round",
        "candidate_name",
    )
    validated_run_metrics = []
    for run, expected_execution in zip(runs, expected_executions):
        run = _mapping(run, "FPS benchmark run")
        if any(run.get(key) != expected_execution[key] for key in identity_keys):
            raise AdmissionInputError(
                "FPS benchmark runs disagree with schedule executions"
            )
        if run.get("passed") is not True:
            raise AdmissionInputError(
                "successful FPS benchmark may not contain a failed run"
            )
        metrics = _mapping(run.get("metrics"), "FPS benchmark run metrics")
        candidate = by_name[run["candidate_name"]]
        if not phase0_compatibility:
            required_run_fields = {
                "requested_execution_mode",
                "profile_path",
                "qualification_mode",
                "returncode",
                "error",
                "metadata_sha256",
            }
            missing_run_fields = sorted(required_run_fields.difference(run))
            if missing_run_fields:
                raise AdmissionInputError(
                    "FPS benchmark run is missing required fields: {!r}".format(
                        missing_run_fields
                    )
                )
            required_metric_fields = {
                "actual_execution_mode",
                "two_stream_fallback_reason",
                "tacker_fallback_reason",
                "qualification_mode_requested",
                "qualification_mode_executed",
                "profile_manifest_sha256",
                "profile_selection_sha256",
                "selected_variant_id",
                "selected_candidate_abi_sha256",
                "persistent_blocks",
                "stable_environment",
                "stable_provenance",
                "provenance",
            }
            missing_metric_fields = sorted(
                required_metric_fields.difference(metrics)
            )
            if missing_metric_fields:
                raise AdmissionInputError(
                    "FPS benchmark run metrics is missing required fields: "
                    "{!r}".format(missing_metric_fields)
                )
            for key, expected in (
                ("requested_execution_mode", candidate["execution_mode"]),
                ("profile_path", candidate.get("profile_path")),
                ("qualification_mode", candidate["qualification_mode"]),
                ("returncode", 0),
                ("error", None),
            ):
                if run.get(key) != expected:
                    raise AdmissionInputError(
                        "FPS benchmark run {} disagrees with candidate {}"
                        .format(key, candidate["name"])
                    )
            if not _is_lower_hex(run.get("metadata_sha256"), 64):
                raise AdmissionInputError(
                    "FPS benchmark run metadata_sha256 must be a lowercase SHA-256"
                )
            run_provenance = _mapping(
                metrics.get("provenance"),
                "FPS benchmark run metrics.provenance",
            )
            profile_hashes = _mapping(
                run_provenance.get("profile_hashes"),
                "FPS benchmark run profile_hashes",
            )
            expected_profile_hash_keys = {
                "active_profile_sha256",
                "tacker_profile_sha256",
                "qualification_profile_sha256",
                "profile_manifest_sha256",
                "profile_selection_sha256",
                "selected_candidate_abi_sha256",
            }
            if set(profile_hashes) != expected_profile_hash_keys:
                raise AdmissionInputError(
                    "FPS benchmark run profile_hashes fields changed"
                )
            expected_profile_sha256 = candidate.get("profile_file_sha256")
            expected_qualification = (
                candidate["qualification_mode"]
                if candidate["execution_mode"] == "tacker"
                else False
            )
            if (
                metrics.get("actual_execution_mode")
                != candidate["execution_mode"]
                or metrics.get("two_stream_fallback_reason") is not None
                or metrics.get("tacker_fallback_reason") is not None
                or metrics.get("qualification_mode_requested")
                is not expected_qualification
                or metrics.get("qualification_mode_executed")
                is not expected_qualification
            ):
                raise AdmissionInputError(
                    "FPS benchmark run physical execution evidence disagrees "
                    "with candidate {}".format(candidate["name"])
                )
            if candidate["execution_mode"] == "tacker":
                active_key = (
                    "qualification_profile_sha256"
                    if candidate["qualification_mode"]
                    else "tacker_profile_sha256"
                )
                inactive_key = (
                    "tacker_profile_sha256"
                    if candidate["qualification_mode"]
                    else "qualification_profile_sha256"
                )
                if (
                    profile_hashes.get("active_profile_sha256")
                    != expected_profile_sha256
                    or profile_hashes.get(active_key) != expected_profile_sha256
                    or profile_hashes.get(inactive_key) is not None
                    or not _is_lower_hex(
                        profile_hashes.get("profile_manifest_sha256"), 64
                    )
                    or metrics.get("profile_manifest_sha256")
                    != profile_hashes.get("profile_manifest_sha256")
                    or metrics.get("profile_selection_sha256")
                    != profile_hashes.get("profile_selection_sha256")
                    or metrics.get("selected_candidate_abi_sha256")
                    != profile_hashes.get("selected_candidate_abi_sha256")
                ):
                    raise AdmissionInputError(
                        "FPS benchmark run profile hashes do not bind candidate "
                        "{} to the measured profile bytes".format(candidate["name"])
                    )
                for optional_hash_key in (
                    "profile_selection_sha256",
                    "selected_candidate_abi_sha256",
                ):
                    optional_hash = profile_hashes.get(optional_hash_key)
                    if optional_hash is not None and not _is_lower_hex(
                        optional_hash, 64
                    ):
                        raise AdmissionInputError(
                            "FPS benchmark run {} must be null or a lowercase "
                            "SHA-256".format(optional_hash_key)
                        )
                if (
                    not isinstance(metrics.get("selected_variant_id"), str)
                    or not metrics["selected_variant_id"]
                    or type(metrics.get("persistent_blocks")) is not int
                    or metrics["persistent_blocks"] < 0
                ):
                    raise AdmissionInputError(
                        "FPS benchmark Tacker run requires selected variant and "
                        "persistent_blocks evidence"
                    )
            elif any(
                profile_hashes.get(key) is not None
                for key in (
                    "active_profile_sha256",
                    "tacker_profile_sha256",
                    "qualification_profile_sha256",
                    "profile_manifest_sha256",
                    "profile_selection_sha256",
                    "selected_candidate_abi_sha256",
                )
            ) or any(
                metrics.get(key) is not None
                for key in (
                    "profile_manifest_sha256",
                    "profile_selection_sha256",
                    "selected_candidate_abi_sha256",
                    "selected_variant_id",
                    "persistent_blocks",
                )
            ):
                raise AdmissionInputError(
                    "FPS benchmark baseline run may not carry Tacker profile hashes"
                )
            validated_run_metrics.append((metrics, run_provenance))
        observed_fps = metrics.get("throughput_fps")
        expected_fps = summaries[run["candidate_name"]][
            "throughput_fps_trials"
        ][run["round_index"]]
        if (
            not _is_finite_number(observed_fps)
            or not math.isclose(
                float(observed_fps), expected_fps, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise AdmissionInputError(
                "FPS benchmark run throughput disagrees with summary round"
            )

    expected_ranking = sorted(
        eligible_names,
        key=lambda name: (-summaries[name]["median_throughput_fps"], name),
    )
    if document.get("ranking") != expected_ranking:
        raise AdmissionInputError(
            "FPS benchmark ranking disagrees with median throughput FPS"
        )
    if document.get("experimental_winner") != expected_ranking[0]:
        raise AdmissionInputError(
            "FPS benchmark experimental_winner disagrees with ranking"
        )
    if (
        "eligible_ranking" in document
        and document.get("eligible_ranking") != expected_ranking
    ):
        raise AdmissionInputError(
            "FPS benchmark eligible_ranking disagrees with median throughput FPS"
        )
    reported_selection = document.get("selection")
    if not phase0_compatibility:
        reported_selection = _mapping(
            reported_selection, "prefiltered FPS benchmark selection"
        )
    elif reported_selection is not None:
        reported_selection = _mapping(
            reported_selection, "FPS benchmark selection"
        )

    raw_comparisons = document.get("paired_comparisons")
    if not isinstance(raw_comparisons, list):
        raise AdmissionInputError(
            "FPS benchmark.paired_comparisons must be an array"
        )
    comparisons = []
    seen_pairs = set()
    for raw_comparison in raw_comparisons:
        comparison = _validate_paired_comparison(
            raw_comparison, summaries, set(eligible_names)
        )
        pair = (comparison["candidate"], comparison["reference"])
        if pair in seen_pairs:
            raise AdmissionInputError(
                "FPS benchmark has duplicate paired comparison {} vs {}".format(
                    *pair
                )
            )
        seen_pairs.add(pair)
        comparisons.append(comparison)

    expected_comparison_pairs = [
        (candidate_name, reference_name)
        for reference_name in ("two_stream", INCUMBENT_BENCHMARK_NAME)
        if reference_name in summaries
        for candidate_name in eligible_names
        if candidate_name != reference_name
    ]
    observed_comparison_pairs = [
        (comparison["candidate"], comparison["reference"])
        for comparison in comparisons
    ]
    if observed_comparison_pairs != expected_comparison_pairs:
        raise AdmissionInputError(
            "FPS benchmark paired_comparisons must contain the producer's "
            "complete ordered direct-comparison set"
        )

    if not phase0_compatibility:
        try:
            expected_selection = _select_fps_candidates(
                summaries,
                candidate_qualifications=qualifications,
                candidate_selection_metadata=selection_metadata,
                incumbent_name=INCUMBENT_BENCHMARK_NAME,
                candidate_names=list(by_name),
                bootstrap_resamples=FORMAL_BOOTSTRAP_RESAMPLES,
                seed=FORMAL_BOOTSTRAP_SEED,
                promotion_min_ratio=PROMOTION_MIN_FPS_RATIO,
                equivalence_fraction=EQUIVALENCE_FRACTION,
            )
        except FPSSelectionError as error:
            raise AdmissionInputError(
                "cannot recompute FPS benchmark selection: {}".format(error)
            )
        if reported_selection != expected_selection:
            raise AdmissionInputError(
                "prefiltered FPS benchmark selection disagrees with recomputed "
                "selector output"
            )
        comparisons_by_pair = {
            (comparison["candidate"], comparison["reference"]): comparison
            for comparison in comparisons
        }
        for evaluation in expected_selection["promotion"][
            "candidate_evaluations"
        ]:
            selection_comparison = evaluation.get("comparison")
            if selection_comparison is None:
                continue
            pair = (
                selection_comparison.get("candidate"),
                selection_comparison.get("reference"),
            )
            if comparisons_by_pair.get(pair) != selection_comparison:
                raise AdmissionInputError(
                    "FPS benchmark selection promotion comparison is not "
                    "identical to its direct paired_comparisons evidence"
                )
        for top_level_key, selection_key in (
            ("eligible_ranking", "eligible_ranking"),
            ("experimental_winner", "experimental_winner"),
            ("deployment_winner", "deployment_winner"),
            ("promotion", "promotion"),
        ):
            if document.get(top_level_key) != expected_selection[selection_key]:
                raise AdmissionInputError(
                    "FPS benchmark {} disagrees with selection.{}".format(
                        top_level_key, selection_key
                    )
                )
    else:
        expected_selection = None

    stable_environment = _mapping(
        document.get("stable_environment"),
        "FPS benchmark stable_environment",
    )
    for key in ("gpu_name", "cuda_runtime", "pytorch_version"):
        value = stable_environment.get(key)
        if not isinstance(value, str) or not value:
            raise AdmissionInputError(
                "FPS benchmark stable_environment.{} must be non-empty".format(
                    key
                )
            )
    stable_provenance = _mapping(
        document.get("stable_provenance"),
        "FPS benchmark stable_provenance",
    )
    if not _is_lower_hex(stable_provenance.get("repository_commit"), 40):
        raise AdmissionInputError(
            "FPS benchmark stable_provenance.repository_commit must be a lowercase commit hash"
        )
    if not phase0_compatibility and (
        not isinstance(stable_provenance.get("repository_commit_source"), str)
        or not stable_provenance["repository_commit_source"]
    ):
        raise AdmissionInputError(
            "prefiltered FPS benchmark stable_provenance.repository_commit_source "
            "must be non-empty"
        )
    if stable_provenance.get("environment") != stable_environment:
        raise AdmissionInputError(
            "FPS benchmark stable provenance environment disagrees with stable_environment"
        )
    if not phase0_compatibility and (
        "submodules" not in stable_provenance
        or not isinstance(stable_provenance["submodules"], list)
    ):
        raise AdmissionInputError(
            "FPS benchmark stable_provenance.submodules must be an explicit array"
        )
    source_files = _mapping(
        stable_provenance.get("source_files"),
        "FPS benchmark stable_provenance.source_files",
    )
    if not source_files or any(
        not isinstance(name, str) or not _is_lower_hex(digest, 64)
        for name, digest in source_files.items()
    ):
        raise AdmissionInputError(
            "FPS benchmark stable provenance source files require lowercase SHA-256 hashes"
        )
    if not phase0_compatibility:
        if type(stable_provenance.get("repository_dirty")) is not bool:
            raise AdmissionInputError(
                "prefiltered FPS benchmark stable_provenance.repository_dirty "
                "must be boolean"
            )
        required_source_files = {
            "profile_render.py",
            "configs",
            "gaussian_renderer/__init__.py",
            "gaussian_renderer/tacker_pipeline.py",
            "diff_gaussian_rasterization/__init__.py",
            "diff_gaussian_rasterization._C",
        }
        missing_source_files = sorted(required_source_files - set(source_files))
        if missing_source_files:
            raise AdmissionInputError(
                "prefiltered FPS benchmark stable provenance is missing source "
                "hashes: {!r}".format(missing_source_files)
            )
        for metrics, run_provenance in validated_run_metrics:
            if metrics.get("stable_environment") != stable_environment:
                raise AdmissionInputError(
                    "FPS benchmark run stable_environment disagrees with report"
                )
            if metrics.get("stable_provenance") != stable_provenance:
                raise AdmissionInputError(
                    "FPS benchmark run stable_provenance disagrees with report"
                )
            run_repository = _mapping(
                run_provenance.get("repository"),
                "FPS benchmark run provenance.repository",
            )
            if (
                "submodules" not in run_repository
                or not isinstance(run_repository["submodules"], list)
            ):
                raise AdmissionInputError(
                    "FPS benchmark run provenance.repository.submodules must "
                    "be an explicit array"
                )
            if (
                run_repository.get("commit")
                != stable_provenance["repository_commit"]
                or run_repository.get("commit_source")
                != stable_provenance["repository_commit_source"]
                or run_repository.get("dirty")
                is not stable_provenance["repository_dirty"]
                or run_repository.get("submodules")
                != stable_provenance.get("submodules")
                or run_repository.get("source_files") != source_files
            ):
                raise AdmissionInputError(
                    "FPS benchmark run repository provenance disagrees with report"
                )
    return {
        "contract": contract,
        "candidates": candidates,
        "candidate_by_name": by_name,
        "correctness_qualifications": qualifications,
        "candidate_selection_metadata": selection_metadata,
        "phase0_compatibility": phase0_compatibility,
        "eligible_names": eligible_names,
        "summaries": summaries,
        "ranking": expected_ranking,
        "paired_comparisons": comparisons,
        "stable_environment": stable_environment,
        "stable_provenance": stable_provenance,
        "generated_at_utc": document.get("generated_at_utc"),
        "runs": document.get("runs") if isinstance(document.get("runs"), list) else [],
        "reported_selection": (
            dict(reported_selection) if reported_selection is not None else None
        ),
        "expected_selection": expected_selection,
    }


def _quality_contract_v2(document):
    document = _mapping(document, "quality input")
    if document.get("schema_version") not in (1, 2):
        raise AdmissionInputError("quality input schema_version must be 1 or 2")
    if document.get("kind") != "4dgaussians_tacker_quality_validation":
        raise AdmissionInputError(
            "quality input kind must be 4dgaussians_tacker_quality_validation"
        )
    if type(document.get("passed")) is not bool:
        raise AdmissionInputError("quality input passed must be boolean")
    _workload(document, "quality input")
    workload = _mapping(document.get("workload"), "quality workload")
    split = workload.get("split")
    frames = workload.get("frames")
    view_indices = workload.get("view_indices")
    model_path = workload.get("model_path")
    source_path = workload.get("source_path")
    if split not in ("train", "test", "video"):
        raise AdmissionInputError("quality workload split is required")
    if type(frames) is not int or frames <= 0:
        raise AdmissionInputError("quality workload frames must be a positive int")
    if (
        not isinstance(view_indices, list)
        or len(view_indices) != frames
        or any(type(index) is not int or index < 0 for index in view_indices)
    ):
        raise AdmissionInputError(
            "quality workload view_indices must contain one index per frame"
        )
    if not isinstance(model_path, str) or not model_path:
        raise AdmissionInputError("quality workload model_path is required")
    if not isinstance(source_path, str) or not source_path:
        raise AdmissionInputError("quality workload source_path is required")
    return {
        "split": split,
        "profile_frames": frames,
        "view_indices": list(view_indices),
        "model_path": model_path,
        "source_path": source_path,
    }


def _candidate_evidence(document, candidate_name, variant_id, overrides):
    keys = (candidate_name, variant_id)
    if candidate_name == INCUMBENT_BENCHMARK_NAME:
        keys = keys + ("tacker",)
    sources = []
    if isinstance(overrides, dict):
        sources.append(overrides)
        nested = overrides.get("candidates")
        if isinstance(nested, dict):
            sources.insert(0, nested)
    candidates = document.get("candidates")
    if isinstance(candidates, dict):
        sources.append(candidates)
    elif isinstance(candidates, list):
        candidates_by_name = {}
        for item in candidates:
            if isinstance(item, dict):
                key = item.get("candidate_name", item.get("variant_id"))
                if isinstance(key, str):
                    candidates_by_name[key] = item
        sources.append(candidates_by_name)
    deltas = document.get("deltas")
    if isinstance(deltas, dict):
        sources.append(deltas)
    for source in sources:
        for key in keys:
            evidence = source.get(key)
            if isinstance(evidence, dict):
                return evidence
    if candidate_name == INCUMBENT_BENCHMARK_NAME:
        measurements = document.get("measurements")
        if isinstance(measurements, dict):
            return measurements
    return None


def _quality_mode(document, candidate_name, execution_mode):
    modes = document.get("modes")
    if not isinstance(modes, dict):
        return (
            (execution_mode, None)
            if execution_mode in ("serial", "two_stream")
            else (None, None)
        )
    keys = [candidate_name]
    if candidate_name == INCUMBENT_BENCHMARK_NAME:
        keys.append("tacker")
    for key in keys:
        mode = modes.get(key)
        if isinstance(mode, dict):
            actual = mode.get(
                "actual_execution_mode",
                mode.get("actual_mode", mode.get("execution_mode")),
            )
            fallback = mode.get("fallback_reason")
            if fallback is None:
                fallback = mode.get("{}_fallback_reason".format(execution_mode))
            return actual, fallback
    return (
        (execution_mode, None)
        if execution_mode in ("serial", "two_stream")
        else (None, None)
    )


def _candidate_correctness(
    quality_document,
    candidate_name,
    variant_id,
    execution_mode,
    overrides,
    correctness_thresholds=None,
    legacy_profiler_error=None,
):
    if correctness_thresholds is None:
        correctness_thresholds = CORRECTNESS_THRESHOLDS
    evidence = _candidate_evidence(
        quality_document, candidate_name, variant_id, overrides
    )
    mode_actual, mode_fallback = _quality_mode(
        quality_document, candidate_name, execution_mode
    )
    source = dict(evidence) if isinstance(evidence, dict) else {}
    if isinstance(evidence, dict):
        nested_measurements = evidence.get("measurements")
        if not isinstance(nested_measurements, dict):
            diagnostics = evidence.get("diagnostics")
            if isinstance(diagnostics, dict) and any(
                key in diagnostics
                for key in ("psnr_drop_db", "ssim_drop", "lpips_increase")
            ):
                nested_measurements = diagnostics
        if isinstance(nested_measurements, dict):
            source.update(nested_measurements)
    actual_mode = source.get(
        "actual_execution_mode",
        source.get("actual_mode", mode_actual),
    )
    fallback_reason = source.get("fallback_reason", mode_fallback)
    reasons = []
    if actual_mode != execution_mode:
        reasons.append("physical execution mode did not match {}".format(execution_mode))
    if fallback_reason is not None:
        reasons.append("physical execution recorded fallback: {}".format(fallback_reason))
    if isinstance(evidence, dict):
        if evidence.get("passed") is False or evidence.get("valid") is False:
            reasons.append("candidate correctness evidence was marked invalid")
        numerics = evidence.get("numerics")
        if isinstance(numerics, dict) and numerics.get("passed") is not True:
            reasons.append("candidate numerical validation did not pass")
    if (
        execution_mode == "tacker"
        and quality_document.get("passed") is False
        and not (
            isinstance(evidence, dict)
            and (
                type(evidence.get("passed")) is bool
                or type(evidence.get("valid")) is bool
            )
        )
    ):
        reasons.append("quality validation report was not marked passed")
    if legacy_profiler_error is not None and execution_mode == "tacker":
        reasons.append(legacy_profiler_error)

    result = {
        "valid": False,
        "actual_execution_mode": actual_mode,
        "fallback_reason": fallback_reason,
        "source": (
            "trusted_baseline"
            if execution_mode in ("serial", "two_stream") and evidence is None
            else "quality_validation"
        ),
        "reasons": reasons,
    }
    if execution_mode == "tacker":
        if candidate_name != INCUMBENT_BENCHMARK_NAME:
            numerics = evidence.get("numerics") if isinstance(evidence, dict) else None
            if not isinstance(numerics, dict) or numerics.get("passed") is not True:
                reasons.append(
                    "candidate kernel-level numerical validation is required"
                )
        quality_fields = (
            ("psnr_drop_db", "psnr_drop_db_max"),
            ("ssim_drop", "ssim_drop_max"),
            ("lpips_increase", "lpips_increase_max"),
        )
        for measurement, threshold_name in quality_fields:
            try:
                value = _finite(
                    source,
                    (measurement,),
                    "candidate {} correctness {}".format(
                        candidate_name, measurement
                    ),
                )
            except AdmissionInputError as error:
                reasons.append(str(error))
                continue
            result[measurement] = value
            if value > correctness_thresholds[threshold_name]:
                reasons.append(
                    "{} exceeds {}".format(measurement, threshold_name)
                )
    result["valid"] = not reasons
    return result


def _legacy_manifest_and_candidate(template):
    manifest_v1, _ = _validate_template(template)
    manifest = {
        "workload_key": WORKLOAD_KEY,
        "rasterizer_commit": manifest_v1["rasterizer_commit"],
        "cuda_arch": manifest_v1["cuda_arch"],
        "compute_capability": list(manifest_v1["compute_capability"]),
        "gpu_name": manifest_v1["gpu_name"],
        "workload": manifest_v1["workload"],
        "iteration": manifest_v1["iteration"],
        "gaussian_count": manifest_v1["gaussian_count"],
        "resolution": list(manifest_v1["resolution"]),
        "persistent_blocks": manifest_v1["persistent_blocks"],
    }
    candidate = {
        "variant_id": LEGACY_VARIANT_ID,
        "execution_mode": "tacker",
        "pair_key": EXPECTED_PAIR_KEY,
        "cuda_symbol": EXPECTED_MIXED_SYMBOL,
        "abi_manifest_sha256": EXPECTED_MIXED_ABI_SHA256,
        "physical_cta_threads": 384,
        "raster_threads": 256,
        "raster_thread_range_inclusive": [0, 255],
        "raster_named_barrier_id": 1,
        "persistent_blocks": manifest_v1["persistent_blocks"],
        "tile_shape": [16, 16],
        "resources": None,
        "fused_nodes": [
            "raster.render_leaf",
            "deformation.pos_deform[1].linear_128x128",
        ],
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
        "backend_subgroups": [
            {
                "name": "pos_deform_l1",
                "thread_range_inclusive": [256, 383],
                "threads": 128,
                "named_barrier_ids": [],
            }
        ],
        "tensor_contract": {
            "input_dtype": "float16",
            "weight_dtype": "float16",
            "bias_dtype": "float32",
            "accumulation_dtype": "float32",
            "output_dtype": "float32",
        },
        "capability_requirements": {
            "cuda_arch": EXPECTED_CUDA_ARCH,
            "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
        },
    }
    return manifest, candidate, dict(CORRECTNESS_THRESHOLDS)


def _descriptor_from_legacy_current_profile(profile, label):
    """Validate an enabled schema-v1 incumbent and expose its v2 descriptor."""

    _validate_template(profile)
    admission = _mapping(profile.get("admission"), "{} admission".format(label))
    if admission != {"enabled": True, "valid": True}:
        raise AdmissionInputError(
            "{} must be an enabled, valid schema-v1 incumbent".format(label)
        )
    measurements = _mapping(
        profile.get("measurements"), "{} measurements".format(label)
    )
    required_measurements = (
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
    for key in required_measurements:
        if not _is_finite_number(measurements.get(key)):
            raise AdmissionInputError(
                "{} measurements.{} must be finite".format(label, key)
            )
    for key in (
        "mixed_p50_ms",
        "solo_raster_p50_ms",
        "solo_head_p50_ms",
        "tacker_end_to_end_p50_ms",
        "two_stream_end_to_end_p50_ms",
    ):
        if float(measurements[key]) <= 0.0:
            raise AdmissionInputError(
                "{} measurements.{} must be > 0".format(label, key)
            )
    _, descriptor, _ = _legacy_manifest_and_candidate(profile)
    descriptor["source_profile_schema_version"] = 1
    return descriptor


def _validate_pos_l1_descriptor(candidate, label):
    candidate = _mapping(candidate, label)
    variant_id = candidate.get("variant_id")
    if not isinstance(variant_id, str) or not variant_id.strip():
        raise AdmissionInputError(
            "{}.variant_id must be a non-empty string".format(label)
        )
    expected = {
        "execution_mode": "tacker",
        "pair_key": EXPECTED_PAIR_KEY,
        "cuda_symbol": EXPECTED_MIXED_SYMBOL,
        "abi_manifest_sha256": EXPECTED_MIXED_ABI_SHA256,
        "physical_cta_threads": 384,
        "raster_threads": 256,
        "raster_thread_range_inclusive": [0, 255],
        "raster_named_barrier_id": 1,
        "tile_shape": [16, 16],
        "fused_nodes": [
            "raster.render_leaf",
            "deformation.pos_deform[1].linear_128x128",
        ],
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
        "backend_subgroups": [
            {
                "name": "pos_deform_l1",
                "thread_range_inclusive": [256, 383],
                "threads": 128,
                "named_barrier_ids": [],
            }
        ],
        "tensor_contract": {
            "input_dtype": "float16",
            "weight_dtype": "float16",
            "bias_dtype": "float32",
            "accumulation_dtype": "float32",
            "output_dtype": "float32",
        },
        "capability_requirements": {
            "cuda_arch": EXPECTED_CUDA_ARCH,
            "compute_capability": EXPECTED_COMPUTE_CAPABILITY,
        },
    }
    for key, value in expected.items():
        if candidate.get(key) != value:
            raise AdmissionInputError(
                "{}.{} must be {!r}".format(label, key, value)
            )
    persistent_blocks = candidate.get("persistent_blocks")
    if type(persistent_blocks) is not int or persistent_blocks < 0:
        raise AdmissionInputError(
            "{}.persistent_blocks must be an int >= 0".format(label)
        )
    resources = candidate.get("resources")
    if "resources" not in candidate or (
        resources is not None and not isinstance(resources, dict)
    ):
        raise AdmissionInputError(
            "{}.resources must be null or an object".format(label)
        )
    if isinstance(resources, dict):
        for name, value in resources.items():
            if value is not None and (
                not _is_finite_number(value) or float(value) < 0.0
            ):
                raise AdmissionInputError(
                    "{}.resources.{} must be null or a finite non-negative "
                    "number".format(label, name)
                )
    return candidate


def _validate_tacker_run_profile_identity(
    benchmark_candidate, descriptor, runs
):
    """Bind every Phase-1 Tacker trial to its SHA-loaded source profile.

    The FPS producer's rows and child metadata are untrusted evidence.  The
    expected identity below comes from the profile bytes loaded by
    ``_descriptor_from_candidate_profile`` (and carried with the descriptor),
    so consistently rewriting a row and all of its runs cannot relabel a
    qualification profile as an enabled deployment, or a different selected
    candidate as the measured kernel.
    """

    candidate_name = benchmark_candidate["name"]
    label = "candidate {} source profile".format(candidate_name)
    source_schema_version = descriptor.get("source_profile_schema_version")
    if source_schema_version not in (1, PROFILE_SCHEMA_VERSION):
        raise AdmissionInputError(
            "{}.schema_version must be 1 or {}".format(
                label, PROFILE_SCHEMA_VERSION
            )
        )
    manifest_hash = descriptor.get("source_profile_manifest_sha256")
    if not _is_lower_hex(manifest_hash, 64):
        raise AdmissionInputError(
            "{}.manifest_sha256 must be a lowercase SHA-256".format(label)
        )
    deployment_enabled = descriptor.get("source_profile_deployment_enabled")
    if type(deployment_enabled) is not bool:
        raise AdmissionInputError(
            "{}.deployment.enabled must be boolean".format(label)
        )

    if source_schema_version == 1:
        if (
            candidate_name != INCUMBENT_BENCHMARK_NAME
            or descriptor.get("variant_id") != LEGACY_VARIANT_ID
            or not deployment_enabled
        ):
            raise AdmissionInputError(
                "schema-v1 source profile is valid only for the enabled "
                "current_tacker incumbent"
            )
        selection_hash = None
        candidate_abi_hash = None
        if descriptor.get("source_profile_selection_sha256") is not None:
            raise AdmissionInputError(
                "schema-v1 source profile may not claim a selection SHA-256"
            )
    else:
        selection_hash = descriptor.get("source_profile_selection_sha256")
        if not _is_lower_hex(selection_hash, 64):
            raise AdmissionInputError(
                "{}.profile_sha256 must be a lowercase SHA-256".format(label)
            )
        # _validate_pos_l1_descriptor pins this to the canonical mixed ABI
        # manifest digest instead of trusting a hash repeated by the run.
        candidate_abi_hash = descriptor["abi_manifest_sha256"]

    expected_qualification_mode = not deployment_enabled
    if (
        benchmark_candidate.get("qualification_mode")
        is not expected_qualification_mode
    ):
        raise AdmissionInputError(
            "FPS benchmark candidate {} qualification_mode disagrees with "
            "its SHA-bound source profile deployment state".format(
                candidate_name
            )
        )

    expected_metrics = {
        "profile_manifest_sha256": manifest_hash,
        "profile_selection_sha256": selection_hash,
        "selected_variant_id": descriptor["variant_id"],
        "selected_candidate_abi_sha256": candidate_abi_hash,
        "persistent_blocks": descriptor["persistent_blocks"],
    }
    hash_fields = {
        "profile_manifest_sha256",
        "profile_selection_sha256",
        "selected_candidate_abi_sha256",
    }
    for run in runs:
        if run.get("candidate_name") != candidate_name:
            continue
        metrics = _mapping(
            run.get("metrics"),
            "FPS benchmark run metrics for {}".format(candidate_name),
        )
        profile_hashes = _mapping(
            _mapping(
                metrics.get("provenance"),
                "FPS benchmark run provenance for {}".format(candidate_name),
            ).get("profile_hashes"),
            "FPS benchmark run profile_hashes for {}".format(candidate_name),
        )
        for field, expected in expected_metrics.items():
            if metrics.get(field) != expected:
                raise AdmissionInputError(
                    "FPS benchmark candidate {} run {} {} disagrees with "
                    "its SHA-bound source profile".format(
                        candidate_name, run.get("run_index"), field
                    )
                )
            if field in hash_fields and profile_hashes.get(field) != expected:
                raise AdmissionInputError(
                    "FPS benchmark candidate {} run {} profile_hashes.{} "
                    "disagrees with its SHA-bound source profile".format(
                        candidate_name, run.get("run_index"), field
                    )
                )


def _validate_source_promotion_evaluation(
    raw_evaluation,
    challenger,
    incumbent,
    by_id,
    label,
):
    """Recompute one sealed incumbent comparison from candidate trial data."""

    evaluation = _mapping(raw_evaluation, label)
    challenger_id = challenger["variant_id"]
    incumbent_id = incumbent["variant_id"]
    if evaluation.get("candidate_variant_id") != challenger_id:
        raise AdmissionInputError(
            "{} candidate order/identity disagrees with equivalence".format(label)
        )
    comparison = _mapping(
        evaluation.get("paired_comparison"),
        "{} paired_comparison".format(label),
    )
    challenger_name = challenger.get("benchmark_candidate_name")
    incumbent_name = incumbent.get("benchmark_candidate_name")
    if (
        not isinstance(challenger_name, str)
        or not challenger_name
        or not isinstance(incumbent_name, str)
        or not incumbent_name
        or comparison.get("candidate") != challenger_name
        or comparison.get("reference") != incumbent_name
    ):
        raise AdmissionInputError(
            "{} paired comparison candidate/reference changed".format(label)
        )
    challenger_performance = challenger["performance"]
    incumbent_performance = incumbent["performance"]
    challenger_trials = [
        float(value)
        for value in challenger_performance["throughput_fps_trials"]
    ]
    incumbent_trials = [
        float(value)
        for value in incumbent_performance["throughput_fps_trials"]
    ]
    challenger_rounds = challenger_performance["round_indices"]
    incumbent_rounds = incumbent_performance["round_indices"]
    if challenger_rounds != incumbent_rounds:
        raise AdmissionInputError(
            "{} candidates do not share paired round indices".format(label)
        )
    if comparison.get("round_indices") != challenger_rounds:
        raise AdmissionInputError(
            "{} paired comparison round indices changed".format(label)
        )
    expected_ratios = [
        candidate_fps / incumbent_fps
        for candidate_fps, incumbent_fps in zip(
            challenger_trials, incumbent_trials
        )
    ]
    observed_ratios = comparison.get("paired_fps_ratios")
    if (
        not isinstance(observed_ratios, list)
        or len(observed_ratios) != len(expected_ratios)
        or any(
            not _is_finite_number(observed)
            or not math.isclose(
                float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
            )
            for observed, expected in zip(observed_ratios, expected_ratios)
        )
    ):
        raise AdmissionInputError(
            "{} paired FPS ratios disagree with trials".format(label)
        )
    median_ratio = (
        float(challenger_performance["median_throughput_fps"])
        / float(incumbent_performance["median_throughput_fps"])
    )
    if (
        not _is_finite_number(comparison.get("median_fps_ratio"))
        or not math.isclose(
            float(comparison["median_fps_ratio"]),
            median_ratio,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not _is_finite_number(comparison.get("median_paired_fps_ratio"))
        or not math.isclose(
            float(comparison["median_paired_fps_ratio"]),
            statistics.median(expected_ratios),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise AdmissionInputError(
            "{} paired comparison median disagrees with trials".format(label)
        )
    interval = _mapping(
        comparison.get("paired_bootstrap_95_ci"),
        "{} paired bootstrap interval".format(label),
    )
    resamples = interval.get("resamples")
    seed = interval.get("seed")
    if (
        resamples != FORMAL_BOOTSTRAP_RESAMPLES
        or seed != FORMAL_BOOTSTRAP_SEED
        or interval.get("confidence") != 0.95
        or interval.get("statistic")
        != "median(candidate_fps)/median(reference_fps)"
        or interval.get("resampling_unit") != "paired_round"
        or interval.get("percentile_method") != "linear_type_7"
    ):
        raise AdmissionInputError(
            "{} paired bootstrap configuration changed; formal evidence "
            "requires 10000 resamples and seed 0".format(label)
        )
    if comparison.get("derived_by_inverting_reported_comparison") is True:
        reverse_lower, reverse_upper = _paired_bootstrap_interval(
            incumbent_trials,
            challenger_trials,
            resamples,
            seed,
            "{}-vs-{}".format(incumbent_name, challenger_name),
        )
        expected_lower = 1.0 / reverse_upper
        expected_upper = 1.0 / reverse_lower
    else:
        expected_lower, expected_upper = _paired_bootstrap_interval(
            challenger_trials,
            incumbent_trials,
            resamples,
            seed,
            "{}-vs-{}".format(challenger_name, incumbent_name),
        )
    for key, expected in (("lower", expected_lower), ("upper", expected_upper)):
        if not _is_finite_number(interval.get(key)) or not math.isclose(
            float(interval[key]), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise AdmissionInputError(
                "{} paired bootstrap {} disagrees with trials".format(label, key)
            )

    lower = expected_lower
    baseline_ratios = {}
    floor_passed = True
    challenger_fps = float(
        challenger_performance["median_throughput_fps"]
    )
    for baseline_id in ("two_stream", incumbent_id):
        baseline = by_id.get(baseline_id)
        if (
            baseline is not None
            and baseline["correctness"]["valid"]
            and baseline.get("performance") is not None
        ):
            baseline_ratio = challenger_fps / float(
                baseline["performance"]["median_throughput_fps"]
            )
            baseline_ratios[baseline_id] = baseline_ratio
            floor_passed = floor_passed and baseline_ratio >= 1.0
    ratio_passed = median_ratio >= PROMOTION_MIN_FPS_RATIO
    ci_passed = lower > PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE
    expected_criteria = {
        "median_fps_ratio": {
            "observed": median_ratio,
            "required_min": PROMOTION_MIN_FPS_RATIO,
            "passed": ratio_passed,
        },
        "paired_bootstrap_95_ci_lower": {
            "observed": lower,
            "required_strictly_greater_than": (
                PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE
            ),
            "passed": ci_passed,
        },
        "baseline_fps_ratios": {
            "observed": baseline_ratios,
            "required_min": 1.0,
            "passed": floor_passed,
        },
    }
    expected_passed = ratio_passed and ci_passed and floor_passed
    if not _evidence_values_equal(
        evaluation.get("criteria"), expected_criteria
    ):
        raise AdmissionInputError(
            "{} promotion criteria disagree with trials".format(label)
        )
    if evaluation.get("passed") is not expected_passed:
        raise AdmissionInputError(
            "{} promotion result disagrees with recomputed gates".format(label)
        )
    return {
        "candidate": challenger,
        "passed": expected_passed,
        "comparison": comparison,
        "criteria": expected_criteria,
        "median_fps_ratio": median_ratio,
        "ci_lower": lower,
    }


def _validate_source_profile_runtime_contract(profile, label):
    """Mirror the schema-v2 runtime invariants for a candidate profile.

    Candidate profile bytes are performance provenance, not a convenient bag
    of descriptor fields.  Validate the whole sealed runtime document before
    extracting its selected descriptor so an invalid manifest or fabricated
    ranking/promotion cannot be laundered into a new admission profile.
    """

    manifest = _mapping(profile.get("manifest"), "{} manifest".format(label))
    expected_manifest = {
        "workload_key": WORKLOAD_KEY,
        "rasterizer_commit": EXPECTED_RASTERIZER_COMMIT,
        "cuda_arch": EXPECTED_CUDA_ARCH,
        "compute_capability": EXPECTED_COMPUTE_CAPABILITY,
        "gpu_name": EXPECTED_GPU_NAME,
        "workload": EXPECTED_SCENE,
        "iteration": EXPECTED_ITERATION,
        "gaussian_count": EXPECTED_GAUSSIANS,
        "resolution": EXPECTED_RESOLUTION,
    }
    for key, expected in expected_manifest.items():
        _require_equal(manifest, key, expected, "{} manifest".format(label))
    persistent_blocks = manifest.get("persistent_blocks")
    if type(persistent_blocks) is not int or persistent_blocks < 0:
        raise AdmissionInputError(
            "{} manifest.persistent_blocks must be an int >= 0".format(label)
        )
    if "pair_key" in manifest:
        raise AdmissionInputError("{} manifest must not contain pair_key".format(label))

    thresholds = _mapping(
        profile.get("correctness_thresholds"),
        "{} correctness_thresholds".format(label),
    )
    for key, locked in CORRECTNESS_THRESHOLDS.items():
        value = thresholds.get(key)
        if not _is_finite_number(value) or float(value) > locked:
            raise AdmissionInputError(
                "{} correctness threshold {} is invalid".format(label, key)
            )

    raw_candidates = profile.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise AdmissionInputError("{} candidates must be non-empty".format(label))
    candidates = []
    by_id = {}
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _mapping(
            raw_candidate, "{} candidate {}".format(label, index)
        )
        variant_id = candidate.get("variant_id")
        if not isinstance(variant_id, str) or not variant_id.strip():
            raise AdmissionInputError(
                "{} candidate.variant_id must be a non-empty string".format(label)
            )
        if variant_id in by_id:
            raise AdmissionInputError(
                "{} has duplicate candidate {!r}".format(label, variant_id)
            )
        mode = candidate.get("execution_mode")
        if mode not in ("serial", "two_stream", "tacker"):
            raise AdmissionInputError(
                "{} candidate {} has invalid execution_mode".format(
                    label, variant_id
                )
            )
        correctness = _mapping(
            candidate.get("correctness"),
            "{} candidate {} correctness".format(label, variant_id),
        )
        if type(correctness.get("valid")) is not bool:
            raise AdmissionInputError(
                "{} candidate {} correctness.valid must be boolean".format(
                    label, variant_id
                )
            )
        if correctness["valid"]:
            if correctness.get("actual_execution_mode") != mode:
                raise AdmissionInputError(
                    "{} candidate {} physical mode changed".format(
                        label, variant_id
                    )
                )
            if correctness.get("fallback_reason") is not None:
                raise AdmissionInputError(
                    "{} candidate {} recorded fallback".format(label, variant_id)
                )
            if mode == "tacker":
                for measurement, threshold_name in (
                    ("psnr_drop_db", "psnr_drop_db_max"),
                    ("ssim_drop", "ssim_drop_max"),
                    ("lpips_increase", "lpips_increase_max"),
                ):
                    value = correctness.get(measurement)
                    if (
                        not _is_finite_number(value)
                        or float(value) > float(thresholds[threshold_name])
                    ):
                        raise AdmissionInputError(
                            "{} candidate {} correctness {} is invalid".format(
                                label, variant_id, measurement
                            )
                        )
        performance = candidate.get("performance")
        if performance is not None:
            performance = _mapping(
                performance,
                "{} candidate {} performance".format(label, variant_id),
            )
            trials = _finite_list(
                performance.get("throughput_fps_trials"),
                "{} candidate {} FPS trials".format(label, variant_id),
                positive=True,
            )
            rounds = performance.get("round_indices")
            if (
                not isinstance(rounds, list)
                or len(rounds) != len(trials)
                or len(set(rounds)) != len(rounds)
                or any(type(value) is not int or value < 0 for value in rounds)
            ):
                raise AdmissionInputError(
                    "{} candidate {} requires one unique round per FPS trial".format(
                        label, variant_id
                    )
                )
            trial_count = performance.get("trial_count")
            if type(trial_count) is not int or trial_count != len(trials):
                raise AdmissionInputError(
                    "{} candidate {} trial_count disagrees with trials".format(
                        label, variant_id
                    )
                )
            median_fps = performance.get("median_throughput_fps")
            if not _is_finite_number(median_fps) or not math.isclose(
                float(median_fps),
                statistics.median(trials),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AdmissionInputError(
                    "{} candidate {} median FPS disagrees with trials".format(
                        label, variant_id
                    )
                )
        diagnostics_value = candidate.get("diagnostics")
        if diagnostics_value is not None and not isinstance(
            diagnostics_value, dict
        ):
            raise AdmissionInputError(
                "{} candidate {} diagnostics must be null or an object".format(
                    label, variant_id
                )
            )
        if mode == "tacker":
            _validate_pos_l1_descriptor(
                candidate, "{} candidate {}".format(label, variant_id)
            )
        candidates.append(candidate)
        by_id[variant_id] = candidate

    for baseline_id in ("serial", "two_stream"):
        baseline = by_id.get(baseline_id)
        if baseline is None or baseline.get("execution_mode") != baseline_id:
            raise AdmissionInputError(
                "{} requires the {} baseline".format(label, baseline_id)
            )
    selected_variant_id = profile.get("selected_variant_id")
    selected = by_id.get(selected_variant_id)
    if selected is None:
        raise AdmissionInputError(
            "{} selected_variant_id must identify one candidate".format(label)
        )
    if (
        selected["execution_mode"] == "tacker"
        and selected.get("persistent_blocks") != persistent_blocks
    ):
        raise AdmissionInputError(
            "{} selected candidate persistent_blocks disagrees with manifest".format(
                label
            )
        )

    deployment = _mapping(
        profile.get("deployment"), "{} deployment".format(label)
    )
    if (
        type(deployment.get("enabled")) is not bool
        or type(deployment.get("valid")) is not bool
        or deployment["enabled"] != deployment["valid"]
    ):
        raise AdmissionInputError(
            "{} deployment enabled/valid must be equal booleans".format(label)
        )
    if not isinstance(profile.get("provenance"), dict):
        raise AdmissionInputError("{} provenance must be an object".format(label))
    if not deployment["enabled"]:
        selection = profile.get("selection")
        if selection is not None and not isinstance(selection, dict):
            raise AdmissionInputError(
                "{} selection must be null or an object".format(label)
            )
        return by_id

    if selected["execution_mode"] != "tacker":
        raise AdmissionInputError(
            "{} enabled deployment must select Tacker".format(label)
        )
    for baseline_id in ("serial", "two_stream"):
        baseline = by_id[baseline_id]
        if (
            not baseline["correctness"]["valid"]
            or baseline.get("performance") is None
        ):
            raise AdmissionInputError(
                "{} enabled deployment requires valid measured {}".format(
                    label, baseline_id
                )
            )
    if (
        not selected["correctness"]["valid"]
        or selected.get("performance") is None
    ):
        raise AdmissionInputError(
            "{} enabled selected candidate must be valid and measured".format(label)
        )

    eligible = [item for item in candidates if item["correctness"]["valid"]]
    if any(item.get("performance") is None for item in eligible):
        raise AdmissionInputError(
            "{} every valid candidate must have performance".format(label)
        )
    benchmark_names = []
    common_rounds = None
    for item in eligible:
        benchmark_name = item.get("benchmark_candidate_name")
        if not isinstance(benchmark_name, str) or not benchmark_name:
            raise AdmissionInputError(
                "{} every valid candidate requires benchmark_candidate_name".format(
                    label
                )
            )
        benchmark_names.append(benchmark_name)
        trials = item["performance"]["throughput_fps_trials"]
        if len(trials) < 10:
            raise AdmissionInputError(
                "{} every valid candidate requires at least 10 FPS trials".format(
                    label
                )
            )
        rounds = item["performance"]["round_indices"]
        if common_rounds is None:
            common_rounds = rounds
        elif rounds != common_rounds:
            raise AdmissionInputError(
                "{} valid candidates must share paired round indices".format(label)
            )
    if len(benchmark_names) != len(set(benchmark_names)):
        raise AdmissionInputError(
            "{} valid candidates require unique benchmark names".format(label)
        )
    eligible.sort(
        key=lambda item: (
            -float(item["performance"]["median_throughput_fps"]),
            item["benchmark_candidate_name"],
        )
    )
    ranking_ids = [item["variant_id"] for item in eligible]
    selection = _mapping(
        profile.get("selection"), "{} selection".format(label)
    )
    for key in ("eligible_variant_ids", "global_median_fps_ranking"):
        if selection.get(key) != ranking_ids:
            raise AdmissionInputError(
                "{} selection {} disagrees with ranking".format(label, key)
            )
    expected_ineligible = {
        item["variant_id"]
        for item in candidates
        if not item["correctness"]["valid"]
    }
    recorded_ineligible = selection.get("ineligible_variant_ids")
    if (
        not isinstance(recorded_ineligible, list)
        or len(recorded_ineligible) != len(set(recorded_ineligible))
        or set(recorded_ineligible) != expected_ineligible
    ):
        raise AdmissionInputError(
            "{} selection ineligible candidates disagree with correctness".format(
                label
            )
        )
    experimental_id = selection.get("experimental_winner_variant_id")
    if experimental_id != ranking_ids[0]:
        raise AdmissionInputError(
            "{} experimental winner disagrees with ranking".format(label)
        )
    if selection.get("deployment_winner_variant_id") != selected_variant_id:
        raise AdmissionInputError(
            "{} deployment winner disagrees with selected_variant_id".format(label)
        )
    top_fps = float(eligible[0]["performance"]["median_throughput_fps"])
    equivalent = [
        item
        for item in eligible
        if (
            top_fps - float(item["performance"]["median_throughput_fps"])
        )
        / top_fps
        <= EQUIVALENCE_FRACTION
    ]
    equivalent.sort(key=_equivalence_preference_key)
    equivalence = _mapping(
        selection.get("equivalence"), "{} selection equivalence".format(label)
    )
    if (
        not _is_finite_number(equivalence.get("fraction"))
        or not math.isclose(
            float(equivalence["fraction"]),
            EQUIVALENCE_FRACTION,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or equivalence.get("candidate_variant_ids_in_preference_order")
        != [item["variant_id"] for item in equivalent]
        or equivalence.get("preferred_variant_id") != equivalent[0]["variant_id"]
    ):
        raise AdmissionInputError(
            "{} equivalence evidence disagrees with candidates".format(label)
        )
    incumbent_id = selection.get("incumbent_variant_id")
    incumbent = by_id.get(incumbent_id)
    if incumbent is None or incumbent["execution_mode"] != "tacker":
        raise AdmissionInputError(
            "{} incumbent must identify a Tacker candidate".format(label)
        )
    promotion = _mapping(
        selection.get("promotion"), "{} selection promotion".format(label)
    )
    minimum_ratio = promotion.get("minimum_median_fps_ratio")
    minimum_lower = promotion.get("minimum_bootstrap_lower_exclusive")
    if (
        type(promotion.get("passed")) is not bool
        or not _is_finite_number(minimum_ratio)
        or not math.isclose(
            float(minimum_ratio),
            PROMOTION_MIN_FPS_RATIO,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not _is_finite_number(minimum_lower)
        or not math.isclose(
            float(minimum_lower),
            PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise AdmissionInputError(
            "{} promotion gate contract changed".format(label)
        )
    if promotion.get("incumbent_variant_id") != incumbent_id:
        raise AdmissionInputError(
            "{} promotion incumbent disagrees with selection".format(label)
        )
    raw_evaluations = promotion.get("candidate_evaluations")
    if not isinstance(raw_evaluations, list):
        raise AdmissionInputError(
            "{} promotion candidate_evaluations must be an array".format(label)
        )

    expected_evaluations = []
    if incumbent["correctness"]["valid"]:
        expected_challengers = [
            item
            for item in equivalent
            if item["variant_id"] != incumbent_id
        ]
        if len(raw_evaluations) != len(expected_challengers):
            raise AdmissionInputError(
                "{} promotion evaluations do not cover equivalent challengers".format(
                    label
                )
            )
        for index, (raw_evaluation, challenger) in enumerate(
            zip(raw_evaluations, expected_challengers)
        ):
            expected_evaluations.append(
                _validate_source_promotion_evaluation(
                    raw_evaluation,
                    challenger,
                    incumbent,
                    by_id,
                    "{} promotion evaluation {}".format(label, index),
                )
            )
        passing_evaluations = [
            item for item in expected_evaluations if item["passed"]
        ]
        selected_evaluation = None
        if passing_evaluations:
            expected_winner = passing_evaluations[0]["candidate"]
            expected_decision = "promoted_equivalent_challenger"
            expected_passed = True
            selected_evaluation = passing_evaluations[0]
        elif experimental_id == incumbent_id:
            expected_winner = incumbent
            expected_decision = "incumbent_is_global_winner"
            expected_passed = True
        else:
            expected_winner = incumbent
            expected_decision = "retained_incumbent"
            expected_passed = False
            preferred_id = equivalent[0]["variant_id"]
            selected_evaluation = next(
                (
                    item
                    for item in expected_evaluations
                    if item["candidate"]["variant_id"] == preferred_id
                ),
                expected_evaluations[0],
            )
    else:
        if raw_evaluations:
            raise AdmissionInputError(
                "{} invalid incumbent must not use promotion bootstrap gates".format(label)
            )
        two_stream = by_id["two_stream"]
        two_stream_fps = float(
            two_stream["performance"]["median_throughput_fps"]
        )
        expected_winner = next(
            (
                item
                for item in equivalent
                if float(item["performance"]["median_throughput_fps"])
                >= two_stream_fps
            ),
            eligible[0],
        )
        expected_decision = "selected_best_valid_candidate_incumbent_invalid"
        expected_passed = True
        selected_evaluation = None

    two_stream = by_id["two_stream"]
    if (
        expected_winner["execution_mode"] == "tacker"
        and float(expected_winner["performance"]["median_throughput_fps"])
        < float(two_stream["performance"]["median_throughput_fps"])
    ):
        expected_winner = two_stream
        expected_decision = "selected_two_stream_floor"
        expected_passed = True
        selected_evaluation = None

    expected_required = equivalent[0]["variant_id"] != incumbent_id
    if expected_decision == "incumbent_is_global_winner":
        expected_required = False
    elif expected_decision == "selected_two_stream_floor":
        expected_required = True
    expected_challenger_id = (
        selected_evaluation["candidate"]["variant_id"]
        if selected_evaluation is not None
        else expected_winner["variant_id"]
    )
    if (
        selected_variant_id != expected_winner["variant_id"]
        or promotion.get("decision") != expected_decision
        or promotion.get("passed") is not expected_passed
        or promotion.get("required") is not expected_required
        or promotion.get("challenger_variant_id") != expected_challenger_id
    ):
        raise AdmissionInputError(
            "{} promotion decision disagrees with recomputed challenger gates".format(
                label
            )
        )
    if selected_evaluation is not None:
        if (
            promotion.get("paired_comparison")
            != selected_evaluation["comparison"]
            or promotion.get("criteria") != selected_evaluation["criteria"]
            or not _is_finite_number(promotion.get("median_fps_ratio"))
            or not math.isclose(
                float(promotion["median_fps_ratio"]),
                selected_evaluation["median_fps_ratio"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not _is_finite_number(
                promotion.get("paired_bootstrap_95_ci_lower")
            )
            or not math.isclose(
                float(promotion["paired_bootstrap_95_ci_lower"]),
                selected_evaluation["ci_lower"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise AdmissionInputError(
                "{} top-level promotion evidence disagrees with recomputed evaluation".format(
                    label
                )
            )
    selected_fps = float(selected["performance"]["median_throughput_fps"])
    if selected_fps < float(
        by_id["two_stream"]["performance"]["median_throughput_fps"]
    ):
        raise AdmissionInputError(
            "{} selected Tacker is slower than two_stream".format(label)
        )
    return by_id


def _v2_template_parts(template):
    template = _mapping(template, "profile template")
    if template.get("schema_version") == 1:
        return _legacy_manifest_and_candidate(template)
    if template.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise AdmissionInputError("profile template schema_version must be 1 or 2")
    _require_equal(template, "workload_key", WORKLOAD_KEY, "profile template")
    _require_equal(
        template,
        "selection_objective",
        SELECTION_OBJECTIVE,
        "profile template",
    )
    manifest = _mapping(template.get("manifest"), "profile template manifest")
    expected = {
        "workload_key": WORKLOAD_KEY,
        "rasterizer_commit": EXPECTED_RASTERIZER_COMMIT,
        "cuda_arch": EXPECTED_CUDA_ARCH,
        "compute_capability": EXPECTED_COMPUTE_CAPABILITY,
        "gpu_name": EXPECTED_GPU_NAME,
        "workload": EXPECTED_SCENE,
        "iteration": EXPECTED_ITERATION,
        "gaussian_count": EXPECTED_GAUSSIANS,
        "resolution": EXPECTED_RESOLUTION,
    }
    for key, value in expected.items():
        _require_equal(manifest, key, value, "profile template manifest")
    if "pair_key" in manifest:
        raise AdmissionInputError(
            "schema-v2 manifest must keep pair_key in its Tacker candidate"
        )
    persistent_blocks = manifest.get("persistent_blocks")
    if type(persistent_blocks) is not int or persistent_blocks < 0:
        raise AdmissionInputError(
            "profile template manifest.persistent_blocks must be an int >= 0"
        )
    if template.get("manifest_sha256") != manifest_sha256(manifest):
        raise AdmissionInputError("profile template manifest SHA-256 mismatch")
    thresholds = _mapping(
        template.get("correctness_thresholds"),
        "profile template correctness_thresholds",
    )
    normalized_thresholds = {}
    for key, locked in CORRECTNESS_THRESHOLDS.items():
        value = _finite(
            thresholds,
            (key,),
            "profile template correctness_thresholds.{}".format(key),
            nonnegative=True,
        )
        if value > locked:
            raise AdmissionInputError(
                "profile template correctness threshold {} weakens {}".format(
                    key, locked
                )
            )
        normalized_thresholds[key] = value
    candidates = template.get("candidates")
    if not isinstance(candidates, list):
        raise AdmissionInputError("profile template candidates must be an array")
    legacy = next(
        (
            dict(candidate)
            for candidate in candidates
            if isinstance(candidate, dict)
            and candidate.get("variant_id") == LEGACY_VARIANT_ID
        ),
        None,
    )
    if legacy is None:
        raise AdmissionInputError(
            "profile template must include the legacy_pos_l1 candidate"
        )
    if legacy.get("execution_mode") != "tacker":
        raise AdmissionInputError("legacy_pos_l1 must use tacker execution mode")
    if legacy.get("pair_key") != EXPECTED_PAIR_KEY:
        raise AdmissionInputError("legacy_pos_l1 pair_key changed")
    if legacy.get("persistent_blocks") != persistent_blocks:
        raise AdmissionInputError(
            "legacy_pos_l1 persistent_blocks disagrees with manifest"
        )
    _validate_pos_l1_descriptor(legacy, "profile template legacy_pos_l1")
    deployment = template.get("deployment")
    if deployment not in (
        {"enabled": False, "valid": False},
        None,
    ):
        raise AdmissionInputError("profile template must be disabled")
    if "profile_sha256" in template:
        if template["profile_sha256"] != profile_sha256(template):
            raise AdmissionInputError("profile template selection SHA-256 mismatch")
    return dict(manifest), legacy, normalized_thresholds


def _completion_diagnostics(benchmark, candidate_name):
    fields = (
        "p50_frame_ms",
        "p95_frame_ms",
        "max_frame_ms",
        "cuda_event_total_render_ms",
    )
    result = {}
    for field in fields:
        values = []
        for run in benchmark["runs"]:
            if not isinstance(run, dict) or run.get("candidate_name") != candidate_name:
                continue
            metrics = run.get("metrics")
            if not isinstance(metrics, dict) or field not in metrics:
                continue
            value = metrics[field]
            if _is_finite_number(value):
                values.append(float(value))
        if values:
            result["{}_trials".format(field)] = values
            result["median_{}".format(field)] = statistics.median(values)
    return result


def _find_paired_comparison(benchmark, candidate_name, reference_name):
    for comparison in benchmark["paired_comparisons"]:
        if (
            comparison["candidate"] == candidate_name
            and comparison["reference"] == reference_name
        ):
            return dict(comparison)
    raise AdmissionInputError(
        "FPS benchmark lacks direct paired comparison {} vs {}".format(
            candidate_name, reference_name
        )
    )


def _variant_id(candidate_name, execution_mode, descriptor):
    if isinstance(descriptor, dict) and isinstance(descriptor.get("variant_id"), str):
        return descriptor["variant_id"]
    if candidate_name == INCUMBENT_BENCHMARK_NAME:
        return LEGACY_VARIANT_ID
    return candidate_name


def _candidate_selection_metadata(
    candidate, benchmark_metadata, phase0_compatibility=False
):
    """Collect the sealed, lower-is-better equivalence tie-break values."""

    result = dict(benchmark_metadata or {})
    resources = candidate.get("resources")
    if not isinstance(resources, dict):
        resources = {}
    for key in (
        "abi_complexity",
        "peak_memory_bytes",
        "registers_per_thread",
        "shared_memory_bytes",
    ):
        descriptor_values = []
        if key in candidate and candidate[key] is not None:
            descriptor_values.append(candidate[key])
        if key in resources and resources[key] is not None:
            descriptor_values.append(resources[key])
        for value in descriptor_values:
            if not _is_finite_number(value) or float(value) < 0.0:
                raise AdmissionInputError(
                    "candidate {} selection metadata {} must be finite and >= 0".format(
                        candidate["variant_id"], key
                    )
                )
            if key in result and not math.isclose(
                float(result[key]), float(value), rel_tol=0.0, abs_tol=0.0
            ):
                raise AdmissionInputError(
                    "candidate {} descriptor {} disagrees with FPS benchmark "
                    "selection metadata".format(candidate["variant_id"], key)
                )
        # A prefiltered FPS report contains the selector's effective metadata.
        # Descriptor resources may verify a reported value, but must not fill
        # an omission and silently change the selector's already-made choice.
        if phase0_compatibility and key not in result and descriptor_values:
            result[key] = float(descriptor_values[0])
    if phase0_compatibility and "abi_complexity" not in result:
        result["abi_complexity"] = {
            "serial": 0.0,
            "two_stream": 1.0,
            "tacker": 2.0,
        }[candidate["execution_mode"]]
    return result


def _equivalence_preference_key(candidate):
    metadata = candidate.get("selection_metadata", {})
    key = []
    for name in (
        "abi_complexity",
        "peak_memory_bytes",
        "registers_per_thread",
        "shared_memory_bytes",
    ):
        value = metadata.get(name)
        key.append((1, 0.0) if value is None else (0, float(value)))
    key.append(candidate["benchmark_candidate_name"])
    return tuple(key)


def evaluate_admission_v2(inputs):
    """Qualify candidates, rank whole-run FPS, and select a deployment.

    Correctness and performance are deliberately separate.  A candidate that
    slows the Raster leaf or loses a microbenchmark remains eligible if its
    correctness evidence passes; only median whole-sequence throughput ranks
    eligible candidates.  Replacing a valid incumbent additionally requires a
    1% ratio and a paired-bootstrap 95% lower bound strictly above 1.0.
    """

    benchmark_key = (
        "fps_benchmark" if "fps_benchmark" in inputs else "benchmark"
    )
    required = (
        "device",
        "quality",
        "raster",
        "leaf",
        "mixed_abi",
        "head_abi",
        "template",
        benchmark_key,
    )
    digests = {}
    for name, value in inputs.items():
        try:
            digests[name] = _source_digest(value)
        except (TypeError, ValueError):
            digests[name] = None
    report = _v2_base_report(digests)
    noncanonical_inputs = sorted(
        name for name, digest in digests.items() if digest is None
    )
    if noncanonical_inputs:
        report["errors"].append(
            "inputs must be finite canonical JSON (invalid: {!r})".format(
                noncanonical_inputs
            )
        )
        return report, None
    try:
        for name in required:
            if name not in inputs:
                raise AdmissionInputError("{} input is required".format(name))
        device = _device(inputs["device"])
        device_contract = _leaf_workload_contract(
            inputs["device"], "device input"
        )
        _validate_mixed_abi(inputs["mixed_abi"])
        _validate_head_abi(inputs["head_abi"])
        manifest, legacy_descriptor, correctness_thresholds = _v2_template_parts(
            inputs["template"]
        )
        benchmark = _validate_fps_benchmark(inputs[benchmark_key])
        quality_contract = _quality_contract_v2(inputs["quality"])
        benchmark_contract = benchmark["contract"]
        for key in (
            "split",
            "profile_frames",
            "view_indices",
            "model_path",
            "source_path",
        ):
            if quality_contract[key] != benchmark_contract[key]:
                raise AdmissionInputError(
                    "quality and FPS benchmark {} must match".format(key)
                )
        for key in ("split", "model_path", "source_path"):
            if device_contract[key] != quality_contract[key]:
                raise AdmissionInputError(
                    "device and quality {} provenance must match".format(key)
                )
        stable_environment = benchmark.get("stable_environment")
        if stable_environment is not None:
            gpu_name = stable_environment.get("gpu_name")
            if gpu_name is not None and gpu_name != device["name"]:
                raise AdmissionInputError(
                    "FPS benchmark GPU does not match device qualification"
                )
        for name in ("quality", "raster", "leaf"):
            _optional_device_name(inputs[name], device["name"], name)
    except (AdmissionInputError, KeyError, TypeError, ValueError) as error:
        report["errors"].append(str(error))
        return report, None

    descriptors = inputs.get("candidate_descriptors", {})
    if not isinstance(descriptors, dict):
        report["errors"].append("candidate_descriptors must be a JSON object")
        return report, None
    current_descriptor_value = descriptors.get(INCUMBENT_BENCHMARK_NAME)
    if current_descriptor_value is None:
        if not benchmark["phase0_compatibility"]:
            report["errors"].append(
                "prefiltered FPS benchmark requires a SHA-bound "
                "current_tacker profile descriptor"
            )
            return report, None
        current_descriptor = dict(legacy_descriptor)
    elif isinstance(current_descriptor_value, dict):
        current_descriptor = dict(current_descriptor_value)
        try:
            _validate_pos_l1_descriptor(
                current_descriptor, "current_tacker descriptor"
            )
        except AdmissionInputError as error:
            report["errors"].append(str(error))
            return report, None
    else:
        report["errors"].append(
            "current_tacker descriptor must be a JSON object"
        )
        return report, None
    if not benchmark["phase0_compatibility"]:
        expected_current_sha256 = benchmark["candidate_by_name"][
            INCUMBENT_BENCHMARK_NAME
        ].get("profile_file_sha256")
        actual_current_sha256 = current_descriptor.get(
            "source_profile_file_sha256"
        )
        if (
            not _is_lower_hex(expected_current_sha256, 64)
            or actual_current_sha256 != expected_current_sha256
        ):
            report["errors"].append(
                "prefiltered FPS benchmark current_tacker descriptor must be "
                "bound to its exact profile_file_sha256"
            )
            return report, None

    # Profiler failures invalidate the physical current Tacker candidate, rather than
    # preventing a correctness-valid baseline from being selected.  Structural
    # provenance mismatches are still surfaced in the reason and sealed output.
    profiler_errors = []
    raster = None
    leaf = None
    try:
        raster = _raster_measurements(inputs["raster"])
        leaf = _leaf_measurements(inputs["leaf"])
        if raster["workload_contract"] != leaf["workload_contract"]:
            raise AdmissionInputError(
                "raster and leaf workload provenance must match exactly"
            )
        if raster["workload_contract"] != device_contract:
            raise AdmissionInputError(
                "device and profiler workload provenance must match exactly"
            )
        for key in ("split", "model_path", "source_path"):
            if leaf["workload_contract"][key] != quality_contract[key]:
                raise AdmissionInputError(
                    "leaf and quality {} provenance must match".format(key)
                )
        if not math.isclose(
            raster["solo_raster_p50_ms"],
            leaf["solo_raster_p50_ms"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise AdmissionInputError(
                "raster and leaf inputs must use the same solo Raster p50"
            )
        for label, persistent_blocks in (
            ("raster", raster["persistent_blocks"]),
            ("leaf", leaf["persistent_blocks"]),
        ):
            if persistent_blocks != current_descriptor["persistent_blocks"]:
                raise AdmissionInputError(
                    "{} persistent_blocks does not match current Tacker {}".format(
                        label, current_descriptor["variant_id"]
                    )
                )
    except (AdmissionInputError, KeyError, TypeError, ValueError) as error:
        profiler_errors.append(str(error))

    diagnostics = {}
    if raster is not None:
        diagnostics.update(
            {
                "solo_raster_p50_ms": raster["solo_raster_p50_ms"],
                "mixed_raster_p50_ms": raster["mixed_raster_p50_ms"],
                "raster_slowdown_pct": raster["raster_slowdown_pct"],
            }
        )
    if leaf is not None:
        solo_sum = leaf["solo_raster_p50_ms"] + leaf["solo_head_p50_ms"]
        diagnostics.update(
            {
                "mixed_p50_ms": leaf["mixed_p50_ms"],
                "solo_raster_p50_ms": leaf["solo_raster_p50_ms"],
                "solo_head_p50_ms": leaf["solo_head_p50_ms"],
                "solo_leaf_sum_p50_ms": solo_sum,
                "mixed_leaf_faster_than_solo_sum": leaf["mixed_p50_ms"] < solo_sum,
            }
        )
    if profiler_errors:
        diagnostics["profiler_errors"] = list(profiler_errors)

    overrides = inputs.get(
        "candidate_correctness", inputs.get("correctness", {})
    )
    candidates = []
    seen_variant_ids = set()
    for benchmark_candidate in benchmark["candidates"]:
        name = benchmark_candidate["name"]
        mode = benchmark_candidate["execution_mode"]
        descriptor = None
        if name == INCUMBENT_BENCHMARK_NAME:
            descriptor = dict(current_descriptor)
        elif mode == "tacker":
            raw_descriptor = descriptors.get(name)
            if raw_descriptor is None:
                raw_descriptor = descriptors.get(
                    benchmark_candidate.get("variant_id", "")
                )
            if isinstance(raw_descriptor, dict):
                descriptor = dict(raw_descriptor)
        if mode == "tacker" and descriptor is not None:
            try:
                _validate_pos_l1_descriptor(
                    descriptor, "candidate {} descriptor".format(name)
                )
            except AdmissionInputError as error:
                report["errors"].append(str(error))
                return report, None
            if (
                not benchmark["phase0_compatibility"]
                and name != INCUMBENT_BENCHMARK_NAME
                and descriptor.get("source_profile_file_sha256")
                != benchmark_candidate.get("profile_file_sha256")
            ):
                report["errors"].append(
                    "prefiltered FPS benchmark candidate {} descriptor must be "
                    "bound to its exact profile_file_sha256".format(name)
                )
                return report, None
            if not benchmark["phase0_compatibility"]:
                try:
                    _validate_tacker_run_profile_identity(
                        benchmark_candidate, descriptor, benchmark["runs"]
                    )
                except AdmissionInputError as error:
                    report["errors"].append(str(error))
                    return report, None
        variant_id = _variant_id(name, mode, descriptor)
        if variant_id in seen_variant_ids:
            report["errors"].append(
                "candidate variant_id {!r} is not unique".format(variant_id)
            )
            return report, None
        seen_variant_ids.add(variant_id)
        profiler_error = None
        if name == INCUMBENT_BENCHMARK_NAME and profiler_errors:
            profiler_error = "; ".join(profiler_errors)
        candidate_overrides = overrides
        if (
            not benchmark["phase0_compatibility"]
            and _candidate_evidence(
                inputs["quality"], name, variant_id, overrides
            )
            is None
        ):
            candidate_overrides = {name: benchmark["correctness_qualifications"][name]}
        correctness = _candidate_correctness(
            inputs["quality"],
            name,
            variant_id,
            mode,
            candidate_overrides,
            correctness_thresholds=correctness_thresholds,
            legacy_profiler_error=profiler_error,
        )
        benchmark_qualification = benchmark["correctness_qualifications"][name]
        if benchmark_qualification["valid"] is False:
            correctness["valid"] = False
            correctness["reasons"].append(
                "FPS benchmark excluded candidate as correctness-invalid"
            )
        correctness["fps_benchmark_qualification"] = dict(
            benchmark_qualification
        )
        if mode == "tacker" and descriptor is None:
            report["errors"].append(
                "candidate {} has no sealed Tacker runtime descriptor".format(
                    name
                )
            )
            return report, None
        performance = (
            dict(benchmark["summaries"][name])
            if name in benchmark["summaries"]
            else None
        )
        candidate = {
            "variant_id": variant_id,
            "benchmark_candidate_name": name,
            "execution_mode": mode,
            "correctness": correctness,
            "performance": performance,
            "diagnostics": _completion_diagnostics(benchmark, name),
        }
        if name == INCUMBENT_BENCHMARK_NAME:
            candidate["diagnostics"].update(diagnostics)
        if descriptor is not None:
            protected = {
                "variant_id",
                "execution_mode",
                "correctness",
                "performance",
                "diagnostics",
                "benchmark_candidate_name",
            }
            for key, value in descriptor.items():
                if key not in protected:
                    candidate[key] = value
        try:
            candidate["selection_metadata"] = _candidate_selection_metadata(
                candidate,
                benchmark["candidate_selection_metadata"].get(name, {}),
                phase0_compatibility=benchmark["phase0_compatibility"],
            )
        except AdmissionInputError as error:
            report["errors"].append(str(error))
            return report, None
        candidates.append(candidate)

    # The deployed runtime always needs both physical fallback baselines.  Do
    # not rely solely on the benchmark's prefilter declaration: independently
    # recomputed mode/fallback evidence must also leave them valid here.
    for baseline_name in ("serial", "two_stream"):
        baseline = next(
            candidate
            for candidate in candidates
            if candidate["benchmark_candidate_name"] == baseline_name
        )
        if (
            not baseline["correctness"]["valid"]
            or baseline["performance"] is None
        ):
            report["candidates"] = candidates
            report["errors"].append(
                "{} baseline failed admission correctness or measurement".format(
                    baseline_name
                )
            )
            return report, None

    eligible = [
        candidate
        for candidate in candidates
        if candidate["correctness"]["valid"]
        and candidate["performance"] is not None
    ]
    if not eligible:
        report["candidates"] = candidates
        report["errors"].append("no correctness-valid candidate is selectable")
        return report, None
    eligible.sort(
        key=lambda candidate: (
            -candidate["performance"]["median_throughput_fps"],
            candidate["benchmark_candidate_name"],
        )
    )
    experimental = eligible[0]
    incumbent = next(
        candidate
        for candidate in candidates
        if candidate["benchmark_candidate_name"] == INCUMBENT_BENCHMARK_NAME
    )
    top_fps = experimental["performance"]["median_throughput_fps"]
    equivalent = [
        candidate
        for candidate in eligible
        if (
            top_fps - candidate["performance"]["median_throughput_fps"]
        )
        / top_fps
        <= EQUIVALENCE_FRACTION
    ]
    equivalent.sort(key=_equivalence_preference_key)
    preferred = equivalent[0]
    candidate_evaluations = []
    promotion = {
        "challenger_variant_id": preferred["variant_id"],
        "incumbent_variant_id": incumbent["variant_id"],
        "minimum_median_fps_ratio": PROMOTION_MIN_FPS_RATIO,
        "minimum_bootstrap_lower_exclusive": PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE,
        "required": preferred["variant_id"] != incumbent["variant_id"],
        "passed": False,
        "decision": "retained_incumbent",
        "candidate_evaluations": candidate_evaluations,
    }
    deployment_winner = incumbent

    if not incumbent["correctness"]["valid"]:
        two_stream = next(
            (
                item
                for item in candidates
                if item["variant_id"] == "two_stream"
                and item["correctness"]["valid"]
                and item["performance"] is not None
            ),
            None,
        )
        deployment_winner = next(
            (
                challenger
                for challenger in equivalent
                if two_stream is None
                or challenger["performance"]["median_throughput_fps"]
                >= two_stream["performance"]["median_throughput_fps"]
            ),
            experimental,
        )
        promotion.update(
            {
                "challenger_variant_id": deployment_winner["variant_id"],
                "passed": True,
                "decision": "selected_best_valid_candidate_incumbent_invalid",
            }
        )
    else:
        passing_challengers = []
        for challenger in equivalent:
            if challenger["variant_id"] == incumbent["variant_id"]:
                continue
            try:
                comparison = _find_paired_comparison(
                    benchmark,
                    challenger["benchmark_candidate_name"],
                    incumbent["benchmark_candidate_name"],
                )
            except AdmissionInputError as error:
                report["candidates"] = candidates
                report["errors"].append(str(error))
                return report, None
            ratio = comparison["median_fps_ratio"]
            lower = comparison["paired_bootstrap_95_ci"]["lower"]
            candidate_fps = challenger["performance"]["median_throughput_fps"]
            baseline_ratios = {}
            floor_passed = True
            for baseline_variant_id in (
                "two_stream",
                incumbent["variant_id"],
            ):
                baseline = next(
                    (
                        item
                        for item in candidates
                        if item["variant_id"] == baseline_variant_id
                        and item["correctness"]["valid"]
                        and item["performance"] is not None
                    ),
                    None,
                )
                if baseline is not None:
                    baseline_ratio = (
                        candidate_fps
                        / baseline["performance"]["median_throughput_fps"]
                    )
                    baseline_ratios[baseline_variant_id] = baseline_ratio
                    floor_passed = floor_passed and baseline_ratio >= 1.0
            ratio_passed = ratio >= PROMOTION_MIN_FPS_RATIO
            ci_passed = lower > PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE
            passed = ratio_passed and ci_passed and floor_passed
            evaluation = {
                "candidate_variant_id": challenger["variant_id"],
                "passed": passed,
                "paired_comparison": comparison,
                "criteria": {
                    "median_fps_ratio": {
                        "observed": ratio,
                        "required_min": PROMOTION_MIN_FPS_RATIO,
                        "passed": ratio_passed,
                    },
                    "paired_bootstrap_95_ci_lower": {
                        "observed": lower,
                        "required_strictly_greater_than": (
                            PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE
                        ),
                        "passed": ci_passed,
                    },
                    "baseline_fps_ratios": {
                        "observed": baseline_ratios,
                        "required_min": 1.0,
                        "passed": floor_passed,
                    },
                },
            }
            candidate_evaluations.append(evaluation)
            if passed:
                passing_challengers.append(challenger)

        if passing_challengers:
            deployment_winner = passing_challengers[0]
            selected_evaluation = next(
                item
                for item in candidate_evaluations
                if item["candidate_variant_id"]
                == deployment_winner["variant_id"]
            )
            promotion.update(
                {
                    "challenger_variant_id": deployment_winner["variant_id"],
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
                    "passed": True,
                    "decision": "promoted_equivalent_challenger",
                }
            )
        elif experimental["variant_id"] == incumbent["variant_id"]:
            promotion.update(
                {
                    "challenger_variant_id": incumbent["variant_id"],
                    "required": False,
                    "passed": True,
                    "decision": "incumbent_is_global_winner",
                }
            )
        elif candidate_evaluations:
            preferred_evaluation = next(
                (
                    item
                    for item in candidate_evaluations
                    if item["candidate_variant_id"] == preferred["variant_id"]
                ),
                candidate_evaluations[0],
            )
            promotion.update(
                {
                    "challenger_variant_id": preferred_evaluation[
                        "candidate_variant_id"
                    ],
                    "paired_comparison": preferred_evaluation[
                        "paired_comparison"
                    ],
                    "median_fps_ratio": preferred_evaluation["criteria"][
                        "median_fps_ratio"
                    ]["observed"],
                    "paired_bootstrap_95_ci_lower": preferred_evaluation[
                        "criteria"
                    ]["paired_bootstrap_95_ci_lower"]["observed"],
                    "criteria": preferred_evaluation["criteria"],
                    "passed": False,
                    "decision": "retained_incumbent",
                }
            )

    # Promotion hysteresis may retain the incumbent, but it may never retain a
    # Tacker deployment that is slower than the valid two-stream fallback.
    two_stream_floor = next(
        item for item in candidates if item["variant_id"] == "two_stream"
    )
    if (
        deployment_winner["execution_mode"] == "tacker"
        and deployment_winner["performance"]["median_throughput_fps"]
        < two_stream_floor["performance"]["median_throughput_fps"]
    ):
        deployment_winner = two_stream_floor
        promotion.update(
            {
                "challenger_variant_id": "two_stream",
                "required": True,
                "passed": True,
                "decision": "selected_two_stream_floor",
            }
        )

    selected_variant_id = deployment_winner["variant_id"]
    selected_persistent_blocks = deployment_winner.get("persistent_blocks")
    if deployment_winner["execution_mode"] == "tacker":
        if type(selected_persistent_blocks) is not int or selected_persistent_blocks < 0:
            report["candidates"] = candidates
            report["errors"].append(
                "selected Tacker candidate requires persistent_blocks"
            )
            return report, None
        manifest = dict(manifest)
        manifest["persistent_blocks"] = selected_persistent_blocks
    selection = {
        "eligible_variant_ids": [candidate["variant_id"] for candidate in eligible],
        "ineligible_variant_ids": [
            candidate["variant_id"]
            for candidate in candidates
            if not candidate["correctness"]["valid"]
        ],
        "global_median_fps_ranking": [
            candidate["variant_id"] for candidate in eligible
        ],
        "experimental_winner_variant_id": experimental["variant_id"],
        "incumbent_variant_id": incumbent["variant_id"],
        "deployment_winner_variant_id": selected_variant_id,
        "equivalence": {
            "fraction": EQUIVALENCE_FRACTION,
            "top_median_throughput_fps": top_fps,
            "candidate_variant_ids_in_preference_order": [
                candidate["variant_id"] for candidate in equivalent
            ],
            "preferred_variant_id": preferred["variant_id"],
            "tie_break_order": [
                "abi_complexity",
                "peak_memory_bytes",
                "registers_per_thread",
                "shared_memory_bytes",
                "benchmark_candidate_name",
            ],
            "missing_metadata_policy": "known_first_then_benchmark_candidate_name",
        },
        "promotion": promotion,
    }
    expected_selector = benchmark.get("expected_selection")
    if expected_selector is not None:
        benchmark_name_by_variant = {
            candidate["variant_id"]: candidate["benchmark_candidate_name"]
            for candidate in candidates
        }
        admission_selector_view = {
            "eligible_candidates": [
                candidate["benchmark_candidate_name"]
                for candidate in candidates
                if candidate["correctness"]["valid"]
                and candidate["performance"] is not None
            ],
            "eligible_ranking": [
                benchmark_name_by_variant[variant_id]
                for variant_id in selection["global_median_fps_ranking"]
            ],
            "experimental_winner": benchmark_name_by_variant[
                selection["experimental_winner_variant_id"]
            ],
            "equivalent_candidates": [
                benchmark_name_by_variant[variant_id]
                for variant_id in selection["equivalence"][
                    "candidate_variant_ids_in_preference_order"
                ]
            ],
            "preferred_candidate": benchmark_name_by_variant[
                selection["equivalence"]["preferred_variant_id"]
            ],
            "deployment_winner": benchmark_name_by_variant[
                selection["deployment_winner_variant_id"]
            ],
        }
        expected_selector_view = {
            "eligible_candidates": expected_selector["eligible_candidates"],
            "eligible_ranking": expected_selector["eligible_ranking"],
            "experimental_winner": expected_selector["experimental_winner"],
            "equivalent_candidates": expected_selector["equivalence"][
                "candidates"
            ],
            "preferred_candidate": expected_selector["equivalence"][
                "preferred_candidate"
            ],
            "deployment_winner": expected_selector["deployment_winner"],
        }
        admission_eligible = set(
            admission_selector_view["eligible_candidates"]
        )
        expected_eligible = set(expected_selector_view["eligible_candidates"])
        if not admission_eligible.issubset(expected_eligible):
            report["candidates"] = candidates
            report["errors"].append(
                "admission may not make a candidate eligible after the sealed "
                "FPS correctness prefilter"
            )
            return report, None
        # Later ABI/numerical/quality validation may conservatively remove a
        # benchmark-valid candidate and select a safe baseline.  When the
        # eligible set is unchanged, however, the independently implemented
        # admission selector must agree exactly with the producer on ranking,
        # equivalence preference, and deployment winner.
        if (
            admission_eligible == expected_eligible
            and admission_selector_view != expected_selector_view
        ):
            report["candidates"] = candidates
            report["errors"].append(
                "admission selection diverges from the sealed FPS selector "
                "result for the same correctness-valid candidates"
            )
            return report, None
    deployment = {
        "enabled": deployment_winner["execution_mode"] == "tacker",
        "valid": deployment_winner["execution_mode"] == "tacker",
    }
    provenance = {
        "generated_at_utc": report["generated_at_utc"],
        "input_sha256": dict(digests),
        "fps_benchmark_generated_at_utc": benchmark["generated_at_utc"],
        "fps_benchmark_stable_environment": benchmark["stable_environment"],
        "fps_benchmark_stable_provenance": benchmark["stable_provenance"],
        "validated_abi": {
            "rasterizer_commit": EXPECTED_RASTERIZER_COMMIT,
            "mixed_abi_version": 1,
            "mixed_abi_manifest_sha256": EXPECTED_MIXED_ABI_SHA256,
            "mixed_kernel_symbol": EXPECTED_MIXED_SYMBOL,
            "head_abi_version": 1,
            "head_solo_kernel_symbol": EXPECTED_HEAD_SOLO_SYMBOL,
            "head_gptb_kernel_symbol": EXPECTED_HEAD_GPTB_SYMBOL,
        },
    }
    common = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "workload_key": WORKLOAD_KEY,
        "selection_objective": SELECTION_OBJECTIVE,
        "selected_variant_id": selected_variant_id,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256(manifest),
        "correctness_thresholds": correctness_thresholds,
        "candidates": candidates,
        "selection": selection,
        "deployment": deployment,
        "provenance": provenance,
    }
    report.update(common)
    report["profile_sha256"] = profile_sha256(report)
    report["passed"] = True
    if not deployment["enabled"]:
        return report, None
    profile = dict(common)
    profile["profile_sha256"] = report["profile_sha256"]
    profile["note"] = (
        "Generated from correctness qualification and whole-run median FPS selection; "
        "Raster/leaf latency values are diagnostics only."
    )
    return report, profile


def evaluate_admission(inputs):
    """Auto-dispatch schema-v2 FPS selection or the legacy v1 evaluator."""

    if "fps_benchmark" in inputs or "benchmark" in inputs:
        return evaluate_admission_v2(inputs)
    return _evaluate_legacy_admission(inputs)


def atomic_write_json(path, value):
    """Atomically replace ``path`` with one fsynced, finite JSON document."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def write_admission_outputs(
    report,
    profile,
    report_path,
    profile_path,
    template_path=None,
):
    """Write a report and, only for a Tacker winner, an enabled profile.

    The disabled template is an immutable input.  ``template_path`` is
    explicit for library callers; CLI callers also carry it in
    ``report.input_paths.template`` so this function itself owns the no-clobber
    guarantee.
    """

    report_target = Path(report_path).expanduser().resolve()
    profile_target = Path(profile_path).expanduser().resolve()
    if report_target == profile_target:
        raise ValueError("report and enabled profile paths must be different")
    input_paths = report.get("input_paths") if isinstance(report, dict) else None
    if template_path is None:
        if isinstance(input_paths, dict):
            template_path = input_paths.get("template")
    protected_paths = []
    pending = []
    if isinstance(input_paths, dict):
        pending.extend(input_paths.values())
    if template_path is not None:
        pending.append(template_path)
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, (str, os.PathLike)):
            protected_paths.append(Path(value).expanduser().resolve())
    if report_target in protected_paths or profile_target in protected_paths:
        raise ValueError(
            "report/enabled profile paths may not overwrite an input artifact "
            "(including the template profile)"
        )

    # Validate canonical/finite serialization before either target changes.
    _canonical_json(report)
    if profile is not None:
        _canonical_json(profile)
    if report.get("passed") and report.get("schema_version") == PROFILE_SCHEMA_VERSION:
        if report.get("profile_sha256") != profile_sha256(report):
            raise ValueError("report selection SHA-256 mismatch")
        if report.get("manifest_sha256") != manifest_sha256(
            report.get("manifest")
        ):
            raise ValueError("report manifest SHA-256 mismatch")
    if profile is not None:
        if not report.get("passed"):
            raise ValueError("cannot write an enabled profile for a failed report")
        if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
            raise ValueError("only schema-v2 profiles may be newly written")
        if profile.get("deployment") != {"enabled": True, "valid": True}:
            raise ValueError("profile is not an enabled, valid Tacker deployment")
        if profile.get("profile_sha256") != profile_sha256(profile):
            raise ValueError("profile selection SHA-256 mismatch")
        if profile.get("profile_sha256") != report.get("profile_sha256"):
            raise ValueError("profile and report selection hashes disagree")
        if profile.get("manifest_sha256") != manifest_sha256(
            profile.get("manifest")
        ):
            raise ValueError("profile manifest SHA-256 mismatch")
        selected = next(
            (
                candidate
                for candidate in profile.get("candidates", [])
                if isinstance(candidate, dict)
                and candidate.get("variant_id") == profile.get("selected_variant_id")
            ),
            None,
        )
        if not isinstance(selected, dict) or selected.get("execution_mode") != "tacker":
            raise ValueError("enabled profile winner must use Tacker execution")
        atomic_write_json(profile_target, profile)
        report = dict(report)
        report["enabled_profile_written"] = True
        report["enabled_profile_path"] = str(profile_target)
    atomic_write_json(report_target, report)
    return report


def _load_json(path, label):
    try:
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as error:
        raise AdmissionInputError("cannot load {}: {}".format(label, error))
    return _mapping(value, label)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_hashed_json(path, label):
    """Parse and hash the same immutable byte snapshot (no load/hash TOCTOU)."""

    try:
        raw = Path(path).expanduser().read_bytes()
        value = json.loads(raw.decode("utf-8"))
        # Python's JSON decoder accepts the non-standard NaN/Infinity tokens by
        # default.  Canonicalize the complete document (not only sealed fields)
        # so candidate profiles are strict finite JSON and fail closed before
        # any manifest/profile hash helper can raise an uncaught exception.
        _canonical_json(value)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise AdmissionInputError("cannot load {}: {}".format(label, error))
    return _mapping(value, label), hashlib.sha256(raw).hexdigest()


def _candidate_profile_specs(values):
    result = {}
    for value in values or []:
        if not isinstance(value, str) or "=" not in value:
            raise AdmissionInputError(
                "--candidate-profile must use NAME=PATH"
            )
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise AdmissionInputError(
                "--candidate-profile must use non-empty NAME=PATH"
            )
        if name in result:
            raise AdmissionInputError(
                "duplicate --candidate-profile name {!r}".format(name)
            )
        result[name] = path
    return result


def _descriptor_from_candidate_profile(
    benchmark_document, candidate_name, profile_path
):
    """Load a SHA-bound current/additional Tacker runtime descriptor."""

    profile, actual_file_sha256 = _load_hashed_json(
        profile_path, "candidate profile {}".format(candidate_name)
    )
    benchmark_candidates = benchmark_document.get("candidates")
    if not isinstance(benchmark_candidates, list):
        raise AdmissionInputError("FPS benchmark candidates must be an array")
    benchmark_row = next(
        (
            row
            for row in benchmark_candidates
            if isinstance(row, dict) and row.get("name") == candidate_name
        ),
        None,
    )
    if benchmark_row is None or benchmark_row.get("execution_mode") != "tacker":
        raise AdmissionInputError(
            "candidate profile {} does not name a Tacker benchmark candidate".format(
                candidate_name
            )
        )
    expected_file_sha256 = benchmark_row.get("profile_file_sha256")
    if (
        not _is_lower_hex(expected_file_sha256, 64)
        or expected_file_sha256 != actual_file_sha256
    ):
        raise AdmissionInputError(
            "candidate profile {} file SHA-256 does not match FPS benchmark".format(
                candidate_name
            )
        )
    if profile.get("schema_version") == 1:
        if candidate_name != INCUMBENT_BENCHMARK_NAME:
            raise AdmissionInputError(
                "only current_tacker may use a schema-v1 candidate profile"
            )
        descriptor = _descriptor_from_legacy_current_profile(
            profile, "candidate profile current_tacker"
        )
        descriptor.update(
            {
                "source_profile_manifest_sha256": profile["manifest_sha256"],
                "source_profile_selection_sha256": None,
                "source_profile_deployment_enabled": True,
            }
        )
        return descriptor, actual_file_sha256
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise AdmissionInputError(
            "candidate profile {} must use schema v2 (or enabled schema v1 "
            "for current_tacker)".format(candidate_name)
        )
    _require_equal(
        profile, "workload_key", WORKLOAD_KEY, "candidate profile"
    )
    _require_equal(
        profile,
        "selection_objective",
        SELECTION_OBJECTIVE,
        "candidate profile",
    )
    manifest = _mapping(
        profile.get("manifest"), "candidate profile manifest"
    )
    if profile.get("manifest_sha256") != manifest_sha256(manifest):
        raise AdmissionInputError(
            "candidate profile {} manifest SHA-256 mismatch".format(
                candidate_name
            )
        )
    if profile.get("profile_sha256") != profile_sha256(profile):
        raise AdmissionInputError(
            "candidate profile {} selection SHA-256 mismatch".format(
                candidate_name
            )
        )
    _validate_source_profile_runtime_contract(
        profile, "candidate profile {}".format(candidate_name)
    )
    candidates = profile.get("candidates")
    if not isinstance(candidates, list):
        raise AdmissionInputError("candidate profile candidates must be an array")
    selected_variant_id = profile.get("selected_variant_id")
    if not isinstance(selected_variant_id, str) or not selected_variant_id.strip():
        raise AdmissionInputError(
            "candidate profile {} selected_variant_id must be non-empty".format(
                candidate_name
            )
        )
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("variant_id") == selected_variant_id
    ]
    if len(matches) != 1 or matches[0].get("execution_mode") != "tacker":
        raise AdmissionInputError(
            "candidate profile {} selected variant must identify exactly one "
            "Tacker descriptor".format(candidate_name)
        )
    if candidate_name == INCUMBENT_BENCHMARK_NAME:
        if profile.get("deployment") != {"enabled": True, "valid": True}:
            raise AdmissionInputError(
                "current_tacker profile must be an enabled, valid deployment"
            )
        selected_correctness = matches[0].get("correctness")
        if (
            not isinstance(selected_correctness, dict)
            or selected_correctness.get("valid") is not True
            or not isinstance(matches[0].get("performance"), dict)
        ):
            raise AdmissionInputError(
                "current_tacker profile selected candidate must have valid "
                "correctness and measured performance"
            )
        source_selection = profile.get("selection")
        if (
            not isinstance(source_selection, dict)
            or source_selection.get("deployment_winner_variant_id")
            != selected_variant_id
        ):
            raise AdmissionInputError(
                "current_tacker profile deployment winner disagrees with selection"
            )
    descriptor = dict(matches[0])
    descriptor.pop("correctness", None)
    descriptor.pop("performance", None)
    descriptor.pop("diagnostics", None)
    descriptor.pop("selection_metadata", None)
    _validate_pos_l1_descriptor(
        descriptor, "candidate profile {} descriptor".format(candidate_name)
    )
    descriptor.update(
        {
            "source_profile_schema_version": PROFILE_SCHEMA_VERSION,
            "source_profile_manifest_sha256": profile["manifest_sha256"],
            "source_profile_selection_sha256": profile["profile_sha256"],
            "source_profile_deployment_enabled": profile["deployment"][
                "enabled"
            ],
        }
    )
    return descriptor, actual_file_sha256


def _parser():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate Tacker measurements and emit a fail-closed profile"
    )
    parser.add_argument("--device-json", required=True)
    parser.add_argument("--quality-json", required=True)
    parser.add_argument("--raster-json", required=True)
    parser.add_argument("--leaf-json", required=True)
    parser.add_argument(
        "--fps-benchmark-json",
        "--benchmark-json",
        dest="fps_benchmark_json",
        help=(
            "successful 4dgaussians_tacker_fps_benchmark report; required "
            "for every new schema-v2 selection"
        ),
    )
    # Accepted only so an old command line receives the explicit read-only
    # migration error in its machine-readable report.
    parser.add_argument("--two-stream-json")
    parser.add_argument("--tacker-json")
    parser.add_argument(
        "--candidate-profile",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "repeatable candidate profile (schema-v2, or enabled schema-v1 "
            "for current_tacker); its exact file SHA-256 must match the named "
            "FPS benchmark candidate"
        ),
    )
    parser.add_argument(
        "--candidate-correctness-json",
        help="optional candidate-name to correctness-evidence JSON mapping",
    )
    parser.add_argument(
        "--mixed-abi-json",
        default=str(
            project_root
            / "submodules"
            / "depth-diff-gaussian-rasterization"
            / "abi"
            / "tacker_mixed_render_head_v1.json"
        ),
    )
    parser.add_argument(
        "--head-abi-json",
        default=str(project_root / "tacker_ext" / "abi" / "head_linear_v1.json"),
    )
    parser.add_argument(
        "--template-profile",
        default=str(project_root / "tacker_profiles" / "raster_head_sm86.json"),
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--enabled-profile", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    paths = {
        "device": args.device_json,
        "quality": args.quality_json,
        "raster": args.raster_json,
        "leaf": args.leaf_json,
        "mixed_abi": args.mixed_abi_json,
        "head_abi": args.head_abi_json,
        "template": args.template_profile,
    }
    if args.fps_benchmark_json:
        paths["fps_benchmark"] = args.fps_benchmark_json
    else:
        if args.two_stream_json:
            paths["two_stream"] = args.two_stream_json
        if args.tacker_json:
            paths["tacker"] = args.tacker_json
    candidate_profile_paths = {}
    try:
        candidate_profile_paths = _candidate_profile_specs(
            args.candidate_profile
        )
        inputs = {
            name: _load_json(path, "{} JSON".format(name))
            for name, path in paths.items()
        }
        if args.candidate_correctness_json:
            inputs["candidate_correctness"] = _load_json(
                args.candidate_correctness_json,
                "candidate correctness JSON",
            )
        if candidate_profile_paths:
            benchmark_document = inputs.get("fps_benchmark")
            if benchmark_document is None:
                raise AdmissionInputError(
                    "--candidate-profile requires --fps-benchmark-json"
                )
            descriptors = {}
            for candidate_name, candidate_path in candidate_profile_paths.items():
                descriptor, file_sha256 = _descriptor_from_candidate_profile(
                    benchmark_document, candidate_name, candidate_path
                )
                descriptor["source_profile_file_sha256"] = file_sha256
                descriptors[candidate_name] = descriptor
            inputs["candidate_descriptors"] = descriptors
        report, profile = evaluate_admission(inputs)
    except AdmissionInputError as error:
        report = _v2_base_report()
        report["errors"].append(str(error))
        profile = None
    report["input_paths"] = {
        name: str(Path(path).expanduser().resolve())
        for name, path in paths.items()
    }
    if candidate_profile_paths:
        report["input_paths"]["candidate_profiles"] = {
            name: str(Path(path).expanduser().resolve())
            for name, path in candidate_profile_paths.items()
        }
    if args.candidate_correctness_json:
        report["input_paths"]["candidate_correctness"] = str(
            Path(args.candidate_correctness_json).expanduser().resolve()
        )
    written_report = write_admission_outputs(
        report,
        profile,
        args.report,
        args.enabled_profile,
        template_path=args.template_profile,
    )
    print(
        "Tacker admission {}. Report: {}".format(
            "passed" if written_report["passed"] else "failed",
            Path(args.report).expanduser().resolve(),
        )
    )
    if written_report.get("enabled_profile_written"):
        print(
            "Enabled profile: {}".format(
                Path(args.enabled_profile).expanduser().resolve()
            )
        )
    return 0 if written_report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
