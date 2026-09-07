#!/usr/bin/env python3
"""Fail-closed admission report/profile generator for the Tacker renderer.

This script does not run a GPU benchmark.  It validates independently
collected JSON measurements, the two CUDA ABI manifests, and the sealed
runtime profile template.  Only a complete passing input set can produce an
``admission.enabled`` profile; every invocation produces a machine-readable
report.

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
import tempfile


SCHEMA_VERSION = 1
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

THRESHOLDS = {
    "raster_slowdown_pct_max": 5.0,
    "mixed_p50_strictly_less_than_solo_sum": True,
    "end_to_end_ratio_max": 1.0,
    "psnr_drop_db_max": 0.05,
    "ssim_drop_max": 0.0001,
    "lpips_increase_max": 0.0001,
}


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
    ).encode("utf-8")


def manifest_sha256(manifest):
    return hashlib.sha256(_canonical_json(manifest)).hexdigest()


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
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
    for key, locked in THRESHOLDS.items():
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


def _gate(name, passed, measured, limit, relation):
    return {
        "name": name,
        "passed": bool(passed),
        "measured": measured,
        "limit": limit,
        "relation": relation,
    }


def _base_report(input_digests=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "4dgaussians_tacker_admission_report",
        "generated_at_utc": _utc_now(),
        "passed": False,
        "workload": {
            "scene": EXPECTED_SCENE,
            "iteration": EXPECTED_ITERATION,
            "resolution": list(EXPECTED_RESOLUTION),
            "gaussian_count": EXPECTED_GAUSSIANS,
        },
        "device": None,
        "thresholds": dict(THRESHOLDS),
        "measurements": None,
        "gates": [],
        "errors": [],
        "input_sha256": input_digests or {},
        "enabled_profile_written": False,
    }


def evaluate_admission(inputs):
    """Evaluate already-loaded JSON inputs.

    Returns ``(report, enabled_profile)``.  The second element is ``None`` for
    every invalid, missing, non-finite, or gate-failing input set.
    """

    required = (
        "device",
        "quality",
        "raster",
        "leaf",
        "two_stream",
        "tacker",
        "mixed_abi",
        "head_abi",
        "template",
    )
    digests = {
        name: _source_digest(inputs[name])
        for name in required
        if name in inputs
    }
    report = _base_report(digests)
    try:
        for name in required:
            if name not in inputs:
                raise AdmissionInputError("{} input is required".format(name))
        device = _device(inputs["device"])
        device_workload_contract = _leaf_workload_contract(
            inputs["device"], "device input"
        )
        report["device"] = device
        for name in ("quality", "raster", "leaf", "two_stream", "tacker"):
            _optional_device_name(inputs[name], device["name"], name)
        _validate_mixed_abi(inputs["mixed_abi"])
        _validate_head_abi(inputs["head_abi"])
        template_manifest, admission_thresholds = _validate_template(
            inputs["template"]
        )
        quality, quality_qualification, quality_contract = _quality_measurements(
            inputs["quality"]
        )
        raster = _raster_measurements(inputs["raster"])
        leaf = _leaf_measurements(inputs["leaf"])
        two_stream_ms, two_stat, two_contract = _end_to_end(
            inputs["two_stream"], "two_stream", "two_stream input"
        )
        tacker_ms, tacker_stat, tacker_contract = _end_to_end(
            inputs["tacker"], "tacker", "tacker input"
        )
        if two_stat != tacker_stat:
            raise AdmissionInputError(
                "two_stream and tacker end-to-end statistics must match"
            )
        if two_contract != tacker_contract:
            raise AdmissionInputError(
                "two_stream and tacker benchmark contracts must match exactly"
            )
        for key in (
            "split",
            "profile_frames",
            "view_indices",
            "model_path",
            "source_path",
        ):
            if quality_contract[key] != two_contract[key]:
                raise AdmissionInputError(
                    "quality and end-to-end {} must match".format(key)
                )
        leaf_contract = leaf["workload_contract"]
        if raster["workload_contract"] != leaf_contract:
            raise AdmissionInputError(
                "raster and leaf workload provenance must match exactly"
            )
        if device_workload_contract != leaf_contract:
            raise AdmissionInputError(
                "device and leaf workload provenance must match exactly"
            )
        for key in ("split", "model_path", "source_path"):
            if leaf_contract[key] != quality_contract[key]:
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
        expected_persistent_blocks = template_manifest["persistent_blocks"]
        for label, measured in (
            ("raster", raster["persistent_blocks"]),
            ("leaf", leaf["persistent_blocks"]),
        ):
            if measured != expected_persistent_blocks:
                raise AdmissionInputError(
                    "{} persistent_blocks does not match the profile manifest".format(
                        label
                    )
                )
        tacker_persistent_blocks = inputs["tacker"].get("persistent_blocks")
        if (
            type(tacker_persistent_blocks) is not int
            or tacker_persistent_blocks != expected_persistent_blocks
        ):
            raise AdmissionInputError(
                "tacker persistent_blocks must match the profile manifest"
            )
        template_digest = inputs["template"].get("manifest_sha256")
        if inputs["tacker"].get("profile_manifest_sha256") != template_digest:
            raise AdmissionInputError(
                "tacker profile_manifest_sha256 must match the template manifest"
            )
    except (AdmissionInputError, KeyError, TypeError, ValueError) as error:
        report["gates"].append(
            _gate("input_contract", False, None, "complete", "valid")
        )
        report["errors"].append(str(error))
        return report, None

    report["gates"].append(
        _gate("input_contract", True, "complete", "complete", "valid")
    )
    report["thresholds"] = dict(admission_thresholds)
    validated_abi = {
        "rasterizer_commit": EXPECTED_RASTERIZER_COMMIT,
        "mixed_abi_version": 1,
        "mixed_kernel_symbol": EXPECTED_MIXED_SYMBOL,
        "head_abi_version": 1,
        "head_solo_kernel_symbol": EXPECTED_HEAD_SOLO_SYMBOL,
        "head_gptb_kernel_symbol": EXPECTED_HEAD_GPTB_SYMBOL,
        "physical_cta_threads": 384,
        "raster_named_barrier_id": 1,
        "raster_named_barrier_participants": 256,
    }
    report["validated_abi"] = dict(validated_abi)
    solo_sum = leaf["solo_raster_p50_ms"] + leaf["solo_head_p50_ms"]
    end_to_end_ratio = tacker_ms / two_stream_ms
    measurements = {
        "raster_slowdown_pct": raster["raster_slowdown_pct"],
        "mixed_p50_ms": leaf["mixed_p50_ms"],
        "solo_raster_p50_ms": leaf["solo_raster_p50_ms"],
        "solo_head_p50_ms": leaf["solo_head_p50_ms"],
        "tacker_end_to_end_p50_ms": tacker_ms,
        "two_stream_end_to_end_p50_ms": two_stream_ms,
        "psnr_drop_db": quality["psnr_drop_db"],
        "ssim_drop": quality["ssim_drop"],
        "lpips_increase": quality["lpips_increase"],
    }
    report["measurements"] = dict(measurements)
    report["derived"] = {
        "mixed_raster_p50_ms": raster["mixed_raster_p50_ms"],
        "solo_leaf_sum_p50_ms": solo_sum,
        "end_to_end_ratio": end_to_end_ratio,
        "end_to_end_statistic": two_stat,
        "quality_was_qualification_run": quality_qualification,
        "benchmark_contract": two_contract,
        "persistent_blocks": template_manifest["persistent_blocks"],
    }
    gates = (
        _gate(
            "raster_qos",
            measurements["raster_slowdown_pct"]
            <= admission_thresholds["raster_slowdown_pct_max"],
            measurements["raster_slowdown_pct"],
            admission_thresholds["raster_slowdown_pct_max"],
            "<=",
        ),
        _gate(
            "mixed_leaf_speedup",
            measurements["mixed_p50_ms"] < solo_sum,
            measurements["mixed_p50_ms"],
            solo_sum,
            "<",
        ),
        _gate(
            "end_to_end",
            end_to_end_ratio <= admission_thresholds["end_to_end_ratio_max"],
            end_to_end_ratio,
            admission_thresholds["end_to_end_ratio_max"],
            "<=",
        ),
        _gate(
            "psnr",
            measurements["psnr_drop_db"]
            <= admission_thresholds["psnr_drop_db_max"],
            measurements["psnr_drop_db"],
            admission_thresholds["psnr_drop_db_max"],
            "<=",
        ),
        _gate(
            "ssim",
            measurements["ssim_drop"]
            <= admission_thresholds["ssim_drop_max"],
            measurements["ssim_drop"],
            admission_thresholds["ssim_drop_max"],
            "<=",
        ),
        _gate(
            "lpips",
            measurements["lpips_increase"]
            <= admission_thresholds["lpips_increase_max"],
            measurements["lpips_increase"],
            admission_thresholds["lpips_increase_max"],
            "<=",
        ),
    )
    report["gates"].extend(gates)
    passed = all(gate["passed"] for gate in report["gates"])
    report["passed"] = passed
    if not passed:
        report["errors"].extend(
            "gate failed: {}".format(gate["name"])
            for gate in gates
            if not gate["passed"]
        )
        return report, None

    # Preserve the runtime-locked manifest byte-for-byte at the data-model
    # level. ABI details validated from separate manifests are provenance, not
    # an independent copy of the runtime contract that could drift.
    manifest = dict(template_manifest)
    profile = {
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256(manifest),
        "thresholds": dict(admission_thresholds),
        "admission": {"enabled": True, "valid": True},
        "measurements": measurements,
        "note": "Enabled only after all locked quality and performance gates passed.",
        "provenance": {
            "generated_at_utc": report["generated_at_utc"],
            "input_sha256": dict(report["input_sha256"]),
            "end_to_end_statistic": two_stat,
            "quality_was_qualification_run": quality_qualification,
            "validated_abi": validated_abi,
        },
    }
    return report, profile


def atomic_write_json(path, value):
    """Atomically replace ``path`` with one fsynced JSON document."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
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


def write_admission_outputs(report, profile, report_path, profile_path):
    """Write a report always, and an enabled profile only after full admission."""

    report_target = Path(report_path).expanduser().resolve()
    profile_target = Path(profile_path).expanduser().resolve()
    if report_target == profile_target:
        raise ValueError("report and enabled profile paths must be different")
    if profile is not None:
        if not report.get("passed"):
            raise ValueError("cannot write an enabled profile for a failed report")
        if profile.get("admission") != {"enabled": True, "valid": True}:
            raise ValueError("profile is not explicitly enabled and valid")
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


def _parser():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate Tacker measurements and emit a fail-closed profile"
    )
    parser.add_argument("--device-json", required=True)
    parser.add_argument("--quality-json", required=True)
    parser.add_argument("--raster-json", required=True)
    parser.add_argument("--leaf-json", required=True)
    parser.add_argument("--two-stream-json", required=True)
    parser.add_argument("--tacker-json", required=True)
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
        "two_stream": args.two_stream_json,
        "tacker": args.tacker_json,
        "mixed_abi": args.mixed_abi_json,
        "head_abi": args.head_abi_json,
        "template": args.template_profile,
    }
    try:
        inputs = {
            name: _load_json(path, "{} JSON".format(name))
            for name, path in paths.items()
        }
        report, profile = evaluate_admission(inputs)
    except AdmissionInputError as error:
        report = _base_report()
        report["gates"].append(
            _gate("input_contract", False, None, "complete", "valid")
        )
        report["errors"].append(str(error))
        profile = None
    report["input_paths"] = {
        name: str(Path(path).expanduser().resolve())
        for name, path in paths.items()
    }
    written_report = write_admission_outputs(
        report, profile, args.report, args.enabled_profile
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
