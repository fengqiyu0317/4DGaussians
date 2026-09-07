#!/usr/bin/env python3
"""Collect admission-grade Raster/head leaf timings on one RTX A6000.

The benchmark is intentionally tied to the first physical 4DGaussians Tacker
workload: ``flame_steak`` at iteration 14000, 111525 Gaussians, and a
1352x1014 render.  Heavy imports are delayed until :func:`run_profile`, which
keeps the schema/statistics/atomic-write helpers testable without PyTorch.

The reported mixed latency is *not* a kernel-only number.  It is the CUDA-event
latency of the public ``GaussianRasterizer.forward_with_head`` call and thus
conservatively includes the same opaque Raster prefix as the legacy full
Raster call plus the physical mixed render/head leaf.  The same mixed p50 is
used for ``mixed_raster_p50_ms`` and ``mixed_p50_ms``; the same legacy full
Raster p50 is used in both admission documents.
"""

from __future__ import print_function

from argparse import ArgumentParser
import datetime
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile


SCHEMA_VERSION = 1
EXPECTED_SCENE = "flame_steak"
EXPECTED_SOURCE_BASENAMES = (
    "flame_steak",
    "flame_steak_4dgs_min",
)
EXPECTED_ITERATION = 14000
EXPECTED_GAUSSIANS = 111525
EXPECTED_RESOLUTION = [1352, 1014]
EXPECTED_GPU_NAME = "NVIDIA RTX A6000"
EXPECTED_COMPUTE_CAPABILITY = [8, 6]
EXPECTED_CUDA_ARCH = "sm_86"
EXPECTED_RASTERIZER_COMMIT = "e49506654e8e11ed8a62d22bcb693e943fdecacf"
EXPECTED_MIXED_SYMBOL = "tacker_mix_render_head_v1"
EXPECTED_HEAD_SOLO_SYMBOL = "tacker_head_linear_solo_v1"
EXPECTED_HEAD_GPTB_SYMBOL = "tacker_head_linear_gptb_v1"

EXPECTED_CAPABILITIES = {
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

NUMERICAL_THRESHOLDS = {
    "raster_color_max_abs": 1.0e-5,
    "raster_depth_max_abs": 1.0e-5,
    "raster_radii_mismatch_count": 0,
    "head_kernel_atol": 2.0e-3,
    "head_kernel_rtol": 2.0e-3,
}

PROJECT_ROOT = Path(__file__).resolve().parent
HEAD_PACKAGE_ROOT = PROJECT_ROOT / "tacker_ext"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HEAD_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(HEAD_PACKAGE_ROOT))


class ProfileContractError(RuntimeError):
    """A fixed workload, extension, numerical, or output contract failed."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _percentile(samples, fraction):
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return (
        ordered[lower_index] * (1.0 - weight)
        + ordered[upper_index] * weight
    )


def summarize_samples(samples):
    """Validate and summarize positive CUDA-event samples in milliseconds."""

    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("timing samples must be a non-empty list")
    values = []
    for value in samples:
        if not _is_finite_number(value):
            raise ValueError("timing samples must all be finite numbers")
        value = float(value)
        if value <= 0.0:
            raise ValueError("timing samples must all be greater than zero")
        values.append(value)
    return {
        "sample_count": len(values),
        "samples_ms": values,
        "p50_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.mean(values)),
        "p95_ms": float(_percentile(values, 0.95)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def atomic_write_json(path, value):
    """Atomically replace one JSON file and reject NaN/Infinity."""

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


def _resolved_output_paths(args):
    outputs = {
        "device": Path(args.device_output).expanduser().resolve(),
        "raster": Path(args.raster_output).expanduser().resolve(),
        "leaf": Path(args.leaf_output).expanduser().resolve(),
        "report": Path(args.report).expanduser().resolve(),
    }
    if len(set(str(path) for path in outputs.values())) != len(outputs):
        raise ValueError("all four output paths must be distinct")
    return outputs


def write_profile_outputs(device, raster, leaf, report, args):
    """Write measurements only on success, but always write the report.

    Existing device/raster/leaf files are intentionally left untouched when
    ``report.passed`` is false.  This prevents a failed numerical run from
    replacing the last admissible measurement set.
    """

    outputs = _resolved_output_paths(args)
    written_report = dict(report)
    passed = report.get("passed") is True
    if passed:
        for label, document in (
            ("device", device),
            ("raster", raster),
            ("leaf", leaf),
        ):
            if not isinstance(document, dict):
                raise ValueError(
                    "passing report requires a {} document".format(label)
                )
        # Validate the complete set before replacing any existing output.
        for document in (device, raster, leaf, written_report):
            json.dumps(document, allow_nan=False)
        atomic_write_json(outputs["device"], device)
        atomic_write_json(outputs["raster"], raster)
        atomic_write_json(outputs["leaf"], leaf)
        written_report["measurement_outputs_written"] = True
    else:
        written_report["measurement_outputs_written"] = False
    written_report["output_paths"] = {
        key: str(path) for key, path in outputs.items()
    }
    atomic_write_json(outputs["report"], written_report)
    return written_report


def _workload_document(args=None):
    return {
        "scene": EXPECTED_SCENE,
        "iteration": EXPECTED_ITERATION,
        "split": getattr(args, "split", None),
        "resolution": list(EXPECTED_RESOLUTION),
        "gaussian_count": EXPECTED_GAUSSIANS,
    }


def _minimal_failure(error, args=None):
    details = getattr(error, "details", None)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "4dgaussians_tacker_leaf_profile_report",
        "generated_at_utc": _utc_now(),
        "passed": False,
        "workload": _workload_document(args),
        "device": None,
        "numerics": None,
        "errors": [str(error)],
        "measurement_outputs_written": False,
    }
    if isinstance(details, dict):
        report.update(details)
    return report


def _common_measurement_fields(workload, device):
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "passed": True,
        "workload": dict(workload),
        "device": dict(device),
    }


def build_measurement_documents(
    workload,
    device,
    extensions,
    numerics,
    solo_raster,
    mixed_full,
    solo_head,
    gptb_head=None,
    sample_view_indices=None,
    persistent_blocks=0,
):
    """Build the exact three JSON inputs consumed by admission validation."""

    for label, summary in (
        ("solo_raster", solo_raster),
        ("mixed_full", mixed_full),
        ("solo_head", solo_head),
    ):
        if not isinstance(summary, dict) or not _is_finite_number(
            summary.get("p50_ms")
        ):
            raise ValueError("{} summary requires finite p50_ms".format(label))
        if float(summary["p50_ms"]) <= 0.0:
            raise ValueError("{} p50_ms must be > 0".format(label))

    if type(persistent_blocks) is not int or persistent_blocks < 0:
        raise ValueError("persistent_blocks must be an int >= 0")
    measurement_config = {"persistent_blocks": persistent_blocks}
    common = _common_measurement_fields(workload, device)
    device_document = dict(common)
    device_document.update(
        {
            "kind": "4dgaussians_tacker_device",
            "device": dict(device),
            "extensions": extensions,
            "measurement_config": dict(measurement_config),
        }
    )

    solo_raster_p50 = float(solo_raster["p50_ms"])
    mixed_full_p50 = float(mixed_full["p50_ms"])
    solo_head_p50 = float(solo_head["p50_ms"])
    semantics = {
        "solo_raster": (
            "CUDA-event latency of the complete legacy rasterize_state call, "
            "including the exact opaque Raster prefix and render leaf"
        ),
        "mixed_full": (
            "CUDA-event latency of the complete public forward_with_head call; "
            "this is a conservative LC completion latency containing the same "
            "opaque Raster prefix plus the physical mixed render/head leaf"
        ),
        "solo_head": (
            "CUDA-event latency of head_linear_solo on the real next-frame "
            "pos_deform[1] FP16 input/weight and FP32 bias"
        ),
        "kernel_only": False,
        "known_raster_synchronization": (
            "The exact rasterizer internally synchronizes its current stream "
            "once to retrieve the opaque-prefix rendered-count; the profiler "
            "adds no per-iteration device synchronization."
        ),
    }

    raster_document = dict(common)
    raster_document.update(
        {
            "kind": "4dgaussians_tacker_raster_profile",
            "measurements": {
                "solo_raster_p50_ms": solo_raster_p50,
                "mixed_raster_p50_ms": mixed_full_p50,
            },
            "timings": {
                "legacy_full_raster": solo_raster,
                "mixed_full_raster_head": mixed_full,
            },
            "measurement_semantics": semantics,
            "sample_view_indices": list(sample_view_indices or []),
            "numerics": numerics,
            "measurement_config": dict(measurement_config),
        }
    )

    leaf_measurements = {
        "mixed_p50_ms": mixed_full_p50,
        "solo_raster_p50_ms": solo_raster_p50,
        "solo_head_p50_ms": solo_head_p50,
    }
    leaf_timings = {
        "mixed_full_raster_head": mixed_full,
        "legacy_full_raster": solo_raster,
        "real_pos_head_solo": solo_head,
    }
    if gptb_head is not None:
        if not isinstance(gptb_head, dict) or not _is_finite_number(
            gptb_head.get("p50_ms")
        ) or float(gptb_head["p50_ms"]) <= 0.0:
            raise ValueError("gptb_head summary requires finite positive p50_ms")
        leaf_timings["real_pos_head_gptb_diagnostic"] = gptb_head
        leaf_measurements["gptb_head_p50_ms_diagnostic"] = float(
            gptb_head["p50_ms"]
        )
    leaf_document = dict(common)
    leaf_document.update(
        {
            "kind": "4dgaussians_tacker_leaf_profile",
            "measurements": leaf_measurements,
            "timings": leaf_timings,
            "measurement_semantics": semantics,
            "sample_view_indices": list(sample_view_indices or []),
            "numerics": numerics,
            "measurement_config": dict(measurement_config),
        }
    )
    return device_document, raster_document, leaf_document


def _select_views(scene, split):
    if split == "train":
        return scene.getTrainCameras()
    if split == "test":
        return scene.getTestCameras()
    return scene.getVideoCameras()


def _selected_pairs(total, start, stride, view_count):
    if view_count < 2:
        raise ValueError("--views must be at least 2")
    if start < 0:
        raise ValueError("--view-start must be non-negative")
    if stride <= 0:
        raise ValueError("--view-stride must be positive")
    sequence = [start + offset * stride for offset in range(view_count + 1)]
    if sequence[-1] >= total:
        raise ValueError(
            "fixed view sequence ends at index {} but split has {} views".format(
                sequence[-1], total
            )
        )
    return list(zip(sequence[:-1], sequence[1:]))


def _load_json(path, label):
    try:
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as error:
        raise ProfileContractError("cannot load {}: {}".format(label, error))
    if not isinstance(value, dict):
        raise ProfileContractError("{} must be a JSON object".format(label))
    return value


def _validate_abi_manifests(mixed, head):
    mixed_expected = {
        "abi_version": 1,
        "rasterizer_upstream_commit": EXPECTED_RASTERIZER_COMMIT,
        "cuda_arch": EXPECTED_CUDA_ARCH,
        "global_kernel_symbol": EXPECTED_MIXED_SYMBOL,
        "python_binding": "rasterize_gaussians_with_head",
        "python_method": "GaussianRasterizer.forward_with_head",
        "capability_query": "tacker_capabilities",
    }
    for key, expected in mixed_expected.items():
        if mixed.get(key) != expected:
            raise ProfileContractError(
                "mixed ABI {} must be {!r}".format(key, expected)
            )
    launch = mixed.get("physical_launch")
    subgroups = mixed.get("subgroups")
    if not isinstance(launch, dict) or launch.get("threads") != 384:
        raise ProfileContractError("mixed ABI must launch 384 threads")
    if not isinstance(subgroups, dict):
        raise ProfileContractError("mixed ABI subgroups are missing")
    raster = subgroups.get("raster")
    head_group = subgroups.get("head")
    if not isinstance(raster, dict) or not isinstance(head_group, dict):
        raise ProfileContractError("mixed ABI Raster/head subgroups are missing")
    if (
        raster.get("thread_range") != [0, 255]
        or raster.get("threads") != 256
        or raster.get("named_barrier_id") != 1
        or raster.get("named_barrier_participants") != 256
    ):
        raise ProfileContractError("mixed ABI Raster thread/barrier layout changed")
    if (
        head_group.get("thread_range") != [256, 383]
        or head_group.get("threads") != 128
        or head_group.get("named_barrier_ids") != []
    ):
        raise ProfileContractError("mixed ABI head thread/barrier layout changed")
    for name, dtype in (
        ("input_dtype", "float16"),
        ("weight_dtype", "float16"),
        ("bias_dtype", "float32"),
        ("accumulation_dtype", "float32"),
        ("output_dtype", "float32"),
    ):
        if head_group.get(name) != dtype:
            raise ProfileContractError(
                "mixed ABI head {} must be {}".format(name, dtype)
            )

    if head.get("abi_version") != 1 or head.get("cuda_arch") != EXPECTED_CUDA_ARCH:
        raise ProfileContractError("head ABI must be version 1 for sm_86")
    if head.get("capability_query") != "tacker_capabilities":
        raise ProfileContractError(
            "head ABI capability_query must be tacker_capabilities"
        )
    logical = head.get("logical_launch")
    symbols = head.get("global_kernel_symbols")
    if not isinstance(logical, dict) or logical.get("block_threads") != 128:
        raise ProfileContractError("head ABI block_threads must be 128")
    if not isinstance(symbols, dict):
        raise ProfileContractError("head ABI global symbols are missing")
    if not isinstance(symbols.get("solo"), dict) or symbols["solo"].get(
        "symbol"
    ) != EXPECTED_HEAD_SOLO_SYMBOL:
        raise ProfileContractError("head ABI solo global symbol changed")
    if not isinstance(symbols.get("gptb"), dict) or symbols["gptb"].get(
        "symbol"
    ) != EXPECTED_HEAD_GPTB_SYMBOL:
        raise ProfileContractError("head ABI GPTB global symbol changed")


def _require_python_symbols(module, module_label, symbols):
    missing = [name for name in symbols if not callable(getattr(module, name, None))]
    if missing:
        raise ProfileContractError(
            "{} is missing callable symbols: {}".format(
                module_label, ", ".join(missing)
            )
        )
    return list(symbols)


def _validate_extensions(rasterizer_module, head_backend, mixed_abi, head_abi):
    _validate_abi_manifests(mixed_abi, head_abi)
    provider = getattr(rasterizer_module, "tacker_capabilities", None)
    if not callable(provider):
        raise ProfileContractError("rasterizer has no tacker_capabilities call")
    capabilities = dict(provider())
    for key, expected in EXPECTED_CAPABILITIES.items():
        if capabilities.get(key) != expected:
            raise ProfileContractError(
                "rasterizer capability {} must be {!r}, got {!r}".format(
                    key, expected, capabilities.get(key)
                )
            )
    raster_backend = getattr(rasterizer_module, "_C", None)
    if raster_backend is None:
        raise ProfileContractError("rasterizer compiled backend is unavailable")
    raster_python_symbols = _require_python_symbols(
        raster_backend,
        "rasterizer extension",
        ("rasterize_gaussians", "rasterize_gaussians_with_head", "tacker_capabilities"),
    )
    head_python_symbols = _require_python_symbols(
        head_backend,
        "head extension",
        (
            "head_linear_solo",
            "head_linear_solo_out",
            "head_linear_gptb",
            "head_linear_gptb_out",
            "tacker_capabilities",
        ),
    )
    head_capabilities = dict(head_backend.tacker_capabilities())
    for key, expected in EXPECTED_HEAD_CAPABILITIES.items():
        if head_capabilities.get(key) != expected:
            raise ProfileContractError(
                "head capability {} must be {!r}, got {!r}".format(
                    key, expected, head_capabilities.get(key)
                )
            )
    return {
        "rasterizer": {
            "path": str(Path(raster_backend.__file__).resolve()),
            "capabilities": capabilities,
            "python_symbols": raster_python_symbols,
            "cuda_global_symbols": [EXPECTED_MIXED_SYMBOL],
        },
        "head": {
            "path": str(Path(head_backend.__file__).resolve()),
            "capabilities": head_capabilities,
            "python_symbols": head_python_symbols,
            "cuda_global_symbols": [
                EXPECTED_HEAD_SOLO_SYMBOL,
                EXPECTED_HEAD_GPTB_SYMBOL,
            ],
        },
        "abi": {
            "mixed_manifest": str(Path(mixed_abi["_path"]).resolve()),
            "head_manifest": str(Path(head_abi["_path"]).resolve()),
        },
    }


def _tensor_error_metrics(actual, reference, torch):
    if tuple(actual.shape) != tuple(reference.shape):
        raise ProfileContractError(
            "numerical tensor shapes differ: {} vs {}".format(
                tuple(actual.shape), tuple(reference.shape)
            )
        )
    actual_float = actual.float()
    reference_float = reference.float()
    if not bool(torch.isfinite(actual_float).all().item()):
        raise ProfileContractError("numerical output contains non-finite values")
    if not bool(torch.isfinite(reference_float).all().item()):
        raise ProfileContractError("numerical reference contains non-finite values")
    difference = actual_float - reference_float
    absolute = difference.abs()
    reference_norm = torch.linalg.vector_norm(reference_float)
    return {
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(difference * difference)).item()),
        "relative_l2": float(
            (
                torch.linalg.vector_norm(difference)
                / torch.clamp(reference_norm, min=1.0e-12)
            ).item()
        ),
    }


def _fp32_linear_reference(input_tensor, weight, bias, torch):
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.mm(input_tensor.float(), weight.float().t()) + bias.float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _mixed_call(bundle, persistent_blocks):
    context = bundle["context"]
    state = bundle["state"]
    task = bundle["task"]
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


def _check_numerics(bundles, rasterize_state, head_linear_solo, torch, persistent_blocks):
    per_pair = []
    failures = []
    network = bundles[0]["pc"]._deformation.deformation_net
    selected_linear = network.pos_deform[1]
    for bundle in bundles:
        legacy = rasterize_state(bundle["context"], bundle["state"])
        mixed_color, mixed_radii, mixed_depth, mixed_head = _mixed_call(
            bundle, persistent_blocks
        )
        task = bundle["task"]
        solo_head = head_linear_solo(
            task.head_input, task.head_weight, task.head_bias
        )
        quantized_reference = _fp32_linear_reference(
            task.head_input, task.head_weight, task.head_bias, torch
        )
        original_input = network.pos_deform[0](task.hidden).float()
        original_reference = _fp32_linear_reference(
            original_input,
            selected_linear.weight.detach(),
            selected_linear.bias.detach(),
            torch,
        )
        torch.cuda.synchronize()

        color = _tensor_error_metrics(mixed_color, legacy.render, torch)
        depth = _tensor_error_metrics(mixed_depth, legacy.depth, torch)
        if tuple(mixed_radii.shape) != tuple(legacy.radii.shape):
            raise ProfileContractError("legacy/mixed radii shapes differ")
        radii_mismatches = int((mixed_radii != legacy.radii).sum().item())
        solo_vs_mixed = _tensor_error_metrics(solo_head, mixed_head, torch)
        solo_vs_quantized = _tensor_error_metrics(
            solo_head, quantized_reference, torch
        )
        mixed_vs_quantized = _tensor_error_metrics(
            mixed_head, quantized_reference, torch
        )
        quantization = _tensor_error_metrics(
            quantized_reference, original_reference, torch
        )

        head_atol = NUMERICAL_THRESHOLDS["head_kernel_atol"]
        head_rtol = NUMERICAL_THRESHOLDS["head_kernel_rtol"]
        checks = {
            "legacy_vs_mixed_color": (
                color["max_abs"]
                <= NUMERICAL_THRESHOLDS["raster_color_max_abs"]
            ),
            "legacy_vs_mixed_depth": (
                depth["max_abs"]
                <= NUMERICAL_THRESHOLDS["raster_depth_max_abs"]
            ),
            "legacy_vs_mixed_radii": (
                radii_mismatches
                <= NUMERICAL_THRESHOLDS["raster_radii_mismatch_count"]
            ),
            "solo_vs_mixed_head": bool(
                torch.allclose(solo_head, mixed_head, rtol=head_rtol, atol=head_atol)
            ),
            "solo_vs_quantized_fp32_reference": bool(
                torch.allclose(
                    solo_head,
                    quantized_reference,
                    rtol=head_rtol,
                    atol=head_atol,
                )
            ),
            "mixed_vs_quantized_fp32_reference": bool(
                torch.allclose(
                    mixed_head,
                    quantized_reference,
                    rtol=head_rtol,
                    atol=head_atol,
                )
            ),
        }
        pair_report = {
            "current_view_index": bundle["current_index"],
            "next_view_index": bundle["next_index"],
            "passed": all(checks.values()),
            "checks": checks,
            "legacy_vs_mixed": {
                "color": color,
                "depth": depth,
                "radii_mismatch_count": radii_mismatches,
            },
            "head": {
                "kernel_reference": (
                    "real FP16 input/weight converted to FP32, FP32 bias, "
                    "TF32 disabled"
                ),
                "solo_vs_mixed": solo_vs_mixed,
                "solo_vs_quantized_fp32_reference": solo_vs_quantized,
                "mixed_vs_quantized_fp32_reference": mixed_vs_quantized,
                "fp16_operand_quantization_vs_original_fp32": quantization,
            },
        }
        if not pair_report["passed"]:
            failures.append(
                "numerical gate failed for view pair {}->{}".format(
                    bundle["current_index"], bundle["next_index"]
                )
            )
        per_pair.append(pair_report)

    report = {
        "passed": not failures,
        "thresholds": dict(NUMERICAL_THRESHOLDS),
        "reference_policy": (
            "Kernel accuracy is judged only against FP16 operands evaluated "
            "with FP32 math and TF32 disabled; original-FP32 error is reported "
            "separately as quantization diagnostics."
        ),
        "per_view_pair": per_pair,
        "errors": failures,
    }
    if failures:
        raise ProfileContractError(
            "; ".join(failures), details={"numerics": report}
        )
    return report


def _time_cuda_calls(functions, warmup, repetitions, torch):
    if warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if not functions:
        raise ValueError("at least one fixed view callable is required")

    stream = torch.cuda.current_stream()
    last_output = None
    for _ in range(warmup):
        for function in functions:
            last_output = function()
    warmup_done = torch.cuda.Event(blocking=False)
    warmup_done.record(stream)
    warmup_done.synchronize()

    sample_count = repetitions * len(functions)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(sample_count)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(sample_count)]
    sample_slots = []
    sample_index = 0
    for _ in range(repetitions):
        for slot, function in enumerate(functions):
            starts[sample_index].record(stream)
            last_output = function()
            ends[sample_index].record(stream)
            sample_slots.append(slot)
            sample_index += 1
    ends[-1].synchronize()
    samples = [
        float(start.elapsed_time(end)) for start, end in zip(starts, ends)
    ]
    # Keep the final allocating-call output live through the terminal event.
    if last_output is None:
        raise RuntimeError("timing callable did not execute")
    summary = summarize_samples(samples)
    summary["warmup_rounds"] = warmup
    summary["repetitions_per_view"] = repetitions
    summary["view_slot_for_each_sample"] = sample_slots
    summary["timing_method"] = "CUDA events on the PyTorch current stream"
    summary["per_iteration_device_synchronize"] = False
    return summary


def run_profile(args, dataset, hyperparam, pipeline):
    """Load the fixed scene, validate numerics, and collect CUDA-event data."""

    import torch
    import diff_gaussian_rasterization as rasterizer_module

    from gaussian_renderer import (
        GaussianModel,
        deform_for_render,
        prepare_render_context,
        rasterize_state,
    )
    from gaussian_renderer.tacker_pipeline import prepare_pos_head_task
    from scene import Scene

    head_api = importlib.import_module("tacker_4dgs_head")
    head_backend = importlib.import_module("tacker_4dgs_head._C")
    head_linear_solo = head_api.head_linear_solo
    head_linear_gptb = head_api.head_linear_gptb

    if args.scene_name.replace("-", "_").lower() != EXPECTED_SCENE:
        raise ProfileContractError("--scene-name must be flame_steak")
    if args.iteration != EXPECTED_ITERATION:
        raise ProfileContractError("--iteration must be 14000")
    if args.persistent_blocks < 0:
        raise ProfileContractError("--persistent-blocks must be >= 0")
    if not torch.cuda.is_available():
        raise ProfileContractError(
            "CUDA is unavailable; run this benchmark on the 4A6000 server"
        )
    torch.cuda.set_device(args.gpu)
    gpu_name = torch.cuda.get_device_name(args.gpu)
    capability = list(torch.cuda.get_device_capability(args.gpu))
    if gpu_name != EXPECTED_GPU_NAME:
        raise ProfileContractError(
            "GPU name must be exactly {!r}, got {!r}".format(
                EXPECTED_GPU_NAME, gpu_name
            )
        )
    if capability != EXPECTED_COMPUTE_CAPABILITY:
        raise ProfileContractError(
            "GPU compute capability must be [8, 6], got {}".format(capability)
        )

    mixed_abi = _load_json(args.mixed_abi, "mixed ABI")
    mixed_abi["_path"] = str(Path(args.mixed_abi).expanduser().resolve())
    head_abi = _load_json(args.head_abi, "head ABI")
    head_abi["_path"] = str(Path(args.head_abi).expanduser().resolve())
    extensions = _validate_extensions(
        rasterizer_module, head_backend, mixed_abi, head_abi
    )

    if bool(getattr(pipeline, "debug", False)):
        raise ProfileContractError("pipeline.debug must be false")
    if bool(getattr(pipeline, "compute_cov3D_python", False)):
        raise ProfileContractError("compute_cov3D_python must be false")
    if bool(getattr(pipeline, "convert_SHs_python", False)):
        raise ProfileContractError("convert_SHs_python must be false")

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=args.iteration,
            shuffle=False,
        )
        gaussians._deformation.eval()
        deformation_network = gaussians._deformation.deformation_net
        deformation_args = getattr(deformation_network, "args", None)
        if bool(getattr(gaussians._deformation, "training", True)):
            raise ProfileContractError("deformation network must be in eval mode")
        if getattr(deformation_network, "W", None) != 128:
            raise ProfileContractError("deformation width W must be 128")
        if getattr(deformation_network, "D", None) != 0:
            raise ProfileContractError("deformation depth D must be 0")
        if deformation_args is None or bool(getattr(deformation_args, "no_dx", True)):
            raise ProfileContractError("the real positional head requires no_dx=false")
        try:
            selected_head = deformation_network.pos_deform[1]
        except (AttributeError, IndexError, TypeError):
            raise ProfileContractError("pos_deform[1] is unavailable")
        selected_weight = getattr(selected_head, "weight", None)
        selected_bias = getattr(selected_head, "bias", None)
        if (
            tuple(getattr(selected_weight, "shape", ())) != (128, 128)
            or tuple(getattr(selected_bias, "shape", ())) != (128,)
        ):
            raise ProfileContractError(
                "pos_deform[1] must be a biased Linear(128, 128)"
            )
        if scene.loaded_iter != EXPECTED_ITERATION:
            raise ProfileContractError("loaded checkpoint is not iteration 14000")
        if scene.dataset_type != "dynerf":
            raise ProfileContractError("flame_steak must use the dynerf loader")
        source_name = Path(dataset.source_path).expanduser().resolve().name
        normalized_source_name = source_name.replace("-", "_").lower()
        if normalized_source_name not in EXPECTED_SOURCE_BASENAMES:
            raise ProfileContractError(
                "dataset source basename must be one of {}, got {!r}".format(
                    ", ".join(EXPECTED_SOURCE_BASENAMES), source_name
                )
            )
        gaussian_count = int(gaussians.get_xyz.shape[0])
        if gaussian_count != EXPECTED_GAUSSIANS:
            raise ProfileContractError(
                "Gaussian count must be {}, got {}".format(
                    EXPECTED_GAUSSIANS, gaussian_count
                )
            )

        all_views = _select_views(scene, args.split)
        pairs = _selected_pairs(
            len(all_views), args.view_start, args.view_stride, args.views
        )
        used_indices = sorted(set(index for pair in pairs for index in pair))
        for index in used_indices:
            view = all_views[index]
            resolution = [int(view.image_width), int(view.image_height)]
            if resolution != EXPECTED_RESOLUTION:
                raise ProfileContractError(
                    "view {} resolution must be {}, got {}".format(
                        index, EXPECTED_RESOLUTION, resolution
                    )
                )

        background = torch.tensor(
            [1, 1, 1] if dataset.white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
        setup_stream = torch.cuda.current_stream()
        cached_weight = None
        cached_bias = None
        bundles = []
        for current_index, next_index in pairs:
            current_context = prepare_render_context(
                all_views[current_index],
                gaussians,
                pipeline,
                background,
                cam_type=scene.dataset_type,
            )
            current_state = deform_for_render(
                current_context, gaussians, stage="fine"
            )
            next_context = prepare_render_context(
                all_views[next_index],
                gaussians,
                pipeline,
                background,
                cam_type=scene.dataset_type,
            )
            task = prepare_pos_head_task(
                next_context,
                gaussians,
                setup_stream,
                head_weight=cached_weight,
                head_bias=cached_bias,
            )
            cached_weight = task.head_weight
            cached_bias = task.head_bias
            if tuple(task.head_input.shape) != (EXPECTED_GAUSSIANS, 128):
                raise ProfileContractError(
                    "real pos_deform[1] input must have shape [111525, 128]"
                )
            if task.head_input.dtype != torch.float16:
                raise ProfileContractError("real head input must be float16")
            if task.head_weight.dtype != torch.float16:
                raise ProfileContractError("cached head weight must be float16")
            if task.head_bias.dtype != torch.float32:
                raise ProfileContractError("cached head bias must be float32")
            bundles.append(
                {
                    "pc": gaussians,
                    "current_index": current_index,
                    "next_index": next_index,
                    "context": current_context,
                    "state": current_state,
                    "task": task,
                }
            )

        # Context/state/task construction, parameter conversion, and all
        # numerical references are setup.  Synchronize them before warmup.
        torch.cuda.synchronize()
        try:
            numerics = _check_numerics(
                bundles,
                rasterize_state,
                head_linear_solo,
                torch,
                args.persistent_blocks,
            )
        except ProfileContractError as error:
            if error.details is None:
                error.details = {
                    "numerics": {
                        "passed": False,
                        "thresholds": dict(NUMERICAL_THRESHOLDS),
                        "errors": [str(error)],
                    }
                }
            raise
        torch.cuda.synchronize()

        legacy_functions = [
            (lambda bundle=bundle: rasterize_state(bundle["context"], bundle["state"]))
            for bundle in bundles
        ]
        mixed_functions = [
            (lambda bundle=bundle: _mixed_call(bundle, args.persistent_blocks))
            for bundle in bundles
        ]
        solo_head_functions = [
            (
                lambda bundle=bundle: head_linear_solo(
                    bundle["task"].head_input,
                    bundle["task"].head_weight,
                    bundle["task"].head_bias,
                )
            )
            for bundle in bundles
        ]

        solo_head_summary = _time_cuda_calls(
            solo_head_functions, args.warmup, args.repetitions, torch
        )
        solo_raster_summary = _time_cuda_calls(
            legacy_functions, args.warmup, args.repetitions, torch
        )
        mixed_summary = _time_cuda_calls(
            mixed_functions, args.warmup, args.repetitions, torch
        )
        gptb_summary = None
        if args.profile_gptb:
            gptb_functions = [
                (
                    lambda bundle=bundle: head_linear_gptb(
                        bundle["task"].head_input,
                        bundle["task"].head_weight,
                        bundle["task"].head_bias,
                        args.persistent_blocks,
                    )
                )
                for bundle in bundles
            ]
            gptb_summary = _time_cuda_calls(
                gptb_functions, args.warmup, args.repetitions, torch
            )

    workload = {
        "scene": EXPECTED_SCENE,
        "iteration": scene.loaded_iter,
        "split": args.split,
        "current_view_indices": [pair[0] for pair in pairs],
        "next_view_indices": [pair[1] for pair in pairs],
        "resolution": list(EXPECTED_RESOLUTION),
        "gaussian_count": gaussian_count,
        "model_path": str(Path(dataset.model_path).expanduser().resolve()),
        "source_path": str(Path(dataset.source_path).expanduser().resolve()),
    }
    device = {
        "name": gpu_name,
        "index": args.gpu,
        "compute_capability": capability,
        "cuda_arch": EXPECTED_CUDA_ARCH,
        "cuda_runtime": torch.version.cuda,
        "pytorch_version": str(torch.__version__),
    }
    sample_view_indices = []
    for _ in range(args.repetitions):
        sample_view_indices.extend(pair[0] for pair in pairs)
    device_document, raster_document, leaf_document = build_measurement_documents(
        workload,
        device,
        extensions,
        numerics,
        solo_raster_summary,
        mixed_summary,
        solo_head_summary,
        gptb_head=gptb_summary,
        sample_view_indices=sample_view_indices,
        persistent_blocks=args.persistent_blocks,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "4dgaussians_tacker_leaf_profile_report",
        "generated_at_utc": _utc_now(),
        "passed": True,
        "workload": workload,
        "device": device,
        "extensions": extensions,
        "numerics": numerics,
        "parameters": {
            "warmup_rounds": args.warmup,
            "repetitions_per_view": args.repetitions,
            "persistent_blocks": args.persistent_blocks,
            "profile_gptb": bool(args.profile_gptb),
        },
        "measurements": {
            "solo_raster_p50_ms": solo_raster_summary["p50_ms"],
            "mixed_raster_p50_ms": mixed_summary["p50_ms"],
            "mixed_p50_ms": mixed_summary["p50_ms"],
            "solo_head_p50_ms": solo_head_summary["p50_ms"],
        },
        "errors": [],
    }
    return device_document, raster_document, leaf_document, report


def _build_parser():
    from arguments import ModelHiddenParams, ModelParams, PipelineParams

    parser = ArgumentParser(
        description=(
            "Profile the fixed 4DGaussians Tacker Raster/head leaves on one "
            "NVIDIA RTX A6000"
        )
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", type=int, default=EXPECTED_ITERATION)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--scene-name", default=EXPECTED_SCENE)
    parser.add_argument(
        "--split", choices=("train", "test", "video"), default="test"
    )
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--view-start", type=int, default=0)
    parser.add_argument("--view-stride", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--persistent-blocks", type=int, default=0)
    parser.add_argument("--profile-gptb", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--mixed-abi",
        default=str(
            PROJECT_ROOT
            / "submodules"
            / "depth-diff-gaussian-rasterization"
            / "abi"
            / "tacker_mixed_render_head_v1.json"
        ),
    )
    parser.add_argument(
        "--head-abi",
        default=str(PROJECT_ROOT / "tacker_ext" / "abi" / "head_linear_v1.json"),
    )
    parser.add_argument("--device-output", required=True)
    parser.add_argument("--raster-output", required=True)
    parser.add_argument("--leaf-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser, model, hyperparam, pipeline


def main():
    parser, model, hyperparam, pipeline = _build_parser()
    from arguments import get_combined_args

    args = get_combined_args(parser)
    device_document = None
    raster_document = None
    leaf_document = None
    try:
        from utils.general_utils import safe_state
        from utils.params_utils import load_config, merge_hparams

        if args.configs:
            args = merge_hparams(args, load_config(args.configs))
        safe_state(args.quiet)
        device_document, raster_document, leaf_document, report = run_profile(
            args,
            model.extract(args),
            hyperparam.extract(args),
            pipeline.extract(args),
        )
    except Exception as error:
        report = _minimal_failure(error, args=args)

    written_report = write_profile_outputs(
        device_document,
        raster_document,
        leaf_document,
        report,
        args,
    )
    print(
        "Tacker leaf profile {}. Report: {}".format(
            "passed" if written_report["passed"] else "failed",
            Path(args.report).expanduser().resolve(),
        )
    )
    for error in written_report.get("errors", []):
        print("- {}".format(error))
    return 0 if written_report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
