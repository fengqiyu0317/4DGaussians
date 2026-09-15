#!/usr/bin/env python3
"""Collect admission-grade Raster/head leaf timings on one RTX A6000.

The benchmark is intentionally tied to the first physical 4DGaussians Tacker
workload: ``flame_steak`` at iteration 14000, 111525 Gaussians, and a
1352x1014 render.  Heavy imports are delayed until :func:`run_profile`, which
keeps the schema/statistics/atomic-write helpers testable without PyTorch.

Without ``--candidate-profile`` the profiler preserves its original schema-v1
positional-head CLI and documents.  An explicit, validated schema-v2 profile
may select the descriptor-based first-linear, packed first-linear, or
whole-head production partition.  It is run as an offline qualification
diagnostic even when that source profile is deployment-disabled.

The reported mixed latency is *not* a kernel-only number.  It is the CUDA-event
latency of the public ``GaussianRasterizer.forward_with_head(s)`` call and thus
conservatively includes the same opaque Raster prefix as the legacy full
Raster call plus the physical mixed render/head leaf.  The same mixed p50 is
used for ``mixed_raster_p50_ms`` and ``mixed_p50_ms``; the same legacy full
Raster p50 is used in both admission documents.  Kernel numerical comparisons
are nevertheless emitted independently for every selected first-linear head.
"""

from __future__ import print_function

from argparse import ArgumentParser
import ast
import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile


SCHEMA_VERSION = 1
VARIANT_SCHEMA_VERSION = 2
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
EXPECTED_MIXED_MULTI_SYMBOL = "tacker_mix_render_heads_v2"
EXPECTED_MIXED_PACKED_SYMBOL = "tacker_mix_render_packed_heads_v3"
EXPECTED_MIXED_WHOLE_SYMBOL = "tacker_mix_render_whole_heads_v4"
EXPECTED_HEAD_MULTI_SOLO_SYMBOL = "tacker_head_linear_multi_solo_v2"
EXPECTED_HEAD_MULTI_GPTB_SYMBOL = "tacker_head_linear_multi_gptb_v2"
EXPECTED_HEAD_PACKED_GPTB_SYMBOL = "tacker_head_linear_packed_gptb_v2"
EXPECTED_HEAD_WHOLE_GPTB_SYMBOL = "tacker_whole_head_gptb_v2"
EXPECTED_FIRST_LINEAR_PARTITION_KIND = "first_linear_heads"
EXPECTED_PACKED_PARTITION_KIND = "packed_first_linear_heads"
EXPECTED_WHOLE_PARTITION_KIND = "whole_heads"
EXPECTED_PARTITION_KINDS = (
    EXPECTED_FIRST_LINEAR_PARTITION_KIND,
    EXPECTED_PACKED_PARTITION_KIND,
    EXPECTED_WHOLE_PARTITION_KIND,
)
EXPECTED_ABI_BY_PARTITION_KIND = {
    EXPECTED_FIRST_LINEAR_PARTITION_KIND: 2,
    EXPECTED_PACKED_PARTITION_KIND: 3,
    EXPECTED_WHOLE_PARTITION_KIND: 4,
}
EXPECTED_FAMILY_BY_PARTITION_KIND = {
    EXPECTED_FIRST_LINEAR_PARTITION_KIND: "first_linear_heads_v2",
    EXPECTED_PACKED_PARTITION_KIND: "packed_first_linear_v3",
    EXPECTED_WHOLE_PARTITION_KIND: "whole_heads_v4",
}
EXPECTED_HEAD_OUTPUT_WIDTHS = {
    "pos": 3,
    "scales": 3,
    "rotations": 4,
    "opacity": 1,
    "shs": 48,
}
EXPECTED_HEAD_ORDER = ("pos", "scales", "rotations", "opacity", "shs")
EXPECTED_V2_WORKER_GROUP_THREADS = [128, 256, 384, 512, 640]
EXPECTED_HEAD_MODULES = {
    "pos": "pos_deform",
    "scales": "scales_deform",
    "rotations": "rotations_deform",
    "opacity": "opacity_deform",
    "shs": "shs_deform",
}

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


def _canonical_json_bytes(value):
    """Return deterministic finite JSON bytes without importing Torch."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value):
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path, label="file"):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise ProfileContractError(
            "cannot hash {} {}: {}".format(label, path, error)
        )
    return digest.hexdigest()


def _reject_json_constant(value):
    raise ValueError("non-finite JSON constant {} is forbidden".format(value))


def _load_json_snapshot(path, label):
    """Read one immutable JSON snapshot and bind it to its exact file bytes."""

    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        value = json.loads(
            raw.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ProfileContractError("cannot load {}: {}".format(label, error))
    if not isinstance(value, dict):
        raise ProfileContractError("{} must be a JSON object".format(label))
    try:
        _canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ProfileContractError(
            "{} contains non-canonical data: {}".format(label, error)
        )
    return value, {
        "path": str(resolved),
        "file_sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
    }


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


def _json_output_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _publish_json_no_clobber(path, value):
    """Publish one complete JSON file without replacing an existing target."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_output_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard-link is an atomic create-if-absent operation.
        # Unlike os.replace, it cannot race into overwriting a previous run.
        os.link(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    # Once the hard link exists, publication is committed.  A failure to
    # remove the staging link must not be reported as a failed publication:
    # callers might otherwise roll back only earlier outputs while leaving
    # this just-linked report/measurement behind.  Retry once, then tolerate
    # an orphan staging name; it shares the exact committed inode and cannot
    # make the four-file result inconsistent.
    for _ in range(2):
        try:
            os.unlink(temporary_name)
            break
        except OSError:
            pass
    return _sha256_bytes(payload)


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
    """No-clobber publication with a hash-bound report committed last.

    A passing run starts only when all four targets are absent.  Measurement
    files are created atomically, their exact bytes are hashed into the report,
    and only then is the report published.  Any injected measurement/report
    publication failure removes files created by this attempt, so an older
    passing report can never be paired with a partial new measurement set.
    """

    outputs = _resolved_output_paths(args)
    written_report = dict(report)
    passed = report.get("passed") is True
    if passed:
        existing = [
            "{}={}".format(label, path)
            for label, path in outputs.items()
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "refusing to overwrite profile outputs: {}".format(
                    ", ".join(existing)
                )
            )
        for label, document in (
            ("device", device),
            ("raster", raster),
            ("leaf", leaf),
        ):
            if not isinstance(document, dict):
                raise ValueError(
                    "passing report requires a {} document".format(label)
                )
        # Validate the complete set before publishing any output.
        for document in (device, raster, leaf, written_report):
            _json_output_bytes(document)
        published = []
        output_hashes = {}
        try:
            for label, document in (
                ("device", device),
                ("raster", raster),
                ("leaf", leaf),
            ):
                output_hashes[label] = _publish_json_no_clobber(
                    outputs[label], document
                )
                published.append(outputs[label])
            written_report["measurement_outputs_written"] = True
            written_report["measurement_output_sha256"] = dict(
                output_hashes
            )
            written_report["publication_policy"] = (
                "strict_no_clobber_measurements_then_report"
            )
            written_report["output_paths"] = {
                key: str(path) for key, path in outputs.items()
            }
            _publish_json_no_clobber(outputs["report"], written_report)
        except Exception:
            for path in published:
                try:
                    path.unlink()
                except OSError:
                    pass
            raise
        return written_report
    else:
        if outputs["report"].exists():
            raise FileExistsError(
                "refusing to overwrite profile report: {}".format(
                    outputs["report"]
                )
            )
        written_report["measurement_outputs_written"] = False
    written_report["output_paths"] = {
        key: str(path) for key, path in outputs.items()
    }
    written_report["publication_policy"] = (
        "strict_no_clobber_measurements_then_report"
    )
    _publish_json_no_clobber(outputs["report"], written_report)
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
    candidate_profile = getattr(args, "candidate_profile", None)
    report = {
        "schema_version": (
            VARIANT_SCHEMA_VERSION
            if candidate_profile is not None
            else SCHEMA_VERSION
        ),
        "kind": "4dgaussians_tacker_leaf_profile_report",
        "generated_at_utc": _utc_now(),
        "passed": False,
        "workload": _workload_document(args),
        "device": None,
        "numerics": None,
        "errors": [str(error)],
        "measurement_outputs_written": False,
    }
    if candidate_profile is not None:
        report.update(
            {
                "candidate_profile_requested": str(
                    Path(candidate_profile).expanduser().resolve()
                ),
                "qualification_profile_execution_requested": True,
                "used_as_deployment": False,
                "candidate_matrix_requested": (
                    str(
                        Path(args.candidate_matrix).expanduser().resolve()
                    )
                    if getattr(args, "candidate_matrix", None) is not None
                    else None
                ),
            }
        )
    if isinstance(details, dict):
        report.update(details)
    return report


def _common_measurement_fields(workload, device, schema_version=SCHEMA_VERSION):
    return {
        "schema_version": schema_version,
        "generated_at_utc": _utc_now(),
        "passed": True,
        "workload": dict(workload),
        "device": dict(device),
    }


def _validate_resource_values(value, section):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("{} resource keys must be strings".format(section))
            _validate_resource_values(item, "{}.{}".format(section, key))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_resource_values(
                item, "{}[{}]".format(section, index)
            )
        return
    if value is None or isinstance(value, (str, bool)):
        return
    if _is_finite_number(value) and float(value) >= 0.0:
        return
    raise ValueError(
        "{} must contain only finite non-negative resource values".format(
            section
        )
    )


def _normalise_variant_metadata(variant, persistent_blocks, head_numerics):
    if not isinstance(variant, dict):
        raise ValueError("variant metadata must be an object")
    required_strings = (
        "variant_id",
        "cuda_symbol",
        "candidate_sha256",
        "profile_file_sha256",
        "profile_sha256",
        "manifest_sha256",
    )
    for key in required_strings:
        value = variant.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError("variant.{} must be a non-empty string".format(key))
    for key in (
        "candidate_sha256",
        "profile_file_sha256",
        "profile_sha256",
        "manifest_sha256",
    ):
        value = variant[key]
        if (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("variant.{} must be a lowercase SHA-256".format(key))
    for key in (
        "source_candidate_sha256_claim",
        "candidate_matrix_sha256",
        "candidate_matrix_file_sha256",
        "source_matrix_sha256_claim",
    ):
        value = variant.get(key)
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                "variant.{} must be null or a lowercase SHA-256".format(key)
            )
    candidate_descriptor_sha256 = variant.get(
        "candidate_descriptor_sha256", variant["candidate_sha256"]
    )
    if (
        not isinstance(candidate_descriptor_sha256, str)
        or len(candidate_descriptor_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in candidate_descriptor_sha256
        )
    ):
        raise ValueError(
            "variant.candidate_descriptor_sha256 must be a lowercase SHA-256"
        )
    candidate_descriptor = variant.get("candidate_descriptor")
    if candidate_descriptor is not None:
        if not isinstance(candidate_descriptor, dict):
            raise ValueError("variant.candidate_descriptor must be an object")
        if _canonical_sha256(candidate_descriptor) != candidate_descriptor_sha256:
            raise ValueError(
                "variant candidate descriptor SHA-256 mismatch"
            )
    selected_heads = variant.get("selected_heads")
    if (
        not isinstance(selected_heads, (list, tuple))
        or not selected_heads
        or any(name not in EXPECTED_HEAD_ORDER for name in selected_heads)
        or len(set(selected_heads)) != len(selected_heads)
        or list(selected_heads)
        != [name for name in EXPECTED_HEAD_ORDER if name in selected_heads]
    ):
        raise ValueError("variant.selected_heads must use canonical head order")
    worker_groups = variant.get("worker_groups")
    if (
        type(worker_groups) is not int
        or worker_groups < 1
        or worker_groups > len(selected_heads)
    ):
        raise ValueError(
            "variant.worker_groups must be in [1, selected head count]"
        )
    variant_blocks = variant.get("persistent_blocks")
    if type(variant_blocks) is not int or variant_blocks < 0:
        raise ValueError("variant.persistent_blocks must be an int >= 0")
    if variant_blocks != persistent_blocks:
        raise ValueError(
            "variant persistent_blocks must match measurement configuration"
        )
    for key in ("abi_version", "physical_cta_threads"):
        if type(variant.get(key)) is not int or variant[key] <= 0:
            raise ValueError("variant.{} must be a positive int".format(key))
    if variant["abi_version"] not in (1, 2, 3, 4):
        raise ValueError("variant ABI must be version 1, 2, 3, or 4")
    abi_contract = {
        1: (EXPECTED_MIXED_SYMBOL, "legacy_pos_l1", "legacy_pos_l1_v1"),
        2: (
            EXPECTED_MIXED_MULTI_SYMBOL,
            "first_linear",
            EXPECTED_FAMILY_BY_PARTITION_KIND[
                EXPECTED_FIRST_LINEAR_PARTITION_KIND
            ],
        ),
        3: (
            EXPECTED_MIXED_PACKED_SYMBOL,
            "packed_first_linear",
            EXPECTED_FAMILY_BY_PARTITION_KIND[EXPECTED_PACKED_PARTITION_KIND],
        ),
        4: (
            EXPECTED_MIXED_WHOLE_SYMBOL,
            "whole_head",
            EXPECTED_FAMILY_BY_PARTITION_KIND[EXPECTED_WHOLE_PARTITION_KIND],
        ),
    }[variant["abi_version"]]
    expected_symbol, expected_backend, expected_family = abi_contract
    if variant["cuda_symbol"] != expected_symbol:
        raise ValueError("variant CUDA symbol disagrees with its physical ABI")
    if variant["abi_version"] == 1 and (
        list(selected_heads) != ["pos"]
        or worker_groups != 1
        or variant["physical_cta_threads"] != 384
    ):
        raise ValueError("legacy/C0 variant must use the exact pos-L1 ABI")
    if variant["abi_version"] != 1 and variant["physical_cta_threads"] != (
        256 + 128 * worker_groups
    ):
        raise ValueError(
            "descriptor-based variant CTA threads disagree with worker_groups"
        )
    if variant.get("backend", expected_backend) != expected_backend:
        raise ValueError("variant backend disagrees with its physical ABI")
    if variant.get("family", expected_family) != expected_family:
        raise ValueError("variant family disagrees with its physical ABI")
    if type(variant.get("used_as_deployment")) is not bool:
        raise ValueError("variant.used_as_deployment must be boolean")
    if variant["used_as_deployment"]:
        raise ValueError("leaf profiler output cannot claim deployment use")
    if variant.get("candidate_identity_kind") not in (
        None,
        "canonical_selected_descriptor",
        "validated_matrix_member",
    ):
        raise ValueError("variant.candidate_identity_kind is invalid")
    for key in (
        "source_candidate_sha256_claim_verified",
        "source_matrix_sha256_claim_verified",
    ):
        if key in variant and type(variant[key]) is not bool:
            raise ValueError("variant.{} must be boolean".format(key))
    resources = variant.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("variant.resources must be an object")
    for key in ("profile_candidate", "rasterizer_runtime", "head_runtime"):
        if not isinstance(resources.get(key), dict):
            raise ValueError(
                "variant.resources.{} must be an object".format(key)
            )
    _validate_resource_values(resources, "variant.resources")
    abi = variant.get("abi")
    if not isinstance(abi, dict):
        raise ValueError("variant.abi must be an object")
    if abi.get("version") != variant["abi_version"]:
        raise ValueError("variant.abi.version must match physical ABI")
    for key in ("mixed_manifest", "head_manifest"):
        if not isinstance(abi.get(key), dict):
            raise ValueError("variant.abi.{} must be an object".format(key))
    if not isinstance(head_numerics, dict) or set(head_numerics) != set(
        selected_heads
    ):
        raise ValueError(
            "head_numerics must contain exactly the selected head names"
        )
    try:
        _canonical_json_bytes(variant)
        _canonical_json_bytes(head_numerics)
    except (TypeError, ValueError) as error:
        raise ValueError("variant metadata must be finite JSON: {}".format(error))
    normalised = dict(variant)
    normalised["selected_heads"] = list(selected_heads)
    normalised["backend"] = expected_backend
    normalised["family"] = expected_family
    return normalised


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
    variant=None,
    head_numerics=None,
    provenance=None,
    head_sample_view_indices=None,
    mixed_sample_view_pairs=None,
):
    """Build legacy-v1 or explicit schema-v2 variant diagnostics.

    The original positional arguments and schema-v1 documents are preserved.
    Supplying ``variant`` opts into the additive schema-v2 fields used by the
    generic first-linear profiler.
    """

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
    schema_version = SCHEMA_VERSION
    variant_metadata = None
    multi_head_abi = False
    if variant is not None:
        variant_metadata = _normalise_variant_metadata(
            variant, persistent_blocks, head_numerics
        )
        schema_version = VARIANT_SCHEMA_VERSION
        multi_head_abi = variant_metadata["abi_version"] in (2, 3, 4)
        measurement_config.update(
            {
                "variant_id": variant_metadata["variant_id"],
                "selected_heads": list(variant_metadata["selected_heads"]),
                "worker_groups": variant_metadata["worker_groups"],
            }
        )
        if not isinstance(provenance, dict):
            raise ValueError("variant measurements require provenance")
        try:
            _canonical_json_bytes(provenance)
        except (TypeError, ValueError) as error:
            raise ValueError("provenance must be finite JSON: {}".format(error))
    common = _common_measurement_fields(
        workload, device, schema_version=schema_version
    )
    if variant_metadata is not None:
        common.update(
            {
                "variant_id": variant_metadata["variant_id"],
                "profile_binding": {
                    "candidate_sha256": variant_metadata["candidate_sha256"],
                    "candidate_descriptor_sha256": variant_metadata.get(
                        "candidate_descriptor_sha256",
                        variant_metadata["candidate_sha256"],
                    ),
                    "source_candidate_sha256_claim": variant_metadata.get(
                        "source_candidate_sha256_claim"
                    ),
                    "source_candidate_sha256_claim_verified": variant_metadata.get(
                        "source_candidate_sha256_claim_verified", False
                    ),
                    "candidate_matrix_sha256": variant_metadata.get(
                        "candidate_matrix_sha256"
                    ),
                    "candidate_matrix_file_sha256": variant_metadata.get(
                        "candidate_matrix_file_sha256"
                    ),
                    "source_matrix_sha256_claim": variant_metadata.get(
                        "source_matrix_sha256_claim"
                    ),
                    "source_matrix_sha256_claim_verified": variant_metadata.get(
                        "source_matrix_sha256_claim_verified", False
                    ),
                    "profile_file_sha256": variant_metadata[
                        "profile_file_sha256"
                    ],
                    "profile_sha256": variant_metadata["profile_sha256"],
                    "manifest_sha256": variant_metadata["manifest_sha256"],
                },
                "variant": dict(variant_metadata),
                "resources": dict(variant_metadata["resources"]),
                "abi": dict(variant_metadata["abi"]),
                "provenance": dict(provenance),
            }
        )
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
    if variant_metadata is not None:
        semantics.update(
            {
                "head_kernel_numerics": (
                    "Each selected mixed output is compared independently "
                    "with the matching solo output and an FP32 "
                    "reference evaluated from the real FP16 operands"
                ),
                "diagnostic_only": True,
                "deployment_inference_forbidden": True,
            }
        )
        if multi_head_abi:
            backend_semantics = {
                "first_linear": (
                    "forward_with_heads",
                    "physical mixed render/multi-first-linear launch",
                    "one head_linear_multi_solo launch covering every selected "
                    "first-linear head",
                ),
                "packed_first_linear": (
                    "forward_with_packed_heads",
                    "physical mixed render/packed-first-linear launch",
                    "one head_linear_packed_gptb launch covering every selected "
                    "packed first-linear head",
                ),
                "whole_head": (
                    "forward_with_whole_heads",
                    "physical mixed render/whole-head launch",
                    "the complete sequence of standalone whole_head_gptb calls "
                    "covering every selected full head",
                ),
            }[variant_metadata["backend"]]
            semantics.update(
                {
                    "mixed_full": (
                        "CUDA-event latency of the complete public "
                        "{} call for the selected variant; "
                        "this contains the opaque Raster prefix and one "
                        "{}"
                    ).format(backend_semantics[0], backend_semantics[1]),
                    "solo_head": (
                        "CUDA-event latency of {}"
                    ).format(backend_semantics[2]),
                }
            )

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
    if variant_metadata is not None:
        raster_document["raster_sample_view_indices"] = list(
            sample_view_indices or []
        )
        raster_document["head_sample_view_indices"] = list(
            head_sample_view_indices or []
        )
        raster_document["mixed_sample_view_pairs"] = list(
            mixed_sample_view_pairs or []
        )
    if multi_head_abi:
        raster_document["timings"]["mixed_full_raster_heads"] = mixed_full
        raster_document["measurements"][
            "mixed_variant_p50_ms"
        ] = mixed_full_p50
        leaf_measurements["multi_head_solo_p50_ms"] = solo_head_p50
        leaf_timings["mixed_full_raster_heads"] = mixed_full
        leaf_timings["real_selected_heads_multi_solo"] = solo_head
    if gptb_head is not None:
        if not isinstance(gptb_head, dict) or not _is_finite_number(
            gptb_head.get("p50_ms")
        ) or float(gptb_head["p50_ms"]) <= 0.0:
            raise ValueError("gptb_head summary requires finite positive p50_ms")
        leaf_timings["real_pos_head_gptb_diagnostic"] = gptb_head
        leaf_measurements["gptb_head_p50_ms_diagnostic"] = float(
            gptb_head["p50_ms"]
        )
        if multi_head_abi:
            leaf_timings[
                "real_selected_heads_multi_gptb_diagnostic"
            ] = gptb_head
            leaf_measurements[
                "multi_head_gptb_p50_ms_diagnostic"
            ] = float(gptb_head["p50_ms"])
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
    if variant_metadata is not None:
        leaf_document["raster_sample_view_indices"] = list(
            sample_view_indices or []
        )
        leaf_document["head_sample_view_indices"] = list(
            head_sample_view_indices or []
        )
        leaf_document["mixed_sample_view_pairs"] = list(
            mixed_sample_view_pairs or []
        )
        leaf_document["selected_head_numerics"] = {
            head_name: head_numerics[head_name]
            for head_name in variant_metadata["selected_heads"]
        }
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
    value, _snapshot = _load_json_snapshot(path, label)
    return value


def _python_base_config(path):
    """Parse, but never execute, one Python config and return its base paths."""

    resolved = Path(path).expanduser().resolve()
    try:
        source = resolved.read_text(encoding="utf-8")
        tree = ast.parse(source, str(resolved))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise ProfileContractError(
            "cannot parse Python configuration {}: {}".format(resolved, error)
        )
    assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_base_"
            for target in node.targets
        ):
            assignments.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_base_"
        ):
            assignments.append(node.value)
    if len(assignments) > 1:
        raise ProfileContractError(
            "configuration {} assigns _base_ more than once".format(resolved)
        )
    if not assignments:
        return []
    try:
        base_value = ast.literal_eval(assignments[0])
    except (ValueError, TypeError) as error:
        raise ProfileContractError(
            "configuration {} has a non-literal _base_: {}".format(
                resolved, error
            )
        )
    if isinstance(base_value, str):
        base_values = [base_value] if base_value else []
    elif isinstance(base_value, (list, tuple)):
        base_values = list(base_value)
    else:
        base_values = None
    if (
        base_values is None
        or any(not isinstance(value, str) or not value for value in base_values)
    ):
        raise ProfileContractError(
            "configuration {} _base_ must be a string or string list".format(
                resolved
            )
        )
    return [(resolved.parent / value).resolve() for value in base_values]


def _snapshot_input_file(path, role, required=True, parse_python=False):
    resolved = Path(path).expanduser().resolve()
    exists = resolved.is_file()
    if required and not exists:
        raise ProfileContractError(
            "required workload input {} is missing: {}".format(role, resolved)
        )
    record = {
        "role": role,
        "path": str(resolved),
        "required": bool(required),
        "exists": exists,
    }
    if exists:
        if parse_python:
            _python_base_config(resolved)
        try:
            record["size_bytes"] = int(resolved.stat().st_size)
        except OSError as error:
            raise ProfileContractError(
                "cannot stat workload input {}: {}".format(resolved, error)
            )
        record["file_sha256"] = _sha256_file(resolved, role)
    else:
        record["size_bytes"] = None
        record["file_sha256"] = None
    return record


def _configuration_chain(path):
    if path is None:
        return []
    chain = []
    stack = []

    def visit(current):
        current = Path(current).expanduser().resolve()
        key = str(current)
        if key in stack:
            raise ProfileContractError(
                "configuration _base_ cycle includes {}".format(current)
            )
        stack.append(key)
        record = _snapshot_input_file(
            current,
            "configuration[{}]".format(len(chain)),
            required=True,
            parse_python=True,
        )
        chain.append(record)
        for base_path in _python_base_config(current):
            visit(base_path)
        stack.pop()

    visit(Path(path).expanduser().resolve())
    return chain


def _capture_execution_source_snapshot(args):
    """Snapshot executable profiler/config bytes before config evaluation."""

    files = [
        _snapshot_input_file(
            Path(__file__).resolve(),
            "execution.profiler_source",
            parse_python=True,
        )
    ]
    model_path = getattr(args, "model_path", None)
    if model_path:
        files.append(
            _snapshot_input_file(
                Path(model_path).expanduser().resolve() / "cfg_args",
                "execution.model_cfg_args",
                parse_python=True,
            )
        )
    files.extend(_configuration_chain(getattr(args, "configs", None)))
    return {
        "captured_before_config_load": True,
        "verified_unchanged_after_measurement": False,
        "files": files,
        "inventories": [],
    }


def _snapshot_inventory(path, role, pattern=None):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ProfileContractError(
            "required dataset directory {} is missing: {}".format(
                role, resolved
            )
        )
    children = resolved.glob(pattern) if pattern is not None else resolved.iterdir()
    entries = []
    for child in sorted(children, key=lambda value: value.name):
        is_file = child.is_file()
        entry = {"name": child.name, "is_file": is_file}
        if is_file:
            try:
                entry["size_bytes"] = int(child.stat().st_size)
            except OSError as error:
                raise ProfileContractError(
                    "cannot stat dataset inventory entry {}: {}".format(
                        child, error
                    )
                )
        entries.append(entry)
    return {
        "role": role,
        "path": str(resolved),
        "pattern": pattern,
        "entries": entries,
        "inventory_sha256": _canonical_sha256(entries),
    }


def _capture_dynerf_dataset_inputs(args, dataset):
    """Bind the fixed DyNeRF camera/time inputs and actual sampled images."""

    source = Path(dataset.source_path).expanduser().resolve()
    files = [
        _snapshot_input_file(
            source / "poses_bounds.npy", "dataset.camera_time_poses"
        ),
        _snapshot_input_file(
            source / "points3D_downsample2.ply", "dataset.aabb_point_cloud"
        ),
    ]
    video_inventory = _snapshot_inventory(
        source, "dataset.video_path_order", pattern="cam*.mp4"
    )
    video_names = [
        entry["name"]
        for entry in video_inventory["entries"]
        if entry["is_file"]
    ]
    if len(video_names) < 2:
        raise ProfileContractError(
            "DyNeRF profiling requires at least two cam*.mp4 paths"
        )
    inventories = [video_inventory]
    image_paths_by_camera = []
    for index, video_name in enumerate(video_names):
        # Mirror Neural3D_NDC_Dataset's video_path.split('.')[0]/images rule.
        video_path = source / video_name
        image_dir = Path(str(video_path).split(".")[0]) / "images"
        inventory = _snapshot_inventory(
            image_dir, "dataset.camera_{}_image_order".format(index)
        )
        image_names = [entry["name"] for entry in inventory["entries"]]
        if not image_names:
            raise ProfileContractError(
                "DyNeRF image directory is empty: {}".format(image_dir)
            )
        inventories.append(inventory)
        image_paths_by_camera.append(
            [image_dir / image_name for image_name in image_names]
        )

    # Scene construction always reads the first train and first test image to
    # establish camera geometry, even when another split is profiled.
    used_images = [image_paths_by_camera[0][0], image_paths_by_camera[1][0]]
    split = getattr(args, "split", "test")
    if split == "train":
        split_images = [
            image_path
            for camera_images in image_paths_by_camera[1:]
            for image_path in camera_images[:300]
        ]
    elif split == "test":
        split_images = image_paths_by_camera[0][:300]
    else:
        # Video cameras reuse the first test image and derive pose/time from
        # poses_bounds.npy; no further image bytes are read.
        split_images = None
    if split_images is not None:
        pairs = _selected_pairs(
            len(split_images),
            int(args.view_start),
            int(args.view_stride),
            int(args.views),
        )
        for index in sorted(set(value for pair in pairs for value in pair)):
            used_images.append(split_images[index])
    unique_images = []
    seen = set()
    for image_path in used_images:
        key = str(image_path.resolve())
        if key not in seen:
            seen.add(key)
            unique_images.append(image_path)
    files.extend(
        _snapshot_input_file(
            image_path,
            "dataset.used_image[{}]".format(index),
        )
        for index, image_path in enumerate(unique_images)
    )
    return files, inventories


def _capture_workload_input_snapshot(args, dataset):
    """Bind every model/config byte that can affect this fixed leaf run."""

    model_path = Path(dataset.model_path).expanduser().resolve()
    iteration_dir = (
        model_path
        / "point_cloud"
        / "iteration_{}".format(EXPECTED_ITERATION)
    )
    files = [
        _snapshot_input_file(
            iteration_dir / "point_cloud.ply", "checkpoint.point_cloud"
        ),
        _snapshot_input_file(
            iteration_dir / "deformation.pth", "checkpoint.deformation"
        ),
        _snapshot_input_file(
            iteration_dir / "deformation_table.pth",
            "checkpoint.deformation_table",
            required=False,
        ),
        _snapshot_input_file(
            iteration_dir / "deformation_accum.pth",
            "checkpoint.deformation_accum",
            required=False,
        ),
        _snapshot_input_file(
            model_path / "cfg_args", "model.cfg_args", parse_python=True
        ),
    ]
    dataset_files, inventories = _capture_dynerf_dataset_inputs(args, dataset)
    files.extend(dataset_files)
    return {
        "iteration_directory": str(iteration_dir),
        "captured_before_cuda": True,
        "verified_unchanged_after_measurement": False,
        "files": files,
        "inventories": inventories,
    }


def _verify_input_snapshot(snapshot):
    if not isinstance(snapshot, dict) or not isinstance(
        snapshot.get("files"), list
    ):
        raise ProfileContractError("workload input snapshot is invalid")
    for expected in snapshot["files"]:
        actual = _snapshot_input_file(
            expected["path"],
            expected["role"],
            required=expected["required"],
            parse_python=expected["role"].startswith("configuration[")
            or expected["role"] == "model.cfg_args",
        )
        for key in ("exists", "size_bytes", "file_sha256"):
            if actual.get(key) != expected.get(key):
                raise ProfileContractError(
                    "workload input changed during profiling: {} ({})".format(
                        expected["path"], key
                    )
                )
    for expected in snapshot.get("inventories", []):
        actual = _snapshot_inventory(
            expected["path"],
            expected["role"],
            pattern=expected.get("pattern"),
        )
        if actual["inventory_sha256"] != expected.get("inventory_sha256"):
            raise ProfileContractError(
                "dataset inventory changed during profiling: {}".format(
                    expected["path"]
                )
            )
    snapshot["verified_unchanged_after_measurement"] = True
    return snapshot


def _verify_bound_file(path, expected_sha256, expected_size, label):
    current = _snapshot_input_file(path, label, required=True)
    if (
        current["file_sha256"] != expected_sha256
        or current["size_bytes"] != expected_size
    ):
        raise ProfileContractError("{} changed during profiling".format(label))


def load_candidate_profile_descriptor(
    path,
    tacker_api=None,
    candidate_matrix_path=None,
    autotune_api=None,
):
    """Resolve an explicit schema-v2 physical qualification profile.

    The profile is validated with the same public loader and partition resolver
    used by runtime dispatch.  Generic first-linear, packed first-linear,
    whole-head, and schema-v2 legacy/C0 candidates are accepted.  Deployment
    state is recorded but deliberately ignored for this offline profiler.

    A source profile's provenance candidate hash is only a claim.  Supplying a
    sealed candidate matrix validates that claim and proves the selected
    descriptor is the exact materialization of one matrix member.  Without a
    matrix, the canonical selected descriptor hash is the primary identity.
    """

    profile, snapshot = _load_json_snapshot(path, "candidate profile")
    if tacker_api is None:
        tacker_api = importlib.import_module(
            "gaussian_renderer.tacker_pipeline"
        )
    try:
        validated = tacker_api.load_tacker_profile(profile_override=profile)
        if validated.get("schema_version") != VARIANT_SCHEMA_VERSION:
            raise ProfileContractError(
                "--candidate-profile must use schema_version 2"
            )
        selected_variant_id = validated.get("selected_variant_id")
        selected_matches = [
            candidate
            for candidate in validated.get("candidates", [])
            if candidate.get("variant_id") == selected_variant_id
        ]
        if len(selected_matches) != 1:
            raise ProfileContractError(
                "candidate profile must identify exactly one selected candidate"
            )
        candidate = tacker_api.selected_tacker_candidate(validated)
        if candidate is not selected_matches[0] and candidate != selected_matches[0]:
            raise ProfileContractError(
                "runtime selected candidate disagrees with profile selection"
            )
        if candidate.get("execution_mode") != "tacker":
            raise ProfileContractError(
                "selected candidate must be a physical Tacker variant"
            )
        partition = tacker_api.resolve_fusion_partition(candidate)
    except ProfileContractError:
        raise
    except Exception as error:
        raise ProfileContractError(
            "invalid candidate profile: {}".format(error)
        )

    variant = getattr(partition, "variant", None)
    fusion_variant_type = getattr(tacker_api, "FusionVariant", None)
    if not isinstance(fusion_variant_type, type) or not isinstance(
        variant, fusion_variant_type
    ):
        raise ProfileContractError(
            "selected candidate did not resolve to FusionVariant"
        )
    selected_heads = tuple(getattr(variant, "selected_heads", ()))
    if (
        not selected_heads
        or any(name not in EXPECTED_HEAD_ORDER for name in selected_heads)
        or list(selected_heads)
        != [name for name in EXPECTED_HEAD_ORDER if name in selected_heads]
    ):
        raise ProfileContractError(
            "selected candidate resolved invalid selected_heads"
        )
    worker_groups = getattr(variant, "worker_groups", None)
    persistent_blocks = getattr(variant, "persistent_blocks", None)
    if (
        type(worker_groups) is not int
        or worker_groups < 1
        or worker_groups > len(selected_heads)
    ):
        raise ProfileContractError(
            "selected candidate resolved invalid worker_groups"
        )
    if type(persistent_blocks) is not int or persistent_blocks < 0:
        raise ProfileContractError(
            "selected candidate resolved invalid persistent_blocks"
        )
    legacy_physical_abi = bool(getattr(variant, "legacy_pos_l1", False))
    partition_descriptor = candidate.get("partition")
    if legacy_physical_abi:
        if (
            partition_descriptor is not None
            or selected_heads != ("pos",)
            or worker_groups != 1
            or getattr(variant, "abi_version", None) != 1
        ):
            raise ProfileContractError(
                "legacy/C0 candidate resolved an invalid v1 physical partition"
            )
    else:
        partition_kind = (
            partition_descriptor.get("kind")
            if isinstance(partition_descriptor, dict)
            else None
        )
        expected_abi = EXPECTED_ABI_BY_PARTITION_KIND.get(partition_kind)
        expected_family = EXPECTED_FAMILY_BY_PARTITION_KIND.get(partition_kind)
        if (
            expected_abi is None
            or getattr(variant, "abi_version", None) != expected_abi
            or getattr(variant, "family", expected_family) != expected_family
        ):
            raise ProfileContractError(
                "candidate must use a supported C1-C4 production partition"
            )

    deployment = validated.get("deployment")
    deployment_enabled = bool(
        isinstance(deployment, dict)
        and deployment.get("enabled") is True
        and deployment.get("valid") is True
    )
    sealed_provenance = validated.get("provenance")
    source_candidate_sha256_claim = (
        sealed_provenance.get("candidate_sha256")
        if isinstance(sealed_provenance, dict)
        else None
    )
    source_matrix_sha256_claim = (
        sealed_provenance.get("matrix_sha256")
        if isinstance(sealed_provenance, dict)
        else None
    )
    for label, value in (
        (
            "profile provenance candidate_sha256",
            source_candidate_sha256_claim,
        ),
        ("profile provenance matrix_sha256", source_matrix_sha256_claim),
    ):
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ProfileContractError("{} must be a lowercase SHA-256".format(label))
    candidate_descriptor_sha256 = _canonical_sha256(candidate)
    matrix = None
    matrix_snapshot = None
    matrix_candidate = None
    if candidate_matrix_path is not None:
        matrix, matrix_snapshot = _load_json_snapshot(
            candidate_matrix_path, "candidate matrix"
        )
        if autotune_api is None:
            autotune_api = importlib.import_module("scripts.tacker_autotune")
        try:
            autotune_api.validate_matrix(matrix)
        except Exception as error:
            raise ProfileContractError(
                "invalid candidate matrix: {}".format(error)
            )
        expected_family = (
            getattr(autotune_api, "LEGACY_ABI_FAMILY", "legacy_pos_l1_v1")
            if legacy_physical_abi
            else EXPECTED_FAMILY_BY_PARTITION_KIND[
                partition_descriptor["kind"]
            ]
        )
        matrix_matches = [
            item
            for item in matrix.get("candidates", [])
            if item.get("variant_id") == variant.variant_id
            and item.get("abi_family") == expected_family
            and item.get("selected_heads") == list(selected_heads)
            and item.get("worker_groups") == worker_groups
            and item.get("persistent_blocks") == persistent_blocks
            and item.get("effective_persistent_blocks") == persistent_blocks
        ]
        if len(matrix_matches) != 1:
            raise ProfileContractError(
                "selected variant/PB/heads is not exactly one candidate matrix member"
            )
        matrix_candidate = matrix_matches[0]
        if source_matrix_sha256_claim != matrix.get("matrix_sha256"):
            raise ProfileContractError(
                "profile provenance matrix_sha256 does not match candidate matrix"
            )
        if source_candidate_sha256_claim != matrix_candidate.get(
            "candidate_sha256"
        ):
            raise ProfileContractError(
                "profile provenance candidate_sha256 does not match matrix member"
            )
        try:
            materialized = autotune_api.materialize_candidate_descriptor(
                matrix_candidate,
                module=tacker_api,
                resources=candidate.get("resources"),
            )
        except Exception as error:
            raise ProfileContractError(
                "cannot materialize selected matrix candidate: {}".format(error)
            )
        if materialized != candidate:
            raise ProfileContractError(
                "selected profile descriptor is not the exact matrix materialization"
            )
    trusted_matrix_candidate_sha256 = (
        matrix_candidate.get("candidate_sha256")
        if matrix_candidate is not None
        else None
    )
    descriptor = {
        "profile": validated,
        "candidate": candidate,
        "partition": partition,
        "variant": variant,
        "variant_id": variant.variant_id,
        "selected_heads": selected_heads,
        "worker_groups": worker_groups,
        "persistent_blocks": persistent_blocks,
        "profile_path": snapshot["path"],
        "profile_file_sha256": snapshot["file_sha256"],
        "profile_file_size_bytes": snapshot["size_bytes"],
        "profile_sha256": validated.get("profile_sha256"),
        "manifest_sha256": validated.get("manifest_sha256"),
        "candidate_sha256": (
            trusted_matrix_candidate_sha256
            if trusted_matrix_candidate_sha256 is not None
            else candidate_descriptor_sha256
        ),
        "candidate_identity_kind": (
            "validated_matrix_member"
            if trusted_matrix_candidate_sha256 is not None
            else "canonical_selected_descriptor"
        ),
        "candidate_descriptor_sha256": candidate_descriptor_sha256,
        "source_candidate_sha256_claim": source_candidate_sha256_claim,
        "source_candidate_sha256_claim_verified": (
            trusted_matrix_candidate_sha256 is not None
        ),
        "candidate_matrix_sha256": (
            matrix.get("matrix_sha256") if matrix is not None else None
        ),
        "source_matrix_sha256_claim": source_matrix_sha256_claim,
        "source_matrix_sha256_claim_verified": matrix is not None,
        "candidate_matrix": matrix,
        "candidate_matrix_path": (
            matrix_snapshot["path"] if matrix_snapshot is not None else None
        ),
        "candidate_matrix_file_sha256": (
            matrix_snapshot["file_sha256"]
            if matrix_snapshot is not None
            else None
        ),
        "candidate_matrix_file_size_bytes": (
            matrix_snapshot["size_bytes"]
            if matrix_snapshot is not None
            else None
        ),
        "matrix_candidate": matrix_candidate,
        "legacy_physical_abi": legacy_physical_abi,
        "deployment_enabled": deployment_enabled,
        "qualification_profile": not deployment_enabled,
        # An explicit profiler run is diagnostic even if its source profile is
        # already deployable.  Consumers must never infer runtime admission
        # from successful leaf measurements.
        "used_as_deployment": False,
    }
    return descriptor


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


def _normalise_raster_runtime_resources(
    raw, abi_version, worker_groups, family=None
):
    if not isinstance(raw, dict):
        raise ProfileContractError("rasterizer resource query returned no object")
    values = dict(raw)
    aliases = {
        "block_threads": ("block_threads", "physical_threads"),
        "registers_per_thread": ("registers_per_thread",),
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
        value = next(
            (values[source] for source in sources if source in values), None
        )
        if type(value) is not int or value < 0:
            raise ProfileContractError(
                "rasterizer resources omit int {} >= 0".format(target)
            )
        values[target] = value
    if values.get("launch_supported") is not True:
        raise ProfileContractError("rasterizer resource query rejects launch")
    if values["active_blocks_per_sm"] < 1:
        raise ProfileContractError("rasterizer reports zero active blocks per SM")
    expected_threads = 256 + 128 * worker_groups
    if values["block_threads"] != expected_threads:
        raise ProfileContractError("rasterizer physical thread count changed")
    if values["max_threads_per_block"] < expected_threads:
        raise ProfileContractError("rasterizer kernel maximum is too small")
    occupancy = values.get("occupancy")
    if not _is_finite_number(occupancy) or not 0.0 < float(occupancy) <= 1.0:
        raise ProfileContractError("rasterizer reports invalid occupancy")
    if values.get("abi_version") != abi_version:
        raise ProfileContractError("rasterizer resource ABI version changed")
    if values.get("worker_groups") != worker_groups:
        raise ProfileContractError("rasterizer resource worker_groups changed")
    if family is not None and values.get("family") != family:
        raise ProfileContractError("rasterizer resource family changed")
    try:
        _canonical_json_bytes(values)
    except (TypeError, ValueError) as error:
        raise ProfileContractError(
            "rasterizer resources contain non-finite data: {}".format(error)
        )
    return values


def _validate_v1_candidate_extensions(
    rasterizer_module,
    head_backend,
    tacker_api,
    candidate_selection,
    mixed_abi_path,
    head_abi_path,
):
    """Validate a schema-v2 C0 descriptor against the physical ABI-v1 stack."""

    candidate = candidate_selection["candidate"]
    mixed_abi, mixed_snapshot = _load_json_snapshot(
        mixed_abi_path, "mixed positional-head ABI"
    )
    head_abi, head_snapshot = _load_json_snapshot(
        head_abi_path, "positional head ABI"
    )
    if mixed_snapshot["file_sha256"] != candidate.get("abi_manifest_sha256"):
        raise ProfileContractError(
            "mixed ABI-v1 file SHA-256 disagrees with selected C0 candidate"
        )
    expected_head_hash = getattr(tacker_api, "HEAD_ABI_SHA256", None)
    if head_snapshot["file_sha256"] != expected_head_hash:
        raise ProfileContractError("head ABI-v1 file SHA-256 changed")
    mixed_for_validation = dict(mixed_abi)
    mixed_for_validation["_path"] = mixed_snapshot["path"]
    head_for_validation = dict(head_abi)
    head_for_validation["_path"] = head_snapshot["path"]
    extensions = _validate_extensions(
        rasterizer_module,
        head_backend,
        mixed_for_validation,
        head_for_validation,
    )
    resource_provider = getattr(
        rasterizer_module, "tacker_resource_requirements", None
    )
    if not callable(resource_provider):
        raise ProfileContractError(
            "rasterizer has no ABI-v1 public resource query"
        )
    runtime_resources = _normalise_raster_runtime_resources(
        dict(resource_provider(abi_version=1, worker_groups=1)), 1, 1
    )
    raster_backend = getattr(rasterizer_module, "_C", None)
    raster_provenance = _module_binary_provenance(
        raster_backend, "rasterizer"
    )
    raster_provenance.update(extensions["rasterizer"])
    head_provenance = _module_binary_provenance(head_backend, "head")
    head_provenance.update(extensions["head"])
    return {
        "rasterizer": raster_provenance,
        "head": head_provenance,
        "abi": {
            "version": 1,
            "mixed_manifest": mixed_snapshot,
            "head_manifest": head_snapshot,
            "candidate_mixed_manifest_sha256": candidate.get(
                "abi_manifest_sha256"
            ),
            "candidate_head_manifest_sha256": expected_head_hash,
            "mixed_symbol": EXPECTED_MIXED_SYMBOL,
            "head_solo_symbol": EXPECTED_HEAD_SOLO_SYMBOL,
            "head_gptb_symbol": EXPECTED_HEAD_GPTB_SYMBOL,
        },
        "resources": {
            "profile_candidate": candidate.get("resources"),
            "rasterizer_runtime": runtime_resources,
            "head_runtime": {
                "abi_version": 1,
                "physical_threads": 128,
                "resource_query_available": False,
                "capabilities": dict(head_backend.tacker_capabilities()),
            },
        },
    }


def _module_binary_provenance(module, label):
    path_value = getattr(module, "__file__", None)
    if not isinstance(path_value, str) or not path_value:
        raise ProfileContractError(
            "{} compiled extension path is unavailable".format(label)
        )
    path = Path(path_value).expanduser().resolve()
    try:
        size_bytes = path.stat().st_size
    except OSError as error:
        raise ProfileContractError(
            "cannot stat {} compiled extension: {}".format(label, error)
        )
    return {
        "path": str(path),
        "file_sha256": _sha256_file(path, "{} extension".format(label)),
        "size_bytes": int(size_bytes),
    }


def _validate_v2_head_capabilities(capabilities):
    expected = {
        "abi_version": 2,
        "sm_target": EXPECTED_CUDA_ARCH,
        "head_features": 128,
        "max_head_tasks": 5,
        "worker_group_threads": 128,
        "max_worker_groups": 5,
        "max_mixed_cta_threads": 896,
        "resource_query": "tacker_resources_v2",
    }
    for key, value in expected.items():
        if capabilities.get(key) != value:
            raise ProfileContractError(
                "head ABI v2 capability {} must be {!r}, got {!r}".format(
                    key, value, capabilities.get(key)
                )
            )
    symbols = capabilities.get("global_kernel_symbols")
    if not isinstance(symbols, dict):
        raise ProfileContractError("head ABI v2 global symbols are missing")
    if symbols.get("multi_solo") != EXPECTED_HEAD_MULTI_SOLO_SYMBOL:
        raise ProfileContractError("head ABI v2 multi-solo symbol changed")
    if symbols.get("multi_gptb") != EXPECTED_HEAD_MULTI_GPTB_SYMBOL:
        raise ProfileContractError("head ABI v2 multi-GPTB symbol changed")
    try:
        supported_task_counts = list(capabilities.get("supported_task_counts", ()))
        supported_worker_groups = list(
            capabilities.get("supported_worker_groups", ())
        )
    except TypeError:
        raise ProfileContractError(
            "head ABI v2 supported counts must be sequences"
        )
    if supported_task_counts != [1, 2, 3, 4, 5]:
        raise ProfileContractError("head ABI v2 task-count contract changed")
    if supported_worker_groups != [1, 2, 3, 4, 5]:
        raise ProfileContractError("head ABI v2 worker-group contract changed")


def _normalise_v2_head_runtime_resources(
    head_resources, worker_groups, raster_resources, backend="first_linear"
):
    """Select and strictly validate standalone facts for one C1--C4 backend."""

    if (
        type(worker_groups) is not int
        or worker_groups < 1
        or worker_groups > 5
    ):
        raise ProfileContractError("head worker_groups must be an int in [1, 5]")
    if not isinstance(head_resources, dict) or head_resources.get(
        "abi_version"
    ) != 2 or not isinstance(head_resources.get("kernels"), dict):
        raise ProfileContractError("head ABI v2 resource query is invalid")
    if not isinstance(raster_resources, dict):
        raise ProfileContractError("rasterizer runtime resources are invalid")
    device_max_threads = raster_resources.get(
        "device_max_threads_per_multiprocessor"
    )
    if type(device_max_threads) is not int or device_max_threads < 1:
        raise ProfileContractError(
            "rasterizer resources omit device max threads per SM"
        )
    physical_threads = worker_groups * 128
    symbols = {
        "first_linear": (
            EXPECTED_HEAD_MULTI_SOLO_SYMBOL,
            EXPECTED_HEAD_MULTI_GPTB_SYMBOL,
        ),
        "packed_first_linear": (EXPECTED_HEAD_PACKED_GPTB_SYMBOL,),
        "whole_head": (EXPECTED_HEAD_WHOLE_GPTB_SYMBOL,),
    }.get(backend)
    if symbols is None:
        raise ProfileContractError("unknown standalone head backend")
    selected = {}
    for symbol in symbols:
        raw = head_resources["kernels"].get(symbol)
        if not isinstance(raw, dict):
            raise ProfileContractError(
                "head resource query is missing {}".format(symbol)
            )
        values = {}
        for field in (
            "registers_per_thread",
            "static_shared_memory_bytes",
            "local_memory_bytes",
            "max_threads_per_block",
            "ptx_version",
            "binary_version",
        ):
            value = raw.get(field)
            if type(value) is not int or value < 0:
                raise ProfileContractError(
                    "head resource {}.{} must be an int >= 0".format(
                        symbol, field
                    )
                )
            values[field] = value
        if values["max_threads_per_block"] < 1:
            raise ProfileContractError(
                "head resource {}.max_threads_per_block must be >= 1".format(
                    symbol
                )
            )
        for field in ("ptx_version", "binary_version"):
            if values[field] != 86:
                raise ProfileContractError(
                    "head resource {}.{} must be 86 for the Phase 3 A6000 "
                    "artifact".format(symbol, field)
                )
        selected_threads = 128 if backend == "whole_head" else physical_threads
        if values["max_threads_per_block"] < selected_threads:
            raise ProfileContractError(
                "head resource {} cannot launch {} threads".format(
                    symbol, selected_threads
                )
            )
        threads_by_group = raw.get("worker_group_threads")
        expected_thread_contract = (
            [128] if backend == "whole_head" else EXPECTED_V2_WORKER_GROUP_THREADS
        )
        if (
            not isinstance(threads_by_group, (list, tuple))
            or list(threads_by_group) != expected_thread_contract
            or any(type(value) is not int for value in threads_by_group)
        ):
            raise ProfileContractError(
                "head resource {} must report the complete WG1-5 CTA thread "
                "contract {}".format(symbol, expected_thread_contract)
            )
        active_by_group = raw.get("active_blocks_per_sm")
        if (
            not isinstance(active_by_group, (list, tuple))
            or len(active_by_group) != len(expected_thread_contract)
            or any(
                type(value) is not int or value < 1
                for value in active_by_group
            )
        ):
            raise ProfileContractError(
                "head resource {} must report five positive integer "
                "active-block counts".format(symbol)
            )
        occupancies = []
        for group_threads, active_blocks in zip(
            threads_by_group, active_by_group
        ):
            if values["max_threads_per_block"] < group_threads:
                raise ProfileContractError(
                    "head resource {} cannot launch the complete WG1-5 "
                    "thread contract".format(symbol)
                )
            occupancy = (
                float(active_blocks) * float(group_threads)
                / float(device_max_threads)
            )
            if (
                not math.isfinite(occupancy)
                or occupancy <= 0.0
                or occupancy > 1.0
            ):
                raise ProfileContractError(
                    "head resource {} has invalid WG1-5 occupancy".format(
                        symbol
                    )
                )
            occupancies.append(occupancy)
        selected_index = 0 if backend == "whole_head" else worker_groups - 1
        active_blocks = active_by_group[selected_index]
        occupancy = occupancies[selected_index]
        values.update(
            {
                "physical_threads": selected_threads,
                "active_blocks_per_sm": active_blocks,
                "occupancy": occupancy,
                "worker_group_threads": list(threads_by_group),
                "active_blocks_per_sm_by_worker_group": list(active_by_group),
                "occupancy_by_worker_group": occupancies,
            }
        )
        selected[symbol] = values
    return {
        "abi_version": 2,
        "selected_worker_groups": worker_groups,
        "selected_physical_threads": (
            128 if backend == "whole_head" else physical_threads
        ),
        "backend": backend,
        "device_max_threads_per_multiprocessor": device_max_threads,
        "kernels": selected,
        "raw_query": head_resources,
    }


def _normalise_v2_raster_runtime_resources(raw, candidate, variant):
    """Apply the support gate's strict A6000/resource-profile contract."""

    family = getattr(variant, "family", None)
    abi_version = getattr(variant, "abi_version", 2)
    values = _normalise_raster_runtime_resources(
        raw, abi_version, variant.worker_groups, family=family
    )
    exact = {
        "block_threads": variant.physical_cta_threads,
        "compute_capability_major": EXPECTED_COMPUTE_CAPABILITY[0],
        "compute_capability_minor": EXPECTED_COMPUTE_CAPABILITY[1],
    }
    for field, expected in exact.items():
        actual = values.get(field)
        if type(actual) is not int or actual != expected:
            raise ProfileContractError(
                "rasterizer runtime resource {} must be {}, got {!r}".format(
                    field, expected, actual
                )
            )
    device_max_threads = values.get("device_max_threads_per_block")
    if (
        type(device_max_threads) is not int
        or device_max_threads < variant.physical_cta_threads
    ):
        raise ProfileContractError(
            "rasterizer device maximum cannot launch selected mixed CTA"
        )
    device_sm_threads = values.get(
        "device_max_threads_per_multiprocessor"
    )
    if type(device_sm_threads) is not int or device_sm_threads < 1:
        raise ProfileContractError(
            "rasterizer resources omit device max threads per SM"
        )

    sealed = candidate.get("resources")
    if not isinstance(sealed, dict):
        raise ProfileContractError(
            "selected Tacker candidate has no sealed measured resources"
        )
    replayable = {
        key: value
        for key, value in values.items()
        if key != "launch_supported" and _is_finite_number(value)
    }
    if replayable != sealed:
        raise ProfileContractError(
            "rasterizer runtime resources changed from the sealed candidate"
        )
    try:
        _canonical_json_bytes(values)
    except (TypeError, ValueError) as error:
        raise ProfileContractError(
            "rasterizer resources contain non-finite data: {}".format(error)
        )
    return values


def _snapshot_v2_support_resources(
    rasterizer_module, candidate_selection
):
    """Capture the strict Raster resource artifact immediately before support."""

    provider = getattr(rasterizer_module, "tacker_variant_resources", None)
    if not callable(provider):
        raise ProfileContractError(
            "rasterizer has no public tacker_variant_resources call"
        )
    variant = candidate_selection["variant"]
    try:
        raw = provider(variant.worker_groups, family=variant.family)
    except TypeError as error:
        if variant.abi_version != 2:
            raise ProfileContractError(
                "rasterizer resource query has no family-aware C3/C4 API: {}"
                .format(error)
            )
        raw = provider(variant.worker_groups)
    return _normalise_v2_raster_runtime_resources(
        dict(raw),
        candidate_selection["candidate"],
        variant,
    )


def _require_v2_raster_snapshot_match(current, support_snapshot):
    if current != support_snapshot:
        raise ProfileContractError(
            "rasterizer resources changed between the support gate and "
            "extension validation"
        )


def _validate_v2_extensions(
    rasterizer_module,
    head_api,
    head_backend,
    candidate_selection,
    mixed_abi_path,
    head_abi_path,
    support_raster_resources,
):
    """Validate and snapshot every binary/ABI used by a C1--C4 leaf run."""

    candidate = candidate_selection["candidate"]
    variant = candidate_selection["variant"]
    partition_kind = candidate["partition"]["kind"]
    backend_contract = {
        EXPECTED_FIRST_LINEAR_PARTITION_KIND: {
            "abi_version": 2,
            "symbol": EXPECTED_MIXED_MULTI_SYMBOL,
            "binding": "rasterize_gaussians_with_heads",
            "method": "GaussianRasterizer.forward_with_heads",
            "python_symbol": "rasterize_gaussians_with_heads",
            "capability_prefix": "mixed_render_heads",
            "capability_symbol": "mixed_multi_symbol",
            "capability_manifest": "mixed_multi_manifest_sha256",
            "capability_head_manifest": "head_multi_manifest_sha256",
            "head_adapter": "tacker_4dgs::head_linear_multi_gptb_device",
        },
        EXPECTED_PACKED_PARTITION_KIND: {
            "abi_version": 3,
            "symbol": EXPECTED_MIXED_PACKED_SYMBOL,
            "binding": "rasterize_gaussians_with_packed_heads",
            "method": "GaussianRasterizer.forward_with_packed_heads",
            "python_symbol": "rasterize_gaussians_with_packed_heads",
            "capability_prefix": "mixed_render_packed_heads",
            "capability_symbol": "mixed_packed_symbol",
            "capability_manifest": "mixed_packed_manifest_sha256",
            "capability_head_manifest": (
                "mixed_packed_head_manifest_sha256"
            ),
            "head_adapter": "tacker_4dgs::head_linear_packed_gptb_device",
        },
        EXPECTED_WHOLE_PARTITION_KIND: {
            "abi_version": 4,
            "symbol": EXPECTED_MIXED_WHOLE_SYMBOL,
            "binding": "rasterize_gaussians_with_whole_heads",
            "method": "GaussianRasterizer.forward_with_whole_heads",
            "python_symbol": "rasterize_gaussians_with_whole_heads",
            "capability_prefix": "mixed_render_whole_heads",
            "capability_symbol": "mixed_whole_symbol",
            "capability_manifest": "mixed_whole_manifest_sha256",
            "capability_head_manifest": (
                "mixed_whole_head_manifest_sha256"
            ),
            "head_adapter": "tacker_4dgs::whole_head_multi_gptb_device",
        },
    }.get(partition_kind)
    if backend_contract is None:
        raise ProfileContractError("unsupported C1-C4 partition kind")
    backend = getattr(variant, "backend", "first_linear")
    mixed_abi, mixed_snapshot = _load_json_snapshot(
        mixed_abi_path, "mixed multi-head ABI"
    )
    head_abi, head_snapshot = _load_json_snapshot(
        head_abi_path, "head multi-linear ABI"
    )
    if mixed_snapshot["file_sha256"] != candidate.get("abi_manifest_sha256"):
        raise ProfileContractError(
            "mixed multi-head ABI file SHA-256 disagrees with selected candidate"
        )
    if head_snapshot["file_sha256"] != candidate.get(
        "head_abi_manifest_sha256"
    ):
        raise ProfileContractError(
            "head multi-linear ABI file SHA-256 disagrees with selected candidate"
        )
    manifest_binding_matches = (
        mixed_abi.get("python_binding") == backend_contract["binding"]
    )
    manifest_method_matches = (
        mixed_abi.get("python_method") == backend_contract["method"]
    )
    if partition_kind == EXPECTED_WHOLE_PARTITION_KIND:
        manifest_binding_matches = mixed_abi.get("python_bindings") == [
            "rasterize_gaussians_with_whole_head",
            backend_contract["binding"],
        ]
        manifest_method_matches = mixed_abi.get("python_methods") == [
            "GaussianRasterizer.forward_with_whole_head",
            backend_contract["method"],
        ]
    if (
        mixed_abi.get("abi_version") != backend_contract["abi_version"]
        or mixed_abi.get("global_kernel_symbol") != backend_contract["symbol"]
        or not manifest_binding_matches
        or not manifest_method_matches
        or mixed_abi.get("capability_query") != "tacker_capabilities"
    ):
        raise ProfileContractError("selected mixed-head ABI manifest changed")
    dependency = mixed_abi.get("tacker_ext_dependency")
    if (
        not isinstance(dependency, dict)
        or dependency.get("manifest_sha256")
        != head_snapshot["file_sha256"]
        or dependency.get("abi_version") != 2
        or dependency.get("device_adapter") != backend_contract["head_adapter"]
    ):
        raise ProfileContractError(
            "mixed multi-head ABI dependency does not bind the head ABI"
        )
    head_symbols = head_abi.get("global_kernel_symbols")
    if (
        head_abi.get("abi_version") != 2
        or head_abi.get("capability_query") != "tacker_capabilities_v2"
        or not isinstance(head_symbols, dict)
        or not isinstance(head_symbols.get("multi_solo"), dict)
        or head_symbols["multi_solo"].get("symbol")
        != EXPECTED_HEAD_MULTI_SOLO_SYMBOL
        or not isinstance(head_symbols.get("multi_gptb"), dict)
        or head_symbols["multi_gptb"].get("symbol")
        != EXPECTED_HEAD_MULTI_GPTB_SYMBOL
    ):
        raise ProfileContractError("head multi-linear ABI manifest changed")
    selected_manifest_symbols = {
        "first_linear": {
            "multi_solo": EXPECTED_HEAD_MULTI_SOLO_SYMBOL,
            "multi_gptb": EXPECTED_HEAD_MULTI_GPTB_SYMBOL,
        },
        "packed_first_linear": {
            "packed_gptb": EXPECTED_HEAD_PACKED_GPTB_SYMBOL,
        },
        "whole_head": {
            "whole_head_gptb": EXPECTED_HEAD_WHOLE_GPTB_SYMBOL,
        },
    }.get(backend)
    if selected_manifest_symbols is None:
        raise ProfileContractError("selected candidate has an unknown head backend")
    for key, symbol in selected_manifest_symbols.items():
        if (
            not isinstance(head_symbols.get(key), dict)
            or head_symbols[key].get("symbol") != symbol
        ):
            raise ProfileContractError(
                "head ABI v2 {} symbol changed".format(key)
            )

    raster_backend = getattr(rasterizer_module, "_C", None)
    if raster_backend is None:
        raise ProfileContractError("rasterizer compiled backend is unavailable")
    raster_python_symbols = _require_python_symbols(
        raster_backend,
        "rasterizer extension",
        (
            "rasterize_gaussians",
            backend_contract["python_symbol"],
            "tacker_capabilities",
            "tacker_resource_requirements",
        ),
    )
    selected_head_python = {
        "first_linear": (
            "head_linear_multi_solo",
            "head_linear_multi_solo_out",
            "head_linear_multi_gptb",
            "head_linear_multi_gptb_out",
        ),
        "packed_first_linear": (
            "head_linear_packed_gptb",
            "head_linear_packed_gptb_out",
        ),
        "whole_head": ("whole_head_gptb",),
    }.get(backend)
    if selected_head_python is None:
        raise ProfileContractError("selected candidate has an unknown head backend")
    head_python_symbols = _require_python_symbols(
        head_backend,
        "head extension",
        selected_head_python + ("tacker_capabilities_v2", "tacker_resources_v2"),
    )
    _require_python_symbols(
        head_api,
        "head Python API",
        tuple(
            name
            for name in selected_head_python
            if not name.endswith("_out")
        )
        + ("tacker_capabilities_v2", "tacker_resources_v2"),
    )

    capability_provider = getattr(rasterizer_module, "tacker_capabilities", None)
    if not callable(capability_provider):
        raise ProfileContractError("rasterizer has no tacker_capabilities call")
    raster_capabilities = dict(capability_provider())
    prefix = backend_contract["capability_prefix"]
    raster_expected = {
        "stream_aware": True,
        "{}_abi".format(prefix): backend_contract["abi_version"],
        prefix: True,
        backend_contract["capability_symbol"]: backend_contract["symbol"],
        backend_contract["capability_manifest"]: mixed_snapshot["file_sha256"],
        backend_contract["capability_head_manifest"]: head_snapshot[
            "file_sha256"
        ],
        "rasterizer_commit": EXPECTED_RASTERIZER_COMMIT,
        "sm_target": EXPECTED_CUDA_ARCH,
    }
    for key, value in raster_expected.items():
        if raster_capabilities.get(key) != value:
            raise ProfileContractError(
                "rasterizer v2 capability {} must be {!r}, got {!r}".format(
                    key, value, raster_capabilities.get(key)
                )
            )
    head_capabilities = dict(head_api.tacker_capabilities_v2())
    _validate_v2_head_capabilities(head_capabilities)
    expected_head_capability_symbol = {
        "first_linear": None,
        "packed_first_linear": ("packed_gptb", EXPECTED_HEAD_PACKED_GPTB_SYMBOL),
        "whole_head": ("whole_head_gptb", EXPECTED_HEAD_WHOLE_GPTB_SYMBOL),
    }[backend]
    if expected_head_capability_symbol is not None:
        symbol_key, expected_symbol = expected_head_capability_symbol
        if head_capabilities.get("global_kernel_symbols", {}).get(
            symbol_key
        ) != expected_symbol:
            raise ProfileContractError(
                "head ABI v2 {} symbol changed".format(symbol_key)
            )
    resource_provider = getattr(
        rasterizer_module, "tacker_variant_resources", None
    )
    if not callable(resource_provider):
        raise ProfileContractError(
            "rasterizer has no public tacker_variant_resources call"
        )
    try:
        raw_raster_resources = resource_provider(
            variant.worker_groups, family=variant.family
        )
    except TypeError as error:
        if variant.abi_version != 2:
            raise ProfileContractError(
                "rasterizer resource query has no family-aware C3/C4 API: {}"
                .format(error)
            )
        raw_raster_resources = resource_provider(variant.worker_groups)
    raster_resources = _normalise_v2_raster_runtime_resources(
        dict(raw_raster_resources), candidate, variant
    )
    _require_v2_raster_snapshot_match(
        raster_resources, support_raster_resources
    )
    head_resources = dict(head_api.tacker_resources_v2())
    selected_head_resources = _normalise_v2_head_runtime_resources(
        head_resources,
        variant.worker_groups,
        raster_resources,
        backend=backend,
    )
    try:
        _canonical_json_bytes(raster_resources)
        _canonical_json_bytes(selected_head_resources)
    except (TypeError, ValueError) as error:
        raise ProfileContractError(
            "extension resources contain non-finite data: {}".format(error)
        )

    raster_provenance = _module_binary_provenance(
        raster_backend, "rasterizer"
    )
    raster_provenance.update(
        {
            "capabilities": raster_capabilities,
            "python_symbols": raster_python_symbols,
            "cuda_global_symbols": [backend_contract["symbol"]],
        }
    )
    head_provenance = _module_binary_provenance(head_backend, "head")
    head_provenance.update(
        {
            "capabilities": head_capabilities,
            "python_symbols": head_python_symbols,
            "cuda_global_symbols": list(
                {
                    "first_linear": (
                        EXPECTED_HEAD_MULTI_SOLO_SYMBOL,
                        EXPECTED_HEAD_MULTI_GPTB_SYMBOL,
                    ),
                    "packed_first_linear": (
                        EXPECTED_HEAD_PACKED_GPTB_SYMBOL,
                    ),
                    "whole_head": (EXPECTED_HEAD_WHOLE_GPTB_SYMBOL,),
                }[backend]
            ),
        }
    )
    return {
        "rasterizer": raster_provenance,
        "head": head_provenance,
        "abi": {
            "version": backend_contract["abi_version"],
            "mixed_manifest": mixed_snapshot,
            "head_manifest": head_snapshot,
            "candidate_mixed_manifest_sha256": candidate.get(
                "abi_manifest_sha256"
            ),
            "candidate_head_manifest_sha256": candidate.get(
                "head_abi_manifest_sha256"
            ),
            "mixed_symbol": backend_contract["symbol"],
            "head_symbols": list(
                {
                    "first_linear": (
                        EXPECTED_HEAD_MULTI_SOLO_SYMBOL,
                        EXPECTED_HEAD_MULTI_GPTB_SYMBOL,
                    ),
                    "packed_first_linear": (
                        EXPECTED_HEAD_PACKED_GPTB_SYMBOL,
                    ),
                    "whole_head": (EXPECTED_HEAD_WHOLE_GPTB_SYMBOL,),
                }[backend]
            ),
            "head_multi_solo_symbol": (
                EXPECTED_HEAD_MULTI_SOLO_SYMBOL
                if backend == "first_linear"
                else None
            ),
            "head_multi_gptb_symbol": (
                EXPECTED_HEAD_MULTI_GPTB_SYMBOL
                if backend == "first_linear"
                else None
            ),
        },
        "resources": {
            "profile_candidate": candidate.get("resources"),
            "rasterizer_runtime": raster_resources,
            "head_runtime": selected_head_resources,
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


def _variant_backend(bundle):
    backend = getattr(bundle["variant"], "backend", "first_linear")
    if backend not in ("first_linear", "packed_first_linear", "whole_head"):
        raise ProfileContractError(
            "selected candidate has an unsupported head backend"
        )
    return backend


def _whole_head_inputs(task, head_count):
    values = task.whole_head_inputs
    if isinstance(values, (list, tuple)):
        values = tuple(values)
    else:
        values = (values,) * head_count
    if len(values) != head_count:
        raise ProfileContractError(
            "whole-head input count does not match selected_heads"
        )
    return values


def _variant_head_operands(bundle):
    """Return per-selected-head operands in a backend-independent form."""

    task = bundle["task"]
    head_count = len(bundle["variant"].selected_heads)
    backend = _variant_backend(bundle)
    if backend == "first_linear":
        inputs = tuple(task.head_inputs)
        first_weights = tuple(task.head_weights)
        first_biases = tuple(task.head_biases)
        tail_weights = (None,) * head_count
        tail_biases = (None,) * head_count
    elif backend == "packed_first_linear":
        inputs = (task.shared_head_input,) * head_count
        first_weights = tuple(
            task.packed_head_weights[index] for index in range(head_count)
        )
        first_biases = tuple(
            task.packed_head_biases[index] for index in range(head_count)
        )
        tail_weights = (None,) * head_count
        tail_biases = (None,) * head_count
    else:
        inputs = _whole_head_inputs(task, head_count)
        first_weights = tuple(task.whole_first_weights)
        first_biases = tuple(task.whole_first_biases)
        tail_weights = tuple(task.whole_tail_weights)
        tail_biases = tuple(task.whole_tail_biases)
    counts = {
        len(inputs),
        len(first_weights),
        len(first_biases),
        len(tail_weights),
        len(tail_biases),
        head_count,
    }
    if counts != {head_count}:
        raise ProfileContractError(
            "selected head operand counts are inconsistent"
        )
    return tuple(
        {
            "input": inputs[index],
            "first_weight": first_weights[index],
            "first_bias": first_biases[index],
            "tail_weight": tail_weights[index],
            "tail_bias": tail_biases[index],
        }
        for index in range(head_count)
    )


def _require_real_cuda_tensor(tensor, label, shape, dtype, torch):
    if tuple(getattr(tensor, "shape", ())) != tuple(shape):
        raise ProfileContractError(
            "{} must have shape {}".format(label, list(shape))
        )
    if getattr(tensor, "dtype", None) != dtype:
        raise ProfileContractError(
            "{} must have dtype {}".format(label, str(dtype))
        )
    if not bool(getattr(tensor, "is_cuda", False)):
        raise ProfileContractError("{} must be a CUDA tensor".format(label))
    contiguous = getattr(tensor, "is_contiguous", None)
    if not callable(contiguous) or not bool(contiguous()):
        raise ProfileContractError("{} must be contiguous".format(label))
    data_ptr = getattr(tensor, "data_ptr", None)
    if not callable(data_ptr) or int(data_ptr()) % 32 != 0:
        raise ProfileContractError("{} must be 32-byte aligned".format(label))


def _validate_variant_task_operands(bundle, torch):
    """Fail closed on the concrete C1--C4 tensor ABI before warmup."""

    variant = bundle["variant"]
    backend = _variant_backend(bundle)
    task = bundle["task"]
    selected_heads = tuple(variant.selected_heads)
    operands = _variant_head_operands(bundle)
    for index, (head_name, operand) in enumerate(zip(selected_heads, operands)):
        prefix = "real {} {}".format(head_name, backend)
        _require_real_cuda_tensor(
            operand["input"],
            "{} input".format(prefix),
            (EXPECTED_GAUSSIANS, 128),
            torch.float16,
            torch,
        )
        _require_real_cuda_tensor(
            operand["first_weight"],
            "{} first weight".format(prefix),
            (128, 128),
            torch.float16,
            torch,
        )
        _require_real_cuda_tensor(
            operand["first_bias"],
            "{} first bias".format(prefix),
            (128,),
            torch.float32,
            torch,
        )
        if backend == "whole_head":
            output_width = EXPECTED_HEAD_OUTPUT_WIDTHS[head_name]
            _require_real_cuda_tensor(
                operand["tail_weight"],
                "{} tail weight".format(prefix),
                (output_width, 128),
                torch.float32,
                torch,
            )
            _require_real_cuda_tensor(
                operand["tail_bias"],
                "{} tail bias".format(prefix),
                (output_width,),
                torch.float32,
                torch,
            )

    if backend == "packed_first_linear":
        head_count = len(selected_heads)
        _require_real_cuda_tensor(
            task.packed_head_weights,
            "packed head weights",
            (head_count, 128, 128),
            torch.float16,
            torch,
        )
        _require_real_cuda_tensor(
            task.packed_head_biases,
            "packed head biases",
            (head_count, 128),
            torch.float32,
            torch,
        )
    elif backend == "whole_head":
        expected_widths = tuple(
            EXPECTED_HEAD_OUTPUT_WIDTHS[name] for name in selected_heads
        )
        if tuple(task.output_widths) != expected_widths:
            raise ProfileContractError(
                "whole-head output widths disagree with selected_heads"
            )
    return operands


def _variant_standalone_outputs(bundle, head_api, diagnostic_gptb=False):
    """Run the ABI-matched standalone adapter for numerical/timing evidence."""

    task = bundle["task"]
    variant = bundle["variant"]
    backend = _variant_backend(bundle)
    if backend == "first_linear":
        if diagnostic_gptb:
            values = head_api.head_linear_multi_gptb(
                task.head_inputs,
                task.head_weights,
                task.head_biases,
                variant.worker_groups,
                variant.persistent_blocks,
            )
        else:
            values = head_api.head_linear_multi_solo(
                task.head_inputs,
                task.head_weights,
                task.head_biases,
                variant.worker_groups,
            )
        if not isinstance(values, (list, tuple)):
            raise ProfileContractError(
                "multi-head standalone API must return one output per head"
            )
        return tuple(values)
    if backend == "packed_first_linear":
        packed = head_api.head_linear_packed_gptb(
            task.shared_head_input,
            task.packed_head_weights,
            task.packed_head_biases,
            variant.worker_groups,
            variant.persistent_blocks,
        )
        return tuple(packed[index] for index in range(len(variant.selected_heads)))

    operands = _variant_head_operands(bundle)
    return tuple(
        head_api.whole_head_gptb(
            operand["input"],
            operand["first_weight"],
            operand["first_bias"],
            operand["tail_weight"],
            operand["tail_bias"],
            variant.persistent_blocks,
        )
        for operand in operands
    )


def _variant_quantized_reference(operand, backend, torch):
    value = _fp32_linear_reference(
        operand["input"],
        operand["first_weight"],
        operand["first_bias"],
        torch,
    )
    if backend == "whole_head":
        value = _fp32_linear_reference(
            torch.relu(value),
            operand["tail_weight"],
            operand["tail_bias"],
            torch,
        )
    return value


def _variant_original_reference(head, hidden, backend, torch):
    value = _fp32_linear_reference(
        head[0](hidden).float(),
        head[1].weight.detach(),
        head[1].bias.detach(),
        torch,
    )
    if backend == "whole_head":
        value = _fp32_linear_reference(
            torch.relu(value),
            head[3].weight.detach(),
            head[3].bias.detach(),
            torch,
        )
    return value


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


def _variant_mixed_call(bundle):
    """Launch one schema-v2 mixed call through the common partition API."""

    return bundle["partition"].launch_mixed(
        bundle["context"], bundle["state"], bundle["task"]
    )


def _packed_mixed_row_count(bundle):
    """Return the agreed C3 row count when an input/context exposes it."""

    task = bundle.get("task")
    context = bundle.get("context")
    sources = (
        ("task.shared_head_input", getattr(task, "shared_head_input", None)),
        ("context.means3D", getattr(context, "means3D", None)),
        ("context.means2D", getattr(context, "means2D", None)),
        (
            "context.screenspace_points",
            getattr(context, "screenspace_points", None),
        ),
    )
    row_counts = []
    for label, value in sources:
        if value is None or not hasattr(value, "shape"):
            continue
        try:
            shape = tuple(value.shape)
        except (TypeError, ValueError) as error:
            raise ProfileContractError(
                "cannot read {} shape: {}".format(label, error)
            )
        if not shape or type(shape[0]) is not int or shape[0] < 0:
            raise ProfileContractError(
                "{} must expose a non-negative integer row count".format(label)
            )
        row_counts.append((label, shape[0]))
    if not row_counts:
        return None
    if len({row_count for _label, row_count in row_counts}) != 1:
        raise ProfileContractError(
            "C3 input/context row counts disagree: {}".format(
                ", ".join(
                    "{}={}".format(label, row_count)
                    for label, row_count in row_counts
                )
            )
        )
    return row_counts[0][1]


def _unpack_packed_mixed_outputs(bundle, packed_outputs, selected_heads):
    """Validate and slice the ABI3 ``[H, N, 128]`` output tensor."""

    if not hasattr(packed_outputs, "shape"):
        raise ProfileContractError(
            "C3 packed mixed-head output must be one rank-3 tensor"
        )
    try:
        shape = tuple(packed_outputs.shape)
    except (TypeError, ValueError) as error:
        raise ProfileContractError(
            "cannot read C3 packed mixed-head output shape: {}".format(error)
        )
    if len(shape) != 3 or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise ProfileContractError(
            "C3 packed mixed-head output must have shape [H, N, 128]"
        )
    expected_heads = len(selected_heads)
    if shape[0] != expected_heads:
        raise ProfileContractError(
            "C3 packed mixed-head output head dimension {} does not match "
            "selected_heads {}".format(shape[0], expected_heads)
        )
    if shape[2] != 128:
        raise ProfileContractError(
            "C3 packed mixed-head output feature dimension must be 128"
        )
    expected_rows = _packed_mixed_row_count(bundle)
    if expected_rows is not None and shape[1] != expected_rows:
        raise ProfileContractError(
            "C3 packed mixed-head output row dimension {} does not match {}".format(
                shape[1], expected_rows
            )
        )
    try:
        return tuple(packed_outputs[index] for index in range(expected_heads))
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ProfileContractError(
            "cannot slice C3 packed mixed-head output: {}".format(error)
        )


def _variant_mixed_outputs(bundle, mixed_outputs):
    try:
        raster_result, head_outputs = bundle["partition"].result_from_mixed(
            bundle["context"], mixed_outputs
        )
    except Exception as error:
        raise ProfileContractError(
            "cannot decode mixed-head output: {}".format(error)
        )
    selected_heads = bundle["variant"].selected_heads
    if _variant_backend(bundle) == "packed_first_linear":
        head_outputs = _unpack_packed_mixed_outputs(
            bundle, head_outputs, selected_heads
        )
    else:
        if not isinstance(head_outputs, (list, tuple)):
            head_outputs = (head_outputs,)
        head_outputs = tuple(head_outputs)
    if len(head_outputs) != len(selected_heads):
        raise ProfileContractError(
            "mixed-head output count does not match selected_heads"
        )
    return raster_result, dict(zip(selected_heads, head_outputs))


def _check_variant_numerics(bundles, rasterize_state, head_api, torch):
    """Validate Raster and every selected C1--C4 output on real operands."""

    selected_heads = tuple(bundles[0]["variant"].selected_heads)
    selected_backend = _variant_backend(bundles[0])
    per_pair = []
    failures = []
    per_selected_head = {
        head_name: {
            "passed": True,
            "thresholds": {
                "atol": NUMERICAL_THRESHOLDS["head_kernel_atol"],
                "rtol": NUMERICAL_THRESHOLDS["head_kernel_rtol"],
            },
            "per_view_pair": [],
        }
        for head_name in selected_heads
    }
    for bundle in bundles:
        if tuple(bundle["variant"].selected_heads) != selected_heads:
            raise ProfileContractError("selected heads changed between view pairs")
        if _variant_backend(bundle) != selected_backend:
            raise ProfileContractError("selected backend changed between view pairs")
        legacy = rasterize_state(bundle["context"], bundle["state"])
        mixed_raw = _variant_mixed_call(bundle)
        mixed_raster, mixed_by_head = _variant_mixed_outputs(bundle, mixed_raw)
        task = bundle["task"]
        standalone_values = _variant_standalone_outputs(bundle, head_api)
        operands = _variant_head_operands(bundle)
        standalone_values = tuple(standalone_values)
        if len(standalone_values) != len(selected_heads):
            raise ProfileContractError(
                "standalone head output count does not match selected_heads"
            )
        standalone_by_head = dict(zip(selected_heads, standalone_values))
        torch.cuda.synchronize()

        color = _tensor_error_metrics(mixed_raster.render, legacy.render, torch)
        depth = _tensor_error_metrics(mixed_raster.depth, legacy.depth, torch)
        if tuple(mixed_raster.radii.shape) != tuple(legacy.radii.shape):
            raise ProfileContractError("legacy/mixed radii shapes differ")
        radii_mismatches = int(
            (mixed_raster.radii != legacy.radii).sum().item()
        )
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
        }
        pair_heads = {}
        network = bundle["pc"]._deformation.deformation_net
        for index, head_name in enumerate(selected_heads):
            head = getattr(network, EXPECTED_HEAD_MODULES[head_name])
            quantized_reference = _variant_quantized_reference(
                operands[index], selected_backend, torch
            )
            original_reference = _variant_original_reference(
                head, task.hidden, selected_backend, torch
            )
            standalone = standalone_by_head[head_name]
            mixed = mixed_by_head[head_name]
            standalone_vs_mixed = _tensor_error_metrics(
                standalone, mixed, torch
            )
            standalone_vs_quantized = _tensor_error_metrics(
                standalone, quantized_reference, torch
            )
            mixed_vs_quantized = _tensor_error_metrics(
                mixed, quantized_reference, torch
            )
            quantization = _tensor_error_metrics(
                quantized_reference, original_reference, torch
            )
            atol = NUMERICAL_THRESHOLDS["head_kernel_atol"]
            rtol = NUMERICAL_THRESHOLDS["head_kernel_rtol"]
            standalone_label = {
                "first_linear": "multi_solo",
                "packed_first_linear": "packed_gptb",
                "whole_head": "whole_head_gptb",
            }[selected_backend]
            head_checks = {
                "{}_vs_mixed".format(standalone_label): bool(
                    torch.allclose(
                        standalone, mixed, rtol=rtol, atol=atol
                    )
                ),
                "{}_vs_quantized_fp32_reference".format(standalone_label): bool(
                    torch.allclose(
                        standalone,
                        quantized_reference,
                        rtol=rtol,
                        atol=atol,
                    )
                ),
                "mixed_vs_quantized_fp32_reference": bool(
                    torch.allclose(
                        mixed,
                        quantized_reference,
                        rtol=rtol,
                        atol=atol,
                    )
                ),
            }
            head_report = {
                "passed": all(head_checks.values()),
                "checks": head_checks,
                "standalone_backend": standalone_label,
                "{}_vs_mixed".format(
                    standalone_label
                ): standalone_vs_mixed,
                "{}_vs_quantized_fp32_reference".format(
                    standalone_label
                ): standalone_vs_quantized,
                "mixed_vs_quantized_fp32_reference": mixed_vs_quantized,
                "fp16_operand_quantization_vs_original_fp32": quantization,
            }
            pair_heads[head_name] = head_report
            per_head_pair = dict(head_report)
            per_head_pair.update(
                {
                    "current_view_index": bundle["current_index"],
                    "next_view_index": bundle["next_index"],
                }
            )
            per_selected_head[head_name]["per_view_pair"].append(
                per_head_pair
            )
            if not head_report["passed"]:
                per_selected_head[head_name]["passed"] = False
                failures.append(
                    "{} head numerical gate failed for view pair {}->{}".format(
                        head_name,
                        bundle["current_index"],
                        bundle["next_index"],
                    )
                )
            checks["{}_head".format(head_name)] = head_report["passed"]

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
            "selected_heads": pair_heads,
        }
        if not (
            checks["legacy_vs_mixed_color"]
            and checks["legacy_vs_mixed_depth"]
            and checks["legacy_vs_mixed_radii"]
        ):
            failures.append(
                "Raster numerical gate failed for view pair {}->{}".format(
                    bundle["current_index"], bundle["next_index"]
                )
            )
        per_pair.append(pair_report)

    report = {
        "passed": not failures,
        "thresholds": dict(NUMERICAL_THRESHOLDS),
        "selected_heads": list(selected_heads),
        "backend": selected_backend,
        "kernel_reference": (
            "Per-head real FP16 first-linear input/weight converted to FP32, "
            "FP32 bias and, for whole-head, FP32 ReLU/tail Linear; TF32 disabled"
        ),
        "per_selected_head": per_selected_head,
        "per_view_pair": per_pair,
        "errors": failures,
    }
    if failures:
        raise ProfileContractError(
            "; ".join(failures), details={"numerics": report}
        )
    return report


def _check_numerics(bundles, rasterize_state, head_linear_solo, torch, persistent_blocks):
    per_pair = []
    failures = []
    network = bundles[0]["pc"]._deformation.deformation_net
    selected_linear = network.pos_deform[1]
    for bundle in bundles:
        legacy = rasterize_state(bundle["context"], bundle["state"])
        if bundle.get("partition") is None:
            mixed_values = _mixed_call(bundle, persistent_blocks)
        else:
            mixed_values = _variant_mixed_call(bundle)
        mixed_color, mixed_radii, mixed_depth, mixed_head = mixed_values
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


def _promote_legacy_candidate_numerics(numerics):
    """Expose C0's physical v1 head checks through the schema-v2 head map."""

    per_head = []
    passed = True
    for pair in numerics.get("per_view_pair", []):
        checks = pair.get("checks", {})
        head_checks = {
            key: value
            for key, value in checks.items()
            if "head" in key or "quantized_fp32_reference" in key
        }
        item = dict(pair.get("head", {}))
        item.update(
            {
                "current_view_index": pair.get("current_view_index"),
                "next_view_index": pair.get("next_view_index"),
                "checks": head_checks,
                "passed": bool(head_checks) and all(head_checks.values()),
            }
        )
        passed = passed and item["passed"]
        per_head.append(item)
    promoted = dict(numerics)
    promoted["selected_heads"] = ["pos"]
    promoted["per_selected_head"] = {
        "pos": {
            "passed": passed,
            "thresholds": {
                "atol": NUMERICAL_THRESHOLDS["head_kernel_atol"],
                "rtol": NUMERICAL_THRESHOLDS["head_kernel_rtol"],
            },
            "physical_abi_version": 1,
            "per_view_pair": per_head,
        }
    }
    return promoted


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


def _optional_file_provenance(path, label):
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        size_bytes = resolved.stat().st_size
    except OSError as error:
        raise ProfileContractError(
            "cannot stat {} {}: {}".format(label, resolved, error)
        )
    return {
        "path": str(resolved),
        "file_sha256": _sha256_file(resolved, label),
        "size_bytes": int(size_bytes),
    }


def _build_variant_provenance(
    args,
    dataset,
    pipeline,
    extensions,
    selection,
    workload_inputs,
    execution_inputs,
):
    execution_files = execution_inputs["files"]
    profiler_source = next(
        item
        for item in execution_files
        if item["role"] == "execution.profiler_source"
    )
    configuration_chain = [
        item
        for item in execution_files
        if item["role"].startswith("configuration[")
    ]
    return {
        "workload_source": {
            "dataset_source_path": str(
                Path(dataset.source_path).expanduser().resolve()
            ),
            "model_path": str(Path(dataset.model_path).expanduser().resolve()),
            "scene": EXPECTED_SCENE,
            "iteration": EXPECTED_ITERATION,
            "split": args.split,
        },
        "workload_inputs": workload_inputs,
        "execution_inputs": execution_inputs,
        "profiler_source": dict(profiler_source),
        "configuration_source": (
            dict(configuration_chain[0]) if configuration_chain else None
        ),
        "configuration_chain": [dict(item) for item in configuration_chain],
        "configuration": {
            "view_start": args.view_start,
            "view_stride": args.view_stride,
            "views": args.views,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "profile_gptb": bool(args.profile_gptb),
            "pipeline": {
                "debug": bool(getattr(pipeline, "debug", False)),
                "compute_cov3D_python": bool(
                    getattr(pipeline, "compute_cov3D_python", False)
                ),
                "convert_SHs_python": bool(
                    getattr(pipeline, "convert_SHs_python", False)
                ),
            },
        },
        "candidate_profile_source": {
            "path": selection["profile_path"],
            "file_sha256": selection["profile_file_sha256"],
            "size_bytes": selection["profile_file_size_bytes"],
            "profile_sha256": selection["profile_sha256"],
            "manifest_sha256": selection["manifest_sha256"],
            "candidate_sha256": selection["candidate_sha256"],
            "candidate_descriptor_sha256": selection[
                "candidate_descriptor_sha256"
            ],
            "source_candidate_sha256_claim": selection[
                "source_candidate_sha256_claim"
            ],
            "source_candidate_sha256_claim_verified": selection[
                "source_candidate_sha256_claim_verified"
            ],
            "candidate_matrix_sha256": selection[
                "candidate_matrix_sha256"
            ],
            "candidate_matrix_path": selection["candidate_matrix_path"],
            "candidate_matrix_file_sha256": selection[
                "candidate_matrix_file_sha256"
            ],
            "candidate_matrix_file_size_bytes": selection[
                "candidate_matrix_file_size_bytes"
            ],
            "source_matrix_sha256_claim": selection[
                "source_matrix_sha256_claim"
            ],
            "source_matrix_sha256_claim_verified": selection[
                "source_matrix_sha256_claim_verified"
            ],
            "deployment_enabled": selection["deployment_enabled"],
            "qualification_profile": selection["qualification_profile"],
            "used_as_deployment": False,
            "sealed_provenance": dict(selection["profile"]["provenance"]),
        },
        "extensions": {
            "rasterizer": {
                "path": extensions["rasterizer"]["path"],
                "file_sha256": extensions["rasterizer"]["file_sha256"],
                "size_bytes": extensions["rasterizer"]["size_bytes"],
            },
            "head": {
                "path": extensions["head"]["path"],
                "file_sha256": extensions["head"]["file_sha256"],
                "size_bytes": extensions["head"]["size_bytes"],
            },
        },
        "abi_manifests": dict(extensions["abi"]),
    }


def _build_variant_metadata(selection, extensions):
    variant = selection["variant"]
    candidate = selection["candidate"]
    return {
        "variant_id": selection["variant_id"],
        "selected_heads": list(selection["selected_heads"]),
        "worker_groups": selection["worker_groups"],
        "persistent_blocks": selection["persistent_blocks"],
        "abi_version": variant.abi_version,
        "cuda_symbol": variant.cuda_symbol,
        "physical_cta_threads": variant.physical_cta_threads,
        "backend": getattr(variant, "backend", "first_linear"),
        "family": getattr(variant, "family", "first_linear_heads_v2"),
        "candidate_sha256": selection["candidate_sha256"],
        "candidate_identity_kind": selection["candidate_identity_kind"],
        "candidate_descriptor_sha256": selection[
            "candidate_descriptor_sha256"
        ],
        "candidate_descriptor": candidate,
        "source_candidate_sha256_claim": selection[
            "source_candidate_sha256_claim"
        ],
        "source_candidate_sha256_claim_verified": selection[
            "source_candidate_sha256_claim_verified"
        ],
        "candidate_matrix_sha256": selection["candidate_matrix_sha256"],
        "candidate_matrix_path": selection["candidate_matrix_path"],
        "candidate_matrix_file_sha256": selection[
            "candidate_matrix_file_sha256"
        ],
        "source_matrix_sha256_claim": selection[
            "source_matrix_sha256_claim"
        ],
        "source_matrix_sha256_claim_verified": selection[
            "source_matrix_sha256_claim_verified"
        ],
        "matrix_candidate": selection["matrix_candidate"],
        "profile_file_sha256": selection["profile_file_sha256"],
        "profile_sha256": selection["profile_sha256"],
        "manifest_sha256": selection["manifest_sha256"],
        "profile_path": selection["profile_path"],
        "profile_schema_version": selection["profile"]["schema_version"],
        "deployment_enabled": selection["deployment_enabled"],
        "qualification_profile": selection["qualification_profile"],
        "used_as_deployment": False,
        "resources": {
            "profile_candidate": candidate.get("resources"),
            "rasterizer_runtime": extensions["resources"][
                "rasterizer_runtime"
            ],
            "head_runtime": extensions["resources"]["head_runtime"],
        },
        "abi": dict(extensions["abi"]),
    }


def run_profile(
    args,
    dataset,
    hyperparam,
    pipeline,
    execution_source_snapshot=None,
    workload_input_snapshot=None,
):
    """Load the fixed scene, validate numerics, and collect CUDA-event data."""

    candidate_profile_path = getattr(args, "candidate_profile", None)
    candidate_matrix_path = getattr(args, "candidate_matrix", None)
    if candidate_matrix_path is not None and candidate_profile_path is None:
        raise ProfileContractError(
            "--candidate-matrix requires --candidate-profile"
        )
    if execution_source_snapshot is None:
        execution_source_snapshot = _capture_execution_source_snapshot(args)
    elif execution_source_snapshot.get("captured_before_config_load") is not True:
        raise ProfileContractError(
            "execution source snapshot was not captured before config load"
        )
    # Detect config/source replacement during load_config itself, then leave
    # the final verified flag for the post-measurement check below.
    _verify_input_snapshot(execution_source_snapshot)
    execution_source_snapshot["verified_unchanged_after_measurement"] = False
    # Workload bytes are captured after config merge selected the actual model
    # and dataset paths, but before CUDA initialization in the CLI path.
    if workload_input_snapshot is None:
        workload_input_snapshot = _capture_workload_input_snapshot(args, dataset)
    else:
        if workload_input_snapshot.get("captured_before_cuda") is not True:
            raise ProfileContractError(
                "workload input snapshot was not captured before CUDA"
            )
        _verify_input_snapshot(workload_input_snapshot)
        workload_input_snapshot["verified_unchanged_after_measurement"] = False

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

    tacker_api = importlib.import_module("gaussian_renderer.tacker_pipeline")
    head_api = importlib.import_module("tacker_4dgs_head")
    head_backend = importlib.import_module("tacker_4dgs_head._C")
    head_linear_solo = head_api.head_linear_solo
    head_linear_gptb = head_api.head_linear_gptb
    candidate_selection = None
    if candidate_profile_path is not None:
        candidate_selection = load_candidate_profile_descriptor(
            candidate_profile_path,
            tacker_api=tacker_api,
            candidate_matrix_path=candidate_matrix_path,
        )
    uses_multi_head_abi = bool(
        candidate_selection is not None
        and not candidate_selection["legacy_physical_abi"]
    )
    support_raster_resources = None

    if args.scene_name.replace("-", "_").lower() != EXPECTED_SCENE:
        raise ProfileContractError("--scene-name must be flame_steak")
    if args.iteration != EXPECTED_ITERATION:
        raise ProfileContractError("--iteration must be 14000")
    if args.persistent_blocks < 0:
        raise ProfileContractError("--persistent-blocks must be >= 0")
    effective_persistent_blocks = args.persistent_blocks
    if candidate_selection is not None:
        effective_persistent_blocks = candidate_selection["persistent_blocks"]
        if args.persistent_blocks not in (0, effective_persistent_blocks):
            raise ProfileContractError(
                "--persistent-blocks must be 0 or match the selected candidate"
            )
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

    if candidate_selection is None:
        mixed_abi = _load_json(args.mixed_abi, "mixed ABI")
        mixed_abi["_path"] = str(Path(args.mixed_abi).expanduser().resolve())
        head_abi = _load_json(args.head_abi, "head ABI")
        head_abi["_path"] = str(Path(args.head_abi).expanduser().resolve())
        extensions = _validate_extensions(
            rasterizer_module, head_backend, mixed_abi, head_abi
        )
    elif candidate_selection["legacy_physical_abi"]:
        extensions = _validate_v1_candidate_extensions(
            rasterizer_module,
            head_backend,
            tacker_api,
            candidate_selection,
            args.mixed_abi,
            args.head_abi,
        )
    else:
        # Bind the exact public resource artifact immediately before the
        # runtime support gate.  Full v2 extension validation deliberately
        # re-queries after that gate and rejects any drift.
        support_raster_resources = _snapshot_v2_support_resources(
            rasterizer_module, candidate_selection
        )
        extensions = None

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
        if candidate_selection is None or candidate_selection[
            "legacy_physical_abi"
        ]:
            if deformation_args is None or bool(
                getattr(deformation_args, "no_dx", True)
            ):
                raise ProfileContractError(
                    "the real positional head requires no_dx=false"
                )
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
        if candidate_selection is not None:
            support_reason = tacker_api.tacker_support_reason(
                gaussians,
                pipeline,
                candidate_selection["profile"],
                stage="fine",
                cam_type=scene.dataset_type,
                workload_name=EXPECTED_SCENE,
                iteration=scene.loaded_iter,
                qualification_mode=True,
            )
            if support_reason is not None:
                raise ProfileContractError(
                    "selected candidate is not runnable: {}".format(
                        support_reason
                    )
                )
            if uses_multi_head_abi:
                partition_kind = candidate_selection["candidate"]["partition"][
                    "kind"
                ]
                mixed_abi_path = {
                    EXPECTED_FIRST_LINEAR_PARTITION_KIND: args.mixed_multi_abi,
                    EXPECTED_PACKED_PARTITION_KIND: args.mixed_packed_abi,
                    EXPECTED_WHOLE_PARTITION_KIND: args.mixed_whole_abi,
                }.get(partition_kind)
                if mixed_abi_path is None:
                    raise ProfileContractError(
                        "selected candidate has no mixed ABI manifest route"
                    )
                extensions = _validate_v2_extensions(
                    rasterizer_module,
                    head_api,
                    head_backend,
                    candidate_selection,
                    mixed_abi_path,
                    args.head_multi_abi,
                    support_raster_resources,
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
        fusion_partition = None
        fusion_variant = None
        cached_head_parameters = None
        if candidate_selection is not None:
            fusion_partition = candidate_selection["partition"]
            fusion_variant = candidate_selection["variant"]
            cached_head_parameters = fusion_partition.cache_parameters(
                gaussians
            )
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
            if candidate_selection is None:
                task = prepare_pos_head_task(
                    next_context,
                    gaussians,
                    setup_stream,
                    head_weight=cached_weight,
                    head_bias=cached_bias,
                )
                cached_weight = task.head_weight
                cached_bias = task.head_bias
                head_inputs = (task.head_input,)
                head_weights = (task.head_weight,)
                head_biases = (task.head_bias,)
                task_head_names = ("pos",)
            else:
                task = fusion_partition.prepare(
                    next_context,
                    gaussians,
                    setup_stream,
                    cached_head_parameters,
                    None,
                )
                if candidate_selection["legacy_physical_abi"]:
                    head_inputs = (task.head_input,)
                    head_weights = (task.head_weight,)
                    head_biases = (task.head_bias,)
                else:
                    _validate_variant_task_operands(
                        {
                            "task": task,
                            "variant": fusion_variant,
                        },
                        torch,
                    )
                    head_inputs = ()
                    head_weights = ()
                    head_biases = ()
                task_head_names = fusion_variant.selected_heads
            if candidate_selection is None or candidate_selection[
                "legacy_physical_abi"
            ]:
                if not (
                    len(head_inputs)
                    == len(head_weights)
                    == len(head_biases)
                    == len(task_head_names)
                ):
                    raise ProfileContractError(
                        "selected head operand counts are inconsistent"
                    )
                for head_name, head_input, head_weight, head_bias in zip(
                    task_head_names, head_inputs, head_weights, head_biases
                ):
                    if tuple(head_input.shape) != (EXPECTED_GAUSSIANS, 128):
                        raise ProfileContractError(
                            "real {} first-linear input must have shape "
                            "[111525, 128]".format(head_name)
                        )
                    if head_input.dtype != torch.float16:
                        raise ProfileContractError(
                            "real {} head input must be float16".format(head_name)
                        )
                    if head_weight.dtype != torch.float16:
                        raise ProfileContractError(
                            "cached {} head weight must be float16".format(
                                head_name
                            )
                        )
                    if head_bias.dtype != torch.float32:
                        raise ProfileContractError(
                            "cached {} head bias must be float32".format(head_name)
                        )
            bundles.append(
                {
                    "pc": gaussians,
                    "current_index": current_index,
                    "next_index": next_index,
                    "context": current_context,
                    "state": current_state,
                    "task": task,
                    "partition": fusion_partition,
                    "variant": fusion_variant,
                }
            )

        # Context/state/task construction, parameter conversion, and all
        # numerical references are setup.  Synchronize them before warmup.
        torch.cuda.synchronize()
        try:
            if candidate_selection is None or candidate_selection[
                "legacy_physical_abi"
            ]:
                numerics = _check_numerics(
                    bundles,
                    rasterize_state,
                    head_linear_solo,
                    torch,
                    effective_persistent_blocks,
                )
                if candidate_selection is not None:
                    numerics = _promote_legacy_candidate_numerics(numerics)
            else:
                numerics = _check_variant_numerics(
                    bundles,
                    rasterize_state,
                    head_api,
                    torch,
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
        if candidate_selection is None:
            mixed_functions = [
                (
                    lambda bundle=bundle: _mixed_call(
                        bundle, effective_persistent_blocks
                    )
                )
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
        elif candidate_selection["legacy_physical_abi"]:
            mixed_functions = [
                (lambda bundle=bundle: _variant_mixed_call(bundle))
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
        else:
            mixed_functions = [
                (lambda bundle=bundle: _variant_mixed_call(bundle))
                for bundle in bundles
            ]
            solo_head_functions = [
                (
                    lambda bundle=bundle: _variant_standalone_outputs(
                        bundle, head_api
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
            if candidate_selection is None or candidate_selection[
                "legacy_physical_abi"
            ]:
                gptb_functions = [
                    (
                        lambda bundle=bundle: head_linear_gptb(
                            bundle["task"].head_input,
                            bundle["task"].head_weight,
                            bundle["task"].head_bias,
                            effective_persistent_blocks,
                        )
                    )
                    for bundle in bundles
                ]
            else:
                gptb_functions = [
                    (
                        lambda bundle=bundle: _variant_standalone_outputs(
                            bundle, head_api, diagnostic_gptb=True
                        )
                    )
                    for bundle in bundles
                ]
            gptb_summary = _time_cuda_calls(
                gptb_functions, args.warmup, args.repetitions, torch
            )

    _verify_input_snapshot(execution_source_snapshot)
    _verify_input_snapshot(workload_input_snapshot)
    if candidate_selection is not None:
        _verify_bound_file(
            candidate_selection["profile_path"],
            candidate_selection["profile_file_sha256"],
            candidate_selection["profile_file_size_bytes"],
            "candidate profile",
        )
        if candidate_selection["candidate_matrix_path"] is not None:
            _verify_bound_file(
                candidate_selection["candidate_matrix_path"],
                candidate_selection["candidate_matrix_file_sha256"],
                candidate_selection["candidate_matrix_file_size_bytes"],
                "candidate matrix",
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
    head_sample_view_indices = []
    mixed_sample_view_pairs = []
    for _ in range(args.repetitions):
        sample_view_indices.extend(pair[0] for pair in pairs)
        head_sample_view_indices.extend(pair[1] for pair in pairs)
        mixed_sample_view_pairs.extend(
            {
                "current_view_index": pair[0],
                "next_view_index": pair[1],
            }
            for pair in pairs
        )
    solo_raster_summary["view_indices_for_each_sample"] = list(
        sample_view_indices
    )
    solo_head_summary["view_indices_for_each_sample"] = list(
        head_sample_view_indices
    )
    mixed_summary["view_pairs_for_each_sample"] = list(
        mixed_sample_view_pairs
    )
    if gptb_summary is not None:
        gptb_summary["view_indices_for_each_sample"] = list(
            head_sample_view_indices
        )
    variant_metadata = None
    variant_provenance = None
    head_numerics = None
    if candidate_selection is not None:
        variant_metadata = _build_variant_metadata(
            candidate_selection, extensions
        )
        variant_provenance = _build_variant_provenance(
            args,
            dataset,
            pipeline,
            extensions,
            candidate_selection,
            workload_input_snapshot,
            execution_source_snapshot,
        )
        head_numerics = numerics["per_selected_head"]
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
        persistent_blocks=effective_persistent_blocks,
        variant=variant_metadata,
        head_numerics=head_numerics,
        provenance=variant_provenance,
        head_sample_view_indices=head_sample_view_indices,
        mixed_sample_view_pairs=mixed_sample_view_pairs,
    )
    report = {
        "schema_version": (
            VARIANT_SCHEMA_VERSION
            if candidate_selection is not None
            else SCHEMA_VERSION
        ),
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
            "persistent_blocks": effective_persistent_blocks,
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
    if candidate_selection is not None:
        report.update(
            {
                "variant_id": variant_metadata["variant_id"],
                "profile_binding": {
                    "candidate_sha256": variant_metadata[
                        "candidate_sha256"
                    ],
                    "candidate_descriptor_sha256": variant_metadata[
                        "candidate_descriptor_sha256"
                    ],
                    "source_candidate_sha256_claim": variant_metadata.get(
                        "source_candidate_sha256_claim"
                    ),
                    "source_candidate_sha256_claim_verified": variant_metadata.get(
                        "source_candidate_sha256_claim_verified", False
                    ),
                    "candidate_matrix_sha256": variant_metadata.get(
                        "candidate_matrix_sha256"
                    ),
                    "candidate_matrix_file_sha256": variant_metadata.get(
                        "candidate_matrix_file_sha256"
                    ),
                    "source_matrix_sha256_claim": variant_metadata.get(
                        "source_matrix_sha256_claim"
                    ),
                    "source_matrix_sha256_claim_verified": variant_metadata.get(
                        "source_matrix_sha256_claim_verified", False
                    ),
                    "profile_file_sha256": variant_metadata[
                        "profile_file_sha256"
                    ],
                    "profile_sha256": variant_metadata["profile_sha256"],
                    "manifest_sha256": variant_metadata[
                        "manifest_sha256"
                    ],
                },
                "variant": variant_metadata,
                "resources": variant_metadata["resources"],
                "abi": variant_metadata["abi"],
                "provenance": variant_provenance,
                "selected_head_numerics": head_numerics,
            }
        )
        report["parameters"].update(
            {
                "selected_heads": list(candidate_selection["selected_heads"]),
                "worker_groups": candidate_selection["worker_groups"],
                "candidate_profile": candidate_selection["profile_path"],
                "candidate_profile_deployment_enabled": candidate_selection[
                    "deployment_enabled"
                ],
                "qualification_profile": candidate_selection[
                    "qualification_profile"
                ],
                "used_as_deployment": False,
            }
        )
        if uses_multi_head_abi:
            report["measurements"][
                "multi_head_solo_p50_ms"
            ] = solo_head_summary["p50_ms"]
        if gptb_summary is not None and uses_multi_head_abi:
            report["measurements"][
                "multi_head_gptb_p50_ms_diagnostic"
            ] = gptb_summary["p50_ms"]
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
    parser.add_argument(
        "--candidate-profile",
        type=str,
        help=(
            "explicit schema-v2 C0--C4 production-partition profile to "
            "measure in offline qualification mode"
        ),
    )
    parser.add_argument(
        "--candidate-matrix",
        type=str,
        help=(
            "optional sealed candidate matrix proving exact profile member "
            "identity; requires --candidate-profile"
        ),
    )
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
    parser.add_argument(
        "--mixed-multi-abi",
        default=str(
            PROJECT_ROOT
            / "submodules"
            / "depth-diff-gaussian-rasterization"
            / "abi"
            / "tacker_mixed_render_heads_v2.json"
        ),
    )
    parser.add_argument(
        "--head-multi-abi",
        default=str(PROJECT_ROOT / "tacker_ext" / "abi" / "head_linear_v2.json"),
    )
    parser.add_argument(
        "--mixed-packed-abi",
        default=str(
            PROJECT_ROOT
            / "submodules"
            / "depth-diff-gaussian-rasterization"
            / "abi"
            / "tacker_mixed_render_packed_heads_v3.json"
        ),
    )
    parser.add_argument(
        "--mixed-whole-abi",
        default=str(
            PROJECT_ROOT
            / "submodules"
            / "depth-diff-gaussian-rasterization"
            / "abi"
            / "tacker_mixed_render_whole_heads_v4.json"
        ),
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

    # Pre-parse only to locate every config source.  Snapshot it before
    # get_combined_args reads/evaluates cfg_args or load_config executes the
    # explicit Python config.
    args = parser.parse_args(sys.argv[1:])
    device_document = None
    raster_document = None
    leaf_document = None
    try:
        from utils.general_utils import safe_state
        from utils.params_utils import load_config, merge_hparams

        execution_source_snapshot = _capture_execution_source_snapshot(args)
        args = get_combined_args(parser)
        if args.configs:
            args = merge_hparams(args, load_config(args.configs))
        _verify_input_snapshot(execution_source_snapshot)
        execution_source_snapshot["verified_unchanged_after_measurement"] = False
        dataset_args = model.extract(args)
        hyperparam_args = hyperparam.extract(args)
        pipeline_args = pipeline.extract(args)
        workload_input_snapshot = _capture_workload_input_snapshot(
            args, dataset_args
        )
        safe_state(args.quiet)
        device_document, raster_document, leaf_document, report = run_profile(
            args,
            dataset_args,
            hyperparam_args,
            pipeline_args,
            execution_source_snapshot=execution_source_snapshot,
            workload_input_snapshot=workload_input_snapshot,
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
