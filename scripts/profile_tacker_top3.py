#!/usr/bin/env python3
"""Profile the top-ranked Phase-3 candidates with Nsight Systems.

The input is a completed ``benchmark_tacker_fps.py`` report.  Its first three
correctness-valid candidates are profiled in ranking order, including serial
or two-stream baselines when they genuinely rank in the top three.  Every
input and output is SHA-bound, and an interrupted run can resume only after
all completed summaries have been revalidated byte-for-byte.

This module intentionally imports only the Python standard library so its
selection, resume, command, and summary contracts remain CPU-testable.
"""

from __future__ import print_function

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


SCHEMA_VERSION = 1
FPS_REPORT_KIND = "4dgaussians_tacker_fps_benchmark"
REPORT_KIND = "4dgaussians_tacker_top3_nsight"
CHECKPOINT_KIND = "4dgaussians_tacker_top3_nsight_checkpoint"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SUPPORTED_MODES = frozenset(("serial", "two_stream", "tacker"))
CHILD_KIND = "4dgaussians_tacker_render_profile"
REQUIRED_PROVENANCE_SOURCE_FILES = (
    "profile_render.py",
    "configs",
    "gaussian_renderer/__init__.py",
    "gaussian_renderer/tacker_pipeline.py",
    "diff_gaussian_rasterization/__init__.py",
    "diff_gaussian_rasterization._C",
)


class Top3ContractError(ValueError):
    """An input, checkpoint, child, or Nsight artifact was not trustworthy."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical_json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value):
    try:
        return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    except (TypeError, ValueError) as error:
        raise Top3ContractError("document must contain finite JSON: {}".format(error))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json_snapshot(path, label):
    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        canonical_json_bytes(value)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise Top3ContractError("cannot load {} {}: {}".format(label, resolved, error))
    if not isinstance(value, dict):
        raise Top3ContractError("{} must be a JSON object".format(label))
    return value, hashlib.sha256(raw).hexdigest()


def atomic_write_json(path, value):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target.name), suffix=".tmp", dir=str(target.parent)
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
        os.replace(temporary_name, str(target))
        _fsync_parent_directory(target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _fsync_parent_directory(path):
    target = Path(path).expanduser().resolve()
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(target.parent), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _publish_lock(path):
    target = Path(path).expanduser().resolve()
    lock_path = target.parent / ".{}.publish.lock".format(target.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_write_json_no_clobber(path, value):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target.name), suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, str(target))
        os.unlink(temporary_name)
        temporary_name = None
        _fsync_parent_directory(target)
    except Exception:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        raise


def atomic_write_json_compare_and_swap(path, value, expected_sha256):
    target = Path(path).expanduser().resolve()
    with _publish_lock(target):
        if not target.is_file() or sha256_file(target) != expected_sha256:
            raise Top3ContractError("published top-3 report changed before compare-and-swap")
        atomic_write_json(target, value)


def atomic_write_text(path, value):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target.name), suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value or "")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(target))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _finite(mapping, key, label, nonnegative=False, positive=False):
    value = mapping.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise Top3ContractError("{}.{} must be finite".format(label, key))
    result = float(value)
    if nonnegative and result < 0.0:
        raise Top3ContractError("{}.{} must be >= 0".format(label, key))
    if positive and result <= 0.0:
        raise Top3ContractError("{}.{} must be > 0".format(label, key))
    return result


def select_top_candidates(report, limit=3):
    """Return normalized top candidates in formal eligible-ranking order."""

    if type(limit) is not int or limit <= 0:
        raise Top3ContractError("limit must be a positive integer")
    if report.get("schema_version") != 1 or report.get("kind") != FPS_REPORT_KIND:
        raise Top3ContractError("FPS report schema/kind is not supported")
    if report.get("passed") is not True:
        raise Top3ContractError("FPS report must have passed")
    ranking = report.get("eligible_ranking")
    candidates = report.get("candidates")
    if not isinstance(ranking, list) or not ranking:
        raise Top3ContractError("FPS report eligible_ranking must be non-empty")
    if not isinstance(candidates, list):
        raise Top3ContractError("FPS report candidates must be an array")
    by_name = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise Top3ContractError("FPS candidates must be objects")
        name = candidate.get("name")
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
            raise Top3ContractError("FPS candidate has an unsafe name")
        if name in by_name:
            raise Top3ContractError("FPS report has duplicate candidate names")
        mode = candidate.get("execution_mode")
        if mode not in SUPPORTED_MODES:
            raise Top3ContractError("candidate {} has unsupported mode".format(name))
        by_name[name] = candidate
    if len(set(ranking)) != len(ranking) or any(name not in by_name for name in ranking):
        raise Top3ContractError("eligible_ranking is not a unique candidate subset")
    summaries = report.get("summaries")
    if not isinstance(summaries, dict):
        raise Top3ContractError("FPS report summaries must be an object")
    result = []
    for rank, name in enumerate(ranking[:limit], 1):
        candidate = by_name[name]
        summary = summaries.get(name)
        if not isinstance(summary, dict):
            raise Top3ContractError("ranked candidate {} has no summary".format(name))
        median_fps = _finite(
            summary, "median_throughput_fps", "summaries[{}]".format(name), positive=True
        )
        mode = candidate["execution_mode"]
        profile_path = candidate.get("profile_path")
        profile_sha = candidate.get("profile_file_sha256")
        qualification_mode = bool(candidate.get("qualification_mode", False))
        if mode == "tacker":
            if not isinstance(profile_path, str) or not profile_path:
                raise Top3ContractError("Tacker candidate {} has no profile".format(name))
            resolved = Path(profile_path).expanduser().resolve()
            if not resolved.is_file():
                raise Top3ContractError(
                    "Tacker candidate {} profile is missing: {}".format(name, resolved)
                )
            profile_document, observed_profile_sha = load_json_snapshot(
                resolved, "Tacker candidate {} profile".format(name)
            )
            if not isinstance(profile_sha, str) or observed_profile_sha != profile_sha:
                raise Top3ContractError(
                    "Tacker candidate {} profile hash changed".format(name)
                )
            profile_schema_version = profile_document.get("schema_version")
            if profile_schema_version not in (1, 2):
                raise Top3ContractError(
                    "Tacker candidate {} profile schema is unsupported".format(name)
                )
            profile_path = str(resolved)
        else:
            if profile_path is not None or profile_sha is not None or qualification_mode:
                raise Top3ContractError(
                    "baseline candidate {} must not carry a Tacker profile".format(name)
                )
            profile_schema_version = None
        result.append(
            {
                "rank": rank,
                "name": name,
                "execution_mode": mode,
                "median_throughput_fps": median_fps,
                "profile_path": profile_path,
                "profile_file_sha256": profile_sha,
                "profile_schema_version": profile_schema_version,
                "qualification_mode": qualification_mode,
            }
        )
    return result


def candidate_output_dir(root, candidate):
    return Path(root) / "rank{:02d}-{}".format(candidate["rank"], candidate["name"])


def build_nsight_command(
    profile_script,
    candidate,
    output_dir,
    model_path,
    config_path,
    source_path,
    gpu,
    frames,
    iteration,
    workload_name,
):
    """Build one argv-only invocation of the existing Nsight collector."""

    if type(gpu) is not int or gpu < 0:
        raise Top3ContractError("gpu must be a non-negative integer")
    if type(frames) is not int or frames <= 0:
        raise Top3ContractError("frames must be a positive integer")
    if type(iteration) is not int:
        raise Top3ContractError("iteration must be an integer")
    return [
        "/bin/bash",
        str(Path(profile_script).expanduser().resolve()),
        str(Path(model_path).expanduser().resolve()),
        str(Path(config_path).expanduser().resolve()),
        str(Path(output_dir).expanduser().resolve()),
        str(gpu),
        str(frames),
        str(iteration),
        str(Path(source_path).expanduser().resolve()),
        candidate["execution_mode"],
        candidate["profile_path"] or "",
        "1" if candidate["qualification_mode"] else "0",
        workload_name,
    ]


def _require_equal(mapping, key, expected, label):
    if key not in mapping or mapping[key] != expected:
        raise Top3ContractError(
            "{}.{} does not match the profiling contract".format(label, key)
        )


def validate_summary_metadata(summary, candidate, expected):
    """Bind the summarized Nsight run to the selected FPS/source contract."""

    metadata = summary.get("metadata")
    if not isinstance(metadata, dict):
        raise Top3ContractError("Nsight summary metadata must be an object")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "kind": CHILD_KIND,
        "passed": True,
        "model_path": expected["model_path"],
        "source_path": expected["source_path"],
        "iteration": expected["iteration"],
        "split": "test",
        "warmup_frames": 10,
        "profile_frames": expected["frames"],
        "execution_mode": candidate["execution_mode"],
        "actual_execution_mode": candidate["execution_mode"],
        "qualification_mode_requested": bool(candidate["qualification_mode"]),
        "qualification_mode_executed": bool(candidate["qualification_mode"]),
    }
    for key, value in fixed.items():
        _require_equal(metadata, key, value, "Nsight summary metadata")
    expected_workload = (
        expected["workload_name"]
        if candidate["execution_mode"] == "tacker"
        else None
    )
    _require_equal(metadata, "workload_name", expected_workload, "Nsight summary metadata")
    expected_tacker_profile = (
        candidate["profile_path"]
        if candidate["execution_mode"] == "tacker"
        and not candidate["qualification_mode"]
        else None
    )
    expected_qualification_profile = (
        candidate["profile_path"] if candidate["qualification_mode"] else None
    )
    _require_equal(metadata, "tacker_profile", expected_tacker_profile, "Nsight summary metadata")
    _require_equal(
        metadata,
        "qualification_profile",
        expected_qualification_profile,
        "Nsight summary metadata",
    )
    _require_equal(metadata, "active_profile_sha256", candidate["profile_file_sha256"], "Nsight summary metadata")
    _require_equal(metadata, "two_stream_fallback_reason", None, "Nsight summary metadata")
    _require_equal(metadata, "tacker_fallback_reason", None, "Nsight summary metadata")
    expected_tacker_sha = (
        candidate["profile_file_sha256"]
        if candidate["execution_mode"] == "tacker"
        and not candidate["qualification_mode"]
        else None
    )
    expected_qualification_sha = (
        candidate["profile_file_sha256"]
        if candidate["qualification_mode"]
        else None
    )
    _require_equal(metadata, "tacker_profile_sha256", expected_tacker_sha, "Nsight summary metadata")
    _require_equal(metadata, "qualification_profile_sha256", expected_qualification_sha, "Nsight summary metadata")
    profile_hashes = metadata.get("profile_hashes")
    if not isinstance(profile_hashes, dict):
        raise Top3ContractError("Nsight summary metadata.profile_hashes must be an object")
    _require_equal(profile_hashes, "active_profile_sha256", candidate["profile_file_sha256"], "Nsight profile_hashes")
    _require_equal(profile_hashes, "tacker_profile_sha256", expected_tacker_sha, "Nsight profile_hashes")
    _require_equal(profile_hashes, "qualification_profile_sha256", expected_qualification_sha, "Nsight profile_hashes")
    runtime_identity_keys = (
        "profile_manifest_sha256",
        "profile_selection_sha256",
        "selected_candidate_abi_sha256",
    )
    runtime_identity = expected["candidate_runtime_identities"].get(candidate["name"])
    if not isinstance(runtime_identity, dict):
        raise Top3ContractError("FPS candidate runtime identity is missing")
    if candidate["execution_mode"] == "tacker":
        manifest_hash = metadata.get("profile_manifest_sha256")
        if not isinstance(manifest_hash, str) or not re.match(
            r"^[0-9a-f]{64}$", manifest_hash
        ):
            raise Top3ContractError(
                "Nsight summary metadata.profile_manifest_sha256 is invalid"
            )
        _require_equal(
            profile_hashes,
            "profile_manifest_sha256",
            manifest_hash,
            "Nsight profile_hashes",
        )
        for key in (
            "profile_selection_sha256",
            "selected_candidate_abi_sha256",
        ):
            value = metadata.get(key)
            if candidate["profile_schema_version"] == 2:
                valid = isinstance(value, str) and re.match(
                    r"^[0-9a-f]{64}$", value
                )
            else:
                valid = value is None or (
                    isinstance(value, str)
                    and re.match(r"^[0-9a-f]{64}$", value)
                )
            if not valid:
                raise Top3ContractError(
                    "Nsight summary metadata.{} is invalid for schema-v{}"
                    .format(key, candidate["profile_schema_version"])
                )
            _require_equal(profile_hashes, key, value, "Nsight profile_hashes")
        if not isinstance(metadata.get("selected_variant_id"), str) or not metadata["selected_variant_id"]:
            raise Top3ContractError("Nsight selected_variant_id is missing")
        if type(metadata.get("persistent_blocks")) is not int or metadata["persistent_blocks"] <= 0:
            raise Top3ContractError("Nsight persistent_blocks is invalid")
    else:
        for key in runtime_identity_keys + ("selected_variant_id", "persistent_blocks"):
            _require_equal(metadata, key, None, "Nsight summary metadata")
    for key in runtime_identity_keys + ("selected_variant_id", "persistent_blocks"):
        _require_equal(
            metadata,
            key,
            runtime_identity.get(key),
            "Nsight summary metadata versus FPS runtime identity",
        )
    source_files = metadata.get("source_files")
    repository = metadata.get("repository")
    if not isinstance(source_files, dict) or not isinstance(repository, dict):
        raise Top3ContractError("Nsight summary source/repository provenance is missing")
    if set(source_files) != set(REQUIRED_PROVENANCE_SOURCE_FILES):
        raise Top3ContractError("Nsight summary source file set changed")
    for name in REQUIRED_PROVENANCE_SOURCE_FILES:
        digest = source_files[name]
        if not isinstance(digest, str) or not re.match(r"^[0-9a-f]{64}$", digest):
            raise Top3ContractError("Nsight source {} has no SHA-256".format(name))
    if source_files["configs"] != expected["config_sha256"]:
        raise Top3ContractError("Nsight config provenance changed")
    if repository.get("source_files") != source_files:
        raise Top3ContractError("Nsight repository source_files disagree")
    stable_provenance = {
        "environment": {
            "gpu_name": metadata.get("gpu_name"),
            "cuda_runtime": metadata.get("cuda_runtime"),
            "pytorch_version": metadata.get("pytorch_version"),
        },
        "repository_commit": repository.get("commit"),
        "repository_commit_source": repository.get("commit_source"),
        "repository_dirty": repository.get("dirty"),
        "submodules": repository.get("submodules"),
        "source_files": source_files,
    }
    if stable_provenance != expected["stable_provenance"]:
        raise Top3ContractError("Nsight source/environment provenance differs from FPS report")
    return metadata


def extract_nsight_diagnostics(summary, candidate, expected=None):
    """Normalize the fields used to compare critical paths and stragglers."""

    if not isinstance(summary, dict):
        raise Top3ContractError("Nsight summary must be an object")
    frame_count = summary.get("frame_count")
    if type(frame_count) is not int or frame_count <= 0:
        raise Top3ContractError("Nsight summary frame_count must be positive")
    if expected is not None and frame_count != expected.get("frames"):
        raise Top3ContractError(
            "Nsight summary frame_count does not match the requested capture"
        )
    render_loop = summary.get("render_loop")
    frame_nvtx = summary.get("frame_nvtx")
    stages = summary.get("main_stages")
    kernels = summary.get("kernels")
    cuda_api = summary.get("cuda_api")
    if not all(isinstance(value, dict) for value in (render_loop, frame_nvtx, stages, kernels, cuda_api)):
        raise Top3ContractError("Nsight summary is missing diagnostic sections")
    fps = _finite(render_loop, "fps", "render_loop", positive=True)
    median_ms = _finite(frame_nvtx, "median_ms", "frame_nvtx", positive=True)
    max_ms = _finite(frame_nvtx, "max_ms", "frame_nvtx", positive=True)
    stddev_ms = _finite(frame_nvtx, "stddev_ms", "frame_nvtx", nonnegative=True)
    stage_values = []
    for name, value in stages.items():
        if not isinstance(name, str) or not name or not isinstance(value, dict):
            raise Top3ContractError("main_stages entries must be objects")
        stage_values.append(
            (name, _finite(value, "gpu_projected_ms_per_frame", "main_stages", nonnegative=True))
        )
    if not stage_values:
        raise Top3ContractError(
            "Nsight summary main_stages must contain a critical-path stage"
        )
    stage_values.sort(key=lambda item: (-item[1], item[0]))
    categories = kernels.get("categories")
    if not isinstance(categories, dict):
        raise Top3ContractError("kernels.categories must be an object")
    category_rows = []
    for name, value in categories.items():
        if not isinstance(value, dict):
            raise Top3ContractError("kernel categories must be objects")
        category_rows.append(
            (name, _finite(value, "ms_per_frame", "kernel category", nonnegative=True))
        )
    category_rows.sort(key=lambda item: (-item[1], item[0]))
    sync = cuda_api.get("stream_synchronize")
    launch = cuda_api.get("kernel_launch")
    if not isinstance(sync, dict) or not isinstance(launch, dict):
        raise Top3ContractError("cuda_api sync/launch diagnostics are missing")
    if expected is not None:
        validate_summary_metadata(summary, candidate, expected)
    else:
        metadata = summary.get("metadata")
        if not isinstance(metadata, dict):
            raise Top3ContractError("Nsight summary metadata must be an object")
        expected_mode = candidate["execution_mode"]
        if metadata.get("execution_mode") != expected_mode:
            raise Top3ContractError("Nsight metadata execution mode changed")
        if metadata.get("actual_execution_mode") != expected_mode:
            raise Top3ContractError("Nsight candidate fell back during profiling")
    return {
        "frame_count": frame_count,
        "nsight_render_fps": fps,
        "benchmark_median_fps": candidate["median_throughput_fps"],
        "critical_path_proxy": {
            "method": "largest NVTX GPU-projected main-stage time per frame",
            "stage": stage_values[0][0] if stage_values else None,
            "gpu_projected_ms_per_frame": stage_values[0][1] if stage_values else None,
        },
        "straggler": {
            "median_frame_ms": median_ms,
            "max_frame_ms": max_ms,
            "stddev_frame_ms": stddev_ms,
            "max_to_median_ratio": max_ms / median_ms,
        },
        "kernel_launches": {
            "total": int(_finite(kernels, "launches", "kernels", nonnegative=True)),
            "per_frame": _finite(kernels, "launches_per_frame", "kernels", nonnegative=True),
            "cuda_api_calls": int(_finite(launch, "calls", "cuda_api.kernel_launch", nonnegative=True)),
        },
        "synchronization": {
            "stream_synchronize_calls": int(_finite(sync, "calls", "cuda_api.stream_synchronize", nonnegative=True)),
            "calls_per_frame": _finite(sync, "calls_per_frame", "cuda_api.stream_synchronize", nonnegative=True),
            "total_ms": _finite(sync, "total_ms", "cuda_api.stream_synchronize", nonnegative=True),
        },
        "dominant_kernel_categories": [
            {"name": name, "ms_per_frame": value} for name, value in category_rows[:5]
        ],
        "idle_or_gap_proxy": {
            "method": "render-loop wall time minus summed CUDA kernel time; overlap can make this only diagnostic",
            "ms_per_frame": max(
                0.0,
                _finite(render_loop, "ms_per_frame", "render_loop", positive=True)
                - _finite(kernels, "ms_per_frame", "kernels", nonnegative=True),
            ),
        },
    }


def _identity(
    fps_path,
    fps_sha,
    candidates,
    profile_script,
    model_path,
    config_path,
    source_path,
    gpu,
    frames,
    iteration,
    workload_name,
):
    script = Path(profile_script).resolve()
    project_root = script.parents[1]
    payload = {
        "driver": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "fps_report": {"path": str(Path(fps_path).resolve()), "sha256": fps_sha},
        "candidates": candidates,
        "profile_script": {
            "path": str(script),
            "sha256": sha256_file(script),
        },
        "model_path": str(Path(model_path).resolve()),
        "config": {
            "path": str(Path(config_path).resolve()),
            "sha256": sha256_file(config_path),
        },
        "source_path": str(Path(source_path).resolve()),
        "gpu": gpu,
        "frames": frames,
        "iteration": iteration,
        "workload_name": workload_name,
        "orchestration_sources": _orchestration_source_identity(project_root),
        "workload_files": _workload_file_identity(
            model_path, source_path, iteration
        ),
    }
    return {"sha256": sha256_json(payload), "payload": payload}


def _orchestration_source_identity(project_root):
    root = Path(project_root).resolve()
    candidates = {
        "profile_render.py": root / "profile_render.py",
        "summarize_nsight_stats.py": root / "scripts" / "summarize_nsight_stats.py",
        "gaussian_renderer/__init__.py": root / "gaussian_renderer" / "__init__.py",
        "gaussian_renderer/tacker_pipeline.py": root / "gaussian_renderer" / "tacker_pipeline.py",
        "diff_gaussian_rasterization/__init__.py": root
        / "submodules"
        / "depth-diff-gaussian-rasterization"
        / "diff_gaussian_rasterization"
        / "__init__.py",
    }
    result = {}
    for name, path in sorted(candidates.items()):
        result[name] = (
            {"path": str(path), "sha256": sha256_file(path)}
            if path.is_file()
            else None
        )
    raster_root = (
        root
        / "submodules"
        / "depth-diff-gaussian-rasterization"
        / "diff_gaussian_rasterization"
    )
    result["diff_gaussian_rasterization._C"] = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in sorted(raster_root.glob("_C*.so"))
        if path.is_file()
    ]
    return result


def _workload_file_identity(model_path, source_path, iteration):
    model_root = Path(model_path).expanduser().resolve()
    source_root = Path(source_path).expanduser().resolve()
    iteration_root = model_root / "point_cloud" / "iteration_{}".format(iteration)
    paths = {
        "cfg_args": model_root / "cfg_args",
        "point_cloud.ply": iteration_root / "point_cloud.ply",
        "deformation.pth": iteration_root / "deformation.pth",
        "deformation_table.pth": iteration_root / "deformation_table.pth",
        "poses_bounds.npy": source_root / "poses_bounds.npy",
    }
    return {
        name: (
            {"path": str(path), "sha256": sha256_file(path)}
            if path.is_file()
            else None
        )
        for name, path in sorted(paths.items())
    }


def _expected_summary_contract(fps_report, identity, candidates):
    contract = fps_report.get("contract")
    provenance = fps_report.get("stable_provenance")
    if not isinstance(contract, dict) or not isinstance(provenance, dict):
        raise Top3ContractError("FPS report lacks contract/stable provenance")
    payload = identity["payload"]
    expected = {
        "model_path": payload["model_path"],
        "source_path": payload["source_path"],
        "iteration": payload["iteration"],
        "frames": payload["frames"],
        "workload_name": payload["workload_name"],
        "config_sha256": payload["config"]["sha256"],
        "stable_provenance": provenance,
        "candidate_runtime_identities": {},
    }
    for key in ("model_path", "source_path", "iteration", "workload_name"):
        if contract.get(key) != expected[key]:
            raise Top3ContractError("Nsight invocation {} differs from FPS contract".format(key))
    runs = fps_report.get("runs")
    if not isinstance(runs, list):
        raise Top3ContractError("FPS report lacks auditable run records")
    runtime_keys = (
        "profile_manifest_sha256",
        "profile_selection_sha256",
        "selected_variant_id",
        "selected_candidate_abi_sha256",
        "persistent_blocks",
    )
    for candidate in candidates:
        identities = []
        for run in runs:
            if not isinstance(run, dict) or run.get("candidate_name") != candidate["name"]:
                continue
            metrics = run.get("metrics")
            if run.get("passed") is not True or not isinstance(metrics, dict):
                continue
            identity_row = {key: metrics.get(key) for key in runtime_keys}
            if identity_row not in identities:
                identities.append(identity_row)
        if len(identities) != 1:
            raise Top3ContractError(
                "FPS runtime identity is missing or unstable for {}".format(
                    candidate["name"]
                )
            )
        expected["candidate_runtime_identities"][candidate["name"]] = identities[0]
    return expected


def _checkpoint(path, identity, report):
    atomic_write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": CHECKPOINT_KIND,
            "identity": identity,
            "report": report,
        },
    )


PROFILE_RECORD_FIELDS = frozenset(
    (
        "candidate",
        "command",
        "output_dir",
        "summary_path",
        "stdout_path",
        "stderr_path",
        "passed",
        "returncode",
        "error",
        "stdout_sha256",
        "stderr_sha256",
        "summary_sha256",
        "profile_metadata_sha256",
        "raw_artifacts",
        "diagnostics",
    )
)


def _make_profile_record(
    candidate,
    root,
    script,
    model_path,
    config,
    source_path,
    gpu,
    frames,
    iteration,
    workload_name,
):
    candidate_dir = candidate_output_dir(root, candidate).resolve()
    mode = candidate["execution_mode"]
    return {
        "candidate": candidate,
        "command": build_nsight_command(
            script,
            candidate,
            candidate_dir,
            model_path,
            config,
            source_path,
            gpu,
            frames,
            iteration,
            workload_name,
        ),
        "output_dir": str(candidate_dir),
        "summary_path": str(candidate_dir / "{}_summary.json".format(mode)),
        "stdout_path": str(candidate_dir / "profile.stdout.txt"),
        "stderr_path": str(candidate_dir / "profile.stderr.txt"),
        "passed": False,
        "returncode": None,
        "error": None,
        "stdout_sha256": None,
        "stderr_sha256": None,
        "summary_sha256": None,
        "profile_metadata_sha256": None,
        "raw_artifacts": None,
        "diagnostics": None,
    }


def _raw_artifact_manifest(candidate_dir):
    """Hash every regular child-created file, rejecting links and escapes."""

    root = Path(candidate_dir).resolve()
    result = {}
    if not root.is_dir():
        raise Top3ContractError("Nsight child did not create its output directory")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise Top3ContractError("Nsight output must not contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise Top3ContractError("Nsight output contains a non-regular artifact")
        relative = path.relative_to(root).as_posix()
        if relative in ("profile.stdout.txt", "profile.stderr.txt"):
            continue
        result[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def _required_raw_names(candidate):
    mode = candidate["execution_mode"]
    return {
        "4dgs_render_{}.nsys-rep".format(mode),
        "4dgs_render_{}_stats.csv".format(mode),
        "{}_profile_metadata.json".format(mode),
        "{}_summary.json".format(mode),
    }


def _validate_raw_artifacts(candidate_dir, candidate, expected_manifest=None):
    manifest = _raw_artifact_manifest(candidate_dir)
    missing = sorted(_required_raw_names(candidate) - set(manifest))
    if missing:
        raise Top3ContractError(
            "Nsight raw artifacts are missing: {}".format(", ".join(missing))
        )
    if expected_manifest is not None and manifest != expected_manifest:
        raise Top3ContractError("recovered Nsight raw artifacts changed")
    return manifest


def _prepare_candidate_attempt(candidate_dir, allow_cleanup):
    path = Path(candidate_dir).resolve()
    if path.exists() or path.is_symlink():
        if not allow_cleanup:
            raise Top3ContractError("candidate output already exists: {}".format(path))
        if path.is_symlink() or not path.is_dir():
            raise Top3ContractError("candidate output is not a safe directory: {}".format(path))
        shutil.rmtree(str(path))


def run_top3_profile(
    fps_report_path,
    output_dir,
    report_path,
    profile_script,
    model_path,
    config_path,
    source_path,
    gpu=0,
    frames=50,
    iteration=14000,
    workload_name="flame_steak",
    limit=3,
    resume=False,
    runner=None,
):
    """Execute or resume the ranking-bound top candidate Nsight suite."""

    runner = subprocess.run if runner is None else runner
    fps_report, fps_sha = load_json_snapshot(fps_report_path, "FPS report")
    candidates = select_top_candidates(fps_report, limit=limit)
    script = Path(profile_script).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    for path, label in ((script, "profile script"), (config, "config")):
        if not path.is_file():
            raise Top3ContractError("{} does not exist: {}".format(label, path))
    root = Path(output_dir).expanduser().resolve()
    report_target = Path(report_path).expanduser().resolve()
    existing_report_sha = None
    if report_target.exists():
        if not resume:
            raise Top3ContractError(
                "refusing to overwrite existing report: {}".format(report_target)
            )
        existing_report, existing_report_sha = load_json_snapshot(
            report_target, "existing top-3 report"
        )
        if existing_report.get("passed") is True:
            raise Top3ContractError("a completed top-3 report cannot be resumed")
    checkpoint_path = root / "top3.checkpoint.json"
    identity = _identity(
        Path(fps_report_path).expanduser().resolve(),
        fps_sha,
        candidates,
        script,
        model_path,
        config,
        source_path,
        gpu,
        frames,
        iteration,
        workload_name,
    )
    expected_summary = _expected_summary_contract(fps_report, identity, candidates)
    if existing_report_sha is not None and (
        existing_report.get("kind") != REPORT_KIND
        or existing_report.get("identity") != identity
    ):
        raise Top3ContractError("existing top-3 report identity does not match")
    recovered = []
    if resume:
        if not checkpoint_path.is_file():
            raise Top3ContractError("resume checkpoint is missing")
        checkpoint, _ = load_json_snapshot(checkpoint_path, "top-3 checkpoint")
        if checkpoint.get("kind") != CHECKPOINT_KIND or checkpoint.get("identity") != identity:
            raise Top3ContractError("top-3 checkpoint identity does not match")
        saved_report = checkpoint.get("report")
        if not isinstance(saved_report, dict) or not isinstance(saved_report.get("profiles"), list):
            raise Top3ContractError("top-3 checkpoint report is invalid")
        if existing_report_sha is not None and existing_report != saved_report:
            raise Top3ContractError(
                "existing top-3 report differs from its checkpoint"
            )
        for index, saved in enumerate(saved_report["profiles"]):
            if index >= len(candidates):
                raise Top3ContractError("top-3 checkpoint candidate order changed")
            expected_record = _make_profile_record(
                candidates[index], root, script, model_path, config, source_path,
                gpu, frames, iteration, workload_name
            )
            if not isinstance(saved, dict) or set(saved) != PROFILE_RECORD_FIELDS:
                raise Top3ContractError("top-3 checkpoint record schema changed")
            for key in (
                "candidate", "command", "output_dir", "summary_path",
                "stdout_path", "stderr_path",
            ):
                if saved[key] != expected_record[key]:
                    raise Top3ContractError(
                        "top-3 checkpoint {} changed".format(key)
                    )
            if type(saved.get("passed")) is not bool:
                raise Top3ContractError("top-3 checkpoint passed must be boolean")
            if saved.get("passed") is not True:
                if index != len(saved_report["profiles"]) - 1:
                    raise Top3ContractError("top-3 failure must be the last checkpoint entry")
                if saved.get("error") is None:
                    raise Top3ContractError("top-3 failed checkpoint has no error")
                break
            if saved.get("returncode") != 0 or saved.get("error") is not None:
                raise Top3ContractError("top-3 recovered success status changed")
            summary_path = Path(saved.get("summary_path", ""))
            if not summary_path.is_file() or saved.get("summary_sha256") != sha256_file(summary_path):
                raise Top3ContractError("recovered Nsight summary changed")
            for hash_key, path_key in (
                ("stdout_sha256", "stdout_path"),
                ("stderr_sha256", "stderr_path"),
            ):
                artifact_path = Path(saved.get(path_key, ""))
                if (
                    not artifact_path.is_file()
                    or saved.get(hash_key) != sha256_file(artifact_path)
                ):
                    raise Top3ContractError("recovered Nsight log changed")
            manifest = _validate_raw_artifacts(
                saved["output_dir"], candidates[index], saved.get("raw_artifacts")
            )
            metadata_path = Path(saved["output_dir"]) / "{}_profile_metadata.json".format(
                candidates[index]["execution_mode"]
            )
            metadata, metadata_sha = load_json_snapshot(
                metadata_path, "recovered Nsight profile metadata"
            )
            if saved.get("profile_metadata_sha256") != metadata_sha:
                raise Top3ContractError("recovered Nsight profile metadata changed")
            summary, _ = load_json_snapshot(summary_path, "recovered Nsight summary")
            if summary.get("metadata") != metadata:
                raise Top3ContractError("Nsight summary metadata differs from raw metadata")
            if extract_nsight_diagnostics(
                summary, candidates[index], expected_summary
            ) != saved.get("diagnostics"):
                raise Top3ContractError("recovered Nsight diagnostics changed")
            recovered.append(dict(saved))
    else:
        root.mkdir(parents=True, exist_ok=False)

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "generated_at_utc": utc_now(),
        "passed": False,
        "identity": identity,
        "fps_report_sha256": fps_sha,
        "selection_policy": "first N candidates in formal eligible_ranking, including baselines",
        "requested_limit": limit,
        "selected_candidates": candidates,
        "resume": {"enabled": bool(resume), "recovered_count": len(recovered)},
        "profiles": recovered,
        "rank1_comparisons": [],
        "errors": [],
    }
    # Persist an empty, identity-bound prefix before the first Nsight child.
    # A SIGINT/host loss during that child can otherwise leave ``root`` and a
    # partial candidate directory behind without any checkpoint that permits a
    # safe ``--resume``.  Resume will validate this document and remove only
    # the deterministic partial directory for the missing first candidate.
    _checkpoint(checkpoint_path, identity, report)
    for candidate in candidates[len(recovered):]:
        candidate_dir = candidate_output_dir(root, candidate)
        _prepare_candidate_attempt(candidate_dir, allow_cleanup=bool(resume))
        record = _make_profile_record(
            candidate, root, script, model_path, config, source_path,
            gpu, frames, iteration, workload_name
        )
        command = record["command"]
        report["profiles"].append(record)
        try:
            completed = runner(
                command,
                cwd=str(script.parents[1]),
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
            record["returncode"] = int(completed.returncode)
            atomic_write_text(record["stdout_path"], completed.stdout or "")
            atomic_write_text(record["stderr_path"], completed.stderr or "")
            record["stdout_sha256"] = sha256_file(record["stdout_path"])
            record["stderr_sha256"] = sha256_file(record["stderr_path"])
            if completed.returncode != 0:
                raise Top3ContractError(
                    "Nsight child exited with status {}".format(completed.returncode)
                )
            summary_path = Path(record["summary_path"])
            summary, summary_sha = load_json_snapshot(summary_path, "Nsight summary")
            if summary.get("frame_count") != frames:
                raise Top3ContractError("Nsight summary frame_count changed")
            metadata_path = candidate_dir / "{}_profile_metadata.json".format(
                candidate["execution_mode"]
            )
            profile_metadata, metadata_sha = load_json_snapshot(
                metadata_path, "Nsight profile metadata"
            )
            if summary.get("metadata") != profile_metadata:
                raise Top3ContractError("Nsight summary metadata differs from raw metadata")
            record["diagnostics"] = extract_nsight_diagnostics(
                summary, candidate, expected_summary
            )
            record["summary_sha256"] = summary_sha
            record["profile_metadata_sha256"] = metadata_sha
            record["raw_artifacts"] = _validate_raw_artifacts(
                candidate_dir, candidate
            )
            record["passed"] = True
        except Exception as error:
            record["error"] = str(error)
            report["errors"].append(
                "candidate {}: {}".format(candidate["name"], error)
            )
            _checkpoint(checkpoint_path, identity, report)
            break
        _checkpoint(checkpoint_path, identity, report)
    report["passed"] = bool(
        not report["errors"]
        and len(report["profiles"]) == len(candidates)
        and all(item.get("passed") is True for item in report["profiles"])
    )
    if report["passed"]:
        baseline = report["profiles"][0]["diagnostics"]
        baseline_fps = baseline["nsight_render_fps"]
        baseline_launches = baseline["kernel_launches"]["per_frame"]
        for record in report["profiles"]:
            diagnostics = record["diagnostics"]
            report["rank1_comparisons"].append(
                {
                    "candidate": record["candidate"]["name"],
                    "rank1_candidate": report["profiles"][0]["candidate"]["name"],
                    "nsight_fps_ratio": diagnostics["nsight_render_fps"] / baseline_fps,
                    "kernel_launches_per_frame_delta": (
                        diagnostics["kernel_launches"]["per_frame"] - baseline_launches
                    ),
                }
            )
    _checkpoint(checkpoint_path, identity, report)
    if existing_report_sha is not None:
        atomic_write_json_compare_and_swap(
            report_target, report, existing_report_sha
        )
    else:
        try:
            atomic_write_json_no_clobber(report_target, report)
        except FileExistsError:
            raise Top3ContractError("top-3 report was created during profiling")
    return report


def _parser():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--profile-script", default=str(project_root / "scripts" / "profile_nsight.sh")
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--iteration", type=int, default=14000)
    parser.add_argument("--workload-name", default="flame_steak")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        report = run_top3_profile(
            args.fps_report,
            args.output_dir,
            args.report,
            args.profile_script,
            args.model_path,
            args.config,
            args.source_path,
            gpu=args.gpu,
            frames=args.frames,
            iteration=args.iteration,
            workload_name=args.workload_name,
            limit=args.limit,
            resume=args.resume,
        )
    except Exception as error:
        print("top-3 Nsight profiling failed closed: {}".format(error), file=os.sys.stderr)
        return 1
    if not report["passed"]:
        for error in report["errors"]:
            print(error, file=os.sys.stderr)
        return 1
    print(
        "top-{} Nsight profiling passed; report: {}".format(
            len(report["profiles"]), Path(args.report).expanduser().resolve()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
