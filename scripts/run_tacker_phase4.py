#!/usr/bin/env python3
"""Run the fail-closed, resumable Phase-4 Tacker qualification.

Phase 4 is a consumer of one already sealed Phase-3.1 run.  It never creates,
extends, ranks a newly generated matrix, or materializes new qualification
profiles.  The only Tacker challengers it may execute are the exact qualified
finalists recovered by ``verify_tacker_phase31.py``.

The coordinator launches the real build, CUDA, profiler, quality, benchmark,
and admission entry points.  JSON produced here is orchestration/provenance
metadata assembled from those real subprocess artifacts; this program never
fabricates CUDA measurements.  ``--dry-run`` is the CPU-only exception: it
prints a deterministic plan and invokes no subprocesses.
"""

from __future__ import print_function

import argparse
from copy import deepcopy
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = 1
STATE_KIND = "4dgaussians_tacker_phase4_checkpoint"
REPORT_KIND = "4dgaussians_tacker_phase4_qualification"
DRY_RUN_KIND = "4dgaussians_tacker_phase4_dry_run_plan"
WORKLOAD_KIND = "4dgaussians_tacker_phase4_workload"
PHASE31_VERIFY_KIND = "4dgaussians_tacker_phase31_postflight_verification"

EXPECTED_PHASE31_IDENTITY = (
    "0f16b7250611891db52cc59dada0f8bd77c40d735c26f98aed7c81047c14d8a2"
)
EXPECTED_PHASE31_MATRIX = (
    "aa45c9dce881a5a71c9d86130fcd6b12070ee2773b83b44a538f2147bbde8577"
)
EXPECTED_PHASE31_FORMAL_SET = (
    "7c592726cb2edfeb9ea7c030b569d213b179c0917d31a9d1b4a56007abc54102"
)
EXPECTED_PHASE31_SELECTION = (
    "1a91a02c2605ae442817bb060b7956ead5d3b940146d274b94a85e00849ec126"
)
EXPECTED_PHASE31_RUN_REPORT_FILE = (
    "b24699941748fff134830bf59e0a0001bc64ac419ed119d10481bb894b956ea9"
)

BASELINE_NAMES = ("serial", "two_stream", "current_tacker")
SEQUENCE_LENGTHS = (1, 2, 50)
FORMAL_TRIALS = 10
FORMAL_FRAMES = 50
FORMAL_WARMUP = 10
FORMAL_SCHEDULE = "abba"
FORMAL_SEED = 0
QUALITY_FRAMES = 50

STAGES = (
    "preflight",
    "build-and-cuda",
    "verify-phase31-seal",
    "finalist-resources-numerics",
    "quality-50-view",
    "formal-10x50-abba",
    "selection-admission",
    "sequence-1-2-50-long",
    "enabled-profile-rerun",
    "fallback-smoke",
    "generalization-workload-1",
    "generalization-workload-2",
    "canary-release-rollback",
)

FORBIDDEN_PROGRAM_BASENAMES = {
    "tacker_autotune.py",
    "generate_tacker_phase2_profiles.py",
    "run_tacker_phase3.py",
    "run_tacker_phase31.py",
}
FORBIDDEN_ARGUMENTS = {
    "phase31-matrix",
    "phase31-c3",
    "phase31-c4",
    "profiles",
    "matrix",
    "extend",
    "generate",
}

RUNTIME_SOURCE_RELATIVE_PATHS = (
    "src/CMakeLists.txt",
    "src/runtime/Scheduler.cc",
    "src/runtime/Scheduler.h",
    "src/runtime/Registry.cc",
    "src/runtime/Registry.h",
    "src/runtime/TaskGraph.cc",
    "src/runtime/TaskGraph.h",
    "src/runtime/ExecutionBackend.h",
    "src/runtime/CudaExecutionBackend.cc",
    "src/runtime/CudaExecutionBackend.h",
    "src/runtime/Kernel.cc",
    "src/runtime/Kernel.h",
    "src/runtime/Runtime.h",
    "src/runtime/Types.h",
    "tests/runtime/runtime_tests.cc",
)

PROJECT_SOURCE_RELATIVE_PATHS = (
    "scripts/run_tacker_phase4.py",
    "scripts/run_tacker_qualification.sh",
    "scripts/run_profile_render_sealed.py",
    "profile_render.py",
    "profile_tacker_leaves.py",
    "scripts/benchmark_tacker_fps.py",
    "scripts/benchmark_tacker_admission.py",
    "scripts/validate_tacker_modes.py",
    "scripts/verify_tacker_phase31.py",
    "gaussian_renderer/__init__.py",
    "gaussian_renderer/tacker_pipeline.py",
    "tacker_ext/abi/head_linear_v1.json",
    "tacker_ext/abi/head_linear_v2.json",
    "submodules/depth-diff-gaussian-rasterization/abi/tacker_mixed_render_head_v1.json",
    "submodules/depth-diff-gaussian-rasterization/abi/tacker_mixed_render_heads_v2.json",
    "submodules/depth-diff-gaussian-rasterization/abi/tacker_mixed_render_packed_heads_v3.json",
    "submodules/depth-diff-gaussian-rasterization/abi/tacker_mixed_render_whole_heads_v4.json",
    "tacker_ext/setup.py",
    "tacker_ext/csrc/bindings.cpp",
    "tacker_ext/csrc/head_linear.cu",
    "tacker_ext/csrc/head_linear_v2.cu",
    "tacker_ext/include/head_linear.h",
    "tacker_ext/include/head_linear_device.cuh",
    "tacker_ext/include/head_linear_kernels.cuh",
    "tacker_ext/include/head_linear_v2.h",
    "tacker_ext/include/head_linear_v2_device.cuh",
    "tacker_ext/include/head_linear_v2_kernels.cuh",
    "tacker_ext/tacker_4dgs_head/__init__.py",
    "submodules/depth-diff-gaussian-rasterization/setup.py",
    "submodules/depth-diff-gaussian-rasterization/ext.cpp",
    "submodules/depth-diff-gaussian-rasterization/rasterize_points.cu",
    "submodules/depth-diff-gaussian-rasterization/rasterize_points.h",
    "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h",
    "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/backward.cu",
    "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/backward.h",
    "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/config.h",
    "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/forward.cu",
    "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/forward.h",
    "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h",
    "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu",
    "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h",
    "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/tacker_forward.cuh",
    "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/tacker_mixed.cu",
    "submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/tacker_mixed.h",
    "submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py",
    "submodules/simple-knn/setup.py",
    "submodules/simple-knn/ext.cpp",
    "submodules/simple-knn/simple_knn.cu",
    "submodules/simple-knn/simple_knn.h",
    "submodules/simple-knn/spatial.cu",
    "submodules/simple-knn/spatial.h",
    "tests/test_run_tacker_phase4.py",
    "tests/test_tacker_qualification_script.py",
    "tests/test_profile_render_modes.py",
    "tests/test_tacker_admission.py",
    "tests/test_tacker_pipeline.py",
    "tests/test_tacker_leaf_profile.py",
    "tests/test_benchmark_tacker_fps.py",
    "tests/test_validate_tacker_modes.py",
    "tacker_ext/tests/test_contract.py",
    "tacker_ext/tests/test_head_linear_cuda.py",
    "tacker_ext/tests/test_head_linear_v2_cuda.py",
    "tacker_ext/tests/test_v2_contract.py",
    "tacker_ext/tests/test_v2_python_api.py",
    "tacker_ext/tests/test_v2_validation.py",
    "tacker_ext/tests/test_validation.py",
    "submodules/depth-diff-gaussian-rasterization/tests/test_stream_aware_legacy_cuda.py",
    "submodules/depth-diff-gaussian-rasterization/tests/test_tacker_mixed_contract.py",
    "submodules/depth-diff-gaussian-rasterization/tests/test_tacker_mixed_cuda.py",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FIRST_PARTY_SOURCE_DIRECTORIES = (
    "arguments",
    "gaussian_renderer",
    "scene",
    "utils",
    "lpipsPyTorch",
)


class Phase4Error(RuntimeError):
    """A Phase-4 orchestration, measurement, or publication gate failed."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical_json_bytes(value):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Phase4Error("value is not finite canonical JSON: {}".format(error))


def sha256_json(value, domain=None):
    payload = value if domain is None else {"domain": domain, "payload": value}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path):
    _raw, _target, _facts, digest = _stable_file_snapshot(
        path, "SHA-256 input", retain_bytes=False
    )
    return digest


def _file_stat_identity(facts):
    return (
        int(facts.st_dev),
        int(facts.st_ino),
        int(facts.st_mode),
        int(facts.st_size),
        int(getattr(facts, "st_mtime_ns", int(facts.st_mtime * 1e9))),
        int(getattr(facts, "st_ctime_ns", int(facts.st_ctime * 1e9))),
    )


def _stable_file_snapshot(path, label, executable=False, retain_bytes=True):
    """Snapshot one regular non-symlink file through a stable descriptor.

    Large model artifacts are hashed without retaining their contents.  JSON
    callers opt into the byte payload while sharing the same identity checks.
    """

    input_path = Path(path).expanduser().absolute()
    descriptor = None
    try:
        before_path = os.lstat(str(input_path))
        if stat.S_ISLNK(before_path.st_mode):
            raise Phase4Error("{} must not be a symlink: {}".format(label, input_path))
        if not stat.S_ISREG(before_path.st_mode):
            raise Phase4Error("{} must be a regular file: {}".format(label, input_path))
        resolved_before = input_path.resolve(strict=True)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(input_path), flags)
        before_fd = os.fstat(descriptor)
        digest = hashlib.sha256()
        chunks = [] if retain_bytes else None
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
        after_fd = os.fstat(descriptor)
        after_path = os.lstat(str(input_path))
        resolved_after = input_path.resolve(strict=True)
    except Phase4Error:
        raise
    except OSError as error:
        raise Phase4Error("{} is unavailable: {}".format(label, error))
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identities = {
        _file_stat_identity(before_path),
        _file_stat_identity(before_fd),
        _file_stat_identity(after_fd),
        _file_stat_identity(after_path),
    }
    if len(identities) != 1 or resolved_before != resolved_after:
        raise Phase4Error("{} changed while being snapshotted: {}".format(label, input_path))
    if executable and not os.access(str(input_path), os.X_OK):
        raise Phase4Error("{} is not executable: {}".format(label, input_path))
    raw = b"".join(chunks) if chunks is not None else None
    return raw, resolved_after, after_fd, digest.hexdigest()


def _stable_file_bytes(path, label, executable=False):
    raw, target, facts, _digest = _stable_file_snapshot(
        path, label, executable=executable, retain_bytes=True
    )
    return raw, target, facts


def _regular_file(path, label, executable=False):
    _raw, target, _facts, _digest = _stable_file_snapshot(
        path, label, executable=executable, retain_bytes=False
    )
    return target


def file_artifact(path, label="artifact"):
    _raw, target, facts, digest = _stable_file_snapshot(
        path, label, retain_bytes=False
    )
    return {
        "path": str(target),
        "sha256": digest,
        "size_bytes": facts.st_size,
    }


def load_json(path, label="JSON"):
    raw, target, _facts = _stable_file_bytes(path, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise Phase4Error("cannot load {} {}: {}".format(label, target, error))
    if not isinstance(value, dict):
        raise Phase4Error("{} must be a JSON object".format(label))
    return value


def _atomic_write_json(path, value, replace=False):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical_json_bytes(value)
    if not replace and target.exists():
        raise Phase4Error("refusing to overwrite existing artifact: {}".format(target))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target.name), suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and target.exists():
            raise Phase4Error(
                "refusing to overwrite concurrently created artifact: {}".format(target)
            )
        os.replace(temporary_name, str(target))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _is_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _require_sha256(value, label):
    if not isinstance(value, str) or not SHA256_RE.match(value):
        raise Phase4Error("{} must be a lowercase SHA-256".format(label))
    return value


def _required_mapping(value, label):
    if not isinstance(value, dict):
        raise Phase4Error("{} must be a JSON object".format(label))
    return value


def _required_list(value, label):
    if not isinstance(value, list):
        raise Phase4Error("{} must be a JSON array".format(label))
    return value


def _finite_number(value, label, minimum=None):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (minimum is not None and float(value) < minimum)
    ):
        raise Phase4Error("{} must be finite{}".format(
            label,
            " and >= {}".format(minimum) if minimum is not None else "",
        ))
    return float(value)


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _git_identity(root, label):
    root = Path(root).expanduser().resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
        status_text = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Phase4Error("cannot capture {} git identity: {}".format(label, error))
    if not re.match(r"^[0-9a-f]{40}$", commit):
        raise Phase4Error("{} git commit is not a full SHA-1".format(label))
    status_entries = [line for line in status_text.splitlines() if line]

    def generated_build_entry(line):
        if not line.startswith("?? "):
            return False
        relative = line[3:]
        parts = Path(relative).parts
        basename = Path(relative).name
        return (
            (basename.startswith("_C.") and basename.endswith(".so"))
            or "build" in parts
            or any(part.endswith(".egg-info") for part in parts)
        )

    source_entries = [line for line in status_entries if not generated_build_entry(line)]
    normalized_status = "\n".join(source_entries)
    if normalized_status:
        normalized_status += "\n"
    return {
        "root": str(root),
        "commit": commit,
        "dirty": bool(source_entries),
        "status_sha256": hashlib.sha256(
            normalized_status.encode("utf-8")
        ).hexdigest(),
        "status_entries": source_entries,
    }


def _runtime_identity(tacker_root):
    root = Path(tacker_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise Phase4Error("TACKER_ROOT must be an existing non-symlink directory")
    sources = {}
    for relative in RUNTIME_SOURCE_RELATIVE_PATHS:
        sources[relative] = file_artifact(root / relative, "Tacker runtime {}".format(relative))
    return {
        "root": str(root),
        "git": _git_identity(root, "Tacker runtime"),
        "sources": sources,
    }


def _configuration_artifact(path):
    return file_artifact(path, "workload config")


def _project_source_paths():
    relative_paths = set(PROJECT_SOURCE_RELATIVE_PATHS)
    for directory in FIRST_PARTY_SOURCE_DIRECTORIES:
        root = PROJECT_ROOT / directory
        for path in root.rglob("*.py"):
            if "__pycache__" not in path.parts:
                relative_paths.add(str(path.relative_to(PROJECT_ROOT)))
    return tuple(sorted(relative_paths))


def _tool_artifact(value, label):
    resolved = shutil.which(value)
    if resolved is None and Path(value).is_absolute():
        resolved = value
    if resolved is None:
        raise Phase4Error("{} executable is unavailable: {}".format(label, value))
    target = Path(resolved).expanduser().resolve(strict=True)
    return {
        "requested": str(value),
        "resolved": file_artifact(target, "{} executable".format(label)),
    }


def _workload_files(spec):
    model = Path(spec["model_path"])
    source = Path(spec["source_path"])
    iteration_root = model / "point_cloud" / "iteration_{}".format(spec["iteration"])
    required = {
        "cfg_args": model / "cfg_args",
        "point_cloud.ply": iteration_root / "point_cloud.ply",
        "deformation.pth": iteration_root / "deformation.pth",
        "deformation_table.pth": iteration_root / "deformation_table.pth",
        "poses_bounds.npy": source / "poses_bounds.npy",
    }
    return {
        name: file_artifact(path, "{} {}".format(spec["name"], name))
        for name, path in sorted(required.items())
    }


def normalize_workload(raw, label, require_files=True):
    document = _required_mapping(raw, label)
    if document.get("schema_version") != 1 or document.get("kind") != WORKLOAD_KIND:
        raise Phase4Error("{} has the wrong schema/kind".format(label))
    result = {}
    for key in ("name", "model_path", "source_path", "config"):
        value = document.get(key)
        if not isinstance(value, str) or not value:
            raise Phase4Error("{} requires non-empty {}".format(label, key))
        result[key] = value
    for key in ("iteration", "image_width", "image_height", "gaussian_count"):
        value = document.get(key)
        if type(value) is not int or value <= 0:
            raise Phase4Error("{} requires positive integer {}".format(label, key))
        result[key] = value
    if document.get("split", "test") != "test":
        raise Phase4Error("{} must use the test split".format(label))
    result["split"] = "test"
    mix = document.get("raster_deformation_mix")
    if mix not in ("raster_heavy", "balanced", "deformation_heavy"):
        raise Phase4Error(
            "{} requires raster_deformation_mix (raster_heavy/balanced/deformation_heavy)"
            .format(label)
        )
    result["raster_deformation_mix"] = mix
    workload_key = document.get("workload_key")
    if not isinstance(workload_key, str) or not workload_key:
        raise Phase4Error("{} requires a non-empty workload_key".format(label))
    result["workload_key"] = workload_key
    profile_args = document.get("profile_args", [])
    if (
        not isinstance(profile_args, list)
        or any(not isinstance(value, str) or not value for value in profile_args)
    ):
        raise Phase4Error("{} profile_args must be an array of non-empty strings".format(label))
    forbidden_driver_options = {
        "--model_path", "--source_path", "--configs", "--iteration", "--split",
        "--warmup", "--frames", "--trials", "--execution-mode", "--execution_mode",
        "--tacker-profile", "--qualification-mode", "--qualification-profile",
        "--workload-name", "--metadata",
    }
    option_names = [value.split("=", 1)[0] for value in profile_args]
    if any(value in forbidden_driver_options for value in option_names):
        raise Phase4Error("{} profile_args override a coordinator-owned option".format(label))
    if profile_args:
        resolution_only = (
            (len(profile_args) == 2 and profile_args[0] == "--resolution")
            or (
                len(profile_args) == 1
                and profile_args[0].startswith("--resolution=")
                and profile_args[0] != "--resolution="
            )
        )
        if not resolution_only:
            raise Phase4Error(
                "{} profile_args may contain only one explicit --resolution value"
                .format(label)
            )
    _validate_safe_command(["profile_render.py"] + profile_args)
    result["profile_args"] = list(profile_args)
    if require_files:
        for key in ("model_path", "source_path"):
            path = Path(result[key]).expanduser().resolve()
            if not path.is_dir() or path.is_symlink():
                raise Phase4Error("{} {} must be an existing directory".format(label, key))
            result[key] = str(path)
        result["config"] = str(_regular_file(result["config"], "{} config".format(label)))
    else:
        for key in ("model_path", "source_path", "config"):
            result[key] = str(Path(result[key]).expanduser().resolve())
    return result


def _primary_workload(args):
    return {
        "schema_version": 1,
        "kind": WORKLOAD_KIND,
        "name": args.workload_name,
        "model_path": args.model_path,
        "source_path": args.source_path,
        "config": args.config,
        "iteration": args.iteration,
        "split": "test",
        "image_width": args.image_width,
        "image_height": args.image_height,
        "gaussian_count": args.gaussian_count,
        "raster_deformation_mix": args.primary_mix,
        "workload_key": args.primary_workload_key,
        "profile_args": list(args.primary_profile_arg),
    }


def _parse_generalization_specs(args, require_files=True):
    if len(args.generalization_workload) != 2:
        raise Phase4Error("Phase 4 requires exactly two --generalization-workload specs")
    result = []
    for index, value in enumerate(args.generalization_workload, 1):
        inline = isinstance(value, str) and value.lstrip().startswith("{")
        path = None if inline else Path(value).expanduser().resolve()
        if inline:
            try:
                document = json.loads(value)
            except ValueError as error:
                raise Phase4Error(
                    "generalization workload {} inline JSON is invalid: {}".format(
                        index, error
                    )
                )
            spec = normalize_workload(
                document,
                "generalization workload {}".format(index),
                require_files=require_files,
            )
            raw = canonical_json_bytes(document)
            spec["spec_artifact"] = {
                "source": "inline_json",
                "path": None,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        elif require_files:
            document = load_json(path, "generalization workload {}".format(index))
            spec = normalize_workload(
                document,
                "generalization workload {}".format(index),
                require_files=True,
            )
            spec["spec_artifact"] = file_artifact(path, "generalization workload spec")
        else:
            spec = {
                "name": "generalization-{}".format(index),
                "spec_path": str(path),
                "raster_deformation_mix": "declared_in_spec",
                "workload_key": "declared_in_spec",
                "profile_args": ["declared_in_spec"],
            }
        result.append(spec)
    names = [item["name"] for item in result]
    if len(set(names)) != 2 or args.workload_name in names:
        raise Phase4Error("generalization workload names must be unique and non-primary")
    if require_files:
        mixes = {args.primary_mix} | {item["raster_deformation_mix"] for item in result}
        if len(mixes) < 2 or result[0]["raster_deformation_mix"] == result[1]["raster_deformation_mix"]:
            raise Phase4Error(
                "the two generalization workloads must declare different Raster/deformation mixes"
            )
    return result


def _phase31_input_identity(args):
    root = Path(args.phase31_run_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise Phase4Error("--phase31-run-root must be an existing non-symlink directory")
    report = file_artifact(root / "phase31-report.json", "sealed Phase-3.1 report")
    state = file_artifact(root / "phase31-state.json", "sealed Phase-3.1 state")
    if report["sha256"] != args.phase31_run_report_sha256:
        raise Phase4Error("Phase-3.1 run report file hash does not match the canonical pin")
    return {"root": str(root), "report": report, "state": state}


def build_identity(args):
    primary = normalize_workload(_primary_workload(args), "primary workload", True)
    generalization = _parse_generalization_specs(args, True)
    project_sources = {
        relative: file_artifact(PROJECT_ROOT / relative, "project source {}".format(relative))
        for relative in _project_source_paths()
    }
    python = file_artifact(
        _regular_file(args.python_executable, "Python executable", executable=True),
        "Python executable",
    )
    template = file_artifact(args.template_profile, "disabled template profile")
    current_profile = file_artifact(
        args.current_tacker_profile, "current Tacker profile"
    )
    phase31 = _phase31_input_identity(args)
    payload = {
        "phase31": phase31,
        "phase31_pins": {
            "identity_sha256": args.phase31_identity_sha256,
            "matrix_sha256": args.phase31_matrix_sha256,
            "formal_set_sha256": args.phase31_formal_set_sha256,
            "selection_sha256": args.phase31_selection_sha256,
            "run_report_file_sha256": args.phase31_run_report_sha256,
        },
        "project_root": str(PROJECT_ROOT),
        "project_git": _git_identity(PROJECT_ROOT, "4DGaussians"),
        "raster_git": _git_identity(
            PROJECT_ROOT / "submodules" / "depth-diff-gaussian-rasterization",
            "Raster submodule",
        ),
        "simple_knn_git": _git_identity(
            PROJECT_ROOT / "submodules" / "simple-knn", "simple-knn submodule"
        ),
        "project_sources": project_sources,
        "runtime": _runtime_identity(args.tacker_root),
        "python_executable": python,
        "tools": {
            "nvidia_smi": _tool_artifact(args.nvidia_smi, "nvidia-smi"),
            "nvcc": _tool_artifact(args.nvcc, "nvcc"),
            "cmake": _tool_artifact(args.cmake, "cmake"),
            "ctest": _tool_artifact(args.ctest, "ctest"),
        },
        "template_profile": template,
        "current_tacker_profile": current_profile,
        "primary_workload": dict(primary, files=_workload_files(primary), config_artifact=_configuration_artifact(primary["config"])),
        "generalization_workloads": [
            dict(item, files=_workload_files(item), config_artifact=_configuration_artifact(item["config"]))
            for item in generalization
        ],
        "protocol": {
            "formal_trials": FORMAL_TRIALS,
            "formal_frames": FORMAL_FRAMES,
            "formal_warmup": FORMAL_WARMUP,
            "formal_schedule": FORMAL_SCHEDULE,
            "formal_seed": FORMAL_SEED,
            "quality_frames": QUALITY_FRAMES,
            "sequence_lengths": list(SEQUENCE_LENGTHS) + [args.long_frames],
            "leaf_views": args.leaf_views,
            "leaf_warmup": args.leaf_warmup,
            "leaf_repetitions": args.leaf_repetitions,
            "sequence_trials": args.sequence_trials,
            "generalization_trials": args.generalization_trials,
            "timeout_seconds": args.timeout_seconds,
        },
        "device": {
            "logical_gpu": args.gpu,
            "physical_gpu": args.physical_gpu,
            "expected_name": args.expected_gpu_name,
            "expected_compute_capability": [8, 6],
            "expected_cuda": args.expected_cuda,
            "expected_torch": args.expected_torch,
            "expected_python": args.expected_python,
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_HOME",
                "CUDACXX",
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "TORCH_HOME",
                "FOURDGS_SOURCE_COMMIT",
                "FOURDGS_RASTERIZER_COMMIT",
                "FOURDGS_SIMPLE_KNN_COMMIT",
            )
        },
    }
    return {
        "sha256": sha256_json(payload, "tacker-phase4-identity-v1"),
        "payload": payload,
    }


def assert_identity_unchanged(args, expected):
    observed = build_identity(args)
    if observed != expected:
        raise Phase4Error(
            "Phase-4 source, runtime, workload, environment, or sealed input changed during the run"
        )


class Journal(object):
    """Durable stage ledger whose successful artifacts are hash-checked on resume."""

    def __init__(self, output_dir, identity, resume):
        self.root = Path(output_dir).expanduser().resolve()
        self.path = self.root / "phase4-state.json"
        self.lock_path = self.root / ".phase4.lock"
        if resume:
            if not self.root.is_dir() or not self.path.is_file():
                raise Phase4Error("--resume requires an existing Phase-4 checkpoint")
        else:
            try:
                self.root.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                raise Phase4Error(
                    "refusing to append to an existing output directory without --resume"
                )
        self.lock_handle = self.lock_path.open("a+")
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock_handle.close()
            self.lock_handle = None
            raise Phase4Error("another Phase-4 coordinator holds the output lock")
        if resume:
            self.state = load_json(self.path, "Phase-4 checkpoint")
            if (
                self.state.get("schema_version") != SCHEMA_VERSION
                or self.state.get("kind") != STATE_KIND
                or self.state.get("identity") != identity
            ):
                raise Phase4Error("Phase-4 resume identity changed")
        else:
            self.state = {
                "schema_version": SCHEMA_VERSION,
                "kind": STATE_KIND,
                "identity": identity,
                "status": "running",
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "stages": [],
            }
            self.save()

    def close(self):
        if getattr(self, "lock_handle", None) is not None:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
            self.lock_handle.close()
            self.lock_handle = None

    def save(self):
        self.state["updated_at_utc"] = utc_now()
        _atomic_write_json(self.path, self.state, replace=True)

    def _record(self, name):
        found = [item for item in self.state["stages"] if item.get("name") == name]
        if len(found) > 1:
            raise Phase4Error("checkpoint contains duplicate stage {}".format(name))
        return found[0] if found else None

    def run(self, name, inputs, action):
        if name not in STAGES:
            raise Phase4Error("unknown Phase-4 stage {}".format(name))
        digest = sha256_json(inputs, "tacker-phase4-stage-input-v1")
        record = self._record(name)
        if record is not None and record.get("input_sha256") != digest:
            raise Phase4Error("stage {} inputs changed during resume".format(name))
        if record is not None and record.get("status") == "succeeded":
            for saved in record.get("artifacts", []):
                if not _is_within(saved.get("path", ""), self.root):
                    raise Phase4Error(
                        "stage {} artifact escapes the output root".format(name)
                    )
                if file_artifact(saved.get("path", ""), "saved stage artifact") != saved:
                    raise Phase4Error("stage {} artifact changed during resume".format(name))
            result_artifact = _required_mapping(
                record.get("result_artifact"),
                "stage {} result artifact".format(name),
            )
            if (
                result_artifact not in record.get("artifacts", [])
                or not _is_within(result_artifact.get("path", ""), self.root)
                or file_artifact(
                    result_artifact.get("path", ""), "saved stage result"
                )
                != result_artifact
            ):
                raise Phase4Error("stage {} result artifact changed during resume".format(name))
            sealed_result = load_json(
                result_artifact["path"], "saved stage {} result".format(name)
            )
            if (
                sealed_result != record.get("result")
                or record.get("result_sha256")
                != sha256_json(sealed_result, "tacker-phase4-stage-result-v1")
            ):
                raise Phase4Error("stage {} checkpoint result changed during resume".format(name))
            return deepcopy(sealed_result)
        attempt = 1 if record is None else int(record.get("attempt", 0)) + 1
        directory = self.root / "attempts" / name / "{:04d}".format(attempt)
        directory.mkdir(parents=True, exist_ok=False)
        if record is None:
            record = {"name": name, "input_sha256": digest}
            self.state["stages"].append(record)
        record.update(
            {
                "status": "running",
                "attempt": attempt,
                "attempt_dir": str(directory),
                "started_at_utc": utc_now(),
                "finished_at_utc": None,
                "error": None,
                "artifacts": [],
                "result": None,
            }
        )
        self.state["status"] = "running"
        self.save()
        try:
            result, paths = action(directory)
            if not isinstance(result, dict):
                raise Phase4Error("stage {} result must be a JSON object".format(name))
            result_path = directory / "stage-result.json"
            _atomic_write_json(result_path, result)
            artifact_paths = list(paths) + [result_path]
            for path in artifact_paths:
                if not _is_within(path, self.root):
                    raise Phase4Error(
                        "stage {} attempted to publish an artifact outside the output root"
                        .format(name)
                    )
            saved = [
                file_artifact(path, "{} stage artifact".format(name))
                for path in artifact_paths
            ]
            result_artifact = file_artifact(
                result_path, "{} sealed stage result".format(name)
            )
        except BaseException as error:
            record["status"] = "failed"
            record["finished_at_utc"] = utc_now()
            record["error"] = "{}: {}".format(type(error).__name__, error)
            self.state["status"] = "failed"
            self.save()
            raise
        record.update(
            {
                "status": "succeeded",
                "finished_at_utc": utc_now(),
                "error": None,
                "artifacts": saved,
                "result": result,
                "result_artifact": result_artifact,
                "result_sha256": sha256_json(
                    result, "tacker-phase4-stage-result-v1"
                ),
            }
        )
        self.state["status"] = "running"
        self.save()
        return deepcopy(result)


def _validate_safe_command(argv):
    if not isinstance(argv, (list, tuple)) or not argv:
        raise Phase4Error("subprocess argv must be a non-empty array")
    values = [str(value) for value in argv]
    basenames = {Path(value).name for value in values}
    forbidden_programs = sorted(basenames & FORBIDDEN_PROGRAM_BASENAMES)
    forbidden_arguments = sorted(set(values) & FORBIDDEN_ARGUMENTS)
    if forbidden_programs or forbidden_arguments:
        raise Phase4Error(
            "Phase 4 forbids candidate generation/extension: programs={!r}, arguments={!r}"
            .format(forbidden_programs, forbidden_arguments)
        )
    return values


def run_command(argv, log_path, cwd=PROJECT_ROOT, env=None, allowed=(0,), timeout=None):
    values = _validate_safe_command(argv)
    log = Path(log_path).expanduser().resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise Phase4Error("refusing to overwrite command log: {}".format(log))
    with log.open("w", encoding="utf-8") as handle:
        handle.write("argv_sha256={}\n".format(
            sha256_json(values, "tacker-phase4-subprocess-argv-v1")
        ))
        handle.flush()
        try:
            completed = subprocess.run(
                values,
                cwd=str(Path(cwd).resolve()),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise Phase4Error("subprocess timed out: {}".format(error))
    if completed.returncode not in allowed:
        raise Phase4Error(
            "subprocess exited {} (expected {!r}); see {}".format(
                completed.returncode, tuple(allowed), log
            )
        )
    return {
        "argv": values,
        "argv_sha256": sha256_json(values, "tacker-phase4-subprocess-argv-v1"),
        "returncode": completed.returncode,
        "log": file_artifact(log, "subprocess log"),
    }


def _script(relative):
    return str((PROJECT_ROOT / relative).resolve())


def _profile_common(workload):
    return [
        "--model_path", workload["model_path"],
        "--source_path", workload["source_path"],
        "--configs", workload["config"],
        "--iteration", str(workload["iteration"]),
        "--split", "test",
    ]


def _dry_run_plan(args):
    generalization = _parse_generalization_specs(args, require_files=False)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": DRY_RUN_KIND,
        "passed": True,
        "dry_run": True,
        "subprocesses_invoked": 0,
        "candidate_generation": "forbidden",
        "sealed_input_policy": {
            "phase31_root": str(Path(args.phase31_run_root).expanduser().resolve()),
            "read_only": True,
            "exact_identity_sha256": args.phase31_identity_sha256,
            "exact_matrix_sha256": args.phase31_matrix_sha256,
            "exact_formal_set_sha256": args.phase31_formal_set_sha256,
            "exact_selection_sha256": args.phase31_selection_sha256,
            "candidate_scope": "exact verified Phase-3.1 valid finalists plus three baselines",
        },
        "protocol": {
            "formal": {
                "trials": FORMAL_TRIALS,
                "frames": FORMAL_FRAMES,
                "warmup": FORMAL_WARMUP,
                "schedule": FORMAL_SCHEDULE,
                "seed": FORMAL_SEED,
            },
            "quality_frames": QUALITY_FRAMES,
            "sequence_lengths": list(SEQUENCE_LENGTHS) + [args.long_frames],
            "fallback_cases": ["missing", "stale_workload", "hash_mismatch"],
            "generalization_workloads": generalization,
        },
        "stages": [
            {"index": index + 1, "name": name, "resume": "hash-bound"}
            for index, name in enumerate(STAGES)
        ],
        "external_entry_points": [
            _script("scripts/verify_tacker_phase31.py"),
            _script("profile_tacker_leaves.py"),
            _script("scripts/validate_tacker_modes.py"),
            _script("scripts/benchmark_tacker_fps.py"),
            _script("scripts/benchmark_tacker_admission.py"),
            _script("scripts/run_profile_render_sealed.py"),
        ],
        "runtime_binding": {
            "root": str(Path(args.tacker_root).expanduser().resolve()),
            "source_paths": list(RUNTIME_SOURCE_RELATIVE_PATHS),
            "built_library": "libtacker_runtime.so",
            "pre_and_post_hash_check": True,
        },
        "publication": {
            "canary": "explicit profile only",
            "release": "selection/profile artifact; no implicit default replacement",
            "rollback_targets": ["current_tacker", "two_stream"],
        },
    }
    payload["plan_sha256"] = sha256_json(payload, "tacker-phase4-dry-run-v1")
    return payload


def _artifact_record_path(record, run_root, label, require_within=True):
    record = _required_mapping(record, "{} record".format(label))
    path = record.get("path")
    if not isinstance(path, str) or not path:
        raise Phase4Error("{} record has no path".format(label))
    target = Path(path).expanduser().resolve()
    if require_within and not _is_within(target, run_root):
        raise Phase4Error("{} escapes the sealed Phase-3.1 root".format(label))
    observed = file_artifact(target, label)
    expected = {
        "path": str(target),
        "sha256": record.get("sha256"),
        "size_bytes": record.get("size_bytes"),
    }
    if observed != expected:
        raise Phase4Error("{} bytes differ from the sealed record".format(label))
    return target


def _schema2_profile_sha256(profile):
    keys = (
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
    payload = {key: profile.get(key) for key in keys}
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        provenance.pop("generated_at_utc", None)
        payload["provenance"] = provenance
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validate_disabled_finalist_profile(path, finalist, matrix_sha256):
    name = finalist["variant_id"]
    profile = load_json(path, "sealed finalist profile {}".format(name))
    if (
        profile.get("schema_version") != 2
        or profile.get("deployment") != {"enabled": False, "valid": False}
        or profile.get("selected_variant_id") != name
    ):
        raise Phase4Error(
            "sealed finalist {} is not an exact disabled schema-v2 profile".format(name)
        )
    manifest = _required_mapping(profile.get("manifest"), "{} manifest".format(name))
    if profile.get("manifest_sha256") != hashlib.sha256(canonical_json_bytes(manifest)).hexdigest():
        raise Phase4Error("sealed finalist {} manifest hash changed".format(name))
    if profile.get("profile_sha256") != _schema2_profile_sha256(profile):
        raise Phase4Error("sealed finalist {} profile hash changed".format(name))
    provenance = _required_mapping(
        profile.get("provenance"), "{} provenance".format(name)
    )
    if (
        provenance.get("phase") != "3.1"
        or provenance.get("matrix_sha256") != matrix_sha256
        or provenance.get("candidate_sha256") != finalist["candidate_sha256"]
    ):
        raise Phase4Error("sealed finalist {} provenance changed".format(name))
    selected = [
        item
        for item in profile.get("candidates", [])
        if isinstance(item, dict) and item.get("variant_id") == name
    ]
    if len(selected) != 1 or selected[0].get("execution_mode") != "tacker":
        raise Phase4Error("sealed finalist {} has no unique Tacker descriptor".format(name))
    if selected[0].get("correctness") != {"valid": False}:
        raise Phase4Error(
            "Phase-3.1 finalist {} must remain qualification-only before Phase 4".format(name)
        )
    return profile


def validate_phase31_postflight(document, args, run_root):
    document = _required_mapping(document, "Phase-3.1 postflight")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != PHASE31_VERIFY_KIND
        or document.get("passed") is not True
        or Path(document.get("run_root", "")).resolve() != Path(run_root).resolve()
    ):
        raise Phase4Error("independent Phase-3.1 verification did not pass for this root")
    if document.get("run_report_sha256") != args.phase31_run_report_sha256:
        raise Phase4Error("Phase-3.1 postflight run-report hash differs from the pin")
    if document.get("identity_sha256") != args.phase31_identity_sha256:
        raise Phase4Error("Phase-3.1 identity differs from the canonical pin")
    checks = _required_mapping(document.get("checks"), "Phase-3.1 checks")
    for name in (
        "state_and_report",
        "stage_artifacts",
        "matrices",
        "screening_ranking",
        "formal_candidate_set",
        "qualification_profiles",
        "qualification_and_backfill",
        "baseline_quality",
        "formal_execution",
        "selection",
        "top3_nsight",
    ):
        if _required_mapping(checks.get(name), "Phase-3.1 {} check".format(name)).get("passed") is not True:
            raise Phase4Error("Phase-3.1 check {} did not pass".format(name))
    if checks["matrices"].get("final_matrix_sha256") != args.phase31_matrix_sha256:
        raise Phase4Error("Phase-3.1 final matrix differs from the canonical pin")
    if checks["matrices"].get("final_candidate_count") != 661:
        raise Phase4Error("canonical Phase-3.1 matrix must contain exactly 661 candidates")
    if checks["matrices"].get("family_candidate_counts") != {
        "c0": 6, "c1": 30, "c2": 450, "c3": 84, "c4": 91
    }:
        raise Phase4Error("canonical Phase-3.1 family coverage changed")
    if checks["formal_candidate_set"].get("formal_set_sha256") != args.phase31_formal_set_sha256:
        raise Phase4Error("Phase-3.1 formal set differs from the canonical pin")
    if checks["formal_candidate_set"].get("baseline_candidates") != list(BASELINE_NAMES):
        raise Phase4Error("Phase-3.1 formal baseline set changed")
    if checks["selection"].get("selection_sha256") != args.phase31_selection_sha256:
        raise Phase4Error("Phase-3.1 selection differs from the canonical pin")
    if checks["screening_ranking"].get("terminal_count") != 661:
        raise Phase4Error("Phase-3.1 sealed matrix lacks terminal screening coverage")
    return checks


def extract_sealed_phase31_inputs(run_root, verification, args):
    root = Path(run_root).expanduser().resolve()
    checks = validate_phase31_postflight(verification, args, root)
    report_path = root / "phase31-report.json"
    report = load_json(report_path, "sealed Phase-3.1 report")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "4dgaussians_tacker_phase31_run"
        or report.get("passed") is not True
        or report.get("identity", {}).get("sha256") != args.phase31_identity_sha256
    ):
        raise Phase4Error("sealed Phase-3.1 report contract changed")
    matrix_summary = _required_mapping(report.get("matrix"), "Phase-3.1 matrix summary")
    if (
        matrix_summary.get("matrix_sha256") != args.phase31_matrix_sha256
        or matrix_summary.get("candidate_count") != 661
        or matrix_summary.get("exhaustive_c2_count") != 450
    ):
        raise Phase4Error("sealed Phase-3.1 matrix summary changed")
    matrix_path = Path(matrix_summary.get("path", "")).expanduser().resolve()
    if not _is_within(matrix_path, root):
        raise Phase4Error("sealed final matrix path escapes the Phase-3.1 root")
    matrix = load_json(matrix_path, "sealed final matrix")
    if (
        matrix.get("schema_version") != 2
        or matrix.get("matrix_sha256") != args.phase31_matrix_sha256
        or len(matrix.get("candidates", [])) != 661
    ):
        raise Phase4Error("sealed final matrix bytes/identity changed")

    formal_summary = _required_mapping(
        report.get("formal_candidate_set"), "Phase-3.1 formal set summary"
    )
    if formal_summary.get("formal_set_sha256") != args.phase31_formal_set_sha256:
        raise Phase4Error("sealed formal-set summary changed")
    formal_path = Path(formal_summary.get("path", "")).expanduser().resolve()
    if not _is_within(formal_path, root):
        raise Phase4Error("sealed formal-set path escapes the Phase-3.1 root")
    formal = load_json(formal_path, "sealed formal candidate set")
    if (
        formal.get("formal_set_sha256") != args.phase31_formal_set_sha256
        or formal.get("baseline_candidates") != list(BASELINE_NAMES)
    ):
        raise Phase4Error("sealed formal candidate-set document changed")
    generated_records = _required_list(
        formal.get("generated_candidate_set", {}).get("candidates"),
        "sealed generated formal candidates",
    )
    generated_names = [item.get("variant_id") for item in generated_records]
    if (
        not generated_names
        or len(generated_names) != len(set(generated_names))
        or any(not isinstance(name, str) or not SAFE_NAME_RE.match(name) for name in generated_names)
    ):
        raise Phase4Error("sealed formal candidate names are empty, duplicated, or unsafe")

    qualification = _required_mapping(report.get("qualification"), "Phase-3.1 qualification")
    qualification_document = _required_mapping(
        qualification.get("document"), "Phase-3.1 qualification document"
    )
    if (
        qualification_document.get("formal_candidate_set_sha256")
        != args.phase31_formal_set_sha256
        or qualification_document.get("matrix_sha256") != args.phase31_matrix_sha256
    ):
        raise Phase4Error("Phase-3.1 qualification is not bound to matrix/formal set")
    finalists = _required_list(
        qualification_document.get("valid_finalists"), "Phase-3.1 valid finalists"
    )
    finalist_names = [item.get("variant_id") for item in finalists]
    if finalist_names != generated_names:
        raise Phase4Error(
            "Phase 4 may consume only the exact ordered Phase-3.1 formal finalists"
        )
    attempted = _required_list(
        qualification_document.get("attempted"), "Phase-3.1 qualification attempts"
    )
    attempts_by_name = {}
    for attempt in attempted:
        candidate = attempt.get("candidate") if isinstance(attempt, dict) else None
        result = attempt.get("result") if isinstance(attempt, dict) else None
        name = candidate.get("variant_id") if isinstance(candidate, dict) else None
        if name in finalist_names and isinstance(result, dict) and result.get("valid") is True:
            if name in attempts_by_name:
                raise Phase4Error("Phase-3.1 qualification has duplicate finalist {}".format(name))
            if result.get("candidate_sha256") != candidate.get("candidate_sha256"):
                raise Phase4Error("Phase-3.1 finalist {} candidate hash changed".format(name))
            attempts_by_name[name] = (candidate, result)
    if set(attempts_by_name) != set(finalist_names):
        raise Phase4Error("Phase-3.1 qualification does not cover every formal finalist")

    profile_records = []
    for finalist in finalists:
        name = finalist["variant_id"]
        candidate, result = attempts_by_name[name]
        if candidate != finalist:
            raise Phase4Error("Phase-3.1 finalist {} descriptor changed".format(name))
        profile_path = _artifact_record_path(
            result.get("profile"), root, "sealed finalist {} profile".format(name)
        )
        _validate_disabled_finalist_profile(profile_path, finalist, args.phase31_matrix_sha256)
        profile_records.append(
            {
                "name": name,
                "candidate_sha256": finalist["candidate_sha256"],
                "search_family": finalist.get("search_family"),
                "abi_family": finalist.get("abi_family"),
                "profile": file_artifact(profile_path, "sealed finalist profile"),
            }
        )

    selection_summary = _required_mapping(report.get("selection"), "Phase-3.1 selection")
    selection = _required_mapping(selection_summary.get("document"), "Phase-3.1 selection document")
    if (
        selection.get("selection_sha256") != args.phase31_selection_sha256
        or selection.get("matrix_sha256") != args.phase31_matrix_sha256
        or selection.get("deployment_winner") not in set(generated_names) | set(BASELINE_NAMES)
    ):
        raise Phase4Error("sealed Phase-3.1 selection changed")
    selection_path = Path(selection_summary.get("path", "")).expanduser().resolve()
    if not _is_within(selection_path, root):
        raise Phase4Error("sealed selection path escapes the Phase-3.1 root")
    selection_file = load_json(selection_path, "sealed Phase-3.1 selection")
    if selection_file != selection:
        raise Phase4Error("sealed selection summary differs from its artifact")

    identity_files = report.get("identity", {}).get("payload", {}).get("files", {})
    current_record = _required_mapping(
        identity_files.get("current_tacker_profile"), "Phase-3.1 current profile record"
    )
    current_cli = file_artifact(args.current_tacker_profile, "current Tacker profile")
    if current_cli["sha256"] != current_record.get("sha256"):
        raise Phase4Error("current Tacker profile bytes differ from Phase-3.1")
    current_profile = load_json(current_cli["path"], "current Tacker profile")
    gate = current_profile.get("admission") if current_profile.get("schema_version") == 1 else current_profile.get("deployment")
    if gate != {"enabled": True, "valid": True}:
        raise Phase4Error("current Tacker rollback profile is not enabled and valid")

    sealed_artifacts = [
        file_artifact(report_path, "sealed Phase-3.1 report"),
        file_artifact(root / "phase31-state.json", "sealed Phase-3.1 state"),
        file_artifact(matrix_path, "sealed Phase-3.1 matrix"),
        file_artifact(formal_path, "sealed Phase-3.1 formal set"),
        file_artifact(selection_path, "sealed Phase-3.1 selection"),
        file_artifact(qualification.get("path"), "sealed Phase-3.1 qualification plan"),
        current_cli,
    ] + [item["profile"] for item in profile_records]
    unique = {}
    for item in sealed_artifacts:
        previous = unique.get(item["path"])
        if previous is not None and previous != item:
            raise Phase4Error("sealed artifact path has conflicting identities")
        unique[item["path"]] = item
    return {
        "phase31_identity_sha256": args.phase31_identity_sha256,
        "matrix_sha256": args.phase31_matrix_sha256,
        "formal_set_sha256": args.phase31_formal_set_sha256,
        "selection_sha256": args.phase31_selection_sha256,
        "phase31_deployment_winner": selection["deployment_winner"],
        "baselines": list(BASELINE_NAMES),
        "finalists": profile_records,
        "matrix_path": str(matrix_path),
        "formal_set_path": str(formal_path),
        "selection_path": str(selection_path),
        "current_tacker_profile": current_cli,
        "sealed_artifacts": [unique[key] for key in sorted(unique)],
        "verification_summary": {
            "succeeded_stage_count": verification.get("succeeded_stage_count"),
            "unique_stage_artifact_count": verification.get("unique_stage_artifact_count"),
            "family_candidate_counts": checks["matrices"]["family_candidate_counts"],
            "screening_terminal_count": checks["screening_ranking"]["terminal_count"],
        },
    }


def verify_sealed_artifacts_unchanged(sealed):
    for expected in sealed.get("sealed_artifacts", []):
        observed = file_artifact(expected.get("path", ""), "sealed Phase-3.1 input")
        if observed != expected:
            raise Phase4Error("sealed Phase-3.1 input changed during Phase 4: {}".format(expected.get("path")))


def validate_device_query(document, args):
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or document.get("kind") != "4dgaussians_tacker_phase4_device_query"
        or document.get("passed") is not True
    ):
        raise Phase4Error("Phase-4 CUDA device query did not pass")
    if (
        document.get("logical_device") != args.gpu
        or document.get("visible_device_count") != 1
        or document.get("name") != args.expected_gpu_name
        or document.get("compute_capability") != [8, 6]
        or str(document.get("torch_version", "")).split("+", 1)[0] != args.expected_torch
        or document.get("torch_cuda") != args.expected_cuda
        or document.get("python_version") != args.expected_python
    ):
        raise Phase4Error("Phase-4 device/PyTorch/CUDA identity changed")
    if type(document.get("multiprocessor_count")) is not int or document["multiprocessor_count"] <= 0:
        raise Phase4Error("Phase-4 device query omitted SM count")
    if type(document.get("total_memory_bytes")) is not int or document["total_memory_bytes"] <= 0:
        raise Phase4Error("Phase-4 device query omitted total memory")
    return document


def _verify_profile_render_provenance(document, allow_missing_profile=False):
    pre_import = _required_mapping(document.get("pre_import"), "profile_render pre_import")
    if (
        pre_import.get("captured_before_heavy_import") is not True
        or pre_import.get("captured_before_config_execution") is not True
    ):
        raise Phase4Error("profile_render provenance was captured too late")
    files = _required_list(pre_import.get("files"), "profile_render pre_import files")
    roles = []
    files_by_role = {}
    for index, item in enumerate(files):
        item = _required_mapping(item, "pre_import file {}".format(index))
        role = item.get("role")
        if not isinstance(role, str) or not role:
            raise Phase4Error("pre_import file has no role")
        if role in files_by_role:
            raise Phase4Error("pre_import file role is duplicated: {}".format(role))
        roles.append(role)
        files_by_role[role] = item
        exists = item.get("exists")
        if type(exists) is not bool:
            raise Phase4Error("pre_import file {} has no boolean exists".format(role))
        if not exists:
            # ``utils`` is a PEP-420-style package in the production tree: it
            # deliberately has no ``__init__.py``.  profile_render seals that
            # absence and installs an empty in-memory namespace so imports
            # cannot fall through to mutable filesystem bytes.  Keep the role
            # mandatory, then verify its synthetic empty-image binding below.
            optional_absence = role in {
                "simple_knn.wrapper",
                "source.namespace.utils",
            } or (allow_missing_profile and role == "profile.tacker")
            if not optional_absence:
                raise Phase4Error("required pre_import file is missing: {}".format(role))
            continue
        _require_sha256(item.get("sha256"), "pre_import {} SHA".format(role))
        if type(item.get("size_bytes")) is not int or item["size_bytes"] < 0:
            raise Phase4Error("pre_import {} has invalid size".format(role))
        _required_mapping(item.get("stat"), "pre_import {} stat".format(role))
    required_roles = {
        "source.profile_render",
        "source.gaussian_renderer.__init__",
        "source.gaussian_renderer.tacker_pipeline",
        "source.namespace.utils",
        "rasterizer.wrapper",
        "rasterizer.binary",
        "head.wrapper",
        "head.binary",
        "simple_knn.wrapper",
        "simple_knn.binary",
    }
    if not required_roles.issubset(set(roles)):
        raise Phase4Error("profile_render pre_import omitted required source/binary roles")
    bootstrap = _required_mapping(
        pre_import.get("bootstrap_binding"), "profile_render bootstrap binding"
    )
    # The bootstrap binding names the exact application source image it
    # compiled, while the bootstrap launcher itself is bound by the runner's
    # project source identity.
    expected_bootstrap = str(_script("profile_render.py"))
    if (
        bootstrap.get("protocol") != 1
        or bootstrap.get("matches_pre_import_snapshot") is not True
        or bootstrap.get("compiled_sha256")
        != files_by_role["source.profile_render"].get("sha256")
        or bootstrap.get("path") != expected_bootstrap
    ):
        raise Phase4Error("profile_render was not executed from its sealed bootstrap bytes")

    binary_bindings = _required_list(
        pre_import.get("binary_bindings"), "profile_render binary bindings"
    )
    expected_binary_roles = {
        "rasterizer": "rasterizer.binary",
        "head": "head.binary",
        "simple_knn": "simple_knn.binary",
    }
    if len(binary_bindings) != len(expected_binary_roles):
        raise Phase4Error("profile_render must bind exactly three extension binaries")
    bindings_by_component = {}
    for index, raw_binding in enumerate(binary_bindings):
        binding = _required_mapping(
            raw_binding, "profile_render binary binding {}".format(index)
        )
        component = binding.get("component")
        if component not in expected_binary_roles or component in bindings_by_component:
            raise Phase4Error("profile_render binary binding component set changed")
        origin_role = expected_binary_roles[component]
        origin = files_by_role[origin_role]
        if (
            binding.get("strategy") != "private_extension_copy_from_pre_import_bytes"
            or binding.get("matches_pre_import_snapshot") is not True
            or binding.get("origin_role") != origin_role
            or binding.get("origin_path") != origin.get("path")
            or binding.get("sha256") != origin.get("sha256")
            or binding.get("size_bytes") != origin.get("size_bytes")
            or not isinstance(binding.get("module"), str)
            or not isinstance(binding.get("loaded_path"), str)
        ):
            raise Phase4Error(
                "loaded {} binary is not bound to its pre-import bytes".format(component)
            )
        _required_mapping(
            binding.get("loaded_stat"), "loaded {} binary stat".format(component)
        )
        _require_sha256(binding.get("sha256"), "loaded {} binary SHA".format(component))
        bindings_by_component[component] = binding

    compatibility_binding = _required_mapping(
        pre_import.get("rasterizer_binding"), "profile_render rasterizer binding"
    )
    for key, value in bindings_by_component["rasterizer"].items():
        if compatibility_binding.get(key) != value:
            raise Phase4Error("legacy Raster binding disagrees with sealed binary binding")
    sealed_modules = _required_list(
        compatibility_binding.get("sealed_source_modules"),
        "loaded Raster sealed source modules",
    )
    if not sealed_modules:
        raise Phase4Error("loaded Raster binding omitted sealed source modules")

    source_binding = _required_mapping(
        pre_import.get("python_source_binding"), "profile_render Python source binding"
    )
    loaded_modules = _required_list(
        source_binding.get("loaded_modules"), "profile_render loaded source modules"
    )
    if (
        source_binding.get("strategy") != "meta_path_loaders_from_pre_import_bytes"
        or source_binding.get("all_loaded_first_party_modules_match_snapshots") is not True
        or source_binding.get("loaded_module_count") != len(loaded_modules)
        or type(source_binding.get("sealed_module_count")) is not int
        or source_binding["sealed_module_count"] < len(loaded_modules)
        or not loaded_modules
    ):
        raise Phase4Error("first-party Python imports are not bound to sealed bytes")
    loaded_names = set()
    for index, raw_module in enumerate(loaded_modules):
        module = _required_mapping(
            raw_module, "profile_render loaded source module {}".format(index)
        )
        name = module.get("module")
        role = module.get("role")
        role_record = files_by_role.get(role)
        synthetic = module.get("synthetic_namespace")
        expected_source_sha256 = (
            hashlib.sha256(b"").hexdigest()
            if synthetic is True
            else (None if role_record is None else role_record.get("sha256"))
        )
        if (
            not isinstance(name, str)
            or not name
            or name in loaded_names
            or role_record is None
            or module.get("origin_path") != role_record.get("path")
            or module.get("sha256") != expected_source_sha256
            or module.get("matches_pre_import_snapshot") is not True
            or type(synthetic) is not bool
            or (synthetic and role_record.get("exists") is not False)
            or (not synthetic and role_record.get("exists") is not True)
        ):
            raise Phase4Error("loaded first-party Python module escaped its snapshot")
        _require_sha256(module.get("sha256"), "loaded Python source SHA")
        loaded_names.add(name)
    post_run = _required_mapping(document.get("post_run"), "profile_render post_run")
    stability = _required_mapping(post_run.get("byte_stability"), "post-run byte stability")
    checked = _required_list(stability.get("files"), "post-run checked files")
    checked_binaries = _required_list(
        stability.get("loaded_binaries"), "post-run loaded binaries"
    )
    runtime_binding = _required_mapping(
        stability.get("runtime_import_binding"), "post-run import binding"
    )
    if (
        stability.get("verified") is not True
        or stability.get("verification_point") != "after_run_before_metadata_publish"
        or stability.get("file_count") != len(checked)
        or len(checked) != len(files)
        or any(not isinstance(item, dict) or item.get("unchanged") is not True for item in checked)
        or stability.get("loaded_binary_count") != 3
        or len(checked_binaries) != 3
    ):
        raise Phase4Error("profile_render source/config/profile bytes were not stable")
    checked_components = set()
    for item in checked_binaries:
        item = _required_mapping(item, "post-run loaded binary")
        component = item.get("component")
        checks = _required_mapping(item.get("checks"), "post-run loaded binary checks")
        if (
            component not in expected_binary_roles
            or component in checked_components
            or item.get("module") != bindings_by_component[component].get("module")
            or item.get("path") != bindings_by_component[component].get("loaded_path")
            or item.get("unchanged") is not True
            or set(checks) != {"path", "stat", "size_bytes", "sha256"}
            or any(value is not True for value in checks.values())
        ):
            raise Phase4Error("loaded extension binary changed during profiling")
        checked_components.add(component)
    if checked_components != set(expected_binary_roles):
        raise Phase4Error("post-run extension binary coverage changed")
    runtime_sources = _required_list(
        runtime_binding.get("source_modules"), "post-run imported source modules"
    )
    runtime_binaries = _required_list(
        runtime_binding.get("binary_modules"), "post-run imported binary modules"
    )
    if (
        runtime_binding.get("verified") is not True
        or runtime_binding.get("finder_is_first") is not True
        or len(runtime_sources) != len(loaded_modules)
        or len(runtime_binaries) != 3
    ):
        raise Phase4Error("sealed import loader binding changed during profiling")
    expected_source_names = set(loaded_names)
    observed_source_names = set()
    for item in runtime_sources:
        item = _required_mapping(item, "post-run imported source module")
        checks = _required_mapping(item.get("checks"), "post-run source import checks")
        if (
            item.get("module") not in expected_source_names
            or item.get("module") in observed_source_names
            or item.get("unchanged") is not True
            or set(checks) != {"module_present", "loader", "origin", "sha256"}
            or any(value is not True for value in checks.values())
        ):
            raise Phase4Error("sealed source import binding changed during profiling")
        observed_source_names.add(item["module"])
    expected_binary_names = {
        item["module"] for item in bindings_by_component.values()
    }
    observed_binary_names = set()
    for item in runtime_binaries:
        item = _required_mapping(item, "post-run imported binary module")
        checks = _required_mapping(item.get("checks"), "post-run binary import checks")
        if (
            item.get("module") not in expected_binary_names
            or item.get("module") in observed_binary_names
            or item.get("unchanged") is not True
            or set(checks) != {"module_present", "loader", "origin"}
            or any(value is not True for value in checks.values())
        ):
            raise Phase4Error("sealed binary import binding changed during profiling")
        observed_binary_names.add(item["module"])
    if (
        observed_source_names != expected_source_names
        or observed_binary_names != expected_binary_names
    ):
        raise Phase4Error("post-run import binding coverage changed")
    return {
        "rasterizer_binary_sha256": bindings_by_component["rasterizer"]["sha256"],
        "rasterizer_origin_path": bindings_by_component["rasterizer"].get("origin_path"),
        "rasterizer_loaded_path": bindings_by_component["rasterizer"].get("loaded_path"),
        "head_binary_sha256": bindings_by_component["head"]["sha256"],
        "simple_knn_binary_sha256": bindings_by_component["simple_knn"]["sha256"],
        "sealed_file_count": len(files),
    }


def _expected_execution_counts(frames):
    mixed = max(frames - 1, 0)
    return {
        "input_frames": frames,
        "full_deformation": 1,
        "prefix": mixed,
        "mixed_launches": mixed,
        "suffix": mixed,
        "solo_raster": 1,
        "outputs": frames,
        "selected_head_evaluations_per_head": frames,
    }


def validate_render_metadata(
    document,
    workload,
    frames,
    trials,
    expected_warmup,
    expected_mode,
    expected_profile=None,
    qualification=False,
    fallback_expected=False,
    allow_missing_profile=False,
):
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or document.get("kind") != "4dgaussians_tacker_render_profile"
        or document.get("passed") is not True
    ):
        raise Phase4Error("profile_render metadata did not pass")
    if (
        document.get("profile_frames") != frames
        or document.get("trial_count") != trials
        or document.get("warmup_frames") != expected_warmup
        or document.get("workload_name") != workload["name"]
        or document.get("iteration") != workload["iteration"]
        or document.get("split") != "test"
        or document.get("execution_mode") != expected_mode
        or document.get("image_width") != workload["image_width"]
        or document.get("image_height") != workload["image_height"]
        or document.get("gaussian_count") != workload["gaussian_count"]
        or Path(document.get("model_path", "")).resolve()
        != Path(workload["model_path"]).resolve()
        or Path(document.get("source_path", "")).resolve()
        != Path(workload["source_path"]).resolve()
    ):
        raise Phase4Error("profile_render metadata workload/protocol changed")
    expected_timing_contract = {
        "unit": "whole_sequence",
        "frames_per_trial": frames,
        "trial_count": trials,
        "primary_metric": "median_throughput_fps",
        "higher_is_better": True,
        "wall_clock": "perf_counter",
        "wall_clock_completion": "cuda_synchronize_after_each_trial",
        "cuda_events": "start_end_and_per_frame_completion",
        "setup_policy": "single_load_prepare_warmup_before_all_trials",
        "io_in_timed_region": False,
        "cuda_peak_memory": "reset_before_each_trial_query_after_wall_boundary",
    }
    if (
        document.get("timing_method") != "perf_counter_with_cuda_synchronize"
        or document.get("frame_timing_method")
        != "cuda_event_consumer_completion_intervals"
        or document.get("io_in_timed_region") is not False
        or document.get("timing_contract") != expected_timing_contract
    ):
        raise Phase4Error("profile_render timing contract changed")
    view_indices = document.get("view_indices")
    if (
        not isinstance(view_indices, list)
        or len(view_indices) != frames
        or any(type(value) is not int or value < 0 for value in view_indices)
    ):
        raise Phase4Error("profile_render output order/view sequence is invalid")
    if frames <= 50 and view_indices != list(range(frames)):
        raise Phase4Error("profile_render 1/2/50 view order changed")
    actual = document.get("actual_execution_mode")
    tacker_reason = document.get("tacker_fallback_reason")
    two_stream_reason = document.get("two_stream_fallback_reason")
    if fallback_expected:
        if actual == "tacker" or not isinstance(tacker_reason, str) or not tacker_reason:
            raise Phase4Error("negative profile smoke did not take a visible Tacker fallback")
    else:
        if actual != expected_mode:
            raise Phase4Error("profile_render used {} instead of {}".format(actual, expected_mode))
        if expected_mode == "tacker" and (tacker_reason is not None or two_stream_reason is not None):
            raise Phase4Error("normal Tacker execution recorded a fallback")
        if expected_mode == "two_stream" and two_stream_reason is not None:
            raise Phase4Error("two_stream execution recorded a serial fallback")
    if document.get("qualification_mode_requested") is not qualification:
        raise Phase4Error("profile_render qualification request flag changed")
    if not fallback_expected and expected_mode == "tacker":
        if document.get("qualification_mode_executed") is not qualification:
            raise Phase4Error("profile_render qualification execution flag changed")
        if document.get("pipeline_execution_counts") != _expected_execution_counts(frames):
            raise Phase4Error("Tacker prefill/steady-state/drain execution counts changed")
    if expected_profile is not None:
        key = "qualification_profile" if qualification else "tacker_profile"
        if Path(document.get(key, "")).resolve() != Path(expected_profile).resolve():
            raise Phase4Error("profile_render did not use the expected profile")
    for key in (
        "p50_frame_ms",
        "p95_frame_ms",
        "max_frame_ms",
        "median_throughput_fps",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    ):
        _finite_number(document.get(key), "profile_render {}".format(key), 0.0)
    if not (
        document["p50_frame_ms"] <= document["p95_frame_ms"] <= document["max_frame_ms"]
        and document["cuda_peak_allocated_bytes"] <= document["cuda_peak_reserved_bytes"]
    ):
        raise Phase4Error("profile_render latency or peak-memory ordering is invalid")
    raw_trials = _required_list(document.get("trials"), "profile_render trials")
    if len(raw_trials) != trials:
        raise Phase4Error("profile_render trial count changed")
    for trial in raw_trials:
        for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
            _finite_number(trial.get(key), "profile_render trial {}".format(key), 0.0)
    provenance = _verify_profile_render_provenance(
        document, allow_missing_profile=allow_missing_profile
    )
    return provenance


def _quality_mode(document, mode):
    modes = _required_mapping(document.get("modes"), "quality modes")
    value = _required_mapping(modes.get(mode), "quality mode {}".format(mode))
    actual = value.get("actual_execution_mode", value.get("actual_mode"))
    fallback = value.get("fallback_reason")
    return value, actual, fallback


def validate_quality_report(
    document,
    workload,
    modes,
    expect_tacker=True,
    frames=QUALITY_FRAMES,
    expected_profile=None,
    qualification=False,
    allow_invalid_tacker=False,
):
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or document.get("kind") != "4dgaussians_tacker_quality_validation"
    ):
        raise Phase4Error("quality report has the wrong schema/kind")
    quality_workload = _required_mapping(document.get("workload"), "quality workload")
    if (
        quality_workload.get("scene") != workload["name"]
        or quality_workload.get("iteration") != workload["iteration"]
        or quality_workload.get("split") != "test"
        or quality_workload.get("frames") != frames
        or quality_workload.get("view_indices") != list(range(frames))
        or quality_workload.get("resolution")
        != [workload["image_width"], workload["image_height"]]
        or quality_workload.get("gaussian_count") != workload["gaussian_count"]
        or Path(quality_workload.get("model_path", "")).resolve()
        != Path(workload["model_path"]).resolve()
        or Path(quality_workload.get("source_path", "")).resolve()
        != Path(workload["source_path"]).resolve()
    ):
        raise Phase4Error("quality report workload/provenance differs from Phase 4")
    if document.get("thresholds") != {
        "psnr_drop_db_max": 0.05,
        "ssim_drop_max": 1e-4,
        "lpips_increase_max": 1e-4,
    }:
        raise Phase4Error("quality report thresholds changed")
    device = _required_mapping(document.get("device"), "quality device")
    if (
        device.get("name") != "NVIDIA RTX A6000"
        or device.get("index") != 0
        or device.get("compute_capability") != [8, 6]
        or device.get("cuda_arch") != "sm_86"
    ):
        raise Phase4Error("quality report did not use logical A6000/sm_86 device zero")
    if "tacker" in modes:
        quality_qualification = _required_mapping(
            document.get("qualification"), "quality qualification"
        )
        if quality_qualification.get("enabled") is not qualification:
            raise Phase4Error("quality qualification flag changed")
        path_key = "profile_override" if qualification else None
        observed_profile = (
            quality_qualification.get(path_key)
            if path_key is not None
            else document.get("tacker_profile")
        )
        if expected_profile is not None and Path(observed_profile or "").resolve() != Path(
            expected_profile
        ).resolve():
            raise Phase4Error("quality report used the wrong Tacker profile")
        if expected_profile is not None:
            expected_profile_artifact = file_artifact(
                expected_profile, "quality Tacker profile"
            )
            tacker_value, _actual, _fallback = _quality_mode(document, "tacker")
            if (
                tacker_value.get("profile_file_sha256")
                != expected_profile_artifact["sha256"]
                or Path(tacker_value.get("profile_source_path", "")).resolve()
                != Path(expected_profile_artifact["path"]).resolve()
            ):
                raise Phase4Error(
                    "quality execution is not bound to the expected profile bytes"
                )
    for mode in modes:
        value, actual, fallback = _quality_mode(document, mode)
        if mode == "tacker" and expect_tacker:
            if not allow_invalid_tacker and (actual != "tacker" or fallback is not None):
                raise Phase4Error("50-view candidate quality fell back")
            if allow_invalid_tacker and (
                (actual == "tacker" and fallback is not None)
                or (actual != "tacker" and not isinstance(fallback, str))
            ):
                raise Phase4Error("invalid Tacker quality has incoherent fallback evidence")
        elif mode != "tacker" and (actual != mode or fallback is not None):
            raise Phase4Error("quality baseline {} did not execute physically".format(mode))
        per_view = _required_list(value.get("per_view"), "{} per-view quality".format(mode))
        if len(per_view) != frames:
            raise Phase4Error("{} quality must contain exactly {} views".format(mode, frames))
        if [item.get("batch_index") for item in per_view] != list(range(frames)):
            raise Phase4Error("{} quality view ordering changed".format(mode))
    if "tacker" in modes and expect_tacker:
        delta = _required_mapping(document.get("deltas", {}).get("tacker"), "Tacker quality delta")
        thresholds = {
            "psnr_drop_db": 0.05,
            "ssim_drop": 1e-4,
            "lpips_increase": 1e-4,
        }
        for key, limit in thresholds.items():
            value = _finite_number(delta.get(key), "quality {}".format(key))
            if value > limit and not allow_invalid_tacker:
                raise Phase4Error("quality {} exceeds {}".format(key, limit))
        if document.get("passed") is not True and not allow_invalid_tacker:
            raise Phase4Error("50-view quality report is not marked passed")
        gates = [item for item in document.get("gates", []) if item.get("mode") == "tacker"]
        if len(gates) != 1:
            raise Phase4Error("50-view quality report has no unique Tacker gate")
        if gates[0].get("passed") is not True and not allow_invalid_tacker:
            raise Phase4Error("50-view Tacker quality gate did not pass")
    return document


def validate_leaf_bundle(paths, candidate_name=None, candidate=None):
    report = load_json(paths["report"], "leaf report")
    device = load_json(paths["device"], "device measurement")
    raster = load_json(paths["raster"], "Raster measurement")
    leaf = load_json(paths["leaf"], "leaf measurement")
    expected_schema = 2 if candidate_name is not None else 1
    if (
        report.get("schema_version") != expected_schema
        or report.get("kind") != "4dgaussians_tacker_leaf_profile_report"
        or report.get("passed") is not True
    ):
        raise Phase4Error("leaf/resource/numerical profiler did not pass")
    numerics = _required_mapping(report.get("numerics"), "leaf numerical validation")
    if numerics.get("passed") is not True:
        raise Phase4Error("kernel-level numerical validation did not pass")
    if candidate_name is not None:
        binding = _required_mapping(
            report.get("profile_binding"), "leaf candidate profile binding"
        )
        if report.get("variant_id") != candidate_name:
            raise Phase4Error("leaf profiler measured the wrong sealed finalist")
        if binding.get("candidate_matrix_sha256") != EXPECTED_PHASE31_MATRIX:
            raise Phase4Error("leaf profiler is not bound to the sealed Phase-3.1 matrix")
        parameters = _required_mapping(
            report.get("parameters"), "leaf candidate parameters"
        )
        if (
            parameters.get("qualification_profile") is not True
            or parameters.get("used_as_deployment") is not False
            or parameters.get("candidate_profile_deployment_enabled") is not False
        ):
            raise Phase4Error("leaf finalist was not executed as qualification-only")
        if candidate is not None:
            profile = _required_mapping(
                candidate.get("profile"), "sealed finalist profile artifact"
            )
            if (
                candidate.get("name") != candidate_name
                or binding.get("candidate_sha256")
                != candidate.get("candidate_sha256")
                or binding.get("profile_file_sha256") != profile.get("sha256")
                or Path(parameters.get("candidate_profile", "")).resolve()
                != Path(profile.get("path", "")).resolve()
            ):
                raise Phase4Error("leaf profiler finalist/profile byte binding changed")
    snapshot = report.get("execution_source_snapshot")
    if isinstance(snapshot, dict) and snapshot.get("verified_unchanged_after_measurement") is not True:
        raise Phase4Error("leaf profiler source/config bytes changed during measurement")
    resources = report.get("resources")
    if not isinstance(resources, dict):
        resources = report.get("selection", {}).get("resources")
    if candidate_name is not None and not isinstance(resources, dict):
        raise Phase4Error("leaf profiler omitted candidate resource/occupancy evidence")
    measurements = dict(
        _required_mapping(report.get("measurements"), "leaf measurements")
    )
    for key in (
        "solo_raster_p50_ms",
        "mixed_raster_p50_ms",
        "solo_head_p50_ms",
        "mixed_p50_ms",
    ):
        _finite_number(measurements.get(key), "leaf measurement {}".format(key), 0.0)
    if candidate_name is not None and "multi_head_solo_p50_ms" in measurements:
        _finite_number(
            measurements.get("multi_head_solo_p50_ms"),
            "leaf measurement multi_head_solo_p50_ms",
            0.0,
        )
        multi_head_source = "reported_multi_head_measurement"
    elif (
        candidate_name is not None
        and candidate is not None
        and candidate.get("abi_family") == "legacy_pos_l1_v1"
    ):
        # The sealed C0 finalist is schema-v2 metadata around the physical ABI-1
        # single-pos-head kernel.  Its producer intentionally retains the v1
        # timing schema, where the one selected-head measurement is exactly the
        # complete selected-head bundle.  Do not invent a zero or accept this
        # omission for any multi-head ABI family.
        measurements["multi_head_solo_p50_ms"] = measurements[
            "solo_head_p50_ms"
        ]
        multi_head_source = "legacy_abi1_solo_head_p50_ms_alias"
    elif candidate_name is not None:
        raise Phase4Error(
            "leaf measurement multi_head_solo_p50_ms is required for "
            "non-legacy finalists"
        )
    elif "multi_head_solo_p50_ms" in measurements:
        _finite_number(
            measurements["multi_head_solo_p50_ms"],
            "leaf measurement multi_head_solo_p50_ms",
            0.0,
        )
        multi_head_source = "reported_multi_head_measurement"
    else:
        # The legacy/current profiler has one selected head and historically
        # published only this field.  It is an exact semantic alias there;
        # sealed multi-head finalists must still provide their own measurement.
        measurements["multi_head_solo_p50_ms"] = measurements[
            "solo_head_p50_ms"
        ]
        multi_head_source = "legacy_solo_head_p50_ms_alias"
    if measurements["solo_raster_p50_ms"] <= 0.0:
        raise Phase4Error("leaf profiler solo Raster latency must be positive")
    diagnostics = {
        "raster_slowdown_fraction": (
            measurements["mixed_raster_p50_ms"]
            / measurements["solo_raster_p50_ms"]
            - 1.0
        ),
        "unfused_leaf_sum_p50_ms": (
            measurements["solo_raster_p50_ms"]
            + measurements["multi_head_solo_p50_ms"]
        ),
        "mixed_minus_unfused_leaf_sum_p50_ms": (
            measurements["mixed_p50_ms"]
            - measurements["solo_raster_p50_ms"]
            - measurements["multi_head_solo_p50_ms"]
        ),
        "qos_gate": False,
        "multi_head_solo_source": multi_head_source,
        "interpretation": "diagnostic_only_not_an_admission_objective",
    }
    return {
        "documents": {"device": device, "raster": raster, "leaf": leaf, "report": report},
        "artifacts": {name: file_artifact(path, "leaf {}".format(name)) for name, path in paths.items()},
        "numerics": numerics,
        "resources": resources,
        "measurements": measurements,
        "diagnostics": diagnostics,
    }


def validate_formal_benchmark(document, sealed, workload, resources=None):
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or document.get("kind") != "4dgaussians_tacker_fps_benchmark"
        or document.get("passed") is not True
    ):
        raise Phase4Error("Phase-4 formal FPS benchmark did not pass")
    names = list(BASELINE_NAMES) + [item["name"] for item in sealed["finalists"]]
    candidates = _required_list(document.get("candidates"), "formal FPS candidates")
    observed_names = [item.get("name") for item in candidates]
    if observed_names != names:
        raise Phase4Error("formal FPS benchmark candidate set/order is not the sealed set")
    contract = _required_mapping(document.get("contract"), "formal FPS contract")
    if (
        contract.get("profile_frames") != FORMAL_FRAMES
        or contract.get("warmup_frames") != FORMAL_WARMUP
        or contract.get("view_indices") != list(range(FORMAL_FRAMES))
        or contract.get("workload_name") != workload["name"]
        or contract.get("iteration") != workload["iteration"]
    ):
        raise Phase4Error("formal benchmark is not the independent 10x50 contract")
    resume_identity = _required_mapping(document.get("resume_identity"), "formal resume identity")
    payload = _required_mapping(resume_identity.get("payload"), "formal resume payload")
    if (
        payload.get("schedule_strategy") != FORMAL_SCHEDULE
        or payload.get("schedule_seed") != FORMAL_SEED
        or payload.get("trials") != FORMAL_TRIALS
    ):
        raise Phase4Error("formal benchmark did not use 10-trial ABBA seed 0")
    qualifications = document.get("correctness_qualifications")
    if qualifications is None:
        eligible_names = list(names)
    else:
        qualifications = _required_mapping(
            qualifications, "formal correctness qualifications"
        )
        if set(qualifications) != set(names):
            raise Phase4Error("formal correctness qualifications changed candidate set")
        for name in names:
            valid = _required_mapping(
                qualifications[name], "formal correctness {}".format(name)
            ).get("valid")
            if type(valid) is not bool:
                raise Phase4Error("formal correctness validity must be boolean")
        eligible_names = [name for name in names if qualifications[name]["valid"]]
        if document.get("eligible_candidates") != eligible_names:
            raise Phase4Error("formal eligible candidate order disagrees with correctness")
        if any(name not in eligible_names for name in ("serial", "two_stream")):
            raise Phase4Error("formal safety baselines must remain correctness-valid")
    expected_count = FORMAL_TRIALS * len(eligible_names)
    runs = _required_list(document.get("runs"), "formal FPS runs")
    if document.get("completed_execution_count") != expected_count or len(runs) != expected_count:
        raise Phase4Error("formal benchmark did not complete every 10x50 execution")
    per_name = {name: 0 for name in eligible_names}
    loaded_hashes = set()
    for run in runs:
        name = run.get("candidate_name")
        if name not in per_name or run.get("error") is not None:
            raise Phase4Error("formal benchmark contains an unknown/failed run")
        per_name[name] += 1
        metadata_path = run.get("metadata_path")
        metadata = load_json(metadata_path, "formal child metadata")
        expected_mode = next(item["execution_mode"] for item in candidates if item["name"] == name)
        profile = next(item.get("profile_path") for item in candidates if item["name"] == name)
        provenance = validate_render_metadata(
            metadata,
            workload,
            FORMAL_FRAMES,
            1,
            FORMAL_WARMUP,
            expected_mode,
            expected_profile=profile,
            qualification=bool(next(item.get("qualification_mode") for item in candidates if item["name"] == name)),
        )
        loaded_hashes.add(provenance["rasterizer_binary_sha256"])
        if run.get("metadata_sha256") != sha256_file(metadata_path):
            raise Phase4Error("formal child metadata hash changed")
    if set(per_name.values()) != {FORMAL_TRIALS}:
        raise Phase4Error("formal benchmark did not execute every candidate 10 times")
    if len(loaded_hashes) != 1:
        raise Phase4Error("formal benchmark loaded different Raster binaries")
    summaries = _required_mapping(document.get("summaries"), "formal summaries")
    if set(summaries) != set(eligible_names):
        raise Phase4Error("formal summaries do not match correctness-valid candidates")
    for name, summary in summaries.items():
        if summary.get("trial_count") != FORMAL_TRIALS:
            raise Phase4Error("formal summary trial count changed for {}".format(name))
    if resources is not None:
        expected_input = _selection_metadata_from_resources(sealed, resources)
        if document.get("candidate_selection_metadata_input") != expected_input:
            raise Phase4Error("formal selection resource metadata input changed")
        effective_metadata = _required_mapping(
            document.get("candidate_selection_metadata"),
            "formal effective selection metadata",
        )
        if set(effective_metadata) != set(names):
            raise Phase4Error("formal selection metadata changed candidate coverage")
        peaks_by_name = {name: [] for name in eligible_names}
        for run in runs:
            peaks_by_name[run["candidate_name"]].append(
                _finite_number(
                    run.get("metrics", {}).get("cuda_peak_reserved_bytes"),
                    "formal peak reserved memory",
                    0.0,
                )
            )
        for name in names:
            entry = _required_mapping(
                effective_metadata.get(name),
                "formal selection metadata {}".format(name),
            )
            for key, value in expected_input[name].items():
                if entry.get(key) != value:
                    raise Phase4Error(
                        "formal selection metadata lost {} for {}".format(key, name)
                    )
            if name in peaks_by_name:
                expected_peak = max(peaks_by_name[name])
                if entry.get("peak_memory_bytes") != expected_peak:
                    raise Phase4Error(
                        "formal peak-memory tie break changed for {}".format(name)
                    )
                if summaries[name].get("max_cuda_peak_reserved_bytes") != expected_peak:
                    raise Phase4Error(
                        "formal peak-memory summary changed for {}".format(name)
                    )
            elif "peak_memory_bytes" in entry:
                raise Phase4Error("invalid candidate acquired unmeasured peak memory")
        selection = _required_mapping(document.get("selection"), "formal selection")
        if selection.get("candidate_selection_metadata") != effective_metadata:
            raise Phase4Error("formal selector did not consume effective metadata")
        if document.get("peak_memory_tie_break") != {
            "metric": "max_cuda_peak_reserved_bytes_across_formal_trials",
            "lower_is_better": True,
            "measured_after_all_interleaved_trials_before_selection": True,
        }:
            raise Phase4Error("formal peak-memory selection contract changed")
    return {
        "loaded_rasterizer_sha256": next(iter(loaded_hashes)),
        "candidate_names": names,
        "eligible_candidate_names": eligible_names,
    }


def _preflight_stage(args, directory):
    artifacts = []
    commands = []
    smi_log = directory / "nvidia-smi.log"
    commands.append(
        run_command(
            [
                args.nvidia_smi,
                "--id={}".format(args.physical_gpu),
                "--query-gpu=index,name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            smi_log,
            timeout=args.timeout_seconds,
        )
    )
    artifacts.append(smi_log)
    smi_text = smi_log.read_text(encoding="utf-8")
    if args.expected_gpu_name not in smi_text or "8.6" not in smi_text:
        raise Phase4Error("nvidia-smi did not report the selected A6000/sm_86")
    nvcc_log = directory / "nvcc-version.log"
    commands.append(
        run_command(
            [args.nvcc, "--version"], nvcc_log, timeout=args.timeout_seconds
        )
    )
    artifacts.append(nvcc_log)
    if "release {}".format(args.expected_cuda) not in nvcc_log.read_text(encoding="utf-8"):
        raise Phase4Error("nvcc release differs from {}".format(args.expected_cuda))
    device_path = directory / "device-query.json"
    device_log = directory / "device-query.log"
    commands.append(
        run_command(
            [
                str(Path(args.python_executable).expanduser().resolve()),
                str(SCRIPT_PATH),
                "_device-query",
                "--gpu", str(args.gpu),
                "--output", str(device_path),
            ],
            device_log,
            timeout=args.timeout_seconds,
        )
    )
    artifacts.extend((device_log, device_path))
    device = validate_device_query(load_json(device_path, "device query"), args)
    report = {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_phase4_preflight",
        "passed": True,
        "physical_gpu": args.physical_gpu,
        "logical_gpu": args.gpu,
        "commands": commands,
        "device": device,
    }
    report_path = directory / "preflight.json"
    _atomic_write_json(report_path, report)
    artifacts.append(report_path)
    return {
        "report": file_artifact(report_path, "preflight report"),
        "device": device,
    }, artifacts


def _build_commands(args, directory):
    runtime_build = directory / "runtime-build"
    simple_build = directory / "simple-knn-build"
    head_build = directory / "head-build"
    raster_build = directory / "raster-build"
    tacker_root = Path(args.tacker_root).expanduser().resolve()
    raster_root = PROJECT_ROOT / "submodules" / "depth-diff-gaussian-rasterization"
    commands = [
        (
            "runtime-configure",
            [
                args.cmake,
                "-S", str(tacker_root / "src"),
                "-B", str(runtime_build),
                "-DTACKER_BUILD_LEGACY=OFF",
                "-DTACKER_ENABLE_CUDA_BACKEND=ON",
                "-DTACKER_BUILD_TESTS=ON",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CUDA_ARCHITECTURES=86",
                "-DCMAKE_CUDA_COMPILER={}".format(args.nvcc),
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            ],
            PROJECT_ROOT,
            None,
        ),
        (
            "runtime-build",
            [args.cmake, "--build", str(runtime_build), "--parallel"],
            PROJECT_ROOT,
            None,
        ),
        (
            "runtime-ctest",
            [args.ctest, "--test-dir", str(runtime_build), "--output-on-failure"],
            PROJECT_ROOT,
            None,
        ),
        (
            "simple-knn-build",
            [args.python_executable, "setup.py", "build_ext", "--inplace", "--force", "--build-temp", str(simple_build)],
            PROJECT_ROOT / "submodules" / "simple-knn",
            None,
        ),
        (
            "head-sm86-build",
            [args.python_executable, "setup.py", "build_ext", "--inplace", "--force", "--build-temp", str(head_build)],
            PROJECT_ROOT / "tacker_ext",
            None,
        ),
        (
            "raster-sm86-build",
            [args.python_executable, "setup.py", "build_ext", "--inplace", "--force", "--build-temp", str(raster_build)],
            raster_root,
            {"TACKER_4DGS_HEAD_INCLUDE": str(PROJECT_ROOT / "tacker_ext" / "include")},
        ),
        (
            "phase4-cpu-contracts",
            [
                args.python_executable, "-m", "unittest",
                "tests.test_run_tacker_phase4",
                "tests.test_tacker_qualification_script",
                "tests.test_profile_render_modes",
                "tests.test_tacker_admission",
                "tests.test_tacker_pipeline",
                "tests.test_tacker_leaf_profile",
                "tests.test_benchmark_tacker_fps",
                "tests.test_validate_tacker_modes",
                "-v",
            ],
            PROJECT_ROOT,
            None,
        ),
        (
            "head-cpu-contracts",
            [
                args.python_executable, "-m", "unittest", "discover",
                "-s", str(PROJECT_ROOT / "tacker_ext" / "tests"),
                "-p", "test_*.py", "-v",
            ],
            PROJECT_ROOT,
            None,
        ),
        (
            "raster-cpu-contracts",
            [
                args.python_executable, "-m", "unittest", "discover",
                "-s", str(raster_root / "tests"),
                "-p", "test_*.py", "-v",
            ],
            PROJECT_ROOT,
            None,
        ),
        (
            "extension-import-query",
            [
                args.python_executable,
                str(SCRIPT_PATH),
                "_extension-query",
                "--output", str(directory / "extension-binaries.json"),
            ],
            PROJECT_ROOT,
            None,
        ),
        (
            "head-cuda-tests",
            [args.python_executable, "-m", "unittest", "tests.test_head_linear_cuda", "tests.test_head_linear_v2_cuda", "-v"],
            PROJECT_ROOT / "tacker_ext",
            None,
        ),
        (
            "raster-cuda-tests",
            [args.python_executable, "-m", "unittest", "tests.test_tacker_mixed_cuda", "tests.test_stream_aware_legacy_cuda", "-v"],
            raster_root,
            None,
        ),
    ]
    return commands, runtime_build


def _build_and_cuda_stage(args, identity, directory):
    records = []
    artifacts = []
    commands, runtime_build = _build_commands(args, directory)
    for label, argv, cwd, extra_env in commands:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        log = directory / "{}.log".format(label)
        record = run_command(
            argv, log, cwd=cwd, env=env, timeout=args.timeout_seconds
        )
        record["label"] = label
        records.append(record)
        artifacts.append(log)
        if label == "extension-import-query":
            artifacts.append(directory / "extension-binaries.json")
    for label in ("head-sm86-build", "raster-sm86-build"):
        log = directory / "{}.log".format(label)
        if "ptxas info" not in log.read_text(encoding="utf-8", errors="replace"):
            raise Phase4Error("{} contains no real ptxas resource report".format(label))
    compile_commands = runtime_build / "compile_commands.json"
    compile_text = _regular_file(
        compile_commands, "Tacker runtime compile_commands"
    ).read_text(encoding="utf-8", errors="replace")
    for name in ("Scheduler.cc", "Registry.cc", "TaskGraph.cc", "CudaExecutionBackend.cc"):
        if name not in compile_text:
            raise Phase4Error("Tacker runtime build omitted {}".format(name))
    library_link = runtime_build / "libtacker_runtime.so"
    if not library_link.is_file():
        raise Phase4Error("runtime build did not produce libtacker_runtime.so")
    library = library_link.resolve()
    if (
        not _is_within(library, runtime_build)
        or not library.is_file()
        or library.is_symlink()
    ):
        raise Phase4Error("libtacker_runtime.so does not resolve to a regular build artifact")
    artifacts.extend((compile_commands, library))
    extension_document = load_json(
        directory / "extension-binaries.json", "built extension binary query"
    )
    if (
        extension_document.get("schema_version") != 1
        or extension_document.get("kind")
        != "4dgaussians_tacker_phase4_extension_binaries"
        or extension_document.get("passed") is not True
    ):
        raise Phase4Error("built extension binary query did not pass")
    extension_binaries = _required_mapping(
        extension_document.get("binaries"), "built extension binaries"
    )
    if set(extension_binaries) != {"head", "rasterizer", "simple_knn"}:
        raise Phase4Error("built extension query omitted an extension binary")
    for name, artifact in extension_binaries.items():
        if file_artifact(artifact.get("path", ""), "{} extension".format(name)) != artifact:
            raise Phase4Error("{} extension changed after build".format(name))
        # ``build_ext --inplace`` deliberately installs these three binaries in
        # the isolated source snapshot so Python imports the exact build we just
        # queried.  They are runtime inputs, not stage publications: their full
        # file artifacts are sealed in build-cuda.json/stage-result.json and are
        # revalidated immediately after this stage (including on resume), then
        # again before and after publication.  Journal artifacts themselves must
        # remain self-contained under the Phase-4 output root.
    if _runtime_identity(args.tacker_root) != identity["payload"]["runtime"]:
        raise Phase4Error("Tacker runtime sources/git state changed during build")
    report = {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_phase4_build_cuda",
        "passed": True,
        "runtime_identity": identity["payload"]["runtime"],
        "runtime_library": file_artifact(library, "built Tacker runtime library"),
        "runtime_library_link": {
            "path": str(library_link),
            "resolved_target": str(library),
        },
        "compile_commands": file_artifact(compile_commands, "runtime compile commands"),
        "extension_binaries": extension_binaries,
        "commands": records,
        "cuda_suites": {
            "runtime_ctest": True,
            "head_cuda": True,
            "raster_abi1_4_cuda": True,
        },
        "ptxas_logs_present": True,
    }
    report_path = directory / "build-cuda.json"
    _atomic_write_json(report_path, report)
    artifacts.append(report_path)
    return {
        "report": file_artifact(report_path, "build/CUDA report"),
        "runtime_library": report["runtime_library"],
        "extension_binaries": extension_binaries,
    }, artifacts


def _phase31_replay_project_root(phase31_report):
    identity = _required_mapping(
        phase31_report.get("identity"), "Phase-3.1 report identity"
    )
    payload = _required_mapping(
        identity.get("payload"), "Phase-3.1 report identity payload"
    )
    scripts = _required_mapping(
        payload.get("scripts"), "Phase-3.1 report script identity"
    )
    expected = {
        "autotune": Path("scripts") / "tacker_autotune.py",
        "benchmark": Path("scripts") / "benchmark_tacker_fps.py",
        "top3": Path("scripts") / "profile_tacker_top3.py",
    }
    paths = {}
    for name in sorted(expected):
        record = _required_mapping(
            scripts.get(name), "Phase-3.1 {} script".format(name)
        )
        raw = record.get("path")
        if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
            raise Phase4Error(
                "Phase-3.1 {} script path is not absolute".format(name)
            )
        paths[name] = Path(os.path.abspath(raw))
    project_root = paths["autotune"].parents[1]
    for name, relative in expected.items():
        if paths[name] != project_root / relative:
            raise Phase4Error(
                "Phase-3.1 replay scripts do not share one sealed project root"
            )
    return project_root


def _verify_phase31_stage(args, directory):
    output = directory / "phase31-postflight.json"
    log = directory / "phase31-postflight.log"
    phase31_report = load_json(
        Path(args.phase31_run_root).expanduser().resolve() / "phase31-report.json",
        "sealed Phase-3.1 report",
    )
    replay_project_root = _phase31_replay_project_root(phase31_report)
    replay_env = os.environ.copy()
    replay_env["TACKER_PHASE31_PROJECT_ROOT"] = str(replay_project_root)
    command = run_command(
        [
            args.python_executable,
            _script("scripts/verify_tacker_phase31.py"),
            "--run-root", str(Path(args.phase31_run_root).expanduser().resolve()),
            "--output", str(output),
            "--python-executable", str(Path(args.python_executable).expanduser().resolve()),
        ],
        log,
        env=replay_env,
        timeout=args.timeout_seconds,
    )
    command["environment"] = {
        "TACKER_PHASE31_PROJECT_ROOT": str(replay_project_root)
    }
    verification = load_json(output, "Phase-3.1 independent verification")
    sealed = extract_sealed_phase31_inputs(
        args.phase31_run_root, verification, args
    )
    sealed_path = directory / "sealed-phase31-inputs.json"
    _atomic_write_json(
        sealed_path,
        {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_phase4_sealed_phase31_inputs",
            "passed": True,
            "read_only": True,
            "candidate_generation_allowed": False,
            "replay_project_root": str(replay_project_root),
            "inputs": sealed,
            "verifier": file_artifact(output, "Phase-3.1 verifier output"),
            "command": command,
        },
    )
    paths = [output, log, sealed_path]
    return dict(sealed, verifier=file_artifact(output, "Phase-3.1 verifier output"), seal_report=file_artifact(sealed_path, "sealed input report")), paths


def _leaf_command(
    args,
    workload,
    outputs,
    candidate=None,
    matrix_path=None,
    current_profile_path=None,
):
    argv = [
        args.python_executable,
        _script("profile_tacker_leaves.py"),
    ] + _profile_common(workload) + [
        "--scene-name", workload["name"],
        "--views", str(args.leaf_views),
        "--warmup", str(args.leaf_warmup),
        "--repetitions", str(args.leaf_repetitions),
        "--gpu", str(args.gpu),
        "--device-output", str(outputs["device"]),
        "--raster-output", str(outputs["raster"]),
        "--leaf-output", str(outputs["leaf"]),
        "--report", str(outputs["report"]),
        "--quiet",
    ]
    if candidate is None:
        if current_profile_path is None:
            raise Phase4Error("current leaf profiling requires the sealed current profile")
        current = load_json(current_profile_path, "sealed current Tacker profile")
        blocks = current.get("manifest", {}).get("persistent_blocks")
        if type(blocks) is not int or blocks <= 0:
            raise Phase4Error("current Tacker profile has no positive persistent_blocks")
        argv.extend(["--persistent-blocks", str(blocks)])
    else:
        argv.extend(
            [
                "--candidate-profile", candidate["profile"]["path"],
                "--candidate-matrix", str(matrix_path),
            ]
        )
    return argv


def _validate_leaf_failure_report(failure, name, candidate):
    expected_schema = 2 if candidate is not None else 1
    if (
        failure.get("schema_version") != expected_schema
        or failure.get("kind")
        != "4dgaussians_tacker_leaf_profile_report"
        or failure.get("passed") is not False
        or not isinstance(failure.get("errors"), list)
        or not failure["errors"]
    ):
        raise Phase4Error(
            "{} leaf profiler failed without a structured reason".format(name)
        )
    return failure


def _finalist_resources_stage(args, identity, sealed, directory):
    workload = identity["payload"]["primary_workload"]
    records = {}
    artifacts = []
    work = [("current_tacker", None)] + [(item["name"], item) for item in sealed["finalists"]]
    for name, candidate in work:
        candidate_dir = directory / name
        candidate_dir.mkdir()
        outputs = {
            key: candidate_dir / "{}.json".format(key)
            for key in ("device", "raster", "leaf", "report")
        }
        log = candidate_dir / "profile.log"
        command = run_command(
            _leaf_command(
                args,
                workload,
                outputs,
                candidate=candidate,
                matrix_path=sealed["matrix_path"],
                current_profile_path=sealed["current_tacker_profile"]["path"],
            ),
            log,
            allowed=(0, 1),
            timeout=args.timeout_seconds,
        )
        if command["returncode"] != 0:
            failure = load_json(outputs["report"], "failed leaf qualification")
            _validate_leaf_failure_report(failure, name, candidate)
            report_artifact = file_artifact(
                outputs["report"], "failed leaf qualification report"
            )
            records[name] = {
                "valid": False,
                "command": command,
                "outputs": {"report": report_artifact},
                "numerics_passed": False,
                "resources": None,
                "complete_resources": None,
                "measurements": None,
                "diagnostics": {
                    "qos_gate": False,
                    "interpretation": "qualification_failure_not_qos_veto",
                    "errors": list(failure["errors"]),
                },
                "reasons": list(failure["errors"]),
            }
            artifacts.extend((log, outputs["report"]))
            continue
        bundle = validate_leaf_bundle(
            outputs,
            candidate_name=(name if candidate else None),
            candidate=candidate,
        )
        selected_resources = bundle["resources"]
        if isinstance(selected_resources, dict) and "profile_candidate" in selected_resources:
            selected_resources = selected_resources.get("profile_candidate")
        resource_summary = {}
        if isinstance(selected_resources, dict):
            for key in (
                "occupancy", "registers_per_thread", "static_shared_bytes",
                "local_bytes_per_thread", "active_blocks_per_multiprocessor",
                "physical_threads", "worker_groups",
            ):
                if key in selected_resources:
                    resource_summary[key] = selected_resources[key]
        records[name] = {
            "valid": True,
            "command": command,
            "outputs": bundle["artifacts"],
            "numerics_passed": True,
            "resources": resource_summary,
            "complete_resources": bundle["resources"],
            "measurements": bundle["measurements"],
            "diagnostics": bundle["diagnostics"],
        }
        artifacts.append(log)
        artifacts.extend(outputs.values())
    if not any(record.get("valid") is True for record in records.values()):
        raise Phase4Error("no Tacker profiler run produced reusable device evidence")
    report = {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_phase4_finalist_resources_numerics",
        "passed": True,
        "matrix_sha256": sealed["matrix_sha256"],
        "candidate_generation_allowed": False,
        "valid_records": sorted(
            name for name, record in records.items() if record.get("valid") is True
        ),
        "invalid_records": sorted(
            name for name, record in records.items() if record.get("valid") is False
        ),
        "records": records,
    }
    report_path = directory / "resources-numerics.json"
    _atomic_write_json(report_path, report)
    artifacts.append(report_path)
    return {
        "report": file_artifact(report_path, "resource/numerics report"),
        "records": records,
    }, artifacts


def _quality_command(
    args,
    workload,
    output,
    modes,
    frames=QUALITY_FRAMES,
    tacker_profile=None,
    qualification_profile=None,
):
    argv = [
        args.python_executable,
        _script("scripts/validate_tacker_modes.py"),
    ] + _profile_common(workload) + [
        "--scene-name", workload["name"],
        "--frames", str(frames),
        "--view-start", "0",
        "--view-stride", "1",
        "--modes",
    ] + list(modes) + [
        "--gpu", str(args.gpu),
        "--output", str(output),
        "--quiet",
    ]
    if tacker_profile is not None:
        argv.extend(["--tacker-profile", str(tacker_profile)])
    if qualification_profile is not None:
        argv.extend(
            [
                "--qualification-mode",
                "--qualification-profile", str(qualification_profile),
            ]
        )
    argv.extend(workload.get("profile_args", []))
    return argv


def _quality_evidence(document, report_path, numerics=None):
    _, actual, fallback = _quality_mode(document, "tacker")
    delta = _required_mapping(document.get("deltas", {}).get("tacker"), "Tacker quality delta")
    measurements = {
        "psnr_drop_db": _finite_number(delta.get("psnr_drop_db"), "PSNR drop"),
        "ssim_drop": _finite_number(delta.get("ssim_drop"), "SSIM drop"),
        "lpips_increase": _finite_number(
            delta.get("lpips_increase"), "LPIPS increase"
        ),
    }
    limits = {
        "psnr_drop_db": 0.05,
        "ssim_drop": 1e-4,
        "lpips_increase": 1e-4,
    }
    gate = [
        item for item in document.get("gates", []) if item.get("mode") == "tacker"
    ]
    numerics_passed = numerics is None or (
        numerics.get("valid") is True
        and numerics.get("numerics_passed") is True
    )
    reasons = []
    if actual != "tacker":
        reasons.append("physical execution mode was {}".format(actual))
    if fallback is not None:
        reasons.append("physical execution recorded fallback: {}".format(fallback))
    for key, limit in limits.items():
        if measurements[key] > limit:
            reasons.append("{} exceeded {}".format(key, limit))
    if len(gate) != 1 or gate[0].get("passed") is not True:
        reasons.append("50-view Tacker quality gate did not pass")
    if not numerics_passed:
        reasons.append("kernel/resource qualification did not pass")
    valid = not reasons
    evidence = {
        "passed": valid,
        "valid": valid,
        "actual_execution_mode": actual,
        "fallback_reason": fallback,
        "psnr_drop_db": measurements["psnr_drop_db"],
        "ssim_drop": measurements["ssim_drop"],
        "lpips_increase": measurements["lpips_increase"],
        "reasons": reasons,
        "quality_report": file_artifact(report_path, "50-view quality report"),
    }
    if numerics is not None:
        evidence["numerics"] = {
            "passed": numerics_passed,
            "source_report": numerics["outputs"]["report"],
        }
    return evidence


def _quality_50_stage(args, identity, sealed, resources, directory):
    workload = identity["payload"]["primary_workload"]
    artifacts = []
    commands = {}
    reports = {}

    baseline_path = directory / "baseline-quality.json"
    baseline_log = directory / "baseline-quality.log"
    commands["baselines"] = run_command(
        _quality_command(
            args,
            workload,
            baseline_path,
            ("serial", "two_stream", "tacker"),
            tacker_profile=sealed["current_tacker_profile"]["path"],
        ),
        baseline_log,
        allowed=(0, 1),
        timeout=args.timeout_seconds,
    )
    baseline = load_json(baseline_path, "Phase-4 baseline quality")
    validate_quality_report(
        baseline,
        workload,
        ("serial", "two_stream", "tacker"),
        True,
        expected_profile=sealed["current_tacker_profile"]["path"],
        qualification=False,
        allow_invalid_tacker=True,
    )
    artifacts.extend((baseline_log, baseline_path))
    reports["baselines"] = file_artifact(baseline_path, "baseline quality")

    correctness = {
        "serial": {"valid": True, "actual_execution_mode": "serial", "fallback_reason": None},
        "two_stream": {"valid": True, "actual_execution_mode": "two_stream", "fallback_reason": None},
        "current_tacker": _quality_evidence(
            baseline,
            baseline_path,
            resources["records"].get("current_tacker"),
        ),
    }
    for candidate in sealed["finalists"]:
        name = candidate["name"]
        output = directory / "{}-quality.json".format(name)
        log = directory / "{}-quality.log".format(name)
        commands[name] = run_command(
            _quality_command(
                args,
                workload,
                output,
                ("serial", "tacker"),
                qualification_profile=candidate["profile"]["path"],
            ),
            log,
            allowed=(0, 1),
            timeout=args.timeout_seconds,
        )
        document = load_json(output, "{} Phase-4 quality".format(name))
        validate_quality_report(
            document,
            workload,
            ("serial", "tacker"),
            True,
            expected_profile=candidate["profile"]["path"],
            qualification=True,
            allow_invalid_tacker=True,
        )
        artifacts.extend((log, output))
        reports[name] = file_artifact(output, "candidate quality")
        evidence = _quality_evidence(
            document, output, resources["records"].get(name)
        )
        evidence["candidate_sha256"] = candidate["candidate_sha256"]
        correctness[name] = evidence
    expected_names = set(BASELINE_NAMES) | {item["name"] for item in sealed["finalists"]}
    if (
        set(correctness) != expected_names
        or correctness["serial"].get("valid") is not True
        or correctness["two_stream"].get("valid") is not True
    ):
        raise Phase4Error("Phase-4 correctness does not cover the exact sealed formal set")
    correctness_path = directory / "formal-correctness.json"
    _atomic_write_json(correctness_path, correctness)
    artifacts.append(correctness_path)
    report = {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_phase4_quality_50_view",
        "passed": True,
        "frames": QUALITY_FRAMES,
        "view_indices": list(range(QUALITY_FRAMES)),
        "matrix_sha256": sealed["matrix_sha256"],
        "formal_set_sha256": sealed["formal_set_sha256"],
        "candidate_generation_allowed": False,
        "valid_candidates": sorted(
            name for name, evidence in correctness.items() if evidence.get("valid") is True
        ),
        "invalid_candidates": sorted(
            name for name, evidence in correctness.items() if evidence.get("valid") is False
        ),
        "reports": reports,
        "correctness": file_artifact(correctness_path, "Phase-4 formal correctness"),
        "commands": commands,
    }
    report_path = directory / "quality-50-view.json"
    _atomic_write_json(report_path, report)
    artifacts.append(report_path)
    return {
        "report": file_artifact(report_path, "50-view quality report"),
        "baseline_quality": reports["baselines"],
        "correctness": report["correctness"],
        "candidate_quality": reports,
    }, artifacts


def _selection_metadata_from_resources(sealed, resources):
    """Build the sealed, QoS-free equivalence tie-break input."""

    names = list(BASELINE_NAMES) + [item["name"] for item in sealed["finalists"]]
    metadata = {
        "serial": {"abi_complexity": 0.0},
        "two_stream": {"abi_complexity": 1.0},
    }
    for name in names:
        if name in metadata:
            continue
        entry = {"abi_complexity": 2.0}
        record = resources["records"].get(name, {})
        measured = record.get("resources")
        if record.get("valid") is True and isinstance(measured, dict):
            registers = measured.get("registers_per_thread")
            shared = measured.get("static_shared_bytes")
            if _is_finite_number(registers) and float(registers) >= 0.0:
                entry["registers_per_thread"] = float(registers)
            if _is_finite_number(shared) and float(shared) >= 0.0:
                entry["shared_memory_bytes"] = float(shared)
        metadata[name] = entry
    if set(metadata) != set(names):
        raise Phase4Error("selection resource metadata changed candidate coverage")
    return metadata


def _formal_benchmark_stage(args, identity, sealed, quality, resources, directory):
    workload = identity["payload"]["primary_workload"]
    output = directory / "formal-fps.json"
    runs_dir = directory / "runs"
    selection_metadata_path = directory / "selection-metadata.json"
    _atomic_write_json(
        selection_metadata_path,
        _selection_metadata_from_resources(sealed, resources),
    )
    argv = [
        args.python_executable,
        _script("scripts/benchmark_tacker_fps.py"),
        "--output", str(output),
        "--runs-dir", str(runs_dir),
        "--run-id", "phase4-formal",
        "--profile-render", _script("scripts/run_profile_render_sealed.py"),
        "--python-executable", str(Path(args.python_executable).expanduser().resolve()),
        "--current-tacker-profile", sealed["current_tacker_profile"]["path"],
        "--correctness-json", quality["correctness"]["path"],
        "--selection-metadata-json", str(selection_metadata_path),
        "--model-path", workload["model_path"],
        "--source-path", workload["source_path"],
        "--configs", workload["config"],
        "--workload-name", workload["name"],
        "--iteration", str(workload["iteration"]),
        "--split", "test",
        "--frames", str(FORMAL_FRAMES),
        "--warmup", str(FORMAL_WARMUP),
        "--expected-image-width", str(workload["image_width"]),
        "--expected-image-height", str(workload["image_height"]),
        "--expected-gaussian-count", str(workload["gaussian_count"]),
        "--expected-view-indices", ",".join(str(value) for value in range(FORMAL_FRAMES)),
        "--trials", str(FORMAL_TRIALS),
        "--schedule", FORMAL_SCHEDULE,
        "--seed", str(FORMAL_SEED),
        "--bootstrap-resamples", "10000",
    ]
    for candidate in sealed["finalists"]:
        argv.extend(
            [
                "--candidate",
                "{}={}".format(candidate["name"], candidate["profile"]["path"]),
            ]
        )
    for value in workload.get("profile_args", []):
        argv.append("--profile-arg={}".format(value))
    if args.timeout_seconds is not None:
        argv.extend(["--timeout-seconds", str(args.timeout_seconds)])
    log = directory / "formal-fps.log"
    command = run_command(argv, log, timeout=None)
    document = load_json(output, "Phase-4 formal FPS")
    validation = validate_formal_benchmark(
        document, sealed, workload, resources=resources
    )
    artifacts = [log, output, selection_metadata_path]
    for run in document.get("runs", []):
        for key in ("metadata_path", "stdout_path", "stderr_path"):
            value = run.get(key)
            if isinstance(value, str) and Path(value).is_file():
                artifacts.append(value)
    checkpoint = document.get("artifacts", {}).get("checkpoint_path")
    if isinstance(checkpoint, str) and Path(checkpoint).is_file():
        artifacts.append(checkpoint)
    return {
        "report": file_artifact(output, "Phase-4 formal FPS"),
        "command": command,
        "deployment_winner": document.get("deployment_winner"),
        "experimental_winner": document.get("experimental_winner"),
        "validation": validation,
    }, artifacts


def _validate_admission(document, output_profile, formal_document):
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 2
        or document.get("kind") != "4dgaussians_tacker_admission_report"
        or document.get("passed") is not True
    ):
        raise Phase4Error("Phase-4 admission did not pass")
    candidates = _required_list(document.get("candidates"), "admission candidates")
    selected_variant = document.get("selected_variant_id")
    selected = [item for item in candidates if item.get("variant_id") == selected_variant]
    if len(selected) != 1:
        raise Phase4Error("admission has no unique deployment winner")
    formal_winner_name = formal_document.get("deployment_winner")
    formal_winner = [
        item for item in candidates
        if item.get("benchmark_candidate_name") == formal_winner_name
    ]
    if len(formal_winner) != 1 or formal_winner[0].get("variant_id") != selected_variant:
        raise Phase4Error("admission recomputation diverged from the sealed FPS selector")
    selection = _required_mapping(document.get("selection"), "admission selection")
    if selection.get("deployment_winner_variant_id") != selected_variant:
        raise Phase4Error("admission selection disagrees with selected_variant_id")
    mode = selected[0].get("execution_mode")
    if mode not in ("serial", "two_stream", "tacker"):
        raise Phase4Error("admission selected an unsupported execution mode")
    deployment = document.get("deployment")
    profile_artifact = None
    output_profile = Path(output_profile).resolve()
    if mode == "tacker":
        if deployment != {"enabled": True, "valid": True} or not output_profile.is_file():
            raise Phase4Error("Tacker admission did not emit an enabled profile")
        profile = load_json(output_profile, "Phase-4 enabled profile")
        if (
            profile.get("schema_version") != 2
            or profile.get("deployment") != {"enabled": True, "valid": True}
            or profile.get("selected_variant_id") != selected_variant
            or profile.get("profile_sha256") != _schema2_profile_sha256(profile)
            or profile.get("profile_sha256") != document.get("profile_sha256")
        ):
            raise Phase4Error("enabled profile does not match recomputed admission")
        profile_artifact = file_artifact(output_profile, "enabled Phase-4 profile")
    else:
        if deployment != {"enabled": False, "valid": False} or output_profile.exists():
            raise Phase4Error("baseline selection must not manufacture a Tacker profile")
    validated_abi = _required_mapping(
        document.get("provenance", {}).get("validated_abi"), "admission validated ABI"
    )
    if mode == "tacker":
        for key in (
            "mixed_abi_version",
            "mixed_abi_manifest_sha256",
            "head_abi_version",
            "head_abi_manifest_sha256",
        ):
            if key not in validated_abi:
                raise Phase4Error("admission omitted validated ABI field {}".format(key))
    return {
        "selected_variant_id": selected_variant,
        "benchmark_candidate_name": formal_winner_name,
        "execution_mode": mode,
        "enabled_profile": profile_artifact,
        "selection_sha256": document.get("profile_sha256"),
        "validated_abi": validated_abi,
    }


def _selection_admission_stage(args, sealed, resources, quality, formal, directory):
    formal_document = load_json(formal["report"]["path"], "Phase-4 formal benchmark")
    ordered_resource_names = ["current_tacker"] + [
        item["name"] for item in sealed["finalists"]
    ]
    usable_resource_names = [
        name
        for name in ordered_resource_names
        if resources["records"].get(name, {}).get("valid") is True
    ]
    if not usable_resource_names:
        raise Phase4Error("admission has no successful device/resource profiler input")
    profiler_source_name = usable_resource_names[0]
    current = resources["records"][profiler_source_name]["outputs"]
    report_path = directory / "admission-report.json"
    profile_path = directory / "enabled-profile.json"
    argv = [
        args.python_executable,
        _script("scripts/benchmark_tacker_admission.py"),
        "--device-json", current["device"]["path"],
        "--quality-json", quality["baseline_quality"]["path"],
        "--raster-json", current["raster"]["path"],
        "--leaf-json", current["leaf"]["path"],
        "--fps-benchmark-json", formal["report"]["path"],
        "--candidate-correctness-json", quality["correctness"]["path"],
        "--head-abi-json", _script("tacker_ext/abi/head_linear_v2.json"),
        "--template-profile", str(Path(args.template_profile).expanduser().resolve()),
        "--report", str(report_path),
        "--enabled-profile", str(profile_path),
    ]
    for relative in (
        "submodules/depth-diff-gaussian-rasterization/abi/tacker_mixed_render_head_v1.json",
        "submodules/depth-diff-gaussian-rasterization/abi/tacker_mixed_render_heads_v2.json",
        "submodules/depth-diff-gaussian-rasterization/abi/tacker_mixed_render_packed_heads_v3.json",
        "submodules/depth-diff-gaussian-rasterization/abi/tacker_mixed_render_whole_heads_v4.json",
    ):
        argv.extend(["--mixed-abi-json", _script(relative)])
    argv.extend(
        [
            "--candidate-profile",
            "current_tacker={}".format(sealed["current_tacker_profile"]["path"]),
        ]
    )
    for candidate in sealed["finalists"]:
        argv.extend(
            [
                "--candidate-profile",
                "{}={}".format(candidate["name"], candidate["profile"]["path"]),
            ]
        )
    log = directory / "admission.log"
    command = run_command(argv, log, timeout=args.timeout_seconds)
    document = load_json(report_path, "Phase-4 admission report")
    validation = _validate_admission(document, profile_path, formal_document)
    generated_profile = validation.get("enabled_profile")
    if validation["benchmark_candidate_name"] == "current_tacker":
        if validation["execution_mode"] != "tacker":
            raise Phase4Error("current_tacker benchmark winner did not map to Tacker")
        validation["enabled_profile"] = sealed["current_tacker_profile"]
        validation["profile_action"] = "retain_incumbent_profile"
        validation["admission_generated_profile"] = generated_profile
    elif validation["execution_mode"] == "tacker":
        validation["profile_action"] = "publish_new_challenger_profile"
        validation["admission_generated_profile"] = generated_profile
    else:
        validation["profile_action"] = "publish_baseline_selection_without_profile"
        validation["admission_generated_profile"] = None
    artifacts = [log, report_path]
    if generated_profile is not None:
        artifacts.append(profile_path)
    return {
        "report": file_artifact(report_path, "Phase-4 admission report"),
        "command": command,
        "profiler_source_candidate": profiler_source_name,
        "deployment": validation,
    }, artifacts


def _render_command(
    args,
    workload,
    output,
    mode,
    frames,
    trials,
    warmup,
    profile=None,
    workload_name=None,
    qualification=False,
):
    argv = [
        args.python_executable,
        _script("scripts/run_profile_render_sealed.py"),
    ] + _profile_common(workload) + [
        "--warmup", str(warmup),
        "--frames", str(frames),
        "--trials", str(trials),
        "--execution-mode", mode,
        "--workload-name", workload["name"] if workload_name is None else workload_name,
        "--metadata", str(output),
        "--quiet",
    ]
    if mode == "tacker":
        if profile is None:
            raise Phase4Error("Tacker render command requires an explicit profile path")
        if qualification:
            argv.extend(
                ["--qualification-mode", "--qualification-profile", str(profile)]
            )
        else:
            argv.extend(["--tacker-profile", str(profile)])
    elif qualification:
        raise Phase4Error("qualification mode requires Tacker execution")
    argv.extend(workload.get("profile_args", []))
    return argv


def _deployment_profile(admission):
    deployment = admission["deployment"]
    profile = deployment.get("enabled_profile")
    return None if profile is None else profile["path"]


def _sequence_regression_stage(args, identity, admission, directory):
    workload = identity["payload"]["primary_workload"]
    mode = admission["deployment"]["execution_mode"]
    profile = _deployment_profile(admission)
    lengths = list(SEQUENCE_LENGTHS) + [args.long_frames]
    records = {}
    artifacts = []
    loaded_hashes = set()
    for frames in lengths:
        metadata_path = directory / "sequence-{}.json".format(frames)
        log = directory / "sequence-{}.log".format(frames)
        command = run_command(
            _render_command(
                args,
                workload,
                metadata_path,
                mode,
                frames,
                args.sequence_trials,
                FORMAL_WARMUP,
                profile=profile,
            ),
            log,
            timeout=args.timeout_seconds,
        )
        metadata = load_json(metadata_path, "{}-frame sequence metadata".format(frames))
        provenance = validate_render_metadata(
            metadata,
            workload,
            frames,
            args.sequence_trials,
            FORMAL_WARMUP,
            mode,
            expected_profile=profile,
            qualification=False,
        )
        loaded_hashes.add(provenance["rasterizer_binary_sha256"])
        quality_artifact = None
        if frames in SEQUENCE_LENGTHS:
            quality_path = directory / "sequence-{}-quality.json".format(frames)
            quality_log = directory / "sequence-{}-quality.log".format(frames)
            if mode == "tacker":
                modes = ("serial", "tacker")
                kwargs = {"tacker_profile": profile}
            elif mode == "two_stream":
                modes = ("serial", "two_stream")
                kwargs = {}
            else:
                modes = None
                kwargs = None
            if modes is None:
                quality_command = None
                quality_artifact = {
                    "passed": True,
                    "reference": "serial_is_the_legacy_renderer",
                    "view_indices": list(range(frames)),
                }
            else:
                quality_command = run_command(
                    _quality_command(
                        args,
                        workload,
                        quality_path,
                        modes,
                        frames=frames,
                        **kwargs
                    ),
                    quality_log,
                    timeout=args.timeout_seconds,
                )
                quality_document = load_json(quality_path, "sequence output-order quality")
                validate_quality_report(
                    quality_document,
                    workload,
                    modes,
                    expect_tacker=(mode == "tacker"),
                    frames=frames,
                    expected_profile=(profile if mode == "tacker" else None),
                    qualification=False,
                )
                quality_artifact = file_artifact(quality_path, "sequence quality")
                artifacts.extend((quality_log, quality_path))
        else:
            quality_command = None
        records[str(frames)] = {
            "metadata": file_artifact(metadata_path, "sequence metadata"),
            "command": command,
            "quality": quality_artifact,
            "quality_command": quality_command,
            "execution_counts": metadata.get("pipeline_execution_counts"),
            "p50_frame_ms": metadata["p50_frame_ms"],
            "p95_frame_ms": metadata["p95_frame_ms"],
            "max_frame_ms": metadata["max_frame_ms"],
            "cuda_peak_allocated_bytes": metadata["cuda_peak_allocated_bytes"],
            "cuda_peak_reserved_bytes": metadata["cuda_peak_reserved_bytes"],
        }
        artifacts.extend((log, metadata_path))
    if len(loaded_hashes) != 1:
        raise Phase4Error("sequence regressions loaded different Raster binaries")
    report = {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_phase4_sequence_regression",
        "passed": True,
        "execution_mode": mode,
        "profile": admission["deployment"].get("enabled_profile"),
        "sequence_lengths": lengths,
        "long_sequence_is_steady_state": args.long_frames > 50,
        "output_order_compared_to_legacy": list(SEQUENCE_LENGTHS),
        "records": records,
        "loaded_rasterizer_sha256": next(iter(loaded_hashes)),
    }
    report_path = directory / "sequence-regression.json"
    _atomic_write_json(report_path, report)
    artifacts.append(report_path)
    return {"report": file_artifact(report_path, "sequence regression"), **report}, artifacts


def _enabled_rerun_stage(args, identity, admission, directory):
    workload = identity["payload"]["primary_workload"]
    mode = admission["deployment"]["execution_mode"]
    profile = _deployment_profile(admission)
    output = directory / "normal-deployment-10x50.json"
    log = directory / "normal-deployment-10x50.log"
    command = run_command(
        _render_command(
            args,
            workload,
            output,
            mode,
            FORMAL_FRAMES,
            FORMAL_TRIALS,
            FORMAL_WARMUP,
            profile=profile,
        ),
        log,
        timeout=args.timeout_seconds,
    )
    document = load_json(output, "normal deployment rerun")
    provenance = validate_render_metadata(
        document,
        workload,
        FORMAL_FRAMES,
        FORMAL_TRIALS,
        FORMAL_WARMUP,
        mode,
        expected_profile=profile,
        qualification=False,
    )
    return {
        "metadata": file_artifact(output, "normal deployment rerun"),
        "command": command,
        "loaded_rasterizer_sha256": provenance["rasterizer_binary_sha256"],
        "execution_mode": mode,
        "profile": admission["deployment"].get("enabled_profile"),
    }, [log, output]


def _fallback_smoke_stage(args, identity, sealed, admission, quality, directory):
    workload = identity["payload"]["primary_workload"]
    correctness = load_json(
        quality["correctness"]["path"], "fallback correctness qualification"
    )
    source_profile = _deployment_profile(admission)
    source_profile_qualification = False
    if source_profile is None:
        if correctness.get("current_tacker", {}).get("valid") is True:
            source_profile = sealed["current_tacker_profile"]["path"]
        else:
            for candidate in sealed["finalists"]:
                if correctness.get(candidate["name"], {}).get("valid") is True:
                    source_profile = candidate["profile"]["path"]
                    source_profile_qualification = True
                    break
    missing_path = directory / "missing-profile.json"
    if missing_path.exists():
        raise Phase4Error("missing-profile negative fixture unexpectedly exists")
    cases = [("missing", missing_path, workload["name"], True)]
    artifacts = []
    if source_profile is not None:
        mismatch_profile = load_json(source_profile, "fallback source profile")
        mismatch_profile["manifest_sha256"] = "0" * 64
        mismatch_path = directory / "hash-mismatch-profile.json"
        _atomic_write_json(mismatch_path, mismatch_profile)
        stale_name = "{}-stale-profile-smoke".format(workload["name"])
        cases.extend(
            [
                ("stale_workload", Path(source_profile), stale_name, False),
                ("hash_mismatch", mismatch_path, workload["name"], False),
            ]
        )
        artifacts.append(mismatch_path)
    records = {}
    loaded_hashes = set()
    for name, profile, requested_workload, allow_missing in cases:
        output = directory / "{}.json".format(name)
        log = directory / "{}.log".format(name)
        command = run_command(
            _render_command(
                args,
                workload,
                output,
                "tacker",
                2,
                1,
                0,
                profile=profile,
                workload_name=requested_workload,
                qualification=(source_profile_qualification and not allow_missing),
            ),
            log,
            timeout=args.timeout_seconds,
        )
        metadata = load_json(output, "{} fallback smoke metadata".format(name))
        expected_workload = dict(workload, name=requested_workload)
        provenance = validate_render_metadata(
            metadata,
            expected_workload,
            2,
            1,
            0,
            "tacker",
            expected_profile=profile,
            qualification=(source_profile_qualification and not allow_missing),
            fallback_expected=True,
            allow_missing_profile=allow_missing,
        )
        reason = metadata["tacker_fallback_reason"].lower()
        expected_words = {
            "missing": ("cannot load", "missing", "no such file"),
            "stale_workload": ("workload", "stale"),
            "hash_mismatch": ("hash", "sha-256", "invalid"),
        }[name]
        if not any(word in reason for word in expected_words):
            raise Phase4Error("{} fallback reason is not specific: {}".format(name, reason))
        loaded_hashes.add(provenance["rasterizer_binary_sha256"])
        records[name] = {
            "metadata": file_artifact(output, "fallback smoke metadata"),
            "command": command,
            "actual_execution_mode": metadata["actual_execution_mode"],
            "fallback_reason": metadata["tacker_fallback_reason"],
            "source_profile": (
                {"path": str(profile), "exists": False}
                if allow_missing
                else file_artifact(profile, "fallback fixture profile")
            ),
        }
        artifacts.extend((log, output))
    if len(loaded_hashes) != 1:
        raise Phase4Error("fallback smoke runs loaded different Raster binaries")
    report = {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_phase4_fallback_smoke",
        "passed": True,
        "cases": records,
        "all_executed_cases_fell_back_with_reason": True,
        "requested_case_count": 3,
        "executed_case_count": len(cases),
        "full_negative_matrix_executed": len(cases) == 3,
        "mutation_cases_unavailable_reason": (
            None
            if source_profile is not None
            else "no correctness-valid sealed Tacker profile was available as a mutation base"
        ),
        "source_profile": (
            None
            if source_profile is None
            else file_artifact(source_profile, "fallback source profile")
        ),
        "source_profile_qualification_mode": source_profile_qualification,
        "loaded_rasterizer_sha256": next(iter(loaded_hashes)),
    }
    report_path = directory / "fallback-smoke.json"
    _atomic_write_json(report_path, report)
    artifacts.append(report_path)
    return {"report": file_artifact(report_path, "fallback smoke"), **report}, artifacts


def _resolution_argument(profile_args):
    values = list(profile_args)
    found = []
    for index, value in enumerate(values):
        if value == "--resolution" and index + 1 < len(values):
            found.append(values[index + 1])
        elif value.startswith("--resolution="):
            found.append(value.split("=", 1)[1])
    if len(found) != 1:
        raise Phase4Error("generalization profile_args require exactly one --resolution")
    try:
        resolution = int(found[0])
    except ValueError:
        raise Phase4Error("generalization --resolution must be an integer")
    if resolution not in (-1, 1, 2, 4, 8):
        raise Phase4Error("generalization --resolution must be one of -1/1/2/4/8")
    return resolution


def _validate_resolution_metadata(metadata, workload):
    argument = _resolution_argument(workload.get("profile_args", []))
    scale = 1 if argument in (-1, 1) else argument
    if (
        metadata.get("profile_resolution_argument") != argument
        or metadata.get("profile_resolution_scale") != scale
        or metadata.get("effective_resolution")
        != [workload["image_width"], workload["image_height"]]
        or metadata.get("image_width") != workload["image_width"]
        or metadata.get("image_height") != workload["image_height"]
    ):
        raise Phase4Error("generalization profiler did not apply the requested resolution")
    original = metadata.get("original_resolution")
    if (
        not isinstance(original, list)
        or len(original) != 2
        or metadata.get("original_image_width") != original[0]
        or metadata.get("original_image_height") != original[1]
        or workload["image_width"] != max(1, int(round(float(original[0]) / scale)))
        or workload["image_height"] != max(1, int(round(float(original[1]) / scale)))
    ):
        raise Phase4Error("generalization effective resolution is not derived from the original view")
    contract = metadata.get("profile_resolution_contract")
    if contract != "bilinear_original_image_only_fov_and_projection_unchanged":
        raise Phase4Error("profile resolution scaling contract changed")
    return {"argument": argument, "scale": scale, "original_resolution": original}


def _generalization_stage(args, identity, index, directory):
    workload = identity["payload"]["generalization_workloads"][index - 1]
    modes = ("serial", "split_serial", "two_stream")
    records = {}
    artifacts = []
    loaded_hashes = set()
    for mode in modes:
        output = directory / "{}.json".format(mode)
        log = directory / "{}.log".format(mode)
        command = run_command(
            _render_command(
                args,
                workload,
                output,
                mode,
                FORMAL_FRAMES,
                args.generalization_trials,
                FORMAL_WARMUP,
            ),
            log,
            timeout=args.timeout_seconds,
        )
        metadata = load_json(output, "generalization {} {}".format(index, mode))
        provenance = validate_render_metadata(
            metadata,
            workload,
            FORMAL_FRAMES,
            args.generalization_trials,
            FORMAL_WARMUP,
            mode,
            qualification=False,
        )
        resolution = _validate_resolution_metadata(metadata, workload)
        loaded_hashes.add(provenance["rasterizer_binary_sha256"])
        records[mode] = {
            "metadata": file_artifact(output, "generalization metadata"),
            "command": command,
            "median_throughput_fps": metadata["median_throughput_fps"],
            "p50_frame_ms": metadata["p50_frame_ms"],
            "p95_frame_ms": metadata["p95_frame_ms"],
            "max_frame_ms": metadata["max_frame_ms"],
            "cuda_peak_allocated_bytes": metadata["cuda_peak_allocated_bytes"],
            "cuda_peak_reserved_bytes": metadata["cuda_peak_reserved_bytes"],
            "resolution": resolution,
        }
        artifacts.extend((log, output))
    if len(loaded_hashes) != 1:
        raise Phase4Error("generalization modes loaded different Raster binaries")
    serial_fps = records["serial"]["median_throughput_fps"]
    split_fps = records["split_serial"]["median_throughput_fps"]
    two_fps = records["two_stream"]["median_throughput_fps"]
    balance_proxy = {
        "kind": "measured_overlap_balance_proxy",
        "two_stream_to_split_serial_fps_ratio": two_fps / split_fps,
        "split_serial_to_serial_fps_ratio": split_fps / serial_fps,
        "interpretation": (
            "whole-run overlap sensitivity proxy; not a direct Raster/deformation kernel-time ratio"
        ),
    }
    report = {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_phase4_generalization_evaluation",
        "passed": True,
        "evaluation_only": True,
        "qualification_variant_evaluated": False,
        "qualification_variant_exclusion_reason": (
            "sealed Phase-3.1 Tacker profiles are bound to the primary workload "
            "key, dimensions, and Gaussian count; Phase 4 forbids candidate "
            "generation or cross-workload profile mutation"
        ),
        "selection_scope": "available_safe_baselines_for_unqualified_workload",
        "candidate_generation_allowed": False,
        "cross_workload_profile_reuse_allowed": False,
        "global_variant_claimed": False,
        "workload": {
            key: workload[key]
            for key in (
                "name", "workload_key", "iteration", "image_width", "image_height",
                "gaussian_count", "raster_deformation_mix", "profile_args",
            )
        },
        "modes": list(modes),
        "records": records,
        "balance_proxy": balance_proxy,
        "baseline_winner": max(modes, key=lambda name: records[name]["median_throughput_fps"]),
        "loaded_rasterizer_sha256": next(iter(loaded_hashes)),
    }
    report_path = directory / "generalization.json"
    _atomic_write_json(report_path, report)
    artifacts.append(report_path)
    return {"report": file_artifact(report_path, "generalization report"), **report}, artifacts


def _sealed_document(kind, payload, domain):
    document = dict(payload)
    document["schema_version"] = 1
    document["kind"] = kind
    document["sha256"] = sha256_json(document, domain)
    return document


def _canary_release_rollback_stage(
    args,
    identity,
    sealed,
    build,
    admission,
    quality,
    sequence,
    rerun,
    fallback,
    generalizations,
    directory,
):
    workload = identity["payload"]["primary_workload"]
    artifacts = []
    drills = {}
    loaded_hashes = set()
    correctness = load_json(
        quality["correctness"]["path"], "rollback correctness qualification"
    )
    current_qualification = _required_mapping(
        correctness.get("current_tacker"), "current Tacker rollback qualification"
    )
    current_available = current_qualification.get("valid") is True
    if not current_available:
        drills["current_tacker"] = {
            "passed": False,
            "eligible": False,
            "executed": False,
            "reason": "current Tacker is correctness-invalid and cannot be a rollback target",
            "qualification": current_qualification,
            "qualification_artifact": quality["correctness"],
        }
    drill_specs = [("two_stream", None)]
    if current_available:
        drill_specs.insert(
            0, ("current_tacker", sealed["current_tacker_profile"]["path"])
        )
    for mode, profile in drill_specs:
        execution_mode = "tacker" if mode == "current_tacker" else "two_stream"
        output = directory / "rollback-{}.json".format(mode)
        log = directory / "rollback-{}.log".format(mode)
        command = run_command(
            _render_command(
                args,
                workload,
                output,
                execution_mode,
                FORMAL_FRAMES,
                args.sequence_trials,
                FORMAL_WARMUP,
                profile=profile,
            ),
            log,
            timeout=args.timeout_seconds,
        )
        metadata = load_json(output, "rollback drill {}".format(mode))
        provenance = validate_render_metadata(
            metadata,
            workload,
            FORMAL_FRAMES,
            args.sequence_trials,
            FORMAL_WARMUP,
            execution_mode,
            expected_profile=profile,
            qualification=False,
        )
        loaded_hashes.add(provenance["rasterizer_binary_sha256"])
        drills[mode] = {
            "metadata": file_artifact(output, "rollback drill metadata"),
            "command": command,
            "passed": True,
            "eligible": True,
            "executed": True,
        }
        artifacts.extend((log, output))
    if len(loaded_hashes) != 1:
        raise Phase4Error("rollback drills loaded different Raster binaries")
    deployment_mode = admission["deployment"]["execution_mode"]
    profile_action = admission["deployment"].get("profile_action")
    canary = _sealed_document(
        "4dgaussians_tacker_phase4_canary",
        {
            "passed": True,
            "scope": (
                "explicit_profile_only"
                if deployment_mode == "tacker"
                else "explicit_execution_mode_only"
            ),
            "default_profile_replaced": False,
            "profile_action": profile_action,
            "deployment": admission["deployment"],
            "normal_10x50": rerun["metadata"],
            "sequence_regression": sequence["report"],
            "fallback_smoke": fallback["report"],
        },
        "tacker-phase4-canary-v1",
    )
    canary_path = directory / "canary.json"
    _atomic_write_json(canary_path, canary)
    artifacts.append(canary_path)
    rollback = _sealed_document(
        "4dgaussians_tacker_phase4_rollback",
        {
            "passed": True,
            "policy": "replace_only_deployment_selection_or_profile",
            "targets": {
                "current_tacker": {
                    "available": current_available,
                    "execution_mode": "tacker",
                    "profile": (
                        sealed["current_tacker_profile"]
                        if current_available
                        else None
                    ),
                    "qualification_artifact": quality["correctness"],
                },
                "two_stream": {
                    "available": True,
                    "execution_mode": "two_stream",
                    "profile": None,
                },
            },
            "drills": drills,
            "loaded_rasterizer_sha256": next(iter(loaded_hashes)),
        },
        "tacker-phase4-rollback-v1",
    )
    rollback_path = directory / "rollback.json"
    _atomic_write_json(rollback_path, rollback)
    artifacts.append(rollback_path)
    release = _sealed_document(
        "4dgaussians_tacker_phase4_release_selection",
        {
            "passed": True,
            "status": "ready_for_explicit_promotion",
            "automatic_default_replacement_performed": False,
            "profile_action": profile_action,
            "phase31": {
                "identity_sha256": sealed["phase31_identity_sha256"],
                "matrix_sha256": sealed["matrix_sha256"],
                "formal_set_sha256": sealed["formal_set_sha256"],
                "selection_sha256": sealed["selection_sha256"],
            },
            "phase4_identity_sha256": identity["sha256"],
            "deployment": admission["deployment"],
            "runtime": {
                "identity": identity["payload"]["runtime"],
                "library": build["runtime_library"],
            },
            "canary": file_artifact(canary_path, "canary artifact"),
            "rollback": file_artifact(rollback_path, "rollback artifact"),
            "generalization": [item["report"] for item in generalizations],
            "workload_selection_policy": "per_workload_key_no_global_variant",
        },
        "tacker-phase4-release-selection-v1",
    )
    release_path = directory / "release-selection.json"
    _atomic_write_json(release_path, release)
    artifacts.append(release_path)
    return {
        "canary": file_artifact(canary_path, "canary artifact"),
        "release": file_artifact(release_path, "release artifact"),
        "rollback": file_artifact(rollback_path, "rollback artifact"),
        "release_sha256": release["sha256"],
    }, artifacts


def _device_query(gpu, output):
    """Run the real CUDA query used by the preflight subprocess."""

    try:
        import torch
    except Exception as error:
        raise Phase4Error("cannot import PyTorch for CUDA preflight: {}".format(error))
    if type(gpu) is not int or gpu != 0:
        raise Phase4Error("Phase 4 requires logical CUDA device zero")
    if not torch.cuda.is_available():
        raise Phase4Error("torch.cuda.is_available() is false")
    visible = int(torch.cuda.device_count())
    if visible != 1:
        raise Phase4Error(
            "Phase 4 requires exactly one CUDA-visible device; observed {}".format(
                visible
            )
        )
    torch.cuda.set_device(gpu)
    properties = torch.cuda.get_device_properties(gpu)
    # This is deliberately a real device operation.  A successful nvidia-smi
    # query alone is not enough to establish that PyTorch can execute CUDA.
    smoke = torch.arange(17, dtype=torch.float32, device="cuda:{}".format(gpu))
    smoke_sum = float(smoke.sum().item())
    torch.cuda.synchronize(gpu)
    document = {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_phase4_device_query",
        "passed": smoke_sum == 136.0,
        "logical_device": gpu,
        "visible_device_count": visible,
        "name": str(torch.cuda.get_device_name(gpu)).strip(),
        "compute_capability": list(torch.cuda.get_device_capability(gpu)),
        "multiprocessor_count": int(properties.multi_processor_count),
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "python_version": "{}.{}".format(sys.version_info[0], sys.version_info[1]),
        "cuda_smoke_sum": smoke_sum,
    }
    _atomic_write_json(output, document)
    return document


def _extension_query(output):
    """Import and hash the exact extension objects selected by Python."""

    try:
        import diff_gaussian_rasterization._C as raster_binary
        import simple_knn._C as simple_knn_binary
        import tacker_4dgs_head._C as head_binary
    except Exception as error:
        raise Phase4Error("cannot import freshly built CUDA extensions: {}".format(error))
    binaries = {}
    for name, module in (
        ("rasterizer", raster_binary),
        ("simple_knn", simple_knn_binary),
        ("head", head_binary),
    ):
        path = getattr(module, "__file__", None)
        if not isinstance(path, str) or not path:
            raise Phase4Error("{} extension has no import path".format(name))
        binaries[name] = file_artifact(path, "imported {} extension".format(name))
    document = {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_phase4_extension_binaries",
        "passed": True,
        "binaries": binaries,
    }
    _atomic_write_json(output, document)
    return document


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Consume one canonical sealed Phase-3.1 run and execute the "
            "fail-closed Phase-4 qualification/release protocol"
        )
    )
    parser.add_argument("--phase31-run-root", required=True)
    parser.add_argument(
        "--phase31-identity-sha256", default=EXPECTED_PHASE31_IDENTITY
    )
    parser.add_argument(
        "--phase31-matrix-sha256", default=EXPECTED_PHASE31_MATRIX
    )
    parser.add_argument(
        "--phase31-formal-set-sha256", default=EXPECTED_PHASE31_FORMAL_SET
    )
    parser.add_argument(
        "--phase31-selection-sha256", default=EXPECTED_PHASE31_SELECTION
    )
    parser.add_argument(
        "--phase31-run-report-sha256",
        default=EXPECTED_PHASE31_RUN_REPORT_FILE,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tacker-root", required=True)
    parser.add_argument("--template-profile", required=True)
    parser.add_argument("--current-tacker-profile", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--configs", dest="config", required=True)
    parser.add_argument("--workload-name", default="flame_steak")
    parser.add_argument(
        "--primary-workload-key",
        default="flame_steak:14000:111525:1352x1014:sm_86",
    )
    parser.add_argument(
        "--primary-mix",
        choices=("raster_heavy", "balanced", "deformation_heavy"),
        default="balanced",
    )
    parser.add_argument("--primary-profile-arg", action="append", default=[])
    parser.add_argument("--iteration", type=int, default=14000)
    parser.add_argument("--image-width", type=int, default=1352)
    parser.add_argument("--image-height", type=int, default=1014)
    parser.add_argument("--gaussian-count", type=int, default=111525)
    parser.add_argument(
        "--generalization-workload",
        action="append",
        required=True,
        help=(
            "repeat exactly twice; JSON carries model/source/config, iteration, "
            "gaussian_count, expected image size, workload_key, and profile_args"
        ),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--nvcc", default="/usr/local/cuda-12.4/bin/nvcc")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ctest", default="ctest")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--expected-gpu-name", default="NVIDIA RTX A6000")
    parser.add_argument("--expected-python", default="3.10")
    parser.add_argument("--expected-torch", default="2.4.1")
    parser.add_argument("--expected-cuda", default="12.4")
    parser.add_argument("--leaf-views", type=int, default=2)
    parser.add_argument("--leaf-warmup", type=int, default=5)
    parser.add_argument("--leaf-repetitions", type=int, default=50)
    parser.add_argument("--sequence-trials", type=int, default=3)
    parser.add_argument("--generalization-trials", type=int, default=3)
    parser.add_argument("--long-frames", type=int, default=500)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=STAGES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plan-output",
        help="optional no-clobber JSON destination for --dry-run",
    )
    return parser


def _validate_cli_args(args, require_files):
    for label in (
        "phase31_identity_sha256",
        "phase31_matrix_sha256",
        "phase31_formal_set_sha256",
        "phase31_selection_sha256",
        "phase31_run_report_sha256",
    ):
        _require_sha256(getattr(args, label), "--{}".format(label.replace("_", "-")))
    canonical_pins = (
        (args.phase31_identity_sha256, EXPECTED_PHASE31_IDENTITY),
        (args.phase31_matrix_sha256, EXPECTED_PHASE31_MATRIX),
        (args.phase31_formal_set_sha256, EXPECTED_PHASE31_FORMAL_SET),
        (args.phase31_selection_sha256, EXPECTED_PHASE31_SELECTION),
        (args.phase31_run_report_sha256, EXPECTED_PHASE31_RUN_REPORT_FILE),
    )
    if any(observed != expected for observed, expected in canonical_pins):
        raise Phase4Error(
            "Phase 4 accepts only the canonical pinned Phase-3.1 identity/matrix/formal/selection/report"
        )
    if args.gpu != 0:
        raise Phase4Error("--gpu must be logical device 0")
    if args.physical_gpu < 0:
        raise Phase4Error("--physical-gpu must be non-negative")
    for label in (
        "iteration", "image_width", "image_height", "gaussian_count",
        "leaf_views", "leaf_repetitions", "sequence_trials",
        "generalization_trials", "timeout_seconds",
    ):
        if type(getattr(args, label)) is not int or getattr(args, label) <= 0:
            raise Phase4Error("--{} must be a positive integer".format(label.replace("_", "-")))
    if type(args.leaf_warmup) is not int or args.leaf_warmup < 0:
        raise Phase4Error("--leaf-warmup must be a non-negative integer")
    if type(args.long_frames) is not int or args.long_frames <= max(SEQUENCE_LENGTHS):
        raise Phase4Error("--long-frames must be greater than 50")
    if not isinstance(args.workload_name, str) or not SAFE_NAME_RE.match(args.workload_name):
        raise Phase4Error("--workload-name contains unsafe characters")
    if not isinstance(args.primary_workload_key, str) or not args.primary_workload_key:
        raise Phase4Error("--primary-workload-key must be non-empty")
    if args.dry_run and args.resume:
        raise Phase4Error("--dry-run and --resume are mutually exclusive")
    output = Path(args.output_dir).expanduser().resolve()
    phase31 = Path(args.phase31_run_root).expanduser().resolve()
    tacker = Path(args.tacker_root).expanduser().resolve()
    if require_files:
        if (
            _is_within(output, PROJECT_ROOT)
            or _is_within(output, tacker)
            or _is_within(output, phase31)
            or _is_within(phase31, output)
        ):
            raise Phase4Error(
                "--output-dir must be outside the project, Tacker runtime, and sealed Phase-3.1 trees"
            )
        _regular_file(args.python_executable, "Python executable", executable=True)
        _regular_file(args.template_profile, "disabled template profile")
        _regular_file(args.current_tacker_profile, "current Tacker profile")
        for label, value in (
            ("nvidia-smi", args.nvidia_smi),
            ("nvcc", args.nvcc),
            ("cmake", args.cmake),
            ("ctest", args.ctest),
        ):
            if shutil.which(value) is None and not (
                Path(value).is_absolute() and Path(value).is_file()
            ):
                raise Phase4Error("--{} executable is unavailable: {}".format(label, value))
        generalization = _parse_generalization_specs(args, require_files=True)
        primary = normalize_workload(_primary_workload(args), "primary workload", True)
        keys = [primary["workload_key"]] + [item["workload_key"] for item in generalization]
        if len(set(keys)) != len(keys):
            raise Phase4Error("primary/generalization workload_key values must be unique")
        pixels = [item["image_width"] * item["image_height"] for item in generalization]
        if len(set(pixels)) != 2:
            raise Phase4Error("generalization workloads must have different effective pixel counts")
        resolutions = [_resolution_argument(item["profile_args"]) for item in generalization]
        scales = [1 if value in (-1, 1) else value for value in resolutions]
        if 1 not in scales or max(scales) <= 1:
            raise Phase4Error(
                "generalization requires one native-resolution and one genuinely scaled workload"
            )
        if all(item["iteration"] == primary["iteration"] for item in generalization):
            raise Phase4Error(
                "at least one generalization workload must use a different checkpoint iteration"
            )
    else:
        _parse_generalization_specs(args, require_files=False)
    return args


def _generalization_cross_validation(generalizations):
    if len(generalizations) != 2:
        raise Phase4Error("exactly two generalization results are required")
    workloads = [item["workload"] for item in generalizations]
    pixels = [item["image_width"] * item["image_height"] for item in workloads]
    if len(set(pixels)) != 2:
        raise Phase4Error("measured generalization resolutions are not distinct")
    proxies = [item["balance_proxy"] for item in generalizations]
    keys = (
        "two_stream_to_split_serial_fps_ratio",
        "split_serial_to_serial_fps_ratio",
    )
    if all(
        math.isclose(
            float(proxies[0][key]),
            float(proxies[1][key]),
            rel_tol=1e-4,
            abs_tol=1e-6,
        )
        for key in keys
    ):
        raise Phase4Error(
            "the two real workloads did not produce measurably different overlap balance proxies"
        )
    return {
        "passed": True,
        "distinct_workload_keys": [item["workload_key"] for item in workloads],
        "effective_pixel_counts": pixels,
        "checkpoint_iterations": [item["iteration"] for item in workloads],
        "gaussian_counts": [item["gaussian_count"] for item in workloads],
        "balance_proxies_are_measurably_different": True,
        "interpretation": (
            "measured whole-run overlap proxy only; not a direct Raster/deformation kernel-time ratio"
        ),
    }


def _artifact_input(value):
    if isinstance(value, dict):
        if set(value) >= {"path", "sha256", "size_bytes"}:
            return {
                "path": value["path"],
                "sha256": value["sha256"],
                "size_bytes": value["size_bytes"],
            }
        return {key: _artifact_input(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_artifact_input(item) for item in value]
    return value


def _result_loaded_raster_hashes(results):
    hashes = set()

    def visit(value, key=None):
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif key in ("loaded_rasterizer_sha256", "rasterizer_binary_sha256"):
            hashes.add(value)

    visit(results)
    for value in hashes:
        _require_sha256(value, "loaded Raster binary SHA")
    if len(hashes) != 1:
        raise Phase4Error(
            "Phase-4 executions were not all bound to one loaded Raster binary: {!r}".format(
                sorted(hashes)
            )
        )
    return next(iter(hashes))


def _verify_build_artifacts_unchanged(build, loaded_raster_sha256=None):
    if file_artifact(
        build["runtime_library"]["path"], "built Tacker runtime library"
    ) != build["runtime_library"]:
        raise Phase4Error("built libtacker_runtime.so changed during Phase 4")
    binaries = _required_mapping(
        build.get("extension_binaries"), "built extension binaries"
    )
    for name, expected in binaries.items():
        if file_artifact(expected.get("path", ""), "built {} extension".format(name)) != expected:
            raise Phase4Error("built {} extension changed during Phase 4".format(name))
    if (
        loaded_raster_sha256 is not None
        and binaries.get("rasterizer", {}).get("sha256") != loaded_raster_sha256
    ):
        raise Phase4Error(
            "profile executions did not load the Raster binary built by Phase 4"
        )


def _completed_report(journal):
    if journal.state.get("status") != "succeeded":
        return None
    stages = journal.state.get("stages")
    if (
        not isinstance(stages, list)
        or [item.get("name") for item in stages] != list(STAGES)
        or any(item.get("status") != "succeeded" for item in stages)
    ):
        raise Phase4Error("completed checkpoint does not contain every succeeded stage")
    for record in stages:
        name = record["name"]
        saved_artifacts = record.get("artifacts")
        if not isinstance(saved_artifacts, list):
            raise Phase4Error("completed stage {} has no artifact list".format(name))
        for saved_artifact in saved_artifacts:
            if (
                not _is_within(saved_artifact.get("path", ""), journal.root)
                or file_artifact(
                    saved_artifact.get("path", ""), "completed stage artifact"
                )
                != saved_artifact
            ):
                raise Phase4Error("completed stage {} artifact changed".format(name))
        result_artifact = _required_mapping(
            record.get("result_artifact"),
            "completed stage {} result artifact".format(name),
        )
        if result_artifact not in saved_artifacts:
            raise Phase4Error("completed stage {} result is not sealed".format(name))
        sealed_result = load_json(
            result_artifact["path"], "completed stage {} result".format(name)
        )
        if (
            sealed_result != record.get("result")
            or record.get("result_sha256")
            != sha256_json(sealed_result, "tacker-phase4-stage-result-v1")
        ):
            raise Phase4Error("completed stage {} result changed".format(name))
    saved = journal.state.get("report")
    if not isinstance(saved, dict):
        raise Phase4Error("completed checkpoint has no report artifact")
    if file_artifact(saved.get("path", ""), "completed Phase-4 report") != saved:
        raise Phase4Error("completed Phase-4 report changed after publication")
    return load_json(saved["path"], "completed Phase-4 report")


def run_phase4(args):
    _validate_cli_args(args, require_files=True)
    identity = build_identity(args)
    journal = Journal(args.output_dir, identity, args.resume)
    try:
        completed = _completed_report(journal)
        if completed is not None:
            return completed

        def execute(name, inputs, action):
            result = journal.run(
                name,
                {"identity_sha256": identity["sha256"], "inputs": _artifact_input(inputs)},
                action,
            )
            if args.stop_after == name:
                raise StopIteration(name)
            return result

        preflight = execute(
            "preflight",
            {"device": identity["payload"]["device"]},
            lambda directory: _preflight_stage(args, directory),
        )
        build = execute(
            "build-and-cuda",
            {"preflight": preflight["report"], "runtime": identity["payload"]["runtime"]},
            lambda directory: _build_and_cuda_stage(args, identity, directory),
        )
        _verify_build_artifacts_unchanged(build)
        sealed = execute(
            "verify-phase31-seal",
            identity["payload"]["phase31"],
            lambda directory: _verify_phase31_stage(args, directory),
        )
        verify_sealed_artifacts_unchanged(sealed)
        resources = execute(
            "finalist-resources-numerics",
            {"seal": sealed["seal_report"], "build": build["report"]},
            lambda directory: _finalist_resources_stage(
                args, identity, sealed, directory
            ),
        )
        quality = execute(
            "quality-50-view",
            {"seal": sealed["seal_report"], "resources": resources["report"]},
            lambda directory: _quality_50_stage(
                args, identity, sealed, resources, directory
            ),
        )
        formal = execute(
            "formal-10x50-abba",
            {
                "seal": sealed["seal_report"],
                "quality": quality["report"],
                "resources": resources["report"],
            },
            lambda directory: _formal_benchmark_stage(
                args, identity, sealed, quality, resources, directory
            ),
        )
        assert_identity_unchanged(args, identity)
        verify_sealed_artifacts_unchanged(sealed)
        admission = execute(
            "selection-admission",
            {
                "formal": formal["report"],
                "quality": quality["report"],
                "resources": resources["report"],
            },
            lambda directory: _selection_admission_stage(
                args, sealed, resources, quality, formal, directory
            ),
        )
        sequence = execute(
            "sequence-1-2-50-long",
            admission["report"],
            lambda directory: _sequence_regression_stage(
                args, identity, admission, directory
            ),
        )
        rerun = execute(
            "enabled-profile-rerun",
            {"admission": admission["report"], "sequence": sequence["report"]},
            lambda directory: _enabled_rerun_stage(
                args, identity, admission, directory
            ),
        )
        fallback = execute(
            "fallback-smoke",
            {
                "admission": admission["report"],
                "seal": sealed["seal_report"],
                "correctness": quality["correctness"],
            },
            lambda directory: _fallback_smoke_stage(
                args, identity, sealed, admission, quality, directory
            ),
        )
        generalization_1 = execute(
            "generalization-workload-1",
            identity["payload"]["generalization_workloads"][0],
            lambda directory: _generalization_stage(
                args, identity, 1, directory
            ),
        )
        generalization_2 = execute(
            "generalization-workload-2",
            identity["payload"]["generalization_workloads"][1],
            lambda directory: _generalization_stage(
                args, identity, 2, directory
            ),
        )
        generalizations = [generalization_1, generalization_2]
        cross_generalization = _generalization_cross_validation(generalizations)
        all_execution_results = {
            "formal": formal,
            "sequence": sequence,
            "rerun": rerun,
            "fallback": fallback,
            "generalization": generalizations,
        }
        loaded_raster_sha256 = _result_loaded_raster_hashes(all_execution_results)
        assert_identity_unchanged(args, identity)
        verify_sealed_artifacts_unchanged(sealed)
        _verify_build_artifacts_unchanged(build, loaded_raster_sha256)
        publication = execute(
            "canary-release-rollback",
            {
                "admission": admission["report"],
                "correctness": quality["correctness"],
                "sequence": sequence["report"],
                "rerun": rerun["metadata"],
                "fallback": fallback["report"],
                "generalization": [item["report"] for item in generalizations],
                "cross_generalization": cross_generalization,
                "loaded_rasterizer_sha256": loaded_raster_sha256,
            },
            lambda directory: _canary_release_rollback_stage(
                args,
                identity,
                sealed,
                build,
                admission,
                quality,
                sequence,
                rerun,
                fallback,
                generalizations,
                directory,
            ),
        )
        loaded_raster_sha256 = _result_loaded_raster_hashes(
            dict(all_execution_results, publication=load_json(
                publication["rollback"]["path"], "rollback publication"
            ))
        )
        assert_identity_unchanged(args, identity)
        verify_sealed_artifacts_unchanged(sealed)
        _verify_build_artifacts_unchanged(build, loaded_raster_sha256)
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "passed": True,
            "generated_at_utc": utc_now(),
            "identity": identity,
            "sealed_phase31": {
                "identity_sha256": sealed["phase31_identity_sha256"],
                "matrix_sha256": sealed["matrix_sha256"],
                "formal_set_sha256": sealed["formal_set_sha256"],
                "selection_sha256": sealed["selection_sha256"],
                "seal_report": sealed["seal_report"],
            },
            "candidate_generation_allowed": False,
            "candidate_scope": (
                "exact Phase-3.1 valid finalists plus serial/two_stream/current_tacker"
            ),
            "runtime": {
                "identity": identity["payload"]["runtime"],
                "built_library": build["runtime_library"],
                "extension_binaries": build["extension_binaries"],
                "verified_unchanged_before_and_after": True,
            },
            "loaded_rasterizer_sha256": loaded_raster_sha256,
            "deployment": admission["deployment"],
            "generalization_cross_validation": cross_generalization,
            "publication": publication,
            "stage_reports": {
                item["name"]: item["result"]
                for item in journal.state["stages"]
                if item.get("status") == "succeeded"
            },
        }
        report_path = journal.root / "phase4-report.json"
        _atomic_write_json(report_path, report)
        journal.state["status"] = "succeeded"
        journal.state["completed_at_utc"] = utc_now()
        journal.state["report"] = file_artifact(report_path, "Phase-4 report")
        journal.save()
        return report
    except StopIteration as stopped:
        return {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_phase4_stopped_checkpoint",
            "passed": True,
            "stopped_after": str(stopped),
            "resume_required": True,
            "checkpoint": file_artifact(journal.path, "Phase-4 checkpoint"),
        }
    finally:
        journal.close()


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in ("_device-query", "_extension-query"):
        internal = values[0]
        child = argparse.ArgumentParser(description="internal Phase-4 device/build query")
        if internal == "_device-query":
            child.add_argument("--gpu", type=int, required=True)
        child.add_argument("--output", required=True)
        child_args = child.parse_args(values[1:])
        try:
            if internal == "_device-query":
                result = _device_query(child_args.gpu, child_args.output)
            else:
                result = _extension_query(child_args.output)
        except Phase4Error as error:
            print("Phase-4 internal query failed: {}".format(error), file=sys.stderr)
            return 2
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    parser = _parser()
    args = parser.parse_args(values)
    try:
        _validate_cli_args(args, require_files=not args.dry_run)
        if args.dry_run:
            plan = _dry_run_plan(args)
            if args.plan_output:
                _atomic_write_json(args.plan_output, plan)
            print(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False))
            return 0
        result = run_phase4(args)
    except (Phase4Error, OSError, ValueError) as error:
        print("Phase-4 qualification failed: {}".format(error), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
