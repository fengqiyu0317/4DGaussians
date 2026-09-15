#!/usr/bin/env python3
"""Run the independent, resumable Phase-3.1 Tacker search on one RTX A6000.

Phase 3.1 deliberately starts a new evidence chain.  The sealed Phase-3
366-candidate run is an input baseline, never an append target.  This runner
builds an exhaustive C0--C2 schema-v2 matrix, stages C3 and C4 generation from
measured parent families, and retains the exact profile/matrix/report binding
used for every short-screening result.

Formal candidates are the global screening top-K union the best successful
representative of every generated C0--C4 family.  A correctness-invalid
candidate is replaced only from its own family.  Every retained candidate is
then measured in one 10x50 ABBA run with serial, two_stream, and the current
Tacker deployment.  Only a new challenger winner causes a disabled winner
qualification profile to be materialized.
"""

from __future__ import print_function

import argparse
from copy import deepcopy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
PHASE3_RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_tacker_phase3.py"
AUTOTUNE_PATH = PROJECT_ROOT / "scripts" / "tacker_autotune.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE3 = _load_module("tacker_phase3_runner_helpers", PHASE3_RUNNER_PATH)
AUTOTUNE = _load_module("tacker_phase31_autotune", AUTOTUNE_PATH)

SCHEMA_VERSION = 1
REPORT_KIND = "4dgaussians_tacker_phase31_run"
STATE_KIND = "4dgaussians_tacker_phase31_checkpoint"
DRY_RUN_KIND = "4dgaussians_tacker_phase31_dry_run_plan"
RESOURCE_KIND = "tacker_phase31_a6000_resource_query"
SCREENING_KIND = "tacker_phase31_screening_batch"
QUALIFICATION_KIND = "tacker_phase31_candidate_qualification"
SELECTION_KIND = "tacker_phase31_selection"
STOP_POINTS = (
    "preflight",
    "base",
    "c3",
    "c4",
    "screen",
    "quality",
    "formal",
    "nsight",
)
SEARCH_FAMILIES = ("c0", "c1", "c2", "c3", "c4")
EXPECTED_WORKLOAD = "flame_steak"
EXPECTED_ITERATION = 14000
EXPECTED_RESOLUTION = (1352, 1014)
EXPECTED_GAUSSIANS = 111525
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

RASTER_FAMILY_CONTRACTS = {
    "legacy_pos_l1_v1": {
        "abi_version": 1,
        "resource_family": "first_linear_head_v1",
        "capability_abi": "mixed_render_head_abi",
        "capability_enabled": "mixed_render_head",
        "capability_symbol": "mixed_symbol",
        "capability_manifest": "mixed_manifest_sha256",
        "symbol": "tacker_mix_render_head_v1",
        "manifest": "tacker_mixed_render_head_v1.json",
        "worker_groups": (1,),
    },
    "first_linear_heads_v2": {
        "abi_version": 2,
        "resource_family": "first_linear_heads_v2",
        "capability_head_manifest": "head_multi_manifest_sha256",
        "capability_abi": "mixed_render_heads_abi",
        "capability_enabled": "mixed_render_heads",
        "capability_symbol": "mixed_multi_symbol",
        "capability_manifest": "mixed_multi_manifest_sha256",
        "symbol": "tacker_mix_render_heads_v2",
        "manifest": "tacker_mixed_render_heads_v2.json",
        "worker_groups": (1, 2, 3, 4, 5),
    },
    "packed_first_linear_v3": {
        "abi_version": 3,
        "resource_family": "packed_first_linear_v3",
        "capability_head_manifest": "mixed_packed_head_manifest_sha256",
        "capability_abi": "mixed_render_packed_heads_abi",
        "capability_enabled": "mixed_render_packed_heads",
        "capability_symbol": "mixed_packed_symbol",
        "capability_manifest": "mixed_packed_manifest_sha256",
        "symbol": "tacker_mix_render_packed_heads_v3",
        "manifest": "tacker_mixed_render_packed_heads_v3.json",
        "worker_groups": (1, 2, 3, 4, 5),
    },
    "whole_heads_v4": {
        "abi_version": 4,
        "resource_family": "whole_heads_v4",
        "capability_head_manifest": "mixed_whole_head_manifest_sha256",
        "capability_abi": "mixed_render_whole_heads_abi",
        "capability_enabled": "mixed_render_whole_heads",
        "capability_symbol": "mixed_whole_symbol",
        "capability_manifest": "mixed_whole_manifest_sha256",
        "symbol": "tacker_mix_render_whole_heads_v4",
        "manifest": "tacker_mixed_render_whole_heads_v4.json",
        "worker_groups": (1, 2, 3, 4, 5),
    },
}

HEAD_SYMBOLS = {
    "multi_solo": "tacker_head_linear_multi_solo_v2",
    "multi_gptb": "tacker_head_linear_multi_gptb_v2",
    "packed_gptb": "tacker_head_linear_packed_gptb_v2",
    "whole_head_gptb": "tacker_whole_head_gptb_v2",
}
REQUIRED_PTXAS_SYMBOLS = tuple(
    contract["symbol"]
    for contract in RASTER_FAMILY_CONTRACTS.values()
) + tuple(HEAD_SYMBOLS.values())
REQUIRED_CUDA_SUITE_SCOPES = {
    "head_extension_cuda": ("head_v2_cuda",),
    "raster_abi1_4_cuda": (
        "raster_abi1",
        "raster_abi2",
        "raster_abi3",
        "raster_abi4",
    ),
    "runtime_c3_c4_fallback": (
        "runtime_c3",
        "runtime_c4",
        "runtime_fallback",
        "execution_counts",
    ),
}
EXPECTED_FORMAL_EXECUTION_COUNTS = {
    "input_frames": 50,
    "full_deformation": 1,
    "prefix": 49,
    "mixed_launches": 49,
    "suffix": 49,
    "solo_raster": 1,
    "outputs": 50,
    "selected_head_evaluations_per_head": 50,
}
SEALED_PHASE3_ROOT = (
    PROJECT_ROOT / "tacker_profiles" / "baselines" / "a6000_phase3_20260912"
).resolve()


class Phase31Error(RuntimeError):
    """A Phase-3.1 orchestration or evidence contract failed closed."""


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
        raise Phase31Error("value is not finite canonical JSON: {}".format(error))


def sha256_json(value, domain=None):
    payload = value if domain is None else {"domain": domain, "payload": value}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path):
    return PHASE3.sha256_file(path)


def load_json(path, label="JSON"):
    try:
        return PHASE3.load_json(path, label)
    except Exception as error:
        raise Phase31Error(str(error))


def atomic_write_json(path, value):
    PHASE3.atomic_write_json(path, value)


def artifact(path):
    try:
        return PHASE3.artifact(path)
    except Exception as error:
        raise Phase31Error(str(error))


def invoke(runner, argv, cwd=PROJECT_ROOT, allowed=(0,), env=None):
    try:
        return PHASE3.invoke(
            runner, argv, cwd=cwd, allowed=allowed, env=env
        )
    except Exception as error:
        raise Phase31Error(str(error))


def _script(name):
    return str((PROJECT_ROOT / "scripts" / name).resolve())


def _is_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _assert_independent_output(output_dir, resume):
    root = Path(output_dir).expanduser().resolve()
    if _is_within(root, SEALED_PHASE3_ROOT):
        raise Phase31Error(
            "Phase 3.1 output must not be inside the sealed Phase-3 evidence root"
        )
    if resume:
        if not (root / "phase31-state.json").is_file():
            raise Phase31Error("--resume requires a Phase-3.1 checkpoint")
    elif root.exists():
        raise Phase31Error(
            "refusing to append to an existing output directory without --resume"
        )
    for parent in (root,) + tuple(root.parents):
        if parent == root:
            continue
        if (parent / "phase3-state.json").is_file():
            raise Phase31Error(
                "Phase 3.1 output must not be nested in a Phase-3 run"
            )
        if parent == PROJECT_ROOT.parent:
            break
    return root


class Journal(object):
    """Hash-bound stage journal with durable artifacts and exact resume."""

    def __init__(self, output_dir, identity, resume):
        self.root = _assert_independent_output(output_dir, resume)
        self.path = self.root / "phase31-state.json"
        self.lock_path = self.root / ".phase31.lock"
        if resume:
            self.state = load_json(self.path, "Phase-3.1 checkpoint")
            if (
                self.state.get("schema_version") != SCHEMA_VERSION
                or self.state.get("kind") != STATE_KIND
                or self.state.get("identity") != identity
            ):
                raise Phase31Error("Phase-3.1 resume identity changed")
        else:
            self.root.mkdir(parents=True)
            self.state = {
                "schema_version": SCHEMA_VERSION,
                "kind": STATE_KIND,
                "identity": identity,
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "status": "running",
                "stages": [],
            }
            self.save()
        self.lock_handle = self.lock_path.open("a+")
        try:
            fcntl.flock(
                self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BaseException:
            self.lock_handle.close()
            self.lock_handle = None
            raise

    def close(self):
        if getattr(self, "lock_handle", None) is not None:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
            self.lock_handle.close()
            self.lock_handle = None

    def save(self):
        self.state["updated_at_utc"] = utc_now()
        atomic_write_json(self.path, self.state)

    def _record(self, name):
        matches = [item for item in self.state["stages"] if item.get("name") == name]
        if len(matches) > 1:
            raise Phase31Error("checkpoint has duplicate stage {}".format(name))
        return matches[0] if matches else None

    def run(self, name, inputs, action):
        if not NAME_RE.match(name):
            raise Phase31Error("unsafe journal stage name")
        digest = sha256_json(inputs, "tacker-phase31-stage-input-v1")
        record = self._record(name)
        if record is not None and record.get("input_sha256") != digest:
            raise Phase31Error("stage {} inputs changed during resume".format(name))
        if record is not None and record.get("status") == "succeeded":
            for saved in record.get("artifacts", []):
                if artifact(saved.get("path", "")) != saved:
                    raise Phase31Error(
                        "stage {} artifact changed during resume".format(name)
                    )
            return deepcopy(record.get("result"))
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
            saved = [artifact(path) for path in paths]
        except BaseException as error:
            record["status"] = "failed"
            record["error"] = "{}: {}".format(type(error).__name__, error)
            record["finished_at_utc"] = utc_now()
            self.state["status"] = "failed"
            self.save()
            raise
        record.update(
            {
                "status": "succeeded",
                "result": result,
                "artifacts": saved,
                "finished_at_utc": utc_now(),
            }
        )
        self.state["status"] = "running"
        self.save()
        return deepcopy(result)


def _manifest_paths():
    raster_abi = PROJECT_ROOT / "submodules" / "depth-diff-gaussian-rasterization" / "abi"
    return {
        family: raster_abi / contract["manifest"]
        for family, contract in RASTER_FAMILY_CONTRACTS.items()
    }, PROJECT_ROOT / "tacker_ext" / "abi" / "head_linear_v2.json"


def _configuration_chain(path):
    try:
        return PHASE3._configuration_chain(path)
    except Exception as error:
        raise Phase31Error(str(error))


def physical_gpu_from_environment():
    try:
        return PHASE3.physical_gpu_from_environment()
    except Exception as error:
        raise Phase31Error(str(error))


def _regular_executable(value, label):
    """Resolve one command to the exact regular executable that will be run."""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise Phase31Error("{} must name an executable".format(label))
    candidate = value
    if os.path.sep not in value:
        candidate = shutil.which(value)
        if candidate is None:
            raise Phase31Error("{} executable was not found: {}".format(label, value))
    target = Path(candidate).expanduser().resolve()
    if not target.is_file() or target.is_symlink() or not os.access(str(target), os.X_OK):
        raise Phase31Error("{} is not a regular executable: {}".format(label, target))
    return target


def _validate_cuda_suite_report(document):
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or document.get("kind")
        != "4dgaussians_tacker_phase31_cuda_validation"
        or document.get("passed") is not True
    ):
        raise Phase31Error(
            "CUDA suite must be a passing Phase-3.1 C0--C4 validation report"
        )
    abi = document.get("abi")
    if not isinstance(abi, dict):
        raise Phase31Error("CUDA suite omitted ABI evidence")
    for key in (
        "head_v2_manifest_sha256",
        "raster_v1_manifest_sha256",
        "raster_v2_manifest_sha256",
        "raster_v3_manifest_sha256",
        "raster_v4_manifest_sha256",
    ):
        if SHA256_RE.match(abi.get(key, "")) is None:
            raise Phase31Error("CUDA suite omitted {}".format(key))
    suites = document.get("test_suites")
    if not isinstance(suites, dict):
        raise Phase31Error("CUDA suite omitted test_suites")
    missing_suites = sorted(set(REQUIRED_CUDA_SUITE_SCOPES) - set(suites))
    if missing_suites:
        raise Phase31Error(
            "CUDA suite omitted required test suites: {}".format(
                ", ".join(missing_suites)
            )
        )
    suite_log_paths = []
    for name, suite in suites.items():
        log_artifact = suite.get("log_artifact") if isinstance(suite, dict) else None
        if (
            not isinstance(name, str)
            or not isinstance(suite, dict)
            or type(suite.get("passed")) is not int
            or type(suite.get("total")) is not int
            or suite["passed"] <= 0
            or suite["passed"] != suite["total"]
            or not isinstance(log_artifact, dict)
            or type(log_artifact.get("size_bytes")) is not int
            or log_artifact["size_bytes"] <= 0
            or artifact(log_artifact.get("path", "")) != log_artifact
        ):
            raise Phase31Error(
                "CUDA suite {} is not an exact pass with a sealed log".format(name)
            )
        suite_log_paths.append(str(Path(log_artifact["path"]).resolve()))
        required_scope = REQUIRED_CUDA_SUITE_SCOPES.get(name)
        if required_scope is not None and suite.get("scope") != list(required_scope):
            raise Phase31Error("CUDA suite {} coverage scope changed".format(name))
    if len(set(suite_log_paths)) != len(suite_log_paths):
        raise Phase31Error("each CUDA test suite must bind a distinct log artifact")
    ptxas = document.get("ptxas")
    ptxas_artifact = ptxas.get("artifact") if isinstance(ptxas, dict) else None
    if (
        not isinstance(ptxas, dict)
        or ptxas.get("passed") is not True
        or ptxas.get("sm_target") != "sm_86"
        or not isinstance(ptxas_artifact, dict)
        or artifact(ptxas_artifact.get("path", "")) != ptxas_artifact
    ):
        raise Phase31Error("CUDA suite omitted sealed sm_86 ptxas evidence")
    kernels = ptxas.get("kernels")
    if not isinstance(kernels, dict):
        raise Phase31Error("ptxas evidence omitted kernel resource records")
    for symbol in REQUIRED_PTXAS_SYMBOLS:
        facts = kernels.get(symbol)
        if not isinstance(facts, dict) or facts.get("accepted") is not True:
            raise Phase31Error("ptxas prefilter rejected or omitted {}".format(symbol))
        for key, minimum in (
            ("registers_per_thread", 1),
            ("static_shared_bytes", 0),
            ("local_bytes_per_thread", 0),
        ):
            _required_int(facts, key, "ptxas {}".format(symbol), minimum)
    return document


def _identity(args):
    physical_gpu = physical_gpu_from_environment()
    model = Path(args.model_path).expanduser().resolve()
    source = Path(args.source_path).expanduser().resolve()
    if not model.is_dir() or not source.is_dir():
        raise Phase31Error("model and source paths must be existing directories")
    nvidia_smi = _regular_executable(args.nvidia_smi, "--nvidia-smi")
    required = {
        "config": Path(args.config).expanduser().resolve(),
        "current_tacker_profile": Path(args.current_tacker_profile).expanduser().resolve(),
        "template_profile": Path(args.template_profile).expanduser().resolve(),
        "cuda_suite_report": Path(args.cuda_suite_report).expanduser().resolve(),
        "python_executable": Path(args.python_executable).expanduser().resolve(),
        "nvidia_smi_executable": nvidia_smi,
    }
    scripts = {
        "runner31": SCRIPT_PATH,
        "runner3_helpers": PHASE3_RUNNER_PATH,
        "autotune": AUTOTUNE_PATH,
        "benchmark": Path(_script("benchmark_tacker_fps.py")),
        "quality": Path(_script("validate_tacker_modes.py")),
        "leaf": PROJECT_ROOT / "profile_tacker_leaves.py",
        "top3": Path(_script("profile_tacker_top3.py")),
        "nsight": Path(args.profile_nsight_script).expanduser().resolve(),
    }
    raster_root = PROJECT_ROOT / "submodules" / "depth-diff-gaussian-rasterization"
    runtime_sources = {
        "profile_render": PROJECT_ROOT / "profile_render.py",
        "tacker_pipeline": PROJECT_ROOT / "gaussian_renderer" / "tacker_pipeline.py",
        "raster_python_binding": raster_root / "diff_gaussian_rasterization" / "__init__.py",
        "raster_extension_binding": raster_root / "ext.cpp",
        "raster_cuda_dispatch": raster_root / "rasterize_points.cu",
        "raster_mixed_kernels": raster_root / "cuda_rasterizer" / "tacker_mixed.cu",
        "raster_mixed_header": raster_root / "cuda_rasterizer" / "tacker_mixed.h",
    }
    raster_manifests, head_manifest = _manifest_paths()
    manifests = dict(raster_manifests)
    manifests["head_linear_v2"] = head_manifest
    for label, path in (
        list(required.items())
        + list(scripts.items())
        + list(runtime_sources.items())
        + list(manifests.items())
    ):
        if not path.is_file() or path.is_symlink():
            raise Phase31Error("{} file is missing or a symlink: {}".format(label, path))
    suite = _validate_cuda_suite_report(load_json(required["cuda_suite_report"], "CUDA suite"))
    suite_abi = suite["abi"]
    expected_suite_hashes = {
        "head_v2_manifest_sha256": sha256_file(head_manifest),
        "raster_v1_manifest_sha256": sha256_file(raster_manifests["legacy_pos_l1_v1"]),
        "raster_v2_manifest_sha256": sha256_file(raster_manifests["first_linear_heads_v2"]),
        "raster_v3_manifest_sha256": sha256_file(raster_manifests["packed_first_linear_v3"]),
        "raster_v4_manifest_sha256": sha256_file(raster_manifests["whole_heads_v4"]),
    }
    if any(suite_abi.get(key) != value for key, value in expected_suite_hashes.items()):
        raise Phase31Error("CUDA suite ABI hashes differ from the source manifests")
    current = load_json(required["current_tacker_profile"], "current profile")
    template = load_json(required["template_profile"], "qualification template")
    current_gate = (
        current.get("admission")
        if current.get("schema_version") == 1
        else current.get("deployment")
    )
    if current_gate != {"enabled": True, "valid": True}:
        raise Phase31Error("current Tacker profile is not admitted")
    if (
        template.get("schema_version") != 2
        or template.get("deployment") != {"enabled": False, "valid": False}
    ):
        raise Phase31Error("qualification template must be disabled schema-v2")
    manifest = current.get("manifest")
    current_pb = manifest.get("persistent_blocks") if isinstance(manifest, dict) else None
    if type(current_pb) is not int or current_pb <= 0:
        raise Phase31Error("current profile has no positive persistent_blocks")
    iteration = model / "point_cloud" / "iteration_{}".format(args.iteration)
    workload_paths = {
        "cfg_args": model / "cfg_args",
        "point_cloud.ply": iteration / "point_cloud.ply",
        "deformation.pth": iteration / "deformation.pth",
        "deformation_table.pth": iteration / "deformation_table.pth",
        "poses_bounds.npy": source / "poses_bounds.npy",
    }
    workload_files = {}
    for name, path in sorted(workload_paths.items()):
        if not path.is_file() or path.is_symlink():
            raise Phase31Error("required workload file is missing: {}".format(path))
        workload_files[name] = {"path": str(path), "sha256": sha256_file(path)}
    payload = {
        "paths": {
            "model": str(model),
            "source": str(source),
            "python": str(required["python_executable"]),
            "nvidia_smi": str(nvidia_smi),
        },
        "files": {name: artifact(path) for name, path in sorted(required.items())},
        "scripts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in sorted(scripts.items())},
        "runtime_sources": {
            name: artifact(path) for name, path in sorted(runtime_sources.items())
        },
        "manifests": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in sorted(manifests.items())},
        "workload_files": workload_files,
        "configuration_chain": _configuration_chain(args.config),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "FOURDGS_SOURCE_COMMIT",
                "FOURDGS_RASTERIZER_COMMIT",
                "FOURDGS_SIMPLE_KNN_COMMIT",
                "NSYS_BIN",
            )
        },
        "workload": {
            "name": args.workload_name,
            "iteration": args.iteration,
            "split": args.split,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "gaussian_count": args.gaussian_count,
            "head_rows": args.head_rows,
            "logical_gpu": args.gpu,
            "physical_gpu": physical_gpu,
        },
        "search": {
            "current_persistent_blocks": current_pb,
            "extra_persistent_blocks": sorted(set(args.persistent_block)),
            "packed_persistent_blocks": sorted(set(args.packed_persistent_block)),
            "whole_head_persistent_blocks": sorted(set(args.whole_head_persistent_block)),
            "c3_top_k": args.c3_top_k,
            "c4_top_k_per_family": args.c4_top_k_per_family,
            "formal_top_k": args.top_k,
            "screen_batch_size": args.screen_batch_size,
            "screen_frames": args.screen_frames,
            "screen_warmup": args.screen_warmup,
            "screen_trials": args.screen_trials,
            "seed": args.seed,
            "leaf_views": args.leaf_views,
            "leaf_warmup": args.leaf_warmup,
            "leaf_repetitions": args.leaf_repetitions,
            "timeout_seconds": args.timeout_seconds,
        },
    }
    return {"sha256": sha256_json(payload, "tacker-phase31-identity-v1"), "payload": payload}


def _assert_identity_unchanged(args, expected):
    if _identity(args) != expected:
        raise Phase31Error("Phase-3.1 code, profile, workload, or ABI inputs changed")


def _required_int(mapping, key, label, minimum=0):
    value = mapping.get(key) if isinstance(mapping, dict) else None
    if type(value) is not int or value < minimum:
        raise Phase31Error("{} requires integer {} >= {}".format(label, key, minimum))
    return value


def _validate_raster_resource(raw, family, worker_groups, logical_gpu, sm_count):
    contract = RASTER_FAMILY_CONTRACTS[family]
    label = "Raster {} WG{}".format(family, worker_groups)
    if not isinstance(raw, dict):
        raise Phase31Error("resource query omitted {}".format(label))
    if (
        raw.get("abi_version") != contract["abi_version"]
        or raw.get("worker_groups") != worker_groups
        or raw.get("device_ordinal") != logical_gpu
        or raw.get("compute_capability_major") != 8
        or raw.get("compute_capability_minor") != 6
        or raw.get("multiprocessor_count") != sm_count
    ):
        raise Phase31Error("{} identity/device facts changed".format(label))
    reported_family = raw.get("family", raw.get("backend_family"))
    expected_family = contract["resource_family"]
    if reported_family is not None and reported_family != expected_family:
        raise Phase31Error("{} reported the wrong backend family".format(label))
    expected_threads = 384 if family == "legacy_pos_l1_v1" else 256 + 128 * worker_groups
    if _required_int(raw, "physical_threads", label, 1) != expected_threads:
        raise Phase31Error("{} physical thread count changed".format(label))
    for key, minimum in (
        ("device_max_threads_per_block", expected_threads),
        ("device_max_threads_per_multiprocessor", 1),
        ("kernel_max_threads_per_block", expected_threads),
        ("registers_per_thread", 1),
        ("static_shared_bytes", 0),
        ("local_bytes_per_thread", 0),
        ("max_dynamic_shared_bytes", 0),
        ("active_blocks_per_multiprocessor", 1),
    ):
        _required_int(raw, key, label, minimum)
    occupancy = raw.get("occupancy")
    if (
        not isinstance(occupancy, (int, float))
        or isinstance(occupancy, bool)
        or not math.isfinite(float(occupancy))
        or not 0.0 < float(occupancy) <= 1.0
        or raw.get("launch_supported") is not True
    ):
        raise Phase31Error("{} is not launchable with valid occupancy".format(label))


def _validate_resource_query(document, logical_gpu, manifest_paths=None):
    """Validate loaded ABI2/ABI3/ABI4 symbols, manifests, and resources."""

    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or document.get("kind") != RESOURCE_KIND
        or document.get("passed") is not True
    ):
        raise Phase31Error("A6000 Phase-3.1 resource query did not pass")
    device = document.get("device")
    if (
        not isinstance(device, dict)
        or device.get("index") != logical_gpu
        or device.get("name") != "NVIDIA RTX A6000"
        or device.get("compute_capability") != [8, 6]
    ):
        raise Phase31Error("resource query is not bound to the selected A6000/sm_86")
    sm_count = _required_int(device, "sm_count", "CUDA device", 1)
    capabilities = document.get("raster_capabilities")
    families = document.get("families")
    if not isinstance(capabilities, dict) or not isinstance(families, dict):
        raise Phase31Error("resource query omitted Raster capabilities/families")
    if capabilities.get("resource_query_family_aware") is not True:
        raise Phase31Error("Raster resource query is not family-aware")
    if list(capabilities.get("supported_mixed_abis", [])) != [1, 2, 3, 4]:
        raise Phase31Error("Raster does not advertise exact ABI1--ABI4 support")
    required_families = [
        "first_linear_heads_v2",
        "packed_first_linear_v3",
        "whole_heads_v4",
    ]
    if list(capabilities.get("supported_backend_families", [])) != required_families:
        raise Phase31Error("Raster backend-family capability list changed")
    if manifest_paths is None:
        manifest_paths, head_manifest_path = _manifest_paths()
    else:
        manifest_paths, head_manifest_path = manifest_paths
    manifest_records = document.get("manifests")
    if not isinstance(manifest_records, dict):
        raise Phase31Error("resource query omitted manifest snapshots")
    for family, contract in RASTER_FAMILY_CONTRACTS.items():
        if (
            capabilities.get(contract["capability_abi"]) != contract["abi_version"]
            or capabilities.get(contract["capability_enabled"]) is not True
            or capabilities.get(contract["capability_symbol"]) != contract["symbol"]
        ):
            raise Phase31Error("loaded Raster {} symbol/ABI changed".format(family))
        expected_hash = sha256_file(manifest_paths[family])
        if (
            capabilities.get(contract["capability_manifest"]) != expected_hash
            or manifest_records.get(family, {}).get("sha256") != expected_hash
        ):
            raise Phase31Error("loaded Raster {} manifest hash changed".format(family))
        head_manifest_capability = contract.get("capability_head_manifest")
        if (
            head_manifest_capability is not None
            and capabilities.get(head_manifest_capability)
            != sha256_file(head_manifest_path)
        ):
            raise Phase31Error(
                "loaded Raster {} head dependency manifest changed".format(family)
            )
        family_doc = families.get(family)
        by_group = family_doc.get("worker_groups") if isinstance(family_doc, dict) else None
        if not isinstance(by_group, dict):
            raise Phase31Error("resource query omitted family {}".format(family))
        for worker_groups in contract["worker_groups"]:
            _validate_raster_resource(
                by_group.get(str(worker_groups)),
                family,
                worker_groups,
                logical_gpu,
                sm_count,
            )
    head = document.get("head_v2")
    head_caps = head.get("capabilities") if isinstance(head, dict) else None
    head_resources = head.get("resources") if isinstance(head, dict) else None
    symbols = head_caps.get("global_kernel_symbols") if isinstance(head_caps, dict) else None
    if (
        not isinstance(head_caps, dict)
        or head_caps.get("abi_version") != 2
        or head_caps.get("sm_target") != "sm_86"
        or symbols != HEAD_SYMBOLS
    ):
        raise Phase31Error("head ABI-v2 multi/packed/whole capabilities changed")
    if manifest_records.get("head_linear_v2", {}).get("sha256") != sha256_file(head_manifest_path):
        raise Phase31Error("head ABI-v2 manifest snapshot changed")
    kernels = head_resources.get("kernels") if isinstance(head_resources, dict) else None
    if not isinstance(kernels, dict) or head_resources.get("abi_version") != 2:
        raise Phase31Error("head ABI-v2 resource query is invalid")
    for name, symbol in HEAD_SYMBOLS.items():
        facts = kernels.get(symbol)
        if not isinstance(facts, dict):
            raise Phase31Error("head resource query omitted {}".format(symbol))
        expected_threads = [128] if name == "whole_head_gptb" else [128, 256, 384, 512, 640]
        if facts.get("worker_group_threads") != expected_threads:
            raise Phase31Error("{} worker-group resource grid changed".format(symbol))
        active = facts.get("active_blocks_per_sm")
        if (
            not isinstance(active, list)
            or len(active) != len(expected_threads)
            or any(type(value) is not int or value < 1 for value in active)
        ):
            raise Phase31Error("{} is not launchable".format(symbol))
        for key, minimum in (
            ("registers_per_thread", 1),
            ("static_shared_memory_bytes", 0),
            ("local_memory_bytes", 0),
            ("max_threads_per_block", expected_threads[-1]),
        ):
            _required_int(facts, key, symbol, minimum)
        if facts.get("ptx_version") != 86 or facts.get("binary_version") != 86:
            raise Phase31Error("{} is not an sm_86 binary".format(symbol))
    return device


def _resource_query(output, gpu):
    """Private child entry point querying the active compiled extensions."""

    try:
        import torch
        from diff_gaussian_rasterization import (
            tacker_capabilities,
            tacker_resource_requirements,
        )

        package_root = PROJECT_ROOT / "tacker_ext"
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        import tacker_4dgs_head

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        physical_gpu_from_environment()
        if gpu != 0 or torch.cuda.device_count() != 1:
            raise RuntimeError("Phase 3.1 requires one visible GPU at logical device 0")
        torch.cuda.set_device(gpu)
        properties = torch.cuda.get_device_properties(gpu)
        name = torch.cuda.get_device_name(gpu)
        capability = list(torch.cuda.get_device_capability(gpu))
        if name != "NVIDIA RTX A6000" or capability != [8, 6]:
            raise RuntimeError("Phase 3.1 requires NVIDIA RTX A6000/sm_86")
        raster_caps = dict(tacker_capabilities())
        families = {}
        for family, contract in RASTER_FAMILY_CONTRACTS.items():
            by_group = {}
            for worker_groups in contract["worker_groups"]:
                by_group[str(worker_groups)] = dict(
                    tacker_resource_requirements(
                        contract["abi_version"],
                        worker_groups,
                        contract["resource_family"],
                    )
                )
            families[family] = {"worker_groups": by_group}
        raster_manifests, head_manifest = _manifest_paths()
        manifests = {
            family: artifact(path) for family, path in raster_manifests.items()
        }
        manifests["head_linear_v2"] = artifact(head_manifest)
        document = {
            "schema_version": 1,
            "kind": RESOURCE_KIND,
            "passed": True,
            "device": {
                "index": gpu,
                "name": name,
                "compute_capability": capability,
                "sm_count": int(properties.multi_processor_count),
                "cuda_runtime": str(torch.version.cuda),
                "pytorch_version": str(torch.__version__),
            },
            "raster_capabilities": raster_caps,
            "families": families,
            "head_v2": {
                "capabilities": dict(tacker_4dgs_head.tacker_capabilities_v2()),
                "resources": dict(tacker_4dgs_head.tacker_resources_v2()),
            },
            "manifests": manifests,
        }
        _validate_resource_query(document, gpu)
    except Exception as error:
        document = {
            "schema_version": 1,
            "kind": RESOURCE_KIND,
            "passed": False,
            "error": str(error),
        }
    atomic_write_json(output, document)
    if not document["passed"]:
        print("Phase-3.1 resource query failed: {}".format(document["error"]), file=sys.stderr)
    return 0 if document["passed"] else 1


def _validate_matrix(matrix, require_c2_450=False):
    try:
        AUTOTUNE.validate_matrix(matrix)
    except Exception as error:
        raise Phase31Error("invalid Phase-3.1 matrix: {}".format(error))
    if matrix.get("schema_version") != 2:
        raise Phase31Error("Phase 3.1 requires a schema-v2 matrix")
    if require_c2_450:
        c2 = [item for item in matrix["candidates"] if item.get("search_family") == "c2"]
        head_sets = {tuple(item["selected_heads"]) for item in c2}
        expected_sets = sum(math.comb(5, size) for size in range(2, 6))
        if len(c2) != 450 or len(head_sets) != expected_sets:
            raise Phase31Error(
                "exhaustive C2 structural screen must contain 450/450 candidates"
            )
        observed = {}
        for item in c2:
            key = tuple(item["selected_heads"])
            observed.setdefault(key, set()).add(
                (item["worker_groups"], item["effective_persistent_blocks"])
            )
        for heads, grid in observed.items():
            if len(grid) != len(heads) * 6:
                raise Phase31Error("C2 head set {} is not exhaustive".format(heads))
    return matrix


def _profile_paths(matrix, profiles_dir):
    try:
        return PHASE3._profile_paths(matrix, profiles_dir)
    except Exception as error:
        raise Phase31Error(str(error))


def _make_profiles(journal, args, runner, matrix_path, matrix, label, resources_path):
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "resources": artifact(resources_path),
        "template": artifact(args.template_profile),
    }

    def action(directory):
        profiles_dir = directory / "profiles"
        completed = invoke(
            runner,
            [
                str(Path(args.python_executable).expanduser().resolve()),
                str(AUTOTUNE_PATH),
                "profiles",
                "--matrix",
                str(Path(matrix_path).resolve()),
                "--resources",
                str(Path(resources_path).resolve()),
                "--template",
                str(Path(args.template_profile).expanduser().resolve()),
                "--output-dir",
                str(profiles_dir.resolve()),
            ],
        )
        del completed
        paths = _profile_paths(matrix, profiles_dir)
        manifest = profiles_dir / "qualification_profiles.json"
        manifest_doc = load_json(manifest, "qualification profile manifest")
        result = {
            "label": label,
            "source_matrix_sha256": matrix["matrix_sha256"],
            "matrix_path": str(Path(matrix_path).resolve()),
            "profiles_dir": str(profiles_dir.resolve()),
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "candidate_count": len(paths),
        }
        if manifest_doc.get("matrix_sha256") != matrix["matrix_sha256"]:
            raise Phase31Error("profile manifest is bound to another matrix")
        return result, [manifest] + [paths[item["variant_id"]] for item in matrix["candidates"]]

    return journal.run("profiles-{}".format(label), inputs, action)


def candidate_batches(matrix, families, batch_size, only_digests=None):
    """Return deterministic family/head-count batches without cross-family mixing."""

    _validate_matrix(matrix)
    if type(batch_size) is not int or batch_size < 1:
        raise Phase31Error("screen_batch_size must be positive")
    requested = tuple(families)
    if not requested or any(family not in SEARCH_FAMILIES for family in requested):
        raise Phase31Error("unknown screening family")
    allowed = None if only_digests is None else set(only_digests)
    groups = {}
    for candidate in matrix["candidates"]:
        if candidate["search_family"] not in requested:
            continue
        if allowed is not None and candidate["candidate_sha256"] not in allowed:
            continue
        key = (candidate["search_family"], len(candidate["selected_heads"]))
        groups.setdefault(key, []).append(candidate)
    result = []
    for family in requested:
        for key in sorted((key for key in groups if key[0] == family), key=lambda value: value[1]):
            values = groups[key]
            for offset in range(0, len(values), batch_size):
                result.append(
                    {
                        "family": family,
                        "head_count": key[1],
                        "batch_index": offset // batch_size,
                        "candidates": values[offset : offset + batch_size],
                    }
                )
    return result


def _screening_qualifications(candidates):
    result = {
        name: {
            "valid": True,
            "scope": "short_screening_execution_precondition_only",
            "formal_correctness": False,
        }
        for name in ("serial", "two_stream", "current_tacker")
    }
    for candidate in candidates:
        result[candidate["variant_id"]] = {
            "valid": True,
            "scope": "phase31_disabled_profile_and_live_abi_preflight",
            "formal_correctness": False,
        }
    return result


def _screen_batch(journal, args, runner, matrix, profiles, batch, label):
    candidates = batch["candidates"]
    profile_paths = _profile_paths(matrix, profiles["profiles_dir"])
    inputs = {
        "source_matrix_sha256": matrix["matrix_sha256"],
        "profile_manifest_sha256": profiles["manifest_sha256"],
        "family": batch["family"],
        "head_count": batch["head_count"],
        "candidate_sha256": [item["candidate_sha256"] for item in candidates],
        "protocol": {
            "frames": args.screen_frames,
            "warmup": args.screen_warmup,
            "trials": args.screen_trials,
            "schedule": "round_robin",
            "seed": args.seed,
        },
    }

    def action(directory):
        session = journal.root / "sessions" / "screen" / label
        session.mkdir(parents=True, exist_ok=True)
        correctness = session / "screening-correctness.json"
        report_path = session / "screening-report.json"
        runs_dir = session / "runs"
        atomic_write_json(correctness, _screening_qualifications(candidates))
        checkpoint = runs_dir / label / "benchmark.checkpoint.json"
        command = PHASE3.build_benchmark_command(
            args,
            report_path,
            runs_dir,
            label,
            [(item["variant_id"], profile_paths[item["variant_id"]]) for item in candidates],
            correctness,
            args.screen_frames,
            args.screen_warmup,
            args.screen_trials,
            "round_robin",
            resume=checkpoint.is_file(),
        )
        command_error = None
        try:
            invoke(runner, command, allowed=(0, 1))
        except Exception as error:
            command_error = str(error)
        report = load_json(report_path, "screening report") if report_path.is_file() else None
        summaries = report.get("summaries") if isinstance(report, dict) else None
        report_facts = artifact(report_path) if report_path.is_file() else None
        expected_names = ["serial", "two_stream", "current_tacker"] + [
            item["variant_id"] for item in candidates
        ]
        contract = report.get("contract") if isinstance(report, dict) else None
        schedule = report.get("schedule") if isinstance(report, dict) else None
        observed_names = [
            item.get("name")
            for item in report.get("candidates", [])
            if isinstance(item, dict)
        ] if isinstance(report, dict) else []
        report_trusted = bool(
            isinstance(report, dict)
            and report.get("passed") is True
            and isinstance(contract, dict)
            and contract.get("profile_frames") == args.screen_frames
            and contract.get("warmup_frames") == args.screen_warmup
            and isinstance(schedule, dict)
            and schedule.get("strategy") == "round_robin"
            and schedule.get("trials_per_candidate") == args.screen_trials
            and observed_names == expected_names
        )
        records = []
        bindings = {}
        for candidate in candidates:
            summary = summaries.get(candidate["variant_id"]) if isinstance(summaries, dict) else None
            score = summary.get("median_throughput_fps") if isinstance(summary, dict) else None
            succeeded = bool(
                report_trusted
                and isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
                and float(score) > 0.0
            )
            profile_facts = artifact(profile_paths[candidate["variant_id"]])
            records.append(
                {
                    "candidate_sha256": candidate["candidate_sha256"],
                    "status": "succeeded" if succeeded else "failed",
                    "score": float(score) if succeeded else None,
                    "error": None if succeeded else (command_error or "screening report omitted a positive score"),
                    "artifact_sha256": report_facts["sha256"] if report_facts is not None else profile_facts["sha256"],
                    "attempt": int(directory.name),
                }
            )
            bindings[candidate["candidate_sha256"]] = {
                "source_matrix_sha256": matrix["matrix_sha256"],
                "profile": profile_facts,
                "report": report_facts,
                "screening_batch": label,
            }
        document = {
            "schema_version": 1,
            "kind": SCREENING_KIND,
            "source_matrix_sha256": matrix["matrix_sha256"],
            "family": batch["family"],
            "head_count": batch["head_count"],
            "candidate_count": len(candidates),
            "protocol": inputs["protocol"],
            "records": records,
            "measurement_source_bindings": bindings,
            "benchmark_report": report_facts,
            "benchmark_passed": report_trusted,
        }
        result_path = directory / "screening-batch.json"
        atomic_write_json(result_path, document)
        paths = [correctness, result_path]
        if report_path.is_file():
            paths.append(report_path)
        if checkpoint.is_file():
            paths.extend(item["path"] for item in PHASE3._regular_artifacts_under(runs_dir / label))
        return {
            "path": str(result_path),
            "records": records,
            "measurement_source_bindings": bindings,
        }, paths

    return journal.run("screen-{}".format(label), inputs, action)


def _screen_families(journal, args, runner, matrix, profiles, families, label, only_digests=None):
    batches = candidate_batches(matrix, families, args.screen_batch_size, only_digests=only_digests)
    if not batches:
        raise Phase31Error("screening stage {} has no candidates".format(label))
    records = []
    bindings = {}
    paths = []
    for sequence, batch in enumerate(batches):
        batch_label = "{}-{}-h{}-b{:03d}".format(
            label, batch["family"], batch["head_count"], sequence
        )
        result = _screen_batch(journal, args, runner, matrix, profiles, batch, batch_label)
        paths.append(result["path"])
        failed = {
            item["candidate_sha256"]
            for item in result["records"]
            if item["status"] == "failed"
        }
        if failed and len(batch["candidates"]) > 1:
            # A benchmark stops at its first failed child.  Never label the
            # unexecuted tail as terminal-failed: retry every unresolved
            # candidate in an isolated one-candidate benchmark.
            retry_records = []
            retry_bindings = {}
            for retry_index, candidate in enumerate(batch["candidates"]):
                if candidate["candidate_sha256"] not in failed:
                    continue
                retry_batch = {
                    "family": batch["family"],
                    "head_count": batch["head_count"],
                    "batch_index": retry_index,
                    "candidates": [candidate],
                }
                retry_label = "{}-retry-{:03d}".format(
                    batch_label, retry_index
                )
                retry = _screen_batch(
                    journal,
                    args,
                    runner,
                    matrix,
                    profiles,
                    retry_batch,
                    retry_label,
                )
                retry_records.extend(retry["records"])
                retry_bindings.update(retry["measurement_source_bindings"])
                paths.append(retry["path"])
            retry_by_digest = {
                item["candidate_sha256"]: item for item in retry_records
            }
            final_records = [
                retry_by_digest.get(item["candidate_sha256"], item)
                for item in result["records"]
            ]
            final_bindings = dict(result["measurement_source_bindings"])
            final_bindings.update(retry_bindings)
        else:
            final_records = result["records"]
            final_bindings = result["measurement_source_bindings"]
        records.extend(final_records)
        bindings.update(final_bindings)
    return {"records": records, "bindings": bindings, "batch_artifacts": paths}


def _write_matrix_stage(journal, name, inputs, builder):
    def action(directory):
        matrix = _validate_matrix(builder(), require_c2_450=True)
        output = directory / "matrix.json"
        atomic_write_json(output, matrix)
        return {
            "matrix_path": str(output),
            "matrix": matrix,
            "matrix_sha256": matrix["matrix_sha256"],
            "candidate_count": len(matrix["candidates"]),
        }, [output]

    return journal.run(name, inputs, action)


def _portable_artifact_identity(facts):
    if facts is None:
        return None
    if not isinstance(facts, dict):
        raise Phase31Error("artifact identity must be an object or null")
    return {
        "sha256": facts.get("sha256"),
        "size_bytes": facts.get("size_bytes"),
    }


def _portable_measurement_bindings(bindings):
    result = {}
    for digest, binding in sorted(bindings.items()):
        result[digest] = {
            "source_matrix_sha256": binding["source_matrix_sha256"],
            "profile": _portable_artifact_identity(binding["profile"]),
            "report": _portable_artifact_identity(binding["report"]),
            "screening_batch": binding["screening_batch"],
        }
    return result


def _ranking_stage(journal, matrix, records, bindings, args):
    portable_inputs = {
        "protocol": {
            "frames": args.screen_frames,
            "warmup": args.screen_warmup,
            "trials": args.screen_trials,
            "schedule": "round_robin",
            "seed": args.seed,
        },
        "measurement_source_bindings": _portable_measurement_bindings(bindings),
    }
    screening_input_sha256 = sha256_json(
        portable_inputs, "tacker-phase31-full-screening-input-v1"
    )
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "screening_input_sha256": screening_input_sha256,
        "records": records,
    }

    def action(directory):
        try:
            ranking = AUTOTUNE.build_screening_ranking(
                matrix,
                records,
                screening_input_sha256=screening_input_sha256,
                maximize=True,
                require_terminal=True,
            )
        except Exception as error:
            raise Phase31Error("cannot seal full screening ranking: {}".format(error))
        if ranking["candidate_count"] != ranking["terminal_count"]:
            raise Phase31Error("screening ranking is not terminal-complete")
        source_path = directory / "measurement-sources.json"
        ranking_path = directory / "screening-ranking.json"
        atomic_write_json(
            source_path,
            {
                "portable_hash_input": portable_inputs,
                "screening_input_sha256": screening_input_sha256,
                "actual_measurement_source_bindings": bindings,
            },
        )
        atomic_write_json(ranking_path, ranking)
        return {
            "ranking": ranking,
            "ranking_path": str(ranking_path),
            "ranking_sha256": ranking["ranking_sha256"],
            "screening_input_sha256": screening_input_sha256,
            "measurement_sources": str(source_path),
        }, [source_path, ranking_path]

    return journal.run("screening-ranking", inputs, action)


def _formal_set_stage(journal, matrix, ranking, top_k):
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "ranking_sha256": ranking["ranking_sha256"],
        "global_top_k": top_k,
        "baselines": ["serial", "two_stream", "current_tacker"],
    }

    def action(directory):
        try:
            formal_set = AUTOTUNE.build_formal_candidate_set(matrix, ranking, top_k)
        except Exception as error:
            raise Phase31Error("cannot build formal candidate set: {}".format(error))
        if formal_set.get("families_without_screening_success"):
            raise Phase31Error(
                "formal set lacks a successful generated family: {}".format(
                    formal_set["families_without_screening_success"]
                )
            )
        if formal_set.get("generated_search_families") != list(SEARCH_FAMILIES):
            raise Phase31Error("formal set must cover generated C0--C4")
        payload = {
            "schema_version": 1,
            "kind": "tacker_phase31_formal_candidate_set",
            "matrix_sha256": matrix["matrix_sha256"],
            "screening_ranking_sha256": ranking["ranking_sha256"],
            "baseline_candidates": ["serial", "two_stream", "current_tacker"],
            "generated_candidate_set": formal_set,
        }
        document = dict(payload)
        document["formal_set_sha256"] = sha256_json(
            payload, "tacker-phase31-formal-set-v1"
        )
        path = directory / "formal-candidate-set.json"
        atomic_write_json(path, document)
        return {
            "formal_set": formal_set,
            "path": str(path),
            "candidate_set_sha256": formal_set["candidate_set_sha256"],
            "formal_set_sha256": document["formal_set_sha256"],
        }, [path]

    return journal.run("formal-candidate-set", inputs, action)


def _validate_quality_report(report, args, profile, returncode):
    try:
        valid = PHASE3._validate_quality_report(report, args, profile, returncode)
    except Exception as error:
        raise Phase31Error(str(error))
    workload = report.get("workload")
    modes = report.get("modes")
    tacker = modes.get("tacker") if isinstance(modes, dict) else None
    indices = workload.get("view_indices") if isinstance(workload, dict) else None
    if (
        workload.get("frames") != 50
        or indices != list(range(50))
        or not isinstance(tacker, dict)
        or tacker.get("actual_mode") != "tacker"
        or tacker.get("fallback_reason") is not None
        or tacker.get("qualification_executed") is not True
    ):
        raise Phase31Error(
            "candidate quality run must return one record for each of 50 unique "
            "requested views and execute Tacker without fallback"
        )
    _validate_per_view_quality(tacker.get("per_view"), "candidate tacker")
    return valid


def _resolved_report_path(value, label):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise Phase31Error("{} must be a non-empty path".format(label))
    return str(Path(value).expanduser().resolve())


def _validate_per_view_quality(records, label):
    if not isinstance(records, list) or len(records) != 50:
        raise Phase31Error("{} must contain exactly 50 per-view records".format(label))
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("batch_index") != index:
            raise Phase31Error(
                "{} per-view batch indices must be exactly 0..49".format(label)
            )
        for metric in ("psnr_db", "ssim", "lpips"):
            value = record.get(metric)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise Phase31Error(
                    "{} per-view {} must be finite".format(label, metric)
                )


def _validate_baseline_quality_report(report, args, returncode):
    """Validate a baseline report while preserving per-baseline invalidity."""

    workload = report.get("workload") if isinstance(report, dict) else None
    modes = report.get("modes") if isinstance(report, dict) else None
    gates = report.get("gates") if isinstance(report, dict) else None
    qualification = report.get("qualification") if isinstance(report, dict) else None
    passed = report.get("passed") if isinstance(report, dict) else None
    expected_profile = str(Path(args.current_tacker_profile).expanduser().resolve())
    expected_model = str(Path(args.model_path).expanduser().resolve())
    expected_source = str(Path(args.source_path).expanduser().resolve())
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "4dgaussians_tacker_quality_validation"
        or type(passed) is not bool
        or (returncode == 0) is not passed
        or not isinstance(workload, dict)
        or workload.get("scene") != EXPECTED_WORKLOAD
        or workload.get("iteration") != EXPECTED_ITERATION
        or workload.get("split") != "test"
        or workload.get("frames") != 50
        or workload.get("view_indices") != list(range(50))
        or workload.get("resolution") != list(EXPECTED_RESOLUTION)
        or workload.get("gaussian_count") != EXPECTED_GAUSSIANS
        or _resolved_report_path(workload.get("model_path"), "quality model_path")
        != expected_model
        or _resolved_report_path(workload.get("source_path"), "quality source_path")
        != expected_source
        or report.get("tacker_profile") != expected_profile
        or not isinstance(qualification, dict)
        or qualification.get("enabled") is not False
        or qualification.get("profile_override") is not None
        or qualification.get("admission_claimed") is not passed
        or not isinstance(modes, dict)
        or set(modes) != {"serial", "two_stream", "tacker"}
        or not isinstance(gates, list)
        or not isinstance(report.get("errors"), list)
        or any(not isinstance(error, str) for error in report["errors"])
    ):
        raise Phase31Error("baseline quality report contract changed")

    gate_by_mode = {}
    for gate in gates:
        if (
            not isinstance(gate, dict)
            or gate.get("mode") not in ("two_stream", "tacker")
        ):
            raise Phase31Error("baseline quality gates changed")
        mode = gate["mode"]
        if mode in gate_by_mode:
            raise Phase31Error("baseline quality gates contain duplicates")
        if any(
            type(gate.get(key)) is not bool
            for key in (
                "actual_mode_passed",
                "qualification_passed",
                "quality_passed",
                "passed",
            )
        ):
            raise Phase31Error("baseline quality gate booleans changed")
        gate_by_mode[mode] = gate
    if set(gate_by_mode) != {"two_stream", "tacker"}:
        raise Phase31Error("baseline quality report omitted a mode gate")
    if passed is not all(gate["passed"] for gate in gate_by_mode.values()):
        raise Phase31Error("baseline quality passed flag disagrees with gates")

    status = {}
    for external, mode in (
        ("serial", "serial"),
        ("two_stream", "two_stream"),
        ("current_tacker", "tacker"),
    ):
        details = modes[mode]
        if (
            not isinstance(details, dict)
            or details.get("requested_mode") != mode
            or type(details.get("qualification_requested")) is not bool
            or type(details.get("qualification_executed")) is not bool
            or details.get("qualification_requested") is not False
            or details.get("qualification_executed") is not False
        ):
            raise Phase31Error("baseline {} mode binding changed".format(mode))
        _validate_per_view_quality(
            details.get("per_view"), "baseline {}".format(mode)
        )
        if mode == "serial":
            valid = bool(
                details.get("actual_mode") == "serial"
                and details.get("fallback_reason") is None
            )
        else:
            gate = gate_by_mode[mode]
            if gate.get("actual_mode") != details.get("actual_mode"):
                raise Phase31Error(
                    "baseline {} gate disagrees with actual mode".format(mode)
                )
            valid = bool(
                gate.get("passed") is True
                and details.get("actual_mode") == mode
                and details.get("fallback_reason") is None
            )
        status[external] = {
            "valid": valid,
            "actual_mode": details.get("actual_mode"),
            "fallback_reason": details.get("fallback_reason"),
        }
    return status


def _qualify_candidate(journal, args, runner, matrix_path, matrix, profiles_dir, candidate):
    digest = candidate["candidate_sha256"]
    profile = Path(profiles_dir) / "{}.json".format(candidate["variant_id"])
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "candidate_sha256": digest,
        "profile": artifact(profile),
        "leaf": {
            "views": args.leaf_views,
            "warmup": args.leaf_warmup,
            "repetitions": args.leaf_repetitions,
            "script": artifact(PROJECT_ROOT / "profile_tacker_leaves.py"),
        },
        "quality": {"frames": 50, "script": artifact(_script("validate_tacker_modes.py"))},
    }

    def action(directory):
        errors = []
        leaf_path = directory / "leaf-report.json"
        quality_path = directory / "quality.json"
        leaf_returncode = 1
        quality_returncode = 1
        try:
            leaf_completed = invoke(
                runner,
                PHASE3.build_leaf_command(args, profile, matrix_path, directory),
                allowed=(0, 1),
            )
            leaf_returncode = int(leaf_completed.returncode)
        except Exception as error:
            errors.append("leaf command: {}".format(error))
        try:
            quality_completed = invoke(
                runner,
                PHASE3.build_quality_command(args, profile, quality_path),
                allowed=(0, 1),
            )
            quality_returncode = int(quality_completed.returncode)
        except Exception as error:
            errors.append("quality command: {}".format(error))
        leaf_valid = False
        quality_valid = False
        if leaf_path.is_file():
            try:
                leaf_valid = PHASE3._validate_leaf_report(
                    load_json(leaf_path, "candidate leaf report"),
                    args,
                    candidate,
                    profile,
                    matrix_path,
                    matrix["matrix_sha256"],
                    leaf_returncode,
                )
            except Exception as error:
                errors.append("leaf evidence: {}".format(error))
        else:
            errors.append("leaf report missing")
        if quality_path.is_file():
            try:
                quality_valid = _validate_quality_report(
                    load_json(quality_path, "candidate quality report"),
                    args,
                    profile,
                    quality_returncode,
                )
            except Exception as error:
                errors.append("quality evidence: {}".format(error))
        else:
            errors.append("quality report missing")
        result = {
            "schema_version": 1,
            "kind": QUALIFICATION_KIND,
            "matrix_sha256": matrix["matrix_sha256"],
            "candidate_sha256": digest,
            "variant_id": candidate["variant_id"],
            "search_family": candidate["search_family"],
            "profile": artifact(profile),
            "leaf_report": artifact(leaf_path) if leaf_path.is_file() else None,
            "quality_report": artifact(quality_path) if quality_path.is_file() else None,
            "leaf_passed": bool(leaf_valid),
            "quality_50_view_passed": bool(quality_valid),
            "quality_actual_tacker_without_fallback": bool(quality_valid),
            "quality_unique_requested_views": bool(quality_valid),
            "quality_exact_per_view_records": bool(quality_valid),
            "valid": bool(leaf_valid and quality_valid),
            "errors": errors,
        }
        output = directory / "qualification.json"
        atomic_write_json(output, result)
        paths = [output]
        for path in (leaf_path, quality_path, directory / "device.json", directory / "raster.json", directory / "leaf.json"):
            if path.is_file():
                paths.append(path)
        return {"result": result, "path": str(output)}, paths

    return journal.run("qualify-{}".format(digest[:20]), inputs, action)


def plan_family_backfill(matrix, ranking, formal_set, qualification_by_digest):
    """Plan one same-family replacement chain for every invalid finalist."""

    selected = [item["candidate"] for item in formal_set["candidates"]]
    selected_digests = {item["candidate_sha256"] for item in selected}
    replacements = []
    for candidate in list(selected):
        digest = candidate["candidate_sha256"]
        result = qualification_by_digest.get(digest)
        if isinstance(result, dict) and result.get("valid") is True:
            continue
        family = candidate["search_family"]
        try:
            alternatives = AUTOTUNE.family_local_backfill_candidates(
                matrix,
                ranking,
                family,
                count=1,
                excluded_candidate_sha256s=selected_digests,
            )
        except Exception as error:
            raise Phase31Error("cannot plan {} backfill: {}".format(family, error))
        replacements.append(
            {
                "invalid_candidate_sha256": digest,
                "search_family": family,
                "alternatives": [item["candidate"] for item in alternatives],
            }
        )
        selected_digests.update(
            item["candidate"]["candidate_sha256"] for item in alternatives
        )
    return replacements


def _baseline_quality(journal, args, runner):
    inputs = {
        "current_profile": artifact(args.current_tacker_profile),
        "frames": 50,
        "modes": ["serial", "two_stream", "tacker"],
        "workload": journal.state["identity"]["payload"]["workload"],
        "configuration_chain": journal.state["identity"]["payload"][
            "configuration_chain"
        ],
    }

    def action(directory):
        output = directory / "baseline-quality.json"
        command = [
            str(Path(args.python_executable).expanduser().resolve()),
            _script("validate_tacker_modes.py"),
            "--model_path", str(Path(args.model_path).expanduser().resolve()),
            "--source_path", str(Path(args.source_path).expanduser().resolve()),
            "--configs", str(Path(args.config).expanduser().resolve()),
            "--iteration", str(args.iteration),
            "--scene-name", args.workload_name,
            "--split", args.split,
            "--frames", "50",
            "--modes", "serial", "two_stream", "tacker",
            "--tacker-profile", str(Path(args.current_tacker_profile).expanduser().resolve()),
            "--gpu", str(args.gpu),
            "--output", str(output.resolve()),
            "--quiet",
        ]
        completed = invoke(runner, command, allowed=(0, 1))
        report = load_json(output, "baseline quality")
        status = _validate_baseline_quality_report(
            report, args, int(completed.returncode)
        )
        report_artifact = artifact(output)
        for details in status.values():
            details["quality_report_sha256"] = report_artifact["sha256"]
        correctness = directory / "baseline-correctness.json"
        atomic_write_json(correctness, status)
        return {
            "quality_report": str(output),
            "correctness_path": str(correctness),
            "correctness": status,
            "all_valid": all(item["valid"] for item in status.values()),
        }, [output, correctness]

    return journal.run("baseline-quality", inputs, action)


def _qualification_plan(
    journal,
    args,
    runner,
    matrix_path,
    matrix,
    profiles,
    ranking,
    formal_set_result,
):
    formal_set = formal_set_result["formal_set"]
    initial = [item["candidate"] for item in formal_set["candidates"]]
    results = {}
    evidence = {}
    ordered = []

    def qualify(candidate, reason):
        digest = candidate["candidate_sha256"]
        if digest not in results:
            qualification = _qualify_candidate(
                journal, args, runner, matrix_path, matrix, profiles["profiles_dir"], candidate
            )
            record = qualification["result"]
            results[digest] = record
            evidence[digest] = artifact(qualification["path"])
            ordered.append({"candidate": candidate, "reason": reason, "result": record})
        return results[digest]

    for candidate in initial:
        qualify(candidate, "formal_set")
    replacement_chains = []
    used = {item["candidate_sha256"] for item in initial}
    for candidate in initial:
        invalid = candidate
        if results[candidate["candidate_sha256"]]["valid"]:
            continue
        family = candidate["search_family"]
        chain = []
        alternatives = AUTOTUNE.family_local_backfill_candidates(
            matrix,
            ranking,
            family,
            count=len(matrix["candidates"]),
            excluded_candidate_sha256s=used,
        )
        replacement = None
        for alternative in alternatives:
            candidate_alt = alternative["candidate"]
            used.add(candidate_alt["candidate_sha256"])
            result = qualify(
                candidate_alt,
                "same_family_backfill_for:{}".format(invalid["candidate_sha256"]),
            )
            chain.append(candidate_alt["candidate_sha256"])
            if result["valid"]:
                replacement = candidate_alt["candidate_sha256"]
                break
        replacement_chains.append(
            {
                "invalid_candidate_sha256": invalid["candidate_sha256"],
                "search_family": family,
                "attempted": chain,
                "replacement_candidate_sha256": replacement,
            }
        )
        if replacement is None:
            raise Phase31Error("no correctness-valid same-family backfill remains for {}".format(family))
    valid = []
    seen = set()
    for item in ordered:
        digest = item["candidate"]["candidate_sha256"]
        if item["result"]["valid"] and digest not in seen:
            valid.append(item["candidate"])
            seen.add(digest)
    family_coverage = {
        family: [item["candidate_sha256"] for item in valid if item["search_family"] == family]
        for family in SEARCH_FAMILIES
    }
    if any(not values for values in family_coverage.values()):
        raise Phase31Error("correctness-valid finalists do not cover C0--C4")
    hash_payload = {
        "schema_version": 1,
        "kind": "tacker_phase31_qualification_plan",
        "matrix_sha256": matrix["matrix_sha256"],
        "screening_ranking_sha256": ranking["ranking_sha256"],
        "formal_candidate_set_sha256": formal_set_result["formal_set_sha256"],
        "autotune_candidate_set_sha256": formal_set["candidate_set_sha256"],
        "attempted": [
            {
                "candidate_sha256": item["candidate"]["candidate_sha256"],
                "search_family": item["candidate"]["search_family"],
                "reason": item["reason"],
                "valid": item["result"]["valid"],
                "leaf_passed": item["result"]["leaf_passed"],
                "quality_50_view_passed": item["result"]["quality_50_view_passed"],
                "qualification_evidence": _portable_artifact_identity(
                    evidence[item["candidate"]["candidate_sha256"]]
                ),
            }
            for item in ordered
        ],
        "same_family_backfill": replacement_chains,
        "valid_finalist_sha256": [item["candidate_sha256"] for item in valid],
        "valid_family_coverage": family_coverage,
    }
    qualification_plan_sha256 = sha256_json(
        hash_payload, "tacker-phase31-qualification-plan-v1"
    )
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "ranking_sha256": ranking["ranking_sha256"],
        "formal_set_sha256": formal_set_result["formal_set_sha256"],
        "candidate_set_sha256": formal_set["candidate_set_sha256"],
        "profile_manifest_sha256": profiles["manifest_sha256"],
        "qualification_plan_sha256": qualification_plan_sha256,
        "qualification_evidence": {
            digest: _portable_artifact_identity(facts)
            for digest, facts in sorted(evidence.items())
        },
    }

    def action(directory):
        document = {
            "schema_version": 1,
            "kind": "tacker_phase31_qualification_plan",
            "matrix_sha256": matrix["matrix_sha256"],
            "screening_ranking_sha256": ranking["ranking_sha256"],
            "formal_candidate_set_sha256": formal_set_result["formal_set_sha256"],
            "autotune_candidate_set_sha256": formal_set["candidate_set_sha256"],
            "attempted": ordered,
            "same_family_backfill": replacement_chains,
            "valid_finalists": valid,
            "valid_family_coverage": family_coverage,
            "portable_hash_input": hash_payload,
            "qualification_plan_sha256": qualification_plan_sha256,
        }
        path = directory / "qualification-plan.json"
        atomic_write_json(path, document)
        return {"document": document, "path": str(path)}, [path]

    return journal.run("qualification-plan", inputs, action)


def _validate_formal_execution_counts(report, finalist_names, runs_root):
    """Seal exact per-sequence scheduler counts for every finalist ABBA run."""

    runs = report.get("runs") if isinstance(report, dict) else None
    if not isinstance(runs, list):
        raise Phase31Error("formal report omitted raw run records")
    root = Path(runs_root).expanduser().resolve()
    finalist_set = set(finalist_names)
    by_candidate = {name: {} for name in finalist_names}
    for run in runs:
        if not isinstance(run, dict) or run.get("candidate_name") not in finalist_set:
            continue
        name = run["candidate_name"]
        round_index = run.get("round_index")
        if type(round_index) is not int or round_index not in range(10):
            raise Phase31Error("formal finalist run has an invalid ABBA round")
        if round_index in by_candidate[name]:
            raise Phase31Error(
                "formal finalist {} has a duplicate ABBA round".format(name)
            )
        if run.get("passed") is not True or run.get("returncode") != 0:
            raise Phase31Error(
                "formal finalist {} round {} did not pass".format(
                    name, round_index
                )
            )
        metadata_path = Path(
            _resolved_report_path(run.get("metadata_path"), "formal metadata_path")
        )
        if not _is_within(metadata_path, root):
            raise Phase31Error("formal child metadata escaped its run directory")
        metadata_artifact = artifact(metadata_path)
        if run.get("metadata_sha256") != metadata_artifact["sha256"]:
            raise Phase31Error("formal child metadata hash changed")
        metadata = load_json(metadata_path, "formal child metadata")
        expected_profile = _resolved_report_path(
            run.get("profile_path"), "formal finalist profile_path"
        )
        if (
            metadata.get("schema_version") != 1
            or metadata.get("kind") != "4dgaussians_tacker_render_profile"
            or metadata.get("passed") is not True
            or metadata.get("execution_mode") != "tacker"
            or metadata.get("actual_execution_mode") != "tacker"
            or metadata.get("profile_frames") != 50
            or metadata.get("view_indices") != list(range(50))
            or metadata.get("two_stream_fallback_reason") is not None
            or metadata.get("tacker_fallback_reason") is not None
            or metadata.get("qualification_mode_requested") is not True
            or metadata.get("qualification_mode_executed") is not True
            or metadata.get("tacker_profile") is not None
            or _resolved_report_path(
                metadata.get("qualification_profile"),
                "formal qualification_profile",
            )
            != expected_profile
            or metadata.get("selected_variant_id") != name
            or metadata.get("pipeline_execution_counts")
            != EXPECTED_FORMAL_EXECUTION_COUNTS
        ):
            raise Phase31Error(
                "formal finalist {} round {} execution counts or binding changed"
                .format(name, round_index)
            )
        by_candidate[name][round_index] = {
            "candidate_name": name,
            "round_index": round_index,
            "metadata": metadata_artifact,
            "pipeline_execution_counts": dict(EXPECTED_FORMAL_EXECUTION_COUNTS),
        }
    expected_rounds = set(range(10))
    for name in finalist_names:
        if set(by_candidate[name]) != expected_rounds:
            raise Phase31Error(
                "formal finalist {} does not have ten count-validated runs".format(
                    name
                )
            )
    return [
        by_candidate[name][round_index]
        for name in finalist_names
        for round_index in range(10)
    ]


def _validate_formal_report(report, args, finalist_names):
    contract = report.get("contract")
    schedule = report.get("schedule")
    names = [
        item.get("name") for item in report.get("candidates", []) if isinstance(item, dict)
    ]
    expected = ["serial", "two_stream", "current_tacker"] + list(finalist_names)
    exit_condition = report.get("phase0_exit_condition")
    if (
        report.get("kind") != "4dgaussians_tacker_fps_benchmark"
        or report.get("passed") is not True
        or not isinstance(exit_condition, dict)
        or exit_condition.get("met") is not True
        or not isinstance(contract, dict)
        or contract.get("profile_frames") != 50
        or contract.get("warmup_frames") != 10
        or not isinstance(schedule, dict)
        or schedule.get("strategy") != "abba"
        or schedule.get("trials_per_candidate") != 10
        or names != expected
    ):
        raise Phase31Error("formal 10x50 ABBA report contract changed")
    comparisons = report.get("paired_comparisons")
    by_pair = {
        (item.get("candidate"), item.get("reference")): item
        for item in comparisons
        if isinstance(item, dict)
    } if isinstance(comparisons, list) else {}
    qualifications = report.get("correctness_qualifications")
    excluded = {
        item.get("name"): item.get("reason")
        for item in report.get("excluded_candidates", [])
        if isinstance(item, dict)
    }
    for name in finalist_names:
        for reference in ("two_stream", "current_tacker"):
            reference_valid = True
            if isinstance(qualifications, dict):
                reference_record = qualifications.get(reference)
                reference_valid = bool(
                    isinstance(reference_record, dict)
                    and reference_record.get("valid") is True
                )
            if not reference_valid:
                if excluded.get(reference) != "correctness_invalid":
                    raise Phase31Error(
                        "invalid baseline {} was not explicitly excluded".format(
                            reference
                        )
                    )
                continue
            comparison = by_pair.get((name, reference))
            interval = comparison.get("paired_bootstrap_95_ci") if isinstance(comparison, dict) else None
            if (
                not isinstance(comparison, dict)
                or not isinstance(interval, dict)
                or not all(isinstance(interval.get(key), (int, float)) for key in ("lower", "upper"))
            ):
                raise Phase31Error("formal report omitted paired CI for {} vs {}".format(name, reference))
    promotion = report.get("promotion")
    winner = report.get("deployment_winner")
    if (
        winner not in expected
        or not isinstance(promotion, dict)
        or not promotion.get("reason")
        or not promotion.get("reason_code")
        or not isinstance(promotion.get("criteria"), dict)
    ):
        raise Phase31Error("formal selection omitted deployment reasoning")
    return winner


def _formal_benchmark(journal, args, runner, matrix, profiles, qualification, baseline):
    finalists = qualification["document"]["valid_finalists"]
    names = [item["variant_id"] for item in finalists]
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "qualification_plan_sha256": qualification["document"]["qualification_plan_sha256"],
        "baseline_correctness": artifact(baseline["correctness_path"]),
        "protocol": {"frames": 50, "warmup": 10, "trials": 10, "schedule": "abba", "seed": args.seed},
        "finalists": [item["candidate_sha256"] for item in finalists],
    }

    def action(directory):
        session = journal.root / "sessions" / "formal"
        session.mkdir(parents=True, exist_ok=True)
        correctness_path = session / "formal-correctness.json"
        qualifications = deepcopy(baseline["correctness"])
        for item in finalists:
            qualifications[item["variant_id"]] = {
                "valid": True,
                "candidate_sha256": item["candidate_sha256"],
                "qualification_plan_sha256": qualification["document"]["qualification_plan_sha256"],
            }
        atomic_write_json(correctness_path, qualifications)
        profile_paths = _profile_paths(matrix, profiles["profiles_dir"])
        report_path = session / "formal-fps.json"
        runs_dir = session / "runs"
        run_id = "phase31-formal"
        checkpoint = runs_dir / run_id / "benchmark.checkpoint.json"
        command = PHASE3.build_benchmark_command(
            args,
            report_path,
            runs_dir,
            run_id,
            [(item["variant_id"], profile_paths[item["variant_id"]]) for item in finalists],
            correctness_path,
            50,
            10,
            10,
            "abba",
            resume=checkpoint.is_file(),
        )
        invoke(runner, command)
        report = load_json(report_path, "formal FPS report")
        winner = _validate_formal_report(report, args, names)
        execution_records = _validate_formal_execution_counts(
            report, names, runs_dir / run_id
        )
        c3_names = [item["variant_id"] for item in finalists if item["search_family"] == "c3"]
        c4_names = [item["variant_id"] for item in finalists if item["search_family"] == "c4"]
        if not c3_names or not c4_names:
            raise Phase31Error("formal ABBA run omitted C3 or C4")
        raw = PHASE3._regular_artifacts_under(runs_dir / run_id)
        raw_manifest = directory / "raw-artifacts.json"
        atomic_write_json(raw_manifest, {"artifacts": raw})
        execution_integrity_path = directory / "execution-integrity.json"
        execution_integrity = {
            "schema_version": 1,
            "kind": "tacker_phase31_formal_execution_integrity",
            "protocol": {
                "frames_per_sequence": 50,
                "trials_per_candidate": 10,
                "schedule": "abba",
            },
            "expected_pipeline_execution_counts": dict(
                EXPECTED_FORMAL_EXECUTION_COUNTS
            ),
            "finalist_names": names,
            "validated_runs": execution_records,
            "claim_scope": (
                "Each finalist child sequence executed Tacker without fallback "
                "and reported exactly one prefill, 49 mixed steps, one drain, "
                "50 outputs, and 50 selected-head evaluations per head."
            ),
        }
        atomic_write_json(execution_integrity_path, execution_integrity)
        result = {
            "fps_report": str(report_path),
            "correctness": str(correctness_path),
            "deployment_winner": winner,
            "formal_candidate_names": names,
            "c3_real_50_view_candidates": c3_names,
            "c4_real_50_view_candidates": c4_names,
            "execution_integrity": {
                "no_fallback_and_exact_scheduler_counts_validated": True,
                "artifact": artifact(execution_integrity_path),
                "scope": "finalist_tacker_children_10x50",
            },
        }
        return result, [
            report_path,
            correctness_path,
            raw_manifest,
            execution_integrity_path,
        ] + [item["path"] for item in raw]

    return journal.run("formal-benchmark", inputs, action)


def _winner_selection(journal, args, matrix, profiles, formal):
    winner = formal["deployment_winner"]
    candidate_by_name = {item["variant_id"]: item for item in matrix["candidates"]}
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "formal_report": artifact(formal["fps_report"]),
        "deployment_winner": winner,
    }

    def action(directory):
        winner_profile = None
        reused_profile = None
        challenger = candidate_by_name.get(winner)
        artifacts = []
        if challenger is not None:
            source = Path(profiles["profiles_dir"]) / "{}.json".format(winner)
            snapshot = load_json(source, "winning qualification profile")
            if (
                snapshot.get("deployment") != {"enabled": False, "valid": False}
                or snapshot.get("provenance", {}).get("matrix_sha256") != matrix["matrix_sha256"]
                or snapshot.get("provenance", {}).get("candidate_sha256") != challenger["candidate_sha256"]
            ):
                raise Phase31Error("winning challenger profile binding changed")
            target = directory / "winner-qualification-profile.json"
            atomic_write_json(target, snapshot)
            winner_profile = artifact(target)
            artifacts.append(target)
        elif winner == "current_tacker":
            reused_profile = artifact(args.current_tacker_profile)
        elif winner not in ("serial", "two_stream"):
            raise Phase31Error("deployment winner is not a known baseline or challenger")
        payload = {
            "schema_version": 1,
            "kind": SELECTION_KIND,
            "matrix_sha256": matrix["matrix_sha256"],
            "formal_report_sha256": sha256_file(formal["fps_report"]),
            "deployment_winner": winner,
            "winner_is_new_challenger": challenger is not None,
            "disabled_winner_qualification_profile": winner_profile,
            "reused_incumbent_profile": reused_profile,
            "baseline_winner_has_no_synthetic_profile": challenger is None,
        }
        document = dict(payload)
        portable_payload = dict(payload)
        portable_payload["disabled_winner_qualification_profile"] = (
            _portable_artifact_identity(winner_profile)
        )
        portable_payload["reused_incumbent_profile"] = (
            _portable_artifact_identity(reused_profile)
        )
        document["portable_hash_input"] = portable_payload
        document["selection_sha256"] = sha256_json(
            portable_payload, "tacker-phase31-selection-v1"
        )
        path = directory / "selection.json"
        atomic_write_json(path, document)
        return {"document": document, "path": str(path)}, [path] + artifacts

    return journal.run("selection", inputs, action)


def _nsight(journal, args, runner, formal):
    physical_gpu = physical_gpu_from_environment()
    inputs = {
        "formal_report": artifact(formal["fps_report"]),
        "frames": 50,
        "limit": 3,
        "physical_gpu": physical_gpu,
    }

    def action(directory):
        session = journal.root / "sessions" / "nsight"
        output_dir = session / "profiles"
        report_path = session / "top3-nsight.json"
        command = [
            str(Path(args.python_executable).expanduser().resolve()),
            _script("profile_tacker_top3.py"),
            "--fps-report", formal["fps_report"],
            "--output-dir", str(output_dir.resolve()),
            "--report", str(report_path.resolve()),
            "--profile-script", str(Path(args.profile_nsight_script).expanduser().resolve()),
            "--model-path", str(Path(args.model_path).expanduser().resolve()),
            "--config", str(Path(args.config).expanduser().resolve()),
            "--source-path", str(Path(args.source_path).expanduser().resolve()),
            "--gpu", str(physical_gpu),
            "--frames", "50",
            "--iteration", str(args.iteration),
            "--workload-name", args.workload_name,
            "--limit", "3",
        ]
        checkpoint = output_dir / "top3.checkpoint.json"
        existing_pass = bool(
            report_path.is_file()
            and load_json(report_path, "existing top-3 Nsight report").get(
                "passed"
            )
            is True
        )
        if not existing_pass:
            if checkpoint.is_file():
                command.append("--resume")
            child_env = os.environ.copy()
            child_env["PYTHON_BIN"] = str(
                Path(args.python_executable).expanduser().resolve()
            )
            invoke(runner, command, env=child_env)
        report = load_json(report_path, "top-3 Nsight report")
        if (
            report.get("kind") != "4dgaussians_tacker_top3_nsight"
            or report.get("passed") is not True
            or len(report.get("profiles", [])) != 3
        ):
            raise Phase31Error("top-3 Nsight report did not pass")
        raw = PHASE3._regular_artifacts_under(output_dir)
        raw_manifest = directory / "raw-artifacts.json"
        atomic_write_json(raw_manifest, {"artifacts": raw})
        return {"report": str(report_path)}, [report_path, raw_manifest] + [item["path"] for item in raw]

    return journal.run("top3-nsight", inputs, action)


def _stopped(journal, point, result):
    journal.state["status"] = "stopped_after_{}".format(point)
    journal.state["stop_result"] = result
    journal.save()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "passed": None,
        "stopped_after": point,
        "checkpoint": str(journal.path),
        "result": result,
    }


def dry_run_plan(args, identity):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": DRY_RUN_KIND,
        "identity": identity,
        "executes_commands": False,
        "isolated_output": True,
        "expected_exhaustive_c2_candidates": 450,
        "formal_protocol": {"frames": 50, "warmup": 10, "trials": 10, "schedule": "abba"},
        "stages": [
            "family-aware A6000 ABI1--ABI4 symbol/resource/manifest preflight",
            "exhaustive schema-v2 C0--C2 matrix and matrix-bound disabled profiles",
            "independent C0/C1/C2 family/head-count screening batches",
            "C3 packed generation from screened C2 top-K head sets",
            "independent C3 screening with regenerated profiles",
            "C4 whole-head generation from C1/C2/C3 top-K-per-family head sets",
            "independent C4 screening with regenerated profiles",
            "terminal full-matrix ranking and reproducible formal candidate set",
            "leaf plus unique 50-view quality qualification with same-family backfill",
            "serial/current/two_stream plus valid finalists 10x50 ABBA benchmark",
            "top-3 Nsight and conditional disabled challenger-winner profile",
        ],
    }


def run_phase31(args, runner=subprocess.run):
    identity = _identity(args)
    if args.dry_run_plan:
        _assert_independent_output(args.output_dir, False)
        return dry_run_plan(args, identity)
    journal = Journal(args.output_dir, identity, args.resume)
    try:
        if journal.state.get("status") == "succeeded":
            saved = journal.state.get("report")
            if not isinstance(saved, dict) or artifact(saved.get("path", "")) != saved:
                raise Phase31Error("completed Phase-3.1 report changed")
            report = load_json(saved["path"], "completed Phase-3.1 report")
            if report.get("kind") != REPORT_KIND or report.get("identity") != identity:
                raise Phase31Error("completed Phase-3.1 report identity changed")
            return report

        physical_gpu = identity["payload"]["workload"]["physical_gpu"]
        preflight_inputs = {
            "logical_gpu": args.gpu,
            "physical_gpu": physical_gpu,
            "runner_sha256": sha256_file(SCRIPT_PATH),
            "cuda_suite": artifact(args.cuda_suite_report),
            "nvidia_smi_executable": identity["payload"]["files"][
                "nvidia_smi_executable"
            ],
        }

        def preflight_action(directory):
            smi = invoke(
                runner,
                [
                    identity["payload"]["paths"]["nvidia_smi"],
                    "--id={}".format(physical_gpu),
                    "--query-gpu=index,name,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
            )
            stdout = PHASE3._completed_text(smi, "stdout")
            if "NVIDIA RTX A6000" not in stdout:
                raise Phase31Error("nvidia-smi did not report an RTX A6000")
            smi_path = directory / "nvidia-smi.json"
            atomic_write_json(smi_path, {"physical_gpu": physical_gpu, "stdout": stdout})
            resources = directory / "resources.json"
            invoke(
                runner,
                [str(Path(args.python_executable).expanduser().resolve()), str(SCRIPT_PATH), "_resource-query", "--gpu", str(args.gpu), "--output", str(resources.resolve())],
            )
            document = load_json(resources, "A6000 Phase-3.1 resources")
            device = _validate_resource_query(document, args.gpu)
            return {"resources": str(resources), "device": device, "nvidia_smi": str(smi_path)}, [resources, smi_path]

        preflight = journal.run("preflight", preflight_inputs, preflight_action)
        if args.stop_after == "preflight":
            return _stopped(journal, "preflight", preflight)

        raster_tiles = ((args.image_width + 15) // 16) * ((args.image_height + 15) // 16)
        backend_blocks = ((args.head_rows + 15) // 16) * 2
        whole_head_blocks = args.head_rows
        base_inputs = {
            "sm_count": preflight["device"]["sm_count"],
            "raster_tile_count": raster_tiles,
            "backend_logical_blocks": backend_blocks,
            "whole_head_logical_blocks": whole_head_blocks,
            "current_persistent_blocks": identity["payload"]["search"]["current_persistent_blocks"],
            "persistent_blocks": identity["payload"]["search"]["extra_persistent_blocks"],
            "packed_persistent_blocks": identity["payload"]["search"]["packed_persistent_blocks"],
            "whole_head_persistent_blocks": identity["payload"]["search"]["whole_head_persistent_blocks"],
        }

        def build_base():
            blocks = AUTOTUNE.derive_persistent_blocks(
                base_inputs["sm_count"],
                raster_tiles,
                backend_blocks,
                current_persistent_blocks=base_inputs["current_persistent_blocks"],
                extra_values=base_inputs["persistent_blocks"],
            )
            return AUTOTUNE.build_phase31_base_matrix(
                blocks,
                sm_count=base_inputs["sm_count"],
                raster_tile_count=raster_tiles,
                backend_logical_blocks=backend_blocks,
                whole_head_logical_blocks=whole_head_blocks,
                packed_persistent_blocks=list(blocks) + base_inputs["packed_persistent_blocks"],
                whole_head_persistent_blocks=list(blocks) + base_inputs["whole_head_persistent_blocks"],
            )

        base = _write_matrix_stage(journal, "base-matrix", base_inputs, build_base)
        base_profiles = _make_profiles(journal, args, runner, base["matrix_path"], base["matrix"], "base", preflight["resources"])
        if args.stop_after == "base":
            return _stopped(journal, "base", {"matrix": base, "profiles": base_profiles})
        base_screen = _screen_families(journal, args, runner, base["matrix"], base_profiles, ("c0", "c1", "c2"), "base")

        c3_inputs = {
            "source_matrix_sha256": base["matrix_sha256"],
            "screening_records_sha256": sha256_json(base_screen["records"], "tacker-phase31-c3-parent-screen-v1"),
            "top_k": args.c3_top_k,
        }
        c3 = _write_matrix_stage(
            journal,
            "c3-matrix",
            c3_inputs,
            lambda: AUTOTUNE.extend_phase31_with_c3(base["matrix"], base_screen["records"], args.c3_top_k),
        )
        base_digests = {item["candidate_sha256"] for item in base["matrix"]["candidates"]}
        c3_digests = {item["candidate_sha256"] for item in c3["matrix"]["candidates"]} - base_digests
        if not c3_digests:
            raise Phase31Error("C3 extension generated no packed candidates")
        c3_profiles = _make_profiles(journal, args, runner, c3["matrix_path"], c3["matrix"], "c3", preflight["resources"])
        if args.stop_after == "c3":
            return _stopped(journal, "c3", {"matrix": c3, "profiles": c3_profiles})
        c3_screen = _screen_families(journal, args, runner, c3["matrix"], c3_profiles, ("c3",), "c3", only_digests=c3_digests)

        c4_parent_records = base_screen["records"] + c3_screen["records"]
        c4_inputs = {
            "source_matrix_sha256": c3["matrix_sha256"],
            "screening_records_sha256": sha256_json(c4_parent_records, "tacker-phase31-c4-parent-screen-v1"),
            "top_k_per_family": args.c4_top_k_per_family,
        }
        c4 = _write_matrix_stage(
            journal,
            "c4-matrix",
            c4_inputs,
            lambda: AUTOTUNE.extend_phase31_with_c4(c3["matrix"], c4_parent_records, args.c4_top_k_per_family),
        )
        c3_all_digests = {item["candidate_sha256"] for item in c3["matrix"]["candidates"]}
        c4_digests = {item["candidate_sha256"] for item in c4["matrix"]["candidates"]} - c3_all_digests
        if not c4_digests:
            raise Phase31Error("C4 extension generated no whole-head candidates")
        c4_profiles = _make_profiles(journal, args, runner, c4["matrix_path"], c4["matrix"], "c4", preflight["resources"])
        if args.stop_after == "c4":
            return _stopped(journal, "c4", {"matrix": c4, "profiles": c4_profiles})
        c4_screen = _screen_families(journal, args, runner, c4["matrix"], c4_profiles, ("c4",), "c4", only_digests=c4_digests)

        all_records = base_screen["records"] + c3_screen["records"] + c4_screen["records"]
        all_bindings = {}
        for result in (base_screen, c3_screen, c4_screen):
            for digest, binding in result["bindings"].items():
                if digest in all_bindings:
                    raise Phase31Error("candidate was short-screened twice")
                all_bindings[digest] = binding
        ranking = _ranking_stage(journal, c4["matrix"], all_records, all_bindings, args)
        final_profiles = _make_profiles(journal, args, runner, c4["matrix_path"], c4["matrix"], "final", preflight["resources"])
        formal_set = _formal_set_stage(journal, c4["matrix"], ranking["ranking"], args.top_k)
        _assert_identity_unchanged(args, identity)
        if args.stop_after == "screen":
            return _stopped(journal, "screen", {"ranking": ranking, "formal_set": formal_set})

        baseline = _baseline_quality(journal, args, runner)
        qualification = _qualification_plan(
            journal,
            args,
            runner,
            c4["matrix_path"],
            c4["matrix"],
            final_profiles,
            ranking["ranking"],
            formal_set,
        )
        _assert_identity_unchanged(args, identity)
        if args.stop_after == "quality":
            return _stopped(journal, "quality", {"baseline": baseline, "qualification": qualification})

        formal = _formal_benchmark(
            journal, args, runner, c4["matrix"], final_profiles, qualification, baseline
        )
        selection = _winner_selection(journal, args, c4["matrix"], final_profiles, formal)
        _assert_identity_unchanged(args, identity)
        if args.stop_after == "formal":
            return _stopped(journal, "formal", {"benchmark": formal, "selection": selection})
        nsight = _nsight(journal, args, runner, formal)
        _assert_identity_unchanged(args, identity)
        if args.stop_after == "nsight":
            return _stopped(journal, "nsight", nsight)

        report_path = journal.root / "phase31-report.json"
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "generated_at_utc": utc_now(),
            "passed": True,
            "identity": identity,
            "isolation": {
                "new_evidence_chain": True,
                "sealed_phase3_root_untouched": str(SEALED_PHASE3_ROOT),
            },
            "matrix": {
                "path": c4["matrix_path"],
                "matrix_sha256": c4["matrix_sha256"],
                "candidate_count": c4["candidate_count"],
                "exhaustive_c2_count": 450,
            },
            "screening": {
                "ranking": ranking["ranking_path"],
                "ranking_sha256": ranking["ranking_sha256"],
                "measurement_sources": ranking["measurement_sources"],
            },
            "formal_candidate_set": {
                "path": formal_set["path"],
                "candidate_set_sha256": formal_set["candidate_set_sha256"],
                "formal_set_sha256": formal_set["formal_set_sha256"],
            },
            "qualification": qualification,
            "baseline": baseline,
            "formal": formal,
            "selection": selection,
            "nsight": nsight,
            "checkpoint": str(journal.path),
        }
        atomic_write_json(report_path, report)
        journal.state["status"] = "succeeded"
        journal.state["report"] = artifact(report_path)
        journal.save()
        return report
    finally:
        journal.close()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--current-tacker-profile", required=True)
    parser.add_argument(
        "--template-profile",
        default=str((PROJECT_ROOT / "tacker_profiles" / "raster_head_sm86.json").resolve()),
    )
    parser.add_argument("--cuda-suite-report", required=True)
    parser.add_argument("--output", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--profile-nsight-script", default=_script("profile_nsight.sh"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workload-name", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--gaussian-count", type=int, required=True)
    parser.add_argument("--head-rows", type=int, required=True)
    parser.add_argument("--persistent-block", type=int, action="append", default=[])
    parser.add_argument("--packed-persistent-block", type=int, action="append", default=[])
    parser.add_argument("--whole-head-persistent-block", type=int, action="append", default=[])
    parser.add_argument("--c3-top-k", type=int, default=4)
    parser.add_argument("--c4-top-k-per-family", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--screen-batch-size", type=int, default=30)
    parser.add_argument("--screen-frames", type=int, default=10)
    parser.add_argument("--screen-warmup", type=int, default=2)
    parser.add_argument("--screen-trials", type=int, default=2)
    parser.add_argument("--leaf-views", type=int, default=2)
    parser.add_argument("--leaf-warmup", type=int, default=5)
    parser.add_argument("--leaf-repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=STOP_POINTS)
    parser.add_argument("--dry-run-plan", action="store_true")
    return parser


def _validate_args(parser, args):
    for name in (
        "image_width",
        "image_height",
        "gaussian_count",
        "head_rows",
        "c3_top_k",
        "c4_top_k_per_family",
        "top_k",
        "screen_batch_size",
        "screen_frames",
        "screen_trials",
        "leaf_views",
        "leaf_repetitions",
    ):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.screen_trials < 2:
        parser.error("--screen-trials must be at least 2")
    if args.timeout_seconds is not None and (
        not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0.0
    ):
        parser.error("--timeout-seconds must be finite and positive")
    if args.screen_warmup < 0 or args.leaf_warmup < 0 or args.gpu != 0:
        parser.error("warmups must be non-negative and --gpu must be logical device 0")
    for values in (
        args.persistent_block,
        args.packed_persistent_block,
        args.whole_head_persistent_block,
    ):
        if any(value < 0 for value in values):
            parser.error("persistent block values must be non-negative")
    if args.resume and args.dry_run_plan:
        parser.error("--resume and --dry-run-plan are mutually exclusive")
    if (
        args.workload_name.replace("-", "_").lower() != EXPECTED_WORKLOAD
        or args.iteration != EXPECTED_ITERATION
        or (args.image_width, args.image_height) != EXPECTED_RESOLUTION
        or args.gaussian_count != EXPECTED_GAUSSIANS
        or args.head_rows != EXPECTED_GAUSSIANS
        or args.split != "test"
    ):
        parser.error(
            "Phase 3.1 is fixed to flame_steak/test iteration 14000, 1352x1014, and 111525 rows"
        )


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "_resource-query":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--gpu", type=int, required=True)
        parser.add_argument("--output", required=True)
        helper = parser.parse_args(raw[1:])
        return _resource_query(helper.output, helper.gpu)
    parser = _parser()
    args = parser.parse_args(raw)
    _validate_args(parser, args)
    try:
        report = run_phase31(args)
    except Exception as error:
        print("Phase-3.1 orchestration failed closed: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
