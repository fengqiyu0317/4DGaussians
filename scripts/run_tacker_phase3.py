#!/usr/bin/env python3
"""Run the offline, resumable Phase-3 Tacker search on one RTX A6000.

The coordinator deliberately keeps measurement and selection policy in the
existing tools.  It invokes ``tacker_autotune.py`` for matrix/beam/DB work,
``benchmark_tacker_fps.py`` for every whole-run measurement,
``profile_tacker_leaves.py`` plus ``validate_tacker_modes.py`` for the final
correctness gate, and ``profile_tacker_top3.py`` for diagnostics.

Every child is launched as an argv vector with ``shell=False``.  The local
checkpoint is an orchestration journal; exact per-candidate screening and
correctness state is additionally sealed by the autotune SQLite database.
"""

from __future__ import print_function

import argparse
import ast
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
import subprocess
import sys
import tempfile


SCHEMA_VERSION = 1
REPORT_KIND = "4dgaussians_tacker_phase3_run"
STATE_KIND = "4dgaussians_tacker_phase3_checkpoint"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEAD_ORDER = ("pos", "scales", "rotations", "opacity", "shs")
STOP_POINTS = ("preflight", "prepare", "screen", "quality", "formal", "nsight")
EXPECTED_WORKLOAD = "flame_steak"
EXPECTED_ITERATION = 14000
EXPECTED_RESOLUTION = (1352, 1014)
EXPECTED_GAUSSIANS = 111525
HEAD_MULTI_SOLO_SYMBOL = "tacker_head_linear_multi_solo_v2"
HEAD_MULTI_GPTB_SYMBOL = "tacker_head_linear_multi_gptb_v2"
HEAD_RESOURCE_SYMBOLS = (HEAD_MULTI_SOLO_SYMBOL, HEAD_MULTI_GPTB_SYMBOL)


class Phase3Error(RuntimeError):
    """A command, artifact, or resume contract failed closed."""


def physical_gpu_from_environment():
    """Resolve the one physical ordinal hidden behind logical CUDA device 0."""

    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(value, str) or re.match(r"^[0-9]+$", value) is None:
        raise Phase3Error(
            "CUDA_VISIBLE_DEVICES must contain exactly one numeric physical GPU ordinal"
        )
    return int(value)


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
        raise Phase3Error("value is not finite canonical JSON: {}".format(error))


def sha256_json(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def input_sha256(value):
    return sha256_json(
        {"domain": "tacker-phase3-orchestration-input-v1", "payload": value}
    )


def sha256_file(path):
    target = Path(path)
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise Phase3Error("cannot hash {}: {}".format(target, error))
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime,
    ):
        raise Phase3Error("file changed while hashing: {}".format(target))
    return digest.hexdigest()


def load_json(path, label="JSON"):
    target = Path(path).expanduser().resolve()
    try:
        raw = target.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        canonical_json_bytes(value)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise Phase3Error("cannot load {} {}: {}".format(label, target, error))
    if not isinstance(value, dict):
        raise Phase3Error("{} must be a JSON object".format(label))
    return value


def atomic_write_json(path, value):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
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
        os.replace(temporary, str(target))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def artifact(path):
    target = Path(path).expanduser().resolve()
    if not target.is_file() or target.is_symlink():
        raise Phase3Error("expected a regular artifact: {}".format(target))
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def _completed_text(completed, name):
    value = getattr(completed, name, "")
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def invoke(runner, argv, cwd=PROJECT_ROOT, allowed=(0,), env=None):
    if not isinstance(argv, list) or not argv or not all(
        isinstance(item, str) and "\x00" not in item for item in argv
    ):
        raise Phase3Error("child command must be a non-empty safe argv list")
    completed = runner(
        argv,
        cwd=str(Path(cwd).resolve()),
        capture_output=True,
        text=True,
        check=False,
        shell=False,
        env=env,
    )
    returncode = int(getattr(completed, "returncode", -1))
    if returncode not in allowed:
        stderr = _completed_text(completed, "stderr").strip()
        raise Phase3Error(
            "command exited with status {}: {}{}".format(
                returncode,
                argv[0],
                " ({})".format(stderr[-1000:]) if stderr else "",
            )
        )
    return completed


class Journal(object):
    def __init__(self, output_dir, identity, resume):
        self.root = Path(output_dir).expanduser().resolve()
        self.path = self.root / "phase3-state.json"
        self.lock_path = self.root / ".phase3.lock"
        if resume:
            if not self.path.is_file():
                raise Phase3Error("--resume requires {}".format(self.path))
            self.state = load_json(self.path, "Phase-3 checkpoint")
            if (
                self.state.get("schema_version") != SCHEMA_VERSION
                or self.state.get("kind") != STATE_KIND
                or self.state.get("identity") != identity
            ):
                raise Phase3Error("Phase-3 resume identity changed")
        else:
            if self.root.exists():
                raise Phase3Error(
                    "refusing to reuse output directory without --resume: {}".format(
                        self.root
                    )
                )
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
        fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

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
            raise Phase3Error("checkpoint has duplicate stage {}".format(name))
        return matches[0] if matches else None

    def run(self, name, inputs, action):
        digest = input_sha256(inputs)
        record = self._record(name)
        if record is not None and record.get("input_sha256") != digest:
            raise Phase3Error("stage {} inputs changed during resume".format(name))
        if record is not None and record.get("status") == "succeeded":
            for saved in record.get("artifacts", []):
                facts = artifact(saved.get("path", ""))
                if facts != saved:
                    raise Phase3Error(
                        "stage {} artifact changed during resume".format(name)
                    )
            return deepcopy(record.get("result"))
        attempt = 1 if record is None else int(record.get("attempt", 0)) + 1
        attempt_dir = self.root / "attempts" / name / "{:04d}".format(attempt)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        if record is None:
            record = {"name": name, "input_sha256": digest}
            self.state["stages"].append(record)
        self.state["status"] = "running"
        record.update(
            {
                "status": "running",
                "attempt": attempt,
                "attempt_dir": str(attempt_dir),
                "started_at_utc": utc_now(),
                "error": None,
                "artifacts": [],
                "result": None,
            }
        )
        self.save()
        try:
            result, artifact_paths = action(attempt_dir)
            saved_artifacts = [artifact(path) for path in artifact_paths]
        except BaseException as error:
            record["status"] = "failed"
            record["error"] = "{}: {}".format(type(error).__name__, error)
            record["finished_at_utc"] = utc_now()
            self.state["status"] = "failed"
            self.save()
            raise
        record["status"] = "succeeded"
        record["result"] = result
        record["artifacts"] = saved_artifacts
        record["finished_at_utc"] = utc_now()
        self.state["status"] = "running"
        self.save()
        return deepcopy(result)


def _script(name):
    return str((PROJECT_ROOT / "scripts" / name).resolve())


def _cuda_suite_passed(document):
    kind = document.get("kind")
    if kind not in (
        "4dgaussians_tacker_phase2_targeted_validation",
        "4dgaussians_tacker_phase3_cuda_validation",
    ):
        return False
    abi = document.get("abi")
    resources = document.get("runtime_resources")
    if kind == "4dgaussians_tacker_phase2_targeted_validation":
        status = document.get("status")
        status_ok = bool(
            document.get("schema_version") == 1
            and isinstance(status, dict)
            and status.get("phase2_exit_condition_complete") is True
        )
        full_suite_key = "remote_full_test_suites"
        exact_counts = True
    else:
        status_ok = bool(
            document.get("schema_version") == 1
            and document.get("passed") is True
        )
        full_suite_key = "full_test_suites"
        exact_counts = False
    if (
        not status_ok
        or not isinstance(abi, dict)
        or abi.get("loaded_raster_binary_capabilities_verified") is not True
        or not isinstance(resources, dict)
    ):
        return False
    for key in (
        "head_v1_manifest_sha256",
        "head_v2_manifest_sha256",
        "raster_v1_manifest_sha256",
        "raster_v2_manifest_sha256",
    ):
        if SHA256_RE.match(abi.get(key, "")) is None:
            return False
    required_suite_counts = {
        "cuda_tests": {"head": 16, "raster_v2_and_legacy": 12},
        full_suite_key: {"head": 60, "raster_v2_and_legacy": 34},
    }
    for suite_group, required in required_suite_counts.items():
        suites = document.get(suite_group)
        if not isinstance(suites, dict):
            return False
        for suite_name, required_count in required.items():
            suite = suites.get(suite_name)
            if not isinstance(suite, dict):
                return False
            passed = suite.get("passed")
            total = suite.get("total")
            if (
                type(passed) is not int
                or type(total) is not int
                or passed != total
                or (
                    passed != required_count
                    if exact_counts
                    else passed < required_count
                )
            ):
                return False
    queries = resources.get("queries")
    if (
        not isinstance(queries, list)
        or [item.get("worker_groups") for item in queries if isinstance(item, dict)]
        != [1, 2, 3, 4, 5]
        or any(item.get("launch_supported") is not True for item in queries)
    ):
        return False
    return True


def _configuration_chain(path):
    """Hash a literal Python ``_base_`` config graph without executing it."""

    chain = []
    active = []

    def visit(raw_path):
        unresolved = Path(raw_path).expanduser()
        if unresolved.is_symlink():
            raise Phase3Error("configuration files must not be symlinks: {}".format(unresolved))
        resolved = unresolved.resolve()
        if not resolved.is_file():
            raise Phase3Error("configuration file is missing: {}".format(resolved))
        key = str(resolved)
        if key in active:
            raise Phase3Error("configuration _base_ cycle includes {}".format(resolved))
        try:
            source = resolved.read_text(encoding="utf-8")
            tree = ast.parse(source, str(resolved))
        except (OSError, UnicodeError, SyntaxError) as error:
            raise Phase3Error("cannot parse configuration {}: {}".format(resolved, error))
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
            raise Phase3Error("configuration assigns _base_ more than once: {}".format(resolved))
        bases = []
        if assignments:
            try:
                value = ast.literal_eval(assignments[0])
            except (TypeError, ValueError) as error:
                raise Phase3Error("configuration has non-literal _base_: {}".format(error))
            values = [value] if isinstance(value, str) else list(value) if isinstance(value, (list, tuple)) else []
            if not values or any(not isinstance(item, str) or not item for item in values):
                raise Phase3Error("configuration _base_ must be a string or string list")
            bases = [(resolved.parent / item) for item in values]
        active.append(key)
        chain.append(
            {
                "path": key,
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
        for base in bases:
            visit(base)
        active.pop()

    visit(path)
    return chain


def _identity(args):
    physical_gpu = physical_gpu_from_environment()
    model_root = Path(args.model_path).expanduser().resolve()
    source_root = Path(args.source_path).expanduser().resolve()
    if not model_root.is_dir() or not source_root.is_dir():
        raise Phase3Error("model and source paths must be existing directories")
    required_files = {
        "config": Path(args.config).expanduser().resolve(),
        "current_tacker_profile": Path(args.current_tacker_profile).expanduser().resolve(),
        "template_profile": Path(args.template_profile).expanduser().resolve(),
        "cuda_suite_report": Path(args.cuda_suite_report).expanduser().resolve(),
        "python_executable": Path(args.python_executable).expanduser().resolve(),
    }
    scripts = {
        name: Path(path)
        for name, path in {
            "runner": SCRIPT_PATH,
            "autotune": _script("tacker_autotune.py"),
            "benchmark": _script("benchmark_tacker_fps.py"),
            "quality": _script("validate_tacker_modes.py"),
            "leaf": str((PROJECT_ROOT / "profile_tacker_leaves.py").resolve()),
            "top3": _script("profile_tacker_top3.py"),
            "nsight": str(Path(args.profile_nsight_script).expanduser().resolve()),
        }.items()
    }
    for label, path in list(required_files.items()) + list(scripts.items()):
        if not path.is_file():
            raise Phase3Error("{} file is missing: {}".format(label, path))
    suite = load_json(required_files["cuda_suite_report"], "CUDA suite report")
    if not _cuda_suite_passed(suite):
        raise Phase3Error("the shared CUDA suite report does not prove a strict pass")
    current = load_json(required_files["current_tacker_profile"], "current profile")
    template = load_json(required_files["template_profile"], "qualification template")
    if required_files["current_tacker_profile"] == required_files["template_profile"]:
        raise Phase3Error("current and qualification template profiles must be distinct")
    if current.get("schema_version") == 1:
        current_gate = current.get("admission")
    elif current.get("schema_version") == 2:
        current_gate = current.get("deployment")
    else:
        current_gate = None
    if current_gate != {"enabled": True, "valid": True}:
        raise Phase3Error(
            "current Tacker profile must be admitted schema-v1 or deployed schema-v2"
        )
    if (
        template.get("schema_version") != 2
        or template.get("deployment") != {"enabled": False, "valid": False}
    ):
        raise Phase3Error("qualification template must be disabled schema-v2")
    manifest = current.get("manifest")
    current_pb = manifest.get("persistent_blocks") if isinstance(manifest, dict) else None
    if type(current_pb) is not int or current_pb <= 0:
        raise Phase3Error("current profile manifest has no positive persistent_blocks")
    iteration_root = model_root / "point_cloud" / "iteration_{}".format(
        args.iteration
    )
    workload_paths = {
        "cfg_args": model_root / "cfg_args",
        "point_cloud.ply": iteration_root / "point_cloud.ply",
        "deformation.pth": iteration_root / "deformation.pth",
        "deformation_table.pth": iteration_root / "deformation_table.pth",
        "poses_bounds.npy": source_root / "poses_bounds.npy",
    }
    workload_files = {}
    for name, path in sorted(workload_paths.items()):
        if path.is_symlink() or not path.is_file():
            raise Phase3Error(
                "required workload file is missing or not a regular file: {}".format(path)
            )
        workload_files[name] = {"path": str(path), "sha256": sha256_file(path)}
    configuration_chain = _configuration_chain(args.config)
    payload = {
        "paths": {
            "model": str(Path(args.model_path).expanduser().resolve()),
            "source": str(Path(args.source_path).expanduser().resolve()),
            "python": str(Path(args.python_executable).expanduser().resolve()),
        },
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(required_files.items())
        },
        "scripts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(scripts.items())
        },
        "workload_files": workload_files,
        "configuration_chain": configuration_chain,
        "execution_environment": {
            name: os.environ.get(name)
            for name in (
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
            "beam_width": args.beam_width,
            "top_k": args.top_k,
            "screen_frames": args.screen_frames,
            "screen_warmup": args.screen_warmup,
            "screen_trials": args.screen_trials,
            "screen_seed": args.seed,
            "leaf_views": args.leaf_views,
            "leaf_warmup": args.leaf_warmup,
            "leaf_repetitions": args.leaf_repetitions,
        },
    }
    return {"sha256": sha256_json(payload), "payload": payload}


def _assert_identity_unchanged(args, expected):
    if _identity(args) != expected:
        raise Phase3Error("Phase-3 code, configuration, profile, or workload inputs changed")


def _required_resource_int(mapping, key, label, minimum=0):
    value = mapping.get(key) if isinstance(mapping, dict) else None
    if type(value) is not int or value < minimum:
        raise Phase3Error(
            "{} requires integer {} >= {}".format(label, key, minimum)
        )
    return value


def _validate_resource_query(document, logical_gpu):
    """Validate every Raster/head resource used by the Phase-3 matrix."""

    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or document.get("kind") != "tacker_phase3_a6000_resource_query"
        or document.get("passed") is not True
    ):
        raise Phase3Error("A6000 resource query did not report a schema-v1 pass")
    device = document.get("device")
    if (
        not isinstance(device, dict)
        or device.get("index") != logical_gpu
        or device.get("name") != "NVIDIA RTX A6000"
        or device.get("compute_capability") != [8, 6]
    ):
        raise Phase3Error(
            "resource query must bind logical GPU {} to NVIDIA RTX A6000/sm_86"
            .format(logical_gpu)
        )
    sm_count = _required_resource_int(device, "sm_count", "resource device", 1)

    families = document.get("families")
    if not isinstance(families, dict):
        raise Phase3Error("resource query omitted Raster ABI families")
    raster_contracts = (
        ("legacy_pos_l1_v1", 1, 1, 384),
    ) + tuple(
        ("first_linear_heads_v2", 2, worker_groups, 256 + 128 * worker_groups)
        for worker_groups in range(1, 6)
    )
    for family_name, abi_version, worker_groups, expected_threads in raster_contracts:
        family = families.get(family_name)
        by_group = family.get("worker_groups") if isinstance(family, dict) else None
        raw = by_group.get(str(worker_groups)) if isinstance(by_group, dict) else None
        label = "Raster {} WG{}".format(family_name, worker_groups)
        if not isinstance(raw, dict):
            raise Phase3Error("resource query omitted {}".format(label))
        if raw.get("abi_version") != abi_version or raw.get("worker_groups") != worker_groups:
            raise Phase3Error("{} ABI/worker-group identity changed".format(label))
        if raw.get("device_ordinal") != logical_gpu:
            raise Phase3Error("{} queried the wrong logical CUDA device".format(label))
        if (
            raw.get("compute_capability_major") != 8
            or raw.get("compute_capability_minor") != 6
        ):
            raise Phase3Error("{} was not queried from sm_86".format(label))
        if raw.get("multiprocessor_count") != sm_count:
            raise Phase3Error("{} SM count differs from the CUDA device".format(label))
        if _required_resource_int(raw, "physical_threads", label, 1) != expected_threads:
            raise Phase3Error("{} physical thread count changed".format(label))
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
            _required_resource_int(raw, key, label, minimum)
        occupancy = raw.get("occupancy")
        if (
            not isinstance(occupancy, (int, float))
            or isinstance(occupancy, bool)
            or not math.isfinite(float(occupancy))
            or not 0.0 < float(occupancy) <= 1.0
            or raw.get("launch_supported") is not True
        ):
            raise Phase3Error("{} is not launchable with valid occupancy".format(label))

    head = document.get("head_v2")
    capabilities = head.get("capabilities") if isinstance(head, dict) else None
    resources = head.get("resources") if isinstance(head, dict) else None
    expected_symbols = {
        "multi_solo": HEAD_MULTI_SOLO_SYMBOL,
        "multi_gptb": HEAD_MULTI_GPTB_SYMBOL,
    }
    observed_symbols = (
        capabilities.get("global_kernel_symbols")
        if isinstance(capabilities, dict)
        else None
    )
    if (
        not isinstance(capabilities, dict)
        or capabilities.get("abi_version") != 2
        or capabilities.get("sm_target") != "sm_86"
        or capabilities.get("resource_query") != "tacker_resources_v2"
        or not isinstance(observed_symbols, dict)
        or any(observed_symbols.get(key) != value for key, value in expected_symbols.items())
        or list(capabilities.get("supported_worker_groups", [])) != [1, 2, 3, 4, 5]
    ):
        raise Phase3Error("head ABI-v2 capabilities are not the sm_86 WG1-5 contract")
    kernels = resources.get("kernels") if isinstance(resources, dict) else None
    if not isinstance(resources, dict) or resources.get("abi_version") != 2 or not isinstance(kernels, dict):
        raise Phase3Error("head tacker_resources_v2 returned an invalid document")
    expected_head_threads = [128, 256, 384, 512, 640]
    for symbol in HEAD_RESOURCE_SYMBOLS:
        facts = kernels.get(symbol)
        label = "head resource {}".format(symbol)
        if not isinstance(facts, dict):
            raise Phase3Error("resource query omitted {}".format(label))
        for key, minimum in (
            ("registers_per_thread", 1),
            ("static_shared_memory_bytes", 0),
            ("local_memory_bytes", 0),
            ("max_threads_per_block", expected_head_threads[-1]),
        ):
            _required_resource_int(facts, key, label, minimum)
        if facts.get("ptx_version") != 86 or facts.get("binary_version") != 86:
            raise Phase3Error("{} is not compiled for PTX/binary sm_86".format(label))
        if facts.get("worker_group_threads") != expected_head_threads:
            raise Phase3Error("{} does not cover exact WG1-5 CTA threads".format(label))
        active_blocks = facts.get("active_blocks_per_sm")
        if (
            not isinstance(active_blocks, list)
            or len(active_blocks) != 5
            or any(type(value) is not int or value < 1 for value in active_blocks)
        ):
            raise Phase3Error("{} does not have launchable WG1-5 occupancy".format(label))
    return device


def _resource_query(output, gpu):
    """Private child entry point: query the active compiled CUDA resources."""

    try:
        import torch
        from diff_gaussian_rasterization import tacker_resource_requirements

        head_package_root = PROJECT_ROOT / "tacker_ext"
        if str(head_package_root) not in sys.path:
            sys.path.insert(0, str(head_package_root))
        import tacker_4dgs_head

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        physical_gpu_from_environment()
        if gpu != 0 or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Phase 3 requires exactly one visible CUDA device and logical --gpu 0"
            )
        torch.cuda.set_device(gpu)
        properties = torch.cuda.get_device_properties(gpu)
        name = torch.cuda.get_device_name(gpu)
        capability = list(torch.cuda.get_device_capability(gpu))
        if name != "NVIDIA RTX A6000" or capability != [8, 6]:
            raise RuntimeError(
                "Phase 3 requires NVIDIA RTX A6000/sm_86, got {} / {}".format(
                    name, capability
                )
            )
        families = {
            "legacy_pos_l1_v1": {
                "worker_groups": {
                    "1": dict(tacker_resource_requirements(1, 1))
                }
            },
            "first_linear_heads_v2": {"worker_groups": {}},
        }
        for worker_groups in range(1, 6):
            families["first_linear_heads_v2"]["worker_groups"][
                str(worker_groups)
            ] = dict(tacker_resource_requirements(2, worker_groups))
        document = {
            "schema_version": 1,
            "kind": "tacker_phase3_a6000_resource_query",
            "passed": True,
            "device": {
                "index": gpu,
                "name": name,
                "compute_capability": capability,
                "sm_count": int(properties.multi_processor_count),
                "cuda_runtime": str(torch.version.cuda),
                "pytorch_version": str(torch.__version__),
            },
            "families": families,
            "head_v2": {
                "capabilities": dict(tacker_4dgs_head.tacker_capabilities_v2()),
                "resources": dict(tacker_4dgs_head.tacker_resources_v2()),
            },
        }
        _validate_resource_query(document, gpu)
    except Exception as error:
        document = {
            "schema_version": 1,
            "kind": "tacker_phase3_a6000_resource_query",
            "passed": False,
            "error": str(error),
        }
    atomic_write_json(output, document)
    if not document["passed"]:
        print(
            "Phase-3 A6000 Raster/head resource query failed: {}".format(
                document["error"]
            ),
            file=sys.stderr,
        )
    return 0 if document["passed"] else 1


def _validate_matrix(matrix):
    if (
        matrix.get("kind") != "tacker_autotune_candidate_matrix"
        or not isinstance(matrix.get("matrix_sha256"), str)
        or SHA256_RE.match(matrix["matrix_sha256"]) is None
        or not isinstance(matrix.get("candidates"), list)
        or not matrix["candidates"]
    ):
        raise Phase3Error("autotune emitted an invalid matrix")
    for candidate in matrix["candidates"]:
        if (
            not isinstance(candidate, dict)
            or not NAME_RE.match(candidate.get("variant_id", ""))
            or SHA256_RE.match(candidate.get("candidate_sha256", "")) is None
        ):
            raise Phase3Error("matrix contains an invalid candidate")
    try:
        validated = _load_autotune_module().validate_matrix(matrix)
    except Exception as error:
        raise Phase3Error("autotune matrix seal is invalid: {}".format(error))
    return validated


def _profile_paths(matrix, profiles_dir):
    result = {}
    for candidate in matrix["candidates"]:
        path = Path(profiles_dir) / "{}.json".format(candidate["variant_id"])
        if not path.is_file():
            raise Phase3Error("qualification profile is missing: {}".format(path))
        profile = load_json(path, "qualification profile")
        if (
            profile.get("selected_variant_id") != candidate["variant_id"]
            or profile.get("deployment") != {"enabled": False, "valid": False}
            or not isinstance(profile.get("provenance"), dict)
            or profile["provenance"].get("candidate_sha256")
            != candidate["candidate_sha256"]
            or profile["provenance"].get("matrix_sha256")
            != matrix["matrix_sha256"]
        ):
            raise Phase3Error("qualification profile binding changed: {}".format(path))
        result[candidate["variant_id"]] = str(path.resolve())
    return result


def build_benchmark_command(
    args,
    output,
    runs_dir,
    run_id,
    candidates,
    correctness,
    frames,
    warmup,
    trials,
    schedule,
    resume=False,
):
    argv = [
        str(Path(args.python_executable).expanduser().resolve()),
        _script("benchmark_tacker_fps.py"),
        "--output",
        str(Path(output).resolve()),
        "--runs-dir",
        str(Path(runs_dir).resolve()),
        "--run-id",
        run_id,
        "--current-tacker-profile",
        str(Path(args.current_tacker_profile).expanduser().resolve()),
    ]
    for name, path in candidates:
        if not NAME_RE.match(name):
            raise Phase3Error("unsafe candidate name: {}".format(name))
        argv.extend(["--candidate", "{}={}".format(name, Path(path).resolve())])
    argv.extend(
        [
            "--correctness-json",
            str(Path(correctness).resolve()),
            "--model-path",
            str(Path(args.model_path).expanduser().resolve()),
            "--source-path",
            str(Path(args.source_path).expanduser().resolve()),
            "--configs",
            str(Path(args.config).expanduser().resolve()),
            "--workload-name",
            args.workload_name,
            "--iteration",
            str(args.iteration),
            "--split",
            args.split,
            "--frames",
            str(frames),
            "--warmup",
            str(warmup),
            "--expected-image-width",
            str(args.image_width),
            "--expected-image-height",
            str(args.image_height),
            "--expected-gaussian-count",
            str(args.gaussian_count),
            "--trials",
            str(trials),
            "--schedule",
            schedule,
            "--seed",
            str(args.seed),
        ]
    )
    if args.timeout_seconds is not None:
        argv.extend(["--timeout-seconds", str(args.timeout_seconds)])
    if resume:
        argv.append("--resume")
    return argv


def _regular_artifacts_under(root):
    base = Path(root).resolve()
    if not base.is_dir():
        raise Phase3Error("artifact directory is missing: {}".format(base))
    result = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise Phase3Error("artifact directory contains a symlink: {}".format(path))
        if path.is_file():
            result.append(artifact(path))
        elif path.exists() and not path.is_dir():
            raise Phase3Error("artifact directory contains a special file: {}".format(path))
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
            "scope": "shared_cuda_suite_and_disabled_profile_preflight",
            "formal_correctness": False,
        }
    return result


def _screen_level(journal, args, runner, matrix, profiles_dir, head_count, label):
    selected = [
        item
        for item in matrix["candidates"]
        if len(item.get("selected_heads", [])) == head_count
        and (head_count != 1 or item.get("search_level") in ("c0", "c1"))
    ]
    if not selected:
        raise Phase3Error("screening level {} has no candidates".format(label))
    profiles = _profile_paths(matrix, profiles_dir)
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "head_count": head_count,
        "candidate_sha256": [item["candidate_sha256"] for item in selected],
        "protocol": {
            "frames": args.screen_frames,
            "warmup": args.screen_warmup,
            "trials": args.screen_trials,
            "schedule": "round_robin",
            "seed": args.seed,
        },
        "cuda_suite_sha256": sha256_file(args.cuda_suite_report),
    }

    def action(directory):
        session = journal.root / "sessions" / label
        session.mkdir(parents=True, exist_ok=True)
        correctness = session / "screening-correctness.json"
        report_path = session / "screening-report.json"
        scores_path = directory / "screening-scores.json"
        qualifications = _screening_qualifications(selected)
        atomic_write_json(correctness, qualifications)
        runs_dir = session / "runs"
        checkpoint = runs_dir / label / "benchmark.checkpoint.json"
        existing_pass = False
        if report_path.is_file():
            existing_pass = load_json(report_path, "existing screening report").get("passed") is True
        if not existing_pass:
            command = build_benchmark_command(
                args,
                report_path,
                runs_dir,
                label,
                [(item["variant_id"], profiles[item["variant_id"]]) for item in selected],
                correctness,
                args.screen_frames,
                args.screen_warmup,
                args.screen_trials,
                "round_robin",
                resume=checkpoint.is_file(),
            )
            invoke(runner, command)
        report = load_json(report_path, "short screening report")
        if report.get("passed") is not True:
            raise Phase3Error("short screening report did not pass")
        contract = report.get("contract")
        schedule_record = report.get("schedule")
        observed_names = [
            item.get("name")
            for item in report.get("candidates", [])
            if isinstance(item, dict)
        ]
        expected_names = ["serial", "two_stream", "current_tacker"] + [
            item["variant_id"] for item in selected
        ]
        if (
            not isinstance(contract, dict)
            or contract.get("profile_frames") != args.screen_frames
            or contract.get("warmup_frames") != args.screen_warmup
            or not isinstance(schedule_record, dict)
            or schedule_record.get("trials_per_candidate") != args.screen_trials
            or schedule_record.get("strategy") != "round_robin"
            or observed_names != expected_names
        ):
            raise Phase3Error("short screening report protocol/candidates changed")
        summaries = report.get("summaries")
        if not isinstance(summaries, dict):
            raise Phase3Error("short screening report omitted summaries")
        scores = {}
        for candidate in selected:
            summary = summaries.get(candidate["variant_id"])
            score = summary.get("median_throughput_fps") if isinstance(summary, dict) else None
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
                or float(score) <= 0.0
            ):
                raise Phase3Error(
                    "screening score is invalid for {}".format(candidate["variant_id"])
                )
            scores[candidate["candidate_sha256"]] = {
                "candidate_sha256": candidate["candidate_sha256"],
                "score": float(score),
                "status": "succeeded",
            }
        atomic_write_json(scores_path, scores)
        profile_artifacts = {
            item["candidate_sha256"]: artifact(profiles[item["variant_id"]])
            for item in selected
        }
        raw_artifacts = _regular_artifacts_under(runs_dir / label)
        raw_manifest = directory / "raw-artifacts.json"
        atomic_write_json(raw_manifest, {"artifacts": raw_artifacts})
        return (
            {
                "source_matrix_sha256": matrix["matrix_sha256"],
                "profiles_dir": str(Path(profiles_dir).resolve()),
                "profile_artifacts": profile_artifacts,
                "report_path": str(report_path),
                "scores_path": str(scores_path),
                "candidate_count": len(selected),
            },
            [correctness, report_path, scores_path, raw_manifest]
            + [item["path"] for item in raw_artifacts],
        )

    return journal.run("screen-{}".format(label), inputs, action)


def _make_profiles(journal, args, runner, matrix_path, matrix, label, resources_path):
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "resources_sha256": sha256_file(resources_path),
        "template_sha256": sha256_file(args.template_profile),
    }

    def action(directory):
        profiles_dir = directory / "profiles"
        command = [
            str(Path(args.python_executable).expanduser().resolve()),
            _script("tacker_autotune.py"),
            "profiles",
            "--matrix",
            str(Path(matrix_path).resolve()),
            "--resources",
            str(Path(resources_path).resolve()),
            "--template",
            str(Path(args.template_profile).expanduser().resolve()),
            "--output-dir",
            str(profiles_dir.resolve()),
        ]
        invoke(runner, command)
        _profile_paths(matrix, profiles_dir)
        manifest = profiles_dir / "qualification_profiles.json"
        return (
            {"profiles_dir": str(profiles_dir.resolve()), "manifest": str(manifest)},
            [manifest]
            + [profiles_dir / "{}.json".format(item["variant_id"]) for item in matrix["candidates"]],
        )

    return journal.run("profiles-{}".format(label), inputs, action)


def _beam(journal, args, runner, matrix_path, matrix, scores_path, target):
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "screening_sha256": sha256_file(scores_path),
        "beam_width": args.beam_width,
        "target_head_count": target,
    }

    def action(directory):
        output = directory / "matrix.json"
        command = [
            str(Path(args.python_executable).expanduser().resolve()),
            _script("tacker_autotune.py"),
            "beam",
            "--matrix",
            str(Path(matrix_path).resolve()),
            "--screening",
            str(Path(scores_path).resolve()),
            "--beam-width",
            str(args.beam_width),
            "--target-head-count",
            str(target),
            "--output",
            str(output.resolve()),
        ]
        invoke(runner, command)
        result = _validate_matrix(load_json(output, "beam matrix"))
        return ({"matrix_path": str(output), "matrix": result}, [output])

    return journal.run("beam-h{}".format(target), inputs, action)


def _load_autotune_module():
    path = Path(_script("tacker_autotune.py"))
    spec = importlib.util.spec_from_file_location("phase3_tacker_autotune", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rank_final(journal, args, matrix, level_results):
    scores = {}
    source_reports = {}
    source_bindings = {}
    for result in level_results:
        level_scores = load_json(result["scores_path"], "screening scores")
        for digest, value in level_scores.items():
            if digest in scores:
                raise Phase3Error("candidate was screened twice: {}".format(digest))
            scores[digest] = value
            source_reports[digest] = result["report_path"]
            profile_facts = result.get("profile_artifacts", {}).get(digest)
            if not isinstance(profile_facts, dict):
                raise Phase3Error("screening profile binding is missing")
            source_bindings[digest] = {
                "source_matrix_sha256": result["source_matrix_sha256"],
                "profile": profile_facts,
                "report": artifact(result["report_path"]),
            }
    if set(scores) != set(item["candidate_sha256"] for item in matrix["candidates"]):
        raise Phase3Error("stratified screening does not cover the final matrix exactly")
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "scores": scores,
        "top_k": args.top_k,
    }

    def action(directory):
        autotune = _load_autotune_module()
        ranked = autotune.rank_screening_candidates(
            matrix["candidates"], scores, top_k=None, maximize=True
        )
        document = {
            "schema_version": 1,
            "kind": "tacker_phase3_screening_ranking",
            "matrix_sha256": matrix["matrix_sha256"],
            "ranked": ranked,
            "correctness_target_k": args.top_k,
            "scores": scores,
            "source_reports": source_reports,
            "source_bindings": source_bindings,
        }
        path = directory / "ranking.json"
        atomic_write_json(path, document)
        return (
            {
                "ranking_path": str(path),
                "ranked": ranked,
                "scores": scores,
                "source_reports": source_reports,
                "source_bindings": source_bindings,
            },
            [path],
        )

    return journal.run("screening-ranking", inputs, action)


def _parse_json_stdout(completed, label):
    try:
        value = json.loads(_completed_text(completed, "stdout"))
    except (TypeError, ValueError) as error:
        raise Phase3Error("{} did not emit JSON: {}".format(label, error))
    if not isinstance(value, dict):
        raise Phase3Error("{} stdout must be a JSON object".format(label))
    return value


def _db_command(args, *items):
    return [
        str(Path(args.python_executable).expanduser().resolve()),
        _script("tacker_autotune.py"),
    ] + list(items)


def _db_claim(args, runner, db, matrix_sha, candidate_sha, stage, inputs, reclaim):
    command = _db_command(
        args,
        "db-claim",
        "--db",
        str(Path(db).resolve()),
        "--matrix-sha256",
        matrix_sha,
        "--candidate-sha256",
        candidate_sha,
        "--stage",
        stage,
        "--inputs",
        str(Path(inputs).resolve()),
    )
    if reclaim:
        command.append("--reclaim-running")
    return _parse_json_stdout(invoke(runner, command), "DB claim")


def _db_complete(
    args, runner, db, matrix_sha, candidate_sha, stage, inputs, token, evidence, result
):
    return _parse_json_stdout(
        invoke(
            runner,
            _db_command(
                args,
                "db-complete",
                "--db",
                str(Path(db).resolve()),
                "--matrix-sha256",
                matrix_sha,
                "--candidate-sha256",
                candidate_sha,
                "--stage",
                stage,
                "--inputs",
                str(Path(inputs).resolve()),
                "--artifact",
                str(Path(evidence).resolve()),
                "--result",
                str(Path(result).resolve()),
                "--claim-token",
                token,
            ),
        ),
        "DB completion",
    )


def _db_fail(args, runner, db, matrix_sha, candidate_sha, stage, inputs, token, error):
    invoke(
        runner,
        _db_command(
            args,
            "db-fail",
            "--db",
            str(Path(db).resolve()),
            "--matrix-sha256",
            matrix_sha,
            "--candidate-sha256",
            candidate_sha,
            "--stage",
            stage,
            "--inputs",
            str(Path(inputs).resolve()),
            "--error",
            str(error),
            "--claim-token",
            token,
        ),
    )


def _record_screening_db(
    journal, args, runner, db, matrix_path, matrix, ranking, profiles_manifest
):
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "workload": journal.state["identity"]["payload"]["workload"],
        "screening_protocol": journal.state["identity"]["payload"]["search"],
        "cuda_suite_report": artifact(args.cuda_suite_report),
        "final_profiles_manifest": artifact(profiles_manifest),
        "actual_measurement_bindings": ranking["source_bindings"],
        "benchmark_driver_sha256": sha256_file(_script("benchmark_tacker_fps.py")),
        "configuration_chain": journal.state["identity"]["payload"]["configuration_chain"],
    }

    def action(directory):
        db_path = Path(db).resolve()
        invoke(
            runner,
            _db_command(
                args,
                "db-register",
                "--db",
                str(db_path),
                "--matrix",
                str(Path(matrix_path).resolve()),
            ),
        )
        inputs_path = directory / "screening-inputs.json"
        atomic_write_json(inputs_path, inputs)
        input_digest = None
        completed_count = 0
        for candidate in matrix["candidates"]:
            digest = candidate["candidate_sha256"]
            claim = _db_claim(
                args,
                runner,
                db_path,
                matrix["matrix_sha256"],
                digest,
                "screening",
                inputs_path,
                False,
            )
            if claim.get("action") == "busy":
                claim = _db_claim(
                    args,
                    runner,
                    db_path,
                    matrix["matrix_sha256"],
                    digest,
                    "screening",
                    inputs_path,
                    True,
                )
            record = claim.get("record")
            if not isinstance(record, dict):
                raise Phase3Error("DB screening claim omitted its record")
            if input_digest is None:
                input_digest = record.get("input_sha256")
            elif input_digest != record.get("input_sha256"):
                raise Phase3Error("DB screening input hashes diverged")
            score_record = ranking["scores"].get(digest)
            if not isinstance(score_record, dict):
                raise Phase3Error("screening score missing for {}".format(digest))
            result_value = {
                "score": float(score_record["score"]),
                "valid": True,
                "scope": "short_e2e_screening",
            }
            if claim.get("action") == "skip":
                if record.get("result") != result_value:
                    raise Phase3Error("DB screening skip result changed")
                completed_count += 1
                continue
            token = claim.get("claim_token")
            if claim.get("action") != "run" or not isinstance(token, str):
                raise Phase3Error("DB screening stage is busy or unclaimable")
            evidence = {
                "schema_version": 1,
                "kind": "tacker_phase3_screening_evidence",
                "matrix_sha256": matrix["matrix_sha256"],
                "candidate_sha256": digest,
                "candidate": candidate,
                "score": result_value["score"],
                "actual_measurement": ranking["source_bindings"][digest],
                "stage_inputs_sha256": record.get("input_sha256"),
            }
            evidence_path = directory / "{}.evidence.json".format(digest)
            result_path = directory / "{}.result.json".format(digest)
            atomic_write_json(evidence_path, evidence)
            atomic_write_json(result_path, result_value)
            _db_complete(
                args,
                runner,
                db_path,
                matrix["matrix_sha256"],
                digest,
                "screening",
                inputs_path,
                token,
                evidence_path,
                result_path,
            )
            completed_count += 1
        marker = {
            "matrix_sha256": matrix["matrix_sha256"],
            "screening_input_sha256": input_digest,
            "completed_candidates": completed_count,
        }
        marker_path = directory / "screening-db.json"
        atomic_write_json(marker_path, marker)
        return (
            {
                "db": str(db_path),
                "screening_input_sha256": input_digest,
                "inputs_path": str(inputs_path),
                "marker": str(marker_path),
            },
            [inputs_path, marker_path],
        )

    return journal.run("screening-db", inputs, action)


def build_leaf_command(args, profile, matrix_path, directory):
    return [
        str(Path(args.python_executable).expanduser().resolve()),
        str((PROJECT_ROOT / "profile_tacker_leaves.py").resolve()),
        "--model_path",
        str(Path(args.model_path).expanduser().resolve()),
        "--source_path",
        str(Path(args.source_path).expanduser().resolve()),
        "--configs",
        str(Path(args.config).expanduser().resolve()),
        "--iteration",
        str(args.iteration),
        "--split",
        args.split,
        "--views",
        str(args.leaf_views),
        "--warmup",
        str(args.leaf_warmup),
        "--repetitions",
        str(args.leaf_repetitions),
        "--candidate-profile",
        str(Path(profile).resolve()),
        "--candidate-matrix",
        str(Path(matrix_path).resolve()),
        "--gpu",
        str(args.gpu),
        "--device-output",
        str((directory / "device.json").resolve()),
        "--raster-output",
        str((directory / "raster.json").resolve()),
        "--leaf-output",
        str((directory / "leaf.json").resolve()),
        "--report",
        str((directory / "leaf-report.json").resolve()),
        "--quiet",
    ]


def build_quality_command(args, profile, output):
    return [
        str(Path(args.python_executable).expanduser().resolve()),
        _script("validate_tacker_modes.py"),
        "--model_path",
        str(Path(args.model_path).expanduser().resolve()),
        "--source_path",
        str(Path(args.source_path).expanduser().resolve()),
        "--configs",
        str(Path(args.config).expanduser().resolve()),
        "--iteration",
        str(args.iteration),
        "--scene-name",
        args.workload_name,
        "--split",
        args.split,
        "--frames",
        "50",
        "--modes",
        "serial",
        "tacker",
        "--qualification-mode",
        "--qualification-profile",
        str(Path(profile).resolve()),
        "--gpu",
        str(args.gpu),
        "--output",
        str(Path(output).resolve()),
        "--quiet",
    ]


def _validate_leaf_report(
    report, args, candidate, profile, matrix_path, matrix_sha256, returncode
):
    workload = report.get("workload")
    if (
        report.get("schema_version") != 2
        or report.get("kind") != "4dgaussians_tacker_leaf_profile_report"
        or type(report.get("passed")) is not bool
        or not isinstance(workload, dict)
        or workload.get("scene") != "flame_steak"
        or workload.get("iteration") != args.iteration
        or workload.get("split") != args.split
        or workload.get("resolution") != [args.image_width, args.image_height]
        or workload.get("gaussian_count") != args.gaussian_count
        or (returncode == 0) != report["passed"]
    ):
        raise Phase3Error("variant leaf report contract changed")
    profile_path = str(Path(profile).resolve())
    matrix_resolved = str(Path(matrix_path).resolve())
    if report["passed"]:
        binding = report.get("profile_binding")
        parameters = report.get("parameters")
        if (
            report.get("variant_id") != candidate["variant_id"]
            or not isinstance(binding, dict)
            or binding.get("candidate_sha256") != candidate["candidate_sha256"]
            or binding.get("candidate_matrix_sha256") != matrix_sha256
            or binding.get("candidate_matrix_file_sha256") != sha256_file(matrix_path)
            or binding.get("profile_file_sha256") != sha256_file(profile)
            or not isinstance(parameters, dict)
            or parameters.get("candidate_profile") != profile_path
            or parameters.get("qualification_profile") is not True
            or parameters.get("used_as_deployment") is not False
            or report.get("measurement_outputs_written") is not True
        ):
            raise Phase3Error("passed leaf report is not bound to the candidate")
    elif (
        report.get("candidate_profile_requested") != profile_path
        or report.get("candidate_matrix_requested") != matrix_resolved
        or report.get("used_as_deployment") is not False
    ):
        raise Phase3Error("failed leaf report is not bound to the candidate request")
    return report["passed"]


def _validate_quality_report(report, args, profile, returncode):
    workload = report.get("workload")
    qualification = report.get("qualification")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "4dgaussians_tacker_quality_validation"
        or type(report.get("passed")) is not bool
        or not isinstance(workload, dict)
        or workload.get("scene") != "flame_steak"
        or workload.get("iteration") != args.iteration
        or workload.get("resolution") != [args.image_width, args.image_height]
        or workload.get("gaussian_count") != args.gaussian_count
        or not isinstance(qualification, dict)
        or qualification.get("enabled") is not True
        or qualification.get("admission_claimed") is not False
        or qualification.get("profile_override") != str(Path(profile).resolve())
        or (returncode == 0) != report["passed"]
    ):
        raise Phase3Error("candidate quality report contract changed")
    if report["passed"]:
        modes = report.get("modes")
        gates = report.get("gates")
        tacker = modes.get("tacker") if isinstance(modes, dict) else None
        tacker_gate = next(
            (
                gate
                for gate in gates
                if isinstance(gate, dict) and gate.get("mode") == "tacker"
            ),
            None,
        ) if isinstance(gates, list) else None
        if (
            set(modes) != {"serial", "tacker"}
            or not isinstance(tacker, dict)
            or tacker.get("actual_mode") != "tacker"
            or tacker.get("qualification_executed") is not True
            or not isinstance(tacker_gate, dict)
            or tacker_gate.get("passed") is not True
        ):
            raise Phase3Error("passed quality report did not execute qualified Tacker")
    return report["passed"]


def _baseline_quality(journal, args, runner):
    inputs = {
        "profile_sha256": sha256_file(args.current_tacker_profile),
        "frames": 50,
        "modes": ["serial", "two_stream", "tacker"],
        "workload": journal.state["identity"]["payload"]["workload"],
        "configuration_chain": journal.state["identity"]["payload"]["configuration_chain"],
    }

    def action(directory):
        output = directory / "baseline-quality.json"
        command = [
            str(Path(args.python_executable).expanduser().resolve()),
            _script("validate_tacker_modes.py"),
            "--model_path",
            str(Path(args.model_path).expanduser().resolve()),
            "--source_path",
            str(Path(args.source_path).expanduser().resolve()),
            "--configs",
            str(Path(args.config).expanduser().resolve()),
            "--iteration",
            str(args.iteration),
            "--scene-name",
            args.workload_name,
            "--split",
            args.split,
            "--frames",
            "50",
            "--modes",
            "serial",
            "two_stream",
            "tacker",
            "--tacker-profile",
            str(Path(args.current_tacker_profile).expanduser().resolve()),
            "--gpu",
            str(args.gpu),
            "--output",
            str(output.resolve()),
            "--quiet",
        ]
        invoke(runner, command, allowed=(0, 1))
        report = load_json(output, "baseline quality report")
        modes = report.get("modes")
        gates = report.get("gates")
        if not isinstance(modes, dict) or not isinstance(gates, list):
            raise Phase3Error("baseline quality report is incomplete")
        gate_by_mode = {
            gate.get("mode"): gate for gate in gates if isinstance(gate, dict)
        }
        serial_valid = isinstance(modes.get("serial"), dict)
        correctness = {
            "serial": {"valid": serial_valid, "quality_report_sha256": sha256_file(output)},
            "two_stream": {
                "valid": bool(gate_by_mode.get("two_stream", {}).get("passed") is True),
                "quality_report_sha256": sha256_file(output),
            },
            "current_tacker": {
                "valid": bool(gate_by_mode.get("tacker", {}).get("passed") is True),
                "quality_report_sha256": sha256_file(output),
            },
        }
        if not all(correctness[name]["valid"] for name in (
            "serial", "two_stream", "current_tacker"
        )):
            raise Phase3Error(
                "serial/two_stream/current_tacker baseline correctness failed"
            )
        correctness_path = directory / "baseline-correctness.json"
        atomic_write_json(correctness_path, correctness)
        return (
            {
                "quality_report": str(output),
                "correctness_path": str(correctness_path),
                "correctness": correctness,
            },
            [output, correctness_path],
        )

    return journal.run("baseline-quality", inputs, action)


def _candidate_correctness(
    journal,
    args,
    runner,
    db,
    matrix_path,
    matrix,
    profiles_dir,
    ranked,
):
    selected = [item["candidate"] for item in ranked]
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "candidate_sha256": [item["candidate_sha256"] for item in selected],
        "leaf": {
            "views": args.leaf_views,
            "warmup": args.leaf_warmup,
            "repetitions": args.leaf_repetitions,
            "profiler_sha256": sha256_file(PROJECT_ROOT / "profile_tacker_leaves.py"),
        },
        "quality": {
            "frames": 50,
            "driver_sha256": sha256_file(_script("validate_tacker_modes.py")),
        },
        "workload": journal.state["identity"]["payload"]["workload"],
        "configuration_chain": journal.state["identity"]["payload"]["configuration_chain"],
    }

    def action(directory):
        inputs_path = directory / "correctness-inputs.json"
        atomic_write_json(inputs_path, inputs)
        input_digest = None
        records = []
        valid_count = 0
        for rank, candidate in enumerate(selected, 1):
            digest = candidate["candidate_sha256"]
            claim = _db_claim(
                args,
                runner,
                db,
                matrix["matrix_sha256"],
                digest,
                "correctness",
                inputs_path,
                False,
            )
            if claim.get("action") == "busy":
                claim = _db_claim(
                    args,
                    runner,
                    db,
                    matrix["matrix_sha256"],
                    digest,
                    "correctness",
                    inputs_path,
                    True,
                )
            db_record = claim.get("record")
            if not isinstance(db_record, dict):
                raise Phase3Error("correctness DB claim omitted record")
            if input_digest is None:
                input_digest = db_record.get("input_sha256")
            elif input_digest != db_record.get("input_sha256"):
                raise Phase3Error("correctness DB input hashes diverged")
            if claim.get("action") == "skip":
                saved_result = db_record.get("result")
                if (
                    not isinstance(saved_result, dict)
                    or type(saved_result.get("valid")) is not bool
                ):
                    raise Phase3Error("resumed correctness result is invalid")
                records.append(
                    {
                        "rank": rank,
                        "candidate_sha256": digest,
                        "variant_id": candidate["variant_id"],
                        "result": saved_result,
                        "resumed_from_db": True,
                    }
                )
                if saved_result["valid"]:
                    valid_count += 1
                if valid_count >= args.top_k:
                    break
                continue
            token = claim.get("claim_token")
            if claim.get("action") != "run" or not isinstance(token, str):
                raise Phase3Error("correctness DB stage is busy or unclaimable")
            candidate_dir = directory / "rank{:02d}-{}".format(rank, candidate["variant_id"])
            candidate_dir.mkdir()
            profile = Path(profiles_dir) / "{}.json".format(candidate["variant_id"])
            try:
                leaf_completed = invoke(
                    runner,
                    build_leaf_command(args, profile, matrix_path, candidate_dir),
                    allowed=(0, 1),
                )
                if leaf_completed.returncode not in (0, 1):
                    raise Phase3Error("unexpected leaf profiler status")
                leaf_report_path = candidate_dir / "leaf-report.json"
                leaf_report = load_json(leaf_report_path, "variant leaf report")
                quality_path = candidate_dir / "quality.json"
                quality_completed = invoke(
                    runner,
                    build_quality_command(args, profile, quality_path),
                    allowed=(0, 1),
                )
                if quality_completed.returncode not in (0, 1):
                    raise Phase3Error("unexpected quality validator status")
                quality_report = load_json(quality_path, "candidate quality report")
                leaf_valid = _validate_leaf_report(
                    leaf_report,
                    args,
                    candidate,
                    profile,
                    matrix_path,
                    matrix["matrix_sha256"],
                    int(leaf_completed.returncode),
                )
                quality_valid = _validate_quality_report(
                    quality_report,
                    args,
                    profile,
                    int(quality_completed.returncode),
                )
                outputs = {}
                for name in ("device", "raster", "leaf"):
                    path = candidate_dir / "{}.json".format(name)
                    if leaf_valid and not path.is_file():
                        raise Phase3Error("passed leaf run omitted {}".format(path))
                    outputs[name] = artifact(path) if path.is_file() else None
                result_value = {
                    "valid": bool(leaf_valid and quality_valid),
                    "leaf_passed": bool(leaf_valid),
                    "quality_passed": bool(quality_valid),
                    "leaf_report_sha256": sha256_file(leaf_report_path),
                    "quality_report_sha256": sha256_file(quality_path),
                }
                evidence = {
                    "schema_version": 1,
                    "kind": "tacker_phase3_correctness_evidence",
                    "matrix_sha256": matrix["matrix_sha256"],
                    "candidate_sha256": digest,
                    "variant_id": candidate["variant_id"],
                    "profile": artifact(profile),
                    "leaf_report": artifact(leaf_report_path),
                    "leaf_outputs": outputs,
                    "quality_report": artifact(quality_path),
                    "result": result_value,
                    "stage_inputs_sha256": db_record.get("input_sha256"),
                }
                evidence_path = candidate_dir / "correctness-evidence.json"
                result_path = candidate_dir / "correctness-result.json"
                atomic_write_json(evidence_path, evidence)
                atomic_write_json(result_path, result_value)
                _db_complete(
                    args,
                    runner,
                    db,
                    matrix["matrix_sha256"],
                    digest,
                    "correctness",
                    inputs_path,
                    token,
                    evidence_path,
                    result_path,
                )
            except BaseException as error:
                try:
                    _db_fail(
                        args,
                        runner,
                        db,
                        matrix["matrix_sha256"],
                        digest,
                        "correctness",
                        inputs_path,
                        token,
                        error,
                    )
                except Exception:
                    pass
                raise
            records.append(
                {
                    "rank": rank,
                    "candidate_sha256": digest,
                    "variant_id": candidate["variant_id"],
                    "result": result_value,
                    "resumed_from_db": False,
                }
            )
            if result_value["valid"]:
                valid_count += 1
            if valid_count >= args.top_k:
                break
        if valid_count < args.top_k:
            raise Phase3Error(
                "only {} correctness-valid candidates remain; {} required".format(
                    valid_count, args.top_k
                )
            )
        marker = {
            "matrix_sha256": matrix["matrix_sha256"],
            "correctness_input_sha256": input_digest,
            "candidates": records,
            "valid_candidates": valid_count,
        }
        marker_path = directory / "correctness-db.json"
        atomic_write_json(marker_path, marker)
        return (
            {
                "correctness_input_sha256": input_digest,
                "records": records,
                "marker": str(marker_path),
                "inputs_path": str(inputs_path),
            },
            [inputs_path, marker_path],
        )

    return journal.run("candidate-correctness", inputs, action)


def _formal(journal, args, runner, db_result, correctness, baseline, matrix, profiles_dir):
    inputs = {
        "matrix_sha256": matrix["matrix_sha256"],
        "screening_input_sha256": db_result["screening_input_sha256"],
        "correctness_input_sha256": correctness["correctness_input_sha256"],
        "baseline_correctness_sha256": sha256_file(baseline["correctness_path"]),
        "top_k": args.top_k,
        "protocol": {"frames": 50, "warmup": 10, "trials": 10, "schedule": "abba"},
    }

    def action(directory):
        session = journal.root / "sessions" / "formal"
        session.mkdir(parents=True, exist_ok=True)
        correctness_output = session / "formal-correctness.json"
        plan_path = session / "formal-plan.json"
        invoke(
            runner,
            _db_command(
                args,
                "formal-plan",
                "--db",
                db_result["db"],
                "--matrix-sha256",
                matrix["matrix_sha256"],
                "--correctness-input-sha256",
                correctness["correctness_input_sha256"],
                "--screening-input-sha256",
                db_result["screening_input_sha256"],
                "--top-k",
                str(args.top_k),
                "--profiles-dir",
                str(Path(profiles_dir).resolve()),
                "--current-tacker-profile",
                str(Path(args.current_tacker_profile).expanduser().resolve()),
                "--baseline-correctness",
                baseline["correctness_path"],
                "--correctness-output",
                str(correctness_output.resolve()),
                "--output",
                str(plan_path.resolve()),
            ),
        )
        plan = load_json(plan_path, "formal benchmark plan")
        fragment = plan.get("benchmark_argv_fragment")
        if not isinstance(fragment, list) or not all(isinstance(item, str) for item in fragment):
            raise Phase3Error("formal plan omitted safe benchmark argv fragment")
        if (
            plan.get("kind") != "tacker_autotune_formal_benchmark_plan"
            or plan.get("matrix_sha256") != matrix["matrix_sha256"]
            or len(plan.get("candidates", [])) != args.top_k
        ):
            raise Phase3Error("formal plan did not contain exactly top-K candidates")
        report_path = session / "formal-fps.json"
        runs_dir = session / "runs"
        checkpoint = runs_dir / "phase3-formal" / "benchmark.checkpoint.json"
        command = [
            str(Path(args.python_executable).expanduser().resolve()),
            _script("benchmark_tacker_fps.py"),
            "--output",
            str(report_path.resolve()),
            "--runs-dir",
            str(runs_dir.resolve()),
            "--run-id",
            "phase3-formal",
        ] + fragment + [
            "--model-path",
            str(Path(args.model_path).expanduser().resolve()),
            "--source-path",
            str(Path(args.source_path).expanduser().resolve()),
            "--configs",
            str(Path(args.config).expanduser().resolve()),
            "--workload-name",
            args.workload_name,
            "--iteration",
            str(args.iteration),
            "--split",
            args.split,
            "--frames",
            "50",
            "--warmup",
            "10",
            "--expected-image-width",
            str(args.image_width),
            "--expected-image-height",
            str(args.image_height),
            "--expected-gaussian-count",
            str(args.gaussian_count),
            "--trials",
            "10",
            "--schedule",
            "abba",
            "--seed",
            str(args.seed),
        ]
        if args.timeout_seconds is not None:
            command.extend(["--timeout-seconds", str(args.timeout_seconds)])
        existing_pass = False
        if report_path.is_file():
            existing_pass = load_json(report_path, "existing formal FPS report").get("passed") is True
        if not existing_pass:
            if checkpoint.is_file():
                command.append("--resume")
            invoke(runner, command)
        report = load_json(report_path, "formal FPS report")
        protocol = report.get("phase0_exit_condition")
        contract = report.get("contract")
        schedule_record = report.get("schedule")
        expected_candidate_count = args.top_k + 3
        if (
            report.get("kind") != "4dgaussians_tacker_fps_benchmark"
            or report.get("passed") is not True
            or not isinstance(protocol, dict)
            or protocol.get("met") is not True
            or not isinstance(contract, dict)
            or contract.get("profile_frames") != 50
            or contract.get("warmup_frames") != 10
            or not isinstance(schedule_record, dict)
            or schedule_record.get("strategy") != "abba"
            or schedule_record.get("trials_per_candidate") != 10
            or len(report.get("candidates", [])) != expected_candidate_count
        ):
            raise Phase3Error("formal 10x50 ABBA benchmark did not pass its protocol")
        planned_names = [item.get("variant_id") for item in plan["candidates"]]
        comparisons = report.get("paired_comparisons")
        observed_pairs = {
            (item.get("candidate"), item.get("reference"))
            for item in comparisons
            if isinstance(item, dict)
        } if isinstance(comparisons, list) else set()
        required_pairs = {
            (name, reference)
            for name in planned_names
            for reference in ("two_stream", "current_tacker")
        }
        report_candidate_names = {
            item.get("name")
            for item in report.get("candidates", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        experimental_winner = report.get("experimental_winner")
        deployment_winner = report.get("deployment_winner")
        promotion = report.get("promotion")
        if (
            not planned_names
            or not all(isinstance(name, str) and name for name in planned_names)
            or not required_pairs.issubset(observed_pairs)
            or experimental_winner not in report_candidate_names
            or deployment_winner not in report_candidate_names
            or not isinstance(promotion, dict)
            or not isinstance(promotion.get("reason"), str)
            or not promotion.get("reason")
            or not isinstance(promotion.get("reason_code"), str)
            or not promotion.get("reason_code")
            or not isinstance(promotion.get("criteria"), dict)
        ):
            raise Phase3Error(
                "formal report omitted winner reasoning or paired baseline ratios"
            )
        raw_artifacts = _regular_artifacts_under(runs_dir / "phase3-formal")
        raw_manifest = directory / "raw-artifacts.json"
        atomic_write_json(raw_manifest, {"artifacts": raw_artifacts})
        return (
            {
                "plan_path": str(plan_path),
                "correctness_path": str(correctness_output),
                "fps_report": str(report_path),
            },
            [plan_path, correctness_output, report_path, raw_manifest]
            + [item["path"] for item in raw_artifacts],
        )

    return journal.run("formal-benchmark", inputs, action)


def _nsight(journal, args, runner, formal):
    physical_gpu = physical_gpu_from_environment()
    inputs = {
        "fps_report": artifact(formal["fps_report"]),
        "frames": 50,
        "limit": 3,
        "logical_gpu": args.gpu,
        "physical_gpu": physical_gpu,
    }

    def action(directory):
        session = journal.root / "sessions" / "nsight"
        output_dir = session / "profiles"
        report_path = session / "top3-nsight.json"
        command = [
            str(Path(args.python_executable).expanduser().resolve()),
            _script("profile_tacker_top3.py"),
            "--fps-report",
            formal["fps_report"],
            "--output-dir",
            str(output_dir.resolve()),
            "--report",
            str(report_path.resolve()),
            "--profile-script",
            str(Path(args.profile_nsight_script).expanduser().resolve()),
            "--model-path",
            str(Path(args.model_path).expanduser().resolve()),
            "--config",
            str(Path(args.config).expanduser().resolve()),
            "--source-path",
            str(Path(args.source_path).expanduser().resolve()),
            "--gpu",
            str(physical_gpu),
            "--frames",
            "50",
            "--iteration",
            str(args.iteration),
            "--workload-name",
            args.workload_name,
            "--limit",
            "3",
        ]
        existing_pass = False
        if report_path.is_file():
            existing_pass = load_json(report_path, "existing top-3 Nsight report").get("passed") is True
        if not existing_pass:
            if (output_dir / "top3.checkpoint.json").is_file():
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
            raise Phase3Error("top-3 Nsight report did not pass")
        raw_artifacts = _regular_artifacts_under(output_dir)
        raw_manifest = directory / "raw-artifacts.json"
        atomic_write_json(raw_manifest, {"artifacts": raw_artifacts})
        return (
            {"report": str(report_path)},
            [report_path, raw_manifest] + [item["path"] for item in raw_artifacts],
        )

    return journal.run("top3-nsight", inputs, action)


def _stopped(journal, stop_after, result):
    if stop_after is None:
        return None
    journal.state["status"] = "stopped_after_{}".format(stop_after)
    journal.state["stop_result"] = result
    journal.save()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "passed": None,
        "stopped_after": stop_after,
        "checkpoint": str(journal.path),
        "result": result,
    }


def dry_run_plan(args, identity):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "4dgaussians_tacker_phase3_dry_run_plan",
        "identity": identity,
        "executes_commands": False,
        "stages": [
            "A6000 preflight and ABI resource query",
            "complete C0/C1/C2 matrix and disabled profiles",
            "short whole-run screen: one-head then 2/3/4/5-head beam levels",
            "exact-input screening DB records and deterministic top-K",
            "top-K variant leaf plus 50-view quality correctness records",
            "top-K plus serial/two_stream/current_tacker 10x50 ABBA benchmark",
            "formal eligible-ranking top-3 Nsight profiling",
        ],
    }


def run_phase3(args, runner=subprocess.run):
    identity = _identity(args)
    physical_gpu = identity["payload"]["workload"]["physical_gpu"]
    if args.dry_run_plan:
        return dry_run_plan(args, identity)
    journal = Journal(args.output_dir, identity, args.resume)
    try:
        if journal.state.get("status") == "succeeded":
            saved = journal.state.get("report")
            if not isinstance(saved, dict) or artifact(saved.get("path", "")) != saved:
                raise Phase3Error("completed Phase-3 report changed")
            report = load_json(saved["path"], "completed Phase-3 report")
            if (
                report.get("kind") != REPORT_KIND
                or report.get("passed") is not True
                or report.get("identity") != identity
            ):
                raise Phase3Error("completed Phase-3 report contract changed")
            return report
        preflight_inputs = {
            "logical_gpu": args.gpu,
            "physical_gpu": physical_gpu,
            "resource_query_source_sha256": sha256_file(SCRIPT_PATH),
            "nvidia_smi": args.nvidia_smi,
        }

        def preflight_action(directory):
            smi = invoke(
                runner,
                [
                    args.nvidia_smi,
                    "--id={}".format(physical_gpu),
                    "--query-gpu=index,name,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
            )
            if "NVIDIA RTX A6000" not in _completed_text(smi, "stdout"):
                raise Phase3Error("nvidia-smi did not report an RTX A6000")
            smi_path = directory / "nvidia-smi.txt"
            smi_path.write_text(_completed_text(smi, "stdout"), encoding="utf-8")
            resources = directory / "resources.json"
            invoke(
                runner,
                [
                    str(Path(args.python_executable).expanduser().resolve()),
                    str(SCRIPT_PATH),
                    "_resource-query",
                    "--gpu",
                    str(args.gpu),
                    "--output",
                    str(resources.resolve()),
                ],
            )
            document = load_json(resources, "A6000 resource query")
            device = _validate_resource_query(document, args.gpu)
            return (
                {"resources": str(resources), "device": device, "nvidia_smi": str(smi_path)},
                [resources, smi_path],
            )

        preflight = journal.run("preflight", preflight_inputs, preflight_action)
        if args.stop_after == "preflight":
            return _stopped(journal, "preflight", preflight)

        raster_tiles = ((args.image_width + 15) // 16) * ((args.image_height + 15) // 16)
        backend_blocks = 0 if args.head_rows == 0 else ((args.head_rows + 15) // 16) * 2
        matrix_inputs = {
            "sm_count": preflight["device"]["sm_count"],
            "raster_tile_count": raster_tiles,
            "backend_logical_blocks": backend_blocks,
            "current_persistent_blocks": identity["payload"]["search"]["current_persistent_blocks"],
            "extra_persistent_blocks": identity["payload"]["search"]["extra_persistent_blocks"],
        }

        def matrix_action(directory):
            output = directory / "matrix.json"
            command = [
                str(Path(args.python_executable).expanduser().resolve()),
                _script("tacker_autotune.py"),
                "matrix",
                "--sm-count",
                str(matrix_inputs["sm_count"]),
                "--raster-tile-count",
                str(raster_tiles),
                "--backend-logical-blocks",
                str(backend_blocks),
                "--current-persistent-blocks",
                str(matrix_inputs["current_persistent_blocks"]),
                "--persistent-block",
                "0",
            ]
            for value in matrix_inputs["extra_persistent_blocks"]:
                command.extend(["--persistent-block", str(value)])
            command.extend(["--output", str(output.resolve())])
            invoke(runner, command)
            matrix = _validate_matrix(load_json(output, "base matrix"))
            return ({"matrix_path": str(output), "matrix": matrix}, [output])

        matrix_result = journal.run("base-matrix", matrix_inputs, matrix_action)
        matrix_path = matrix_result["matrix_path"]
        matrix = matrix_result["matrix"]
        profiles = _make_profiles(
            journal, args, runner, matrix_path, matrix, "base", preflight["resources"]
        )
        _assert_identity_unchanged(args, identity)
        if args.stop_after == "prepare":
            return _stopped(
                journal,
                "prepare",
                {"matrix": matrix_result, "profiles": profiles},
            )

        level_results = []
        level_results.append(
            _screen_level(journal, args, runner, matrix, profiles["profiles_dir"], 1, "h1")
        )
        level_results.append(
            _screen_level(journal, args, runner, matrix, profiles["profiles_dir"], 2, "h2")
        )
        parent_scores = level_results[-1]["scores_path"]
        for target in (3, 4, 5):
            beam = _beam(
                journal, args, runner, matrix_path, matrix, parent_scores, target
            )
            matrix_path = beam["matrix_path"]
            matrix = beam["matrix"]
            profiles = _make_profiles(
                journal,
                args,
                runner,
                matrix_path,
                matrix,
                "h{}".format(target),
                preflight["resources"],
            )
            screened = _screen_level(
                journal,
                args,
                runner,
                matrix,
                profiles["profiles_dir"],
                target,
                "h{}".format(target),
            )
            level_results.append(screened)
            parent_scores = screened["scores_path"]
        ranking = _rank_final(journal, args, matrix, level_results)
        db_path = journal.root / "profile.sqlite"
        db_result = _record_screening_db(
            journal,
            args,
            runner,
            db_path,
            matrix_path,
            matrix,
            ranking,
            profiles["manifest"],
        )
        _assert_identity_unchanged(args, identity)
        if args.stop_after == "screen":
            return _stopped(
                journal,
                "screen",
                {"ranking": ranking, "database": db_result},
            )

        baseline = _baseline_quality(journal, args, runner)
        correctness = _candidate_correctness(
            journal,
            args,
            runner,
            db_path,
            matrix_path,
            matrix,
            profiles["profiles_dir"],
            ranking["ranked"],
        )
        _assert_identity_unchanged(args, identity)
        if args.stop_after == "quality":
            return _stopped(
                journal,
                "quality",
                {"baseline": baseline, "candidates": correctness},
            )

        formal = _formal(
            journal,
            args,
            runner,
            db_result,
            correctness,
            baseline,
            matrix,
            profiles["profiles_dir"],
        )
        _assert_identity_unchanged(args, identity)
        if args.stop_after == "formal":
            return _stopped(journal, "formal", formal)
        nsight = _nsight(journal, args, runner, formal)
        _assert_identity_unchanged(args, identity)
        report_path = journal.root / "phase3-report.json"
        if report_path.exists():
            report = load_json(report_path, "Phase-3 report")
            if (
                report.get("kind") != REPORT_KIND
                or report.get("passed") is not True
                or report.get("identity") != identity
            ):
                raise Phase3Error("existing Phase-3 report changed")
        else:
            report = {
                "schema_version": SCHEMA_VERSION,
                "kind": REPORT_KIND,
                "generated_at_utc": utc_now(),
                "passed": True,
                "identity": identity,
                "matrix": {
                    "path": matrix_path,
                    "matrix_sha256": matrix["matrix_sha256"],
                    "candidate_count": len(matrix["candidates"]),
                },
                "screening": {
                    "ranking": ranking["ranking_path"],
                    "database": db_result,
                },
                "correctness": correctness,
                "formal": formal,
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
        help="disabled schema-v2 profile used only as a qualification template",
    )
    parser.add_argument("--cuda-suite-report", required=True)
    parser.add_argument("--output", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument(
        "--profile-nsight-script", default=_script("profile_nsight.sh")
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workload-name", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--gaussian-count", type=int, required=True)
    parser.add_argument("--head-rows", type=int, required=True)
    parser.add_argument("--persistent-block", type=int, action="append", default=[])
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
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
    positive = (
        "image_width",
        "image_height",
        "gaussian_count",
        "head_rows",
        "beam_width",
        "top_k",
        "screen_frames",
        "screen_trials",
        "leaf_views",
        "leaf_repetitions",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.screen_trials < 2:
        parser.error("--screen-trials must be at least 2")
    if args.screen_warmup < 0 or args.leaf_warmup < 0 or args.gpu < 0:
        parser.error("warmup and GPU values must be non-negative")
    if args.gpu != 0:
        parser.error(
            "--gpu must be logical device 0; select one physical GPU with "
            "CUDA_VISIBLE_DEVICES=<ordinal>"
        )
    if any(value < 0 for value in args.persistent_block):
        parser.error("--persistent-block must be non-negative")
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
            "Phase 3 is fixed to flame_steak/test iteration 14000, "
            "1352x1014, and 111525 Gaussian/head rows"
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
        report = run_phase3(args)
    except Exception as error:
        print("Phase-3 orchestration failed closed: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
