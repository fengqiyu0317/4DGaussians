#!/usr/bin/env python3
"""Independently replay and verify a completed Tacker Phase-3.1 run.

The verifier treats the run directory as untrusted input.  It first validates
the checkpoint artifact ledger, then reconstructs the staged candidate space,
screening ranking, formal set, qualification profiles, and final selection.
It never writes inside the run directory; only ``--output`` is atomically
published after verification.
"""

from __future__ import print_function

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import types


PROJECT_ROOT_ENV = "TACKER_PHASE31_PROJECT_ROOT"
PROJECT_ROOT = Path(
    os.environ.get(PROJECT_ROOT_ENV, str(Path(__file__).resolve().parents[1]))
).expanduser().resolve()
AUTOTUNE_PATH = PROJECT_ROOT / "scripts" / "tacker_autotune.py"
BENCHMARK_PATH = PROJECT_ROOT / "scripts" / "benchmark_tacker_fps.py"
TOP3_PATH = PROJECT_ROOT / "scripts" / "profile_tacker_top3.py"
STATE_KIND = "4dgaussians_tacker_phase31_checkpoint"
RUN_REPORT_KIND = "4dgaussians_tacker_phase31_run"
VERIFY_REPORT_KIND = "4dgaussians_tacker_phase31_postflight_verification"
FORMAL_SET_KIND = "tacker_phase31_formal_candidate_set"
SELECTION_KIND = "tacker_phase31_selection"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEARCH_FAMILIES = ("c0", "c1", "c2", "c3", "c4")
BASELINES = ("serial", "two_stream", "current_tacker")
EXPECTED_WORKLOAD = {
    "name": "flame_steak",
    "iteration": 14000,
    "split": "test",
    "image_width": 1352,
    "image_height": 1014,
    "gaussian_count": 111525,
    "head_rows": 111525,
    "logical_gpu": 0,
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


AUTOTUNE = None
BENCHMARK = None
TOP3 = None
_ACTIVE_SNAPSHOT_STORE = None


class VerificationError(RuntimeError):
    """The completed-run evidence failed independent verification."""


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
        raise VerificationError("non-canonical JSON value: {}".format(error))


def sha256_json(value, domain):
    return hashlib.sha256(
        canonical_json_bytes({"domain": domain, "payload": value})
    ).hexdigest()


def _stat_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        getattr(value, "st_mtime_ns", int(value.st_mtime * 1000000000)),
    )


def _lexical_path(path, label, directory=False):
    if not isinstance(path, (str, os.PathLike)) or "\x00" in str(path):
        raise VerificationError("{} path is invalid".format(label))
    raw = Path(path).expanduser()
    lexical = Path(os.path.abspath(str(raw)))
    try:
        resolved = lexical.resolve(strict=True)
        status = os.lstat(str(lexical))
    except OSError as error:
        raise VerificationError("{} is missing: {}".format(label, error))
    if resolved != lexical or stat.S_ISLNK(status.st_mode):
        raise VerificationError("{} contains a symbolic link".format(label))
    expected = stat.S_ISDIR(status.st_mode) if directory else stat.S_ISREG(status.st_mode)
    if not expected:
        raise VerificationError("{} is not a regular {}".format(
            label, "directory" if directory else "file"
        ))
    return lexical


class SnapshotStore(object):
    """One-fd immutable snapshots for every artifact used by verification."""

    def __init__(self, after_read_hook=None):
        self.entries = {}
        self.after_read_hook = after_read_hook

    def snapshot(self, path, label="artifact", capture_bytes=False):
        target = _lexical_path(path, label)
        key = str(target)
        existing = self.entries.get(key)
        if existing is not None:
            if capture_bytes and existing.get("bytes") is None:
                raise VerificationError(
                    "{} was not byte-snapshotted on first access".format(label)
                )
            return existing
        capture = bool(
            capture_bytes or target.suffix.lower() in (".json", ".py")
        )
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = None
        try:
            before_path = os.lstat(str(target))
            descriptor = os.open(str(target), flags)
            before_fd = os.fstat(descriptor)
            digest = hashlib.sha256()
            chunks = [] if capture else None
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            if self.after_read_hook is not None:
                self.after_read_hook(target)
            after_fd = os.fstat(descriptor)
            after_path = os.lstat(str(target))
        except OSError as error:
            raise VerificationError("cannot snapshot {} {}: {}".format(label, target, error))
        finally:
            if descriptor is not None:
                os.close(descriptor)
        identities = (
            _stat_identity(before_path),
            _stat_identity(before_fd),
            _stat_identity(after_fd),
            _stat_identity(after_path),
        )
        if len(set(identities)) != 1 or not stat.S_ISREG(after_path.st_mode):
            raise VerificationError("{} changed while being snapshotted".format(label))
        entry = {
            "path": key,
            "sha256": digest.hexdigest(),
            "size_bytes": before_fd.st_size,
            "stat_identity": _stat_identity(before_fd),
            "bytes": b"".join(chunks) if chunks is not None else None,
            "json": None,
        }
        self.entries[key] = entry
        return entry

    def copy_to(self, path, destination, label="artifact"):
        """Copy an already-verified snapshot without exposing a consumer to it."""

        target = _lexical_path(path, label)
        key = str(target)
        entry = self.entries.get(key)
        if entry is None:
            entry = self.snapshot(target, label)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if entry.get("bytes") is not None:
            destination.write_bytes(entry["bytes"])
            return
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = None
        output = None
        try:
            before_path = os.lstat(str(target))
            descriptor = os.open(str(target), flags)
            before_fd = os.fstat(descriptor)
            output = os.open(
                str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                offset = 0
                while offset < len(chunk):
                    offset += os.write(output, chunk[offset:])
            after_fd = os.fstat(descriptor)
            after_path = os.lstat(str(target))
        except OSError as error:
            raise VerificationError("cannot copy {} {}: {}".format(label, target, error))
        finally:
            if output is not None:
                os.close(output)
            if descriptor is not None:
                os.close(descriptor)
        identities = {
            _stat_identity(before_path),
            _stat_identity(before_fd),
            _stat_identity(after_fd),
            _stat_identity(after_path),
            entry["stat_identity"],
        }
        if (
            len(identities) != 1
            or digest.hexdigest() != entry["sha256"]
            or size != entry["size_bytes"]
        ):
            raise VerificationError("{} changed after its verified snapshot".format(label))

    def revalidate(self, path, label="external input"):
        """Re-hash a live external input before/after a subprocess call."""

        target = _lexical_path(path, label)
        entry = self.entries.get(str(target))
        if entry is None:
            entry = self.snapshot(target, label)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = None
        try:
            before_path = os.lstat(str(target))
            descriptor = os.open(str(target), flags)
            before_fd = os.fstat(descriptor)
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
            after_fd = os.fstat(descriptor)
            after_path = os.lstat(str(target))
        except OSError as error:
            raise VerificationError(
                "cannot revalidate {} {}: {}".format(label, target, error)
            )
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if (
            len(
                {
                    _stat_identity(before_path),
                    _stat_identity(before_fd),
                    _stat_identity(after_fd),
                    _stat_identity(after_path),
                    entry["stat_identity"],
                }
            )
            != 1
            or digest.hexdigest() != entry["sha256"]
            or size != entry["size_bytes"]
        ):
            raise VerificationError("{} changed across replay".format(label))
        return entry

    def json(self, path, label="JSON"):
        entry = self.snapshot(path, label, capture_bytes=True)
        if entry["json"] is None:
            try:
                value = json.loads(
                    entry["bytes"].decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_constant,
                )
                canonical_json_bytes(value)
            except VerificationError:
                raise
            except (UnicodeError, TypeError, ValueError) as error:
                raise VerificationError(
                    "cannot parse {} {}: {}".format(label, entry["path"], error)
                )
            if not isinstance(value, dict):
                raise VerificationError("{} must be a JSON object".format(label))
            entry["json"] = value
        return deepcopy(entry["json"])


def _snapshot_store():
    return _ACTIVE_SNAPSHOT_STORE if _ACTIVE_SNAPSHOT_STORE is not None else SnapshotStore()


def sha256_file(path):
    return _snapshot_store().snapshot(path)["sha256"]


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError("duplicate JSON key {!r}".format(key))
        result[key] = value
    return result


def _reject_constant(value):
    raise VerificationError("non-finite JSON constant {!r}".format(value))


def load_json(path, label="JSON"):
    return _snapshot_store().json(path, label)


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
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _runner_pretty_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _within(path, root):
    try:
        Path(os.path.abspath(str(Path(path).expanduser()))).relative_to(
            Path(os.path.abspath(str(Path(root).expanduser())))
        )
        return True
    except ValueError:
        return False


def _artifact_facts(path):
    snapshot = _snapshot_store().snapshot(path, "artifact")
    return {
        "path": snapshot["path"],
        "sha256": snapshot["sha256"],
        "size_bytes": snapshot["size_bytes"],
    }


def _verify_artifact(record, root, label, allow_external=False):
    if not isinstance(record, dict):
        raise VerificationError("{} artifact record is not an object".format(label))
    required = {"path", "sha256", "size_bytes"}
    if set(record) != required:
        raise VerificationError("{} artifact fields changed".format(label))
    if SHA256_RE.match(record.get("sha256", "")) is None:
        raise VerificationError("{} artifact hash is invalid".format(label))
    path = _lexical_path(record.get("path", ""), label)
    if not allow_external and not _within(path, root):
        raise VerificationError("{} artifact escapes the run root".format(label))
    observed = _artifact_facts(path)
    if observed != record:
        raise VerificationError("{} artifact bytes changed".format(label))
    return observed


def _verify_identity_record(record, label):
    if not isinstance(record, dict) or not {"path", "sha256"}.issubset(record):
        raise VerificationError("{} identity artifact is invalid".format(label))
    path = _lexical_path(record["path"], label)
    snapshot = _snapshot_store().snapshot(path, label)
    if snapshot["sha256"] != record["sha256"]:
        raise VerificationError("{} identity artifact hash changed".format(label))
    if "size_bytes" in record and snapshot["size_bytes"] != record["size_bytes"]:
        raise VerificationError("{} identity artifact size changed".format(label))
    return str(path)


def _identity_paths(identity):
    if not isinstance(identity, dict) or set(identity) != {"sha256", "payload"}:
        raise VerificationError("run identity schema changed")
    payload = identity["payload"]
    if not isinstance(payload, dict):
        raise VerificationError("identity payload is invalid")
    expected_hash = sha256_json(payload, "tacker-phase31-identity-v1")
    if identity["sha256"] != expected_hash:
        raise VerificationError("run identity SHA-256 changed")
    workload = payload.get("workload")
    if not isinstance(workload, dict) or any(
        workload.get(key) != value for key, value in EXPECTED_WORKLOAD.items()
    ):
        raise VerificationError("identity is not the fixed Phase-3.1 workload")
    if type(workload.get("physical_gpu")) is not int or workload["physical_gpu"] < 0:
        raise VerificationError("identity physical GPU is invalid")
    paths = set()
    for section in (
        "files",
        "scripts",
        "runtime_sources",
        "manifests",
        "workload_files",
    ):
        records = payload.get(section)
        if not isinstance(records, dict):
            raise VerificationError("identity omitted {}".format(section))
        for name, record in sorted(records.items()):
            paths.add(_verify_identity_record(record, "identity.{}.{}".format(section, name)))
    chain = payload.get("configuration_chain")
    if not isinstance(chain, list) or not chain:
        raise VerificationError("identity configuration chain is empty")
    for index, record in enumerate(chain):
        paths.add(_verify_identity_record(record, "identity.configuration_chain[{}]".format(index)))
    scripts = payload["scripts"]
    for name, expected_path in (
        ("autotune", AUTOTUNE_PATH),
        ("benchmark", BENCHMARK_PATH),
        ("top3", TOP3_PATH),
    ):
        record = scripts.get(name)
        observed_path = _verify_identity_record(
            record, "identity.scripts.{}".format(name)
        )
        if observed_path != str(expected_path.resolve()):
            raise VerificationError(
                "replay module {} is not the identity-sealed source".format(name)
            )
    return paths


def _module_from_snapshot(name, path, payload):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__loader__ = None
    sys.modules[name] = module
    try:
        code = compile(payload, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_replay_modules(identity):
    global AUTOTUNE, BENCHMARK, TOP3
    scripts = identity["payload"]["scripts"]
    loaded = []
    for key, module_name, expected_path in (
        ("autotune", "tacker_phase31_postflight_autotune", AUTOTUNE_PATH),
        ("benchmark", "tacker_phase31_postflight_benchmark", BENCHMARK_PATH),
        ("top3", "tacker_phase31_postflight_top3", TOP3_PATH),
    ):
        record = scripts[key]
        snapshot = _snapshot_store().snapshot(
            record["path"], "sealed replay module {}".format(key), capture_bytes=True
        )
        if (
            snapshot["path"] != str(expected_path)
            or snapshot["sha256"] != record["sha256"]
        ):
            raise VerificationError("sealed replay module {} changed".format(key))
        loaded.append(
            _module_from_snapshot(module_name, expected_path, snapshot["bytes"])
        )
    AUTOTUNE, BENCHMARK, TOP3 = loaded


def _verify_preflight(stages, identity, verified, run_root):
    result = _require_stage(stages, "preflight")["result"]
    resources_path = _verified_path(
        result.get("resources", ""), verified, run_root, "preflight resources"
    )
    resources = load_json(resources_path, "preflight resources")
    device = resources.get("device")
    workload = identity["payload"]["workload"]
    if (
        resources.get("schema_version") != 1
        or resources.get("kind") != "tacker_phase31_a6000_resource_query"
        or resources.get("passed") is not True
        or not isinstance(device, dict)
        or device != result.get("device")
        or device.get("index") != workload["logical_gpu"]
        or device.get("name") != "NVIDIA RTX A6000"
        or device.get("compute_capability") != [8, 6]
        or type(device.get("sm_count")) is not int
        or device["sm_count"] <= 0
    ):
        raise VerificationError("preflight is not a sealed A6000/sm_86 pass")
    smi_path = _verified_path(
        result.get("nvidia_smi", ""), verified, run_root, "nvidia-smi evidence"
    )
    smi = load_json(smi_path, "nvidia-smi evidence")
    if (
        smi.get("physical_gpu") != workload["physical_gpu"]
        or "NVIDIA RTX A6000" not in str(smi.get("stdout", ""))
    ):
        raise VerificationError("nvidia-smi evidence is not bound to the A6000")
    manifest_records = resources.get("manifests")
    identity_manifests = identity["payload"]["manifests"]
    if not isinstance(manifest_records, dict):
        raise VerificationError("preflight omitted manifest snapshots")
    for name, record in identity_manifests.items():
        sealed = manifest_records.get(name)
        if not isinstance(sealed, dict) or sealed.get("sha256") != record.get("sha256"):
            raise VerificationError("preflight manifest {} changed".format(name))
    capabilities = resources.get("raster_capabilities")
    if (
        not isinstance(capabilities, dict)
        or capabilities.get("resource_query_family_aware") is not True
        or capabilities.get("supported_mixed_abis") != [1, 2, 3, 4]
        or capabilities.get("supported_backend_families")
        != [
            "first_linear_heads_v2",
            "packed_first_linear_v3",
            "whole_heads_v4",
        ]
    ):
        raise VerificationError("preflight family/ABI capabilities changed")
    return result


def _stage_map(state):
    stages = state.get("stages")
    if not isinstance(stages, list) or not stages:
        raise VerificationError("checkpoint has no stages")
    result = {}
    for stage in stages:
        if not isinstance(stage, dict) or not isinstance(stage.get("name"), str):
            raise VerificationError("checkpoint contains an invalid stage")
        name = stage["name"]
        if name in result:
            raise VerificationError("checkpoint contains duplicate stage {}".format(name))
        if stage.get("status") != "succeeded":
            raise VerificationError("completed checkpoint has non-success stage {}".format(name))
        result[name] = stage
    return result


def _audit_stage_artifacts(stages, run_root):
    verified = {}
    artifact_count = 0
    for name, stage in sorted(stages.items()):
        records = stage.get("artifacts")
        if not isinstance(records, list):
            raise VerificationError("stage {} artifact ledger is invalid".format(name))
        for index, record in enumerate(records):
            facts = _verify_artifact(
                record,
                run_root,
                "stage {} artifact {}".format(name, index),
            )
            previous = verified.get(facts["path"])
            if previous is not None and previous != facts:
                raise VerificationError("conflicting artifact ledger entry")
            verified[facts["path"]] = facts
            artifact_count += 1
    return verified, artifact_count


def _require_stage(stages, name):
    stage = stages.get(name)
    if stage is None or not isinstance(stage.get("result"), dict):
        raise VerificationError("required succeeded stage {} is missing".format(name))
    return stage


def _verified_path(path, verified, run_root, label):
    target = str(_lexical_path(path, label))
    if not _within(target, run_root):
        raise VerificationError("{} escapes the run root".format(label))
    if target not in verified:
        raise VerificationError("{} is not in the succeeded-stage ledger".format(label))
    return Path(target)


def _matrix_from_stage(stages, name, verified, run_root):
    stage = _require_stage(stages, name)
    result = stage["result"]
    path = _verified_path(result.get("matrix_path", ""), verified, run_root, name)
    matrix = load_json(path, name)
    if result.get("matrix") != matrix:
        raise VerificationError("{} embedded matrix differs from its artifact".format(name))
    try:
        AUTOTUNE.validate_matrix(matrix)
    except Exception as error:
        raise VerificationError("{} matrix is non-canonical: {}".format(name, error))
    expected_bytes = _runner_pretty_bytes(matrix)
    if _snapshot_store().snapshot(path, name, capture_bytes=True)["bytes"] != expected_bytes:
        raise VerificationError("{} matrix bytes are not canonical".format(name))
    if (
        result.get("matrix_sha256") != matrix.get("matrix_sha256")
        or result.get("candidate_count") != len(matrix.get("candidates", []))
    ):
        raise VerificationError("{} result summary changed".format(name))
    return matrix, path


def _portable_artifact(record):
    if record is None:
        return None
    return {"sha256": record.get("sha256"), "size_bytes": record.get("size_bytes")}


def _portable_bindings(bindings):
    result = {}
    for digest, binding in sorted(bindings.items()):
        result[digest] = {
            "source_matrix_sha256": binding["source_matrix_sha256"],
            "profile": _portable_artifact(binding["profile"]),
            "report": _portable_artifact(binding["report"]),
            "screening_batch": binding["screening_batch"],
        }
    return result


def _candidate_count(matrix, family):
    return sum(item.get("search_family") == family for item in matrix["candidates"])


def _validate_exhaustive_c2(matrix):
    c2 = [item for item in matrix["candidates"] if item.get("search_family") == "c2"]
    head_sets = {tuple(item["selected_heads"]) for item in c2}
    expected_sets = 26  # C(5,2) + C(5,3) + C(5,4) + C(5,5)
    if len(c2) != 450 or len(head_sets) != expected_sets:
        raise VerificationError("C2 is not the exhaustive 450/450 matrix")
    for heads in head_sets:
        grid = {
            (item["worker_groups"], item["effective_persistent_blocks"])
            for item in c2
            if tuple(item["selected_heads"]) == heads
        }
        if len(grid) != len(heads) * 6:
            raise VerificationError("C2 head-set launch grid is incomplete")


def _record_for_binding(
    stages,
    digest,
    binding,
    verified,
    run_root,
    expected_protocol,
    document_cache,
):
    label = binding.get("screening_batch")
    if not isinstance(label, str) or not label:
        raise VerificationError("screening binding omitted batch label")
    stage = _require_stage(stages, "screen-{}".format(label))
    result = stage["result"]
    result_path = _verified_path(
        result.get("path", ""), verified, run_root, "screening batch"
    )
    cache_key = str(result_path)
    document = document_cache.get(cache_key)
    if document is None:
        document = load_json(result_path, "screening batch")
        document_cache[cache_key] = document
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "tacker_phase31_screening_batch"
        or document.get("records") != result.get("records")
        or document.get("measurement_source_bindings")
        != result.get("measurement_source_bindings")
        or document.get("protocol") != expected_protocol
        or document.get("candidate_count") != len(document.get("records", []))
    ):
        raise VerificationError("screening stage differs from its sealed batch artifact")
    bound = result.get("measurement_source_bindings")
    if not isinstance(bound, dict) or bound.get(digest) != binding:
        raise VerificationError("screening binding differs from its source stage")
    matches = [
        item
        for item in result.get("records", [])
        if isinstance(item, dict) and item.get("candidate_sha256") == digest
    ]
    if len(matches) != 1:
        raise VerificationError("screening source stage has no unique terminal record")
    return matches[0]


def _verify_binding_artifact(record, verified, run_root, label, required=True):
    if record is None:
        if required:
            raise VerificationError("{} artifact is missing".format(label))
        return None
    facts = _verify_artifact(record, run_root, label)
    ledger = verified.get(facts["path"])
    if ledger != facts:
        raise VerificationError("{} artifact is absent from the stage ledger".format(label))
    return facts


def _collect_screening_records(
    stages,
    identity,
    measurement_document,
    matrices,
    verified,
    run_root,
):
    bindings = measurement_document.get("actual_measurement_source_bindings")
    if not isinstance(bindings, dict):
        raise VerificationError("measurement source document omitted bindings")
    final = matrices["c4"]
    candidate_by_digest = {
        item["candidate_sha256"]: item for item in final["candidates"]
    }
    if set(bindings) != set(candidate_by_digest):
        raise VerificationError("measurement bindings do not cover final matrix exactly")
    base_digests = {item["candidate_sha256"] for item in matrices["base"]["candidates"]}
    c3_all = {item["candidate_sha256"] for item in matrices["c3"]["candidates"]}
    expected_source = {}
    for digest in candidate_by_digest:
        if digest in base_digests:
            expected_source[digest] = matrices["base"]["matrix_sha256"]
        elif digest in c3_all:
            expected_source[digest] = matrices["c3"]["matrix_sha256"]
        else:
            expected_source[digest] = matrices["c4"]["matrix_sha256"]
    records = []
    by_family_stage = {"base": [], "c3": [], "c4": []}
    search = identity["payload"]["search"]
    expected_protocol = {
        "frames": search["screen_frames"],
        "warmup": search["screen_warmup"],
        "trials": search["screen_trials"],
        "schedule": "round_robin",
        "seed": search["seed"],
    }
    document_cache = {}
    for digest, binding in sorted(bindings.items()):
        if not isinstance(binding, dict):
            raise VerificationError("screening binding is not an object")
        if binding.get("source_matrix_sha256") != expected_source[digest]:
            raise VerificationError("screening profile used the wrong staged matrix")
        profile_facts = _verify_binding_artifact(
            binding.get("profile"), verified, run_root, "screening profile"
        )
        profile = load_json(profile_facts["path"], "screening profile")
        candidate = candidate_by_digest[digest]
        provenance = profile.get("provenance")
        if (
            profile.get("selected_variant_id") != candidate["variant_id"]
            or profile.get("deployment") != {"enabled": False, "valid": False}
            or not isinstance(provenance, dict)
            or provenance.get("candidate_sha256") != digest
            or provenance.get("matrix_sha256") != expected_source[digest]
        ):
            raise VerificationError("screening profile provenance changed")
        report_facts = _verify_binding_artifact(
            binding.get("report"),
            verified,
            run_root,
            "screening benchmark report",
            required=False,
        )
        record = _record_for_binding(
            stages,
            digest,
            binding,
            verified,
            run_root,
            expected_protocol,
            document_cache,
        )
        expected_artifact_sha256 = (
            report_facts["sha256"]
            if report_facts is not None
            else profile_facts["sha256"]
        )
        if record.get("artifact_sha256") != expected_artifact_sha256:
            raise VerificationError("screening terminal record evidence hash changed")
        if record.get("status") == "succeeded":
            if report_facts is None:
                raise VerificationError("successful screening lacks bound benchmark evidence")
            benchmark = load_json(report_facts["path"], "screening benchmark")
            names = [
                item.get("name")
                for item in benchmark.get("candidates", [])
                if isinstance(item, dict)
            ]
            contract = benchmark.get("contract")
            schedule = benchmark.get("schedule")
            if (
                benchmark.get("passed") is not True
                or candidate["variant_id"] not in names
                or candidate["variant_id"] not in benchmark.get("summaries", {})
                or not isinstance(contract, dict)
                or contract.get("profile_frames") != expected_protocol["frames"]
                or contract.get("warmup_frames") != expected_protocol["warmup"]
                or not isinstance(schedule, dict)
                or schedule.get("strategy") != "round_robin"
                or schedule.get("trials_per_candidate")
                != expected_protocol["trials"]
            ):
                raise VerificationError("successful screening candidate was not measured")
            summary = benchmark["summaries"][candidate["variant_id"]]
            if record.get("score") != summary.get("median_throughput_fps"):
                raise VerificationError("screening score differs from benchmark summary")
        elif record.get("status") != "failed":
            raise VerificationError("screening record is not terminal")
        elif report_facts is not None:
            benchmark = load_json(report_facts["path"], "failed screening benchmark")
            names = [
                item.get("name")
                for item in benchmark.get("candidates", [])
                if isinstance(item, dict)
            ]
            if candidate["variant_id"] not in names:
                raise VerificationError("failed screening candidate was never scheduled")
        records.append(record)
        key = "base" if digest in base_digests else "c3" if digest in c3_all else "c4"
        by_family_stage[key].append(record)
    return records, by_family_stage, bindings


def _compare_matrix(label, rebuilt, stored):
    if rebuilt != stored:
        raise VerificationError("{} matrix replay differs from sealed matrix".format(label))
    return rebuilt["matrix_sha256"]


def _rebuild_matrices(identity, preflight, stored, staged_records):
    payload = identity["payload"]
    workload = payload.get("workload")
    search = payload.get("search")
    device = preflight.get("device")
    if not all(isinstance(item, dict) for item in (workload, search, device)):
        raise VerificationError("identity/preflight launch geometry is incomplete")
    width = workload.get("image_width")
    height = workload.get("image_height")
    rows = workload.get("head_rows")
    sm_count = device.get("sm_count")
    if any(type(value) is not int or value <= 0 for value in (width, height, rows, sm_count)):
        raise VerificationError("launch geometry contains invalid values")
    raster_tiles = ((width + 15) // 16) * ((height + 15) // 16)
    backend_blocks = ((rows + 15) // 16) * 2
    blocks = AUTOTUNE.derive_persistent_blocks(
        sm_count,
        raster_tiles,
        backend_blocks,
        current_persistent_blocks=search["current_persistent_blocks"],
        extra_values=search["extra_persistent_blocks"],
    )
    base = AUTOTUNE.build_phase31_base_matrix(
        blocks,
        sm_count=sm_count,
        raster_tile_count=raster_tiles,
        backend_logical_blocks=backend_blocks,
        whole_head_logical_blocks=rows,
        packed_persistent_blocks=list(blocks) + search["packed_persistent_blocks"],
        whole_head_persistent_blocks=list(blocks) + search["whole_head_persistent_blocks"],
    )
    _compare_matrix("base", base, stored["base"])
    c3 = AUTOTUNE.extend_phase31_with_c3(
        base, staged_records["base"], search["c3_top_k"]
    )
    _compare_matrix("C3", c3, stored["c3"])
    c4 = AUTOTUNE.extend_phase31_with_c4(
        c3,
        staged_records["base"] + staged_records["c3"],
        search["c4_top_k_per_family"],
    )
    _compare_matrix("C4", c4, stored["c4"])
    _validate_exhaustive_c2(c4)
    return {
        "base_matrix_sha256": base["matrix_sha256"],
        "c3_matrix_sha256": c3["matrix_sha256"],
        "final_matrix_sha256": c4["matrix_sha256"],
        "final_candidate_count": len(c4["candidates"]),
        "c2_candidate_count": _candidate_count(c4, "c2"),
        "family_candidate_counts": {
            family: _candidate_count(c4, family) for family in SEARCH_FAMILIES
        },
    }


def _verify_ranking(
    stages,
    identity,
    final_matrix,
    records,
    bindings,
    measurement_document,
    verified,
    run_root,
):
    search = identity["payload"]["search"]
    portable_input = {
        "protocol": {
            "frames": search["screen_frames"],
            "warmup": search["screen_warmup"],
            "trials": search["screen_trials"],
            "schedule": "round_robin",
            "seed": search["seed"],
        },
        "measurement_source_bindings": _portable_bindings(bindings),
    }
    digest = sha256_json(portable_input, "tacker-phase31-full-screening-input-v1")
    if (
        measurement_document.get("portable_hash_input") != portable_input
        or measurement_document.get("screening_input_sha256") != digest
    ):
        raise VerificationError("screening input hash replay differs")
    rebuilt = AUTOTUNE.build_screening_ranking(
        final_matrix,
        records,
        screening_input_sha256=digest,
        maximize=True,
        require_terminal=True,
    )
    stage = _require_stage(stages, "screening-ranking")
    result = stage["result"]
    ranking_path = _verified_path(
        result.get("ranking_path", ""), verified, run_root, "screening ranking"
    )
    stored = load_json(ranking_path, "screening ranking")
    if rebuilt != stored or result.get("ranking") != stored:
        raise VerificationError("screening ranking replay differs")
    if (
        _snapshot_store().snapshot(
            ranking_path, "screening ranking", capture_bytes=True
        )["bytes"]
        != _runner_pretty_bytes(stored)
    ):
        raise VerificationError("screening ranking bytes are not canonical")
    return rebuilt, {
        "screening_input_sha256": digest,
        "ranking_sha256": rebuilt["ranking_sha256"],
        "candidate_count": rebuilt["candidate_count"],
        "terminal_count": rebuilt["terminal_count"],
        "successful_count": rebuilt["successful_count"],
        "failed_count": rebuilt["failed_count"],
    }


def _formal_wrapper(matrix, ranking, top_k):
    generated = AUTOTUNE.build_formal_candidate_set(matrix, ranking, top_k)
    payload = {
        "schema_version": 1,
        "kind": FORMAL_SET_KIND,
        "matrix_sha256": matrix["matrix_sha256"],
        "screening_ranking_sha256": ranking["ranking_sha256"],
        "baseline_candidates": list(BASELINES),
        "generated_candidate_set": generated,
    }
    document = dict(payload)
    document["formal_set_sha256"] = sha256_json(
        payload, "tacker-phase31-formal-set-v1"
    )
    return document


def _verify_formal_set(stages, identity, matrix, ranking, verified, run_root):
    expected = _formal_wrapper(
        matrix, ranking, identity["payload"]["search"]["formal_top_k"]
    )
    stage = _require_stage(stages, "formal-candidate-set")
    result = stage["result"]
    path = _verified_path(result.get("path", ""), verified, run_root, "formal set")
    stored = load_json(path, "formal candidate set")
    if stored != expected:
        raise VerificationError("formal candidate set replay differs")
    if (
        result.get("formal_set") != expected["generated_candidate_set"]
        or result.get("formal_set_sha256") != expected["formal_set_sha256"]
        or result.get("candidate_set_sha256")
        != expected["generated_candidate_set"]["candidate_set_sha256"]
    ):
        raise VerificationError("formal candidate set stage summary changed")
    return expected, {
        "formal_set_sha256": expected["formal_set_sha256"],
        "candidate_set_sha256": expected["generated_candidate_set"]["candidate_set_sha256"],
        "generated_candidate_count": len(expected["generated_candidate_set"]["candidates"]),
        "baseline_candidates": list(BASELINES),
    }


def _profile_manifest(stage, matrix, verified, run_root):
    result = stage["result"]
    manifest_path = _verified_path(
        result.get("manifest", ""), verified, run_root, "final profile manifest"
    )
    manifest = load_json(manifest_path, "final profile manifest")
    if (
        manifest.get("kind") != "tacker_autotune_qualification_profiles"
        or manifest.get("matrix_sha256") != matrix["matrix_sha256"]
        or len(manifest.get("profiles", [])) != len(matrix["candidates"])
    ):
        raise VerificationError("final profile manifest contract changed")
    manifest_payload = dict(manifest)
    observed_manifest_sha256 = manifest_payload.pop("manifest_sha256", None)
    if observed_manifest_sha256 != AUTOTUNE.canonical_sha256(
        manifest_payload, "tacker-autotune-profile-manifest-v1"
    ):
        raise VerificationError("final profile manifest hash changed")
    by_digest = {}
    for item in manifest["profiles"]:
        if not isinstance(item, dict):
            raise VerificationError("invalid profile manifest entry")
        path = _verified_path(item.get("path", ""), verified, run_root, "final profile")
        facts = _artifact_facts(path)
        if (
            item.get("artifact_sha256") != facts["sha256"]
            or item.get("artifact_size") != facts["size_bytes"]
            or item.get("path") != facts["path"]
        ):
            raise VerificationError("final profile manifest artifact changed")
        digest = item.get("candidate_sha256")
        if digest in by_digest:
            raise VerificationError("duplicate final profile candidate")
        by_digest[digest] = {"entry": item, "path": path, "facts": facts}
    expected = {item["candidate_sha256"] for item in matrix["candidates"]}
    if set(by_digest) != expected:
        raise VerificationError("final profiles do not cover matrix exactly")
    return manifest, by_digest


def _default_materialize_profiles(
    matrix_path,
    resources_path,
    template_path,
    output_dir,
    python_executable,
    runner,
    autotune_path=None,
):
    script = AUTOTUNE_PATH if autotune_path is None else Path(autotune_path)
    command = [
        str(Path(python_executable).expanduser().resolve()),
        str(script),
        "profiles",
        "--matrix",
        str(matrix_path),
        "--resources",
        str(resources_path),
        "--template",
        str(template_path),
        "--output-dir",
        str(output_dir),
    ]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    completed = runner(
        command,
        cwd=str(PROJECT_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    if int(getattr(completed, "returncode", -1)) != 0:
        raise VerificationError(
            "profile replay failed: {}".format(
                str(getattr(completed, "stderr", ""))[-1000:]
            )
        )


def _verify_profiles(
    stages,
    identity,
    matrix,
    matrix_path,
    verified,
    run_root,
    python_executable,
    runner,
    profile_materializer,
):
    final_stage = _require_stage(stages, "profiles-final")
    manifest, originals = _profile_manifest(final_stage, matrix, verified, run_root)
    preflight = _require_stage(stages, "preflight")["result"]
    resources = _verified_path(
        preflight.get("resources", ""), verified, run_root, "preflight resources"
    )
    template_record = identity["payload"]["files"].get("template_profile")
    template = Path(_verify_identity_record(template_record, "template profile"))
    python_record = identity["payload"]["files"].get("python_executable")
    sealed_python = Path(_verify_identity_record(python_record, "Python executable"))
    # Command-line Python launchers are commonly symlinks (venv/conda).  Bind the
    # replay to the identity-sealed real executable, then watch that regular file.
    try:
        requested_python_real = Path(python_executable).expanduser().resolve(strict=True)
    except OSError as error:
        raise VerificationError("profile replay Python is missing: {}".format(error))
    requested_python = _lexical_path(
        requested_python_real, "profile replay Python"
    )
    if requested_python != sealed_python:
        raise VerificationError("profile replay Python differs from run identity")
    watched = [
        (sealed_python, "profile replay Python"),
        (AUTOTUNE_PATH, "profile replay autotune source"),
    ]
    for name, record in sorted(identity["payload"]["runtime_sources"].items()):
        watched.append(
            (
                Path(_verify_identity_record(record, "runtime source {}".format(name))),
                "profile replay runtime source {}".format(name),
            )
        )
    renderer_init = PROJECT_ROOT / "gaussian_renderer" / "__init__.py"
    _snapshot_store().snapshot(renderer_init, "gaussian renderer package source")
    watched.append((renderer_init, "gaussian renderer package source"))
    for watched_path, watched_label in watched:
        _snapshot_store().revalidate(watched_path, watched_label)
    with tempfile.TemporaryDirectory(prefix="tacker-phase31-postflight-") as temporary:
        temporary_root = Path(temporary)
        input_root = temporary_root / "inputs"
        sealed_root = temporary_root / "sealed-project"
        copied_matrix = input_root / "matrix.json"
        copied_resources = input_root / "resources.json"
        copied_template = input_root / "template.json"
        copied_autotune = sealed_root / "scripts" / "tacker_autotune.py"
        _snapshot_store().copy_to(matrix_path, copied_matrix, "final matrix")
        _snapshot_store().copy_to(resources, copied_resources, "preflight resources")
        _snapshot_store().copy_to(template, copied_template, "template profile")
        _snapshot_store().copy_to(
            AUTOTUNE_PATH, copied_autotune, "sealed autotune source"
        )
        output_dir = Path(temporary) / "profiles"
        try:
            if profile_materializer is None:
                _default_materialize_profiles(
                    copied_matrix,
                    copied_resources,
                    copied_template,
                    output_dir,
                    sealed_python,
                    runner,
                    autotune_path=copied_autotune,
                )
            else:
                profile_materializer(
                    copied_matrix, copied_resources, copied_template, output_dir
                )
        finally:
            for watched_path, watched_label in watched:
                _snapshot_store().revalidate(watched_path, watched_label)
        replay_manifest = load_json(
            output_dir / "qualification_profiles.json", "replayed profile manifest"
        )
        replay_entries = {
            item["candidate_sha256"]: item
            for item in replay_manifest.get("profiles", [])
            if isinstance(item, dict)
        }
        mismatches = []
        replay_hashes = {}
        for candidate in matrix["candidates"]:
            digest = candidate["candidate_sha256"]
            replay_path = output_dir / "{}.json".format(candidate["variant_id"])
            if not replay_path.is_file():
                mismatches.append({"candidate_sha256": digest, "reason": "missing"})
                continue
            replay_hash = sha256_file(replay_path)
            original_hash = originals[digest]["facts"]["sha256"]
            replay_hashes[digest] = replay_hash
            entry = replay_entries.get(digest)
            if (
                replay_hash != original_hash
                or not isinstance(entry, dict)
                or entry.get("artifact_sha256") != replay_hash
            ):
                mismatches.append(
                    {
                        "candidate_sha256": digest,
                        "expected_sha256": original_hash,
                        "replayed_sha256": replay_hash,
                    }
                )
        if set(replay_entries) != set(originals):
            raise VerificationError("replayed profile manifest coverage changed")
        if mismatches:
            raise VerificationError(
                "replayed qualification profile hashes differ for {} candidates".format(
                    len(mismatches)
                )
            )
    profile_set_sha256 = sha256_json(
        {digest: replay_hashes[digest] for digest in sorted(replay_hashes)},
        "tacker-phase31-profile-set-v1",
    )
    return {
        "matrix_sha256": matrix["matrix_sha256"],
        "profile_count": len(originals),
        "profile_hash_mismatch_count": 0,
        "profile_set_sha256": profile_set_sha256,
        "source_manifest_sha256": manifest["manifest_sha256"],
    }


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and value not in (float("inf"), float("-inf"))
    )


def _validate_per_view(records, label):
    if not isinstance(records, list) or len(records) != 50:
        raise VerificationError("{} does not contain exactly 50 views".format(label))
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("batch_index") != index:
            raise VerificationError("{} view indices are not exactly 0..49".format(label))
        if any(not _finite_number(record.get(key)) for key in ("psnr_db", "ssim", "lpips")):
            raise VerificationError("{} contains a non-finite quality metric".format(label))


def _validate_candidate_quality(report, identity, profile_path):
    workload = identity["payload"]["workload"]
    report_workload = report.get("workload")
    qualification = report.get("qualification")
    modes = report.get("modes")
    gates = report.get("gates")
    tacker = modes.get("tacker") if isinstance(modes, dict) else None
    tacker_gate = next(
        (
            item
            for item in gates
            if isinstance(item, dict) and item.get("mode") == "tacker"
        ),
        None,
    ) if isinstance(gates, list) else None
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "4dgaussians_tacker_quality_validation"
        or report.get("passed") is not True
        or not isinstance(report_workload, dict)
        or report_workload.get("scene") != workload["name"]
        or report_workload.get("iteration") != workload["iteration"]
        or report_workload.get("split") != workload["split"]
        or report_workload.get("frames") != 50
        or report_workload.get("view_indices") != list(range(50))
        or report_workload.get("resolution")
        != [workload["image_width"], workload["image_height"]]
        or report_workload.get("gaussian_count") != workload["gaussian_count"]
        or not isinstance(qualification, dict)
        or qualification.get("enabled") is not True
        or qualification.get("admission_claimed") is not False
        or qualification.get("profile_override") != str(Path(profile_path).resolve())
        or not isinstance(modes, dict)
        or set(modes) != {"serial", "tacker"}
        or not isinstance(tacker, dict)
        or tacker.get("actual_mode") != "tacker"
        or tacker.get("fallback_reason") is not None
        or tacker.get("qualification_executed") is not True
        or not isinstance(tacker_gate, dict)
        or tacker_gate.get("passed") is not True
    ):
        raise VerificationError("valid finalist 50-view quality contract changed")
    _validate_per_view(tacker.get("per_view"), "valid finalist quality")


def _validate_candidate_leaf(report, identity, candidate, profile_facts, matrix_facts):
    workload = identity["payload"]["workload"]
    report_workload = report.get("workload")
    binding = report.get("profile_binding")
    parameters = report.get("parameters")
    if (
        report.get("schema_version") != 2
        or report.get("kind") != "4dgaussians_tacker_leaf_profile_report"
        or report.get("passed") is not True
        or not isinstance(report_workload, dict)
        or report_workload.get("scene") != workload["name"]
        or report_workload.get("iteration") != workload["iteration"]
        or report_workload.get("split") != workload["split"]
        or report_workload.get("resolution")
        != [workload["image_width"], workload["image_height"]]
        or report_workload.get("gaussian_count") != workload["gaussian_count"]
        or report.get("variant_id") != candidate["variant_id"]
        or not isinstance(binding, dict)
        or binding.get("candidate_sha256") != candidate["candidate_sha256"]
        or binding.get("candidate_matrix_sha256") != matrix_facts["matrix_sha256"]
        or binding.get("candidate_matrix_file_sha256") != matrix_facts["file_sha256"]
        or binding.get("profile_file_sha256") != profile_facts["sha256"]
        or not isinstance(parameters, dict)
        or parameters.get("candidate_profile") != profile_facts["path"]
        or parameters.get("qualification_profile") is not True
        or parameters.get("used_as_deployment") is not False
        or report.get("measurement_outputs_written") is not True
    ):
        raise VerificationError("valid finalist leaf evidence binding changed")


def _verify_qualification_result(
    stages,
    identity,
    matrix,
    matrix_path,
    candidate,
    expected_result,
    originals,
    verified,
    run_root,
):
    digest = candidate["candidate_sha256"]
    stage = _require_stage(stages, "qualify-{}".format(digest[:20]))
    wrapper = stage["result"]
    output_path = _verified_path(
        wrapper.get("path", ""), verified, run_root, "candidate qualification"
    )
    result = load_json(output_path, "candidate qualification")
    if wrapper.get("result") != result or expected_result != result:
        raise VerificationError("qualification stage/result artifact differs")
    if (
        result.get("schema_version") != 1
        or result.get("kind") != "tacker_phase31_candidate_qualification"
        or result.get("matrix_sha256") != matrix["matrix_sha256"]
        or result.get("candidate_sha256") != digest
        or result.get("variant_id") != candidate["variant_id"]
        or result.get("search_family") != candidate["search_family"]
        or type(result.get("valid")) is not bool
        or type(result.get("leaf_passed")) is not bool
        or type(result.get("quality_50_view_passed")) is not bool
        or not isinstance(result.get("errors"), list)
    ):
        raise VerificationError("candidate qualification schema changed")
    profile_facts = _verify_binding_artifact(
        result.get("profile"), verified, run_root, "qualification profile"
    )
    if profile_facts != originals[digest]["facts"]:
        raise VerificationError("qualification used a non-final profile artifact")
    leaf_facts = _verify_binding_artifact(
        result.get("leaf_report"),
        verified,
        run_root,
        "candidate leaf report",
        required=False,
    )
    quality_facts = _verify_binding_artifact(
        result.get("quality_report"),
        verified,
        run_root,
        "candidate quality report",
        required=False,
    )
    leaf_report = load_json(leaf_facts["path"], "candidate leaf report") if leaf_facts else None
    quality_report = load_json(quality_facts["path"], "candidate quality report") if quality_facts else None
    expected_valid = bool(result["leaf_passed"] and result["quality_50_view_passed"])
    if (
        result["valid"] is not expected_valid
        or result.get("quality_actual_tacker_without_fallback") is not result["quality_50_view_passed"]
        or result.get("quality_unique_requested_views") is not result["quality_50_view_passed"]
        or result.get("quality_exact_per_view_records") is not result["quality_50_view_passed"]
    ):
        raise VerificationError("candidate qualification gate flags disagree")
    if result["valid"]:
        if leaf_report is None or quality_report is None:
            raise VerificationError("valid finalist omitted leaf/quality evidence")
        if leaf_report.get("passed") is not True or quality_report.get("passed") is not True:
            raise VerificationError("valid finalist report did not pass")
        _validate_candidate_leaf(
            leaf_report,
            identity,
            candidate,
            profile_facts,
            {
                "matrix_sha256": matrix["matrix_sha256"],
                "file_sha256": sha256_file(matrix_path),
            },
        )
        _validate_candidate_quality(quality_report, identity, profile_facts["path"])
    elif not result["errors"]:
        raise VerificationError("invalid qualification omitted its explicit error")
    return result, _artifact_facts(output_path)


def _verify_qualification_plan(
    stages,
    identity,
    matrix,
    matrix_path,
    ranking,
    formal_wrapper,
    originals,
    verified,
    run_root,
):
    stage = _require_stage(stages, "qualification-plan")
    wrapper = stage["result"]
    path = _verified_path(
        wrapper.get("path", ""), verified, run_root, "qualification plan"
    )
    document = load_json(path, "qualification plan")
    if wrapper.get("document") != document:
        raise VerificationError("qualification plan differs from stage result")
    attempted = document.get("attempted")
    if not isinstance(attempted, list) or not attempted:
        raise VerificationError("qualification plan has no attempts")
    matrix_by_digest = {
        item["candidate_sha256"]: item for item in matrix["candidates"]
    }
    seen = set()
    normalized = []
    evidence = {}
    for index, item in enumerate(attempted):
        if not isinstance(item, dict) or not isinstance(item.get("candidate"), dict):
            raise VerificationError("qualification attempt is invalid")
        candidate = item["candidate"]
        digest = candidate.get("candidate_sha256")
        if digest in seen or matrix_by_digest.get(digest) != candidate:
            raise VerificationError("qualification attempt is duplicate or non-matrix")
        seen.add(digest)
        result, facts = _verify_qualification_result(
            stages,
            identity,
            matrix,
            matrix_path,
            candidate,
            item.get("result"),
            originals,
            verified,
            run_root,
        )
        normalized.append(
            {"candidate": candidate, "reason": item.get("reason"), "result": result}
        )
        evidence[digest] = facts
    initial = [
        item["candidate"]
        for item in formal_wrapper["generated_candidate_set"]["candidates"]
    ]
    if len(normalized) < len(initial) or any(
        normalized[index]["candidate"] != candidate
        or normalized[index]["reason"] != "formal_set"
        for index, candidate in enumerate(initial)
    ):
        raise VerificationError("qualification plan does not start with the formal set")
    by_digest = {
        item["candidate"]["candidate_sha256"]: item for item in normalized
    }
    used = {item["candidate_sha256"] for item in initial}
    cursor = len(initial)
    expected_chains = []
    for candidate in initial:
        digest = candidate["candidate_sha256"]
        if by_digest[digest]["result"]["valid"]:
            continue
        alternatives = AUTOTUNE.family_local_backfill_candidates(
            matrix,
            ranking,
            candidate["search_family"],
            count=len(matrix["candidates"]),
            excluded_candidate_sha256s=used,
        )
        chain = []
        replacement = None
        for alternative in alternatives:
            expected = alternative["candidate"]
            if cursor >= len(normalized):
                raise VerificationError("same-family backfill evidence is incomplete")
            item = normalized[cursor]
            if (
                item["candidate"] != expected
                or item["reason"]
                != "same_family_backfill_for:{}".format(digest)
            ):
                raise VerificationError("same-family backfill order changed")
            cursor += 1
            used.add(expected["candidate_sha256"])
            chain.append(expected["candidate_sha256"])
            if item["result"]["valid"]:
                replacement = expected["candidate_sha256"]
                break
        if replacement is None:
            raise VerificationError("invalid finalist has no valid same-family backfill")
        expected_chains.append(
            {
                "invalid_candidate_sha256": digest,
                "search_family": candidate["search_family"],
                "attempted": chain,
                "replacement_candidate_sha256": replacement,
            }
        )
    if cursor != len(normalized) or document.get("same_family_backfill") != expected_chains:
        raise VerificationError("qualification backfill chain changed")
    valid = []
    valid_seen = set()
    for item in normalized:
        digest = item["candidate"]["candidate_sha256"]
        if item["result"]["valid"] and digest not in valid_seen:
            valid.append(item["candidate"])
            valid_seen.add(digest)
    family_coverage = {
        family: [
            item["candidate_sha256"]
            for item in valid
            if item["search_family"] == family
        ]
        for family in SEARCH_FAMILIES
    }
    if any(not values for values in family_coverage.values()):
        raise VerificationError("valid finalists do not cover C0--C4")
    hash_payload = {
        "schema_version": 1,
        "kind": "tacker_phase31_qualification_plan",
        "matrix_sha256": matrix["matrix_sha256"],
        "screening_ranking_sha256": ranking["ranking_sha256"],
        "formal_candidate_set_sha256": formal_wrapper["formal_set_sha256"],
        "autotune_candidate_set_sha256": formal_wrapper[
            "generated_candidate_set"
        ]["candidate_set_sha256"],
        "attempted": [
            {
                "candidate_sha256": item["candidate"]["candidate_sha256"],
                "search_family": item["candidate"]["search_family"],
                "reason": item["reason"],
                "valid": item["result"]["valid"],
                "leaf_passed": item["result"]["leaf_passed"],
                "quality_50_view_passed": item["result"]["quality_50_view_passed"],
                "qualification_evidence": _portable_artifact(
                    evidence[item["candidate"]["candidate_sha256"]]
                ),
            }
            for item in normalized
        ],
        "same_family_backfill": expected_chains,
        "valid_finalist_sha256": [item["candidate_sha256"] for item in valid],
        "valid_family_coverage": family_coverage,
    }
    digest = sha256_json(hash_payload, "tacker-phase31-qualification-plan-v1")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "tacker_phase31_qualification_plan"
        or document.get("matrix_sha256") != matrix["matrix_sha256"]
        or document.get("screening_ranking_sha256") != ranking["ranking_sha256"]
        or document.get("formal_candidate_set_sha256")
        != formal_wrapper["formal_set_sha256"]
        or document.get("autotune_candidate_set_sha256")
        != formal_wrapper["generated_candidate_set"]["candidate_set_sha256"]
        or document.get("valid_finalists") != valid
        or document.get("valid_family_coverage") != family_coverage
        or document.get("portable_hash_input") != hash_payload
        or document.get("qualification_plan_sha256") != digest
    ):
        raise VerificationError("qualification plan portable replay differs")
    return document, {
        "qualification_plan_sha256": digest,
        "attempted_candidate_count": len(normalized),
        "valid_finalist_count": len(valid),
        "backfill_chain_count": len(expected_chains),
        "valid_family_coverage": family_coverage,
    }


def _verify_baseline_quality(stages, identity, verified, run_root):
    stage = _require_stage(stages, "baseline-quality")
    result = stage["result"]
    quality_path = _verified_path(
        result.get("quality_report", ""), verified, run_root, "baseline quality"
    )
    correctness_path = _verified_path(
        result.get("correctness_path", ""),
        verified,
        run_root,
        "baseline correctness",
    )
    report = load_json(quality_path, "baseline quality")
    correctness = load_json(correctness_path, "baseline correctness")
    workload = identity["payload"]["workload"]
    report_workload = report.get("workload")
    modes = report.get("modes")
    gates = report.get("gates")
    qualification = report.get("qualification")
    expected_profile = str(
        Path(identity["payload"]["files"]["current_tacker_profile"]["path"]).resolve()
    )
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "4dgaussians_tacker_quality_validation"
        or type(report.get("passed")) is not bool
        or not isinstance(report_workload, dict)
        or report_workload.get("scene") != workload["name"]
        or report_workload.get("iteration") != workload["iteration"]
        or report_workload.get("split") != workload["split"]
        or report_workload.get("frames") != 50
        or report_workload.get("view_indices") != list(range(50))
        or report_workload.get("resolution")
        != [workload["image_width"], workload["image_height"]]
        or report_workload.get("gaussian_count") != workload["gaussian_count"]
        or report_workload.get("model_path")
        != str(Path(identity["payload"]["paths"]["model"]).resolve())
        or report_workload.get("source_path")
        != str(Path(identity["payload"]["paths"]["source"]).resolve())
        or report.get("tacker_profile") != expected_profile
        or not isinstance(qualification, dict)
        or qualification.get("enabled") is not False
        or qualification.get("profile_override") is not None
        or qualification.get("admission_claimed") is not report["passed"]
        or not isinstance(modes, dict)
        or set(modes) != {"serial", "two_stream", "tacker"}
        or not isinstance(gates, list)
        or not isinstance(report.get("errors"), list)
    ):
        raise VerificationError("baseline 50-view quality contract changed")
    gate_by_mode = {}
    for gate in gates:
        if not isinstance(gate, dict) or gate.get("mode") not in ("two_stream", "tacker"):
            raise VerificationError("baseline quality gate changed")
        if gate["mode"] in gate_by_mode or any(
            type(gate.get(key)) is not bool
            for key in (
                "actual_mode_passed",
                "qualification_passed",
                "quality_passed",
                "passed",
            )
        ):
            raise VerificationError("baseline quality gates are invalid")
        gate_by_mode[gate["mode"]] = gate
    if set(gate_by_mode) != {"two_stream", "tacker"}:
        raise VerificationError("baseline quality omitted a gate")
    if report["passed"] is not all(item["passed"] for item in gate_by_mode.values()):
        raise VerificationError("baseline quality passed flag disagrees with gates")
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
            or details.get("qualification_requested") is not False
            or details.get("qualification_executed") is not False
        ):
            raise VerificationError("baseline mode binding changed")
        _validate_per_view(details.get("per_view"), "baseline {}".format(mode))
        if mode == "serial":
            valid = bool(
                details.get("actual_mode") == "serial"
                and details.get("fallback_reason") is None
            )
        else:
            gate = gate_by_mode[mode]
            if gate.get("actual_mode") != details.get("actual_mode"):
                raise VerificationError("baseline gate actual mode changed")
            valid = bool(
                gate.get("passed") is True
                and details.get("actual_mode") == mode
                and details.get("fallback_reason") is None
            )
        status[external] = {
            "valid": valid,
            "actual_mode": details.get("actual_mode"),
            "fallback_reason": details.get("fallback_reason"),
            "quality_report_sha256": sha256_file(quality_path),
        }
    if not status["serial"]["valid"] or not status["two_stream"]["valid"]:
        raise VerificationError("serial/two_stream safety baseline is invalid")
    if (
        correctness != status
        or result.get("correctness") != status
        or result.get("all_valid") is not all(item["valid"] for item in status.values())
    ):
        raise VerificationError("baseline correctness summary changed")
    return status, {
        "quality_report_sha256": sha256_file(quality_path),
        "correctness_sha256": sha256_file(correctness_path),
        "current_tacker_valid": status["current_tacker"]["valid"],
    }


def _formal_candidate_entries(formal, finalist_names, originals, identity):
    candidates = formal.get("candidates")
    if not isinstance(candidates, list):
        raise VerificationError("formal FPS candidates are missing")
    names = [item.get("name") for item in candidates if isinstance(item, dict)]
    if names != list(BASELINES) + finalist_names:
        raise VerificationError("formal FPS candidate order changed")
    for entry in candidates:
        if set(entry) != {
            "name",
            "execution_mode",
            "profile_path",
            "qualification_mode",
            "profile_file_sha256",
        }:
            raise VerificationError("formal FPS candidate fields changed")
        name = entry["name"]
        if name == "serial":
            expected_mode = "serial"
        elif name == "two_stream":
            expected_mode = "two_stream"
        else:
            expected_mode = "tacker"
        if entry.get("execution_mode") != expected_mode:
            raise VerificationError("formal FPS execution mode changed")
        if name in finalist_names:
            facts = next(
                value["facts"]
                for value in originals.values()
                if value["entry"]["variant_id"] == name
            )
            if (
                entry.get("profile_path") != facts["path"]
                or entry.get("profile_file_sha256") != facts["sha256"]
                or entry.get("qualification_mode") is not True
            ):
                raise VerificationError("formal finalist profile binding changed")
        elif name == "current_tacker":
            current = identity["payload"]["files"]["current_tacker_profile"]
            if (
                entry.get("profile_path") != current["path"]
                or entry.get("profile_file_sha256") != current["sha256"]
                or bool(entry.get("qualification_mode"))
            ):
                raise VerificationError("formal incumbent profile binding changed")
        elif (
            entry.get("profile_path") is not None
            or entry.get("profile_file_sha256") is not None
            or bool(entry.get("qualification_mode"))
        ):
            raise VerificationError("formal non-Tacker baseline has a profile")
    return candidates


def _path_hash(record):
    if not isinstance(record, dict):
        return None
    return {"path": record.get("path"), "sha256": record.get("sha256")}


def _verify_formal_resume_identity(
    formal,
    candidates,
    expected_schedule,
    identity,
    formal_path,
    correctness_path,
    verified,
    run_root,
):
    resume = formal.get("resume_identity")
    if not isinstance(resume, dict) or set(resume) != {"sha256", "payload"}:
        raise VerificationError("formal resume identity is invalid")
    payload = resume.get("payload")
    if not isinstance(payload, dict):
        raise VerificationError("formal resume identity payload is invalid")
    try:
        digest = BENCHMARK._sha256_json(payload)
    except Exception as error:
        raise VerificationError("cannot hash formal resume identity: {}".format(error))
    if resume.get("sha256") != digest:
        raise VerificationError("formal resume identity hash changed")
    schedule = formal["schedule"]
    bootstrap = formal["bootstrap"]
    expected_scalars = {
        "schema_version": 1,
        "report_kind": "4dgaussians_tacker_fps_benchmark",
        "driver_sha256": identity["payload"]["scripts"]["benchmark"]["sha256"],
        "contract": formal["contract"],
        "candidates": candidates,
        "correctness_qualifications": formal["correctness_qualifications"],
        "candidate_selection_metadata": formal["candidate_selection_metadata"],
        "schedule": expected_schedule,
        "schedule_strategy": "abba",
        "schedule_seed": schedule["seed"],
        "trials": 10,
        "bootstrap_resamples": bootstrap["resamples"],
        "incumbent_name": "current_tacker",
        "promotion_min_ratio": float(BENCHMARK.DEFAULT_PROMOTION_MIN_RATIO),
        "equivalence_fraction": float(BENCHMARK.DEFAULT_EQUIVALENCE_FRACTION),
        "project_root": str(PROJECT_ROOT),
        "profile_args": [],
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise VerificationError("formal resume identity {} changed".format(key))
    phase = identity["payload"]
    if payload.get("profile_render") != _path_hash(
        phase["runtime_sources"].get("profile_render")
    ):
        raise VerificationError("formal profile_render provenance changed")
    if payload.get("python_executable") != _path_hash(
        phase["files"].get("python_executable")
    ):
        raise VerificationError("formal Python executable identity changed")
    if payload.get("configs") != _path_hash(phase["files"].get("config")):
        raise VerificationError("formal config identity changed")
    expected_chain = [_path_hash(item) for item in phase["configuration_chain"]]
    observed_chain = payload.get("configs_chain")
    if not isinstance(observed_chain, list) or [
        _path_hash(item) for item in observed_chain
    ] != expected_chain:
        raise VerificationError("formal config chain identity changed")
    phase_workload = {
        name: _path_hash(record)
        for name, record in phase["workload_files"].items()
    }
    if payload.get("workload_files") != phase_workload:
        raise VerificationError("formal workload file identity changed")
    selection_inputs = payload.get("selection_inputs")
    expected_selection_inputs = {
        "correctness_json": {
            "path": str(correctness_path),
            "sha256": sha256_file(correctness_path),
        }
    }
    if (
        selection_inputs != expected_selection_inputs
        or formal.get("selection_inputs") != expected_selection_inputs
    ):
        raise VerificationError("formal selection input identity changed")
    artifacts = formal.get("artifacts")
    expected_session = run_root / "sessions" / "formal" / "runs" / "phase31-formal"
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("session_dir") != str(expected_session)
        or artifacts.get("driver_sha256")
        != phase["scripts"]["benchmark"]["sha256"]
        or artifacts.get("profile_render_sha256")
        != phase["runtime_sources"]["profile_render"]["sha256"]
        or artifacts.get("report_path") != str(formal_path)
    ):
        raise VerificationError("formal report artifact provenance changed")
    checkpoint_path = _verified_path(
        artifacts.get("checkpoint_path", ""),
        verified,
        run_root,
        "formal benchmark checkpoint",
    )
    if checkpoint_path != expected_session / BENCHMARK.CHECKPOINT_FILE_NAME:
        raise VerificationError("formal checkpoint path changed")
    sources = payload.get("required_provenance_sources")
    if not isinstance(sources, dict) or set(sources) != set(
        BENCHMARK.REQUIRED_PROVENANCE_SOURCE_FILES
    ):
        raise VerificationError("formal required provenance sources changed")
    phase_source_map = {
        "profile_render.py": phase["runtime_sources"].get("profile_render"),
        "configs": phase["files"].get("config"),
        "gaussian_renderer/tacker_pipeline.py": phase["runtime_sources"].get(
            "tacker_pipeline"
        ),
        "diff_gaussian_rasterization/__init__.py": phase["runtime_sources"].get(
            "raster_python_binding"
        ),
    }
    for name, record in sources.items():
        if name == "diff_gaussian_rasterization._C":
            if not isinstance(record, list):
                raise VerificationError("formal raster binary provenance changed")
            for index, item in enumerate(record):
                _verify_identity_record(
                    item, "formal provenance {}[{}]".format(name, index)
                )
        else:
            if record is None:
                raise VerificationError("formal provenance source {} is missing".format(name))
            _verify_identity_record(record, "formal provenance {}".format(name))
            phase_record = phase_source_map.get(name)
            if phase_record is not None and _path_hash(record) != _path_hash(phase_record):
                raise VerificationError(
                    "formal provenance source {} differs from run identity".format(name)
                )
    return sources


def _verify_stable_provenance(stable, identity, required_sources):
    if not isinstance(stable, dict):
        raise VerificationError("formal stable provenance is missing")
    source_files = stable.get("source_files")
    environment = stable.get("environment")
    if not isinstance(source_files, dict) or set(source_files) != set(
        BENCHMARK.REQUIRED_PROVENANCE_SOURCE_FILES
    ):
        raise VerificationError("formal source-file provenance changed")
    if (
        not isinstance(environment, dict)
        or environment.get("gpu_name") != "NVIDIA RTX A6000"
    ):
        raise VerificationError("formal stable provenance is not the A6000")
    for name, record in required_sources.items():
        if name == "diff_gaussian_rasterization._C":
            allowed = {item.get("sha256") for item in record if isinstance(item, dict)}
            if source_files[name] not in allowed:
                raise VerificationError("formal loaded raster binary is not source-sealed")
        elif source_files[name] != record.get("sha256"):
            raise VerificationError("formal source hash {} changed".format(name))
    configured_commit = identity["payload"].get("environment", {}).get(
        "FOURDGS_SOURCE_COMMIT"
    )
    if configured_commit is not None and stable.get("repository_commit") != configured_commit:
        raise VerificationError("formal repository commit differs from run identity")


def _verify_formal_execution(
    formal,
    candidates,
    finalist_names,
    result,
    identity,
    formal_path,
    correctness_path,
    verified,
    run_root,
):
    runs = formal.get("runs")
    if not isinstance(runs, list):
        raise VerificationError("formal report omitted raw run records")
    expected_root = run_root / "sessions" / "formal" / "runs" / "phase31-formal"
    candidate_by_name = {item["name"]: item for item in candidates}
    names = [item["name"] for item in candidates]
    qualifications = formal["correctness_qualifications"]
    eligible = [name for name in names if qualifications[name]["valid"] is True]
    schedule = formal["schedule"]
    bootstrap = formal["bootstrap"]
    try:
        expected_schedule = BENCHMARK.build_schedule(
            eligible, 10, strategy="abba", seed=schedule["seed"]
        )
    except Exception as error:
        raise VerificationError("cannot replay formal ABBA schedule: {}".format(error))
    if (
        schedule.get("executions") != expected_schedule
        or schedule.get("base_order") != BENCHMARK._stable_order(eligible, schedule["seed"])
        or formal.get("eligible_candidates") != eligible
        or formal.get("excluded_candidates")
        != [
            {"name": name, "reason": "correctness_invalid"}
            for name in names
            if not qualifications[name]["valid"]
        ]
        or formal.get("expected_execution_count") != len(expected_schedule)
        or formal.get("completed_execution_count") != len(expected_schedule)
        or len(runs) != len(expected_schedule)
    ):
        raise VerificationError("formal ABBA schedule/execution count changed")
    required_sources = _verify_formal_resume_identity(
        formal,
        candidates,
        expected_schedule,
        identity,
        formal_path,
        correctness_path,
        verified,
        run_root,
    )
    resume_payload = formal["resume_identity"]["payload"]
    profile_render = resume_payload["profile_render"]["path"]
    python_executable = resume_payload["python_executable"]["path"]
    configs = resume_payload["configs"]["path"]
    profile_args = resume_payload["profile_args"]
    contract = formal["contract"]
    by_name = {name: {} for name in finalist_names}
    used_paths = set()
    stable_environment = None
    stable_provenance = None
    validated_metrics = []
    for index, (run, item) in enumerate(zip(runs, expected_schedule)):
        if not isinstance(run, dict) or set(run) != set(BENCHMARK.RUN_RECORD_FIELDS):
            raise VerificationError("formal raw run record fields changed")
        candidate = candidate_by_name[item["candidate_name"]]
        expected_record = BENCHMARK._make_run_record(
            item,
            candidate,
            expected_root,
            profile_render,
            python_executable,
            contract,
            configs,
            profile_args,
        )
        for key in (
            "run_index",
            "round_index",
            "position_in_round",
            "candidate_name",
            "requested_execution_mode",
            "profile_path",
            "qualification_mode",
            "command",
            "metadata_path",
            "stdout_path",
            "stderr_path",
        ):
            if run.get(key) != expected_record[key]:
                raise VerificationError("formal run {} {} changed".format(index, key))
        if (
            run.get("passed") is not True
            or run.get("returncode") != 0
            or run.get("error") is not None
        ):
            raise VerificationError("formal run {} did not pass".format(index))
        paths = {}
        for path_key, hash_key in (
            ("metadata_path", "metadata_sha256"),
            ("stdout_path", "stdout_sha256"),
            ("stderr_path", "stderr_sha256"),
        ):
            artifact_path = _verified_path(
                run[path_key], verified, run_root, "formal run {} {}".format(index, path_key)
            )
            if not _within(artifact_path, expected_root) or str(artifact_path) in used_paths:
                raise VerificationError("formal child artifact path is reused or escaped")
            used_paths.add(str(artifact_path))
            if run.get(hash_key) != sha256_file(artifact_path):
                raise VerificationError("formal run {} {} changed".format(index, hash_key))
            paths[path_key] = artifact_path
        metadata = load_json(paths["metadata_path"], "formal run metadata")
        try:
            metrics = BENCHMARK.validate_child_metadata(metadata, candidate, contract)
        except Exception as error:
            raise VerificationError(
                "formal child metadata validation failed: {}".format(error)
            )
        if run.get("metrics") != metrics:
            raise VerificationError("formal child normalized metrics changed")
        if stable_environment is None:
            stable_environment = metrics["stable_environment"]
            stable_provenance = metrics["stable_provenance"]
        elif (
            metrics["stable_environment"] != stable_environment
            or metrics["stable_provenance"] != stable_provenance
        ):
            raise VerificationError("formal child environment/provenance is unstable")
        validated_metrics.append(run)
        name = run["candidate_name"]
        if name in by_name:
            if (
                metadata.get("pipeline_execution_counts")
                != EXPECTED_FORMAL_EXECUTION_COUNTS
                or run["round_index"] in by_name[name]
            ):
                raise VerificationError("formal finalist execution metadata changed")
            by_name[name][run["round_index"]] = {
                "candidate_name": name,
                "round_index": run["round_index"],
                "metadata": _artifact_facts(paths["metadata_path"]),
                "pipeline_execution_counts": dict(EXPECTED_FORMAL_EXECUTION_COUNTS),
            }
    if (
        formal.get("stable_environment") != stable_environment
        or formal.get("stable_provenance") != stable_provenance
    ):
        raise VerificationError("formal top-level stable provenance changed")
    _verify_stable_provenance(stable_provenance, identity, required_sources)
    try:
        summaries, ranking, comparisons = BENCHMARK.aggregate_runs(
            validated_metrics,
            eligible,
            10,
            bootstrap["resamples"],
            schedule["seed"],
        )
    except Exception as error:
        raise VerificationError("cannot aggregate formal raw runs: {}".format(error))
    if (
        formal.get("summaries") != summaries
        or formal.get("ranking") != ranking
        or formal.get("paired_comparisons") != comparisons
    ):
        raise VerificationError("formal raw-run aggregate/ranking/CI replay differs")
    selector = _recompute_benchmark_selection(formal, summaries=summaries)
    if any(set(rounds) != set(range(10)) for rounds in by_name.values()):
        raise VerificationError("formal finalist does not have ten 50-view runs")
    validated_runs = [
        by_name[name][round_index]
        for name in finalist_names
        for round_index in range(10)
    ]
    integrity_summary = result.get("execution_integrity")
    facts = _verify_binding_artifact(
        integrity_summary.get("artifact") if isinstance(integrity_summary, dict) else None,
        verified,
        run_root,
        "formal execution integrity",
    )
    integrity = load_json(facts["path"], "formal execution integrity")
    expected_integrity = {
        "schema_version": 1,
        "kind": "tacker_phase31_formal_execution_integrity",
        "protocol": {
            "frames_per_sequence": 50,
            "trials_per_candidate": 10,
            "schedule": "abba",
        },
        "expected_pipeline_execution_counts": dict(EXPECTED_FORMAL_EXECUTION_COUNTS),
        "finalist_names": finalist_names,
        "validated_runs": validated_runs,
        "claim_scope": (
            "Each finalist child sequence executed Tacker without fallback "
            "and reported exactly one prefill, 49 mixed steps, one drain, "
            "50 outputs, and 50 selected-head evaluations per head."
        ),
    }
    if (
        integrity_summary.get("no_fallback_and_exact_scheduler_counts_validated")
        is not True
        or integrity_summary.get("scope") != "finalist_tacker_children_10x50"
        or integrity != expected_integrity
    ):
        raise VerificationError("formal execution-integrity replay differs")
    return len(validated_runs), facts["sha256"], selector


def _verify_formal_evidence(
    stages,
    identity,
    qualification,
    baseline,
    originals,
    verified,
    run_root,
):
    stage = _require_stage(stages, "formal-benchmark")
    result = stage["result"]
    formal_path = _verified_path(
        result.get("fps_report", ""), verified, run_root, "formal FPS report"
    )
    correctness_path = _verified_path(
        result.get("correctness", ""), verified, run_root, "formal correctness"
    )
    formal = load_json(formal_path, "formal FPS report")
    correctness = load_json(correctness_path, "formal correctness")
    finalists = qualification["valid_finalists"]
    finalist_names = [item["variant_id"] for item in finalists]
    expected_correctness = dict(baseline)
    for candidate in finalists:
        expected_correctness[candidate["variant_id"]] = {
            "valid": True,
            "candidate_sha256": candidate["candidate_sha256"],
            "qualification_plan_sha256": qualification[
                "qualification_plan_sha256"
            ],
        }
    workload = identity["payload"]["workload"]
    try:
        expected_contract = BENCHMARK.make_contract(
            identity["payload"]["paths"]["model"],
            identity["payload"]["paths"]["source"],
            workload["name"],
            workload["iteration"],
            workload["split"],
            10,
            50,
            workload["image_width"],
            workload["image_height"],
            workload["gaussian_count"],
            view_indices=list(range(50)),
        )
    except Exception as error:
        raise VerificationError("cannot rebuild formal contract: {}".format(error))
    contract = formal.get("contract")
    schedule = formal.get("schedule")
    bootstrap = formal.get("bootstrap")
    if (
        formal.get("schema_version") != 1
        or formal.get("kind") != "4dgaussians_tacker_fps_benchmark"
        or formal.get("passed") is not True
        or formal.get("phase0_exit_condition", {}).get("met") is not True
        or contract != expected_contract
        or not isinstance(schedule, dict)
        or schedule.get("strategy") != "abba"
        or schedule.get("trials_per_candidate") != 10
        or schedule.get("seed") != identity["payload"]["search"]["seed"]
        or not isinstance(bootstrap, dict)
        or bootstrap.get("confidence") != 0.95
        or bootstrap.get("seed") != schedule.get("seed")
        or bootstrap.get("resampling_unit") != "paired_round"
        or type(bootstrap.get("resamples")) is not int
        or bootstrap["resamples"] <= 0
        or formal.get("selection_objective") != BENCHMARK.SELECTION_OBJECTIVE
        or formal.get("errors") != []
        or correctness != expected_correctness
        or formal.get("correctness_qualifications") != expected_correctness
        or result.get("formal_candidate_names") != finalist_names
        or result.get("deployment_winner") != formal.get("deployment_winner")
    ):
        raise VerificationError("formal 10x50 ABBA contract changed")
    candidates = _formal_candidate_entries(
        formal, finalist_names, originals, identity
    )
    c3_names = [item["variant_id"] for item in finalists if item["search_family"] == "c3"]
    c4_names = [item["variant_id"] for item in finalists if item["search_family"] == "c4"]
    if (
        not c3_names
        or not c4_names
        or result.get("c3_real_50_view_candidates") != c3_names
        or result.get("c4_real_50_view_candidates") != c4_names
    ):
        raise VerificationError("formal benchmark omitted a real C3/C4 finalist")
    validated_count, integrity_sha, selector = _verify_formal_execution(
        formal,
        candidates,
        finalist_names,
        result,
        identity,
        formal_path,
        correctness_path,
        verified,
        run_root,
    )
    if result.get("deployment_winner") != selector["deployment_winner"]:
        raise VerificationError("formal stage winner differs from raw-run replay")
    return formal, {
        "formal_report_sha256": sha256_file(formal_path),
        "formal_correctness_sha256": sha256_file(correctness_path),
        "valid_finalist_count": len(finalist_names),
        "validated_50_view_sequence_count": validated_count,
        "execution_integrity_sha256": integrity_sha,
        "c3_real_50_view_candidate_count": len(c3_names),
        "c4_real_50_view_candidate_count": len(c4_names),
    }


def _select_top_candidates_snapshot(formal):
    replay = deepcopy(formal)
    originals = {
        item.get("name"): item
        for item in formal.get("candidates", [])
        if isinstance(item, dict)
    }
    with tempfile.TemporaryDirectory(prefix="tacker-phase31-top3-inputs-") as temporary:
        root = Path(temporary)
        for candidate in replay.get("candidates", []):
            profile = candidate.get("profile_path") if isinstance(candidate, dict) else None
            if profile is None:
                continue
            destination = root / "{}.json".format(candidate["name"])
            _snapshot_store().copy_to(
                profile, destination, "top-3 candidate profile"
            )
            candidate["profile_path"] = str(destination)
        try:
            selected = TOP3.select_top_candidates(replay, limit=3)
        except Exception as error:
            raise VerificationError("cannot replay top-3 selection: {}".format(error))
    for candidate in selected:
        candidate["profile_path"] = originals[candidate["name"]].get("profile_path")
    return selected


def _verify_top3_identity(report, formal_path, candidates, identity):
    record = report.get("identity")
    if not isinstance(record, dict) or set(record) != {"sha256", "payload"}:
        raise VerificationError("top-3 identity schema changed")
    payload = record.get("payload")
    if not isinstance(payload, dict) or record.get("sha256") != TOP3.sha256_json(payload):
        raise VerificationError("top-3 identity hash changed")
    phase = identity["payload"]
    expected = {
        "driver": _path_hash(phase["scripts"].get("top3")),
        "fps_report": {"path": str(formal_path), "sha256": sha256_file(formal_path)},
        "candidates": candidates,
        "profile_script": _path_hash(phase["scripts"].get("nsight")),
        "model_path": phase["paths"]["model"],
        "config": _path_hash(phase["files"].get("config")),
        "source_path": phase["paths"]["source"],
        "gpu": phase["workload"]["physical_gpu"],
        "frames": 50,
        "iteration": phase["workload"]["iteration"],
        "workload_name": phase["workload"]["name"],
        "workload_files": {
            name: _path_hash(item) for name, item in phase["workload_files"].items()
        },
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise VerificationError("top-3 identity {} changed".format(key))
    if set(payload) != set(expected) | {"orchestration_sources"}:
        raise VerificationError("top-3 identity fields changed")
    sources = payload.get("orchestration_sources")
    if not isinstance(sources, dict):
        raise VerificationError("top-3 orchestration provenance is missing")
    try:
        current_sources = TOP3._orchestration_source_identity(PROJECT_ROOT)
    except Exception as error:
        raise VerificationError(
            "cannot rebuild top-3 orchestration provenance: {}".format(error)
        )
    if sources != current_sources:
        raise VerificationError("top-3 orchestration source identity changed")
    phase_source_map = {
        "profile_render.py": phase["runtime_sources"].get("profile_render"),
        "gaussian_renderer/tacker_pipeline.py": phase["runtime_sources"].get(
            "tacker_pipeline"
        ),
        "diff_gaussian_rasterization/__init__.py": phase["runtime_sources"].get(
            "raster_python_binding"
        ),
    }
    for name, source in sources.items():
        values = source if isinstance(source, list) else [source]
        for index, value in enumerate(values):
            if value is None:
                continue
            _verify_identity_record(
                value, "top-3 source {}[{}]".format(name, index)
            )
        phase_source = phase_source_map.get(name)
        if phase_source is not None and _path_hash(source) != _path_hash(phase_source):
            raise VerificationError("top-3 source {} differs from run identity".format(name))
    return record


def _load_sealed_nsight_summarizer(top3_identity):
    record = top3_identity["payload"]["orchestration_sources"].get(
        "summarize_nsight_stats.py"
    )
    if not isinstance(record, dict):
        raise VerificationError("top-3 identity omitted the Nsight summarizer")
    snapshot = _snapshot_store().snapshot(
        record["path"], "sealed Nsight summarizer", capture_bytes=True
    )
    if snapshot["sha256"] != record.get("sha256"):
        raise VerificationError("sealed Nsight summarizer source changed")
    return _module_from_snapshot(
        "tacker_phase31_postflight_nsight_summarizer",
        Path(record["path"]),
        snapshot["bytes"],
    )


def _verify_nsight(stages, formal, formal_path, identity, verified, run_root):
    stage = _require_stage(stages, "top3-nsight")
    result = stage["result"]
    path = _verified_path(
        result.get("report", ""), verified, run_root, "top-3 Nsight report"
    )
    report = load_json(path, "top-3 Nsight report")
    expected = _select_top_candidates_snapshot(formal)
    profiles = report.get("profiles")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "4dgaussians_tacker_top3_nsight"
        or report.get("passed") is not True
        or report.get("fps_report_sha256") != sha256_file(formal_path)
        or report.get("requested_limit") != 3
        or report.get("selection_policy")
        != "first N candidates in formal eligible_ranking, including baselines"
        or report.get("selected_candidates") != expected
        or report.get("errors") != []
        or not isinstance(profiles, list)
        or len(profiles) != 3
    ):
        raise VerificationError("top-3 Nsight report contract changed")
    top3_identity = _verify_top3_identity(report, formal_path, expected, identity)
    summarizer = _load_sealed_nsight_summarizer(top3_identity)
    try:
        expected_summary = TOP3._expected_summary_contract(
            formal, top3_identity, expected
        )
    except Exception as error:
        raise VerificationError("cannot rebuild top-3 summary contract: {}".format(error))
    formal_sources = formal["stable_provenance"]["source_files"]
    orchestration = top3_identity["payload"]["orchestration_sources"]
    for name in (
        "profile_render.py",
        "gaussian_renderer/__init__.py",
        "gaussian_renderer/tacker_pipeline.py",
        "diff_gaussian_rasterization/__init__.py",
    ):
        sealed = orchestration.get(name)
        if not isinstance(sealed, dict) or formal_sources.get(name) != sealed.get("sha256"):
            raise VerificationError("top-3/formal source provenance {} differs".format(name))
    if formal_sources.get("configs") != top3_identity["payload"]["config"]["sha256"]:
        raise VerificationError("top-3/formal config provenance differs")
    raster_binaries = orchestration.get("diff_gaussian_rasterization._C")
    if (
        not isinstance(raster_binaries, list)
        or formal_sources.get("diff_gaussian_rasterization._C")
        not in {
            item.get("sha256")
            for item in raster_binaries
            if isinstance(item, dict)
        }
    ):
        raise VerificationError("top-3/formal raster binary provenance differs")
    output_root = run_root / "sessions" / "nsight" / "profiles"
    used_paths = set()
    diagnostics = []
    for index, profile in enumerate(profiles):
        if (
            not isinstance(profile, dict)
            or set(profile) != set(TOP3.PROFILE_RECORD_FIELDS)
            or profile.get("candidate") != expected[index]
            or profile.get("passed") is not True
            or profile.get("returncode") != 0
            or profile.get("error") is not None
        ):
            raise VerificationError("top-3 Nsight profile result changed")
        candidate = expected[index]
        expected_record = TOP3._make_profile_record(
            candidate,
            output_root,
            top3_identity["payload"]["profile_script"]["path"],
            top3_identity["payload"]["model_path"],
            top3_identity["payload"]["config"]["path"],
            top3_identity["payload"]["source_path"],
            top3_identity["payload"]["gpu"],
            top3_identity["payload"]["frames"],
            top3_identity["payload"]["iteration"],
            top3_identity["payload"]["workload_name"],
        )
        for key in (
            "candidate",
            "command",
            "output_dir",
            "summary_path",
            "stdout_path",
            "stderr_path",
        ):
            if profile.get(key) != expected_record[key]:
                raise VerificationError("top-3 Nsight profile {} changed".format(key))
        candidate_dir = _lexical_path(
            profile["output_dir"], "Nsight candidate directory", directory=True
        )
        if candidate_dir != TOP3.candidate_output_dir(output_root, candidate):
            raise VerificationError("top-3 Nsight output directory changed")
        for path_key, hash_key in (
            ("summary_path", "summary_sha256"),
            ("stdout_path", "stdout_sha256"),
            ("stderr_path", "stderr_sha256"),
        ):
            artifact_path = _verified_path(
                profile.get(path_key, ""), verified, run_root, "Nsight profile artifact"
            )
            if str(artifact_path) in used_paths:
                raise VerificationError("Nsight profile artifact was reused")
            used_paths.add(str(artifact_path))
            if (
                not _within(artifact_path, candidate_dir)
                or profile.get(hash_key) != sha256_file(artifact_path)
            ):
                raise VerificationError("Nsight profile artifact hash changed")
        summary = load_json(profile["summary_path"], "Nsight summary")
        mode = candidate["execution_mode"]
        metadata_path = _verified_path(
            candidate_dir / "{}_profile_metadata.json".format(mode),
            verified,
            run_root,
            "Nsight raw profile metadata",
        )
        metadata = load_json(metadata_path, "Nsight raw profile metadata")
        stats_path = _verified_path(
            candidate_dir / "4dgs_render_{}_stats.csv".format(mode),
            verified,
            run_root,
            "Nsight stats CSV",
        )
        if (
            profile.get("profile_metadata_sha256") != sha256_file(metadata_path)
            or summary.get("metadata") != metadata
            or summary.get("source") != str(stats_path)
        ):
            raise VerificationError("Nsight raw/summary metadata binding changed")
        manifest = profile.get("raw_artifacts")
        if not isinstance(manifest, dict):
            raise VerificationError("Nsight raw artifact manifest is missing")
        with tempfile.TemporaryDirectory(
            prefix="tacker-phase31-nsight-raw-"
        ) as temporary:
            replay_dir = Path(temporary) / "candidate"
            replay_dir.mkdir()
            for relative, facts in sorted(manifest.items()):
                relative_path = Path(relative)
                if (
                    not isinstance(relative, str)
                    or not relative
                    or relative_path.is_absolute()
                    or ".." in relative_path.parts
                    or relative_path.as_posix() != relative
                    or not isinstance(facts, dict)
                    or set(facts) != {"sha256", "size_bytes"}
                ):
                    raise VerificationError("Nsight raw artifact entry is unsafe")
                source = _verified_path(
                    candidate_dir / relative_path,
                    verified,
                    run_root,
                    "Nsight raw artifact",
                )
                if str(source) in used_paths:
                    if source != Path(profile["summary_path"]):
                        raise VerificationError("Nsight raw artifact was reused")
                else:
                    used_paths.add(str(source))
                observed = _artifact_facts(source)
                if {
                    "sha256": observed["sha256"],
                    "size_bytes": observed["size_bytes"],
                } != facts:
                    raise VerificationError("Nsight raw artifact manifest changed")
                _snapshot_store().copy_to(
                    source,
                    replay_dir / relative_path,
                    "Nsight raw artifact",
                )
            try:
                replay_manifest = TOP3._validate_raw_artifacts(
                    replay_dir, candidate, expected_manifest=manifest
                )
            except Exception as error:
                raise VerificationError(
                    "Nsight raw artifact replay failed: {}".format(error)
                )
            if replay_manifest != manifest:
                raise VerificationError("Nsight raw artifact replay differs")
            try:
                rebuilt_summary = summarizer.summarize(
                    replay_dir / stats_path.name,
                    replay_dir / metadata_path.name,
                )
            except Exception as error:
                raise VerificationError(
                    "Nsight summary reconstruction failed: {}".format(error)
                )
            rebuilt_summary["source"] = str(stats_path)
            if canonical_json_bytes(rebuilt_summary) != canonical_json_bytes(summary):
                raise VerificationError("Nsight summary reconstruction differs")
        try:
            replay_diagnostics = TOP3.extract_nsight_diagnostics(
                rebuilt_summary, candidate, expected_summary
            )
        except Exception as error:
            raise VerificationError("Nsight diagnostics validation failed: {}".format(error))
        if profile.get("diagnostics") != replay_diagnostics:
            raise VerificationError("Nsight diagnostics replay differs")
        diagnostics.append(replay_diagnostics)
    baseline_fps = diagnostics[0]["nsight_render_fps"]
    baseline_launches = diagnostics[0]["kernel_launches"]["per_frame"]
    expected_comparisons = [
        {
            "candidate": expected[index]["name"],
            "rank1_candidate": expected[0]["name"],
            "nsight_fps_ratio": item["nsight_render_fps"] / baseline_fps,
            "kernel_launches_per_frame_delta": (
                item["kernel_launches"]["per_frame"] - baseline_launches
            ),
        }
        for index, item in enumerate(diagnostics)
    ]
    if report.get("rank1_comparisons") != expected_comparisons:
        raise VerificationError("Nsight rank-1 comparisons replay differs")
    checkpoint_path = _verified_path(
        output_root / "top3.checkpoint.json",
        verified,
        run_root,
        "top-3 checkpoint",
    )
    checkpoint = load_json(checkpoint_path, "top-3 checkpoint")
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("kind") != TOP3.CHECKPOINT_KIND
        or checkpoint.get("identity") != top3_identity
        or checkpoint.get("report") != report
    ):
        raise VerificationError("top-3 checkpoint differs from completed report")
    return {
        "report_sha256": sha256_file(path),
        "profile_count": 3,
        "selected_candidate_names": [item["name"] for item in expected],
        "raw_artifact_count": sum(
            len(item["raw_artifacts"]) for item in profiles
        ),
        "diagnostics_replayed": True,
    }


def _recompute_benchmark_selection(report, summaries=None):
    candidates = report.get("candidates")
    names = [item.get("name") for item in candidates if isinstance(item, dict)] if isinstance(candidates, list) else []
    bootstrap = report.get("bootstrap")
    schedule = report.get("schedule")
    stored_selection = report.get("selection")
    if not names or not all(isinstance(item, dict) for item in (bootstrap, schedule, stored_selection)):
        raise VerificationError("formal FPS selection inputs are incomplete")
    equivalence = stored_selection.get("equivalence")
    if (
        not isinstance(equivalence, dict)
        or equivalence.get("fraction") != BENCHMARK.DEFAULT_EQUIVALENCE_FRACTION
    ):
        raise VerificationError("formal selector equivalence policy changed")
    try:
        rebuilt = BENCHMARK.select_candidates(
            report.get("summaries") if summaries is None else summaries,
            candidate_qualifications=report.get("correctness_qualifications"),
            candidate_selection_metadata=report.get("candidate_selection_metadata"),
            incumbent_name="current_tacker",
            candidate_names=names,
            bootstrap_resamples=bootstrap["resamples"],
            seed=schedule["seed"],
            promotion_min_ratio=BENCHMARK.DEFAULT_PROMOTION_MIN_RATIO,
            equivalence_fraction=BENCHMARK.DEFAULT_EQUIVALENCE_FRACTION,
        )
    except Exception as error:
        raise VerificationError("cannot replay formal selector: {}".format(error))
    if rebuilt != stored_selection:
        raise VerificationError("formal benchmark selector replay differs")
    if (
        report.get("experimental_winner") != rebuilt["experimental_winner"]
        or report.get("deployment_winner") != rebuilt["deployment_winner"]
        or report.get("promotion") != rebuilt["promotion"]
        or report.get("eligible_ranking") != rebuilt["eligible_ranking"]
    ):
        raise VerificationError("formal FPS top-level selection differs")
    return rebuilt


def _verify_selection(stages, identity, matrix, verified, run_root):
    formal_stage = _require_stage(stages, "formal-benchmark")
    formal_result = formal_stage["result"]
    formal_path = _verified_path(
        formal_result.get("fps_report", ""), verified, run_root, "formal FPS report"
    )
    formal = load_json(formal_path, "formal FPS report")
    if formal.get("passed") is not True:
        raise VerificationError("formal FPS report did not pass")
    selector = _recompute_benchmark_selection(formal)
    selection_stage = _require_stage(stages, "selection")
    selection_result = selection_stage["result"]
    selection_path = _verified_path(
        selection_result.get("path", ""), verified, run_root, "selection manifest"
    )
    selection = load_json(selection_path, "selection manifest")
    portable = selection.get("portable_hash_input")
    expected_portable = {
        key: selection.get(key)
        for key in (
            "schema_version",
            "kind",
            "matrix_sha256",
            "formal_report_sha256",
            "deployment_winner",
            "winner_is_new_challenger",
            "disabled_winner_qualification_profile",
            "reused_incumbent_profile",
            "baseline_winner_has_no_synthetic_profile",
        )
    }
    expected_portable["disabled_winner_qualification_profile"] = _portable_artifact(
        expected_portable["disabled_winner_qualification_profile"]
    )
    expected_portable["reused_incumbent_profile"] = _portable_artifact(
        expected_portable["reused_incumbent_profile"]
    )
    if (
        selection.get("schema_version") != 1
        or selection.get("kind") != SELECTION_KIND
        or not isinstance(portable, dict)
        or portable != expected_portable
        or selection.get("selection_sha256")
        != sha256_json(portable, "tacker-phase31-selection-v1")
    ):
        raise VerificationError("selection portable hash changed")
    winner = selector["deployment_winner"]
    if (
        selection.get("matrix_sha256") != matrix["matrix_sha256"]
        or selection.get("formal_report_sha256") != sha256_file(formal_path)
        or selection.get("deployment_winner") != winner
        or selection_result.get("document") != selection
    ):
        raise VerificationError("selection is not bound to formal winner/matrix")
    candidate = next(
        (item for item in matrix["candidates"] if item["variant_id"] == winner),
        None,
    )
    winner_profile = selection.get("disabled_winner_qualification_profile")
    if candidate is not None:
        facts = _verify_binding_artifact(
            winner_profile, verified, run_root, "winner qualification profile"
        )
        profile = load_json(facts["path"], "winner qualification profile")
        provenance = profile.get("provenance")
        if (
            selection.get("winner_is_new_challenger") is not True
            or selection.get("baseline_winner_has_no_synthetic_profile") is not False
            or selection.get("reused_incumbent_profile") is not None
            or profile.get("deployment") != {"enabled": False, "valid": False}
            or profile.get("selected_variant_id") != winner
            or not isinstance(provenance, dict)
            or provenance.get("matrix_sha256") != matrix["matrix_sha256"]
            or provenance.get("candidate_sha256") != candidate["candidate_sha256"]
            or portable.get("disabled_winner_qualification_profile")
            != _portable_artifact(facts)
        ):
            raise VerificationError("challenger winner profile changed")
    else:
        if winner not in BASELINES or winner_profile is not None:
            raise VerificationError("baseline winner profile policy changed")
        if (
            selection.get("winner_is_new_challenger") is not False
            or selection.get("baseline_winner_has_no_synthetic_profile") is not True
        ):
            raise VerificationError("baseline winner mislabeled as challenger")
        reused = selection.get("reused_incumbent_profile")
        if winner == "current_tacker":
            current_record = identity["payload"]["files"].get(
                "current_tacker_profile"
            )
            current_path = _verify_identity_record(
                current_record, "current Tacker profile"
            )
            expected_reused = _artifact_facts(current_path)
            if reused != expected_reused:
                raise VerificationError("reused incumbent artifact changed")
        elif reused is not None:
            raise VerificationError("serial/two-stream winner has a synthetic profile")
    return {
        "formal_report_sha256": sha256_file(formal_path),
        "experimental_winner": selector["experimental_winner"],
        "deployment_winner": winner,
        "selection_sha256": selection["selection_sha256"],
        "winner_is_new_challenger": candidate is not None,
    }


def _cross_check_report(report, state, stages, verified, run_root):
    if report.get("checkpoint") != str((run_root / "phase31-state.json").resolve()):
        raise VerificationError("top-level checkpoint path changed")
    mappings = (
        ("qualification", "qualification-plan"),
        ("baseline", "baseline-quality"),
        ("formal", "formal-benchmark"),
        ("selection", "selection"),
        ("nsight", "top3-nsight"),
    )
    for report_key, stage_name in mappings:
        if report.get(report_key) != _require_stage(stages, stage_name)["result"]:
            raise VerificationError("top-level {} differs from stage result".format(report_key))
    c4_result = _require_stage(stages, "c4-matrix")["result"]
    matrix_summary = report.get("matrix")
    if (
        not isinstance(matrix_summary, dict)
        or matrix_summary.get("path") != c4_result.get("matrix_path")
        or matrix_summary.get("matrix_sha256") != c4_result.get("matrix_sha256")
        or matrix_summary.get("candidate_count") != c4_result.get("candidate_count")
        or matrix_summary.get("exhaustive_c2_count") != 450
    ):
        raise VerificationError("top-level matrix summary differs from stage")
    ranking_result = _require_stage(stages, "screening-ranking")["result"]
    screening = report.get("screening")
    if (
        not isinstance(screening, dict)
        or screening.get("ranking") != ranking_result.get("ranking_path")
        or screening.get("ranking_sha256") != ranking_result.get("ranking_sha256")
        or screening.get("measurement_sources") != ranking_result.get("measurement_sources")
    ):
        raise VerificationError("top-level screening summary differs from stage")
    formal_result = _require_stage(stages, "formal-candidate-set")["result"]
    if report.get("formal_candidate_set") != {
        "path": formal_result.get("path"),
        "candidate_set_sha256": formal_result.get("candidate_set_sha256"),
        "formal_set_sha256": formal_result.get("formal_set_sha256"),
    }:
        raise VerificationError("top-level formal set differs from stage")


def _verify_phase31_active(
    run_root,
    python_executable=None,
    runner=subprocess.run,
    profile_materializer=None,
):
    root = _lexical_path(run_root, "run root", directory=True)
    state_path = root / "phase31-state.json"
    state = load_json(state_path, "Phase-3.1 checkpoint")
    if (
        state.get("schema_version") != 1
        or state.get("kind") != STATE_KIND
        or state.get("status") != "succeeded"
    ):
        raise VerificationError("checkpoint is not a completed Phase-3.1 run")
    identity = state.get("identity")
    identity_paths = _identity_paths(identity)
    _load_replay_modules(identity)
    stages = _stage_map(state)
    verified, artifact_count = _audit_stage_artifacts(stages, root)
    report_facts = _verify_artifact(
        state.get("report"), root, "completed run report"
    )
    if report_facts["path"] != str((root / "phase31-report.json").resolve()):
        raise VerificationError("checkpoint points to a non-canonical run report path")
    report = load_json(report_facts["path"], "completed Phase-3.1 report")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != RUN_REPORT_KIND
        or report.get("passed") is not True
        or report.get("identity") != identity
    ):
        raise VerificationError("completed run report contract changed")
    _cross_check_report(report, state, stages, verified, root)
    preflight = _verify_preflight(stages, identity, verified, root)

    base, base_path = _matrix_from_stage(stages, "base-matrix", verified, root)
    c3, c3_path = _matrix_from_stage(stages, "c3-matrix", verified, root)
    c4, c4_path = _matrix_from_stage(stages, "c4-matrix", verified, root)
    del base_path, c3_path
    ranking_stage = _require_stage(stages, "screening-ranking")["result"]
    measurement_path = _verified_path(
        ranking_stage.get("measurement_sources", ""),
        verified,
        root,
        "measurement sources",
    )
    measurement = load_json(measurement_path, "measurement sources")
    records, staged_records, bindings = _collect_screening_records(
        stages,
        identity,
        measurement,
        {"base": base, "c3": c3, "c4": c4},
        verified,
        root,
    )
    matrix_checks = _rebuild_matrices(
        identity,
        preflight,
        {"base": base, "c3": c3, "c4": c4},
        staged_records,
    )
    ranking, ranking_checks = _verify_ranking(
        stages,
        identity,
        c4,
        records,
        bindings,
        measurement,
        verified,
        root,
    )
    formal, formal_checks = _verify_formal_set(
        stages, identity, c4, ranking, verified, root
    )
    executable = (
        python_executable
        if python_executable is not None
        else identity["payload"]["paths"]["python"]
    )
    profile_checks = _verify_profiles(
        stages,
        identity,
        c4,
        c4_path,
        verified,
        root,
        executable,
        runner,
        profile_materializer,
    )
    _, original_profiles = _profile_manifest(
        _require_stage(stages, "profiles-final"), c4, verified, root
    )
    qualification, qualification_checks = _verify_qualification_plan(
        stages,
        identity,
        c4,
        c4_path,
        ranking,
        formal,
        original_profiles,
        verified,
        root,
    )
    baseline, baseline_checks = _verify_baseline_quality(
        stages, identity, verified, root
    )
    formal_report, formal_evidence_checks = _verify_formal_evidence(
        stages,
        identity,
        qualification,
        baseline,
        original_profiles,
        verified,
        root,
    )
    selection_checks = _verify_selection(stages, identity, c4, verified, root)
    formal_path = _verified_path(
        _require_stage(stages, "formal-benchmark")["result"].get(
            "fps_report", ""
        ),
        verified,
        root,
        "formal FPS report",
    )
    nsight_checks = _verify_nsight(
        stages, formal_report, formal_path, identity, verified, root
    )
    return {
        "schema_version": 1,
        "kind": VERIFY_REPORT_KIND,
        "passed": True,
        "run_root": str(root),
        "run_report_sha256": report_facts["sha256"],
        "identity_sha256": identity["sha256"],
        "identity_input_artifact_count": len(identity_paths),
        "succeeded_stage_count": len(stages),
        "stage_artifact_ledger_entry_count": artifact_count,
        "unique_stage_artifact_count": len(verified),
        "checks": {
            "state_and_report": {"passed": True},
            "stage_artifacts": {"passed": True, "all_succeeded_artifacts_verified": True},
            "matrices": dict({"passed": True}, **matrix_checks),
            "screening_ranking": dict({"passed": True}, **ranking_checks),
            "formal_candidate_set": dict({"passed": True}, **formal_checks),
            "qualification_profiles": dict({"passed": True}, **profile_checks),
            "qualification_and_backfill": dict(
                {"passed": True}, **qualification_checks
            ),
            "baseline_quality": dict({"passed": True}, **baseline_checks),
            "formal_execution": dict(
                {"passed": True}, **formal_evidence_checks
            ),
            "selection": dict({"passed": True}, **selection_checks),
            "top3_nsight": dict({"passed": True}, **nsight_checks),
        },
    }


def verify_phase31(
    run_root,
    python_executable=None,
    runner=subprocess.run,
    profile_materializer=None,
    snapshot_after_read_hook=None,
):
    global _ACTIVE_SNAPSHOT_STORE
    if _ACTIVE_SNAPSHOT_STORE is not None:
        raise VerificationError("nested Phase-3.1 verification is unsupported")
    _ACTIVE_SNAPSHOT_STORE = SnapshotStore(
        after_read_hook=snapshot_after_read_hook
    )
    try:
        return _verify_phase31_active(
            run_root,
            python_executable=python_executable,
            runner=runner,
            profile_materializer=profile_materializer,
        )
    finally:
        _ACTIVE_SNAPSHOT_STORE = None


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "To run this verifier from outside the clean source checkout, set "
            "{}=/absolute/path/to/4DGaussians before launching it."
        ).format(PROJECT_ROOT_ENV),
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python-executable")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    output = Path(os.path.abspath(str(Path(args.output).expanduser())))
    run_root = Path(os.path.abspath(str(Path(args.run_root).expanduser())))
    if _within(output, run_root):
        parser.error("--output must be outside the read-only run root")
    try:
        report = verify_phase31(
            run_root,
            python_executable=args.python_executable,
        )
    except Exception as error:
        report = {
            "schema_version": 1,
            "kind": VERIFY_REPORT_KIND,
            "passed": False,
            "run_root": str(run_root),
            "error": "{}: {}".format(type(error).__name__, error),
        }
        atomic_write_json(output, report)
        print("Phase-3.1 postflight verification failed: {}".format(error), file=sys.stderr)
        return 1
    atomic_write_json(output, report)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
