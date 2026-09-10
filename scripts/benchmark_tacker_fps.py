#!/usr/bin/env python3
"""Interleaved, fail-closed whole-run FPS benchmark for Tacker candidates.

The driver launches ``profile_render.py`` once per candidate and trial.  A
single child invocation measures one complete render sequence; candidates are
interleaved here so that clock and temperature drift is paired by round.  The
module intentionally uses only the Python standard library, which also keeps
its ordering, validation, and statistical contracts CPU-testable.

Example::

    python scripts/benchmark_tacker_fps.py \
      --output /tmp/tacker-fps/report.json \
      --current-tacker-profile tacker_profiles/current.json \
      --candidate wider_head=/tmp/profiles/wider-head.json \
      --correctness-json /tmp/tacker-fps/correctness.json \
      --model-path /data/model --source-path /data/scene \
      --configs arguments/dynerf/flame_steak.py \
      --workload-name flame_steak --iteration 14000 \
      --expected-image-width 1352 --expected-image-height 1014 \
      --expected-gaussian-count 111525 \
      --profile-arg=--resolution --profile-arg=1

Every value supplied with ``--profile-arg`` is passed as one argv item.  The
driver-owned workload, timing, mode, profile, and metadata arguments cannot be
overridden.  Commands are always executed as argv lists with ``shell=False``.
"""

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import uuid


SCHEMA_VERSION = 1
REPORT_KIND = "4dgaussians_tacker_fps_benchmark"
CHILD_KIND = "4dgaussians_tacker_render_profile"
SELECTION_OBJECTIVE = "median_throughput_fps"
TIMING_METHOD = "perf_counter_with_cuda_synchronize"
FRAME_TIMING_METHOD = "cuda_event_consumer_completion_intervals"
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_PROMOTION_MIN_RATIO = 1.01
DEFAULT_EQUIVALENCE_FRACTION = 0.005
REQUIRED_PROVENANCE_SOURCE_FILES = (
    "profile_render.py",
    "configs",
    "gaussian_renderer/__init__.py",
    "gaussian_renderer/tacker_pipeline.py",
    "diff_gaussian_rasterization/__init__.py",
    "diff_gaussian_rasterization._C",
)
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
RESERVED_CANDIDATE_NAMES = frozenset(
    ("serial", "two_stream", "current_tacker")
)
TIMING_CONTRACT = {
    "unit": "whole_sequence",
    "primary_metric": "median_throughput_fps",
    "higher_is_better": True,
    "wall_clock": "perf_counter",
    "wall_clock_completion": "cuda_synchronize_after_each_trial",
    "cuda_events": "start_end_and_per_frame_completion",
    "setup_policy": "single_load_prepare_warmup_before_all_trials",
    "io_in_timed_region": False,
}

# These options define the comparison contract or a candidate's identity.  If
# callers could smuggle a second copy through --profile-arg, argparse's
# last-value-wins behaviour could silently benchmark a different workload.
CONTROLLED_PROFILE_OPTIONS = frozenset(
    (
        "-m",
        "--model_path",
        "--model-path",
        "-s",
        "--source_path",
        "--source-path",
        "--iteration",
        "--configs",
        "--split",
        "--warmup",
        "--frames",
        "--trials",
        "--execution-mode",
        "--execution_mode",
        "--tacker-profile",
        "--qualification-mode",
        "--qualification-profile",
        "--correctness-json",
        "--selection-metadata-json",
        "--workload-name",
        "--metadata",
        "--quiet",
        "-h",
        "--help",
    )
)


class BenchmarkContractError(ValueError):
    """Raised when configuration or child metadata violates the contract."""


def utc_now():
    """Return an ISO-8601 UTC timestamp."""

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical_json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, value):
    """Atomically replace ``path`` with one fsynced, finite JSON document."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target.name),
        suffix=".tmp",
        dir=str(target.parent),
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
    """Persist the directory entry created by an atomic publish."""

    target = Path(path).expanduser().resolve()
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(target.parent), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json_no_clobber(path, value):
    """Atomically publish finite JSON only when ``path`` is still absent.

    The hard-link commit is an atomic create-if-absent operation on the same
    filesystem.  This closes the race between the CLI's early existence check
    and the end of a long benchmark run.
    """

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target.name),
        suffix=".tmp",
        dir=str(target.parent),
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


def atomic_write_text(path, value):
    """Atomically persist captured child output beside its metadata."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target.name),
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value or "")
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


def _stable_order(names, seed):
    """Return a cross-process deterministic seed-dependent base order."""

    def key(item):
        index, name = item
        material = "{}\0{}\0{}".format(seed, index, name).encode("utf-8")
        return hashlib.sha256(material).hexdigest(), index

    return [name for _, name in sorted(enumerate(names), key=key)]


def build_schedule(candidate_names, trials, strategy="abba", seed=0):
    """Build deterministic rounds containing each candidate exactly once.

    ``round_robin`` rotates the seed-derived base order every round.  ``abba``
    uses a forward order and its reverse, then rotates before the next pair of
    rounds.  With two candidates this is the familiar A-B / B-A pattern.
    """

    names = list(candidate_names)
    if not names:
        raise BenchmarkContractError("at least one candidate is required")
    if len(set(names)) != len(names):
        raise BenchmarkContractError("candidate names must be unique")
    if type(trials) is not int or trials <= 0:
        raise BenchmarkContractError("trials must be a positive integer")
    if strategy not in ("round_robin", "abba"):
        raise BenchmarkContractError(
            "schedule strategy must be round_robin or abba"
        )
    if type(seed) is not int:
        raise BenchmarkContractError("schedule seed must be an integer")

    base = _stable_order(names, seed)
    schedule = []
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
            schedule.append(
                {
                    "run_index": run_index,
                    "round_index": round_index,
                    "position_in_round": position,
                    "candidate_name": candidate_name,
                }
            )
            run_index += 1
    return schedule


def _is_finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _finite(mapping, key, label, positive=False, nonnegative=False):
    if key not in mapping:
        raise BenchmarkContractError("{}.{} is required".format(label, key))
    value = mapping[key]
    if not _is_finite_number(value):
        raise BenchmarkContractError(
            "{}.{} must be a finite number".format(label, key)
        )
    value = float(value)
    if positive and value <= 0.0:
        raise BenchmarkContractError("{}.{} must be > 0".format(label, key))
    if nonnegative and value < 0.0:
        raise BenchmarkContractError("{}.{} must be >= 0".format(label, key))
    return value


def _mapping(value, label):
    if not isinstance(value, dict):
        raise BenchmarkContractError("{} must be a JSON object".format(label))
    return value


def load_json_mapping_snapshot(path, label):
    """Parse and hash one immutable finite-JSON byte snapshot."""

    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise BenchmarkContractError(
            "cannot read {} {}: {}".format(label, resolved, error)
        )
    document = _mapping(document, label)
    try:
        canonical_json_bytes(document)
    except (TypeError, ValueError) as error:
        raise BenchmarkContractError("{} must contain finite JSON: {}".format(label, error))
    return document, hashlib.sha256(raw).hexdigest()


def load_json_mapping(path, label):
    """Load one finite-JSON object for a driver-owned selection input."""

    document, _ = load_json_mapping_snapshot(path, label)
    return document


def load_correctness_qualifications(path, include_sha256=False):
    """Load direct ``name -> {valid}`` or a top-level candidates mapping."""

    document, digest = load_json_mapping_snapshot(path, "correctness JSON")
    if "candidates" not in document:
        result = document
    else:
        candidates = document["candidates"]
        if not isinstance(candidates, dict):
            raise BenchmarkContractError(
                "correctness JSON.candidates must be a JSON object"
            )
        # Preserve the unambiguous direct-map case where a candidate is literally
        # named "candidates": {"valid": ...}.  Otherwise this is the wrapper form.
        result = document if type(candidates.get("valid")) is bool else candidates
    return (result, digest) if include_sha256 else result


def load_candidate_selection_metadata(path, include_sha256=False):
    """Load the direct ``candidate name -> tie-break metadata`` mapping."""

    document, digest = load_json_mapping_snapshot(path, "selection metadata JSON")
    return (document, digest) if include_sha256 else document


def _list(value, label):
    if not isinstance(value, list):
        raise BenchmarkContractError("{} must be a JSON array".format(label))
    return value


def _same_number(actual, expected, label, rel_tol=1e-9, abs_tol=1e-9):
    if not _is_finite_number(actual) or not math.isclose(
        float(actual), float(expected), rel_tol=rel_tol, abs_tol=abs_tol
    ):
        raise BenchmarkContractError(
            "{} mismatch: expected {!r}, got {!r}".format(label, expected, actual)
        )


def _require_equal(mapping, key, expected, label):
    if key not in mapping:
        raise BenchmarkContractError(
            "{}.{} is required".format(label, key)
        )
    actual = mapping[key]
    if actual != expected:
        raise BenchmarkContractError(
            "{}.{} mismatch: expected {!r}, got {!r}".format(
                label, key, expected, actual
            )
        )


def _resolved_string(value, label):
    if not isinstance(value, str) or not value:
        raise BenchmarkContractError("{} must be a non-empty path".format(label))
    return str(Path(value).expanduser().resolve())


def validate_child_metadata(metadata, candidate, contract):
    """Validate one profile_render result and return normalized trial data."""

    document = _mapping(metadata, "child metadata")
    _require_equal(document, "schema_version", 1, "child metadata")
    _require_equal(document, "kind", CHILD_KIND, "child metadata")
    _require_equal(document, "passed", True, "child metadata")

    expected_fields = (
        ("workload_name", contract["workload_name"]),
        ("iteration", contract["iteration"]),
        ("split", contract["split"]),
        ("warmup_frames", contract["warmup_frames"]),
        ("profile_frames", contract["profile_frames"]),
        ("view_indices", contract["view_indices"]),
        ("image_width", contract["image_width"]),
        ("image_height", contract["image_height"]),
        ("gaussian_count", contract["gaussian_count"]),
        ("timing_method", TIMING_METHOD),
        ("frame_timing_method", FRAME_TIMING_METHOD),
        ("io_in_timed_region", False),
        ("execution_mode", candidate["execution_mode"]),
        ("actual_execution_mode", candidate["execution_mode"]),
        ("two_stream_fallback_reason", None),
        ("tacker_fallback_reason", None),
        (
            "qualification_mode_requested",
            bool(candidate.get("qualification_mode", False)),
        ),
        (
            "qualification_mode_executed",
            bool(candidate.get("qualification_mode", False)),
        ),
    )
    for key, expected in expected_fields:
        _require_equal(document, key, expected, "child metadata")

    timing_contract = _mapping(
        document.get("timing_contract"), "child metadata.timing_contract"
    )
    expected_timing_contract = dict(TIMING_CONTRACT)
    expected_timing_contract.update(
        {"frames_per_trial": contract["profile_frames"], "trial_count": 1}
    )
    if timing_contract != expected_timing_contract:
        raise BenchmarkContractError(
            "child metadata.timing_contract does not match the whole-sequence contract"
        )
    _require_equal(
        document,
        "aggregate_method",
        "median_across_whole_sequence_trials",
        "child metadata",
    )
    _require_equal(
        document, "primary_metric", SELECTION_OBJECTIVE, "child metadata"
    )
    _require_equal(
        document, "primary_metric_higher_is_better", True, "child metadata"
    )

    model_path = _resolved_string(document.get("model_path"), "model_path")
    source_path = _resolved_string(document.get("source_path"), "source_path")
    if model_path != contract["model_path"]:
        raise BenchmarkContractError(
            "child metadata.model_path mismatch: expected {!r}, got {!r}".format(
                contract["model_path"], model_path
            )
        )
    if source_path != contract["source_path"]:
        raise BenchmarkContractError(
            "child metadata.source_path mismatch: expected {!r}, got {!r}".format(
                contract["source_path"], source_path
            )
        )

    # Fallbacks turn a nominal mode into a different physical benchmark.  They
    # are therefore hard failures even if a child accidentally reports the
    # requested actual_execution_mode as well.
    if candidate["execution_mode"] == "two_stream":
        _require_equal(
            document,
            "two_stream_fallback_reason",
            None,
            "child metadata",
        )
    if candidate["execution_mode"] == "tacker":
        _require_equal(
            document, "tacker_fallback_reason", None, "child metadata"
        )
        _require_equal(
            document,
            "two_stream_fallback_reason",
            None,
            "child metadata",
        )
        qualification_mode = bool(candidate.get("qualification_mode", False))
        _require_equal(
            document,
            "qualification_mode_requested",
            qualification_mode,
            "child metadata",
        )
        _require_equal(
            document,
            "qualification_mode_executed",
            qualification_mode,
            "child metadata",
        )
        if qualification_mode:
            _require_equal(document, "tacker_profile", None, "child metadata")
            reported_profile = _resolved_string(
                document.get("qualification_profile"),
                "qualification_profile",
            )
            profile_label = "qualification_profile"
            profile_hash_key = "qualification_profile_sha256"
            inactive_hash_key = "tacker_profile_sha256"
        else:
            _require_equal(
                document, "qualification_profile", None, "child metadata"
            )
            reported_profile = _resolved_string(
                document.get("tacker_profile"), "tacker_profile"
            )
            profile_label = "tacker_profile"
            profile_hash_key = "tacker_profile_sha256"
            inactive_hash_key = "qualification_profile_sha256"
        if reported_profile != candidate["profile_path"]:
            raise BenchmarkContractError(
                "child metadata.{} mismatch: expected {!r}, got {!r}".format(
                    profile_label, candidate["profile_path"], reported_profile
                )
            )
        manifest_hash = document.get("profile_manifest_sha256")
        if (
            not isinstance(manifest_hash, str)
            or not re.match(r"^[0-9a-f]{64}$", manifest_hash)
        ):
            raise BenchmarkContractError(
                "child metadata.profile_manifest_sha256 must be a lowercase SHA-256"
            )
        selected_variant_id = document.get("selected_variant_id")
        if not isinstance(selected_variant_id, str) or not selected_variant_id:
            raise BenchmarkContractError(
                "child metadata.selected_variant_id must be non-empty for Tacker"
            )
        persistent_blocks = document.get("persistent_blocks")
        if type(persistent_blocks) is not int or persistent_blocks < 0:
            raise BenchmarkContractError(
                "child metadata.persistent_blocks must be an int >= 0 for Tacker"
            )
        for optional_hash_key in (
            "profile_selection_sha256",
            "selected_candidate_abi_sha256",
        ):
            if optional_hash_key not in document:
                raise BenchmarkContractError(
                    "child metadata.{} is required".format(optional_hash_key)
                )
            optional_hash = document[optional_hash_key]
            if optional_hash is not None and (
                not isinstance(optional_hash, str)
                or not re.match(r"^[0-9a-f]{64}$", optional_hash)
            ):
                raise BenchmarkContractError(
                    "child metadata.{} must be null or a lowercase SHA-256"
                    .format(optional_hash_key)
                )
        reported_file_hash = document.get(profile_hash_key)
        if reported_file_hash != candidate["profile_file_sha256"]:
            raise BenchmarkContractError(
                "child metadata.{} does not match the candidate file".format(
                    profile_hash_key
                )
            )
        _require_equal(document, inactive_hash_key, None, "child metadata")
        _require_equal(
            document,
            "active_profile_sha256",
            candidate["profile_file_sha256"],
            "child metadata",
        )
    else:
        for key in (
            "tacker_profile",
            "qualification_profile",
            "active_profile_sha256",
            "tacker_profile_sha256",
            "qualification_profile_sha256",
            "profile_manifest_sha256",
            "profile_selection_sha256",
            "selected_variant_id",
            "selected_candidate_abi_sha256",
            "persistent_blocks",
        ):
            _require_equal(document, key, None, "child metadata")

    _require_equal(document, "trial_count", 1, "child metadata")
    trials = _list(document.get("trials"), "child metadata.trials")
    if len(trials) != 1:
        raise BenchmarkContractError(
            "child metadata.trials must contain exactly one whole-run trial"
        )
    trial = _mapping(trials[0], "child metadata.trials[0]")
    _require_equal(trial, "trial_index", 1, "child metadata.trials[0]")
    _require_equal(trial, "fallback_reason", None, "child metadata.trials[0]")

    elapsed_seconds = _finite(
        trial, "elapsed_seconds", "child metadata.trials[0]", positive=True
    )
    total_render_ms = _finite(
        trial, "total_render_ms", "child metadata.trials[0]", positive=True
    )
    throughput_fps = _finite(
        trial, "throughput_fps", "child metadata.trials[0]", positive=True
    )
    mean_frame_ms = _finite(
        trial, "mean_frame_ms", "child metadata.trials[0]", positive=True
    )
    p50_frame_ms = _finite(
        trial, "p50_frame_ms", "child metadata.trials[0]", nonnegative=True
    )
    p95_frame_ms = _finite(
        trial, "p95_frame_ms", "child metadata.trials[0]", nonnegative=True
    )
    max_frame_ms = _finite(
        trial, "max_frame_ms", "child metadata.trials[0]", nonnegative=True
    )
    cuda_event_total_render_ms = _finite(
        trial,
        "cuda_event_total_render_ms",
        "child metadata.trials[0]",
        nonnegative=True,
    )
    cuda_event_mean_frame_ms = _finite(
        trial,
        "cuda_event_mean_frame_ms",
        "child metadata.trials[0]",
        nonnegative=True,
    )
    if not p50_frame_ms <= p95_frame_ms <= max_frame_ms:
        raise BenchmarkContractError(
            "child frame statistics must satisfy p50 <= p95 <= max"
        )

    completion = _list(
        trial.get("frame_completion_ms"),
        "child metadata.trials[0].frame_completion_ms",
    )
    if len(completion) != contract["profile_frames"]:
        raise BenchmarkContractError(
            "child frame_completion_ms length must equal profile_frames"
        )
    for index, value in enumerate(completion):
        if not _is_finite_number(value) or float(value) < 0.0:
            raise BenchmarkContractError(
                "child frame_completion_ms[{}] must be finite and >= 0".format(
                    index
                )
            )
    cumulative = _list(
        trial.get("cumulative_frame_completion_ms"),
        "child metadata.trials[0].cumulative_frame_completion_ms",
    )
    if len(cumulative) != contract["profile_frames"]:
        raise BenchmarkContractError(
            "child cumulative_frame_completion_ms length must equal profile_frames"
        )
    previous = 0.0
    for index, value in enumerate(cumulative):
        if not _is_finite_number(value) or float(value) < previous:
            raise BenchmarkContractError(
                "child cumulative_frame_completion_ms[{}] must be finite and monotonic"
                .format(index)
            )
        _same_number(
            float(value) - previous,
            completion[index],
            "frame completion interval {}".format(index),
            rel_tol=1e-7,
            abs_tol=1e-5,
        )
        previous = float(value)

    expected_fps = contract["profile_frames"] / elapsed_seconds
    expected_mean_ms = total_render_ms / contract["profile_frames"]
    _same_number(total_render_ms, elapsed_seconds * 1000.0, "total_render_ms")
    _same_number(throughput_fps, expected_fps, "throughput_fps")
    _same_number(mean_frame_ms, expected_mean_ms, "mean_frame_ms")
    _same_number(
        cuda_event_mean_frame_ms,
        sum(float(value) for value in completion) / contract["profile_frames"],
        "cuda_event_mean_frame_ms",
        rel_tol=1e-7,
        abs_tol=1e-5,
    )
    _same_number(
        p50_frame_ms,
        statistics.median(float(value) for value in completion),
        "p50_frame_ms",
        rel_tol=1e-7,
        abs_tol=1e-5,
    )
    ordered_completion = sorted(float(value) for value in completion)
    _same_number(
        p95_frame_ms,
        _percentile(ordered_completion, 0.95),
        "p95_frame_ms",
        rel_tol=1e-7,
        abs_tol=1e-5,
    )
    _same_number(
        max_frame_ms,
        max(ordered_completion),
        "max_frame_ms",
        rel_tol=1e-7,
        abs_tol=1e-5,
    )

    # With one child trial, every top-level compatibility/median field must
    # equal that trial.  This catches accidental use of completion interval p50
    # as the FPS selection objective.
    top_level_pairs = (
        ("elapsed_seconds", elapsed_seconds),
        ("total_render_ms", total_render_ms),
        ("throughput_fps", throughput_fps),
        ("mean_frame_ms", mean_frame_ms),
        ("cuda_event_total_render_ms", cuda_event_total_render_ms),
        ("cuda_event_mean_frame_ms", cuda_event_mean_frame_ms),
        ("p50_frame_ms", p50_frame_ms),
        ("p95_frame_ms", p95_frame_ms),
        ("max_frame_ms", max_frame_ms),
        ("median_elapsed_seconds", elapsed_seconds),
        ("median_total_render_ms", total_render_ms),
        ("median_throughput_fps", throughput_fps),
        ("median_mean_frame_ms", mean_frame_ms),
        ("median_cuda_event_total_render_ms", cuda_event_total_render_ms),
        ("median_cuda_event_mean_frame_ms", cuda_event_mean_frame_ms),
        ("median_p50_frame_ms", p50_frame_ms),
        ("median_p95_frame_ms", p95_frame_ms),
        ("median_max_frame_ms", max_frame_ms),
    )
    for key, expected in top_level_pairs:
        if key not in document:
            raise BenchmarkContractError(
                "child metadata.{} is required".format(key)
            )
        _same_number(document[key], expected, "child metadata.{}".format(key))

    top_completion = _list(
        document.get("frame_completion_ms"),
        "child metadata.frame_completion_ms",
    )
    if top_completion != completion:
        raise BenchmarkContractError(
            "child top-level frame_completion_ms must equal its only trial"
        )
    top_cumulative = _list(
        document.get("cumulative_frame_completion_ms"),
        "child metadata.cumulative_frame_completion_ms",
    )
    if top_cumulative != cumulative:
        raise BenchmarkContractError(
            "child top-level cumulative_frame_completion_ms must equal its only trial"
        )

    environment = _mapping(
        document.get("environment"), "child metadata.environment"
    )
    repository = _mapping(
        document.get("repository"), "child metadata.repository"
    )
    source_files = _mapping(
        document.get("source_files"), "child metadata.source_files"
    )
    for source_name in REQUIRED_PROVENANCE_SOURCE_FILES:
        source_sha256 = source_files.get(source_name)
        if (
            not isinstance(source_sha256, str)
            or not re.match(r"^[0-9a-f]{64}$", source_sha256)
        ):
            raise BenchmarkContractError(
                "child metadata.source_files.{} must be a lowercase SHA-256"
                .format(source_name)
            )
    profile_render_sha256 = source_files["profile_render.py"]
    if document.get("profile_render_sha256") != profile_render_sha256:
        raise BenchmarkContractError(
            "child metadata.profile_render_sha256 disagrees with source_files"
        )
    if repository.get("source_files") != source_files:
        raise BenchmarkContractError(
            "child metadata.repository.source_files disagrees with source_files"
        )
    repository_commit = repository.get("commit")
    if (
        not isinstance(repository_commit, str)
        or not re.match(r"^[0-9a-f]{40}$", repository_commit)
    ):
        raise BenchmarkContractError(
            "child metadata.repository.commit must be a lowercase Git commit"
        )
    repository_commit_source = repository.get("commit_source")
    if not isinstance(repository_commit_source, str) or not repository_commit_source:
        raise BenchmarkContractError(
            "child metadata.repository.commit_source must be non-empty"
        )
    repository_dirty = repository.get("dirty")
    if type(repository_dirty) is not bool:
        raise BenchmarkContractError(
            "child metadata.repository.dirty must be a boolean"
        )
    if "submodules" not in repository or not isinstance(
        repository["submodules"], list
    ):
        raise BenchmarkContractError(
            "child metadata.repository.submodules must be an explicit array"
        )
    if document.get("repository_dirty") is not repository_dirty:
        raise BenchmarkContractError(
            "child metadata.repository_dirty disagrees with repository.dirty"
        )
    profile_hashes = _mapping(
        document.get("profile_hashes"), "child metadata.profile_hashes"
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
        raise BenchmarkContractError(
            "child metadata.profile_hashes fields changed"
        )
    metadata_collection_errors = _list(
        document.get("metadata_collection_errors"),
        "child metadata.metadata_collection_errors",
    )
    if any(not isinstance(value, str) for value in metadata_collection_errors):
        raise BenchmarkContractError(
            "child metadata.metadata_collection_errors must contain strings"
        )
    # nvidia-smi/NVML data is deliberately not a validity gate: PyTorch has
    # already completed the requested CUDA path, and optional telemetry can be
    # unavailable (for example during a driver/library mismatch).  Stable
    # runtime identity is checked using the compatibility fields below.
    stable_environment = {
        "gpu_name": document.get("gpu_name"),
        "cuda_runtime": document.get("cuda_runtime"),
        "pytorch_version": document.get("pytorch_version"),
    }
    for key, value in stable_environment.items():
        if not isinstance(value, str) or not value:
            raise BenchmarkContractError(
                "child metadata.{} must be a non-empty string".format(key)
            )
    for key in ("cuda_runtime", "pytorch_version"):
        if environment.get(key) != stable_environment[key]:
            raise BenchmarkContractError(
                "child environment.{} disagrees with its top-level value".format(
                    key
                )
            )
    environment_gpu = environment.get("gpu")
    if environment_gpu is not None:
        environment_gpu = _mapping(environment_gpu, "child metadata.environment.gpu")
        if environment_gpu.get("name") != stable_environment["gpu_name"]:
            raise BenchmarkContractError(
                "child environment.gpu.name disagrees with gpu_name"
            )
    for key in (
        "tacker_profile_sha256",
        "qualification_profile_sha256",
        "active_profile_sha256",
        "profile_manifest_sha256",
        "profile_selection_sha256",
        "selected_candidate_abi_sha256",
    ):
        if profile_hashes.get(key) != document.get(key):
            raise BenchmarkContractError(
                "child profile_hashes.{} disagrees with its top-level value".format(
                    key
                )
            )

    stable_provenance = {
        "environment": stable_environment,
        "repository_commit": repository_commit,
        "repository_commit_source": repository_commit_source,
        "repository_dirty": repository_dirty,
        "submodules": repository["submodules"],
        "source_files": source_files,
    }

    return {
        "throughput_fps": throughput_fps,
        "elapsed_seconds": elapsed_seconds,
        "total_render_ms": total_render_ms,
        "mean_frame_ms": mean_frame_ms,
        "p50_frame_ms": p50_frame_ms,
        "p95_frame_ms": p95_frame_ms,
        "max_frame_ms": max_frame_ms,
        "cuda_event_total_render_ms": cuda_event_total_render_ms,
        "cuda_event_mean_frame_ms": cuda_event_mean_frame_ms,
        "actual_execution_mode": document.get("actual_execution_mode"),
        "two_stream_fallback_reason": document.get(
            "two_stream_fallback_reason"
        ),
        "tacker_fallback_reason": document.get("tacker_fallback_reason"),
        "qualification_mode_requested": document.get(
            "qualification_mode_requested"
        ),
        "qualification_mode_executed": document.get(
            "qualification_mode_executed"
        ),
        "profile_manifest_sha256": document.get("profile_manifest_sha256"),
        "profile_selection_sha256": document.get("profile_selection_sha256"),
        "selected_variant_id": document.get("selected_variant_id"),
        "selected_candidate_abi_sha256": document.get(
            "selected_candidate_abi_sha256"
        ),
        "persistent_blocks": document.get("persistent_blocks"),
        "stable_environment": stable_environment,
        "stable_provenance": stable_provenance,
        "provenance": {
            "environment": environment,
            "repository": repository,
            "profile_hashes": profile_hashes,
            "metadata_collection_errors": metadata_collection_errors,
        },
    }


def _percentile(sorted_values, probability):
    if not sorted_values:
        raise BenchmarkContractError("cannot compute a percentile of no values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


class _DeterministicGenerator(object):
    """Small SHA-256 counter generator with version-independent sampling."""

    def __init__(self, seed, label):
        self._key = canonical_json_bytes({"seed": seed, "label": label})
        self._counter = 0

    def index(self, stop):
        if stop <= 0:
            raise ValueError("stop must be positive")
        material = self._key + self._counter.to_bytes(16, byteorder="big")
        self._counter += 1
        return int.from_bytes(hashlib.sha256(material).digest(), "big") % stop


def paired_bootstrap_ratio(
    candidate_fps,
    reference_fps,
    resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
    seed=0,
    label="comparison",
):
    """Return paired FPS ratios and a deterministic bootstrap 95% CI.

    Each bootstrap replicate resamples *round indices*, preserving the pairing,
    and evaluates ``median(candidate) / median(reference)``.
    """

    candidate_values = [float(value) for value in candidate_fps]
    reference_values = [float(value) for value in reference_fps]
    if not candidate_values or len(candidate_values) != len(reference_values):
        raise BenchmarkContractError(
            "paired samples must be non-empty and have equal length"
        )
    if type(resamples) is not int or resamples <= 0:
        raise BenchmarkContractError("bootstrap resamples must be positive")
    for value in candidate_values + reference_values:
        if not math.isfinite(value) or value <= 0.0:
            raise BenchmarkContractError("paired FPS values must be finite and > 0")

    paired_ratios = [
        candidate / reference
        for candidate, reference in zip(candidate_values, reference_values)
    ]
    median_fps_ratio = statistics.median(candidate_values) / statistics.median(
        reference_values
    )
    generator = _DeterministicGenerator(seed, label)
    bootstrapped = []
    count = len(candidate_values)
    for _ in range(resamples):
        indices = [generator.index(count) for _ in range(count)]
        sampled_candidate = [candidate_values[index] for index in indices]
        sampled_reference = [reference_values[index] for index in indices]
        bootstrapped.append(
            statistics.median(sampled_candidate)
            / statistics.median(sampled_reference)
        )
    bootstrapped.sort()
    return {
        "paired_fps_ratios": paired_ratios,
        "median_paired_fps_ratio": statistics.median(paired_ratios),
        "median_fps_ratio": median_fps_ratio,
        "paired_bootstrap_95_ci": {
            "lower": _percentile(bootstrapped, 0.025),
            "upper": _percentile(bootstrapped, 0.975),
            "confidence": 0.95,
            "resamples": resamples,
            "seed": seed,
            "statistic": "median(candidate_fps)/median(reference_fps)",
            "resampling_unit": "paired_round",
            "percentile_method": "linear_type_7",
        },
    }


def normalize_correctness_qualifications(candidate_names, qualifications=None):
    """Return explicit, deterministic correctness status for every candidate.

    ``None`` is the Phase-0 compatibility mode: candidates which predate the
    split correctness report are treated as valid.  Once a mapping is
    supplied it must cover the candidate set exactly, so a misspelled or
    omitted qualification cannot silently enter the performance ranking.

    Only the strict boolean ``valid`` member affects selection.  Other members
    (including Raster slowdown and leaf timings) are preserved as diagnostics
    but are deliberately not interpreted here.
    """

    names = list(candidate_names)
    if not names or len(set(names)) != len(names):
        raise BenchmarkContractError(
            "correctness qualifications require unique candidate names"
        )
    if qualifications is None:
        return {
            name: {
                "valid": True,
                "source": "phase0_compatibility_default",
            }
            for name in names
        }
    if not isinstance(qualifications, dict):
        raise BenchmarkContractError(
            "candidate_qualifications must be a JSON object"
        )
    expected = set(names)
    observed = set(qualifications)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise BenchmarkContractError(
            "candidate_qualifications must cover candidates exactly "
            "(missing={!r}, extra={!r})".format(missing, extra)
        )

    normalized = {}
    for name in names:
        entry = qualifications[name]
        if not isinstance(entry, dict):
            raise BenchmarkContractError(
                "candidate_qualifications[{!r}] must be a JSON object".format(
                    name
                )
            )
        if type(entry.get("valid")) is not bool:
            raise BenchmarkContractError(
                "candidate_qualifications[{!r}].valid must be a boolean".format(
                    name
                )
            )
        normalized[name] = dict(entry)
    return normalized


_SELECTION_METADATA_KEYS = (
    "abi_complexity",
    "peak_memory_bytes",
    "registers_per_thread",
    "shared_memory_bytes",
)


def _selection_number(metadata, key, candidate_name):
    """Normalize one optional, lower-is-better tie-break measurement."""

    if key not in metadata:
        return None
    value = metadata[key]
    if not _is_finite_number(value) or float(value) < 0.0:
        raise BenchmarkContractError(
            "candidate_selection_metadata[{!r}].{} must be finite and >= 0"
            .format(candidate_name, key)
        )
    return float(value)


def _optional_selection_key(value):
    # Known measurements sort before missing measurements.  If both are
    # missing the next criterion, and ultimately the candidate name, decides.
    return (1, 0.0) if value is None else (0, value)


def candidate_tie_break_key(candidate_name, candidate_selection_metadata=None):
    """Return the stable Phase-1 lower-is-better equivalence key.

    The optional metadata contract is intentionally small and metric-neutral:
    ``abi_complexity``, ``peak_memory_bytes``, ``registers_per_thread``, and
    ``shared_memory_bytes``.  Missing measurements never make a candidate
    invalid and fall through deterministically to the remaining fields/name.
    """

    all_metadata = (
        {} if candidate_selection_metadata is None else candidate_selection_metadata
    )
    if not isinstance(all_metadata, dict):
        raise BenchmarkContractError(
            "candidate_selection_metadata must be a JSON object"
        )
    metadata = all_metadata.get(candidate_name, {})
    if not isinstance(metadata, dict):
        raise BenchmarkContractError(
            "candidate_selection_metadata[{!r}] must be a JSON object".format(
                candidate_name
            )
        )
    unknown = sorted(set(metadata) - set(_SELECTION_METADATA_KEYS))
    if unknown:
        raise BenchmarkContractError(
            "candidate_selection_metadata[{!r}] contains unknown fields: {!r}"
            .format(candidate_name, unknown)
        )
    values = [
        _selection_number(metadata, key, candidate_name)
        for key in _SELECTION_METADATA_KEYS
    ]
    return tuple(_optional_selection_key(value) for value in values) + (
        candidate_name,
    )


def selection_metadata_with_defaults(
    candidate_names,
    candidate_selection_metadata=None,
    execution_modes=None,
):
    """Inject conservative ABI-complexity defaults for stable tie-breaking."""

    names = list(candidate_names)
    supplied = (
        {} if candidate_selection_metadata is None else candidate_selection_metadata
    )
    if not isinstance(supplied, dict):
        raise BenchmarkContractError(
            "candidate_selection_metadata must be a JSON object"
        )
    unknown = sorted(set(supplied) - set(names))
    if unknown:
        raise BenchmarkContractError(
            "candidate_selection_metadata contains unknown candidates: {!r}".format(
                unknown
            )
        )
    modes = {} if execution_modes is None else execution_modes
    effective = {}
    for name in names:
        entry = supplied.get(name, {})
        if not isinstance(entry, dict):
            raise BenchmarkContractError(
                "candidate_selection_metadata[{!r}] must be a JSON object".format(
                    name
                )
            )
        entry = dict(entry)
        unknown = sorted(set(entry) - set(_SELECTION_METADATA_KEYS))
        if unknown:
            raise BenchmarkContractError(
                "candidate_selection_metadata[{!r}] contains unknown fields: {!r}"
                .format(name, unknown)
            )
        for key in _SELECTION_METADATA_KEYS:
            if key in entry:
                _selection_number(entry, key, name)
        if name == "serial" or modes.get(name) == "serial":
            default_complexity = 0.0
        elif name == "two_stream" or modes.get(name) == "two_stream":
            default_complexity = 1.0
        else:
            # current_tacker and every additional Tacker candidate share the
            # conservative default unless explicit ABI metadata overrides it.
            default_complexity = 2.0
        if "abi_complexity" not in entry:
            entry["abi_complexity"] = default_complexity
        effective[name] = entry
    return effective


def _validate_selection_summary(name, summary):
    if not isinstance(summary, dict):
        raise BenchmarkContractError(
            "summary for candidate {!r} must be a JSON object".format(name)
        )
    median_fps = _finite(
        summary,
        "median_throughput_fps",
        "summaries[{!r}]".format(name),
        positive=True,
    )
    trials = _list(
        summary.get("throughput_fps_trials"),
        "summaries[{!r}].throughput_fps_trials".format(name),
    )
    if not trials:
        raise BenchmarkContractError(
            "summaries[{!r}].throughput_fps_trials must not be empty".format(
                name
            )
        )
    trial_values = []
    for index, value in enumerate(trials):
        if not _is_finite_number(value) or float(value) <= 0.0:
            raise BenchmarkContractError(
                "summaries[{!r}].throughput_fps_trials[{}] must be finite and > 0"
                .format(name, index)
            )
        trial_values.append(float(value))
    _same_number(
        median_fps,
        statistics.median(trial_values),
        "summaries[{!r}].median_throughput_fps".format(name),
    )
    return median_fps, trial_values


def _select_with_valid_incumbent(
    equivalent_candidates,
    fps_by_name,
    trials_by_name,
    incumbent_name,
    promotion_min_ratio,
    bootstrap_resamples,
    seed,
):
    preferred_candidate = equivalent_candidates[0]
    promotion = {
        "incumbent": incumbent_name,
        "incumbent_valid": True,
        "challenger": preferred_candidate,
        "decision": "retain_incumbent",
        "promoted": False,
        "requirements": {
            "median_fps_ratio_min": float(promotion_min_ratio),
            "paired_bootstrap_95_ci_lower_strictly_greater_than": 1.0,
            "not_slower_than": ["two_stream", "current_tacker"],
        },
        "criteria": {},
        "comparison": None,
        "candidate_evaluations": [],
        "reason": None,
        "reason_code": None,
    }
    passing_challengers = []
    for challenger in equivalent_candidates:
        if challenger == incumbent_name:
            continue
        comparison = paired_bootstrap_ratio(
            trials_by_name[challenger],
            trials_by_name[incumbent_name],
            resamples=bootstrap_resamples,
            seed=seed,
            # Use the exact identity emitted by ``aggregate_runs`` and later
            # revalidated by admission/runtime.  A label suffix would change
            # the deterministic bootstrap sample stream and could change a
            # boundary decision for identical trials.
            label="{}-vs-{}".format(challenger, incumbent_name),
        )
        comparison.update(
            {
                "candidate": challenger,
                "reference": incumbent_name,
                "round_indices": list(range(len(trials_by_name[challenger]))),
            }
        )
        median_ratio = comparison["median_fps_ratio"]
        ci_lower = comparison["paired_bootstrap_95_ci"]["lower"]
        ratio_passed = median_ratio >= float(promotion_min_ratio)
        ci_passed = ci_lower > 1.0
        floor_observed = {
            name: fps_by_name[challenger] / fps_by_name[name]
            for name in ("two_stream", "current_tacker")
        }
        floor_passed = all(value >= 1.0 for value in floor_observed.values())
        criteria = {
            "median_fps_ratio": {
                "observed": median_ratio,
                "required_min": float(promotion_min_ratio),
                "passed": ratio_passed,
            },
            "paired_bootstrap_95_ci_lower": {
                "observed": ci_lower,
                "required_strictly_greater_than": 1.0,
                "passed": ci_passed,
            },
            "baseline_fps_ratios": {
                "observed": floor_observed,
                "required_min": 1.0,
                "passed": floor_passed,
            },
        }
        failed = []
        if not ratio_passed:
            failed.append("median_fps_ratio")
        if not ci_passed:
            failed.append("paired_bootstrap_95_ci_lower")
        if not floor_passed:
            failed.append("baseline_fps_floor")
        promotion["candidate_evaluations"].append(
            {
                "candidate": challenger,
                "passed": not failed,
                "failed_criteria": failed,
                "criteria": criteria,
                "comparison": comparison,
            }
        )
        if not failed:
            passing_challengers.append(challenger)

    deployment_winner = incumbent_name
    if passing_challengers:
        deployment_winner = passing_challengers[0]
        selected_evaluation = next(
            item
            for item in promotion["candidate_evaluations"]
            if item["candidate"] == deployment_winner
        )
        promotion["challenger"] = deployment_winner
        promotion["criteria"] = selected_evaluation["criteria"]
        promotion["comparison"] = selected_evaluation["comparison"]
        promotion["decision"] = "promote_challenger"
        promotion["promoted"] = True
        promotion["reason"] = "challenger passed every incumbent promotion gate"
        promotion["reason_code"] = "promotion_gates_passed"
    elif fps_by_name[incumbent_name] < fps_by_name["two_stream"]:
        incumbent_floor_ratio = (
            fps_by_name[incumbent_name] / fps_by_name["two_stream"]
        )
        deployment_winner = "two_stream"
        promotion["challenger"] = "two_stream"
        promotion["decision"] = "selected_two_stream_floor"
        promotion["criteria"] = {
            "incumbent_two_stream_fps_ratio": {
                "observed": incumbent_floor_ratio,
                "required_min": 1.0,
                "passed": False,
            }
        }
        promotion["comparison"] = None
        promotion["reason"] = (
            "incumbent is slower than the correctness-valid two_stream baseline"
        )
        promotion["reason_code"] = "incumbent_below_two_stream"
    elif preferred_candidate == incumbent_name:
        promotion["reason"] = (
            "incumbent is the stable preference within the top FPS equivalence set"
        )
        promotion["reason_code"] = "incumbent_preferred"
    elif promotion["candidate_evaluations"]:
        preferred_evaluation = next(
            item
            for item in promotion["candidate_evaluations"]
            if item["candidate"] == preferred_candidate
        )
        promotion["criteria"] = preferred_evaluation["criteria"]
        promotion["comparison"] = preferred_evaluation["comparison"]
        promotion["reason"] = "challenger failed: {}".format(
            ", ".join(preferred_evaluation["failed_criteria"])
        )
        promotion["reason_code"] = "promotion_gates_failed"
    else:
        promotion["reason"] = "no non-incumbent candidate is in the top FPS set"
        promotion["reason_code"] = "no_challenger"
    return deployment_winner, promotion


def _select_with_invalid_incumbent(
    equivalent_candidates,
    fps_by_name,
    incumbent_name,
):
    """Replace an invalid incumbent without applying stability promotion gates."""

    evaluations = []
    passing_candidates = []
    for candidate in equivalent_candidates:
        two_stream_ratio = fps_by_name[candidate] / fps_by_name["two_stream"]
        floor_passed = two_stream_ratio >= 1.0
        criteria = {
            "two_stream_fps_ratio": {
                "observed": two_stream_ratio,
                "required_min": 1.0,
                "passed": floor_passed,
            }
        }
        evaluations.append(
            {
                "candidate": candidate,
                "passed": floor_passed,
                "failed_criteria": [] if floor_passed else ["two_stream_fps_floor"],
                "criteria": criteria,
                "comparison": None,
            }
        )
        if floor_passed:
            passing_candidates.append(candidate)
    # The exact global FPS argmax is in the equivalence set and cannot be below
    # the valid two-stream baseline, so this is guaranteed by prior validation.
    if not passing_candidates:
        raise BenchmarkContractError(
            "no correctness-valid replacement meets the two_stream FPS floor"
        )
    deployment_winner = passing_candidates[0]
    selected_evaluation = next(
        item for item in evaluations if item["candidate"] == deployment_winner
    )
    promotion = {
        "incumbent": incumbent_name,
        "incumbent_valid": False,
        "challenger": deployment_winner,
        "decision": "replace_invalid_incumbent",
        "promoted": True,
        "requirements": {
            "not_slower_than": ["two_stream"],
            "waived": [
                "median_fps_ratio_min",
                "paired_bootstrap_95_ci_lower",
            ],
            "waiver_reason": "incumbent_invalid",
        },
        "criteria": selected_evaluation["criteria"],
        "comparison": None,
        "candidate_evaluations": evaluations,
        "reason": "incumbent_invalid",
        "reason_code": "incumbent_invalid",
    }
    return deployment_winner, promotion


def select_candidates(
    summaries,
    candidate_qualifications=None,
    candidate_selection_metadata=None,
    incumbent_name="current_tacker",
    candidate_names=None,
    bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
    seed=0,
    promotion_min_ratio=DEFAULT_PROMOTION_MIN_RATIO,
    equivalence_fraction=DEFAULT_EQUIVALENCE_FRACTION,
):
    """Select the experimental argmax and the conservative deployment winner.

    Correctness is a qualification gate; performance and tie-break metadata
    cannot make an invalid candidate eligible.  Conversely, Raster slowdown,
    leaf timing, occupancy, and similar diagnostics are never read by this
    function.  The exact FPS argmax remains visible as ``experimental_winner``.
    A separate 0.5%-equivalent preference supplies the stable promotion
    challenger, ordered by ABI complexity, peak memory, resource usage, then
    name.
    """

    if not isinstance(summaries, dict):
        raise BenchmarkContractError("summaries must be a JSON object")
    names = list(summaries) if candidate_names is None else list(candidate_names)
    if not names or len(set(names)) != len(names):
        raise BenchmarkContractError("candidate_names must be unique and non-empty")
    unknown_summaries = sorted(set(summaries) - set(names))
    if unknown_summaries:
        raise BenchmarkContractError(
            "summaries contain unknown candidates: {!r}".format(unknown_summaries)
        )
    if not isinstance(incumbent_name, str) or incumbent_name not in names:
        raise BenchmarkContractError("incumbent must name a benchmark candidate")
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise BenchmarkContractError("bootstrap_resamples must be positive")
    if not _is_finite_number(promotion_min_ratio) or float(promotion_min_ratio) < 1.0:
        raise BenchmarkContractError("promotion_min_ratio must be finite and >= 1")
    if (
        not _is_finite_number(equivalence_fraction)
        or float(equivalence_fraction) < 0.0
        or float(equivalence_fraction) >= 1.0
    ):
        raise BenchmarkContractError(
            "equivalence_fraction must be finite and in [0, 1)"
        )

    required_baselines = ("serial", "two_stream", "current_tacker")
    missing_baselines = [name for name in required_baselines if name not in names]
    if missing_baselines:
        raise BenchmarkContractError(
            "candidate set is missing required baselines: {!r}".format(
                missing_baselines
            )
        )

    qualifications = normalize_correctness_qualifications(
        names, candidate_qualifications
    )
    invalid_safety_baselines = [
        name
        for name in ("serial", "two_stream")
        if not qualifications[name]["valid"]
    ]
    if invalid_safety_baselines:
        raise BenchmarkContractError(
            "serial and two_stream baselines must be correctness-valid: {!r}".format(
                invalid_safety_baselines
            )
        )
    eligible = [name for name in names if qualifications[name]["valid"]]
    if not eligible:
        raise BenchmarkContractError("no correctness-valid candidate is available")

    candidate_selection_metadata = selection_metadata_with_defaults(
        names, candidate_selection_metadata
    )

    fps_by_name = {}
    trials_by_name = {}
    for name in eligible:
        if name not in summaries:
            raise BenchmarkContractError(
                "correctness-valid candidate {!r} has no FPS summary".format(name)
            )
        median_fps, trial_values = _validate_selection_summary(
            name, summaries[name]
        )
        fps_by_name[name] = median_fps
        trials_by_name[name] = trial_values

    eligible_ranking = sorted(
        eligible,
        key=lambda name: (-fps_by_name[name], name),
    )
    experimental_winner = eligible_ranking[0]
    top_fps = fps_by_name[experimental_winner]
    equivalence_limit = float(equivalence_fraction)
    equivalent_candidates = [
        name
        for name in eligible
        if (top_fps - fps_by_name[name]) / top_fps <= equivalence_limit
    ]
    equivalent_candidates.sort(
        key=lambda name: candidate_tie_break_key(
            name, candidate_selection_metadata
        )
    )
    preferred_candidate = equivalent_candidates[0]
    if qualifications[incumbent_name]["valid"]:
        deployment_winner, promotion = _select_with_valid_incumbent(
            equivalent_candidates,
            fps_by_name,
            trials_by_name,
            incumbent_name,
            promotion_min_ratio,
            bootstrap_resamples,
            seed,
        )
    else:
        deployment_winner, promotion = _select_with_invalid_incumbent(
            equivalent_candidates,
            fps_by_name,
            incumbent_name,
        )

    return {
        "selection_objective": SELECTION_OBJECTIVE,
        "correctness_qualifications": qualifications,
        "candidate_selection_metadata": candidate_selection_metadata,
        "diagnostics_excluded_from_selection": [
            "raster_slowdown_pct",
            "mixed_p50_ms",
            "solo_leaf_sum_p50_ms",
            "leaf_savings_ms",
            "frame_latency_p50_p95_max",
            "occupancy",
        ],
        "eligible_candidates": eligible,
        "excluded_candidates": [
            {
                "name": name,
                "reason": "correctness_invalid",
            }
            for name in names
            if not qualifications[name]["valid"]
        ],
        "eligible_ranking": eligible_ranking,
        "performance_argmax": experimental_winner,
        "experimental_winner": experimental_winner,
        "equivalence": {
            "fraction": equivalence_limit,
            "top_median_throughput_fps": top_fps,
            "candidates": equivalent_candidates,
            "preferred_candidate": preferred_candidate,
            "tie_break_order": [
                "abi_complexity",
                "peak_memory_bytes",
                "registers_per_thread",
                "shared_memory_bytes",
                "name",
            ],
            "default_abi_complexity": {
                "serial": 0.0,
                "two_stream": 1.0,
                "tacker": 2.0,
            },
            "missing_metadata_policy": "known_first_then_name",
        },
        "deployment_winner": deployment_winner,
        "promotion": promotion,
    }


def aggregate_runs(runs, candidate_names, trials, bootstrap_resamples, seed):
    """Aggregate complete validated runs into summaries and comparisons."""

    by_candidate = {name: {} for name in candidate_names}
    for run in runs:
        if run.get("passed") is not True:
            raise BenchmarkContractError("cannot aggregate a failed child run")
        name = run["candidate_name"]
        round_index = run["round_index"]
        if name not in by_candidate:
            raise BenchmarkContractError("run references an unknown candidate")
        if round_index in by_candidate[name]:
            raise BenchmarkContractError(
                "candidate {!r} has duplicate round {}".format(name, round_index)
            )
        by_candidate[name][round_index] = run

    expected_rounds = set(range(trials))
    summaries = {}
    for name in candidate_names:
        observed_rounds = set(by_candidate[name])
        if observed_rounds != expected_rounds:
            raise BenchmarkContractError(
                "candidate {!r} does not have one run in every round".format(name)
            )
        ordered = [by_candidate[name][index] for index in range(trials)]
        fps_values = [item["metrics"]["throughput_fps"] for item in ordered]
        elapsed_values = [item["metrics"]["elapsed_seconds"] for item in ordered]
        summaries[name] = {
            "trial_count": len(ordered),
            "round_indices": list(range(trials)),
            "throughput_fps_trials": fps_values,
            "median_throughput_fps": statistics.median(fps_values),
            "min_throughput_fps": min(fps_values),
            "max_throughput_fps": max(fps_values),
            "elapsed_seconds_trials": elapsed_values,
            "median_total_render_ms": statistics.median(
                item["metrics"]["total_render_ms"] for item in ordered
            ),
            "metadata_paths": [item["metadata_path"] for item in ordered],
            "profile_manifest_sha256_values": sorted(
                set(
                    item["metrics"]["profile_manifest_sha256"]
                    for item in ordered
                    if item["metrics"]["profile_manifest_sha256"] is not None
                )
            ),
        }

    ranking = sorted(
        candidate_names,
        key=lambda name: (-summaries[name]["median_throughput_fps"], name),
    )
    comparisons = []
    references = [
        name for name in ("two_stream", "current_tacker") if name in summaries
    ]
    for reference in references:
        reference_fps = summaries[reference]["throughput_fps_trials"]
        for candidate_name in candidate_names:
            if candidate_name == reference:
                continue
            candidate_fps = summaries[candidate_name]["throughput_fps_trials"]
            comparison_label = "{}-vs-{}".format(candidate_name, reference)
            comparison = paired_bootstrap_ratio(
                candidate_fps,
                reference_fps,
                resamples=bootstrap_resamples,
                seed=seed,
                label=comparison_label,
            )
            comparison.update(
                {
                    "candidate": candidate_name,
                    "reference": reference,
                    "round_indices": list(range(trials)),
                }
            )
            comparisons.append(comparison)
    return summaries, ranking, comparisons


def parse_candidate_spec(spec):
    if not isinstance(spec, str) or "=" not in spec:
        raise BenchmarkContractError(
            "candidate must use NAME=PROFILE_PATH syntax"
        )
    name, profile_path = spec.split("=", 1)
    if not NAME_PATTERN.match(name):
        raise BenchmarkContractError(
            "candidate name must match {}".format(NAME_PATTERN.pattern)
        )
    if name in RESERVED_CANDIDATE_NAMES:
        raise BenchmarkContractError(
            "candidate name {!r} is reserved".format(name)
        )
    if not profile_path:
        raise BenchmarkContractError("candidate profile path must not be empty")
    return name, profile_path


def _candidate_qualification_mode(profile, candidate_name):
    """Detect disabled schema-v2 profiles from the already-hashed snapshot."""

    if not isinstance(profile, dict):
        raise BenchmarkContractError(
            "candidate {!r} profile must be a JSON object".format(candidate_name)
        )
    if profile.get("schema_version") != 2:
        # Schema v1 and pre-schema Phase-0 test profiles use the normal
        # --tacker-profile path.  Their full runtime contract is validated by
        # profile_render/TackerRenderer rather than duplicated here.
        return False
    deployment = profile.get("deployment")
    if not isinstance(deployment, dict) or type(deployment.get("enabled")) is not bool:
        raise BenchmarkContractError(
            "schema-v2 candidate {!r} requires deployment.enabled boolean".format(
                candidate_name
            )
        )
    if candidate_name == "current_tacker" and not deployment["enabled"]:
        raise BenchmarkContractError(
            "current_tacker schema-v2 incumbent must be deployment-enabled"
        )
    return not deployment["enabled"]


def make_candidates(current_tacker_profile, candidate_specs):
    current_path = Path(current_tacker_profile).expanduser().resolve()
    candidates = [
        {
            "name": "serial",
            "execution_mode": "serial",
            "profile_path": None,
            "qualification_mode": False,
        },
        {
            "name": "two_stream",
            "execution_mode": "two_stream",
            "profile_path": None,
            "qualification_mode": False,
        },
        {
            "name": "current_tacker",
            "execution_mode": "tacker",
            "profile_path": str(current_path),
            "qualification_mode": False,
        },
    ]
    seen = set(item["name"] for item in candidates)
    for spec in candidate_specs or []:
        name, raw_path = parse_candidate_spec(spec)
        if name in seen:
            raise BenchmarkContractError(
                "duplicate candidate name {!r}".format(name)
            )
        seen.add(name)
        candidates.append(
            {
                "name": name,
                "execution_mode": "tacker",
                "profile_path": str(Path(raw_path).expanduser().resolve()),
                "qualification_mode": False,
            }
        )
    for candidate in candidates:
        profile_path = candidate["profile_path"]
        if profile_path is None:
            candidate["profile_file_sha256"] = None
            continue
        if not Path(profile_path).is_file():
            raise BenchmarkContractError(
                "candidate {!r} profile does not exist: {}".format(
                    candidate["name"], profile_path
                )
            )
        profile_snapshot, profile_file_sha256 = load_json_mapping_snapshot(
            profile_path,
            "candidate {!r} profile".format(candidate["name"]),
        )
        candidate["qualification_mode"] = _candidate_qualification_mode(
            profile_snapshot, candidate["name"]
        )
        candidate["profile_file_sha256"] = profile_file_sha256
    return candidates


def validate_profile_args(profile_args):
    result = []
    for value in profile_args or []:
        if not isinstance(value, str) or not value:
            raise BenchmarkContractError("profile arguments must be non-empty strings")
        option = value.split("=", 1)[0]
        if option in CONTROLLED_PROFILE_OPTIONS:
            raise BenchmarkContractError(
                "profile argument {!r} is controlled by the benchmark driver".format(
                    option
                )
            )
        if value == "--":
            raise BenchmarkContractError("a bare -- is not a profile argument")
        result.append(value)
    return result


def make_contract(
    model_path,
    source_path,
    workload_name,
    iteration,
    split,
    warmup_frames,
    profile_frames,
    image_width,
    image_height,
    gaussian_count,
    view_indices=None,
):
    integer_fields = {
        "iteration": iteration,
        "warmup_frames": warmup_frames,
        "profile_frames": profile_frames,
        "image_width": image_width,
        "image_height": image_height,
        "gaussian_count": gaussian_count,
    }
    for key, value in integer_fields.items():
        if type(value) is not int:
            raise BenchmarkContractError("{} must be an integer".format(key))
    if profile_frames <= 0:
        raise BenchmarkContractError("profile_frames must be positive")
    if warmup_frames < 0:
        raise BenchmarkContractError("warmup_frames must be non-negative")
    if image_width <= 0 or image_height <= 0 or gaussian_count <= 0:
        raise BenchmarkContractError(
            "image dimensions and gaussian_count must be positive"
        )
    if split not in ("train", "test", "video"):
        raise BenchmarkContractError("split must be train, test, or video")
    if not isinstance(workload_name, str) or not workload_name:
        raise BenchmarkContractError("workload_name must be non-empty")
    if view_indices is None:
        view_indices = list(range(profile_frames))
    else:
        view_indices = list(view_indices)
    if (
        len(view_indices) != profile_frames
        or any(type(index) is not int or index < 0 for index in view_indices)
    ):
        raise BenchmarkContractError(
            "view_indices must contain profile_frames non-negative integers"
        )
    return {
        "model_path": str(Path(model_path).expanduser().resolve()),
        "source_path": str(Path(source_path).expanduser().resolve()),
        "workload_name": workload_name,
        "iteration": iteration,
        "split": split,
        "warmup_frames": warmup_frames,
        "profile_frames": profile_frames,
        "view_indices": view_indices,
        "image_width": image_width,
        "image_height": image_height,
        "gaussian_count": gaussian_count,
        "timing_method": TIMING_METHOD,
        "frame_timing_method": FRAME_TIMING_METHOD,
        "io_in_timed_region": False,
        "throughput_definition": "profile_frames / elapsed_seconds",
    }


def build_child_command(
    python_executable,
    profile_render_path,
    candidate,
    contract,
    metadata_path,
    configs=None,
    profile_args=None,
):
    command = [str(python_executable), str(profile_render_path)]
    command.extend(validate_profile_args(profile_args))
    command.extend(
        [
            "--model_path",
            contract["model_path"],
            "--source_path",
            contract["source_path"],
            "--iteration",
            str(contract["iteration"]),
            "--split",
            contract["split"],
            "--warmup",
            str(contract["warmup_frames"]),
            "--frames",
            str(contract["profile_frames"]),
            "--trials",
            "1",
            "--workload-name",
            contract["workload_name"],
            "--execution-mode",
            candidate["execution_mode"],
            "--metadata",
            str(metadata_path),
            "--quiet",
        ]
    )
    if configs is not None:
        command.extend(["--configs", str(configs)])
    if candidate["execution_mode"] == "tacker":
        if candidate.get("qualification_mode", False):
            command.extend(
                [
                    "--qualification-mode",
                    "--qualification-profile",
                    candidate["profile_path"],
                ]
            )
        else:
            command.extend(["--tacker-profile", candidate["profile_path"]])
    return command


def _load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise BenchmarkContractError(
            "cannot read child metadata {}: {}".format(path, error)
        )


def _safe_run_stem(schedule_item):
    return "{:04d}-round{:03d}-{}".format(
        schedule_item["run_index"],
        schedule_item["round_index"],
        schedule_item["candidate_name"],
    )


def _completed_output(completed, name):
    value = getattr(completed, name, "")
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_interleaved_benchmark(
    candidates,
    contract,
    trials,
    strategy,
    seed,
    bootstrap_resamples,
    session_dir,
    profile_render_path,
    python_executable=sys.executable,
    project_root=None,
    configs=None,
    profile_args=None,
    timeout_seconds=None,
    runner=None,
    candidate_qualifications=None,
    candidate_selection_metadata=None,
    incumbent_name="current_tacker",
    promotion_min_ratio=DEFAULT_PROMOTION_MIN_RATIO,
    equivalence_fraction=DEFAULT_EQUIVALENCE_FRACTION,
):
    """Execute the benchmark and return a report, including on child failure."""

    if type(trials) is not int or trials < 2:
        raise BenchmarkContractError("trials must be at least 2 for paired inference")
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise BenchmarkContractError("bootstrap_resamples must be positive")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise BenchmarkContractError("timeout_seconds must be positive")
    runner = subprocess.run if runner is None else runner
    profile_render = Path(profile_render_path).expanduser().resolve()
    if not profile_render.is_file():
        raise BenchmarkContractError(
            "profile_render.py does not exist: {}".format(profile_render)
        )
    root = (
        profile_render.parent
        if project_root is None
        else Path(project_root).expanduser().resolve()
    )
    session = Path(session_dir).expanduser().resolve()
    candidate_by_name = {item["name"]: item for item in candidates}
    if len(candidate_by_name) != len(candidates):
        raise BenchmarkContractError("candidate names must be unique")
    candidate_names = [item["name"] for item in candidates]
    correctness_qualifications = normalize_correctness_qualifications(
        candidate_names, candidate_qualifications
    )
    # All three baselines must exist.  serial/two_stream are safety fallbacks
    # and must remain valid; current_tacker may be invalid, in which case it is
    # excluded and replaced without applying the normal stability threshold.
    required_baselines = ("serial", "two_stream", "current_tacker")
    missing_baselines = [
        name for name in required_baselines if name not in candidate_by_name
    ]
    invalid_safety_baselines = [
        name
        for name in ("serial", "two_stream")
        if name in correctness_qualifications
        and not correctness_qualifications[name]["valid"]
    ]
    if missing_baselines:
        raise BenchmarkContractError(
            "candidate set is missing required baselines: {!r}".format(
                missing_baselines
            )
        )
    if invalid_safety_baselines:
        raise BenchmarkContractError(
            "serial and two_stream baselines must be correctness-valid: {!r}".format(
                invalid_safety_baselines
            )
        )
    if incumbent_name not in candidate_by_name:
        raise BenchmarkContractError("incumbent must name a benchmark candidate")
    selection_metadata = selection_metadata_with_defaults(
        candidate_names,
        candidate_selection_metadata,
        execution_modes={
            item["name"]: item["execution_mode"] for item in candidates
        },
    )
    for name in candidate_names:
        candidate_tie_break_key(name, selection_metadata)
    eligible_names = [
        name for name in candidate_names if correctness_qualifications[name]["valid"]
    ]
    schedule = build_schedule(eligible_names, trials, strategy, seed)
    session.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "passed": False,
        "selection_objective": SELECTION_OBJECTIVE,
        "generated_at_utc": utc_now(),
        "contract": contract,
        "candidates": candidates,
        "correctness_qualifications": correctness_qualifications,
        "candidate_selection_metadata": selection_metadata,
        "eligible_candidates": eligible_names,
        "excluded_candidates": [
            {
                "name": name,
                "reason": "correctness_invalid",
            }
            for name in candidate_names
            if not correctness_qualifications[name]["valid"]
        ],
        "schedule": {
            "strategy": strategy,
            "seed": seed,
            "trials_per_candidate": trials,
            "base_order": _stable_order(eligible_names, seed),
            "executions": schedule,
        },
        "bootstrap": {
            "confidence": 0.95,
            "resamples": bootstrap_resamples,
            "seed": seed,
            "resampling_unit": "paired_round",
        },
        "phase0_exit_condition": {
            "required_trials_per_candidate": 10,
            "required_frames_per_trial": 50,
            "met": bool(trials >= 10 and contract["profile_frames"] == 50),
        },
        "artifacts": {
            "session_dir": str(session),
            "driver_sha256": sha256_file(Path(__file__).resolve()),
            "profile_render_sha256": sha256_file(profile_render),
        },
        "runs": [],
        "summaries": {},
        "ranking": [],
        "eligible_ranking": [],
        "experimental_winner": None,
        "deployment_winner": None,
        "promotion": None,
        "selection": None,
        "paired_comparisons": [],
        "errors": [],
    }

    stable_environment = None
    stable_provenance = None
    for schedule_item in schedule:
        candidate = candidate_by_name[schedule_item["candidate_name"]]
        stem = _safe_run_stem(schedule_item)
        metadata_path = session / "{}.metadata.json".format(stem)
        stdout_path = session / "{}.stdout.txt".format(stem)
        stderr_path = session / "{}.stderr.txt".format(stem)
        command = build_child_command(
            python_executable,
            profile_render,
            candidate,
            contract,
            metadata_path,
            configs=configs,
            profile_args=profile_args,
        )
        run_record = dict(schedule_item)
        run_record.update(
            {
                "requested_execution_mode": candidate["execution_mode"],
                "profile_path": candidate["profile_path"],
                "qualification_mode": bool(
                    candidate.get("qualification_mode", False)
                ),
                "command": command,
                "metadata_path": str(metadata_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "passed": False,
                "returncode": None,
                "error": None,
            }
        )
        report["runs"].append(run_record)
        try:
            completed = runner(
                command,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
            run_record["returncode"] = int(completed.returncode)
            atomic_write_text(stdout_path, _completed_output(completed, "stdout"))
            atomic_write_text(stderr_path, _completed_output(completed, "stderr"))
            if completed.returncode != 0:
                raise BenchmarkContractError(
                    "child exited with status {}".format(completed.returncode)
                )
            metadata, metadata_hash = load_json_mapping_snapshot(
                metadata_path, "child metadata"
            )
            metrics = validate_child_metadata(metadata, candidate, contract)
            if stable_environment is None:
                stable_environment = metrics["stable_environment"]
            elif metrics["stable_environment"] != stable_environment:
                raise BenchmarkContractError(
                    "stable GPU/CUDA/PyTorch environment changed between runs"
                )
            if stable_provenance is None:
                stable_provenance = metrics["stable_provenance"]
            elif metrics["stable_provenance"] != stable_provenance:
                raise BenchmarkContractError(
                    "repository/submodule/source provenance changed between runs"
                )
            run_record["metrics"] = metrics
            run_record["metadata_sha256"] = metadata_hash
            run_record["passed"] = True
        except subprocess.TimeoutExpired as error:
            atomic_write_text(stdout_path, _completed_output(error, "stdout"))
            atomic_write_text(stderr_path, _completed_output(error, "stderr"))
            run_record["error"] = "child timed out after {} seconds".format(
                timeout_seconds
            )
            report["errors"].append(
                "run {} ({}): {}".format(
                    schedule_item["run_index"],
                    candidate["name"],
                    run_record["error"],
                )
            )
            break
        except Exception as error:
            # Preserve the expected paths even when the child never created its
            # metadata.  This makes failed remote runs straightforward to audit.
            if not stdout_path.exists():
                atomic_write_text(stdout_path, "")
            if not stderr_path.exists():
                atomic_write_text(stderr_path, "")
            run_record["error"] = str(error)
            report["errors"].append(
                "run {} ({}): {}".format(
                    schedule_item["run_index"],
                    candidate["name"],
                    run_record["error"],
                )
            )
            break

    report["completed_execution_count"] = sum(
        1 for item in report["runs"] if item["passed"]
    )
    report["expected_execution_count"] = len(schedule)
    report["stable_environment"] = stable_environment
    report["stable_provenance"] = stable_provenance
    if not report["errors"] and len(report["runs"]) == len(schedule):
        try:
            summaries, ranking, comparisons = aggregate_runs(
                report["runs"],
                eligible_names,
                trials,
                bootstrap_resamples,
                seed,
            )
            selection = select_candidates(
                summaries,
                candidate_qualifications=correctness_qualifications,
                candidate_selection_metadata=selection_metadata,
                incumbent_name=incumbent_name,
                candidate_names=candidate_names,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
                promotion_min_ratio=promotion_min_ratio,
                equivalence_fraction=equivalence_fraction,
            )
            report["summaries"] = summaries
            report["ranking"] = ranking
            report["eligible_ranking"] = selection["eligible_ranking"]
            report["experimental_winner"] = selection["experimental_winner"]
            report["deployment_winner"] = selection["deployment_winner"]
            report["promotion"] = selection["promotion"]
            report["selection"] = selection
            report["paired_comparisons"] = comparisons
            report["passed"] = True
        except Exception as error:
            report["errors"].append("aggregation failed: {}".format(error))
    return report


def _parse_view_indices(value, frames):
    if value is None:
        return list(range(frames))
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        raise BenchmarkContractError(
            "--expected-view-indices must be comma-separated integers"
        )
    if len(indices) != frames or any(index < 0 for index in indices):
        raise BenchmarkContractError(
            "--expected-view-indices must contain exactly --frames non-negative integers"
        )
    return indices


def _parser():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Interleave whole-run FPS trials for Tacker candidates"
    )
    parser.add_argument("--output", required=True, help="aggregate JSON report")
    parser.add_argument(
        "--runs-dir",
        help="artifact parent (default: <output stem>-runs beside the report)",
    )
    parser.add_argument(
        "--run-id",
        help="optional unique artifact directory name; must not already exist",
    )
    parser.add_argument(
        "--profile-render",
        default=str(project_root / "profile_render.py"),
        help="profile_render.py path",
    )
    parser.add_argument(
        "--python-executable", default=sys.executable, help="Python executable"
    )
    parser.add_argument("--current-tacker-profile", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=PROFILE",
        help="additional Tacker candidate; repeat for multiple candidates",
    )
    parser.add_argument(
        "--correctness-json",
        required=True,
        help=(
            "candidate correctness mapping; accepts name-to-object "
            "or a document whose candidates member is that mapping"
        ),
    )
    parser.add_argument(
        "--selection-metadata-json",
        help="optional candidate metadata mapping for stable equivalence tie-breaks",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument(
        "--configs",
        required=True,
        help="exact experiment config; its content hash is part of provenance",
    )
    parser.add_argument("--workload-name", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--split", choices=("train", "test", "video"), default="test"
    )
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--expected-image-width", type=int, required=True)
    parser.add_argument("--expected-image-height", type=int, required=True)
    parser.add_argument("--expected-gaussian-count", type=int, required=True)
    parser.add_argument(
        "--expected-view-indices",
        help="comma-separated exact view sequence (default: 0..frames-1)",
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--schedule",
        choices=("round_robin", "abba"),
        default="abba",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--profile-arg",
        action="append",
        default=[],
        help=(
            "one extra profile_render.py argv item; use --profile-arg=VALUE and "
            "repeat (driver-owned options are rejected)"
        ),
    )
    return parser


def _session_dir(args, output_path):
    if args.runs_dir:
        runs_parent = Path(args.runs_dir).expanduser().resolve()
    else:
        runs_parent = output_path.parent / "{}-runs".format(output_path.stem)
    if args.run_id:
        if not NAME_PATTERN.match(args.run_id):
            raise BenchmarkContractError(
                "--run-id must match {}".format(NAME_PATTERN.pattern)
            )
        run_id = args.run_id
    else:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        run_id = "{}-{}-{}".format(timestamp, os.getpid(), uuid.uuid4().hex[:8])
    return runs_parent / run_id


def main(argv=None):
    args = _parser().parse_args(argv)
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        print(
            "refusing to overwrite existing FPS benchmark report: {}".format(
                output_path
            ),
            file=sys.stderr,
        )
        return 1
    selection_inputs = {}
    try:
        view_indices = _parse_view_indices(args.expected_view_indices, args.frames)
        contract = make_contract(
            args.model_path,
            args.source_path,
            args.workload_name,
            args.iteration,
            args.split,
            args.warmup,
            args.frames,
            args.expected_image_width,
            args.expected_image_height,
            args.expected_gaussian_count,
            view_indices=view_indices,
        )
        candidates = make_candidates(
            args.current_tacker_profile, args.candidate
        )
        candidate_qualifications, correctness_digest = (
            load_correctness_qualifications(
                args.correctness_json, include_sha256=True
            )
        )
        selection_inputs["correctness_json"] = {
            "path": str(Path(args.correctness_json).expanduser().resolve()),
            "sha256": correctness_digest,
        }
        if args.selection_metadata_json is None:
            candidate_selection_metadata = None
        else:
            candidate_selection_metadata, selection_metadata_digest = (
                load_candidate_selection_metadata(
                    args.selection_metadata_json, include_sha256=True
                )
            )
            selection_inputs["selection_metadata_json"] = {
                "path": str(
                    Path(args.selection_metadata_json).expanduser().resolve()
                ),
                "sha256": selection_metadata_digest,
            }
        profile_args = validate_profile_args(args.profile_arg)
        session_dir = _session_dir(args, output_path)
        report = run_interleaved_benchmark(
            candidates,
            contract,
            args.trials,
            args.schedule,
            args.seed,
            args.bootstrap_resamples,
            session_dir,
            args.profile_render,
            python_executable=args.python_executable,
            project_root=Path(args.profile_render).expanduser().resolve().parent,
            configs=(
                str(Path(args.configs).expanduser().resolve())
                if args.configs is not None
                else None
            ),
            profile_args=profile_args,
            timeout_seconds=args.timeout_seconds,
            candidate_qualifications=candidate_qualifications,
            candidate_selection_metadata=candidate_selection_metadata,
        )
    except Exception as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "passed": False,
            "selection_objective": SELECTION_OBJECTIVE,
            "generated_at_utc": utc_now(),
            "runs": [],
            "summaries": {},
            "ranking": [],
            "eligible_ranking": [],
            "experimental_winner": None,
            "deployment_winner": None,
            "promotion": None,
            "selection": None,
            "paired_comparisons": [],
            "errors": [str(error)],
        }
    report["selection_inputs"] = selection_inputs
    report["artifacts"] = dict(report.get("artifacts", {}))
    report["artifacts"]["report_path"] = str(output_path)
    try:
        atomic_write_json_no_clobber(output_path, report)
    except FileExistsError:
        print(
            "refusing to overwrite FPS benchmark report created during run: {}"
            .format(output_path),
            file=sys.stderr,
        )
        return 1
    if report.get("passed"):
        winner = report["experimental_winner"]
        winner_fps = report["summaries"][winner]["median_throughput_fps"]
        print(
            "FPS benchmark passed: {} is the experimental winner at median "
            "{:.3f} FPS; deployment winner: {}; report: {}".format(
                winner,
                winner_fps,
                report["deployment_winner"],
                output_path,
            )
        )
        return 0
    print(
        "FPS benchmark failed closed; report: {}".format(output_path),
        file=sys.stderr,
    )
    for error in report.get("errors", []):
        print("  {}".format(error), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
