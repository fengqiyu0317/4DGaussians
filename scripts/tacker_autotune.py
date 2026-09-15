#!/usr/bin/env python3
"""Deterministic candidate search and resumable profiling state for Tacker.

This module intentionally has no top-level dependency on PyTorch, CUDA, or
``gaussian_renderer``.  Candidate *specifications* can therefore be generated,
ranked, hashed, and persisted on a CPU-only coordinator.  Profile descriptors
are materialized only when :func:`materialize_candidate_descriptor` is called;
the v2 descriptor is always produced by
``gaussian_renderer.tacker_pipeline.first_linear_candidate_contract``.

The SQLite database is a coordinator/checkpoint database, not an admission
profile.  A successful stage is reusable only for an exact matrix hash,
candidate hash, stage name, input hash, and still-matching artifact hash.
"""

import argparse
from copy import deepcopy
import datetime
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import tempfile
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MATRIX_SCHEMA_VERSION = 1
CANDIDATE_SCHEMA_VERSION = 1
PHASE31_MATRIX_SCHEMA_VERSION = 2
PHASE31_CANDIDATE_SCHEMA_VERSION = 2
DB_SCHEMA_VERSION = 3
MATRIX_KIND = "tacker_autotune_candidate_matrix"

HEAD_ORDER = ("pos", "scales", "rotations", "opacity", "shs")
LEGACY_ABI_FAMILY = "legacy_pos_l1_v1"
FIRST_LINEAR_ABI_FAMILY = "first_linear_heads_v2"
PACKED_FIRST_LINEAR_ABI_FAMILY = "packed_first_linear_v3"
WHOLE_HEAD_ABI_FAMILY = "whole_heads_v4"

SEARCH_FAMILY_C0 = "c0"
SEARCH_FAMILY_C1 = "c1"
SEARCH_FAMILY_C2 = "c2"
SEARCH_FAMILY_C3 = "c3"
SEARCH_FAMILY_C4 = "c4"
SEARCH_FAMILIES = (
    SEARCH_FAMILY_C0,
    SEARCH_FAMILY_C1,
    SEARCH_FAMILY_C2,
    SEARCH_FAMILY_C3,
    SEARCH_FAMILY_C4,
)

STAGE_RUNNING = "running"
STAGE_SUCCEEDED = "succeeded"
STAGE_FAILED = "failed"
STAGE_STATUSES = (STAGE_RUNNING, STAGE_SUCCEEDED, STAGE_FAILED)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CLAIM_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
_STAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MISSING = object()


class AutotuneError(ValueError):
    """Base class for fail-closed autotuner contract errors."""


class ProfileDBError(AutotuneError):
    """Base class for profile database errors."""


class ProfileDBCorruptionError(ProfileDBError):
    """The database or a referenced artifact failed integrity validation."""


class ProfileDBConflictError(ProfileDBError):
    """Existing state conflicts with the requested transition."""


def _assert_json_value(value, section="value"):
    """Reject values whose JSON representation is ambiguous or non-canonical."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AutotuneError("{} contains a non-finite number".format(section))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_value(item, "{}[{}]".format(section, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AutotuneError("{} has a non-string object key".format(section))
            _assert_json_value(item, "{}.{}".format(section, key))
        return
    raise AutotuneError(
        "{} contains unsupported JSON type {}".format(
            section, type(value).__name__
        )
    )


def canonical_json_bytes(value):
    """Return the unique compact UTF-8 representation used for all hashes."""

    _assert_json_value(value)
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise AutotuneError("value is not canonical JSON: {}".format(error))
    return rendered.encode("utf-8")


def canonical_sha256(value, domain=None):
    """Hash canonical JSON with optional domain separation."""

    payload = value
    if domain is not None:
        if not isinstance(domain, str) or not domain:
            raise AutotuneError("hash domain must be a non-empty string")
        payload = {"domain": domain, "payload": value}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def input_sha256(value):
    """Return a domain-separated stage input hash."""

    return canonical_sha256(value, "tacker-autotune-stage-input-v1")


def _require_sha256(value, section):
    if not isinstance(value, str) or _SHA256_RE.match(value) is None:
        raise AutotuneError("{} must be a lowercase SHA-256".format(section))
    return value


def _normalize_head_order(head_order):
    if not isinstance(head_order, (list, tuple)) or not head_order:
        raise AutotuneError("head_order must be a non-empty list")
    result = []
    for name in head_order:
        if not isinstance(name, str) or not name:
            raise AutotuneError("head_order entries must be non-empty strings")
        if name in result:
            raise AutotuneError("head_order entries must be unique")
        result.append(name)
    if tuple(result) != HEAD_ORDER:
        raise AutotuneError(
            "head_order must exactly match runtime HEAD_ORDER {!r}".format(
                HEAD_ORDER
            )
        )
    return tuple(result)


def normalize_persistent_blocks(values):
    """Validate, deduplicate, and numerically sort persistent block counts."""

    if not isinstance(values, (list, tuple, set)):
        raise AutotuneError("persistent_blocks must be a sequence")
    result = set()
    for value in values:
        if type(value) is not int or value < 0:
            raise AutotuneError(
                "persistent block values must be integers greater than or equal to 0"
            )
        result.add(value)
    if not result:
        raise AutotuneError("persistent_blocks must not be empty")
    return tuple(sorted(result))


def derive_persistent_blocks(
    sm_count,
    raster_tile_count,
    backend_logical_blocks,
    current_persistent_blocks=7000,
    extra_values=None,
):
    """Build the Phase-3 PB scan from device and workload shape facts.

    The set covers 1x/2x/4x SM count, the actual Raster tile count, the current
    deployed value, and the backend logical-block count.  Callers may add
    explicitly justified values; canonical sorting removes overlap.
    """

    facts = (
        ("sm_count", sm_count, False),
        ("raster_tile_count", raster_tile_count, True),
        ("backend_logical_blocks", backend_logical_blocks, True),
        ("current_persistent_blocks", current_persistent_blocks, True),
    )
    for name, value, allow_zero in facts:
        if type(value) is not int or value < 0 or (not allow_zero and value == 0):
            suffix = ">= 0" if allow_zero else "> 0"
            raise AutotuneError("{} must be an integer {}".format(name, suffix))
    values = [
        sm_count,
        2 * sm_count,
        4 * sm_count,
        raster_tile_count,
        current_persistent_blocks,
        backend_logical_blocks,
    ]
    if extra_values is not None:
        if not isinstance(extra_values, (list, tuple, set)):
            raise AutotuneError("extra_values must be a sequence")
        values.extend(extra_values)
    return normalize_persistent_blocks(values)


def raster_logical_blocks(image_width, image_height, tile_width=16, tile_height=16):
    """Return the mixed kernel's Raster logical grid size."""

    for name, value in (
        ("image_width", image_width),
        ("image_height", image_height),
        ("tile_width", tile_width),
        ("tile_height", tile_height),
    ):
        if type(value) is not int or value <= 0:
            raise AutotuneError("{} must be a positive integer".format(name))
    return ((image_width + tile_width - 1) // tile_width) * (
        (image_height + tile_height - 1) // tile_height
    )


def first_linear_logical_blocks(rows):
    """Return the v1/v2 first-Linear logical grid: ceil(rows/16) * 2."""

    if type(rows) is not int or rows < 0:
        raise AutotuneError("head rows must be an integer >= 0")
    return 0 if rows == 0 else ((rows + 15) // 16) * 2


def effective_persistent_blocks(
    requested_persistent_blocks,
    sm_count,
    raster_tile_count,
    backend_logical_blocks,
):
    """Mirror ``tacker_mixed.cu`` physical-block resolution exactly."""

    for name, value, allow_zero in (
        ("requested_persistent_blocks", requested_persistent_blocks, True),
        ("sm_count", sm_count, False),
        ("raster_tile_count", raster_tile_count, True),
        ("backend_logical_blocks", backend_logical_blocks, True),
    ):
        if type(value) is not int or value < 0 or (not allow_zero and value == 0):
            suffix = ">= 0" if allow_zero else "> 0"
            raise AutotuneError("{} must be an integer {}".format(name, suffix))
    logical_blocks = max(raster_tile_count, backend_logical_blocks)
    if logical_blocks == 0:
        return 0
    physical = (
        sm_count
        if requested_persistent_blocks == 0
        else requested_persistent_blocks
    )
    return max(1, min(physical, logical_blocks))


def _canonical_selected_heads(selected_heads, head_order):
    order = _normalize_head_order(head_order)
    if not isinstance(selected_heads, (list, tuple)) or not selected_heads:
        raise AutotuneError("selected_heads must be a non-empty sequence")
    names = list(selected_heads)
    if any(not isinstance(name, str) or name not in order for name in names):
        raise AutotuneError("selected_heads contains an unknown head")
    if len(set(names)) != len(names):
        raise AutotuneError("selected_heads must not contain duplicates")
    canonical = [name for name in order if name in names]
    if names != canonical:
        raise AutotuneError("selected_heads must use canonical head order")
    return tuple(names)


def _search_level(abi_family, head_count):
    if abi_family == LEGACY_ABI_FAMILY:
        return "c0"
    if head_count == 1:
        return "c1"
    if head_count == 2:
        return "c2"
    return "c2_h{}".format(head_count)


def _phase31_search_family(abi_family, head_count):
    """Return the coarse C0--C4 family used by Phase-3.1 selection."""

    if abi_family == LEGACY_ABI_FAMILY:
        return SEARCH_FAMILY_C0
    if abi_family == FIRST_LINEAR_ABI_FAMILY:
        return SEARCH_FAMILY_C1 if head_count == 1 else SEARCH_FAMILY_C2
    if abi_family == PACKED_FIRST_LINEAR_ABI_FAMILY:
        return SEARCH_FAMILY_C3
    if abi_family == WHOLE_HEAD_ABI_FAMILY:
        return SEARCH_FAMILY_C4
    raise AutotuneError("unknown ABI family {!r}".format(abi_family))


def candidate_search_family(candidate):
    """Return a stable C0--C4 label for either candidate schema version."""

    validate_candidate(candidate)
    if candidate["candidate_schema_version"] == PHASE31_CANDIDATE_SCHEMA_VERSION:
        return candidate["search_family"]
    return _phase31_search_family(
        candidate["abi_family"], len(candidate["selected_heads"])
    )


def _phase31_search_level(abi_family, head_count):
    family = _phase31_search_family(abi_family, head_count)
    if family in (SEARCH_FAMILY_C0, SEARCH_FAMILY_C1):
        return family
    return "{}_h{}".format(family, head_count)


def _variant_id(
    abi_family, selected_heads, worker_groups, effective_block_count
):
    names = "_".join(selected_heads)
    if abi_family == LEGACY_ABI_FAMILY:
        return "c0_legacy_pos_l1_pb{}".format(effective_block_count)
    prefix = "c{}".format(len(selected_heads))
    if len(selected_heads) > 2:
        prefix = "c2h{}".format(len(selected_heads))
    return "{}_{}_l1_wg{}_pb{}".format(
        prefix, names, worker_groups, effective_block_count
    )


def _candidate_identity_payload(
    abi_family, selected_heads, worker_groups, effective_block_count
):
    return {
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "abi_family": abi_family,
        "selected_heads": list(selected_heads),
        "worker_groups": worker_groups,
        "effective_persistent_blocks": effective_block_count,
    }


def _phase31_variant_id(
    abi_family, selected_heads, worker_groups, effective_block_count
):
    """Create a readable stable id while keeping the physical ABI explicit."""

    if abi_family in (LEGACY_ABI_FAMILY, FIRST_LINEAR_ABI_FAMILY):
        return _variant_id(
            abi_family, selected_heads, worker_groups, effective_block_count
        )
    names = "_".join(selected_heads)
    if abi_family == PACKED_FIRST_LINEAR_ABI_FAMILY:
        prefix = "c3_packed_first_linear"
    elif abi_family == WHOLE_HEAD_ABI_FAMILY:
        prefix = "c4_whole_heads"
    else:
        raise AutotuneError("unknown ABI family {!r}".format(abi_family))
    return "{}_{}_wg{}_pb{}".format(
        prefix, names, worker_groups, effective_block_count
    )


def _phase31_candidate_identity_payload(
    abi_family, selected_heads, worker_groups, effective_block_count
):
    return {
        "candidate_schema_version": PHASE31_CANDIDATE_SCHEMA_VERSION,
        "abi_family": abi_family,
        "search_family": _phase31_search_family(
            abi_family, len(selected_heads)
        ),
        "selected_heads": list(selected_heads),
        "worker_groups": worker_groups,
        "effective_persistent_blocks": effective_block_count,
    }


def make_candidate(
    abi_family,
    selected_heads,
    worker_groups,
    persistent_blocks,
    head_order=HEAD_ORDER,
    effective_block_count=None,
    requested_persistent_blocks=None,
):
    """Create one validated, hash-identified candidate specification."""

    order = _normalize_head_order(head_order)
    names = _canonical_selected_heads(selected_heads, order)
    if abi_family not in (LEGACY_ABI_FAMILY, FIRST_LINEAR_ABI_FAMILY):
        raise AutotuneError("unknown ABI family {!r}".format(abi_family))
    if type(worker_groups) is not int or not 1 <= worker_groups <= len(names):
        raise AutotuneError(
            "worker_groups must be in [1, selected head count]"
        )
    if type(persistent_blocks) is not int or persistent_blocks < 0:
        raise AutotuneError("persistent_blocks must be an int >= 0")
    if effective_block_count is None:
        effective_block_count = persistent_blocks
    if type(effective_block_count) is not int or effective_block_count < 0:
        raise AutotuneError("effective_block_count must be an int >= 0")
    if requested_persistent_blocks is None:
        requested = (persistent_blocks,)
    else:
        requested = normalize_persistent_blocks(requested_persistent_blocks)
        if persistent_blocks not in requested:
            raise AutotuneError(
                "persistent_blocks must be represented in requested PB aliases"
            )
    if abi_family == LEGACY_ABI_FAMILY and (
        names != ("pos",) or worker_groups != 1
    ):
        raise AutotuneError(
            "legacy ABI family is restricted to the current pos-L1 worker"
        )

    identity = _candidate_identity_payload(
        abi_family, names, worker_groups, effective_block_count
    )
    digest = canonical_sha256(identity, "tacker-autotune-candidate-v1")
    result = dict(identity)
    result.update(
        {
            "candidate_sha256": digest,
            "search_level": _search_level(abi_family, len(names)),
            # Profiles launch the canonical effective count explicitly.  In
            # particular, a requested 0 alias never leaves the same candidate
            # hash with device-dependent profile bytes.
            "persistent_blocks": effective_block_count,
            "requested_persistent_blocks": list(requested),
            "variant_id": _variant_id(
                abi_family, names, worker_groups, effective_block_count
            ),
        }
    )
    return result


def make_phase31_candidate(
    abi_family,
    selected_heads,
    worker_groups,
    persistent_blocks,
    head_order=HEAD_ORDER,
    effective_block_count=None,
    requested_persistent_blocks=None,
):
    """Create one schema-v2 C0--C4 candidate.

    Requested PB aliases are deliberately excluded from identity.  A physical
    launch is identified by ABI family, selected head set, worker-group count,
    and the effective PB count; aliases that clamp to that same launch merge.
    """

    order = _normalize_head_order(head_order)
    names = _canonical_selected_heads(selected_heads, order)
    supported = (
        LEGACY_ABI_FAMILY,
        FIRST_LINEAR_ABI_FAMILY,
        PACKED_FIRST_LINEAR_ABI_FAMILY,
        WHOLE_HEAD_ABI_FAMILY,
    )
    if abi_family not in supported:
        raise AutotuneError("unknown ABI family {!r}".format(abi_family))
    if type(worker_groups) is not int or not 1 <= worker_groups <= len(names):
        raise AutotuneError("worker_groups must be in [1, selected head count]")
    if type(persistent_blocks) is not int or persistent_blocks < 0:
        raise AutotuneError("persistent_blocks must be an int >= 0")
    if effective_block_count is None:
        effective_block_count = persistent_blocks
    if type(effective_block_count) is not int or effective_block_count < 0:
        raise AutotuneError("effective_block_count must be an int >= 0")
    if requested_persistent_blocks is None:
        requested = (persistent_blocks,)
    else:
        requested = normalize_persistent_blocks(requested_persistent_blocks)
        if persistent_blocks not in requested:
            raise AutotuneError(
                "persistent_blocks must be represented in requested PB aliases"
            )
    if abi_family == LEGACY_ABI_FAMILY and (
        names != ("pos",) or worker_groups != 1
    ):
        raise AutotuneError(
            "legacy ABI family is restricted to the current pos-L1 worker"
        )
    if abi_family == FIRST_LINEAR_ABI_FAMILY and len(names) > len(order):
        raise AutotuneError("first-linear candidate has too many heads")
    if abi_family == PACKED_FIRST_LINEAR_ABI_FAMILY and len(names) < 2:
        raise AutotuneError("packed first-linear candidates require 2--5 heads")

    identity = _phase31_candidate_identity_payload(
        abi_family, names, worker_groups, effective_block_count
    )
    digest = canonical_sha256(identity, "tacker-autotune-candidate-v2")
    result = dict(identity)
    result.update(
        {
            "candidate_sha256": digest,
            "search_level": _phase31_search_level(abi_family, len(names)),
            "persistent_blocks": effective_block_count,
            "requested_persistent_blocks": list(requested),
            "variant_id": _phase31_variant_id(
                abi_family, names, worker_groups, effective_block_count
            ),
        }
    )
    return result


def _validate_phase31_candidate(candidate, head_order=HEAD_ORDER):
    required = {
        "candidate_schema_version",
        "candidate_sha256",
        "abi_family",
        "search_family",
        "search_level",
        "selected_heads",
        "worker_groups",
        "persistent_blocks",
        "requested_persistent_blocks",
        "effective_persistent_blocks",
        "variant_id",
    }
    if set(candidate) != required:
        missing = sorted(required - set(candidate))
        extra = sorted(set(candidate) - required)
        raise AutotuneError(
            "candidate fields changed (missing={}, extra={})".format(
                missing, extra
            )
        )
    expected = make_phase31_candidate(
        candidate.get("abi_family"),
        candidate.get("selected_heads"),
        candidate.get("worker_groups"),
        (
            candidate.get("requested_persistent_blocks", [None])[0]
            if candidate.get("requested_persistent_blocks")
            else None
        ),
        head_order=head_order,
        effective_block_count=candidate.get("effective_persistent_blocks"),
        requested_persistent_blocks=candidate.get("requested_persistent_blocks"),
    )
    if candidate != expected:
        if candidate.get("candidate_sha256") != expected["candidate_sha256"]:
            raise AutotuneError(
                "candidate_sha256 does not match candidate parameters"
            )
        raise AutotuneError("candidate derived fields are non-canonical")
    return candidate


def validate_candidate(candidate, head_order=HEAD_ORDER):
    """Validate every candidate field and its canonical identity hash."""

    if not isinstance(candidate, dict):
        raise AutotuneError("candidate must be a JSON object")
    if candidate.get("candidate_schema_version") == PHASE31_CANDIDATE_SCHEMA_VERSION:
        return _validate_phase31_candidate(candidate, head_order=head_order)
    required = {
        "candidate_schema_version",
        "candidate_sha256",
        "abi_family",
        "search_level",
        "selected_heads",
        "worker_groups",
        "persistent_blocks",
        "requested_persistent_blocks",
        "effective_persistent_blocks",
        "variant_id",
    }
    if set(candidate) != required:
        missing = sorted(required - set(candidate))
        extra = sorted(set(candidate) - required)
        raise AutotuneError(
            "candidate fields changed (missing={}, extra={})".format(missing, extra)
        )
    if candidate.get("candidate_schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise AutotuneError("candidate_schema_version is unsupported")
    expected = make_candidate(
        candidate.get("abi_family"),
        candidate.get("selected_heads"),
        candidate.get("worker_groups"),
        (
            candidate.get("requested_persistent_blocks", [None])[0]
            if candidate.get("requested_persistent_blocks")
            else None
        ),
        head_order=head_order,
        effective_block_count=candidate.get("effective_persistent_blocks"),
        requested_persistent_blocks=candidate.get("requested_persistent_blocks"),
    )
    if candidate != expected:
        if candidate.get("candidate_sha256") != expected["candidate_sha256"]:
            raise AutotuneError("candidate_sha256 does not match candidate parameters")
        raise AutotuneError("candidate derived fields are non-canonical")
    return candidate


def candidate_sort_key(candidate, head_order=HEAD_ORDER):
    validate_candidate(candidate, head_order=head_order)
    order = _normalize_head_order(head_order)
    indices = tuple(order.index(name) for name in candidate["selected_heads"])
    family_rank = {
        LEGACY_ABI_FAMILY: 0,
        FIRST_LINEAR_ABI_FAMILY: 1,
        PACKED_FIRST_LINEAR_ABI_FAMILY: 2,
        WHOLE_HEAD_ABI_FAMILY: 3,
    }[candidate["abi_family"]]
    return (
        family_rank,
        len(candidate["selected_heads"]),
        indices,
        candidate["worker_groups"],
        candidate["effective_persistent_blocks"],
        candidate["persistent_blocks"],
        candidate["candidate_sha256"],
    )


def deduplicate_candidates(candidates, head_order=HEAD_ORDER):
    """Deduplicate by canonical SHA-256 and return a stable ordering."""

    if not isinstance(candidates, (list, tuple)):
        raise AutotuneError("candidates must be a sequence")
    by_digest = {}
    for candidate in candidates:
        validate_candidate(candidate, head_order=head_order)
        digest = candidate["candidate_sha256"]
        previous = by_digest.get(digest)
        if previous is None:
            by_digest[digest] = deepcopy(candidate)
            continue
        identity_fields = (
            "candidate_schema_version",
            "candidate_sha256",
            "abi_family",
            "search_family",
            "search_level",
            "selected_heads",
            "worker_groups",
            "effective_persistent_blocks",
            "variant_id",
        )
        if any(
            previous.get(name) != candidate.get(name) for name in identity_fields
        ):
            raise AutotuneError("candidate SHA-256 collision or conflicting duplicate")
        requested = normalize_persistent_blocks(
            previous["requested_persistent_blocks"]
            + candidate["requested_persistent_blocks"]
        )
        maker = (
            make_phase31_candidate
            if candidate["candidate_schema_version"]
            == PHASE31_CANDIDATE_SCHEMA_VERSION
            else make_candidate
        )
        by_digest[digest] = maker(
            candidate["abi_family"],
            candidate["selected_heads"],
            (
                1
                if candidate["abi_family"] == LEGACY_ABI_FAMILY
                else candidate["worker_groups"]
            ),
            min(requested),
            head_order=head_order,
            effective_block_count=candidate["effective_persistent_blocks"],
            requested_persistent_blocks=requested,
        )
    return sorted(
        by_digest.values(), key=lambda item: candidate_sort_key(item, head_order)
    )


def enumerate_base_candidates(
    persistent_blocks,
    head_order=HEAD_ORDER,
    sm_count=None,
    raster_tile_count=None,
    backend_logical_blocks=None,
):
    """Enumerate every C0 PB, every C1, and every C2 pair/WG/PB.

    C0 uses the original/current pos-L1 ABI.  All C1 and C2 candidates use the
    multi-head v2 ABI.  For each selected head set, every legal worker group
    count and every PB value is present.
    """

    order = _normalize_head_order(head_order)
    if "pos" not in order:
        raise AutotuneError("head_order must contain pos for the C0 scan")
    blocks = normalize_persistent_blocks(persistent_blocks)
    geometry = (sm_count, raster_tile_count, backend_logical_blocks)
    if any(value is not None for value in geometry) and not all(
        value is not None for value in geometry
    ):
        raise AutotuneError(
            "sm_count, raster_tile_count, and backend_logical_blocks "
            "must be provided together"
        )

    def effective(requested):
        if all(value is not None for value in geometry):
            return effective_persistent_blocks(
                requested, sm_count, raster_tile_count, backend_logical_blocks
            )
        return requested

    result = []
    for block_count in blocks:
        result.append(
            make_candidate(
                LEGACY_ABI_FAMILY,
                ["pos"],
                1,
                block_count,
                head_order=order,
                effective_block_count=effective(block_count),
            )
        )
    for head_name in order:
        for block_count in blocks:
            result.append(
                make_candidate(
                    FIRST_LINEAR_ABI_FAMILY,
                    [head_name],
                    1,
                    block_count,
                    head_order=order,
                    effective_block_count=effective(block_count),
                )
            )
    for selected_heads in itertools.combinations(order, 2):
        for worker_groups in (1, 2):
            for block_count in blocks:
                result.append(
                    make_candidate(
                        FIRST_LINEAR_ABI_FAMILY,
                        selected_heads,
                        worker_groups,
                        block_count,
                        head_order=order,
                        effective_block_count=effective(block_count),
                    )
                )
    return deduplicate_candidates(result, head_order=order)


def _require_complete_geometry(
    sm_count, raster_tile_count, backend_logical_blocks
):
    geometry = (sm_count, raster_tile_count, backend_logical_blocks)
    if any(value is not None for value in geometry) and not all(
        value is not None for value in geometry
    ):
        raise AutotuneError(
            "sm_count, raster_tile_count, and backend_logical_blocks "
            "must be provided together"
        )
    return all(value is not None for value in geometry)


def _effective_pb_for_geometry(
    requested,
    sm_count,
    raster_tile_count,
    backend_logical_blocks,
):
    if _require_complete_geometry(
        sm_count, raster_tile_count, backend_logical_blocks
    ):
        return effective_persistent_blocks(
            requested,
            sm_count,
            raster_tile_count,
            backend_logical_blocks,
        )
    return requested


def enumerate_exhaustive_c2_candidates(
    persistent_blocks,
    head_order=HEAD_ORDER,
    sm_count=None,
    raster_tile_count=None,
    backend_logical_blocks=None,
):
    """Enumerate the complete Phase-3.1 C2 H2--H5 Cartesian product.

    For each non-singleton head set, every legal worker-group count and every
    requested PB is emitted before physical-identity deduplication.  With five
    heads and the current six effective A6000 PB values this is exactly 450
    unique candidates: ``6 * sum(comb(5, h) * h, h=2..5)``.
    """

    order = _normalize_head_order(head_order)
    blocks = normalize_persistent_blocks(persistent_blocks)
    _require_complete_geometry(
        sm_count, raster_tile_count, backend_logical_blocks
    )
    result = []
    for head_count in range(2, len(order) + 1):
        for selected_heads in itertools.combinations(order, head_count):
            for worker_groups in range(1, head_count + 1):
                for requested in blocks:
                    result.append(
                        make_phase31_candidate(
                            FIRST_LINEAR_ABI_FAMILY,
                            selected_heads,
                            worker_groups,
                            requested,
                            head_order=order,
                            effective_block_count=_effective_pb_for_geometry(
                                requested,
                                sm_count,
                                raster_tile_count,
                                backend_logical_blocks,
                            ),
                        )
                    )
    return deduplicate_candidates(result, head_order=order)


def validate_exhaustive_c2_candidates(
    candidates,
    persistent_blocks,
    head_order=HEAD_ORDER,
    sm_count=None,
    raster_tile_count=None,
    backend_logical_blocks=None,
):
    """Fail closed unless ``candidates`` is exactly the canonical C2 grid."""

    order = _normalize_head_order(head_order)
    actual = deduplicate_candidates(candidates, head_order=order)
    if any(
        item["candidate_schema_version"] != PHASE31_CANDIDATE_SCHEMA_VERSION
        or item["search_family"] != SEARCH_FAMILY_C2
        for item in actual
    ):
        raise AutotuneError("exhaustive C2 set contains a non-C2 candidate")
    expected = enumerate_exhaustive_c2_candidates(
        persistent_blocks,
        head_order=order,
        sm_count=sm_count,
        raster_tile_count=raster_tile_count,
        backend_logical_blocks=backend_logical_blocks,
    )
    if actual != expected:
        raise AutotuneError(
            "C2 candidates do not exactly cover every H2--H5 "
            "head-set/worker-group/effective-PB combination"
        )
    effective_values = {
        item["effective_persistent_blocks"] for item in expected
    }
    expected_count = len(effective_values) * sum(
        len(tuple(itertools.combinations(order, head_count))) * head_count
        for head_count in range(2, len(order) + 1)
    )
    if len(expected) != expected_count:
        raise AutotuneError("C2 physical identity deduplication is incomplete")
    if len(order) == 5 and len(effective_values) == 6 and len(expected) != 450:
        raise AutotuneError("current five-head/six-PB C2 grid must contain 450")
    return actual


def enumerate_exhaustive_first_linear_candidates(
    persistent_blocks,
    head_order=HEAD_ORDER,
    sm_count=None,
    raster_tile_count=None,
    backend_logical_blocks=None,
):
    """Enumerate schema-v2 C0/C1 plus the exhaustive C2 H2--H5 grid."""

    order = _normalize_head_order(head_order)
    blocks = normalize_persistent_blocks(persistent_blocks)
    result = []
    for requested in blocks:
        effective = _effective_pb_for_geometry(
            requested,
            sm_count,
            raster_tile_count,
            backend_logical_blocks,
        )
        result.append(
            make_phase31_candidate(
                LEGACY_ABI_FAMILY,
                ("pos",),
                1,
                requested,
                head_order=order,
                effective_block_count=effective,
            )
        )
        for head_name in order:
            result.append(
                make_phase31_candidate(
                    FIRST_LINEAR_ABI_FAMILY,
                    (head_name,),
                    1,
                    requested,
                    head_order=order,
                    effective_block_count=effective,
                )
            )
    result.extend(
        enumerate_exhaustive_c2_candidates(
            blocks,
            head_order=order,
            sm_count=sm_count,
            raster_tile_count=raster_tile_count,
            backend_logical_blocks=backend_logical_blocks,
        )
    )
    return deduplicate_candidates(result, head_order=order)


def validate_exhaustive_first_linear_candidates(
    candidates,
    persistent_blocks,
    head_order=HEAD_ORDER,
    sm_count=None,
    raster_tile_count=None,
    backend_logical_blocks=None,
):
    """Validate the exact canonical schema-v2 C0--C2 candidate set."""

    order = _normalize_head_order(head_order)
    actual = deduplicate_candidates(candidates, head_order=order)
    expected = enumerate_exhaustive_first_linear_candidates(
        persistent_blocks,
        head_order=order,
        sm_count=sm_count,
        raster_tile_count=raster_tile_count,
        backend_logical_blocks=backend_logical_blocks,
    )
    if actual != expected:
        raise AutotuneError(
            "first-linear candidates do not exactly cover canonical C0--C2"
        )
    c2 = [
        item for item in actual if item["search_family"] == SEARCH_FAMILY_C2
    ]
    validate_exhaustive_c2_candidates(
        c2,
        persistent_blocks,
        head_order=order,
        sm_count=sm_count,
        raster_tile_count=raster_tile_count,
        backend_logical_blocks=backend_logical_blocks,
    )
    effective_values = {
        item["effective_persistent_blocks"] for item in actual
    }
    if len(order) == 5 and len(effective_values) == 6 and len(actual) != 486:
        raise AutotuneError("current five-head/six-PB C0--C2 grid must contain 486")
    return actual


def _normalize_launch_geometry(launch_geometry):
    if launch_geometry is None:
        return None
    if not isinstance(launch_geometry, dict) or set(launch_geometry) != {
        "sm_count",
        "raster_tile_count",
        "backend_logical_blocks",
    }:
        raise AutotuneError("launch_geometry has unexpected fields")
    # Calling the resolver performs the exact type/range validation.
    effective_persistent_blocks(
        0,
        launch_geometry["sm_count"],
        launch_geometry["raster_tile_count"],
        launch_geometry["backend_logical_blocks"],
    )
    return {
        "sm_count": launch_geometry["sm_count"],
        "raster_tile_count": launch_geometry["raster_tile_count"],
        "backend_logical_blocks": launch_geometry["backend_logical_blocks"],
    }


def _matrix_payload(
    head_order, persistent_blocks, candidates, launch_geometry=None
):
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "kind": MATRIX_KIND,
        "head_order": list(head_order),
        "persistent_blocks": list(persistent_blocks),
        "launch_geometry": _normalize_launch_geometry(launch_geometry),
        "candidates": candidates,
    }


def seal_matrix(
    head_order, persistent_blocks, candidates, launch_geometry=None
):
    """Create a strictly validated and SHA-256-sealed matrix document."""

    order = _normalize_head_order(head_order)
    blocks = normalize_persistent_blocks(persistent_blocks)
    normalized = deduplicate_candidates(candidates, head_order=order)
    aliases = [
        requested
        for item in normalized
        for requested in item["requested_persistent_blocks"]
    ]
    if any(requested not in blocks for requested in aliases):
        raise AutotuneError("candidate uses requested PB outside matrix PB set")
    used = set(aliases)
    if used != set(blocks):
        raise AutotuneError("matrix PB set contains values with no candidate")
    requested_to_effective = {}
    for item in normalized:
        for requested in item["requested_persistent_blocks"]:
            previous = requested_to_effective.get(requested)
            effective = item["effective_persistent_blocks"]
            if previous is not None and previous != effective:
                raise AutotuneError(
                    "one requested PB maps to conflicting effective launch counts"
                )
            requested_to_effective[requested] = effective
    geometry = _normalize_launch_geometry(launch_geometry)
    for requested, effective in requested_to_effective.items():
        expected_effective = requested
        if geometry is not None:
            expected_effective = effective_persistent_blocks(
                requested,
                geometry["sm_count"],
                geometry["raster_tile_count"],
                geometry["backend_logical_blocks"],
            )
        if effective != expected_effective:
            raise AutotuneError(
                "requested/effective PB mapping disagrees with launch geometry"
            )
    geometry_arguments = {
        "sm_count": None,
        "raster_tile_count": None,
        "backend_logical_blocks": None,
    }
    if geometry is not None:
        geometry_arguments.update(geometry)
    expected_base = enumerate_base_candidates(
        blocks,
        head_order=order,
        **geometry_arguments
    )
    actual_base = [
        item
        for item in normalized
        if item["abi_family"] == LEGACY_ABI_FAMILY
        or len(item["selected_heads"]) <= 2
    ]
    if actual_base != expected_base:
        raise AutotuneError(
            "matrix must contain the complete canonical C0/C1/C2 "
            "PB/head/worker-group Cartesian product"
        )
    payload = _matrix_payload(order, blocks, normalized, geometry)
    result = dict(payload)
    result["matrix_sha256"] = canonical_sha256(
        payload, "tacker-autotune-matrix-v1"
    )
    return result


def build_base_matrix(
    persistent_blocks,
    head_order=HEAD_ORDER,
    sm_count=None,
    raster_tile_count=None,
    backend_logical_blocks=None,
):
    blocks = normalize_persistent_blocks(persistent_blocks)
    geometry_values = (sm_count, raster_tile_count, backend_logical_blocks)
    if any(value is not None for value in geometry_values) and not all(
        value is not None for value in geometry_values
    ):
        raise AutotuneError(
            "sm_count, raster_tile_count, and backend_logical_blocks "
            "must be provided together"
        )
    launch_geometry = None
    if all(value is not None for value in geometry_values):
        launch_geometry = {
            "sm_count": sm_count,
            "raster_tile_count": raster_tile_count,
            "backend_logical_blocks": backend_logical_blocks,
        }
    return seal_matrix(
        head_order,
        blocks,
        enumerate_base_candidates(
            blocks,
            head_order=head_order,
            sm_count=sm_count,
            raster_tile_count=raster_tile_count,
            backend_logical_blocks=backend_logical_blocks,
        ),
        launch_geometry=launch_geometry,
    )


def _normalize_phase31_launch_geometry(launch_geometry):
    if not isinstance(launch_geometry, dict) or set(launch_geometry) != {
        "sm_count",
        "raster_tile_count",
        "backend_logical_blocks",
        "whole_head_logical_blocks",
    }:
        raise AutotuneError("Phase-3.1 launch_geometry has unexpected fields")
    effective_persistent_blocks(
        0,
        launch_geometry["sm_count"],
        launch_geometry["raster_tile_count"],
        launch_geometry["backend_logical_blocks"],
    )
    effective_persistent_blocks(
        0,
        launch_geometry["sm_count"],
        launch_geometry["raster_tile_count"],
        launch_geometry["whole_head_logical_blocks"],
    )
    return {
        "sm_count": launch_geometry["sm_count"],
        "raster_tile_count": launch_geometry["raster_tile_count"],
        "backend_logical_blocks": launch_geometry["backend_logical_blocks"],
        "whole_head_logical_blocks": launch_geometry[
            "whole_head_logical_blocks"
        ],
    }


def _normalize_phase31_pb_grids(persistent_blocks, grids=None):
    base = normalize_persistent_blocks(persistent_blocks)
    if grids is None:
        grids = {}
    if not isinstance(grids, dict) or not set(grids).issubset(set(SEARCH_FAMILIES)):
        raise AutotuneError(
            "persistent_blocks_by_family must only contain C0--C4 keys"
        )
    result = {}
    for family in SEARCH_FAMILIES:
        result[family] = normalize_persistent_blocks(grids.get(family, base))
    for family in (SEARCH_FAMILY_C0, SEARCH_FAMILY_C1, SEARCH_FAMILY_C2):
        if result[family] != base:
            raise AutotuneError(
                "C0--C2 must share the exhaustive base requested PB grid"
            )
    return result


def _phase31_backend_logical_blocks(search_family, launch_geometry):
    if search_family == SEARCH_FAMILY_C4:
        return launch_geometry["whole_head_logical_blocks"]
    return launch_geometry["backend_logical_blocks"]


def _phase31_effective_pb(search_family, requested, launch_geometry):
    return effective_persistent_blocks(
        requested,
        launch_geometry["sm_count"],
        launch_geometry["raster_tile_count"],
        _phase31_backend_logical_blocks(search_family, launch_geometry),
    )


def _phase31_effective_pb_grids(requested_grids, launch_geometry):
    return {
        family: list(
            sorted(
                set(
                    _phase31_effective_pb(family, requested, launch_geometry)
                    for requested in requested_grids[family]
                )
            )
        )
        for family in SEARCH_FAMILIES
    }


def _enumerate_phase31_family_head_sets(
    abi_family,
    selected_head_sets,
    persistent_blocks,
    launch_geometry,
    head_order=HEAD_ORDER,
):
    order = _normalize_head_order(head_order)
    blocks = normalize_persistent_blocks(persistent_blocks)
    canonical_sets = set()
    for selected_heads in selected_head_sets:
        names = _canonical_selected_heads(selected_heads, order)
        family = _phase31_search_family(abi_family, len(names))
        if abi_family == PACKED_FIRST_LINEAR_ABI_FAMILY and len(names) < 2:
            raise AutotuneError("C3 head sets must contain at least two heads")
        if abi_family == WHOLE_HEAD_ABI_FAMILY and len(names) < 1:
            raise AutotuneError("C4 head sets must not be empty")
        if family not in (SEARCH_FAMILY_C3, SEARCH_FAMILY_C4):
            raise AutotuneError("family head-set generator only supports C3/C4")
        canonical_sets.add(names)
    result = []
    for names in sorted(
        canonical_sets,
        key=lambda item: tuple(order.index(name) for name in item),
    ):
        family = _phase31_search_family(abi_family, len(names))
        for worker_groups in range(1, len(names) + 1):
            for requested in blocks:
                result.append(
                    make_phase31_candidate(
                        abi_family,
                        names,
                        worker_groups,
                        requested,
                        head_order=order,
                        effective_block_count=_phase31_effective_pb(
                            family, requested, launch_geometry
                        ),
                    )
                )
    return deduplicate_candidates(result, head_order=order)


def _validate_phase31_generated_family_grid(
    candidates,
    abi_family,
    persistent_blocks,
    launch_geometry,
    head_order,
):
    selected_sets = sorted(
        set(tuple(item["selected_heads"]) for item in candidates),
        key=lambda item: tuple(head_order.index(name) for name in item),
    )
    expected = _enumerate_phase31_family_head_sets(
        abi_family,
        selected_sets,
        persistent_blocks,
        launch_geometry,
        head_order=head_order,
    )
    if candidates != expected:
        family = _phase31_search_family(
            abi_family, len(selected_sets[0]) if selected_sets else 1
        )
        raise AutotuneError(
            "{} candidates must cover the complete head-set/WG/PB grid".format(
                family.upper()
            )
        )


def _phase31_matrix_payload(
    head_order,
    persistent_blocks,
    requested_grids,
    effective_grids,
    candidates,
    launch_geometry,
):
    return {
        "schema_version": PHASE31_MATRIX_SCHEMA_VERSION,
        "kind": MATRIX_KIND,
        "head_order": list(head_order),
        "persistent_blocks": list(persistent_blocks),
        "persistent_blocks_by_family": {
            family: list(requested_grids[family]) for family in SEARCH_FAMILIES
        },
        "effective_persistent_blocks_by_family": {
            family: list(effective_grids[family]) for family in SEARCH_FAMILIES
        },
        "launch_geometry": launch_geometry,
        "candidates": candidates,
    }


def seal_phase31_matrix(
    head_order,
    persistent_blocks,
    candidates,
    launch_geometry,
    persistent_blocks_by_family=None,
):
    """Seal a schema-v2 matrix with exhaustive C0--C2 and staged C3/C4."""

    order = _normalize_head_order(head_order)
    base = normalize_persistent_blocks(persistent_blocks)
    geometry = _normalize_phase31_launch_geometry(launch_geometry)
    requested_grids = _normalize_phase31_pb_grids(
        base, persistent_blocks_by_family
    )
    effective_grids = _phase31_effective_pb_grids(requested_grids, geometry)
    normalized = deduplicate_candidates(candidates, head_order=order)
    if any(
        item["candidate_schema_version"] != PHASE31_CANDIDATE_SCHEMA_VERSION
        for item in normalized
    ):
        raise AutotuneError("schema-v2 matrix cannot contain schema-v1 candidates")

    for item in normalized:
        family = item["search_family"]
        aliases = item["requested_persistent_blocks"]
        if any(value not in requested_grids[family] for value in aliases):
            raise AutotuneError(
                "candidate uses requested PB outside its family PB grid"
            )
        for requested in aliases:
            expected_effective = _phase31_effective_pb(
                family, requested, geometry
            )
            if expected_effective != item["effective_persistent_blocks"]:
                raise AutotuneError(
                    "requested/effective PB mapping disagrees with family geometry"
                )

    first_linear = [
        item
        for item in normalized
        if item["search_family"]
        in (SEARCH_FAMILY_C0, SEARCH_FAMILY_C1, SEARCH_FAMILY_C2)
    ]
    validate_exhaustive_first_linear_candidates(
        first_linear,
        base,
        head_order=order,
        sm_count=geometry["sm_count"],
        raster_tile_count=geometry["raster_tile_count"],
        backend_logical_blocks=geometry["backend_logical_blocks"],
    )
    for abi_family, family in (
        (PACKED_FIRST_LINEAR_ABI_FAMILY, SEARCH_FAMILY_C3),
        (WHOLE_HEAD_ABI_FAMILY, SEARCH_FAMILY_C4),
    ):
        family_candidates = [
            item for item in normalized if item["search_family"] == family
        ]
        if family_candidates:
            _validate_phase31_generated_family_grid(
                family_candidates,
                abi_family,
                requested_grids[family],
                geometry,
                order,
            )

    payload = _phase31_matrix_payload(
        order,
        base,
        requested_grids,
        effective_grids,
        normalized,
        geometry,
    )
    result = dict(payload)
    result["matrix_sha256"] = canonical_sha256(
        payload, "tacker-autotune-matrix-v2"
    )
    return result


def build_phase31_base_matrix(
    persistent_blocks,
    head_order=HEAD_ORDER,
    sm_count=None,
    raster_tile_count=None,
    backend_logical_blocks=None,
    whole_head_logical_blocks=None,
    packed_persistent_blocks=None,
    whole_head_persistent_blocks=None,
):
    """Build the new-run Phase-3.1 matrix with exhaustive C0--C2.

    C3 and C4 PB grids are declared now so later staged extensions cannot
    silently change launch identities.  The whole-head row-logical count is
    added only to C4's requested grid and therefore never expands C2.
    """

    if not _require_complete_geometry(
        sm_count, raster_tile_count, backend_logical_blocks
    ):
        raise AutotuneError("Phase-3.1 matrix requires launch geometry")
    if whole_head_logical_blocks is None:
        whole_head_logical_blocks = backend_logical_blocks
    geometry = _normalize_phase31_launch_geometry(
        {
            "sm_count": sm_count,
            "raster_tile_count": raster_tile_count,
            "backend_logical_blocks": backend_logical_blocks,
            "whole_head_logical_blocks": whole_head_logical_blocks,
        }
    )
    base = normalize_persistent_blocks(persistent_blocks)
    packed = normalize_persistent_blocks(
        base if packed_persistent_blocks is None else packed_persistent_blocks
    )
    whole_values = list(
        base
        if whole_head_persistent_blocks is None
        else normalize_persistent_blocks(whole_head_persistent_blocks)
    )
    whole_values.append(whole_head_logical_blocks)
    whole = normalize_persistent_blocks(whole_values)
    grids = {
        SEARCH_FAMILY_C0: base,
        SEARCH_FAMILY_C1: base,
        SEARCH_FAMILY_C2: base,
        SEARCH_FAMILY_C3: packed,
        SEARCH_FAMILY_C4: whole,
    }
    candidates = enumerate_exhaustive_first_linear_candidates(
        base,
        head_order=head_order,
        sm_count=sm_count,
        raster_tile_count=raster_tile_count,
        backend_logical_blocks=backend_logical_blocks,
    )
    return seal_phase31_matrix(
        head_order,
        base,
        candidates,
        geometry,
        persistent_blocks_by_family=grids,
    )


def _validate_phase31_matrix(matrix):
    required = {
        "schema_version",
        "kind",
        "head_order",
        "persistent_blocks",
        "persistent_blocks_by_family",
        "effective_persistent_blocks_by_family",
        "launch_geometry",
        "candidates",
        "matrix_sha256",
    }
    if set(matrix) != required:
        raise AutotuneError("Phase-3.1 matrix fields changed")
    if matrix.get("kind") != MATRIX_KIND:
        raise AutotuneError("matrix kind is unsupported")
    _require_sha256(matrix.get("matrix_sha256"), "matrix.matrix_sha256")
    expected = seal_phase31_matrix(
        matrix.get("head_order"),
        matrix.get("persistent_blocks"),
        matrix.get("candidates"),
        matrix.get("launch_geometry"),
        persistent_blocks_by_family=matrix.get("persistent_blocks_by_family"),
    )
    if matrix != expected:
        if matrix.get("matrix_sha256") != expected["matrix_sha256"]:
            raise AutotuneError("matrix_sha256 does not match matrix contents")
        raise AutotuneError("matrix is not canonical")
    return matrix


def validate_matrix(matrix):
    """Validate a sealed matrix, including all candidate hashes and ordering."""

    if not isinstance(matrix, dict):
        raise AutotuneError("matrix must be a JSON object")
    if matrix.get("schema_version") == PHASE31_MATRIX_SCHEMA_VERSION:
        return _validate_phase31_matrix(matrix)
    required = {
        "schema_version",
        "kind",
        "head_order",
        "persistent_blocks",
        "launch_geometry",
        "candidates",
        "matrix_sha256",
    }
    if set(matrix) != required:
        raise AutotuneError("matrix fields changed")
    if matrix.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise AutotuneError("matrix schema_version is unsupported")
    if matrix.get("kind") != MATRIX_KIND:
        raise AutotuneError("matrix kind is unsupported")
    _require_sha256(matrix.get("matrix_sha256"), "matrix.matrix_sha256")
    expected = seal_matrix(
        matrix.get("head_order"),
        matrix.get("persistent_blocks"),
        matrix.get("candidates"),
        launch_geometry=matrix.get("launch_geometry"),
    )
    if matrix != expected:
        if matrix.get("matrix_sha256") != expected["matrix_sha256"]:
            raise AutotuneError("matrix_sha256 does not match matrix contents")
        raise AutotuneError("matrix is not canonically sorted")
    return matrix


def _normalize_screening_results(candidates, screening_results):
    candidate_by_digest = {}
    for candidate in candidates:
        validate_candidate(candidate)
        candidate_by_digest[candidate["candidate_sha256"]] = candidate

    if isinstance(screening_results, dict):
        items = []
        for digest, value in screening_results.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("candidate_sha256", digest)
            else:
                record = {"candidate_sha256": digest, "score": value}
            items.append(record)
    elif isinstance(screening_results, list):
        items = list(screening_results)
    else:
        raise AutotuneError("screening_results must be an object or list")

    normalized = {}
    for record in items:
        if not isinstance(record, dict):
            raise AutotuneError("screening result entries must be objects")
        # Phase 3.1 screening journals carry the same terminal evidence fields
        # consumed later by ``build_screening_ranking``.  Staged C3/C4 parent
        # selection needs only status/score, but must accept (and validate)
        # those sealed records without callers weakening them first.
        allowed = {
            "candidate_sha256",
            "score",
            "status",
            "error",
            "artifact_sha256",
            "attempt",
        }
        if not set(record).issubset(allowed):
            raise AutotuneError("screening result contains unknown fields")
        digest = _require_sha256(
            record.get("candidate_sha256"),
            "screening_result.candidate_sha256",
        )
        if digest not in candidate_by_digest:
            raise AutotuneError("screening result references an unknown candidate")
        status = record.get("status", STAGE_SUCCEEDED)
        if status not in (STAGE_SUCCEEDED, STAGE_FAILED):
            raise AutotuneError("screening status must be succeeded or failed")
        error = record.get("error")
        if error is not None and not isinstance(error, str):
            raise AutotuneError("screening failure error must be a string or null")
        artifact_sha256 = record.get("artifact_sha256")
        if artifact_sha256 is not None:
            _require_sha256(
                artifact_sha256, "screening_result.artifact_sha256"
            )
        attempt = record.get("attempt")
        if attempt is not None and (type(attempt) is not int or attempt < 1):
            raise AutotuneError(
                "screening result attempt must be a positive integer"
            )
        score = record.get("score")
        if status == STAGE_SUCCEEDED:
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
            ):
                raise AutotuneError(
                    "successful screening result requires a finite numeric score"
                )
            if error is not None:
                raise AutotuneError(
                    "successful screening result cannot contain an error"
                )
            normalized_record = {
                "candidate_sha256": digest,
                "score": float(score),
                "status": status,
            }
        else:
            if score is not None:
                raise AutotuneError("failed screening result score must be null")
            normalized_record = {
                "candidate_sha256": digest,
                "score": None,
                "status": status,
            }
        previous = normalized.get(digest)
        if previous is not None and previous != normalized_record:
            raise AutotuneError("conflicting screening results for one candidate")
        normalized[digest] = normalized_record
    return normalized


def rank_screening_candidates(
    candidates, screening_results, top_k=None, maximize=True
):
    """Return successful screened candidates in deterministic score order."""

    normalized_candidates = deduplicate_candidates(candidates)
    scores = _normalize_screening_results(
        normalized_candidates, screening_results
    )
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise AutotuneError("top_k must be a positive integer or null")
    if type(maximize) is not bool:
        raise AutotuneError("maximize must be a bool")

    ranked = []
    for candidate in normalized_candidates:
        record = scores.get(candidate["candidate_sha256"])
        if record is None or record["status"] != STAGE_SUCCEEDED:
            continue
        ranked.append(
            {
                "candidate_sha256": candidate["candidate_sha256"],
                "score": record["score"],
                "candidate": deepcopy(candidate),
            }
        )

    def ranking_key(item):
        score_key = -item["score"] if maximize else item["score"]
        return (score_key, candidate_sort_key(item["candidate"]))

    ranked.sort(key=ranking_key)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


def rank_screening_head_sets(
    candidates,
    screening_results,
    search_family,
    top_k=None,
    maximize=True,
    head_order=HEAD_ORDER,
):
    """Rank unique head sets by their best successful physical candidate."""

    if search_family not in SEARCH_FAMILIES:
        raise AutotuneError("search_family must be one of C0--C4")
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise AutotuneError("top_k must be a positive integer or null")
    order = _normalize_head_order(head_order)
    normalized = deduplicate_candidates(candidates, head_order=order)
    if (
        isinstance(screening_results, dict)
        and screening_results.get("kind")
        == "tacker_autotune_screening_ranking"
    ):
        validate_screening_ranking(
            screening_results, require_complete=False
        )
        screening_results = [
            {
                "candidate_sha256": item["candidate_sha256"],
                "status": item["status"],
                "score": item["score"],
            }
            for item in screening_results["terminal_status"]
            if item["status"] in (STAGE_SUCCEEDED, STAGE_FAILED)
        ]
    family_candidates = [
        item
        for item in normalized
        if candidate_search_family(item) == search_family
    ]
    scores = _normalize_screening_results(normalized, screening_results)
    family_scores = {
        item["candidate_sha256"]: scores[item["candidate_sha256"]]
        for item in family_candidates
        if item["candidate_sha256"] in scores
    }
    ranked_candidates = rank_screening_candidates(
        family_candidates,
        family_scores,
        top_k=None,
        maximize=maximize,
    )
    result = []
    seen = set()
    for ranked_item in ranked_candidates:
        selected_heads = tuple(ranked_item["candidate"]["selected_heads"])
        if selected_heads in seen:
            continue
        seen.add(selected_heads)
        result.append(
            {
                "head_set_rank": len(result) + 1,
                "selected_heads": list(selected_heads),
                "representative_candidate_sha256": ranked_item[
                    "candidate_sha256"
                ],
                "score": ranked_item["score"],
                "representative_candidate": deepcopy(
                    ranked_item["candidate"]
                ),
            }
        )
        if top_k is not None and len(result) >= top_k:
            break
    return result


def _require_phase31_matrix(matrix):
    validate_matrix(matrix)
    if matrix.get("schema_version") != PHASE31_MATRIX_SCHEMA_VERSION:
        raise AutotuneError("operation requires a Phase-3.1 schema-v2 matrix")
    return matrix


def extend_phase31_with_c3(
    matrix,
    screening_results,
    top_k,
    maximize=True,
):
    """Generate C3 packed grids from top-K successful C2 head sets."""

    _require_phase31_matrix(matrix)
    ranked_sets = rank_screening_head_sets(
        matrix["candidates"],
        screening_results,
        SEARCH_FAMILY_C2,
        top_k=top_k,
        maximize=maximize,
        head_order=matrix["head_order"],
    )
    if not ranked_sets:
        raise AutotuneError("C3 generation requires a successful C2 screening")
    children = _enumerate_phase31_family_head_sets(
        PACKED_FIRST_LINEAR_ABI_FAMILY,
        [item["selected_heads"] for item in ranked_sets],
        matrix["persistent_blocks_by_family"][SEARCH_FAMILY_C3],
        matrix["launch_geometry"],
        head_order=matrix["head_order"],
    )
    return seal_phase31_matrix(
        matrix["head_order"],
        matrix["persistent_blocks"],
        list(matrix["candidates"]) + children,
        matrix["launch_geometry"],
        persistent_blocks_by_family=matrix["persistent_blocks_by_family"],
    )


def extend_phase31_with_c4(
    matrix,
    screening_results,
    top_k_per_family,
    maximize=True,
):
    """Generate C4 from top head sets in each generated C1/C2/C3 family.

    A successful C1 source necessarily contributes a single-head whole-head
    grid.  A successful C2 or C3 source contributes a multi-head grid.  Thus
    both shapes are retained whenever successful source inputs of both kinds
    exist, even when their absolute scores differ substantially.
    """

    _require_phase31_matrix(matrix)
    if type(top_k_per_family) is not int or top_k_per_family < 1:
        raise AutotuneError("top_k_per_family must be a positive integer")
    selected_sets = []
    source_summary = {}
    for family in (
        SEARCH_FAMILY_C1,
        SEARCH_FAMILY_C2,
        SEARCH_FAMILY_C3,
    ):
        generated = [
            item
            for item in matrix["candidates"]
            if candidate_search_family(item) == family
        ]
        if not generated:
            continue
        ranked = rank_screening_head_sets(
            matrix["candidates"],
            screening_results,
            family,
            top_k=top_k_per_family,
            maximize=maximize,
            head_order=matrix["head_order"],
        )
        source_summary[family] = ranked
        selected_sets.extend(item["selected_heads"] for item in ranked)
    if not selected_sets:
        raise AutotuneError("C4 generation requires a successful C1/C2/C3 screening")
    children = _enumerate_phase31_family_head_sets(
        WHOLE_HEAD_ABI_FAMILY,
        selected_sets,
        matrix["persistent_blocks_by_family"][SEARCH_FAMILY_C4],
        matrix["launch_geometry"],
        head_order=matrix["head_order"],
    )
    has_single_source = any(
        len(item["selected_heads"]) == 1
        for item in source_summary.get(SEARCH_FAMILY_C1, [])
    )
    has_multi_source = any(
        len(item["selected_heads"]) > 1
        for family in (SEARCH_FAMILY_C2, SEARCH_FAMILY_C3)
        for item in source_summary.get(family, [])
    )
    if has_single_source and not any(
        len(item["selected_heads"]) == 1 for item in children
    ):
        raise AutotuneError("C4 generation dropped successful single-head input")
    if has_multi_source and not any(
        len(item["selected_heads"]) > 1 for item in children
    ):
        raise AutotuneError("C4 generation dropped successful multi-head input")
    return seal_phase31_matrix(
        matrix["head_order"],
        matrix["persistent_blocks"],
        list(matrix["candidates"]) + children,
        matrix["launch_geometry"],
        persistent_blocks_by_family=matrix["persistent_blocks_by_family"],
    )


def _normalize_terminal_screening_results(candidates, screening_results):
    candidate_by_digest = {
        item["candidate_sha256"]: item for item in candidates
    }
    if isinstance(screening_results, dict):
        items = []
        for digest, value in screening_results.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("candidate_sha256", digest)
            else:
                record = {"candidate_sha256": digest, "score": value}
            items.append(record)
    elif isinstance(screening_results, list):
        items = list(screening_results)
    else:
        raise AutotuneError("screening_results must be an object or list")

    normalized = {}
    for record in items:
        if not isinstance(record, dict):
            raise AutotuneError("screening result entries must be objects")
        allowed = {
            "candidate_sha256",
            "score",
            "status",
            "error",
            "artifact_sha256",
            "attempt",
        }
        if not set(record).issubset(allowed):
            raise AutotuneError("screening result contains unknown fields")
        digest = _require_sha256(
            record.get("candidate_sha256"),
            "screening_result.candidate_sha256",
        )
        if digest not in candidate_by_digest:
            raise AutotuneError("screening result references an unknown candidate")
        status = record.get("status", STAGE_SUCCEEDED)
        if status not in (STAGE_SUCCEEDED, STAGE_FAILED):
            raise AutotuneError(
                "full screening status must be terminal succeeded or failed"
            )
        score = record.get("score")
        error = record.get("error")
        if status == STAGE_SUCCEEDED:
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
            ):
                raise AutotuneError(
                    "successful screening result requires a finite numeric score"
                )
            if error is not None:
                raise AutotuneError(
                    "successful screening result cannot contain an error"
                )
            score = float(score)
        else:
            if score is not None:
                raise AutotuneError("failed screening result score must be null")
            if error is not None and not isinstance(error, str):
                raise AutotuneError("screening failure error must be a string or null")
        artifact_sha256 = record.get("artifact_sha256")
        if artifact_sha256 is not None:
            _require_sha256(
                artifact_sha256, "screening_result.artifact_sha256"
            )
        attempt = record.get("attempt")
        if attempt is not None and (type(attempt) is not int or attempt < 1):
            raise AutotuneError("screening result attempt must be a positive integer")
        normalized_record = {
            "candidate_sha256": digest,
            "status": status,
            "score": score,
            "error": error,
            "artifact_sha256": artifact_sha256,
            "attempt": attempt,
        }
        previous = normalized.get(digest)
        if previous is not None and previous != normalized_record:
            raise AutotuneError("conflicting screening results for one candidate")
        normalized[digest] = normalized_record
    return normalized


def build_screening_ranking(
    matrix,
    screening_results,
    screening_input_sha256=None,
    maximize=True,
    require_terminal=True,
):
    """Seal a deterministic full ranking and per-candidate terminal status."""

    validate_matrix(matrix)
    if type(maximize) is not bool or type(require_terminal) is not bool:
        raise AutotuneError("maximize and require_terminal must be bools")
    if screening_input_sha256 is not None:
        _require_sha256(screening_input_sha256, "screening_input_sha256")
    candidates = deduplicate_candidates(
        matrix["candidates"], head_order=matrix["head_order"]
    )
    records = _normalize_terminal_screening_results(
        candidates, screening_results
    )
    missing = [
        item["candidate_sha256"]
        for item in candidates
        if item["candidate_sha256"] not in records
    ]
    if missing and require_terminal:
        raise AutotuneError(
            "full screening ranking requires terminal status for all candidates"
        )

    terminal_status = []
    successful = []
    for candidate in candidates:
        digest = candidate["candidate_sha256"]
        record = records.get(digest)
        if record is None:
            terminal_status.append(
                {
                    "candidate_sha256": digest,
                    "variant_id": candidate["variant_id"],
                    "search_family": candidate_search_family(candidate),
                    "status": "missing",
                    "score": None,
                    "error": None,
                    "artifact_sha256": None,
                    "attempt": None,
                }
            )
            continue
        terminal_status.append(
            {
                "candidate_sha256": digest,
                "variant_id": candidate["variant_id"],
                "search_family": candidate_search_family(candidate),
                "status": record["status"],
                "score": record["score"],
                "error": record["error"],
                "artifact_sha256": record["artifact_sha256"],
                "attempt": record["attempt"],
            }
        )
        if record["status"] == STAGE_SUCCEEDED:
            successful.append((candidate, record["score"]))

    def ranking_key(pair):
        score_key = -pair[1] if maximize else pair[1]
        return (score_key, candidate_sort_key(pair[0], matrix["head_order"]))

    successful.sort(key=ranking_key)
    ranking = []
    for candidate, score in successful:
        ranking.append(
            {
                "rank": len(ranking) + 1,
                "candidate_sha256": candidate["candidate_sha256"],
                "variant_id": candidate["variant_id"],
                "search_family": candidate_search_family(candidate),
                "score": score,
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "tacker_autotune_screening_ranking",
        "matrix_sha256": matrix["matrix_sha256"],
        "screening_input_sha256": screening_input_sha256,
        "selection_objective": "screening_score",
        "higher_is_better": maximize,
        "complete": not missing,
        "candidate_count": len(candidates),
        "terminal_count": len(records),
        "successful_count": len(successful),
        "failed_count": sum(
            1 for item in records.values() if item["status"] == STAGE_FAILED
        ),
        "ranking": ranking,
        "terminal_status": terminal_status,
    }
    result = dict(payload)
    result["ranking_sha256"] = canonical_sha256(
        payload, "tacker-autotune-screening-ranking-v1"
    )
    return result


def validate_screening_ranking(ranking, matrix=None, require_complete=True):
    if not isinstance(ranking, dict):
        raise AutotuneError("screening ranking must be an object")
    required = {
        "schema_version",
        "kind",
        "matrix_sha256",
        "screening_input_sha256",
        "selection_objective",
        "higher_is_better",
        "complete",
        "candidate_count",
        "terminal_count",
        "successful_count",
        "failed_count",
        "ranking",
        "terminal_status",
        "ranking_sha256",
    }
    if set(ranking) != required:
        raise AutotuneError("screening ranking fields changed")
    if ranking.get("schema_version") != 1 or ranking.get("kind") != (
        "tacker_autotune_screening_ranking"
    ):
        raise AutotuneError("screening ranking schema/kind is unsupported")
    _require_sha256(ranking.get("matrix_sha256"), "ranking.matrix_sha256")
    _require_sha256(ranking.get("ranking_sha256"), "ranking.ranking_sha256")
    payload = dict(ranking)
    del payload["ranking_sha256"]
    if canonical_sha256(
        payload, "tacker-autotune-screening-ranking-v1"
    ) != ranking["ranking_sha256"]:
        raise AutotuneError("ranking_sha256 does not match ranking contents")
    if require_complete and ranking.get("complete") is not True:
        raise AutotuneError("screening ranking is not terminal-complete")
    if matrix is not None:
        validate_matrix(matrix)
        if ranking["matrix_sha256"] != matrix["matrix_sha256"]:
            raise AutotuneError("screening ranking is bound to another matrix")
        raw = []
        for item in ranking["terminal_status"]:
            if item.get("status") == "missing":
                continue
            raw.append(
                {
                    name: item.get(name)
                    for name in (
                        "candidate_sha256",
                        "status",
                        "score",
                        "error",
                        "artifact_sha256",
                        "attempt",
                    )
                }
            )
        expected = build_screening_ranking(
            matrix,
            raw,
            screening_input_sha256=ranking["screening_input_sha256"],
            maximize=ranking["higher_is_better"],
            require_terminal=ranking["complete"],
        )
        if ranking != expected:
            raise AutotuneError("screening ranking is non-canonical")
    return ranking


def build_screening_ranking_from_db(
    database,
    matrix_sha256,
    screening_input_sha256,
    screening_stage="screening",
    maximize=True,
    require_terminal=True,
):
    """Build the ranking from exact-input, artifact-verified DB rows."""

    if not isinstance(database, ProfileDB):
        raise AutotuneError("database must be a ProfileDB")
    _require_sha256(screening_input_sha256, "screening_input_sha256")
    database._validate_stage_name(screening_stage)
    matrix = database.get_matrix(matrix_sha256)
    records = []
    for candidate in matrix["candidates"]:
        stage = database.get_stage(
            matrix_sha256,
            candidate["candidate_sha256"],
            screening_stage,
            input_sha256=screening_input_sha256,
            verify_artifact=True,
        )
        if stage is None:
            continue
        if stage["status"] == STAGE_RUNNING:
            if require_terminal:
                raise AutotuneError("screening stage still has a running candidate")
            continue
        result = stage["result"]
        if stage["status"] == STAGE_SUCCEEDED:
            if not isinstance(result, dict):
                raise ProfileDBCorruptionError(
                    "successful screening stage requires an object result"
                )
            score = result.get("score")
            error = None
        else:
            score = None
            error = result.get("error") if isinstance(result, dict) else None
        records.append(
            {
                "candidate_sha256": candidate["candidate_sha256"],
                "status": stage["status"],
                "score": score,
                "error": error,
                "artifact_sha256": stage["artifact_sha256"],
                "attempt": stage["attempt"],
            }
        )
    return build_screening_ranking(
        matrix,
        records,
        screening_input_sha256=screening_input_sha256,
        maximize=maximize,
        require_terminal=require_terminal,
    )


def ranked_family_candidates(matrix, screening_ranking, search_family):
    """Return successful candidates in sealed family-local screening order."""

    validate_matrix(matrix)
    validate_screening_ranking(screening_ranking, matrix=matrix)
    if search_family not in SEARCH_FAMILIES:
        raise AutotuneError("search_family must be one of C0--C4")
    candidate_by_digest = {
        item["candidate_sha256"]: item for item in matrix["candidates"]
    }
    return [
        {
            "rank": item["rank"],
            "score": item["score"],
            "candidate": deepcopy(candidate_by_digest[item["candidate_sha256"]]),
        }
        for item in screening_ranking["ranking"]
        if item["search_family"] == search_family
    ]


def next_family_backfill_candidate(
    matrix,
    screening_ranking,
    search_family,
    excluded_candidate_sha256s=(),
):
    """Return the next screening-successful candidate within one family."""

    if not isinstance(excluded_candidate_sha256s, (list, tuple, set)):
        raise AutotuneError("excluded_candidate_sha256s must be a sequence")
    excluded = set(excluded_candidate_sha256s)
    for item in ranked_family_candidates(
        matrix, screening_ranking, search_family
    ):
        if item["candidate"]["candidate_sha256"] not in excluded:
            return item
    return None


def family_local_backfill_candidates(
    matrix,
    screening_ranking,
    search_family,
    count=1,
    excluded_candidate_sha256s=(),
):
    """Return up to ``count`` next successes without crossing family bounds."""

    if type(count) is not int or count < 0:
        raise AutotuneError("count must be an integer >= 0")
    if not isinstance(excluded_candidate_sha256s, (list, tuple, set)):
        raise AutotuneError("excluded_candidate_sha256s must be a sequence")
    if count == 0:
        return []
    excluded = set(excluded_candidate_sha256s)
    result = []
    for item in ranked_family_candidates(
        matrix, screening_ranking, search_family
    ):
        digest = item["candidate"]["candidate_sha256"]
        if digest in excluded:
            continue
        result.append(item)
        excluded.add(digest)
        if len(result) == count:
            break
    return result


def build_formal_candidate_set(matrix, screening_ranking, global_top_k):
    """Seal global top-K union one best successful C0--C4 representative."""

    _require_phase31_matrix(matrix)
    validate_screening_ranking(screening_ranking, matrix=matrix)
    if type(global_top_k) is not int or global_top_k < 1:
        raise AutotuneError("global_top_k must be a positive integer")
    candidate_by_digest = {
        item["candidate_sha256"]: item for item in matrix["candidates"]
    }
    ranking_by_digest = {
        item["candidate_sha256"]: item for item in screening_ranking["ranking"]
    }
    selected_reasons = {}
    for item in screening_ranking["ranking"][:global_top_k]:
        selected_reasons.setdefault(item["candidate_sha256"], []).append(
            "global_top_k"
        )
    generated_families = [
        family
        for family in SEARCH_FAMILIES
        if any(
            candidate_search_family(item) == family
            for item in matrix["candidates"]
        )
    ]
    representative_by_family = {}
    families_without_success = []
    for family in generated_families:
        best = next(
            (
                item
                for item in screening_ranking["ranking"]
                if item["search_family"] == family
            ),
            None,
        )
        if best is None:
            representative_by_family[family] = None
            families_without_success.append(family)
            continue
        digest = best["candidate_sha256"]
        representative_by_family[family] = digest
        selected_reasons.setdefault(digest, []).append(
            "family_best:{}".format(family)
        )
    ordered_digests = sorted(
        selected_reasons,
        key=lambda digest: ranking_by_digest[digest]["rank"],
    )
    entries = []
    for digest in ordered_digests:
        ranked = ranking_by_digest[digest]
        entries.append(
            {
                "formal_set_rank": len(entries) + 1,
                "screening_rank": ranked["rank"],
                "screening_score": ranked["score"],
                "candidate_sha256": digest,
                "variant_id": ranked["variant_id"],
                "search_family": ranked["search_family"],
                "selection_reasons": selected_reasons[digest],
                "candidate": deepcopy(candidate_by_digest[digest]),
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "tacker_autotune_formal_candidate_set",
        "matrix_sha256": matrix["matrix_sha256"],
        "screening_ranking_sha256": screening_ranking["ranking_sha256"],
        "global_top_k": global_top_k,
        "selection_rule": (
            "global_screening_top_k_union_best_successful_per_generated_family"
        ),
        "generated_search_families": generated_families,
        "representative_by_family": representative_by_family,
        "families_without_screening_success": families_without_success,
        "candidates": entries,
    }
    result = dict(payload)
    result["candidate_set_sha256"] = canonical_sha256(
        payload, "tacker-autotune-formal-candidate-set-v1"
    )
    return result


def validate_formal_candidate_set(matrix, screening_ranking, candidate_set):
    if not isinstance(candidate_set, dict):
        raise AutotuneError("formal candidate set must be an object")
    if candidate_set.get("kind") != "tacker_autotune_formal_candidate_set":
        raise AutotuneError("formal candidate set kind is unsupported")
    expected = build_formal_candidate_set(
        matrix, screening_ranking, candidate_set.get("global_top_k")
    )
    if candidate_set != expected:
        if candidate_set.get("candidate_set_sha256") != expected.get(
            "candidate_set_sha256"
        ):
            raise AutotuneError(
                "candidate_set_sha256 does not match candidate-set contents"
            )
        raise AutotuneError("formal candidate set is non-canonical")
    return candidate_set


# Descriptive aliases used by orchestrators and downstream report builders.
build_full_screening_ranking = build_screening_ranking
family_local_backfill_candidate = next_family_backfill_candidate


def expand_beam_candidates(
    candidates,
    screening_results,
    beam_width,
    target_head_count,
    persistent_blocks=None,
    head_order=HEAD_ORDER,
    maximize=True,
):
    """Expand top-K physical parents into a 3--5-head candidate level.

    Parents must have exactly ``target_head_count - 1`` selected heads.  Each
    retained parent is extended by every missing head.  For every unique child
    head set, the returned level covers every legal worker group count and
    every requested PB value.  Canonical SHA-256 deduplication collapses child
    sets reached from multiple parents.
    """

    order = _normalize_head_order(head_order)
    if type(beam_width) is not int or beam_width < 1:
        raise AutotuneError("beam_width must be a positive integer")
    if (
        type(target_head_count) is not int
        or target_head_count < 3
        or target_head_count > min(5, len(order))
    ):
        raise AutotuneError("target_head_count must be in [3, 5]")
    normalized_candidates = deduplicate_candidates(candidates, head_order=order)
    scores = _normalize_screening_results(
        normalized_candidates, screening_results
    )
    parents = [
        item
        for item in normalized_candidates
        if item["abi_family"] == FIRST_LINEAR_ABI_FAMILY
        and len(item["selected_heads"]) == target_head_count - 1
    ]
    parent_scores = {
        item["candidate_sha256"]: scores[item["candidate_sha256"]]
        for item in parents
        if item["candidate_sha256"] in scores
    }
    ranked = rank_screening_candidates(
        parents,
        parent_scores,
        top_k=beam_width,
        maximize=maximize,
    )
    if not ranked:
        raise AutotuneError("beam has no successful scored parent candidates")

    if persistent_blocks is None:
        blocks = normalize_persistent_blocks(
            [
                requested
                for item in normalized_candidates
                for requested in item["requested_persistent_blocks"]
            ]
        )
    else:
        blocks = normalize_persistent_blocks(persistent_blocks)
    effective_by_request = {}
    for item in normalized_candidates:
        for requested in item["requested_persistent_blocks"]:
            previous = effective_by_request.get(requested)
            effective = item["effective_persistent_blocks"]
            if previous is not None and previous != effective:
                raise AutotuneError(
                    "candidate set has conflicting requested/effective PB mapping"
                )
            effective_by_request[requested] = effective
    missing_mappings = [value for value in blocks if value not in effective_by_request]
    if missing_mappings:
        raise AutotuneError(
            "beam PB values lack effective launch mapping: {}".format(
                missing_mappings
            )
        )

    selected_sets = set()
    for ranked_item in ranked:
        parent_names = ranked_item["candidate"]["selected_heads"]
        for name in order:
            if name in parent_names:
                continue
            child = tuple(item for item in order if item in parent_names or item == name)
            if len(child) == target_head_count:
                selected_sets.add(child)

    result = []
    for selected_heads in sorted(
        selected_sets, key=lambda names: tuple(order.index(name) for name in names)
    ):
        for worker_groups in range(1, target_head_count + 1):
            for block_count in blocks:
                result.append(
                    make_candidate(
                        FIRST_LINEAR_ABI_FAMILY,
                        selected_heads,
                        worker_groups,
                        block_count,
                        head_order=order,
                        effective_block_count=effective_by_request[block_count],
                    )
                )
    return deduplicate_candidates(result, head_order=order)


def extend_matrix_with_beam(
    matrix,
    screening_results,
    beam_width,
    target_head_count,
    maximize=True,
):
    validate_matrix(matrix)
    children = expand_beam_candidates(
        matrix["candidates"],
        screening_results,
        beam_width,
        target_head_count,
        persistent_blocks=matrix["persistent_blocks"],
        head_order=matrix["head_order"],
        maximize=maximize,
    )
    return seal_matrix(
        matrix["head_order"],
        matrix["persistent_blocks"],
        list(matrix["candidates"]) + children,
        launch_geometry=matrix["launch_geometry"],
    )


def run_beam_search(
    base_matrix,
    screening_by_parent_head_count,
    beam_width,
    max_heads=5,
    maximize=True,
):
    """Apply screening-driven beam expansions successively through 3--5 heads."""

    validate_matrix(base_matrix)
    if not isinstance(screening_by_parent_head_count, dict):
        raise AutotuneError("screening_by_parent_head_count must be an object")
    if type(max_heads) is not int or max_heads < 3 or max_heads > 5:
        raise AutotuneError("max_heads must be in [3, 5]")
    matrix = deepcopy(base_matrix)
    for target in range(3, max_heads + 1):
        parent_count = target - 1
        scores = screening_by_parent_head_count.get(str(parent_count), _MISSING)
        if scores is _MISSING:
            scores = screening_by_parent_head_count.get(parent_count, _MISSING)
        if scores is _MISSING:
            raise AutotuneError(
                "missing screening scores for {}-head parents".format(parent_count)
            )
        matrix = extend_matrix_with_beam(
            matrix,
            scores,
            beam_width,
            target,
            maximize=maximize,
        )
    return matrix


def _load_tacker_pipeline():
    """Delay the only gaussian_renderer import until descriptor materialization."""

    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return importlib.import_module("gaussian_renderer.tacker_pipeline")


def materialize_candidate_descriptor(
    candidate,
    module=None,
    resources=None,
    legacy_candidate=None,
):
    """Materialize one profile descriptor from the runtime-owned contracts.

    ``legacy_candidate`` may be supplied by tests/offline callers.  Otherwise
    the current legacy descriptor is copied from the runtime's validated
    template.  Non-legacy candidates always call the matching runtime-owned
    contract; no structural profile fields are duplicated in this tuner.
    """

    validate_candidate(candidate)
    if module is None:
        module = _load_tacker_pipeline()
    runtime_head_order = getattr(module, "HEAD_ORDER", None)
    if tuple(runtime_head_order or ()) != HEAD_ORDER:
        raise AutotuneError(
            "runtime HEAD_ORDER disagrees with autotune candidate contract"
        )
    family = candidate["abi_family"]
    expected_backend = None
    if family == FIRST_LINEAR_ABI_FAMILY:
        contract = getattr(module, "first_linear_candidate_contract", None)
        if not callable(contract):
            raise AutotuneError(
                "runtime omitted first_linear_candidate_contract"
            )
        descriptor = contract(
            candidate["variant_id"],
            candidate["selected_heads"],
            worker_groups=candidate["worker_groups"],
            persistent_blocks=candidate["persistent_blocks"],
            resources=resources,
        )
        expected_backend = "first_linear"
    elif family == PACKED_FIRST_LINEAR_ABI_FAMILY:
        contract = getattr(
            module, "packed_first_linear_candidate_contract", None
        )
        if not callable(contract):
            raise AutotuneError(
                "runtime omitted packed_first_linear_candidate_contract"
            )
        descriptor = contract(
            candidate["variant_id"],
            candidate["selected_heads"],
            worker_groups=candidate["worker_groups"],
            persistent_blocks=candidate["persistent_blocks"],
            resources=resources,
        )
        expected_backend = "packed_first_linear"
    elif family == WHOLE_HEAD_ABI_FAMILY:
        contract = getattr(module, "whole_head_candidate_contract", None)
        if not callable(contract):
            raise AutotuneError("runtime omitted whole_head_candidate_contract")
        descriptor = contract(
            candidate["variant_id"],
            candidate["selected_heads"],
            worker_groups=candidate["worker_groups"],
            persistent_blocks=candidate["persistent_blocks"],
            resources=resources,
        )
        expected_backend = "whole_head"
    else:
        if legacy_candidate is None:
            loader = getattr(module, "load_tacker_profile", None)
            if not callable(loader):
                raise AutotuneError("runtime omitted load_tacker_profile")
            profile = loader()
            legacy_id = getattr(module, "LEGACY_VARIANT_ID", "legacy_pos_l1")
            matches = [
                item
                for item in profile.get("candidates", [])
                if item.get("variant_id") == legacy_id
            ]
            if len(matches) != 1:
                raise AutotuneError(
                    "runtime template must contain exactly one legacy candidate"
                )
            legacy_candidate = matches[0]
        if not isinstance(legacy_candidate, dict):
            raise AutotuneError("legacy_candidate must be a profile object")
        descriptor = deepcopy(legacy_candidate)
        descriptor["variant_id"] = candidate["variant_id"]
        descriptor["persistent_blocks"] = candidate["persistent_blocks"]
        if resources is not None:
            descriptor["resources"] = deepcopy(resources)

    if not isinstance(descriptor, dict):
        raise AutotuneError("runtime descriptor contract returned a non-object")
    if descriptor.get("variant_id") != candidate["variant_id"]:
        raise AutotuneError("runtime descriptor returned the wrong variant_id")
    if descriptor.get("persistent_blocks") != candidate["persistent_blocks"]:
        raise AutotuneError("runtime descriptor returned the wrong PB count")
    descriptor_family = descriptor.get("abi_family")
    if descriptor_family is not None and descriptor_family != family:
        raise AutotuneError("runtime descriptor returned the wrong ABI family")
    if family != LEGACY_ABI_FAMILY:
        partition = descriptor.get("partition")
        if not isinstance(partition, dict):
            raise AutotuneError("runtime descriptor omitted partition")
        if partition.get("selected_heads") != candidate["selected_heads"]:
            raise AutotuneError("runtime descriptor returned the wrong heads")
        if partition.get("worker_groups") != candidate["worker_groups"]:
            raise AutotuneError("runtime descriptor returned the wrong worker_groups")
        backend = partition.get("backend")
        if backend is not None and backend != expected_backend:
            raise AutotuneError("runtime descriptor returned the wrong backend")
    return descriptor


def materialize_matrix_descriptors(matrix, module=None, resource_provider=None):
    """Materialize descriptors in candidate order with optional resource facts."""

    validate_matrix(matrix)
    if module is None:
        module = _load_tacker_pipeline()
    if resource_provider is not None and not callable(resource_provider):
        raise AutotuneError("resource_provider must be callable")
    descriptors = []
    for candidate in matrix["candidates"]:
        resources = None
        if resource_provider is not None:
            resources = resource_provider(
                candidate["abi_family"], candidate["worker_groups"]
            )
        descriptors.append(
            materialize_candidate_descriptor(
                candidate, module=module, resources=resources
            )
        )
    return descriptors


def normalize_runtime_resources(raw):
    """Normalize the runtime resource-query aliases used by qualification.

    The extension's boolean ``launch_supported`` is consumed as a gate and is
    intentionally not copied into the numeric profile resource object.
    """

    if not isinstance(raw, dict):
        raise AutotuneError("runtime resources must be a JSON object")
    values = dict(raw)
    if values.pop("launch_supported", None) is not True:
        raise AutotuneError("runtime resource query did not confirm launch support")
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
    normalized = {}
    for target, sources in aliases.items():
        value = next((values[name] for name in sources if name in values), None)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise AutotuneError(
                "runtime resources omitted finite non-negative {}".format(target)
            )
        normalized[target] = value
    if int(normalized["active_blocks_per_sm"]) < 1:
        raise AutotuneError("runtime resources report zero active blocks per SM")
    for name, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise AutotuneError(
                    "runtime resource {} must be finite and non-negative".format(
                        name
                    )
                )
            normalized[name] = value
    normalized.update(
        {name: value for name, value in normalized.items()}
    )
    return normalized


def build_disabled_qualification_profile(
    matrix,
    candidate,
    resources,
    module=None,
    template=None,
):
    """Build one replayable, disabled, hash-sealed qualification profile."""

    validate_matrix(matrix)
    validate_candidate(candidate, head_order=matrix["head_order"])
    registered = {
        item["candidate_sha256"]: item for item in matrix["candidates"]
    }.get(candidate["candidate_sha256"])
    if registered != candidate:
        raise AutotuneError("candidate is not an exact member of matrix")
    if module is None:
        module = _load_tacker_pipeline()
    if template is None:
        loader = getattr(module, "load_tacker_profile", None)
        if not callable(loader):
            raise AutotuneError("runtime omitted load_tacker_profile")
        template = loader()
    if not isinstance(template, dict):
        raise AutotuneError("qualification template must be an object")
    normalized_resources = normalize_runtime_resources(resources)
    legacy_source = None
    if candidate["abi_family"] == LEGACY_ABI_FAMILY:
        legacy_id = getattr(module, "LEGACY_VARIANT_ID", "legacy_pos_l1")
        legacy_matches = [
            item
            for item in template.get("candidates", [])
            if item.get("variant_id") == legacy_id
        ]
        if len(legacy_matches) != 1:
            raise AutotuneError(
                "qualification template must contain one legacy candidate"
            )
        legacy_source = legacy_matches[0]
    descriptor = materialize_candidate_descriptor(
        candidate,
        module=module,
        resources=normalized_resources,
        legacy_candidate=legacy_source,
    )
    baselines = [
        deepcopy(item)
        for item in template.get("candidates", [])
        if item.get("execution_mode") in ("serial", "two_stream")
    ]
    if [item.get("execution_mode") for item in baselines] != [
        "serial",
        "two_stream",
    ]:
        raise AutotuneError(
            "qualification template must contain serial and two_stream baselines"
        )
    profile = deepcopy(template)
    profile["candidates"] = baselines + [descriptor]
    profile["selected_variant_id"] = candidate["variant_id"]
    manifest = profile.get("manifest")
    if not isinstance(manifest, dict):
        raise AutotuneError("qualification template omitted manifest")
    manifest["persistent_blocks"] = candidate["persistent_blocks"]
    manifest_hasher = getattr(module, "manifest_sha256", None)
    profile_hasher = getattr(module, "profile_sha256", None)
    validator = getattr(module, "validate_tacker_profile", None)
    if not all(callable(item) for item in (manifest_hasher, profile_hasher, validator)):
        raise AutotuneError("runtime omitted profile hash/validation contracts")
    profile["manifest_sha256"] = manifest_hasher(manifest)
    profile["selection"] = None
    profile["deployment"] = {"enabled": False, "valid": False}
    profile["provenance"] = {
        "template": True,
        "phase": (
            "3.1"
            if matrix["schema_version"] == PHASE31_MATRIX_SCHEMA_VERSION
            else 3
        ),
        "generated_by": "scripts/tacker_autotune.py",
        "matrix_sha256": matrix["matrix_sha256"],
        "candidate_sha256": candidate["candidate_sha256"],
        "abi_family": candidate["abi_family"],
        "requested_persistent_blocks": candidate["requested_persistent_blocks"],
        "effective_persistent_blocks": candidate["effective_persistent_blocks"],
        "launch_geometry": deepcopy(matrix["launch_geometry"]),
    }
    phase_label = (
        "Phase-3.1"
        if matrix["schema_version"] == PHASE31_MATRIX_SCHEMA_VERSION
        else "Phase-3"
    )
    profile["note"] = (
        "Disabled {} qualification profile; it is not deployable until "
        "correctness and formal whole-run selection evidence are sealed."
    ).format(phase_label)
    profile["profile_sha256"] = profile_hasher(profile)
    try:
        validator(profile)
    except Exception as error:
        raise AutotuneError(
            "runtime rejected generated qualification profile: {}".format(error)
        )
    return profile


def build_qualification_profiles(
    matrix,
    resource_provider,
    module=None,
    template=None,
):
    """Materialize every matrix member, caching resources by ABI/WG."""

    validate_matrix(matrix)
    if not callable(resource_provider):
        raise AutotuneError("resource_provider must be callable")
    if module is None:
        module = _load_tacker_pipeline()
    if template is None:
        template = module.load_tacker_profile()
    cache = {}
    profiles = {}
    for candidate in matrix["candidates"]:
        resource_key = (
            candidate["abi_family"],
            candidate["worker_groups"],
        )
        if resource_key not in cache:
            cache[resource_key] = resource_provider(*resource_key)
        profile = build_disabled_qualification_profile(
            matrix,
            candidate,
            cache[resource_key],
            module=module,
            template=template,
        )
        profiles[candidate["variant_id"]] = profile
    return profiles


def _resource_provider_from_document(document):
    """Return an ABI-family/WG provider backed by a finite JSON document."""

    if not isinstance(document, dict):
        raise AutotuneError("resource document must be an object")
    families = document.get("families", document)
    if not isinstance(families, dict):
        raise AutotuneError("resource document families must be an object")

    def provider(abi_family, worker_groups):
        family = families.get(abi_family)
        if not isinstance(family, dict):
            raise AutotuneError(
                "resource document omitted ABI family {}".format(abi_family)
            )
        by_group = family.get("worker_groups", family)
        if not isinstance(by_group, dict):
            raise AutotuneError("resource worker_groups must be an object")
        raw = by_group.get(str(worker_groups), by_group.get(worker_groups))
        if raw is None:
            raise AutotuneError(
                "resource document omitted {} worker_groups={}".format(
                    abi_family, worker_groups
                )
            )
        return raw

    return provider


def _raise_duplicate_json_key(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AutotuneError("duplicate JSON object key {!r}".format(key))
        result[key] = value
    return result


def _reject_json_constant(value):
    raise AutotuneError("non-finite JSON constant {!r}".format(value))


def parse_json_text(text, section="JSON"):
    if not isinstance(text, str):
        raise AutotuneError("{} must be text".format(section))
    try:
        value = json.loads(
            text,
            object_pairs_hook=_raise_duplicate_json_key,
            parse_constant=_reject_json_constant,
        )
    except AutotuneError:
        raise
    except (TypeError, ValueError) as error:
        raise AutotuneError("invalid {}: {}".format(section, error))
    _assert_json_value(value, section)
    return value


def load_json_file(path):
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise AutotuneError("cannot read JSON {}: {}".format(path, error))
    return parse_json_text(text, str(path))


def pretty_json_text(value):
    _assert_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def atomic_write_bytes(path, payload):
    """Durably replace one file after writing and fsyncing a sibling temp file."""

    path = Path(path)
    if not isinstance(payload, bytes):
        raise AutotuneError("atomic payload must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
        try:
            directory_descriptor = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def atomic_write_json(path, value):
    atomic_write_bytes(path, pretty_json_text(value).encode("utf-8"))


def hash_artifact_file(path):
    """Hash a regular file and reject concurrent in-place mutation."""

    path = Path(path)
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not os.path.isfile(str(path)):
                raise ProfileDBCorruptionError(
                    "artifact is not a regular file: {}".format(path)
                )
            digest = hashlib.sha256()
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except ProfileDBCorruptionError:
        raise
    except OSError as error:
        raise ProfileDBCorruptionError(
            "cannot read artifact {}: {}".format(path, error)
        )
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        getattr(before, "st_mtime_ns", int(before.st_mtime * 1000000000)),
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        getattr(after, "st_mtime_ns", int(after.st_mtime * 1000000000)),
    )
    if identity_before != identity_after:
        raise ProfileDBCorruptionError("artifact changed while it was hashed")
    return {
        "artifact_sha256": digest.hexdigest(),
        "artifact_size": before.st_size,
        "artifact_path": str(path.resolve()),
    }


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


class ProfileDB(object):
    """Transactional, hash-validating checkpoint database for profile stages."""

    _EXPECTED_TABLES = {"metadata", "matrices", "candidates", "stages"}

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.artifact_store = Path(str(self.path) + ".artifacts")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        try:
            self._connection = sqlite3.connect(
                str(self.path), timeout=30.0, isolation_level=None
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._initialize_or_validate_schema()
            self._quick_check()
        except ProfileDBError:
            self.close()
            raise
        except sqlite3.DatabaseError as error:
            self.close()
            raise ProfileDBCorruptionError(
                "cannot open profile DB {}: {}".format(self.path, error)
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _require_open(self):
        if self._connection is None:
            raise ProfileDBError("profile DB is closed")

    def _begin(self):
        self._require_open()
        self._connection.execute("BEGIN IMMEDIATE")

    def _commit(self):
        self._connection.execute("COMMIT")

    def _rollback(self):
        try:
            self._connection.execute("ROLLBACK")
        except sqlite3.DatabaseError:
            pass

    def _initialize_or_validate_schema(self):
        rows = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        existing = set(row[0] for row in rows)
        if existing and existing != self._EXPECTED_TABLES:
            raise ProfileDBCorruptionError(
                "profile DB has an incomplete or unknown schema"
            )
        if not existing:
            self._begin()
            try:
                self._connection.execute(
                    "CREATE TABLE metadata ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                self._connection.execute(
                    "CREATE TABLE matrices ("
                    "matrix_sha256 TEXT PRIMARY KEY, "
                    "matrix_json TEXT NOT NULL, created_utc TEXT NOT NULL)"
                )
                self._connection.execute(
                    "CREATE TABLE candidates ("
                    "matrix_sha256 TEXT NOT NULL, "
                    "candidate_sha256 TEXT NOT NULL, "
                    "candidate_json TEXT NOT NULL, "
                    "PRIMARY KEY (matrix_sha256, candidate_sha256), "
                    "FOREIGN KEY (matrix_sha256) REFERENCES matrices(matrix_sha256))"
                )
                self._connection.execute(
                    "CREATE TABLE stages ("
                    "matrix_sha256 TEXT NOT NULL, "
                    "candidate_sha256 TEXT NOT NULL, "
                    "stage_name TEXT NOT NULL, "
                    "input_sha256 TEXT NOT NULL, "
                    "status TEXT NOT NULL CHECK "
                    "(status IN ('running','succeeded','failed')), "
                    "claim_token TEXT, "
                    "artifact_sha256 TEXT, artifact_path TEXT, artifact_size INTEGER, "
                    "source_artifact_path TEXT, "
                    "result_json TEXT, attempt INTEGER NOT NULL CHECK (attempt >= 1), "
                    "updated_utc TEXT NOT NULL, "
                    "PRIMARY KEY "
                    "(matrix_sha256, candidate_sha256, stage_name, input_sha256), "
                    "FOREIGN KEY (matrix_sha256, candidate_sha256) REFERENCES "
                    "candidates(matrix_sha256, candidate_sha256), "
                    "CHECK ((status = 'succeeded' AND artifact_sha256 IS NOT NULL "
                    "AND artifact_path IS NOT NULL AND artifact_size IS NOT NULL "
                    "AND source_artifact_path IS NOT NULL) "
                    "OR status != 'succeeded'), "
                    "CHECK ((status = 'running' AND claim_token IS NOT NULL) "
                    "OR (status != 'running' AND claim_token IS NULL)))"
                )
                self._connection.execute(
                    "CREATE INDEX stages_lookup ON stages "
                    "(matrix_sha256, candidate_sha256, stage_name, status)"
                )
                self._connection.execute(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    ("schema_version", str(DB_SCHEMA_VERSION)),
                )
                self._commit()
            except Exception:
                self._rollback()
                raise
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is None or row[0] != str(DB_SCHEMA_VERSION):
            raise ProfileDBCorruptionError(
                "profile DB schema version is missing or unsupported"
            )
        extra_metadata = self._connection.execute(
            "SELECT key FROM metadata WHERE key != 'schema_version'"
        ).fetchall()
        if extra_metadata:
            raise ProfileDBCorruptionError("profile DB metadata contains unknown keys")

    def _quick_check(self):
        rows = self._connection.execute("PRAGMA quick_check").fetchall()
        if len(rows) != 1 or rows[0][0] != "ok":
            raise ProfileDBCorruptionError("SQLite quick_check failed")

    def _foreign_key_and_orphan_check(self):
        violations = self._connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ProfileDBCorruptionError(
                "SQLite foreign_key_check found orphaned rows"
            )
        orphan_candidates = self._connection.execute(
            "SELECT COUNT(*) FROM candidates AS c LEFT JOIN matrices AS m "
            "ON m.matrix_sha256 = c.matrix_sha256 "
            "WHERE m.matrix_sha256 IS NULL"
        ).fetchone()[0]
        orphan_stages = self._connection.execute(
            "SELECT COUNT(*) FROM stages AS s LEFT JOIN candidates AS c "
            "ON c.matrix_sha256 = s.matrix_sha256 "
            "AND c.candidate_sha256 = s.candidate_sha256 "
            "WHERE c.candidate_sha256 IS NULL"
        ).fetchone()[0]
        if orphan_candidates or orphan_stages:
            raise ProfileDBCorruptionError(
                "profile DB contains orphan candidates or stages"
            )

    def register_matrix(self, matrix):
        """Register one sealed matrix; an exact re-registration is idempotent."""

        validate_matrix(matrix)
        matrix_digest = matrix["matrix_sha256"]
        matrix_json = canonical_json_bytes(matrix).decode("utf-8")
        self._begin()
        try:
            row = self._connection.execute(
                "SELECT matrix_json FROM matrices WHERE matrix_sha256 = ?",
                (matrix_digest,),
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO matrices(matrix_sha256, matrix_json, created_utc) "
                    "VALUES (?, ?, ?)",
                    (matrix_digest, matrix_json, _utc_now()),
                )
            elif row[0] != matrix_json:
                raise ProfileDBConflictError(
                    "matrix hash is already bound to different contents"
                )
            for candidate in matrix["candidates"]:
                digest = candidate["candidate_sha256"]
                candidate_json = canonical_json_bytes(candidate).decode("utf-8")
                existing = self._connection.execute(
                    "SELECT candidate_json FROM candidates "
                    "WHERE matrix_sha256 = ? AND candidate_sha256 = ?",
                    (matrix_digest, digest),
                ).fetchone()
                if existing is None:
                    self._connection.execute(
                        "INSERT INTO candidates(matrix_sha256, candidate_sha256, "
                        "candidate_json) VALUES (?, ?, ?)",
                        (matrix_digest, digest, candidate_json),
                    )
                elif existing[0] != candidate_json:
                    raise ProfileDBConflictError(
                        "candidate hash is already bound to different contents"
                    )
            count = self._connection.execute(
                "SELECT COUNT(*) FROM candidates WHERE matrix_sha256 = ?",
                (matrix_digest,),
            ).fetchone()[0]
            if count != len(matrix["candidates"]):
                raise ProfileDBConflictError(
                    "registered matrix has an unexpected candidate set"
                )
            self._commit()
        except Exception:
            self._rollback()
            raise
        return matrix_digest

    def _load_registered_matrix(self, matrix_digest):
        _require_sha256(matrix_digest, "matrix_sha256")
        row = self._connection.execute(
            "SELECT matrix_json FROM matrices WHERE matrix_sha256 = ?",
            (matrix_digest,),
        ).fetchone()
        if row is None:
            raise ProfileDBConflictError("matrix is not registered")
        try:
            matrix = parse_json_text(row[0], "stored matrix")
            validate_matrix(matrix)
        except AutotuneError as error:
            raise ProfileDBCorruptionError(
                "stored matrix failed validation: {}".format(error)
            )
        if matrix["matrix_sha256"] != matrix_digest:
            raise ProfileDBCorruptionError("stored matrix row key disagrees with JSON")
        return matrix

    def get_matrix(self, matrix_sha256):
        """Return a validated copy of one registered matrix."""

        return deepcopy(self._load_registered_matrix(matrix_sha256))

    def _load_registered_candidate(self, matrix_digest, candidate_digest):
        _require_sha256(candidate_digest, "candidate_sha256")
        matrix = self._load_registered_matrix(matrix_digest)
        row = self._connection.execute(
            "SELECT candidate_json FROM candidates "
            "WHERE matrix_sha256 = ? AND candidate_sha256 = ?",
            (matrix_digest, candidate_digest),
        ).fetchone()
        if row is None:
            raise ProfileDBConflictError("candidate is not registered in matrix")
        try:
            candidate = parse_json_text(row[0], "stored candidate")
            validate_candidate(candidate, head_order=matrix["head_order"])
        except AutotuneError as error:
            raise ProfileDBCorruptionError(
                "stored candidate failed validation: {}".format(error)
            )
        expected = {
            item["candidate_sha256"]: item for item in matrix["candidates"]
        }.get(candidate_digest)
        if candidate != expected:
            raise ProfileDBCorruptionError(
                "stored candidate row disagrees with registered matrix"
            )
        return candidate

    @staticmethod
    def _validate_stage_name(stage_name):
        if not isinstance(stage_name, str) or _STAGE_RE.match(stage_name) is None:
            raise AutotuneError(
                "stage_name must be 1-128 safe ASCII identifier characters"
            )
        return stage_name

    @staticmethod
    def _resolve_input_hash(inputs, explicit_input_sha256):
        if inputs is not _MISSING and explicit_input_sha256 is not None:
            raise AutotuneError("provide inputs or input_sha256, not both")
        if inputs is _MISSING and explicit_input_sha256 is None:
            raise AutotuneError("stage inputs are required")
        if explicit_input_sha256 is not None:
            return _require_sha256(explicit_input_sha256, "input_sha256")
        return input_sha256(inputs)

    @classmethod
    def _row_to_record(cls, row):
        if row is None:
            return None
        result = None
        if row["result_json"] is not None:
            try:
                result = parse_json_text(row["result_json"], "stored stage result")
            except AutotuneError as error:
                raise ProfileDBCorruptionError(
                    "stored stage result is invalid: {}".format(error)
                )
        record = {
            "matrix_sha256": row["matrix_sha256"],
            "candidate_sha256": row["candidate_sha256"],
            "stage_name": row["stage_name"],
            "input_sha256": row["input_sha256"],
            "status": row["status"],
            "claim_token": row["claim_token"],
            "artifact_sha256": row["artifact_sha256"],
            "artifact_path": row["artifact_path"],
            "artifact_size": row["artifact_size"],
            "source_artifact_path": row["source_artifact_path"],
            "result": result,
            "attempt": row["attempt"],
            "updated_utc": row["updated_utc"],
        }
        cls._validate_record_claim_state(record)
        return record

    @staticmethod
    def _validate_claim_token(claim_token):
        if not isinstance(claim_token, str) or _CLAIM_TOKEN_RE.match(claim_token) is None:
            raise AutotuneError("claim_token must be a lowercase 128-bit token")
        return claim_token

    @classmethod
    def _validate_record_claim_state(cls, record):
        token = record.get("claim_token")
        if record.get("status") == STAGE_RUNNING:
            try:
                cls._validate_claim_token(token)
            except AutotuneError as error:
                raise ProfileDBCorruptionError(
                    "running stage has an invalid claim token: {}".format(error)
                )
        elif token is not None:
            raise ProfileDBCorruptionError(
                "non-running stage retained a claim token"
            )

    @staticmethod
    def _stat_identity(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            getattr(value, "st_mtime_ns", int(value.st_mtime * 1000000000)),
            getattr(value, "st_ctime_ns", int(value.st_ctime * 1000000000)),
        )

    def _prepare_artifact_store(self):
        try:
            self.artifact_store.mkdir(parents=True, exist_ok=True)
            if self.artifact_store.is_symlink() or not self.artifact_store.is_dir():
                raise ProfileDBCorruptionError(
                    "profile DB artifact store is not a real directory"
                )
        except ProfileDBError:
            raise
        except OSError as error:
            raise ProfileDBError(
                "cannot prepare profile DB artifact store: {}".format(error)
            )

    def _fsync_artifact_store(self):
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        descriptor = os.open(str(self.artifact_store), flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _snapshot_artifact(self, source_path):
        """Copy a stable source read into the DB's content-addressed store."""

        self._prepare_artifact_store()
        source = Path(source_path).resolve()
        temporary_name = None
        try:
            with source.open("rb") as source_handle:
                before = os.fstat(source_handle.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ProfileDBCorruptionError(
                        "stage artifact source is not a regular file"
                    )
                path_before = os.stat(str(source))
                if self._stat_identity(path_before) != self._stat_identity(before):
                    raise ProfileDBCorruptionError(
                        "stage artifact source path changed before snapshot"
                    )
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".snapshot-",
                    suffix=".tmp",
                    dir=str(self.artifact_store),
                )
                digest = hashlib.sha256()
                with os.fdopen(descriptor, "wb") as destination:
                    while True:
                        chunk = source_handle.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        destination.write(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
                after = os.fstat(source_handle.fileno())
                path_after = os.stat(str(source))
                identity = self._stat_identity(before)
                if (
                    self._stat_identity(after) != identity
                    or self._stat_identity(path_after) != identity
                ):
                    raise ProfileDBCorruptionError(
                        "stage artifact source changed while snapshotting"
                    )

            artifact_sha256 = digest.hexdigest()
            target = self.artifact_store / artifact_sha256
            created = False
            try:
                os.link(temporary_name, str(target))
                created = True
            except FileExistsError:
                created = False
            if created:
                os.chmod(str(target), 0o444)
                self._fsync_artifact_store()
            target_stat = os.lstat(str(target))
            if not stat.S_ISREG(target_stat.st_mode):
                raise ProfileDBCorruptionError(
                    "content-addressed artifact target is not a regular file"
                )
            if target_stat.st_mode & 0o222:
                raise ProfileDBCorruptionError(
                    "content-addressed artifact target is writable"
                )
            snapshot = hash_artifact_file(target)
            if (
                snapshot["artifact_sha256"] != artifact_sha256
                or snapshot["artifact_size"] != before.st_size
                or snapshot["artifact_path"] != str(target.resolve())
            ):
                raise ProfileDBCorruptionError(
                    "content-addressed artifact snapshot failed verification"
                )
            snapshot["source_artifact_path"] = str(source)
            return snapshot
        except ProfileDBError:
            raise
        except OSError as error:
            raise ProfileDBCorruptionError(
                "cannot snapshot stage artifact {}: {}".format(source, error)
            )
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass

    def _verify_success_artifact(self, record):
        if record["status"] != STAGE_SUCCEEDED:
            return
        _require_sha256(record["artifact_sha256"], "artifact_sha256")
        expected_path = self.artifact_store / record["artifact_sha256"]
        if record["artifact_path"] != str(expected_path.resolve()):
            raise ProfileDBCorruptionError(
                "successful stage does not reference its DB-owned CAS artifact"
            )
        if (
            not isinstance(record.get("source_artifact_path"), str)
            or not record["source_artifact_path"]
        ):
            raise ProfileDBCorruptionError(
                "successful stage omitted source artifact provenance"
            )
        try:
            target_stat = os.lstat(record["artifact_path"])
        except OSError as error:
            raise ProfileDBCorruptionError(
                "cannot stat content-addressed artifact: {}".format(error)
            )
        if not stat.S_ISREG(target_stat.st_mode) or target_stat.st_mode & 0o222:
            raise ProfileDBCorruptionError(
                "content-addressed artifact is not immutable regular storage"
            )
        facts = hash_artifact_file(record["artifact_path"])
        if (
            facts["artifact_sha256"] != record["artifact_sha256"]
            or facts["artifact_size"] != record["artifact_size"]
            or facts["artifact_path"] != record["artifact_path"]
        ):
            raise ProfileDBCorruptionError(
                "successful stage artifact no longer matches its sealed hash"
            )

    @staticmethod
    def _artifact_facts_match(observed, expected):
        return all(
            observed.get(name) == expected.get(name)
            for name in ("artifact_sha256", "artifact_size", "artifact_path")
        )

    def claim_stage(
        self,
        matrix_sha256,
        candidate_sha256,
        stage_name,
        inputs=_MISSING,
        input_sha256=None,
        reclaim_running=False,
    ):
        """Claim a stage or return ``skip`` for an intact exact success.

        An existing running row is ``busy`` by default, which prevents two
        workers from silently profiling the same stage.  After a known worker
        crash, callers may explicitly set ``reclaim_running=True``; this is
        recorded as another attempt and returns ``run``.
        """

        self._validate_stage_name(stage_name)
        digest = self._resolve_input_hash(inputs, input_sha256)
        if type(reclaim_running) is not bool:
            raise AutotuneError("reclaim_running must be a bool")
        self._begin()
        try:
            self._load_registered_candidate(matrix_sha256, candidate_sha256)
            row = self._connection.execute(
                "SELECT * FROM stages WHERE matrix_sha256 = ? "
                "AND candidate_sha256 = ? AND stage_name = ? "
                "AND input_sha256 = ?",
                (matrix_sha256, candidate_sha256, stage_name, digest),
            ).fetchone()
            if row is not None:
                record = self._row_to_record(row)
                if record["status"] == STAGE_SUCCEEDED:
                    self._verify_success_artifact(record)
                    self._commit()
                    return {
                        "action": "skip",
                        "claim_token": None,
                        "record": record,
                    }
                if record["status"] == STAGE_RUNNING and not reclaim_running:
                    record = dict(record)
                    record["claim_token"] = None
                    self._commit()
                    return {
                        "action": "busy",
                        "claim_token": None,
                        "record": record,
                    }
                attempt = record["attempt"] + 1
                claim_token = uuid.uuid4().hex
                self._connection.execute(
                    "UPDATE stages SET status = ?, artifact_sha256 = NULL, "
                    "artifact_path = NULL, artifact_size = NULL, "
                    "source_artifact_path = NULL, result_json = NULL, "
                    "claim_token = ?, attempt = ?, updated_utc = ? "
                    "WHERE matrix_sha256 = ? AND candidate_sha256 = ? "
                    "AND stage_name = ? AND input_sha256 = ?",
                    (
                        STAGE_RUNNING,
                        claim_token,
                        attempt,
                        _utc_now(),
                        matrix_sha256,
                        candidate_sha256,
                        stage_name,
                        digest,
                    ),
                )
                action = "run"
            else:
                other_running = self._connection.execute(
                    "SELECT * FROM stages WHERE matrix_sha256 = ? "
                    "AND candidate_sha256 = ? AND stage_name = ? "
                    "AND status = 'running' ORDER BY input_sha256 LIMIT 1",
                    (matrix_sha256, candidate_sha256, stage_name),
                ).fetchone()
                if other_running is not None:
                    record = self._row_to_record(other_running)
                    record = dict(record)
                    record["claim_token"] = None
                    self._commit()
                    return {
                        "action": "busy",
                        "claim_token": None,
                        "record": record,
                    }
                attempt = 1
                claim_token = uuid.uuid4().hex
                self._connection.execute(
                    "INSERT INTO stages(matrix_sha256, candidate_sha256, "
                    "stage_name, input_sha256, status, claim_token, artifact_sha256, "
                    "artifact_path, artifact_size, source_artifact_path, result_json, "
                    "attempt, updated_utc) "
                    "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?, ?)",
                    (
                        matrix_sha256,
                        candidate_sha256,
                        stage_name,
                        digest,
                        STAGE_RUNNING,
                        claim_token,
                        attempt,
                        _utc_now(),
                    ),
                )
                action = "run"
            row = self._connection.execute(
                "SELECT * FROM stages WHERE matrix_sha256 = ? "
                "AND candidate_sha256 = ? AND stage_name = ? "
                "AND input_sha256 = ?",
                (matrix_sha256, candidate_sha256, stage_name, digest),
            ).fetchone()
            record = self._row_to_record(row)
            self._commit()
            return {
                "action": action,
                "claim_token": claim_token,
                "record": record,
            }
        except Exception:
            self._rollback()
            raise

    def complete_stage(
        self,
        matrix_sha256,
        candidate_sha256,
        stage_name,
        artifact_path,
        claim_token=None,
        inputs=_MISSING,
        input_sha256=None,
        result=None,
    ):
        """Atomically mark a claimed stage successful with a verified artifact."""

        self._validate_stage_name(stage_name)
        self._validate_claim_token(claim_token)
        digest = self._resolve_input_hash(inputs, input_sha256)
        artifact = self._snapshot_artifact(artifact_path)
        result_json = canonical_json_bytes(result).decode("utf-8")
        self._begin()
        try:
            self._load_registered_candidate(matrix_sha256, candidate_sha256)
            row = self._connection.execute(
                "SELECT * FROM stages WHERE matrix_sha256 = ? "
                "AND candidate_sha256 = ? AND stage_name = ? "
                "AND input_sha256 = ?",
                (matrix_sha256, candidate_sha256, stage_name, digest),
            ).fetchone()
            if row is None:
                raise ProfileDBConflictError("stage must be claimed before completion")
            record = self._row_to_record(row)
            if record["status"] != STAGE_RUNNING:
                raise ProfileDBConflictError(
                    "only a running stage can be completed"
                )
            if record["claim_token"] != claim_token:
                raise ProfileDBConflictError(
                    "claim token does not own the current stage attempt"
                )
            before_update = hash_artifact_file(artifact["artifact_path"])
            if not self._artifact_facts_match(before_update, artifact):
                raise ProfileDBCorruptionError(
                    "artifact snapshot changed before DB transaction update"
                )
            cursor = self._connection.execute(
                "UPDATE stages SET status = ?, artifact_sha256 = ?, "
                "artifact_path = ?, artifact_size = ?, source_artifact_path = ?, "
                "result_json = ?, claim_token = NULL, "
                "updated_utc = ? WHERE matrix_sha256 = ? "
                "AND candidate_sha256 = ? AND stage_name = ? "
                "AND input_sha256 = ? AND status = ? AND claim_token = ?",
                (
                    STAGE_SUCCEEDED,
                    artifact["artifact_sha256"],
                    artifact["artifact_path"],
                    artifact["artifact_size"],
                    artifact["source_artifact_path"],
                    result_json,
                    _utc_now(),
                    matrix_sha256,
                    candidate_sha256,
                    stage_name,
                    digest,
                    STAGE_RUNNING,
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ProfileDBConflictError(
                    "stage attempt changed before completion"
                )
            before_commit = hash_artifact_file(artifact["artifact_path"])
            if not self._artifact_facts_match(before_commit, artifact):
                raise ProfileDBCorruptionError(
                    "artifact snapshot changed before successful DB commit"
                )
            row = self._connection.execute(
                "SELECT * FROM stages WHERE matrix_sha256 = ? "
                "AND candidate_sha256 = ? AND stage_name = ? "
                "AND input_sha256 = ?",
                (matrix_sha256, candidate_sha256, stage_name, digest),
            ).fetchone()
            record = self._row_to_record(row)
            self._commit()
        except Exception:
            self._rollback()
            raise
        try:
            after_commit = hash_artifact_file(artifact["artifact_path"])
            if not self._artifact_facts_match(after_commit, artifact):
                raise ProfileDBCorruptionError(
                    "artifact snapshot changed during successful DB commit"
                )
        except Exception as error:
            self._invalidate_completed_stage_after_artifact_race(
                matrix_sha256,
                candidate_sha256,
                stage_name,
                digest,
                artifact,
            )
            if isinstance(error, ProfileDBError):
                raise
            raise ProfileDBCorruptionError(
                "cannot verify artifact after DB commit: {}".format(error)
            )
        return record

    def _invalidate_completed_stage_after_artifact_race(
        self,
        matrix_sha256,
        candidate_sha256,
        stage_name,
        input_digest,
        artifact,
    ):
        """Fail closed if an artifact changes at the DB commit boundary."""

        self._begin()
        try:
            row = self._connection.execute(
                "SELECT * FROM stages WHERE matrix_sha256 = ? "
                "AND candidate_sha256 = ? AND stage_name = ? "
                "AND input_sha256 = ?",
                (matrix_sha256, candidate_sha256, stage_name, input_digest),
            ).fetchone()
            record = self._row_to_record(row)
            if (
                record is not None
                and record["status"] == STAGE_SUCCEEDED
                and record["artifact_sha256"] == artifact["artifact_sha256"]
                and record["artifact_path"] == artifact["artifact_path"]
                and record["artifact_size"] == artifact["artifact_size"]
            ):
                result_json = canonical_json_bytes(
                    {"error": "artifact changed during completion commit"}
                ).decode("utf-8")
                self._connection.execute(
                    "UPDATE stages SET status = ?, artifact_sha256 = NULL, "
                    "artifact_path = NULL, artifact_size = NULL, "
                    "source_artifact_path = NULL, result_json = ?, "
                    "claim_token = NULL, updated_utc = ? "
                    "WHERE matrix_sha256 = ? AND candidate_sha256 = ? "
                    "AND stage_name = ? AND input_sha256 = ?",
                    (
                        STAGE_FAILED,
                        result_json,
                        _utc_now(),
                        matrix_sha256,
                        candidate_sha256,
                        stage_name,
                        input_digest,
                    ),
                )
            self._commit()
        except Exception:
            self._rollback()
            raise

    def fail_stage(
        self,
        matrix_sha256,
        candidate_sha256,
        stage_name,
        error,
        claim_token=None,
        inputs=_MISSING,
        input_sha256=None,
    ):
        """Record a failed attempt; a later exact claim will retry it."""

        self._validate_stage_name(stage_name)
        self._validate_claim_token(claim_token)
        digest = self._resolve_input_hash(inputs, input_sha256)
        result_json = canonical_json_bytes({"error": error}).decode("utf-8")
        self._begin()
        try:
            self._load_registered_candidate(matrix_sha256, candidate_sha256)
            row = self._connection.execute(
                "SELECT * FROM stages WHERE matrix_sha256 = ? "
                "AND candidate_sha256 = ? AND stage_name = ? "
                "AND input_sha256 = ?",
                (matrix_sha256, candidate_sha256, stage_name, digest),
            ).fetchone()
            record = self._row_to_record(row)
            if record is None or record["status"] != STAGE_RUNNING:
                raise ProfileDBConflictError(
                    "only a running stage can be marked failed"
                )
            if record["claim_token"] != claim_token:
                raise ProfileDBConflictError(
                    "claim token does not own the current stage attempt"
                )
            cursor = self._connection.execute(
                "UPDATE stages SET status = ?, artifact_sha256 = NULL, "
                "artifact_path = NULL, artifact_size = NULL, "
                "source_artifact_path = NULL, result_json = ?, "
                "claim_token = NULL, updated_utc = ? WHERE matrix_sha256 = ? "
                "AND candidate_sha256 = ? AND stage_name = ? "
                "AND input_sha256 = ? AND status = ? AND claim_token = ?",
                (
                    STAGE_FAILED,
                    result_json,
                    _utc_now(),
                    matrix_sha256,
                    candidate_sha256,
                    stage_name,
                    digest,
                    STAGE_RUNNING,
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ProfileDBConflictError(
                    "stage attempt changed before failure recording"
                )
            row = self._connection.execute(
                "SELECT * FROM stages WHERE matrix_sha256 = ? "
                "AND candidate_sha256 = ? AND stage_name = ? "
                "AND input_sha256 = ?",
                (matrix_sha256, candidate_sha256, stage_name, digest),
            ).fetchone()
            record = self._row_to_record(row)
            self._commit()
            return record
        except Exception:
            self._rollback()
            raise

    def get_stage(
        self,
        matrix_sha256,
        candidate_sha256,
        stage_name,
        inputs=_MISSING,
        input_sha256=None,
        verify_artifact=True,
    ):
        self._validate_stage_name(stage_name)
        digest = self._resolve_input_hash(inputs, input_sha256)
        self._load_registered_candidate(matrix_sha256, candidate_sha256)
        row = self._connection.execute(
            "SELECT * FROM stages WHERE matrix_sha256 = ? "
            "AND candidate_sha256 = ? AND stage_name = ? AND input_sha256 = ?",
            (matrix_sha256, candidate_sha256, stage_name, digest),
        ).fetchone()
        record = self._row_to_record(row)
        if record is not None and verify_artifact:
            self._verify_success_artifact(record)
        return record

    def list_stages(self, matrix_sha256=None, verify_artifacts=True):
        if matrix_sha256 is None:
            rows = self._connection.execute(
                "SELECT * FROM stages ORDER BY matrix_sha256, candidate_sha256, "
                "stage_name, input_sha256"
            ).fetchall()
        else:
            self._load_registered_matrix(matrix_sha256)
            rows = self._connection.execute(
                "SELECT * FROM stages WHERE matrix_sha256 = ? "
                "ORDER BY candidate_sha256, stage_name, input_sha256",
                (matrix_sha256,),
            ).fetchall()
        records = [self._row_to_record(row) for row in rows]
        if verify_artifacts:
            for record in records:
                self._verify_success_artifact(record)
        return records

    def validate(self, verify_artifacts=True):
        """Run physical and logical integrity checks; raise on the first defect."""

        self._quick_check()
        self._foreign_key_and_orphan_check()
        rows = self._connection.execute(
            "SELECT matrix_sha256, matrix_json FROM matrices ORDER BY matrix_sha256"
        ).fetchall()
        for row in rows:
            try:
                matrix = parse_json_text(row["matrix_json"], "stored matrix")
                validate_matrix(matrix)
            except AutotuneError as error:
                raise ProfileDBCorruptionError(
                    "stored matrix failed validation: {}".format(error)
                )
            if row["matrix_sha256"] != matrix["matrix_sha256"]:
                raise ProfileDBCorruptionError("matrix row key mismatch")
            candidate_rows = self._connection.execute(
                "SELECT candidate_sha256, candidate_json FROM candidates "
                "WHERE matrix_sha256 = ? ORDER BY candidate_sha256",
                (row["matrix_sha256"],),
            ).fetchall()
            expected = {
                item["candidate_sha256"]: item for item in matrix["candidates"]
            }
            if set(item["candidate_sha256"] for item in candidate_rows) != set(expected):
                raise ProfileDBCorruptionError("candidate rows do not match matrix")
            for candidate_row in candidate_rows:
                candidate = parse_json_text(
                    candidate_row["candidate_json"], "stored candidate"
                )
                if candidate != expected[candidate_row["candidate_sha256"]]:
                    raise ProfileDBCorruptionError(
                        "candidate row contents do not match matrix"
                    )
        records = self.list_stages(verify_artifacts=verify_artifacts)
        for record in records:
            try:
                _require_sha256(record["matrix_sha256"], "matrix_sha256")
                _require_sha256(record["candidate_sha256"], "candidate_sha256")
                _require_sha256(record["input_sha256"], "input_sha256")
                self._validate_stage_name(record["stage_name"])
            except AutotuneError as error:
                raise ProfileDBCorruptionError(
                    "stored stage key failed validation: {}".format(error)
                )
            if record["status"] not in STAGE_STATUSES:
                raise ProfileDBCorruptionError("stored stage status is invalid")
        return True


def publish_qualification_profiles(matrix, profiles, output_dir):
    """Atomically publish one disabled profile per candidate and a hash manifest."""

    validate_matrix(matrix)
    if not isinstance(profiles, dict):
        raise AutotuneError("profiles must be a variant_id mapping")
    expected_names = [item["variant_id"] for item in matrix["candidates"]]
    if set(profiles) != set(expected_names):
        raise AutotuneError("profile set does not exactly cover matrix candidates")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    candidate_by_name = {
        item["variant_id"]: item for item in matrix["candidates"]
    }
    for variant_id in expected_names:
        profile = profiles[variant_id]
        candidate = candidate_by_name[variant_id]
        if not isinstance(profile, dict):
            raise AutotuneError("qualification profile must be an object")
        if profile.get("selected_variant_id") != variant_id:
            raise AutotuneError("qualification profile selected the wrong variant")
        deployment = profile.get("deployment")
        if deployment != {"enabled": False, "valid": False}:
            raise AutotuneError("qualification profile must remain disabled")
        provenance = profile.get("provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("matrix_sha256") != matrix["matrix_sha256"]
            or provenance.get("candidate_sha256")
            != candidate["candidate_sha256"]
        ):
            raise AutotuneError("qualification profile provenance is not matrix-bound")
        path = output_dir / "{}.json".format(variant_id)
        atomic_write_json(path, profile)
        facts = hash_artifact_file(path)
        artifacts.append(
            {
                "candidate_sha256": candidate["candidate_sha256"],
                "variant_id": variant_id,
                "path": facts["artifact_path"],
                "artifact_sha256": facts["artifact_sha256"],
                "artifact_size": facts["artifact_size"],
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "tacker_autotune_qualification_profiles",
        "matrix_sha256": matrix["matrix_sha256"],
        "profiles": artifacts,
    }
    manifest = dict(payload)
    manifest["manifest_sha256"] = canonical_sha256(
        payload, "tacker-autotune-profile-manifest-v1"
    )
    atomic_write_json(output_dir / "qualification_profiles.json", manifest)
    return manifest


def _validate_baseline_correctness(
    baseline_correctness, require_safe_baselines=True
):
    """Normalize baseline gates, optionally preserving explicit invalid rows.

    Schema-v1 formal planning retains its historical requirement that serial
    and two-stream are valid.  Phase-3.1 planning records invalid baselines in
    the sealed plan so fail-closed replacement logic can handle them without
    pretending they passed correctness.
    """

    if type(require_safe_baselines) is not bool:
        raise AutotuneError("require_safe_baselines must be a bool")
    if not isinstance(baseline_correctness, dict):
        raise AutotuneError("baseline correctness must be an object")
    expected = {"serial", "two_stream", "current_tacker"}
    if set(baseline_correctness) != expected:
        raise AutotuneError(
            "baseline correctness must exactly cover serial, two_stream, "
            "and current_tacker"
        )
    result = {}
    for name in sorted(expected):
        entry = baseline_correctness[name]
        if not isinstance(entry, dict) or type(entry.get("valid")) is not bool:
            raise AutotuneError(
                "baseline correctness {} requires boolean valid".format(name)
            )
        if (
            require_safe_baselines
            and name in ("serial", "two_stream")
            and entry["valid"] is not True
        ):
            raise AutotuneError(
                "formal benchmark requires {} correctness valid=true".format(
                    name
                )
            )
        result[name] = deepcopy(entry)
    return result


def build_formal_benchmark_plan(
    database,
    matrix_sha256,
    correctness_input_sha256,
    screening_input_sha256,
    top_k,
    profiles_dir,
    current_tacker_profile,
    baseline_correctness,
    correctness_json_path,
    correctness_stage="correctness",
    screening_stage="screening",
):
    """Select exact-input DB successes and emit a formal benchmark plan.

    The result contains the complete correctness mapping expected by
    ``benchmark_tacker_fps.py`` and a directly appendable argv fragment.  The
    caller/CLI supplies the workload-specific benchmark arguments separately.
    """

    if not isinstance(database, ProfileDB):
        raise AutotuneError("database must be a ProfileDB")
    _require_sha256(correctness_input_sha256, "correctness_input_sha256")
    _require_sha256(screening_input_sha256, "screening_input_sha256")
    if type(top_k) is not int or top_k < 1:
        raise AutotuneError("top_k must be a positive integer")
    database._validate_stage_name(correctness_stage)
    database._validate_stage_name(screening_stage)
    matrix = database.get_matrix(matrix_sha256)
    if matrix["launch_geometry"] is None:
        raise AutotuneError(
            "formal benchmark requires matrix launch_geometry"
        )
    baseline_entries = _validate_baseline_correctness(
        baseline_correctness,
        require_safe_baselines=(
            matrix["schema_version"] == MATRIX_SCHEMA_VERSION
        ),
    )

    eligible_candidates = []
    score_mapping = {}
    correctness_by_digest = {}
    checkpoint_bindings = {}
    for candidate in matrix["candidates"]:
        digest = candidate["candidate_sha256"]
        correctness_record = database.get_stage(
            matrix_sha256,
            digest,
            correctness_stage,
            input_sha256=correctness_input_sha256,
            verify_artifact=True,
        )
        if (
            correctness_record is None
            or correctness_record["status"] != STAGE_SUCCEEDED
        ):
            continue
        correctness_result = correctness_record["result"]
        if (
            not isinstance(correctness_result, dict)
            or type(correctness_result.get("valid")) is not bool
        ):
            raise ProfileDBCorruptionError(
                "successful correctness stage requires result.valid boolean"
            )
        if not correctness_result["valid"]:
            continue
        screening_record = database.get_stage(
            matrix_sha256,
            digest,
            screening_stage,
            input_sha256=screening_input_sha256,
            verify_artifact=True,
        )
        if screening_record is None or screening_record["status"] != STAGE_SUCCEEDED:
            continue
        screening_result = screening_record["result"]
        if not isinstance(screening_result, dict):
            raise ProfileDBCorruptionError(
                "successful screening stage requires an object result"
            )
        score = screening_result.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ProfileDBCorruptionError(
                "successful screening stage requires finite result.score"
            )
        eligible_candidates.append(candidate)
        score_mapping[digest] = float(score)
        correctness_by_digest[digest] = deepcopy(correctness_result)
        checkpoint_bindings[digest] = {
            "correctness_artifact_sha256": correctness_record[
                "artifact_sha256"
            ],
            "screening_artifact_sha256": screening_record["artifact_sha256"],
        }

    ranked = rank_screening_candidates(
        eligible_candidates, score_mapping, top_k=top_k, maximize=True
    )
    if not ranked:
        raise AutotuneError(
            "no correctness-valid candidate has an exact-input screening success"
        )

    profile_root = Path(profiles_dir).resolve()
    current_facts = hash_artifact_file(current_tacker_profile)
    correctness_path = str(Path(correctness_json_path).resolve())
    correctness = dict(baseline_entries)
    plan_candidates = []
    candidate_argv = []
    for ranked_item in ranked:
        candidate = ranked_item["candidate"]
        variant_id = candidate["variant_id"]
        profile_path = profile_root / "{}.json".format(variant_id)
        profile = load_json_file(profile_path)
        if profile.get("selected_variant_id") != variant_id:
            raise AutotuneError(
                "candidate profile {} selected another variant".format(profile_path)
            )
        if profile.get("deployment") != {"enabled": False, "valid": False}:
            raise AutotuneError(
                "formal candidate profile must be a disabled qualification profile"
            )
        provenance = profile.get("provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("matrix_sha256") != matrix_sha256
            or provenance.get("candidate_sha256")
            != candidate["candidate_sha256"]
        ):
            raise AutotuneError("formal candidate profile provenance mismatch")
        profile_facts = hash_artifact_file(profile_path)
        correctness[variant_id] = correctness_by_digest[
            candidate["candidate_sha256"]
        ]
        candidate_argv.extend(
            ["--candidate", "{}={}".format(variant_id, profile_facts["artifact_path"])]
        )
        plan_candidates.append(
            {
                "rank": len(plan_candidates) + 1,
                "candidate_sha256": candidate["candidate_sha256"],
                "variant_id": variant_id,
                "screening_score": ranked_item["score"],
                "profile_path": profile_facts["artifact_path"],
                "profile_file_sha256": profile_facts["artifact_sha256"],
                "checkpoint_artifacts": checkpoint_bindings[
                    candidate["candidate_sha256"]
                ],
            }
        )

    correctness_bytes = pretty_json_text(correctness).encode("utf-8")
    argv_fragment = [
        "--current-tacker-profile",
        current_facts["artifact_path"],
    ] + candidate_argv + ["--correctness-json", correctness_path]
    payload = {
        "schema_version": 1,
        "kind": "tacker_autotune_formal_benchmark_plan",
        "matrix_sha256": matrix_sha256,
        "correctness_stage_input_sha256": correctness_input_sha256,
        "screening_stage_input_sha256": screening_input_sha256,
        "top_k": top_k,
        "selection_objective": "screening_score",
        "higher_is_better": True,
        "required_baselines": ["serial", "two_stream", "current_tacker"],
        "current_tacker_profile": {
            "path": current_facts["artifact_path"],
            "artifact_sha256": current_facts["artifact_sha256"],
        },
        "candidates": plan_candidates,
        "correctness_json": {
            "path": correctness_path,
            "artifact_sha256": hashlib.sha256(correctness_bytes).hexdigest(),
        },
        "benchmark_driver": str(
            (PROJECT_ROOT / "scripts" / "benchmark_tacker_fps.py").resolve()
        ),
        "benchmark_argv_fragment": argv_fragment,
        "correctness_qualifications": correctness,
    }
    plan_hash_domain = "tacker-autotune-formal-plan-v1"
    if matrix["schema_version"] == PHASE31_MATRIX_SCHEMA_VERSION:
        payload["schema_version"] = 2
        payload["baseline_terminal_status"] = {
            name: (
                "correctness_valid"
                if baseline_entries[name]["valid"]
                else "correctness_invalid"
            )
            for name in ("serial", "two_stream", "current_tacker")
        }
        payload["invalid_baselines"] = [
            name
            for name in ("serial", "two_stream", "current_tacker")
            if not baseline_entries[name]["valid"]
        ]
        plan_hash_domain = "tacker-autotune-formal-plan-v2"
    plan = dict(payload)
    plan["plan_sha256"] = canonical_sha256(
        payload, plan_hash_domain
    )
    return plan


def _write_or_print_json(path, value):
    if path == "-":
        sys.stdout.write(pretty_json_text(value))
    else:
        atomic_write_json(path, value)


def _inputs_from_args(args):
    if getattr(args, "input_sha256", None) is not None:
        return (_MISSING, args.input_sha256)
    return (load_json_file(args.inputs), None)


def _add_stage_key_arguments(parser):
    parser.add_argument("--db", required=True)
    parser.add_argument("--matrix-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--stage", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inputs", help="JSON file containing exact stage inputs")
    group.add_argument("--input-sha256")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    matrix_parser = subparsers.add_parser(
        "matrix", help="emit deterministic C0/C1/C2 base candidates"
    )
    matrix_parser.add_argument("--sm-count", type=int)
    matrix_parser.add_argument("--raster-tile-count", type=int)
    matrix_parser.add_argument("--backend-logical-blocks", type=int)
    matrix_parser.add_argument("--current-persistent-blocks", type=int, default=7000)
    matrix_parser.add_argument(
        "--persistent-block", type=int, action="append", default=[]
    )
    matrix_parser.add_argument("--output", default="-")

    phase31_matrix_parser = subparsers.add_parser(
        "phase31-matrix",
        help="emit schema-v2 exhaustive C0--C2 and declare C3/C4 PB grids",
    )
    phase31_matrix_parser.add_argument("--sm-count", type=int, required=True)
    phase31_matrix_parser.add_argument(
        "--raster-tile-count", type=int, required=True
    )
    phase31_matrix_parser.add_argument(
        "--backend-logical-blocks", type=int, required=True
    )
    phase31_matrix_parser.add_argument(
        "--whole-head-logical-blocks", type=int, required=True
    )
    phase31_matrix_parser.add_argument(
        "--current-persistent-blocks", type=int, default=7000
    )
    phase31_matrix_parser.add_argument(
        "--persistent-block", type=int, action="append", default=[]
    )
    phase31_matrix_parser.add_argument(
        "--packed-persistent-block", type=int, action="append", default=[]
    )
    phase31_matrix_parser.add_argument(
        "--whole-head-persistent-block", type=int, action="append", default=[]
    )
    phase31_matrix_parser.add_argument("--output", default="-")

    phase31_c3_parser = subparsers.add_parser(
        "phase31-c3",
        help="append C3 packed candidates from top screened C2 head sets",
    )
    phase31_c3_parser.add_argument("--matrix", required=True)
    phase31_c3_parser.add_argument("--screening", required=True)
    phase31_c3_parser.add_argument("--top-k", type=int, required=True)
    phase31_c3_parser.add_argument("--minimize", action="store_true")
    phase31_c3_parser.add_argument("--output", default="-")

    phase31_c4_parser = subparsers.add_parser(
        "phase31-c4",
        help="append C4 whole-head candidates from top C1/C2/C3 head sets",
    )
    phase31_c4_parser.add_argument("--matrix", required=True)
    phase31_c4_parser.add_argument("--screening", required=True)
    phase31_c4_parser.add_argument(
        "--top-k-per-family", type=int, required=True
    )
    phase31_c4_parser.add_argument("--minimize", action="store_true")
    phase31_c4_parser.add_argument("--output", default="-")

    ranking_parser = subparsers.add_parser(
        "screening-ranking",
        help="seal full screening ranking and candidate terminal statuses",
    )
    ranking_parser.add_argument("--matrix", required=True)
    ranking_source = ranking_parser.add_mutually_exclusive_group(required=True)
    ranking_source.add_argument("--screening")
    ranking_source.add_argument("--db")
    ranking_parser.add_argument("--screening-input-sha256")
    ranking_parser.add_argument("--screening-stage", default="screening")
    ranking_parser.add_argument("--allow-incomplete", action="store_true")
    ranking_parser.add_argument("--minimize", action="store_true")
    ranking_parser.add_argument("--output", default="-")

    formal_set_parser = subparsers.add_parser(
        "formal-set",
        help="seal global screening top-K union best C0--C4 representatives",
    )
    formal_set_parser.add_argument("--matrix", required=True)
    formal_set_parser.add_argument("--screening-ranking", required=True)
    formal_set_parser.add_argument("--top-k", type=int, required=True)
    formal_set_parser.add_argument("--output", default="-")

    beam_parser = subparsers.add_parser(
        "beam", help="append one screening-selected 3--5-head beam level"
    )
    beam_parser.add_argument("--matrix", required=True)
    beam_parser.add_argument("--screening", required=True)
    beam_parser.add_argument("--beam-width", required=True, type=int)
    beam_parser.add_argument("--target-head-count", required=True, type=int)
    beam_parser.add_argument("--minimize", action="store_true")
    beam_parser.add_argument("--output", default="-")

    search_parser = subparsers.add_parser(
        "beam-search", help="append successive beam levels through 3--5 heads"
    )
    search_parser.add_argument("--matrix", required=True)
    search_parser.add_argument("--screening-by-level", required=True)
    search_parser.add_argument("--beam-width", required=True, type=int)
    search_parser.add_argument("--max-heads", type=int, default=5)
    search_parser.add_argument("--minimize", action="store_true")
    search_parser.add_argument("--output", default="-")

    descriptor_parser = subparsers.add_parser(
        "descriptors", help="materialize runtime-owned profile descriptors"
    )
    descriptor_parser.add_argument("--matrix", required=True)
    descriptor_parser.add_argument("--output", default="-")

    profiles_parser = subparsers.add_parser(
        "profiles", help="publish disabled qualification profiles for a matrix"
    )
    profiles_parser.add_argument("--matrix", required=True)
    profiles_parser.add_argument(
        "--resources",
        required=True,
        help=(
            "JSON mapping ABI family then worker_groups to runtime resource facts"
        ),
    )
    profiles_parser.add_argument("--template")
    profiles_parser.add_argument("--output-dir", required=True)

    register_parser = subparsers.add_parser(
        "db-register", help="transactionally register a sealed matrix"
    )
    register_parser.add_argument("--db", required=True)
    register_parser.add_argument("--matrix", required=True)

    claim_parser = subparsers.add_parser(
        "db-claim", help="claim, resume, or skip an exact stage"
    )
    _add_stage_key_arguments(claim_parser)
    claim_parser.add_argument("--reclaim-running", action="store_true")

    complete_parser = subparsers.add_parser(
        "db-complete", help="seal a successful stage artifact"
    )
    _add_stage_key_arguments(complete_parser)
    complete_parser.add_argument("--artifact", required=True)
    complete_parser.add_argument("--claim-token", required=True)
    complete_parser.add_argument("--result", help="optional JSON result file")

    fail_parser = subparsers.add_parser(
        "db-fail", help="record a failed stage attempt"
    )
    _add_stage_key_arguments(fail_parser)
    fail_parser.add_argument("--error", required=True)
    fail_parser.add_argument("--claim-token", required=True)

    status_parser = subparsers.add_parser(
        "db-status", help="validate and list profile stage checkpoints"
    )
    status_parser.add_argument("--db", required=True)
    status_parser.add_argument("--matrix-sha256")

    validate_parser = subparsers.add_parser(
        "db-validate", help="validate DB contents and all successful artifacts"
    )
    validate_parser.add_argument("--db", required=True)

    formal_parser = subparsers.add_parser(
        "formal-plan",
        help="select exact DB successes for the formal interleaved benchmark",
    )
    formal_parser.add_argument("--db", required=True)
    formal_parser.add_argument("--matrix-sha256", required=True)
    formal_parser.add_argument("--correctness-input-sha256", required=True)
    formal_parser.add_argument("--screening-input-sha256", required=True)
    formal_parser.add_argument("--top-k", type=int, required=True)
    formal_parser.add_argument("--profiles-dir", required=True)
    formal_parser.add_argument("--current-tacker-profile", required=True)
    formal_parser.add_argument("--baseline-correctness", required=True)
    formal_parser.add_argument("--correctness-output", required=True)
    formal_parser.add_argument("--correctness-stage", default="correctness")
    formal_parser.add_argument("--screening-stage", default="screening")
    formal_parser.add_argument("--output", default="-")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required")

    if args.command == "matrix":
        fact_values = (
            args.sm_count,
            args.raster_tile_count,
            args.backend_logical_blocks,
        )
        if all(value is not None for value in fact_values):
            blocks = derive_persistent_blocks(
                args.sm_count,
                args.raster_tile_count,
                args.backend_logical_blocks,
                current_persistent_blocks=args.current_persistent_blocks,
                extra_values=args.persistent_block,
            )
        elif any(value is not None for value in fact_values):
            parser.error(
                "--sm-count, --raster-tile-count, and "
                "--backend-logical-blocks must be provided together"
            )
        else:
            parser.error(
                "matrix generation requires SM, Raster, and backend logical-block facts"
            )
        if all(value is not None for value in fact_values):
            matrix = build_base_matrix(
                blocks,
                sm_count=args.sm_count,
                raster_tile_count=args.raster_tile_count,
                backend_logical_blocks=args.backend_logical_blocks,
            )
        else:
            matrix = build_base_matrix(blocks)
        _write_or_print_json(args.output, matrix)
        return 0

    if args.command == "phase31-matrix":
        blocks = derive_persistent_blocks(
            args.sm_count,
            args.raster_tile_count,
            args.backend_logical_blocks,
            current_persistent_blocks=args.current_persistent_blocks,
            extra_values=args.persistent_block,
        )
        packed_blocks = normalize_persistent_blocks(
            list(blocks) + args.packed_persistent_block
        )
        whole_blocks = normalize_persistent_blocks(
            list(blocks) + args.whole_head_persistent_block
        )
        matrix = build_phase31_base_matrix(
            blocks,
            sm_count=args.sm_count,
            raster_tile_count=args.raster_tile_count,
            backend_logical_blocks=args.backend_logical_blocks,
            whole_head_logical_blocks=args.whole_head_logical_blocks,
            packed_persistent_blocks=packed_blocks,
            whole_head_persistent_blocks=whole_blocks,
        )
        _write_or_print_json(args.output, matrix)
        return 0

    if args.command == "phase31-c3":
        matrix = load_json_file(args.matrix)
        screening = load_json_file(args.screening)
        result = extend_phase31_with_c3(
            matrix,
            screening,
            args.top_k,
            maximize=not args.minimize,
        )
        _write_or_print_json(args.output, result)
        return 0

    if args.command == "phase31-c4":
        matrix = load_json_file(args.matrix)
        screening = load_json_file(args.screening)
        result = extend_phase31_with_c4(
            matrix,
            screening,
            args.top_k_per_family,
            maximize=not args.minimize,
        )
        _write_or_print_json(args.output, result)
        return 0

    if args.command == "screening-ranking":
        matrix = load_json_file(args.matrix)
        validate_matrix(matrix)
        if args.db is not None:
            if args.screening_input_sha256 is None:
                parser.error(
                    "screening-ranking --db requires --screening-input-sha256"
                )
            with ProfileDB(args.db) as database:
                registered = database.get_matrix(matrix["matrix_sha256"])
                if registered != matrix:
                    raise AutotuneError(
                        "ranking matrix differs from the registered matrix"
                    )
                result = build_screening_ranking_from_db(
                    database,
                    matrix["matrix_sha256"],
                    args.screening_input_sha256,
                    screening_stage=args.screening_stage,
                    maximize=not args.minimize,
                    require_terminal=not args.allow_incomplete,
                )
        else:
            screening = load_json_file(args.screening)
            result = build_screening_ranking(
                matrix,
                screening,
                screening_input_sha256=args.screening_input_sha256,
                maximize=not args.minimize,
                require_terminal=not args.allow_incomplete,
            )
        _write_or_print_json(args.output, result)
        return 0

    if args.command == "formal-set":
        matrix = load_json_file(args.matrix)
        ranking = load_json_file(args.screening_ranking)
        result = build_formal_candidate_set(matrix, ranking, args.top_k)
        _write_or_print_json(args.output, result)
        return 0

    if args.command == "beam":
        matrix = load_json_file(args.matrix)
        screening = load_json_file(args.screening)
        result = extend_matrix_with_beam(
            matrix,
            screening,
            args.beam_width,
            args.target_head_count,
            maximize=not args.minimize,
        )
        _write_or_print_json(args.output, result)
        return 0

    if args.command == "beam-search":
        matrix = load_json_file(args.matrix)
        screening = load_json_file(args.screening_by_level)
        result = run_beam_search(
            matrix,
            screening,
            args.beam_width,
            max_heads=args.max_heads,
            maximize=not args.minimize,
        )
        _write_or_print_json(args.output, result)
        return 0

    if args.command == "descriptors":
        matrix = load_json_file(args.matrix)
        descriptors = materialize_matrix_descriptors(matrix)
        _write_or_print_json(
            args.output,
            {
                "matrix_sha256": matrix["matrix_sha256"],
                "descriptors": descriptors,
            },
        )
        return 0

    if args.command == "profiles":
        matrix = load_json_file(args.matrix)
        resources = load_json_file(args.resources)
        provider = _resource_provider_from_document(resources)
        module = _load_tacker_pipeline()
        if args.template is None:
            template = module.load_tacker_profile()
        else:
            template = module.load_tacker_profile(profile_path=args.template)
        profiles = build_qualification_profiles(
            matrix,
            provider,
            module=module,
            template=template,
        )
        manifest = publish_qualification_profiles(
            matrix, profiles, args.output_dir
        )
        _write_or_print_json("-", manifest)
        return 0

    if args.command == "db-register":
        matrix = load_json_file(args.matrix)
        with ProfileDB(args.db) as database:
            digest = database.register_matrix(matrix)
        _write_or_print_json("-", {"matrix_sha256": digest, "registered": True})
        return 0

    if args.command in ("db-claim", "db-complete", "db-fail"):
        inputs, explicit_digest = _inputs_from_args(args)
        with ProfileDB(args.db) as database:
            if args.command == "db-claim":
                record = database.claim_stage(
                    args.matrix_sha256,
                    args.candidate_sha256,
                    args.stage,
                    inputs=inputs,
                    input_sha256=explicit_digest,
                    reclaim_running=args.reclaim_running,
                )
            elif args.command == "db-complete":
                result = None
                if args.result is not None:
                    result = load_json_file(args.result)
                record = database.complete_stage(
                    args.matrix_sha256,
                    args.candidate_sha256,
                    args.stage,
                    args.artifact,
                    claim_token=args.claim_token,
                    inputs=inputs,
                    input_sha256=explicit_digest,
                    result=result,
                )
            else:
                record = database.fail_stage(
                    args.matrix_sha256,
                    args.candidate_sha256,
                    args.stage,
                    args.error,
                    claim_token=args.claim_token,
                    inputs=inputs,
                    input_sha256=explicit_digest,
                )
        _write_or_print_json("-", record)
        return 0

    if args.command == "db-status":
        with ProfileDB(args.db) as database:
            database.validate(verify_artifacts=True)
            records = database.list_stages(
                matrix_sha256=args.matrix_sha256, verify_artifacts=True
            )
        _write_or_print_json("-", {"stages": records, "valid": True})
        return 0

    if args.command == "db-validate":
        with ProfileDB(args.db) as database:
            database.validate(verify_artifacts=True)
        _write_or_print_json("-", {"valid": True})
        return 0


    if args.command == "formal-plan":
        baselines = load_json_file(args.baseline_correctness)
        with ProfileDB(args.db) as database:
            plan = build_formal_benchmark_plan(
                database,
                args.matrix_sha256,
                args.correctness_input_sha256,
                args.screening_input_sha256,
                args.top_k,
                args.profiles_dir,
                args.current_tacker_profile,
                baselines,
                args.correctness_output,
                correctness_stage=args.correctness_stage,
                screening_stage=args.screening_stage,
            )
        atomic_write_json(
            args.correctness_output, plan["correctness_qualifications"]
        )
        _write_or_print_json(args.output, plan)
        return 0

    raise AssertionError("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
