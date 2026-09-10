"""Inference-only Raster(t) + deformation-head(t+1) Tacker pipeline.

The production path is deliberately fail closed.  It is enabled only by a
hashed schema-v2 profile whose selected candidate passed correctness
qualification and offline whole-run FPS selection, plus an exact
model/rasterizer contract.  Read-only schema-v1 profiles remain runnable
during migration, but their historical Raster-QoS diagnostics are not
reinterpreted as runtime gates.
When any part of that contract is missing, :class:`TackerRenderer` delegates
the complete sequence to ``TwoStreamRenderer`` and exposes the reason.

Only the first ``Linear(128, 128)`` in ``pos_deform`` is physically fused.
The prefix produces its real activation input, the mixed rasterizer produces
the real FP32 Linear output, and the suffix consumes that output to construct
the next frame's ``GaussianRenderState``.  The selected Linear is therefore
never executed a second time by PyTorch.
"""

from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

import diff_gaussian_rasterization as _rasterizer_module

try:
    from . import (
        GaussianRasterizer,
        GaussianRenderState,
        RenderResult,
        TwoStreamRenderer,
        _record_context_stream,
        deform_for_render,
        prepare_render_context,
        rasterize_state,
    )
except (ImportError, ValueError):  # Supports direct import in CPU contract tests.
    from gaussian_renderer import (  # type: ignore
        GaussianRasterizer,
        GaussianRenderState,
        RenderResult,
        TwoStreamRenderer,
        _record_context_stream,
        deform_for_render,
        prepare_render_context,
        rasterize_state,
    )


PROFILE_SCHEMA_VERSION = 2
LEGACY_PROFILE_SCHEMA_VERSION = 1
RASTERIZER_COMMIT = "e49506654e8e11ed8a62d22bcb693e943fdecacf"
PAIR_KEY = "raster.render_leaf+deformation.pos_deform[1].linear_128x128"
LEGACY_VARIANT_ID = "legacy_pos_l1"
SELECTION_OBJECTIVE = "median_throughput_fps"
WORKLOAD_KEY = "flame_steak:14000:111525:1352x1014:sm_86"
EQUIVALENCE_FRACTION = 0.005
PROMOTION_MIN_FPS_RATIO = 1.01
PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE = 1.0
FORMAL_BOOTSTRAP_RESAMPLES = 10000
FORMAL_BOOTSTRAP_SEED = 0
MIN_SELECTION_TRIALS = 10
MIXED_ABI_SHA256 = (
    "231c90c429321b2673b88ecd09efb40b6aedda7a23f3e061a2bcedec06d44426"
)
DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tacker_profiles"
    / "raster_head_sm86.json"
)

_CORRECTNESS_LIMITS = {
    "psnr_drop_db_max": 0.05,
    "ssim_drop_max": 0.0001,
    "lpips_increase_max": 0.0001,
}

# Schema-v1 profiles remain readable during migration.  Their old performance
# limits are checked for structural compatibility only; Phase 1 deliberately
# does not use them as runtime admission gates.
_LEGACY_LOCKED_LIMITS = {
    "raster_slowdown_pct_max": 5.0,
    "end_to_end_ratio_max": 1.0,
    **_CORRECTNESS_LIMITS,
}

_MEASUREMENT_KEYS = (
    "raster_slowdown_pct",
    "mixed_p50_ms",
    "solo_raster_p50_ms",
    "solo_head_p50_ms",
    "tacker_end_to_end_p50_ms",
    "two_stream_end_to_end_p50_ms",
    "psnr_drop_db",
    "ssim_drop",
    "lpips_increase",
)


class TackerProfileError(ValueError):
    """Raised when an admission profile is malformed or has a bad hash."""


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(manifest):
    """Return the stable SHA-256 used by the profile's sealed manifest."""

    return hashlib.sha256(_canonical_json(manifest)).hexdigest()


_PROFILE_SEALED_KEYS = (
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


def profile_sha256(profile):
    """Hash every schema-v2 field that can change dispatch or selection.

    Human notes and generation timestamps are intentionally outside this
    payload.  Candidate trials, correctness results, input hashes, the chosen
    variant, and the sealed workload/ABI manifest are all covered.
    """

    if not isinstance(profile, dict):
        raise TackerProfileError("profile must be a JSON object")
    payload = {key: profile.get(key) for key in _PROFILE_SEALED_KEYS}
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        provenance.pop("generated_at_utc", None)
        payload["provenance"] = provenance
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _is_finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _require_exact(mapping, key, expected, section):
    if key not in mapping:
        raise TackerProfileError("{}.{} is required".format(section, key))
    if mapping[key] != expected:
        raise TackerProfileError(
            "{}.{} must be {!r}".format(section, key, expected)
        )


def _validate_legacy_tacker_profile(profile):
    """Validate the read-only schema-v1 profile contract."""

    if not isinstance(profile, dict):
        raise TackerProfileError("profile must be a JSON object")
    _require_exact(
        profile,
        "schema_version",
        LEGACY_PROFILE_SCHEMA_VERSION,
        "profile",
    )

    manifest = profile.get("manifest")
    if not isinstance(manifest, dict):
        raise TackerProfileError("profile.manifest must be an object")
    digest = profile.get("manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise TackerProfileError("profile.manifest_sha256 must be a SHA-256 hex string")
    expected_digest = manifest_sha256(manifest)
    if digest.lower() != expected_digest:
        raise TackerProfileError("profile manifest SHA-256 mismatch")

    _require_exact(manifest, "rasterizer_commit", RASTERIZER_COMMIT, "manifest")
    _require_exact(manifest, "pair_key", PAIR_KEY, "manifest")
    _require_exact(manifest, "cuda_arch", "sm_86", "manifest")
    _require_exact(manifest, "compute_capability", [8, 6], "manifest")
    _require_exact(manifest, "gpu_name", "NVIDIA RTX A6000", "manifest")
    _require_exact(manifest, "workload", "flame_steak", "manifest")
    _require_exact(manifest, "iteration", 14000, "manifest")
    _require_exact(manifest, "gaussian_count", 111525, "manifest")
    _require_exact(manifest, "resolution", [1352, 1014], "manifest")
    _require_exact(manifest, "physical_cta_threads", 384, "manifest")
    _require_exact(manifest, "raster_thread_range_inclusive", [0, 255], "manifest")
    _require_exact(manifest, "head_thread_range_inclusive", [256, 383], "manifest")
    _require_exact(manifest, "raster_named_barrier_id", 1, "manifest")
    _require_exact(manifest, "head_named_barrier_ids", [], "manifest")
    _require_exact(manifest, "head_input_dtype", "float16", "manifest")
    _require_exact(manifest, "head_weight_dtype", "float16", "manifest")
    _require_exact(manifest, "head_bias_dtype", "float32", "manifest")
    _require_exact(manifest, "head_accumulation_dtype", "float32", "manifest")
    _require_exact(manifest, "head_output_dtype", "float32", "manifest")

    persistent_blocks = manifest.get("persistent_blocks")
    if (
        not isinstance(persistent_blocks, int)
        or isinstance(persistent_blocks, bool)
        or persistent_blocks < 0
    ):
        raise TackerProfileError("manifest.persistent_blocks must be an int >= 0")

    thresholds = profile.get("thresholds")
    if not isinstance(thresholds, dict):
        raise TackerProfileError("profile.thresholds must be an object")
    for key, locked_limit in _LEGACY_LOCKED_LIMITS.items():
        value = thresholds.get(key)
        if not _is_finite_number(value):
            raise TackerProfileError("thresholds.{} must be finite".format(key))
        if float(value) > locked_limit:
            raise TackerProfileError(
                "thresholds.{} weakens the locked limit {}".format(
                    key, locked_limit
                )
            )
    _require_exact(
        thresholds,
        "mixed_p50_strictly_less_than_solo_sum",
        True,
        "thresholds",
    )

    admission = profile.get("admission")
    if not isinstance(admission, dict):
        raise TackerProfileError("profile.admission must be an object")
    for key in ("enabled", "valid"):
        if type(admission.get(key)) is not bool:
            raise TackerProfileError("admission.{} must be boolean".format(key))

    measurements = profile.get("measurements")
    if measurements is not None and not isinstance(measurements, dict):
        raise TackerProfileError("profile.measurements must be null or an object")
    if admission["enabled"] or admission["valid"]:
        if not isinstance(measurements, dict):
            raise TackerProfileError(
                "enabled/valid profiles require measured admission data"
            )
        for key in _MEASUREMENT_KEYS:
            if not _is_finite_number(measurements.get(key)):
                raise TackerProfileError(
                    "measurements.{} must be finite".format(key)
                )
        for key in (
            "mixed_p50_ms",
            "solo_raster_p50_ms",
            "solo_head_p50_ms",
            "tacker_end_to_end_p50_ms",
            "two_stream_end_to_end_p50_ms",
        ):
            if float(measurements[key]) <= 0.0:
                raise TackerProfileError(
                    "measurements.{} must be greater than zero".format(key)
                )
    return profile


def _require_sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TackerProfileError("{} must be a lowercase SHA-256".format(label))


def _validate_v2_manifest(manifest, workload_key):
    if not isinstance(manifest, dict):
        raise TackerProfileError("profile.manifest must be an object")
    expected = {
        "workload_key": workload_key,
        "rasterizer_commit": RASTERIZER_COMMIT,
        "cuda_arch": "sm_86",
        "compute_capability": [8, 6],
        "gpu_name": "NVIDIA RTX A6000",
        "workload": "flame_steak",
        "iteration": 14000,
        "gaussian_count": 111525,
        "resolution": [1352, 1014],
    }
    for key, value in expected.items():
        _require_exact(manifest, key, value, "manifest")
    persistent_blocks = manifest.get("persistent_blocks")
    if (
        type(persistent_blocks) is not int
        or persistent_blocks < 0
    ):
        raise TackerProfileError("manifest.persistent_blocks must be an int >= 0")


def _validate_pos_l1_v2_candidate(candidate):
    """Validate a current-ABI pos-L1 variant (including block-count scans)."""

    section = "candidate {}".format(candidate.get("variant_id", "<unknown>"))
    expected = {
        "execution_mode": "tacker",
        "pair_key": PAIR_KEY,
        "cuda_symbol": "tacker_mix_render_head_v1",
        "abi_manifest_sha256": MIXED_ABI_SHA256,
        "physical_cta_threads": 384,
        "raster_threads": 256,
        "raster_thread_range_inclusive": [0, 255],
        "raster_named_barrier_id": 1,
        "persistent_blocks": candidate.get("persistent_blocks"),
    }
    for key, value in expected.items():
        _require_exact(candidate, key, value, section)
    persistent_blocks = candidate.get("persistent_blocks")
    if type(persistent_blocks) is not int or persistent_blocks < 0:
        raise TackerProfileError(
            "{}.persistent_blocks must be an int >= 0".format(section)
        )
    fused_nodes = candidate.get("fused_nodes")
    if fused_nodes != [
        "raster.render_leaf",
        "deformation.pos_deform[1].linear_128x128",
    ]:
        raise TackerProfileError("{}.fused_nodes changed".format(section))
    task_graph = {
        "parallel_nodes": [
            "deformation.scales_deform",
            "deformation.rotations_deform",
            "deformation.opacity_deform",
            "deformation.shs_deform",
        ],
        "suffix_nodes": [
            "deformation.pos_deform[2]",
            "deformation.pos_deform[3]",
            "deformation.apply_residuals",
        ],
        "required_outputs": [
            "raster.color",
            "raster.depth",
            "raster.radii",
            "deformation.pos_delta",
        ],
    }
    for key, value in task_graph.items():
        _require_exact(candidate, key, value, section)
    subgroups = candidate.get("backend_subgroups")
    if subgroups != [
        {
            "name": "pos_deform_l1",
            "thread_range_inclusive": [256, 383],
            "threads": 128,
            "named_barrier_ids": [],
        }
    ]:
        raise TackerProfileError("{}.backend_subgroups changed".format(section))
    tensor_contract = candidate.get("tensor_contract")
    if not isinstance(tensor_contract, dict):
        raise TackerProfileError("{}.tensor_contract must be an object".format(section))
    expected_dtypes = {
        "input_dtype": "float16",
        "weight_dtype": "float16",
        "bias_dtype": "float32",
        "accumulation_dtype": "float32",
        "output_dtype": "float32",
    }
    for key, value in expected_dtypes.items():
        _require_exact(tensor_contract, key, value, "{}.tensor_contract".format(section))
    capability = candidate.get("capability_requirements")
    if not isinstance(capability, dict):
        raise TackerProfileError(
            "{}.capability_requirements must be an object".format(section)
        )
    _require_exact(capability, "cuda_arch", "sm_86", "{}.capability_requirements".format(section))
    _require_exact(capability, "compute_capability", [8, 6], "{}.capability_requirements".format(section))
    _require_exact(candidate, "tile_shape", [16, 16], section)
    if "resources" not in candidate:
        raise TackerProfileError("{}.resources is required".format(section))
    resources = candidate["resources"]
    if resources is not None:
        if not isinstance(resources, dict):
            raise TackerProfileError(
                "{}.resources must be null or an object".format(section)
            )
        for name, value in resources.items():
            if value is not None and (
                not _is_finite_number(value) or float(value) < 0.0
            ):
                raise TackerProfileError(
                    "{}.resources.{} must be null or a finite non-negative number"
                    .format(section, name)
                )


def _validate_v2_candidate(candidate, correctness_limits):
    if not isinstance(candidate, dict):
        raise TackerProfileError("profile.candidates entries must be objects")
    variant_id = candidate.get("variant_id")
    if not isinstance(variant_id, str) or not variant_id:
        raise TackerProfileError("candidate.variant_id must be a non-empty string")
    execution_mode = candidate.get("execution_mode")
    if execution_mode not in ("serial", "two_stream", "tacker"):
        raise TackerProfileError(
            "candidate {} has an invalid execution_mode".format(variant_id)
        )
    correctness = candidate.get("correctness")
    if not isinstance(correctness, dict) or type(correctness.get("valid")) is not bool:
        raise TackerProfileError(
            "candidate {} correctness.valid must be boolean".format(variant_id)
        )
    if correctness["valid"]:
        if correctness.get("actual_execution_mode") != execution_mode:
            raise TackerProfileError(
                "candidate {} did not execute its declared physical mode".format(
                    variant_id
                )
            )
        if correctness.get("fallback_reason") is not None:
            raise TackerProfileError(
                "candidate {} correctness recorded a fallback".format(variant_id)
            )
        if execution_mode == "tacker":
            quality_keys = {
                "psnr_drop_db": "psnr_drop_db_max",
                "ssim_drop": "ssim_drop_max",
                "lpips_increase": "lpips_increase_max",
            }
            for measurement, limit_name in quality_keys.items():
                value = correctness.get(measurement)
                if not _is_finite_number(value):
                    raise TackerProfileError(
                        "candidate {} correctness.{} must be finite".format(
                            variant_id, measurement
                        )
                    )
                if float(value) > correctness_limits[limit_name]:
                    raise TackerProfileError(
                        "candidate {} exceeds correctness threshold {}".format(
                            variant_id, limit_name
                        )
                    )

    performance = candidate.get("performance")
    if performance is not None:
        if not isinstance(performance, dict):
            raise TackerProfileError(
                "candidate {} performance must be null or an object".format(
                    variant_id
                )
            )
        trials = performance.get("throughput_fps_trials")
        if not isinstance(trials, list) or not trials:
            raise TackerProfileError(
                "candidate {} requires whole-run FPS trials".format(variant_id)
            )
        if any(not _is_finite_number(value) or float(value) <= 0.0 for value in trials):
            raise TackerProfileError(
                "candidate {} FPS trials must be finite and > 0".format(variant_id)
            )
        median_fps = performance.get("median_throughput_fps")
        expected_median = statistics.median(float(value) for value in trials)
        if (
            not _is_finite_number(median_fps)
            or not math.isclose(
                float(median_fps), expected_median, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise TackerProfileError(
                "candidate {} median_throughput_fps disagrees with trials".format(
                    variant_id
                )
            )
    diagnostics = candidate.get("diagnostics")
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise TackerProfileError(
            "candidate {} diagnostics must be null or an object".format(variant_id)
        )
    if execution_mode == "tacker":
        _validate_pos_l1_v2_candidate(candidate)
    return candidate


def _same_finite_number(actual, expected, label):
    if not _is_finite_number(actual) or not math.isclose(
        float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise TackerProfileError("{} disagrees with whole-run trials".format(label))
    return float(actual)


def _percentile(sorted_values, probability):
    if not sorted_values:
        raise TackerProfileError("cannot compute an empty bootstrap percentile")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


class _DeterministicGenerator(object):
    """Version-independent SHA-256 sampler shared with Phase-1 admission."""

    def __init__(self, seed, label):
        self._key = _canonical_json({"seed": seed, "label": label})
        self._counter = 0

    def index(self, stop):
        material = self._key + self._counter.to_bytes(16, byteorder="big")
        self._counter += 1
        return int.from_bytes(hashlib.sha256(material).digest(), "big") % stop


def _paired_bootstrap_interval(
    candidate_values, reference_values, resamples, seed, label
):
    """Recompute the paired-round ratio-of-medians bootstrap interval."""

    if (
        not candidate_values
        or len(candidate_values) != len(reference_values)
        or type(resamples) is not int
        or resamples <= 0
        or type(seed) is not int
    ):
        raise TackerProfileError("paired bootstrap metadata is invalid")
    generator = _DeterministicGenerator(seed, label)
    count = len(candidate_values)
    values = []
    for _ in range(resamples):
        indices = [generator.index(count) for _ in range(count)]
        values.append(
            statistics.median(candidate_values[index] for index in indices)
            / statistics.median(reference_values[index] for index in indices)
        )
    values.sort()
    return _percentile(values, 0.025), _percentile(values, 0.975)


def _performance_by_round(candidate):
    """Return a sealed round -> FPS mapping for one eligible candidate."""

    variant_id = candidate["variant_id"]
    performance = candidate["performance"]
    trials = [float(value) for value in performance["throughput_fps_trials"]]
    trial_count = performance.get("trial_count")
    if type(trial_count) is not int or trial_count != len(trials):
        raise TackerProfileError(
            "candidate {} trial_count disagrees with FPS trials".format(variant_id)
        )
    if trial_count < MIN_SELECTION_TRIALS:
        raise TackerProfileError(
            "candidate {} requires at least {} whole-run trials".format(
                variant_id, MIN_SELECTION_TRIALS
            )
        )
    rounds = performance.get("round_indices")
    if (
        not isinstance(rounds, list)
        or len(rounds) != trial_count
        or any(type(value) is not int or value < 0 for value in rounds)
        or len(set(rounds)) != len(rounds)
    ):
        raise TackerProfileError(
            "candidate {} requires one unique round index per FPS trial".format(
                variant_id
            )
        )
    return list(rounds), dict(zip(rounds, trials))


def _validate_paired_promotion_evidence(comparison, challenger, incumbent):
    """Validate and independently recompute one challenger comparison."""

    if not isinstance(comparison, dict):
        raise TackerProfileError("promotion paired_comparison must be an object")
    challenger_name = challenger.get("benchmark_candidate_name")
    incumbent_name = incumbent.get("benchmark_candidate_name")
    if not isinstance(challenger_name, str) or not challenger_name:
        raise TackerProfileError(
            "candidate {} benchmark_candidate_name is required".format(
                challenger["variant_id"]
            )
        )
    if not isinstance(incumbent_name, str) or not incumbent_name:
        raise TackerProfileError(
            "candidate {} benchmark_candidate_name is required".format(
                incumbent["variant_id"]
            )
        )
    if (
        comparison.get("candidate") != challenger_name
        or comparison.get("reference") != incumbent_name
    ):
        raise TackerProfileError(
            "promotion paired comparison names the wrong candidate/reference"
        )

    challenger_rounds, challenger_by_round = _performance_by_round(challenger)
    incumbent_rounds, incumbent_by_round = _performance_by_round(incumbent)
    rounds = comparison.get("round_indices")
    if rounds != challenger_rounds or rounds != incumbent_rounds:
        raise TackerProfileError(
            "promotion paired comparison must cover every shared FPS round"
        )
    challenger_values = [challenger_by_round[index] for index in rounds]
    incumbent_values = [incumbent_by_round[index] for index in rounds]
    expected_ratios = [
        challenger_value / incumbent_value
        for challenger_value, incumbent_value in zip(
            challenger_values, incumbent_values
        )
    ]
    ratios = comparison.get("paired_fps_ratios")
    if (
        not isinstance(ratios, list)
        or len(ratios) != len(expected_ratios)
        or any(not _is_finite_number(value) or float(value) <= 0.0 for value in ratios)
        or any(
            not math.isclose(
                float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
            )
            for actual, expected in zip(ratios, expected_ratios)
        )
    ):
        raise TackerProfileError(
            "promotion paired FPS ratios disagree with whole-run trials"
        )
    expected_ratio = (
        float(challenger["performance"]["median_throughput_fps"])
        / float(incumbent["performance"]["median_throughput_fps"])
    )
    ratio = _same_finite_number(
        comparison.get("median_fps_ratio"),
        expected_ratio,
        "promotion median_fps_ratio",
    )
    _same_finite_number(
        comparison.get("median_paired_fps_ratio"),
        statistics.median(expected_ratios),
        "promotion median_paired_fps_ratio",
    )

    interval = comparison.get("paired_bootstrap_95_ci")
    if not isinstance(interval, dict):
        raise TackerProfileError(
            "promotion paired_bootstrap_95_ci must be an object"
        )
    lower = interval.get("lower")
    upper = interval.get("upper")
    if (
        not _is_finite_number(lower)
        or float(lower) <= 0.0
        or not _is_finite_number(upper)
        or float(upper) <= 0.0
        or float(lower) > float(upper)
    ):
        raise TackerProfileError("promotion paired bootstrap interval is invalid")
    if not _is_finite_number(interval.get("confidence")) or not math.isclose(
        float(interval["confidence"]), 0.95, rel_tol=0.0, abs_tol=1e-15
    ):
        raise TackerProfileError("promotion paired bootstrap confidence must be 0.95")
    for key, expected in (
        ("statistic", "median(candidate_fps)/median(reference_fps)"),
        ("resampling_unit", "paired_round"),
        ("percentile_method", "linear_type_7"),
    ):
        if interval.get(key) != expected:
            raise TackerProfileError(
                "promotion paired bootstrap {} changed".format(key)
            )
    resamples = interval.get("resamples")
    seed = interval.get("seed")
    if (
        resamples != FORMAL_BOOTSTRAP_RESAMPLES
        or seed != FORMAL_BOOTSTRAP_SEED
    ):
        raise TackerProfileError(
            "promotion paired bootstrap must use 10000 resamples and seed 0"
        )
    if comparison.get("derived_by_inverting_reported_comparison") is True:
        reverse_lower, reverse_upper = _paired_bootstrap_interval(
            incumbent_values,
            challenger_values,
            resamples,
            seed,
            "{}-vs-{}".format(incumbent_name, challenger_name),
        )
        expected_lower = 1.0 / reverse_upper
        expected_upper = 1.0 / reverse_lower
    else:
        expected_lower, expected_upper = _paired_bootstrap_interval(
            challenger_values,
            incumbent_values,
            resamples,
            seed,
            "{}-vs-{}".format(challenger_name, incumbent_name),
        )
    lower = _same_finite_number(
        lower, expected_lower, "promotion paired bootstrap lower"
    )
    _same_finite_number(
        upper, expected_upper, "promotion paired bootstrap upper"
    )
    return ratio, lower


def _validate_promotion_criterion(criterion, observed, required_key, required):
    if not isinstance(criterion, dict):
        raise TackerProfileError("promotion criterion must be an object")
    _same_finite_number(
        criterion.get("observed"), observed, "promotion criterion observed value"
    )
    _same_finite_number(
        criterion.get(required_key), required, "promotion criterion requirement"
    )


def _validate_promotion_evaluation(
    evaluation, challenger, incumbent, two_stream
):
    if not isinstance(evaluation, dict):
        raise TackerProfileError("promotion candidate evaluation must be an object")
    if evaluation.get("candidate_variant_id") != challenger["variant_id"]:
        raise TackerProfileError("promotion candidate evaluation changed order")
    if type(evaluation.get("passed")) is not bool:
        raise TackerProfileError("promotion candidate evaluation passed must be boolean")
    ratio, ci_lower = _validate_paired_promotion_evidence(
        evaluation.get("paired_comparison"), challenger, incumbent
    )
    challenger_fps = float(
        challenger["performance"]["median_throughput_fps"]
    )
    incumbent_fps = float(incumbent["performance"]["median_throughput_fps"])
    two_stream_fps = float(two_stream["performance"]["median_throughput_fps"])
    ratio_passed = ratio >= PROMOTION_MIN_FPS_RATIO
    ci_passed = ci_lower > PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE
    floor_ratios = {
        "two_stream": challenger_fps / two_stream_fps,
        incumbent["variant_id"]: challenger_fps / incumbent_fps,
    }
    floor_passed = all(value >= 1.0 for value in floor_ratios.values())
    expected_passed = ratio_passed and ci_passed and floor_passed
    if evaluation["passed"] != expected_passed:
        raise TackerProfileError(
            "promotion candidate pass result disagrees with recomputed gates"
        )

    criteria = evaluation.get("criteria")
    if not isinstance(criteria, dict):
        raise TackerProfileError("promotion evaluation criteria must be an object")
    ratio_criterion = criteria.get("median_fps_ratio")
    _validate_promotion_criterion(
        ratio_criterion, ratio, "required_min", PROMOTION_MIN_FPS_RATIO
    )
    if ratio_criterion.get("passed") is not ratio_passed:
        raise TackerProfileError("promotion ratio criterion result changed")
    ci_criterion = criteria.get("paired_bootstrap_95_ci_lower")
    _validate_promotion_criterion(
        ci_criterion,
        ci_lower,
        "required_strictly_greater_than",
        PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE,
    )
    if ci_criterion.get("passed") is not ci_passed:
        raise TackerProfileError("promotion bootstrap criterion result changed")
    floor_criterion = criteria.get("baseline_fps_ratios")
    if not isinstance(floor_criterion, dict):
        raise TackerProfileError(
            "promotion baseline_fps_ratios criterion must be an object"
        )
    observed_floor = floor_criterion.get("observed")
    if not isinstance(observed_floor, dict) or set(observed_floor) != set(floor_ratios):
        raise TackerProfileError("promotion baseline FPS ratio set changed")
    for baseline_id, expected in floor_ratios.items():
        _same_finite_number(
            observed_floor.get(baseline_id),
            expected,
            "promotion baseline {} FPS ratio".format(baseline_id),
        )
    _same_finite_number(
        floor_criterion.get("required_min"),
        1.0,
        "promotion baseline FPS requirement",
    )
    if floor_criterion.get("passed") is not floor_passed:
        raise TackerProfileError("promotion baseline FPS criterion result changed")
    return expected_passed, ratio, ci_lower


def _validate_v2_selection(selection, candidates, selected_variant_id):
    """Recompute ranking, equivalence and every incumbent-promotion gate."""

    by_id = {candidate["variant_id"]: candidate for candidate in candidates}
    eligible = [
        candidate
        for candidate in candidates
        if candidate["correctness"]["valid"]
    ]
    if not eligible:
        raise TackerProfileError("selection has no correctness-valid candidate")
    if any(candidate.get("performance") is None for candidate in eligible):
        raise TackerProfileError(
            "every correctness-valid candidate requires whole-run performance"
        )

    benchmark_names = []
    shared_rounds = None
    for candidate in eligible:
        benchmark_name = candidate.get("benchmark_candidate_name")
        if not isinstance(benchmark_name, str) or not benchmark_name:
            raise TackerProfileError(
                "candidate {} benchmark_candidate_name is required".format(
                    candidate["variant_id"]
                )
            )
        benchmark_names.append(benchmark_name)
        rounds, _ = _performance_by_round(candidate)
        if shared_rounds is None:
            shared_rounds = rounds
        elif rounds != shared_rounds:
            raise TackerProfileError(
                "correctness-valid candidates must share the same paired rounds"
            )
    if len(benchmark_names) != len(set(benchmark_names)):
        raise TackerProfileError(
            "correctness-valid candidates require unique benchmark_candidate_name values"
        )

    ranking = sorted(
        eligible,
        key=lambda candidate: (
            -float(candidate["performance"]["median_throughput_fps"]),
            candidate["benchmark_candidate_name"],
        ),
    )
    ranking_ids = [candidate["variant_id"] for candidate in ranking]
    if selection.get("global_median_fps_ranking") != ranking_ids:
        raise TackerProfileError(
            "selection ranking disagrees with correctness-valid median FPS"
        )
    if selection.get("eligible_variant_ids") != ranking_ids:
        raise TackerProfileError(
            "selection eligible_variant_ids disagrees with ranking"
        )
    ineligible = selection.get("ineligible_variant_ids")
    expected_ineligible = {
        candidate["variant_id"]
        for candidate in candidates
        if not candidate["correctness"]["valid"]
    }
    if (
        not isinstance(ineligible, list)
        or len(ineligible) != len(set(ineligible))
        or set(ineligible) != expected_ineligible
    ):
        raise TackerProfileError(
            "selection ineligible_variant_ids disagrees with correctness"
        )
    experimental_id = selection.get("experimental_winner_variant_id")
    if experimental_id != ranking_ids[0]:
        raise TackerProfileError(
            "experimental winner is not the global median FPS argmax"
        )
    top_fps = float(ranking[0]["performance"]["median_throughput_fps"])
    equivalent_ids = [
        candidate["variant_id"]
        for candidate in ranking
        if (
            top_fps
            - float(candidate["performance"]["median_throughput_fps"])
        )
        / top_fps
        <= EQUIVALENCE_FRACTION
    ]
    equivalence = selection.get("equivalence")
    if not isinstance(equivalence, dict):
        raise TackerProfileError("selection.equivalence must be an object")
    fraction = equivalence.get("fraction")
    if not _is_finite_number(fraction) or not math.isclose(
        float(fraction), EQUIVALENCE_FRACTION, rel_tol=0.0, abs_tol=1e-15
    ):
        raise TackerProfileError("selection equivalence fraction must be 0.005")

    def preference_key(candidate):
        metadata = candidate.get("selection_metadata", {})
        if not isinstance(metadata, dict):
            raise TackerProfileError(
                "candidate selection_metadata must be an object"
            )
        key = []
        for name in (
            "abi_complexity",
            "peak_memory_bytes",
            "registers_per_thread",
            "shared_memory_bytes",
        ):
            value = metadata.get(name)
            if value is not None and (
                not _is_finite_number(value) or float(value) < 0.0
            ):
                raise TackerProfileError(
                    "candidate selection metadata must be finite and non-negative"
                )
            key.append((1, 0.0) if value is None else (0, float(value)))
        key.append(candidate["benchmark_candidate_name"])
        return tuple(key)

    preferred_equivalent = sorted(
        (by_id[variant_id] for variant_id in equivalent_ids),
        key=preference_key,
    )
    preferred_ids = [candidate["variant_id"] for candidate in preferred_equivalent]
    recorded_equivalent = equivalence.get(
        "candidate_variant_ids_in_preference_order",
        equivalence.get("candidates", equivalence.get("equivalent_variant_ids")),
    )
    if recorded_equivalent != preferred_ids:
        raise TackerProfileError(
            "selection equivalence preference disagrees with candidate resources"
        )
    preferred_id = equivalence.get(
        "preferred_variant_id", equivalence.get("preferred_candidate")
    )
    if preferred_id != preferred_ids[0]:
        raise TackerProfileError("selection preferred equivalent candidate changed")
    if selection.get("deployment_winner_variant_id") != selected_variant_id:
        raise TackerProfileError(
            "selected_variant_id disagrees with deployment winner"
        )

    incumbent_id = selection.get("incumbent_variant_id")
    if (
        not isinstance(incumbent_id, str)
        or incumbent_id not in by_id
        or by_id[incumbent_id]["execution_mode"] != "tacker"
    ):
        raise TackerProfileError(
            "selection incumbent_variant_id must identify a Tacker candidate"
        )
    promotion = selection.get("promotion")
    if not isinstance(promotion, dict) or type(promotion.get("passed")) is not bool:
        raise TackerProfileError("selection.promotion.passed must be boolean")
    _same_finite_number(
        promotion.get("minimum_median_fps_ratio"),
        PROMOTION_MIN_FPS_RATIO,
        "selection promotion minimum_median_fps_ratio",
    )
    _same_finite_number(
        promotion.get("minimum_bootstrap_lower_exclusive"),
        PROMOTION_MIN_BOOTSTRAP_LOWER_EXCLUSIVE,
        "selection promotion minimum_bootstrap_lower_exclusive",
    )
    if promotion.get("incumbent_variant_id") != incumbent_id:
        raise TackerProfileError(
            "selection promotion incumbent disagrees with selection"
        )

    incumbent = by_id[incumbent_id]
    selected = by_id[selected_variant_id]
    two_stream = by_id["two_stream"]
    selected_fps = float(selected["performance"]["median_throughput_fps"])
    two_stream_fps = float(
        two_stream["performance"]["median_throughput_fps"]
    )
    # This floor applies before every early-return case, including a retained
    # incumbent and an invalid-incumbent replacement.
    if selected_fps < two_stream_fps:
        raise TackerProfileError("deployment winner is slower than two_stream")

    if not incumbent["correctness"]["valid"]:
        replacements = [
            variant_id
            for variant_id in preferred_ids
            if float(by_id[variant_id]["performance"]["median_throughput_fps"])
            >= two_stream_fps
        ]
        if not replacements or selected_variant_id != replacements[0]:
            raise TackerProfileError(
                "invalid incumbent replacement disagrees with the two_stream floor"
            )
        if promotion.get("candidate_evaluations") != []:
            raise TackerProfileError(
                "invalid incumbent replacement must not claim bootstrap evaluations"
            )
        if (
            promotion.get("challenger_variant_id") != selected_variant_id
            or promotion.get("decision")
            != "selected_best_valid_candidate_incumbent_invalid"
            or promotion["passed"] is not True
            or promotion.get("required") is not True
        ):
            raise TackerProfileError(
                "invalid incumbent replacement decision is not sealed"
            )
        return

    incumbent_rounds, _ = _performance_by_round(incumbent)
    if incumbent_rounds != shared_rounds:
        raise TackerProfileError("incumbent does not share the paired FPS rounds")
    evaluation_ids = [
        variant_id for variant_id in preferred_ids if variant_id != incumbent_id
    ]
    evaluations = promotion.get("candidate_evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != len(evaluation_ids):
        raise TackerProfileError(
            "promotion candidate evaluations must cover every equivalent challenger"
        )
    recomputed = {}
    passing_ids = []
    for variant_id, evaluation in zip(evaluation_ids, evaluations):
        challenger = by_id[variant_id]
        passed, ratio, ci_lower = _validate_promotion_evaluation(
            evaluation, challenger, incumbent, two_stream
        )
        recomputed[variant_id] = {
            "evaluation": evaluation,
            "ratio": ratio,
            "ci_lower": ci_lower,
        }
        if passed:
            passing_ids.append(variant_id)

    summary_id = None
    if passing_ids:
        expected_selected_id = passing_ids[0]
        expected_decision = "promoted_equivalent_challenger"
        expected_passed = True
        expected_required = True
        summary_id = expected_selected_id
    elif experimental_id == incumbent_id:
        expected_selected_id = incumbent_id
        expected_decision = "incumbent_is_global_winner"
        expected_passed = True
        expected_required = False
    else:
        expected_selected_id = incumbent_id
        expected_decision = "retained_incumbent"
        expected_passed = False
        expected_required = preferred_id != incumbent_id
        if evaluation_ids:
            summary_id = (
                preferred_id if preferred_id in recomputed else evaluation_ids[0]
            )

    if selected_variant_id != expected_selected_id:
        raise TackerProfileError(
            "deployment winner disagrees with recomputed incumbent promotion"
        )
    if (
        promotion["passed"] is not expected_passed
        or promotion.get("decision") != expected_decision
        or promotion.get("required") is not expected_required
    ):
        raise TackerProfileError(
            "selection promotion decision disagrees with recomputed gates"
        )
    expected_challenger_id = (
        summary_id
        if summary_id is not None
        else incumbent_id
    )
    if promotion.get("challenger_variant_id") != expected_challenger_id:
        raise TackerProfileError(
            "selection promotion challenger disagrees with recomputed gates"
        )

    if summary_id is None:
        for key in (
            "paired_comparison",
            "median_fps_ratio",
            "paired_bootstrap_95_ci_lower",
            "criteria",
        ):
            if key in promotion:
                raise TackerProfileError(
                    "incumbent-winner promotion contains unexpected {}".format(key)
                )
        return

    summary = recomputed[summary_id]
    evaluation = summary["evaluation"]
    if promotion.get("paired_comparison") != evaluation["paired_comparison"]:
        raise TackerProfileError(
            "selection promotion comparison disagrees with candidate evaluation"
        )
    if promotion.get("criteria") != evaluation["criteria"]:
        raise TackerProfileError(
            "selection promotion criteria disagree with candidate evaluation"
        )
    _same_finite_number(
        promotion.get("median_fps_ratio"),
        summary["ratio"],
        "selection promotion median_fps_ratio",
    )
    _same_finite_number(
        promotion.get("paired_bootstrap_95_ci_lower"),
        summary["ci_lower"],
        "selection promotion paired_bootstrap_95_ci_lower",
    )


def _validate_v2_tacker_profile(profile):
    _require_exact(profile, "schema_version", PROFILE_SCHEMA_VERSION, "profile")
    _require_exact(profile, "workload_key", WORKLOAD_KEY, "profile")
    _require_exact(
        profile, "selection_objective", SELECTION_OBJECTIVE, "profile"
    )
    manifest = profile.get("manifest")
    _validate_v2_manifest(manifest, profile["workload_key"])
    digest = profile.get("manifest_sha256")
    _require_sha256(digest, "profile.manifest_sha256")
    if digest != manifest_sha256(manifest):
        raise TackerProfileError("profile manifest SHA-256 mismatch")

    correctness_limits = profile.get("correctness_thresholds")
    if not isinstance(correctness_limits, dict):
        raise TackerProfileError("profile.correctness_thresholds must be an object")
    for key, locked_limit in _CORRECTNESS_LIMITS.items():
        value = correctness_limits.get(key)
        if not _is_finite_number(value):
            raise TackerProfileError(
                "correctness_thresholds.{} must be finite".format(key)
            )
        if float(value) > locked_limit:
            raise TackerProfileError(
                "correctness_thresholds.{} weakens the locked limit {}".format(
                    key, locked_limit
                )
            )

    candidates = profile.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise TackerProfileError("profile.candidates must be a non-empty array")
    by_id = {}
    for candidate in candidates:
        _validate_v2_candidate(candidate, correctness_limits)
        variant_id = candidate["variant_id"]
        if variant_id in by_id:
            raise TackerProfileError(
                "profile.candidates has duplicate variant_id {!r}".format(variant_id)
            )
        by_id[variant_id] = candidate
    for baseline_id in ("serial", "two_stream"):
        if baseline_id not in by_id:
            raise TackerProfileError(
                "profile.candidates must include the {} baseline".format(baseline_id)
            )
        if by_id[baseline_id]["execution_mode"] != baseline_id:
            raise TackerProfileError(
                "candidate {} must execute the {} baseline".format(
                    baseline_id, baseline_id
                )
            )

    selected_variant_id = profile.get("selected_variant_id")
    if not isinstance(selected_variant_id, str) or selected_variant_id not in by_id:
        raise TackerProfileError(
            "profile.selected_variant_id must identify exactly one candidate"
        )
    selected = by_id[selected_variant_id]
    if selected["execution_mode"] == "tacker":
        if selected["persistent_blocks"] != manifest["persistent_blocks"]:
            raise TackerProfileError(
                "selected candidate persistent_blocks disagrees with manifest"
            )

    deployment = profile.get("deployment")
    if not isinstance(deployment, dict):
        raise TackerProfileError("profile.deployment must be an object")
    for key in ("enabled", "valid"):
        if type(deployment.get(key)) is not bool:
            raise TackerProfileError("deployment.{} must be boolean".format(key))
    if deployment["enabled"] != deployment["valid"]:
        raise TackerProfileError("deployment.enabled and valid must agree")
    selection = profile.get("selection")
    if selection is not None and not isinstance(selection, dict):
        raise TackerProfileError("profile.selection must be null or an object")
    if deployment["enabled"]:
        if selected["execution_mode"] != "tacker":
            raise TackerProfileError(
                "deployable Tacker profiles cannot select a baseline mode"
            )
        for baseline_id in ("serial", "two_stream"):
            baseline = by_id[baseline_id]
            if (
                not baseline["correctness"]["valid"]
                or baseline.get("performance") is None
            ):
                raise TackerProfileError(
                    "deployed profile requires valid measured {} baseline".format(
                        baseline_id
                    )
                )
        if not selected["correctness"]["valid"]:
            raise TackerProfileError(
                "deployed selected candidate is not correctness-valid"
            )
        if selected.get("performance") is None:
            raise TackerProfileError(
                "deployed selected candidate has no whole-run performance"
            )
        if not isinstance(selection, dict):
            raise TackerProfileError("deployed profile requires selection evidence")
        _validate_v2_selection(selection, candidates, selected_variant_id)

    provenance = profile.get("provenance")
    if not isinstance(provenance, dict):
        raise TackerProfileError("profile.provenance must be an object")
    sealed_digest = profile.get("profile_sha256")
    _require_sha256(sealed_digest, "profile.profile_sha256")
    if sealed_digest != profile_sha256(profile):
        raise TackerProfileError("profile selection SHA-256 mismatch")
    return profile


def validate_tacker_profile(profile):
    """Validate a schema-v2 profile or a read-only schema-v1 legacy profile."""

    if not isinstance(profile, dict):
        raise TackerProfileError("profile must be a JSON object")
    version = profile.get("schema_version")
    try:
        # Python's decoder accepts non-standard NaN/Infinity tokens.  Validate
        # the complete document, including unsealed human-readable fields, so
        # every accepted profile is finite JSON rather than only hashable in
        # the dispatch-relevant subset.
        _canonical_json(profile)
        if version == LEGACY_PROFILE_SCHEMA_VERSION:
            return _validate_legacy_tacker_profile(profile)
        if version == PROFILE_SCHEMA_VERSION:
            return _validate_v2_tacker_profile(profile)
    except (TypeError, ValueError) as error:
        if isinstance(error, TackerProfileError):
            raise
        raise TackerProfileError("profile contains non-canonical data: {}".format(error))
    raise TackerProfileError(
        "profile.schema_version must be {} or read-only legacy {}".format(
            PROFILE_SCHEMA_VERSION, LEGACY_PROFILE_SCHEMA_VERSION
        )
    )


def selected_tacker_candidate(profile):
    """Return the selected physical candidate in either supported schema."""

    if profile.get("schema_version") == LEGACY_PROFILE_SCHEMA_VERSION:
        candidate = dict(profile["manifest"])
        candidate.update(
            {
                "variant_id": LEGACY_VARIANT_ID,
                "execution_mode": "tacker",
            }
        )
        return candidate
    selected_variant_id = profile["selected_variant_id"]
    for candidate in profile["candidates"]:
        if candidate["variant_id"] == selected_variant_id:
            return candidate
    raise TackerProfileError("selected candidate disappeared after validation")


def load_tacker_profile(profile_path=None, profile_override=None):
    """Load and structurally validate one profile.

    ``profile_override`` is an explicit injection point for tests and for a
    caller that has just produced a measured profile.  It does not bypass the
    manifest/profile hashes or correctness qualification.
    """

    if profile_path is not None and profile_override is not None:
        raise TackerProfileError(
            "profile_path and profile_override are mutually exclusive"
        )
    if profile_override is not None:
        profile = deepcopy(profile_override)
    else:
        path = DEFAULT_PROFILE_PATH if profile_path is None else Path(profile_path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                profile = json.load(handle)
        except (OSError, ValueError) as error:
            raise TackerProfileError(
                "cannot load Tacker profile {}: {}".format(path, error)
            )
    return validate_tacker_profile(profile)


def tacker_profile_admission_reason(profile):
    """Return ``None`` only for a qualified, selected Tacker deployment.

    Performance selection is already sealed into schema v2.  Runtime does not
    reinterpret Raster slowdown, mixed-leaf savings, or per-frame latency.
    Schema-v1 profiles keep only their correctness and enabled-state checks.
    """

    try:
        validate_tacker_profile(profile)
    except TackerProfileError as error:
        return "invalid Tacker profile: {}".format(error)

    if profile["schema_version"] == LEGACY_PROFILE_SCHEMA_VERSION:
        admission = profile["admission"]
        if not admission["enabled"]:
            return "Tacker profile is disabled"
        if not admission["valid"]:
            return "Tacker profile is not marked valid"
        thresholds = profile["thresholds"]
        measured = profile["measurements"]
        if measured["psnr_drop_db"] > thresholds["psnr_drop_db_max"]:
            return "PSNR drop exceeds the profile threshold"
        if measured["ssim_drop"] > thresholds["ssim_drop_max"]:
            return "SSIM drop exceeds the profile threshold"
        if measured["lpips_increase"] > thresholds["lpips_increase_max"]:
            return "LPIPS increase exceeds the profile threshold"
        return None

    deployment = profile["deployment"]
    if not deployment["enabled"]:
        return "Tacker profile deployment is disabled"
    if not deployment["valid"]:
        return "Tacker profile deployment is not marked valid"
    selected = selected_tacker_candidate(profile)
    if selected["execution_mode"] != "tacker":
        return "selected deployment winner is the {} baseline".format(
            selected["execution_mode"]
        )
    if not selected["correctness"]["valid"]:
        return "selected Tacker candidate is not correctness-valid"
    return None


def _capability_value(capabilities, name, default=None):
    if isinstance(capabilities, dict):
        return capabilities.get(name, default)
    return getattr(capabilities, name, default)


def _query_rasterizer_capabilities():
    provider = getattr(_rasterizer_module, "tacker_capabilities", None)
    if provider is None:
        backend = getattr(_rasterizer_module, "_C", None)
        provider = getattr(backend, "tacker_capabilities", None)
    if provider is None:
        raise RuntimeError("rasterizer has no Tacker capability query")
    return provider() if callable(provider) else provider


def _module_name(module):
    return type(module).__name__


def _linear_shape_reason(module, in_features, out_features, label):
    if _module_name(module) != "Linear":
        return "{} must be Linear".format(label)
    if getattr(module, "in_features", None) != in_features:
        return "{} has the wrong input width".format(label)
    if getattr(module, "out_features", None) != out_features:
        return "{} has the wrong output width".format(label)
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    if weight is None or tuple(getattr(weight, "shape", ())) != (
        out_features,
        in_features,
    ):
        return "{} weight shape is not [{}, {}]".format(
            label, out_features, in_features
        )
    if bias is None or tuple(getattr(bias, "shape", ())) != (out_features,):
        return "{} bias shape is not [{}]".format(label, out_features)
    return None


def _head_structure_reason(network):
    heads = (
        ("pos_deform", 3),
        ("scales_deform", 3),
        ("rotations_deform", 4),
        ("opacity_deform", 1),
        ("shs_deform", 48),
    )
    for name, output_width in heads:
        head = getattr(network, name, None)
        if head is None or _module_name(head) != "Sequential":
            return "{} must be an exact Sequential head".format(name)
        try:
            modules = list(head)
        except TypeError:
            return "{} is not iterable".format(name)
        if len(modules) != 4:
            return "{} must contain exactly four modules".format(name)
        if _module_name(modules[0]) != "ReLU" or _module_name(modules[2]) != "ReLU":
            return "{} must be ReLU/Linear/ReLU/Linear".format(name)
        reason = _linear_shape_reason(modules[1], 128, 128, "{}[1]".format(name))
        if reason is not None:
            return reason
        reason = _linear_shape_reason(
            modules[3], 128, output_width, "{}[3]".format(name)
        )
        if reason is not None:
            return reason
    return None


def tacker_support_reason(
    pc,
    pipe,
    profile,
    stage="fine",
    cam_type=None,
    override_color=None,
    workload_name=None,
    iteration=None,
    qualification_mode=False,
):
    """Return the first reason the physical two-task path cannot be admitted."""

    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if torch.is_grad_enabled():
        return "autograd is enabled"
    if stage != "fine":
        return "only the exact fine inference stage is supported"
    if cam_type == "PanopticSports":
        return "PanopticSports is not supported"
    if cam_type != "dynerf":
        return "the first Tacker profile is restricted to dynerf"
    if bool(getattr(pipe, "debug", False)):
        return "rasterizer debug mode is unsupported"
    if bool(getattr(pipe, "compute_cov3D_python", False)):
        return "compute_cov3D_python requires fallback"
    if bool(getattr(pipe, "convert_SHs_python", False)):
        return "convert_SHs_python requires fallback"
    if override_color is not None:
        return "override_color requires fallback"

    if qualification_mode:
        try:
            validate_tacker_profile(profile)
        except TackerProfileError as error:
            return "invalid Tacker qualification profile: {}".format(error)
    else:
        profile_reason = tacker_profile_admission_reason(profile)
        if profile_reason is not None:
            return profile_reason
    selected_candidate = selected_tacker_candidate(profile)
    if selected_candidate.get("execution_mode") != "tacker":
        return "selected profile candidate is not a physical Tacker variant"
    manifest = profile["manifest"]
    if workload_name != manifest["workload"]:
        return "workload name does not match the admitted profile"
    if iteration != manifest["iteration"]:
        return "checkpoint iteration does not match the admitted profile"

    deformation = getattr(pc, "_deformation", None)
    if deformation is None:
        return "model has no deformation network"
    if bool(getattr(deformation, "training", True)):
        return "deformation network is not in eval mode"
    network = getattr(deformation, "deformation_net", None)
    if network is None:
        return "model has no inner deformation network"
    if getattr(network, "W", None) != 128:
        return "deformation width W must be 128"
    if getattr(network, "D", None) != 0:
        return "defor_depth must be 0"
    args = getattr(network, "args", None)
    if args is None:
        return "deformation arguments are unavailable"
    if bool(getattr(args, "no_dx", False)):
        return "no_dx must be false for the selected positional head"
    structure_reason = _head_structure_reason(network)
    if structure_reason is not None:
        return structure_reason

    xyz_shape = tuple(getattr(getattr(pc, "get_xyz", None), "shape", ()))
    expected_count = manifest["gaussian_count"]
    if not xyz_shape or xyz_shape[0] != expected_count:
        return "Gaussian count does not match the admitted profile"

    try:
        capabilities = _query_rasterizer_capabilities()
    except Exception as error:
        return "rasterizer capability query failed: {}".format(error)
    if not bool(_capability_value(capabilities, "stream_aware", False)):
        return "rasterizer extension is not stream-aware"
    mixed_abi = _capability_value(
        capabilities,
        "mixed_render_head_abi",
        _capability_value(capabilities, "mixed_abi", 0),
    )
    if not isinstance(mixed_abi, int) or mixed_abi < 1:
        return "rasterizer mixed ABI version is below 1"
    if _capability_value(capabilities, "sm_target", None) != "sm_86":
        return "rasterizer mixed ABI was not built for sm_86"
    expected_capabilities = {
        "mixed_threads": 384,
        "raster_threads": 256,
        "head_threads": 128,
        "head_thread_base": 256,
        "raster_named_barrier_id": 1,
    }
    for key, expected in expected_capabilities.items():
        if _capability_value(capabilities, key, None) != expected:
            return "rasterizer capability {} does not match {}".format(
                key, expected
            )
    if not callable(getattr(GaussianRasterizer, "forward_with_head", None)):
        return "GaussianRasterizer.forward_with_head is unavailable"

    device = getattr(getattr(pc, "get_xyz", None), "device", None)
    try:
        compute_capability = tuple(torch.cuda.get_device_capability(device))
    except Exception as error:
        return "cannot query GPU compute capability: {}".format(error)
    if compute_capability != (8, 6):
        return "the first Tacker path requires GPU capability 8.6"
    try:
        gpu_name = " ".join(str(torch.cuda.get_device_name(device)).split())
    except Exception as error:
        return "cannot query GPU name: {}".format(error)
    if gpu_name != manifest["gpu_name"]:
        return "GPU name does not match the admitted NVIDIA RTX A6000 profile"

    return None


def _poc_fre(input_data, poc_buf):
    """Exact local copy of ``scene.deformation.poc_fre``."""

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    return torch.cat([input_data, input_data_sin, input_data_cos], -1)


def _record_tensor_stream(value, stream):
    if bool(getattr(value, "is_cuda", False)) and callable(
        getattr(value, "record_stream", None)
    ):
        value.record_stream(stream)


def _view_profile_reason(view, profile):
    """Validate the first camera against the profile-bound raster workload."""

    manifest = profile["manifest"]
    expected_width, expected_height = manifest["resolution"]
    width = getattr(view, "image_width", None)
    height = getattr(view, "image_height", None)
    if width != expected_width or height != expected_height:
        return "camera resolution does not match the admitted profile"
    return None


@dataclass
class PosHeadTask:
    """Real D(t+1) prefix/other-head values surrounding the mixed Linear."""

    context: object
    point_emb: object
    scales_emb: object
    rotations_emb: object
    opacity_emb: object
    shs_emb: object
    hidden: object
    mask: object
    head_input: object
    head_weight: object
    head_bias: object
    scale_delta: object
    rotation_delta: object
    opacity_delta: object
    shs_delta: object
    prefix_ready: object
    head_output: object = None

    def record_stream(self, stream):
        for field in fields(self):
            _record_tensor_stream(getattr(self, field.name), stream)


def _cache_pos_head_parameters(pc):
    selected = pc._deformation.deformation_net.pos_deform[1]
    weight = selected.weight.detach().to(dtype=torch.float16).contiguous()
    bias = selected.bias.detach().to(dtype=torch.float32).contiguous()
    return weight, bias


def prepare_pos_head_task(
    context,
    pc,
    deform_stream,
    head_weight=None,
    head_bias=None,
    prefix_ready=None,
):
    """Enqueue the exact D(t+1) prefix and four non-selected full heads.

    ``prefix_ready`` is recorded immediately after the real selected-head input
    is materialized, before the other four heads are submitted.  This is the
    point at which the Raster stream may start the physical mixed leaf.
    """

    if prefix_ready is None:
        prefix_ready = torch.cuda.Event(blocking=False)
    if head_weight is None or head_bias is None:
        head_weight, head_bias = _cache_pos_head_parameters(pc)

    with torch.cuda.stream(deform_stream):
        deformation = pc._deformation
        network = deformation.deformation_net

        point_emb = _poc_fre(context.means3D, deformation.pos_poc)
        scales_emb = _poc_fre(context.scales, deformation.rotation_scaling_poc)
        rotations_emb = _poc_fre(
            context.rotations, deformation.rotation_scaling_poc
        )
        hidden = network.query_time(
            point_emb,
            scales_emb,
            rotations_emb,
            None,
            context.timestamp,
        )

        args = network.args
        if bool(getattr(args, "static_mlp", False)):
            mask = network.static_mlp(hidden)
        elif bool(getattr(args, "empty_voxel", False)):
            mask = network.empty_voxel(point_emb[:, :3])
        else:
            mask = torch.ones_like(context.opacity[:, 0]).unsqueeze(-1)

        # pos_deform[1] is intentionally not called here or in the suffix.
        head_input = (
            network.pos_deform[0](hidden)
            .to(dtype=torch.float16)
            .contiguous()
        )
        prefix_ready.record(deform_stream)

        # These are independent of the selected positional head and remain on
        # the deformation stream, overlapping the Raster+head mixed kernel.
        scale_delta = network.scales_deform(hidden)
        rotation_delta = network.rotations_deform(hidden)
        opacity_delta = network.opacity_deform(hidden)
        shs_delta = network.shs_deform(hidden)

    return PosHeadTask(
        context=context,
        point_emb=point_emb,
        scales_emb=scales_emb,
        rotations_emb=rotations_emb,
        opacity_emb=context.opacity,
        shs_emb=context.shs,
        hidden=hidden,
        mask=mask,
        head_input=head_input,
        head_weight=head_weight,
        head_bias=head_bias,
        scale_delta=scale_delta,
        rotation_delta=rotation_delta,
        opacity_delta=opacity_delta,
        shs_delta=shs_delta,
        prefix_ready=prefix_ready,
    )


def _batch_quaternion_multiply(left, right):
    from utils.graphics_utils import batch_quaternion_multiply

    return batch_quaternion_multiply(left, right)


def finish_pos_head_task(task, pc, head_output):
    """Consume the physical FP32 head output and build the next render state."""

    dtype = getattr(head_output, "dtype", None)
    if dtype is not None and str(dtype) not in ("float32", "float", "torch.float32"):
        raise TypeError("mixed positional-head output must be FP32")

    network = pc._deformation.deformation_net
    args = network.args

    # This is pos_deform[2] (ReLU) followed by pos_deform[3] (Linear 128->3).
    # The selected pos_deform[1] Linear is never re-executed.
    dx = network.pos_deform[2:](head_output)
    pts = torch.zeros_like(task.point_emb[:, :3])
    pts = task.point_emb[:, :3] * task.mask + dx

    if bool(getattr(args, "no_ds", False)):
        scales = task.scales_emb[:, :3]
    else:
        scales = torch.zeros_like(task.scales_emb[:, :3])
        scales = task.scales_emb[:, :3] * task.mask + task.scale_delta

    if bool(getattr(args, "no_dr", False)):
        rotations = task.rotations_emb[:, :4]
    else:
        rotations = torch.zeros_like(task.rotations_emb[:, :4])
        if bool(getattr(args, "apply_rotation", False)):
            rotations = _batch_quaternion_multiply(
                task.rotations_emb, task.rotation_delta
            )
        else:
            rotations = task.rotations_emb[:, :4] + task.rotation_delta

    if bool(getattr(args, "no_do", False)):
        opacity = task.opacity_emb[:, :1]
    else:
        opacity = torch.zeros_like(task.opacity_emb[:, :1])
        opacity = task.opacity_emb[:, :1] * task.mask + task.opacity_delta

    if bool(getattr(args, "no_dshs", False)):
        shs = task.shs_emb
    else:
        dshs = task.shs_delta.reshape([task.shs_emb.shape[0], 16, 3])
        shs = torch.zeros_like(task.shs_emb)
        shs = task.shs_emb * task.mask.unsqueeze(-1) + dshs

    return GaussianRenderState(
        means3D=pts,
        scales=pc.scaling_activation(scales),
        rotations=pc.rotation_activation(rotations),
        opacities=pc.opacity_activation(opacity),
        shs=shs,
    )


def _forward_with_head(context, state, task, persistent_blocks):
    """Small adapter around the fixed physical binding API."""

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


def _result_from_mixed(context, mixed_outputs):
    image, radii, depth, head_output = mixed_outputs
    result = RenderResult(
        render=image,
        viewspace_points=context.screenspace_points,
        visibility_filter=radii > 0,
        radii=radii,
        depth=depth,
    )
    return result, head_output


@dataclass
class _TackerSlot:
    ready: object
    prefix_ready: object
    mixed_done: object
    raster_done: object
    context: object = None
    state: object = None
    task: object = None
    has_raster_done: bool = False


class TackerRenderer:
    """Two-slot physical pipeline for ``R(t) + head(t+1)``.

    Unsupported or unmeasured configurations run through a real
    ``TwoStreamRenderer`` instance.  ``last_fallback_reason`` distinguishes
    that case from an admitted physical Tacker execution.
    """

    def __init__(
        self,
        pc,
        pipe,
        bg_color,
        scaling_modifier=1.0,
        stage="fine",
        cam_type=None,
        profile_path=None,
        profile_override=None,
        workload_name=None,
        iteration=None,
        qualification_mode=False,
    ):
        self.pc = pc
        self.pipe = pipe
        self.bg_color = bg_color
        self.scaling_modifier = scaling_modifier
        self.stage = stage
        self.cam_type = cam_type
        self.workload_name = workload_name
        self.iteration = iteration
        self.qualification_mode = qualification_mode
        self.device = getattr(pc.get_xyz, "device", None)

        self._profile_error = None
        self.profile = None
        if type(qualification_mode) is not bool:
            self._profile_error = "qualification_mode must be boolean"
        elif qualification_mode and (
            profile_override is None or profile_path is not None
        ):
            self._profile_error = (
                "qualification_mode requires an explicit profile_override and "
                "forbids profile_path/default-profile admission"
            )
        try:
            if self._profile_error is None:
                self.profile = load_tacker_profile(
                    profile_path=profile_path,
                    profile_override=profile_override,
                )
        except TackerProfileError as error:
            self.profile = None
            self._profile_error = str(error)

        # Immutable inference parameters are converted once at renderer
        # construction, outside the render loop and its timing ranges.
        self.head_weight = None
        self.head_bias = None
        self._cache_ready = None
        self._cache_error = None
        try:
            self.head_weight, self.head_bias = _cache_pos_head_parameters(pc)
            if bool(getattr(self.head_weight, "is_cuda", False)):
                self._cache_ready = torch.cuda.Event(blocking=False)
                self._cache_ready.record(torch.cuda.current_stream(self.device))
        except Exception as error:
            self._cache_error = str(error)

        self.persistent_blocks = 0
        if self.profile is not None:
            self.persistent_blocks = selected_tacker_candidate(self.profile).get(
                "persistent_blocks", 0
            )

        self.raster_stream = None
        self.deform_stream = None
        self.slots = None
        self._active = False
        self._last_fallback_reason = None
        self._last_used_tacker = False
        self._last_qualification_mode = False
        self._last_caller_stream = None
        self._fallback_renderer = TwoStreamRenderer(
            pc,
            pipe,
            bg_color,
            scaling_modifier=scaling_modifier,
            stage=stage,
            cam_type=cam_type,
        )

    @property
    def fallback_reason(self):
        if self._profile_error is not None:
            return "invalid Tacker profile: {}".format(self._profile_error)
        reason = tacker_support_reason(
            self.pc,
            self.pipe,
            self.profile,
            stage=self.stage,
            cam_type=self.cam_type,
            override_color=None,
            workload_name=self.workload_name,
            iteration=self.iteration,
            qualification_mode=self.qualification_mode,
        )
        if reason is None and self._cache_error is not None:
            return "cannot cache selected-head parameters: {}".format(
                self._cache_error
            )
        return reason

    @property
    def last_fallback_reason(self):
        return self._last_fallback_reason

    @property
    def last_used_tacker(self):
        return self._last_used_tacker

    @property
    def last_qualification_mode(self):
        """Whether the last physical run bypassed measured admission explicitly."""

        return self._last_qualification_mode

    @property
    def fallback_backend_reason(self):
        """Why the delegated two-stream renderer itself selected serial."""

        if self._last_used_tacker:
            return None
        return getattr(self._fallback_renderer, "last_fallback_reason", None)

    @property
    def actual_execution_mode(self):
        """Return ``not_run``, ``tacker``, ``two_stream``, or ``serial``."""

        if self._last_fallback_reason is None and not self._last_used_tacker:
            return "not_run"
        if self._last_used_tacker:
            return "tacker"
        if self.fallback_backend_reason is not None:
            return "serial"
        return "two_stream"

    def _ensure_cuda_resources(self):
        if self.raster_stream is not None:
            return
        self.raster_stream = torch.cuda.Stream(device=self.device)
        self.deform_stream = torch.cuda.Stream(device=self.device)
        self.slots = [
            _TackerSlot(
                ready=torch.cuda.Event(blocking=False),
                prefix_ready=torch.cuda.Event(blocking=False),
                mixed_done=torch.cuda.Event(blocking=False),
                raster_done=torch.cuda.Event(blocking=False),
            )
            for _ in range(2)
        ]
        if self._cache_ready is not None:
            self.raster_stream.wait_event(self._cache_ready)
            self.deform_stream.wait_event(self._cache_ready)

    def prepare(self):
        """Finish one-time parameter/resource setup before timed rendering."""

        reason = self.fallback_reason
        if reason is None:
            self._ensure_cuda_resources()
        else:
            prepare_fallback = getattr(self._fallback_renderer, "prepare", None)
            if callable(prepare_fallback):
                prepare_fallback()
        if self._cache_ready is not None:
            self._cache_ready.synchronize()
        return reason

    @staticmethod
    def _stream_identity(stream):
        return (
            getattr(stream, "device", None),
            getattr(stream, "cuda_stream", id(stream)),
        )

    def _require_caller_stream(self, caller_stream):
        current = torch.cuda.current_stream(self.device)
        if self._stream_identity(current) != self._stream_identity(caller_stream):
            raise RuntimeError(
                "TackerRenderer.render_sequence() must be consumed from the "
                "same CUDA stream that started it"
            )

    def _require_profile_view(self, view):
        reason = _view_profile_reason(view, self.profile)
        if reason is not None:
            # Work already queued for earlier fixed-workload frames must finish
            # before reporting the heterogeneous-view contract violation.
            self.synchronize()
            raise RuntimeError("Tacker workload changed mid-sequence: {}".format(reason))

    def _prepare_context(self, view):
        return prepare_render_context(
            view,
            self.pc,
            self.pipe,
            self.bg_color,
            scaling_modifier=self.scaling_modifier,
            override_color=None,
            cam_type=self.cam_type,
        )

    def _wait_slot_reuse(self, slot):
        if slot.has_raster_done:
            self.deform_stream.wait_event(slot.raster_done)

    def _enqueue_full_deformation(self, view, slot):
        with torch.cuda.stream(self.deform_stream):
            self._wait_slot_reuse(slot)
            slot.context = self._prepare_context(view)
            slot.state = deform_for_render(slot.context, self.pc, stage=self.stage)
            slot.task = None
            _record_context_stream(slot.context, self.deform_stream)
            slot.state.record_stream(self.deform_stream)
            slot.ready.record(self.deform_stream)

    def _enqueue_prefix(self, view, slot):
        with torch.cuda.stream(self.deform_stream):
            self._wait_slot_reuse(slot)
            slot.context = self._prepare_context(view)
            slot.state = None
            slot.task = prepare_pos_head_task(
                slot.context,
                self.pc,
                self.deform_stream,
                head_weight=self.head_weight,
                head_bias=self.head_bias,
                prefix_ready=slot.prefix_ready,
            )
            _record_context_stream(slot.context, self.deform_stream)
            slot.task.record_stream(self.deform_stream)

    def _enqueue_suffix(self, slot):
        with torch.cuda.stream(self.deform_stream):
            self.deform_stream.wait_event(slot.mixed_done)
            _record_tensor_stream(slot.task.head_output, self.deform_stream)
            slot.state = finish_pos_head_task(
                slot.task,
                self.pc,
                slot.task.head_output,
            )
            slot.state.record_stream(self.deform_stream)
            slot.ready.record(self.deform_stream)

    def _enqueue_mixed(self, current_slot, next_slot):
        with torch.cuda.stream(self.raster_stream):
            self.raster_stream.wait_event(current_slot.ready)
            self.raster_stream.wait_event(next_slot.prefix_ready)
            current_slot.state.record_stream(self.raster_stream)
            _record_context_stream(current_slot.context, self.raster_stream)
            next_slot.task.record_stream(self.raster_stream)

            mixed = _forward_with_head(
                current_slot.context,
                current_slot.state,
                next_slot.task,
                self.persistent_blocks,
            )
            result, head_output = _result_from_mixed(current_slot.context, mixed)
            next_slot.task.head_output = head_output
            _record_tensor_stream(head_output, self.raster_stream)
            result.record_stream(self.raster_stream)

            next_slot.mixed_done.record(self.raster_stream)
            current_slot.raster_done.record(self.raster_stream)
            current_slot.has_raster_done = True
        return result

    def _enqueue_solo_raster(self, slot):
        with torch.cuda.stream(self.raster_stream):
            self.raster_stream.wait_event(slot.ready)
            slot.state.record_stream(self.raster_stream)
            _record_context_stream(slot.context, self.raster_stream)
            result = rasterize_state(slot.context, slot.state)
            result.record_stream(self.raster_stream)
            slot.raster_done.record(self.raster_stream)
            slot.has_raster_done = True
        return result

    def render_sequence(self, views):
        """Yield input-ordered frames using prefill, mixed steady state, drain."""

        if self._active:
            raise RuntimeError(
                "TackerRenderer does not support interleaved or reentrant sequences"
            )
        self._active = True
        try:
            iterator = iter(views)
            try:
                current_view = next(iterator)
            except StopIteration:
                return

            fallback_reason = self.fallback_reason
            if fallback_reason is None:
                fallback_reason = _view_profile_reason(current_view, self.profile)
            self._last_fallback_reason = fallback_reason
            self._last_used_tacker = fallback_reason is None
            self._last_qualification_mode = bool(
                fallback_reason is None and self.qualification_mode
            )
            if fallback_reason is not None:
                for output in self._fallback_renderer.render_sequence(
                    _prepend(current_view, iterator)
                ):
                    yield output
                return

            self._ensure_cuda_resources()
            caller_stream = torch.cuda.current_stream(self.device)
            self._last_caller_stream = caller_stream
            self.raster_stream.wait_stream(caller_stream)
            self.deform_stream.wait_stream(caller_stream)

            # D(0) is the only full deformation prefill.  Every steady-state
            # D(t+1) below is split around the physical selected head.
            self._enqueue_full_deformation(current_view, self.slots[0])
            try:
                next_view = next(iterator)
                self._require_profile_view(next_view)
                has_next = True
            except StopIteration:
                next_view = None
                has_next = False

            frame_index = 0
            while True:
                self._require_caller_stream(caller_stream)
                current_slot = self.slots[frame_index % 2]

                if has_next:
                    next_slot = self.slots[(frame_index + 1) % 2]
                    self._enqueue_prefix(next_view, next_slot)
                    result = self._enqueue_mixed(current_slot, next_slot)
                    self._enqueue_suffix(next_slot)
                else:
                    result = self._enqueue_solo_raster(current_slot)

                self._require_caller_stream(caller_stream)
                caller_stream.wait_event(current_slot.raster_done)
                result.record_stream(caller_stream)
                yield result.as_dict()

                self._require_caller_stream(caller_stream)
                if not has_next:
                    return
                current_view = next_view
                frame_index += 1
                try:
                    next_view = next(iterator)
                    self._require_profile_view(next_view)
                    has_next = True
                except StopIteration:
                    next_view = None
                    has_next = False
        finally:
            self._active = False

    def synchronize(self):
        """Wait for the backend used by the most recent sequence."""

        if self.actual_execution_mode == "not_run":
            # Parameter conversion is intentionally constructor-time work. An
            # explicit pre-warmup synchronize keeps it outside timed regions.
            if self._cache_ready is not None:
                self._cache_ready.synchronize()
            return
        if self._last_used_tacker:
            if self.raster_stream is not None:
                self.raster_stream.synchronize()
                self.deform_stream.synchronize()
            if self._last_caller_stream is not None:
                self._last_caller_stream.synchronize()
            return
        synchronize_fallback = getattr(self._fallback_renderer, "synchronize", None)
        if callable(synchronize_fallback):
            synchronize_fallback()


def _prepend(first, iterator):
    """Python-3.7-compatible one-element chain without materialising views."""

    yield first
    for item in iterator:
        yield item


__all__ = [
    "DEFAULT_PROFILE_PATH",
    "LEGACY_PROFILE_SCHEMA_VERSION",
    "LEGACY_VARIANT_ID",
    "PAIR_KEY",
    "PROFILE_SCHEMA_VERSION",
    "SELECTION_OBJECTIVE",
    "WORKLOAD_KEY",
    "PosHeadTask",
    "TackerProfileError",
    "TackerRenderer",
    "finish_pos_head_task",
    "load_tacker_profile",
    "manifest_sha256",
    "profile_sha256",
    "prepare_pos_head_task",
    "tacker_profile_admission_reason",
    "tacker_support_reason",
    "selected_tacker_candidate",
    "validate_tacker_profile",
]
