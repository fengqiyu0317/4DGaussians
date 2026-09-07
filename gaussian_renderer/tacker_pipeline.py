"""Inference-only Raster(t) + deformation-head(t+1) Tacker pipeline.

The production path is deliberately fail closed.  It is enabled only by a
hashed, measured admission profile and by an exact model/rasterizer contract.
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


PROFILE_SCHEMA_VERSION = 1
RASTERIZER_COMMIT = "e49506654e8e11ed8a62d22bcb693e943fdecacf"
PAIR_KEY = "raster.render_leaf+deformation.pos_deform[1].linear_128x128"
DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tacker_profiles"
    / "raster_head_sm86.json"
)

_LOCKED_LIMITS = {
    "raster_slowdown_pct_max": 5.0,
    "end_to_end_ratio_max": 1.0,
    "psnr_drop_db_max": 0.05,
    "ssim_drop_max": 0.0001,
    "lpips_increase_max": 0.0001,
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
    ).encode("utf-8")


def manifest_sha256(manifest):
    """Return the stable SHA-256 used by the profile's sealed manifest."""

    return hashlib.sha256(_canonical_json(manifest)).hexdigest()


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_exact(mapping, key, expected, section):
    if key not in mapping:
        raise TackerProfileError("{}.{} is required".format(section, key))
    if mapping[key] != expected:
        raise TackerProfileError(
            "{}.{} must be {!r}".format(section, key, expected)
        )


def validate_tacker_profile(profile):
    """Validate profile schema, sealed manifest, and locked gate strength.

    This validates structure rather than admission measurements.  Use
    :func:`tacker_profile_admission_reason` for the measured gate decision.
    """

    if not isinstance(profile, dict):
        raise TackerProfileError("profile must be a JSON object")
    _require_exact(
        profile,
        "schema_version",
        PROFILE_SCHEMA_VERSION,
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
    for key, locked_limit in _LOCKED_LIMITS.items():
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


def load_tacker_profile(profile_path=None, profile_override=None):
    """Load and structurally validate one profile.

    ``profile_override`` is an explicit injection point for tests and for a
    caller that has just produced a measured profile.  It does not bypass the
    manifest hash or any admission threshold.
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
    """Return ``None`` only when every measured profile gate passes."""

    try:
        validate_tacker_profile(profile)
    except TackerProfileError as error:
        return "invalid Tacker profile: {}".format(error)

    admission = profile["admission"]
    if not admission["enabled"]:
        return "Tacker profile is disabled"
    if not admission["valid"]:
        return "Tacker profile is not marked valid"

    thresholds = profile["thresholds"]
    measured = profile["measurements"]
    if measured["raster_slowdown_pct"] > thresholds["raster_slowdown_pct_max"]:
        return "Raster QoS slowdown exceeds the profile threshold"
    solo_sum = measured["solo_raster_p50_ms"] + measured["solo_head_p50_ms"]
    if measured["mixed_p50_ms"] >= solo_sum:
        return "mixed p50 is not strictly faster than the solo-kernel sum"
    end_to_end_ratio = (
        measured["tacker_end_to_end_p50_ms"]
        / measured["two_stream_end_to_end_p50_ms"]
    )
    if end_to_end_ratio > thresholds["end_to_end_ratio_max"]:
        return "Tacker end-to-end p50 is slower than two_stream"
    if measured["psnr_drop_db"] > thresholds["psnr_drop_db_max"]:
        return "PSNR drop exceeds the profile threshold"
    if measured["ssim_drop"] > thresholds["ssim_drop_max"]:
        return "SSIM drop exceeds the profile threshold"
    if measured["lpips_increase"] > thresholds["lpips_increase_max"]:
        return "LPIPS increase exceeds the profile threshold"
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
            self.persistent_blocks = self.profile["manifest"]["persistent_blocks"]

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
    "PAIR_KEY",
    "PROFILE_SCHEMA_VERSION",
    "PosHeadTask",
    "TackerProfileError",
    "TackerRenderer",
    "finish_pos_head_task",
    "load_tacker_profile",
    "manifest_sha256",
    "prepare_pos_head_task",
    "tacker_profile_admission_reason",
    "tacker_support_reason",
    "validate_tacker_profile",
]
