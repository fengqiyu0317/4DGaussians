#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""Gaussian deformation and rasterization entry points.

``render`` remains the compatibility entry point.  The split API makes the
dependency between deformation and rasterization explicit so inference code
can pipeline deformation of frame t+1 with rasterization of frame t.
"""

from dataclasses import dataclass, fields
import math

import torch

import diff_gaussian_rasterization as _rasterizer_module
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.profiling_utils import nvtx_range
from utils.sh_utils import eval_sh


@dataclass
class RenderContext:
    """Per-view inputs that do not depend on the deformation result."""

    viewpoint_camera: object
    screenspace_points: torch.Tensor
    rasterizer: object
    means3D: torch.Tensor
    means2D: torch.Tensor
    scales: object
    rotations: object
    opacity: torch.Tensor
    shs: torch.Tensor
    timestamp: torch.Tensor
    cov3D_precomp: object
    colors_precomp: object
    convert_shs_python: bool


@dataclass
class GaussianRenderState:
    """Activated Gaussian attributes consumed by the rasterizer."""

    means3D: torch.Tensor
    scales: object
    rotations: object
    opacities: torch.Tensor
    shs: object

    def record_stream(self, stream):
        """Keep storage live until work submitted to ``stream`` completes."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(stream)


@dataclass
class RenderResult:
    """Typed form of the dictionary historically returned by ``render``."""

    render: torch.Tensor
    viewspace_points: torch.Tensor
    visibility_filter: torch.Tensor
    radii: torch.Tensor
    depth: torch.Tensor

    def as_dict(self):
        return {
            "render": self.render,
            "viewspace_points": self.viewspace_points,
            "visibility_filter": self.visibility_filter,
            "radii": self.radii,
            "depth": self.depth,
        }

    def record_stream(self, stream):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(stream)


def _record_context_stream(context, stream):
    for field in fields(context):
        value = getattr(context, field.name)
        if isinstance(value, torch.Tensor) and value.is_cuda:
            value.record_stream(stream)

    # Camera matrices, background, and camera position live inside the
    # rasterizer's NamedTuple rather than as direct RenderContext fields.  A
    # slot may drop its old rasterizer object immediately after enqueueing a
    # wait for ``raster_done``; explicitly recording these nested tensors keeps
    # the caching allocator from recycling them while Raster still reads them.
    rasterizer = getattr(context, "rasterizer", None)
    settings = getattr(rasterizer, "raster_settings", None)
    if settings is not None:
        for value in settings:
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(stream)


def prepare_render_context(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    cam_type=None,
):
    """Build camera and rasterizer inputs for one frame."""
    with nvtx_range("renderer/setup"):
        screenspace_points = torch.zeros_like(
            pc.get_xyz,
            dtype=pc.get_xyz.dtype,
            requires_grad=True,
            device="cuda",
        ) + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        means3D = pc.get_xyz
        if cam_type != "PanopticSports":
            tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.image_height),
                image_width=int(viewpoint_camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_camera.world_view_transform.cuda(),
                projmatrix=viewpoint_camera.full_proj_transform.cuda(),
                sh_degree=pc.active_sh_degree,
                campos=viewpoint_camera.camera_center.cuda(),
                prefiltered=False,
                debug=pipe.debug,
            )
            timestamp_value = viewpoint_camera.time
        else:
            raster_settings = viewpoint_camera["camera"]
            timestamp_value = viewpoint_camera["time"]

        # Keep the legacy copy/detach semantics.  In particular, ``as_tensor``
        # would retain an input tensor's autograd history during training.
        timestamp = torch.tensor(timestamp_value).to(means3D.device).repeat(
            means3D.shape[0], 1
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        opacity = pc._opacity
        shs = pc.get_features

        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc._scaling
            rotations = pc._rotation

    return RenderContext(
        viewpoint_camera=viewpoint_camera,
        screenspace_points=screenspace_points,
        rasterizer=rasterizer,
        means3D=means3D,
        means2D=screenspace_points,
        scales=scales,
        rotations=rotations,
        opacity=opacity,
        shs=shs,
        timestamp=timestamp,
        cov3D_precomp=cov3D_precomp,
        colors_precomp=override_color,
        convert_shs_python=bool(pipe.convert_SHs_python),
    )


def deform_for_render(context: RenderContext, pc: GaussianModel, stage="fine"):
    """Run temporal deformation and renderer-facing activations."""
    with nvtx_range("renderer/deformation"):
        if "coarse" in stage:
            means3D_final = context.means3D
            scales_final = context.scales
            rotations_final = context.rotations
            opacity_final = context.opacity
            shs_final = context.shs
        elif "fine" in stage:
            (
                means3D_final,
                scales_final,
                rotations_final,
                opacity_final,
                shs_final,
            ) = pc._deformation(
                context.means3D,
                context.scales,
                context.rotations,
                context.opacity,
                context.shs,
                context.timestamp,
            )
        else:
            raise NotImplementedError("unsupported render stage: {}".format(stage))

    with nvtx_range("renderer/activation"):
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity_final = pc.opacity_activation(opacity_final)

    # Preserve the original execution order: Python SH conversion happens
    # after deformation/activation and immediately before rasterization.
    if context.colors_precomp is None and context.convert_shs_python:
        with nvtx_range("renderer/sh_to_rgb_python"):
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            camera_center = context.viewpoint_camera.camera_center.cuda()
            dir_pp = pc.get_xyz - camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            context.colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    return GaussianRenderState(
        means3D=means3D_final,
        scales=scales_final,
        rotations=rotations_final,
        opacities=opacity_final,
        shs=shs_final,
    )


def rasterize_state(context: RenderContext, state: GaussianRenderState):
    """Rasterize one already-deformed Gaussian state."""
    with nvtx_range("renderer/rasterization"):
        rendered_image, radii, depth = context.rasterizer(
            means3D=state.means3D,
            means2D=context.means2D,
            shs=state.shs,
            colors_precomp=context.colors_precomp,
            opacities=state.opacities,
            scales=state.scales,
            rotations=state.rotations,
            cov3D_precomp=context.cov3D_precomp,
        )
    return RenderResult(
        render=rendered_image,
        viewspace_points=context.screenspace_points,
        visibility_filter=radii > 0,
        radii=radii,
        depth=depth,
    )


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    stage="fine",
    cam_type=None,
):
    """Compatibility renderer; signature and return dictionary are unchanged."""
    context = prepare_render_context(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier=scaling_modifier,
        override_color=override_color,
        cam_type=cam_type,
    )
    state = deform_for_render(context, pc, stage=stage)
    return rasterize_state(context, state).as_dict()


def two_stream_support_reason(pipe, stage="fine", cam_type=None, override_color=None):
    """Return ``None`` when the experimental inference pipeline is supported."""
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if torch.is_grad_enabled():
        return "autograd is enabled"
    if "fine" not in stage:
        return "only the fine inference stage is supported"
    if cam_type == "PanopticSports":
        return "PanopticSports uses an opaque camera/rasterizer path"
    if pipe.compute_cov3D_python:
        return "compute_cov3D_python requires the serial path"
    if pipe.convert_SHs_python:
        return "convert_SHs_python requires the serial path"
    if override_color is not None:
        return "override_color requires the serial path"
    capability_provider = getattr(_rasterizer_module, "tacker_capabilities", None)
    if capability_provider is None:
        rasterizer_backend = getattr(_rasterizer_module, "_C", None)
        capability_provider = getattr(rasterizer_backend, "tacker_capabilities", None)
    if capability_provider is None:
        return "rasterizer extension does not advertise stream-aware execution"
    try:
        capabilities = (
            capability_provider()
            if callable(capability_provider)
            else capability_provider
        )
    except Exception as error:
        return "rasterizer capability query failed: {}".format(error)
    if isinstance(capabilities, dict):
        stream_aware = capabilities.get("stream_aware", False)
    else:
        stream_aware = getattr(capabilities, "stream_aware", False)
    if not bool(stream_aware):
        return "rasterizer extension is not stream-aware"
    return None


@dataclass
class _PipelineSlot:
    ready: object
    raster_done: object
    context: object = None
    state: object = None
    has_raster_done: bool = False


class TwoStreamRenderer:
    """Two-slot inference pipeline for ``R(t) || D(t + 1)``.

    Rasterization and deformation use independent non-default streams. The
    rasterizer extension receives PyTorch's current stream, so its complete
    preprocess/sort/render chain follows ``raster_stream``. Events order slot
    reuse and producer/consumer dependencies.
    """

    def __init__(self, pc, pipe, bg_color, scaling_modifier=1.0, stage="fine", cam_type=None):
        self.pc = pc
        self.pipe = pipe
        self.bg_color = bg_color
        self.scaling_modifier = scaling_modifier
        self.stage = stage
        self.cam_type = cam_type
        # CUDA resources are intentionally lazy. Constructing this object must
        # remain safe on CPU-only hosts and for configurations that will use the
        # serial fallback.
        self.device = getattr(pc.get_xyz, "device", None)
        self.raster_stream = None
        self.deform_stream = None
        self.slots = None
        self._active = False
        self._last_fallback_reason = None
        self._last_caller_stream = None

    @property
    def fallback_reason(self):
        """Current support result; recomputed rather than cached at construction."""
        return two_stream_support_reason(
            self.pipe,
            stage=self.stage,
            cam_type=self.cam_type,
            override_color=None,
        )

    @property
    def last_fallback_reason(self):
        """Support result used by the most recently started sequence."""
        return self._last_fallback_reason

    def _ensure_cuda_resources(self):
        if self.raster_stream is not None:
            return
        raster_stream = torch.cuda.Stream(device=self.device)
        deform_stream = torch.cuda.Stream(device=self.device)
        slots = [
            _PipelineSlot(
                ready=torch.cuda.Event(blocking=False),
                raster_done=torch.cuda.Event(blocking=False),
            )
            for _ in range(2)
        ]
        self.raster_stream = raster_stream
        self.deform_stream = deform_stream
        self.slots = slots

    def prepare(self):
        """Create supported stream/event resources outside a timed sequence."""

        reason = self.fallback_reason
        if reason is None:
            self._ensure_cuda_resources()
        return reason

    @staticmethod
    def _stream_identity(stream):
        """Return a stable CUDA-stream identity for consumer-switch checks."""
        return (
            getattr(stream, "device", None),
            getattr(stream, "cuda_stream", id(stream)),
        )

    def _require_caller_stream(self, caller_stream):
        current_stream = torch.cuda.current_stream(self.device)
        if self._stream_identity(current_stream) != self._stream_identity(caller_stream):
            raise RuntimeError(
                "TwoStreamRenderer.render_sequence() must be consumed from the "
                "same CUDA stream that started it"
            )

    def _enqueue_deformation(self, view, slot):
        with torch.cuda.stream(self.deform_stream):
            if slot.has_raster_done:
                self.deform_stream.wait_event(slot.raster_done)
            slot.context = prepare_render_context(
                view,
                self.pc,
                self.pipe,
                self.bg_color,
                scaling_modifier=self.scaling_modifier,
                override_color=None,
                cam_type=self.cam_type,
            )
            slot.state = deform_for_render(slot.context, self.pc, stage=self.stage)
            _record_context_stream(slot.context, self.deform_stream)
            slot.state.record_stream(self.deform_stream)
            slot.ready.record(self.deform_stream)

    def render_sequence(self, views):
        """Yield render dictionaries in input order using prefill/steady/drain."""
        if self._active:
            raise RuntimeError(
                "TwoStreamRenderer does not support interleaved or reentrant sequences"
            )
        self._active = True
        try:
            view_iterator = iter(views)
            try:
                current_view = next(view_iterator)
            except StopIteration:
                return

            # Evaluate inference mode, pipeline flags, and extension capability
            # at execution time. A renderer can therefore be constructed before
            # entering ``torch.no_grad()``.
            fallback_reason = self.fallback_reason
            self._last_fallback_reason = fallback_reason
            caller_stream = None
            if torch.cuda.is_available():
                caller_stream = torch.cuda.current_stream(self.device)
            self._last_caller_stream = caller_stream

            if fallback_reason is not None:
                while True:
                    if caller_stream is not None:
                        self._require_caller_stream(caller_stream)
                    serial_result = render(
                        current_view,
                        self.pc,
                        self.pipe,
                        self.bg_color,
                        scaling_modifier=self.scaling_modifier,
                        stage=self.stage,
                        cam_type=self.cam_type,
                    )
                    yield serial_result
                    if caller_stream is not None:
                        self._require_caller_stream(caller_stream)
                    try:
                        current_view = next(view_iterator)
                    except StopIteration:
                        return

            self._ensure_cuda_resources()
            # Model/camera preparation may have been submitted on whichever
            # stream actually starts this sequence, not the constructor's stream.
            self.raster_stream.wait_stream(caller_stream)
            self.deform_stream.wait_stream(caller_stream)

            # One-element lookahead gives D0,D1,R0,D2,R1,... without
            # materialising arbitrary camera iterables.
            self._enqueue_deformation(current_view, self.slots[0])
            try:
                next_view = next(view_iterator)
                has_next = True
            except StopIteration:
                next_view = None
                has_next = False

            frame_index = 0
            while True:
                self._require_caller_stream(caller_stream)
                current_slot = self.slots[frame_index % 2]

                # Enqueue D(t+1) before entering R(t): rasterization performs a
                # synchronous count copy that can temporarily block Python.
                if has_next:
                    self._enqueue_deformation(next_view, self.slots[(frame_index + 1) % 2])

                with torch.cuda.stream(self.raster_stream):
                    self.raster_stream.wait_event(current_slot.ready)
                    current_slot.state.record_stream(self.raster_stream)
                    _record_context_stream(current_slot.context, self.raster_stream)
                    result = rasterize_state(current_slot.context, current_slot.state)
                    result.record_stream(self.raster_stream)
                    current_slot.raster_done.record(self.raster_stream)
                    current_slot.has_raster_done = True

                # Preserve normal PyTorch stream semantics for consumers. The
                # wait establishes readiness; record_stream protects allocator
                # lifetime for work the consumer submits after the yield.
                self._require_caller_stream(caller_stream)
                caller_stream.wait_event(current_slot.raster_done)
                result.record_stream(caller_stream)
                yield result.as_dict()

                # Resumption is another consumer interaction. Reject a stream
                # switch before consuming more input or submitting D(t+2).
                self._require_caller_stream(caller_stream)
                if not has_next:
                    return
                current_view = next_view
                frame_index += 1
                try:
                    next_view = next(view_iterator)
                    has_next = True
                except StopIteration:
                    next_view = None
                    has_next = False
        finally:
            self._active = False

    def synchronize(self):
        if self.raster_stream is not None:
            self.raster_stream.synchronize()
            self.deform_stream.synchronize()
        if self._last_caller_stream is not None:
            self._last_caller_stream.synchronize()


def render_sequence_two_stream(
    views,
    pc,
    pipe,
    bg_color,
    scaling_modifier=1.0,
    stage="fine",
    cam_type=None,
):
    """Convenience wrapper around :class:`TwoStreamRenderer`."""
    renderer = TwoStreamRenderer(
        pc,
        pipe,
        bg_color,
        scaling_modifier=scaling_modifier,
        stage=stage,
        cam_type=cam_type,
    )
    try:
        yield from renderer.render_sequence(views)
    finally:
        renderer.synchronize()


# Import only after the split renderer and TwoStreamRenderer are fully defined:
# tacker_pipeline intentionally reuses those public building blocks.
from .tacker_pipeline import (  # noqa: E402,F401
    TackerProfileError,
    TackerRenderer,
    load_tacker_profile,
    tacker_profile_admission_reason,
    tacker_support_reason,
)
