"""Experimental gsplat RGB+D + next-frame ABI-v5 or legacy device-v7 MLP.

The parent full_mlp_overlap_graph qualification validates the reused workload,
MLP layout and graph schedule; it does NOT qualify the new gsplat leaf. Its
correctness and resources are recorded separately by the mixed benchmark.
"""

from copy import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from gsplat_mixed_ext import (
    CANDIDATE_AWARE_VARIANTS, KERNEL_VARIANTS, LATEST_KERNEL_VARIANT, WIDE_WARP_SPLITS,
)

from . import (RenderResult, _record_context_stream, prepare_render_context,
               deform_for_render)
from .tacker_pipeline import (
    _HexPlaneCudaGraphSlot, _cache_full_mlp_parameters, finish_full_mlp_task,
    fusion_variant_from_candidate, load_tacker_profile,
    prepare_full_mlp_graph_task, tacker_support_reason,
)


@dataclass
class PreparedGsplat:
    context: object
    inputs: tuple
    radii: torch.Tensor
    width: int
    height: int

    def result(self, image):
        return RenderResult(
            render=image[0, ..., :3].permute(2, 0, 1).contiguous(),
            viewspace_points=self.context.screenspace_points,
            visibility_filter=self.radii > 0, radii=self.radii,
            depth=image[0, ..., 3].unsqueeze(0).contiguous(),
        )


def prepare_gsplat(context, state):
    """Exact gsplat 1.5.3 dense/classic/RGB+D prefix, stopping before pixels."""
    from gsplat.cuda._wrapper import (fully_fused_projection, spherical_harmonics,
                                      isect_tiles, isect_offset_encode)
    rasterizer = context.rasterizer
    settings = rasterizer.raster_settings
    if rasterizer.packed or context.cov3D_precomp is not None:
        raise ValueError("mixed prototype requires dense scales/quaternions inputs")
    width, height = settings.image_width, settings.image_height
    opacity = state.opacities.squeeze(-1)
    scales = state.scales
    if settings.scale_modifier != 1:
        scales = scales * settings.scale_modifier
    radii, means, depths, conics, _ = fully_fused_projection(
        state.means3D, None, state.rotations, scales,
        rasterizer.viewmats, rasterizer.Ks, width, height,
        near_plane=0.2, far_plane=1e10, eps2d=0.3, packed=False,
        radius_clip=0, opacities=opacity,
    )
    opacity = opacity.unsqueeze(0)
    if context.colors_precomp is None:
        campos = torch.inverse(rasterizer.viewmats)[..., :3, 3]
        dirs = state.means3D.unsqueeze(0) - campos[:, None, :]
        colors = spherical_harmonics(settings.sh_degree, dirs,
                                    state.shs.unsqueeze(0), masks=(radii > 0).all(dim=-1))
        colors = torch.clamp_min(colors + 0.5, 0)
    else:
        colors = context.colors_precomp.unsqueeze(0)
    colors = torch.cat((colors, depths[..., None]), dim=-1)
    background = torch.cat((settings.bg, settings.bg.new_zeros(1))).unsqueeze(0)
    tw, th = (width + 15) // 16, (height + 15) // 16
    _, intersections, ids = isect_tiles(means, radii, depths, 16, tw, th, packed=False)
    offsets = isect_offset_encode(intersections, 1, tw, th).reshape(1, th, tw)
    return PreparedGsplat(context, (means, conics, colors, opacity, offsets, ids, background),
                          radii[0].amax(dim=-1), width, height)


def render_prepared(prepared):
    from gsplat.cuda._wrapper import rasterize_to_pixels
    means, conics, colors, opacity, offsets, ids, background = prepared.inputs
    image, _ = rasterize_to_pixels(means, conics, colors, opacity,
                                   prepared.width, prepared.height, 16, offsets, ids,
                                   backgrounds=background, packed=False)
    return prepared.result(image)


class GsplatMixedRenderer:
    """Two slots, HexPlane CUDA Graph, and one physical mixed leaf."""

    def __init__(self, pc, pipe, background, qualification_profile,
                 workload_name="flame_steak", iteration=14000, physical_blocks=0,
                 fused=True, cam_type="dynerf", kernel_variant=LATEST_KERNEL_VARIANT, candidate_pairing=True):
        if torch.is_grad_enabled():
            raise RuntimeError("gsplat mixed is inference-only")
        if getattr(pipe, "raster_backend", None) != "gsplat" or getattr(pipe, "gsplat_packed", False):
            raise ValueError("gsplat mixed requires raster_backend=gsplat and dense layout")
        import gsplat
        if gsplat.__version__ != "1.5.3":
            raise RuntimeError("mixed prototype is pinned to gsplat 1.5.3")
        for name in ("compute_cov3D_python", "convert_SHs_python"):
            if getattr(pipe, name, False):
                raise ValueError("gsplat mixed does not support " + name)
        self.parent_profile = load_tacker_profile(qualification_profile)
        candidates = self.parent_profile["candidates"]
        candidate = next(c for c in candidates if c["variant_id"] == self.parent_profile["selected_variant_id"])
        if candidate["partition"]["kind"] != "full_mlp_overlap_graph":
            raise ValueError("an explicit full_mlp_overlap_graph parent qualification is required")
        admission_pipe = copy(pipe)
        admission_pipe.raster_backend = "diff_gaussian"
        reason = tacker_support_reason(pc, admission_pipe, self.parent_profile,
                                       cam_type=cam_type, workload_name=workload_name,
                                       iteration=iteration, qualification_mode=True)
        if reason is not None:
            raise RuntimeError("parent Graph workload qualification failed: " + reason)
        self.variant = fusion_variant_from_candidate(candidate)
        self.pc, self.pipe, self.background = pc, pipe, background
        self.cam_type = cam_type
        self.physical_blocks, self.fused = physical_blocks, fused
        if kernel_variant not in KERNEL_VARIANTS:
            raise ValueError("unknown gsplat mixed kernel variant: " + str(kernel_variant))
        self.kernel_variant = kernel_variant
        self.candidate_pairing = bool(candidate_pairing)
        self.qualification_profile = str(Path(qualification_profile).resolve())
        self.qualification_sha256 = hashlib.sha256(Path(qualification_profile).read_bytes()).hexdigest()
        self.cached = _cache_full_mlp_parameters(pc)
        self.mlp_implementation = "device_v7" if kernel_variant in (
            "tensor_v7", *CANDIDATE_AWARE_VARIANTS) else "abi_v5"
        self.leaf_cached = self.cached
        if self.mlp_implementation == "device_v7":
            from tacker_4dgs_head import pack_deform_tail_weights_v3
            self.leaf_cached = (*self.cached[:4], pack_deform_tail_weights_v3(self.cached[4]), self.cached[5])
        self.raster_stream = torch.cuda.Stream(device=pc.get_xyz.device)
        self.deform_stream = torch.cuda.Stream(device=pc.get_xyz.device)
        self.graphs = None
        self._active = False
        self.last_execution_counts = {}
        self.last_fallback_reason = None

    def prepare(self):
        from gsplat_mixed_ext import load
        self.extension = load(self.kernel_variant)
        if self.graphs is not None:
            return
        torch.cuda.synchronize()
        timestamp = self.pc.get_xyz.new_zeros((self.pc.get_xyz.shape[0], 1))
        self.graphs = [_HexPlaneCudaGraphSlot(self.pc._deformation.deformation_net,
                                            self.pc.get_xyz, timestamp, self.deform_stream)
                       for _ in range(2)]
        self.ready = [torch.cuda.Event() for _ in range(2)]
        self.prefix_ready = [torch.cuda.Event() for _ in range(2)]
        self.mixed_done = [torch.cuda.Event() for _ in range(2)]
        self.raster_done = [torch.cuda.Event() for _ in range(2)]
        torch.cuda.synchronize()

    def render_sequence(self, views):
        if self._active:
            raise RuntimeError("gsplat mixed renderer is not reentrant")
        if not len(views):
            return
        if self.graphs is None:
            self.prepare()
        self._active = True
        caller = torch.cuda.current_stream()
        self.raster_stream.wait_stream(caller)
        self.deform_stream.wait_stream(caller)
        counts = {"input_frames": len(views), "full_deformation": 1, "prefix_graph_replays": 0,
                  "mixed_launches": 0, "separate_leaf_pairs": 0, "raster_prepares": 0,
                  "solo_raster": 0, "outputs": 0}
        self.last_execution_counts = counts
        contexts, states = [None, None], [None, None]

        def context(view):
            return prepare_render_context(view, self.pc, self.pipe, self.background, cam_type=self.cam_type)

        try:
            contexts[0] = context(views[0])
            _record_context_stream(contexts[0], self.deform_stream)
            self.deform_stream.wait_stream(caller)
            with torch.cuda.stream(self.deform_stream):
                states[0] = deform_for_render(contexts[0], self.pc)
                self.ready[0].record()
            for i in range(len(views)):
                current, next_slot = i % 2, (i + 1) % 2
                has_next = i + 1 < len(views)
                task = None
                if has_next:
                    contexts[next_slot] = context(views[i + 1])
                    _record_context_stream(contexts[next_slot], self.deform_stream)
                    self.deform_stream.wait_stream(caller)
                    if i > 0:
                        self.deform_stream.wait_event(self.raster_done[next_slot])
                    task = prepare_full_mlp_graph_task(
                        contexts[next_slot], self.pc, self.deform_stream, self.variant,
                        self.cached, self.graphs[next_slot], self.prefix_ready[next_slot])
                    counts["prefix_graph_replays"] += 1
                with torch.cuda.stream(self.raster_stream):
                    self.raster_stream.wait_event(self.ready[current])
                    _record_context_stream(contexts[current], self.raster_stream)
                    states[current].record_stream(self.raster_stream)
                    prepared = prepare_gsplat(contexts[current], states[current])
                    counts["raster_prepares"] += 1
                    if has_next:
                        self.raster_stream.wait_event(task.prefix_ready)
                        task.deform_features.record_stream(self.raster_stream)
                        mlp = [task.deform_features, *self.leaf_cached]
                        if self.fused:
                            if self.kernel_variant in CANDIDATE_AWARE_VARIANTS:
                                raw = self.extension.forward(prepared.inputs, mlp, prepared.width,
                                    prepared.height, self.physical_blocks, 2, False, self.candidate_pairing)
                            else:
                                raw = self.extension.forward(prepared.inputs, mlp, prepared.width,
                                                             prepared.height, self.physical_blocks)
                            result = prepared.result(raw[0])
                            deltas = raw[3:8]
                            counts["mixed_launches"] += 1
                        else:
                            # Same custom MLP, prefix and residual rules; only
                            # the physical fusion differs from the mixed arm.
                            result = self.render_prepared_raster(prepared)
                            if self.mlp_implementation == "device_v7":
                                deltas = self.extension.mlp_forward(mlp)
                            else:
                                from tacker_4dgs_head import deform_mlp_delta_v1 as deform_mlp_delta
                                deltas = deform_mlp_delta(*mlp)
                            counts["separate_leaf_pairs"] += 1
                        self.mixed_done[next_slot].record()
                    else:
                        result = self.render_prepared_raster(prepared)
                        counts["solo_raster"] += 1
                    self.raster_done[current].record()
                if has_next:
                    self.deform_stream.wait_event(self.mixed_done[next_slot])
                    for delta in deltas:
                        delta.record_stream(self.deform_stream)
                    with torch.cuda.stream(self.deform_stream):
                        states[next_slot] = finish_full_mlp_task(task, self.pc, deltas)
                        self.ready[next_slot].record()
                caller.wait_event(self.raster_done[current])
                result.record_stream(caller)
                counts["outputs"] += 1
                yield result.as_dict()
        finally:
            self._active = False

    def render_prepared_raster(self, prepared):
        if self.kernel_variant in WIDE_WARP_SPLITS:
            image, _, _ = self.extension.raster_forward(prepared.inputs, prepared.width, prepared.height)
            return prepared.result(image)
        return render_prepared(prepared)

    def synchronize(self):
        self.raster_stream.synchronize()
        self.deform_stream.synchronize()
