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

from typing import NamedTuple
import json
import os
from pathlib import Path
import threading
import torch.nn as nn
import torch
from . import _C


_TILE_LOAD_TRACE_PATH = os.environ.get("TACKER_TILE_LOAD_TRACE")
_TILE_LOAD_TRACE_LOCK = threading.Lock()
_TILE_LOAD_TRACE_INDEX = 0


def _record_raster_tile_loads(image_buffer, num_rendered, raster_settings):
    """Append one profiling-only tile-load record when explicitly enabled."""

    global _TILE_LOAD_TRACE_INDEX
    if not _TILE_LOAD_TRACE_PATH:
        return
    if not hasattr(_C, "tacker_image_state"):
        raise RuntimeError(
            "TACKER_TILE_LOAD_TRACE requires the tacker_image_state binding"
        )
    height = int(raster_settings.image_height)
    width = int(raster_settings.image_width)
    grid_y = (height + 15) // 16
    grid_x = (width + 15) // 16
    ranges, contributors = _C.tacker_image_state(image_buffer, height, width)
    range_counts = (
        ranges[:, 1].to(dtype=torch.int64)
        - ranges[:, 0].to(dtype=torch.int64)
    )

    padded_height = grid_y * 16
    padded_width = grid_x * 16
    padded_contributors = torch.zeros(
        (padded_height, padded_width),
        dtype=torch.int64,
        device=contributors.device,
    )
    padded_contributors[:height, :width] = contributors.to(dtype=torch.int64)
    contributor_tiles = (
        padded_contributors.view(grid_y, 16, grid_x, 16)
        .permute(0, 2, 1, 3)
        .reshape(grid_y * grid_x, 256)
    )
    contributor_max = contributor_tiles.max(dim=1).values
    contributor_sum = contributor_tiles.sum(dim=1)

    valid_pixels = torch.ones(
        (height, width), dtype=torch.int64, device=contributors.device
    )
    padded_valid = torch.zeros(
        (padded_height, padded_width),
        dtype=torch.int64,
        device=contributors.device,
    )
    padded_valid[:height, :width] = valid_pixels
    valid_per_tile = (
        padded_valid.view(grid_y, 16, grid_x, 16)
        .permute(0, 2, 1, 3)
        .reshape(grid_y * grid_x, 256)
        .sum(dim=1)
    )
    packed = torch.stack(
        (range_counts, contributor_max, contributor_sum, valid_per_tile),
        dim=1,
    ).cpu()

    with _TILE_LOAD_TRACE_LOCK:
        frame_index = _TILE_LOAD_TRACE_INDEX
        _TILE_LOAD_TRACE_INDEX += 1
        target = Path(_TILE_LOAD_TRACE_PATH).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "frame_index": frame_index,
            "image_width": width,
            "image_height": height,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "tile_count": grid_x * grid_y,
            "num_rendered": int(num_rendered),
            "range_counts": packed[:, 0].tolist(),
            "contributor_max": packed[:, 1].tolist(),
            "contributor_sum": packed[:, 2].tolist(),
            "valid_pixels": packed[:, 3].tolist(),
        }
        with target.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def tacker_capabilities():
    """Return the compiled Raster+head ABI, or an explicit legacy marker."""

    if not hasattr(_C, "tacker_capabilities"):
        return {
            "stream_aware": False,
            "mixed_render_head_abi": 0,
            "reason": "installed extension predates the Tacker mixed ABI",
        }
    return dict(_C.tacker_capabilities())


def tacker_resource_requirements(
    abi_version=2, worker_groups=1, family=None
):
    """Query launch resources/occupancy for one compiled mixed variant."""

    for name, value in (
        ("abi_version", abi_version),
        ("worker_groups", worker_groups),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int")
    if family is not None and not isinstance(family, str):
        raise TypeError("family must be a str or None")
    if not hasattr(_C, "tacker_resource_requirements"):
        raise RuntimeError(
            "installed diff_gaussian_rasterization extension does not "
            "provide the Tacker resource query; rebuild the extension"
        )
    if family is None:
        return dict(
            _C.tacker_resource_requirements(abi_version, worker_groups)
        )
    return dict(
        _C.tacker_resource_requirements(
            abi_version, worker_groups, family
        )
    )


def tacker_variant_resources(
    worker_groups, family=None
):
    """Return the schema-v2 runtime's normalized resource vocabulary."""

    family_to_abi = {
        "first_linear_heads_v2": 2,
        "packed_first_linear_v3": 3,
        "whole_heads_v4": 4,
        "full_deform_mlp_v5": 5,
    }
    if family is None:
        raw = tacker_resource_requirements(
            abi_version=2, worker_groups=worker_groups
        )
        family = "first_linear_heads_v2"
    elif not isinstance(family, str):
        raise TypeError("family must be a str or None")
    else:
        try:
            abi_version = family_to_abi[family]
        except KeyError:
            raise ValueError("unsupported Tacker mixed backend family")
        raw = tacker_resource_requirements(
            abi_version=abi_version,
            worker_groups=worker_groups,
            family=family,
        )
    normalized = dict(raw)
    normalized.update(
        {
            "block_threads": raw["physical_threads"],
            "registers_per_thread": raw["registers_per_thread"],
            "static_shared_memory_bytes": raw["static_shared_bytes"],
            "max_threads_per_block": raw["kernel_max_threads_per_block"],
            "active_blocks_per_sm": raw[
                "active_blocks_per_multiprocessor"
            ],
        }
    )
    return normalized

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        _record_raster_tile_loads(imgBuffer, num_rendered, raster_settings)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool


class TackerPreparedRaster(NamedTuple):
    """Own every tensor whose storage is consumed by a deferred render leaf."""

    num_rendered: int
    color: torch.Tensor
    depth: torch.Tensor
    radii: torch.Tensor
    geom_buffer: torch.Tensor
    binning_buffer: torch.Tensor
    image_buffer: torch.Tensor
    colors_precomp: torch.Tensor

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

    def forward_with_head(
        self,
        means3D,
        means2D,
        opacities,
        head_input,
        head_weight,
        head_bias,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        persistent_blocks=0,
    ):
        """Inference-only physical fusion of Raster and one 128x128 head.

        ``head_input`` and ``head_weight`` are contiguous CUDA FP16 tensors;
        ``head_bias`` is contiguous CUDA FP32.  The returned head tensor is
        FP32 and implements ``head_input @ head_weight.T + head_bias``.
        ``means2D`` is retained for call-site compatibility but is not part of
        the inference-only C++ ABI.
        """

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer.forward_with_head is inference-only; "
                "call it under torch.no_grad()"
            )
        if not hasattr(_C, "rasterize_gaussians_with_head"):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide the Tacker mixed ABI; rebuild the extension"
            )
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either SHs or precomputed colors!"
            )
        if (
            (scales is None or rotations is None) and cov3D_precomp is None
        ) or (
            (scales is not None or rotations is not None)
            and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or "
                "precomputed 3D covariance!"
            )

        # Unlike the legacy autograd path, keep placeholders on the raster
        # tensor's device.  They remain empty and are never dereferenced.
        empty = means3D.new_empty((0,))
        if shs is None:
            shs = empty
        if colors_precomp is None:
            colors_precomp = empty
        if scales is None:
            scales = empty
        if rotations is None:
            rotations = empty
        if cov3D_precomp is None:
            cov3D_precomp = empty

        raster_settings = self.raster_settings
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            head_input,
            head_weight,
            head_bias,
            persistent_blocks,
        )

        (
            _num_rendered,
            color,
            depth,
            radii,
            _geom_buffer,
            _binning_buffer,
            _image_buffer,
            head_output,
        ) = _C.rasterize_gaussians_with_head(*args)
        return color, radii, depth, head_output

    def forward_with_heads(
        self,
        means3D,
        means2D,
        opacities,
        head_inputs,
        head_weights,
        head_biases,
        worker_groups=1,
        persistent_blocks=0,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        """Fuse Raster with 1--5 first-linear deformation-head tasks.

        Each task computes ``input @ weight.T + bias`` and returns a distinct
        FP32 ``[N, 128]`` tensor.  Read-only operands may alias across tasks,
        which permits all five heads to share one hidden activation.  Worker
        group ``g`` evaluates task indices ``g, g + worker_groups, ...``.
        """

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer.forward_with_heads is inference-only; "
                "call it under torch.no_grad()"
            )
        if not hasattr(_C, "rasterize_gaussians_with_heads"):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide the Tacker mixed ABI v2; rebuild the extension"
            )
        for name, values in (
            ("head_inputs", head_inputs),
            ("head_weights", head_weights),
            ("head_biases", head_biases),
        ):
            if not isinstance(values, (list, tuple)):
                raise TypeError(f"{name} must be a list or tuple of tensors")
        head_inputs = tuple(head_inputs)
        head_weights = tuple(head_weights)
        head_biases = tuple(head_biases)
        if not isinstance(worker_groups, int) or isinstance(worker_groups, bool):
            raise TypeError("worker_groups must be an int")
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if worker_groups < 1:
            raise ValueError("worker_groups must be >= 1")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either SHs or precomputed colors!"
            )
        if (
            (scales is None or rotations is None) and cov3D_precomp is None
        ) or (
            (scales is not None or rotations is not None)
            and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or "
                "precomputed 3D covariance!"
            )

        empty = means3D.new_empty((0,))
        if shs is None:
            shs = empty
        if colors_precomp is None:
            colors_precomp = empty
        if scales is None:
            scales = empty
        if rotations is None:
            rotations = empty
        if cov3D_precomp is None:
            cov3D_precomp = empty

        raster_settings = self.raster_settings
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            head_inputs,
            head_weights,
            head_biases,
            worker_groups,
            persistent_blocks,
        )

        (
            _num_rendered,
            color,
            depth,
            radii,
            _geom_buffer,
            _binning_buffer,
            _image_buffer,
            head_outputs,
        ) = _C.rasterize_gaussians_with_heads(*args)
        return color, radii, depth, tuple(head_outputs)

    def forward_with_packed_heads(
        self,
        means3D,
        means2D,
        opacities,
        head_input,
        packed_head_weights,
        packed_head_biases,
        worker_groups=1,
        persistent_blocks=0,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        """Fuse Raster with C3 packed shared-input first-linear heads."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer.forward_with_packed_heads is "
                "inference-only; call it under torch.no_grad()"
            )
        if not hasattr(_C, "rasterize_gaussians_with_packed_heads"):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide the Tacker packed mixed ABI v3; rebuild the extension"
            )
        for name, value in (
            ("worker_groups", worker_groups),
            ("persistent_blocks", persistent_blocks),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
        if worker_groups < 1:
            raise ValueError("worker_groups must be >= 1")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either SHs or precomputed colors!"
            )
        if (
            (scales is None or rotations is None) and cov3D_precomp is None
        ) or (
            (scales is not None or rotations is not None)
            and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or "
                "precomputed 3D covariance!"
            )

        empty = means3D.new_empty((0,))
        if shs is None:
            shs = empty
        if colors_precomp is None:
            colors_precomp = empty
        if scales is None:
            scales = empty
        if rotations is None:
            rotations = empty
        if cov3D_precomp is None:
            cov3D_precomp = empty

        raster_settings = self.raster_settings
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            head_input,
            packed_head_weights,
            packed_head_biases,
            worker_groups,
            persistent_blocks,
        )
        (
            _num_rendered,
            color,
            depth,
            radii,
            _geom_buffer,
            _binning_buffer,
            _image_buffer,
            packed_head_outputs,
        ) = _C.rasterize_gaussians_with_packed_heads(*args)
        return color, radii, depth, packed_head_outputs

    def forward_with_whole_heads(
        self,
        means3D,
        means2D,
        opacities,
        head_inputs,
        first_weights,
        first_biases,
        tail_weights,
        tail_biases,
        output_widths,
        worker_groups=1,
        persistent_blocks=0,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        """Fuse Raster with one to five complete C4 deformation heads."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer.forward_with_whole_heads is "
                "inference-only; call it under torch.no_grad()"
            )
        if not hasattr(_C, "rasterize_gaussians_with_whole_heads"):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide the Tacker whole-head mixed ABI v4; rebuild the extension"
            )
        sequences = (
            ("head_inputs", head_inputs),
            ("first_weights", first_weights),
            ("first_biases", first_biases),
            ("tail_weights", tail_weights),
            ("tail_biases", tail_biases),
            ("output_widths", output_widths),
        )
        for name, values in sequences:
            if not isinstance(values, (list, tuple)):
                raise TypeError(f"{name} must be a list or tuple")
        (
            head_inputs,
            first_weights,
            first_biases,
            tail_weights,
            tail_biases,
            output_widths,
        ) = tuple(tuple(values) for _name, values in sequences)
        task_count = len(head_inputs)
        if task_count < 1 or task_count > 5:
            raise ValueError("whole-head task count must be in [1, 5]")
        if any(
            len(values) != task_count
            for values in (
                first_weights,
                first_biases,
                tail_weights,
                tail_biases,
                output_widths,
            )
        ):
            raise ValueError(
                "whole-head inputs, parameters, and output_widths must have "
                "equal lengths"
            )
        for index, width in enumerate(output_widths):
            if not isinstance(width, int) or isinstance(width, bool):
                raise TypeError(f"output_widths[{index}] must be an int")
            if width < 1 or width > 128:
                raise ValueError(f"output_widths[{index}] must be in [1, 128]")
            if not isinstance(tail_weights[index], torch.Tensor) or not isinstance(
                tail_biases[index], torch.Tensor
            ):
                raise TypeError("tail_weights and tail_biases must contain tensors")
            if tail_weights[index].dim() != 2 or tail_weights[index].size(0) != width:
                raise ValueError(
                    f"output_widths[{index}] must match tail_weights[{index}]"
                )
            if tail_biases[index].dim() != 1 or tail_biases[index].size(0) != width:
                raise ValueError(
                    f"output_widths[{index}] must match tail_biases[{index}]"
                )
        for name, value in (
            ("worker_groups", worker_groups),
            ("persistent_blocks", persistent_blocks),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
        if worker_groups < 1 or worker_groups > task_count:
            raise ValueError("worker_groups must be in [1, task_count]")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either SHs or precomputed colors!"
            )
        if (
            (scales is None or rotations is None) and cov3D_precomp is None
        ) or (
            (scales is not None or rotations is not None)
            and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or "
                "precomputed 3D covariance!"
            )

        empty = means3D.new_empty((0,))
        if shs is None:
            shs = empty
        if colors_precomp is None:
            colors_precomp = empty
        if scales is None:
            scales = empty
        if rotations is None:
            rotations = empty
        if cov3D_precomp is None:
            cov3D_precomp = empty

        raster_settings = self.raster_settings
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            head_inputs,
            first_weights,
            first_biases,
            tail_weights,
            tail_biases,
            worker_groups,
            persistent_blocks,
        )
        (
            _num_rendered,
            color,
            depth,
            radii,
            _geom_buffer,
            _binning_buffer,
            _image_buffer,
            head_outputs,
        ) = _C.rasterize_gaussians_with_whole_heads(*args)
        return color, radii, depth, tuple(head_outputs)

    def forward_with_deform_mlp(
        self,
        means3D,
        means2D,
        opacities,
        deform_features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        tail_weights,
        tail_biases,
        persistent_blocks=0,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        """Directly fuse Raster with the complete five-head deformation MLP.

        ``persistent_blocks=0`` deliberately selects the direct logical grid
        for ABI v5.  Positive values opt into grid-stride persistence.
        """

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer.forward_with_deform_mlp is "
                "inference-only; call it under torch.no_grad()"
            )
        if not hasattr(_C, "rasterize_gaussians_with_deform_mlp"):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide the Tacker full-deform mixed ABI v5; rebuild the extension"
            )
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either SHs or precomputed colors!"
            )
        if (
            (scales is None or rotations is None) and cov3D_precomp is None
        ) or (
            (scales is not None or rotations is not None)
            and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or "
                "precomputed 3D covariance!"
            )

        empty = means3D.new_empty((0,))
        if shs is None:
            shs = empty
        if colors_precomp is None:
            colors_precomp = empty
        if scales is None:
            scales = empty
        if rotations is None:
            rotations = empty
        if cov3D_precomp is None:
            cov3D_precomp = empty

        raster_settings = self.raster_settings
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            deform_features,
            backbone_weight,
            backbone_bias,
            first_weights,
            first_biases,
            tail_weights,
            tail_biases,
            persistent_blocks,
        )
        (
            num_rendered,
            color,
            depth,
            radii,
            _geom_buffer,
            _binning_buffer,
            _image_buffer,
            deform_outputs,
        ) = _C.rasterize_gaussians_with_deform_mlp(*args)
        _record_raster_tile_loads(
            _image_buffer, num_rendered, raster_settings
        )
        return color, radii, depth, tuple(deform_outputs)

    def prepare_for_deform_mlp_overlap(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        handoff=None,
    ):
        """Prepare Raster state while leaving the final render leaf deferred."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer.prepare_for_deform_mlp_overlap is "
                "inference-only; call it under torch.no_grad()"
            )
        if not hasattr(_C, "prepare_gaussians_for_deform_mlp_overlap"):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide prepared Raster overlap; rebuild the extension"
            )
        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either SHs or precomputed colors!"
            )
        if (
            (scales is None or rotations is None) and cov3D_precomp is None
        ) or (
            (scales is not None or rotations is not None)
            and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or "
                "precomputed 3D covariance!"
            )

        empty = means3D.new_empty((0,))
        if shs is None:
            shs = empty
        if colors_precomp is None:
            colors_precomp = empty
        if scales is None:
            scales = empty
        if rotations is None:
            rotations = empty
        if cov3D_precomp is None:
            cov3D_precomp = empty
        colors_precomp = colors_precomp.contiguous()

        raster_settings = self.raster_settings
        prepare = _C.prepare_gaussians_for_deform_mlp_overlap
        extra = ()
        if handoff is not None:
            if not hasattr(_C, "prepare_gaussians_for_deform_mlp_overlap_handoff"):
                raise RuntimeError("installed Raster extension lacks prepare handoff; rebuild it")
            if len(handoff) != 3 or any(type(value) is not int or value <= 0 for value in handoff):
                raise ValueError("handoff requires positive stream, deform-event, duplicate-event handles")
            prepare = _C.prepare_gaussians_for_deform_mlp_overlap_handoff
            extra = tuple(handoff)
        prepared = prepare(
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            *extra,
        )
        return TackerPreparedRaster(*prepared, colors_precomp)

    def render_prepared_with_deform_mlp_overlap(
        self,
        prepared,
        deform_features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        tail_weights,
        tail_biases,
        persistent_blocks=0,
    ):
        """Launch the unchanged ABI-v5 mixed leaf from prepared Raster state."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer.render_prepared_with_deform_mlp_overlap "
                "is inference-only; call it under torch.no_grad()"
            )
        if not hasattr(
            _C, "render_prepared_gaussians_with_deform_mlp_overlap"
        ):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide the prepared ABI-v5 leaf; rebuild the extension"
            )
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")
        if not isinstance(prepared, TackerPreparedRaster):
            raise TypeError("prepared must be a TackerPreparedRaster")

        raster_settings = self.raster_settings
        deform_outputs = (
            _C.render_prepared_gaussians_with_deform_mlp_overlap(
                raster_settings.bg,
                prepared.colors_precomp,
                prepared.color,
                prepared.depth,
                prepared.radii,
                prepared.geom_buffer,
                prepared.binning_buffer,
                prepared.image_buffer,
                prepared.num_rendered,
                raster_settings.image_height,
                raster_settings.image_width,
                raster_settings.debug,
                deform_features,
                backbone_weight,
                backbone_bias,
                first_weights,
                first_biases,
                tail_weights,
                tail_biases,
                persistent_blocks,
            )
        )
        _record_raster_tile_loads(
            prepared.image_buffer, prepared.num_rendered, raster_settings
        )
        return (
            prepared.color,
            prepared.radii,
            prepared.depth,
            tuple(deform_outputs),
        )

    def render_prepared_with_deform_mlp_tensor_tail_overlap(
        self,
        prepared,
        deform_features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        packed_tail_weights,
        tail_biases,
        persistent_blocks=0,
    ):
        """Launch experimental padded Tensor-tail mixed ABI-v6.

        This method is deliberately separate from the qualified ABI-v5 path.
        ``packed_tail_weights`` must be cached FP16 storage with shape
        ``[112, 128]``; packing is not part of the mixed kernel.
        """

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer."
                "render_prepared_with_deform_mlp_tensor_tail_overlap is "
                "inference-only; call it under torch.no_grad()"
            )
        if not hasattr(
            _C,
            "render_prepared_gaussians_with_deform_mlp_tensor_tail_overlap",
        ):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide experimental mixed ABI-v6; rebuild the extension"
            )
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")
        if not isinstance(prepared, TackerPreparedRaster):
            raise TypeError("prepared must be a TackerPreparedRaster")

        raster_settings = self.raster_settings
        deform_outputs = (
            _C.render_prepared_gaussians_with_deform_mlp_tensor_tail_overlap(
                raster_settings.bg,
                prepared.colors_precomp,
                prepared.color,
                prepared.depth,
                prepared.radii,
                prepared.geom_buffer,
                prepared.binning_buffer,
                prepared.image_buffer,
                prepared.num_rendered,
                raster_settings.image_height,
                raster_settings.image_width,
                raster_settings.debug,
                deform_features,
                backbone_weight,
                backbone_bias,
                first_weights,
                first_biases,
                packed_tail_weights,
                tail_biases,
                persistent_blocks,
            )
        )
        _record_raster_tile_loads(
            prepared.image_buffer, prepared.num_rendered, raster_settings
        )
        return (
            prepared.color,
            prepared.radii,
            prepared.depth,
            tuple(deform_outputs),
        )

    def render_prepared_with_deform_mlp_tensor_tail_low_register_overlap(
        self,
        prepared,
        deform_features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        packed_tail_weights,
        tail_biases,
        persistent_blocks=0,
    ):
        """Launch experimental low-register padded Tensor-tail mixed ABI-v7."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer."
                "render_prepared_with_deform_mlp_tensor_tail_low_register_overlap "
                "is inference-only; call it under torch.no_grad()"
            )
        binding_name = (
            "render_prepared_gaussians_with_deform_mlp_tensor_tail_"
            "low_register_overlap"
        )
        if not hasattr(_C, binding_name):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide experimental mixed ABI-v7; rebuild the extension"
            )
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")
        if not isinstance(prepared, TackerPreparedRaster):
            raise TypeError("prepared must be a TackerPreparedRaster")

        raster_settings = self.raster_settings
        deform_outputs = getattr(_C, binding_name)(
            raster_settings.bg,
            prepared.colors_precomp,
            prepared.color,
            prepared.depth,
            prepared.radii,
            prepared.geom_buffer,
            prepared.binning_buffer,
            prepared.image_buffer,
            prepared.num_rendered,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug,
            deform_features,
            backbone_weight,
            backbone_bias,
            first_weights,
            first_biases,
            packed_tail_weights,
            tail_biases,
            persistent_blocks,
        )
        _record_raster_tile_loads(
            prepared.image_buffer, prepared.num_rendered, raster_settings
        )
        return (
            prepared.color,
            prepared.radii,
            prepared.depth,
            tuple(deform_outputs),
        )

    def render_prepared_with_deform_mlp_tensor_tail_four_cta_overlap(
        self,
        prepared,
        deform_features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        packed_tail_weights,
        tail_biases,
        persistent_blocks=0,
    ):
        """Launch experimental four-CTA padded Tensor-tail mixed ABI-v8."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer."
                "render_prepared_with_deform_mlp_tensor_tail_four_cta_overlap "
                "is inference-only; call it under torch.no_grad()"
            )
        binding_name = (
            "render_prepared_gaussians_with_deform_mlp_tensor_tail_"
            "four_cta_overlap"
        )
        if not hasattr(_C, binding_name):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide experimental mixed ABI-v8; rebuild the extension"
            )
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")
        if not isinstance(prepared, TackerPreparedRaster):
            raise TypeError("prepared must be a TackerPreparedRaster")

        raster_settings = self.raster_settings
        deform_outputs = getattr(_C, binding_name)(
            raster_settings.bg,
            prepared.colors_precomp,
            prepared.color,
            prepared.depth,
            prepared.radii,
            prepared.geom_buffer,
            prepared.binning_buffer,
            prepared.image_buffer,
            prepared.num_rendered,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug,
            deform_features,
            backbone_weight,
            backbone_bias,
            first_weights,
            first_biases,
            packed_tail_weights,
            tail_biases,
            persistent_blocks,
        )
        _record_raster_tile_loads(
            prepared.image_buffer, prepared.num_rendered, raster_settings
        )
        return (
            prepared.color,
            prepared.radii,
            prepared.depth,
            tuple(deform_outputs),
        )

    def render_prepared_with_deform_mlp_tensor_tail_balanced_overlap(
        self,
        prepared,
        deform_features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        packed_tail_weights,
        tail_biases,
        persistent_blocks=0,
    ):
        """Launch balanced-work Tensor-tail mixed ABI-v9."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer."
                "render_prepared_with_deform_mlp_tensor_tail_balanced_overlap "
                "is inference-only; call it under torch.no_grad()"
            )
        binding_name = (
            "render_prepared_gaussians_with_deform_mlp_tensor_tail_"
            "balanced_overlap"
        )
        if not hasattr(_C, binding_name):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide experimental mixed ABI-v9; rebuild the extension"
            )
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")
        if not isinstance(prepared, TackerPreparedRaster):
            raise TypeError("prepared must be a TackerPreparedRaster")

        raster_settings = self.raster_settings
        deform_outputs = getattr(_C, binding_name)(
            raster_settings.bg,
            prepared.colors_precomp,
            prepared.color,
            prepared.depth,
            prepared.radii,
            prepared.geom_buffer,
            prepared.binning_buffer,
            prepared.image_buffer,
            prepared.num_rendered,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug,
            deform_features,
            backbone_weight,
            backbone_bias,
            first_weights,
            first_biases,
            packed_tail_weights,
            tail_biases,
            persistent_blocks,
        )
        _record_raster_tile_loads(
            prepared.image_buffer, prepared.num_rendered, raster_settings
        )
        return (
            prepared.color,
            prepared.radii,
            prepared.depth,
            tuple(deform_outputs),
        )

    def render_prepared_with_deform_mlp_tensor_tail_raster_optimized_overlap(
        self,
        prepared,
        deform_features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        packed_tail_weights,
        tail_biases,
        persistent_blocks=0,
    ):
        """Launch exact Raster-optimized Tensor-tail mixed ABI-v10."""

        return self._render_prepared_with_deform_mlp_tensor_tail_raster_variant(
            prepared=prepared,
            deform_features=deform_features,
            backbone_weight=backbone_weight,
            backbone_bias=backbone_bias,
            first_weights=first_weights,
            first_biases=first_biases,
            packed_tail_weights=packed_tail_weights,
            tail_biases=tail_biases,
            persistent_blocks=persistent_blocks,
            binding_suffix="raster_optimized_overlap",
            abi_name="ABI-v10",
        )

    def render_prepared_with_deform_mlp_tensor_tail_raster_fast_overlap(
        self,
        prepared,
        deform_features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        packed_tail_weights,
        tail_biases,
        persistent_blocks=0,
    ):
        """Launch approximate fast-Raster Tensor-tail mixed ABI-v11."""

        return self._render_prepared_with_deform_mlp_tensor_tail_raster_variant(
            prepared=prepared,
            deform_features=deform_features,
            backbone_weight=backbone_weight,
            backbone_bias=backbone_bias,
            first_weights=first_weights,
            first_biases=first_biases,
            packed_tail_weights=packed_tail_weights,
            tail_biases=tail_biases,
            persistent_blocks=persistent_blocks,
            binding_suffix="raster_fast_overlap",
            abi_name="ABI-v11",
        )

    def render_prepared_raster_fast_range(
        self,
        prepared,
        raster_start,
        raster_end,
    ):
        """Launch the ABI-v12 fast-Raster-only half-open tile range."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer experimental ABI-v12 Raster range is "
                "inference-only; call it under torch.no_grad()"
            )
        binding_name = "render_prepared_gaussians_raster_fast_range"
        if not hasattr(_C, binding_name):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide experimental split Raster ABI-v12; rebuild the "
                "extension"
            )
        if not isinstance(prepared, TackerPreparedRaster):
            raise TypeError("prepared must be a TackerPreparedRaster")
        for name, value in (
            ("raster_start", raster_start),
            ("raster_end", raster_end),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
        raster_settings = self.raster_settings
        raster_blocks = (
            (int(raster_settings.image_width) + 15) // 16
        ) * ((int(raster_settings.image_height) + 15) // 16)
        if not 0 <= raster_start <= raster_end <= raster_blocks:
            raise ValueError(
                "Raster range must satisfy 0 <= start <= end <= "
                f"{raster_blocks}"
            )
        getattr(_C, binding_name)(
            raster_settings.bg,
            prepared.colors_precomp,
            prepared.color,
            prepared.depth,
            prepared.radii,
            prepared.geom_buffer,
            prepared.binning_buffer,
            prepared.image_buffer,
            prepared.num_rendered,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug,
            raster_start,
            raster_end,
        )

    def render_prepared_with_hexplane_sampling(
        self,
        prepared,
        coords,
        planes,
        axes,
        *,
        hex_tasks_per_cta=17,
        raster_start=0,
        raster_end=None,
        outputs=None,
        block_fusion=False,
    ):
        """Spatial fusion: one 512-thread CTA = 256 Raster + 256 HexPlane.

        coords: already-normalized contiguous FP32 [N,4]. Each plane is
        [1,16,H,W]; axes follow grid_sample's (x,y) order. A Hex logical task
        samples 256 points from ONE plane, not all planes. Its subgroup runs
        hex_tasks_per_cta consecutive tasks, independently of the Raster half.
        The default 17 is an untuned candidate, NOT a measured optimum.

        Only [raster_start,raster_end) is rendered; callers must separately
        render any remaining tiles. Grid=max(range length,ceil(Hex tasks/q));
        extra CTAs have an idle half. Outputs are [N,16] views of [16,N] storage
        and may be supplied preallocated for Graph replay. The call does not
        normalize coordinates, multiply/concatenate planes, or run the MLP.
        Consumers on another CUDA stream must wait for this launch to finish.
        """
        if torch.is_grad_enabled():
            raise RuntimeError("Raster/HexPlane mixed kernel is inference-only; use no_grad()")
        if not isinstance(prepared, TackerPreparedRaster):
            raise TypeError("prepared must be a TackerPreparedRaster")
        if type(block_fusion) is not bool:
            raise TypeError("block_fusion must be a bool")
        if block_fusion and hex_tasks_per_cta != 1:
            raise ValueError("block fusion requires one sampling task per CTA")
        binding_name = ("render_prepared_gaussians_with_hexplane_blocks" if block_fusion
                        else "render_prepared_gaussians_with_hexplane_sampling")
        if not hasattr(_C, binding_name):
            raise RuntimeError("rebuild diff_gaussian_rasterization for the Raster/HexPlane mixed leaf")
        for name, value in (("hex_tasks_per_cta", hex_tasks_per_cta),
                            ("raster_start", raster_start)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
        if not 0 < hex_tasks_per_cta <= 2**31 - 1:
            raise ValueError("hex_tasks_per_cta must be positive and int32-safe")
        settings = self.raster_settings
        tile_count = ((int(settings.image_width) + 15) // 16) * (
            (int(settings.image_height) + 15) // 16)
        if raster_end is None:
            raster_end = tile_count
        if not isinstance(raster_end, int) or isinstance(raster_end, bool):
            raise TypeError("raster_end must be an int")
        if not 0 <= raster_start <= raster_end <= tile_count:
            raise ValueError(f"Raster range must satisfy 0 <= start <= end <= {tile_count}")
        planes, axes = tuple(planes), tuple(axes)
        if not 1 <= len(planes) <= 12 or len(axes) != len(planes):
            raise ValueError("provide 1..12 planes and one axis pair per plane")
        if outputs is None:
            outputs = tuple(torch.empty((16, coords.shape[0]),
                                        device=coords.device, dtype=coords.dtype).t()
                            for _ in planes)
        else:
            outputs = tuple(outputs)
        if len(outputs) != len(planes):
            raise ValueError("provide one preallocated output per plane")
        getattr(_C, binding_name)(
            settings.bg, prepared.colors_precomp, prepared.color, prepared.depth,
            prepared.radii, prepared.geom_buffer, prepared.binning_buffer,
            prepared.image_buffer, prepared.num_rendered,
            settings.image_height, settings.image_width,
            raster_start, raster_end, coords, planes, axes, outputs, hex_tasks_per_cta,
        )
        return outputs

    def render_prepared_with_hexplane_blocks(
        self, prepared, coords, planes, axes, *, raster_start=0,
        raster_end=None, outputs=None,
    ):
        """C: one launch; whole 256-thread blocks own Hex or Raster tasks.

        Grid = planes*ceil(N/256) + Raster range length. Task domains are
        interleaved; each Raster task receives its own logical tile ID.
        Both inputs must be ready. Consumers wait for the complete launch.
        """
        return self.render_prepared_with_hexplane_sampling(
            prepared, coords, planes, axes, hex_tasks_per_cta=1,
            raster_start=raster_start, raster_end=raster_end,
            outputs=outputs, block_fusion=True)

    @staticmethod
    def raster_hexplane_block_resources():
        keys = ("threads", "registers_per_thread", "local_bytes_per_thread",
                "static_shared_bytes", "active_blocks_per_sm", "binary_version")
        return dict(zip(keys, _C.tacker_raster_hexplane_block_resources()))

    @staticmethod
    def raster_hexplane_resources():
        """Actual compiler/device resources, not the Hex-only 512 probe."""
        keys = ("threads", "registers_per_thread", "local_bytes_per_thread",
                "static_shared_bytes", "active_blocks_per_sm", "binary_version")
        return dict(zip(keys, _C.tacker_raster_hexplane_resources()))

    def render_prepared_with_deform_mlp_tensor_tail_split_raster_overlap(
        self,
        prepared,
        deform_features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        packed_tail_weights,
        tail_biases,
        raster_start,
        persistent_blocks=0,
    ):
        """Launch the ABI-v12 remaining Raster range + complete Deform."""

        return self._render_prepared_with_deform_mlp_tensor_tail_raster_variant(
            prepared=prepared,
            deform_features=deform_features,
            backbone_weight=backbone_weight,
            backbone_bias=backbone_bias,
            first_weights=first_weights,
            first_biases=first_biases,
            packed_tail_weights=packed_tail_weights,
            tail_biases=tail_biases,
            persistent_blocks=persistent_blocks,
            binding_suffix="split_raster_overlap",
            abi_name="ABI-v12",
            raster_start=raster_start,
        )

    def _render_prepared_with_deform_mlp_tensor_tail_raster_variant(
        self,
        *,
        prepared,
        deform_features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        packed_tail_weights,
        tail_biases,
        persistent_blocks,
        binding_suffix,
        abi_name,
        raster_start=None,
    ):
        if torch.is_grad_enabled():
            raise RuntimeError(
                f"GaussianRasterizer experimental {abi_name} Raster path is "
                "inference-only; call it under torch.no_grad()"
            )
        binding_name = (
            "render_prepared_gaussians_with_deform_mlp_tensor_tail_"
            + binding_suffix
        )
        if not hasattr(_C, binding_name):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                f"provide experimental mixed {abi_name}; rebuild the extension"
            )
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")
        if not isinstance(prepared, TackerPreparedRaster):
            raise TypeError("prepared must be a TackerPreparedRaster")

        raster_settings = self.raster_settings
        binding_arguments = [
            raster_settings.bg,
            prepared.colors_precomp,
            prepared.color,
            prepared.depth,
            prepared.radii,
            prepared.geom_buffer,
            prepared.binning_buffer,
            prepared.image_buffer,
            prepared.num_rendered,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug,
            deform_features,
            backbone_weight,
            backbone_bias,
            first_weights,
            first_biases,
            packed_tail_weights,
            tail_biases,
            persistent_blocks,
        ]
        if raster_start is not None:
            if not isinstance(raster_start, int) or isinstance(
                raster_start, bool
            ):
                raise TypeError("raster_start must be an int")
            raster_blocks = (
                (int(raster_settings.image_width) + 15) // 16
            ) * ((int(raster_settings.image_height) + 15) // 16)
            if not 0 <= raster_start <= raster_blocks:
                raise ValueError(
                    f"raster_start must be in [0, {raster_blocks}]"
                )
            binding_arguments.append(raster_start)
        deform_outputs = getattr(_C, binding_name)(*binding_arguments)
        _record_raster_tile_loads(
            prepared.image_buffer, prepared.num_rendered, raster_settings
        )
        return (
            prepared.color,
            prepared.radii,
            prepared.depth,
            tuple(deform_outputs),
        )

    def profile_prepared_deform_mlp_tensor_tail_four_cta_overlap(
        self,
        prepared,
        deform_features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        packed_tail_weights,
        tail_biases,
        persistent_blocks=0,
        mode="both",
    ):
        """Measure ABI-v8 Raster/Deform subgroup lifetimes per physical CTA.

        ``mode`` is ``"both"``, ``"raster_only"``, or ``"deform_only"``.
        The returned timing tensor has int64 columns
        ``[raster_start, raster_end, deform_start, deform_end]`` in SM cycles.
        Disabled subgroup columns are zero.  This profiling path is separate
        from the qualified production ABI-v8 kernel.
        """

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer."
                "profile_prepared_deform_mlp_tensor_tail_four_cta_overlap "
                "is inference-only; call it under torch.no_grad()"
            )
        binding_name = (
            "render_prepared_gaussians_with_deform_mlp_tensor_tail_"
            "four_cta_diagnostic"
        )
        if not hasattr(_C, binding_name):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide ABI-v8 subgroup diagnostics; rebuild the extension"
            )
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")
        if not isinstance(prepared, TackerPreparedRaster):
            raise TypeError("prepared must be a TackerPreparedRaster")
        mode_ids = {"both": 0, "raster_only": 1, "deform_only": 2}
        if mode not in mode_ids:
            raise ValueError(
                "mode must be 'both', 'raster_only', or 'deform_only'"
            )

        raster_settings = self.raster_settings
        results = getattr(_C, binding_name)(
            raster_settings.bg,
            prepared.colors_precomp,
            prepared.color,
            prepared.depth,
            prepared.radii,
            prepared.geom_buffer,
            prepared.binning_buffer,
            prepared.image_buffer,
            prepared.num_rendered,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug,
            deform_features,
            backbone_weight,
            backbone_bias,
            first_weights,
            first_biases,
            packed_tail_weights,
            tail_biases,
            persistent_blocks,
            mode_ids[mode],
        )
        deform_outputs = tuple(results[:-1])
        timing_cycles = results[-1]
        if mode != "deform_only":
            _record_raster_tile_loads(
                prepared.image_buffer, prepared.num_rendered, raster_settings
            )
        return (
            prepared.color,
            prepared.radii,
            prepared.depth,
            deform_outputs,
            timing_cycles,
        )

    def profile_prepared_deform_mlp_tensor_tail_raster_overlap(
        self,
        prepared,
        deform_features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        packed_tail_weights,
        tail_biases,
        persistent_blocks=0,
        abi_version=10,
        mode="both",
    ):
        """Measure ABI-v10/v11 balanced Raster/Deform subgroup lifetimes.

        All modes retain the production physical grid and balanced one-or-two
        Deform-tile mapping.  ``abi_version`` is 10 or 11; ``mode`` is
        ``"both"``, ``"raster_only"``, or ``"deform_only"``.
        """

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer."
                "profile_prepared_deform_mlp_tensor_tail_raster_overlap "
                "is inference-only; call it under torch.no_grad()"
            )
        binding_name = (
            "render_prepared_gaussians_with_deform_mlp_tensor_tail_"
            "raster_diagnostic"
        )
        if not hasattr(_C, binding_name):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide ABI-v10/v11 subgroup diagnostics; rebuild the extension"
            )
        if abi_version not in (10, 11):
            raise ValueError("abi_version must be 10 or 11")
        if not isinstance(persistent_blocks, int) or isinstance(
            persistent_blocks, bool
        ):
            raise TypeError("persistent_blocks must be an int")
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be >= 0")
        if not isinstance(prepared, TackerPreparedRaster):
            raise TypeError("prepared must be a TackerPreparedRaster")
        mode_ids = {"both": 0, "raster_only": 1, "deform_only": 2}
        if mode not in mode_ids:
            raise ValueError(
                "mode must be 'both', 'raster_only', or 'deform_only'"
            )

        raster_settings = self.raster_settings
        results = getattr(_C, binding_name)(
            raster_settings.bg,
            prepared.colors_precomp,
            prepared.color,
            prepared.depth,
            prepared.radii,
            prepared.geom_buffer,
            prepared.binning_buffer,
            prepared.image_buffer,
            prepared.num_rendered,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug,
            deform_features,
            backbone_weight,
            backbone_bias,
            first_weights,
            first_biases,
            packed_tail_weights,
            tail_biases,
            persistent_blocks,
            abi_version,
            mode_ids[mode],
        )
        deform_outputs = tuple(results[:-1])
        timing_cycles = results[-1]
        if mode != "deform_only":
            _record_raster_tile_loads(
                prepared.image_buffer, prepared.num_rendered, raster_settings
            )
        return (
            prepared.color,
            prepared.radii,
            prepared.depth,
            deform_outputs,
            timing_cycles,
        )

    def render_prepared_overlap(self, prepared):
        """Launch the ordinary Raster leaf for the final drained frame."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "GaussianRasterizer.render_prepared_overlap is inference-only; "
                "call it under torch.no_grad()"
            )
        if not hasattr(_C, "render_prepared_gaussians_overlap"):
            raise RuntimeError(
                "installed diff_gaussian_rasterization extension does not "
                "provide the prepared Raster leaf; rebuild the extension"
            )
        if not isinstance(prepared, TackerPreparedRaster):
            raise TypeError("prepared must be a TackerPreparedRaster")

        raster_settings = self.raster_settings
        _C.render_prepared_gaussians_overlap(
            raster_settings.bg,
            prepared.colors_precomp,
            prepared.color,
            prepared.depth,
            prepared.radii,
            prepared.geom_buffer,
            prepared.binning_buffer,
            prepared.image_buffer,
            prepared.num_rendered,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug,
        )
        _record_raster_tile_loads(
            prepared.image_buffer, prepared.num_rendered, raster_settings
        )
        return prepared.color, prepared.radii, prepared.depth

    def forward_with_whole_head(
        self,
        means3D,
        means2D,
        opacities,
        head_input,
        first_weight,
        first_bias,
        tail_weight,
        tail_bias,
        persistent_blocks=0,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        """Strict one-task convenience method for the C4 physical backend."""

        result = self.forward_with_whole_heads(
            means3D=means3D,
            means2D=means2D,
            opacities=opacities,
            head_inputs=(head_input,),
            first_weights=(first_weight,),
            first_biases=(first_bias,),
            tail_weights=(tail_weight,),
            tail_biases=(tail_bias,),
            output_widths=(tail_weight.size(0),),
            worker_groups=1,
            persistent_blocks=persistent_blocks,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        color, radii, depth, head_outputs = result
        return color, radii, depth, head_outputs[0]
