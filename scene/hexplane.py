import itertools
import logging as log
import os
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            capture_safe: bool = False,
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    reduction_threads = None
    # Limit this experiment to the qualified Graph query. Serial rendering
    # and the eager prefill remain an independent, unfused reference.
    if capture_safe and os.environ.get("TACKER_HEXPLANE_FUSED_REDUCTION", "0") != "0":
        if torch.is_grad_enabled():
            raise RuntimeError("TACKER_HEXPLANE_FUSED_REDUCTION is inference-only")
        from tacker_4dgs_head.hexplane import fused_reduction_threads, reduce_features
        reduction_threads = fused_reduction_threads()
        if (grid_dimensions != 2 or pts.ndim != 2 or pts.shape[1] != 4
                or not concat_features or num_levels != 2
                or any(len(level) != 6 or any(plane.shape[1] != 16 for plane in level)
                       for level in ms_grids[:num_levels])):
            raise ValueError("fused HexPlane reduction requires two levels of six 16-channel 2-D planes")
    reduction_samples = [] if reduction_threads is not None else None
    lowreg = os.environ.get("TACKER_HEXPLANE_LOWREG") == "1" and not torch.is_grad_enabled()
    if lowreg:
        if grid_dimensions != 2 or pts.ndim != 2 or pts.shape[1] != 4:
            raise ValueError("TACKER_HEXPLANE_LOWREG requires 2-D planes and [N,4] coords")
        from tacker_4dgs_head.hexplane import sample_plane
        pts = pts.contiguous()
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            if lowreg:
                interp_out_plane = sample_plane(pts, grid[ci], *coo_comb)
                if reduction_samples is not None:
                    reduction_samples.append(interp_out_plane)
                else:
                    interp_space = interp_space * interp_out_plane
                continue
            if capture_safe:
                # ``pts[..., coo_comb]`` constructs an implicit CUDA index
                # tensor and is not graph-capture safe in PyTorch 2.5.  Scalar
                # integer views plus stack preserve the coordinate order and
                # numerical grid_sample contract without that hidden copy for
                # either 2-D or 3-D grids.
                plane_coords = torch.stack(
                    tuple(pts[..., axis] for axis in coo_comb),
                    dim=-1,
                )
            else:
                plane_coords = pts[..., coo_comb]
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], plane_coords)
                .view(-1, feature_dim)
            )
            # compute product over planes
            if reduction_samples is not None:
                reduction_samples.append(interp_out_plane)
            else:
                interp_space = interp_space * interp_out_plane

        # combine over scales
        if reduction_samples is not None:
            continue
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if reduction_samples is not None:
        return reduce_features(reduction_samples, threads=reduction_threads)
    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class HexPlaneField(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(
        self,
        pts: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        capture_safe: bool = False,
    ):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None,
            capture_safe=capture_safe)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)


        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features

    def forward_cuda_graph(
        self,
        pts: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ):
        """Capture-safe inference equivalent of :meth:`forward`."""

        return self.get_density(pts, timestamps, capture_safe=True)
