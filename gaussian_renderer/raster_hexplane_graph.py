"""Opt-in split Graph prefix for Raster + HexPlane sampling experiments.

Prepared Raster buffers change address between frames, so the physical mixed
leaf is submitted normally, between captured coordinate and reduction graphs.
It is NOT described as a replay of a graph containing the entire mixed leaf.
Unpaired planes may be sampled early on the deform stream. The separate
control uses the same plane split and dependency boundaries; this distinguishes
spatial fusion from changing sampling math or the early submission window.

batched (B) starts one all-plane sampler after coordinates and preserves the
early prefix window. block_fused (C) waits for Raster prepare and coordinates,
then launches one 256-thread/block kernel over both complete task domains.
The stage A/B/C benchmark captures the entire combined leaf in a Graph;
this renderer adapter keeps the existing prepared-buffer submission boundary.
"""

import itertools

import torch


def sampling_plane_split(mixed_planes):
    """Disjoint plane ownership; each plane contains ceil(N/256) tasks."""
    if type(mixed_planes) is not int or not 1 <= mixed_planes <= 12:
        raise ValueError("mixed_planes must be an integer in [1,12]")
    boundary = 12 - mixed_planes
    return tuple(range(boundary)), tuple(range(boundary, 12))


class RasterHexPlaneGraphSlot:
    deferred_sampling = True

    def __init__(self, network, xyz, timestamp, stream, *, mode, tasks_per_cta,
                 mixed_planes=12):
        if mode not in ("fused", "separate", "batched", "block_fused"):
            raise ValueError("unknown Raster/HexPlane experiment mode")
        if mode in ("batched", "block_fused") and (mixed_planes != 12 or tasks_per_cta != 1):
            raise ValueError("B/C require all 12 planes and one sampling task per CTA")
        if torch.is_grad_enabled():
            raise RuntimeError("Raster/HexPlane experiment is inference-only")
        self.mode = mode
        self.tasks_per_cta = tasks_per_cta
        self.early_plane_ids, self.mixed_plane_ids = sampling_plane_split(mixed_planes)
        self._device = xyz.device
        self._shape = tuple(xyz.shape)
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] <= 0
                or tuple(timestamp.shape) != (xyz.shape[0], 1)):
            raise ValueError("Raster/HexPlane prefix requires nonempty [N,3] xyz and [N,1] time")
        if (xyz.dtype != torch.float32 or timestamp.dtype != torch.float32
                or not xyz.is_cuda or timestamp.device != xyz.device):
            raise TypeError("Raster/HexPlane prefix requires same-device CUDA FP32 inputs")
        grid = network.grid
        if (grid.grid_config[0]["grid_dimensions"] != 2
                or not grid.concat_features or len(grid.grids) != 2
                or any(len(level) != 6 for level in grid.grids)):
            raise ValueError("Raster/HexPlane experiment requires two levels of six 2-D planes")
        self.planes = tuple(plane for level in grid.grids for plane in level)
        if any(plane.dtype != torch.float32 or plane.device != xyz.device
               or not plane.is_contiguous() or plane.ndim != 4
               or tuple(plane.shape[:2]) != (1, 16) for plane in self.planes):
            raise TypeError("Raster/HexPlane experiment requires contiguous CUDA FP32 [1,16,H,W] planes")
        self.axes = tuple(itertools.combinations(range(4), 2)) * 2
        self.xyz = torch.empty_like(xyz)
        self.timestamp = torch.empty_like(timestamp)
        self.samples = tuple(torch.zeros(16, xyz.shape[0], device=xyz.device).t()
                             for _ in self.planes)
        self.coords_ready = torch.cuda.Event(blocking=False)
        self.sampling_done = torch.cuda.Event(blocking=False)
        self.prepare_ready = torch.cuda.Event(blocking=False)
        self.hex_ctas = (mixed_planes * ((xyz.shape[0] + 255) // 256) + tasks_per_cta - 1) // tasks_per_cta

        # Import once outside capture; the standalone control uses the same
        # device sampler that is inlined into the mixed kernel's second half.
        from tacker_4dgs_head.hexplane import (
            sample_plane_out, sample_planes, fused_reduction_threads, reduce_features)
        self.reduction_threads = fused_reduction_threads()
        from scene.hexplane import normalize_aabb

        def coordinates():
            return torch.cat((normalize_aabb(self.xyz, grid.aabb), self.timestamp), dim=-1).contiguous()

        def reduction():
            if self.reduction_threads is not None:
                return reduce_features(self.samples, threads=self.reduction_threads)
            levels = []
            for begin in (0, 6):
                product = 1.0
                for sample in self.samples[begin:begin + 6]:
                    product = product * sample
                levels.append(product)
            return torch.cat(levels, dim=-1).contiguous()

        with torch.cuda.stream(stream):
            self.xyz.copy_(xyz, non_blocking=True)
            self.timestamp.copy_(timestamp, non_blocking=True)
        torch.cuda.synchronize(xyz.device)
        with torch.cuda.stream(stream):
            for _ in range(3):
                coordinates()
                reduction()
        stream.synchronize()
        self.coordinate_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.coordinate_graph, stream=stream, capture_error_mode="thread_local"):
            self.coords = coordinates()
        self.reduction_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.reduction_graph, stream=stream, capture_error_mode="thread_local"):
            self.output = reduction()

        def sample_selected_planes(plane_ids):
            if self.mode in ("batched", "block_fused"):
                sample_planes(self.coords, tuple(self.planes[i] for i in plane_ids),
                              tuple(self.axes[i] for i in plane_ids),
                              outputs=tuple(self.samples[i] for i in plane_ids))
                return
            for index in plane_ids:
                sample_plane_out(self.coords, self.planes[index],
                                 self.samples[index], *self.axes[index])

        # These writes have no Raster-buffer dependency and do not overlap
        # mixed-plane outputs. Queue them immediately after coordinates, not
        # after prepare_ready. Reduction remains on this same deform stream,
        # so it waits for BOTH early work and sampling_done without a GPU spin.
        self.early_sampling_graph = None
        if self.early_plane_ids:
            with torch.cuda.stream(stream):
                sample_selected_planes(self.early_plane_ids)
            stream.synchronize()
            self.early_sampling_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.early_sampling_graph, stream=stream, capture_error_mode="thread_local"):
                sample_selected_planes(self.early_plane_ids)
        self.sampling_graph = None
        if mode in ("separate", "batched"):
            with torch.cuda.stream(stream):
                sample_selected_planes(self.mixed_plane_ids)
            stream.synchronize()
            self.sampling_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.sampling_graph, stream=stream, capture_error_mode="thread_local"):
                sample_selected_planes(self.mixed_plane_ids)
        stream.synchronize()

    def enqueue(self, xyz, timestamp, stream):
        if (tuple(xyz.shape) != self._shape
                or tuple(timestamp.shape) != (self._shape[0], 1)
                or xyz.device != self._device or timestamp.device != self._device
                or xyz.dtype != torch.float32 or timestamp.dtype != torch.float32):
            raise ValueError("Raster/HexPlane prefix input contract changed")
        if torch.cuda.current_stream(self._device).cuda_stream != stream.cuda_stream:
            raise RuntimeError("Raster/HexPlane coordinate graph must replay on its deform stream")
        self.xyz.copy_(xyz, non_blocking=True)
        self.timestamp.copy_(timestamp, non_blocking=True)
        self.coordinate_graph.replay()
        self.coords_ready.record(stream)
        if self.early_sampling_graph is not None:
            self.early_sampling_graph.replay()
        if self.mode == "batched":
            # B retains the original early prefix window: no Raster-prepare wait.
            self.sampling_graph.replay()
            self.reduction_graph.replay()
        # Output becomes valid only after complete_sampling records prefix_ready.
        return self.output

    def complete_sampling(self, context, prepared, split_tile, raster_stream,
                          deform_stream, prefix_ready, prefix_end=None):
        if split_tile is None:
            raise RuntimeError("Raster/HexPlane experiment requires the v12 Raster split")
        if self.mode == "batched":
            with torch.cuda.stream(deform_stream):
                prefix_ready.record(deform_stream)
                if prefix_end is not None:
                    prefix_end.record(deform_stream)
            return 0  # Caller launches the complete independent Raster prepass.
        paired_tiles = 0
        if self.mode in ("fused", "block_fused"):
            paired_tiles = split_tile if self.mode == "block_fused" else min(split_tile, self.hex_ctas)
            raster_stream.wait_event(self.coords_ready)
            mixed_planes = tuple(self.planes[i] for i in self.mixed_plane_ids)
            mixed_samples = tuple(self.samples[i] for i in self.mixed_plane_ids)
            mixed_axes = tuple(self.axes[i] for i in self.mixed_plane_ids)
            for tensor in (self.coords,) + mixed_planes + mixed_samples:
                tensor.record_stream(raster_stream)
            if self.mode == "block_fused":
                context.rasterizer.render_prepared_with_hexplane_blocks(
                    prepared, self.coords, mixed_planes, mixed_axes,
                    raster_start=0, raster_end=paired_tiles, outputs=mixed_samples)
            else:
                context.rasterizer.render_prepared_with_hexplane_sampling(
                    prepared, self.coords, mixed_planes, mixed_axes,
                    hex_tasks_per_cta=self.tasks_per_cta,
                    raster_start=0, raster_end=paired_tiles, outputs=mixed_samples)
            self.sampling_done.record(raster_stream)
        else:
            # Only the mixed-owned planes wait for Raster prepare in this
            # matched control. Early planes use exactly the same schedule.
            self.prepare_ready.record(raster_stream)
        with torch.cuda.stream(deform_stream):
            if self.mode in ("fused", "block_fused"):
                deform_stream.wait_event(self.sampling_done)
            else:
                deform_stream.wait_event(self.prepare_ready)
                self.sampling_graph.replay()
            self.reduction_graph.replay()
            prefix_ready.record(deform_stream)
            if prefix_end is not None:
                prefix_end.record(deform_stream)
        return paired_tiles
