#!/usr/bin/env python3
"""Swap actual legacy and gsplat tile membership into one sealed RGBD kernel."""
import copy
import hashlib
import math
import os
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

RUNTIME = Path(os.environ['GSPLAT_REFERENCE_PROJECT'])
sys.path.insert(0, str(RUNTIME / 'scripts'))
from analyze_gsplat_mixed import load_workload, save, sha
from benchmark_gsplat_v7 import node_resources
from profile_gsplat_leaf_resources import binary

ARMS = ('gsplat_tiles', 'legacy_tiles')
TIMING_VIEWS = (0, 10, 20, 30, 40)


def legacy_membership(prepared, width, height, tiles):
    import torch

    # The sealed legacy BinningState stores sorted IDs first, then unsorted
    # IDs, then sorted uint64 keys. Each field starts on a 128-byte boundary.
    def third_field(buffer, size):
        assert buffer.data_ptr() % 128 == 0 and buffer.element_size() == 1
        return 2 * ((size + 127) // 128) * 128

    n = prepared.num_rendered
    buf = prepared.binning_buffer
    keys_start = third_field(buf, n * 4)
    ids = buf[:n * 4].view(torch.int32)
    keys = buf[keys_start:keys_start + n * 8].view(torch.int64)
    tile_ids = keys >> 32
    assert bool(((tile_ids >= 0) & (tile_ids < tiles)).all())
    counts = torch.bincount(tile_ids, minlength=tiles)
    # Check the independently stored ImageState ranges, including empty tiles.
    buf = prepared.image_buffer
    ranges_start = third_field(buf, width * height * 4)
    ranges = buf[ranges_start:ranges_start + tiles * 8].view(torch.int32).reshape(tiles, 2)
    offsets = counts.cumsum(0) - counts
    assert torch.equal(ranges[:, 1] - ranges[:, 0], counts)
    assert torch.equal(ranges[counts != 0, 0], offsets[counts != 0])
    return tile_ids, ids, offsets


def paired_inputs(context, state, gsplat, legacy):
    import torch
    from gsplat.cuda._wrapper import spherical_harmonics

    r, w, h = gsplat.inputs, gsplat.width, gsplat.height
    means, conics, colors, opacity, offsets, ids, background = r
    native_valid = gsplat.radii > 0
    total = state.means3D.shape[0]
    tile, old_ids, old_offsets = legacy_membership(legacy, w, h, offsets.numel())
    assert bool(((old_ids >= 0) & (old_ids < total)).all())
    assert bool(native_valid[ids].all())
    missing = torch.zeros_like(native_valid)
    missing[old_ids] = True
    missing &= ~native_valid
    # Preserve the original gsplat tensors for every native-visible Gaussian.
    # Fill only the additional IDs that the opacity-aware projection omitted.
    means, conics, colors = (a.clone() for a in (means, conics, colors))
    rasterizer = context.rasterizer
    settings = rasterizer.raster_settings
    radii = torch.empty((1, total, 2), device=means.device, dtype=torch.int32)
    extra_means, extra_conics = torch.empty_like(means), torch.empty_like(conics)
    extra_depths = torch.empty_like(opacity)
    scales = state.scales
    if settings.scale_modifier != 1:
        scales = scales * settings.scale_modifier
    torch.ops.gsplat_tile_ablation.project(
        [state.means3D.contiguous(), state.rotations.contiguous(), scales.contiguous(),
         rasterizer.viewmats, rasterizer.Ks], w, h, radii, extra_means, extra_depths, extra_conics)
    assert bool((radii[0, missing] > 0).all())
    means[:, missing] = extra_means[:, missing]
    conics[:, missing] = extra_conics[:, missing]
    colors[:, missing, 3] = extra_depths[:, missing]
    if context.colors_precomp is None:
        campos = torch.inverse(rasterizer.viewmats)[..., :3, 3]
        dirs = state.means3D.unsqueeze(0) - campos[:, None, :]
        extra_colors = spherical_harmonics(settings.sh_degree, dirs, state.shs.unsqueeze(0),
                                           masks=missing.unsqueeze(0))
        colors[:, missing, :3] = torch.clamp_min(extra_colors[:, missing] + 0.5, 0)

    # Common depth sort and stable Gaussian-ID tiebreaker. Prove the native
    # list already has this order, preserving it byte for byte.
    depth_bits = colors[0, :, 3].contiguous().view(torch.int32)
    id_bits = (total - 1).bit_length()

    def key(tiles, points):
        depth = depth_bits[points].to(torch.int64)
        assert bool((depth > 0).all())
        return (tiles << (32 + id_bits)) | (depth << id_bits) | points.to(torch.int64)

    native_offsets = offsets.flatten()
    ends = torch.cat((native_offsets[1:], native_offsets.new_tensor([ids.numel()])))
    native_tiles = torch.repeat_interleave(
        torch.arange(offsets.numel(), device=ids.device), ends - native_offsets, output_size=ids.numel())
    native_key = key(native_tiles, ids)
    assert bool((native_key[1:] > native_key[:-1]).all())
    old_ids = old_ids[key(tile, old_ids).argsort()].contiguous()
    old_offsets = old_offsets.to(torch.int32).reshape_as(offsets)
    common = means, conics, colors, opacity
    arms = dict(zip(ARMS, ((*common, offsets, ids, background),
                          (*common, old_offsets, old_ids, background))))
    return arms, int(missing.sum())


def main():
    import torch

    native = binary('gsplat_cuda', os.environ['GSPLAT_NATIVE_BINARY'])
    sys.modules['gsplat.csrc'] = native
    with patch('torch.utils.cpp_extension._jit_compile', side_effect=AssertionError('unexpected JIT')):
        from gsplat.cuda import _backend
    assert _backend._C is native
    import gsplat_mixed_ext
    import diff_gaussian_rasterization
    extension = binary('gsplat_mixed_tensor_v7_ca', os.environ['GSPLAT_MIXED_BINARY'])
    gsplat_mixed_ext._extensions['tensor_v7_ca'] = extension
    root = Path(__file__).resolve().parent
    prefetch = Path(os.environ['GSPLAT_PREFETCH_BINARY'])
    projector = root / 'build/project.so'
    torch.ops.load_library(str(prefetch))
    torch.ops.load_library(str(projector))
    args, pc, scene, pipe, background, views, report = load_workload()
    from gaussian_renderer import prepare_render_context, deform_for_render
    from gaussian_renderer.gsplat_mixed import GsplatMixedRenderer, prepare_gsplat
    from gaussian_renderer.tacker_pipeline import (
        prepare_full_mlp_graph_task, _prepare_raster_for_deform_mlp_overlap)
    from sampling_prepass_priority import CudaDriver

    renderer = GsplatMixedRenderer(pc, pipe, background, args.qualification_profile,
        kernel_variant='tensor_v7_ca', physical_blocks=5440, cam_type=scene.dataset_type)
    renderer.prepare()
    assert len(renderer.graphs) == 2 and all(slot.graph for slot in renderer.graphs)
    next_context = prepare_render_context(views[1], pc, pipe, background, cam_type=scene.dataset_type)
    slot = renderer.graphs[1]
    graph = slot.graph

    class ReplayAudit:
        calls = 0

        def replay(self):
            self.calls += 1
            return graph.replay()

    audit = ReplayAudit()
    slot.graph = audit
    renderer.deform_stream.wait_stream(torch.cuda.current_stream())
    task = prepare_full_mlp_graph_task(next_context, pc, renderer.deform_stream,
        renderer.variant, renderer.cached, slot, renderer.prefix_ready[1])
    torch.cuda.current_stream().wait_event(task.prefix_ready)
    slot.graph = graph
    assert audit.calls == 1
    report['graph'] = {'qualification': 'full_mlp_overlap_graph_pb0', 'slots': 2,
                       'actual_prefix_replays': 1, 'timed_scope': 'isolated pixel kernel'}
    assert report['parent_qualification_sha256'] == (
        'a13d87f6c4a22a24b20f83cb3c66a009e759b739b498c9ff55f8c35c9b82d23a')
    provenance = [Path(native.__file__), Path(extension.__file__),
                  Path(diff_gaussian_rasterization._C.__file__), prefetch, projector,
                  root / 'build/project.cu', Path(__file__), root / 'build_gsplat_tile_ablation.py']
    report['ablation_sha256'] = {str(p.resolve()): sha(p) for p in provenance}
    driver, stream = CudaDriver(), torch.cuda.Stream()
    capture = os.environ.get('GSPLAT_NCU_CAPTURE') == '1'
    old_pipe = copy.copy(pipe)
    old_pipe.raster_backend = 'diff_gaussian'
    report['frames'] = []
    for i in (range(1) if capture else range(50)):
        context = prepare_render_context(views[i], pc, pipe, background, cam_type=scene.dataset_type)
        state = deform_for_render(context, pc)
        gsplat = prepare_gsplat(context, state)
        old_context = prepare_render_context(views[i], pc, old_pipe, background, cam_type=scene.dataset_type)
        legacy = _prepare_raster_for_deform_mlp_overlap(old_context, state)
        arms, supplemented = paired_inputs(context, state, gsplat, legacy)
        w, h = gsplat.width, gsplat.height
        r = gsplat.inputs
        image = torch.empty((1, h, w, 4), device=r[2].device, dtype=r[2].dtype)
        alpha = torch.empty((1, h, w, 1), device=r[2].device, dtype=r[2].dtype)
        last_ids = torch.empty((1, h, w), device=r[2].device, dtype=torch.int32)
        frame = {'view': i, 'supplemented_gaussians': supplemented,
                 'intersections': {arm: arms[arm][5].numel() for arm in ARMS}}
        graphs = {}
        if i == 0:
            assert list(frame['intersections'].values()) == [1383688, 4363472], frame
            frame['membership_hashes'] = {
                arm: hashlib.sha256(b''.join(x.cpu().numpy().tobytes() for x in arms[arm][4:6])).hexdigest()
                for arm in ARMS}
        torch.cuda.synchronize()
        stream.wait_stream(torch.cuda.current_stream())
        launch = torch.ops.gsplat_rgbd_ablation.prefetch
        for arm in ARMS:
            inputs = list(arms[arm])
            with torch.cuda.stream(stream):
                launch(inputs, w, h, image, alpha, last_ids)
            stream.synchronize()
            owner = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(owner, stream=stream):
                launch(inputs, w, h, image, alpha, last_ids)
            graphs[arm] = owner
            nodes = driver.kernel_nodes(owner.raw_cuda_graph())
            assert len(nodes) == 1 and math.prod(nodes[0]['grid']) == 5440, nodes
            assert 'rasterize_rgbd_prefetch_kernel' in nodes[0]['name'], nodes
            assert math.prod(nodes[0]['block']) == 256, nodes
            if i == 0:
                report.setdefault('arms', {})[arm] = {
                    'nodes': nodes, 'resources': node_resources(driver, owner)[0]}

        if capture:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for arm in ARMS:
                with torch.cuda.stream(stream):
                    graphs[arm].replay()
                stream.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        elif i in TIMING_VIEWS:
            samples = {arm: [] for arm in ARMS}
            for trial in range(6):
                for arm in ARMS[::1 if trial % 2 == 0 else -1]:
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        for _ in range(5):
                            graphs[arm].replay()
                        start.record(stream)
                        for _ in range(20):
                            graphs[arm].replay()
                        stop.record(stream)
                    stop.synchronize()
                    samples[arm].append(start.elapsed_time(stop) / 20)
            frame['unprofiled_ms'] = samples
            frame['unprofiled_median_ms'] = {arm: statistics.median(v) for arm, v in samples.items()}

        # Output addresses are also shared by both measured Graphs.
        with torch.cuda.stream(stream):
            graphs['gsplat_tiles'].replay()
            base_image, base_alpha = image.clone(), alpha.clone()
            graphs['legacy_tiles'].replay()
        stream.synchronize()
        native_image, native_alpha, _ = native.rasterize_to_pixels_3dgs_fwd(
            *r[:4], r[6], None, w, h, 16, r[4], r[5])
        assert torch.equal(base_image.view(torch.int32), native_image.view(torch.int32))
        assert torch.equal(base_alpha.view(torch.int32), native_alpha.view(torch.int32))
        frame['quality'] = {}
        for label, a, b in (
            ('rgb', base_image[..., :3], image[..., :3]),
            ('depth', base_image[..., 3:4], image[..., 3:4]),
            ('alpha', base_alpha, alpha),
        ):
            diff = (a - b).abs()
            frame['quality'][label] = {
                'mae': float(diff.mean()), 'max': float(diff.max()),
                'different_pixels': int((diff != 0).any(dim=-1).sum()),
                'bitwise_equal': torch.equal(a.view(torch.int32), b.view(torch.int32))}
        report['frames'].append(frame)
        if i in TIMING_VIEWS:
            print('FRAME', i, frame, flush=True)
    for key in ('source_and_checkpoint_sha256', 'ablation_sha256'):
        assert all(sha(p) == digest for p, digest in report[key].items())
    report['completed'] = True
    save(args.output / 'report.json', report)
    print('DONE', args.output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
