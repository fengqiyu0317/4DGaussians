#!/usr/bin/env python3
"""Compare sealed and wide RGBD prefetch on each frozen candidate list."""
import copy
import hashlib
import math
import os
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

from profile_gsplat_tile_ablation import binary, load_workload, node_resources, paired_inputs, save, sha

LISTS = ('legacy_tiles', 'gsplat_tiles')
LAYOUTS = ('prefetch', 'wide')
ARMS = tuple((candidates, layout) for candidates in LISTS for layout in LAYOUTS)
TIMING_VIEWS = (0, 10, 20, 30, 40)


def label(arm):
    return '_'.join(arm)


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
    projector = Path(os.environ['GSPLAT_CANDIDATE_PROJECTOR'])
    wide = root / 'build/wide.so'
    for library in (prefetch, projector, wide):
        torch.ops.load_library(str(library))
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
    prefix_graph = slot.graph

    class ReplayAudit:
        calls = 0

        def replay(self):
            self.calls += 1
            return prefix_graph.replay()

    audit = ReplayAudit()
    slot.graph = audit
    renderer.deform_stream.wait_stream(torch.cuda.current_stream())
    task = prepare_full_mlp_graph_task(next_context, pc, renderer.deform_stream,
        renderer.variant, renderer.cached, slot, renderer.prefix_ready[1])
    torch.cuda.current_stream().wait_event(task.prefix_ready)
    slot.graph = prefix_graph
    assert audit.calls == 1
    report['graph'] = {'qualification': 'full_mlp_overlap_graph_pb0', 'slots': 2,
                       'actual_prefix_replays': 1, 'timed_scope': 'isolated pixel kernel'}
    assert report['parent_qualification_sha256'] == (
        'a13d87f6c4a22a24b20f83cb3c66a009e759b739b498c9ff55f8c35c9b82d23a')
    provenance = [Path(native.__file__), Path(extension.__file__),
                  Path(diff_gaussian_rasterization._C.__file__), prefetch, projector, wide,
                  projector.with_suffix('.cu'), wide.with_suffix('.cu'),
                  Path(__file__), root / 'profile_gsplat_tile_ablation.py',
                  root / 'build_gsplat_wide_load_ablation.py']
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
        inputs, supplemented = paired_inputs(context, state, gsplat, legacy)
        w, h = gsplat.width, gsplat.height
        image = torch.empty((1, h, w, 4), device='cuda', dtype=torch.float32)
        alpha = torch.empty((1, h, w, 1), device='cuda', dtype=torch.float32)
        last_ids = torch.empty((1, h, w), device='cuda', dtype=torch.int32)
        frame = {'view': i, 'supplemented_gaussians': supplemented,
                 'intersections': {candidate: inputs[candidate][5].numel() for candidate in LISTS}}
        if i == 0:
            frame['membership_hashes'] = {
                candidate: hashlib.sha256(b''.join(x.cpu().numpy().tobytes()
                    for x in inputs[candidate][4:6])).hexdigest() for candidate in LISTS}
        torch.cuda.synchronize()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = {}
        for candidates, layout in ARMS:
            raster_inputs = list(inputs[candidates])
            launch = getattr(torch.ops.gsplat_rgbd_ablation, layout)
            with torch.cuda.stream(stream):
                launch(raster_inputs, w, h, image, alpha, last_ids)
            stream.synchronize()
            owner = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(owner, stream=stream):
                launch(raster_inputs, w, h, image, alpha, last_ids)
            arm = candidates, layout
            graphs[arm] = owner
            nodes = driver.kernel_nodes(owner.raw_cuda_graph())
            assert len(nodes) == 1 and math.prod(nodes[0]['grid']) == 5440, nodes
            assert f'rasterize_rgbd_{layout}_kernel' in nodes[0]['name'], nodes
            assert math.prod(nodes[0]['block']) == 256, nodes
            if i == 0:
                resources = node_resources(driver, owner)[0]
                assert resources['dynamic_shared_bytes'] == 11264
                report.setdefault('arms', {})[label(arm)] = {'nodes': nodes, 'resources': resources}

        if capture:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for arm in ARMS:
                with torch.cuda.stream(stream):
                    graphs[arm].replay()
                stream.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        elif i in TIMING_VIEWS:
            samples = {label(arm): [] for arm in ARMS}
            for trial in range(6):
                for arm in (ARMS if trial % 2 == 0 else reversed(ARMS)):
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
                    samples[label(arm)].append(start.elapsed_time(stop) / 20)
            frame['unprofiled_ms'] = samples
            frame['unprofiled_median_ms'] = {arm: statistics.median(v) for arm, v in samples.items()}
        frame['quality'] = {}
        for candidates in LISTS:
            with torch.cuda.stream(stream):
                graphs[candidates, 'prefetch'].replay()
                base_image, base_alpha, base_ids = image.clone(), alpha.clone(), last_ids.clone()
                graphs[candidates, 'wide'].replay()
            stream.synchronize()
            metrics = {'last_ids_equal': torch.equal(base_ids, last_ids)}
            for channel, before, after in (
                ('rgb', base_image[..., :3], image[..., :3]),
                ('depth', base_image[..., 3:4], image[..., 3:4]),
                ('alpha', base_alpha, alpha),
            ):
                diff = (before - after).abs()
                metrics[channel] = {'mae': float(diff.mean()), 'max': float(diff.max()),
                                    'bitwise_equal': torch.equal(before.view(torch.int32), after.view(torch.int32))}
            frame['quality'][candidates] = metrics
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
