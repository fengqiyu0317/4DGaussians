#!/usr/bin/env python3
"""Measure RGBD load placement with shared gsplat inputs and real Graph nodes."""
from itertools import permutations
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

ARMS = ('on_demand', 'prefetch', 'padded')
TIMING_VIEWS = (0, 10, 20, 30, 40)


def main():
    import torch

    native = binary('gsplat_cuda', os.environ['GSPLAT_NATIVE_BINARY'])
    sys.modules['gsplat.csrc'] = native
    with patch('torch.utils.cpp_extension._jit_compile', side_effect=AssertionError('unexpected JIT')):
        from gsplat.cuda import _backend
    assert _backend._C is native
    build = Path(__file__).resolve().parent / 'build'
    for arm in ARMS[1:]:
        torch.ops.load_library(str(build / f'{arm}.so'))
    import gsplat_mixed_ext
    extension = binary('gsplat_mixed_tensor_v7_ca', os.environ['GSPLAT_MIXED_BINARY'])
    gsplat_mixed_ext._extensions['tensor_v7_ca'] = extension

    args, pc, scene, pipe, background, views, report = load_workload()
    from gaussian_renderer import prepare_render_context, deform_for_render
    from gaussian_renderer.gsplat_mixed import GsplatMixedRenderer, prepare_gsplat
    from gaussian_renderer.tacker_pipeline import prepare_full_mlp_graph_task
    from sampling_prepass_priority import CudaDriver

    renderer = GsplatMixedRenderer(pc, pipe, background, args.qualification_profile,
        kernel_variant='tensor_v7_ca', physical_blocks=5440, cam_type=scene.dataset_type)
    renderer.prepare()
    assert len(renderer.graphs) == 2 and all(slot.graph for slot in renderer.graphs)
    next_context = prepare_render_context(views[1], pc, pipe, background, cam_type=scene.dataset_type)
    slot = renderer.graphs[1]
    owner = slot.graph

    class ReplayAudit:
        calls = 0

        def replay(self):
            self.calls += 1
            return owner.replay()

    audit = ReplayAudit()
    slot.graph = audit
    renderer.deform_stream.wait_stream(torch.cuda.current_stream())
    task = prepare_full_mlp_graph_task(next_context, pc, renderer.deform_stream,
        renderer.variant, renderer.cached, slot, renderer.prefix_ready[1])
    torch.cuda.current_stream().wait_event(task.prefix_ready)
    slot.graph = owner
    assert audit.calls == 1
    report['graph'] = {'qualification': 'full_mlp_overlap_graph_pb0', 'slots': 2,
                       'actual_prefix_replays': audit.calls, 'timed_scope': 'isolated pixel kernel'}
    assert report['parent_qualification_sha256'] == (
        'a13d87f6c4a22a24b20f83cb3c66a009e759b739b498c9ff55f8c35c9b82d23a')
    provenance = [Path(native.__file__), Path(extension.__file__),
                  build / 'prefetch.so', build / 'padded.so',
                  build / 'RasterizeToPixels3DGSFwd.cu', build / 'prefetch.cu', build / 'padded.cu',
                  Path(__file__), build.parent / 'build_gsplat_rgbd_ablation.py']
    report['ablation_sha256'] = {str(p.resolve()): sha(p) for p in provenance}
    driver = CudaDriver()
    stream = torch.cuda.Stream()
    capture = os.environ.get('GSPLAT_NCU_CAPTURE') == '1'
    report['frames'] = []
    for i in (range(1) if capture else range(50)):
        context = prepare_render_context(views[i], pc, pipe, background, cam_type=scene.dataset_type)
        prepared = prepare_gsplat(context, deform_for_render(context, pc))
        r, w, h = prepared.inputs, prepared.width, prepared.height
        assert r[2].is_contiguous() and r[2].shape[-1] == 4
        assert r[2].data_ptr() % 16 == 0
        torch.cuda.synchronize()
        stream.wait_stream(torch.cuda.current_stream())
        timed = i in TIMING_VIEWS
        graphs, outputs = {}, {}
        frame = {'view': i, 'intersections': r[5].numel(), 'differences': {}}
        if i == 0:
            assert frame['intersections'] == 1383688, frame
        for arm in ARMS:
            def launch():
                if arm == 'on_demand':
                    return native.rasterize_to_pixels_3dgs_fwd(
                        *r[:4], r[6], None, w, h, 16, r[4], r[5])
                image = torch.empty((1, h, w, 4), device=r[2].device, dtype=r[2].dtype)
                alpha = torch.empty((1, h, w, 1), device=r[2].device, dtype=r[2].dtype)
                last_ids = torch.empty((1, h, w), device=r[2].device, dtype=torch.int32)
                getattr(torch.ops.gsplat_rgbd_ablation, arm)(list(r), w, h, image, alpha, last_ids)
                return image, alpha, last_ids

            with torch.cuda.stream(stream):
                outputs[arm] = launch()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                outputs[arm] = launch()
            graphs[arm] = graph
            nodes = driver.kernel_nodes(graph.raw_cuda_graph())
            assert len(nodes) == 1 and math.prod(nodes[0]['grid']) == 5440, nodes
            assert math.prod(nodes[0]['block']) == 256, nodes
            if i == 0:
                report.setdefault('arms', {})[arm] = {
                    'nodes': nodes, 'resources': node_resources(driver, graph)[0]}
            with torch.cuda.stream(stream):
                graph.replay()
        stream.synchronize()

        def check_outputs():
            base, base_alpha, base_ids = outputs['on_demand']
            for arm in ARMS[1:]:
                image, alpha, ids = outputs[arm]
                metrics = {}
                for label, a, b in (
                    ('rgb', base[..., :3], image[..., :3]), ('depth', base[..., 3], image[..., 3]),
                    ('alpha', base_alpha, alpha),
                ):
                    diff = (a - b).abs()
                    metrics[label] = {'mae': float(diff.mean()), 'max': float(diff.max()),
                                      'bitwise_equal': torch.equal(a.view(torch.int32), b.view(torch.int32))}
                metrics['last_ids_equal'] = torch.equal(base_ids, ids)
                frame['differences'][arm] = metrics

        if capture:
            check_outputs()
            before_ncu = frame['differences'].copy()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for arm in ARMS:
                with torch.cuda.stream(stream):
                    graphs[arm].replay()
                stream.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        elif timed:
            samples = {arm: [] for arm in ARMS}
            for order in permutations(ARMS):
                for arm in order:
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
        check_outputs()
        if capture:
            assert before_ncu == frame['differences']
        report['frames'].append(frame)
        if timed:
            print('FRAME', i, frame.get('unprofiled_median_ms'), flush=True)
    for key in ('source_and_checkpoint_sha256', 'ablation_sha256'):
        assert all(sha(p) == digest for p, digest in report[key].items())
    report['completed'] = True
    save(args.output / 'report.json', report)
    print('DONE', args.output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
