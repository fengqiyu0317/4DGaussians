#!/usr/bin/env python3
"""Matched Graph and end-to-end benchmarks for candidate-aware wide Raster."""
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

from analyze_gsplat_mixed import load_workload, save, sha
from benchmark_gsplat_candidate import verify_counts
from benchmark_gsplat_v7 import measure, node_resources, summarize_trials, v7_state

CONFIGS = {
    'independent': ('tensor_v7_ca', False),
    'mixed': ('tensor_v7_ca', True),
    'wide_independent': ('tensor_v7_ca_wide', False),
    'wide_mixed': ('tensor_v7_ca_wide', True),
}
NCU_ARMS = ('raster', 'wide_raster', 'mixed', 'wide_mixed')


def load_binary(name, variable):
    spec = importlib.util.spec_from_file_location(name, os.environ[variable])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReplayAudit:
    def __init__(self, owner):
        self.owner, self.calls = owner, 0

    def replay(self):
        self.calls += 1
        return self.owner.replay()


def forward(renderer, prepared, mlp, mode=2, profile=False):
    return renderer.extension.forward(prepared.inputs, mlp, prepared.width, prepared.height,
                                      5440, mode, profile, True)


def bitwise(a, b):
    import torch
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def graph_features(renderer, context, pc):
    import torch
    from gaussian_renderer.tacker_pipeline import prepare_full_mlp_graph_task

    slot = renderer.graphs[1]
    audit = ReplayAudit(slot.graph)
    slot.graph = audit
    renderer.deform_stream.wait_stream(torch.cuda.current_stream())
    try:
        task = prepare_full_mlp_graph_task(context, pc, renderer.deform_stream, renderer.variant,
            renderer.cached, slot, renderer.prefix_ready[1])
        torch.cuda.current_stream().wait_event(task.prefix_ready)
        assert audit.calls == 1
        return task.deform_features.clone().contiguous()
    finally:
        slot.graph = audit.owner


def leaf(pc, scene, pipe, background, views, report, runners, capture):
    import torch
    from gaussian_renderer import prepare_render_context, deform_for_render
    from gaussian_renderer.gsplat_mixed import prepare_gsplat
    from gsplat.cuda._wrapper import rasterize_to_pixels
    from sampling_prepass_priority import CudaDriver
    from tests.test_gsplat_tensor_v7_cuda import LegacyV7

    base, wide = runners['mixed'], runners['wide_mixed']
    oracle = LegacyV7()
    report['leaf_cases'] = []
    for index in ((0,) if capture else (0, 24, 48)):
        context = prepare_render_context(views[index], pc, pipe, background, cam_type=scene.dataset_type)
        state = deform_for_render(context, pc) if index == 0 else v7_state(context, pc, base)
        prepared = prepare_gsplat(context, state)
        r, w, h = prepared.inputs, prepared.width, prepared.height
        next_context = prepare_render_context(views[index + 1], pc, pipe, background, cam_type=scene.dataset_type)
        mlp = [graph_features(base, next_context, pc), *base.leaf_cached]
        raster = lambda: rasterize_to_pixels(*r[:4], w, h, 16, r[4], r[5], backgrounds=r[6])
        wide_raster = lambda: wide.extension.raster_forward(r, w, h)
        deform = lambda: base.extension.mlp_forward(mlp)
        wide_deform = lambda: wide.extension.mlp_forward(mlp)
        arms = {
            'raster': raster, 'wide_raster': wide_raster,
            'mlp': deform, 'wide_mlp': wide_deform,
            'independent': lambda: (*raster(), *deform()),
            'wide_independent': lambda: (*wide_raster()[:2], *wide_deform()),
            'mixed': lambda: forward(base, prepared, mlp),
            'wide_mixed': lambda: forward(wide, prepared, mlp),
            'mixed_raster_only': lambda: forward(base, prepared, mlp, mode=0)[:3],
            'wide_mixed_raster_only': lambda: forward(wide, prepared, mlp, mode=0)[:3],
            'mixed_mlp_only': lambda: forward(base, prepared, mlp, mode=1)[3:8],
            'wide_mixed_mlp_only': lambda: forward(wide, prepared, mlp, mode=1)[3:8],
        }
        expected_raster, expected_mlp = raster(), oracle(mlp)
        base_raw = forward(base, prepared, mlp, profile=True)
        wide_raw = forward(wide, prepared, mlp, profile=True)
        assignments = base_raw[9]
        assert torch.equal(assignments, wide_raw[9])
        assert int(assignments[:, 0].sum()) == 6971
        assert int((assignments[:, 1] >= 0).sum()) == 1531 and bool(assignments[:, 2].all())
        if index == 0:
            # Exercise the scalar fallback with an otherwise valid contiguous RGBD tensor.
            colors = r[2]
            shifted = torch.empty(colors.numel() + 1, dtype=colors.dtype, device=colors.device)[1:]
            shifted = shifted.view_as(colors)
            shifted.copy_(colors)
            assert shifted.is_contiguous() and shifted.data_ptr() % 16 != 0
            unaligned = (*r[:2], shifted, *r[3:])
            assert all(bitwise(a, b) for a, b in zip(
                wide.extension.raster_forward(unaligned, w, h), base_raw[:3]))
        case = {'raster_view': index, 'deform_view': index + 1, 'actual_prefix_replays': 1,
                'intersections': r[5].numel(),
                'candidate_assignment_sha256': hashlib.sha256(assignments.cpu().numpy().tobytes()).hexdigest(),
                'candidate_extra_workers': 1531, 'deform_tiles': 6971, 'arms': {}}
        driver, stream = CudaDriver(), torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs, outputs = {}, {}
        for name in (NCU_ARMS if capture else arms):
            callback = arms[name]
            with torch.cuda.stream(stream):
                callback()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                outputs[name] = callback()
            nodes = driver.kernel_nodes(graph.raw_cuda_graph())
            assert len(nodes) == (2 if name.endswith('independent') else 1), nodes
            graphs[name] = graph
            case['arms'][name] = {'nodes': nodes, 'resources': node_resources(driver, graph)}
            with torch.cuda.stream(stream):
                graph.replay()
        stream.synchronize()

        def check_outputs():
            for name, out in outputs.items():
                if name.endswith('mlp') or name.endswith('mlp_only'):
                    expected = expected_mlp
                elif name.endswith('independent'):
                    expected = (*expected_raster, *expected_mlp)
                elif name.endswith('mixed'):
                    out = out[:8]
                    expected = (*base_raw[:3], *expected_mlp)
                else:
                    expected = expected_raster if name == 'raster' else base_raw[:3]
                assert len(out) == len(expected) and all(bitwise(a, b) for a, b in zip(out, expected)), name
                case['arms'][name]['bitwise_equal'] = True

        check_outputs()
        if capture:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for graph in graphs.values():
                with torch.cuda.stream(stream):
                    graph.replay()
                stream.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            check_outputs()
        else:
            samples = {name: [] for name in graphs}
            for trial in range(6):
                for name in (graphs if trial % 2 == 0 else reversed(graphs)):
                    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        for _ in range(5):
                            graphs[name].replay()
                        start.record(stream)
                        for _ in range(20):
                            graphs[name].replay()
                        stop.record(stream)
                    stop.synchronize()
                    samples[name].append(start.elapsed_time(stop) / 20)
            for name, values in samples.items():
                case['arms'][name].update(samples_ms=values, median_ms=statistics.median(values))
        report['leaf_cases'].append(case)
        print('LEAF', index, {name: v.get('median_ms') for name, v in case['arms'].items()}, flush=True)
    if not capture:
        report['leaf_ms'] = {name: statistics.mean(c['arms'][name]['median_ms'] for c in report['leaf_cases'])
                             for name in arms}


def formal(args, pc, scene, pipe, background, views, report, runners):
    import torch
    from gaussian_renderer import prepare_render_context
    from gaussian_renderer.gsplat_mixed import prepare_gsplat

    def consume(renderer, selected):
        for _ in renderer.render_sequence(selected):
            pass

    def repeat(arms, trials, key, counts=None):
        result = {name: {'trials': []} for name in arms}
        for trial in range(trials):
            for name in (arms if trial % 2 == 0 else reversed(arms)):
                row = measure(arms[name], len(views))
                if counts:
                    result[name]['execution_counts'] = counts(name)
                result[name]['trials'].append(row)
                print(key, trial, name, row, flush=True)
        for row in result.values():
            row['median'] = summarize_trials(row['trials'])
        report[key] = result
        save(args.output / 'partial.json', report)

    for renderer in runners.values():
        consume(renderer, views[:10])
    repeat({name: lambda r=r: consume(r, views) for name, r in runners.items()}, 10, 'end_to_end',
           counts=lambda name: verify_counts(runners[name], len(views)))

    baseline = runners['independent']
    contexts = [prepare_render_context(v, pc, pipe, background, cam_type=scene.dataset_type) for v in views]
    states = [v7_state(c, pc, baseline) for c in contexts]
    dummy = pc.get_xyz.new_empty((pc.get_xyz.shape[0], 32))
    mlp = [dummy, *baseline.leaf_cached]

    def raster_sequence(renderer):
        for context, state in zip(contexts, states):
            prepared = prepare_gsplat(context, state)
            if renderer.fused:
                prepared.result(forward(renderer, prepared, mlp, mode=0)[0])
            else:
                renderer.render_prepared_raster(prepared)

    for renderer in runners.values():
        raster_sequence(renderer)
    repeat({name: lambda r=r: raster_sequence(r) for name, r in runners.items()}, 6, 'prepare_pixels_layout')
    del contexts, states, mlp, dummy
    torch.cuda.synchronize()

    reference = []
    report['quality'] = {}
    for name, renderer in runners.items():
        audits = [ReplayAudit(slot.graph) for slot in renderer.graphs]
        for slot, audit in zip(renderer.graphs, audits):
            slot.graph = audit
        rows = []
        try:
            for index, result in enumerate(renderer.render_sequence(views)):
                outputs = result['render'].cpu(), result['depth'].cpu()
                if name == 'independent':
                    reference.append(outputs)
                row = {'view': index}
                for channel, output, expected in zip(('rgb', 'depth'), outputs, reference[index]):
                    diff = (output - expected).abs()
                    row[channel] = {'mae': float(diff.mean()), 'max': float(diff.max()),
                                    'bitwise_equal': bitwise(output, expected)}
                rows.append(row)
        finally:
            for slot, audit in zip(renderer.graphs, audits):
                slot.graph = audit.owner
        replays = [a.calls for a in audits]
        assert sum(replays) == 49 and len(rows) == 50, (name, replays, len(rows))
        report['quality'][name] = {'rows': rows, 'actual_prefix_replays_per_slot': replays,
                                   'execution_counts': verify_counts(renderer, len(views))}


def main():
    import torch
    import gsplat_mixed_ext
    native = load_binary('gsplat_cuda', 'GSPLAT_NATIVE_BINARY')
    sys.modules['gsplat.csrc'] = native
    with patch('torch.utils.cpp_extension._jit_compile', side_effect=AssertionError('unexpected JIT')):
        from gsplat.cuda import _backend
    assert _backend._C is native
    for variant, variable in (('tensor_v7_ca', 'GSPLAT_MIXED_BINARY'),
                              ('tensor_v7_ca_wide', 'GSPLAT_WIDE_MIXED_BINARY')):
        gsplat_mixed_ext._extensions[variant] = load_binary('gsplat_mixed_' + variant, variable)
    args, pc, scene, pipe, background, views, report = load_workload()
    from gaussian_renderer.gsplat_mixed import GsplatMixedRenderer
    assert report['parent_qualification_sha256'] == (
        'a13d87f6c4a22a24b20f83cb3c66a009e759b739b498c9ff55f8c35c9b82d23a')
    runners = {name: GsplatMixedRenderer(pc, copy.copy(pipe), background, args.qualification_profile,
        kernel_variant=variant, fused=fused, physical_blocks=5440, cam_type=scene.dataset_type)
        for name, (variant, fused) in CONFIGS.items()}
    for renderer in runners.values():
        renderer.prepare()
        assert len(renderer.graphs) == 2 and all(slot.graph for slot in renderer.graphs)
        assert all(bitwise(a, b) for a, b in zip(renderer.leaf_cached, runners['mixed'].leaf_cached))
    report['profile'] = 'full_mlp_overlap_graph_pb0'
    report['configurations'] = CONFIGS
    report['binary_sha256'] = {str(Path(m.__file__).resolve()): sha(m.__file__)
        for m in (native, *gsplat_mixed_ext._extensions.values())}
    oracle_path = Path(os.environ['GSPLAT_LEGACY_V7_LIBRARY'])
    report['binary_sha256'][str(oracle_path)] = sha(oracle_path)
    extra_sources = [Path(__file__), Path(gsplat_mixed_ext.__file__).with_name('mixed_tensor_v7_ca.cu'),
                     Path(gsplat_mixed_ext.__file__).with_name('bindings_tensor_v7_ca.cpp')]
    report['source_and_checkpoint_sha256'].update({str(p.resolve()): sha(p) for p in extra_sources})
    for source in extra_sources:
        (args.output / 'runtime_sources' / source.name).write_bytes(source.read_bytes())
    if args.stage == 'formal':
        formal(args, pc, scene, pipe, background, views, report, runners)
    else:
        leaf(pc, scene, pipe, background, views, report, runners, os.environ.get('GSPLAT_NCU_CAPTURE') == '1')
    for key in ('binary_sha256', 'source_and_checkpoint_sha256'):
        assert all(sha(p) == digest for p, digest in report[key].items())
    report['completed'] = True
    save(args.output / 'report.json', report)
    print('DONE', args.output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
