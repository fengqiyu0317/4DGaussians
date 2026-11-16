#!/usr/bin/env python3
"""Frozen old-v11 versus gsplat leaf resources on the same real inputs."""
import copy
import hashlib
import importlib.util
import json
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


def binary(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    import torch
    native = binary('gsplat_cuda', os.environ['GSPLAT_NATIVE_BINARY'])
    # Supply the sealed binary to gsplat's regular precompiled import path.
    sys.modules['gsplat.csrc'] = native
    def sealed_jit(name, *args, **kwargs):
        assert name == 'gsplat_cuda', name
        return native
    with patch('torch.utils.cpp_extension._jit_compile', sealed_jit):
        from gsplat.cuda import _backend
    assert _backend._C is native
    import gsplat_mixed_ext
    extension = binary('gsplat_mixed_tensor_v7_ca', os.environ['GSPLAT_MIXED_BINARY'])
    gsplat_mixed_ext._extensions['tensor_v7_ca'] = extension

    args, pc, scene, pipe, background, views, report = load_workload()
    from gaussian_renderer import prepare_render_context, deform_for_render
    from gaussian_renderer.gsplat_mixed import GsplatMixedRenderer, prepare_gsplat
    from gaussian_renderer.tacker_pipeline import (
        prepare_full_mlp_graph_task, _prepare_raster_for_deform_mlp_overlap)
    from gsplat.cuda._wrapper import rasterize_to_pixels
    from tests.test_gsplat_tensor_v7_cuda import LegacyV7
    from sampling_prepass_priority import CudaDriver
    import diff_gaussian_rasterization

    renderer = GsplatMixedRenderer(pc, pipe, background, args.qualification_profile,
        kernel_variant='tensor_v7_ca', physical_blocks=5440, cam_type=scene.dataset_type)
    renderer.prepare()
    context = prepare_render_context(views[0], pc, pipe, background, cam_type=scene.dataset_type)
    state = deform_for_render(context, pc)
    prepared = prepare_gsplat(context, state)
    old_pipe = copy.copy(pipe); old_pipe.raster_backend = 'diff_gaussian'
    old_context = prepare_render_context(views[0], pc, old_pipe, background, cam_type=scene.dataset_type)
    old = _prepare_raster_for_deform_mlp_overlap(old_context, state)
    old_rasterizer = old_context.rasterizer
    assert hasattr(diff_gaussian_rasterization._C, 'render_prepared_gaussians_raster_fast_range')

    next_context = prepare_render_context(views[1], pc, pipe, background, cam_type=scene.dataset_type)
    class ReplayAudit:
        def __init__(self, owner): self.owner, self.calls = owner, 0
        def replay(self): self.calls += 1; return self.owner.replay()
    slot = renderer.graphs[1]
    audit = ReplayAudit(slot.graph); slot.graph = audit
    renderer.deform_stream.wait_stream(torch.cuda.current_stream())
    task = prepare_full_mlp_graph_task(next_context, pc, renderer.deform_stream,
        renderer.variant, renderer.cached, slot, renderer.prefix_ready[1])
    torch.cuda.current_stream().wait_event(task.prefix_ready)
    features = task.deform_features.clone().contiguous()
    slot.graph = audit.owner
    assert audit.calls == 1
    mlp = [features, *renderer.leaf_cached]
    fields = ('deform_features', 'backbone_weight', 'backbone_bias', 'first_weights',
              'first_biases', 'packed_tail_weights', 'tail_biases')
    old_kwargs = dict(zip(fields, mlp))
    old_kwargs.update(prepared=old, persistent_blocks=0)
    def old_mixed():
        return old_rasterizer.render_prepared_with_deform_mlp_tensor_tail_raster_fast_overlap(**old_kwargs)
    r = prepared.inputs
    def pixels():
        return rasterize_to_pixels(*r[:4], prepared.width, prepared.height, 16, r[4], r[5], backgrounds=r[6])
    def mixed():
        return extension.forward(r, mlp, prepared.width, prepared.height, 5440, 2, False, True)
    arms = {
        'old_raster': lambda: old_rasterizer.render_prepared_raster_fast_range(old, 0, 5440),
        'gsplat_raster': pixels,
        'old_mixed_v11': old_mixed,
        'gsplat_mixed_ca': mixed,
    }
    torch.cuda.synchronize()
    expected_old = old_mixed()
    old_rgb, old_depth = expected_old[0].clone(), expected_old[2].clone()
    expected_heads = [t.clone() for t in LegacyV7()(mlp)]
    assert all(torch.equal(a, b) for a, b in zip(expected_old[3], expected_heads))
    image, alpha = pixels(); expected_image, expected_alpha = image.clone(), alpha.clone()
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    driver = CudaDriver(); graphs, outputs = {}, {}
    report['arms'] = {}
    for label, callback in arms.items():
        with torch.cuda.stream(stream):
            for _ in range(3): callback()
        stream.synchronize()
        owner = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(owner, stream=stream): outputs[label] = callback()
        graphs[label] = owner
        nodes = driver.kernel_nodes(owner.raw_cuda_graph())
        assert len(nodes) == 1 and math.prod(nodes[0]['grid']) == 5440, nodes
        assert math.prod(nodes[0]['block']) == (256 if label.endswith('raster') else 384), nodes
        report['arms'][label] = {'nodes': nodes, 'resources': node_resources(driver, owner)}

    def check(label):
        raw = outputs[label]
        if label.startswith('old_'):
            assert torch.equal(old.color, old_rgb) and torch.equal(old.depth, old_depth), label
            if label == 'old_mixed_v11':
                assert all(torch.equal(a, b) for a, b in zip(raw[3], expected_heads))
        else:
            assert torch.equal(raw[0], expected_image) and torch.equal(raw[1], expected_alpha), label
            if label == 'gsplat_mixed_ca':
                assert all(torch.equal(a, b) for a, b in zip(raw[3:8], expected_heads))
    for label in arms:
        with torch.cuda.stream(stream): graphs[label].replay()
        stream.synchronize(); check(label)
    if os.environ.get('GSPLAT_NCU_CAPTURE') == '1':
        torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart()
        for label in arms:
            with torch.cuda.stream(stream): graphs[label].replay()
            stream.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    else:
        report['unprofiled_ms'] = {k: [] for k in arms}
        for trial in range(6):
            for label in (list(arms) if trial % 2 == 0 else list(reversed(arms))):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    for _ in range(5): graphs[label].replay()
                    a.record(stream)
                    for _ in range(20): graphs[label].replay()
                    b.record(stream)
                b.synchronize()
                report['unprofiled_ms'][label].append(a.elapsed_time(b) / 20)
        report['unprofiled_median_ms'] = {k: statistics.median(v) for k, v in report['unprofiled_ms'].items()}
    for label in arms:
        with torch.cuda.stream(stream): graphs[label].replay()
        stream.synchronize(); check(label)
        report['arms'][label]['output_checks_passed'] = True
    raw_mapping = extension.forward(r, mlp, prepared.width, prepared.height, 5440, 2, True, True)[9].cpu()
    assert int(raw_mapping[:, 0].sum()) == 6971 and int((raw_mapping[:, 1] >= 0).sum()) == 1531
    assert bool(raw_mapping[:, 2].all())
    report.update(kind='old_v11_vs_gsplat_leaf_ncu', completed=True,
        scope='isolated Graph leaf nodes; same Raster(0) activated state and Graph-prefix MLP(1) input; no prefix/prepare/alloc/JIT in profiled interval; not end-to-end FPS',
        graph={'parent': 'full_mlp_overlap_graph_pb0', 'slots': 2, 'actual_prefix_replays': audit.calls},
        intersections={'old': old.num_rendered, 'gsplat': r[5].numel()},
        candidate_pairing={'active': True, 'extra_workers': 1531, 'deform_tiles': 6971},
        backend_rgb_difference={'mae': float((old_rgb - expected_image[0, ..., :3].permute(2, 0, 1)).abs().mean()),
                                'max': float((old_rgb - expected_image[0, ..., :3].permute(2, 0, 1)).abs().max())})
    binaries = [Path(native.__file__), Path(extension.__file__), Path(diff_gaussian_rasterization._C.__file__),
                Path(os.environ['GSPLAT_LEGACY_V7_LIBRARY'])]
    report['binary_sha256'] = {str(p.resolve()): sha(p) for p in binaries}
    # Include the new driver and exact candidate source as provenance, without rebuilding.
    for p in [Path(__file__), RUNTIME/'gsplat_mixed_ext/mixed_tensor_v7_ca.cu']:
        report['source_and_checkpoint_sha256'][str(p)] = sha(p)
        dest = args.output/'runtime_sources'/p.name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    report['inputs_unchanged'] = all(sha(p) == value for p, value in report['source_and_checkpoint_sha256'].items())
    assert report['inputs_unchanged']
    save(args.output/'report.json', report)
    print('RESOURCE_CHECKS', json.dumps(report['arms']), flush=True)
    print('DONE', args.output/'report.json', flush=True)


if __name__ == '__main__': main()
