"""MLP-only comparison using real checkpoint weights and HexPlane features.

The native control is standalone Tensor-tail v4, not the v11 mixed adapter.
V11 full-renderer comparisons must use full_mlp_overlap_graph qualification.
"""
import argparse
import contextlib
import json
import hashlib
import importlib.util
import os
from pathlib import Path
import statistics
import sys
import time

from benchmark_original_deform_compile import gpu_snapshot, source_hashes, write_json, compiler_compatibility_patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "model-path", "source-path", "output", "native-extension"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    before = gpu_snapshot(0, require_idle=True)
    root = Path(args.project_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tacker_ext"))
    for key in list(os.environ):
        if key.startswith("TACKER_"):
            os.environ.pop(key)
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "2"
    import torch
    extension_spec = importlib.util.spec_from_file_location("tacker_4dgs_head._C", args.native_extension)
    extension = importlib.util.module_from_spec(extension_spec)
    extension_spec.loader.exec_module(extension)
    assert hasattr(extension, "deform_mlp_tensor_tail_out_v4")
    sys.modules["tacker_4dgs_head._C"] = extension
    from arguments import ModelParams, ModelHiddenParams, PipelineParams, get_combined_args
    from utils.params_utils import load_config, merge_hparams
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    import scene.deformation as deformation_module
    from gaussian_renderer.tacker_pipeline import _cache_full_mlp_parameters
    from tacker_4dgs_head import deform_mlp_tensor_tail_out_v4, pack_deform_tail_weights_v3
    deformation_module.nvtx_range = contextlib.nullcontext
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load_parser = argparse.ArgumentParser()
    mp, hp, pp = ModelParams(load_parser, sentinel=True), ModelHiddenParams(load_parser), PipelineParams(load_parser)
    saved = sys.argv
    sys.argv = [saved[0], "--model_path", args.model_path, "--source_path", args.source_path]
    try:
        config = get_combined_args(load_parser)
    finally:
        sys.argv = saved
    config = merge_hparams(config, load_config(root / "arguments/dynerf/flame_steak.py"))
    hashes = source_hashes(root, Path(args.model_path))
    with torch.no_grad():
        pc = GaussianModel(mp.extract(config).sh_degree, hp.extract(config))
        scene = Scene(mp.extract(config), pc, load_iteration=14000, shuffle=False)
        net = pc._deformation.eval().requires_grad_(False).deformation_net
        assert len(pc.get_xyz) == 111525 and len(net.feature_out) == 1
        views = [scene.getTestCameras()[i] for i in (0, 12, 24, 36, 49)]
        features = [net.grid.forward_cuda_graph(pc.get_xyz, torch.full(
            (len(pc.get_xyz), 1), float(v.time), device="cuda")).contiguous() for v in views]
        heads = [net.pos_deform, net.scales_deform, net.rotations_deform, net.opacity_deform, net.shs_deform]

        def eager(x):
            hidden = net.feature_out(x)
            return tuple(head(hidden) for head in heads)

        parameters = _cache_full_mlp_parameters(pc)
        packed_tail = pack_deform_tail_weights_v3(parameters[4])
        native_outputs = [torch.empty((len(pc.get_xyz), width), device="cuda") for width in (3, 3, 4, 1, 48)]

        def native(x):
            return deform_mlp_tensor_tail_out_v4(x, *parameters[:4], packed_tail,
                                                parameters[5], native_outputs, 0)

        patch = compiler_compatibility_patch(torch)
        import torch._inductor.config as inductor_config
        inductor_config.triton.cudagraphs = False
        compiled = torch.compile(eager, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
        begin = time.perf_counter()
        compiled(features[0])
        torch.cuda.synchronize()
        setup = time.perf_counter() - begin
        print("COMPILED", setup, flush=True)
        quality = []
        for x in features:
            expected, actual, custom = eager(x), compiled(x), native(x)
            row = {"compile_max_abs": [], "native_max_abs": [], "native_relative_l2": []}
            for reference, candidate, kernel in zip(expected, actual, custom):
                torch.testing.assert_close(candidate, reference, atol=5e-4, rtol=2e-3)
                assert torch.isfinite(kernel).all()
                row["compile_max_abs"].append(float((reference - candidate).abs().max()))
                row["native_max_abs"].append(float((reference - kernel).abs().max()))
                row["native_relative_l2"].append(float((reference - kernel).norm() / reference.norm()))
            quality.append(row)
        static = features[0].clone()

        def capture(fn):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fn(static)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = fn(static)
            stream.synchronize()

            def replay(x):
                static.copy_(x, non_blocking=True)
                graph.replay()
                return result

            return replay

        functions = {"eager_fp32": eager, "compile_fp32": compiled,
                     "compile_graph_fp32": capture(compiled),
                     "native_tensor_tail_v4_graph": capture(native)}
        for fn in functions.values():
            for i in range(20):
                fn(features[i % 5])
            torch.cuda.synchronize()
        records = {name: [] for name in functions}
        names = list(functions)
        for round_index in range(10):
            order = names if round_index % 2 == 0 else list(reversed(names))
            for name in order:
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                wall = time.perf_counter()
                for i in range(50):
                    result = functions[name](features[i % 5])
                    del result
                end.record()
                end.synchronize()
                records[name].append({"wall_ms": (time.perf_counter() - wall) * 1000 / 50,
                                      "cuda_ms": start.elapsed_time(end) / 50})
            print("ROUND", round_index + 1, flush=True)
        summary = {name: {"median_wall_ms": statistics.median(r["wall_ms"] for r in rows),
                          "median_cuda_ms": statistics.median(r["cuda_ms"] for r in rows), "trials": rows}
                   for name, rows in records.items()}
        assert source_hashes(root, Path(args.model_path)) == hashes
        from torch._dynamo.utils import counters
        assert counters["stats"]["unique_graphs"] == 1 and not counters.get("graph_break")
        report = {"passed": True, "scope": "MLP-only: backbone + five heads; no HexPlane, residuals or renderer activations",
                  "points": len(pc.get_xyz), "feature_shape": list(features[0].shape),
                  "feature_view_indices": [0, 12, 24, 36, 49], "independent_processes": 1,
                  "protocol": "10 mirrored rounds x 50 calls; 20 warmup per arm; Graph input copy included",
                  "native_variant": "standalone Tensor-tail v4; not v11 mixed adapter",
                  "precision": "eager/compile FP32 no TF32; native uses existing FP16 Tensor Core boundaries with FP32 accumulation/output",
                  "compiler_internal_cudagraphs": False, "compiler_fullgraph": True,
                  "compiler_compatibility_patch": patch, "compile_setup_seconds": setup,
                  "quality": quality, "measurements": summary, "source_sha256": hashes,
                  "gpu_before": before, "gpu_after": gpu_snapshot(0), "torch": torch.__version__,
                  "loaded_extension": __import__('tacker_4dgs_head')._extension().__file__}
        report["loaded_extension_sha256"] = hashlib.sha256(Path(args.native_extension).read_bytes()).hexdigest()
        write_json(args.output, report)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
