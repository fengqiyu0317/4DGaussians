"""Synthetic, MLP-only Jetson comparison of exact v11 device-v7 and PyTorch.

The native wrapper is not the mixed renderer. A separate 384-thread control
retains its thread/shared-memory envelope, but executes no Raster work/pairing.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from benchmark_original_deform_compile import (
    compiler_compatibility_patch, gpu_snapshot, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=111525)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--calls", type=int, default=40)
    args = parser.parse_args()
    before = gpu_snapshot(0, require_idle=True)
    source_paths = [Path(__file__), Path(args.header), Path(args.library),
                    Path(__file__).with_name("deform_v11_standalone_adapter.cu")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    benchmark_stream = torch.cuda.Stream()
    torch.cuda.set_stream(benchmark_stream)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    widths = (3, 3, 4, 1, 48)
    offsets = (0, 3, 6, 10, 11, 59)

    # Generate once on CPU. Tensor sizes, seed, distributions, and every source
    # byte are shared across all arms and independent process repeats.
    cpu = {
        "features": torch.randn(args.rows, 32),
        "backbone_weight": torch.randn(32, 128) / (32 ** 0.5),
        "backbone_bias": torch.randn(128) * 0.05,
        "first_weights": torch.randn(5, 128, 128) / (128 ** 0.5),
        "first_biases": torch.randn(5, 128) * 0.05,
        "tail_weights": torch.randn(59, 128) / (128 ** 0.5),
        "tail_biases": torch.randn(59) * 0.05,
    }
    data_hashes = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in cpu.items()}
    tensors = {k: v.cuda() for k, v in cpu.items()}
    x, bw, bb, fw, fb, tw, tb = (tensors[k] for k in cpu)
    fwh, twh = fw.half(), tw.half()
    packed = torch.zeros(112, 128, dtype=torch.float16, device="cuda")
    for head, start in enumerate((0, 16, 32, 48, 64)):
        packed[start:start + widths[head]].copy_(twh[offsets[head]:offsets[head + 1]])
    first = tuple(fw[h] for h in range(5))
    first_half = tuple(fwh[h] for h in range(5))
    first_bias = tuple(fb[h] for h in range(5))
    tails = tuple(tw[offsets[h]:offsets[h+1]] for h in range(5))
    tails_half = tuple(twh[offsets[h]:offsets[h+1]] for h in range(5))
    tail_bias = tuple(tb[offsets[h]:offsets[h+1]] for h in range(5))

    # This Jetson Torch 2.8 wheel implements mm(out_dtype=float32), but its
    # mm.dtype Meta kernel has the old two-argument signature. Keep the actual
    # CUDA operation intact and expose its shape/dtype through a registered
    # opaque op. Inductor can fuse the surrounding activation/bias operations.
    # This compatibility bridge is used in every matched-precision arm.
    @torch.library.custom_op("v11_bench::mm_half_float", mutates_args=())
    def mm_half_float(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mm(a, b, out_dtype=torch.float32)

    @mm_half_float.register_fake
    def mm_half_float_fake(a, b):
        return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.float32)

    def eager_fp32(features):
        hidden = F.linear(features, bw.T, bb)
        return tuple(F.linear(F.relu(F.linear(F.relu(hidden), first[h], first_bias[h])),
                              tails[h], tail_bias[h]) for h in range(5))

    def eager_matched(features):
        # Match v11's exact FP16 boundaries: FP32 backbone; half activation and
        # weights; FP32 GEMM output; bias/ReLU; half activation; FP32 tail output.
        hidden = F.relu(F.linear(features, bw.T, bb)).half()
        outputs = []
        for h in range(5):
            inner = mm_half_float(hidden, first_half[h].T)
            active = F.relu(inner + first_bias[h]).half()
            result = mm_half_float(active, tails_half[h].T)
            outputs.append(result + tail_bias[h])
        return tuple(outputs)

    library = ctypes.CDLL(str(Path(args.library).resolve()))
    launch = library.deform_v11_launch
    launch.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
    launch.restype = ctypes.c_int
    resources = library.deform_v11_resources
    resources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    resources.restype = ctypes.c_int
    resource_info = {}
    for control in (0, 1):
        values = (ctypes.c_int * 5)()
        assert resources(control, values) == 0
        resource_info[str(control)] = dict(zip(
            ("registers_per_thread", "shared_bytes", "local_bytes", "max_threads", "active_ctas_per_sm"),
            list(values)))
    output_sets = [tuple(torch.empty(args.rows, w, device="cuda") for w in widths) for _ in range(2)]
    stream_pointer = torch.cuda.current_stream().cuda_stream
    native_pointers = [v.data_ptr() for v in (x, bw, bb, fwh, fb, packed, tb)]

    def native(control):
        outputs = output_sets[control]
        pointers = native_pointers + [v.data_ptr() for v in outputs]
        def run(_features):
            assert _features.data_ptr() == x.data_ptr()
            status = launch(*pointers, args.rows, control, stream_pointer)
            if status:
                raise RuntimeError(f"CUDA native launch error {status}")
            return outputs
        return run

    def compare(reference, candidate, atol, rtol):
        result = []
        for a, b in zip(reference, candidate):
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
            delta = (a-b).float()
            result.append({"max_abs": float(delta.abs().max()),
                           "relative_l2": float(delta.norm() / a.float().norm()),
                           "exact": bool(torch.equal(a, b))})
        return result

    capture_setup = {}
    def capture(name, fn):
        # Capture on the current stream because the ctypes adapter explicitly
        # receives that stream. No hidden framework stream assumption is used.
        for _ in range(4):
            fn(x)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            outputs = fn(x)
        torch.cuda.synchronize()
        capture_setup[name] = time.perf_counter() - begin
        def replay(_features):
            assert _features.data_ptr() == x.data_ptr()
            graph.replay()
            return outputs
        return replay

    with torch.inference_mode():
        quality = {}
        reference_fp32, reference_matched = eager_fp32(x), eager_matched(x)
        kernel128, kernel384 = native(0), native(1)
        quality["native_vs_matched"] = compare(reference_matched, kernel128(x), 2e-3, 3e-3)
        quality["native_384_vs_128"] = compare(kernel128(x), kernel384(x), 2e-5, 2e-5)
        quality["native_vs_original_fp32"] = compare(reference_fp32, kernel128(x), 5e-3, 5e-3)
        patch = compiler_compatibility_patch(torch)
        import torch._inductor.config as config
        config.triton.cudagraphs = False
        functions = {"eager_fp32": eager_fp32, "eager_graph_fp32": capture("eager_fp32", eager_fp32)}
        compile_setup = {}
        for precision, fn, reference in (("fp32", eager_fp32, reference_fp32),
                                          ("matched", eager_matched, reference_matched)):
            compiled = torch.compile(fn, fullgraph=True, dynamic=False,
                                     mode="max-autotune-no-cudagraphs")
            begin = time.perf_counter()
            candidate = compiled(x)
            torch.cuda.synchronize()
            compile_setup[precision] = time.perf_counter() - begin
            quality["compile_" + precision] = compare(reference, candidate, 2e-3, 3e-3)
            functions["compile_" + precision] = compiled
            functions["compile_graph_" + precision] = capture("compile_" + precision, compiled)
            print("COMPILED", precision, compile_setup[precision], flush=True)
        functions.update({"eager_matched": eager_matched,
                          "eager_graph_matched": capture("eager_matched", eager_matched),
                          "v11_standalone": kernel128,
                          "v11_standalone_graph": capture("native128", kernel128),
                          "v11_resource_control_graph": capture("native384", kernel384)})
        for name, fn in functions.items():
            ref = reference_fp32 if name.endswith("fp32") else reference_matched
            quality["arm_" + name] = compare(ref, fn(x), 2e-3, 3e-3)
            for _ in range(12):
                fn(x)
            torch.cuda.synchronize()
        records = {name: [] for name in functions}
        names = list(functions)
        for round_index in range(args.rounds):
            order = names if round_index % 2 == 0 else names[::-1]
            for name in order:
                fn = functions[name]
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter()
                start.record()
                for _ in range(args.calls):
                    outputs = fn(x)
                end.record()
                end.synchronize()
                records[name].append({"cuda_ms": start.elapsed_time(end) / args.calls,
                                      "wall_ms": (time.perf_counter()-wall)*1000 / args.calls})
            print("ROUND", round_index + 1, flush=True)
        summary = {name: {"median_cuda_ms": statistics.median(r["cuda_ms"] for r in trials),
                          "median_wall_ms": statistics.median(r["wall_ms"] for r in trials),
                          "trials": trials} for name, trials in records.items()}
        from torch._dynamo.utils import counters
        assert counters["stats"]["unique_graphs"] == 2, dict(counters)
        assert not counters.get("graph_break"), dict(counters)
        assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths} == hashes
        report = {"passed": True, "scope": "MLP-only: 32->128 backbone; five 128->128->(3,3,4,1,48) heads",
                  "excluded": ["HexPlane", "position encoding", "residual application", "Raster", "input/output transfers", "weight packing", "compilation and capture"],
                  "qualification_profile": None, "mixed_renderer_executed": False,
                  "v11_device_entry": "deform_mlp_tensor_tail_gptb_device_v7",
                  "native_adapter": "128 threads; same v11 device source, extracted standalone; one 16-row tile per CTA",
                  "resource_control": "384 threads/25216B shared; no Raster, no candidate-aware production pairing; one 16-row tile per CTA",
                  "rows": args.rows, "seed": args.seed, "data_sha256": data_hashes,
                  "distribution": "features N(0,1), backbone weight N(0,1/sqrt(32)), head/tail weights N(0,1/sqrt(128)), biases N(0,.05)",
                  "precision_fp32": "FP32 weights/activations/output, TF32 disabled",
                  "precision_matched": "FP32 backbone, FP16 head weights and activations, FP32 accumulation/bias/output; torch.mm out_dtype=float32",
                  "matched_precision_compiler_bridge": "registered opaque mm_half_float op with fake shape/dtype; runtime calls unchanged torch.mm(out_dtype=float32); needed for wheel's outdated mm.dtype Meta signature; used in all matched arms",
                  "fp16_reduced_precision_reduction": False,
                  "protocol": f"{args.rounds} mirrored rounds x {args.calls} calls, fixed input; 12 warmup per arm; CUDA events and synchronized wall time",
                  "compiler_mode": "max-autotune-no-cudagraphs", "compiler_fullgraph": True,
                  "compiler_dynamic": False, "compiler_internal_cudagraphs": False,
                  "compiler_compatibility_patch": patch, "compile_setup_seconds": compile_setup,
                  "capture_setup_seconds": capture_setup, "dynamo_unique_graphs": 2,
                  "cuda_graph_arms": [name for name in functions if "graph" in name],
                  "quality": quality, "measurements": summary, "resources": resource_info,
                  "source_sha256": hashes, "gpu_before": before, "gpu_after": gpu_snapshot(0),
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "device": torch.cuda.get_device_name(0), "pid": os.getpid()}
        write_json(args.output, report)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
