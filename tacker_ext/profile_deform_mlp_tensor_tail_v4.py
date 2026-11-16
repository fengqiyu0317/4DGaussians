"""Compare unpadded Tensor-Core-tail v3 with padded shared-input v4."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import torch

from profile_deform_mlp_tensor_tail_v3 import WIDTHS, make_case, make_outputs
from tacker_4dgs_head import (
    deform_mlp_tensor_tail_out_v3,
    deform_mlp_tensor_tail_out_v4,
    pack_deform_tail_weights_v3,
)


def launch_v3(args, packed, outputs, blocks: int) -> None:
    deform_mlp_tensor_tail_out_v3(
        *args[:5], packed, args[6], outputs, blocks)


def launch_v4(args, packed, outputs, blocks: int) -> None:
    deform_mlp_tensor_tail_out_v4(
        *args[:5], packed, args[6], outputs, blocks)


def comparison(v3, v4) -> dict[str, object]:
    per_head = []
    total_error2 = 0.0
    total_reference2 = 0.0
    for head, (baseline, candidate) in enumerate(zip(v3, v4)):
        error = candidate - baseline
        error2 = float(torch.sum(error.double() * error.double()).item())
        reference2 = float(
            torch.sum(baseline.double() * baseline.double()).item())
        total_error2 += error2
        total_reference2 += reference2
        per_head.append({
            "head": head,
            "width": WIDTHS[head],
            "max_absolute_error": float(error.abs().max().item()),
            "mean_absolute_error": float(error.abs().mean().item()),
            "rmse": math.sqrt(error2 / max(1, error.numel())),
            "relative_l2": math.sqrt(error2 / reference2)
            if reference2 else 0.0,
            "bitwise_equal": bool(torch.equal(baseline, candidate)),
        })
    return {
        "baseline": "deform_mlp_tensor_tail_v3",
        "candidate": "deform_mlp_tensor_tail_padded_v4",
        "overall_relative_l2": math.sqrt(total_error2 / total_reference2)
        if total_reference2 else 0.0,
        "all_heads_bitwise_equal": all(
            item["bitwise_equal"] for item in per_head),
        "per_head": per_head,
    }


def timed(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def benchmark(args, packed, v3_outputs, v4_outputs, blocks, warmup,
              iterations, cycles) -> dict[str, object]:
    v3_fn = lambda: launch_v3(args, packed, v3_outputs, blocks)
    v4_fn = lambda: launch_v4(args, packed, v4_outputs, blocks)
    for index in range(warmup):
        (v3_fn if index % 2 == 0 else v4_fn)()
    torch.cuda.synchronize()

    samples = {"v3": [], "v4": []}
    for _ in range(cycles):
        for name, fn in (("v3", v3_fn), ("v4", v4_fn),
                         ("v4", v4_fn), ("v3", v3_fn)):
            samples[name].append(timed(fn, iterations))
    medians = {name: statistics.median(values)
               for name, values in samples.items()}
    return {
        "design": "repeated ABBA",
        "cycles": cycles,
        "iterations_per_sample": iterations,
        "samples_milliseconds": samples,
        "median_milliseconds": medians,
        "v4_speedup_over_v3": medians["v3"] / medians["v4"],
        "v4_time_reduction_percent":
            (1.0 - medians["v4"] / medians["v3"]) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "benchmark", "profile"),
                        required=True)
    parser.add_argument("--variant", choices=("v3", "v4"), default="v4")
    parser.add_argument("--rows", type=int, default=111525)
    parser.add_argument("--persistent-blocks", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--cycles", type=int, default=5)
    parsed = parser.parse_args()
    if parsed.rows < 0 or parsed.persistent_blocks < 0:
        parser.error("rows and persistent-blocks must be non-negative")
    if parsed.warmup < 0 or parsed.iterations <= 0 or parsed.cycles <= 0:
        parser.error("warmup must be non-negative; iterations/cycles positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    expected = os.environ.get("TACKER_CUDA_ARCH", "sm_86")
    actual = "sm_{}{}".format(*torch.cuda.get_device_capability(0))
    if actual != expected:
        raise RuntimeError("build target {} does not match {}".format(
            expected, actual))

    with torch.no_grad():
        args = make_case(parsed.rows)
        packed = pack_deform_tail_weights_v3(args[5])
        v3_outputs = make_outputs(parsed.rows)
        v4_outputs = make_outputs(parsed.rows)
        blocks = parsed.persistent_blocks
        environment = {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": actual,
            "rows": parsed.rows,
            "tiles": (parsed.rows + 15) // 16,
            "persistent_blocks": blocks,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }

        if parsed.mode == "check":
            launch_v3(args, packed, v3_outputs, blocks)
            launch_v4(args, packed, v4_outputs, blocks)
            torch.cuda.synchronize()
            result = {
                "mode": "check",
                "environment": environment,
                "accuracy": comparison(v3_outputs, v4_outputs),
            }
        elif parsed.mode == "benchmark":
            result = {
                "mode": "benchmark",
                "environment": environment,
                "timing": benchmark(
                    args, packed, v3_outputs, v4_outputs, blocks,
                    parsed.warmup, parsed.iterations, parsed.cycles),
            }
        else:
            fn = (lambda: launch_v3(args, packed, v3_outputs, blocks)) \
                if parsed.variant == "v3" else \
                (lambda: launch_v4(args, packed, v4_outputs, blocks))
            for _ in range(parsed.warmup):
                fn()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(
                "tensor_tail_{}_profile".format(parsed.variant))
            fn()
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            result = {
                "mode": "profile",
                "environment": environment,
                "variant": parsed.variant,
                "target_launches": 1,
            }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
