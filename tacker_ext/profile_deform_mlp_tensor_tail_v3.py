"""Compare the retained scalar-tail v1 with experimental Tensor-Core-tail v3."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import torch

from tacker_4dgs_head import (
    deform_mlp_delta_out_v1,
    deform_mlp_tensor_tail_out_v3,
    pack_deform_tail_weights_v3,
)


WIDTHS = (3, 3, 4, 1, 48)


def make_case(rows: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260927 + rows)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", generator=generator)

    return (
        randn(rows, 32),
        randn(32, 128) / math.sqrt(32),
        randn(128) * 0.1,
        (randn(5, 128, 128) / math.sqrt(128)).half().contiguous(),
        randn(5, 128) * 0.1,
        (randn(59, 128) / math.sqrt(128)).contiguous(),
        randn(59) * 0.1,
    )


def make_outputs(rows: int) -> tuple[torch.Tensor, ...]:
    return tuple(torch.empty((rows, width), device="cuda") for width in WIDTHS)


def launch_v1(args, outputs, blocks: int) -> None:
    deform_mlp_delta_out_v1(*args, outputs, blocks)


def launch_v3(args, packed, outputs, blocks: int) -> None:
    deform_mlp_tensor_tail_out_v3(
        *args[:5], packed, args[6], outputs, blocks)


def comparison(v1, v3) -> dict[str, object]:
    per_head = []
    total_error2 = 0.0
    total_reference2 = 0.0
    for head, (baseline, candidate) in enumerate(zip(v1, v3)):
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
        })
    return {
        "baseline": "deform_mlp_delta_v1",
        "candidate": "deform_mlp_tensor_tail_v3",
        "overall_relative_l2": math.sqrt(total_error2 / total_reference2)
        if total_reference2 else 0.0,
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


def benchmark(args, packed, v1_outputs, v3_outputs, blocks, warmup,
              iterations, cycles) -> dict[str, object]:
    v1_fn = lambda: launch_v1(args, v1_outputs, blocks)
    v3_fn = lambda: launch_v3(args, packed, v3_outputs, blocks)
    for index in range(warmup):
        (v1_fn if index % 2 == 0 else v3_fn)()
    torch.cuda.synchronize()

    samples = {"v1": [], "v3": []}
    # Repeated ABBA gives both variants the same exposure to drift.
    for _ in range(cycles):
        for name, fn in (("v1", v1_fn), ("v3", v3_fn),
                         ("v3", v3_fn), ("v1", v1_fn)):
            samples[name].append(timed(fn, iterations))
    medians = {name: statistics.median(values)
               for name, values in samples.items()}
    return {
        "design": "repeated ABBA",
        "cycles": cycles,
        "iterations_per_sample": iterations,
        "samples_milliseconds": samples,
        "median_milliseconds": medians,
        "v3_speedup_over_v1": medians["v1"] / medians["v3"],
        "v3_time_reduction_percent":
            (1.0 - medians["v3"] / medians["v1"]) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "benchmark", "profile"),
                        required=True)
    parser.add_argument("--variant", choices=("v1", "v3"), default="v3")
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
        v1_outputs = make_outputs(parsed.rows)
        v3_outputs = make_outputs(parsed.rows)
        environment = {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": actual,
            "rows": parsed.rows,
            "tiles": (parsed.rows + 15) // 16,
            "persistent_blocks": parsed.persistent_blocks,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }

        if parsed.mode == "check":
            launch_v1(args, v1_outputs, parsed.persistent_blocks)
            launch_v3(args, packed, v3_outputs, parsed.persistent_blocks)
            torch.cuda.synchronize()
            result = {"mode": "check", "environment": environment,
                      "accuracy": comparison(v1_outputs, v3_outputs)}
        elif parsed.mode == "benchmark":
            result = {
                "mode": "benchmark",
                "environment": environment,
                "timing": benchmark(
                    args, packed, v1_outputs, v3_outputs,
                    parsed.persistent_blocks, parsed.warmup,
                    parsed.iterations, parsed.cycles),
            }
        else:
            fn = (lambda: launch_v1(
                args, v1_outputs, parsed.persistent_blocks)) \
                if parsed.variant == "v1" else (lambda: launch_v3(
                    args, packed, v3_outputs, parsed.persistent_blocks))
            for _ in range(parsed.warmup):
                fn()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push("tensor_tail_{}_profile".format(
                parsed.variant))
            fn()
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            result = {"mode": "profile", "environment": environment,
                      "variant": parsed.variant, "target_launches": 1}

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
