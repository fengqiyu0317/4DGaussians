"""Correctness, timing, and Nsight Compute driver for deform_mlp_delta_v1.

Run this script on the target sm_86 GPU after building the extension in place.
The ``profile`` mode intentionally launches exactly one target kernel after the
requested warm-up launches so Nsight Compute can select it with
``--launch-skip WARMUP --launch-count 1``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Iterable, Sequence

import torch

from tacker_4dgs_head import deform_mlp_delta_out_v1, deform_mlp_delta_v1


WIDTHS = (3, 3, 4, 1, 48)
OFFSETS = (0, 3, 6, 10, 11, 59)


def make_case(rows: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(20260916 + rows)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, generator=generator)

    return (
        randn(rows, 32),
        randn(32, 128) / math.sqrt(32),
        randn(128) * 0.1,
        (randn(5, 128, 128) / math.sqrt(128)).half().contiguous(),
        randn(5, 128) * 0.1,
        randn(59, 128) / math.sqrt(128),
        randn(59) * 0.1,
    )


def make_outputs(rows: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(torch.empty((rows, width), device=device) for width in WIDTHS)


def reference(args: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    features, w0, b0, weights, biases, tails, tail_bias = args
    activated = torch.relu(features @ w0 + b0).half().float()
    values = []
    for head in range(5):
        hidden = torch.relu(
            activated @ weights[head].float().t() + biases[head]
        )
        lo, hi = OFFSETS[head : head + 2]
        values.append(hidden @ tails[lo:hi].t() + tail_bias[lo:hi])
    return tuple(values)


def validate(
    args: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]
) -> dict[str, object]:
    expected = reference(args)
    maximum_absolute_errors = []
    for actual, wanted in zip(outputs, expected):
        torch.testing.assert_close(actual, wanted, rtol=2e-3, atol=5e-4)
        maximum_absolute_errors.append(float((actual - wanted).abs().max().item()))
    return {
        "passed": True,
        "rtol": 2e-3,
        "atol": 5e-4,
        "maximum_absolute_errors": maximum_absolute_errors,
    }


def unique(values: Iterable[int]) -> list[int]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def benchmark_one(
    args: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    persistent_blocks: int,
    warmup: int,
    iterations: int,
    rounds: int,
) -> dict[str, object]:
    for _ in range(warmup):
        deform_mlp_delta_out_v1(*args, outputs, persistent_blocks)
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            deform_mlp_delta_out_v1(*args, outputs, persistent_blocks)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) / iterations)

    return {
        "persistent_blocks": persistent_blocks,
        "milliseconds_per_launch": samples,
        "median_milliseconds": statistics.median(samples),
        "min_milliseconds": min(samples),
        "max_milliseconds": max(samples),
    }


def cache_pair_benchmark(
    args: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    persistent_blocks: int,
    warmup: int,
    pairs: int,
    eviction_mib: int,
) -> dict[str, object]:
    eviction_elements = eviction_mib * 1024 * 1024 // 4
    eviction = torch.ones(eviction_elements, device=outputs[0].device)
    for _ in range(warmup):
        deform_mlp_delta_out_v1(*args, outputs, persistent_blocks)
    torch.cuda.synchronize()

    cold_samples = []
    warm_samples = []
    for _ in range(pairs):
        # Touch much more than the A6000 L2 before recording the cold launch.
        # Stream ordering guarantees that this finishes before cold_start, but
        # its own time is outside the measured event interval.
        eviction.add_(1.0)
        cold_start = torch.cuda.Event(enable_timing=True)
        cold_end = torch.cuda.Event(enable_timing=True)
        warm_start = torch.cuda.Event(enable_timing=True)
        warm_end = torch.cuda.Event(enable_timing=True)
        cold_start.record()
        deform_mlp_delta_out_v1(*args, outputs, persistent_blocks)
        cold_end.record()
        warm_start.record()
        deform_mlp_delta_out_v1(*args, outputs, persistent_blocks)
        warm_end.record()
        warm_end.synchronize()
        cold_samples.append(float(cold_start.elapsed_time(cold_end)))
        warm_samples.append(float(warm_start.elapsed_time(warm_end)))

    cold_median = statistics.median(cold_samples)
    warm_median = statistics.median(warm_samples)
    return {
        "persistent_blocks": persistent_blocks,
        "pairs": pairs,
        "eviction_mib": eviction_mib,
        "cold_milliseconds": cold_samples,
        "warm_milliseconds": warm_samples,
        "cold_median_milliseconds": cold_median,
        "warm_median_milliseconds": warm_median,
        "cold_over_warm_ratio": cold_median / warm_median,
    }


def environment(rows: int) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "sm_count": properties.multi_processor_count,
        "rows": rows,
        "tiles": (rows + 15) // 16,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("check", "benchmark", "cache", "profile"), required=True
    )
    parser.add_argument("--rows", type=int, default=111525)
    parser.add_argument("--persistent-blocks", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--eviction-mib", type=int, default=64)
    parsed = parser.parse_args()

    if parsed.rows < 0 or parsed.persistent_blocks < 0:
        parser.error("rows and persistent-blocks must be non-negative")
    if parsed.warmup < 0 or parsed.iterations <= 0 or parsed.rounds <= 0:
        parser.error("warmup must be non-negative; iterations and rounds must be positive")
    if parsed.eviction_mib <= 0:
        parser.error("eviction-mib must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_capability(0) != (8, 6):
        raise RuntimeError("deform_mlp_delta_v1 profiling requires sm_86")

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", 0)

    try:
        with torch.no_grad():
            args = make_case(parsed.rows, device)
            outputs = make_outputs(parsed.rows, device)
            metadata = environment(parsed.rows)

            if parsed.mode == "check":
                actual = deform_mlp_delta_v1(*args, parsed.persistent_blocks)
                result = {
                    "mode": "check",
                    "environment": metadata,
                    "correctness": validate(args, actual),
                }
            elif parsed.mode == "benchmark":
                sm_count = int(metadata["sm_count"])
                tiles = int(metadata["tiles"])
                requested = unique((0, sm_count, 2 * sm_count, 4 * sm_count,
                                    8 * sm_count, tiles))
                timings = [
                    benchmark_one(
                        args,
                        outputs,
                        blocks,
                        parsed.warmup,
                        parsed.iterations,
                        parsed.rounds,
                    )
                    for blocks in requested
                ]
                result = {
                    "mode": "benchmark",
                    "environment": metadata,
                    "warmup": parsed.warmup,
                    "iterations_per_round": parsed.iterations,
                    "rounds": parsed.rounds,
                    "timings": timings,
                }
            elif parsed.mode == "cache":
                result = {
                    "mode": "cache",
                    "environment": metadata,
                    "cache_comparison": cache_pair_benchmark(
                        args,
                        outputs,
                        parsed.persistent_blocks,
                        parsed.warmup,
                        parsed.iterations,
                        parsed.eviction_mib,
                    ),
                }
            else:
                for _ in range(parsed.warmup):
                    deform_mlp_delta_out_v1(
                        *args, outputs, parsed.persistent_blocks
                    )
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("deform_mlp_profile")
                deform_mlp_delta_out_v1(*args, outputs, parsed.persistent_blocks)
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                result = {
                    "mode": "profile",
                    "environment": metadata,
                    "warmup": parsed.warmup,
                    "persistent_blocks": parsed.persistent_blocks,
                    "target_launches": 1,
                }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
