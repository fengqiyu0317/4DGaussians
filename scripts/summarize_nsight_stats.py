#!/usr/bin/env python3
"""Summarize the combined CSV stream emitted by ``nsys stats``."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


REPORT_RE = re.compile(r"/([^/]+)\.py\]\.{3}\s*$")
MAIN_STAGES = (
    "renderer/setup",
    "renderer/deformation",
    "renderer/activation",
    "renderer/rasterization",
)
DEFORMATION_STAGES = (
    "deformation/positional_encoding",
    "deformation/hexplane_feature_sampling",
    "deformation/backbone_mlp",
    "deformation/heads_and_residuals",
)


def parse_number(value):
    value = value.strip()
    if not value:
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def parse_reports(path):
    reports = {}
    current_name = None
    current_lines = []

    def finish():
        nonlocal current_name, current_lines
        if current_name is not None and current_lines:
            reports[current_name] = [
                {key: parse_number(value or "") for key, value in row.items()}
                for row in csv.DictReader(current_lines)
            ]
        current_name = None
        current_lines = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("Processing ["):
            finish()
            match = REPORT_RE.search(raw_line)
            current_name = match.group(1) if match else None
        elif raw_line.startswith("Generating SQLite file"):
            continue
        elif current_name is not None:
            if raw_line.strip():
                current_lines.append(raw_line)
            else:
                finish()
    finish()
    return reports


def index_by(rows, key):
    return {str(row[key]).lstrip(":"): row for row in rows}


def ns_to_ms(value):
    return float(value) / 1_000_000.0


def percent(part, total):
    return 100.0 * part / total if total else 0.0


def kernel_category(name):
    lower = name.lower()
    if "tacker_mix_render_head" in lower:
        return "tacker_mixed_render_head"
    if "tacker_head_linear_solo" in lower:
        return "tacker_head_solo"
    if "tacker_head_linear_gptb" in lower:
        return "tacker_head_gptb"
    if "rendercuda" in lower:
        return "raster_render"
    if "preprocesscuda" in lower:
        return "raster_preprocess"
    if "duplicatewithkeys" in lower:
        return "duplicate_with_keys"
    if "deviceradixsort" in lower:
        return "radix_sort"
    if "grid_sampler" in lower:
        return "hexplane_grid_sample"
    if "sgemm" in lower or "gemm" in lower:
        return "gemm"
    if "catarraybatchedcopy" in lower:
        return "concat_copy"
    if "relu" in lower or "clamp" in lower:
        return "activation"
    if "vectorized_elementwise" in lower or "elementwise_kernel" in lower:
        return "elementwise"
    if "copy" in lower:
        return "copy"
    return "other"


def summarize(path, metadata_path=None):
    reports = parse_reports(path)
    required = {
        "nvtx_sum",
        "nvtx_gpu_proj_sum",
        "cuda_api_sum",
        "cuda_gpu_kern_sum",
        "cuda_gpu_mem_time_sum",
    }
    missing = sorted(required - reports.keys())
    if missing:
        raise ValueError("missing Nsight reports: {}".format(", ".join(missing)))

    nvtx = index_by(reports["nvtx_sum"], "Range")
    projected = index_by(reports["nvtx_gpu_proj_sum"], "Range")
    loop = nvtx["profile/render_loop"]
    loop_projected = projected["profile/render_loop"]
    frame_rows = [row for name, row in nvtx.items() if name.startswith("profile/frame_")]
    frame_count = len(frame_rows)
    if frame_count == 0:
        raise ValueError("no profile/frame_* NVTX ranges found")

    loop_ms = ns_to_ms(loop["Total Time (ns)"])
    projected_loop_ms = ns_to_ms(loop_projected["Total Proj Time (ns)"])
    frame_values_ms = sorted(ns_to_ms(row["Total Time (ns)"]) for row in frame_rows)

    def stage_summary(name):
        cpu_row = nvtx[name]
        gpu_row = projected[name]
        cpu_total = ns_to_ms(cpu_row["Total Time (ns)"])
        gpu_total = ns_to_ms(gpu_row["Total Proj Time (ns)"])
        return {
            "instances": int(cpu_row["Instances"]),
            "cpu_total_ms": cpu_total,
            "cpu_ms_per_frame": cpu_total / frame_count,
            "cpu_loop_percent": percent(cpu_total, loop_ms),
            "gpu_projected_total_ms": gpu_total,
            "gpu_projected_ms_per_frame": gpu_total / frame_count,
            "gpu_projected_loop_percent": percent(gpu_total, projected_loop_ms),
            "gpu_ops": int(gpu_row["Total GPU Ops"]),
        }

    kernel_rows = reports["cuda_gpu_kern_sum"]
    kernel_total_ns = sum(float(row["Total Time (ns)"]) for row in kernel_rows)
    kernel_categories = defaultdict(lambda: {"time_ns": 0.0, "instances": 0})
    for row in kernel_rows:
        bucket = kernel_categories[kernel_category(str(row["Name"]))]
        bucket["time_ns"] += float(row["Total Time (ns)"])
        bucket["instances"] += int(row["Instances"])

    categories = {
        name: {
            "instances": int(values["instances"]),
            "total_ms": ns_to_ms(values["time_ns"]),
            "ms_per_frame": ns_to_ms(values["time_ns"]) / frame_count,
            "kernel_percent": percent(values["time_ns"], kernel_total_ns),
        }
        for name, values in sorted(
            kernel_categories.items(),
            key=lambda item: float(item[1]["time_ns"]),
            reverse=True,
        )
    }
    top_kernels = [
        {
            "name": str(row["Name"]),
            "instances": int(row["Instances"]),
            "total_ms": ns_to_ms(row["Total Time (ns)"]),
            "ms_per_frame": ns_to_ms(row["Total Time (ns)"]) / frame_count,
            "kernel_percent": percent(float(row["Total Time (ns)"]), kernel_total_ns),
        }
        for row in sorted(kernel_rows, key=lambda row: float(row["Total Time (ns)"]), reverse=True)[:20]
    ]

    api_rows = reports["cuda_api_sum"]
    api_total_ns = sum(float(row["Total Time (ns)"]) for row in api_rows)
    api_by_name = index_by(api_rows, "Name")
    sync = api_by_name.get("cudaStreamSynchronize")
    launch = api_by_name.get("cudaLaunchKernel")
    mem_total_ns = sum(float(row["Total Time (ns)"]) for row in reports["cuda_gpu_mem_time_sum"])

    main_stage_values = {name: stage_summary(name) for name in MAIN_STAGES}
    marked_cpu_ms = sum(stage["cpu_total_ms"] for stage in main_stage_values.values())
    marked_gpu_ms = sum(stage["gpu_projected_total_ms"] for stage in main_stage_values.values())
    deformation_values = {
        name: stage_summary(name)
        for name in DEFORMATION_STAGES
        if name in nvtx and name in projected
    }

    result = {
        "source": str(path),
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path else {},
        "frame_count": frame_count,
        "render_loop": {
            "total_ms": loop_ms,
            "ms_per_frame": loop_ms / frame_count,
            "fps": 1000.0 * frame_count / loop_ms,
            "gpu_projected_total_ms": projected_loop_ms,
            "gpu_projected_ms_per_frame": projected_loop_ms / frame_count,
        },
        "frame_nvtx": {
            "mean_ms": sum(frame_values_ms) / frame_count,
            "stddev_ms": statistics.pstdev(frame_values_ms),
            "min_ms": frame_values_ms[0],
            "median_ms": statistics.median(frame_values_ms),
            "max_ms": frame_values_ms[-1],
        },
        "main_stages": main_stage_values,
        "main_stage_unmarked": {
            "cpu_total_ms": loop_ms - marked_cpu_ms,
            "cpu_ms_per_frame": (loop_ms - marked_cpu_ms) / frame_count,
            "cpu_loop_percent": percent(loop_ms - marked_cpu_ms, loop_ms),
            "gpu_projected_total_ms": projected_loop_ms - marked_gpu_ms,
            "gpu_projected_ms_per_frame": (projected_loop_ms - marked_gpu_ms) / frame_count,
            "gpu_projected_loop_percent": percent(projected_loop_ms - marked_gpu_ms, projected_loop_ms),
        },
        "deformation_stages": deformation_values,
        "kernels": {
            "total_ms": ns_to_ms(kernel_total_ns),
            "ms_per_frame": ns_to_ms(kernel_total_ns) / frame_count,
            "launches": sum(int(row["Instances"]) for row in kernel_rows),
            "launches_per_frame": sum(int(row["Instances"]) for row in kernel_rows) / frame_count,
            "categories": categories,
            "top": top_kernels,
        },
        "cuda_api": {
            "total_ms": ns_to_ms(api_total_ns),
            "stream_synchronize": {
                "calls": int(sync["Num Calls"]) if sync else 0,
                "calls_per_frame": int(sync["Num Calls"]) / frame_count if sync else 0.0,
                "total_ms": ns_to_ms(sync["Total Time (ns)"]) if sync else 0.0,
            },
            "kernel_launch": {
                "calls": int(launch["Num Calls"]) if launch else 0,
                "calls_per_frame": int(launch["Num Calls"]) / frame_count if launch else 0.0,
                "total_ms": ns_to_ms(launch["Total Time (ns)"]) if launch else 0.0,
            },
        },
        "gpu_memory_operations": {
            "total_ms": ns_to_ms(mem_total_ns),
            "ms_per_frame": ns_to_ms(mem_total_ns) / frame_count,
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats_csv", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    result = summarize(args.stats_csv, args.metadata)
    payload = json.dumps(result, ensure_ascii=False, indent=args.indent) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
