#!/usr/bin/env python3
"""Estimate persistent-block candidates from real tile loads and CTA timings."""

import argparse
import bisect
import csv
import heapq
import json
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_CANDIDATES = (
    3486, 3584, 3712, 3840, 3968, 4096, 4224, 4352, 4480,
    4608, 4736, 4800, 4864, 4992, 5120, 5248, 5376, 5440,
    5600, 5760, 6000, 6400, 6971,
)


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class TimingCurves:
    def __init__(self, path):
        raw = defaultdict(lambda: defaultdict(list))
        with path.open("r", encoding="utf-8") as handle:
            rows = (line for line in handle if not line.startswith("#"))
            for row in csv.DictReader(rows):
                k0 = int(row["raster_k0"])
                k1 = int(row["raster_k1"])
                raster_tasks = int(row["raster_tasks"])
                deform_tasks = int(row["deform_tasks"])
                if k0 != k1:
                    continue
                if raster_tasks == 0:
                    workload = 0
                elif raster_tasks == 1:
                    workload = k0
                elif raster_tasks == 2:
                    workload = k0 + k1
                else:
                    raise ValueError("timing table contains unsupported R count")
                raw[(raster_tasks, deform_tasks)][workload].append(
                    float(row["median_raw_us"])
                )
        self.curves = {}
        for key, by_workload in raw.items():
            points = sorted(
                (workload, statistics.median(samples))
                for workload, samples in by_workload.items()
            )
            self.curves[key] = points

    def evaluate(self, raster_tasks, deform_tasks, raster_workload):
        points = self.curves[(raster_tasks, deform_tasks)]
        if len(points) == 1:
            return points[0][1]
        workloads = [point[0] for point in points]
        upper = bisect.bisect_left(workloads, raster_workload)
        if upper == 0:
            lower, upper = 0, 1
        elif upper == len(points):
            lower, upper = len(points) - 2, len(points) - 1
        else:
            lower = upper - 1
        x0, y0 = points[lower]
        x1, y1 = points[upper]
        if x0 == x1:
            return y0
        fraction = (raster_workload - x0) / float(x1 - x0)
        return y0 + (y1 - y0) * fraction


def validate_pair_sum_model(curves, path):
    errors = []
    with path.open("r", encoding="utf-8") as handle:
        rows = (line for line in handle if not line.startswith("#"))
        for row in csv.DictReader(rows):
            if int(row["raster_tasks"]) != 2:
                continue
            deform_tasks = int(row["deform_tasks"])
            workload = int(row["raster_k0"]) + int(row["raster_k1"])
            observed = float(row["median_raw_us"])
            predicted = curves.evaluate(2, deform_tasks, workload)
            errors.append(abs(predicted - observed) / observed)
    return {
        "samples": len(errors),
        "median_absolute_percentage_error": statistics.median(errors),
        "p90_absolute_percentage_error": percentile(errors, 0.90),
        "max_absolute_percentage_error": max(errors),
    }


def task_count(logical_tasks, physical_block, persistent_blocks):
    if physical_block >= logical_tasks:
        return 0
    return 1 + (logical_tasks - 1 - physical_block) // persistent_blocks


def greedy_makespan(costs, lanes):
    heap = [0.0] * lanes
    heapq.heapify(heap)
    for cost in costs:
        earliest = heapq.heappop(heap)
        heapq.heappush(heap, earliest + cost)
    return max(heap)


def summarize_candidate(records, curves, persistent_blocks, deform_tasks, lanes):
    tile_count = records[0]["tile_count"]
    frame_totals = []
    frame_greedy = []
    frame_maxima = []
    all_costs = []
    type_counts = defaultdict(int)
    raster_dominant = 0
    active_ctas = 0

    for record in records:
        counts = record["range_counts"]
        costs = []
        for physical_block in range(persistent_blocks):
            raster_count = task_count(
                tile_count, physical_block, persistent_blocks
            )
            deform_count = task_count(
                deform_tasks, physical_block, persistent_blocks
            )
            if raster_count > 2 or deform_count > 2:
                raise ValueError(
                    "candidate requires a workload outside the measured 0/1/2 grid"
                )
            raster_indices = [
                physical_block + task * persistent_blocks
                for task in range(raster_count)
            ]
            raster_workload = sum(counts[index] for index in raster_indices)
            cost = curves.evaluate(
                raster_count, deform_count, raster_workload
            )
            costs.append(cost)
            type_counts["{}R+{}D".format(raster_count, deform_count)] += 1
            raster_only = curves.evaluate(raster_count, 0, raster_workload)
            deform_only = curves.evaluate(0, deform_count, 0)
            raster_dominant += raster_only > deform_only
            active_ctas += 1
        frame_totals.append(sum(costs))
        frame_greedy.append(greedy_makespan(costs, lanes))
        frame_maxima.append(max(costs))
        all_costs.extend(costs)

    frame_count = len(records)
    normalized_types = {
        name: count // frame_count for name, count in sorted(type_counts.items())
    }
    median_total = statistics.median(frame_totals)
    return {
        "persistent_blocks": persistent_blocks,
        "cta_types_per_frame": normalized_types,
        "waves_at_measured_residency": persistent_blocks / float(lanes),
        "estimated_cta_work_us": {
            "median_total": median_total,
            "min_total": min(frame_totals),
            "max_total": max(frame_totals),
            "mean_per_cta": median_total / persistent_blocks,
            "p95_per_cta": percentile(all_costs, 0.95),
            "median_frame_max": statistics.median(frame_maxima),
        },
        "greedy_{}_lane_makespan_us".format(lanes): {
            "median": statistics.median(frame_greedy),
            "min": min(frame_greedy),
            "max": max(frame_greedy),
        },
        "raster_dominant_cta_fraction": raster_dominant / active_ctas,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("symmetric_timings", type=Path)
    parser.add_argument("--pair-timings", type=Path)
    parser.add_argument("--deform-tasks", type=int, default=6971)
    parser.add_argument("--sm-count", type=int, default=84)
    parser.add_argument("--active-blocks-per-sm", type=int, default=4)
    parser.add_argument(
        "--persistent-blocks",
        default=",".join(str(value) for value in DEFAULT_CANDIDATES),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    with args.trace.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError("trace contains no frames")
    candidates = [int(value) for value in args.persistent_blocks.split(",")]
    lanes = args.sm_count * args.active_blocks_per_sm
    curves = TimingCurves(args.symmetric_timings)
    results = [
        summarize_candidate(
            records, curves, candidate, args.deform_tasks, lanes
        )
        for candidate in candidates
    ]
    ranked = sorted(
        results,
        key=lambda result: (
            result["greedy_{}_lane_makespan_us".format(lanes)]["median"],
            result["estimated_cta_work_us"]["median_total"],
        ),
    )
    output = {
        "schema_version": 1,
        "source_trace": str(args.trace.resolve()),
        "symmetric_timings": str(args.symmetric_timings.resolve()),
        "frames": len(records),
        "raster_tasks": int(records[0]["tile_count"]),
        "deform_tasks": args.deform_tasks,
        "sm_count": args.sm_count,
        "active_blocks_per_sm": args.active_blocks_per_sm,
        "model_limitations": [
            "CTA timings are isolated hot-cache measurements",
            "greedy lanes approximate scheduling and omit co-resident contention",
            "ranking must be validated end-to-end",
        ],
        "pair_sum_model_validation": (
            validate_pair_sum_model(curves, args.pair_timings)
            if args.pair_timings is not None
            else None
        ),
        "ranking": [result["persistent_blocks"] for result in ranked],
        "candidates": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
