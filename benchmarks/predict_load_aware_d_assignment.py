#!/usr/bin/env python3
"""Estimate the ceiling from assigning D quotas by measured Raster load."""

import argparse
import bisect
import csv
import heapq
import json
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_CANDIDATES = (
    2720, 2800, 2944, 3072, 3200, 3328, 3456, 3486,
    3584, 3712, 3840, 3968, 4096, 4224, 4352, 4480,
)


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class PairTimingCurves:
    def __init__(self, path):
        grouped = defaultdict(lambda: defaultdict(list))
        with path.open("r", encoding="utf-8") as handle:
            rows = (line for line in handle if not line.startswith("#"))
            for row in csv.DictReader(rows):
                k0 = int(row["raster_k0"])
                k1 = int(row["raster_k1"])
                raster_tasks = int(row["raster_tasks"])
                deform_tasks = int(row["deform_tasks"])
                if raster_tasks == 0:
                    workload = 0
                elif raster_tasks == 1:
                    workload = k0
                elif raster_tasks == 2:
                    workload = k0 + k1
                else:
                    raise ValueError("timing table contains more than 2R")
                grouped[(raster_tasks, deform_tasks)][workload].append(
                    float(row["median_raw_us"])
                )
        self.curves = {
            key: sorted(
                (workload, statistics.median(samples))
                for workload, samples in by_workload.items()
            )
            for key, by_workload in grouped.items()
        }

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
        fraction = (raster_workload - x0) / float(x1 - x0)
        return y0 + (y1 - y0) * fraction


def task_count(logical_tasks, physical_block, persistent_blocks):
    if physical_block >= logical_tasks:
        return 0
    return 1 + (logical_tasks - 1 - physical_block) // persistent_blocks


def lpt_makespan(costs, lanes):
    heap = [0.0] * lanes
    heapq.heapify(heap)
    for cost in sorted(costs, reverse=True):
        earliest = heapq.heappop(heap)
        heapq.heappush(heap, earliest + cost)
    return max(heap)


def summarize(records, curves, persistent_blocks, deform_tasks, lanes):
    tile_count = records[0]["tile_count"]
    base_d, extra_d = divmod(deform_tasks, persistent_blocks)
    current_totals = []
    aware_totals = []
    current_lpt = []
    aware_lpt = []
    current_equivalent = []
    aware_equivalent = []
    global_equivalent_lower_bound = []
    changed_assignments = []

    for record in records:
        tile_loads = record["range_counts"]
        raster_counts = []
        raster_workloads = []
        for block in range(persistent_blocks):
            count = task_count(tile_count, block, persistent_blocks)
            if count > 2:
                raise ValueError(
                    "PB {} requires {}R, outside measured curves".format(
                        persistent_blocks, count
                    )
                )
            raster_counts.append(count)
            raster_workloads.append(sum(
                tile_loads[block + task * persistent_blocks]
                for task in range(count)
            ))

        current_d = [
            base_d + (1 if block < extra_d else 0)
            for block in range(persistent_blocks)
        ]
        if max(current_d, default=0) > 3:
            raise ValueError(
                "PB {} requires more than measured 3D".format(
                    persistent_blocks
                )
            )

        # Under the quotient/remainder constraint every block receives base_d
        # tasks and exactly extra_d blocks receive one additional task.  The
        # independent optimum is therefore to place those extras where their
        # measured marginal CTA cost is smallest.
        marginal = []
        for block, (raster_count, raster_workload) in enumerate(
            zip(raster_counts, raster_workloads)
        ):
            base_cost = curves.evaluate(
                raster_count, base_d, raster_workload
            )
            extra_cost = curves.evaluate(
                raster_count, base_d + 1, raster_workload
            )
            marginal.append((extra_cost - base_cost, block))
        selected = {
            block for _cost, block in sorted(marginal)[:extra_d]
        }
        aware_d = [
            base_d + (1 if block in selected else 0)
            for block in range(persistent_blocks)
        ]
        changed_assignments.append(sum(
            left != right for left, right in zip(current_d, aware_d)
        ))

        current_costs = []
        aware_costs = []
        for raster_count, raster_workload, old_d, new_d in zip(
            raster_counts, raster_workloads, current_d, aware_d
        ):
            current_costs.append(curves.evaluate(
                raster_count, old_d, raster_workload
            ))
            aware_costs.append(curves.evaluate(
                raster_count, new_d, raster_workload
            ))
        current_totals.append(sum(current_costs))
        aware_totals.append(sum(aware_costs))
        current_lpt.append(lpt_makespan(current_costs, lanes))
        aware_lpt.append(lpt_makespan(aware_costs, lanes))

        current_equivalent.append(sum(
            max(raster_workload, 512 * deform_count)
            for raster_workload, deform_count in zip(
                raster_workloads, current_d
            )
        ))
        # The raw-equivalent optimum can differ slightly from the measured
        # timing optimum, so calculate its exact quota assignment separately.
        equivalent_marginal = sorted(
            (
                max(raster_workload, 512 * (base_d + 1))
                - max(raster_workload, 512 * base_d),
                block,
            )
            for block, raster_workload in enumerate(raster_workloads)
        )
        equivalent_selected = {
            block for _cost, block in equivalent_marginal[:extra_d]
        }
        aware_equivalent.append(sum(
            max(
                raster_workload,
                512 * (base_d + (1 if block in equivalent_selected else 0)),
            )
            for block, raster_workload in enumerate(raster_workloads)
        ))
        global_equivalent_lower_bound.append(max(
            sum(raster_workloads), 512 * deform_tasks
        ))

    def median(values):
        return statistics.median(values)

    current_total = median(current_totals)
    aware_total = median(aware_totals)
    current_lane = median(current_lpt)
    aware_lane = median(aware_lpt)
    current_eq = median(current_equivalent)
    aware_eq = median(aware_equivalent)
    lower_eq = median(global_equivalent_lower_bound)
    return {
        "persistent_blocks": persistent_blocks,
        "d_quota": {
            "base_per_cta": base_d,
            "extra_task_ctas": extra_d,
            "max_per_cta": base_d + (1 if extra_d else 0),
        },
        "r_quota": {
            "base_per_cta": tile_count // persistent_blocks,
            "extra_task_ctas": tile_count % persistent_blocks,
            "max_per_cta": (tile_count + persistent_blocks - 1) // persistent_blocks,
        },
        "median_reassigned_ctas": median(changed_assignments),
        "measured_cta_cost_model": {
            "current_total_us": current_total,
            "load_aware_total_us": aware_total,
            "total_reduction_fraction": 1.0 - aware_total / current_total,
            "current_lpt_{}_lane_us".format(lanes): current_lane,
            "load_aware_lpt_{}_lane_us".format(lanes): aware_lane,
            "lpt_reduction_fraction": 1.0 - aware_lane / current_lane,
        },
        "zero_overhead_equivalent_model": {
            "current_sum_max": current_eq,
            "load_aware_sum_max": aware_eq,
            "global_lower_bound": lower_eq,
            "current_to_load_aware_reduction_fraction": 1.0 - aware_eq / current_eq,
            "current_to_global_ceiling_fraction": 1.0 - lower_eq / current_eq,
            "load_aware_gap_above_global_lower_bound_fraction": aware_eq / lower_eq - 1.0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("timings", type=Path)
    parser.add_argument("--deform-tasks", type=int, default=6971)
    parser.add_argument("--lanes", type=int, default=336)
    parser.add_argument(
        "--persistent-blocks",
        default=",".join(str(value) for value in DEFAULT_CANDIDATES),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    curves = PairTimingCurves(args.timings)
    candidates = [
        summarize(
            records, curves, int(value), args.deform_tasks, args.lanes
        )
        for value in args.persistent_blocks.split(",")
    ]
    output = {
        "schema_version": 1,
        "source_trace": str(args.trace.resolve()),
        "timings": str(args.timings.resolve()),
        "frames": len(records),
        "deform_tasks": args.deform_tasks,
        "lanes": args.lanes,
        "model_scope": [
            "R grouping is unchanged; only the quotient/remainder D quota is reassigned",
            "zero-overhead model treats one D task as 512 Raster Gaussian references",
            "measured model interpolates isolated hot-cache 0/1/2R + 0/1/2/3D CTA timings",
            "LPT is an optimistic lane schedule and is not the current CUDA block order",
            "scheduler construction overhead and co-resident contention are excluded",
        ],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
