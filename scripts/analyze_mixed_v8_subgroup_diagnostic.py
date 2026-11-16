#!/usr/bin/env python3
"""Summarize ABI-v8 both/Raster-only/Deform-only clock64 JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def load_records(root, mode):
    path = root / "{}.jsonl".format(mode)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not records or any(record.get("mode") != mode for record in records):
        raise ValueError("{} has missing or mismatched records".format(path))
    return records


def aggregate(records, metric):
    means = [
        float(record[metric]["mean"])
        for record in records
        if record[metric].get("count", 0)
    ]
    medians = [
        float(record[metric]["median"])
        for record in records
        if record[metric].get("count", 0)
    ]
    if not means:
        return None
    return {
        "records": len(means),
        "mean_cycles": statistics.mean(means),
        "median_of_cta_medians_cycles": statistics.median(medians),
        "min_record_mean_cycles": min(means),
        "max_record_mean_cycles": max(means),
        "record_mean_span_percent": (
            (max(means) - min(means)) / statistics.mean(means) * 100.0
        ),
    }


def aggregate_kernel_elapsed_ms(records):
    values = [
        float(record["kernel_elapsed_ms"])
        for record in records
        if record.get("kernel_elapsed_ms") is not None
    ]
    if not values:
        return None
    return {
        "records": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "span_percent": (
            (max(values) - min(values)) / statistics.mean(values) * 100.0
        ),
    }


def summarize(root, sm_clock_hz=None):
    both = load_records(root, "both")
    raster_only = load_records(root, "raster_only")
    deform_only = load_records(root, "deform_only")

    both_raster = aggregate(both, "raster_duration_cycles")
    both_deform_all = aggregate(both, "deform_duration_cycles")
    both_deform_overlap = aggregate(both, "overlap_window_cycles")
    solo_raster = aggregate(raster_only, "raster_duration_cycles")
    solo_deform = aggregate(deform_only, "deform_duration_cycles")
    overlap_fractions = [
        float(record["overlap_fraction_of_shorter"]["mean"])
        for record in both
    ]
    both_kernel = aggregate_kernel_elapsed_ms(both)
    raster_only_kernel = aggregate_kernel_elapsed_ms(raster_only)
    deform_only_kernel = aggregate_kernel_elapsed_ms(deform_only)

    raster_solo_cycles = solo_raster["mean_cycles"]
    deform_solo_cycles = solo_deform["mean_cycles"]
    raster_both_cycles = both_raster["mean_cycles"]
    deform_both_overlap_cycles = both_deform_overlap["mean_cycles"]
    serial_cycles = raster_solo_cycles + deform_solo_cycles
    mixed_critical_cycles = max(
        raster_both_cycles, deform_both_overlap_cycles
    )
    ideal_overlap_cycles = max(raster_solo_cycles, deform_solo_cycles)
    event_serial_ms = None
    event_ideal_ms = None
    if raster_only_kernel is not None and deform_only_kernel is not None:
        event_serial_ms = (
            raster_only_kernel["mean_ms"] + deform_only_kernel["mean_ms"]
        )
        event_ideal_ms = max(
            raster_only_kernel["mean_ms"], deform_only_kernel["mean_ms"]
        )

    calibrated_time = None
    if sm_clock_hz is not None:
        if sm_clock_hz <= 0:
            raise ValueError("sm_clock_hz must be positive")

        def mean_us(aggregate):
            return aggregate["mean_cycles"] / sm_clock_hz * 1.0e6

        calibrated_time = {
            "sm_clock_hz": sm_clock_hz,
            "cycle_period_ns": 1.0e9 / sm_clock_hz,
            "both_raster_mean_us": mean_us(both_raster),
            "both_deform_all_ctas_mean_us": mean_us(both_deform_all),
            "both_deform_overlapping_ctas_mean_us": mean_us(
                both_deform_overlap
            ),
            "raster_only_mean_us": mean_us(solo_raster),
            "deform_only_mean_us": mean_us(solo_deform),
        }

    return {
        "kind": "mixed_v8_subgroup_diagnostic_summary",
        "input_root": str(root.resolve()),
        "record_counts": {
            "both": len(both),
            "raster_only": len(raster_only),
            "deform_only": len(deform_only),
        },
        "logical_blocks": {
            "raster": int(both[0]["raster_logical_blocks"]),
            "deform": int(both[0]["deform_logical_blocks"]),
            "physical": int(both[0]["physical_blocks"]),
        },
        "aggregates": {
            "both_raster": both_raster,
            "both_deform_all_ctas": both_deform_all,
            "both_deform_overlapping_ctas": both_deform_overlap,
            "raster_only": solo_raster,
            "deform_only": solo_deform,
        },
        "cuda_event_kernel_time": {
            "both": both_kernel,
            "raster_only": raster_only_kernel,
            "deform_only": deform_only_kernel,
            "serial_solo_sum_mean_ms": event_serial_ms,
            "ideal_overlap_lower_bound_mean_ms": event_ideal_ms,
            "measured_speedup_vs_serial_solo": (
                event_serial_ms
                / both_kernel["mean_ms"]
                if both_kernel is not None
                and event_serial_ms is not None
                else None
            ),
            "ideal_speedup_vs_serial_solo": (
                event_serial_ms / event_ideal_ms
                if event_serial_ms is not None and event_ideal_ms is not None
                else None
            ),
            "measured_time_reduction_vs_serial_percent": (
                (1.0 - both_kernel["mean_ms"] / event_serial_ms) * 100.0
                if both_kernel is not None and event_serial_ms is not None
                else None
            ),
            "potential_overlap_saving_realized_percent": (
                (event_serial_ms - both_kernel["mean_ms"])
                / (event_serial_ms - event_ideal_ms)
                * 100.0
                if both_kernel is not None
                and event_serial_ms is not None
                and event_ideal_ms is not None
                else None
            ),
            "penalty_vs_ideal_overlap_percent": (
                (both_kernel["mean_ms"] / event_ideal_ms - 1.0) * 100.0
                if both_kernel is not None and event_ideal_ms is not None
                else None
            ),
        },
        "clock64_time_at_fixed_sm_clock": calibrated_time,
        "contention": {
            "raster_slowdown_percent":
                (raster_both_cycles / raster_solo_cycles - 1.0) * 100.0,
            "deform_overlapping_cta_slowdown_percent":
                (deform_both_overlap_cycles / deform_solo_cycles - 1.0)
                * 100.0,
            "deform_all_cta_slowdown_percent":
                (both_deform_all["mean_cycles"] / deform_solo_cycles - 1.0)
                * 100.0,
            "mean_overlap_fraction_of_shorter": statistics.mean(
                overlap_fractions
            ),
        },
        "overlap_model": {
            "solo_serial_cycles_per_overlapping_cta": serial_cycles,
            "measured_mixed_critical_cycles_per_overlapping_cta":
                mixed_critical_cycles,
            "ideal_no_contention_cycles_per_overlapping_cta":
                ideal_overlap_cycles,
            "measured_speedup_vs_serial": serial_cycles / mixed_critical_cycles,
            "ideal_speedup_vs_serial": serial_cycles / ideal_overlap_cycles,
            "measured_time_reduction_vs_serial_percent":
                (1.0 - mixed_critical_cycles / serial_cycles) * 100.0,
            "ideal_time_reduction_vs_serial_percent":
                (1.0 - ideal_overlap_cycles / serial_cycles) * 100.0,
            "potential_overlap_saving_realized_percent":
                (serial_cycles - mixed_critical_cycles)
                / (serial_cycles - ideal_overlap_cycles)
                * 100.0,
            "contention_penalty_vs_ideal_overlap_percent":
                (mixed_critical_cycles / ideal_overlap_cycles - 1.0) * 100.0,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--sm-clock-hz",
        type=int,
        default=None,
        help="fixed SM clock used to convert clock64 cycles to time",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            summarize(args.root, sm_clock_hz=args.sm_clock_hz),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
