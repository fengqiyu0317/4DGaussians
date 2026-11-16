#!/usr/bin/env python3
"""Validate and summarize the A6000 mixed-tail v5/v7/v8 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from benchmark_tacker_fps import paired_bootstrap_ratio  # noqa: E402


SCENES = (
    "flame_steak",
    "coffee_martini",
    "cook_spinach",
    "flame_salmon_1",
)
VARIANTS = ("v5", "v7", "v8")
EXPECTED_COUNTS = {
    "input_frames": 50,
    "full_deformation": 1,
    "prefix": 49,
    "mixed_launches": 49,
    "suffix": 49,
    "solo_raster": 1,
    "outputs": 50,
    "selected_head_evaluations_per_head": 50,
}
RUN_NAME = re.compile(r"\d+-round(?P<round>\d+)-(?P<variant>v[578])\.metadata\.json$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize_values(values):
    mean = statistics.mean(values)
    return {
        "samples": len(values),
        "values": values,
        "median_fps": statistics.median(values),
        "median_ms_per_frame": 1000.0 / statistics.median(values),
        "mean_fps": mean,
        "min_fps": min(values),
        "max_fps": max(values),
        "sample_stdev_fps": statistics.stdev(values),
        "coefficient_of_variation": statistics.stdev(values) / mean,
    }


def validate_fps_scene(scene_root: Path, scene: str):
    by_variant = {variant: {} for variant in VARIANTS}
    paths = sorted((scene_root / "fps" / "runs").glob("*.metadata.json"))
    require(len(paths) == 30, f"{scene}: expected 30 FPS metadata files")
    for path in paths:
        match = RUN_NAME.match(path.name)
        require(match is not None, f"{scene}: malformed run filename {path.name}")
        round_index = int(match.group("round"))
        variant = match.group("variant")
        document = load_json(path)
        require(document.get("passed") is True, f"{path}: child did not pass")
        require(document.get("actual_execution_mode") == "tacker", f"{path}: not tacker")
        require(document.get("tacker_fallback_reason") is None, f"{path}: fallback")
        require(document.get("qualification_mode_executed") is True, f"{path}: no qualification")
        require(document.get("pipeline_execution_counts") == EXPECTED_COUNTS, f"{path}: counts")
        require(document.get("warmup_frames") == 10, f"{path}: warmup")
        require(document.get("profile_frames") == 50, f"{path}: frame count")
        require(document.get("gpu_name") == "NVIDIA RTX A6000", f"{path}: GPU")
        require(document.get("image_width") == 1352, f"{path}: width")
        require(document.get("image_height") == 1014, f"{path}: height")
        require(round_index not in by_variant[variant], f"{scene}: duplicate round")
        by_variant[variant][round_index] = float(document["median_throughput_fps"])
    for variant in VARIANTS:
        require(set(by_variant[variant]) == set(range(10)), f"{scene}/{variant}: rounds")

    ordered = {
        variant: [by_variant[variant][round_index] for round_index in range(10)]
        for variant in VARIANTS
    }
    comparisons = {}
    for candidate, reference in (("v7", "v5"), ("v8", "v5"), ("v8", "v7")):
        key = f"{candidate}_vs_{reference}"
        comparisons[key] = paired_bootstrap_ratio(
            ordered[candidate],
            ordered[reference],
            resamples=10000,
            seed=0,
            label=f"a6000-{scene}-{candidate}-vs-{reference}",
        )
    return {
        "protocol": {
            "rounds": 10,
            "warmup_frames_per_child": 10,
            "timed_frames_per_child": 50,
            "children": 30,
            "schedule": "fixed seed-0 three-way ABBA",
        },
        "variants": {variant: summarize_values(ordered[variant]) for variant in VARIANTS},
        "comparisons": comparisons,
    }


def validate_quality_scene(scene_root: Path, scene: str):
    result = {}
    documents = {}
    for variant in VARIANTS:
        path = scene_root / "quality" / f"{variant}.json"
        document = load_json(path)
        documents[variant] = document
        require(document.get("passed") is True, f"{path}: quality gate failed")
        require(document.get("errors") == [], f"{path}: quality errors")
        modes = document.get("modes", {})
        require(modes.get("serial", {}).get("actual_mode") == "serial", f"{path}: serial mode")
        tacker = modes.get("tacker", {})
        require(tacker.get("actual_mode") == "tacker", f"{path}: tacker mode")
        require(tacker.get("fallback_reason") is None, f"{path}: quality fallback")
        require(tacker.get("qualification_executed") is True, f"{path}: quality qualification")
        require(tacker.get("pipeline_execution_counts") == EXPECTED_COUNTS, f"{path}: quality counts")
        result[variant] = {
            "passed": True,
            "serial_means": modes["serial"]["means"],
            "tacker_means": tacker["means"],
            "tacker_minus_serial": document["deltas"]["tacker"],
        }
    result["v8_vs_v7"] = {
        "means_equal": (
            documents["v8"]["modes"]["tacker"]["means"]
            == documents["v7"]["modes"]["tacker"]["means"]
        ),
        "per_view_equal": (
            documents["v8"]["modes"]["tacker"]["per_view"]
            == documents["v7"]["modes"]["tacker"]["per_view"]
        ),
    }
    require(result["v8_vs_v7"]["per_view_equal"], f"{scene}: v7/v8 per-view mismatch")
    return result


def read_ncu_metrics(ncu_root: Path):
    wanted = {
        ("GPU Speed Of Light Throughput", "Duration"),
        ("GPU Speed Of Light Throughput", "Compute (SM) Throughput"),
        ("GPU Speed Of Light Throughput", "Memory Throughput"),
        ("Compute Workload Analysis", "Issue Slots Busy"),
        ("Scheduler Statistics", "One or More Eligible"),
        ("Scheduler Statistics", "No Eligible"),
        ("Scheduler Statistics", "Active Warps Per Scheduler"),
        ("Scheduler Statistics", "Eligible Warps Per Scheduler"),
        ("Warp State Statistics", "Warp Cycles Per Issued Instruction"),
        ("Instruction Statistics", "Executed Instructions"),
        ("Launch Statistics", "Registers Per Thread"),
        ("Occupancy", "Theoretical Occupancy"),
        ("Occupancy", "Achieved Occupancy"),
        ("Occupancy", "Achieved Active Warps Per SM"),
    }
    result = {}
    raw_wanted = {
        "gpu__time_duration.sum",
        "launch__registers_per_thread",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_eligible.avg.per_cycle_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        "smsp__inst_executed.sum",
    }
    for variant in VARIANTS:
        label = f"mixed_{variant}"
        path = ncu_root / f"{label}.details.csv"
        values = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["Section Name"], row["Metric Name"])
                if key in wanted:
                    values[f"{key[0]}::{key[1]}"] = {
                        "value": float(row["Metric Value"].replace(",", "")),
                        "unit": row["Metric Unit"],
                    }
        require(len(values) == len(wanted), f"{path}: missing NCU metrics")
        raw_path = ncu_root / f"{label}.raw.csv"
        with raw_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        require(len(rows) == 3, f"{raw_path}: unexpected raw CSV shape")
        header, units, row = rows
        raw = {}
        for metric in raw_wanted:
            require(metric in header, f"{raw_path}: missing {metric}")
            index = header.index(metric)
            raw[metric] = {
                "value": float(row[index].replace(",", "")),
                "unit": units[index],
            }
        result[variant] = {"details": values, "raw": raw}
    durations = {
        variant: result[variant]["raw"]["gpu__time_duration.sum"]["value"]
        for variant in VARIANTS
    }
    result["duration_comparisons"] = {
        "v7_vs_v5_speedup": durations["v5"] / durations["v7"],
        "v8_vs_v5_speedup": durations["v5"] / durations["v8"],
        "v8_vs_v7_speedup": durations["v7"] / durations["v8"],
        "v8_minus_v7_ms": durations["v8"] - durations["v7"],
    }
    return result


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    e2e_root = experiment_root / "results" / "a6000_multiscene_e2e_quality_v1"
    ncu_root = experiment_root / "results" / "a6000_ncu_v1"
    gate = load_json(
        experiment_root / "results" / "a6000_resource_accuracy_v1" / "resource_accuracy_gate.json"
    )
    require(gate.get("device") == "NVIDIA RTX A6000", "resource gate device")
    require(gate.get("compute_capability") == "sm_86", "resource gate architecture")
    require(all(gate["v8_vs_v7_bitwise_equal"]), "resource gate v7/v8 mismatch")

    scenes = {}
    for scene in SCENES:
        scene_root = e2e_root / scene
        scenes[scene] = {
            "fps": validate_fps_scene(scene_root, scene),
            "quality": validate_quality_scene(scene_root, scene),
        }

    aggregate = {}
    for candidate, reference in (("v7", "v5"), ("v8", "v5"), ("v8", "v7")):
        key = f"{candidate}_vs_{reference}"
        ratios = [
            scenes[scene]["fps"]["comparisons"][key]["median_fps_ratio"]
            for scene in SCENES
        ]
        aggregate[key] = {
            "per_scene_median_fps_ratios": dict(zip(SCENES, ratios)),
            "geometric_mean_ratio": geometric_mean(ratios),
            "minimum_ratio": min(ratios),
            "maximum_ratio": max(ratios),
            "all_per_scene_bootstrap_ci_lower_above_one": all(
                scenes[scene]["fps"]["comparisons"][key]["paired_bootstrap_95_ci"]["lower"]
                > 1.0
                for scene in SCENES
            ),
        }

    summary = {
        "schema_version": 1,
        "kind": "a6000_mixed_tensor_tail_v8_multiscene_summary",
        "passed": True,
        "experiment": {
            "device": "NVIDIA RTX A6000",
            "compute_capability": "sm_86",
            "scenes": list(SCENES),
            "variants": list(VARIANTS),
            "fps_child_runs": 120,
            "quality_runs": 12,
        },
        "resource_accuracy_gate": gate,
        "ncu": read_ncu_metrics(ncu_root),
        "scenes": scenes,
        "multiscene_aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
