"""Analyze measured kernel-replay resources without claiming concurrent attribution."""
import argparse
import csv
import json
from pathlib import Path

from analyze_sampling_kernel_activity import foreign_python_processes


METRICS = {
    "theoretical_occupancy_percent": "sm__maximum_warps_per_active_cycle_pct",
    "achieved_occupancy_percent": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm_throughput_percent": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex_throughput_percent_active": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l2_throughput_percent_elapsed": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "issue_active_percent": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "eligible_warps_per_scheduler_cycle": "smsp__warps_eligible.avg.per_cycle_active",
    "l1_sector_hit_percent": "l1tex__t_sector_hit_rate.pct",
    "l2_sector_hit_percent": "lts__t_sector_hit_rate.pct",
}


def raw_csv(path):
    with path.open() as handle:
        reader = csv.DictReader(handle)
        units = next(reader)
        rows = [r for r in reader if r.get("ID", "").isdigit()]
    return units, rows


def number(value):
    if value in (None, "", "n/a", "N/A"):
        return None
    return float(value.replace(",", ""))


def duration_ms(row, units):
    value = number(row["gpu__time_duration.sum"])
    factor = {"ns": 1e-6, "us": 1e-3, "ms": 1, "s": 1e3}
    return value * factor[units["gpu__time_duration.sum"]]


def source_and_audit(directory):
    assert (directory / "completed.txt").exists(), directory
    for stem in ("before", "after"):
        lines = (directory / (stem + ".source_check.txt")).read_text().splitlines()
        assert lines and all(line.endswith(": OK") for line in lines), directory
    audit_text = (directory / "during_processes.txt").read_text()
    own_path = "/green-handoff-20261010-4vqFSK/scripts/run_profile_render_sealed.py"
    own_pids = {line.split()[0] for line in audit_text.splitlines()
                if own_path in line and len(line.split(None, 3)) == 4
                and line.split(None, 3)[2].startswith("python")}
    bad = foreign_python_processes(audit_text)
    # App-range relaunches Python; ps can catch its teardown as [python3].
    # Only accept these transient entries for PIDs independently seen running
    # our exact sealed launcher in another snapshot of the same audit.
    bad = [line for line in bad if own_path not in line and not (
        line.split()[0] in own_pids
        and line.split(None, 3)[3] in ("[python3]", "[python3] <defunct>"))]
    assert not bad, bad
    meta = json.loads((directory / "renderer.metadata.json").read_text())
    assert meta["passed"] and meta["actual_execution_mode"] == "tacker"
    assert meta["qualification_mode_executed"]
    assert meta["selected_variant_id"] == "full_mlp_overlap_graph_pb0"
    assert meta["tacker_fallback_reason"] is None
    assert not meta["green_contexts"]["enabled"]
    assert meta["post_run"]["byte_stability"]["verified"]
    assert (meta["gaussian_count"], meta["image_width"], meta["image_height"]) == (111525, 1352, 1014)
    return {"qualification_sha256": meta["qualification_profile_sha256"],
            "source_and_loaded_binary_stable": True, "foreign_python_processes": bad}


def summarize_group(name, indices, rows, units):
    selected = [rows[i] for i in indices]
    durations = [duration_ms(row, units) for row in selected]
    total = sum(durations)
    result = {"name": name, "count": len(selected), "profile_ids": indices,
              "isolated_kernel_sum_ms": total, "metrics": {}, "resources": []}
    configurations = {}
    for row in selected:
        key = (row["Block Size"], row["Grid Size"],
               row["launch__registers_per_thread"],
               row["launch__shared_mem_per_block_static"])
        if key not in configurations:
            smem = number(row["launch__shared_mem_per_block_static"])
            factor = {"byte/block": 1, "Kbyte/block": 1000}[units["launch__shared_mem_per_block_static"]]
            configurations[key] = {
                "block": key[0], "grid": key[1], "registers_per_thread": int(key[2]),
                "static_shared_bytes_per_block": round(smem * factor),
                "occupancy_limits_ctas_per_sm": {
                    kind: number(row["launch__occupancy_limit_" + kind])
                    for kind in ("blocks", "registers", "shared_mem", "warps")},
                "launches": 0,
            }
        configurations[key]["launches"] += 1
    result["resources"] = list(configurations.values())
    for label, metric in METRICS.items():
        values = [number(row.get(metric)) for row in selected]
        result["metrics"][label] = None if any(v is None for v in values) else {
            "duration_weighted_mean": sum(v * t for v, t in zip(values, durations)) / total,
            "min": min(values), "max": max(values),
        }
    stalls = {}
    for metric in units:
        if "average_warps_issue_stalled_" in metric:
            values = [number(row.get(metric)) for row in selected]
            if all(v is not None for v in values):
                reason = metric.split("stalled_")[1].split("_per_issue")[0]
                stalls[reason] = sum(v * t for v, t in zip(values, durations)) / total
    result["stall_ratios_not_percent"] = dict(sorted(stalls.items(), key=lambda item: -item[1]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    inventory = root / "ncu-unpartitioned-20261010-inventory"
    kernels = root / "ncu-unpartitioned-20261010-kernels"
    audits = {"inventory": source_and_audit(inventory), "kernels": source_and_audit(kernels)}
    _, reference = raw_csv(inventory / "report.raw.csv")
    units, rows = raw_csv(kernels / "report.raw.csv")
    assert len(rows) == 82
    for row, expected in zip(rows, reference[102:184]):
        for key in ("Kernel Name", "Grid Size", "Block Size", "Stream"):
            assert row[key] == expected[key], (row["ID"], key, row[key], expected[key])
    assert sum("grid_sampler_2d_kernel" in row["Kernel Name"] for row in rows) == 12
    assert "hexplane_reduce_features_tiled_v1<256>" in rows[35]["Kernel Name"]
    assert rows[54]["Kernel Name"] == "tacker_render_raster_fast_range_v12"
    assert rows[61]["Kernel Name"] == "tacker_mix_render_deform_mlp_tensor_tail_split_raster_v12"
    groups = {
        "deform_coordinates": range(0, 11),
        "plane_coordinate_gather": range(11, 35, 2),
        "hexplane_sampling": range(12, 35, 2),
        "feature_reduction": [35], "deform_prefix_mask": [36],
        "raster_buffer_init": range(37, 41), "raster_preprocess": [41],
        "scan_init": [42], "raster_scan": [43], "raster_duplicate": [44],
        "sort_histogram": [45], "sort_exclusive_sum": [46],
        "sort_onesweep": range(47, 53), "identify_ranges": [53],
        "raster_prepass": [54], "mlp_input_pack": range(55, 61),
        "mixed_mlp_raster_tail": [61], "visibility_mask": [62],
        "deform_suffix": range(63, 82),
    }
    covered = sorted(i for indices in groups.values() for i in indices)
    assert covered == list(range(82))
    result = {
        "variant": "v12 + original sampling + fused reduction(256), Green disabled",
        "actual_variant": "full_mlp_overlap_graph_pb0",
        "qualification_profile": "/data/qyfeng/4DGaussians-tacker-v12-split-20261001/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json",
        "kernel_replay_scope": "One complete steady pair Raster(0)/Deform(1), excluding 102 prefill launches. 82 profiled launches, 16 replay passes per launch. Cache-control default all. These are isolated launch counters, not concurrent per-kernel attribution or formal FPS.",
        "metric_aggregation": "For multi-launch groups, displayed means are duration-weighted across isolated launches; they are not a separately measured stage-wide counter.",
        "dram_counters": "Not exposed in the collected Orin section results; absence is not zero usage. L1TEX/L2 counters do not measure external DRAM bandwidth.",
        "stall_units": "NCU stalled-warps per issue-active ratios, NOT percentage of wall time.",
        "clock_hz": 1300500000, "audits": audits,
        "groups": {name: summarize_group(name, list(indices), rows, units)
                   for name, indices in groups.items()},
        "normal_graph_timeline_reference": "../green8_8_prepare_only_renderer/summary.json (baseline arm; historical separate batch)",
        "nvidia_replay_documentation": "https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay",
    }
    range_dir = root / "ncu-unpartitioned-20261010-range-v2"
    if (range_dir / "completed.txt").exists():
        result["audits"]["application_range"] = source_and_audit(range_dir)
        range_units, range_rows = raw_csv(range_dir / "report.raw.csv")
        range_row = range_rows[0]
        active_cycles = number(range_row["sm__cycles_active.avg"])
        elapsed_cycles = number(range_row["sm__cycles_elapsed.avg"])
        result["concurrent_application_range"] = {
            "scope": "Whole 20-frame trial including one prefill and one drain, not pure steady state. Graph and cross-stream concurrency preserved; metrics attributed only to the whole range.",
            "rows": range_rows, "units": range_units,
            "metrics": {key: number(range_row.get(metric)) for key, metric in METRICS.items()},
            "sm_active_cycles_fraction_percent": 100 * active_cycles / elapsed_cycles,
            "range_duration_ms": duration_ms(range_row, range_units),
        }
    baseline = root / "ncu-unpartitioned-20261010-baseline"
    if (baseline / "completed.txt").exists():
        result["audits"]["baseline"] = source_and_audit(baseline)
        meta = json.loads((baseline / "renderer.metadata.json").read_text())
        result["unprofiled_sanity"] = {
            "throughput_fps": meta["throughput_fps"],
            "scope": "Two 50-frame trials, 20 warmup frames; sanity check only, not new ABBA comparison.",
        }
    baseline20 = root / "ncu-unpartitioned-20261010-baseline20"
    if (baseline20 / "completed.txt").exists():
        result["audits"]["baseline20"] = source_and_audit(baseline20)
        off = json.loads((baseline20 / "renderer.metadata.json").read_text())
        on = json.loads((range_dir / "renderer.metadata.json").read_text())
        result["application_range_perturbation_sanity"] = {
            "unprofiled_20_frame_fps": off["throughput_fps"],
            "last_profiled_replay_20_frame_fps": on["throughput_fps"],
            "fps_change_percent": 100 * (on["throughput_fps"] / off["throughput_fps"] - 1),
            "accepted_as_low_perturbation_normal_run_counters": False,
            "scope": "Same warmup and frames, but 2 off trials versus 1 last replay trial, no ABBA, and per-pass metadata is overwritten. This indicates substantial instrumentation perturbation, NOT a controlled FPS comparison. Range counters are auxiliary only and do not claim exact normal-runtime utilization.",
        }
    assert {audit["qualification_sha256"] for audit in result["audits"].values()} == {
        "a13d87f6c4a22a24b20f83cb3c66a009e759b739b498c9ff55f8c35c9b82d23a"}
    (root / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for name, group in result["groups"].items():
        values = group["metrics"]
        print(name, "count", group["count"], "isolated_ms", round(group["isolated_kernel_sum_ms"], 4),
              {key: round(values[key]["duration_weighted_mean"], 2) if values[key] else None
               for key in ("theoretical_occupancy_percent", "achieved_occupancy_percent",
                           "sm_throughput_percent", "l2_throughput_percent_elapsed",
                           "issue_active_percent")})


if __name__ == "__main__":
    main()
