"""Summarize completed isolated partial-sampling experiments (stdlib only)."""
import argparse
import json
from pathlib import Path
import re
import sqlite3
import statistics


def configuration(path):
    match = re.fullmatch(r"\d+-(baseline|lowreg|fused|separate)-mp(\d+)-q(\d+)", path.stem)
    if match is None:
        raise ValueError("unexpected run filename: " + str(path))
    mode, planes, ratio = match.groups()
    key = mode if mode in ("baseline", "lowreg") else mode + "_mp" + planes
    return key, mode, int(planes), int(ratio)


def summarize_runs(directory):
    groups = {}
    identity = None
    for path in sorted(directory.glob("[0-9][0-9]-*.json")):
        data = json.loads(path.read_text())
        key, mode, planes, ratio = configuration(path)
        assert data["passed"] and data["actual_execution_mode"] == "tacker", path
        assert data["qualification_mode_executed"] and data["tacker_fallback_reason"] is None, path
        assert data["selected_variant_id"] == "full_mlp_overlap_graph_pb0", path
        assert data["warmup_frames"] == 20 and data["profile_frames"] == 50, path
        assert data["post_run"]["byte_stability"]["verified"], path
        current_identity = (data["qualification_profile_sha256"],
                            data["environment"]["cuda_visible_devices"],
                            data["environment"]["gpu"]["name"])
        if identity is None:
            identity = current_identity
        assert current_identity == identity, "mixed targets or qualification in " + str(directory)
        counts = data["pipeline_execution_counts"]
        assert counts["mixed_launches"] == 49 and counts["outputs"] == 50, counts
        if mode in ("fused", "separate"):
            assert counts["raster_hexplane_fused_launches"] == (49 if mode == "fused" else 0), counts
            assert counts["raster_hexplane_sampling_graph_replays"] == (49 if mode == "separate" else 0), counts
            assert counts.get("raster_hexplane_early_sampling_graph_replays", 0) == (49 if planes < 12 else 0), counts
        groups.setdefault(key, []).append({"source": str(path.relative_to(directory.parent)), "fps": data["throughput_fps"]})
    assert groups and "baseline" in groups, directory
    baseline = statistics.median(run["fps"] for run in groups["baseline"])
    result = {}
    for key, runs in groups.items():
        values = [run["fps"] for run in runs]
        median = statistics.median(values)
        result[key] = {"runs": runs, "run_count": len(runs), "median_fps": median,
                       "mean_ms_per_frame_from_throughput": 1000 / median,
                       "relative_to_baseline_fps_percent": 100 * (median / baseline - 1),
                       "min_fps": min(values), "max_fps": max(values)}
    return result


def summarize_events(directory):
    result = {}
    for path in sorted(directory.glob("*.timeline.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        leaves = [row for row in rows if row["kind"] == "mixed_pipeline_leaf_cuda_event"][-49:]
        assert len(leaves) == 49, path
        result[path.name] = {
            "measured_mixed_frames": 49,
            "prefix_ready_before_prepass_start_frames": sum(row["deform_prefix_ready_before_prepass_start"] for row in leaves),
            "median_ms": {key: statistics.median(row[key] for row in leaves) for key in (
                "raster_prepare_elapsed_ms", "prepass_elapsed_ms", "deform_prepare_elapsed_ms",
                "mixed_remainder_elapsed_ms", "prefix_wait_after_prepass_ms")}}
    return result


def summarize_trace(path):
    con = sqlite3.connect("file:" + str(path.resolve()) + "?mode=ro", uri=True)
    rows = list(con.execute("SELECT s.value,k.start,k.end,k.registersPerThread,k.blockX,k.gridX "
                            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start"))
    con.close()
    result = {"kernels": {}}
    for name in dict.fromkeys(row[0] for row in rows if row[0].startswith("tacker_") or row[0].startswith("hexplane_")):
        matching = [row for row in rows if row[0] == name]
        result["kernels"][name] = {"launches": len(matching),
            "median_duration_ms": statistics.median((row[2] - row[1]) / 1e6 for row in matching),
            "registers_per_thread": matching[0][3], "threads": matching[0][4], "grid_ctas": matching[0][5]}
    mixed = [row for row in rows if row[0] == "tacker_mix_raster_hexplane_512_v1"]
    early = [row for row in rows if row[0].startswith("hexplane_sample_plane_solo_v1")]
    if mixed and early:
        assert len(mixed) == 9 and len(early) % 9 == 0, path
        per_frame = len(early) // 9
        groups = [early[i:i + per_frame] for i in range(0, len(early), per_frame)]
        result["early_sampling"] = {
            "planes_per_frame": per_frame,
            "early_start_before_mixed_start_frames": sum(group[0][1] < late[1] for group, late in zip(groups, mixed)),
            "early_complete_before_mixed_start_frames": sum(max(row[2] for row in group) <= late[1] for group, late in zip(groups, mixed)),
            "median_early_start_relative_to_mixed_start_ms": statistics.median((group[0][1] - late[1]) / 1e6 for group, late in zip(groups, mixed)),
            "median_early_end_relative_to_mixed_start_ms": statistics.median((max(row[2] for row in group) - late[1]) / 1e6 for group, late in zip(groups, mixed))}
        prepare_start = [row for row in rows if "preprocessCUDA<" in row[0]]
        prepare_end = [row for row in rows if row[0].startswith("identifyTileRanges")]
        assert len(prepare_start) == len(prepare_end) == 10, path
        overlaps = [sum(max(0, min(row[2], end[2]) - max(row[1], start[1])) / 1e6
                        for row in group)
                    for group, start, end in zip(groups, prepare_start, prepare_end)]
        result["early_sampling"]["sampling_complete_before_raster_gpu_prepare_frames"] = sum(
            max(row[2] for row in group) <= start[1] for group, start in zip(groups, prepare_start))
        result["early_sampling"]["median_overlap_with_raster_gpu_prepare_span_ms"] = statistics.median(overlaps)
        result["early_sampling"]["overlap_warning"] = "This is the instrumented GPU interval from preprocessCUDA to identifyTileRanges, not CPU prepare time or proof of unprofiled overlap."
    return result


def summarize(root):
    result = {"kind": "raster_hexplane_partial_sampling_experiment", "schema_version": 1,
              "root": str(root.resolve()), "qualification_variant": "full_mlp_overlap_graph_pb0",
              "production_admission_claimed": False, "default_enabled": False,
              "graph_scope": "Graph-qualified coordinate/early-sampling/reduction scheduling; new mixed leaf is still a native launch, not whole-leaf Graph replay",
              "timing_policy": "Formal FPS uses unprofiled 50-frame whole-sequence wall timing after 20 warmup frames; compare same-platform medians only",
              "sweeps": {}, "confirmations": {}, "quality": {}, "diagnostic_events": {}, "nsight": {}}
    reference_path = next(root.glob("confirm-*/00-baseline-*.json"), None)
    if reference_path is not None:
        reference = json.loads(reference_path.read_text())
        result["qualification_profile"] = {"path": reference["qualification_profile"],
                                          "sha256": reference["qualification_profile_sha256"]}
        result["runtime"] = {"gpu": reference["environment"]["gpu"],
                             "cuda_visible_devices": reference["environment"]["cuda_visible_devices"],
                             "pytorch": reference["pytorch_version"], "cuda_runtime": reference["cuda_runtime"]}
    result["source_checks"] = {}
    for directory in sorted(root.iterdir()):
        if directory.is_dir() and directory.name.startswith(("cuda-", "quality-", "sweep-", "confirm-", "diagnostic-", "trace-")):
            checks = list(directory.glob("*.source_check.txt"))
            assert (directory / "final.source_check.txt") in checks, directory
            for path in checks:
                lines = path.read_text().splitlines()
                assert lines and all(line.endswith(": OK") for line in lines), path
            result["source_checks"][directory.name] = {"checks": len(checks), "passed": True}
    result["memory_checks"] = {}
    for path in sorted(root.glob("memcheck-*.log")):
        text = path.read_text()
        assert "ERROR SUMMARY: 0 errors" in text and "Ran 8 tests" in text and "\nOK\n" in text, path
        resource_line = next(line for line in text.splitlines() if line.startswith("RASTER_HEXPLANE_MIXED_RESOURCES="))
        result["memory_checks"][path.name] = {"tests": 8, "errors": 0, "resources": json.loads(resource_line.split("=", 1)[1])}
    for directory in sorted(root.glob("sweep-*")):
        result["sweeps"][directory.name] = summarize_runs(directory)
    for directory in sorted(root.glob("confirm-*")):
        result["confirmations"][directory.name] = summarize_runs(directory)
    for directory in sorted(root.glob("quality-*")):
        for path in sorted(directory.glob("[0-9][0-9]-*.json")):
            data = json.loads(path.read_text())
            assert data["passed"] and all(gate["passed"] for gate in data["gates"]), path
            result["quality"][str(path.relative_to(root))] = {"passed": True, "deltas": data["deltas"]}
    for directory in sorted(root.glob("diagnostic-*")):
        result["diagnostic_events"][directory.name] = summarize_events(directory)
    for directory in sorted(root.glob("trace-*")):
        result["nsight"][directory.name] = {path.name: summarize_trace(path) for path in sorted(directory.glob("*.sqlite"))}
    result["diagnostic_warning"] = "CUDA event spans include stream waits; Nsight durations are concurrent observations, not isolated timings. Neither diagnostic FPS nor sums of stage medians are formal end-to-end results."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = summarize(args.results)
    rendered = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    for phase, variants in data["confirmations"].items():
        print(phase)
        for name, metrics in variants.items():
            print(name, "n=", metrics["run_count"], "FPS=", round(metrics["median_fps"], 6),
                  "delta%=", round(metrics["relative_to_baseline_fps_percent"], 6))
