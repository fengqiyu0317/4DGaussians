"""Validate a complete outer A(default), B(high), B(high), A(default) batch."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from analyze_sampling_kernel_activity import foreign_python_processes
from benchmark_sampling_prepass_graph import QUALIFICATION_SHA, analyze_trace, summarize_abba


VERSION_NAMES = {"default": "v13", "high": "v13 (sampling-high-priority experiment)"}


def paired_summary(records):
    if [row["arm"] for row in records] != ["A", "B", "B", "A"]:
        raise ValueError("one complete predeclared priority ABBA block is required")
    default = statistics.mean(row["both_ms"] for row in records if row["arm"] == "A")
    high = statistics.mean(row["both_ms"] for row in records if row["arm"] == "B")
    return {"default_both_ms": default, "high_both_ms": high,
            "saved_ms": default - high, "time_saved_percent": 100 * (1 - high / default),
            "high_over_default": high / default, "speedup": default / high}


def trace_waits(path, repetitions=8):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    sampling = sorted((r for r in rows if "grid_sampler_2d_kernel" in r["name"]),
                      key=lambda r: int(r["start_ns"]))
    raster = sorted((r for r in rows if r["name"] == "tacker_render_raster_fast_range_v12"),
                    key=lambda r: int(r["start_ns"]))
    assert len(sampling) == 12 * repetitions and len(raster) == repetitions
    records = []
    for index in range(repetitions):
        group = sampling[12 * index:12 * (index + 1)]
        start = int(group[0]["start_ns"])
        end = max(int(r["end_ns"]) for r in group)
        raster_start = int(raster[index]["start_ns"])
        window_start = min(start, raster_start)
        gaps = sum(max(0, int(b["start_ns"]) - int(a["end_ns"]))
                   for a, b in zip(group, group[1:]))
        records.append({"sampling_first_start_delay_ms": (start - window_start) / 1e6,
                        "sampling_inter_kernel_gaps_ms": gaps / 1e6,
                        "sampling_first_to_last_span_ms": (end - start) / 1e6})
    return {"samples": records,
            "mean": {key: statistics.mean(r[key] for r in records) for key in records[0]}}


def validate_run(root, index, arm, identities, fingerprints):
    run = root / ("run%d_%s" % (index, arm))
    assert (run / "completed.txt").exists(), run
    data = json.loads((run / "benchmark.json").read_text())
    metadata = json.loads((run / "renderer.metadata.json").read_text())
    assert data["sampling_priority"] == ("default" if arm == "A" else "high")
    assert data["node_priority_backend"] and data["correctness_passed"]
    assert metadata["passed"] and metadata["qualification_mode_executed"]
    assert metadata["actual_execution_mode"] == "tacker" and metadata["tacker_fallback_reason"] is None
    assert metadata["post_run"]["byte_stability"]["verified"] and not metadata["green_contexts"]["enabled"]
    assert data["actual_variant"] == metadata["selected_variant_id"] == "full_mlp_overlap_graph_pb0"
    assert data["qualification_sha256"] == metadata["qualification_profile_sha256"] == QUALIFICATION_SHA
    identity = (data["driver_sha256"], data["trace_library_sha256"], metadata["qualification_profile_sha256"])
    if identities:
        assert identities[0] == identity
    identities.append(identity)
    for stage in ("before", "after"):
        checks = (run / (stage + ".source_check.txt")).read_text().splitlines()
        assert checks and all(line.endswith(": OK") for line in checks)
        assert "GR3D_FREQ 0%" in (run / (stage + ".tegrastats.txt")).read_text()
        assert not foreign_python_processes((run / (stage + ".processes.txt")).read_text())
    assert all(line.endswith(": OK") for line in (root / ("run%d.batch_source_check.txt" % index)).read_text().splitlines())
    driver_path = str(root.parent / "scripts/benchmark_sampling_prepass_graph.py")
    # The archive can be local, so resolve the actual remote driver from command.txt.
    import shlex
    command = shlex.split((run / "command.txt").read_text())
    driver_path = command[1]
    assert driver_path.endswith("/scripts/benchmark_sampling_prepass_graph.py")
    assert not [line for line in foreign_python_processes((run / "during_processes.txt").read_text())
                if driver_path not in line]
    cases = []
    assert len(data["cases"]) == 2
    for case in data["cases"]:
        workload = case["workload"]
        assert workload["graph_replay_backend"] == "cuda_driver_use_node_priority"
        assert workload["stream_priorities"] == {"control": 0, "sampling": 0 if arm == "A" else -3, "prepass": 0}
        assert workload["sampling_serial_on_one_stream"]
        assert (workload["points"], workload["resolution"], workload["raster_tile_range"]) == (111525, [1352, 1014], [0, 1954])
        if case["case"] in fingerprints:
            assert fingerprints[case["case"]] == workload["workload_fingerprints"], "workload changed across arms"
        fingerprints[case["case"]] = workload["workload_fingerprints"]
        assert len(workload["graph_priority_audits"]) == 5
        for audit in workload["graph_priority_audits"].values():
            assert audit["verified"] and audit["use_node_priority"]
            assert audit["requested_instantiation_flags"] == 8
            if audit["instantiate_flags"] != 8:
                assert audit["behavioral_control"] == {"default_update_status": 0, "default_update_result": 0,
                                                       "priority_update_status": 910, "priority_update_result": 8}
        assert summarize_abba(case["records"]) == case["summary"]
        assert case["summary"]["block_count"] == 4
        assert all(len(row["both_timing"]["samples_ms"]) == 40 for row in case["records"] if row["arm"] == "B")
        traces = {}
        for trace in case["diagnostic_traces"]:
            path = run / ("case%d.%s.kernels.csv" % (case["case"], trace["mode"]))
            assert trace == analyze_trace(path, trace["mode"], 8, 1954)
            traces[trace["mode"]] = trace["mean"]
        assert set(traces) == {"sampling", "prepass", "both", "both_reverse"}
        cases.append({"case": case["case"], "unprofiled": case["summary"]["median"],
                      "diagnostic_traces": traces,
                      "diagnostic_waits": trace_waits(run / ("case%d.both.kernels.csv" % case["case"])),
                      "fingerprints": workload["workload_fingerprints"]})
    return {"index": index, "arm": arm, "directory": run.name,
            "unprofiled": {key: data[key] for key in ("sampling_ms", "prepass_ms", "standalone_sum_ms", "both_ms")},
            "both_ms": data["both_ms"], "cases": cases,
            "artifact_sha256": {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                                for name in ("benchmark.json", "renderer.metadata.json")}}


def analyze(root):
    assert (root / "completed.txt").exists(), "incomplete ABBA batch"
    identities, fingerprints = [], {}
    records = [validate_run(root, index, arm, identities, fingerprints)
               for index, arm in enumerate(("A", "B", "B", "A"))]
    comparison = paired_summary(records)
    groups = {}
    for arm, label in (("A", "default"), ("B", "high")):
        selected = [r for r in records if r["arm"] == arm]
        cases = [c for r in selected for c in r["cases"]]
        primary = {key: statistics.mean(r["unprofiled"][key] for r in selected)
                   for key in selected[0]["unprofiled"]}
        traces = {mode: {key: statistics.mean(c["diagnostic_traces"][mode][key] for c in cases)
                         for key in cases[0]["diagnostic_traces"][mode]}
                  for mode in ("sampling", "prepass", "both", "both_reverse")}
        waits = {key: statistics.mean(c["diagnostic_waits"]["mean"][key] for c in cases)
                 for key in cases[0]["diagnostic_waits"]["mean"]}
        groups[label] = {"unprofiled": primary, "separate_CUPTI_diagnostics": traces,
                         "sampling_waits_in_both": waits,
                         "both_process_mean_range_ms": [min(r["both_ms"] for r in selected), max(r["both_ms"] for r in selected)]}
    return {"status": "complete_and_validated", "date": "2026-10-13", "timezone": "Asia/Shanghai",
            "scope": "Jetson Orin 16 SM; fixed real-input Graph stage microbenchmark, NOT whole-frame FPS",
            "qualification_profile": "full_mlp_overlap_graph", "actual_variant": "full_mlp_overlap_graph_pb0",
            "qualification_sha256": QUALIFICATION_SHA,
            "version_names": dict(VERSION_NAMES),
            "protocol": {"outer_order": ["default", "high", "high", "default"], "outer_ABBA_blocks": 1,
                         "cases_per_process": 2, "inner_standalone_vs_both_ABBA_blocks_per_case": 4,
                         "unprofiled_replays_per_timing_batch": 40, "warmup_replays": 10,
                         "separate_CUPTI_replays_per_case_and_mode": 8,
                         "graph_backend_both_arms": "same CUDA Driver flags=8; only captured sampling priority differs (0 versus -3)",
                         "sampling_serial": True, "SM_partition": False,
                         "workload_fingerprints_identical_across_four_processes": True,
                         "continuous_foreign_process_and_source_binary_audits": "passed",
                         "gpu_clock_hz_checked_before_after_each_process": 1300500000},
            "comparison_unprofiled": comparison, "groups": groups, "runs": records,
            "warnings": ["sampling kernel sum and prepass duration in both come from separate CUPTI diagnostics; both total comes from unprofiled CUDA events.",
                         "Kernel lifetime overlap includes stalls, not instruction-active overlap. Sampling span contains gaps and excludes waiting before the first sampling kernel.",
                         "One outer ABBA block is a bounded experiment, not a confidence interval or a full-renderer FPS claim."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.results)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"comparison": report["comparison_unprofiled"], "groups": report["groups"]}, indent=2))
