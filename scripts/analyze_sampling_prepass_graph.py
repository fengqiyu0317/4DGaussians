"""Validate the completed clean batch; never admit rejected timing attempts."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from analyze_sampling_kernel_activity import foreign_python_processes
from benchmark_sampling_prepass_graph import analyze_trace, summarize_abba, QUALIFICATION_SHA


def analyze(root):
    run = root / "run6"
    assert (run / "completed.txt").exists()
    data = json.loads((run / "benchmark.json").read_text())
    metadata = json.loads((run / "renderer.metadata.json").read_text())
    expected_hashes = {
        "benchmark.json": "45456b36a9ab9b9c094f0228f2f517916e57b6293dd5c94e55c437c58e5e8593",
        "renderer.metadata.json": "99be96e82198d02c29d36bc05b032a50822e51ac0ac17dab30c40223ff484da2",
    }
    for name, expected in expected_hashes.items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == expected
    assert data["actual_variant"] == metadata["selected_variant_id"] == "full_mlp_overlap_graph_pb0"
    assert data["qualification_sha256"] == metadata["qualification_profile_sha256"] == QUALIFICATION_SHA
    assert metadata["passed"] and metadata["qualification_mode_executed"]
    assert metadata["actual_execution_mode"] == "tacker" and metadata["tacker_fallback_reason"] is None
    assert metadata["post_run"]["byte_stability"]["verified"]
    assert not metadata["green_contexts"]["enabled"] and data["correctness_passed"]
    assert (metadata["gaussian_count"], metadata["image_width"], metadata["image_height"]) == (111525, 1352, 1014)
    for name in ("before", "after"):
        lines = (run / (name + ".source_check.txt")).read_text().splitlines()
        assert lines and all(line.endswith(": OK") for line in lines)
        assert "GR3D_FREQ 0%" in (run / (name + ".tegrastats.txt")).read_text()
        assert not foreign_python_processes((run / (name + ".processes.txt")).read_text())
    own_driver = "/home/jetson/sampling-prepass-graph-20261012-27jBBs/scripts/benchmark_sampling_prepass_graph.py"
    foreign = [line for line in foreign_python_processes((run / "during_processes.txt").read_text())
               if own_driver not in line]
    assert not foreign, foreign
    cases = []
    for result in data["cases"]:
        summary = summarize_abba(result["records"])
        assert summary == result["summary"]
        traces = {}
        for recorded in result["diagnostic_traces"]:
            mode = recorded["mode"]
            path = run / ("case%d.%s.kernels.csv" % (result["case"], mode))
            computed = analyze_trace(path, mode, 8, 1954)
            assert recorded == computed
            traces[mode] = recorded["mean"]
        assert set(traces) == {"sampling", "prepass", "both", "both_reverse"}
        cases.append({"case": result["case"], "workload": result["workload"],
                      "unprofiled_ABBA": summary,
                      "unprofiled_serial_control_ms": result["serial_control"]["mean_ms"],
                      "unprofiled_reverse_order_control_ms": result["reverse_branch_order_control"]["mean_ms"],
                      "diagnostic_trace_means": traces})
    assert len(cases) == 2
    mean_trace = {mode: {key: statistics.mean(case["diagnostic_trace_means"][mode][key] for case in cases)
                         for key in cases[0]["diagnostic_trace_means"][mode]}
                  for mode in ("sampling", "prepass", "both", "both_reverse")}
    sampling_extra = mean_trace["both"]["sampling_kernel_sum_ms"] - mean_trace["sampling"]["sampling_kernel_sum_ms"]
    raster_extra = mean_trace["both"]["prepass_ms"] - mean_trace["prepass"]["prepass_ms"]
    result = {
        "status": "completed_and_validated", "date": "2026-10-12", "timezone": "Asia/Shanghai",
        "scope": "Jetson Orin 16 SM, no partition, exact original 12 native sampling kernels plus v12 Raster prepass 1954 tiles; fixed real inputs, warmed isolated Graph microbenchmark, NOT full-renderer FPS.",
        "qualification_profile": data["qualification_profile"], "qualification_sha256": QUALIFICATION_SHA,
        "actual_variant": data["actual_variant"], "gpu_clock_hz_checked_before_after": 1300500000,
        "admitted_batch": "run6", "continuous_foreign_process_audit": "passed",
        "source_binary_and_correctness_checks": "passed", "download_sha256_verified": expected_hashes,
        "rejected_attempts": {
            "run1": "no valid timings: root filesystem full during sealed private binary snapshots",
            "run2": "no valid timings: same root filesystem exhaustion",
            "run3": "NOT pure sampling: .out overload inserts 12 memcpy32_post kernels; trace init also failed without sudo",
            "run4": "NOT pure sampling: trace strictly rejected 24 kernels instead of 12; all timings excluded",
            "run5": "correct kernels, but continuous process audit caught foreign QwenMMDiT import process; entire batch excluded regardless of measured times"},
        "protocol": {"real_input_cases": 2, "ABBA_blocks_per_case": 4, "replays_per_timing_batch": 40,
                     "warmup_replays_per_timing_batch": 10, "formal_timing_profiler_attached": False,
                     "both_Graph": "two independent default-priority stream branches forked from and joined to one control stream; no CPU submission gap between branch launches",
                     "standalone_sum": "isolated sampling Graph elapsed + isolated prepass Graph elapsed",
                     "trace": "separate CUPTI concurrent-kernel activity, only AFTER all unprofiled timing controls; 8 replays per arm per case, exactly 12/1 kernels and distinct streams verified",
                     "no_production_source_or_binary_edits": True},
        "unprofiled_summary": {key: data[key] for key in (
            "sampling_ms", "prepass_ms", "standalone_sum_ms", "both_ms", "both_over_standalone_sum", "time_saved_percent", "speedup")},
        "cases": cases, "diagnostic_trace_means": mean_trace,
        "diagnostic_interpretation": {
            "kernel_lifetime_overlap_ms": mean_trace["both"]["sampling_prepass_kernel_overlap_ms"],
            "sampling_kernel_sum_increase_ms": sampling_extra,
            "sampling_kernel_sum_increase_percent": 100 * sampling_extra / mean_trace["sampling"]["sampling_kernel_sum_ms"],
            "prepass_kernel_duration_increase_ms": raster_extra,
            "prepass_kernel_duration_increase_percent": 100 * raster_extra / mean_trace["prepass"]["prepass_ms"],
            "sampling_span_in_both_ms": mean_trace["both"]["sampling_span_ms"],
            "warning": "CUPTI intervals include in-kernel waits, NOT instruction-active overlap. Sampling span also includes gaps between its 12 kernels. Diagnostic counters/durations are not substituted for unprofiled primary timings."},
        "conclusion": "Small measured stage-level gain (~1.75%, ~0.110 ms); only partial kernel-lifetime overlap and concurrent slowdowns offset much of it. Not evidence of a corresponding whole-frame FPS improvement."}
    result["unprofiled_summary"]["saved_ms"] = data["standalone_sum_ms"] - data["both_ms"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = analyze(options.results)
    options.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["unprofiled_summary"], indent=2))
    print(json.dumps(result["diagnostic_interpretation"], indent=2))
