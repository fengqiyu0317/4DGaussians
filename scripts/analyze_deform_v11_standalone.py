"""Aggregate independent synthetic v11 standalone processes, excluding pilots."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paths = sorted(args.directory.glob("run_*.json"))
    assert paths, args.directory
    reports = [json.loads(path.read_text()) for path in paths]
    first = reports[0]
    for report in reports:
        assert report["passed"] and not report["mixed_renderer_executed"]
        for key in ("rows", "seed", "data_sha256", "source_sha256", "precision_fp32",
                    "precision_matched", "compiler_internal_cudagraphs", "resources", "protocol"):
            assert report[key] == first[key], key
        assert report["compiler_internal_cudagraphs"] is False
        assert report["dynamo_unique_graphs"] == 2
        for snapshot in (report["gpu_before"], report["gpu_after"]):
            assert set(snapshot["jetson_frequency_hz"].values()) == {1300500000}
    measurements = {}
    for mode in first["measurements"]:
        medians = [r["measurements"][mode]["median_cuda_ms"] for r in reports]
        wall = [r["measurements"][mode]["median_wall_ms"] for r in reports]
        trials = [t["cuda_ms"] for r in reports for t in r["measurements"][mode]["trials"]]
        measurements[mode] = {"median_cuda_ms": statistics.median(medians),
                              "min_process_median_cuda_ms": min(medians),
                              "max_process_median_cuda_ms": max(medians),
                              "process_medians_cuda_ms": medians,
                              "median_wall_ms": statistics.median(wall),
                              "pooled_trial_median_cuda_ms": statistics.median(trials),
                              "trial_count": len(trials)}
    def ms(mode):
        return measurements[mode]["median_cuda_ms"]
    native = ms("v11_standalone_graph")
    comparisons = {}
    for precision in ("fp32", "matched"):
        candidate = ms("compile_graph_" + precision)
        eager = ms("eager_" + precision)
        comparisons[precision] = {
            "compile_graph_speedup_over_eager": eager / candidate,
            "compile_graph_latency_reduction_percent": (1-candidate/eager)*100,
            "native_graph_speedup_over_compile_graph": candidate / native,
            "native_graph_latency_reduction_percent_vs_compile_graph": (1-native/candidate)*100,
            "adding_graph_to_compile_latency_reduction_percent":
                (1-candidate/ms("compile_" + precision))*100,
        }
    quality = {key: {"max_abs_across_heads_and_processes": max(
                        head["max_abs"] for report in reports for head in report["quality"][key]),
                    "max_relative_l2_across_heads_and_processes": max(
                        head["relative_l2"] for report in reports for head in report["quality"][key])}
               for key in ("native_vs_matched", "native_vs_original_fp32", "native_384_vs_128",
                           "compile_fp32", "compile_matched")}
    output = {"scope": first["scope"], "rows": first["rows"], "seed": first["seed"],
              "device": first["device"], "torch": first["torch"], "cuda": first["cuda"],
              "independent_processes": len(reports), "protocol_per_process": first["protocol"],
              "aggregation": "median of independent process medians; CUDA event time per call",
              "measurements": measurements, "comparisons": comparisons, "quality": quality,
              "precision_fp32": first["precision_fp32"], "precision_matched": first["precision_matched"],
              "matched_precision_compiler_bridge": first["matched_precision_compiler_bridge"],
              "native_adapter": first["native_adapter"], "resource_control": first["resource_control"],
              "qualification_profile": None, "mixed_renderer_executed": False,
              "excluded": first["excluded"], "data_sha256": first["data_sha256"],
              "source_sha256": first["source_sha256"], "resources": first["resources"],
              "source_files": [str(p) for p in paths]}
    (args.directory / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    with (args.directory / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "median_cuda_ms", "min_process_median_cuda_ms",
                         "max_process_median_cuda_ms", "median_wall_ms"])
        for mode, value in measurements.items():
            writer.writerow([mode] + [value[key] for key in
                                      ("median_cuda_ms", "min_process_median_cuda_ms",
                                       "max_process_median_cuda_ms", "median_wall_ms")])
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
