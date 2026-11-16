"""Aggregate same-computation dedicated kernels against production Both."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.glob("run*/benchmark.json"))
    assert len(paths) == 3, paths
    processes, fingerprints, resources = [], [], []
    for path in paths:
        d = json.loads(path.read_text())
        assert d["passed"] and d["rounds"] == 6 and d["repeats"] == 30
        assert len(d["cases"]) == 2 and d["actual_variant"] == "full_mlp_overlap_graph_pb0"
        assert (path.parent / "completed.txt").exists()
        metadata = json.loads((path.parent / "renderer.metadata.json").read_text())
        assert metadata["passed"] and metadata["tacker_fallback_reason"] is None
        assert metadata["qualification_mode_executed"] and metadata["post_run"]["byte_stability"]["verified"]
        freq = re.findall(r"^\d{9,10}$", (path.parent / "during_processes_and_frequency.txt").read_text(), re.M)
        assert freq and set(freq) == {"1300500000"}, set(freq)
        fp = [{k:v for k,v in c["workload"].items() if k.endswith("sha256") or k == "step"}
              for c in d["cases"]]
        fingerprints.append(fp)
        values = {}
        arms = sorted({row["arm"] for c in d["cases"] for row in c["batches"]})
        for arm in arms:
            rows = [row for c in d["cases"] for row in c["batches"] if row["arm"] == arm]
            assert len(rows) == 12 and all(len(row["samples_ms"]) == 30 for row in rows)
            values[arm] = statistics.mean(row["mean_ms"] for row in rows)
        processes.append({"run":path.parent.name,"means_ms":values,"frequency_samples":len(freq),
                          "pipeline_execution_counts":metadata["pipeline_execution_counts"]})
        resources.append(d["cases"][0]["workload"]["graph_audits"])
    assert all(fp == fingerprints[0] for fp in fingerprints), "independent process inputs differ"
    means = {arm: statistics.mean(p["means_ms"][arm] for p in processes) for arm in processes[0]["means_ms"]}
    comparisons = {}
    for name, serial, mixed, raster in (
        ("v11_full", "serial_full", "both_v11", "raster_full"),
        ("v13_tail", "serial_tail", "both_v13", "raster_tail"),
        ("v13_full_including_prepass", "serial_full", "complete_v13", "raster_full"),
    ):
        baseline, fused = means[serial], means[mixed]
        total = means["deform"] + means[raster]
        comparisons[name] = {"independent_deform_ms":means["deform"], "independent_raster_ms":means[raster],
                             "standalone_sum_ms":total, "independent_serial_graph_ms":baseline,
                             "mixed_graph_ms":fused, "saved_vs_serial_ms":baseline-fused,
                             "time_reduction_vs_serial_percent":100*(1-fused/baseline),
                             "speedup_vs_serial":baseline/fused,
                             "speed_increase_vs_serial_percent":100*(baseline/fused-1),
                             "time_reduction_vs_standalone_sum_percent":100*(1-fused/total),
                             "baseline_arm":serial, "mixed_arm":mixed}
    report = {"passed":True,"date":"2026-10-15","device":"Jetson AGX Orin sm_87, 16 SM, 1.3005 GHz",
              "workload":"flame_steak iteration 14000, test split, 1352x1014, 111525 Deform rows / 6971 tiles",
              "qualification_profile":"full_mlp_overlap_graph","actual_variant":"full_mlp_overlap_graph_pb0",
              "qualification_sha256":"a13d87f6c4a22a24b20f83cb3c66a009e759b739b498c9ff55f8c35c9b82d23a",
              "same_computation":"Dedicated Deform calls exact mixed device-v7; dedicated Raster calls exact mixed general_ptb_render_raster_optimized<NUM_CHANNELS,true>; all outputs bitwise identical on frozen real inputs.",
              "normal_resources":"Deform 128 threads and own descriptor/scratch; Raster 256 threads and own Raster shared arrays; no mixed warp padding or resource cap. Actual compiled attributes in graph_audits.",
              "scope":"fixed real inputs, hot Driver Graph flags=8 priority 0; excludes prefix, prepare, extraction, packing, compile and capture; complete_v13 is sequential prepass then production Both when both inputs are ready; not full-frame FPS",
              "aggregation":"mean of three independent process means, two input cases/process, six mirrored rounds/case, 30 timed replays/arm/batch, 10 warmup replays/batch",
              "samples_per_arm":1080,"independent_processes":3,"means_ms":means,
              "process_mean_ranges_ms":{arm:[min(p["means_ms"][arm] for p in processes),max(p["means_ms"][arm] for p in processes)] for arm in means},
              "comparisons":comparisons,"per_process":processes,"input_fingerprints":fingerprints[0],
              "graph_audits":resources[0],
              "correctness":"All nine arms checked after output poisoning before and after timings; Deform/color/depth bitwise equality, outside-range outputs remain NaN; Graph nodes and real resource attributes inspected; all three process functional fingerprints match.",
              "raw_report_sha256":{str(p.relative_to(args.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.root / "summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"means_ms":means,"comparisons":comparisons},indent=2))


if __name__ == "__main__": main()
