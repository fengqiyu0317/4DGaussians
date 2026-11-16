"""Validate and aggregate independently collected Graph mixed-tail controls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root
    reports = sorted(root.glob("run*/benchmark.json"))
    assert len(reports) == 3, reports
    per_process, fingerprints, resource_audits, metadata_refs = [], [], [], []
    for path in reports:
        report = json.loads(path.read_text())
        assert report["passed"] and report["actual_variant"] == "full_mlp_overlap_graph_pb0"
        assert report["rounds"] == 6 and report["repeats"] == 30 and len(report["cases"]) == 2
        metadata_path = path.parent / "renderer.metadata.json"
        metadata = json.loads(metadata_path.read_text())
        assert metadata["passed"] and metadata["tacker_fallback_reason"] is None
        assert metadata["qualification_mode_executed"] and metadata["selected_variant_id"] == "full_mlp_overlap_graph_pb0"
        assert metadata["post_run"]["byte_stability"]["verified"]
        assert (path.parent / "completed.txt").exists()
        text = (path.parent / "during_processes_and_frequency.txt").read_text()
        frequencies = [int(x) for x in re.findall(r"^\d{9,10}$", text, re.M)]
        assert frequencies and set(frequencies) == {1300500000}, set(frequencies)
        data = {"run": path.parent.name, "frequency_samples": len(frequencies), "variants": {}}
        fp = []
        for case in report["cases"]:
            w = case["workload"]
            fp.append({key: value for key, value in w.items() if key.endswith("sha256") or key == "pipeline_step"})
            for key, audit in w["graph_audits"].items():
                assert audit["requested_instantiation_flags"] == 8 and len(audit["nodes"]) == 1
                resource_audits.append({"run": path.parent.name, "case": case["case"], "arm": key, **audit})
        fingerprints.append(fp)
        for variant in ("v11", "v13"):
            values = {}
            for mode in ("deform_only", "raster_only", "both"):
                rows = [batch for case in report["cases"] for batch in case["batches"]
                        if batch["variant"] == variant and batch["mode"] == mode]
                assert len(rows) == 12 and all(len(row["samples_ms"]) == 30 for row in rows)
                values[mode + "_ms"] = statistics.mean(row["mean_ms"] for row in rows)
            values["production_both_ms"] = statistics.mean(
                case["production_controls"][variant]["mean_ms"] for case in report["cases"])
            values["diagnostic_both_overhead_percent"] = 100 * (
                values["both_ms"] / values["production_both_ms"] - 1)
            data["variants"][variant] = values
        per_process.append(data)
        metadata_refs.append({"run": path.parent.name, "qualification_profile": metadata["qualification_profile"],
                              "qualification_sha256": metadata["qualification_profile_sha256"],
                              "pipeline_execution_counts": metadata["pipeline_execution_counts"]})
    assert all(fp == fingerprints[0] for fp in fingerprints), "input/output fingerprints differ across processes"
    summary = {"passed": True, "date": "2026-10-15", "device": "Jetson AGX Orin sm_87, 16 SM",
               "workload": "flame_steak, iteration 14000, test split, 1352x1014, 111525 Deform rows",
               "qualification_profile": "full_mlp_overlap_graph", "actual_variant": "full_mlp_overlap_graph_pb0",
               "graph": "CUDA Driver Graph, instantiate flags=8 USE_NODE_PRIORITY, stream priority 0; one audited kernel per replay",
               "timing": "CUDA events; arithmetic mean of three independent process means; each process two real frozen inputs, six mirrored rounds, 30 replays per arm; 10 warmup replays per batch",
               "independent_processes": 3, "samples_per_variant_mode": 1080,
               "scope": "same-envelope diagnostic mixed tail; v11 Raster [0,5440), v13 Raster [1954,5440); both process all 6971 Deform tiles; excludes v13 prepass, prefix, preparation, packing, extraction, compilation and capture",
               "correctness": "both and standalone enabled outputs bitwise match production; outputs poisoned before replay; disabled branch outputs remain NaN; same input/output fingerprints across all three processes",
               "per_process": per_process, "variants": {}, "input_fingerprints": fingerprints[0],
               "qualification_audits": metadata_refs, "resource_audits": resource_audits,
               "limitations": ["Hot fixed-input Graph microbenchmark, not full-frame latency or FPS.",
                               "v13 has 1954 fewer Raster tiles in this tail because they run in its earlier prepass.",
                               "Standalone means the other warp group is disabled while retaining the complete mixed CTA resources.",
                               "Production control is measured separately; primary three-mode results use a common diagnostic symbol per version."]}
    for variant in ("v11", "v13"):
        values = {}
        for mode in ("deform_only", "raster_only", "both", "production_both"):
            samples = [row["variants"][variant][mode + "_ms"] for row in per_process]
            values[mode + "_ms"] = statistics.mean(samples)
            values[mode + "_process_range_ms"] = [min(samples), max(samples)]
        total = values["deform_only_ms"] + values["raster_only_ms"]
        values.update(standalone_sum_ms=total, saved_ms=total-values["both_ms"],
                      fusion_efficiency_percent=100*(total-values["both_ms"])/min(values["deform_only_ms"], values["raster_only_ms"]),
                      diagnostic_both_overhead_percent=100*(values["both_ms"]/values["production_both_ms"]-1))
        summary["variants"][variant] = values
    summary["raw_report_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in reports}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": summary["passed"], "variants": summary["variants"],
                      "input_fingerprints_match": True}, indent=2))


if __name__ == "__main__":
    main()
