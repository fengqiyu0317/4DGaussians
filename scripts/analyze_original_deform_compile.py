"""Summarize the four-arm inference experiment with paired process bootstrap."""
import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

MODES = ("eager", "eager_graph", "compile", "compile_graph")


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def comparison(records, target, reference, scope):
    by_round = {}
    for record in records:
        by_round.setdefault(record["index"] // 4, {})[record["mode"]] = record
    ratios = [r[target]["measurements"][scope]["median_fps"] /
              r[reference]["measurements"][scope]["median_fps"]
              for r in by_round.values()]
    rng = random.Random(0)
    bootstraps = [statistics.median(rng.choices(ratios, k=len(ratios)))
                  for _ in range(10000)]
    return {"paired_round_ratios": ratios, "median_paired_speedup": statistics.median(ratios),
            "paired_bootstrap_95ci": [quantile(bootstraps, .025), quantile(bootstraps, .975)],
            "bootstrap_replicates": len(bootstraps), "bootstrap_seed": 0}


def analyze(device):
    formal_path, quality_path = device / "formal_10x50/summary.json", device / "quality_v2/summary.json"
    formal, quality = json.loads(formal_path.read_text()), json.loads(quality_path.read_text())
    assert formal["passed"] and quality["passed"]
    assert len(formal["records"]) == 40 and len(quality["records"]) == 4
    rows, reports = [], []
    for record in formal["records"]:
        report = json.loads((device / "formal_10x50" / Path(record["path"]).name).read_text())
        assert report["passed"] and report["frames"] == 50 and report["warmup_frames"] == 20
        assert report["source_stable"] and not report["mixed_kernel"]
        assert not report["compiler_internal_cudagraphs"]
        for scope in ("deform", "end_to_end"):
            assert report["measurements"][scope]["trials"][0]["graph_replays"] == (
                50 if record["mode"].endswith("_graph") else 0)
        reports.append(report)
    quality_reports = []
    for record in quality["records"]:
        report = json.loads((device / "quality_v2" / Path(record["path"]).name).read_text())
        assert report["passed"] and report["quality"]["passed"]
        assert len(report["quality"]["per_view"]) == 50
        quality_reports.append(report)
    # Model and benchmark implementation must be frozen across all arms and phases.
    frozen = [{Path(path).name: digest for path, digest in r["source_sha256"].items()}
              for r in reports + quality_reports]
    assert all(hashes == frozen[0] for hashes in frozen)
    environment = {k: reports[0][k] for k in ("gpu_name", "torch", "cuda_runtime", "triton", "precision",
                    "scope", "renderer_activations", "graph_input_policy", "source_sha256")}
    environment["compiler_compatibility_patch"] = next(
        (r["compiler_compatibility_patch"] for r in reports if r["compiler_compatibility_patch"]), None)
    output = {"passed": True, "environment": environment, "by_mode": formal["by_mode"],
              "quality": {}, "comparisons": {}, "setup_times": {}, "memory_mib": {}}
    for mode in MODES:
        children = [r for r in reports if r["mode"] == mode]
        checked = next(r for r in quality_reports if r["mode"] == mode)
        output["quality"][mode] = {k: checked["quality"][k] for k in (
            "passed", "means", "candidate_minus_reference", "max_abs_deform_errors", "max_image_rmse")}
        output["setup_times"][mode] = {
            "median_warm_cache_compile_seconds": statistics.median(r["compile_setup_seconds"] for r in children),
            "median_graph_capture_seconds": statistics.median(r["graph_setup_seconds"] for r in children),
            "quality_process_compile_seconds": checked["compile_setup_seconds"]}
        output["memory_mib"][mode] = {scope: {
            "median_peak_allocated": statistics.median(
                r["measurements"][scope]["peak_allocated_bytes"] for r in children) / 2**20,
            "median_peak_reserved": statistics.median(
                r["measurements"][scope]["peak_reserved_bytes"] for r in children) / 2**20}
            for scope in ("deform", "end_to_end")}
        for scope in ("deform", "end_to_end"):
            rows.append({"device": device.name, "mode": mode, "scope": scope,
                         **formal["by_mode"][mode][scope]})
    pairs = (("eager_graph", "eager"), ("compile", "eager"), ("compile_graph", "eager"),
             ("compile_graph", "compile"), ("compile_graph", "eager_graph"))
    for target, reference in pairs:
        output["comparisons"][target + "_vs_" + reference] = {
            scope: comparison(formal["records"], target, reference, scope)
            for scope in ("deform", "end_to_end")}
    output["artifact_sha256"] = {str(p.relative_to(device)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(device.rglob("*.json"))}
    control_path = device / "capture_safe_eager_control.json"
    if control_path.exists():
        control = json.loads(control_path.read_text())
        assert control["passed"] and control["diagnostic_only"]
        output["capture_safe_eager_control"] = {
            "independent_processes": 1,
            "trials_per_scope": len(control["measurements"]["deform"]["trials"]),
            "measurements": control["measurements"],
            "note": "Supplementary within-process control, not included in the formal paired bootstrap.",
            "comparisons": {scope: {
                "safe_indexing_speedup_vs_original_eager": (
                    control["measurements"][scope]["median_fps"] /
                    formal["by_mode"]["eager"][scope]["median_fps"]),
                "graph_speedup_vs_safe_indexing_eager": (
                    formal["by_mode"]["eager_graph"][scope]["median_fps"] /
                    control["measurements"][scope]["median_fps"])}
                for scope in ("deform", "end_to_end")}}
    return output, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--devices", nargs="+", default=["jetson"])
    args = parser.parse_args()
    result = {"workload": "flame_steak iteration 14000; 111525 Gaussians; 1352x1014; test views 0..49",
              "protocol": "4 arms x 10 independent processes x 50 frames; 20 warmup frames per scope; mirrored order",
              "timing": "whole-sequence wall time with final CUDA synchronization; no per-frame I/O; setup excluded",
              "deployment_boundary": "inference with fixed Gaussian count and attributes; no default renderer change",
              "attribution_boundary": "eager uses original coordinate indexing; other arms use equivalent capture-safe stack indexing",
              "quality_gate": {"psnr_drop_max_db": .05, "ssim_drop_max": 1e-4, "lpips_increase_max": 1e-4},
              "devices": {}}
    rows = []
    for name in args.devices:
        result["devices"][name], device_rows = analyze(args.result_root / name)
        rows.extend(device_rows)
    (args.result_root / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    with (args.result_root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, device in result["devices"].items():
        print(name)
        for mode, measurements in device["by_mode"].items():
            print(mode, "deform_ms", round(measurements["deform"]["median_wall_ms"], 4),
                  "deform_speedup", round(measurements["deform"]["speedup_vs_eager"], 4),
                  "fps", round(measurements["end_to_end"]["median_fps"], 3),
                  "frame_speedup", round(measurements["end_to_end"]["speedup_vs_eager"], 4))


if __name__ == "__main__":
    main()
