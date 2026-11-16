"""Audited fixed-8+8 versus duplicate-boundary 8->16 prepare comparison."""
import argparse
import json
import math
from pathlib import Path
import statistics

from analyze_green_contexts import load_run, validate_completed
from profile_hexplane_reduction_activity import analyze as analyze_kernels


def historical_prepare_breakdown(path, summary, name):
    """Reapply the current duplicate/sort boundary to the exact old trace cohort."""
    directory = path.parent / summary.get("trace_directory", "green-trace")
    validate_completed(directory)
    samples = []
    for index in (5, 6):
        row = load_run(directory, index, True, "on")
        assert row["identity"][0] == summary["qualification_sha256"]
        assert not row["foreign_processes"]
        for trial in row["activity"]["traces"]:
            samples.extend(analyze_kernels(directory / Path(trial["path"]).name,
                                           50, row["activity"]["config"]))
    assert len(samples) == summary["timelines"][name]["sample_count"] == 196
    # Check that this is the same cohort/definition, not a different old run.
    for field in ("prepare_span_ms", "sampling_kernel_sum_ms", "remaining_prepass_ms"):
        assert math.isclose(statistics.mean(s[field] for s in samples),
                            summary["timelines"][name]["mean"][field], abs_tol=1e-12)
    fields = ("prepare_through_duplicate_ms", "duplicate_kernel_ms",
              "duplicate_to_sort_gap_ms", "sort_kernel_sum_ms",
              "sort_through_ranges_span_ms", "identify_ranges_kernel_ms")
    return {"sample_count": len(samples), "trace_directory": str(directory),
            "mean": {field: statistics.mean(s[field] for s in samples) for field in fields},
            "scope": "Local reanalysis of the original archived kernel timestamps; no new GPU measurement."}


def arm(row):
    resources, config = row["green_resources"], row["activity"]["config"]
    enabled = resources["prepare_handoff_enabled"]
    assert type(enabled) is bool and config["green_prepare_handoff"] == enabled
    assert resources["deform_sms"] == resources["raster_prepare_sms"] == 8
    assert resources["raster_prepass_sms"] == resources["tail_sms"] == 16
    assert resources["raster_prepare_suffix_sms"] == (16 if enabled else 8)
    assert resources["prepare_handoff_boundary"] == ("after_duplicateWithKeys" if enabled else None)
    return "handoff8_to16" if enabled else "fixed8_8"


def read_row(directory, index, enabled, collector):
    stem = "{:02d}-green1-handoff{}-{}".format(index, int(enabled), collector)
    row = load_run(directory, index, True, collector, stem_override=stem)
    assert arm(row) == ("handoff8_to16" if enabled else "fixed8_8")
    meta = json.loads((directory / (stem + ".json")).read_text())
    assert meta["pipeline_execution_counts"].get("raster_prepare_handoffs", 0) == (49 if enabled else 0)
    return row


def summarize(root, historical_four=None, historical_eight=None):
    formal_dir = root / "handoff-formal"
    validate_completed(formal_dir)
    rows = [read_row(formal_dir, i, bool(enabled), "off")
            for i, enabled in enumerate([0, 1, 1, 0, 0, 1, 1, 0])]
    identity = rows[0]["identity"]
    assert all(row["identity"] == identity and not row["activity"]["traces"] for row in rows)
    accepted, rejected = [], {}
    for offset in (0, 4):
        block = rows[offset:offset + 4]
        bad = {row["source"]: row["foreign_processes"] for row in block if row["foreign_processes"]}
        if bad:
            rejected[str(offset // 4)] = bad
        else:
            accepted.extend(block)
    assert accepted, "All formal ABBA blocks contaminated"
    result = {"kind": "jetson_green_prepare_duplicate_handoff", "default_enabled": False,
        "actual_variant": "full_mlp_overlap_graph_pb0", "qualification_sha256": identity[0],
        "pipeline_sha256": identity[3], "rasterizer_binary_sha256": identity[4],
        "qualification_profile": rows[0]["qualification_profile"],
        "workload": {"scene": "flame_steak", "iteration": 14000, "gaussians": 111525,
                     "resolution": [1352, 1014]},
        "gpu_clock_hz_required_and_checked": 1300500000,
        "formal_policy": "Two ABBA blocks, 4 trials x 50 frames/process, 20 warmup, collector off. Reject whole blocks only on foreign-process audits.",
        "rejected_formal_blocks": rejected,
        "formal_process_audits": {row["source"]: row["process_ownership"] for row in rows},
        "formal": {}, "formal_abba_blocks": [], "quality": {}, "timelines": {}}
    for offset in (0, 4):
        if str(offset // 4) in rejected:
            continue
        block = rows[offset:offset + 4]
        fixed = statistics.median(row["fps"] for row in block if arm(row) == "fixed8_8")
        handoff = statistics.median(row["fps"] for row in block if arm(row) == "handoff8_to16")
        result["formal_abba_blocks"].append({"block": offset // 4,
            "fixed8_8_median_fps": fixed, "handoff8_to16_median_fps": handoff,
            "fps_change_percent": 100 * (handoff / fixed - 1)})
    for name in ("fixed8_8", "handoff8_to16"):
        selected = [row for row in accepted if arm(row) == name]
        fps = statistics.median(row["fps"] for row in selected)
        result["formal"][name] = {"median_fps": fps, "ms_per_frame_from_fps": 1000 / fps,
            "process_count": len(selected), "min_process_fps": min(row["fps"] for row in selected),
            "max_process_fps": max(row["fps"] for row in selected),
            "runs": [{"source": row["source"], "fps": row["fps"]} for row in selected]}
    result["fps_change_percent_vs_fixed8_8"] = 100 * (
        result["formal"]["handoff8_to16"]["median_fps"] / result["formal"]["fixed8_8"]["median_fps"] - 1)
    quality_dir = root / "handoff-quality"
    validate_completed(quality_dir)
    for index, name in enumerate(("fixed8_8", "handoff8_to16")):
        d = json.loads((quality_dir / ("{:02d}-green1-handoff{}-off.json".format(index, index))).read_text())
        assert d["passed"] and all(gate["passed"] for gate in d["gates"])
        g = d["modes"]["tacker"]["green_contexts"]
        assert g["enabled"] and g["prepare_handoff_enabled"] == bool(index)
        assert g["deform_sms"] == g["raster_prepare_sms"] == 8
        assert g["raster_prepare_suffix_sms"] == (16 if index else 8)
        result["quality"][name] = d["deltas"]
    trace_dir = root / "handoff-trace"
    validate_completed(trace_dir)
    schedule = [(0, "off"), (0, "on"), (0, "on"), (0, "off"),
                (1, "off"), (1, "on"), (1, "on"), (1, "off")]
    traces = [read_row(trace_dir, i, bool(enabled), collector)
              for i, (enabled, collector) in enumerate(schedule)]
    assert all(row["identity"] == identity for row in traces)
    result["trace_process_audits"] = {row["source"]: row["process_ownership"] for row in traces}
    for offset, name in ((0, "fixed8_8"), (4, "handoff8_to16")):
        block = traces[offset:offset + 4]
        assert not any(row["foreign_processes"] for row in block), "Contaminated whole trace block: " + name
        off = statistics.median(row["fps"] for row in block if row["collector"] == "off")
        on = statistics.median(row["fps"] for row in block if row["collector"] == "on")
        delta = 100 * (on / off - 1)
        assert abs(delta) <= 2, "Collector perturbation bound exceeded: " + name
        samples = [sample for row in block if row["collector"] == "on"
                   for trial in row["activity"]["traces"] for sample in trial["samples"]]
        assert len(samples) == 196
        fields = [k for k in samples[0] if k.endswith("_ms") or k == "merge_kernel_count"]
        result["timelines"][name] = {"sample_count": len(samples),
            "collector_fps_delta_percent": delta,
            "mean": {k: statistics.mean(s[k] for s in samples) for k in fields},
            "median": {k: statistics.median(s[k] for s in samples) for k in fields},
            "stream_identities": samples[0]["stream_identity_context_and_stream"],
            "sampling_prepare_overlap_percent": 100 * sum(s["sampling_prepare_kernel_overlap_ms"] for s in samples) /
                sum(s["sampling_kernel_sum_ms"] for s in samples)}
    result["historical_references"] = {}
    for name, path, key in (("four_plus_twelve", historical_four, "green4_12"),
                             ("eight_plus_eight", historical_eight, "green8_8")):
        if path is None:
            continue
        old = json.loads(path.read_text())
        assert old["qualification_sha256"] == identity[0] and old["workload"] == result["workload"]
        assert old["partition"]["raster_prepass_sms"] == old["partition"]["tail_sms"] == 16
        result["historical_references"][name] = {"path": str(path), "formal": old["formal"][key],
            "timeline": old["timelines"][key], "baseline_formal": old["formal"]["baseline"],
            "prepare_boundary_reanalysis": historical_prepare_breakdown(path, old, key),
            "scope": "Historical separate batch; native host scheduling has since changed. Not a fresh direct ABBA against this configuration."}
        result["historical_references"][name]["handoff_fps_change_percent"] = 100 * (
            result["formal"]["handoff8_to16"]["median_fps"] / old["formal"][key]["median_fps"] - 1)
        if name == "eight_plus_eight":
            result["historical_references"][name]["fixed8_8_repeat_fps_drift_percent"] = 100 * (
                result["formal"]["fixed8_8"]["median_fps"] / old["formal"][key]["median_fps"] - 1)
    result["scope"] = "Concurrent Graph pipeline timings, not standalone. Full-device stream eligibility is not guaranteed occupancy. No new NCU bandwidth analysis. Default remains off."
    fixed = result["timelines"]["fixed8_8"]["mean"]
    handoff = result["timelines"]["handoff8_to16"]["mean"]
    result["implementation"] = {
        "environment": "TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF=1",
        "requires": {"TACKER_EXPERIMENTAL_GREEN_CONTEXTS": "1", "TACKER_GREEN_DEFORM_SMS": "8"},
        "boundary": "after_duplicateWithKeys",
        "dependencies": "Full-device stream waits for duplicate completion and the complete next-frame Deform prefix, including feature merge and mask.",
        "mechanism": "Submit subsequent sort and range kernels to the ordinary primary-context stream. Do not resize Green Context partitions or change already-running kernels.",
        "drain": "No next-frame Deform prefix on the last frame; its prepare remains on the 8-SM stream.",
        "extra_cpu_or_device_synchronize": False,
        "existing_num_rendered_d2h_synchronize_removed": False
    }
    result["conclusion"] = {
        "promote_to_default": False,
        "handoff_stream_boundary_and_both_dependencies_verified": True,
        "prepare_span_ms_saved": fixed["prepare_span_ms"] - handoff["prepare_span_ms"],
        "prepare_suffix_span_ms_saved": fixed["sort_through_ranges_span_ms"] - handoff["sort_through_ranges_span_ms"],
        "sort_kernel_ms_saved": fixed["sort_kernel_sum_ms"] - handoff["sort_kernel_sum_ms"],
        "identify_ranges_kernel_ms_saved": fixed["identify_ranges_kernel_ms"] - handoff["identify_ranges_kernel_ms"],
        "interpretation": "Small throughput improvement in this Graph qualification workload, from faster prepare suffix kernels; sampling, prepass and mixed tail remain nearly unchanged. Historical 4+12 is a separate batch, not a new direct ABBA.",
        "correctness_scope": "Ten-frame serial-versus-Tacker quality gates in both arms. Source/binary byte stability checks verify files, not cross-arm bitwise image equality. No new compute-sanitizer run in this experiment."
    }
    cpu_path = root / "local_cpu_checks.json"
    if cpu_path.exists():
        result["local_cpu_checks"] = json.loads(cpu_path.read_text())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-four", type=Path)
    parser.add_argument("--historical-eight", type=Path)
    args = parser.parse_args()
    report = summarize(args.root, args.historical_four, args.historical_eight)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
