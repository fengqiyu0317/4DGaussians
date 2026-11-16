"""Aggregate complete, sealed kernel-only traces without mixing FPS cohorts."""
import argparse
import json
from pathlib import Path
import statistics


def foreign_python_processes(text):
    allowed = ("profile_sampling_kernel_activity.py", "/usr/local/bin/jtop",
               "/usr/share/nvpmodel_indicator/", "/opt/nvidia/pva-allow-2/",
               "/usr/bin/networkd-dispatcher", "/usr/bin/update-manager",
               "-c import json", "-c import pathlib,json")
    found = set()
    for line in text.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[0].isdigit() and parts[2].startswith("python"):
            if not any(item in parts[3] for item in allowed):
                found.add(line.strip())
    return sorted(found)


def sampling_summary(samples):
    total = sum(row["sampling_ms"] for row in samples)
    prepare = sum(row["sampling_prepare_kernel_overlap_ms"] for row in samples)
    prepass = sum(row["sampling_prepass_kernel_overlap_ms"] for row in samples)
    return {
        "sample_count": len(samples),
        "mean_sampling_kernel_sum_ms": total / len(samples),
        "median_sampling_kernel_sum_ms": statistics.median(row["sampling_ms"] for row in samples),
        "mean_raster_prepare_kernel_sum_ms": statistics.mean(row["prepare_kernel_sum_ms"] for row in samples),
        "mean_raster_prepass_kernel_ms": statistics.mean(row["prepass_ms"] for row in samples),
        "mean_overlap_prepare_ms": prepare / len(samples),
        "mean_overlap_prepass_ms": prepass / len(samples),
        "fraction_sampling_overlapped_prepare_percent": 100 * prepare / total,
        "fraction_sampling_overlapped_prepass_percent": 100 * prepass / total,
        "fraction_prepare_overlapped_sampling_percent": 100 * prepare / sum(row["prepare_kernel_sum_ms"] for row in samples),
        "fraction_prepass_overlapped_sampling_percent": 100 * prepass / sum(row["prepass_ms"] for row in samples),
        "frames_with_prepass_overlap": sum(row["sampling_prepass_kernel_overlap_ms"] > 0 for row in samples),
        "mean_first_to_last_sampling_span_overlap_prepass_ms": statistics.mean(
            max(0, min(row["sampling_intervals_ns"][-1][1], row["prepass_interval_ns"][1])
                - max(row["sampling_intervals_ns"][0][0], row["prepass_interval_ns"][0])) / 1e6
            for row in samples),
        "sampling_complete_before_prepass_frames": sum(row["sampling_end_relative_to_prepass_start_ms"] <= 0 for row in samples),
        "median_sampling_end_relative_to_prepass_start_ms": statistics.median(row["sampling_end_relative_to_prepass_start_ms"] for row in samples),
    }


def summarize(directory):
    assert (directory / "completed.txt").exists(), "incomplete/rejected batch: " + str(directory)
    checks = list(directory.glob("*.source_check.txt"))
    assert directory / "final.source_check.txt" in checks, directory
    assert all(all(line.endswith(": OK") for line in path.read_text().splitlines())
               for path in checks), directory
    assert "GR3D_FREQ 0%" in (directory / "final.tegrastats.txt").read_text(), directory
    audits = list(directory.glob("*.during_processes.txt"))
    assert audits, "continuous host-process audit missing"
    foreign = {path.name: foreign_python_processes(path.read_text()) for path in audits}
    # Reject a whole predeclared ABBA block on process-audit evidence, NEVER
    # individual runs selected by FPS. An intact later ABBA block remains valid.
    rejected_blocks = {}
    all_paths = sorted(directory.glob("*.activity.json"))
    assert len(all_paths) % 4 == 0, "expected complete four-process ABBA blocks"
    for start in range(0, len(all_paths), 4):
        block = all_paths[start:start + 4]
        assert [json.loads(path.read_text())["mode"] for path in block] == ["off", "on", "on", "off"]
        reasons = {}
        for path in block:
            audit_name = path.name.replace(".activity.json", ".during_processes.txt")
            assert audit_name in foreign, path
            if foreign[audit_name]:
                reasons[audit_name] = foreign[audit_name]
        if reasons:
            rejected_blocks[start // 4] = {"runs": [path.name for path in block],
                                          "process_audit_reasons": reasons}
    groups = {}
    identity = None
    for path_index, path in enumerate(all_paths):
        data = json.loads(path.read_text())
        metadata_path = path.with_name(path.name.replace(".activity.json", ".json"))
        meta = json.loads(metadata_path.read_text())
        assert meta["passed"] and meta["actual_execution_mode"] == "tacker", path
        assert meta["qualification_mode_executed"] and meta["tacker_fallback_reason"] is None, path
        assert meta["selected_variant_id"] == "full_mlp_overlap_graph_pb0", path
        assert meta["post_run"]["byte_stability"]["verified"], path
        frames = meta["profile_frames"]
        counts = meta["pipeline_execution_counts"]
        assert counts["prefix"] == counts["mixed_launches"] == counts["raster_prepasses"] == frames - 1
        assert counts["outputs"] == frames
        current = (meta["qualification_profile_sha256"], data["driver_sha256"],
                   data["library_sha256"], meta["environment"]["gpu"]["name"], frames)
        if identity is None:
            identity = current
        assert current == identity, directory
        samples = [row for trial in data["traces"] for row in trial["samples"]]
        assert len(samples) == (0 if data["mode"] == "off" else (frames - 1) * meta["trial_count"]), path
        if path_index // 4 in rejected_blocks:
            continue
        groups.setdefault(data["mode"], []).append({"source": path.name,
            "fps": data["throughput_fps"], "samples": samples})
    baseline = statistics.median(row["fps"] for row in groups["off"])
    result = {}
    for mode, runs in groups.items():
        fps = [row["fps"] for row in runs]
        samples = [row for run in runs for row in run["samples"]]
        delta = 100 * (statistics.median(fps) / baseline - 1)
        result[mode] = {"process_count": len(runs), "median_fps": statistics.median(fps),
            "relative_to_off_fps_percent": delta,
            "within_predeclared_2_percent_perturbation_bound": abs(delta) <= 2,
            "runs": [{key: value for key, value in row.items() if key != "samples"} for row in runs],
            "sampling": sampling_summary(samples) if samples else None,
            "first_nine_pipeline_steps": sampling_summary([row for row in samples if row["pipeline_step"] < 9]) if samples else None}
    return {"kind": "original_v12_gpu_concurrent_kernel_only_overlap",
        "actual_variant": "full_mlp_overlap_graph_pb0", "qualification_sha256": identity[0],
        "qualification_profile": meta["qualification_profile"],
        "gpu_name": identity[3], "gpu_clock_hz_required_and_checked": 1300500000,
        "driver_sha256": identity[1], "library_sha256": identity[2], "frames_per_trial": identity[4],
        "sudo_both_on_and_off": True, "admitted_blocks_python_process_audit_passed": True,
        "admitted_ABBA_blocks": [i for i in range(len(all_paths) // 4) if i not in rejected_blocks],
        "rejected_ABBA_blocks": rejected_blocks,
        "exclusion_policy": "whole predefined four-process ABBA block rejected solely on continuous process-audit evidence, never by measured FPS",
        "continuous_host_audit_files": len(audits), "modes": result,
        "scope": "Concurrent-kernel activity only; no CPU/runtime/NVTX tracing or inserted Graph event nodes. Kernel intervals can include in-kernel stalls, so geometric overlap is NOT effective both/standalone-sum speedup. FPS calibration is required; it does not prove zero internal scheduling perturbation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.results)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
