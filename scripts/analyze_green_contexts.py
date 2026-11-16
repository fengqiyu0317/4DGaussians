"""Compare sealed Green Context ABBA cohorts without using trace FPS as formal FPS."""
import argparse
import json
from pathlib import Path
import shlex
import statistics

from analyze_sampling_kernel_activity import foreign_python_processes


def resource_partition(resources):
    """Validate the actual partition rather than assuming the former 4+12."""
    deform, raster = resources["deform_sms"], resources["raster_sms"]
    prepass = resources.get("raster_prepass_sms", raster)
    assert deform in (4, 8, 12) and raster == 16 - deform
    assert resources.get("raster_prepare_sms", raster) == raster
    assert resources["scope"] == ("deform_prefix_raster_prepare" if prepass == 16 else "prefix_only")
    assert prepass in (16, raster) and resources["tail_sms"] == 16
    return {"deform_prefix_sms": deform, "raster_prepare_sms": raster,
            "raster_prepass_sms": prepass, "tail_sms": 16}


def cohort_partition(rows, expected_deform_sms=None):
    partitions = [row["partition"] for row in rows if row["green"]]
    assert partitions and all(partition == partitions[0] for partition in partitions)
    if expected_deform_sms is not None:
        assert partitions[0]["deform_prefix_sms"] == expected_deform_sms, "Wrong requested SM split"
    return partitions[0]


def audit_process_ownership(audit, source_tree, stem):
    """Recognize this exact run, including its observed argv-less exit row.

    Do not waive unknown bracketed processes or another run of the same driver.
    A terminal row is ours only after its PID/user/comm was seen with this run's
    exact driver and metadata arguments, and after the last such live row.
    """
    driver = source_tree + "/scripts/profile_hexplane_reduction_activity.py"
    metadata = source_tree + "/" + stem + ".json"
    rows, live = [], []
    for index, line in enumerate(audit.splitlines()):
        parts = line.split(None, 3)
        if len(parts) != 4 or not parts[0].isdigit():
            continue
        key, command = tuple(parts[:3]), parts[3]
        try:
            args = shlex.split(command)
        except ValueError:
            args = []
        is_own = (len(args) > 1 and args[1] == driver
                  and any(args[i] == "--metadata" and args[i + 1] == metadata
                          for i in range(len(args) - 1)))
        row = (index, line.strip(), key, command, is_own)
        rows.append(row)
        if is_own:
            live.append(row)
    assert live, "This run's profiler PID was never observed: " + stem
    identities = {row[2] for row in live}
    assert len(identities) == 1, "Multiple profiler identities for one run: " + stem
    own_identity = next(iter(identities))
    last_live = live[-1][0]
    allowed, terminal = set(), set()
    for index, line, key, command, is_own in rows:
        if is_own:
            allowed.add(line)
        elif key == own_identity and index > last_live and command == "[" + key[2] + "]":
            allowed.add(line)
            terminal.add(line)
    return {"foreign_processes": [line for line in foreign_python_processes(audit) if line not in allowed],
            "own_process_identity_pid_user_comm": list(own_identity),
            "proven_own_terminal_rows": sorted(terminal)}


def load_run(directory, index, green, collector, stem_override=None):
    stem = stem_override or "{:02d}-green{}-{}".format(index, int(green), collector)
    activity = json.loads((directory / (stem + ".activity.json")).read_text())
    meta = json.loads((directory / (stem + ".json")).read_text())
    checks = (directory / (stem + ".source_check.txt")).read_text().splitlines()
    assert checks and all(line.endswith(": OK") for line in checks), stem
    assert meta["passed"] and meta["actual_execution_mode"] == "tacker", stem
    assert meta["qualification_mode_executed"] and meta["tacker_fallback_reason"] is None, stem
    assert meta["selected_variant_id"] == "full_mlp_overlap_graph_pb0", stem
    assert meta["post_run"]["byte_stability"]["verified"], stem
    assert meta["profile_frames"] == 50 and meta["warmup_frames"] == 20, stem
    assert (meta["gaussian_count"], meta["image_width"], meta["image_height"]) == (111525, 1352, 1014), stem
    assert meta["trial_count"] == (4 if directory.name.endswith("formal") else 2), stem
    counts = meta["pipeline_execution_counts"]
    assert counts["prefix"] == counts["raster_prepasses"] == counts["mixed_launches"] == counts["suffix"] == 49, stem
    assert counts["full_deformation"] == counts["solo_raster"] == 1 and counts["outputs"] == 50, stem
    assert "raster_hexplane_fused_launches" not in counts, stem
    assert activity["mode"] == collector and activity["config"]["green_contexts"] == green, stem
    assert activity["config"]["sampling"] == "original" and activity["config"]["reduction"] and activity["config"]["threads"] == 256, stem
    assert meta["green_contexts"]["enabled"] == green, stem
    partition = None
    if green:
        resources = meta["green_contexts"]
        partition = resource_partition(resources)
        assert activity["config"]["green_deform_sms"] == partition["deform_prefix_sms"], stem
        assert activity["config"].get("green_prepass_sms", partition["raster_prepass_sms"]) == partition["raster_prepass_sms"], stem
        assert resources["stream_resource_binding_verified"], stem
        assert resources["library_sha256"] == meta["source_files"]["green_contexts.binary"], stem
    identity = (meta["qualification_profile_sha256"], activity["driver_sha256"],
                activity["library_sha256"], meta["source_files"]["gaussian_renderer/tacker_pipeline.py"],
                meta["source_files"]["diff_gaussian_rasterization._C"], meta["view_indices"])
    audit = (directory / (stem + ".during_processes.txt")).read_text()
    ownership = audit_process_ownership(audit, meta["source_tree"], directory.name + "/" + stem)
    return {"source": stem, "green": green, "collector": collector,
            "fps": meta["throughput_fps"], "activity": activity,
            "identity": identity, "foreign_processes": ownership["foreign_processes"],
            "process_ownership": ownership,
            "green_resources": meta["green_contexts"],
            "partition": partition,
            "peak_allocated_bytes": meta["cuda_peak_allocated_bytes"],
            "peak_reserved_bytes": meta["cuda_peak_reserved_bytes"],
            "qualification_profile": meta["qualification_profile"]}


def validate_completed(directory):
    assert (directory / "completed.txt").exists(), "Incomplete batch: " + str(directory)
    checks = (directory / "final.source_check.txt").read_text().splitlines()
    assert checks and all(line.endswith(": OK") for line in checks), directory


def trace_path(root, directory_name="green-trace"):
    """Select a fresh resumed cohort without overwriting interrupted traces."""
    if not directory_name or directory_name in (".", "..") or Path(directory_name).name != directory_name:
        raise ValueError("Trace directory must be one child directory name")
    return root / directory_name


def summarize(root, expected_deform_sms=None, trace_directory="green-trace"):
    trace_dir = trace_path(root, trace_directory)
    formal_dir = root / "green-formal"
    validate_completed(formal_dir)
    formal = [load_run(formal_dir, i, bool(green), "off")
              for i, green in enumerate([0, 1, 1, 0, 0, 1, 1, 0])]
    identity = formal[0]["identity"]
    partition = cohort_partition(formal, expected_deform_sms)
    prepass_sms = partition["raster_prepass_sms"]
    green_name = "green{}_{}".format(partition["deform_prefix_sms"], partition["raster_prepare_sms"])
    assert all(row["identity"] == identity and not row["activity"]["traces"] for row in formal)
    rejected, accepted = {}, []
    for offset in (0, 4):
        block = formal[offset:offset + 4]
        bad = {row["source"]: row["foreign_processes"] for row in block if row["foreign_processes"]}
        if bad:
            rejected[str(offset // 4)] = bad
        else:
            accepted.extend(block)
    assert accepted, "All formal ABBA blocks rejected on foreign-process evidence"
    result = {"kind": "jetson_green_contexts_prepare_only" if prepass_sms == 16 else "jetson_green_contexts_prefix_only", "default_enabled": False,
        "actual_variant": "full_mlp_overlap_graph_pb0", "qualification_profile": formal[0]["qualification_profile"],
        "qualification_sha256": identity[0], "pipeline_sha256": identity[3],
        "workload": {"scene": "flame_steak", "iteration": 14000, "gaussians": 111525, "resolution": [1352, 1014]},
        "gpu_clock_hz_required_and_checked": 1300500000,
        "partition": partition,
        "hexplane_fused_reduction_threads_both_arms": 256,
        "hexplane_raster_mixed_disabled_both_arms": True,
        "formal_policy": "Two predeclared ABBA blocks; 4 trials of 50 frames/process, 20 warmup; no CUPTI or added Graph event timing; reject whole blocks on foreign-process audits, never select by FPS.",
        "audit_policy": "Only the exact driver plus this run's metadata argument identifies our profiler. An argv-less terminal row is attributed to it only for the same previously observed PID/user/comm after its last live row; unknown or other-run processes remain foreign.",
        "formal_process_audits": {row["source"]: row["process_ownership"] for row in formal},
        "rejected_formal_blocks": rejected, "formal": {}, "timelines": {}}
    for green, name in ((False, "baseline"), (True, green_name)):
        rows = [row for row in accepted if row["green"] == green]
        fps = statistics.median(row["fps"] for row in rows)
        result["formal"][name] = {"process_count": len(rows), "median_fps": fps,
            "ms_per_frame_from_fps": 1000 / fps, "min_process_fps": min(row["fps"] for row in rows),
            "max_process_fps": max(row["fps"] for row in rows),
            "median_peak_allocated_bytes": statistics.median(row["peak_allocated_bytes"] for row in rows),
            "median_peak_reserved_bytes": statistics.median(row["peak_reserved_bytes"] for row in rows),
            "runs": [{"source": row["source"], "fps": row["fps"]} for row in rows]}
    result["fps_change_percent"] = 100 * (result["formal"][green_name]["median_fps"] / result["formal"]["baseline"]["median_fps"] - 1)
    quality_dir = root / "green-quality"
    validate_completed(quality_dir)
    result["quality"] = {}
    for index, green in enumerate([False, True]):
        path = quality_dir / ("{:02d}-green{}-off.json".format(index, int(green)))
        report = json.loads(path.read_text())
        assert report["passed"] and all(gate["passed"] for gate in report["gates"])
        assert report["modes"]["tacker"]["green_contexts"]["enabled"] == green
        if green:
            assert resource_partition(report["modes"]["tacker"]["green_contexts"]) == partition
        result["quality"][green_name if green else "baseline"] = report["deltas"]

    result["trace_directory"] = trace_dir.name
    if trace_dir.exists():
        validate_completed(trace_dir)
        traces = [load_run(trace_dir, i, bool(green), collector)
                  for i, (green, collector) in enumerate([(0, "off"), (0, "on"), (0, "on"), (0, "off"),
                                                         (1, "off"), (1, "on"), (1, "on"), (1, "off")])]
        assert all(row["identity"] == identity for row in traces)
        assert cohort_partition(traces, expected_deform_sms) == partition
        result["trace_process_audits"] = {row["source"]: row["process_ownership"] for row in traces}
        for offset, name in ((0, "baseline"), (4, green_name)):
            block = traces[offset:offset + 4]
            bad = {row["source"]: row["foreign_processes"] for row in block if row["foreign_processes"]}
            if bad:
                result["timelines"][name] = {"rejected_whole_block": bad}
                continue
            off = statistics.median(row["fps"] for row in block if row["collector"] == "off")
            on = statistics.median(row["fps"] for row in block if row["collector"] == "on")
            delta = 100 * (on / off - 1)
            samples = [sample for row in block if row["collector"] == "on"
                       for trial in row["activity"]["traces"] for sample in trial["samples"]]
            assert len(samples) == 196
            fields = [key for key in samples[0] if key.endswith("_ms") or key == "merge_kernel_count"]
            result["timelines"][name] = {"sample_count": len(samples), "collector_fps_delta_percent": delta,
                "within_predeclared_2_percent_perturbation_bound": abs(delta) <= 2,
                "mean": {key: statistics.mean(sample[key] for sample in samples) for key in fields},
                "median": {key: statistics.median(sample[key] for sample in samples) for key in fields},
                "fraction_sampling_intervals_overlapping_prepare_percent": 100 * sum(
                    sample["sampling_prepare_kernel_overlap_ms"] for sample in samples) / sum(
                    sample["sampling_kernel_sum_ms"] for sample in samples),
                "stream_identities": samples[0]["stream_identity_context_and_stream"]}
    cpu_checks = root / "local_cpu_checks.json"
    if cpu_checks.exists():
        result["local_cpu_checks"] = json.loads(cpu_checks.read_text())
    result["conclusion"] = {
        "promote_to_default": False,
        "resource_partition_verified": True,
        "prepass_full_device": prepass_sms == 16,
        "interpretation": "Resource partition and prepass stream routing are verified independently of measured FPS. Increased timestamp overlap alone is not throughput improvement; formal FPS and separately calibrated stage timings must be compared.",
        "unmeasured_causes": "No new NCU bandwidth/cache study was performed; the measurements do not isolate bandwidth contention, nor prove that Green Contexts cannot help with a different split or scope.",
        "qualification_only": True
    }
    result["lifecycle"] = "Graph executables are reset after work completes; bounded per-configuration Green stream pairs stay alive until process exit for PyTorch allocator record_stream references."
    result["scope"] = "Qualification-mode experiment, not production admission. Timestamp overlap includes in-kernel stalls; it is not effective concurrency or an isolated standalone speedup. Trace FPS is used only for collector calibration, never substituted for formal FPS."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deform-sms", type=int, choices=(4, 8, 12), help="Require this actual Deform SM count")
    parser.add_argument("--trace-directory", default="green-trace", help="Completed trace cohort child directory; preserves interrupted cohorts")
    args = parser.parse_args()
    report = summarize(args.root, args.deform_sms, args.trace_directory)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
