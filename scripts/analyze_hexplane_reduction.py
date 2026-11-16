"""Sealed mirrored/ABBA experiment summary; formal FPS never uses trace FPS."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from analyze_sampling_kernel_activity import foreign_python_processes


CONFIGS = ("original-r0", "original-r128", "original-r256", "mixed-r0", "mixed-r128", "mixed-r256")


def config_key(config):
    return config["sampling"] + "-r" + str(config["threads"] if config["reduction"] else 0)


def checked_runs(directory, block_size, *, allow_incomplete=False):
    completed = (directory / "completed.txt").exists()
    assert completed or allow_incomplete, "incomplete batch: " + str(directory)
    checks = list(directory.glob("*.source_check.txt"))
    assert checks and (allow_incomplete or directory / "final.source_check.txt" in checks)
    for path in checks:
        lines = path.read_text().splitlines()
        assert lines and all(line.endswith(": OK") for line in lines), path
    paths = sorted(directory.glob("*.activity.json"))
    assert paths and (allow_incomplete or len(paths) % block_size == 0)
    blocks, rejected = [], {}
    complete_count = len(paths) - len(paths) % block_size
    if complete_count != len(paths):
        rejected[complete_count // block_size] = {
            "reason": "incomplete trailing ABBA block; never reuse individual runs",
            "runs": [p.name for p in paths[complete_count:]],
        }
    identity = None
    for start in range(0, complete_count, block_size):
        block, reasons = [], {}
        for path in paths[start:start + block_size]:
            data = json.loads(path.read_text())
            source_check = path.with_name(path.name.replace(".activity.json", ".source_check.txt"))
            assert source_check in checks, path
            meta = json.loads(path.with_name(path.name.replace(".activity.json", ".json")).read_text())
            assert meta["passed"] and meta["actual_execution_mode"] == "tacker"
            assert meta["qualification_mode_executed"] and meta["tacker_fallback_reason"] is None
            assert meta["selected_variant_id"] == "full_mlp_overlap_graph_pb0"
            assert meta["post_run"]["byte_stability"]["verified"]
            counts = meta["pipeline_execution_counts"]
            frames = meta["profile_frames"]
            assert frames == 50 and meta["warmup_frames"] == 20
            assert (meta["gaussian_count"], meta["image_width"], meta["image_height"]) == (111525, 1352, 1014)
            assert counts["outputs"] == frames and counts["mixed_launches"] == counts["raster_prepasses"] == frames - 1
            assert data["config"]["mixed_planes"] == 12 and data["config"]["tasks_per_cta"] == 3
            if data["config"]["sampling"] == "mixed":
                assert counts["raster_hexplane_fused_launches"] == frames - 1
            current = (meta["qualification_profile_sha256"], data["driver_sha256"],
                       data["library_sha256"], meta["environment"]["gpu"]["name"])
            if identity is None:
                identity = current
            assert current == identity, path
            audit = path.with_name(path.name.replace(".activity.json", ".during_processes.txt"))
            assert audit.exists(), path
            bad = [line for line in foreign_python_processes(audit.read_text())
                   if "profile_hexplane_reduction_activity.py" not in line]
            if bad:
                reasons[audit.name] = bad
            block.append({"path": path.name, "batch": directory.name,
                          "key": config_key(data["config"]), "data": data,
                          "peak_allocated_bytes": meta["cuda_peak_allocated_bytes"],
                          "peak_reserved_bytes": meta["cuda_peak_reserved_bytes"],
                          "qualification_profile": meta["qualification_profile"]})
        if block_size == 12:
            assert [r["key"] for r in block] == list(CONFIGS) + list(reversed(CONFIGS))
            assert all(r["data"]["mode"] == "off" for r in block)
        else:
            assert len({r["key"] for r in block}) == 1
            assert [r["data"]["mode"] for r in block] == ["off", "on", "on", "off"]
        if reasons:
            rejected[start // block_size] = reasons
        else:
            blocks.extend(block)
    assert blocks or allow_incomplete, "all blocks rejected on process-audit evidence"
    return blocks, rejected, identity


def select_trace_blocks(batches, identity):
    """Select first clean COMPLETE ABBA, never select by measured performance.

    A batch interrupted by an idle check is not called completed. Earlier
    complete blocks may be used only with every run's provenance stability
    and process audit verified. Incomplete trailing blocks are always omitted.
    """
    selected, rejected, duplicate, incomplete = {}, {}, [], []
    for directory in batches:
        done = (directory / "completed.txt").exists()
        runs, bad, current = checked_runs(directory, 4, allow_incomplete=not done)
        if current is not None:
            assert current == identity, directory
        if bad:
            rejected[directory.name] = bad
        if not done:
            incomplete.append(directory.name)
        for offset in range(0, len(runs), 4):
            block = runs[offset:offset + 4]
            key = block[0]["key"]
            assert len(block) == 4 and all(r["key"] == key for r in block)
            if key in selected:
                duplicate.append({"batch": directory.name, "config": key,
                                  "reason": "an earlier clean complete ABBA was already selected"})
            else:
                selected[key] = block
    assert set(selected) == set(CONFIGS), "missing clean complete ABBA: " + str(set(CONFIGS) - set(selected))
    return [r for key in CONFIGS for r in selected[key]], rejected, duplicate, incomplete


def summarize(root):
    formal, rejected_formal, identity = checked_runs(root / "formal", 12)
    result = {"kind": "jetson_hexplane_fused_reduction", "actual_variant": "full_mlp_overlap_graph_pb0",
        "qualification_profile": formal[0]["qualification_profile"], "qualification_sha256": identity[0],
        "gpu_name": identity[3], "gpu_clock_hz": 1300500000,
        "workload": {"name": "flame_steak", "gaussians": 111525, "width": 1352, "height": 1014},
        "default_enabled": False, "sampling_ratio_changed": False, "mixed_planes": 12, "tasks_per_cta": 3,
        "formal_policy": "50 frames/trial, 20 warmup, four trials/process; two mirrored six-configuration blocks; formal FPS contains no CUPTI or per-frame synchronization",
        "rejected_formal_blocks": rejected_formal, "formal": {}, "timelines": {}}
    for key in CONFIGS:
        runs = [r for r in formal if r["key"] == key]
        assert runs, key
        fps = statistics.median(r["data"]["throughput_fps"] for r in runs)
        result["formal"][key] = {"process_count": len(runs), "median_fps": fps,
            "ms_per_frame_from_fps": 1000 / fps,
            "median_peak_allocated_bytes": statistics.median(r["peak_allocated_bytes"] for r in runs),
            "median_peak_reserved_bytes": statistics.median(r["peak_reserved_bytes"] for r in runs),
            "runs": [{"source": r["path"], "fps": r["data"]["throughput_fps"]} for r in runs]}
    baseline = result["formal"]["original-r0"]["median_fps"]
    for key, metrics in result["formal"].items():
        control = result["formal"][key.split("-r")[0] + "-r0"]["median_fps"]
        metrics["relative_to_original_percent"] = 100 * (metrics["median_fps"] / baseline - 1)
        metrics["reduction_gain_with_same_sampling_percent"] = 100 * (metrics["median_fps"] / control - 1)
    # Preserved directory mtimes order the actual sequential batches. Report
    # the selected source batches explicitly; no FPS-dependent selection.
    trace_dirs = sorted((p for p in root.glob("trace*")
                         if p.is_dir() and any(p.glob("*.activity.json"))),
                        key=lambda p: (p.stat().st_mtime_ns, p.name))
    if trace_dirs:
        traces, rejected, duplicates, incomplete = select_trace_blocks(trace_dirs, identity)
        result["rejected_trace_blocks"] = rejected
        result["trace_batch_order"] = [p.name for p in trace_dirs]
        result["incomplete_trace_batches"] = incomplete
        result["unused_duplicate_trace_blocks"] = duplicates
        result["trace_selection_policy"] = "Earliest clean complete per-config ABBA in chronological batch order; reject entire blocks on process audit; never select on FPS. Interrupted batches are explicitly retained as incomplete."
        for key in CONFIGS:
            runs = [r for r in traces if r["key"] == key]
            assert len(runs) == 4, key
            off = statistics.median(r["data"]["throughput_fps"] for r in runs if r["data"]["mode"] == "off")
            on = statistics.median(r["data"]["throughput_fps"] for r in runs if r["data"]["mode"] == "on")
            delta = 100 * (on / off - 1)
            samples = [s for r in runs if r["data"]["mode"] == "on" for trial in r["data"]["traces"] for s in trial["samples"]]
            assert len(samples) == 196
            keys = [k for k in samples[0] if k.endswith("_ms") or k == "merge_kernel_count"]
            result["timelines"][key] = {"sample_count": len(samples), "off_fps": off, "on_fps": on,
                "source_batch": runs[0]["batch"],
                "source_runs": [r["path"] for r in runs],
                "collector_fps_delta_percent": delta, "within_predeclared_2_percent_perturbation_bound": abs(delta) <= 2,
                "mean": {k: statistics.mean(s[k] for s in samples) for k in keys},
                "median": {k: statistics.median(s[k] for s in samples) for k in keys}}
            if runs[0]["data"]["config"]["reduction"]:
                launches = []
                for run in runs:
                    for trial in run["data"]["traces"]:
                        path = root / run["batch"] / Path(trial["path"]).name
                        with path.open(newline="") as handle:
                            launches.extend(r for r in csv.DictReader(handle)
                                            if "hexplane_reduce_features_tiled_v1" in r["name"])
                threads = runs[0]["data"]["config"]["threads"]
                assert len(launches) == len(samples)
                assert all(int(r["block_x"]) == threads and int(r["grid_x"]) == 3486
                           and int(r["graph_node_id"]) > 0 for r in launches)
                result["timelines"][key]["observed_reduction_launch"] = {
                    "count": len(launches), "threads": threads, "grid_x": 3486,
                    "registers_per_thread": sorted({int(r["registers"]) for r in launches}),
                    "all_graph_nodes": True}
    result["validation"] = {}
    for phase in ("reduction-cuda-final", "memcheck", "quality"):
        directory = root / phase
        assert (directory / "completed.txt").exists(), phase
        result["validation"][phase] = {"completed": True}
        if phase != "quality":
            log = (directory / "reduction.log").read_text()
            assert "Ran 8 tests" in log and "\nOK\n" in log
            if phase == "memcheck":
                assert "ERROR SUMMARY: 0 errors" in log
        else:
            for path in sorted(directory.glob("[0-9][0-9]-*.json")):
                quality = json.loads(path.read_text())
                assert quality["passed"] and all(g["passed"] for g in quality["gates"])
                result["validation"][phase][path.name] = quality["deltas"]
    result["scope"] = "Experimental Graph scheduling, not production admission. Mixed sampling leaf remains a native launch; fused reduction is a captured Graph node. Kernel timestamp intervals include stalls; no claim that interval overlap equals speedup."
    cpu = root / "local_cpu_checks.json"
    if cpu.exists():
        result["validation"]["local_cpu"] = json.loads(cpu.read_text())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered)
