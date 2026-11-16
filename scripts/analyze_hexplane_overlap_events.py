"""Validate and summarize event-only Jetson Graph experiments (stdlib)."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_hexplane_overlap_events import summarize_samples


def summarize(root):
    phases = {}
    qualification_identity = None
    for phase in ("pilot", "confirm", "coverage"):
        directory = root / ("event-probe-" + phase)
        if not directory.exists():
            continue
        checks = list(directory.glob("*.source_check.txt"))
        assert directory / "final.source_check.txt" in checks, directory
        assert all(all(line.endswith(": OK") for line in path.read_text().splitlines())
                   for path in checks), directory
        groups = {}
        for path in sorted(directory.glob("*.events.json")):
            match = re.fullmatch(r"(\d+)-(off|last|all)-f(\d+)\.events\.json", path.name)
            assert match, path
            mode, frames = match.group(2), int(match.group(3))
            event = json.loads(path.read_text())
            meta_path = path.with_name(path.name.replace(".events.json", ".json"))
            meta = json.loads(meta_path.read_text())
            assert event["mode"] == mode and meta["profile_frames"] == frames, path
            assert meta["passed"] and meta["actual_execution_mode"] == "tacker", path
            assert meta["qualification_mode_executed"] and meta["tacker_fallback_reason"] is None, path
            assert meta["selected_variant_id"] == "full_mlp_overlap_graph_pb0", path
            assert meta["post_run"]["byte_stability"]["verified"], path
            counts = meta["pipeline_execution_counts"]
            assert counts["prefix"] == counts["mixed_launches"] == counts["raster_prepasses"] == frames - 1, path
            assert counts["outputs"] == frames, path
            current = (meta["qualification_profile"], meta["qualification_profile_sha256"],
                       meta["environment"]["gpu"]["name"], meta["environment"]["cuda_visible_devices"],
                       event["probe_sha256"])
            if qualification_identity is None:
                qualification_identity = current
            assert current == qualification_identity, "different code/qualification/target"
            samples = [sample for trial in event["trials"] for sample in trial["samples"]]
            assert len(samples) == (0 if mode == "off" else 2 * meta["trial_count"]), path
            assert all(sample["plane_ids"] == (list(range(12)) if mode == "all" else [11])
                       for sample in samples), path
            assert all(len(sample["sampling_intervals_ms"]) == (12 if mode == "all" else 1)
                       for sample in samples), path
            key = "f" + str(frames)
            groups.setdefault(key, {}).setdefault(mode, []).append({
                "source": str(path.relative_to(root)), "fps": event["throughput_fps"],
                "samples": samples})
        result = {}
        for frame_key, modes in groups.items():
            baseline = statistics.median(row["fps"] for row in modes["off"])
            result[frame_key] = {}
            for mode, runs in modes.items():
                fps = [run["fps"] for run in runs]
                samples = [sample for run in runs for sample in run["samples"]]
                delta = 100 * (statistics.median(fps) / baseline - 1)
                result[frame_key][mode] = {
                    "process_count": len(runs), "median_fps": statistics.median(fps),
                    "min_fps": min(fps), "max_fps": max(fps),
                    "relative_to_off_fps_percent": delta,
                    "within_predeclared_2_percent_fps_perturbation_bound": abs(delta) <= 2,
                    "runs": [{k: value for k, value in run.items() if k != "samples"} for run in runs],
                    "sampling": summarize_samples(samples) if samples else None,
                }
        phases[phase] = result
    assert qualification_identity is not None, root
    samples = []
    for phase in ("confirm", "coverage"):
        for path in (root / ("event-probe-" + phase)).glob("*-all-*.events.json"):
            report = json.loads(path.read_text())
            samples.extend(row for trial in report["trials"] for row in trial["samples"])
    cuda_log = root / "event-probe-cuda/cuda.log"
    assert "Ran 1 test" in cuda_log.read_text() and "\nOK\n" in cuda_log.read_text(), cuda_log
    return {
        "kind": "original_v12_sampling_event_only_overlap", "schema_version": 1,
        "qualification_profile": qualification_identity[0],
        "qualification_profile_sha256": qualification_identity[1],
        "actual_variant": "full_mlp_overlap_graph_pb0", "gpu_name": qualification_identity[2],
        "probe_sha256": qualification_identity[4],
        "gpu_clock_hz_required_and_checked": 1300500000,
        "cuda_graph_external_timing_and_bitwise_output_test_passed": True,
        "phases": phases,
        "all_plane_samples_confirm_and_coverage": summarize_samples(samples) if samples else None,
        "scope": {
            "no_nsight": True, "no_per_frame_sync_or_io_added": True,
            "production_defaults_changed": False,
            "sampled_replays": "last two slot replays per complete trial; varied sequence lengths check earlier/middle frame positions",
            "sampling": "sum of twelve event brackets immediately around original ATen grid_sample calls, not first-to-last span",
            "prepare_overlap": "intersection with Raster prepare stage span (includes original CPU/D2H gaps); an upper bound on actual prepare-kernel intersection",
            "prepass_overlap": "intersection with event bracket of single unchanged prepass kernel",
            "fps_gate": "2% bound is a perturbation check, not proof that internal scheduling is identical to zero-instrumentation execution",
            "not_measured": "isolated standalone times or effective both/standalone-sum speedup; these events measure geometric interval overlap only",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.results)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
