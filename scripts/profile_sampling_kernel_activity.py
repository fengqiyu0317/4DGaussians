"""Sealed original-v12 benchmark with ONLY concurrent GPU kernel timestamps.

This is a separate calibration experiment, not a zero-profiler claim.  It adds
no CUDA events or CPU/API tracing; FPS controls determine its admissibility.
"""
import argparse
import csv
import ctypes
import hashlib
import inspect
import json
from pathlib import Path
import statistics

from run_profile_render_sealed import _read_stable_source


def overlaps(aa, bb):
    return sum(max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
               for a in aa for b in bb) / 1e6


def analyze(path, frame_count):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({"name": row["name"], "start": int(row["start_ns"]),
                         "end": int(row["end_ns"]), "stream": int(row["stream_id"]),
                         "node": int(row["graph_node_id"])})
    rows.sort(key=lambda row: row["start"])
    h = [row for row in rows if "grid_sampler_2d_kernel" in row["name"]]
    starts = [row for row in rows if "preprocessCUDA" in row["name"]]
    ends = [row for row in rows if "identifyTileRanges" in row["name"]]
    prepasses = [row for row in rows if "tacker_render_raster_fast_range_v12" in row["name"]]
    assert (len(h), len(starts), len(ends), len(prepasses)) == (
        12 * frame_count, frame_count, frame_count, frame_count - 1), path
    assert all(row["node"] == 0 for row in h[:12]), path
    h = h[12:]
    assert all(row["node"] > 0 for row in h), path
    result = []
    for step in range(frame_count - 1):
        sampling = h[12 * step:12 * (step + 1)]
        start, end, prepass = starts[step], ends[step], prepasses[step]
        prepare = [row for row in rows if row["stream"] == start["stream"]
                   and row["start"] >= start["start"] and row["end"] <= end["end"]]
        duration = sum(row["end"] - row["start"] for row in sampling) / 1e6
        result.append({
            "pipeline_step": step, "sampling_ms": duration,
            "prepare_kernel_sum_ms": sum(row["end"] - row["start"] for row in prepare) / 1e6,
            "prepass_ms": (prepass["end"] - prepass["start"]) / 1e6,
            "sampling_prepare_kernel_overlap_ms": overlaps(sampling, prepare),
            "sampling_prepass_kernel_overlap_ms": overlaps(sampling, [prepass]),
            "sampling_end_relative_to_prepass_start_ms": (sampling[-1]["end"] - prepass["start"]) / 1e6,
            "sampling_intervals_ns": [[row["start"], row["end"]] for row in sampling],
            "prepass_interval_ns": [prepass["start"], prepass["end"]],
        })
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kernel-mode", choices=("off", "on"), required=True)
    parser.add_argument("--kernel-library", type=Path, required=True)
    parser.add_argument("--kernel-output", type=Path, required=True)
    options, args = parser.parse_known_args()
    import os
    assert os.environ.get("TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12") == "1"
    assert os.environ.get("TACKER_EXPERIMENTAL_RASTER_HEXPLANE", "0") == "0"
    assert os.environ.get("TACKER_HEXPLANE_LOWREG", "0") == "0"
    assert all(not os.environ.get(name) for name in (
        "TACKER_PROFILE_MIXED_PIPELINE_JSONL", "TACKER_PROFILE_MIXED_V12_SPLIT_JSONL",
        "TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE"))
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    binary_hash = hashlib.sha256(options.kernel_library.read_bytes()).hexdigest()
    record, raw = _read_stable_source(str(Path(__file__).resolve().parents[1] / "profile_render.py"))
    namespace = {"__name__": "profile_render_kernel_activity_diagnostic",
                 "__file__": record["path"], "__package__": None,
                 "_PROFILE_RENDER_BOOTSTRAP": {
                     "protocol": 1, "record": record, "bytes": raw,
                     "execution": {"compiled_sha256": record["sha256"],
                                   "compile_mode": "exec", "dont_inherit": True}}}
    exec(compile(raw, record["path"], "exec", dont_inherit=True), namespace)
    traces = []
    lib = None
    if options.kernel_mode == "on":
        lib = ctypes.CDLL(str(options.kernel_library.resolve()))
        lib.sampling_activity_write.argtypes = [ctypes.c_char_p]
        activate = namespace["_activate_runtime_imports"]
        measure = namespace["_measure_trial"]

        def checked(result):
            if result != 0:
                raise RuntimeError("kernel-only CUPTI collector failed: " + str(result))

        def activation(*aa, **kk):
            activate(*aa, **kk)
            checked(lib.sampling_activity_init())

        def measured(*aa, **kk):
            bound = inspect.signature(measure).bind(*aa, **kk).arguments
            checked(lib.sampling_activity_clear())
            result = measure(*aa, **kk)
            path = options.kernel_output.with_name(options.kernel_output.stem +
                ".trial" + str(result["trial_index"]) + ".kernels.csv")
            checked(lib.sampling_activity_write(str(path).encode()))
            traces.append({"trial_index": result["trial_index"], "path": str(path),
                           "samples": analyze(path, len(bound["views"]))})
            return result

        namespace["_activate_runtime_imports"] = activation
        namespace["_measure_trial"] = measured
    namespace["_cli_main"](args)
    if lib is not None:
        assert lib.sampling_activity_stop() == 0
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == own_hash
    assert hashlib.sha256(options.kernel_library.read_bytes()).hexdigest() == binary_hash
    metadata_path = Path(args[args.index("--metadata") + 1])
    meta = json.loads(metadata_path.read_text())
    assert meta["passed"] and meta["actual_execution_mode"] == "tacker"
    assert meta["qualification_mode_executed"] and meta["tacker_fallback_reason"] is None
    assert meta["selected_variant_id"] == "full_mlp_overlap_graph_pb0"
    assert meta["post_run"]["byte_stability"]["verified"]
    samples = [row for trial in traces for row in trial["samples"]]
    total = sum(row["sampling_ms"] for row in samples)
    report = {
        "mode": options.kernel_mode, "driver_sha256": own_hash, "library_sha256": binary_hash,
        "throughput_fps": meta["throughput_fps"], "metadata": str(metadata_path),
        "qualification_profile_sha256": meta["qualification_profile_sha256"],
        "policy": "ONLY CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL; no API/CPU/NVTX trace, no added Graph events, no per-frame sync or I/O; records flushed after original timed trial ends",
        "traces": traces,
        "summary": None if not samples else {
            "sample_count": len(samples),
            "mean_sampling_ms": statistics.mean(row["sampling_ms"] for row in samples),
            "median_sampling_ms": statistics.median(row["sampling_ms"] for row in samples),
            "prepare_overlap_ms_mean": statistics.mean(row["sampling_prepare_kernel_overlap_ms"] for row in samples),
            "prepass_overlap_ms_mean": statistics.mean(row["sampling_prepass_kernel_overlap_ms"] for row in samples),
            "sampling_prepare_kernel_overlap_percent": 100 * sum(row["sampling_prepare_kernel_overlap_ms"] for row in samples) / total,
            "sampling_prepass_kernel_overlap_percent": 100 * sum(row["sampling_prepass_kernel_overlap_ms"] for row in samples) / total,
            "frames_with_prepass_kernel_overlap": sum(row["sampling_prepass_kernel_overlap_ms"] > 0 for row in samples),
            "sampling_complete_before_prepass_frames": sum(row["sampling_end_relative_to_prepass_start_ms"] <= 0 for row in samples),
            "median_sampling_end_relative_to_prepass_start_ms": statistics.median(row["sampling_end_relative_to_prepass_start_ms"] for row in samples),
        }}
    options.kernel_output.write_text(json.dumps(report, indent=2) + "\n")
    print("KERNEL_ONLY_PROBE", options.kernel_mode, json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
