"""Matched reduction controls, optionally collecting GPU kernel timestamps only."""
import argparse
import csv
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path

from run_profile_render_sealed import _read_stable_source


def analyze(path, frames, config):
    with path.open(newline="") as handle:
        rows = [{"name": r["name"], "s": int(r["start_ns"]), "e": int(r["end_ns"]),
                 "stream": (int(r.get("context_id", 0)), int(r["stream_id"])),
                 "node": int(r["graph_node_id"])}
                for r in csv.DictReader(handle)]
    rows.sort(key=lambda r: r["s"])
    starts = [r for r in rows if "preprocessCUDA" in r["name"]]
    ends = [r for r in rows if "identifyTileRanges" in r["name"]]
    tails = [r for r in rows if "tacker_mix_render_deform_mlp_tensor_tail_split_raster_v12" in r["name"]]
    prepasses = [r for r in rows if "tacker_render_raster_fast_range_v12" in r["name"]]
    mixed = [r for r in rows if "tacker_mix_raster_hexplane_512_v1" in r["name"]]
    reductions = [r for r in rows if "hexplane_reduce_features_tiled_v1" in r["name"]]
    samplers = [r for r in rows if "grid_sampler_2d_kernel" in r["name"]]
    assert (len(starts), len(ends), len(tails), len(prepasses)) == (frames, frames, frames - 1, frames - 1), path
    is_mixed = config["sampling"] == "mixed"
    handoff = config.get("green_prepare_handoff", False)
    assert len(mixed) == (frames - 1 if is_mixed else 0), path
    assert len(samplers) == (12 if is_mixed else frames * 12), path
    # Eager prefill and serial rendering intentionally keep the old reduction.
    assert len(reductions) == (frames - 1 if config["reduction"] else 0), path
    assert all(r["node"] > 0 for r in reductions), path
    # Green Contexts keep prefill on the ordinary stream, but capture/replay
    # steady-state sampling on a distinct partitioned stream.
    dstream = samplers[12 if len(samplers) > 12 else 0]["stream"]
    samples = []
    for i, tail in enumerate(tails):
        start, end, prepass = starts[i], ends[i], prepasses[i]
        sampling_rows = samplers[12 * (i + 1):12 * (i + 2)] if not is_mixed else []
        prepare_streams = (start["stream"], end["stream"]) if handoff else (start["stream"],)
        prepare_rows = [r for r in rows if r["stream"] in prepare_streams
                        and r["s"] >= start["s"] and r["e"] <= end["e"]]
        if config.get("green_contexts"):
            if handoff:
                assert end["stream"] == prepass["stream"] == tail["stream"], path
            if config.get("green_prepass_sms") == 16:
                assert start["stream"] != prepass["stream"] == tail["stream"], path
                assert end["e"] <= prepass["s"], path
            else:
                # Historical prefix-only traces kept prepass partitioned.
                assert start["stream"] == prepass["stream"] != tail["stream"], path
            assert dstream not in (start["stream"], prepass["stream"], tail["stream"], samplers[0]["stream"]), path
            assert all(r["node"] > 0 and r["stream"] == dstream for r in sampling_rows), path
        region_start = mixed[i]["s"] if is_mixed else prepass["s"]
        lower = mixed[i]["e"] if is_mixed else samplers[12 * (i + 1)]["s"]
        operations = [r for r in rows if r["stream"] == dstream and r["s"] >= lower
                      and r["e"] <= tail["s"] and r["node"] > 0]
        if config["reduction"]:
            reduction = reductions[i]
            merge_ops = [reduction]
        else:
            mul = [r for r in operations if "MulFunctor" in r["name"]]
            cats = [r for r in operations if "CatArrayBatchedCopy" in r["name"]]
            assert len(mul) == 12 and cats, (path, i, len(mul), len(cats))
            merge_ops = mul + cats[-1:]
            reduction = {"s": min(r["s"] for r in merge_ops), "e": max(r["e"] for r in merge_ops)}
        duplicates = [r for r in prepare_rows if "duplicateWithKeys" in r["name"]]
        sorts = [r for r in prepare_rows if "DeviceRadixSort" in r["name"]]
        assert len(duplicates) == 1 and sorts, path
        duplicate, first_sort = duplicates[0], min(sorts, key=lambda r: r["s"])
        assert duplicate["stream"] == start["stream"], path
        assert all(r["stream"] == end["stream"] for r in sorts), path
        if handoff:
            assert duplicate["e"] <= first_sort["s"] and reduction["e"] <= first_sort["s"], path
            assert all(r["e"] <= first_sort["s"] for r in operations), path
        samples.append({
            "pipeline_step": i,
            "prepare_span_ms": (end["e"] - start["s"]) / 1e6,
            "prepare_kernel_sum_ms": sum(r["e"] - r["s"] for r in prepare_rows) / 1e6,
            "prepare_through_duplicate_ms": (duplicate["e"] - start["s"]) / 1e6,
            "duplicate_kernel_ms": (duplicate["e"] - duplicate["s"]) / 1e6,
            "duplicate_to_sort_gap_ms": (first_sort["s"] - duplicate["e"]) / 1e6,
            "sort_kernel_sum_ms": sum(r["e"] - r["s"] for r in sorts) / 1e6,
            "sort_through_ranges_span_ms": (end["e"] - first_sort["s"]) / 1e6,
            "identify_ranges_kernel_ms": (end["e"] - end["s"]) / 1e6,
            "sampling_kernel_sum_ms": sum(r["e"] - r["s"] for r in sampling_rows) / 1e6,
            "sampling_span_ms": (sampling_rows[-1]["e"] - sampling_rows[0]["s"]) / 1e6 if sampling_rows else None,
            "sampling_prepare_kernel_overlap_ms": sum(max(0, min(s["e"], p["e"]) - max(s["s"], p["s"]))
                for s in sampling_rows for p in prepare_rows) / 1e6,
            "sampling_prepass_overlap_ms": sum(max(0, min(s["e"], prepass["e"]) - max(s["s"], prepass["s"])) for s in sampling_rows) / 1e6,
            "sampling_end_relative_to_prepass_start_ms": (sampling_rows[-1]["e"] - prepass["s"]) / 1e6 if sampling_rows else None,
            "stream_identity_context_and_stream": {"deform_prefix": dstream,
                "raster_prepare": start["stream"], "raster_prepare_suffix": end["stream"],
                "raster_prefix": prepass["stream"], "mixed_tail": tail["stream"]},
            "raster_prefix_region_ms": (prepass["e"] - region_start) / 1e6,
            "sampling_mixed_ms": (mixed[i]["e"] - mixed[i]["s"]) / 1e6 if is_mixed else 0,
            "remaining_prepass_ms": (prepass["e"] - prepass["s"]) / 1e6,
            "post_prepass_to_mlp_start_ms": (tail["s"] - prepass["e"]) / 1e6,
            "merge_kernel_sum_ms": sum(r["e"] - r["s"] for r in merge_ops) / 1e6,
            "merge_kernel_count": len(merge_ops),
            "merge_span_ms": (reduction["e"] - reduction["s"]) / 1e6,
            "merge_end_relative_to_prepass_end_ms": (reduction["e"] - prepass["e"]) / 1e6,
            "mlp_raster_tail_ms": (tail["e"] - tail["s"]) / 1e6,
            "gpu_prepare_start_to_tail_end_ms": (tail["e"] - start["s"]) / 1e6,
            "prepare_interval_ns": [start["s"], end["e"]],
            "raster_prefix_interval_ns": [region_start, prepass["e"]],
            "merge_intervals_ns": [[r["s"], r["e"]] for r in sorted(merge_ops, key=lambda r: r["s"])],
            "mlp_raster_tail_interval_ns": [tail["s"], tail["e"]],
        })
    return samples


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kernel-mode", choices=("off", "on"), required=True)
    parser.add_argument("--kernel-library", type=Path, required=True)
    parser.add_argument("--kernel-output", type=Path, required=True)
    options, args = parser.parse_known_args()
    sampling = os.environ.get("TACKER_EXPERIMENTAL_RASTER_HEXPLANE", "0")
    assert sampling in ("0", "fused")
    config = {"sampling": "mixed" if sampling == "fused" else "original",
              "reduction": os.environ.get("TACKER_HEXPLANE_FUSED_REDUCTION", "0") == "1",
              "threads": int(os.environ.get("TACKER_HEXPLANE_REDUCTION_THREADS", "256")),
              "mixed_planes": int(os.environ.get("TACKER_RASTER_HEXPLANE_MIXED_PLANES", "12")),
              "tasks_per_cta": int(os.environ.get("TACKER_RASTER_HEXPLANE_TASKS_PER_CTA", "3"))}
    config["green_contexts"] = os.environ.get("TACKER_EXPERIMENTAL_GREEN_CONTEXTS", "0") == "1"
    config["green_prepare_handoff"] = os.environ.get("TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF", "0") == "1"
    config["green_deform_sms"] = int(os.environ.get("TACKER_GREEN_DEFORM_SMS", "4")) if config["green_contexts"] else None
    config["green_prepass_sms"] = 16 if config["green_contexts"] else None
    assert config["mixed_planes"] == 12 and config["tasks_per_cta"] == 3
    assert os.environ.get("TACKER_HEXPLANE_LOWREG", "0") == "0"
    assert os.environ.get("TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12") == "1"
    assert not any(os.environ.get(k) for k in ("TACKER_PROFILE_MIXED_PIPELINE_JSONL",
        "TACKER_PROFILE_MIXED_V12_SPLIT_JSONL", "TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE"))
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    binary_hash = hashlib.sha256(options.kernel_library.read_bytes()).hexdigest()
    record, raw = _read_stable_source(str(Path(__file__).resolve().parents[1] / "profile_render.py"))
    namespace = {"__name__": "profile_render_reduction_activity", "__file__": record["path"],
        "__package__": None, "_PROFILE_RENDER_BOOTSTRAP": {"protocol": 1, "record": record,
        "bytes": raw, "execution": {"compiled_sha256": record["sha256"], "compile_mode": "exec", "dont_inherit": True}}}
    exec(compile(raw, record["path"], "exec", dont_inherit=True), namespace)
    traces = []
    lib = None
    if options.kernel_mode == "on":
        lib = ctypes.CDLL(str(options.kernel_library.resolve()))
        lib.sampling_activity_write.argtypes = [ctypes.c_char_p]
        activate, measure = namespace["_activate_runtime_imports"], namespace["_measure_trial"]

        def checked(result):
            if result:
                raise RuntimeError("concurrent-kernel collector failed: " + str(result))

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
                           "samples": analyze(path, len(bound["views"]), config)})
            return result

        namespace["_activate_runtime_imports"], namespace["_measure_trial"] = activation, measured
    namespace["_cli_main"](args)
    if lib is not None:
        checked(lib.sampling_activity_stop())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == own_hash
    assert hashlib.sha256(options.kernel_library.read_bytes()).hexdigest() == binary_hash
    meta = json.loads(Path(args[args.index("--metadata") + 1]).read_text())
    assert meta["passed"] and meta["actual_execution_mode"] == "tacker" and meta["tacker_fallback_reason"] is None
    assert meta["qualification_mode_executed"] and meta["selected_variant_id"] == "full_mlp_overlap_graph_pb0"
    assert meta["post_run"]["byte_stability"]["verified"]
    assert meta["green_contexts"]["enabled"] == config["green_contexts"]
    if config["green_contexts"]:
        assert meta["green_contexts"]["scope"] == "deform_prefix_raster_prepare"
        assert meta["green_contexts"]["deform_sms"] == config["green_deform_sms"]
        assert meta["green_contexts"]["raster_sms"] == 16 - config["green_deform_sms"]
        assert meta["green_contexts"]["raster_prepass_sms"] == 16
        assert meta["green_contexts"]["tail_sms"] == 16
        assert meta["green_contexts"].get("prepare_handoff_enabled", False) == config["green_prepare_handoff"]
        if config["green_prepare_handoff"]:
            assert meta["green_contexts"]["raster_prepare_suffix_sms"] == 16
            assert meta["pipeline_execution_counts"]["raster_prepare_handoffs"] == meta["profile_frames"] - 1
    report = {"mode": options.kernel_mode, "config": config, "throughput_fps": meta["throughput_fps"],
              "driver_sha256": own_hash, "library_sha256": binary_hash,
              "qualification_profile_sha256": meta["qualification_profile_sha256"], "traces": traces,
              "policy": "ONLY concurrent GPU kernels, no added Graph event nodes, no per-frame synchronization or I/O; CSV flush after original timed trial ends"}
    options.kernel_output.write_text(json.dumps(report, indent=2) + "\n")
    print("REDUCTION_PROBE", config, options.kernel_mode, meta["throughput_fps"], flush=True)


if __name__ == "__main__":
    main()
