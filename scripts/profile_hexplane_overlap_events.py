"""Opt-in event-only probe of the ORIGINAL v12 HexPlane CUDA Graph.

No production modules are edited.  The normal sealed benchmark is executed
from its original bytes; hooks are installed only after its sealed imports.
Each slot's external Graph events are overwritten by replay, so ONLY its last
replay per trial is read, after the benchmark's existing sequence-end sync.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

from run_profile_render_sealed import _read_stable_source


def interval_overlap(first, second):
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def summarize_samples(samples):
    durations = [sum(b - a for a, b in row["sampling_intervals_ms"]) for row in samples]
    overlaps = {
        name: [sum(interval_overlap(interval, row[name + "_interval_ms"])
                   for interval in row["sampling_intervals_ms"]) for row in samples]
        for name in ("raster_prepare", "raster_prepass")
    }
    total = sum(durations)
    return {
        "sample_count": len(samples),
        "mean_sampling_bracket_sum_ms": statistics.mean(durations),
        "median_sampling_bracket_sum_ms": statistics.median(durations),
        "overlap": {name: {
            "mean_ms": statistics.mean(values),
            "fraction_of_sampling_brackets_percent": 100 * sum(values) / total,
            "frames_with_overlap": sum(value > 0 for value in values),
        } for name, values in overlaps.items()},
        "sampling_complete_before_prepass_frames": sum(
            row["sampling_intervals_ms"][-1][1] <= row["raster_prepass_interval_ms"][0]
            for row in samples),
        "median_sampling_end_relative_to_prepass_start_ms": statistics.median(
            row["sampling_intervals_ms"][-1][1] - row["raster_prepass_interval_ms"][0]
            for row in samples),
        "median_raster_prepare_span_ms": statistics.median(
            row["raster_prepare_interval_ms"][1] - row["raster_prepare_interval_ms"][0]
            for row in samples),
        "median_prepass_bracket_ms": statistics.median(
            row["raster_prepass_interval_ms"][1] - row["raster_prepass_interval_ms"][0]
            for row in samples),
    }


class CapturedSamplingEvents:
    """Preinitialized external events in an otherwise unchanged linear Graph."""

    def __init__(self, torch, stream, mode):
        if mode not in ("all", "last"):
            raise ValueError("sampling event mode must be all or last")
        self.torch = torch
        self.stream = stream
        self.plane_ids = tuple(range(12)) if mode == "all" else (11,)
        self.events = {i: (torch.cuda.Event(enable_timing=True, external=True),
                           torch.cuda.Event(enable_timing=True, external=True))
                       for i in self.plane_ids}
        self.capture_count = 0
        self.last_step = None
        # CUDA Event objects initialize lazily; do that before Graph capture.
        for pair in self.events.values():
            for event in pair:
                event.record(stream)
        stream.synchronize()

    def sample(self, function, *args, **kwargs):
        if not self.torch.cuda.is_current_stream_capturing():
            return function(*args, **kwargs)
        plane = self.capture_count
        self.capture_count += 1
        if plane >= 12:
            raise RuntimeError("expected exactly twelve captured sampling calls")
        pair = self.events.get(plane)
        if pair is not None:
            pair[0].record(self.stream)
        result = function(*args, **kwargs)
        if pair is not None:
            pair[1].record(self.stream)
        return result


class EventOnlyRecorder:
    def __init__(self, torch, mode):
        self.torch = torch
        self.mode = mode
        self.banks = {}
        self.capture_bank = None
        self.renderer = None
        self.step = -1
        self.prepare_count = 0
        self.prepass_count = 0
        self.trials = []

    def install(self):
        import scene.hexplane as hexplane
        import gaussian_renderer.tacker_pipeline as pipeline

        original_sample = hexplane.grid_sample_wrapper
        original_init = pipeline._HexPlaneCudaGraphSlot.__init__
        original_replay = pipeline._HexPlaneCudaGraphSlot.enqueue
        original_prefix = pipeline.TackerRenderer._enqueue_prefix
        original_prepare = pipeline.FullMlpOverlapPartition.prepare_raster
        original_prepass = pipeline.FullMlpOverlapPartition.launch_prepared_raster_prefix
        recorder = self

        def sample(*args, **kwargs):
            if recorder.capture_bank is None:
                return original_sample(*args, **kwargs)
            return recorder.capture_bank.sample(original_sample, *args, **kwargs)

        def initialize(slot, network, xyz, timestamp, stream):
            if recorder.renderer is not None:
                raise RuntimeError("Graph construction occurred during a timed trial")
            bank = CapturedSamplingEvents(recorder.torch, stream, recorder.mode)
            if recorder.capture_bank is not None:
                raise RuntimeError("nested HexPlane capture")
            recorder.capture_bank = bank
            try:
                original_init(slot, network, xyz, timestamp, stream)
            finally:
                recorder.capture_bank = None
            if bank.capture_count != 12:
                raise RuntimeError("HexPlane Graph must contain twelve ATen sampling calls")
            recorder.banks[id(slot)] = bank

        def replay(slot, *args, **kwargs):
            result = original_replay(slot, *args, **kwargs)
            if recorder.renderer is not None:
                recorder.banks[id(slot)].last_step = recorder.step
            return result

        def prefix(renderer, *args, **kwargs):
            if renderer is recorder.renderer:
                recorder.step += 1
            return original_prefix(renderer, *args, **kwargs)

        def prepare(partition, *args, **kwargs):
            if recorder.renderer is None:
                return original_prepare(partition, *args, **kwargs)
            pair = recorder.prepare_events[recorder.prepare_count]
            recorder.prepare_count += 1
            stream = recorder.torch.cuda.current_stream()
            pair[0].record(stream)
            result = original_prepare(partition, *args, **kwargs)
            pair[1].record(stream)
            return result

        def prepass(partition, *args, **kwargs):
            if recorder.renderer is None:
                return original_prepass(partition, *args, **kwargs)
            pair = recorder.prepass_events[recorder.prepass_count]
            recorder.prepass_count += 1
            stream = recorder.torch.cuda.current_stream()
            pair[0].record(stream)
            result = original_prepass(partition, *args, **kwargs)
            pair[1].record(stream)
            return result

        hexplane.grid_sample_wrapper = sample
        pipeline._HexPlaneCudaGraphSlot.__init__ = initialize
        pipeline._HexPlaneCudaGraphSlot.enqueue = replay
        pipeline.TackerRenderer._enqueue_prefix = prefix
        pipeline.FullMlpOverlapPartition.prepare_raster = prepare
        pipeline.FullMlpOverlapPartition.launch_prepared_raster_prefix = prepass

    def begin(self, renderer, frame_count):
        if len(self.banks) != 2:
            raise RuntimeError("warmup must have built both original Graph slots")
        self.step = -1
        self.prepare_count = self.prepass_count = 0
        for bank in self.banks.values():
            bank.last_step = None
        self.prepare_events = [(self.torch.cuda.Event(enable_timing=True),
                                self.torch.cuda.Event(enable_timing=True))
                               for _ in range(frame_count)]
        self.prepass_events = [(self.torch.cuda.Event(enable_timing=True),
                                self.torch.cuda.Event(enable_timing=True))
                               for _ in range(frame_count - 1)]
        caller = self.torch.cuda.current_stream()
        for pair in self.prepare_events + self.prepass_events:
            for event in pair:
                event.record(caller)
        self.anchor = self.torch.cuda.Event(enable_timing=True)
        self.anchor.record(caller)
        # Setup-only sync BEFORE either trial timing clock starts.
        self.torch.cuda.synchronize()
        self.renderer = renderer

    def finish(self, result, frame_count):
        self.renderer = None
        if (self.step + 1, self.prepare_count, self.prepass_count) != (
                frame_count - 1, frame_count, frame_count - 1):
            raise RuntimeError("unexpected prefix/prepare/prepass execution counts")
        samples = []
        for bank in sorted(self.banks.values(), key=lambda value: value.last_step):
            step = bank.last_step
            if step not in (frame_count - 3, frame_count - 2):
                raise RuntimeError("event bank does not identify its final replay")
            pairs = [bank.events[i] for i in bank.plane_ids]
            prepare_pair = self.prepare_events[step]
            prepass_pair = self.prepass_events[step]
            # The ORIGINAL _measure_trial has already synchronized the device.
            # Query proves we do not silently introduce a per-frame wait here.
            if not all(event.query() for pair in pairs + [prepare_pair, prepass_pair]
                       for event in pair):
                raise RuntimeError("sequence-end sync did not complete timing events")
            intervals = [[self.anchor.elapsed_time(a), self.anchor.elapsed_time(b)]
                         for a, b in pairs]
            if any(b < a for a, b in intervals):
                raise RuntimeError("negative sampling interval")
            if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
                raise RuntimeError("sampling brackets overlap on the same stream")
            samples.append({
                "pipeline_step": step,
                "deform_input_frame": step + 1,
                "plane_ids": list(bank.plane_ids),
                "sampling_intervals_ms": intervals,
                "raster_prepare_interval_ms": [self.anchor.elapsed_time(e) for e in prepare_pair],
                "raster_prepass_interval_ms": [self.anchor.elapsed_time(e) for e in prepass_pair],
            })
        self.trials.append({"trial_index": result["trial_index"],
                            "throughput_fps": result["throughput_fps"], "samples": samples})


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--event-mode", choices=("off", "last", "all"), required=True)
    parser.add_argument("--event-output", type=Path, required=True)
    options, profile_args = parser.parse_known_args()
    for flag in ("--execution-mode", "--qualification-mode", "--qualification-profile", "--metadata"):
        if flag not in profile_args:
            raise ValueError("event experiment requires " + flag)
    if profile_args[profile_args.index("--execution-mode") + 1] != "tacker":
        raise ValueError("only Graph-qualified tacker mode is supported")
    import os
    for name in ("TACKER_PROFILE_MIXED_PIPELINE_JSONL", "TACKER_PROFILE_MIXED_V12_SPLIT_JSONL",
                 "TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE"):
        if os.environ.get(name):
            raise ValueError("old synchronized diagnostics must be disabled: " + name)
    if os.environ.get("TACKER_EXPERIMENTAL_RASTER_HEXPLANE", "0") != "0":
        raise ValueError("this probe is for original v12, not the new mixed sampler")
    if os.environ.get("TACKER_HEXPLANE_LOWREG", "0") != "0":
        raise ValueError("this probe requires the original ATen sampler")
    if os.environ.get("TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12") != "1":
        raise ValueError("ABI-v12 must be explicitly enabled")

    root = Path(__file__).resolve().parents[1]
    record, raw = _read_stable_source(str(root / "profile_render.py"))
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    namespace = {
        "__name__": "profile_render_event_only_diagnostic",
        "__file__": str(root / "profile_render.py"), "__package__": None,
        "_PROFILE_RENDER_BOOTSTRAP": {
            "protocol": 1, "record": record, "bytes": raw,
            "execution": {"compiled_sha256": record["sha256"],
                          "compile_mode": "exec", "dont_inherit": True}},
    }
    exec(compile(raw, namespace["__file__"], "exec", dont_inherit=True), namespace)
    recorder_box = []
    if options.event_mode != "off":
        activate = namespace["_activate_runtime_imports"]
        measure = namespace["_measure_trial"]

        def activate_with_probe(*args, **kwargs):
            activate(*args, **kwargs)
            recorder = EventOnlyRecorder(namespace["torch"], options.event_mode)
            recorder.install()
            recorder_box.append(recorder)

        def measure_with_probe(*args, **kwargs):
            import inspect
            bound = inspect.signature(measure).bind(*args, **kwargs).arguments
            recorder = recorder_box[0]
            frame_count = len(bound["views"])
            recorder.begin(bound["pipeline_renderer"], frame_count)
            result = measure(*args, **kwargs)
            recorder.finish(result, frame_count)
            return result

        namespace["_activate_runtime_imports"] = activate_with_probe
        namespace["_measure_trial"] = measure_with_probe
    namespace["_cli_main"](profile_args)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != own_hash:
        raise RuntimeError("event probe source changed during execution")
    metadata_path = Path(profile_args[profile_args.index("--metadata") + 1])
    metadata = json.loads(metadata_path.read_text())
    if not (metadata["passed"] and metadata["actual_execution_mode"] == "tacker"
            and metadata["qualification_mode_executed"]
            and metadata["selected_variant_id"] == "full_mlp_overlap_graph_pb0"
            and metadata["post_run"]["byte_stability"]["verified"]
            and metadata["tacker_fallback_reason"] is None):
        raise RuntimeError("probe did not hit the sealed, qualified original Graph path")
    trials = recorder_box[0].trials if recorder_box else []
    samples = [row for trial in trials for row in trial["samples"]]
    report = {
        "schema_version": 1, "mode": options.event_mode,
        "probe_sha256": own_hash, "qualification_profile": metadata["qualification_profile"],
        "qualification_profile_sha256": metadata["qualification_profile_sha256"],
        "selected_variant_id": metadata["selected_variant_id"],
        "benchmark_metadata": str(metadata_path), "throughput_fps": metadata["throughput_fps"],
        "policy": {
            "nsight_or_cupti_tool_attached": False,
            "per_frame_synchronization_added": False,
            "event_nodes_per_graph_replay": {"off": 0, "last": 2, "all": 24}[options.event_mode],
            "raster_event_records_per_steady_step": 0 if options.event_mode == "off" else 4,
            "samples": "last replay of each of two Graph slots, per trial; overwritten earlier replays are NOT read",
            "sampling_bracket": "external CUDA events immediately around unchanged grid_sample call; includes event/Graph scheduling effects, not instruction-active time",
            "raster_prepare": "GPU stage event span includes original D2H/CPU gaps; NOT the union of Raster kernel intervals",
            "prepass": "event bracket around the unchanged single Raster prepass kernel",
            "off_control": "same sealed benchmark bytes without installing any probe hooks",
        },
        "trials": trials, "summary": summarize_samples(samples) if samples else None,
    }
    options.event_output.parent.mkdir(parents=True, exist_ok=True)
    options.event_output.write_text(json.dumps(report, indent=2) + "\n")
    print("EVENT_ONLY_PROBE", options.event_mode,
          json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
