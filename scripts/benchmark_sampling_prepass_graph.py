"""Isolated, Graph-qualified ATen sampling / v12 prepass dual-stream benchmark.

Production sources and binaries are read-only. Extract the last workload from
each of the two real Graph slots, then freeze all inputs. No prepare, coordinate
gather, feature reduction, MLP, or mixed kernel is included in microbench timings.
"""
import argparse
import csv
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys


QUALIFICATION_SHA = "a13d87f6c4a22a24b20f83cb3c66a009e759b739b498c9ff55f8c35c9b82d23a"


def summarize_abba(records):
    blocks = []
    for start in range(0, len(records), 4):
        group = records[start:start + 4]
        if [row["arm"] for row in group] != ["A", "B", "B", "A"]:
            raise ValueError("only complete predeclared ABBA blocks are admitted")
        singles = [row for row in group if row["arm"] == "A"]
        both = statistics.mean(row["both_ms"] for row in group if row["arm"] == "B")
        sampling = statistics.mean(row["sampling_ms"] for row in singles)
        raster = statistics.mean(row["prepass_ms"] for row in singles)
        total = sampling + raster
        blocks.append({"sampling_ms": sampling, "prepass_ms": raster,
                       "standalone_sum_ms": total, "both_ms": both,
                       "saved_ms": total - both,
                       "both_over_standalone_sum": both / total,
                       "speedup": total / both,
                       "time_saved_percent": 100 * (1 - both / total)})
    return {"block_count": len(blocks), "blocks": blocks,
            "median": {key: statistics.median(row[key] for row in blocks)
                       for key in blocks[0]},
            "both_over_sum_range": [min(row["both_over_standalone_sum"] for row in blocks),
                                    max(row["both_over_standalone_sum"] for row in blocks)]}


def analyze_trace(path, mode, repetitions, split_tile):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    sampling = sorted((row for row in rows if "grid_sampler_2d_kernel" in row["name"]),
                      key=lambda row: int(row["start_ns"]))
    raster = sorted((row for row in rows if row["name"] == "tacker_render_raster_fast_range_v12"),
                    key=lambda row: int(row["start_ns"]))
    expected_s = 0 if mode == "prepass" else repetitions * 12
    expected_r = 0 if mode == "sampling" else repetitions
    assert (len(sampling), len(raster), len(rows)) == (expected_s, expected_r, expected_s + expected_r)
    assert all(int(row["graph_node_id"]) > 0 for row in rows)
    assert all((int(row["grid_x"]), int(row["block_x"]), int(row["registers"])) == (436, 256, 48)
               for row in sampling)
    assert all((int(row["grid_x"]), int(row["block_x"]), int(row["registers"])) == (split_tile, 256, 36)
               for row in raster)
    if mode.startswith("both"):
        assert {row["stream_id"] for row in sampling}.isdisjoint(row["stream_id"] for row in raster)
    samples = []
    for index in range(repetitions):
        ss = sampling[index * 12:(index + 1) * 12]
        rr = raster[index:index + 1]
        intervals = [(int(row["start_ns"]), int(row["end_ns"])) for row in ss]
        raster_intervals = [(int(row["start_ns"]), int(row["end_ns"])) for row in rr]
        overlap = sum(max(0, min(b, d) - max(a, c))
                      for a, b in intervals for c, d in raster_intervals)
        all_intervals = intervals + raster_intervals
        samples.append({"sampling_kernel_sum_ms": sum(b - a for a, b in intervals) / 1e6,
                        "sampling_span_ms": ((intervals[-1][1] - intervals[0][0]) / 1e6
                                             if intervals else 0),
                        "prepass_ms": sum(b - a for a, b in raster_intervals) / 1e6,
                        "kernel_window_ms": (max(b for a, b in all_intervals)
                                             - min(a for a, b in all_intervals)) / 1e6,
                        "sampling_prepass_kernel_overlap_ms": overlap / 1e6})
    return {"mode": mode, "kernel_count": len(rows), "graph_nodes_verified": True,
            "stream_ids": {"sampling": sorted({row["stream_id"] for row in sampling}),
                           "prepass": sorted({row["stream_id"] for row in raster})},
            "samples": samples,
            "mean": {key: statistics.mean(row[key] for row in samples) for key in samples[0]}}


class WorkloadExtractor:
    def __init__(self, torch):
        self.torch = torch
        self.banks = {}
        self.current_capture = None
        self.latest_slot = None
        self.active = False
        self.prepasses = {}
        self.prepass_index = 0

    def install(self):
        import scene.hexplane as hexplane
        import gaussian_renderer.tacker_pipeline as pipeline
        sample = hexplane.grid_sample_wrapper
        initialize = pipeline._HexPlaneCudaGraphSlot.__init__
        enqueue = pipeline._HexPlaneCudaGraphSlot.enqueue
        prepass = pipeline.FullMlpOverlapPartition.launch_prepared_raster_prefix
        extractor = self

        def wrapped_sample(grid, coords, align_corners=True):
            output = sample(grid, coords, align_corners)
            if extractor.current_capture is not None and extractor.torch.cuda.is_current_stream_capturing():
                extractor.current_capture.append((grid, coords, align_corners, output))
            return output

        def wrapped_initialize(slot, *args, **kwargs):
            if extractor.current_capture is not None:
                raise RuntimeError("nested HexPlane capture")
            extractor.current_capture = []
            try:
                initialize(slot, *args, **kwargs)
                calls = extractor.current_capture
                assert len(calls) == 12
                extractor.banks[id(slot)] = {"slot": slot, "calls": calls}
            finally:
                extractor.current_capture = None

        def wrapped_enqueue(slot, *args, **kwargs):
            result = enqueue(slot, *args, **kwargs)
            if extractor.active:
                extractor.latest_slot = id(slot)
            return result

        def wrapped_prepass(partition, context, prepared, split_tile):
            result = prepass(partition, context, prepared, split_tile)
            if extractor.active:
                assert extractor.latest_slot in extractor.banks
                # Only the LAST use of a slot is retained. Its coordinate and
                # sample buffers then still describe this exact next frame.
                extractor.prepasses[extractor.latest_slot] = {
                    "context": context, "prepared": prepared, "split_tile": split_tile,
                    "step": extractor.prepass_index,
                    "bank": extractor.banks[extractor.latest_slot]}
                extractor.prepass_index += 1
            return result

        hexplane.grid_sample_wrapper = wrapped_sample
        pipeline._HexPlaneCudaGraphSlot.__init__ = wrapped_initialize
        pipeline._HexPlaneCudaGraphSlot.enqueue = wrapped_enqueue
        pipeline.FullMlpOverlapPartition.launch_prepared_raster_prefix = wrapped_prepass


class FrozenCase:
    def __init__(self, torch, source, sampling_priority="default", node_priority_backend=False):
        self.torch = torch
        self.step = source["step"]
        self.context = source["context"]
        self.split_tile = source["split_tile"]
        self.prepared = type(source["prepared"])(*(
            value.clone() if isinstance(value, torch.Tensor) else value
            for value in source["prepared"]))
        self.calls = [(grid, coords.clone(), align) for grid, coords, align, _ in source["bank"]["calls"]]
        self.samples_by_mode = {}
        self.latest_samples = None
        self.priority_driver = None
        self.priority_policy = sampling_priority
        self.priority_range = None
        self.priorities = {key: 0 for key in ("control", "sampling", "prepass")}
        if sampling_priority not in ("default", "high"):
            raise ValueError("unknown sampling priority: " + sampling_priority)
        if sampling_priority == "high" or node_priority_backend:
            from sampling_prepass_priority import CudaDriver
            self.priority_driver = CudaDriver()
            self.priority_range = self.priority_driver.priority_range()
            if sampling_priority == "high":
                self.priorities["sampling"] = self.priority_range["greatest"]
        self.requested_priorities = dict(self.priorities)
        self.streams = {key: torch.cuda.Stream(priority=value) for key, value in self.priorities.items()}
        if self.priority_driver is not None:
            from sampling_prepass_priority import validate_stream_priorities
            actual = {key: self.priority_driver.stream_priority(stream) for key, stream in self.streams.items()}
            if sampling_priority == "high":
                self.priorities = validate_stream_priorities(actual, self.priority_range)
            elif actual != self.priorities:
                raise RuntimeError("default-priority control did not retain priority 0")
        self.graphs = {}
        self.priority_audits = {}
        self.events_keepalive = []
        self.raster_reference = None
        self.sampling_reference = None
        torch.cuda.synchronize()
        self.make_references()
        for mode in ("sampling", "prepass", "serial", "both", "both_reverse"):
            self.capture(mode)
        for mode in self.graphs:
            self.check_outputs(mode)

    def sample(self):
        # Use the EXACT original functional call. In this installed PyTorch,
        # the .out overload is a functional wrapper plus a device copy and
        # therefore adds twelve memcpy32_post kernels: it is NOT admissible.
        from scene.hexplane import grid_sample_wrapper
        self.latest_samples = [grid_sample_wrapper(grid, coords, align)
                               for grid, coords, align in self.calls]

    def raster(self):
        self.context.rasterizer.render_prepared_raster_fast_range(
            self.prepared, 0, self.split_tile)

    def make_references(self):
        from scene.hexplane import grid_sample_wrapper
        self.sampling_reference = [grid_sample_wrapper(grid, coords, align).clone()
                                   for grid, coords, align in self.calls]
        self.raster()
        self.torch.cuda.synchronize()
        self.raster_reference = {key: getattr(self.prepared, key).clone()
                                 for key in ("color", "depth", "image_buffer")}

    def capture(self, mode):
        torch = self.torch
        control, sampling, raster = (self.streams[key] for key in ("control", "sampling", "prepass"))
        fork, sampled, rendered = (torch.cuda.Event() for _ in range(3))
        self.events_keepalive.extend((fork, sampled, rendered))
        for event in (fork, sampled, rendered):
            event.record(control)
        torch.cuda.synchronize()
        for _ in range(3):
            self.sample()
            self.raster()
        torch.cuda.synchronize()
        graph = (torch.cuda.CUDAGraph(keep_graph=True) if self.priority_driver is not None
                 else torch.cuda.CUDAGraph())
        with torch.cuda.graph(graph, stream=control, capture_error_mode="thread_local"):
            fork.record(control)
            if mode in ("sampling", "serial", "both", "both_reverse"):
                sampling.wait_event(fork)
            if mode in ("prepass", "both", "both_reverse"):
                raster.wait_event(fork)
            if mode == "both_reverse":
                with torch.cuda.stream(raster):
                    self.raster()
                    rendered.record(raster)
            if mode != "prepass":
                with torch.cuda.stream(sampling):
                    self.sample()
                    sampled.record(sampling)
            if mode == "serial":
                raster.wait_event(sampled)
            if mode in ("prepass", "serial", "both"):
                with torch.cuda.stream(raster):
                    self.raster()
                    rendered.record(raster)
            if mode != "prepass":
                control.wait_event(sampled)
            if mode != "sampling":
                control.wait_event(rendered)
        control.synchronize()
        if self.priority_driver is not None:
            graph = self.wrap_captured_graph(graph, mode)
            self.priority_audits[mode] = graph.audit
        self.graphs[mode] = graph
        self.samples_by_mode[mode] = None if mode == "prepass" else self.latest_samples

    def wrap_captured_graph(self, graph, mode):
        from sampling_prepass_priority import PriorityGraph
        return PriorityGraph(self.torch, graph, self.priority_driver, mode,
                             self.priorities, self.split_tile)

    def check_outputs(self, mode):
        # Poison only the regions this arm promises to overwrite, making a
        # no-op graph fail correctness instead of reusing a previous result.
        torch = self.torch
        if mode != "prepass":
            for sample in self.samples_by_mode[mode]:
                sample.fill_(float("nan"))
        settings = self.context.rasterizer.raster_settings
        width, height = int(settings.image_width), int(settings.image_height)
        valid = None
        if mode != "sampling":
            y = torch.arange(height, device="cuda").view(-1, 1)
            x = torch.arange(width, device="cuda").view(1, -1)
            valid = ((y // 16) * ((width + 15) // 16) + x // 16) < self.split_tile
            self.prepared.color[:, valid] = float("nan")
            self.prepared.depth.reshape(height, width)[valid] = float("nan")
        torch.cuda.synchronize()
        with torch.cuda.stream(self.streams["control"]):
            self.graphs[mode].replay()
        self.streams["control"].synchronize()
        if mode != "prepass":
            for actual, expected in zip(self.samples_by_mode[mode], self.sampling_reference):
                self.check_sampling_output(actual, expected, mode)
        if mode != "sampling":
            for key, expected in self.raster_reference.items():
                assert torch.equal(getattr(self.prepared, key), expected), "raster output mismatch: " + mode + " " + key

    def check_sampling_output(self, actual, expected, mode):
        assert self.torch.equal(actual, expected), "sampling output mismatch: " + mode

    def measure(self, mode, repeats, warmup):
        torch = self.torch
        control = self.streams["control"]
        graph = self.graphs[mode]
        pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                 for _ in range(repeats)]
        with torch.cuda.stream(control):
            for a, b in pairs:
                a.record(control)
                b.record(control)
            for _ in range(warmup):
                graph.replay()
        control.synchronize()
        with torch.cuda.stream(control):
            for a, b in pairs:
                a.record(control)
                graph.replay()
                b.record(control)
        control.synchronize()
        values = [a.elapsed_time(b) for a, b in pairs]
        return {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
                "min_ms": min(values), "max_ms": max(values), "samples_ms": values}

    def description(self):
        settings = self.context.rasterizer.raster_settings
        # Outside all timing and tracing. Compare deterministic functional
        # inputs/outputs across independent ABBA processes, never padding bytes
        # in opaque Raster buffers (which need not be initialized).
        def tensor_bytes(tensor):
            return tensor.detach().cpu().contiguous().numpy().tobytes()

        sampling_inputs, sampling_outputs = hashlib.sha256(), hashlib.sha256()
        for grid, coords, align in self.calls:
            sampling_inputs.update(repr((tuple(grid.shape), tuple(coords.shape), align)).encode())
            sampling_inputs.update(tensor_bytes(grid))
            sampling_inputs.update(tensor_bytes(coords))
        for tensor in self.sampling_reference:
            sampling_outputs.update(tensor_bytes(tensor))
        width, height = int(settings.image_width), int(settings.image_height)
        y = self.torch.arange(height).view(-1, 1)
        x = self.torch.arange(width).view(1, -1)
        valid = ((y // 16) * ((width + 15) // 16) + x // 16) < self.split_tile
        raster_outputs = hashlib.sha256()
        raster_outputs.update(tensor_bytes(self.raster_reference["color"].cpu()[:, valid]))
        raster_outputs.update(tensor_bytes(self.raster_reference["depth"].cpu().reshape(height, width)[valid]))
        return {"pipeline_step": self.step, "sampling_input_frame": self.step + 1,
                "raster_input_frame": self.step,
                "sampling_planes": len(self.calls), "points": self.calls[0][1].shape[0],
                "resolution": [int(settings.image_width), int(settings.image_height)],
                "raster_tile_range": [0, self.split_tile],
                "num_rendered": self.prepared.num_rendered,
                "plane_shapes": [list(grid.shape) for grid, _, _ in self.calls],
                "coordinate_strides": [list(coords.stride()) for _, coords, _ in self.calls],
                "sampling_output_layout": "original [N,16] view with stride (1,N), underlying contiguous [16,N]",
                "sampling_stream": self.streams["sampling"].cuda_stream,
                "prepass_stream": self.streams["prepass"].cuda_stream,
                "priority_policy": self.priority_policy, "stream_priorities": self.priorities,
                "requested_stream_priorities": self.requested_priorities,
                "device_priority_range": self.priority_range,
                "graph_priority_audits": self.priority_audits,
                "graph_replay_backend": ("cuda_driver_use_node_priority" if self.priority_driver is not None
                                         else "original_torch_graph"),
                "workload_fingerprints": {"sampling_inputs_sha256": sampling_inputs.hexdigest(),
                                          "sampling_outputs_sha256": sampling_outputs.hexdigest(),
                                          "raster_valid_outputs_sha256": raster_outputs.hexdigest()},
                "sampling_serial_on_one_stream": True,
                "correctness": "all five graphs bitwise equal to original wrapper / prepass outputs after poisoning"}

    def close(self):
        self.torch.cuda.synchronize()
        for graph in self.graphs.values():
            if self.priority_driver is not None:
                graph.close()
            else:
                graph.reset()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--abba-blocks", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--bench-warmup", type=int, default=10)
    parser.add_argument("--trace-library", type=Path)
    parser.add_argument("--trace-repeats", type=int, default=8)
    parser.add_argument("--sampling-priority", choices=("default", "high"), default="default")
    parser.add_argument("--node-priority-backend", action="store_true",
                        help="Use the same driver/flags=8 backend for the priority-0 control")
    options, args = parser.parse_known_args()
    assert options.abba_blocks > 0 and options.repeats > 0
    for flag in ("--execution-mode", "--qualification-mode", "--qualification-profile", "--metadata"):
        assert flag in args, flag
    assert args[args.index("--execution-mode") + 1] == "tacker"
    assert os.environ.get("TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12") == "1"
    for key in ("TACKER_EXPERIMENTAL_GREEN_CONTEXTS", "TACKER_EXPERIMENTAL_RASTER_HEXPLANE",
                "TACKER_HEXPLANE_LOWREG", "TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF"):
        assert os.environ.get(key, "0") == "0", key
    root = options.runtime_root.resolve()
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root))
    from run_profile_render_sealed import _read_stable_source
    record, raw = _read_stable_source(str(root / "profile_render.py"))
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    namespace = {"__name__": "profile_render_sampling_prepass_extractor",
                 "__file__": record["path"], "__package__": None,
                 "_PROFILE_RENDER_BOOTSTRAP": {
                     "protocol": 1, "record": record, "bytes": raw,
                     "execution": {"compiled_sha256": record["sha256"],
                                   "compile_mode": "exec", "dont_inherit": True}}}
    exec(compile(raw, record["path"], "exec", dont_inherit=True), namespace)
    activate, measure = namespace["_activate_runtime_imports"], namespace["_measure_trial"]
    extractor_box = []

    def activation(*aa, **kk):
        activate(*aa, **kk)
        extractor = WorkloadExtractor(namespace["torch"])
        extractor.install()
        extractor_box.append(extractor)

    def measured(*aa, **kk):
        bound = inspect.signature(measure).bind(*aa, **kk).arguments
        extractor = extractor_box[0]
        extractor.active = True
        try:
            result = measure(*aa, **kk)
        finally:
            extractor.active = False
        assert extractor.prepass_index == len(bound["views"]) - 1
        return result

    namespace["_activate_runtime_imports"] = activation
    namespace["_measure_trial"] = measured
    namespace["_cli_main"](args)
    # The production --quiet wrapper suppresses stdout globally; restore it
    # for this separate microbenchmark's own progress and failure diagnostics.
    sys.stdout = sys.__stdout__
    metadata_path = Path(args[args.index("--metadata") + 1])
    metadata = json.loads(metadata_path.read_text())
    assert (metadata["passed"] and metadata["actual_execution_mode"] == "tacker"
            and metadata["qualification_mode_executed"]
            and metadata["selected_variant_id"] == "full_mlp_overlap_graph_pb0"
            and metadata["tacker_fallback_reason"] is None
            and metadata["post_run"]["byte_stability"]["verified"]
            and metadata["qualification_profile_sha256"] == QUALIFICATION_SHA
            and not metadata["green_contexts"]["enabled"])
    torch = namespace["torch"]
    extractor = extractor_box[0]
    assert len(extractor.prepasses) == 2
    report = {"schema_version": 1, "benchmark": "sampling_prepass_dual_stream_graph",
              "qualification_profile": metadata["qualification_profile"],
              "qualification_sha256": metadata["qualification_profile_sha256"],
              "actual_variant": metadata["selected_variant_id"], "driver_sha256": own_hash,
              "workload_extraction_metadata": str(metadata_path),
              "sampling_priority": options.sampling_priority,
              "node_priority_backend": options.node_priority_backend or options.sampling_priority == "high",
              "policy": {
                  "formal_timings_profiler_attached": False,
                  "fixed_input_warmed_graph_microbenchmark": True,
                  "identical_input_addresses_across_arms": True,
                  "output_addresses": "sampling outputs fixed per captured Graph, private addresses differ across arms; original layout and exact native call retained",
                  "both": "one captured fork/join Graph; independent sampling/prepass streams, shared all SMs",
                  "priorities": ("sampling high priority (actual value read back; PyTorch may clamp CUDA range); prepass/control priority 0; verified USE_NODE_PRIORITY"
                                 if options.sampling_priority == "high" else
                                 ("all streams priority 0; same driver/flags=8 backend as high-priority arm"
                                  if options.node_priority_backend else "all streams default priority; original torch Graph replay")),
                  "sampling_serial_on_one_stream": True,
                  "A": "sampling-only Graph and prepass-only Graph measured separately, then summed",
                  "B": "both Graph; sampling branch captured before prepass branch (predeclared primary)",
                  "both_reverse": "prepass branch captured first; secondary launch-order control, not best-of selection",
                  "excludes": ["coordinate preparation/gather", "raster prepare", "feature reduction", "MLP", "mixed kernel"],
                  "scope_warning": "Not full-renderer FPS or normal concurrent counters; fixed inputs are replayed hot, unlike NCU cache-flushed isolated launches.",
                  "standalone_sum": "sum of isolated Graph elapsed times, NOT sum of concurrent event spans",
                  "launch_implementation": "exact original scene.hexplane.grid_sample_wrapper / F.grid_sample call, bilinear/border/align_corners; CUPTI admits exactly 12 sampler kernels and 1 prepass, no extra copy"},
              "cases": []}
    with torch.no_grad():
        cases = [FrozenCase(torch, value, options.sampling_priority, options.node_priority_backend)
                 for value in sorted(extractor.prepasses.values(), key=lambda row: row["step"])]
        for case_index, case in enumerate(cases):
            assert case.split_tile == 1954 and len(case.calls) == 12
            result = {"case": case_index, "workload": case.description(), "records": []}
            for block in range(options.abba_blocks):
                for position, arm in enumerate(("A", "B", "B", "A")):
                    entry = {"block": block, "position": position, "arm": arm}
                    if arm == "A":
                        order = ("sampling", "prepass") if position == 0 else ("prepass", "sampling")
                        entry["standalone_order"] = list(order)
                        for mode in order:
                            timing = case.measure(mode, options.repeats, options.bench_warmup)
                            entry[mode + "_ms"] = timing["mean_ms"]
                            entry[mode + "_timing"] = timing
                    else:
                        timing = case.measure("both", options.repeats, options.bench_warmup)
                        entry["both_ms"] = timing["mean_ms"]
                        entry["both_timing"] = timing
                    result["records"].append(entry)
                    print("BENCH", case_index, block, position, arm,
                          {key: value for key, value in entry.items() if key.endswith("_ms")}, flush=True)
            result["summary"] = summarize_abba(result["records"])
            result["serial_control"] = case.measure("serial", options.repeats, options.bench_warmup)
            result["reverse_branch_order_control"] = case.measure("both_reverse", options.repeats, options.bench_warmup)
            report["cases"].append(result)
        # Keep primary timings even if an optional diagnostic collector fails.
        options.benchmark_output.with_name("benchmark.primary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        # Attach CUPTI ONLY after all primary/secondary unprofiled timings.
        if options.trace_library is not None:
            library = ctypes.CDLL(str(options.trace_library.resolve()))
            library.sampling_activity_write.argtypes = [ctypes.c_char_p]
            status = library.sampling_activity_init()
            if status != 0:
                raise RuntimeError("CUPTI concurrent-kernel activity init returned " + str(status))
            report["trace_library_sha256"] = hashlib.sha256(options.trace_library.read_bytes()).hexdigest()
            for case, result in zip(cases, report["cases"]):
                result["diagnostic_traces"] = []
                for mode in ("sampling", "prepass", "both", "both_reverse"):
                    with torch.cuda.stream(case.streams["control"]):
                        for _ in range(5):
                            case.graphs[mode].replay()
                    case.streams["control"].synchronize()
                    assert library.sampling_activity_clear() == 0
                    with torch.cuda.stream(case.streams["control"]):
                        for _ in range(options.trace_repeats):
                            case.graphs[mode].replay()
                    case.streams["control"].synchronize()
                    path = options.benchmark_output.with_name("case%d.%s.kernels.csv" % (result["case"], mode))
                    assert library.sampling_activity_write(str(path).encode()) == 0
                    result["diagnostic_traces"].append(analyze_trace(path, mode, options.trace_repeats, case.split_tile))
                for mode in case.graphs:
                    case.check_outputs(mode)
            assert library.sampling_activity_stop() == 0
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == own_hash
    report["sampling_ms"] = statistics.mean(case["summary"]["median"]["sampling_ms"] for case in report["cases"])
    report["prepass_ms"] = statistics.mean(case["summary"]["median"]["prepass_ms"] for case in report["cases"])
    report["both_ms"] = statistics.mean(case["summary"]["median"]["both_ms"] for case in report["cases"])
    report["standalone_sum_ms"] = report["sampling_ms"] + report["prepass_ms"]
    report["both_over_standalone_sum"] = report["both_ms"] / report["standalone_sum_ms"]
    report["time_saved_percent"] = 100 * (1 - report["both_over_standalone_sum"])
    report["speedup"] = 1 / report["both_over_standalone_sum"]
    report["correctness_passed"] = True
    for case in cases:
        for mode in case.graphs:
            case.check_outputs(mode)
        case.close()
    options.benchmark_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("SUMMARY", json.dumps({key: report[key] for key in (
        "sampling_ms", "prepass_ms", "standalone_sum_ms", "both_ms", "both_over_standalone_sum", "time_saved_percent", "speedup")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print("BENCHMARK_FAILURE", type(error).__name__, str(error), file=sys.stderr, flush=True)
        trace = error.__traceback__
        while trace is not None:
            print("FAILURE_LOCATION", trace.tb_frame.f_code.co_filename,
                  trace.tb_lineno, trace.tb_frame.f_code.co_name, file=sys.stderr, flush=True)
            trace = trace.tb_next
        raise
