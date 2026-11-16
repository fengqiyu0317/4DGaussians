"""A/B/C fixed-input Graph experiment extracted from full_mlp_overlap_graph.

A: original 12 ATen sampling kernels + independent prepass (two streams).
B: one all-plane sampling kernel + independent prepass (two streams).
C: one 256-thread/block mixed kernel, one task type per whole block.
All primary timings wait for both outputs. This measures a stage, not FPS.
"""
import argparse
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import statistics
import sys

from benchmark_sampling_prepass_graph import FrozenCase, WorkloadExtractor
from sampling_prepass_trace import split_graph_kernels


def summarize_abc(records):
    if not records or len(records) % 6:
        raise ValueError("require complete A,B,C,C,B,A blocks")
    blocks = []
    for begin in range(0, len(records), 6):
        rows = records[begin:begin + 6]
        if [row["arm"] for row in rows] != list("ABCCBA"):
            raise ValueError("require predeclared A,B,C,C,B,A order")
        times = {arm: statistics.mean(row["mean_ms"] for row in rows if row["arm"] == arm)
                 for arm in "ABC"}
        if any(value <= 0 for value in times.values()):
            raise ValueError("timings must be positive")
        blocks.append({**{arm + "_ms": value for arm, value in times.items()},
                       "B_over_A": times["B"] / times["A"],
                       "C_over_B": times["C"] / times["B"],
                       "C_over_A": times["C"] / times["A"]})
    return {"block_count": len(blocks), "blocks": blocks,
            "median": {key: statistics.median(row[key] for row in blocks) for key in blocks[0]}}


def validate_abc_nodes(nodes, arm, mode, points, split_tile):
    """Validate exact capture geometry and priority before Driver replay."""
    if arm not in ("A", "B", "C") or mode not in ("sampling", "prepass", "serial", "both", "both_reverse"):
        raise ValueError("unknown A/B/C Graph arm or mode")
    point_tiles = (points + 255) // 256
    fused = arm == "C" and mode in ("both", "both_reverse")
    expected = {"sampling": 0 if mode == "prepass" or fused else (12 if arm == "A" else 1),
                "prepass": 0 if mode == "sampling" or fused else 1,
                "mixed": int(fused)}
    counts = {key: 0 for key in expected}
    for node in nodes:
        name = node["name"]
        if "grid_sampler_2d_kernel" in name and arm == "A":
            kind, grid = "sampling", point_tiles
        elif name == "hexplane_sample_planes_batched_v1" and arm in ("B", "C") and not fused:
            kind, grid = "sampling", 12 * point_tiles
        elif name == "tacker_render_raster_fast_range_v12" and not fused:
            kind, grid = "prepass", split_tile
        elif name == "tacker_mix_raster_hexplane_blocks_v1" and fused:
            kind, grid = "mixed", 12 * point_tiles + split_tile
        else:
            raise RuntimeError("unexpected A/B/C Graph kernel: " + name)
        if (node["priority"] != 0 or node["grid"] != [grid, 1, 1]
                or node["block"] != [256, 1, 1]):
            raise RuntimeError("A/B/C Graph priority or geometry mismatch: " + str(node))
        counts[kind] += 1
    if counts != expected:
        raise RuntimeError("A/B/C Graph node counts mismatch: " + str(counts))
    return counts


class ABCDriverGraph:
    """Same USE_NODE_PRIORITY backend as v13; capture owner retains storage."""
    def __init__(self, case, owner, mode):
        from sampling_prepass_priority import verify_instantiation
        self.torch, self.owner, self.driver = case.torch, owner, case.priority_driver
        graph = owner.raw_cuda_graph()
        nodes = self.driver.kernel_nodes(graph)
        points = case.calls[0][1].shape[0]
        counts = validate_abc_nodes(nodes, case.arm, mode, points, case.split_tile)
        self.executable = self.driver.instantiate(graph)
        try:
            flags = verify_instantiation(self.driver, self.executable, graph)
        except BaseException:
            self.driver.destroy(self.executable)
            self.executable = None
            raise
        self.audit = {"verified": True, **flags, "use_node_priority": True,
                      "kernel_counts": counts, "nodes": nodes}

    def replay(self):
        if self.executable is None:
            raise RuntimeError("A/B/C Driver Graph has been closed")
        self.driver.launch(self.executable, self.torch.cuda.current_stream())

    def close(self):
        if self.executable is not None:
            self.torch.cuda.synchronize()
            self.driver.destroy(self.executable)
            self.executable = None
            self.owner.reset()


class ComparisonCase(FrozenCase):
    def __init__(self, torch, source, arm="A", node_priority_backend=False):
        self.arm = arm
        super().__init__(torch, source, node_priority_backend=node_priority_backend)

    def wrap_captured_graph(self, graph, mode):
        return ABCDriverGraph(self, graph, mode)


class BatchedCase(ComparisonCase):
    def __init__(self, torch, source, arm, node_priority_backend=False):
        if arm not in ("B", "C"):
            raise ValueError("custom case must be B or C")
        self.arm = arm
        calls = source["bank"]["calls"]
        if len(calls) != 12:
            raise ValueError("B/C require two levels of six planes")
        self.planes = tuple(call[0] for call in calls)
        self.axes = tuple(itertools.combinations(range(4), 2)) * 2
        if "bc_coords" not in source:
            coords = torch.empty((calls[0][1].shape[0], 4), device=calls[0][1].device,
                                 dtype=torch.float32)
            written = set()
            for (_, pair, align, _), axes in zip(calls, self.axes):
                if not align or pair.shape != (coords.shape[0], 2):
                    raise ValueError("B/C require align_corners=True and [N,2] plane coords")
                for column, axis in enumerate(axes):
                    if axis in written:
                        if not torch.equal(coords[:, axis], pair[:, column]):
                            raise ValueError("plane coordinate order does not match HexPlane axes")
                    else:
                        coords[:, axis].copy_(pair[:, column])
                        written.add(axis)
            source["bc_coords"] = coords
        self.coords = source["bc_coords"]  # Same input address for B and C.
        self.outputs = tuple(torch.empty((16, self.coords.shape[0]), device=self.coords.device).t()
                             for _ in self.planes)
        super().__init__(torch, source, arm, node_priority_backend=node_priority_backend)

    def sample(self):
        from tacker_4dgs_head.hexplane import sample_planes
        self.latest_samples = sample_planes(self.coords, self.planes, self.axes, outputs=self.outputs)

    def capture(self, mode):
        if self.arm != "C" or mode not in ("both", "both_reverse"):
            return super().capture(mode)
        torch = self.torch
        control = self.streams["control"]
        def mixed():
            self.context.rasterizer.render_prepared_with_hexplane_blocks(
                self.prepared, self.coords, self.planes, self.axes,
                raster_start=0, raster_end=self.split_tile, outputs=self.outputs)
        torch.cuda.synchronize()
        with torch.cuda.stream(control):
            for _ in range(3):
                mixed()
        control.synchronize()
        graph = (torch.cuda.CUDAGraph(keep_graph=True) if self.priority_driver is not None
                 else torch.cuda.CUDAGraph())
        with torch.cuda.graph(graph, stream=control):
            mixed()
        if self.priority_driver is not None:
            graph = self.wrap_captured_graph(graph, mode)
            self.priority_audits[mode] = graph.audit
        self.graphs[mode] = graph
        self.samples_by_mode[mode] = self.outputs

    def check_sampling_output(self, actual, expected, mode):
        # Same arithmetic in B/C; original ATen may contract multiply-adds.
        self.torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6,
                                       msg="sampling output mismatch: " + self.arm + " " + mode)

    def description(self):
        result = super().description()
        result.update(implementation=self.arm,
                      correctness="poisoned outputs; sampling FP32-close to ATen, Raster bitwise equal",
                      sampling_implementation="hexplane_sample_task_device shared by B/C",
                      sampling_kernel_count=1,
                      combined_kernel_count=2 if self.arm == "B" else 1,
                      unified_coords_address=self.coords.data_ptr())
        return result


def audit_kernels(case, arm, path):
    torch = case.torch
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
    ) as prof:
        with torch.cuda.stream(case.streams["control"]):
            case.graphs["both"].replay()
        case.streams["control"].synchronize()
        prof.step()
        with torch.cuda.stream(case.streams["control"]):
            case.graphs["both"].replay()
        case.streams["control"].synchronize()
        prof.step()
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    driver_audit = case.priority_audits.get('both')
    body, outside = split_graph_kernels(trace, driver_audit=driver_audit)
    kernels = [row["name"] for row in body]
    expected = {"A": 13, "B": 2, "C": 1}[arm]
    if len(kernels) != expected:
        raise RuntimeError(f"{arm}: expected {expected} kernels, observed {kernels}")
    if arm == "A":
        valid = sum("grid_sampler_2d_kernel" in name for name in kernels) == 12
    elif arm == "B":
        valid = kernels.count("hexplane_sample_planes_batched_v1") == 1
    else:
        valid = kernels == ["tacker_mix_raster_hexplane_blocks_v1"]
    if arm in ("A", "B"):
        valid = valid and kernels.count("tacker_render_raster_fast_range_v12") == 1
    if not valid:
        raise RuntimeError(f"{arm}: unexpected kernel symbols: {kernels}")
    trace_nodes = [{'name': row['name'], 'priority': 0,
                    'grid': row['args']['grid'], 'block': row['args']['block']} for row in body]
    validate_abc_nodes(trace_nodes, arm, 'both', case.calls[0][1].shape[0], case.split_tile)
    return {"graph_kernel_count": len(kernels), "symbols": kernels,
            "outside_graph_kernel_count": len(outside),
            "outside_graph_symbols": [row["name"] for row in outside],
            "total_replay_kernel_count": len(kernels) + len(outside),
            "runtime_bookkeeping_included_in_primary_timing": True,
            "independent_driver_node_audit": driver_audit is not None,
            "trace": str(path)}


def extract_workloads(root, args):
    from run_profile_render_sealed import _read_stable_source
    record, raw = _read_stable_source(str(root / "profile_render.py"))
    namespace = {"__name__": "profile_render_sampling_prepass_bc_extractor",
                 "__file__": record["path"], "__package__": None,
                 "_PROFILE_RENDER_BOOTSTRAP": {
                     "protocol": 1, "record": record, "bytes": raw,
                     "execution": {"compiled_sha256": record["sha256"],
                                   "compile_mode": "exec", "dont_inherit": True}}}
    exec(compile(raw, record["path"], "exec", dont_inherit=True), namespace)
    activate, measure = namespace["_activate_runtime_imports"], namespace["_measure_trial"]
    extractors = []
    def activation(*aa, **kk):
        activate(*aa, **kk)
        extractor = WorkloadExtractor(namespace["torch"])
        extractor.install()
        extractors.append(extractor)
    def measured(*aa, **kk):
        bound = inspect.signature(measure).bind(*aa, **kk).arguments
        extractor = extractors[0]
        extractor.active = True
        try:
            result = measure(*aa, **kk)
        finally:
            extractor.active = False
        if extractor.prepass_index != len(bound["views"]) - 1:
            raise RuntimeError("unexpected Raster prepass count in qualified extraction")
        return result
    namespace["_activate_runtime_imports"] = activation
    namespace["_measure_trial"] = measured
    namespace["_cli_main"](args)
    sys.stdout = sys.__stdout__
    metadata_path = Path(args[args.index("--metadata") + 1])
    metadata = json.loads(metadata_path.read_text())
    qualification = Path(args[args.index("--qualification-profile") + 1])
    if not (metadata["passed"] and metadata["qualification_mode_executed"]
            and metadata["actual_execution_mode"] == "tacker"
            and metadata["selected_variant_id"] == "full_mlp_overlap_graph_pb0"
            and metadata["tacker_fallback_reason"] is None
            and metadata["post_run"]["byte_stability"]["verified"]
            and metadata["qualification_profile_sha256"] == hashlib.sha256(qualification.read_bytes()).hexdigest()):
        raise RuntimeError("extraction did not execute the required full_mlp_overlap_graph qualification")
    sources = sorted(extractors[0].prepasses.values(), key=lambda row: row["step"])
    if len(sources) != 2:
        raise RuntimeError("require both real Graph slot workloads; use at least three frames")
    return namespace["torch"], metadata, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--abba-blocks", type=int, default=4,
                        help="Number of symmetric A,B,C,C,B,A timing blocks")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--bench-warmup", type=int, default=10)
    parser.add_argument("--node-priority-backend", action="store_true",
                        help="Use v13's CUDA Driver USE_NODE_PRIORITY backend for all three arms")
    options, args = parser.parse_known_args()
    if options.abba_blocks < 1 or options.repeats < 1 or options.bench_warmup < 0:
        parser.error("counts must be positive (warmup may be zero)")
    for flag in ("--execution-mode", "--qualification-mode", "--qualification-profile", "--metadata"):
        if flag not in args:
            parser.error("required renderer argument: " + flag)
    if args[args.index("--execution-mode") + 1] != "tacker":
        parser.error("extraction requires --execution-mode tacker")
    if os.environ.get("TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12") != "1":
        parser.error("set TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1")
    for key in ("TACKER_EXPERIMENTAL_GREEN_CONTEXTS", "TACKER_EXPERIMENTAL_RASTER_HEXPLANE",
                "TACKER_HEXPLANE_LOWREG", "TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF"):
        if os.environ.get(key, "0") != "0":
            parser.error("baseline extraction requires " + key + "=0")
    root = options.runtime_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    output = options.benchmark_output.resolve()
    if output.exists():
        parser.error("refusing to overwrite benchmark output")
    output.parent.mkdir(parents=True, exist_ok=True)
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch, metadata, sources = extract_workloads(root, args)
    report = {"schema_version": 1, "benchmark": "sampling_prepass_ABC_graph",
              "qualification_profile": metadata["qualification_profile"],
              "qualification_sha256": metadata["qualification_profile_sha256"],
              "actual_variant": metadata["selected_variant_id"],
              "driver_sha256": own_hash,
              "scope": "fixed-input warmed Graph stage microbenchmark; not whole-frame FPS",
              "policy": {"order": "ABCCBA", "streams_priority": 0, "SM_partition": False,
                         "graph_backend": ("cuda_driver_use_node_priority" if options.node_priority_backend
                                           else "torch.cuda.CUDAGraph (same for all three arms)"),
                         "requested_graph_instantiate_flags": 8 if options.node_priority_backend else None,
                         "A": "12 original sampling kernels + independent prepass; two streams",
                         "B": "one all-plane sampling kernel + independent prepass; two streams",
                         "C": "one whole-block mixed kernel; 256 threads per block",
                         "sampling_inputs": "identical values; B/C share the unified [N,4] coordinate address",
                         "timing": "both inputs ready through completion of both sampling and prepass",
                         "excludes": ["coordinate preparation", "raster prepare", "reduction", "MLP"],
                         "profiler": "kernel-count audits run only after all primary timings"},
              "cases": []}
    all_cases = []
    try:
        with torch.no_grad():
            for source in sources:
                cases = {"A": ComparisonCase(torch, source, node_priority_backend=options.node_priority_backend),
                         "B": BatchedCase(torch, source, "B", options.node_priority_backend),
                         "C": BatchedCase(torch, source, "C", options.node_priority_backend)}
                all_cases.append(cases)
                result = {"case": len(all_cases) - 1,
                          "workloads": {arm: case.description() for arm, case in cases.items()},
                          "records": []}
                for block in range(options.abba_blocks):
                    for position, arm in enumerate("ABCCBA"):
                        timing = cases[arm].measure("both", options.repeats, options.bench_warmup)
                        result["records"].append({"block": block, "position": position,
                                                  "arm": arm, **timing})
                        print("BENCH", result["case"], block, position, arm, timing["mean_ms"], flush=True)
                result["summary"] = summarize_abc(result["records"])
                report["cases"].append(result)
            output.with_suffix(".primary.json").write_text(json.dumps(report, indent=2) + "\n")
            for cases, result in zip(all_cases, report["cases"]):
                result["isolated_controls"] = {
                    "A_sampling": cases["A"].measure("sampling", options.repeats, options.bench_warmup),
                    "B_sampling": cases["B"].measure("sampling", options.repeats, options.bench_warmup),
                    "prepass": cases["A"].measure("prepass", options.repeats, options.bench_warmup),
                    "A_reverse": cases["A"].measure("both_reverse", options.repeats, options.bench_warmup),
                    "B_reverse": cases["B"].measure("both_reverse", options.repeats, options.bench_warmup),
                }
            for cases, result in zip(all_cases, report["cases"]):
                result["kernel_audits"] = {}
                for arm, case in cases.items():
                    case.check_outputs("both")
                    result["kernel_audits"][arm] = audit_kernels(
                        case, arm, output.with_name(f"case{result['case']}.{arm}.trace.json"))
                for b, c in zip(cases["B"].outputs, cases["C"].outputs):
                    torch.testing.assert_close(b, c, rtol=0, atol=0)
                result["B_C_sampling_bitwise_equal"] = True
            from tacker_4dgs_head.hexplane import batch_resources
            report["resources"] = {"B_sampling": batch_resources(),
                                   "C_mixed": all_cases[0]["C"].context.rasterizer.raster_hexplane_block_resources()}
            report["correctness_passed"] = True
            if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != own_hash:
                raise RuntimeError("benchmark source changed during execution")
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    finally:
        for cases in all_cases:
            for case in cases.values():
                case.close()


if __name__ == "__main__":
    main()
