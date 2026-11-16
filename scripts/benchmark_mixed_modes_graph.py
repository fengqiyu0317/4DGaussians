"""Graph-qualified v11/v13 mixed-tail controls on identical frozen real inputs.

Requires the isolated binary from prepare_mixed_modes_graph_experiment.py.
The v13 Raster control contains only the post-prepass tile range. Compilation,
input extraction, preparation, output checks and profiling are outside timing.
"""
import argparse
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys


QUALIFICATION_SHA = "a13d87f6c4a22a24b20f83cb3c66a009e759b739b498c9ff55f8c35c9b82d23a"
MODE_ENV = "TACKER_BENCHMARK_MIXED_MODE"
SYMBOLS = {
    "v11": "tacker_mix_render_deform_mlp_tensor_tail_raster_fast_v11",
    "v13": "tacker_mix_render_deform_mlp_tensor_tail_split_raster_v12",
}


class Extractor:
    def __init__(self, torch):
        self.torch, self.active, self.calls, self.cases = torch, False, 0, []

    def install(self):
        import gaussian_renderer.tacker_pipeline as pipeline
        original, extractor = pipeline._forward_prepared_with_deform_mlp, self

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if extractor.active:
                values = inspect.signature(original).bind(*args, **kwargs).arguments
                assert values["persistent_blocks"] == 0
                assert values["raster_start"] == 1954
                task, prepared = values["task"], values["prepared_raster"]
                # The caller already joined prefix_ready on the Raster stream.
                # Clone here before the other Graph slot can reuse the buffers.
                copied = type(prepared)(*(value.clone() if isinstance(value, extractor.torch.Tensor)
                                          else value for value in prepared))
                from tacker_4dgs_head import pack_deform_tail_weights_v3
                extractor.cases.append({
                    "step": extractor.calls, "rasterizer": values["context"].rasterizer,
                    "prepared": copied, "split_tile": values["raster_start"],
                    "arguments": {
                        "deform_features": task.deform_features.clone(),
                        "backbone_weight": task.backbone_weight,
                        "backbone_bias": task.backbone_bias,
                        "first_weights": task.full_first_weights,
                        "first_biases": task.full_first_biases,
                        "packed_tail_weights": pack_deform_tail_weights_v3(task.full_tail_weights),
                        "tail_biases": task.full_tail_biases,
                        "persistent_blocks": 0,
                    },
                })
                extractor.cases = extractor.cases[-2:]
                extractor.calls += 1
            return result

        pipeline._forward_prepared_with_deform_mlp = wrapped


class Case:
    def __init__(self, torch, source):
        from sampling_prepass_priority import CudaDriver
        self.torch, self.source = torch, source
        self.stream, self.driver = torch.cuda.Stream(priority=0), CudaDriver()
        self.graphs, self.outputs, self.audits, self.references = {}, {}, {}, {}
        torch.cuda.synchronize()
        with torch.cuda.stream(self.stream):
            for variant in SYMBOLS:
                result = self.call(variant, "production")
                self.references[variant] = {
                    "color": result[0].clone(), "depth": result[2].clone(),
                    "heads": tuple(value.clone() for value in result[3]),
                }
        self.stream.synchronize()
        for key in ("color", "depth"):
            assert torch.equal(self.references["v11"][key], self.references["v13"][key]), key
        assert all(torch.equal(a, b) for a, b in zip(self.references["v11"]["heads"],
                                                    self.references["v13"]["heads"]))
        for variant in SYMBOLS:
            for mode in ("production", "both", "raster_only", "deform_only"):
                self.capture(variant, mode)
                self.check(variant, mode)

    def call(self, variant, mode):
        if mode == "production":
            os.environ.pop(MODE_ENV, None)
        else:
            os.environ[MODE_ENV] = str({"both": 0, "raster_only": 1, "deform_only": 2}[mode])
        kwargs = {"prepared": self.source["prepared"], **self.source["arguments"]}
        suffix = "raster_fast_overlap" if variant == "v11" else "split_raster_overlap"
        if variant == "v13":
            kwargs["raster_start"] = self.source["split_tile"]
        return getattr(self.source["rasterizer"],
                       "render_prepared_with_deform_mlp_tensor_tail_" + suffix)(**kwargs)

    def capture(self, variant, mode):
        from sampling_prepass_priority import verify_instantiation
        torch, key = self.torch, (variant, mode)
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self.call(variant, mode)
        self.stream.synchronize()
        owner = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(owner, stream=self.stream, capture_error_mode="thread_local"):
            self.outputs[key] = self.call(variant, mode)
        self.stream.synchronize()
        os.environ.pop(MODE_ENV, None)
        nodes = self.driver.kernel_nodes(owner.raw_cuda_graph())
        assert len(nodes) == 1, nodes
        expected = SYMBOLS[variant] + ("" if mode == "production" else "_modes_benchmark")
        node = nodes[0]
        assert node["name"] == expected, node
        assert node["grid"] == [5440 if variant == "v11" else 3486, 1, 1], node
        assert node["block"] == [384, 1, 1] and node["priority"] == 0, node
        executable = self.driver.instantiate(owner.raw_cuda_graph())
        audit = verify_instantiation(self.driver, executable, owner.raw_cuda_graph())
        # Read the actual compiled attributes for each captured function.
        from sampling_prepass_priority import KernelParamsV1
        params = KernelParamsV1()
        self.driver.call("cuGraphKernelNodeGetParams", ctypes.c_void_p(
            self.driver.kernel_handles(owner.raw_cuda_graph())[0]), ctypes.byref(params))
        getter = self.driver.library.cuFuncGetAttribute
        getter.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
        getter.restype = ctypes.c_int
        attrs = {}
        for name, code in (("registers", 4), ("static_shared_bytes", 1), ("local_bytes", 3), ("binary_version", 6)):
            value = ctypes.c_int()
            assert getter(ctypes.byref(value), code, params.func) == 0
            attrs[name] = value.value
        assert attrs["registers"] == 40 and attrs["static_shared_bytes"] == (25232 if variant == "v11" else 25216), attrs
        assert attrs["local_bytes"] == 0 and attrs["binary_version"] == 87, attrs
        if mode != "production":
            assert attrs == self.audits[variant + "/production"]["resources"], attrs
        self.graphs[key] = owner, executable
        self.audits[variant + "/" + mode] = {**audit, "nodes": nodes, "resources": attrs}

    def replay(self, key):
        self.driver.launch(self.graphs[key][1], self.stream)

    def check(self, variant, mode):
        torch, key = self.torch, (variant, mode)
        settings = self.source["rasterizer"].raster_settings
        width, height = int(settings.image_width), int(settings.image_height)
        with torch.cuda.stream(self.stream):
            y = torch.arange(height, device="cuda").view(-1, 1)
            x = torch.arange(width, device="cuda").view(1, -1)
            mask = ((y // 16) * ((width + 15) // 16) + x // 16) >= (
                0 if variant == "v11" else self.source["split_tile"])
            prepared = self.source["prepared"]
            prepared.color[:, mask] = float("nan")
            prepared.depth.reshape(height, width)[mask] = float("nan")
            for output in self.outputs[key][3]:
                output.fill_(float("nan"))
            self.replay(key)
        self.stream.synchronize()
        if mode == "deform_only":
            assert torch.isnan(prepared.color[:, mask]).all()
            assert torch.isnan(prepared.depth.reshape(height, width)[mask]).all()
        else:
            reference = self.references[variant]
            assert torch.equal(prepared.color[:, mask], reference["color"][:, mask]), key
            assert torch.equal(prepared.depth.reshape(height, width)[mask],
                               reference["depth"].reshape(height, width)[mask]), key
        for actual, expected in zip(self.outputs[key][3], self.references[variant]["heads"]):
            if mode == "raster_only":
                assert torch.isnan(actual).all()
            else:
                assert torch.equal(actual, expected), key

    def measure(self, variant, mode, repeats, warmup):
        torch, key = self.torch, (variant, mode)
        pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                 for _ in range(repeats)]
        with torch.cuda.stream(self.stream):
            for a, b in pairs:
                a.record(self.stream)
                b.record(self.stream)
            for _ in range(warmup):
                self.replay(key)
        self.stream.synchronize()
        with torch.cuda.stream(self.stream):
            for a, b in pairs:
                a.record(self.stream)
                self.replay(key)
                b.record(self.stream)
        self.stream.synchronize()
        samples = [a.elapsed_time(b) for a, b in pairs]
        return {"mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
                "samples_ms": samples}

    def description(self):
        settings = self.source["rasterizer"].raster_settings
        feature = self.source["arguments"]["deform_features"]
        reference = self.references["v11"]
        def sha(tensor):
            return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return {"pipeline_step": self.source["step"], "resolution": [int(settings.image_width), int(settings.image_height)],
                "deform_rows": feature.shape[0], "deform_tiles": (feature.shape[0] + 15) // 16,
                "v11_raster_tiles": [0, 5440], "v13_raster_tiles": [1954, 5440],
                "excluded_v13_prepass_tiles": [0, 1954],
                "features_sha256": sha(feature), "raster_color_sha256": sha(reference["color"]),
                "deform_outputs_sha256": [sha(x) for x in reference["heads"]],
                "inputs_shared_between_variants_and_modes": True, "correctness": "bitwise production equality; enabled outputs poisoned before replay; disabled outputs stay NaN",
                "graph_audits": self.audits}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--bench-warmup", type=int, default=10)
    options, args = parser.parse_known_args()
    assert MODE_ENV not in os.environ
    root = options.runtime_root.resolve()
    sys.path[:0] = [str(root), str(root / "scripts")]
    from run_profile_render_sealed import _read_stable_source
    record, raw = _read_stable_source(str(root / "profile_render.py"))
    namespace = {"__name__": "profile_render_mixed_modes_extractor", "__file__": record["path"],
                 "__package__": None, "_PROFILE_RENDER_BOOTSTRAP": {
                     "protocol": 1, "record": record, "bytes": raw,
                     "execution": {"compiled_sha256": record["sha256"], "compile_mode": "exec", "dont_inherit": True}}}
    exec(compile(raw, record["path"], "exec", dont_inherit=True), namespace)
    activate, measure = namespace["_activate_runtime_imports"], namespace["_measure_trial"]
    boxes = []
    def activation(*aa, **kk):
        activate(*aa, **kk)
        extractor = Extractor(namespace["torch"])
        extractor.install()
        boxes.append(extractor)
    def measured(*aa, **kk):
        boxes[0].active = True
        try:
            return measure(*aa, **kk)
        finally:
            boxes[0].active = False
    namespace["_activate_runtime_imports"], namespace["_measure_trial"] = activation, measured
    namespace["_cli_main"](args)
    sys.stdout = sys.__stdout__
    metadata_path = Path(args[args.index("--metadata") + 1])
    metadata = json.loads(metadata_path.read_text())
    assert metadata["passed"] and metadata["qualification_mode_executed"]
    assert metadata["selected_variant_id"] == "full_mlp_overlap_graph_pb0"
    assert metadata["qualification_profile_sha256"] == QUALIFICATION_SHA
    assert metadata["tacker_fallback_reason"] is None
    assert metadata["post_run"]["byte_stability"]["verified"]
    assert metadata["pipeline_execution_counts"]["raster_prepasses"] == 4
    assert boxes[0].calls == 4 and len(boxes[0].cases) == 2
    torch = namespace["torch"]
    report = {"passed": False, "qualification_profile": metadata["qualification_profile"],
              "qualification_sha256": QUALIFICATION_SHA, "actual_variant": metadata["selected_variant_id"],
              "scope": "isolated warmed mixed tail, real Graph-qualified inputs; CUDA Driver Graph flags=8, priority 0",
              "not_full_frame_fps": True, "v13_base_kernel": "v12", "profiler_attached": False,
              "rounds": options.rounds, "repeats": options.repeats, "warmup": options.bench_warmup,
              "cases": []}
    with torch.no_grad():
        for case_index, source in enumerate(boxes[0].cases):
            case = Case(torch, source)
            result = {"case": case_index, "workload": case.description(), "batches": []}
            arms = [(variant, mode) for variant in SYMBOLS
                    for mode in ("deform_only", "raster_only", "both")]
            for round_index in range(options.rounds):
                for variant, mode in (arms if round_index % 2 == 0 else arms[::-1]):
                    values = case.measure(variant, mode, options.repeats, options.bench_warmup)
                    result["batches"].append({"round": round_index, "variant": variant,
                                               "mode": mode, **values})
                print("ROUND", case_index, round_index + 1, flush=True)
            result["production_controls"] = {variant: case.measure(variant, "production", options.repeats, options.bench_warmup)
                                             for variant in SYMBOLS}
            for variant in SYMBOLS:
                for mode in ("production", "both", "raster_only", "deform_only"):
                    case.check(variant, mode)
            report["cases"].append(result)
            options.benchmark_output.write_text(json.dumps(report, indent=2) + "\n")
            for owner, executable in case.graphs.values():
                case.driver.destroy(executable)
                owner.reset()
        report["passed"] = True
    options.benchmark_output.write_text(json.dumps(report, indent=2) + "\n")
    print("BENCHMARK_DONE", options.benchmark_output, flush=True)


if __name__ == "__main__":
    main()
