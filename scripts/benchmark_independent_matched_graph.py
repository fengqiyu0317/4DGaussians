"""Dedicated same-device-code Deform/Raster versus production mixed kernels."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys


ARMS = ("deform", "raster_full", "raster_tail", "prepass", "serial_full",
        "serial_tail", "both_v11", "both_v13", "complete_v13")
DEFORM = "tacker_deform_mlp_v7_standalone_matched"
RASTER = "tacker_render_raster_fast_range_v12"
MIXED = {"v11": "tacker_mix_render_deform_mlp_tensor_tail_raster_fast_v11",
         "v13": "tacker_mix_render_deform_mlp_tensor_tail_split_raster_v12"}


class Case:
    def __init__(self, torch, source, library):
        from sampling_prepass_priority import CudaDriver
        self.torch, self.source = torch, source
        self.stream, self.driver = torch.cuda.Stream(priority=0), CudaDriver()
        self.graphs, self.outputs, self.audits = {}, {}, {}
        self.native = library.deform_matched_launch
        self.native.argtypes = [ctypes.c_void_p]*12 + [ctypes.c_int, ctypes.c_size_t]
        self.native.restype = ctypes.c_int
        a = source["arguments"]
        self.input_ptrs = [a[key].data_ptr() for key in (
            "deform_features", "backbone_weight", "backbone_bias", "first_weights",
            "first_biases", "packed_tail_weights", "tail_biases")]
        self.rows = a["deform_features"].shape[0]
        self.native_outputs = tuple(torch.empty((self.rows, width), device="cuda")
                                    for width in (3, 3, 4, 1, 48))
        self.native_ptrs = self.input_ptrs + [x.data_ptr() for x in self.native_outputs]
        torch.cuda.synchronize()
        with torch.cuda.stream(self.stream):
            ref = self.mixed("v11")
            self.reference = {"color": ref[0].clone(), "depth": ref[2].clone(),
                              "heads": tuple(x.clone() for x in ref[3])}
            ref13 = self.mixed("v13")
        self.stream.synchronize()
        assert torch.equal(ref13[0], self.reference["color"])
        assert torch.equal(ref13[2], self.reference["depth"])
        assert all(torch.equal(a, b) for a, b in zip(ref13[3], self.reference["heads"]))
        for arm in ARMS:
            self.capture(arm)
            self.check(arm)

    def deform(self):
        status = self.native(*self.native_ptrs, self.rows, self.torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError("independent Deform launch error " + str(status))
        return self.native_outputs

    def raster(self, start, end):
        self.source["rasterizer"].render_prepared_raster_fast_range(
            self.source["prepared"], start, end)

    def mixed(self, variant):
        kwargs = {"prepared": self.source["prepared"], **self.source["arguments"]}
        suffix = "raster_fast_overlap" if variant == "v11" else "split_raster_overlap"
        if variant == "v13":
            kwargs["raster_start"] = self.source["split_tile"]
        return getattr(self.source["rasterizer"],
                       "render_prepared_with_deform_mlp_tensor_tail_"+suffix)(**kwargs)

    def call(self, arm):
        if arm == "deform": return self.deform()
        if arm == "raster_full": self.raster(0, 5440)
        elif arm == "raster_tail": self.raster(1954, 5440)
        elif arm == "prepass": self.raster(0, 1954)
        elif arm in ("serial_full", "serial_tail"):
            self.raster(0 if arm == "serial_full" else 1954, 5440)
            return self.deform()
        elif arm in ("both_v11", "both_v13"):
            return self.mixed(arm[-3:])[3]
        elif arm == "complete_v13":
            self.raster(0, 1954)
            return self.mixed("v13")[3]
        return ()

    def capture(self, arm):
        from sampling_prepass_priority import KernelParamsV1, verify_instantiation
        torch = self.torch
        with torch.cuda.stream(self.stream):
            for _ in range(3): self.call(arm)
        self.stream.synchronize()
        owner = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(owner, stream=self.stream, capture_error_mode="thread_local"):
            self.outputs[arm] = self.call(arm)
        self.stream.synchronize()
        nodes = self.driver.kernel_nodes(owner.raw_cuda_graph())
        expected = {
            "deform": [DEFORM], "raster_full": [RASTER], "raster_tail": [RASTER],
            "prepass": [RASTER], "serial_full": [RASTER, DEFORM], "serial_tail": [RASTER, DEFORM],
            "both_v11": [MIXED["v11"]], "both_v13": [MIXED["v13"]],
            "complete_v13": [RASTER, MIXED["v13"]],
        }[arm]
        assert sorted(n["name"] for n in nodes) == sorted(expected), nodes
        handles = self.driver.kernel_handles(owner.raw_cuda_graph())
        attrs_by_name = {}
        for handle in handles:
            params = KernelParamsV1()
            self.driver.call("cuGraphKernelNodeGetParams", ctypes.c_void_p(handle), ctypes.byref(params))
            name = ctypes.c_char_p()
            self.driver.call("cuFuncGetName", ctypes.byref(name), params.func)
            getter = self.driver.library.cuFuncGetAttribute
            getter.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
            getter.restype = ctypes.c_int
            attrs = {}
            for key, code in (("registers",4),("static_shared_bytes",1),("local_bytes",3),("binary_version",6)):
                value = ctypes.c_int()
                assert getter(ctypes.byref(value), code, params.func) == 0
                attrs[key] = value.value
            attrs_by_name[name.value.decode()] = attrs
        for node in nodes:
            name, grid = node["name"], node["grid"]
            assert node["priority"] == 0 and grid[1:] == [1,1]
            if name == DEFORM:
                assert node["block"] == [128,1,1] and grid[0] == 6971
                assert attrs_by_name[name]["static_shared_bytes"] == 14976
            elif name == RASTER:
                assert node["block"] == [256,1,1]
                assert grid[0] == (1954 if arm in ("prepass","complete_v13") else
                                   3486 if arm in ("raster_tail","serial_tail") else 5440)
                assert attrs_by_name[name]["static_shared_bytes"] == 10240
            else:
                assert node["block"] == [384,1,1]
                assert grid[0] == (5440 if name == MIXED["v11"] else 3486)
            assert attrs_by_name[name]["local_bytes"] == 0, attrs_by_name[name]
            assert attrs_by_name[name]["binary_version"] == 87
        executable = self.driver.instantiate(owner.raw_cuda_graph())
        audit = verify_instantiation(self.driver, executable, owner.raw_cuda_graph())
        self.graphs[arm] = owner, executable
        self.audits[arm] = {**audit, "nodes":nodes, "compiled_resources":attrs_by_name}

    def replay(self, arm):
        self.driver.launch(self.graphs[arm][1], self.stream)

    def check(self, arm):
        torch, p = self.torch, self.source["prepared"]
        with torch.cuda.stream(self.stream):
            p.color.fill_(float("nan")); p.depth.fill_(float("nan"))
            for output in self.outputs[arm]: output.fill_(float("nan"))
            self.replay(arm)
        self.stream.synchronize()
        for actual, expected in zip(self.outputs[arm], self.reference["heads"]):
            assert torch.equal(actual, expected), "Deform mismatch: "+arm
        settings = self.source["rasterizer"].raster_settings
        width, height = int(settings.image_width), int(settings.image_height)
        y, x = torch.arange(height, device="cuda").view(-1,1), torch.arange(width,device="cuda").view(1,-1)
        tile = (y//16)*((width+15)//16)+x//16
        lo, hi = (0,0) if arm == "deform" else ((0,1954) if arm == "prepass" else
                  ((1954,5440) if arm in ("raster_tail","serial_tail","both_v13") else (0,5440)))
        mask = (tile>=lo)&(tile<hi)
        for actual, expected in ((p.color, self.reference["color"]),
                                 (p.depth.reshape(1,height,width), self.reference["depth"].reshape(1,height,width))):
            assert torch.equal(actual[:,mask], expected[:,mask]), "Raster mismatch: "+arm
            assert torch.isnan(actual[:,~mask]).all(), "Raster outside range: "+arm

    def measure(self, arm, repeats, warmup):
        torch = self.torch
        pairs = [(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(repeats)]
        with torch.cuda.stream(self.stream):
            for a,b in pairs: a.record(self.stream); b.record(self.stream)
            for _ in range(warmup): self.replay(arm)
        self.stream.synchronize()
        with torch.cuda.stream(self.stream):
            for a,b in pairs:
                a.record(self.stream); self.replay(arm); b.record(self.stream)
        self.stream.synchronize()
        samples=[a.elapsed_time(b) for a,b in pairs]
        return {"mean_ms":statistics.mean(samples), "samples_ms":samples}

    def description(self):
        def sha(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return {"step":self.source["step"], "deform_rows":self.rows,
                "features_sha256":sha(self.source["arguments"]["deform_features"]),
                "raster_color_sha256":sha(self.reference["color"]),
                "deform_outputs_sha256":[sha(x) for x in self.reference["heads"]],
                "graph_audits":self.audits,
                "correctness":"all dedicated/serial/mixed enabled outputs bitwise identical; poisoned outputs and outside-range NaN checks"}


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-root",type=Path,required=True)
    parser.add_argument("--benchmark-output",type=Path,required=True)
    parser.add_argument("--library",type=Path,default=Path(__file__).with_name("libdeform_matched.so"))
    parser.add_argument("--rounds",type=int,default=6)
    parser.add_argument("--repeats",type=int,default=30)
    parser.add_argument("--bench-warmup",type=int,default=10)
    options,args=parser.parse_known_args()
    assert "TACKER_BENCHMARK_MIXED_MODE" not in os.environ
    root=options.runtime_root.resolve()
    sys.path[:0]=[str(root),str(root/"scripts")]
    from benchmark_mixed_modes_graph import Extractor, QUALIFICATION_SHA
    from run_profile_render_sealed import _read_stable_source
    record,raw=_read_stable_source(str(root/"profile_render.py"))
    namespace={"__name__":"profile_render_independent_matched_extractor","__file__":record["path"],"__package__":None,
               "_PROFILE_RENDER_BOOTSTRAP":{"protocol":1,"record":record,"bytes":raw,
                 "execution":{"compiled_sha256":record["sha256"],"compile_mode":"exec","dont_inherit":True}}}
    exec(compile(raw,record["path"],"exec",dont_inherit=True),namespace)
    activate,measure=namespace["_activate_runtime_imports"],namespace["_measure_trial"]
    boxes=[]
    def activation(*a,**k):
        activate(*a,**k); extractor=Extractor(namespace["torch"]); extractor.install(); boxes.append(extractor)
    def measured(*a,**k):
        boxes[0].active=True
        try:return measure(*a,**k)
        finally:boxes[0].active=False
    namespace["_activate_runtime_imports"],namespace["_measure_trial"]=activation,measured
    namespace["_cli_main"](args)
    sys.stdout=sys.__stdout__
    metadata=json.loads(Path(args[args.index("--metadata")+1]).read_text())
    assert metadata["passed"] and metadata["qualification_mode_executed"]
    assert metadata["selected_variant_id"]=="full_mlp_overlap_graph_pb0"
    assert metadata["qualification_profile_sha256"]==QUALIFICATION_SHA
    assert metadata["tacker_fallback_reason"] is None and metadata["post_run"]["byte_stability"]["verified"]
    assert boxes[0].calls==4 and len(boxes[0].cases)==2
    torch=namespace["torch"]; library=ctypes.CDLL(str(options.library.resolve()))
    report={"passed":False,"qualification_profile":metadata["qualification_profile"],"qualification_sha256":QUALIFICATION_SHA,
            "actual_variant":metadata["selected_variant_id"],"rounds":options.rounds,"repeats":options.repeats,
            "cases":[],"primary_profiler_attached":False,
            "same_device_code":"Deform calls exact deform_mlp_tensor_tail_gptb_device_v7; dedicated Raster and mixed call exact general_ptb_render_raster_optimized<NUM_CHANNELS,true>; no algorithm or precision changes",
            "scope":"frozen real inputs; hot single-stream Driver Graph flags=8 priority 0; complete_v13 is prepass then production Both with both inputs ready; prefix/prepare excluded"}
    with torch.no_grad():
        for index,source in enumerate(boxes[0].cases):
            case=Case(torch,source,library)
            result={"case":index,"workload":case.description(),"batches":[]}
            for round_index in range(options.rounds):
                for arm in (ARMS if round_index%2==0 else ARMS[::-1]):
                    result["batches"].append({"round":round_index,"arm":arm,**case.measure(arm,options.repeats,options.bench_warmup)})
                print("ROUND",index,round_index+1,flush=True)
            for arm in ARMS:case.check(arm)
            report["cases"].append(result)
            options.benchmark_output.write_text(json.dumps(report,indent=2)+"\n")
            for owner,exe in case.graphs.values():case.driver.destroy(exe); owner.reset()
        report["passed"]=True
    options.benchmark_output.write_text(json.dumps(report,indent=2)+"\n")
    print("BENCHMARK_DONE",options.benchmark_output,flush=True)


if __name__=="__main__":main()
