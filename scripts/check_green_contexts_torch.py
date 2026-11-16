"""PyTorch/Graph control and optional Green Context Graph/SM-isolation probe.

Without --smid-library, only an ordinary stream Graph control is tested.
With it, Green Context ExternalStreams, Graphs and physical SM sets are tested.
Run only after an external GPU-idleness check on a shared server.
"""

import argparse
import ctypes
import gc
import json
import platform
import traceback
from pathlib import Path

import torch


def green_graph_check(library, smid_library_path):
    library.probe_green_pair_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.probe_green_pair_create.restype = ctypes.c_int
    library.probe_green_pair_destroy.argtypes = [ctypes.c_void_p]
    library.probe_green_pair_destroy.restype = ctypes.c_int
    library.probe_green_pair_stream.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.probe_green_pair_stream.restype = ctypes.c_size_t
    library.probe_green_pair_sm_count.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.probe_green_pair_sm_count.restype = ctypes.c_uint
    library.probe_green_last_stage.restype = ctypes.c_char_p
    kernel = ctypes.CDLL(str(Path(smid_library_path).resolve()))
    kernel.probe_launch_smid.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint]
    kernel.probe_launch_smid.restype = ctypes.c_int
    pair = ctypes.c_void_p()
    code = library.probe_green_pair_create(ctypes.byref(pair))
    if code:
        return {"passed": False, "create_code": code,
                "stopped_at": library.probe_green_last_stage().decode()}
    streams, graphs, inputs, outputs, ids = [], [], [], [], []
    report = {"passed": False, "partitions": []}
    stage = "setup"
    try:
        for index in range(2):
            handle = library.probe_green_pair_stream(pair, index)
            count = library.probe_green_pair_sm_count(pair, index)
            stream = torch.cuda.ExternalStream(handle, device=0)
            streams.append(stream)
            x = torch.zeros(65536, device="cuda", dtype=torch.float32)
            y = torch.empty_like(x)
            smids = torch.full((2048,), -1, device="cuda", dtype=torch.int32)
            inputs.append(x)
            outputs.append(y)
            ids.append(smids)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    torch.mul(x, 2.0, out=y)
                    torch.add(y, 1.0, out=y)
                launch_code = kernel.probe_launch_smid(handle, smids.data_ptr(), smids.numel())
                if launch_code:
                    raise RuntimeError(f"SM-ID warmup launch failed: {launch_code}")
            stream.synchronize()
            warmup_sms = sorted(set(smids.cpu().tolist()))
            report.setdefault("warmup_observed_sm_ids", []).append(warmup_sms)
            graph = torch.cuda.CUDAGraph()
            stage = f"partition_{index}.capture_begin"
            with torch.cuda.graph(graph, stream=stream), torch.no_grad():
                try:
                    stage = f"partition_{index}.capture_mul"
                    torch.mul(x, 2.0, out=y)
                    stage = f"partition_{index}.capture_add"
                    torch.add(y, 1.0, out=y)
                    stage = f"partition_{index}.capture_smid"
                    launch_code = kernel.probe_launch_smid(handle, smids.data_ptr(), smids.numel())
                    if launch_code:
                        raise RuntimeError(f"SM-ID capture launch failed: {launch_code}")
                    stage = f"partition_{index}.capture_end"
                except Exception as error:
                    report["first_capture_error"] = repr(error)
                    raise
            graphs.append(graph)
            stage = f"partition_{index}.replay"
            pointer = y.data_ptr()
            replays = []
            for value in [0.0, 1.0, -2.0, 13.0]:
                x.fill_(value)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph.replay()
                stream.synchronize()
                actual = y.cpu()
                correct = bool(torch.equal(actual, torch.full_like(actual, 2.0 * value + 1.0)))
                observed = sorted(set(smids.cpu().tolist()))
                row = {"input": value, "correct": correct,
                       "stable_output_pointer": y.data_ptr() == pointer,
                       "observed_sm_ids": observed}
                replays.append(row)
                if not correct or not row["stable_output_pointer"] or -1 in observed or len(observed) != count:
                    raise RuntimeError(f"Green Graph validation failed: {row}, expected {count} SMs")
            report["partitions"].append({"requested_sm_count": count,
                "observed_sm_ids": replays[-1]["observed_sm_ids"], "replays": replays})

        observed_sets = [set(row["observed_sm_ids"]) for row in report["partitions"]]
        report["physical_sm_sets_disjoint"] = observed_sets[0].isdisjoint(observed_sets[1])
        if not report["physical_sm_sets_disjoint"]:
            raise RuntimeError("Green Graphs used overlapping SM partitions")
        # Submit both Graphs before joining either. This is a functional test,
        # not a claim that both made simultaneous progress or improved FPS.
        for index in range(2):
            inputs[index].fill_(20.0 + index)
            streams[index].wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(streams[index]):
                graphs[index].replay()
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.current_stream().synchronize()
        for index in range(2):
            actual = outputs[index].cpu()
            assert torch.equal(actual, torch.full_like(actual, 41.0 + 2.0 * index))
            assert set(ids[index].cpu().tolist()) == observed_sets[index]
        report["dual_graph_submission_pass"] = True
        event = torch.cuda.Event()
        with torch.cuda.stream(streams[0]):
            inputs[1].fill_(7.0)
            event.record()
        streams[1].wait_event(event)
        with torch.cuda.stream(streams[1]):
            graphs[1].replay()
        streams[1].synchronize()
        actual = outputs[1].cpu()
        assert torch.equal(actual, torch.full_like(actual, 15.0))
        report["green_to_green_event_dependency_pass"] = True
        # Exercise captured allocations and a HexPlane-shaped PyTorch workload,
        # not just pointwise operations with preallocated out= buffers. This is
        # a representative smoke test, NOT the project's custom sampler path.
        planes = [torch.randn(1, 16, 32, 32, device="cuda") for _ in range(12)]
        grid = torch.empty(1, 4096, 1, 2, device="cuda").fill_(0.0)

        def sample_merge():
            features = []
            for scale in range(2):
                value = None
                for plane in planes[scale * 6:(scale + 1) * 6]:
                    sampled = torch.nn.functional.grid_sample(
                        plane, grid, mode="bilinear", padding_mode="zeros", align_corners=True
                    ).reshape(16, 4096).transpose(0, 1)
                    value = sampled if value is None else value * sampled
                features.append(value)
            return torch.cat(features, dim=1)

        streams[0].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[0]):
            for _ in range(3):
                sample_merge()
        streams[0].synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=streams[0]):
            merged = sample_merge()
        graphs.append(graph)
        pointer = merged.data_ptr()
        sample_checks = []
        for value in [0.0, 0.25, -0.5, 0.9]:
            grid.fill_(value)
            expected = sample_merge()
            streams[0].wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(streams[0]):
                graph.replay()
            streams[0].synchronize()
            actual_cpu, expected_cpu = merged.cpu(), expected.cpu()
            correct = bool(torch.equal(actual_cpu, expected_cpu))
            sample_checks.append({"grid_coordinate": value, "bitwise_equal": correct,
                "stable_output_pointer": merged.data_ptr() == pointer,
                "max_abs_difference": (actual_cpu - expected_cpu).abs().max().item()})
            if not correct or merged.data_ptr() != pointer:
                raise RuntimeError(f"Sampling/merge Graph failed: {sample_checks[-1]}")
        report["representative_sampling_merge"] = {
            "passed": True, "partition_sm_count": 4, "points": 4096,
            "planes": 12, "channels_per_scale": 16, "output_shape": list(merged.shape),
            "captured_allocations": True, "project_custom_sampler_tested": False,
            "replays": sample_checks,
        }
        report["passed"] = True
    except Exception as error:
        report["error"] = repr(error)
        report["stopped_at"] = stage
        report["traceback"] = traceback.format_exc()
    finally:
        for stream in streams:
            stream.synchronize()
        # Graph executables must be destroyed BEFORE their owning contexts.
        graph = None
        graphs.clear()
        event = None
        gc.collect()
        streams.clear()
        stream = None
        report["cleanup_code"] = library.probe_green_pair_destroy(pair)
        if report["cleanup_code"]:
            report["passed"] = False
    return report


def check(library_path, smid_library_path=None):
    torch.cuda.set_device(0)
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(0)
    report = {
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "device": properties.name,
        "sm_count": properties.multi_processor_count,
        "compute_capability": [properties.major, properties.minor],
        "ordinary_graph_pass": False,
        "green_graph_tested": False,
    }
    library = ctypes.CDLL(str(Path(library_path).resolve()))
    library.probe_sm_resource.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    library.probe_sm_resource.restype = ctypes.c_int
    count = ctypes.c_uint()
    result = library.probe_sm_resource(ctypes.byref(count))
    library.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    library.cuGetErrorName.restype = ctypes.c_int
    name = ctypes.c_char_p()
    library.cuGetErrorName(result, ctypes.byref(name))
    report["resource_query_after_torch_init"] = {
        "api": "cuDeviceGetDevResource.SM",
        "code": result,
        "name": name.value.decode() if name.value else "unknown",
        "sm_count": count.value,
    }

    static_input = torch.empty(65536, device="cuda", dtype=torch.float32)
    static_output = torch.empty_like(static_input)
    static_input.fill_(0.0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(3):
            torch.mul(static_input, 2.0, out=static_output)
            torch.add(static_output, 1.0, out=static_output)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), torch.no_grad():
        torch.mul(static_input, 2.0, out=static_output)
        torch.add(static_output, 1.0, out=static_output)
    pointer = static_output.data_ptr()
    checks = []
    for value in [0.0, 1.0, -2.0, 13.0]:
        static_input.fill_(value)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        actual = static_output.cpu()
        correct = bool(torch.equal(actual, torch.full_like(actual, 2.0 * value + 1.0)))
        stable_pointer = static_output.data_ptr() == pointer
        checks.append({"input": value, "correct": correct, "stable_output_pointer": stable_pointer})
        if not correct or not stable_pointer:
            raise RuntimeError(f"Ordinary Graph replay failed: {checks[-1]}")
    report["ordinary_graph_replays"] = checks
    report["ordinary_graph_pass"] = True
    report["green_graph_status"] = (
        "blocked_by_unsupported_resource_api" if result == 801
        else "not_tested_by_this_ordinary_stream_control"
    )
    if result == 0 and smid_library_path:
        report["green_graph_tested"] = True
        report["green_graph"] = green_graph_check(library, smid_library_path)
        report["green_graph_status"] = "passed" if report["green_graph"]["passed"] else "failed"
    torch.cuda.synchronize()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--smid-library")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = check(args.library, args.smid_library)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["ordinary_graph_pass"] or (
        result["green_graph_tested"] and not result["green_graph"]["passed"]
    ):
        raise SystemExit(2)
