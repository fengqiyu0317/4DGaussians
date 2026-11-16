"""Small Jetson compatibility probe; all artifacts must be on the data disk."""
import argparse
import ctypes
import json
from pathlib import Path

from sampling_prepass_priority import AttributeValue, CudaDriver, PRIORITY_ATTRIBUTE


class InstantiateParams(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_ulonglong), ("stream", ctypes.c_void_p),
                ("error_node", ctypes.c_void_p), ("result", ctypes.c_int)]


class UpdateInfo(ctypes.Structure):
    _fields_ = [("result", ctypes.c_int), ("error_node", ctypes.c_void_p),
                ("error_from_node", ctypes.c_void_p)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    a = torch.ones(16, device="cuda")  # Establish the current primary context.
    driver = CudaDriver()
    device_range = driver.priority_range()
    high = torch.cuda.Stream(priority=device_range["greatest"])
    normal, control = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    runtime = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so.12")
    ptr = ctypes.c_void_p
    driver.library.cuGraphKernelNodeSetAttribute.argtypes = [ptr, ctypes.c_int, ctypes.POINTER(AttributeValue)]
    driver.library.cuGraphExecUpdate_v2.argtypes = [ptr, ptr, ctypes.POINTER(UpdateInfo)]
    driver.library.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver_version = ctypes.c_int()
    driver.call("cuDriverGetVersion", ctypes.byref(driver_version))
    for library, prefix in ((driver.library, "cu"), (runtime, "cuda")):
        declarations = {
            "GraphInstantiateWithFlags": [ctypes.POINTER(ptr), ptr, ctypes.c_ulonglong],
            "GraphInstantiateWithParams": [ctypes.POINTER(ptr), ptr, ctypes.POINTER(InstantiateParams)],
            "GraphExecGetFlags": [ptr, ctypes.POINTER(ctypes.c_ulonglong)],
            "GraphExecDestroy": [ptr],
        }
        for suffix, signature in declarations.items():
            function = getattr(library, prefix + suffix)
            function.argtypes, function.restype = signature, ctypes.c_int
    report = {"driver_version": driver_version.value, "torch_version": torch.__version__,
              "torch_cuda_version": torch.version.cuda, "device_priority_range": device_range,
              "actual_priorities": {"sampling": driver.stream_priority(high),
                                    "prepass": driver.stream_priority(normal)}, "graphs": []}
    for heterogeneous in (False, True):
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=control, capture_error_mode="thread_local"):
            high.wait_stream(control)
            with torch.cuda.stream(high):
                b = a + 1
            if heterogeneous:
                normal.wait_stream(control)
                with torch.cuda.stream(normal):
                    c = a + 2
                control.wait_stream(normal)
            control.wait_stream(high)
        entry = {"heterogeneous": heterogeneous, "nodes": driver.kernel_nodes(graph.raw_cuda_graph()),
                 "attempts": []}
        node_count = ctypes.c_size_t()
        driver.call("cuGraphGetNodes", ptr(graph.raw_cuda_graph()), None, ctypes.byref(node_count))
        handles = (ptr * node_count.value)()
        driver.call("cuGraphGetNodes", ptr(graph.raw_cuda_graph()), handles, ctypes.byref(node_count))
        first_kernel = None
        for handle in handles:
            kind = ctypes.c_int()
            driver.call("cuGraphNodeGetType", handle, ctypes.byref(kind))
            if kind.value == 0:
                first_kernel = handle
                break
        original_priority = AttributeValue()
        driver.call("cuGraphKernelNodeGetAttribute", first_kernel, PRIORITY_ATTRIBUTE, ctypes.byref(original_priority))
        changed_priority = AttributeValue(priority=0 if original_priority.priority != 0 else -3)
        for library, prefix in ((driver.library, "cu"), (runtime, "cuda")):
            for suffix in ("WithFlags", "WithParams"):
                for flags in (0, 1, 8, 9):
                    executable = ptr()
                    params = InstantiateParams(flags=flags)
                    last = ctypes.c_ulonglong(flags) if suffix == "WithFlags" else ctypes.byref(params)
                    status = getattr(library, prefix + "GraphInstantiate" + suffix)(
                        ctypes.byref(executable), ptr(graph.raw_cuda_graph()), last)
                    attempt = {"entry": prefix + "GraphInstantiate" + suffix,
                               "requested_flags": flags, "status": status}
                    if status == 0:
                        try:
                            actual = ctypes.c_ulonglong(0xFFFF)
                            attempt["query_status"] = getattr(library, prefix + "GraphExecGetFlags")(
                                executable, ctypes.byref(actual))
                            attempt["actual_flags"] = actual.value
                            driver.launch(executable, control)
                            control.synchronize()
                            attempt["outputs_correct"] = bool(torch.equal(b, a + 1))
                            if heterogeneous:
                                attempt["outputs_correct"] &= bool(torch.equal(c, a + 2))
                            torch.cuda.synchronize()
                            # With USE_NODE_PRIORITY, changing a node's priority
                            # must make whole-graph update fail (CUDA 12.6 docs).
                            driver.call("cuGraphKernelNodeSetAttribute", first_kernel,
                                        PRIORITY_ATTRIBUTE, ctypes.byref(changed_priority))
                            try:
                                info = UpdateInfo()
                                attempt["priority_change_update_status"] = driver.library.cuGraphExecUpdate_v2(
                                    executable, ptr(graph.raw_cuda_graph()), ctypes.byref(info))
                                attempt["priority_change_update_result"] = info.result
                            finally:
                                driver.call("cuGraphKernelNodeSetAttribute", first_kernel,
                                            PRIORITY_ATTRIBUTE, ctypes.byref(original_priority))
                        finally:
                            driver.destroy(executable)
                    entry["attempts"].append(attempt)
        graph.reset()
        report["graphs"].append(entry)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
