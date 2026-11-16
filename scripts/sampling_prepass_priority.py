"""Opt-in node-priority replay for the deterministic sampling/prepass benchmark.

Not a general replacement for torch.cuda.CUDAGraph.replay (e.g. RNG handling).
Keep the PyTorch capture owner alive so its private allocator pool and tensors
outlive the separately instantiated CUDA executable. No CUDA calls at import.
"""
import ctypes


USE_NODE_PRIORITY = 8  # CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY, CUDA 12.6
PRIORITY_ATTRIBUTE = 8  # CU_LAUNCH_ATTRIBUTE_PRIORITY


def validate_stream_priorities(actual, device_range):
    # PyTorch's pooled range can be narrower than CUDA's (Orin: -3 vs -5).
    # Accept that documented mapping, but never silently accept priority 0.
    if (actual["control"] != 0 or actual["prepass"] != 0
            or not device_range["greatest"] <= actual["sampling"] < 0):
        raise RuntimeError("actual stream priorities do not implement sampling-high/prepass-default: "
                           + str(actual))
    return dict(actual)


class KernelParamsV1(ctypes.Structure):
    # The unversioned cuGraphKernelNodeGetParams entry point uses the v1 ABI.
    _fields_ = [("func", ctypes.c_void_p),
                ("grid_x", ctypes.c_uint), ("grid_y", ctypes.c_uint), ("grid_z", ctypes.c_uint),
                ("block_x", ctypes.c_uint), ("block_y", ctypes.c_uint), ("block_z", ctypes.c_uint),
                ("shared_bytes", ctypes.c_uint),
                ("kernel_params", ctypes.c_void_p), ("extra", ctypes.c_void_p)]


class AttributeValue(ctypes.Union):
    # CUlaunchAttributeValue has 64-byte padding and pointer alignment.
    _fields_ = [("priority", ctypes.c_int), ("padding", ctypes.c_byte * 64),
                ("alignment", ctypes.c_void_p)]


class UpdateInfo(ctypes.Structure):
    _fields_ = [("result", ctypes.c_int), ("error_node", ctypes.c_void_p),
                ("error_from_node", ctypes.c_void_p)]


def verify_instantiation(driver, executable, graph):
    flags = driver.flags(executable)
    audit = {"instantiate_flags": flags, "requested_instantiation_flags": USE_NODE_PRIORITY}
    if flags == USE_NODE_PRIORITY:
        audit["verification_method"] = "direct_flags_readback"
    elif flags == 0:
        # Orin CUDA 12.6 returns 0 for ALL queried flags, including AutoFree.
        # Do not simply waive verification: prove the documented flag-specific
        # update restriction on temporary clones, with a flags=0 control.
        evidence = driver.priority_update_control(graph)
        if (evidence.get("default_update_status"), evidence.get("default_update_result"),
                evidence.get("priority_update_status"), evidence.get("priority_update_result")) != (0, 0, 910, 8):
            raise RuntimeError("USE_NODE_PRIORITY behavioral verification failed: " + str(evidence))
        audit.update(verification_method="priority_update_constraint_with_default_control",
                     flags_readback_quirk=True, behavioral_control=evidence)
    else:
        raise RuntimeError("Graph executable did not enable USE_NODE_PRIORITY; flags=" + str(flags))
    return audit


def validate_nodes(nodes, mode, sampling_priority, prepass_priority, split_tile):
    expected = {"sampling": 0 if mode == "prepass" else 12,
                "prepass": 0 if mode == "sampling" else 1}
    counts = {key: 0 for key in expected}
    for node in nodes:
        name = node["name"]
        if "grid_sampler_2d_kernel" in name:
            kind, priority, grid = "sampling", sampling_priority, 436
        elif name == "tacker_render_raster_fast_range_v12":
            kind, priority, grid = "prepass", prepass_priority, split_tile
        else:
            raise RuntimeError("unexpected kernel in priority Graph: " + name)
        if node["priority"] != priority:
            raise RuntimeError("captured priority mismatch for " + kind)
        if node["grid"] != [grid, 1, 1] or node["block"] != [256, 1, 1]:
            raise RuntimeError("priority Graph changed launch geometry for " + kind)
        counts[kind] += 1
    if counts != expected:
        raise RuntimeError("priority Graph kernel counts mismatch: " + str(counts))
    return counts


class CudaDriver:
    def __init__(self):
        self.library = ctypes.CDLL("libcuda.so.1")
        pointer = ctypes.c_void_p
        signatures = {
            "cuCtxGetStreamPriorityRange": [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)],
            "cuStreamGetPriority": [pointer, ctypes.POINTER(ctypes.c_int)],
            "cuGraphGetNodes": [pointer, ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_size_t)],
            "cuGraphNodeGetType": [pointer, ctypes.POINTER(ctypes.c_int)],
            "cuGraphKernelNodeGetParams": [pointer, ctypes.POINTER(KernelParamsV1)],
            "cuFuncGetName": [ctypes.POINTER(ctypes.c_char_p), pointer],
            "cuGraphKernelNodeGetAttribute": [pointer, ctypes.c_int, ctypes.POINTER(AttributeValue)],
            "cuGraphInstantiateWithFlags": [ctypes.POINTER(pointer), pointer, ctypes.c_ulonglong],
            "cuGraphExecGetFlags": [pointer, ctypes.POINTER(ctypes.c_ulonglong)],
            "cuGraphClone": [ctypes.POINTER(pointer), pointer],
            "cuGraphDestroy": [pointer],
            "cuGraphKernelNodeSetAttribute": [pointer, ctypes.c_int, ctypes.POINTER(AttributeValue)],
            "cuGraphExecUpdate_v2": [pointer, pointer, ctypes.POINTER(UpdateInfo)],
            "cuGraphLaunch": [pointer, pointer],
            "cuGraphExecDestroy": [pointer],
        }
        for name, signature in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = signature, ctypes.c_int

    def call(self, name, *args):
        status = getattr(self.library, name)(*args)
        if status:
            raise RuntimeError("%s returned CUDA driver error %d" % (name, status))

    def priority_range(self):
        least, greatest = ctypes.c_int(), ctypes.c_int()
        self.call("cuCtxGetStreamPriorityRange", ctypes.byref(least), ctypes.byref(greatest))
        if greatest.value >= 0:
            raise RuntimeError("device has no priority above the default priority 0")
        return {"least": least.value, "greatest": greatest.value}

    def stream_priority(self, stream):
        value = ctypes.c_int()
        self.call("cuStreamGetPriority", ctypes.c_void_p(stream.cuda_stream), ctypes.byref(value))
        return value.value

    def kernel_handles(self, graph):
        handle = ctypes.c_void_p(graph)
        count = ctypes.c_size_t()
        self.call("cuGraphGetNodes", handle, None, ctypes.byref(count))
        handles = (ctypes.c_void_p * count.value)()
        self.call("cuGraphGetNodes", handle, handles, ctypes.byref(count))
        kernels = []
        for node in handles[:count.value]:
            kind = ctypes.c_int()
            self.call("cuGraphNodeGetType", node, ctypes.byref(kind))
            if kind.value == 0:  # CU_GRAPH_NODE_TYPE_KERNEL
                kernels.append(node)
        return kernels

    def kernel_nodes(self, graph):
        nodes = []
        for node in self.kernel_handles(graph):
            params, attr, name = KernelParamsV1(), AttributeValue(), ctypes.c_char_p()
            self.call("cuGraphKernelNodeGetParams", node, ctypes.byref(params))
            self.call("cuFuncGetName", ctypes.byref(name), params.func)
            self.call("cuGraphKernelNodeGetAttribute", node, PRIORITY_ATTRIBUTE, ctypes.byref(attr))
            nodes.append({"name": name.value.decode(), "priority": attr.priority,
                          "grid": [params.grid_x, params.grid_y, params.grid_z],
                          "block": [params.block_x, params.block_y, params.block_z]})
        return nodes

    def instantiate(self, graph, flags=USE_NODE_PRIORITY):
        executable = ctypes.c_void_p()
        self.call("cuGraphInstantiateWithFlags", ctypes.byref(executable),
                  ctypes.c_void_p(graph), flags)
        return executable

    def priority_update_control(self, graph):
        clone = ctypes.c_void_p()
        self.call("cuGraphClone", ctypes.byref(clone), ctypes.c_void_p(graph))
        executables = []
        try:
            for flags in (0, USE_NODE_PRIORITY):
                executables.append(self.instantiate(clone.value, flags))
            node = self.kernel_handles(clone.value)[0]
            previous = AttributeValue()
            self.call("cuGraphKernelNodeGetAttribute", node, PRIORITY_ATTRIBUTE, ctypes.byref(previous))
            changed = AttributeValue(priority=-1 if previous.priority == 0 else 0)
            self.call("cuGraphKernelNodeSetAttribute", node, PRIORITY_ATTRIBUTE, ctypes.byref(changed))
            evidence = {}
            for key, executable in zip(("default", "priority"), executables):
                info = UpdateInfo()
                evidence[key + "_update_status"] = self.library.cuGraphExecUpdate_v2(
                    executable, clone, ctypes.byref(info))
                evidence[key + "_update_result"] = info.result
            return evidence
        finally:
            for executable in executables:
                self.destroy(executable)
            self.call("cuGraphDestroy", clone)

    def flags(self, executable):
        value = ctypes.c_ulonglong()
        self.call("cuGraphExecGetFlags", executable, ctypes.byref(value))
        return value.value

    def launch(self, executable, stream):
        self.call("cuGraphLaunch", executable, ctypes.c_void_p(stream.cuda_stream))

    def destroy(self, executable):
        self.call("cuGraphExecDestroy", executable)


class PriorityGraph:
    def __init__(self, torch, owner, driver, mode, priorities, split_tile):
        self.torch, self.owner, self.driver = torch, owner, driver
        self.executable = None
        nodes = driver.kernel_nodes(owner.raw_cuda_graph())
        counts = validate_nodes(nodes, mode, priorities["sampling"], priorities["prepass"], split_tile)
        executable = driver.instantiate(owner.raw_cuda_graph())
        try:
            flag_audit = verify_instantiation(driver, executable, owner.raw_cuda_graph())
        except BaseException:
            driver.destroy(executable)
            raise
        self.executable = executable
        self.audit = {"verified": True, **flag_audit, "use_node_priority": True,
                      "kernel_counts": counts, "nodes": nodes}

    def replay(self):
        if self.executable is None:
            raise RuntimeError("priority Graph has been closed")
        self.driver.launch(self.executable, self.torch.cuda.current_stream())

    def close(self):
        if self.executable is not None:
            self.torch.cuda.synchronize()
            self.driver.destroy(self.executable)
            self.executable = None
            self.owner.reset()
