"""Separate captured Graph kernels from runtime replay bookkeeping by CUPTI ID."""


def split_graph_kernels(trace, driver_audit=None):
    events = trace["traceEvents"]
    # Driver replay (v13 flags=8) has no cudaGraphLaunch runtime call.
    # Prefer its direct cuGraphLaunch record when present.
    driver = [row for row in events if row.get("cat") == "cuda_driver"
              and row.get("name", "").startswith("cuGraphLaunch")]
    launches = driver or [row for row in events if row.get("cat") == "cuda_runtime"
                         and row.get("name", "").startswith("cudaGraphLaunch")]
    if not launches and driver_audit is not None:
        # Jetson Kineto 2.8 omits the direct Driver API callback, but retains
        # its kernel correlation ID. Admit only one positive correlation group
        # with exactly the independently verified Driver Graph node count.
        kernels = [row for row in events if row.get('cat') == 'kernel']
        correlations = {row.get('args', {}).get('correlation') for row in kernels}
        if not (driver_audit.get('verified') and driver_audit.get('use_node_priority')
                and driver_audit.get('requested_instantiation_flags') == 8
                and len(correlations) == 1 and all(isinstance(value, int) and value > 0 for value in correlations)
                and len(kernels) == sum(driver_audit.get('kernel_counts', {}).values()) > 0):
            raise ValueError('uncorrelated Driver trace or unverified Graph node audit')
        return kernels, []
    if len(launches) != 1 or "correlation" not in launches[0].get("args", {}):
        raise ValueError("require exactly one correlated cudaGraphLaunch in the active trace")
    correlation = launches[0]["args"]["correlation"]
    kernels = [row for row in events if row.get("cat") == "kernel"]
    body = [row for row in kernels if row.get("args", {}).get("correlation") == correlation]
    outside = [row for row in kernels if row.get("args", {}).get("correlation") != correlation]
    if not body:
        raise ValueError("trace does not correlate any kernels with the Graph launch")
    return body, outside
