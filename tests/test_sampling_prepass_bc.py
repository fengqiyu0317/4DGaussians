import importlib.util
from pathlib import Path
import sys
import unittest


scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("sampling_prepass_bc", scripts / "benchmark_sampling_prepass_bc.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
from sampling_prepass_trace import split_graph_kernels


class SummaryTests(unittest.TestCase):
    def test_driver_graph_checks_exact_counts_geometry_and_default_priority(self):
        def node(name, grid, priority=0):
            return {"name": name, "priority": priority, "grid": [grid, 1, 1], "block": [256, 1, 1]}
        native = [node("grid_sampler_2d_kernel_float", 436) for _ in range(12)]
        prepass = node("tacker_render_raster_fast_range_v12", 1954)
        batch = node("hexplane_sample_planes_batched_v1", 5232)
        mixed = node("tacker_mix_raster_hexplane_blocks_v1", 7186)
        for arm, nodes in (("A", native + [prepass]), ("B", [batch, prepass]), ("C", [mixed])):
            counts = bench.validate_abc_nodes(nodes, arm, "both", 111525, 1954)
            self.assertEqual(sum(counts.values()), {"A": 13, "B": 2, "C": 1}[arm])
        for nodes in ([dict(mixed, priority=-1)], [dict(mixed, grid=[7185, 1, 1])],
                      [mixed, mixed], [node("memcpy32_post", 1)], [batch, prepass]):
            with self.assertRaises(RuntimeError):
                bench.validate_abc_nodes(nodes, "C", "both", 111525, 1954)

    def test_separates_sampling_changes_from_mixed_kernel_changes(self):
        times = {"A": 6., "B": 5.5, "C": 5.6}
        records = [{"arm": arm, "mean_ms": times[arm]} for arm in "ABCCBA"]
        result = bench.summarize_abc(records)["median"]
        self.assertLess(result["B_over_A"], 1)
        self.assertGreater(result["C_over_B"], 1)
        self.assertLess(result["C_over_A"], 1)
        self.assertEqual(result["C_ms"], 5.6)

    def test_reject_partial_reordered_or_nonpositive_results(self):
        for order in ("", "ABC", "ABCABC", "ABBCBA"):
            with self.subTest(order=order), self.assertRaises(ValueError):
                bench.summarize_abc([{"arm": arm, "mean_ms": 1} for arm in order])
        with self.assertRaises(ValueError):
            bench.summarize_abc([{"arm": arm, "mean_ms": 0} for arm in "ABCCBA"])

    def test_trace_accounts_for_runtime_kernels_without_weakening_graph_audit(self):
        trace = {"traceEvents": [
            {"cat": "cuda_runtime", "name": "cudaGraphLaunch", "args": {"correlation": 53}},
            {"cat": "kernel", "name": "runtime_fill", "args": {"correlation": 44}},
            {"cat": "kernel", "name": "mixed", "args": {"correlation": 53}},
            {"cat": "kernel", "name": "unexpected_copy", "args": {"correlation": 53}},
        ]}
        body, outside = split_graph_kernels(trace)
        self.assertEqual([row["name"] for row in body], ["mixed", "unexpected_copy"])
        self.assertEqual([row["name"] for row in outside], ["runtime_fill"])
        with self.assertRaises(ValueError):
            split_graph_kernels({"traceEvents": trace["traceEvents"][1:]})
        trace['traceEvents'][0].update(cat='cuda_driver', name='cuGraphLaunch')
        driver_body, driver_outside = split_graph_kernels(trace)
        self.assertEqual(driver_body, body)
        self.assertEqual(driver_outside, outside)

    def test_driver_trace_without_api_marker_requires_independent_verified_nodes(self):
        trace = {'traceEvents': [{'cat': 'kernel', 'name': 'mixed', 'args': {'correlation': 19}}]}
        audit = {'verified': True, 'use_node_priority': True, 'requested_instantiation_flags': 8,
                 'kernel_counts': {'mixed': 1}}
        self.assertEqual(len(split_graph_kernels(trace, driver_audit=audit)[0]), 1)
        for bad in (dict(audit, verified=False), dict(audit, kernel_counts={'mixed': 2})):
            with self.assertRaises(ValueError):
                split_graph_kernels(trace, driver_audit=bad)
        trace['traceEvents'].append({'cat': 'kernel', 'name': 'extra', 'args': {'correlation': 20}})
        with self.assertRaises(ValueError):
            split_graph_kernels(trace, driver_audit=dict(audit, kernel_counts={'mixed': 2}))


if __name__ == "__main__":
    unittest.main()
