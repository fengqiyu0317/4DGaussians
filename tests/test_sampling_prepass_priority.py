import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


path = Path(__file__).resolve().parents[1] / "scripts/sampling_prepass_priority.py"
spec = importlib.util.spec_from_file_location("sampling_prepass_priority", path)
priority = importlib.util.module_from_spec(spec)
spec.loader.exec_module(priority)


def nodes(mode="both", sampling_priority=-1):
    result = []
    if mode != "prepass":
        result.extend({"name": "grid_sampler_2d_kernel_fp32", "priority": sampling_priority,
                       "grid": [436, 1, 1], "block": [256, 1, 1]} for _ in range(12))
    if mode != "sampling":
        result.append({"name": "tacker_render_raster_fast_range_v12", "priority": 0,
                       "grid": [1954, 1, 1], "block": [256, 1, 1]})
    return result


class FakeDriver:
    def __init__(self, flags=priority.USE_NODE_PRIORITY, behavioral_control=None):
        self.value = flags
        self.behavioral_control = behavioral_control
        self.destroyed = []
        self.launched = []

    def kernel_nodes(self, handle):
        return nodes()

    def instantiate(self, handle):
        return 123

    def flags(self, handle):
        return self.value

    def destroy(self, handle):
        self.destroyed.append(handle)

    def launch(self, handle, stream):
        self.launched.append((handle, stream))

    def priority_update_control(self, handle):
        if self.behavioral_control is None:
            raise RuntimeError("USE_NODE_PRIORITY verification unavailable")
        return self.behavioral_control


class PriorityTests(unittest.TestCase):
    def test_pytorch_priority_pool_can_clamp_cuda_range(self):
        actual = {"control": 0, "sampling": -3, "prepass": 0}
        self.assertEqual(priority.validate_stream_priorities(actual, {"least": 0, "greatest": -5}), actual)

    def test_priority_mapping_must_not_silently_become_default(self):
        for actual in ({"control": 0, "sampling": 0, "prepass": 0},
                       {"control": 0, "sampling": -3, "prepass": -3},
                       {"control": 0, "sampling": -6, "prepass": 0}):
            with self.assertRaisesRegex(RuntimeError, "actual stream priorities"):
                priority.validate_stream_priorities(actual, {"least": 0, "greatest": -5})

    def test_abi_layout(self):
        self.assertEqual(ctypes.sizeof(priority.AttributeValue), 64)
        self.assertEqual(ctypes.sizeof(priority.KernelParamsV1), 56)
        self.assertEqual(priority.KernelParamsV1.kernel_params.offset, 40)

    def test_all_five_graphs_are_admitted(self):
        for mode in ("sampling", "prepass", "serial", "both", "both_reverse"):
            result = priority.validate_nodes(nodes(mode), mode, -1, 0, 1954)
            self.assertEqual(result["sampling"], 0 if mode == "prepass" else 12)
            self.assertEqual(result["prepass"], 0 if mode == "sampling" else 1)

    def test_stream_priority_alone_is_not_enough(self):
        with self.assertRaisesRegex(RuntimeError, "priority mismatch"):
            priority.validate_nodes(nodes(sampling_priority=0), "both", -1, 0, 1954)

    def test_reject_extra_kernel_or_changed_geometry(self):
        bad = nodes() + [dict(nodes()[0], name="memcpy32_post")]
        with self.assertRaisesRegex(RuntimeError, "unexpected kernel"):
            priority.validate_nodes(bad, "both", -1, 0, 1954)
        bad = nodes()
        bad[0] = dict(bad[0], block=[128, 1, 1])
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            priority.validate_nodes(bad, "both", -1, 0, 1954)
        with self.assertRaisesRegex(RuntimeError, "counts"):
            priority.validate_nodes(nodes()[:-1], "both", -1, 0, 1954)

    def make_graph(self, driver):
        self.events = []
        owner = SimpleNamespace(raw_cuda_graph=lambda: 99,
                                reset=lambda: self.events.append("reset_owner"))
        torch = SimpleNamespace(cuda=SimpleNamespace(
            current_stream=lambda: "current", synchronize=lambda: self.events.append("synchronize")))
        return priority.PriorityGraph(torch, owner, driver, "both", {"sampling": -1, "prepass": 0}, 1954)

    def test_executable_flags_verified_and_owner_retained(self):
        driver = FakeDriver()
        graph = self.make_graph(driver)
        self.assertIsNotNone(graph.owner)
        self.assertEqual(graph.audit["instantiate_flags"], priority.USE_NODE_PRIORITY)
        graph.replay()
        self.assertEqual(driver.launched, [(123, "current")])
        graph.close()
        graph.close()
        self.assertEqual(driver.destroyed, [123])
        self.assertEqual(self.events, ["synchronize", "reset_owner"])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            graph.replay()

    def test_bad_flags_destroy_executable_and_fail_closed(self):
        driver = FakeDriver(flags=0)
        with self.assertRaisesRegex(RuntimeError, "USE_NODE_PRIORITY"):
            self.make_graph(driver)
        self.assertEqual(driver.destroyed, [123])

    def test_orin_zero_flags_requires_independent_positive_and_negative_controls(self):
        evidence = {"default_update_status": 0, "default_update_result": 0,
                    "priority_update_status": 910, "priority_update_result": 8}
        graph = self.make_graph(FakeDriver(flags=0, behavioral_control=evidence))
        self.assertTrue(graph.audit["verified"])
        self.assertEqual(graph.audit["instantiate_flags"], 0)
        self.assertEqual(graph.audit["requested_instantiation_flags"], 8)
        self.assertTrue(graph.audit["flags_readback_quirk"])
        graph.close()

    def test_zero_flags_with_no_differential_behavior_fails_closed(self):
        evidence = {"default_update_status": 0, "default_update_result": 0,
                    "priority_update_status": 0, "priority_update_result": 0}
        driver = FakeDriver(flags=0, behavioral_control=evidence)
        with self.assertRaisesRegex(RuntimeError, "behavioral verification failed"):
            self.make_graph(driver)
        self.assertEqual(driver.destroyed, [123])


if __name__ == "__main__":
    unittest.main()
