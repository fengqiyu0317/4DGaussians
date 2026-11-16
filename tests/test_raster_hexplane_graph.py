"""CPU checks for disjoint early/mixed sampling and submission dependencies."""
import contextlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


def load_slot_module():
    fake_torch = types.SimpleNamespace(float32="float32")
    path = Path(__file__).resolve().parents[1] / "gaussian_renderer/raster_hexplane_graph.py"
    spec = importlib.util.spec_from_file_location("raster_hexplane_cpu_contract", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        spec.loader.exec_module(module)
    return module, fake_torch


class PartialSamplingTests(unittest.TestCase):
    def setUp(self):
        self.module, self.torch = load_slot_module()
        self.log = []
        self.deform = types.SimpleNamespace(cuda_stream=11, name="deform")
        self.raster = types.SimpleNamespace(
            cuda_stream=10, name="raster",
            wait_event=lambda event: self.log.append(("raster_wait", event.name)))
        self.torch.cuda = types.SimpleNamespace(
            current_stream=lambda device: self.deform,
            stream=lambda stream: contextlib.nullcontext(stream))

    def event(self, name):
        return types.SimpleNamespace(name=name, record=lambda stream:
                                     self.log.append(("record", name, stream.name)))

    def graph(self, name):
        return types.SimpleNamespace(replay=lambda: self.log.append(("replay", name)))

    def tensor(self, name, shape=None):
        return types.SimpleNamespace(
            name=name, shape=shape, device="cuda", dtype="float32",
            copy_=lambda value, non_blocking: self.log.append(("copy", name)),
            record_stream=lambda stream: self.log.append(("retain", name)))

    def slot(self, mode="fused", mixed_planes=3):
        slot = object.__new__(self.module.RasterHexPlaneGraphSlot)
        slot.mode = mode
        slot.tasks_per_cta = 3
        slot._device = "cuda"
        slot._shape = (111525, 3)
        slot.early_plane_ids, slot.mixed_plane_ids = self.module.sampling_plane_split(mixed_planes)
        slot.xyz = self.tensor("xyz", slot._shape)
        slot.timestamp = self.tensor("time", (111525, 1))
        slot.coords = self.tensor("coords")
        slot.planes = tuple(self.tensor("plane%d" % i) for i in range(12))
        slot.samples = tuple(self.tensor("sample%d" % i) for i in range(12))
        slot.axes = tuple((i % 3, 3) for i in range(12))
        slot.hex_ctas = (mixed_planes * 436 + 2) // 3
        slot.coords_ready = self.event("coords")
        slot.sampling_done = self.event("sampling_done")
        slot.prepare_ready = self.event("prepare_ready")
        slot.coordinate_graph = self.graph("coordinate")
        slot.early_sampling_graph = self.graph("early") if slot.early_plane_ids else None
        slot.sampling_graph = self.graph("late_separate") if mode == "separate" else None
        slot.reduction_graph = self.graph("reduction")
        slot.output = object()
        return slot

    def test_plane_domains_are_disjoint_and_cover_all_tasks(self):
        for count in range(1, 13):
            early, late = self.module.sampling_plane_split(count)
            self.assertFalse(set(early) & set(late))
            self.assertEqual(sorted(early + late), list(range(12)))
            self.assertEqual(len(late), count)
            self.assertEqual((len(early) + len(late)) * 436, 5232)
        for count in (0, 13, -1, True, 1.5, "6"):
            with self.assertRaises(ValueError):
                self.module.sampling_plane_split(count)

    def test_early_sampling_is_queued_before_raster_dependency(self):
        slot = self.slot()
        output = slot.enqueue(self.tensor("input_xyz", slot._shape),
                              self.tensor("input_time", (111525, 1)), self.deform)
        self.assertIs(output, slot.output)
        self.assertLess(self.log.index(("record", "coords", "deform")),
                        self.log.index(("replay", "early")))
        self.assertNotIn(("record", "prepare_ready", "raster"), self.log)

    def test_mixed_writes_only_late_planes_and_reduction_joins_both(self):
        slot = self.slot()
        slot.enqueue(self.tensor("input_xyz", slot._shape),
                     self.tensor("input_time", (111525, 1)), self.deform)
        def launch(prepared, coords, planes, axes, **kwargs):
            self.log.append(("mixed", len(planes)))
            self.assertEqual(tuple(t.name for t in planes), ("plane9", "plane10", "plane11"))
            self.assertEqual(tuple(t.name for t in kwargs["outputs"]), ("sample9", "sample10", "sample11"))
            self.assertEqual(kwargs["raster_end"], 436)
        context = types.SimpleNamespace(rasterizer=types.SimpleNamespace(
            render_prepared_with_hexplane_sampling=launch))
        self.deform.wait_event = lambda event: self.log.append(("deform_wait", event.name))
        paired = slot.complete_sampling(context, object(), 1954, self.raster,
                                        self.deform, self.event("prefix_ready"))
        self.assertEqual(paired, 436)
        self.assertLess(self.log.index(("replay", "early")), self.log.index(("mixed", 3)))
        self.assertLess(self.log.index(("deform_wait", "sampling_done")), self.log.index(("replay", "reduction")))
        self.assertLess(self.log.index(("replay", "reduction")), self.log.index(("record", "prefix_ready", "deform")))
        self.assertEqual([x for x in self.log if x[0] == "raster_wait"], [("raster_wait", "coords")])

    def test_separate_control_preserves_early_work_and_only_defers_late(self):
        slot = self.slot(mode="separate", mixed_planes=6)
        slot.enqueue(self.tensor("input_xyz", slot._shape),
                     self.tensor("input_time", (111525, 1)), self.deform)
        self.deform.wait_event = lambda event: self.log.append(("deform_wait", event.name))
        paired = slot.complete_sampling(None, None, 1954, self.raster,
                                        self.deform, self.event("prefix_ready"))
        self.assertEqual(paired, 0)
        self.assertLess(self.log.index(("replay", "early")), self.log.index(("record", "prepare_ready", "raster")))
        self.assertLess(self.log.index(("deform_wait", "prepare_ready")), self.log.index(("replay", "late_separate")))
        self.assertLess(self.log.index(("replay", "late_separate")), self.log.index(("replay", "reduction")))

    def test_all_mixed_control_does_not_replay_an_early_graph(self):
        slot = self.slot(mixed_planes=12)
        slot.enqueue(self.tensor("input_xyz", slot._shape),
                     self.tensor("input_time", (111525, 1)), self.deform)
        self.assertNotIn(("replay", "early"), self.log)

    def test_batched_sampling_starts_without_waiting_for_raster_prepare(self):
        slot = self.slot(mode="batched", mixed_planes=12)
        slot.sampling_graph = self.graph("batch")
        slot.enqueue(self.tensor("input_xyz", slot._shape),
                     self.tensor("input_time", (111525, 1)), self.deform)
        self.assertEqual(self.log[-2:], [("replay", "batch"), ("replay", "reduction")])
        paired = slot.complete_sampling(None, None, 1954, self.raster,
                                        self.deform, self.event("prefix_ready"))
        self.assertEqual(paired, 0)
        self.assertFalse(any(x[0].endswith("wait") for x in self.log))
        self.assertEqual(self.log.count(("replay", "reduction")), 1)
        self.assertEqual(self.log[-1], ("record", "prefix_ready", "deform"))

    def test_block_fused_renders_complete_prepass_and_joins_before_reduction(self):
        slot = self.slot(mode="block_fused", mixed_planes=12)
        self.deform.wait_event = lambda event: self.log.append(("deform_wait", event.name))
        def launch(prepared, coords, planes, axes, **kwargs):
            self.assertEqual(len(planes), 12)
            self.assertEqual(kwargs["raster_end"], 1954)
            self.log.append(("block_fused",))
        context = types.SimpleNamespace(rasterizer=types.SimpleNamespace(
            render_prepared_with_hexplane_blocks=launch))
        paired = slot.complete_sampling(context, object(), 1954, self.raster,
                                        self.deform, self.event("prefix_ready"))
        self.assertEqual(paired, 1954)
        self.assertLess(self.log.index(("block_fused",)),
                        self.log.index(("deform_wait", "sampling_done")))
        self.assertLess(self.log.index(("deform_wait", "sampling_done")),
                        self.log.index(("replay", "reduction")))


if __name__ == "__main__":
    unittest.main()
