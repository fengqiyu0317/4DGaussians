"""CPU checks for real-kernel interval accounting and Graph/prefill separation."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("kernel_activity_cpu", scripts / "profile_sampling_kernel_activity.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class KernelActivityTests(unittest.TestCase):
    def test_kernel_intersections_exclude_queue_gaps(self):
        aa = [{"start": 0, "end": 10}, {"start": 100, "end": 110}]
        self.assertEqual(probe.overlaps(aa, [{"start": 20, "end": 90}]), 0)
        self.assertEqual(probe.overlaps(aa, [{"start": 5, "end": 105}]), 10 / 1e6)

    def make_csv(self, path, omit_plane=False, wrong_graph=False):
        rows = []
        def add(start, end, name, stream=17, node=0):
            rows.append([start, end, stream, node, 1, 256, 48, name])
        for i in range(12):
            add(1 + i * 3, 2 + i * 3, "grid_sampler_2d_kernel")
        for i in range(11 if omit_plane else 12):
            add(100 + i * 20, 110 + i * 20, "grid_sampler_2d_kernel",
                node=0 if wrong_graph else i + 1)
        add(90, 95, "preprocessCUDA", stream=13)
        add(106, 109, "DeviceScan", stream=13)
        add(230, 235, "identifyTileRanges", stream=13)
        add(250, 280, "tacker_render_raster_fast_range_v12", stream=13)
        add(1000, 1005, "preprocessCUDA", stream=13)
        add(1006, 1010, "identifyTileRanges", stream=13)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("start_ns", "end_ns", "stream_id", "graph_node_id",
                             "grid_x", "block_x", "registers", "name"))
            writer.writerows(rows)

    def test_prefill_is_excluded_and_actual_kernel_durations_are_summed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.make_csv(path)
            rows = probe.analyze(path, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sampling_ms"], 120 / 1e6)
        self.assertEqual(rows[0]["sampling_prepare_kernel_overlap_ms"], 3 / 1e6)
        self.assertEqual(rows[0]["sampling_prepass_kernel_overlap_ms"], 10 / 1e6)
        self.assertGreater(rows[0]["sampling_end_relative_to_prepass_start_ms"], 0)

    def test_missing_sampling_or_non_graph_record_fails(self):
        for kwargs in ({"omit_plane": True}, {"wrong_graph": True}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.csv"
                self.make_csv(path, **kwargs)
                with self.assertRaises(AssertionError):
                    probe.analyze(path, 2)


if __name__ == "__main__":
    unittest.main()
