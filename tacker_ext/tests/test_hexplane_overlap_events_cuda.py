"""Check timed external Graph records and unchanged FP32 sampling numerics."""
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from profile_hexplane_overlap_events import CapturedSamplingEvents


class GraphTimingTests(unittest.TestCase):
    def test_external_timed_events_work_and_do_not_change_sampling(self):
        for mode in ("last", "all"):
            stream = torch.cuda.Stream()
            grid = torch.rand((1, 16, 32, 32), device="cuda")
            coords = torch.rand((1, 1, 111525, 2), device="cuda") * 2 - 1
            static = coords.clone()
            torch.cuda.synchronize()
            bank = CapturedSamplingEvents(torch, stream, mode)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    F.grid_sample(grid, static, align_corners=True, padding_mode="border")
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = [bank.sample(F.grid_sample, grid, static,
                                       align_corners=True, padding_mode="border")
                           for _ in range(12)]
                product = outputs[0]
                for output in outputs[1:]:
                    product = product * output
            stream.synchronize()
            self.assertEqual(bank.capture_count, 12)
            for replay in range(3):
                changed = (coords + replay * 0.125).clamp(-1, 1)
                torch.cuda.synchronize()
                anchor = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    static.copy_(changed)
                    anchor.record(stream)
                    graph.replay()
                stream.synchronize()
                reference = F.grid_sample(grid, changed, align_corners=True,
                                          padding_mode="border")
                expected = reference
                for _ in range(11):
                    expected = expected * reference
                self.assertTrue(torch.equal(product, expected))
                intervals = []
                for plane in bank.plane_ids:
                    start, end = bank.events[plane]
                    self.assertTrue(start.query() and end.query())
                    self.assertGreater(start.elapsed_time(end), 0)
                    intervals.append((anchor.elapsed_time(start), anchor.elapsed_time(end)))
                self.assertTrue(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])))
                self.assertGreater(intervals[0][0], 0)


if __name__ == "__main__":
    unittest.main()
