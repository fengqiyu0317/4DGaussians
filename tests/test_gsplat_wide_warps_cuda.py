"""Check mixed warp splits against the 8:4 wide kernel."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class GsplatWideWarpsCudaTest(unittest.TestCase):
    def test_partial_tiles_multiple_batches_and_graph_replay(self):
        from gsplat_mixed_ext import WIDE_WARP_SPLITS, load
        from tests.test_gsplat_tensor_v7_cuda import GsplatTensorV7Test

        baseline = load("tensor_v7_ca_wide")
        with torch.no_grad():
            raster, mlp = GsplatTensorV7Test.inputs(self, n=300, rows=139)
            # Ensure a second batch is needed even for pixels near the tile center.
            raster[3].fill_(0.008)
            self.assertGreater(raster[5].numel(), 6 * 256)
            for variant, warp_split in WIDE_WARP_SPLITS.items():
                extension = load(variant)
                resources = extension.resources()
                self.assertEqual(resources[3], sum(warp_split) * 32)
                if variant.endswith("_3cta") and torch.cuda.get_device_capability() == (8, 7):
                    self.assertLessEqual(resources[0], 40)
                    self.assertEqual(resources[4], 3)
                if variant.endswith("_lean") and torch.cuda.get_device_capability() == (8, 7):
                    self.assertLessEqual(resources[0], 40)
                    self.assertEqual(resources[2], 0)
                    self.assertEqual(resources[4], 3)
                # Exercise candidate assignment and the two-worker grid stride.
                for blocks, paired in ((0, True), (0, False), (1, True)):
                    with self.subTest(variant=variant, blocks=blocks, paired=paired):
                        expected = baseline.forward(raster, mlp, 37, 29, blocks, 2, True, paired)
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            actual = extension.forward(raster, mlp, 37, 29, blocks, 2, True, paired)
                        with torch.cuda.stream(stream):
                            graph.replay()
                        stream.synchronize()
                        for a, b in zip((*actual[:8], actual[9]), (*expected[:8], expected[9])):
                            self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)))
                        if variant.endswith("_lean") and blocks == 0 and paired:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=stream):
                                lean = extension.forward(raster, mlp, 37, 29, blocks, 2, False, paired)
                            with torch.cuda.stream(stream):
                                graph.replay()
                            stream.synchronize()
                            for a, b in zip(lean[:8], expected[:8]):
                                self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)))


if __name__ == "__main__":
    unittest.main()
