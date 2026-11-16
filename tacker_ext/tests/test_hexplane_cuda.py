"""Numerical, buffer, and CUDA Graph checks for the specialized sampler."""

import itertools
import unittest

import torch
import torch.nn.functional as F

from tacker_4dgs_head.hexplane import sample_plane, sample_plane_out, resources


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class HexPlaneCudaTest(unittest.TestCase):
    @staticmethod
    def reference(coords, plane, axes):
        grid = coords[:, list(axes)].reshape(1, 1, -1, 2)
        return F.grid_sample(plane, grid, mode="bilinear", padding_mode="border",
                             align_corners=True).reshape(16, -1).t()

    @torch.no_grad()
    def test_axes_borders_partial_tiles_and_512_mapping(self):
        torch.manual_seed(20261007)
        for rows in (1, 255, 256, 257, 1025):
            coords = torch.rand(rows, 4, device="cuda") * 4 - 2
            if rows > 8:
                coords[:8] = torch.tensor(
                    [[-1] * 4, [1] * 4, [0] * 4, [-2] * 4, [2] * 4,
                     [float("nan")] * 4, [float("inf")] * 4,
                     [float("-inf")] * 4], device="cuda")
            for height, width in ((1, 1), (1, 64), (64, 1), (64, 64), (150, 128)):
                plane = torch.randn(1, 16, height, width, device="cuda")
                for axes in itertools.combinations(range(4), 2):
                    expected = self.reference(coords, plane, axes)
                    for probe in (False, True):
                        with self.subTest(rows=rows, shape=(height, width),
                                          axes=axes, probe=probe):
                            actual = sample_plane(coords, plane, *axes, probe_512=probe)
                            torch.testing.assert_close(actual, expected,
                                                       rtol=2e-6, atol=2e-6)

    @torch.no_grad()
    def test_out_guards_repeated_writes_and_empty(self):
        for rows in (0, 1, 257):
            coords = torch.zeros(rows, 4, device="cuda")
            plane = torch.randn(1, 16, 64, 64, device="cuda")
            buffer = torch.full((16 * rows + 8,), -12345., device="cuda")
            output = buffer[4:-4].view(16, rows).t()
            for probe in (False, True):
                result = sample_plane_out(coords, plane, output, 0, 3, probe_512=probe)
                self.assertEqual(result.data_ptr(), output.data_ptr())
                self.assertTrue((buffer[:4] == -12345).all().item())
                self.assertTrue((buffer[-4:] == -12345).all().item())
                if rows:
                    torch.testing.assert_close(result, self.reference(coords, plane, (0, 3)),
                                               rtol=2e-6, atol=2e-6)
                    coords.add_(0.1)

    @torch.no_grad()
    def test_graph_replay_uses_updated_coords_and_planes(self):
        coords = torch.rand(257, 4, device="cuda") * 2 - 1
        plane = torch.randn(1, 16, 150, 64, device="cuda")
        output = torch.empty(16, 257, device="cuda").t()
        for probe in (False, True):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    sample_plane_out(coords, plane, output, 0, 3, probe_512=probe)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                sample_plane_out(coords, plane, output, 0, 3, probe_512=probe)
            for _ in range(3):
                coords.mul_(0.7)
                plane.add_(0.03)
                graph.replay()
                torch.testing.assert_close(output, self.reference(coords, plane, (0, 3)),
                                           rtol=2e-6, atol=2e-6)

    def test_rejects_training_wrong_layout_and_wrong_axes(self):
        coords = torch.zeros(257, 4, device="cuda")
        plane = torch.zeros(1, 16, 64, 64, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            sample_plane(coords, plane, 0, 1)
        with torch.no_grad():
            for axes in ((0, 0), (-1, 1), (0, 4)):
                with self.assertRaisesRegex(RuntimeError, "axes"):
                    sample_plane(coords, plane, *axes)
            with self.assertRaisesRegex(RuntimeError, "coords"):
                sample_plane(coords[:, ::2], plane, 0, 1)
            with self.assertRaisesRegex(RuntimeError, "output"):
                sample_plane_out(coords, plane, torch.empty(257, 16, device="cuda"), 0, 1)

    def test_resource_query(self):
        info = resources()
        self.assertEqual(info["solo"]["threads"], 256)
        self.assertEqual(info["probe_512"]["threads"], 512)
        for item in info.values():
            self.assertGreater(item["registers_per_thread"], 0)
            self.assertEqual(item["static_shared_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
