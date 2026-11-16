"""Bitwise left-fold, layout, buffer guards, and Graph replay validation."""
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

from tacker_4dgs_head.hexplane import reduce_features, reduction_resources


def reference(samples):
    levels = []
    for begin in (0, 6):
        product = 1.0
        for sample in samples[begin:begin + 6]:
            product = product * sample
        levels.append(product)
    return torch.cat(levels, dim=-1).contiguous()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class HexPlaneReductionCudaTests(unittest.TestCase):
    @staticmethod
    def samples(rows):
        return tuple((torch.rand(16, rows, device="cuda") * 4 - 2).t()
                     for _ in range(12))

    @torch.no_grad()
    def test_partial_tiles_real_size_and_bitwise_results(self):
        torch.manual_seed(20261009)
        for rows in (0, 1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 111525):
            samples = self.samples(rows)
            expected = reference(samples)
            for threads in (128, 256):
                with self.subTest(rows=rows, threads=threads):
                    actual = reduce_features(samples, threads=threads)
                    self.assertTrue(actual.is_contiguous())
                    self.assertEqual(tuple(actual.shape), (rows, 32))
                    self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))

    @torch.no_grad()
    def test_signed_zero_subnormal_overflow_and_nan(self):
        samples = list(self.samples(33))
        special = torch.tensor([0., -0., 2. ** -140, -(2. ** -140),
                                1.e20, -1.e20, float("inf"), float("nan")], device="cuda")
        for sample in samples:
            sample[:8, 0] = special
        samples[0][:, 1] = 0.
        samples[1][:, 1] = float("inf")
        expected = reference(samples)
        for threads in (128, 256):
            actual = reduce_features(samples, threads=threads)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            keep = ~torch.isnan(expected)
            self.assertTrue(torch.equal(actual.view(torch.int32)[keep], expected.view(torch.int32)[keep]))

    @torch.no_grad()
    def test_single_point_contiguous_view_is_also_column_major(self):
        samples = tuple(sample.contiguous() for sample in self.samples(1))
        for threads in (128, 256):
            self.assertTrue(torch.equal(reduce_features(samples, threads=threads), reference(samples)))

    @torch.no_grad()
    def test_out_guards_repeated_writes(self):
        for rows in (0, 1, 33, 257):
            samples = self.samples(rows)
            buffer = torch.full((rows * 32 + 16,), -12345., device="cuda")
            output = buffer[8:-8].view(rows, 32)
            for threads in (128, 256):
                result = reduce_features(samples, threads=threads, output=output)
                self.assertEqual(result.data_ptr(), output.data_ptr())
                self.assertTrue(torch.equal(output, reference(samples)))
                self.assertTrue((buffer[:8] == -12345.).all().item())
                self.assertTrue((buffer[-8:] == -12345.).all().item())
                for sample in samples:
                    sample.mul_(0.9)

    @torch.no_grad()
    def test_graph_replay_on_nondefault_stream_with_changed_inputs(self):
        for threads in (128, 256):
            samples = self.samples(257)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    reduce_features(samples, threads=threads)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = reduce_features(samples, threads=threads)
            address = output.data_ptr()
            for _ in range(4):
                with torch.cuda.stream(stream):
                    for sample in samples:
                        sample.mul_(0.9).add_(0.01)
                    graph.replay()
                stream.synchronize()
                self.assertEqual(output.data_ptr(), address)
                self.assertTrue(torch.equal(output, reference(samples)))

    def test_rejects_training_threads_layout_dtype_shape_and_alias(self):
        samples = self.samples(33)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            reduce_features(samples)
        with torch.no_grad():
            for threads in (0, 64, 512):
                with self.assertRaisesRegex(RuntimeError, "threads"):
                    reduce_features(samples, threads=threads)
            with self.assertRaisesRegex(RuntimeError, "12 planes"):
                reduce_features(samples[:11])
            for wrong in (samples[0].contiguous(), samples[0].half(), samples[0][:32], samples[0].cpu()):
                with self.assertRaisesRegex(RuntimeError, "samples"):
                    reduce_features((wrong,) + samples[1:])
            with self.assertRaisesRegex(RuntimeError, "output"):
                reduce_features(samples, output=torch.empty(32, 33, device="cuda").t())
            storage = torch.empty(33 * 32, device="cuda")
            aliased = storage[:33 * 16].view(16, 33).t()
            with self.assertRaisesRegex(RuntimeError, "alias"):
                reduce_features((aliased,) + samples[1:], output=storage.view(33, 32))

    @torch.no_grad()
    def test_original_sampling_and_split_graph_integration(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scene.hexplane import HexPlaneField
        from gaussian_renderer.raster_hexplane_graph import RasterHexPlaneGraphSlot
        grid = HexPlaneField(1.6, {"grid_dimensions": 2, "input_coordinate_dim": 4,
                                 "output_coordinate_dim": 16, "resolution": [8, 8, 8, 4]}, [1, 2]).cuda()
        xyz = torch.rand(257, 3, device="cuda")
        timestamp = torch.rand(257, 1, device="cuda")
        with mock.patch.dict(os.environ, {"TACKER_HEXPLANE_FUSED_REDUCTION": "0", "TACKER_HEXPLANE_LOWREG": "0"}):
            expected = grid.forward_cuda_graph(xyz, timestamp)
        for threads in (128, 256):
            with mock.patch.dict(os.environ, {"TACKER_HEXPLANE_FUSED_REDUCTION": "1",
                                              "TACKER_HEXPLANE_REDUCTION_THREADS": str(threads), "TACKER_HEXPLANE_LOWREG": "0"}):
                actual = grid.forward_cuda_graph(xyz, timestamp)
                self.assertTrue(torch.equal(actual, expected))
                # Eager/serial reference is not switched to the new kernel.
                with mock.patch("tacker_4dgs_head.hexplane.reduce_features", side_effect=AssertionError("eager reference used fused reduction")):
                    grid(xyz, timestamp)
                deform, raster = torch.cuda.Stream(), torch.cuda.Stream()
                slot = RasterHexPlaneGraphSlot(types.SimpleNamespace(grid=grid), xyz, timestamp,
                                              deform, mode="separate", tasks_per_cta=3, mixed_planes=9)
                ready = torch.cuda.Event()
                with torch.cuda.stream(deform):
                    slot.enqueue(xyz, timestamp, deform)
                with torch.cuda.stream(raster):
                    slot.complete_sampling(None, None, 1954, raster, deform, ready)
                ready.synchronize()
                torch.testing.assert_close(slot.output, expected, rtol=2e-5, atol=3e-6)

    def test_resource_query(self):
        info = reduction_resources()
        print("HEXPLANE_REDUCTION_RESOURCES=" + str(info), flush=True)
        for threads in (128, 256):
            self.assertEqual(info[threads]["threads"], threads)
            self.assertEqual(info[threads]["static_shared_bytes"], 32 * 33 * 4)
            self.assertEqual(info[threads]["local_bytes_per_thread"], 0)


if __name__ == "__main__":
    unittest.main()
