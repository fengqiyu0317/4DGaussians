"""B/C leaf and captured Graph correctness; no throughput/qualification claim."""
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from tacker_4dgs_head.hexplane import sample_plane, sample_planes, batch_resources
import test_raster_hexplane_mixed_cuda as raster_tests
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from sampling_prepass_trace import split_graph_kernels


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class BatchedSamplingTests(unittest.TestCase):
    @torch.no_grad()
    def test_planes_borders_empty_and_partial_point_tiles(self):
        helper = raster_tests.RasterHexPlaneMixedTests()
        for rows, count in itertools.product((0, 1, 255, 256, 257, 1025), (1, 6, 12)):
            with self.subTest(rows=rows, planes=count):
                coords, planes, axes = helper.bundle(rows, count)
                if rows >= 6:
                    coords[:6] = torch.tensor([
                        [-1.] * 4, [1.] * 4, [0.] * 4, [float("nan")] * 4,
                        [float("inf")] * 4, [float("-inf")] * 4], device="cuda")
                buffers = [torch.full((16 * rows + 8,), -12345., device="cuda") for _ in planes]
                outputs = [buffer[4:-4].view(16, rows).t() for buffer in buffers]
                for _ in range(2):
                    actual = sample_planes(coords, planes, axes, outputs=outputs)
                    helper.check_hex(actual, coords, planes, axes)
                    for value, plane, pair, buffer in zip(actual, planes, axes, buffers):
                        torch.testing.assert_close(value, sample_plane(coords, plane, *pair), rtol=0, atol=0)
                        self.assertTrue(bool((buffer[:4] == -12345).all()))
                        self.assertTrue(bool((buffer[-4:] == -12345).all()))
                    coords.mul_(0.7)
                allocated = sample_planes(coords, planes, axes)
                helper.check_hex(allocated, coords, planes, axes)

    @torch.no_grad()
    def test_reject_aliases_bad_output_layout_and_axis_domains(self):
        coords, planes, axes = raster_tests.RasterHexPlaneMixedTests().bundle(257, 2)
        output = torch.empty(16, 257, device="cuda").t()
        with self.assertRaisesRegex(RuntimeError, "alias"):
            sample_planes(coords, planes, axes, outputs=(output, output))
        with self.assertRaisesRegex(RuntimeError, "strides"):
            sample_planes(coords, planes[:1], axes[:1], outputs=(torch.empty(257, 16, device="cuda"),))
        for bad_axes in ([], [(0,)], [(0, 0)], [(0, 4)]):
            with self.assertRaises(RuntimeError):
                sample_planes(coords, planes[:1], bad_axes)


@unittest.skipUnless(torch.cuda.is_available() and raster_tests.GaussianRasterizer is not None,
                     "CUDA and rebuilt Raster extension required")
class BlockMixedTests(unittest.TestCase):
    scene = raster_tests.RasterHexPlaneMixedTests.scene
    bundle = raster_tests.RasterHexPlaneMixedTests.bundle
    check_hex = raster_tests.RasterHexPlaneMixedTests.check_hex
    check_raster = raster_tests.RasterHexPlaneMixedTests.check_raster
    setUp = raster_tests.RasterHexPlaneMixedTests.setUp
    tearDown = raster_tests.RasterHexPlaneMixedTests.tearDown

    def test_partial_ranges_and_unequal_task_domains(self):
        rasterizer, args = self.scene(use_sh=True, opacity=0.9)
        expected = rasterizer.prepare_for_deform_mlp_overlap(**args)
        rasterizer.render_prepared_raster_fast_range(expected, 0, 12)
        for rows, count, start, end in ((1, 1, 0, 12), (257, 12, 1, 11),
                                        (1025, 6, 5, 6), (111525, 12, 0, 12)):
            with self.subTest(rows=rows, count=count, start=start, end=end):
                actual = rasterizer.prepare_for_deform_mlp_overlap(**args)
                actual.color.fill_(-7)
                actual.depth.fill_(-9)
                coords, planes, axes = self.bundle(rows, count)
                outputs = rasterizer.render_prepared_with_hexplane_blocks(
                    actual, coords, planes, axes, raster_start=start, raster_end=end)
                self.check_hex(outputs, coords, planes, axes)
                tile = (torch.arange(35, device="cuda")[:, None] // 16 * 4 +
                        torch.arange(49, device="cuda")[None, :] // 16)
                untouched = (tile < start) | (tile >= end)
                self.assertTrue(bool((actual.color[:, untouched] == -7).all()))
                self.assertTrue(bool((actual.depth[:, untouched] == -9).all()))
                rasterizer.render_prepared_raster_fast_range(actual, 0, start)
                rasterizer.render_prepared_raster_fast_range(actual, end, 12)
                self.check_raster(actual, expected, 35, 49)

    def test_empty_domains_and_invisible_gaussians(self):
        for points, rows, start, end, visible in ((0, 257, 0, 12, True),
                                                 (600, 257, 5, 5, True),
                                                 (600, 0, 0, 12, True),
                                                 (0, 0, 0, 12, True),
                                                 (600, 257, 0, 12, False)):
            with self.subTest(points=points, rows=rows, visible=visible):
                rasterizer, args = self.scene(points=points, visible=visible)
                expected = rasterizer.prepare_for_deform_mlp_overlap(**args)
                rasterizer.render_prepared_raster_fast_range(expected, start, end)
                actual = rasterizer.prepare_for_deform_mlp_overlap(**args)
                coords, planes, axes = self.bundle(rows)
                outputs = rasterizer.render_prepared_with_hexplane_blocks(
                    actual, coords, planes, axes, raster_start=start, raster_end=end)
                self.check_hex(outputs, coords, planes, axes)
                if points:
                    self.check_raster(actual, expected, 35, 49)
                else:
                    # Empty scenes do not allocate a Raster ImageState.
                    torch.testing.assert_close(actual.color, expected.color, rtol=0, atol=0)
                    torch.testing.assert_close(actual.depth, expected.depth, rtol=0, atol=0)

    def test_graph_replays_updated_inputs_and_exact_kernel_counts(self):
        rasterizer, args = self.scene()
        coords, planes, axes = self.bundle(257)
        expected = rasterizer.prepare_for_deform_mlp_overlap(**args)
        rasterizer.render_prepared_raster_fast_range(expected, 0, 12)
        actual = rasterizer.prepare_for_deform_mlp_overlap(**args)
        outputs = tuple(torch.empty(16, 257, device="cuda").t() for _ in planes)
        streams = [torch.cuda.Stream() for _ in range(3)]
        control, sampling, raster = streams

        def launch(mode):
            if mode == "C":
                rasterizer.render_prepared_with_hexplane_blocks(
                    actual, coords, planes, axes, outputs=outputs)
            else:
                fork, sampled, rendered = events[mode]
                fork.record(control)
                sampling.wait_event(fork)
                raster.wait_event(fork)
                with torch.cuda.stream(sampling):
                    sample_planes(coords, planes, axes, outputs=outputs)
                    sampled.record(sampling)
                with torch.cuda.stream(raster):
                    rasterizer.render_prepared_raster_fast_range(actual, 0, 12)
                    rendered.record(raster)
                control.wait_event(sampled)
                control.wait_event(rendered)

        # Native coordinate preparation is outside capture so A contains only
        # its original 12 sampler kernels, with no gather/contiguous kernels.
        native_coords = [coords[:, a].contiguous() for a in axes]
        def native_launch():
            return [torch.nn.functional.grid_sample(
                p, c.reshape(1, 1, -1, 2), mode="bilinear", padding_mode="border",
                align_corners=True).reshape(16, -1).t() for p, c in zip(planes, native_coords)]

        native = {}
        events = {mode: [torch.cuda.Event() for _ in range(3)] for mode in ("A", "B")}
        # A's launch is captured separately below to keep coordinate gathers out.
        graphs = {}
        torch.cuda.synchronize()
        for mode in ("B", "C"):
            control.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(control):
                for _ in range(3):
                    launch(mode)
            control.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=control):
                launch(mode)
            graphs[mode] = graph
        with torch.cuda.stream(control):
            native["A"] = native_launch()
        control.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=control):
            fork, sampled, rendered = events["A"]
            fork.record(control)
            sampling.wait_event(fork)
            raster.wait_event(fork)
            with torch.cuda.stream(sampling):
                native["A"] = native_launch()
                sampled.record(sampling)
            with torch.cuda.stream(raster):
                rasterizer.render_prepared_raster_fast_range(actual, 0, 12)
                rendered.record(raster)
            control.wait_event(sampled)
            control.wait_event(rendered)
        graphs["A"] = graph
        for _ in range(3):
            coords.mul_(0.8)
            for plane in planes:
                plane.add_(0.03)
            for destination, pair in zip(native_coords, axes):
                destination.copy_(coords[:, pair])
            for mode in ("A", "B", "C"):
                values = native["A"] if mode == "A" else outputs
                for value in values:
                    value.fill_(float("nan"))
                actual.color.fill_(float("nan"))
                actual.depth.fill_(float("nan"))
                graphs[mode].replay()
                torch.cuda.synchronize()
                self.check_hex(values, coords, planes, axes)
                self.check_raster(actual, expected, 35, 49)

        with tempfile.TemporaryDirectory() as directory:
            for mode, expected_count in (("A", 13), ("B", 2), ("C", 1)):
                # Warm Kineto before the active interval. PyTorch may also
                # enqueue replay bookkeeping outside the captured Graph.
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
                ) as prof:
                    graphs[mode].replay()
                    torch.cuda.synchronize()
                    prof.step()
                    graphs[mode].replay()
                    torch.cuda.synchronize()
                    prof.step()
                trace_directory = Path(os.environ.get("TACKER_BC_TEST_TRACE_DIR", directory))
                trace_directory.mkdir(parents=True, exist_ok=True)
                path = trace_directory / (mode + ".json")
                prof.export_chrome_trace(str(path))
                kernels, outside = split_graph_kernels(json.loads(path.read_text()))
                self.assertEqual(len(kernels), expected_count, [row["name"] for row in kernels])
                print("GRAPH_KERNEL_AUDIT", mode, "body", len(kernels),
                      "outside_graph", len(outside), "total", len(kernels) + len(outside))
                if mode == "C":
                    self.assertEqual(kernels[0]["name"], "tacker_mix_raster_hexplane_blocks_v1")
        self.assertEqual(batch_resources()["threads"], 256)
        self.assertEqual(rasterizer.raster_hexplane_block_resources()["threads"], 256)


if __name__ == "__main__":
    unittest.main()
