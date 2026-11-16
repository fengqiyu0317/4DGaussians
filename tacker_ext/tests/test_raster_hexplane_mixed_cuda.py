"""Leaf correctness only: not a v11/v12 qualification or FPS benchmark.

Run with the freshly built Raster extension first on PYTHONPATH. Warmup and
capture cover only the new prepared leaf, not the full rendering pipeline.
"""
import itertools
import json
import unittest

import torch
import torch.nn.functional as F

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer, _C,
    )
except ImportError:
    GaussianRasterizer = None


@unittest.skipUnless(torch.cuda.is_available() and GaussianRasterizer is not None,
                     "requires CUDA and rebuilt Raster extension")
class RasterHexPlaneMixedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261008)
        self.no_grad = torch.no_grad()
        self.no_grad.__enter__()

    def tearDown(self):
        self.no_grad.__exit__(None, None, None)

    def scene(self, *, height=35, width=49, points=600, use_sh=False,
              opacity=0.01, visible=True):
        settings = GaussianRasterizationSettings(
            image_height=height, image_width=width, tanfovx=1.0, tanfovy=1.0,
            bg=torch.tensor([0.1, 0.2, 0.3], device="cuda"), scale_modifier=1.0,
            viewmatrix=torch.eye(4, device="cuda"),
            projmatrix=torch.eye(4, device="cuda"), sh_degree=0,
            campos=torch.zeros(3, device="cuda"), prefiltered=False, debug=False,
        )
        rasterizer = GaussianRasterizer(settings)
        means = torch.rand(points, 3, device="cuda") * 0.6 - 0.3
        means[:, 2] = 2.0 if visible else -2.0
        rotations = torch.zeros(points, 4, device="cuda")
        rotations[:, 0] = 1
        args = dict(means3D=means, means2D=torch.zeros_like(means),
                    opacities=torch.full((points, 1), opacity, device="cuda"),
                    scales=torch.full((points, 3), 0.4, device="cuda"),
                    rotations=rotations)
        if use_sh:
            args["shs"] = torch.randn(points, 1, 3, device="cuda")
        else:
            args["colors_precomp"] = torch.rand(points, 3, device="cuda")
        return rasterizer, args

    def bundle(self, rows, plane_count=12):
        coords = (torch.rand(rows, 4, device="cuda") * 4 - 2).contiguous()
        if rows >= 8:
            coords[:8] = torch.tensor([
                [-1, -1, -1, -1], [1, 1, 1, 1], [0, 0, 0, 0],
                [-2, 2, -2, 2], [float("nan"), 0, 0, 0],
                [0, float("inf"), 0, 0], [0, 0, -float("inf"), 0],
                [0.1, -0.2, 0.3, -0.4],
            ], device="cuda")
        pairs = list(itertools.combinations(range(4), 2))
        axes = (pairs * 2)[:plane_count]
        planes = [torch.randn(1, 16, 1 if i % 5 == 0 else 7 + i,
                              1 if i % 7 == 0 else 11 + i, device="cuda")
                  for i in range(plane_count)]
        return coords, planes, axes

    def check_hex(self, outputs, coords, planes, axes):
        self.assertEqual(len(outputs), len(planes))
        for actual, plane, pair in zip(outputs, planes, axes):
            self.assertEqual(tuple(actual.shape), (coords.shape[0], 16))
            if coords.shape[0] == 0:
                continue
            reference = F.grid_sample(
                plane, coords[:, pair].reshape(1, 1, -1, 2),
                mode="bilinear", padding_mode="border", align_corners=True,
            ).reshape(16, -1).t()
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=3e-6)

    def check_raster(self, actual, expected, height, width):
        torch.testing.assert_close(actual.color, expected.color, rtol=0, atol=0)
        torch.testing.assert_close(actual.depth, expected.depth, rtol=0, atol=0)
        _, contrib = _C.tacker_image_state(actual.image_buffer, height, width)
        _, ref_contrib = _C.tacker_image_state(expected.image_buffer, height, width)
        torch.testing.assert_close(contrib, ref_contrib, rtol=0, atol=0)

    def test_real_raster_body_ratios_and_tail_mappings(self):
        rasterizer, args = self.scene()
        expected = rasterizer.prepare_for_deform_mlp_overlap(**args)
        self.assertGreater(expected.num_rendered, 256)
        rasterizer.render_prepared_raster_fast_range(expected, 0, 12)
        for rows, ratio in itertools.product((1, 255, 256, 257, 1003),
                                              (1, 2, 4, 17, 10000, 2**31 - 1)):
            with self.subTest(rows=rows, ratio=ratio):
                actual = rasterizer.prepare_for_deform_mlp_overlap(**args)
                coords, planes, axes = self.bundle(rows)
                output = rasterizer.render_prepared_with_hexplane_sampling(
                    actual, coords, planes, axes, hex_tasks_per_cta=ratio)
                self.check_hex(output, coords, planes, axes)
                self.check_raster(actual, expected, 35, 49)

    def test_partial_raster_range_preserves_other_tiles(self):
        rasterizer, args = self.scene(use_sh=True, opacity=0.9)
        expected = rasterizer.prepare_for_deform_mlp_overlap(**args)
        rasterizer.render_prepared_raster_fast_range(expected, 0, 12)
        actual = rasterizer.prepare_for_deform_mlp_overlap(**args)
        actual.color.fill_(-7)
        actual.depth.fill_(-9)
        coords, planes, axes = self.bundle(257, 6)
        output = rasterizer.render_prepared_with_hexplane_sampling(
            actual, coords, planes, axes, raster_start=1, raster_end=11,
            hex_tasks_per_cta=4)
        mask = torch.arange(35, device="cuda")[:, None] // 16 * 4 + (
            torch.arange(49, device="cuda")[None, :] // 16)
        untouched = (mask == 0) | (mask == 11)
        self.assertTrue(bool((actual.color[:, untouched] == -7).all()))
        self.assertTrue(bool((actual.depth[:, untouched] == -9).all()))
        self.check_hex(output, coords, planes, axes)
        rasterizer.render_prepared_raster_fast_range(actual, 0, 1)
        rasterizer.render_prepared_raster_fast_range(actual, 11, 12)
        self.check_raster(actual, expected, 35, 49)

    def test_early_graph_and_late_mixed_have_disjoint_plane_outputs(self):
        from tacker_4dgs_head.hexplane import sample_plane_out
        rasterizer, args = self.scene(height=65, width=81)
        expected = rasterizer.prepare_for_deform_mlp_overlap(**args)
        rasterizer.render_prepared_raster_fast_range(expected, 0, 30)
        for mixed_planes in (1, 3, 6, 9, 12):
            with self.subTest(mixed_planes=mixed_planes):
                coords, planes, axes = self.bundle(111525)
                outputs = tuple(torch.empty(16, coords.shape[0], device="cuda").t()
                                for _ in planes)
                early_count = 12 - mixed_planes
                deform_stream = torch.cuda.Stream()
                raster_stream = torch.cuda.Stream()
                deform_stream.wait_stream(torch.cuda.current_stream())
                raster_stream.wait_stream(torch.cuda.current_stream())
                early_graph = None
                if early_count:
                    with torch.cuda.stream(deform_stream):
                        for _ in range(2):
                            for i in range(early_count):
                                sample_plane_out(coords, planes[i], outputs[i], *axes[i])
                    deform_stream.synchronize()
                    early_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(early_graph, stream=deform_stream):
                        for i in range(early_count):
                            sample_plane_out(coords, planes[i], outputs[i], *axes[i])
                for _ in range(2):
                    actual = rasterizer.prepare_for_deform_mlp_overlap(**args)
                    for output in outputs:
                        output.fill_(-12345)
                    coords.mul_(0.8)
                    deform_stream.wait_stream(torch.cuda.current_stream())
                    raster_stream.wait_stream(torch.cuda.current_stream())
                    coords_ready = torch.cuda.Event()
                    sampling_done = torch.cuda.Event()
                    with torch.cuda.stream(deform_stream):
                        coords_ready.record(deform_stream)
                        if early_graph is not None:
                            early_graph.replay()
                    paired = min(30, (mixed_planes * 436 + 2) // 3)
                    with torch.cuda.stream(raster_stream):
                        raster_stream.wait_event(coords_ready)
                        rasterizer.render_prepared_with_hexplane_sampling(
                            actual, coords, planes[early_count:], axes[early_count:],
                            outputs=outputs[early_count:], hex_tasks_per_cta=3,
                            raster_start=0, raster_end=paired)
                        sampling_done.record(raster_stream)
                        rasterizer.render_prepared_raster_fast_range(actual, paired, 30)
                    with torch.cuda.stream(deform_stream):
                        deform_stream.wait_event(sampling_done)
                        features = torch.cat((torch.stack(outputs[:6]).prod(0),
                                              torch.stack(outputs[6:]).prod(0)), dim=1)
                    torch.cuda.synchronize()
                    self.check_hex(outputs, coords, planes, axes)
                    reference = [F.grid_sample(
                        plane, coords[:, pair].reshape(1, 1, -1, 2), mode="bilinear",
                        padding_mode="border", align_corners=True).reshape(16, -1).t()
                        for plane, pair in zip(planes, axes)]
                    expected_features = torch.cat((torch.stack(reference[:6]).prod(0),
                                                   torch.stack(reference[6:]).prod(0)), dim=1)
                    torch.testing.assert_close(features, expected_features, rtol=1e-4, atol=3e-5)
                    self.check_raster(actual, expected, 65, 81)

    def test_empty_raster_hex_only_and_empty_hex_raster_only(self):
        for points, rows, start, end in ((0, 257, 0, 12),
                                        (600, 257, 5, 5),
                                        (600, 0, 0, 12),
                                        (0, 0, 0, 12)):
            with self.subTest(points=points, rows=rows, start=start):
                rasterizer, args = self.scene(points=points)
                actual = rasterizer.prepare_for_deform_mlp_overlap(**args)
                color_before = actual.color.clone()
                expected = rasterizer.prepare_for_deform_mlp_overlap(**args)
                rasterizer.render_prepared_raster_fast_range(expected, start, end)
                coords, planes, axes = self.bundle(rows, 1)
                output = rasterizer.render_prepared_with_hexplane_sampling(
                    actual, coords, planes, axes, raster_start=start, raster_end=end)
                self.check_hex(output, coords, planes, axes)
                if points and start != end:
                    self.check_raster(actual, expected, 35, 49)
                else:
                    torch.testing.assert_close(actual.color, color_before, rtol=0, atol=0)

    def test_visible_count_zero(self):
        rasterizer, args = self.scene(visible=False)
        prepared = rasterizer.prepare_for_deform_mlp_overlap(**args)
        self.assertEqual(prepared.num_rendered, 0)
        coords, planes, axes = self.bundle(257)
        output = rasterizer.render_prepared_with_hexplane_sampling(prepared, coords, planes, axes)
        self.check_hex(output, coords, planes, axes)
        torch.testing.assert_close(prepared.color,
                                   rasterizer.raster_settings.bg[:, None, None].expand(3, 35, 49),
                                   rtol=0, atol=0)

    def test_graph_replay_on_nondefault_stream_with_changed_inputs(self):
        # 323 Raster tiles vs ceil(12*436/17)=308 Hex CTAs: most physical
        # blocks exercise BOTH halves, unlike a tiny image with a Hex-only tail.
        rasterizer, args = self.scene(height=257, width=289)
        expected = rasterizer.prepare_for_deform_mlp_overlap(**args)
        rasterizer.render_prepared_raster_fast_range(expected, 0, 323)
        actual = rasterizer.prepare_for_deform_mlp_overlap(**args)
        coords, planes, axes = self.bundle(111525)
        outputs = tuple(torch.empty(16, coords.shape[0], device="cuda").t() for _ in planes)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                rasterizer.render_prepared_with_hexplane_sampling(
                    actual, coords, planes, axes, outputs=outputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                returned = rasterizer.render_prepared_with_hexplane_sampling(
                    actual, coords, planes, axes, outputs=outputs)
            for _ in range(3):
                coords.mul_(0.8)
                for plane in planes:
                    plane.add_(0.1)
                graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        self.assertTrue(all(a is b for a, b in zip(returned, outputs)))
        self.check_hex(outputs, coords, planes, axes)
        self.check_raster(actual, expected, 257, 289)

    def test_invalid_arguments_and_aliases_rejected(self):
        rasterizer, args = self.scene()
        prepared = rasterizer.prepare_for_deform_mlp_overlap(**args)
        coords, planes, axes = self.bundle(257, 2)
        invoke = lambda **kw: rasterizer.render_prepared_with_hexplane_sampling(
            prepared, coords, planes, axes, **kw)
        for ratio in (0, -1, 2**31):
            with self.assertRaises(ValueError):
                invoke(hex_tasks_per_cta=ratio)
        with self.assertRaises(TypeError):
            invoke(hex_tasks_per_cta=True)
        with self.assertRaises(ValueError):
            invoke(raster_start=12, raster_end=13)
        with self.assertRaises(RuntimeError):
            invoke(outputs=[torch.empty(257, 16, device="cuda") for _ in planes])
        duplicate = torch.empty(16, 257, device="cuda").t()
        with self.assertRaises(RuntimeError):
            invoke(outputs=[duplicate, duplicate])
        overlapping = torch.empty(16 * 257 + 17, device="cuda")
        view_a = overlapping[:-17].view(16, 257).t()
        view_b = overlapping[17:].view(16, 257).t()
        with self.assertRaises(RuntimeError):
            invoke(outputs=[view_a, view_b])
        alias_plane = torch.randn(1, 16, 1, 257, device="cuda")
        with self.assertRaises(RuntimeError):
            rasterizer.render_prepared_with_hexplane_sampling(
                prepared, coords, [alias_plane], [(0, 1)],
                outputs=[alias_plane.reshape(16, 257).t()])
        with self.assertRaises(RuntimeError):
            rasterizer.render_prepared_with_hexplane_sampling(
                prepared, torch.empty(257, 8, device="cuda")[:, ::2], planes, axes)
        with self.assertRaises(RuntimeError):
            rasterizer.render_prepared_with_hexplane_sampling(
                prepared, coords.double(), planes, axes)
        with self.assertRaises(RuntimeError):
            rasterizer.render_prepared_with_hexplane_sampling(
                prepared, coords, planes, [(0, 0), (1, 2)])
        with torch.enable_grad(), self.assertRaises(RuntimeError):
            invoke()

    def test_resources(self):
        resources = GaussianRasterizer.raster_hexplane_resources()
        print("RASTER_HEXPLANE_MIXED_RESOURCES=" + json.dumps(resources, sort_keys=True))
        self.assertEqual(resources["threads"], 512)
        self.assertEqual(resources["static_shared_bytes"], 10240)
        self.assertGreater(resources["active_blocks_per_sm"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
