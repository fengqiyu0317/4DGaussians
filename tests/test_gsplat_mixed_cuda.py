"""Real CUDA tests for one-launch gsplat + Deform MLP physical fusion."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class GsplatMixedCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gsplat_mixed_ext import load
        cls.ext = load("v1")
        cls.candidate = load("v2")
        cls.grid_constant = load("v3")

    def inputs(self, n=73, rows=35, width=37, height=29, opaque=False):
        from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode
        torch.manual_seed(91)
        means = torch.rand(1, n, 2, device="cuda") * torch.tensor([width, height], device="cuda")
        conics = torch.zeros(1, n, 3, device="cuda")
        conics[..., 0] = 0.05
        conics[..., 2] = 0.07
        opacity = torch.rand(1, n, device="cuda") * 0.8
        if opaque:
            means[:] = torch.tensor([width / 2, height / 2], device="cuda")
            conics[..., 0] = conics[..., 2] = 0.001
            opacity.fill_(0.99)
        radii = torch.full((1, n, 2), max(width, height), dtype=torch.int32, device="cuda")
        depths = torch.arange(n, device="cuda", dtype=torch.float32).unsqueeze(0) + 1
        tw, th = (width + 15) // 16, (height + 15) // 16
        _, isects, ids = isect_tiles(means, radii, depths, 16, tw, th)
        offsets = isect_offset_encode(isects, 1, tw, th).reshape(1, th, tw)
        colors = torch.rand(1, n, 4, device="cuda")
        background = torch.tensor([[0.2, 0.4, 0.8, 0]], device="cuda")
        raster = [means, conics, colors, opacity, offsets, ids, background]
        mlp = [torch.randn(rows,32,device="cuda") * 0.1,
               torch.randn(32,128,device="cuda") * 0.03,
               torch.randn(128,device="cuda") * 0.03,
               (torch.randn(5,128,128,device="cuda") * 0.03).half(),
               torch.randn(5,128,device="cuda") * 0.03,
               torch.randn(59,128,device="cuda") * 0.03,
               torch.randn(59,device="cuda") * 0.03]
        return raster, mlp

    def compare(self, *, opaque=False):
        from gsplat.cuda._wrapper import rasterize_to_pixels
        from tacker_4dgs_head import deform_mlp_delta_v1 as deform_mlp_delta
        r, d = self.inputs(opaque=opaque)
        image, alpha = rasterize_to_pixels(*r[:4], 37, 29, 16, r[4], r[5], backgrounds=r[6])
        expected_mlp = deform_mlp_delta(*d)
        for extension in (self.ext, self.candidate, self.grid_constant):
            for blocks in (0, 1, 4):
                result = extension.forward(r, d, 37, 29, blocks, 2, True)
                torch.testing.assert_close(result[0], image, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(result[1], alpha, atol=2e-6, rtol=2e-6)
                for actual, expected in zip(result[3:8], expected_mlp):
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                self.assertTrue(torch.isfinite(result[0]).all())
                self.assertTrue((result[8][:, :2] >= 0).all())

    def test_resource_attribution_controls(self):
        with torch.no_grad():
            r, d = self.inputs()
            expected = self.ext.forward(r, d, 37, 29)
            for mode in (0, 1, 3, 4, 5):
                for blocks in (0, 1):
                    result = self.ext.forward(r, d, 37, 29, blocks, mode)
                    if mode not in (1, 5):
                        for i in (0, 1): torch.testing.assert_close(result[i], expected[i])
                    if mode not in (0, 4):
                        for a,b in zip(result[3:8], expected[3:8]): torch.testing.assert_close(a,b)
            with self.assertRaisesRegex(RuntimeError, "profiling requires"):
                self.ext.forward(r,d,37,29,0,3,True)

    def test_odd_shapes_and_grid_stride_mapping(self):
        with torch.no_grad():
            self.compare()

    def test_early_termination(self):
        with torch.no_grad():
            self.compare(opaque=True)

    def test_empty_intersections_and_zero_deform_rows(self):
        with torch.no_grad():
            r, d = self.inputs(n=0, rows=0)
            result = self.ext.forward(r, d, 37, 29)
            torch.testing.assert_close(result[0], r[6][:, None, None, :].expand(1,29,37,4))
            self.assertEqual(result[3].shape, (0,3))
            self.assertEqual(result[1].count_nonzero(), 0)

    def test_inference_and_layout_guards(self):
        with torch.no_grad():
            r, d = self.inputs()
            bad = list(r)
            bad[2] = bad[2].double()
            with self.assertRaisesRegex(RuntimeError, "dtype/layout"):
                self.ext.forward(bad, d, 37, 29)
        with torch.enable_grad():
            with self.assertRaisesRegex(RuntimeError, "inference-only"):
                self.ext.forward(r, d, 37, 29)


if __name__ == "__main__":
    unittest.main()
