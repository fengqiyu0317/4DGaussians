"""Run on CUDA with both rasterizers installed; CPU-only discovery is safe."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA and both rasterizers")
class GsplatBackendCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        path = Path(__file__).resolve().parents[1] / "gaussian_renderer" / "gsplat_backend.py"
        spec = importlib.util.spec_from_file_location("gsplat_backend_cuda_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.adapter = module.GsplatRasterizer
        cls.settings_type = GaussianRasterizationSettings
        cls.legacy_type = GaussianRasterizer

    def fixture(self, background, scale_modifier):
        from utils.graphics_utils import getProjectionMatrix
        torch.manual_seed(42)
        settings = self.settings_type(
            image_height=48, image_width=64, tanfovx=0.7, tanfovy=0.525,
            bg=torch.full((3,), background, device="cuda"), scale_modifier=scale_modifier,
            viewmatrix=torch.eye(4, device="cuda"),
            projmatrix=getProjectionMatrix(0.01, 100, 2 * __import__("math").atan(0.7),
                                          2 * __import__("math").atan(0.525)).T.cuda(),
            sh_degree=3, campos=torch.zeros(3, device="cuda"), prefiltered=False, debug=False)
        xyz = torch.randn(64, 3, device="cuda") * 0.45
        xyz[:, 2] += 3
        quats = torch.nn.functional.normalize(torch.randn(64, 4, device="cuda"), dim=-1)
        inputs = dict(means3D=xyz, means2D=torch.zeros_like(xyz),
                      scales=torch.rand(64, 3, device="cuda") * 0.12 + 0.02,
                      rotations=quats, opacities=torch.rand(64, 1, device="cuda") * 0.7,
                      shs=torch.randn(64, 16, 3, device="cuda") * 0.15,
                      colors_precomp=None, cov3D_precomp=None)
        return settings, inputs

    @unittest.skipIf(torch is None, "requires torch")
    def test_rgb_depth_and_packed_agree_for_anisotropic_rotated_gaussians(self):
        with torch.no_grad():
            for background, modifier in ((0.0, 1.0), (1.0, 1.3)):
                with self.subTest(background=background, modifier=modifier):
                    settings, inputs = self.fixture(background, modifier)
                    reference = self.legacy_type(raster_settings=settings)(**inputs)
                    dense = self.adapter(settings)(**inputs)
                    packed = self.adapter(settings, packed=True)(**inputs)
                    self.assertEqual(dense[0].shape, (3, 48, 64))
                    self.assertEqual(dense[1].shape, (64,))
                    self.assertEqual(dense[2].shape, (1, 48, 64))
                    # Different footprint bounds can cause small truncation
                    # differences; a half-pixel shift or matrix transpose bug
                    # causes errors far larger than these limits.
                    self.assertLess((dense[0] - reference[0]).abs().mean().item(), 3e-4)
                    self.assertLess((dense[2] - reference[2]).abs().mean().item(), 1e-3)
                    torch.testing.assert_close(dense[0], packed[0], atol=2e-6, rtol=2e-6)
                    torch.testing.assert_close(dense[2], packed[2], atol=2e-6, rtol=2e-6)

    def test_training_is_explicitly_rejected(self):
        with torch.enable_grad():
            with self.assertRaisesRegex(RuntimeError, "inference only"):
                self.adapter(SimpleNamespace())

    def test_nonidentity_camera_and_precomputed_covariance(self):
        from utils.general_utils import build_scaling_rotation, strip_symmetric
        with torch.no_grad():
            settings, inputs = self.fixture(1.0, 1.3)
            view = torch.eye(4, device="cuda")
            angle = 0.3
            cosine, sine = __import__("math").cos(angle), __import__("math").sin(angle)
            view[:3, :3] = torch.tensor([[cosine, 0, sine], [0, 1, 0],
                                         [-sine, 0, cosine]], device="cuda")
            view[:3, 3] = torch.tensor([0.1, -0.1, 0.2], device="cuda")
            projection = settings.projmatrix
            settings = settings._replace(viewmatrix=view.T.contiguous(),
                                         projmatrix=view.T @ projection,
                                         campos=torch.linalg.inv(view)[:3, 3])
            transform = build_scaling_rotation(inputs["scales"] * settings.scale_modifier,
                                                inputs["rotations"])
            covariance = strip_symmetric(transform @ transform.transpose(1, 2))
            reference = self.legacy_type(raster_settings=settings)(**inputs)
            dense = self.adapter(settings)(**inputs)
            cov_inputs = dict(inputs, scales=None, rotations=None, cov3D_precomp=covariance)
            with_covariance = self.adapter(settings)(**cov_inputs)
            self.assertLess((dense[0] - reference[0]).abs().mean().item(), 3e-4)
            self.assertLess((dense[2] - reference[2]).abs().mean().item(), 1e-3)
            torch.testing.assert_close(dense[0], with_covariance[0], atol=3e-6, rtol=3e-6)
            torch.testing.assert_close(dense[2], with_covariance[2], atol=3e-6, rtol=3e-6)


if __name__ == "__main__":
    unittest.main()
