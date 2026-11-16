"""GPU smoke test for legacy training on a non-default PyTorch stream."""

import unittest

try:
    import torch
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except (ImportError, OSError):
    torch = None
    GaussianRasterizationSettings = None
    GaussianRasterizer = None


def _legacy_cuda_available():
    return (
        torch is not None
        and torch.cuda.is_available()
        and GaussianRasterizationSettings is not None
        and GaussianRasterizer is not None
    )


@unittest.skipUnless(_legacy_cuda_available(), "legacy CUDA extension required")
class StreamAwareLegacyCudaTest(unittest.TestCase):
    def test_forward_and_backward_complete_on_nondefault_stream(self):
        device = torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.Stream(device=device)
        caller_stream = torch.cuda.current_stream(device)
        stream.wait_stream(caller_stream)

        with torch.cuda.stream(stream):
            settings = GaussianRasterizationSettings(
                image_height=32,
                image_width=32,
                tanfovx=1.0,
                tanfovy=1.0,
                bg=torch.zeros(3, device=device, dtype=torch.float32),
                scale_modifier=1.0,
                viewmatrix=torch.eye(4, device=device, dtype=torch.float32),
                projmatrix=torch.eye(4, device=device, dtype=torch.float32),
                sh_degree=0,
                campos=torch.zeros(3, device=device, dtype=torch.float32),
                prefiltered=False,
                debug=False,
            )
            rasterizer = GaussianRasterizer(settings)
            means3d = torch.tensor(
                [[0.0, 0.0, 2.0]],
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )
            means2d = torch.zeros(
                (1, 3), device=device, dtype=torch.float32, requires_grad=True
            )
            colors = torch.tensor(
                [[0.8, 0.2, 0.1]],
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )
            opacities = torch.tensor(
                [[0.9]], device=device, dtype=torch.float32, requires_grad=True
            )
            scales = torch.tensor(
                [[0.1, 0.1, 0.1]],
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )
            rotations = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]],
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )

            color, radii, depth = rasterizer(
                means3D=means3d,
                means2D=means2d,
                opacities=opacities,
                colors_precomp=colors,
                scales=scales,
                rotations=rotations,
            )
            loss = color.square().sum() + depth.square().sum()
            loss.backward()
            complete = torch.cuda.Event(blocking=False)
            complete.record(stream)

        caller_stream.wait_event(complete)
        complete.synchronize()

        self.assertGreater(int((radii > 0).sum().item()), 0)
        self.assertTrue(bool(torch.isfinite(color).all().item()))
        self.assertTrue(bool(torch.isfinite(depth).all().item()))
        for name, tensor in (
            ("means3D", means3d),
            ("means2D", means2d),
            ("colors", colors),
            ("opacities", opacities),
            ("scales", scales),
            ("rotations", rotations),
        ):
            self.assertIsNotNone(tensor.grad, "{} gradient is missing".format(name))
            self.assertTrue(
                bool(torch.isfinite(tensor.grad).all().item()),
                "{} gradient is non-finite".format(name),
            )


if __name__ == "__main__":
    unittest.main()
