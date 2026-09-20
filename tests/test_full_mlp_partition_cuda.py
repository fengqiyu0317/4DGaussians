"""End-to-end CUDA boundary test for ``FullMlpPartition``.

Run this test only after rebuilding the rasterizer with mixed ABI v5.  The
CPU-only contract tests cover scheduler accounting; this file verifies that
the real partition crosses both CUDA streams without falling back to any
Python MLP layer.
"""

import math
import types
import unittest

try:
    import torch
    import torch.nn as nn
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from gaussian_renderer import GaussianRenderState
    from gaussian_renderer import tacker_pipeline
except (ImportError, OSError):
    torch = None
    nn = None
    GaussianRasterizationSettings = None
    GaussianRasterizer = None
    GaussianRenderState = None
    tacker_pipeline = None


WIDTHS = (3, 3, 4, 1, 48)
OFFSETS = (0, 3, 6, 10, 11, 59)


def _full_mlp_available():
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        capabilities = tacker_pipeline._query_rasterizer_capabilities()
    except Exception:
        return False
    return (
        capabilities.get("mixed_render_deform_mlp_abi") == 5
        and capabilities.get("mixed_deform_mlp_family")
        == tacker_pipeline.FULL_MLP_ABI_FAMILY
    )


if nn is not None:

    class CountingLinear(nn.Linear):
        def __init__(self, in_features, out_features):
            super().__init__(in_features, out_features)
            self.forward_calls = 0

        def forward(self, value):
            self.forward_calls += 1
            return super().forward(value)


    class FixedGrid(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.register_buffer("features", features)
            self.forward_calls = 0
            self.feat_dim = features.shape[1]

        def forward(self, points, timestamp):
            self.forward_calls += 1
            if points.shape[0] != self.features.shape[0]:
                raise ValueError("fixed grid row count changed")
            if timestamp.shape != (points.shape[0], 1):
                raise ValueError("timestamp shape changed")
            return self.features


def _make_head(output_width):
    return nn.Sequential(
        nn.ReLU(),
        CountingLinear(128, 128),
        nn.ReLU(),
        CountingLinear(128, output_width),
    )


def _make_model(rows):
    generator = torch.Generator(device="cuda").manual_seed(20260918 + rows)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    network = nn.Module()
    network.W = 128
    network.D = 0
    network.no_grid = False
    network.grid_pe = 0
    network.args = types.SimpleNamespace(
        static_mlp=False,
        empty_voxel=False,
        no_dx=False,
        no_ds=False,
        no_dr=False,
        no_do=False,
        no_dshs=False,
        apply_rotation=False,
    )
    network.grid = FixedGrid(randn(rows, 32))
    network.feature_out = nn.Sequential(CountingLinear(32, 128))
    for head_name, width in zip(tacker_pipeline.HEAD_ORDER, WIDTHS):
        setattr(
            network,
            tacker_pipeline.HEAD_MODULES[head_name],
            _make_head(width),
        )
    network.cuda()

    with torch.no_grad():
        network.feature_out[0].weight.copy_(
            randn(128, 32) / math.sqrt(32)
        )
        network.feature_out[0].bias.copy_(randn(128) * 0.1)
        for head_name in tacker_pipeline.HEAD_ORDER:
            head = getattr(network, tacker_pipeline.HEAD_MODULES[head_name])
            head[1].weight.copy_(randn(128, 128) / math.sqrt(128))
            head[1].bias.copy_(randn(128) * 0.1)
            head[3].weight.copy_(
                randn(head[3].out_features, 128) / math.sqrt(128)
            )
            head[3].bias.copy_(randn(head[3].out_features) * 0.1)

    pc = types.SimpleNamespace(
        _deformation=types.SimpleNamespace(
            deformation_net=network,
            training=False,
        ),
        scaling_activation=lambda value: value,
        rotation_activation=lambda value: value,
        opacity_activation=lambda value: value,
    )
    raw = types.SimpleNamespace(
        means3D=randn(rows, 3),
        scales=randn(rows, 3),
        rotations=randn(rows, 4),
        opacity=randn(rows, 1),
        shs=randn(rows, 16, 3),
        timestamp=randn(rows, 1),
    )
    return pc, network, raw


def _make_empty_raster_context():
    float_options = {"device": "cuda", "dtype": torch.float32}
    settings = GaussianRasterizationSettings(
        image_height=16,
        image_width=16,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, **float_options),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, **float_options),
        projmatrix=torch.eye(4, **float_options),
        sh_degree=0,
        campos=torch.zeros(3, **float_options),
        prefiltered=False,
        debug=False,
    )
    empty3 = torch.empty((0, 3), **float_options)
    state = GaussianRenderState(
        means3D=empty3,
        scales=empty3,
        rotations=torch.empty((0, 4), **float_options),
        opacities=torch.empty((0, 1), **float_options),
        shs=torch.empty((0, 16, 3), **float_options),
    )
    context = types.SimpleNamespace(
        rasterizer=GaussianRasterizer(raster_settings=settings),
        means2D=torch.empty((0, 3), **float_options),
        colors_precomp=None,
        cov3D_precomp=None,
        screenspace_points=torch.empty((0, 3), **float_options),
    )
    return context, state


def _make_deform_context(raw):
    return types.SimpleNamespace(
        means3D=raw.means3D,
        scales=raw.scales,
        rotations=raw.rotations,
        opacity=raw.opacity,
        shs=raw.shs,
        timestamp=raw.timestamp,
    )


def _reference(features, cached):
    w0, b0, first_weights, first_biases, tails, tail_biases = cached
    activated = torch.relu(features @ w0 + b0).half().float()
    outputs = []
    for index in range(5):
        hidden = torch.relu(
            activated @ first_weights[index].float().t()
            + first_biases[index]
        )
        lo, hi = OFFSETS[index : index + 2]
        outputs.append(hidden @ tails[lo:hi].t() + tail_biases[lo:hi])
    return tuple(outputs)


@unittest.skipUnless(
    _full_mlp_available(), "mixed full-deform CUDA ABI v5 required"
)
class FullMlpPartitionCudaTest(unittest.TestCase):
    def setUp(self):
        self.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.previous_tf32

    def test_real_partition_crosses_streams_without_python_mlp(self):
        rows = 33
        pc, network, raw = _make_model(rows)
        deform_context = _make_deform_context(raw)
        raster_context, raster_state = _make_empty_raster_context()
        deform_stream = torch.cuda.Stream()
        raster_stream = torch.cuda.Stream()

        for persistent_blocks in (0, 2):
            with self.subTest(persistent_blocks=persistent_blocks), torch.no_grad():
                variant = tacker_pipeline.FusionVariant(
                    variant_id="cuda_full_mlp_pb{}".format(persistent_blocks),
                    selected_heads=tacker_pipeline.HEAD_ORDER,
                    worker_groups=1,
                    persistent_blocks=persistent_blocks,
                    abi_version=5,
                    cuda_symbol="tacker_mix_render_deform_mlp_v5",
                    physical_cta_threads=384,
                    backend=tacker_pipeline.FULL_MLP_BACKEND,
                    family=tacker_pipeline.FULL_MLP_ABI_FAMILY,
                )
                partition = tacker_pipeline.FullMlpPartition(variant)
                cached = partition.cache_parameters(pc)
                cache_ready = torch.cuda.Event(blocking=False)
                cache_ready.record(torch.cuda.current_stream())
                expected = _reference(network.grid.features, cached)
                prefix_ready = torch.cuda.Event(blocking=False)
                mixed_done = torch.cuda.Event(blocking=False)
                suffix_ready = torch.cuda.Event(blocking=False)

                task = partition.prepare(
                    deform_context,
                    pc,
                    deform_stream,
                    cached,
                    prefix_ready,
                )
                with torch.cuda.stream(raster_stream):
                    raster_stream.wait_event(cache_ready)
                    raster_stream.wait_event(prefix_ready)
                    mixed = partition.launch_mixed(
                        raster_context, raster_state, task
                    )
                    result, outputs = partition.result_from_mixed(
                        raster_context, mixed
                    )
                    mixed_done.record(raster_stream)
                with torch.cuda.stream(deform_stream):
                    deform_stream.wait_event(mixed_done)
                    final_state = partition.finish(task, pc, outputs)
                    suffix_ready.record(deform_stream)

                torch.cuda.current_stream().wait_event(suffix_ready)
                torch.cuda.synchronize()

                self.assertEqual(
                    tuple(task.full_mlp_outputs), tacker_pipeline.HEAD_ORDER
                )
                for head_name, output in zip(
                    tacker_pipeline.HEAD_ORDER, outputs
                ):
                    self.assertEqual(
                        task.full_mlp_outputs[head_name].data_ptr(),
                        output.data_ptr(),
                    )
                self.assertEqual(result.render.shape, (3, 16, 16))
                for actual, reference in zip(outputs, expected):
                    torch.testing.assert_close(
                        actual, reference, atol=5e-4, rtol=2e-3
                    )
                torch.testing.assert_close(
                    final_state.means3D,
                    raw.means3D + expected[0],
                    atol=5e-4,
                    rtol=2e-3,
                )
                torch.testing.assert_close(
                    final_state.scales,
                    raw.scales + expected[1],
                    atol=5e-4,
                    rtol=2e-3,
                )
                torch.testing.assert_close(
                    final_state.rotations,
                    raw.rotations + expected[2],
                    atol=5e-4,
                    rtol=2e-3,
                )
                torch.testing.assert_close(
                    final_state.opacities,
                    raw.opacity + expected[3],
                    atol=5e-4,
                    rtol=2e-3,
                )
                torch.testing.assert_close(
                    final_state.shs,
                    raw.shs + expected[4].reshape(rows, 16, 3),
                    atol=5e-4,
                    rtol=2e-3,
                )

        self.assertEqual(network.grid.forward_calls, 2)
        self.assertEqual(network.feature_out[0].forward_calls, 0)
        for head_name in tacker_pipeline.HEAD_ORDER:
            head = getattr(network, tacker_pipeline.HEAD_MODULES[head_name])
            self.assertEqual(head[1].forward_calls, 0)
            self.assertEqual(head[3].forward_calls, 0)


if __name__ == "__main__":
    unittest.main()
