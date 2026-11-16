"""Resource and numerical gate for experimental mixed Tensor-tail ABI-v6.

The check intentionally uses an empty Raster grid so it isolates the Deform
subgroup's math.  Real Raster+Deform contention is measured separately by NCU
on the flame_steak prepared mixed leaf; this script never measures end-to-end
FPS.
"""

from __future__ import annotations

import json
import math

import torch

from diff_gaussian_rasterization import _C
from tacker_4dgs_head import pack_deform_tail_weights_v3


WIDTHS = (3, 3, 4, 1, 48)


def empty_prepared(rows: int):
    options = {"device": "cuda", "dtype": torch.float32}
    background = torch.zeros((3,), **options)
    means = torch.empty((0, 3), **options)
    colors = torch.empty((0, 3), **options)
    opacity = torch.empty((0, 1), **options)
    placeholder = torch.empty((0,), **options)
    covariance = torch.empty((0, 6), **options)
    prepared = _C.prepare_gaussians_for_deform_mlp_overlap(
        background,
        means,
        colors,
        opacity,
        placeholder,
        placeholder,
        1.0,
        covariance,
        torch.eye(4, **options),
        torch.eye(4, **options),
        1.0,
        1.0,
        16,
        16,
        placeholder,
        0,
        torch.zeros((3,), **options),
        False,
        False,
    )
    features = torch.randn((rows, 32), **options) * 0.05
    backbone_weight = torch.randn((32, 128), **options) * 0.05
    backbone_bias = torch.randn((128,), **options) * 0.05
    first_weights = (
        torch.randn((5, 128, 128), device="cuda", dtype=torch.float16)
        * 0.05
    )
    first_biases = torch.randn((5, 128), **options) * 0.05
    tail_weights = torch.randn((59, 128), **options) * 0.05
    tail_biases = torch.randn((59,), **options) * 0.05
    packed_tail_weights = pack_deform_tail_weights_v3(tail_weights)
    return (
        background,
        colors,
        prepared,
        features,
        backbone_weight,
        backbone_bias,
        first_weights,
        first_biases,
        tail_weights,
        packed_tail_weights,
        tail_biases,
    )


def prepared_prefix(background, colors, prepared):
    rendered, out_color, out_depth, radii, geom, binning, image = prepared
    return (
        background,
        colors,
        out_color,
        out_depth,
        radii,
        geom,
        binning,
        image,
        rendered,
        16,
        16,
        False,
    )


def compare(reference, candidate):
    per_head = []
    total_error2 = 0.0
    total_reference2 = 0.0
    for head, (left, right) in enumerate(zip(reference, candidate)):
        error = right.double() - left.double()
        error2 = float(torch.sum(error * error).item())
        reference2 = float(torch.sum(left.double() * left.double()).item())
        total_error2 += error2
        total_reference2 += reference2
        per_head.append(
            {
                "head": head,
                "width": WIDTHS[head],
                "max_absolute_error": float(error.abs().max().item()),
                "mean_absolute_error": float(error.abs().mean().item()),
                "relative_l2": math.sqrt(error2 / reference2)
                if reference2
                else 0.0,
            }
        )
    return {
        "reference": "mixed_scalar_tail_abi_v5",
        "candidate": "mixed_padded_tensor_tail_abi_v6",
        "overall_relative_l2": math.sqrt(total_error2 / total_reference2)
        if total_reference2
        else 0.0,
        "per_head": per_head,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(20260927)
    rows = 111525
    with torch.no_grad():
        (
            background,
            colors,
            prepared,
            features,
            backbone_weight,
            backbone_bias,
            first_weights,
            first_biases,
            tail_weights,
            packed_tail_weights,
            tail_biases,
        ) = empty_prepared(rows)
        prefix = prepared_prefix(background, colors, prepared)
        v5 = _C.render_prepared_gaussians_with_deform_mlp_overlap(
            *prefix,
            features,
            backbone_weight,
            backbone_bias,
            first_weights,
            first_biases,
            tail_weights,
            tail_biases,
            0,
        )
        v6 = _C.render_prepared_gaussians_with_deform_mlp_tensor_tail_overlap(
            *prefix,
            features,
            backbone_weight,
            backbone_bias,
            first_weights,
            first_biases,
            packed_tail_weights,
            tail_biases,
            0,
        )
        torch.cuda.synchronize()

    result = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": "sm_{}{}".format(
            *torch.cuda.get_device_capability(0)
        ),
        "rows": rows,
        "resources": {
            "v5": _C.tacker_resource_requirements(
                5, 1, "full_deform_mlp_v5"
            ),
            "v6": _C.tacker_resource_requirements(
                6, 1, "full_deform_mlp_tensor_tail_v6"
            ),
        },
        "accuracy": compare(v5, v6),
        "capabilities": {
            key: value
            for key, value in _C.tacker_capabilities().items()
            if "tensor_tail" in key
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
