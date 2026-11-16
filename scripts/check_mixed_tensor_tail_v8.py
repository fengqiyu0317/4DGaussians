"""Resource and numerical gate for four-CTA mixed Tensor-tail ABI-v8.

The empty Raster grid isolates the Deform subgroup.  This script compares v8
with preserved v5/v7 symbols and never reports end-to-end FPS.
"""

from __future__ import annotations

import json

import torch

from check_mixed_tensor_tail_v6 import compare, empty_prepared, prepared_prefix
from diff_gaussian_rasterization import _C


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(20260928)
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
        v7 = (
            _C.render_prepared_gaussians_with_deform_mlp_tensor_tail_low_register_overlap(
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
        )
        v8 = (
            _C.render_prepared_gaussians_with_deform_mlp_tensor_tail_four_cta_overlap(
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
        )
        torch.cuda.synchronize()

    v7_accuracy = compare(v5, v7)
    v7_accuracy["candidate"] = "mixed_low_register_tensor_tail_abi_v7"
    v8_accuracy = compare(v5, v8)
    v8_accuracy["candidate"] = "mixed_four_cta_tensor_tail_abi_v8"
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
            "v7": _C.tacker_resource_requirements(
                7, 1, "full_deform_mlp_tensor_tail_low_register_v7"
            ),
            "v8": _C.tacker_resource_requirements(
                8, 1, "full_deform_mlp_tensor_tail_four_cta_v8"
            ),
        },
        "accuracy_v7_vs_v5": v7_accuracy,
        "accuracy_v8_vs_v5": v8_accuracy,
        "v8_vs_v7_bitwise_equal": [
            bool(torch.equal(left, right)) for left, right in zip(v7, v8)
        ],
        "capabilities": {
            key: value
            for key, value in _C.tacker_capabilities().items()
            if "tensor_tail" in key
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
