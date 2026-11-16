"""Resource and numerical smoke gate for balanced-work mixed ABI-v9.

The empty Raster launch verifies that v9 preserves ABI-v8's Deform math and
resource envelope.  A real prepared Raster scene is still required to exercise
the one-or-two-tile balanced mapping.
"""

from __future__ import annotations

import json

import torch

from check_mixed_tensor_tail_v6 import compare, empty_prepared, prepared_prefix
from diff_gaussian_rasterization import _C


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(20260929)
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
            _tail_weights,
            packed_tail_weights,
            tail_biases,
        ) = empty_prepared(rows)
        prefix = prepared_prefix(background, colors, prepared)
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
        v9 = (
            _C.render_prepared_gaussians_with_deform_mlp_tensor_tail_balanced_overlap(
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

    resources = {
        "v8": _C.tacker_resource_requirements(
            8, 1, "full_deform_mlp_tensor_tail_four_cta_v8"
        ),
        "v9": _C.tacker_resource_requirements(
            9, 1, "full_deform_mlp_tensor_tail_balanced_v9"
        ),
    }
    result = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": "sm_{}{}".format(
            *torch.cuda.get_device_capability(0)
        ),
        "rows": rows,
        "resources": resources,
        "accuracy_v9_vs_v8": compare(v8, v9),
        "v9_vs_v8_bitwise_equal": [
            bool(torch.equal(left, right)) for left, right in zip(v8, v9)
        ],
        "resource_envelope_equal": all(
            resources["v8"][key] == resources["v9"][key]
            for key in (
                "physical_threads",
                "registers_per_thread",
                "static_shared_bytes",
                "local_bytes_per_thread",
                "active_blocks_per_multiprocessor",
                "active_warps_per_multiprocessor",
            )
        ),
    }
    if not all(result["v9_vs_v8_bitwise_equal"]):
        raise RuntimeError("ABI-v9 changed empty-Raster Deform output")
    if not result["resource_envelope_equal"]:
        raise RuntimeError("ABI-v9 changed the ABI-v8 resource envelope")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
