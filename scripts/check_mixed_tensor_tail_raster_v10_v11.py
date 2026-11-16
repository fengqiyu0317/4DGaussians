"""Resource and empty-Raster numerical gate for mixed ABIs v10/v11.

The empty Raster launch isolates the Deform subgroup, so exact equality with
ABI-v9 proves the Raster changes did not alter Deform math or output routing.
A real prepared scene is measured separately for Raster correctness/timing.
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
        common = (
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
        v9 = (
            _C.render_prepared_gaussians_with_deform_mlp_tensor_tail_balanced_overlap(
                *common
            )
        )
        v10 = (
            _C.render_prepared_gaussians_with_deform_mlp_tensor_tail_raster_optimized_overlap(
                *common
            )
        )
        v11 = (
            _C.render_prepared_gaussians_with_deform_mlp_tensor_tail_raster_fast_overlap(
                *common
            )
        )
        torch.cuda.synchronize()

    resources = {
        "v9": _C.tacker_resource_requirements(
            9, 1, "full_deform_mlp_tensor_tail_balanced_v9"
        ),
        "v10": _C.tacker_resource_requirements(
            10, 1, "full_deform_mlp_tensor_tail_raster_optimized_v10"
        ),
        "v11": _C.tacker_resource_requirements(
            11, 1, "full_deform_mlp_tensor_tail_raster_fast_v11"
        ),
    }
    envelope_keys = (
        "physical_threads",
        "registers_per_thread",
        "local_bytes_per_thread",
        "active_blocks_per_multiprocessor",
        "active_warps_per_multiprocessor",
        "occupancy",
    )
    result = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": "sm_{}{}".format(
            *torch.cuda.get_device_capability(0)
        ),
        "rows": rows,
        "resources": resources,
        "deform_v10_vs_v9": compare(v9, v10),
        "deform_v11_vs_v9": compare(v9, v11),
        "v10_vs_v9_bitwise_equal": [
            bool(torch.equal(left, right)) for left, right in zip(v9, v10)
        ],
        "v11_vs_v9_bitwise_equal": [
            bool(torch.equal(left, right)) for left, right in zip(v9, v11)
        ],
        "v10_v11_preserve_v9_occupancy_envelope": all(
            resources[variant][key] == resources["v9"][key]
            for variant in ("v10", "v11")
            for key in envelope_keys
        ),
    }
    if not all(result["v10_vs_v9_bitwise_equal"]):
        raise RuntimeError("ABI-v10 changed empty-Raster Deform output")
    if not all(result["v11_vs_v9_bitwise_equal"]):
        raise RuntimeError("ABI-v11 changed empty-Raster Deform output")
    if not result["v10_v11_preserve_v9_occupancy_envelope"]:
        raise RuntimeError("ABI-v10/v11 changed the ABI-v9 occupancy envelope")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
