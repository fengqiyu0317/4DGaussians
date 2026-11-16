"""Correctness and timestamp smoke test for the ABI-v8 diagnostic kernel.

This deliberately uses an empty Raster grid.  It validates the profiling ABI,
mode gating, and Deform numerical equivalence without claiming Raster/Deform
contention results; contention measurements require a prepared real scene.
"""

from __future__ import annotations

import json

import torch

from check_mixed_tensor_tail_v6 import compare, empty_prepared, prepared_prefix
from diff_gaussian_rasterization import _C


def timing_summary(timing):
    host = timing.detach().cpu()
    durations = host[:, (1, 3)] - host[:, (0, 2)]
    return {
        "shape": list(host.shape),
        "dtype": str(host.dtype),
        "raster_nonzero_records": int((host[:, 0] != 0).sum().item()),
        "deform_nonzero_records": int((host[:, 2] != 0).sum().item()),
        "raster_positive_durations": int((durations[:, 0] > 0).sum().item()),
        "deform_positive_durations": int((durations[:, 1] > 0).sum().item()),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    binding = (
        "render_prepared_gaussians_with_deform_mlp_tensor_tail_"
        "four_cta_diagnostic"
    )
    if not hasattr(_C, binding):
        raise RuntimeError("ABI-v8 subgroup diagnostic binding is unavailable")

    torch.manual_seed(20260929)
    rows = 4097
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
        production = (
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
        diagnostic = getattr(_C, binding)
        mode_results = {}
        for mode_name, mode_id in (
            ("both", 0),
            ("raster_only", 1),
            ("deform_only", 2),
        ):
            values = diagnostic(
                *prefix,
                features,
                backbone_weight,
                backbone_bias,
                first_weights,
                first_biases,
                packed_tail_weights,
                tail_biases,
                0,
                mode_id,
            )
            torch.cuda.synchronize()
            mode_results[mode_name] = values

    both = mode_results["both"]
    deform_only = mode_results["deform_only"]
    result = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": "sm_{}{}".format(
            *torch.cuda.get_device_capability(0)
        ),
        "rows": rows,
        "production_vs_both": compare(production, both[:5]),
        "production_vs_deform_only": compare(production, deform_only[:5]),
        "production_vs_both_bitwise_equal": all(
            torch.equal(left, right) for left, right in zip(production, both[:5])
        ),
        "production_vs_deform_only_bitwise_equal": all(
            torch.equal(left, right)
            for left, right in zip(production, deform_only[:5])
        ),
        "timing": {
            name: timing_summary(values[-1])
            for name, values in mode_results.items()
        },
        "diagnostic_capability": bool(
            _C.tacker_capabilities().get(
                "prepared_raster_deform_mlp_tensor_tail_four_cta_diagnostic"
            )
        ),
    }
    if not result["production_vs_both_bitwise_equal"]:
        raise RuntimeError("diagnostic both mode changed Deform output")
    if not result["production_vs_deform_only_bitwise_equal"]:
        raise RuntimeError("diagnostic deform-only mode changed Deform output")
    if result["timing"]["raster_only"]["deform_nonzero_records"] != 0:
        raise RuntimeError("Raster-only mode recorded Deform timestamps")
    if result["timing"]["deform_only"]["raster_nonzero_records"] != 0:
        raise RuntimeError("Deform-only mode recorded Raster timestamps")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
