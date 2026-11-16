"""Lazy JIT loader for gsplat + Deform MLP mixed-kernel variants."""

from pathlib import Path

LATEST_KERNEL_VARIANT = "tensor_v7_ca"
WIDE_KERNEL_VARIANT = "tensor_v7_ca_wide"
WIDE_WARP_SPLITS = {
    WIDE_KERNEL_VARIANT: (8, 4),
    "tensor_v7_ca_wide_r6": (6, 4),
    "tensor_v7_ca_wide_r4": (4, 4),
    "tensor_v7_ca_wide_r4_d8": (4, 8),
    "tensor_v7_ca_wide_r8_d8": (8, 8),
    "tensor_v7_ca_wide_r8_d8_3cta": (8, 8),
    "tensor_v7_ca_wide_r8_d8_reuse": (8, 8),
    "tensor_v7_ca_wide_r8_d8_lean": (8, 8),
}
CANDIDATE_AWARE_VARIANTS = ("tensor_v7_ca", *WIDE_WARP_SPLITS)
KERNEL_VARIANTS = ("v1", "v2", "v3", "tensor_v7", *CANDIDATE_AWARE_VARIANTS)

_extensions = {}


def load(variant=LATEST_KERNEL_VARIANT):
    """Default to candidate-aware device-v7; older variants remain explicit.

    v2 asks ptxas for four resident 384-thread CTAs.
    v3 additionally passes descriptors through grid-constant kernel parameters.
    The register cap is experimental and device dependent. Separate module
    names keep the baseline and candidate loaded together for paired trials.
    """
    if variant not in KERNEL_VARIANTS:
        raise ValueError("unknown gsplat mixed kernel variant: " + str(variant))
    if variant not in _extensions:
        from torch.utils.cpp_extension import load as jit_load
        directory = Path(__file__).resolve().parent
        sources = ["bindings.cpp", "mixed_grid_constant.cu" if variant == "v3" else "mixed.cu"]
        if variant == "tensor_v7":
            sources = ["bindings_tensor_v7.cpp", "mixed_tensor_v7.cu"]
        elif variant in CANDIDATE_AWARE_VARIANTS:
            sources = ["bindings_tensor_v7_ca.cpp", "mixed_tensor_v7_ca.cu"]
        min_blocks = {
            "v1": 1,
            "tensor_v7_ca_wide_r8_d8_3cta": 3,
            "tensor_v7_ca_wide_r8_d8_reuse": 1,
            "tensor_v7_ca_wide_r8_d8_lean": 1,
        }.get(variant, 4)
        cuda_flags = ["-O3", "--use_fast_math", "-lineinfo",
                      f"-DGSPLAT_MIXED_MIN_BLOCKS={min_blocks}", "--ptxas-options=-v"]
        if variant == "tensor_v7_ca_wide_r8_d8_3cta":
            cuda_flags.append("-DGSPLAT_MIXED_BACKBONE_UNROLL=8")
        if variant in ("tensor_v7_ca_wide_r8_d8_reuse", "tensor_v7_ca_wide_r8_d8_lean"):
            cuda_flags.append("-DGSPLAT_MIXED_REUSE_WEIGHTS=1")
        if variant == "tensor_v7_ca_wide_r8_d8_lean":
            cuda_flags.append("-DGSPLAT_MIXED_PAIRED_LEAN=1")
        if variant in WIDE_WARP_SPLITS:
            raster_warps, deform_warps = WIDE_WARP_SPLITS[variant]
            cuda_flags.extend([
                "-DGSPLAT_MIXED_WIDE_RASTER",
                f"-DGSPLAT_MIXED_RASTER_WARPS={raster_warps}",
            ])
            if deform_warps == 8:
                cuda_flags.append("-DGSPLAT_MIXED_DEFORM_GROUPS=2")
        _extensions[variant] = jit_load(
            name="gsplat_mixed_" + variant, sources=[str(directory / file) for file in sources],
            extra_include_paths=[str(directory.parent / "tacker_ext" / "include")],
            extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags,
        )
    return _extensions[variant]
