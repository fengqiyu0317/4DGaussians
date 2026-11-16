#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT = Path(__file__).resolve().parent
GLM_INCLUDE = ROOT / "third_party" / "glm"
HEAD_INCLUDE = Path(
    os.environ.get(
        "TACKER_4DGS_HEAD_INCLUDE",
        str(ROOT.parent.parent / "tacker_ext" / "include"),
    )
).resolve()

if not (HEAD_INCLUDE / "head_linear_device.cuh").is_file() or not (
    HEAD_INCLUDE / "head_linear_v2_device.cuh"
).is_file() or not (HEAD_INCLUDE / "deform_mlp_device.cuh").is_file() or not (
    HEAD_INCLUDE / "hexplane_sample_device.cuh"
).is_file():
    raise RuntimeError(
        "head_linear_device.cuh, head_linear_v2_device.cuh, and "
        "deform_mlp_device.cuh, and hexplane_sample_device.cuh are required; "
        "set TACKER_4DGS_HEAD_INCLUDE to the 4DGaussians/tacker_ext/include "
        "directory"
    )


def deployment_target():
    cuda_arch = os.environ.get("TACKER_CUDA_ARCH", "sm_86").strip().lower()
    if (
        len(cuda_arch) != 5
        or not cuda_arch.startswith("sm_")
        or not cuda_arch[3:].isdigit()
    ):
        raise RuntimeError("TACKER_CUDA_ARCH must use the form sm_XX")
    major, minor = int(cuda_arch[3]), int(cuda_arch[4])
    return cuda_arch, major, minor


# The A6000 remains the default qualified deployment.  Separate builds can
# opt into another exact target without changing or overwriting that binary.
CUDA_ARCH, CC_MAJOR, CC_MINOR = deployment_target()
os.environ["TORCH_CUDA_ARCH_LIST"] = "{}.{}".format(CC_MAJOR, CC_MINOR)

setup(
    name="diff_gaussian_rasterization",
    packages=["diff_gaussian_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                str(ROOT / "cuda_rasterizer" / "rasterizer_impl.cu"),
                str(ROOT / "cuda_rasterizer" / "forward.cu"),
                str(ROOT / "cuda_rasterizer" / "backward.cu"),
                str(ROOT / "cuda_rasterizer" / "tacker_mixed.cu"),
                str(ROOT / "cuda_rasterizer" / "tacker_raster_hexplane.cu"),
                str(ROOT / "rasterize_points.cu"),
                str(ROOT / "ext.cpp"),
            ],
            include_dirs=[str(GLM_INCLUDE), str(HEAD_INCLUDE)],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-DTACKER_SM_TARGET={}".format(CUDA_ARCH),
                ],
                "nvcc": ["-O3", "-lineinfo", "-Xptxas=-v"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
