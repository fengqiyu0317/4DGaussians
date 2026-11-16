from pathlib import Path
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent


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


# Keep the artifact deterministic instead of silently compiling for whichever
# GPU happens to be visible on the build host.  The qualified A6000 target is
# the default; Jetson deployments opt in with TACKER_CUDA_ARCH=sm_87.
CUDA_ARCH, CC_MAJOR, CC_MINOR = deployment_target()
os.environ["TORCH_CUDA_ARCH_LIST"] = "{}.{}".format(CC_MAJOR, CC_MINOR)
TARGET_DEFINES = [
    "-DTACKER_SM_TARGET={}".format(CUDA_ARCH),
    "-DTACKER_CC_MAJOR={}".format(CC_MAJOR),
    "-DTACKER_CC_MINOR={}".format(CC_MINOR),
]


setup(
    name="tacker-4dgs-head",
    version="0.2.0",
    description="Functional 4DGaussians deformation-head kernels for Tacker",
    packages=find_packages(),
    ext_modules=([
        CUDAExtension(
            name="tacker_4dgs_head._C",
            sources=[
                str(ROOT / "csrc" / "bindings.cpp"),
                str(ROOT / "csrc" / "head_linear.cu"),
                str(ROOT / "csrc" / "head_linear_v2.cu"),
                str(ROOT / "csrc" / "deform_mlp.cu"),
            ],
            include_dirs=[str(ROOT / "include")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + TARGET_DEFINES,
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                    "-Xptxas=-v",
                    "-gencode=arch=compute_{0}{1},code=sm_{0}{1}".format(
                        CC_MAJOR, CC_MINOR
                    ),
                ] + TARGET_DEFINES,
            },
        )
    ] if os.environ.get("TACKER_BUILD_HEXPLANE_ONLY") != "1" else []) + [
        CUDAExtension(
            name="tacker_4dgs_head._hexplane",
            sources=[str(ROOT / "csrc" / "hexplane_bindings.cpp"),
                     str(ROOT / "csrc" / "hexplane_sample.cu"),
                     str(ROOT / "csrc" / "hexplane_reduce.cu")],
            include_dirs=[str(ROOT / "include")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + TARGET_DEFINES,
                "nvcc": ["-O3", "-std=c++17", "-lineinfo", "-Xptxas=-v"]
                        + TARGET_DEFINES,
            },
        )
    ],
    # Keep the normal CPython SOABI suffix so an in-place binary cannot be
    # silently imported by an incompatible Python interpreter.
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.7",
    zip_safe=False,
)
