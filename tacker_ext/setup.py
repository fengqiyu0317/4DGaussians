from pathlib import Path
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

# RTX A6000 is Ampere (SM 8.6).  Keep the artifact deterministic and avoid
# silently producing cubins for the build host's GPU.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")


setup(
    name="tacker-4dgs-head",
    version="0.2.0",
    description="Functional 4DGaussians deformation-head kernels for Tacker",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="tacker_4dgs_head._C",
            sources=[
                str(ROOT / "csrc" / "bindings.cpp"),
                str(ROOT / "csrc" / "head_linear.cu"),
                str(ROOT / "csrc" / "head_linear_v2.cu"),
            ],
            include_dirs=[str(ROOT / "include")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                    "-Xptxas=-v",
                    "-gencode=arch=compute_86,code=sm_86",
                ],
            },
        )
    ],
    # Keep the normal CPython SOABI suffix so an in-place binary cannot be
    # silently imported by an incompatible Python interpreter.
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.7",
    zip_safe=False,
)
