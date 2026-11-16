#!/usr/bin/env bash
# Build the vendored extensions for one explicit architecture, without Git submodules.
set -euo pipefail
if [[ ${1:-} == --help || $# != 1 ]]; then
    echo 'Usage: bash scripts/build_inference_extensions.sh sm_86|sm_87'
    echo 'Install CUDA PyTorch, torchvision, requirements-inference.txt and gsplat first.'
    exit 0
fi
case "$1" in
    sm_86) torch_arch=8.6 ;;
    sm_87) torch_arch=8.7 ;;
    *) echo 'Supported targets: sm_86 (RTX A6000), sm_87 (Jetson Orin)' >&2; exit 2 ;;
esac
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python3}
export TACKER_CUDA_ARCH="$1" TORCH_CUDA_ARCH_LIST="$torch_arch"
export MAX_JOBS=${MAX_JOBS:-2}
export TACKER_4DGS_HEAD_INCLUDE="$root/tacker_ext/include"
unset TACKER_BUILD_HEXPLANE_ONLY
"$python_bin" -c 'import torch; assert torch.version.cuda, "A CUDA-enabled PyTorch installation is required"; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)'
for extension in submodules/simple-knn tacker_ext submodules/depth-diff-gaussian-rasterization; do
    "$python_bin" -m pip install --no-build-isolation "$root/$extension"
done
echo 'Extensions installed. gsplat mixed kernels are JIT compiled on first use.'
