#!/usr/bin/env bash

# One-click RTX A6000 entry point for the sealed Phase-4 qualification.
#
# The shell owns only the remote environment and canonical workload defaults.
# scripts/run_tacker_phase4.py owns all validation, real subprocess execution,
# hash-bound resume checkpoints, and atomic publication.  In particular this
# wrapper never generates a candidate matrix or manufactures measurement JSON.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

TACKER_ROOT="${TACKER_ROOT:-${PROJECT_ROOT}/../Tacker}"
PHASE31_RUN_ROOT="${PHASE31_RUN_ROOT:-/data/qyfeng/tacker_phase31_validation/20260913-codex-phase31/run-complete-v3}"
MODEL_PATH="${MODEL_PATH:-/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak}"
SOURCE_PATH="${SOURCE_PATH:-/data/qyfeng/datasets/n3dv/flame_steak}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/arguments/dynerf/flame_steak.py}"
TEMPLATE_PROFILE="${TEMPLATE_PROFILE:-${PROJECT_ROOT}/tacker_profiles/raster_head_sm86.json}"
CURRENT_TACKER_PROFILE="${CURRENT_TACKER_PROFILE:-/home/qyfeng/tacker_phase31_code/20260913-codex-phase31/tacker_profiles/incumbents/current-v1.json}"
PYTHON_BIN="${PYTHON_BIN:-/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python3.10}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
NVCC_BIN="${NVCC_BIN:-${CUDA_HOME}/bin/nvcc}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
CTEST_BIN="${CTEST_BIN:-ctest}"
TORCH_HOME="${TORCH_HOME:-/data/qyfeng/cache/torch}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
QUALIFICATION_RUN_ID="${QUALIFICATION_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-phase4}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/qyfeng/tacker_phase4_validation/${QUALIFICATION_RUN_ID}}"

[[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]] || {
    printf 'run_tacker_qualification: CUDA_VISIBLE_DEVICES must be one numeric GPU ordinal\n' >&2
    exit 2
}
[[ "${QUALIFICATION_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    printf 'run_tacker_qualification: unsafe QUALIFICATION_RUN_ID\n' >&2
    exit 2
}

export CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME
export CUDACXX="${NVCC_BIN}"
export TORCH_HOME
export TORCH_CUDA_ARCH_LIST=8.6
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PROJECT_ROOT}/tacker_ext:${PROJECT_ROOT}/submodules/depth-diff-gaussian-rasterization:${PROJECT_ROOT}/submodules/simple-knn:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Inline workload documents are hashed as Phase-4 inputs.  They avoid creating
# mutable setup files beside the sealed Phase-3.1 evidence.  The first workload
# changes checkpoint size at native resolution; the second changes Raster load
# with the real profiling-only resolution scale.  Neither is admitted and no
# primary-workload Tacker profile is reused for them.
if [[ -z "${GENERALIZATION_WORKLOAD_1:-}" ]]; then
    printf -v GENERALIZATION_WORKLOAD_1 \
        '{"schema_version":1,"kind":"4dgaussians_tacker_phase4_workload","name":"flame_steak_iteration_3000_native","model_path":"%s","source_path":"%s","config":"%s","iteration":3000,"split":"test","image_width":1352,"image_height":1014,"gaussian_count":92999,"raster_deformation_mix":"raster_heavy","workload_key":"dynerf/flame_steak/iteration_3000/test/1352x1014/resolution_1","profile_args":["--resolution","-1"]}' \
        "${MODEL_PATH}" "${SOURCE_PATH}" "${CONFIG_PATH}"
fi
if [[ -z "${GENERALIZATION_WORKLOAD_2:-}" ]]; then
    printf -v GENERALIZATION_WORKLOAD_2 \
        '{"schema_version":1,"kind":"4dgaussians_tacker_phase4_workload","name":"flame_steak_iteration_14000_scale4","model_path":"%s","source_path":"%s","config":"%s","iteration":14000,"split":"test","image_width":338,"image_height":254,"gaussian_count":111525,"raster_deformation_mix":"deformation_heavy","workload_key":"dynerf/flame_steak/iteration_14000/test/338x254/resolution_4","profile_args":["--resolution","4"]}' \
        "${MODEL_PATH}" "${SOURCE_PATH}" "${CONFIG_PATH}"
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_tacker_phase4.py" \
    --phase31-run-root "${PHASE31_RUN_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --tacker-root "${TACKER_ROOT}" \
    --template-profile "${TEMPLATE_PROFILE}" \
    --current-tacker-profile "${CURRENT_TACKER_PROFILE}" \
    --model-path "${MODEL_PATH}" \
    --source-path "${SOURCE_PATH}" \
    --configs "${CONFIG_PATH}" \
    --workload-name flame_steak \
    --primary-workload-key flame_steak:14000:111525:1352x1014:sm_86 \
    --primary-mix balanced \
    --iteration 14000 \
    --image-width 1352 \
    --image-height 1014 \
    --gaussian-count 111525 \
    --generalization-workload "${GENERALIZATION_WORKLOAD_1}" \
    --generalization-workload "${GENERALIZATION_WORKLOAD_2}" \
    --python-executable "${PYTHON_BIN}" \
    --nvidia-smi "${NVIDIA_SMI_BIN}" \
    --nvcc "${NVCC_BIN}" \
    --cmake "${CMAKE_BIN}" \
    --ctest "${CTEST_BIN}" \
    --gpu 0 \
    --physical-gpu "${CUDA_VISIBLE_DEVICES}" \
    --expected-gpu-name "NVIDIA RTX A6000" \
    --expected-python 3.10 \
    --expected-torch 2.4.1 \
    --expected-cuda 12.4 \
    --leaf-views 2 \
    --leaf-warmup 5 \
    --leaf-repetitions 50 \
    --sequence-trials 3 \
    --generalization-trials 3 \
    --long-frames 500 \
    "$@"
