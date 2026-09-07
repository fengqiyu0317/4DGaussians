#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 MODEL_PATH CONFIG_PATH [OUTPUT_DIR] [GPU] [FRAMES] [ITERATION] [SOURCE_PATH] [EXECUTION_MODE] [PROFILE_PATH] [QUALIFICATION_MODE] [WORKLOAD_NAME]"
    exit 2
fi

MODEL_PATH=$1
CONFIG_PATH=$2
OUTPUT_DIR=${3:-nsight_reports}
GPU_INDEX=${4:-0}
PROFILE_FRAMES=${5:-50}
LOAD_ITERATION=${6:--1}
SOURCE_PATH=${7:-${SOURCE_PATH:-}}
EXECUTION_MODE=${8:-${EXECUTION_MODE:-serial}}
TACKER_PROFILE_PATH=${9:-${TACKER_PROFILE_PATH:-}}
QUALIFICATION_MODE=${10:-${QUALIFICATION_MODE:-0}}
WORKLOAD_NAME=${11:-${WORKLOAD_NAME:-}}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
NSYS_BIN=${NSYS_BIN:-nsys}

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)
case "${EXECUTION_MODE}" in
    serial|split_serial|two_stream|tacker) ;;
    *)
        echo "Unsupported execution mode: ${EXECUTION_MODE}" >&2
        exit 2
        ;;
esac

REPORT_BASE="${OUTPUT_DIR}/4dgs_render_${EXECUTION_MODE}"
METADATA_PATH="${OUTPUT_DIR}/${EXECUTION_MODE}_profile_metadata.json"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export FOURDGS_NVTX=1
export NSYS_TMPDIR="${NSYS_TMPDIR:-/tmp}"
PROJECT_MODULE_PATHS="${PROJECT_ROOT}/submodules/depth-diff-gaussian-rasterization:${PROJECT_ROOT}/submodules/simple-knn"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_MODULE_PATHS}"

PROFILE_ARGS=(
    --model_path "${MODEL_PATH}"
    --configs "${CONFIG_PATH}"
    --iteration "${LOAD_ITERATION}"
    --split test
    --warmup 10
    --frames "${PROFILE_FRAMES}"
    --execution-mode "${EXECUTION_MODE}"
    --metadata "${METADATA_PATH}"
)
if [[ -n "${SOURCE_PATH}" ]]; then
    PROFILE_ARGS+=(--source_path "${SOURCE_PATH}")
fi
if [[ "${EXECUTION_MODE}" == "tacker" ]]; then
    if [[ -z "${TACKER_PROFILE_PATH}" || -z "${WORKLOAD_NAME}" ]]; then
        echo "tacker mode requires PROFILE_PATH and WORKLOAD_NAME" >&2
        exit 2
    fi
    PROFILE_ARGS+=(--workload-name "${WORKLOAD_NAME}")
    case "${QUALIFICATION_MODE,,}" in
        1|true|yes|on)
            PROFILE_ARGS+=(
                --qualification-mode
                --qualification-profile "${TACKER_PROFILE_PATH}"
            )
            ;;
        0|false|no|off)
            PROFILE_ARGS+=(--tacker-profile "${TACKER_PROFILE_PATH}")
            ;;
        *)
            echo "QUALIFICATION_MODE must be 0/1 or false/true" >&2
            exit 2
            ;;
    esac
fi

"${NSYS_BIN}" profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output="${REPORT_BASE}" \
    "${PYTHON_BIN}" profile_render.py "${PROFILE_ARGS[@]}"

"${NSYS_BIN}" stats \
    --report nvtx_sum,nvtx_gpu_proj_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
    --format csv \
    --force-export=true \
    "${REPORT_BASE}.nsys-rep" > "${REPORT_BASE}_stats.csv"

"${PYTHON_BIN}" scripts/summarize_nsight_stats.py \
    "${REPORT_BASE}_stats.csv" \
    --metadata "${METADATA_PATH}" \
    --output "${OUTPUT_DIR}/${EXECUTION_MODE}_summary.json"

echo "Nsight report: ${REPORT_BASE}.nsys-rep"
echo "CSV statistics: ${REPORT_BASE}_stats.csv"
echo "Structured summary: ${OUTPUT_DIR}/${EXECUTION_MODE}_summary.json"
