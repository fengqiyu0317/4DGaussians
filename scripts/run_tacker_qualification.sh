#!/usr/bin/env bash

# Run the complete, fail-closed Tacker qualification on the configured 4A6000
# host.  This script is intentionally a remote-host entry point: it does not
# open an SSH connection and it never manufactures measurement JSON.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
readonly HEAD_EXTENSION_DIR="${PROJECT_ROOT}/tacker_ext"
readonly RASTER_EXTENSION_DIR="${PROJECT_ROOT}/submodules/depth-diff-gaussian-rasterization"
readonly SIMPLE_KNN_DIR="${PROJECT_ROOT}/submodules/simple-knn"

TACKER_ROOT="${TACKER_ROOT:-${PROJECT_ROOT}/../Tacker-4DGS-runtime}"
TACKER_BUILD_DIR="${TACKER_BUILD_DIR:-${TACKER_ROOT}/build-runtime-a6000}"
MODEL_PATH="${MODEL_PATH:-/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak}"
SOURCE_PATH="${SOURCE_PATH:-/data/qyfeng/datasets/n3dv/flame_steak}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/qyfeng/tacker_admission}"
TORCH_HOME="${TORCH_HOME:-/data/qyfeng/cache/torch}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/arguments/dynerf/flame_steak.py}"
TEMPLATE_PROFILE="${TEMPLATE_PROFILE:-${PROJECT_ROOT}/tacker_profiles/raster_head_sm86.json}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
PYTHON_BIN="${PYTHON_BIN:-/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python}"
NVCC_BIN="${NVCC_BIN:-${CUDA_HOME}/bin/nvcc}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
CTEST_BIN="${CTEST_BIN:-ctest}"

# CUDA_VISIBLE_DEVICES is the only device selector.  Every Python entry point
# therefore addresses the selected physical card as logical device zero.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export TORCH_CUDA_ARCH_LIST=8.6
export TORCH_HOME
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

DEVICE_JSON="${DEVICE_JSON:-${OUTPUT_DIR}/device.json}"
RASTER_JSON="${RASTER_JSON:-${OUTPUT_DIR}/raster.json}"
LEAF_JSON="${LEAF_JSON:-${OUTPUT_DIR}/leaf.json}"
LEAF_REPORT="${LEAF_REPORT:-${OUTPUT_DIR}/leaf-profile-report.json}"
QUALITY_JSON="${QUALITY_JSON:-${OUTPUT_DIR}/quality.json}"
TWO_STREAM_JSON="${TWO_STREAM_JSON:-${OUTPUT_DIR}/two-stream.json}"
TACKER_JSON="${TACKER_JSON:-${OUTPUT_DIR}/tacker.json}"
ADMISSION_REPORT="${ADMISSION_REPORT:-${OUTPUT_DIR}/admission-report.json}"
ADMITTED_PROFILE="${ADMITTED_PROFILE:-${PROJECT_ROOT}/tacker_profiles/raster_head_sm86.admitted.json}"
ADMITTED_TACKER_JSON="${ADMITTED_TACKER_JSON:-${OUTPUT_DIR}/tacker-admitted-verification.json}"
HEAD_BUILD_LOG="${HEAD_BUILD_LOG:-${OUTPUT_DIR}/head-sm86-build.log}"
RASTER_BUILD_LOG="${RASTER_BUILD_LOG:-${OUTPUT_DIR}/raster-sm86-build.log}"
SIMPLE_KNN_BUILD_LOG="${SIMPLE_KNN_BUILD_LOG:-${OUTPUT_DIR}/simple-knn-sm86-build.log}"

readonly ITERATION=14000
readonly PROFILE_FRAMES=50
readonly E2E_WARMUP=10
readonly LEAF_VIEWS=2
readonly LEAF_WARMUP=5
readonly LEAF_REPETITIONS=50
readonly PERSISTENT_BLOCKS="${PERSISTENT_BLOCKS:-7000}"
QUALIFICATION_RUN_ID="${QUALIFICATION_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
HEAD_BUILD_TEMP="${HEAD_BUILD_TEMP:-${OUTPUT_DIR}/build/${QUALIFICATION_RUN_ID}/head}"
RASTER_BUILD_TEMP="${RASTER_BUILD_TEMP:-${OUTPUT_DIR}/build/${QUALIFICATION_RUN_ID}/raster}"
SIMPLE_KNN_BUILD_TEMP="${SIMPLE_KNN_BUILD_TEMP:-${OUTPUT_DIR}/build/${QUALIFICATION_RUN_ID}/simple-knn}"

step() {
    printf '\n[tacker-qualification] %s\n' "$1"
}

die() {
    printf '[tacker-qualification] ERROR: %s\n' "$1" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || die "required file is missing: $1"
}

require_directory() {
    [[ -d "$1" ]] || die "required directory is missing: $1"
}

step "1/10 exact device and environment preflight"
printf '[tacker-qualification] host: %s\n' "$(hostname -f 2>/dev/null || hostname)"

[[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]] || \
    die "CUDA_VISIBLE_DEVICES must select exactly one numeric GPU ordinal"
[[ "${PERSISTENT_BLOCKS}" =~ ^[1-9][0-9]*$ ]] || \
    die "PERSISTENT_BLOCKS must be a positive integer"
[[ "${QUALIFICATION_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "QUALIFICATION_RUN_ID may contain only letters, digits, dot, underscore, and dash"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
    die "Python executable was not found: ${PYTHON_BIN}"
command -v "${NVCC_BIN}" >/dev/null 2>&1 || \
    die "nvcc executable was not found: ${NVCC_BIN}"
command -v "${NVIDIA_SMI_BIN}" >/dev/null 2>&1 || \
    die "nvidia-smi executable was not found: ${NVIDIA_SMI_BIN}"
command -v "${CMAKE_BIN}" >/dev/null 2>&1 || \
    die "CMake executable was not found: ${CMAKE_BIN}"
command -v "${CTEST_BIN}" >/dev/null 2>&1 || \
    die "CTest executable was not found: ${CTEST_BIN}"
readonly NVCC_PATH="$(command -v "${NVCC_BIN}")"
export CUDACXX="${NVCC_PATH}"

require_directory "${TACKER_ROOT}"
require_file "${TACKER_ROOT}/src/CMakeLists.txt"
require_directory "${MODEL_PATH}"
require_directory "${SOURCE_PATH}"
mkdir -p "${TORCH_HOME}"
require_file "${CONFIG_PATH}"
require_file "${TEMPLATE_PROFILE}"
require_file "${HEAD_EXTENSION_DIR}/include/head_linear_device.cuh"
require_file "${RASTER_EXTENSION_DIR}/abi/tacker_mixed_render_head_v1.json"

readonly NVCC_RELEASE="$(
    "${NVCC_PATH}" --version \
        | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
        | tail -n 1
)"
[[ "${NVCC_RELEASE}" == "12.4" ]] || \
    die "CUDA toolkit 12.4 is required; nvcc reports ${NVCC_RELEASE:-unknown}"

readonly SMI_GPU_NAME="$(
    "${NVIDIA_SMI_BIN}" -i "${CUDA_VISIBLE_DEVICES}" \
        --query-gpu=name --format=csv,noheader \
        | tr -d '\r' \
        | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
)"
[[ "${SMI_GPU_NAME}" == "NVIDIA RTX A6000" ]] || \
    die "selected physical GPU must be NVIDIA RTX A6000; got ${SMI_GPU_NAME:-unknown}"

mapfile -t SMI_ALL_GPU_NAMES < <(
    "${NVIDIA_SMI_BIN}" --query-gpu=name --format=csv,noheader \
        | tr -d '\r' \
        | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
)
[[ "${#SMI_ALL_GPU_NAMES[@]}" -eq 4 ]] || \
    die "the qualification host must expose exactly four physical GPUs; got ${#SMI_ALL_GPU_NAMES[@]}"
for physical_gpu_name in "${SMI_ALL_GPU_NAMES[@]}"; do
    [[ "${physical_gpu_name}" == "NVIDIA RTX A6000" ]] || \
        die "all four physical GPUs must be NVIDIA RTX A6000; got ${physical_gpu_name:-unknown}"
done

[[ ! -e "${ADMITTED_PROFILE}" ]] || \
    die "refusing to reuse an existing admitted profile; choose a new ADMITTED_PROFILE: ${ADMITTED_PROFILE}"

"${PYTHON_BIN}" - "${TEMPLATE_PROFILE}" "${ADMITTED_PROFILE}" \
    "${PERSISTENT_BLOCKS}" <<'PY'
from __future__ import print_function

import json
import os
import sys

try:
    import torch
except Exception as error:
    print("PyTorch import failed: {}".format(error), file=sys.stderr)
    raise SystemExit(1)


def fail(message):
    print("Tacker device preflight failed: {}".format(message), file=sys.stderr)
    raise SystemExit(1)


if os.path.realpath(sys.argv[1]) == os.path.realpath(sys.argv[2]):
    fail("the admitted profile must not overwrite the disabled template")
try:
    with open(sys.argv[1], "r") as handle:
        template = json.load(handle)
    template_blocks = template["manifest"]["persistent_blocks"]
except Exception as error:
    fail("cannot read template persistent_blocks: {}".format(error))
if template_blocks != int(sys.argv[3]):
    fail(
        "PERSISTENT_BLOCKS {} does not match template manifest {}".format(
            sys.argv[3], template_blocks
        )
    )
if sys.version_info[:2] != (3, 10):
    fail("the qualification environment must use Python 3.10")

torch_public_version = str(torch.__version__).split("+", 1)[0]
if torch_public_version != "2.4.1":
    fail("PyTorch 2.4.1 is required, got {}".format(torch.__version__))
if torch.version.cuda != "12.4":
    fail("the PyTorch CUDA runtime must be 12.4, got {}".format(torch.version.cuda))
if not torch.cuda.is_available():
    fail("torch.cuda.is_available() is false")
if torch.cuda.device_count() != 1:
    fail(
        "exactly one CUDA device must be visible through CUDA_VISIBLE_DEVICES, got {}"
        .format(torch.cuda.device_count())
    )

torch.cuda.set_device(0)
gpu_name = torch.cuda.get_device_name(0).strip()
compute_capability = tuple(torch.cuda.get_device_capability(0))
if gpu_name != "NVIDIA RTX A6000":
    fail("logical GPU 0 must be NVIDIA RTX A6000, got {}".format(gpu_name))
if compute_capability != (8, 6):
    fail("logical GPU 0 must have compute capability 8.6, got {}".format(
        compute_capability
    ))

print(
    "Tacker device preflight passed: host={}, visible={}, gpu={}, cc=8.6, "
    "python={}.{}.{}, torch={}, torch_cuda={}".format(
        os.uname()[1],
        os.environ["CUDA_VISIBLE_DEVICES"],
        gpu_name,
        sys.version_info[0],
        sys.version_info[1],
        sys.version_info[2],
        torch.__version__,
        torch.version.cuda,
    )
)
PY

mkdir -p "${OUTPUT_DIR}"
[[ -w "${OUTPUT_DIR}" ]] || die "output directory is not writable: ${OUTPUT_DIR}"

step "2/10 build and test the reusable Tacker CUDA runtime"
"${CMAKE_BIN}" \
    -S "${TACKER_ROOT}/src" \
    -B "${TACKER_BUILD_DIR}" \
    -DTACKER_BUILD_LEGACY=OFF \
    -DTACKER_ENABLE_CUDA_BACKEND=ON \
    -DTACKER_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_COMPILER="${NVCC_PATH}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
"${CMAKE_BIN}" --build "${TACKER_BUILD_DIR}" --parallel
(
    cd "${TACKER_BUILD_DIR}"
    "${CTEST_BIN}" --output-on-failure
)
require_file "${TACKER_BUILD_DIR}/compile_commands.json"
grep -Fq "CudaExecutionBackend.cc" "${TACKER_BUILD_DIR}/compile_commands.json" || \
    die "Tacker was built without the required CUDA execution backend"

step "3/10 build simple-knn, head, and mixed Raster extensions for sm_86"
(
    cd "${SIMPLE_KNN_DIR}"
    "${PYTHON_BIN}" setup.py build_ext --inplace --force \
        --build-temp "${SIMPLE_KNN_BUILD_TEMP}" 2>&1 \
        | tee "${SIMPLE_KNN_BUILD_LOG}"
)
(
    cd "${HEAD_EXTENSION_DIR}"
    "${PYTHON_BIN}" setup.py build_ext --inplace --force \
        --build-temp "${HEAD_BUILD_TEMP}" 2>&1 \
        | tee "${HEAD_BUILD_LOG}"
)
(
    cd "${RASTER_EXTENSION_DIR}"
    TACKER_4DGS_HEAD_INCLUDE="${HEAD_EXTENSION_DIR}/include" \
        "${PYTHON_BIN}" setup.py build_ext --inplace --force \
        --build-temp "${RASTER_BUILD_TEMP}" 2>&1 \
        | tee "${RASTER_BUILD_LOG}"
)
grep -Fq "ptxas info" "${HEAD_BUILD_LOG}" || \
    die "head build log contains no ptxas resource report"
grep -Fq "ptxas info" "${RASTER_BUILD_LOG}" || \
    die "Raster build log contains no ptxas resource report"

export PYTHONPATH="${HEAD_EXTENSION_DIR}:${RASTER_EXTENSION_DIR}:${SIMPLE_KNN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Import the just-built entry points explicitly.  The GPU unittest modules use
# skip decorators when an extension cannot be imported; this guard makes such
# a missing binary a hard failure rather than a successful all-skipped run.
"${PYTHON_BIN}" - <<'PY'
from __future__ import print_function

import sys

try:
    import torch
    from diff_gaussian_rasterization import _C as raster_backend
    from simple_knn._C import distCUDA2
    from tacker_4dgs_head import head_linear_solo
except Exception as error:
    print("built extension import failed: {}".format(error), file=sys.stderr)
    raise SystemExit(1)

if not torch.cuda.is_available():
    raise SystemExit("CUDA became unavailable after the extension build")
if not callable(head_linear_solo):
    raise SystemExit("head_linear_solo is not callable")
if not callable(distCUDA2):
    raise SystemExit("simple_knn.distCUDA2 is not callable")
if not hasattr(raster_backend, "rasterize_gaussians_with_head"):
    raise SystemExit("mixed Raster entry point is unavailable")
PY

step "4/10 run the head CUDA tests"
(
    cd "${HEAD_EXTENSION_DIR}"
    "${PYTHON_BIN}" -m unittest tests.test_head_linear_cuda -v
)

step "5/10 run the mixed and legacy stream-aware Raster CUDA tests"
(
    cd "${RASTER_EXTENSION_DIR}"
    "${PYTHON_BIN}" -m unittest tests.test_tacker_mixed_cuda -v
    "${PYTHON_BIN}" -m unittest tests.test_stream_aware_legacy_cuda -v
)

step "6/10 collect canonical device, Raster, and mixed-leaf measurements"
(
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" profile_tacker_leaves.py \
        --model_path "${MODEL_PATH}" \
        --source_path "${SOURCE_PATH}" \
        --configs "${CONFIG_PATH}" \
        --scene-name flame_steak \
        --iteration "${ITERATION}" \
        --split test \
        --views "${LEAF_VIEWS}" \
        --warmup "${LEAF_WARMUP}" \
        --repetitions "${LEAF_REPETITIONS}" \
        --persistent-blocks "${PERSISTENT_BLOCKS}" \
        --gpu 0 \
        --device-output "${DEVICE_JSON}" \
        --raster-output "${RASTER_JSON}" \
        --leaf-output "${LEAF_JSON}" \
        --report "${LEAF_REPORT}"
)

step "7/10 validate serial, two_stream, and qualification Tacker quality"
(
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" scripts/validate_tacker_modes.py \
        --model_path "${MODEL_PATH}" \
        --source_path "${SOURCE_PATH}" \
        --configs "${CONFIG_PATH}" \
        --iteration "${ITERATION}" \
        --scene-name flame_steak \
        --split test \
        --frames "${PROFILE_FRAMES}" \
        --modes serial two_stream tacker \
        --qualification-mode \
        --qualification-profile "${TEMPLATE_PROFILE}" \
        --gpu 0 \
        --output "${QUALITY_JSON}"
)

step "8/10 collect comparable two_stream and qualification Tacker E2E timings"
(
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" profile_render.py \
        --model_path "${MODEL_PATH}" \
        --source_path "${SOURCE_PATH}" \
        --configs "${CONFIG_PATH}" \
        --iteration "${ITERATION}" \
        --split test \
        --warmup "${E2E_WARMUP}" \
        --frames "${PROFILE_FRAMES}" \
        --execution-mode two_stream \
        --workload-name flame_steak \
        --metadata "${TWO_STREAM_JSON}"

    "${PYTHON_BIN}" profile_render.py \
        --model_path "${MODEL_PATH}" \
        --source_path "${SOURCE_PATH}" \
        --configs "${CONFIG_PATH}" \
        --iteration "${ITERATION}" \
        --split test \
        --warmup "${E2E_WARMUP}" \
        --frames "${PROFILE_FRAMES}" \
        --execution-mode tacker \
        --workload-name flame_steak \
        --qualification-mode \
        --qualification-profile "${TEMPLATE_PROFILE}" \
        --metadata "${TACKER_JSON}"
)

step "9/10 run fail-closed admission"
(
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" scripts/benchmark_tacker_admission.py \
        --device-json "${DEVICE_JSON}" \
        --quality-json "${QUALITY_JSON}" \
        --raster-json "${RASTER_JSON}" \
        --leaf-json "${LEAF_JSON}" \
        --two-stream-json "${TWO_STREAM_JSON}" \
        --tacker-json "${TACKER_JSON}" \
        --mixed-abi-json "${RASTER_EXTENSION_DIR}/abi/tacker_mixed_render_head_v1.json" \
        --head-abi-json "${HEAD_EXTENSION_DIR}/abi/head_linear_v1.json" \
        --template-profile "${TEMPLATE_PROFILE}" \
        --report "${ADMISSION_REPORT}" \
        --enabled-profile "${ADMITTED_PROFILE}"
)

step "10/10 verify normal Tacker execution with the admitted profile"
(
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" profile_render.py \
        --model_path "${MODEL_PATH}" \
        --source_path "${SOURCE_PATH}" \
        --configs "${CONFIG_PATH}" \
        --iteration "${ITERATION}" \
        --split test \
        --warmup "${E2E_WARMUP}" \
        --frames "${PROFILE_FRAMES}" \
        --execution-mode tacker \
        --workload-name flame_steak \
        --tacker-profile "${ADMITTED_PROFILE}" \
        --metadata "${ADMITTED_TACKER_JSON}"
)

# profile_render.py records fallbacks instead of treating them as a process
# failure.  Inspect its real metadata so the final normal-mode check also fails
# closed.  This block reads profiler output; it does not create or alter it.
"${PYTHON_BIN}" - "${ADMITTED_TACKER_JSON}" "${ADMITTED_PROFILE}" <<'PY'
from __future__ import print_function

import json
import os
import sys


def load_json(path, label):
    try:
        with open(path, "r") as handle:
            value = json.load(handle)
    except Exception as error:
        raise SystemExit("cannot load {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise SystemExit("{} must be a JSON object".format(label))
    return value


metadata_path = os.path.realpath(sys.argv[1])
profile_path = os.path.realpath(sys.argv[2])
metadata = load_json(metadata_path, "verification metadata")
profile = load_json(profile_path, "admitted profile")

if profile.get("admission") != {"enabled": True, "valid": True}:
    raise SystemExit("the generated profile is not enabled and valid")
if metadata.get("passed") is not True:
    raise SystemExit("normal Tacker verification metadata is not marked passed")
if metadata.get("actual_execution_mode") != "tacker":
    raise SystemExit(
        "normal Tacker verification fell back to {}: {}".format(
            metadata.get("actual_execution_mode"),
            metadata.get("tacker_fallback_reason"),
        )
    )
if metadata.get("qualification_mode_requested") is not False:
    raise SystemExit("final verification unexpectedly requested qualification mode")
if metadata.get("qualification_mode_executed") is not False:
    raise SystemExit("final verification unexpectedly executed qualification mode")
if metadata.get("tacker_profile") != profile_path:
    raise SystemExit("final verification did not use the generated admitted profile")
if metadata.get("profile_manifest_sha256") != profile.get("manifest_sha256"):
    raise SystemExit("final verification profile manifest hash does not match")
if metadata.get("tacker_fallback_reason") is not None:
    raise SystemExit("normal Tacker verification recorded a Tacker fallback")
if metadata.get("two_stream_fallback_reason") is not None:
    raise SystemExit("normal Tacker verification recorded a two_stream fallback")

print("Normal admitted-profile Tacker verification passed: {}".format(metadata_path))
PY

printf '\n[tacker-qualification] admission passed\n'
printf '[tacker-qualification] report: %s\n' "${ADMISSION_REPORT}"
printf '[tacker-qualification] admitted profile: %s\n' "${ADMITTED_PROFILE}"
printf '[tacker-qualification] normal verification: %s\n' "${ADMITTED_TACKER_JSON}"
