#!/usr/bin/env bash
# Run in a prepared environment; never install packages or modify the model.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo 'usage: PROJECT_ROOT DEPENDENCY_ROOT OUTPUT_ROOT PHYSICAL_GPU' >&2
    exit 2
fi
project_root=$(realpath "$1")
dependency_root=$(realpath "$2")
output_root=$(realpath -m "$3")
physical_gpu=$4
[[ "$physical_gpu" =~ ^[0-9]+$ ]] || exit 2
[[ ! -e "$output_root" ]] || { echo 'output already exists' >&2; exit 2; }
python_bin=${FOURDGS_PYTHON:-/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python}
qualification=${FOURDGS_GRAPH_QUALIFICATION:-/data/qyfeng/4DGaussians-flame-steak-full/benchmarks/results/full_mlp_graph_formal_20260925/full_mlp_overlap_graph_pb0.qualification.json}
model_path=${FOURDGS_MODEL_PATH:-/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak}
source_path=${FOURDGS_DATASET_PATH:-/data/qyfeng/datasets/n3dv/flame_steak}
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export PATH="$dependency_root/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$dependency_root:$project_root:$project_root/tacker_ext:$project_root/submodules/depth-diff-gaussian-rasterization:$project_root/submodules/simple-knn"
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
export TACKER_CUDA_ARCH=sm_86 TACKER_GPU_NAME='NVIDIA RTX A6000'
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99 FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d
export FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
# Do not inherit experimental mixed-kernel switches from another experiment.
while IFS='=' read -r variable _; do
    case "$variable" in TACKER_EXPERIMENTAL_*|TACKER_PROFILE_*|TACKER_MIXED_*|TACKER_HEXPLANE_*|TACKER_RASTER_HEXPLANE_*) unset "$variable" ;; esac
done < <(env)

# A healthy driver uses its normal libraries. Use the documented isolated
# preload only when nvidia-smi actually reports a driver/library mismatch.
if ! gpu_status=$(env -u LD_PRELOAD nvidia-smi 2>&1); then
    if [[ "$gpu_status" == *'Driver/library version mismatch'* ]] &&
       [[ $(cat /sys/module/nvidia/version) == 570.124.06 ]]; then
        export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06
    else
        echo "$gpu_status" >&2
        exit 2
    fi
else
    unset LD_PRELOAD
fi
mkdir -p "$output_root"
nvidia-smi > "$output_root/gpu_before.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$output_root/processes_before.csv"
"$python_bin" - "$physical_gpu" <<'PY'
import csv, subprocess, sys
gpu = list(csv.reader(subprocess.check_output([
    'nvidia-smi', '-i', sys.argv[1], '--query-gpu=uuid,memory.used,utilization.gpu',
    '--format=csv,noheader,nounits'], text=True).splitlines()))[0]
apps = list(csv.reader(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
    text=True).splitlines()))
assert not any(r and r[0].strip() == gpu[0].strip() for r in apps), apps
assert int(gpu[1]) < 256 and int(gpu[2]) <= 5, gpu
print('GPU_IDLE', gpu, flush=True)
PY
"$python_bin" "$project_root/scripts/benchmark_gsplat_backend.py" \
    --model_path "$model_path" --source_path "$source_path" \
    --configs "$project_root/arguments/dynerf/flame_steak.py" \
    --iteration 14000 --frames 50 --warmup 10 --trials 10 \
    --modes diff_serial gsplat_dense gsplat_packed diff_two_stream gsplat_two_stream tacker_graph \
    --qualification-profile "$qualification" --output "$output_root" \
    > "$output_root/run.log" 2>&1
nvidia-smi > "$output_root/gpu_after.txt"
