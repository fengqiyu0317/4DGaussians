#!/usr/bin/env bash
set -euo pipefail
umask 022

if [[ $# -ne 3 ]]; then
    echo "usage: $0 PROJECT_ROOT OUTPUT_ROOT PHYSICAL_GPU" >&2
    exit 2
fi
if [[ $(id -u) -eq 0 ]]; then
    echo "launch this script as the project user; it uses sudo -n only for NCU" >&2
    exit 2
fi

project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
physical_gpu=$3
ncu_bin=/usr/local/cuda/bin/ncu
python_bin=/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python
model_path=/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak
source_path=/data/qyfeng/datasets/n3dv/flame_steak
config_path="${project_root}/arguments/dynerf/flame_steak.py"
qualification_profile="${project_root}/benchmarks/results/full_mlp_graph_formal_20260925/full_mlp_overlap_pb0.qualification.json"
driver_preload=/usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06
python_path="${project_root}/tacker_ext:${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}/submodules/simple-knn:${project_root}"

for required in "$ncu_bin" "$python_bin" "$model_path" "$source_path" \
    "$config_path" "$qualification_profile"; do
    if [[ ! -e "$required" ]]; then
        echo "required path is missing: $required" >&2
        exit 2
    fi
done
sudo -n true

mkdir -p "$output_root"
if find "$output_root" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite non-empty output root: $output_root" >&2
    exit 2
fi

date --iso-8601=seconds > "${output_root}/started_at.txt"
env LD_PRELOAD="$driver_preload" /usr/bin/nvidia-smi > "${output_root}/nvidia_smi_before.txt"
ps -eo user,pid,etime,cmd > "${output_root}/processes_before.txt"

base_env=(
    CUDA_VISIBLE_DEVICES="$physical_gpu"
    LD_PRELOAD="$driver_preload"
    TACKER_CUDA_ARCH=sm_86
    TACKER_GPU_NAME="NVIDIA RTX A6000"
    PYTHONPATH="$python_path"
    FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99
    FOURDGS_SOURCE_DIRTY=true
    FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d
    FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
)
common_ncu=(
    "$ncu_bin"
    --target-processes all
    --replay-mode kernel
    --set full
    --launch-count 1
    --force-overwrite
    --profile-from-start off
    --launch-skip 0
)
profile_command=(
    "$python_bin" "${project_root}/scripts/run_profile_render_sealed.py"
    --model_path "$model_path"
    --source_path "$source_path"
    --configs "$config_path"
    --iteration 14000
    --split test
    --warmup 3
    --frames 2
    --trials 1
    --workload-name flame_steak
    --execution-mode tacker
    --qualification-mode
    --qualification-profile "$qualification_profile"
    --quiet
)

run_ncu() {
    local label=$1
    local kernel_regex=$2
    shift 2
    echo "[$(date --iso-8601=seconds)] NCU ${label}"
    sudo -n env "${base_env[@]}" "$@" \
        "${common_ncu[@]}" \
        --kernel-name "regex:${kernel_regex}" \
        --export "${output_root}/${label}" \
        "${profile_command[@]}" \
        --metadata "${output_root}/${label}.metadata.json" \
        > "${output_root}/${label}.log" 2>&1
}

run_ncu mixed_v5 tacker_mix_render_deform_mlp_v5
run_ncu mixed_v7 tacker_mix_render_deform_mlp_tensor_tail_low_register_v7 \
    TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7=1
run_ncu mixed_v8 tacker_mix_render_deform_mlp_tensor_tail_four_cta_v8 \
    TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8=1

for label in mixed_v5 mixed_v7 mixed_v8; do
    "$ncu_bin" --import "${output_root}/${label}.ncu-rep" \
        --page details --csv > "${output_root}/${label}.details.csv" 2>&1
    "$ncu_bin" --import "${output_root}/${label}.ncu-rep" \
        --page raw --csv > "${output_root}/${label}.raw.csv" 2>&1
done

env LD_PRELOAD="$driver_preload" /usr/bin/nvidia-smi > "${output_root}/nvidia_smi_after.txt"
date --iso-8601=seconds > "${output_root}/completed_at.txt"
sudo -n chmod -R a+rX "$output_root"
echo "A6000 mixed ABI-v5/v7/v8 NCU collection complete: $output_root"
