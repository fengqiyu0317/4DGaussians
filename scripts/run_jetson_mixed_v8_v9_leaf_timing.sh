#!/usr/bin/env bash
set -euo pipefail
umask 022

if [[ $# -ne 2 ]]; then
    echo "usage: $0 PROJECT_ROOT OUTPUT_ROOT" >&2
    exit 2
fi

project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
python_bin=/usr/bin/python3
model_path=/data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak
source_path=/data/qyfeng/datasets/n3dv/flame_steak_4dgs_min
config_path="${project_root}/arguments/dynerf/flame_steak.py"
qualification_profile="${project_root}/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap.qualification.json"
gpu_freq_root=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu

mkdir -p "$output_root"
if find "$output_root" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite non-empty output root: $output_root" >&2
    exit 2
fi
mkdir -p "${output_root}/runs"

export CUDA_VISIBLE_DEVICES=0
export TACKER_CUDA_ARCH=sm_87
export TACKER_GPU_NAME=Orin
export PYTHONPATH="${project_root}/_python_deps_compat:${project_root}/tacker_ext:${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}/submodules/simple-knn:${project_root}:/home/jetson/.local/lib/python3.10/site-packages"
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99
export FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d
export FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503

date --iso-8601=seconds > "${output_root}/started_at.txt"
nvpmodel -q > "${output_root}/nvpmodel.txt" 2>&1
nvidia-smi > "${output_root}/nvidia_smi_before.txt" 2>&1
timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_before.txt" 2>&1 || true
if ! grep -q "GR3D_FREQ 0%" "${output_root}/tegrastats_before.txt"; then
    echo "GPU was not idle before v8/v9 leaf timing" >&2
    exit 3
fi

frequency_stop="${output_root}/frequency_monitor.stop"
(
    echo "epoch_ns,cur_freq_hz"
    while [[ ! -e "$frequency_stop" ]]; do
        read -r sample_freq < "${gpu_freq_root}/cur_freq"
        printf '%s,%s\n' "$(date +%s%N)" "$sample_freq"
        sleep 0.02
    done
) > "${output_root}/gpu_frequency.csv" &
frequency_pid=$!

cleanup_monitor() {
    touch "$frequency_stop"
    wait "$frequency_pid" 2>/dev/null || true
    rm -f "$frequency_stop"
}
trap cleanup_monitor EXIT

profile_common=(
    "$python_bin" "${project_root}/scripts/run_profile_render_sealed.py"
    --model_path "$model_path"
    --source_path "$source_path"
    --configs "$config_path"
    --iteration 14000
    --split test
    --warmup 3
    --frames 10
    --trials 1
    --workload-name flame_steak
    --execution-mode tacker
    --qualification-mode
    --qualification-profile "$qualification_profile"
    --quiet
)

schedule=(v8 v9 v9 v8)
for run_index in "${!schedule[@]}"; do
    variant=${schedule[$run_index]}
    stem=$(printf "%02d-%s" "$run_index" "$variant")
    variant_flag=TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8=1
    if [[ "$variant" == "v9" ]]; then
        variant_flag=TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9=1
    fi
    echo "[$(date --iso-8601=seconds)] run=${run_index} variant=${variant}"
    env \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9 \
        "$variant_flag" \
        TACKER_PROFILE_MIXED_LEAF_JSONL="${output_root}/${variant}.jsonl" \
        timeout 600 "${profile_common[@]}" \
        --metadata "${output_root}/runs/${stem}.metadata.json" \
        > "${output_root}/runs/${stem}.stdout.txt" \
        2> "${output_root}/runs/${stem}.stderr.txt"
done

cleanup_monitor
trap - EXIT
timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_after.txt" 2>&1 || true
nvidia-smi > "${output_root}/nvidia_smi_after.txt" 2>&1
date --iso-8601=seconds > "${output_root}/completed_at.txt"
chmod -R a+rX "$output_root"
echo "mixed ABI-v8/v9 production-leaf timing complete: $output_root"
