#!/usr/bin/env bash
set -euo pipefail
umask 022

if [[ $# -ne 2 ]]; then
    echo "usage: $0 PROJECT_ROOT OUTPUT_ROOT" >&2
    exit 2
fi
if [[ $(id -u) -ne 0 ]]; then
    echo "Jetson NCU performance-counter collection must run as root" >&2
    exit 2
fi

project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
ncu_bin=/usr/local/cuda/bin/ncu
python_bin=/usr/bin/python3
model_path=/data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak
source_path=/data/qyfeng/datasets/n3dv/flame_steak_4dgs_min
config_path="${project_root}/arguments/dynerf/flame_steak.py"
qualification_profile="${project_root}/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap.qualification.json"
gpu_freq_root=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
kernel_regex=tacker_mix_render_deform_mlp_tensor_tail_raster_fast_v11_diagnostic

for required in "$ncu_bin" "$python_bin" "$model_path" "$source_path" \
    "$config_path" "$qualification_profile" \
    "${project_root}/scripts/run_profile_render_sealed.py"; do
    if [[ ! -e "$required" ]]; then
        echo "required path is missing: $required" >&2
        exit 2
    fi
done

if [[ -e "$output_root" ]]; then
    echo "refusing to overwrite existing output root: $output_root" >&2
    exit 2
fi
mkdir -p "$output_root"

export CUDA_VISIBLE_DEVICES=0
export TACKER_CUDA_ARCH=sm_87
export TACKER_GPU_NAME=Orin
export PYTHONPATH="${project_root}/_python_deps_compat:${project_root}/tacker_ext:${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}/submodules/simple-knn:${project_root}:/home/jetson/.local/lib/python3.10/site-packages"
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99
export FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d
export FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503

check_idle_and_clock() {
    local idle_sample
    idle_sample=$(mktemp)
    timeout 3 tegrastats --interval 1000 > "$idle_sample" 2>&1 || true
    if ! grep -q "GR3D_FREQ 0%" "$idle_sample"; then
        echo "Jetson GPU is not idle" >&2
        cat "$idle_sample" >&2
        rm -f "$idle_sample"
        exit 3
    fi
    cat "$idle_sample" >> "${output_root}/idle_checks.txt"
    rm -f "$idle_sample"
    for bound in min_freq max_freq; do
        local frequency
        read -r frequency < "${gpu_freq_root}/${bound}"
        if [[ "$frequency" != "1300500000" ]]; then
            echo "unexpected ${bound}: ${frequency}" >&2
            exit 3
        fi
    done
}

date --iso-8601=seconds > "${output_root}/started_at.txt"
nvpmodel -q > "${output_root}/nvpmodel.txt" 2>&1
jetson_clocks --show > "${output_root}/jetson_clocks.txt" 2>&1
nvidia-smi > "${output_root}/nvidia_smi_before.txt" 2>&1
ps -eo user,pid,etime,cmd > "${output_root}/processes_before.txt"
check_idle_and_clock

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
tegrastats --interval 100 > "${output_root}/tegrastats_during.txt" 2>&1 &
tegrastats_pid=$!

cleanup_monitors() {
    touch "$frequency_stop"
    wait "$frequency_pid" 2>/dev/null || true
    kill "$tegrastats_pid" 2>/dev/null || true
    wait "$tegrastats_pid" 2>/dev/null || true
    rm -f "$frequency_stop"
}
trap cleanup_monitors EXIT

common_ncu=(
    "$ncu_bin"
    --target-processes all
    --replay-mode kernel
    --set full
    --launch-count 1
    --force-overwrite
    --profile-from-start off
    --launch-skip 0
    --import-source yes
    --source-folders "$project_root"
    --kernel-name "regex:${kernel_regex}"
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

run_mode() {
    local mode=$1
    local label="v11_${mode}"
    check_idle_and_clock
    echo "[$(date --iso-8601=seconds)] NCU v11 mode=${mode}"
    env \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 \
        TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11=1 \
        TACKER_PROFILE_MIXED_BALANCED_SUBGROUP_MODE="$mode" \
        TACKER_PROFILE_MIXED_BALANCED_SUBGROUP_JSONL="${output_root}/${label}.jsonl" \
        "${common_ncu[@]}" \
        --export "${output_root}/${label}" \
        "${profile_command[@]}" \
        --metadata "${output_root}/${label}.metadata.json" \
        > "${output_root}/${label}.log" 2>&1
}

cd "$project_root"
for mode in both raster_only deform_only; do
    run_mode "$mode"
done

for mode in both raster_only deform_only; do
    label="v11_${mode}"
    "$ncu_bin" --import "${output_root}/${label}.ncu-rep" \
        --page details --csv > "${output_root}/${label}.details.csv" 2>&1
    "$ncu_bin" --import "${output_root}/${label}.ncu-rep" \
        --page raw --csv > "${output_root}/${label}.raw.csv" 2>&1
    "$ncu_bin" --import "${output_root}/${label}.ncu-rep" \
        --page source --print-source cuda,sass --csv \
        > "${output_root}/${label}.source.csv" 2>&1 || true
done

cleanup_monitors
trap - EXIT
timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_after.txt" 2>&1 || true
nvidia-smi > "${output_root}/nvidia_smi_after.txt" 2>&1
date --iso-8601=seconds > "${output_root}/completed_at.txt"
chmod -R a+rX "$output_root"
echo "Jetson v11 subgroup NCU collection complete: $output_root"
