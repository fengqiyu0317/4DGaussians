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
qualification_profile="${project_root}/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json"
gpu_freq_root=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu

for required in \
    "$ncu_bin" \
    "$python_bin" \
    "$model_path" \
    "$source_path" \
    "$config_path" \
    "$qualification_profile" \
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
    local bound
    local frequency
    local idle_sample
    idle_sample=$(mktemp)
    timeout 4 tegrastats --interval 1000 > "$idle_sample" 2>&1 || true
    if grep "GR3D_FREQ" "$idle_sample" | grep -qv "GR3D_FREQ 0%"; then
        echo "Jetson GPU is not continuously idle" >&2
        cat "$idle_sample" >&2
        rm -f "$idle_sample"
        exit 3
    fi
    cat "$idle_sample" >> "${output_root}/idle_checks.txt"
    rm -f "$idle_sample"
    for bound in min_freq max_freq cur_freq; do
        read -r frequency < "${gpu_freq_root}/${bound}"
        if [[ "$frequency" != "1300500000" ]]; then
            echo "unexpected ${bound}: ${frequency}" >&2
            exit 3
        fi
    done
}

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

common_ncu=(
    "$ncu_bin"
    --target-processes all
    --replay-mode kernel
    --graph-profiling node
    --profile-from-start off
    --launch-skip 0
    --force-overwrite
)

run_inventory() {
    local label=inventory
    check_idle_and_clock
    echo "[$(date --iso-8601=seconds)] NCU ${label}"
    env \
        -u TACKER_MIXED_V12_SPLIT_TILE \
        -u TACKER_MIXED_V12_SPLIT_FRACTION \
        -u TACKER_PROFILE_MIXED_PIPELINE_JSONL \
        -u TACKER_PROFILE_MIXED_V12_SPLIT_JSONL \
        -u TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11 \
        TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1 \
        "${common_ncu[@]}" \
        --section LaunchStats \
        --launch-count 300 \
        --export "${output_root}/${label}" \
        "${profile_command[@]}" \
        --metadata "${output_root}/${label}.metadata.json" \
        > "${output_root}/${label}.log" 2>&1
}

run_target() {
    local label=$1
    local kernel_regex=$2
    local launch_count=$3
    check_idle_and_clock
    echo "[$(date --iso-8601=seconds)] NCU ${label} regex=${kernel_regex} count=${launch_count}"
    env \
        -u TACKER_MIXED_V12_SPLIT_TILE \
        -u TACKER_MIXED_V12_SPLIT_FRACTION \
        -u TACKER_PROFILE_MIXED_PIPELINE_JSONL \
        -u TACKER_PROFILE_MIXED_V12_SPLIT_JSONL \
        -u TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11 \
        TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1 \
        "${common_ncu[@]}" \
        --set full \
        --kernel-name "regex:${kernel_regex}" \
        --launch-count "$launch_count" \
        --export "${output_root}/${label}" \
        "${profile_command[@]}" \
        --metadata "${output_root}/${label}.metadata.json" \
        > "${output_root}/${label}.log" 2>&1
}

date --iso-8601=seconds > "${output_root}/started_at.txt"
nvpmodel -q > "${output_root}/nvpmodel.txt" 2>&1
jetson_clocks --show > "${output_root}/jetson_clocks.txt" 2>&1
nvidia-smi > "${output_root}/nvidia_smi_before.txt" 2>&1 || true
ps -eo user,pid,etime,cmd > "${output_root}/processes_before.txt"

cd "$project_root"
run_inventory
run_target hexplane_grid_sampler 'grid_sampler_2d' 12
run_target raster_preprocess 'preprocessCUDA' 1
run_target raster_scan_sort '(DeviceScan|DeviceRadixSort)' 16
run_target raster_duplicate 'duplicateWithKeys' 1
run_target raster_identify 'identifyTileRanges' 1
run_target raster_prepass 'tacker_render_raster_fast_range_v12' 1

for report in "${output_root}"/*.ncu-rep; do
    label=$(basename "$report" .ncu-rep)
    "$ncu_bin" --import "$report" --page details --csv \
        > "${output_root}/${label}.details.csv" 2>&1
    "$ncu_bin" --import "$report" --page raw --csv \
        > "${output_root}/${label}.raw.csv" 2>&1
done

check_idle_and_clock
nvidia-smi > "${output_root}/nvidia_smi_after.txt" 2>&1 || true
date --iso-8601=seconds > "${output_root}/completed_at.txt"
chmod -R a+rX "$output_root"
echo "Jetson external-overlap NCU collection complete: $output_root"
