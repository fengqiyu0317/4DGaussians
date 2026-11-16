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

for required in \
    "$python_bin" \
    "$model_path" \
    "$source_path" \
    "$config_path" \
    "${project_root}/scripts/run_profile_render_sealed.py" \
    "$qualification_profile" \
    "${gpu_freq_root}/cur_freq"; do
    if [[ ! -e "$required" ]]; then
        echo "required path is missing: $required" >&2
        exit 2
    fi
done

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
    echo "GPU was not idle before subgroup timing collection" >&2
    exit 3
fi
ps -eo user,pid,etime,cmd > "${output_root}/processes_before.txt"
for name in governor min_freq max_freq cur_freq available_frequencies; do
    if [[ -r "${gpu_freq_root}/${name}" ]]; then
        cp "${gpu_freq_root}/${name}" "${output_root}/${name}.txt"
    fi
done

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

# Two mirrored passes reduce sensitivity to process startup and thermal drift.
schedule=(both raster_only deform_only deform_only raster_only both)
echo "run_index,mode,freq_before_hz,freq_after_hz" > \
    "${output_root}/run_frequencies.csv"

for run_index in "${!schedule[@]}"; do
    mode=${schedule[$run_index]}
    stem=$(printf "%02d-%s" "$run_index" "$mode")
    read -r freq_before < "${gpu_freq_root}/cur_freq"
    echo "[$(date --iso-8601=seconds)] run=${run_index} mode=${mode} freq=${freq_before}"
    env \
        TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8=1 \
        TACKER_PROFILE_MIXED_V8_SUBGROUP_MODE="$mode" \
        TACKER_PROFILE_MIXED_V8_SUBGROUP_JSONL="${output_root}/${mode}.jsonl" \
        timeout 600 "${profile_common[@]}" \
        --metadata "${output_root}/runs/${stem}.metadata.json" \
        > "${output_root}/runs/${stem}.stdout.txt" \
        2> "${output_root}/runs/${stem}.stderr.txt"
    read -r freq_after < "${gpu_freq_root}/cur_freq"
    printf '%s,%s,%s,%s\n' \
        "$run_index" "$mode" "$freq_before" "$freq_after" >> \
        "${output_root}/run_frequencies.csv"
done

cleanup_monitors
trap - EXIT
timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_after.txt" 2>&1 || true
nvidia-smi > "${output_root}/nvidia_smi_after.txt" 2>&1
date --iso-8601=seconds > "${output_root}/completed_at.txt"
chmod -R a+rX "$output_root"
echo "mixed ABI-v8 subgroup CUDA-event timing collection complete: $output_root"
