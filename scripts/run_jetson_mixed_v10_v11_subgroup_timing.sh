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
    "$qualification_profile" \
    "${project_root}/scripts/run_profile_render_sealed.py"; do
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

validate_metadata() {
    "$python_bin" -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    doc = json.load(handle)
assert doc.get("passed") is True, doc.get("errors")
assert doc.get("profile_frames") == 10, doc.get("profile_frames")
assert doc.get("warmup_frames") == 3, doc.get("warmup_frames")
assert doc.get("actual_execution_mode") == "tacker", doc.get("actual_execution_mode")
assert doc.get("two_stream_fallback_reason") is None, doc.get("two_stream_fallback_reason")
assert doc.get("tacker_fallback_reason") is None, doc.get("tacker_fallback_reason")
assert doc.get("qualification_mode_requested") is True
assert doc.get("qualification_mode_executed") is True
counts = doc.get("pipeline_execution_counts", {})
assert counts.get("input_frames") == 10, counts
assert counts.get("mixed_launches") == 9, counts
assert counts.get("solo_raster") == 1, counts
assert counts.get("outputs") == 10, counts
' "$1"
}

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

date --iso-8601=seconds > "${output_root}/started_at.txt"
nvpmodel -q > "${output_root}/nvpmodel.txt" 2>&1
nvidia-smi > "${output_root}/nvidia_smi_before.txt" 2>&1
timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_before.txt" 2>&1 || true
if ! grep -q "GR3D_FREQ 0%" "${output_root}/tegrastats_before.txt"; then
    echo "GPU was not idle before v10/v11 subgroup diagnostics" >&2
    exit 3
fi
for bound in min_freq max_freq; do
    read -r frequency < "${gpu_freq_root}/${bound}"
    if [[ "$frequency" != "1300500000" ]]; then
        echo "unexpected ${bound}: ${frequency}" >&2
        exit 3
    fi
done
ps -eo user,pid,etime,cmd > "${output_root}/processes_before.txt"
if [[ -r /tmp/mixed_v10_v11_subgroup_build_sm87.log ]]; then
    cp /tmp/mixed_v10_v11_subgroup_build_sm87.log "${output_root}/build_sm87.log"
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

# Each variant/mode pair runs twice, in a mirrored order, so process startup
# and thermal drift do not consistently favor one variant or diagnostic mode.
schedule=(
    v10:both v11:both
    v10:raster_only v11:raster_only
    v10:deform_only v11:deform_only
    v11:deform_only v10:deform_only
    v11:raster_only v10:raster_only
    v11:both v10:both
)
echo "run_index,variant,mode,freq_before_hz,freq_after_hz" > \
    "${output_root}/run_frequencies.csv"

for run_index in "${!schedule[@]}"; do
    entry=${schedule[$run_index]}
    variant=${entry%%:*}
    mode=${entry#*:}
    stem=$(printf "%02d-%s-%s" "$run_index" "$variant" "$mode")
    metadata="${output_root}/runs/${stem}.metadata.json"
    case "$variant" in
        v10) variant_flag=TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10=1 ;;
        v11) variant_flag=TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11=1 ;;
        *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
    esac

    read -r freq_before < "${gpu_freq_root}/cur_freq"
    echo "[$(date --iso-8601=seconds)] run=${run_index}/11 variant=${variant} mode=${mode} freq=${freq_before}"
    env \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11 \
        -u TACKER_PROFILE_MIXED_LEAF_JSONL \
        -u TACKER_PROFILE_MIXED_V8_SUBGROUP_MODE \
        -u TACKER_PROFILE_MIXED_V8_SUBGROUP_JSONL \
        "$variant_flag" \
        TACKER_PROFILE_MIXED_BALANCED_SUBGROUP_MODE="$mode" \
        TACKER_PROFILE_MIXED_BALANCED_SUBGROUP_JSONL="${output_root}/${variant}_${mode}.jsonl" \
        timeout 600 "${profile_common[@]}" \
        --metadata "$metadata" \
        > "${output_root}/runs/${stem}.stdout.txt" \
        2> "${output_root}/runs/${stem}.stderr.txt"
    validate_metadata "$metadata"
    read -r freq_after < "${gpu_freq_root}/cur_freq"
    if [[ "$freq_before" != "1300500000" || "$freq_after" != "1300500000" ]]; then
        echo "GPU frequency changed during run ${run_index}: ${freq_before} -> ${freq_after}" >&2
        exit 3
    fi
    printf '%s,%s,%s,%s,%s\n' \
        "$run_index" "$variant" "$mode" "$freq_before" "$freq_after" >> \
        "${output_root}/run_frequencies.csv"
done

cleanup_monitors
trap - EXIT
timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_after.txt" 2>&1 || true
nvidia-smi > "${output_root}/nvidia_smi_after.txt" 2>&1
date --iso-8601=seconds > "${output_root}/completed_at.txt"
chmod -R a+rX "$output_root"
echo "mixed ABI-v10/v11 subgroup diagnostics complete: $output_root"
