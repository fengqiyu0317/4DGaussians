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
mkdir -p "${output_root}/fps/runs"

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
import json, math, sys
path, expected_mode, tacker_text = sys.argv[1:]
is_tacker = tacker_text == "1"
with open(path, encoding="utf-8") as handle:
    doc = json.load(handle)
assert doc.get("passed") is True, doc.get("errors")
assert doc.get("profile_frames") == 50
assert doc.get("warmup_frames") == 10
assert doc.get("actual_execution_mode") == expected_mode, doc.get("actual_execution_mode")
assert doc.get("two_stream_fallback_reason") is None, doc.get("two_stream_fallback_reason")
assert doc.get("tacker_fallback_reason") is None, doc.get("tacker_fallback_reason")
assert doc.get("qualification_mode_requested") is is_tacker
assert doc.get("qualification_mode_executed") is is_tacker
fps = doc.get("throughput_fps")
assert isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0
if is_tacker:
    counts = doc.get("pipeline_execution_counts", {})
    assert counts.get("input_frames") == 50, counts
    assert counts.get("mixed_launches") == 49, counts
    assert counts.get("solo_raster") == 1, counts
    assert counts.get("outputs") == 50, counts
' "$1" "$2" "$3"
}

profile_common=(
    "$python_bin" "${project_root}/scripts/run_profile_render_sealed.py"
    --model_path "$model_path"
    --source_path "$source_path"
    --configs "$config_path"
    --iteration 14000
    --split test
    --warmup 10
    --frames 50
    --trials 1
    --workload-name flame_steak
    --quiet
)

# Deterministic seed-0 ABBA schedule generated for the ordered candidates
# [serial, two_stream, v9, v10, v11].  Every round contains all candidates;
# adjacent rounds reverse the rotated base order.
schedule=(
    v10 serial v9 v11 two_stream
    two_stream v11 v9 serial v10
    serial v9 v11 two_stream v10
    v10 two_stream v11 v9 serial
    v9 v11 two_stream v10 serial
    serial v10 two_stream v11 v9
    v11 two_stream v10 serial v9
    v9 serial v10 two_stream v11
    two_stream v10 serial v9 v11
    v11 v9 serial v10 two_stream
)

date --iso-8601=seconds > "${output_root}/started_at.txt"
nvpmodel -q > "${output_root}/nvpmodel.txt" 2>&1
nvidia-smi > "${output_root}/nvidia_smi_before.txt" 2>&1
timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_before.txt" 2>&1 || true
if ! grep -q "GR3D_FREQ 0%" "${output_root}/tegrastats_before.txt"; then
    echo "GPU was not idle before end-to-end collection" >&2
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
echo "run_index,round,position,variant" > "${output_root}/schedule.csv"
echo "run_index,variant,freq_before_hz,freq_after_hz" > "${output_root}/run_frequencies.csv"

for run_index in "${!schedule[@]}"; do
    variant=${schedule[$run_index]}
    round=$((run_index / 5))
    position=$((run_index % 5))
    stem=$(printf "%04d-round%03d-%s" "$run_index" "$round" "$variant")
    metadata="${output_root}/fps/runs/${stem}.metadata.json"
    stdout_path="${output_root}/fps/runs/${stem}.stdout.txt"
    stderr_path="${output_root}/fps/runs/${stem}.stderr.txt"
    printf '%s,%s,%s,%s\n' "$run_index" "$round" "$position" "$variant" >> \
        "${output_root}/schedule.csv"

    variant_env=(
        env
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11
        -u TACKER_PROFILE_MIXED_LEAF_JSONL
        -u TACKER_PROFILE_MIXED_V8_SUBGROUP_MODE
        -u TACKER_PROFILE_MIXED_V8_SUBGROUP_JSONL
    )
    extra_args=()
    is_tacker=0
    case "$variant" in
        serial)
            execution_mode=serial
            ;;
        two_stream)
            execution_mode=two_stream
            ;;
        v9)
            execution_mode=tacker
            is_tacker=1
            variant_env+=(TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9=1)
            ;;
        v10)
            execution_mode=tacker
            is_tacker=1
            variant_env+=(TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10=1)
            ;;
        v11)
            execution_mode=tacker
            is_tacker=1
            variant_env+=(TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11=1)
            ;;
        *)
            echo "unknown variant: ${variant}" >&2
            exit 2
            ;;
    esac
    if [[ "$is_tacker" == 1 ]]; then
        extra_args+=(
            --qualification-mode
            --qualification-profile "$qualification_profile"
        )
    fi

    read -r freq_before < "${gpu_freq_root}/cur_freq"
    echo "[$(date --iso-8601=seconds)] FPS run=${run_index}/49 round=${round} position=${position} variant=${variant} freq=${freq_before}"
    "${variant_env[@]}" timeout 600 \
        "${profile_common[@]}" \
        --execution-mode "$execution_mode" \
        "${extra_args[@]}" \
        --metadata "$metadata" \
        > "$stdout_path" \
        2> "$stderr_path"
    validate_metadata "$metadata" "$execution_mode" "$is_tacker"
    read -r freq_after < "${gpu_freq_root}/cur_freq"
    if [[ "$freq_before" != "1300500000" || "$freq_after" != "1300500000" ]]; then
        echo "GPU frequency changed during run ${run_index}: ${freq_before} -> ${freq_after}" >&2
        exit 3
    fi
    printf '%s,%s,%s,%s\n' "$run_index" "$variant" "$freq_before" "$freq_after" >> \
        "${output_root}/run_frequencies.csv"
done

timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_after.txt" 2>&1 || true
nvidia-smi > "${output_root}/nvidia_smi_after.txt" 2>&1
date --iso-8601=seconds > "${output_root}/completed_at.txt"
chmod -R a+rX "$output_root"
echo "serial/two-stream/v9/v10/v11 formal end-to-end collection complete: $output_root"
