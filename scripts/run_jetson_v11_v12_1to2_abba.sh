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
qualification_profile="${project_root}/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json"
gpu_freq_root=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
pipeline_source="${project_root}/gaussian_renderer/tacker_pipeline.py"

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

pipeline_sha256=$(sha256sum "$pipeline_source" | awk '{print $1}')
printf '%s  %s\n' "$pipeline_sha256" "$pipeline_source" \
    > "${output_root}/runtime_source.sha256"

export CUDA_VISIBLE_DEVICES=0
export TACKER_CUDA_ARCH=sm_87
export TACKER_GPU_NAME=Orin
export PYTHONPATH="${project_root}/_python_deps_compat:${project_root}/tacker_ext:${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}/submodules/simple-knn:${project_root}:/home/jetson/.local/lib/python3.10/site-packages"
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99
export FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d
export FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503

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
    --execution-mode tacker
    --qualification-mode
    --qualification-profile "$qualification_profile"
    --quiet
)

check_gpu_idle() {
    local stem=$1
    local minimum maximum current
    if [[ "$(sha256sum "$pipeline_source" | awk '{print $1}')" != "$pipeline_sha256" ]]; then
        echo "runtime source changed before ${stem}: ${pipeline_source}" >&2
        exit 4
    fi
    minimum=$(cat "${gpu_freq_root}/min_freq")
    maximum=$(cat "${gpu_freq_root}/max_freq")
    current=$(cat "${gpu_freq_root}/cur_freq")
    printf 'min=%s\nmax=%s\ncur=%s\n' "$minimum" "$maximum" "$current" \
        > "${output_root}/runs/${stem}.gpu_freq_before.txt"
    if [[ "$minimum" != 1300500000 || "$maximum" != 1300500000 || "$current" != 1300500000 ]]; then
        echo "GPU frequency is not fixed at 1300500000 before ${stem}" >&2
        exit 3
    fi
    timeout 2 tegrastats --interval 1000 \
        > "${output_root}/runs/${stem}.tegrastats_before.txt" 2>&1 || [[ $? -eq 124 ]]
    if ! grep -q "GR3D_FREQ 0%" "${output_root}/runs/${stem}.tegrastats_before.txt"; then
        echo "GPU was not idle before ${stem}" >&2
        exit 3
    fi
}

validate_metadata() {
    local metadata=$1
    local variant=$2
    "$python_bin" -c '
import json, math, sys
path, variant = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    doc = json.load(handle)
assert doc.get("passed") is True, doc.get("errors")
assert doc.get("actual_execution_mode") == "tacker"
assert doc.get("tacker_fallback_reason") is None
assert doc.get("qualification_mode_executed") is True
assert doc.get("profile_frames") == 50
assert doc.get("warmup_frames") == 10
fps = doc.get("throughput_fps")
assert isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0
counts = doc.get("pipeline_execution_counts", {})
assert counts.get("mixed_launches") == 49, counts
assert counts.get("outputs") == 50, counts
expected_prepasses = 49 if variant == "v12" else 0
assert counts.get("raster_prepasses") == expected_prepasses, counts
' "$metadata" "$variant"
}

# Ten rounds.  Consecutive round pairs form ABBA: v12,v11,v11,v12.
schedule=(
    v12 v11
    v11 v12
    v12 v11
    v11 v12
    v12 v11
    v11 v12
    v12 v11
    v11 v12
    v12 v11
    v11 v12
)

date --iso-8601=seconds > "${output_root}/started_at.txt"
nvpmodel -q > "${output_root}/nvpmodel.txt" 2>&1

for index in "${!schedule[@]}"; do
    round=$((index / 2))
    position=$((index % 2))
    variant=${schedule[$index]}
    stem=$(printf '%04d-round%02d-pos%d-%s' "$index" "$round" "$position" "$variant")
    metadata="${output_root}/runs/${stem}.metadata.json"
    check_gpu_idle "$stem"
    echo "[$(date --iso-8601=seconds)] ${stem}"

    variant_env=(
        env
        -u TACKER_MIXED_V12_SPLIT_TILE
        -u TACKER_MIXED_V12_SPLIT_FRACTION
        -u TACKER_PROFILE_MIXED_PIPELINE_JSONL
        -u TACKER_PROFILE_MIXED_V12_SPLIT_JSONL
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12
    )
    if [[ "$variant" == v11 ]]; then
        variant_env+=(TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11=1)
    else
        variant_env+=(TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1)
    fi

    "${variant_env[@]}" "${profile_common[@]}" --metadata "$metadata" \
        > "${output_root}/runs/${stem}.log" 2>&1
    validate_metadata "$metadata" "$variant"
done

if [[ "$(sha256sum "$pipeline_source" | awk '{print $1}')" != "$pipeline_sha256" ]]; then
    echo "runtime source changed during the ABBA run: ${pipeline_source}" >&2
    exit 4
fi

date --iso-8601=seconds > "${output_root}/finished_at.txt"
timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_after.txt" 2>&1 || true
if ! grep -q "GR3D_FREQ 0%" "${output_root}/tegrastats_after.txt"; then
    echo "GPU did not return to idle after the ABBA run" >&2
    exit 3
fi
