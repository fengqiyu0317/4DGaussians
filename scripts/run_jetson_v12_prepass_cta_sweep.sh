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
pipeline_source="${project_root}/gaussian_renderer/tacker_pipeline.py"
mixed_source="${project_root}/submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/tacker_mixed.cu"
raster_binary=$(find \
    "${project_root}/submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization" \
    -maxdepth 1 -type f -name '_C*.so' -print -quit)
gpu_freq_root=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu

for required in \
    "$python_bin" \
    "$model_path" \
    "$source_path" \
    "$config_path" \
    "$qualification_profile" \
    "$pipeline_source" \
    "$mixed_source" \
    "$raster_binary" \
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
mkdir -p "${output_root}/runs"

pipeline_sha256=$(sha256sum "$pipeline_source" | awk '{print $1}')
mixed_sha256=$(sha256sum "$mixed_source" | awk '{print $1}')
binary_sha256=$(sha256sum "$raster_binary" | awk '{print $1}')
printf '%s  %s\n' "$pipeline_sha256" "$pipeline_source" \
    > "${output_root}/runtime_sources.sha256"
printf '%s  %s\n' "$mixed_sha256" "$mixed_source" \
    >> "${output_root}/runtime_sources.sha256"
printf '%s  %s\n' "$binary_sha256" "$raster_binary" \
    >> "${output_root}/runtime_sources.sha256"

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
    --warmup 3
    --frames 10
    --trials 1
    --workload-name flame_steak
    --execution-mode tacker
    --qualification-mode
    --qualification-profile "$qualification_profile"
    --quiet
)

check_sources() {
    local stem=$1
    if [[ "$(sha256sum "$pipeline_source" | awk '{print $1}')" != "$pipeline_sha256" || \
          "$(sha256sum "$mixed_source" | awk '{print $1}')" != "$mixed_sha256" || \
          "$(sha256sum "$raster_binary" | awk '{print $1}')" != "$binary_sha256" ]]; then
        echo "runtime source or binary changed before ${stem}" >&2
        exit 4
    fi
}

check_gpu_idle() {
    local stem=$1
    local minimum maximum current
    check_sources "$stem"
    read -r minimum < "${gpu_freq_root}/min_freq"
    read -r maximum < "${gpu_freq_root}/max_freq"
    read -r current < "${gpu_freq_root}/cur_freq"
    printf 'min=%s\nmax=%s\ncur=%s\n' "$minimum" "$maximum" "$current" \
        > "${output_root}/runs/${stem}.gpu_freq_before.txt"
    if [[ "$minimum" != 1300500000 || "$maximum" != 1300500000 || "$current" != 1300500000 ]]; then
        echo "GPU frequency is not fixed at 1300500000 before ${stem}" >&2
        exit 3
    fi
    timeout 3 tegrastats --interval 1000 \
        > "${output_root}/runs/${stem}.tegrastats_before.txt" 2>&1 || [[ $? -eq 124 ]]
    if grep "GR3D_FREQ" "${output_root}/runs/${stem}.tegrastats_before.txt" | grep -qv "GR3D_FREQ 0%"; then
        echo "GPU was not continuously idle before ${stem}" >&2
        exit 3
    fi
}

validate_run() {
    local metadata=$1
    local timeline=$2
    "$python_bin" -c '
import json, math, sys
metadata_path, timeline_path = sys.argv[1:]
with open(metadata_path, encoding="utf-8") as handle:
    doc = json.load(handle)
assert doc.get("passed") is True, doc.get("errors")
assert doc.get("actual_execution_mode") == "tacker"
assert doc.get("selected_variant_id") == "full_mlp_overlap_graph_pb0"
assert doc.get("qualification_mode_executed") is True
assert doc.get("tacker_fallback_reason") is None
assert doc.get("two_stream_fallback_reason") is None
assert doc.get("profile_frames") == 10
assert doc.get("warmup_frames") == 3
counts = doc.get("pipeline_execution_counts", {})
assert counts.get("mixed_launches") == 9, counts
assert counts.get("raster_prepasses") == 9, counts
assert counts.get("outputs") == 10, counts
fps = doc.get("throughput_fps")
assert isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0
with open(timeline_path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
leaf = [row for row in rows if row.get("kind") == "mixed_pipeline_leaf_cuda_event"]
assert len(leaf) == 11, len(leaf)
assert all(row.get("profile_schedule") == "both" for row in leaf)
' "$metadata" "$timeline"
}

# Mirrored order balances thermal/time drift while giving two independent runs
# for each physical-grid size.  Zero preserves the unbounded legacy launch.
schedule=(0 80 64 48 48 64 80 0)

date --iso-8601=seconds > "${output_root}/started_at.txt"
nvpmodel -q > "${output_root}/nvpmodel.txt" 2>&1

for index in "${!schedule[@]}"; do
    blocks=${schedule[$index]}
    if [[ "$blocks" == 0 ]]; then
        label=unbounded
    else
        label="blocks${blocks}"
    fi
    stem=$(printf '%02d-%s' "$index" "$label")
    metadata="${output_root}/runs/${stem}.metadata.json"
    timeline="${output_root}/runs/${stem}.timeline.jsonl"
    check_gpu_idle "$stem"
    echo "[$(date --iso-8601=seconds)] ${stem}"

    env \
        -u TACKER_MIXED_V12_SPLIT_TILE \
        -u TACKER_MIXED_V12_SPLIT_FRACTION \
        -u TACKER_PROFILE_MIXED_V12_SPLIT_JSONL \
        -u TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11 \
        TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1 \
        TACKER_MIXED_V12_PREPASS_PHYSICAL_BLOCKS="$blocks" \
        TACKER_PROFILE_MIXED_PIPELINE_JSONL="$timeline" \
        "${profile_common[@]}" --metadata "$metadata" \
        > "${output_root}/runs/${stem}.log" 2>&1
    validate_run "$metadata" "$timeline"
done

check_sources finished
date --iso-8601=seconds > "${output_root}/finished_at.txt"
timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_after.txt" 2>&1 || true
if grep "GR3D_FREQ" "${output_root}/tegrastats_after.txt" | grep -qv "GR3D_FREQ 0%"; then
    echo "GPU did not return to continuous idle after the sweep" >&2
    exit 3
fi
echo "Jetson v12 prepass CTA sweep complete: ${output_root}"
