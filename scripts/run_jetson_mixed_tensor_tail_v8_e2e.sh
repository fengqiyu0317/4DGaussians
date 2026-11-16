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

for required in \
    "$python_bin" \
    "$model_path" \
    "$source_path" \
    "$config_path" \
    "${project_root}/scripts/run_profile_render_sealed.py" \
    "${project_root}/scripts/validate_tacker_modes.py" \
    "$qualification_profile"; do
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
mkdir -p "${output_root}/fps/runs" "${output_root}/quality"

export CUDA_VISIBLE_DEVICES=0
export TACKER_CUDA_ARCH=sm_87
export TACKER_GPU_NAME=Orin
export PYTHONPATH="${project_root}/_python_deps_compat:${project_root}/tacker_ext:${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}/submodules/simple-knn:${project_root}:/home/jetson/.local/lib/python3.10/site-packages"
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99
export FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d
export FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503

run_variant() {
    local variant=$1
    shift
    local -a variant_env=(
        env
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8
    )
    case "$variant" in
        v5) ;;
        v7) variant_env+=(TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7=1) ;;
        v8) variant_env+=(TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8=1) ;;
        *)
            echo "unknown variant: $variant" >&2
            return 2
            ;;
    esac
    "${variant_env[@]}" "$@"
}

validate_profile_metadata() {
    "$python_bin" -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    doc = json.load(handle)
assert doc.get("passed") is True, doc.get("errors")
assert doc.get("actual_execution_mode") == "tacker", doc.get("actual_execution_mode")
assert doc.get("tacker_fallback_reason") is None, doc.get("tacker_fallback_reason")
assert doc.get("qualification_mode_executed") is True
assert doc.get("pipeline_execution_counts", {}).get("mixed_launches") == 49
' "$1"
}

date --iso-8601=seconds > "${output_root}/started_at.txt"
nvpmodel -q > "${output_root}/nvpmodel.txt" 2>&1
timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_before.txt" 2>&1 || true
if ! grep -q "GR3D_FREQ 0%" "${output_root}/tegrastats_before.txt"; then
    echo "GPU was not idle before end-to-end collection" >&2
    exit 3
fi
ps -eo user,pid,etime,cmd > "${output_root}/processes_before.txt"

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

# Same deterministic seed-0 ABBA schedule produced by benchmark_tacker_fps.py
# for the ordered candidate set [v5, v7, v8].  Every round contains each
# candidate exactly once and adjacent rounds reverse the current base order.
schedule=(
    v8 v5 v7
    v7 v5 v8
    v5 v7 v8
    v8 v7 v5
    v7 v8 v5
    v5 v8 v7
    v8 v5 v7
    v7 v5 v8
    v5 v7 v8
    v8 v7 v5
)

run_index=0
for round in $(seq 0 9); do
    for position in $(seq 0 2); do
        variant=${schedule[$((round * 3 + position))]}
        stem=$(printf "%04d-round%03d-%s" "$run_index" "$round" "$variant")
        metadata="${output_root}/fps/runs/${stem}.metadata.json"
        echo "[$(date --iso-8601=seconds)] FPS round=${round} position=${position} variant=${variant}"
        run_variant "$variant" timeout 300 \
            "${profile_common[@]}" --metadata "$metadata" \
            > "${output_root}/fps/runs/${stem}.stdout.txt" \
            2> "${output_root}/fps/runs/${stem}.stderr.txt"
        validate_profile_metadata "$metadata"
        run_index=$((run_index + 1))
    done
done

quality_common=(
    "$python_bin" "${project_root}/scripts/validate_tacker_modes.py"
    -s "$source_path"
    --model_path "$model_path"
    --configs "$config_path"
    --scene-name flame_steak
    --iteration 14000
    --split test
    --frames 50
    --view-start 0
    --view-stride 1
    --expected-image-width 1352
    --expected-image-height 1014
    --expected-gaussian-count 111525
    --gpu 0
    --lpips-net vgg
    --modes serial tacker
    --qualification-mode
    --qualification-profile "$qualification_profile"
    --quiet
)

for variant in v5 v7 v8; do
    echo "[$(date --iso-8601=seconds)] quality variant=${variant}"
    run_variant "$variant" timeout 1200 \
        "${quality_common[@]}" --output "${output_root}/quality/${variant}.json" \
        > "${output_root}/quality/${variant}.stdout.txt" \
        2> "${output_root}/quality/${variant}.stderr.txt"
done

timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_after.txt" 2>&1 || true
date --iso-8601=seconds > "${output_root}/completed_at.txt"
chmod -R a+rX "$output_root"
echo "mixed ABI-v5/v7/v8 end-to-end and quality collection complete: $output_root"
