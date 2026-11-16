#!/usr/bin/env bash
set -euo pipefail
umask 022

if [[ $# -ne 3 ]]; then
    echo "usage: $0 PROJECT_ROOT OUTPUT_ROOT PHYSICAL_GPU" >&2
    exit 2
fi

project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
physical_gpu=$3
python_bin=/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python
model_root=/data/qyfeng/4DGaussians-flame-steak-full/outputs
dataset_root=/data/qyfeng/datasets/n3dv
driver_libcuda=/usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06
driver_libnvidia_ml=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06

declare -A gaussian_counts=(
    [flame_steak]=111525
    [coffee_martini]=121436
    [cook_spinach]=127047
    [flame_salmon_1]=132708
)

profile_for_scene() {
    local scene=$1
    if [[ "$scene" == flame_steak ]]; then
        printf '%s\n' "${project_root}/benchmarks/results/full_mlp_graph_formal_20260925/full_mlp_overlap_pb0.qualification.json"
    else
        printf '%s\n' "${project_root}/benchmarks/results/multiscene_formal_20260925/${scene}/profiles/full_mlp_overlap.qualification.json"
    fi
}

for required in \
    "$python_bin" \
    "$driver_libcuda" \
    "$driver_libnvidia_ml" \
    "${project_root}/scripts/run_profile_render_sealed.py" \
    "${project_root}/scripts/validate_tacker_modes.py"; do
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

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export LD_PRELOAD="${driver_libcuda}:${driver_libnvidia_ml}"
export TACKER_CUDA_ARCH=sm_86
export TACKER_GPU_NAME="NVIDIA RTX A6000"
export PYTHONPATH="${project_root}/tacker_ext:${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}/submodules/simple-knn:${project_root}"
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

date --iso-8601=seconds > "${output_root}/started_at.txt"
/usr/bin/nvidia-smi > "${output_root}/nvidia_smi_before.txt"
ps -eo user,pid,etime,cmd > "${output_root}/processes_before.txt"

for scene in flame_steak coffee_martini cook_spinach flame_salmon_1; do
    scene_root="${output_root}/${scene}"
    mkdir -p "${scene_root}/fps/runs" "${scene_root}/quality"
    model_path="${model_root}/n3dv_${scene}"
    source_path="${dataset_root}/${scene}"
    config_path="${project_root}/arguments/dynerf/${scene}.py"
    qualification_profile=$(profile_for_scene "$scene")
    expected_gaussians=${gaussian_counts[$scene]}
    for required in "$model_path" "$source_path" "$config_path" "$qualification_profile"; do
        if [[ ! -e "$required" ]]; then
            echo "required scene path is missing: $required" >&2
            exit 2
        fi
    done

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
        --workload-name "$scene"
        --execution-mode tacker
        --qualification-mode
        --qualification-profile "$qualification_profile"
        --quiet
    )

    run_index=0
    for round in $(seq 0 9); do
        for position in $(seq 0 2); do
            variant=${schedule[$((round * 3 + position))]}
            stem=$(printf "%04d-round%03d-%s" "$run_index" "$round" "$variant")
            metadata="${scene_root}/fps/runs/${stem}.metadata.json"
            echo "[$(date --iso-8601=seconds)] scene=${scene} FPS round=${round} position=${position} variant=${variant}"
            run_variant "$variant" timeout 300 \
                "${profile_common[@]}" --metadata "$metadata" \
                > "${scene_root}/fps/runs/${stem}.stdout.txt" \
                2> "${scene_root}/fps/runs/${stem}.stderr.txt"
            validate_profile_metadata "$metadata"
            run_index=$((run_index + 1))
        done
    done

    quality_common=(
        "$python_bin" "${project_root}/scripts/validate_tacker_modes.py"
        -s "$source_path"
        --model_path "$model_path"
        --configs "$config_path"
        --scene-name "$scene"
        --iteration 14000
        --split test
        --frames 50
        --view-start 0
        --view-stride 1
        --expected-image-width 1352
        --expected-image-height 1014
        --expected-gaussian-count "$expected_gaussians"
        --gpu 0
        --lpips-net vgg
        --modes serial tacker
        --qualification-mode
        --qualification-profile "$qualification_profile"
        --quiet
    )
    for variant in v5 v7 v8; do
        echo "[$(date --iso-8601=seconds)] scene=${scene} quality variant=${variant}"
        run_variant "$variant" timeout 1200 \
            "${quality_common[@]}" --output "${scene_root}/quality/${variant}.json" \
            > "${scene_root}/quality/${variant}.stdout.txt" \
            2> "${scene_root}/quality/${variant}.stderr.txt"
    done
done

/usr/bin/nvidia-smi > "${output_root}/nvidia_smi_after.txt"
date --iso-8601=seconds > "${output_root}/completed_at.txt"
chmod -R a+rX "$output_root"
echo "A6000 mixed ABI-v5/v7/v8 multi-scene validation complete: $output_root"
