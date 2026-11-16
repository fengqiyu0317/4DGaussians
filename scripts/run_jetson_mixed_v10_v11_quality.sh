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

mkdir -p "$output_root"
if find "$output_root" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite non-empty output root: $output_root" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=0
export TACKER_CUDA_ARCH=sm_87
export TACKER_GPU_NAME=Orin
export PYTHONPATH="${project_root}/_python_deps_compat:${project_root}/tacker_ext:${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}/submodules/simple-knn:${project_root}:/home/jetson/.local/lib/python3.10/site-packages"

nvidia-smi > "${output_root}/nvidia_smi_before.txt" 2>&1
timeout 3 tegrastats --interval 1000 > "${output_root}/tegrastats_before.txt" 2>&1 || true
if ! grep -q "GR3D_FREQ 0%" "${output_root}/tegrastats_before.txt"; then
    echo "GPU was not idle before v10/v11 quality validation" >&2
    exit 3
fi

common=(
    "$python_bin" "${project_root}/scripts/validate_tacker_modes.py"
    -m "$model_path"
    -s "$source_path"
    --configs "$config_path"
    --iteration 14000
    --scene-name flame_steak
    --expected-image-width 1352
    --expected-image-height 1014
    --expected-gaussian-count 111525
    --split test
    --frames 10
    --view-start 0
    --view-stride 1
    --modes serial tacker
    --qualification-mode
    --qualification-profile "$qualification_profile"
    --lpips-net vgg
    --gpu 0
    --quiet
)

for variant in v10 v11; do
    case "$variant" in
        v10) variant_flag=TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10=1 ;;
        v11) variant_flag=TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11=1 ;;
        *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
    esac
    echo "[$(date --iso-8601=seconds)] quality variant=${variant}"
    env \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 \
        -u TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11 \
        "$variant_flag" \
        timeout 1200 "${common[@]}" \
        --output "${output_root}/${variant}.json" \
        > "${output_root}/${variant}.stdout.txt" \
        2> "${output_root}/${variant}.stderr.txt"
done

nvidia-smi > "${output_root}/nvidia_smi_after.txt" 2>&1
date --iso-8601=seconds > "${output_root}/completed_at.txt"
chmod -R a+rX "$output_root"
echo "mixed ABI-v10/v11 quality validation complete: $output_root"
