#!/usr/bin/env bash
set -u

if [ "$#" -ne 4 ]; then
    echo "usage: $0 SCENE PHYSICAL_GPU GAUSSIAN_COUNT RESULT_ROOT" >&2
    exit 2
fi

scene="$1"
physical_gpu="$2"
gaussian_count="$3"
result_root="$4"
project_root="$(cd "$(dirname "$0")/.." && pwd)"
python_executable="${FOURDGS_PYTHON:-/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python}"
dataset_root="${FOURDGS_DATASET_ROOT:-/data/qyfeng/datasets/n3dv}"
model_root="${FOURDGS_MODEL_ROOT:-/data/qyfeng/4DGaussians-flame-steak-full/outputs}"
quality_dir="${result_root}/quality"
profile_dir="${FOURDGS_PROFILE_DIR:-${result_root}/profiles}"
dataset_path="${FOURDGS_DATASET_PATH:-${dataset_root}/${scene}}"
model_path="${FOURDGS_MODEL_PATH:-${model_root}/n3dv_${scene}}"

mkdir -p "$quality_dir"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
if [[ -n "${FOURDGS_LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="${FOURDGS_LD_PRELOAD}"
elif [[ -f /usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06 ]] && \
     [[ -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06 ]]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06"
else
    unset LD_PRELOAD
fi
export PYTHONPATH="${project_root}/tacker_ext:${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}/submodules/simple-knn:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"

common=(
    -s "$dataset_path"
    --model_path "$model_path"
    --configs "${project_root}/arguments/dynerf/${scene}.py"
    --scene-name "$scene"
    --iteration 14000
    --split test
    --frames 50
    --view-start 0
    --view-stride 1
    --expected-image-width 1352
    --expected-image-height 1014
    --expected-gaussian-count "$gaussian_count"
    --gpu 0
    --lpips-net vgg
    --quiet
)

status=0
baseline="${quality_dir}/serial_two_stream.json"
if [ -e "$baseline" ]; then
    echo "refusing to overwrite $baseline" >&2
    exit 2
fi
"$python_executable" "${project_root}/scripts/validate_tacker_modes.py" \
    "${common[@]}" --modes serial two_stream --output "$baseline" || status=1

for candidate in full_mlp full_mlp_overlap full_mlp_overlap_graph; do
    output="${quality_dir}/${candidate}.json"
    profile="${profile_dir}/${candidate}.qualification.json"
    if [ -e "$output" ]; then
        echo "refusing to overwrite $output" >&2
        exit 2
    fi
    "$python_executable" "${project_root}/scripts/validate_tacker_modes.py" \
        "${common[@]}" --modes serial tacker --qualification-mode \
        --qualification-profile "$profile" --output "$output" || status=1
done

exit "$status"
