#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "usage: $0 SCENE PHYSICAL_GPU GAUSSIAN_COUNT LABEL TRIALS RESULT_ROOT" >&2
    exit 2
fi

scene="$1"
physical_gpu="$2"
gaussian_count="$3"
label="$4"
trials="$5"
result_root="$6"
project_root="$(cd "$(dirname "$0")/.." && pwd)"
python_executable="${FOURDGS_PYTHON:-/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python}"
dataset_root="${FOURDGS_DATASET_ROOT:-/data/qyfeng/datasets/n3dv}"
model_root="${FOURDGS_MODEL_ROOT:-/data/qyfeng/4DGaussians-flame-steak-full/outputs}"
output_dir="${result_root}/${label}"
profile_dir="${result_root}/profiles"
current_profile="${project_root}/tacker_profiles/baselines/a6000_phase4_20260914/run-complete-v5/attempts/selection-admission/0001/enabled-profile.json"

if [ -e "${output_dir}/report.json" ] || [ -e "${output_dir}/runs" ]; then
    echo "refusing to overwrite existing benchmark artifacts in ${output_dir}" >&2
    exit 2
fi
mkdir -p "$output_dir"

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06"
export PYTHONPATH="${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
# The preserved benchmark snapshot intentionally has no .git directory.  Bind
# it to the source commits it was copied from; strict per-file hashes below
# still identify all modified runtime sources and compiled binaries.
export FOURDGS_SOURCE_COMMIT="${FOURDGS_SOURCE_COMMIT:-60018d41f4f57b693b0e0dc4052f85b0c9d31a99}"
export FOURDGS_SOURCE_DIRTY="${FOURDGS_SOURCE_DIRTY:-true}"
export FOURDGS_RASTERIZER_COMMIT="${FOURDGS_RASTERIZER_COMMIT:-464096373b869fff713265bcdc34b1822277980d}"
export FOURDGS_SIMPLE_KNN_COMMIT="${FOURDGS_SIMPLE_KNN_COMMIT:-44f764299fa305faf6ec5ebd99939e0508331503}"

exec "$python_executable" "${project_root}/scripts/benchmark_tacker_fps.py" \
    --output "${output_dir}/report.json" \
    --runs-dir "${output_dir}/runs" \
    --run-id "${label}-abba-seed0" \
    --profile-render "${project_root}/scripts/run_profile_render_sealed.py" \
    --python-executable "$python_executable" \
    --current-tacker-profile "$current_profile" \
    --candidate "full_mlp=${profile_dir}/full_mlp.qualification.json" \
    --candidate "full_mlp_overlap=${profile_dir}/full_mlp_overlap.qualification.json" \
    --candidate "full_mlp_overlap_graph=${profile_dir}/full_mlp_overlap_graph.qualification.json" \
    --correctness-json "${result_root}/correctness.json" \
    --model-path "${model_root}/n3dv_${scene}" \
    --source-path "${dataset_root}/${scene}" \
    --configs "${project_root}/arguments/dynerf/${scene}.py" \
    --workload-name "$scene" \
    --iteration 14000 \
    --split test \
    --frames 50 \
    --warmup 10 \
    --expected-image-width 1352 \
    --expected-image-height 1014 \
    --expected-gaussian-count "$gaussian_count" \
    --trials "$trials" \
    --schedule abba \
    --seed 0 \
    --bootstrap-resamples 10000 \
    --timeout-seconds 300
