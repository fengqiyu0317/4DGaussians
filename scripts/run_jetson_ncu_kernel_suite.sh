#!/usr/bin/env bash
set -euo pipefail
umask 022

if [[ $# -ne 2 ]]; then
    echo "usage: $0 PROJECT_ROOT OUTPUT_ROOT" >&2
    exit 2
fi

if [[ $(id -u) -ne 0 ]]; then
    echo "Jetson NCU collection must run as root" >&2
    exit 2
fi

project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
ncu_bin=/usr/local/cuda/bin/ncu
python_bin=/usr/bin/python3
model_path=/data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak
source_path=/data/qyfeng/datasets/n3dv/flame_steak_4dgs_min
config_path="${project_root}/arguments/dynerf/flame_steak.py"
profile_root="${project_root}/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources"

for required in \
    "$ncu_bin" \
    "$python_bin" \
    "$model_path" \
    "$source_path" \
    "$config_path" \
    "${project_root}/tacker_ext/profile_deform_mlp_ncu.py" \
    "${project_root}/scripts/run_profile_render_sealed.py" \
    "${profile_root}/full_mlp.qualification.json"; do
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

export CUDA_VISIBLE_DEVICES=0
export TACKER_CUDA_ARCH=sm_87
export TACKER_GPU_NAME=Orin
export PYTHONPATH="${project_root}/_python_deps_compat:${project_root}/tacker_ext:${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}/submodules/simple-knn:${project_root}:/home/jetson/.local/lib/python3.10/site-packages"
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99
export FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d
export FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503

date --iso-8601=seconds > "${output_root}/started_at.txt"
jetson_clocks --show > "${output_root}/jetson_clocks.txt" 2>&1
timeout 2 tegrastats --interval 1000 > "${output_root}/tegrastats_before.txt" 2>&1 || true
ps -eo user,pid,etime,cmd > "${output_root}/processes_before.txt"

common_ncu=(
    "$ncu_bin"
    --target-processes all
    --replay-mode kernel
    --set full
    --launch-count 1
    --force-overwrite
)

echo "[$(date --iso-8601=seconds)] standalone_deform_mlp"
"${common_ncu[@]}" \
    --kernel-name regex:tacker_deform_mlp_delta_v1 \
    --launch-skip 3 \
    --export "${output_root}/standalone_deform_mlp" \
    "$python_bin" "${project_root}/tacker_ext/profile_deform_mlp_ncu.py" \
        --mode profile --rows 111525 --persistent-blocks 0 --warmup 3 \
    > "${output_root}/standalone_deform_mlp.log" 2>&1

echo "[$(date --iso-8601=seconds)] solo_raster_render"
"${common_ncu[@]}" \
    --profile-from-start off \
    --kernel-name regex:renderCUDA \
    --launch-skip 0 \
    --export "${output_root}/solo_raster_render" \
    "$python_bin" "${project_root}/scripts/run_profile_render_sealed.py" \
        --model_path "$model_path" \
        --source_path "$source_path" \
        --configs "$config_path" \
        --iteration 14000 \
        --split test \
        --warmup 3 \
        --frames 1 \
        --trials 1 \
        --workload-name flame_steak \
        --execution-mode serial \
        --metadata "${output_root}/solo_raster_render.metadata.json" \
        --quiet \
    > "${output_root}/solo_raster_render.log" 2>&1

echo "[$(date --iso-8601=seconds)] mixed_render_deform_mlp"
"${common_ncu[@]}" \
    --profile-from-start off \
    --kernel-name regex:tacker_mix_render_deform_mlp_v5 \
    --launch-skip 0 \
    --export "${output_root}/mixed_render_deform_mlp" \
    "$python_bin" "${project_root}/scripts/run_profile_render_sealed.py" \
        --model_path "$model_path" \
        --source_path "$source_path" \
        --configs "$config_path" \
        --iteration 14000 \
        --split test \
        --warmup 3 \
        --frames 2 \
        --trials 1 \
        --workload-name flame_steak \
        --execution-mode tacker \
        --qualification-mode \
        --qualification-profile "${profile_root}/full_mlp.qualification.json" \
        --metadata "${output_root}/mixed_render_deform_mlp.metadata.json" \
        --quiet \
    > "${output_root}/mixed_render_deform_mlp.log" 2>&1

for label in standalone_deform_mlp solo_raster_render mixed_render_deform_mlp; do
    "$ncu_bin" --import "${output_root}/${label}.ncu-rep" \
        --page details --csv > "${output_root}/${label}.details.csv" 2>&1
    "$ncu_bin" --import "${output_root}/${label}.ncu-rep" \
        --page raw --csv > "${output_root}/${label}.raw.csv" 2>&1
done

timeout 2 tegrastats --interval 1000 > "${output_root}/tegrastats_after.txt" 2>&1 || true
date --iso-8601=seconds > "${output_root}/completed_at.txt"
chmod -R a+rX "$output_root"
echo "NCU kernel suite complete: $output_root"
