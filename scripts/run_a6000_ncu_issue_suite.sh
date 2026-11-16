#!/usr/bin/env bash
set -euo pipefail
umask 022

if [[ $# -ne 3 ]]; then
    echo "usage: $0 PROJECT_ROOT OUTPUT_ROOT PHYSICAL_GPU" >&2
    exit 2
fi

if [[ $(id -u) -ne 0 ]]; then
    echo "A6000 NCU collection must run as root" >&2
    exit 2
fi

project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
physical_gpu="$3"

ncu_bin=${NCU_BIN:-/usr/local/cuda-12.4/bin/ncu}
python_bin=${FOURDGS_PYTHON:-/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python}
model_path=${FOURDGS_MODEL_PATH:-/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak}
source_path=${FOURDGS_SOURCE_PATH:-/data/qyfeng/datasets/n3dv/flame_steak}
config_path=${FOURDGS_CONFIG_PATH:-${project_root}/arguments/dynerf/flame_steak.py}
qualification_profile=${FOURDGS_QUALIFICATION_PROFILE:-/data/qyfeng/4DGaussians-flame-steak-full/benchmarks/results/full_mlp_graph_formal_20260925/full_mlp_legacy_pb0.qualification.json}
driver_preload=/usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06

for required in \
    "$ncu_bin" \
    "$python_bin" \
    "$model_path" \
    "$source_path" \
    "$config_path" \
    "$qualification_profile" \
    "${project_root}/tacker_ext/profile_deform_mlp_ncu.py" \
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
mkdir -p "$output_root"

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export LD_PRELOAD="$driver_preload"
export PYTHONPATH="${project_root}/tacker_ext:${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}/submodules/simple-knn:${project_root}"
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99
export FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d
export FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503

date --iso-8601=seconds > "${output_root}/started_at.txt"
/usr/bin/nvidia-smi > "${output_root}/nvidia_smi_before.txt"
ps -eo user,pid,etime,cmd > "${output_root}/processes_before.txt"

common_ncu=(
    "$ncu_bin"
    --target-processes all
    --replay-mode kernel
    --set full
    --launch-count 1
    --force-overwrite
)

cd "$project_root"

echo "[$(date --iso-8601=seconds)] standalone_deform_mlp on physical GPU ${physical_gpu}"
"${common_ncu[@]}" \
    --kernel-name regex:tacker_deform_mlp_delta_v1 \
    --launch-skip 3 \
    --export "${output_root}/standalone_deform_mlp" \
    "$python_bin" "${project_root}/tacker_ext/profile_deform_mlp_ncu.py" \
        --mode profile --rows 111525 --persistent-blocks 0 --warmup 3 \
    > "${output_root}/standalone_deform_mlp.log" 2>&1

echo "[$(date --iso-8601=seconds)] solo_raster_render on physical GPU ${physical_gpu}"
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

echo "[$(date --iso-8601=seconds)] mixed_render_deform_mlp on physical GPU ${physical_gpu}"
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
        --qualification-profile "$qualification_profile" \
        --metadata "${output_root}/mixed_render_deform_mlp.metadata.json" \
        --quiet \
    > "${output_root}/mixed_render_deform_mlp.log" 2>&1

for label in standalone_deform_mlp solo_raster_render mixed_render_deform_mlp; do
    "$ncu_bin" --import "${output_root}/${label}.ncu-rep" \
        --page details --csv > "${output_root}/${label}.details.csv" 2>&1
    "$ncu_bin" --import "${output_root}/${label}.ncu-rep" \
        --page raw --csv > "${output_root}/${label}.raw.csv" 2>&1
done

/usr/bin/nvidia-smi > "${output_root}/nvidia_smi_after.txt"
date --iso-8601=seconds > "${output_root}/completed_at.txt"
chmod -R a+rX "$output_root"
if [[ -n ${SUDO_USER:-} && ${SUDO_USER} != root ]]; then
    chown -R "$SUDO_USER": "$output_root"
fi
echo "A6000 NCU issue suite complete: $output_root"
