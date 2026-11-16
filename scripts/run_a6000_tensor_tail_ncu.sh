#!/usr/bin/env bash
set -euo pipefail
umask 022

if [[ $# -ne 3 ]]; then
    echo "usage: $0 EXPERIMENT_ROOT OUTPUT_ROOT PHYSICAL_GPU" >&2
    exit 2
fi
if [[ $(id -u) -ne 0 ]]; then
    echo "NCU performance-counter collection must run as root" >&2
    exit 2
fi

experiment_root=$(realpath "$1")
output_root=$(realpath -m "$2")
physical_gpu="$3"
python_bin=${FOURDGS_PYTHON:-/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python}
ncu_bin=${NCU_BIN:-/usr/local/cuda-12.4/bin/ncu}
driver_preload=/usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06
profile_script=${experiment_root}/tacker_ext/profile_deform_mlp_tensor_tail_v3.py

for required in "$python_bin" "$ncu_bin" "$profile_script"; do
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
export PYTHONPATH="${experiment_root}/tacker_ext"
export TACKER_CUDA_ARCH=sm_86

date --iso-8601=seconds > "${output_root}/started_at.txt"
/usr/bin/nvidia-smi > "${output_root}/nvidia_smi_before.txt"
ps -eo user,pid,etime,cmd > "${output_root}/processes_before.txt"

common=(
    "$ncu_bin"
    --target-processes all
    --replay-mode kernel
    --set full
    --launch-skip 3
    --launch-count 1
    --force-overwrite
)

cd "${experiment_root}/tacker_ext"
"${common[@]}" \
    --kernel-name regex:tacker_deform_mlp_delta_v1 \
    --export "${output_root}/deform_mlp_v1" \
    "$python_bin" "$profile_script" \
        --mode profile --variant v1 --rows 111525 --warmup 3 \
    > "${output_root}/deform_mlp_v1.log" 2>&1

"${common[@]}" \
    --kernel-name regex:tacker_deform_mlp_tensor_tail_v3 \
    --export "${output_root}/deform_mlp_tensor_tail_v3" \
    "$python_bin" "$profile_script" \
        --mode profile --variant v3 --rows 111525 --warmup 3 \
    > "${output_root}/deform_mlp_tensor_tail_v3.log" 2>&1

for label in deform_mlp_v1 deform_mlp_tensor_tail_v3; do
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
echo "Tensor-tail NCU comparison complete: $output_root"
