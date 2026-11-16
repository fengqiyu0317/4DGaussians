#!/usr/bin/env bash
set -euo pipefail
umask 022

if [[ $# -ne 2 ]]; then
    echo "usage: $0 EXPERIMENT_ROOT OUTPUT_ROOT" >&2
    exit 2
fi
if [[ $(id -u) -ne 0 ]]; then
    echo "Jetson NCU performance-counter collection must run as root" >&2
    exit 2
fi

experiment_root=$(realpath "$1")
output_root=$(realpath -m "$2")
python_bin=/usr/bin/python3
ncu_bin=/usr/local/cuda/bin/ncu
profile_script=${experiment_root}/tacker_ext/profile_deform_mlp_tensor_tail_v4.py

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

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${experiment_root}/tacker_ext:/home/jetson/.local/lib/python3.10/site-packages"
export TACKER_CUDA_ARCH=sm_87

date --iso-8601=seconds > "${output_root}/started_at.txt"
jetson_clocks --show > "${output_root}/jetson_clocks.txt" 2>&1
timeout 2 tegrastats --interval 1000 > "${output_root}/tegrastats_before.txt" 2>&1 || true
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
    --kernel-name regex:tacker_deform_mlp_tensor_tail_v3 \
    --export "${output_root}/deform_mlp_tensor_tail_v3" \
    "$python_bin" "$profile_script" \
        --mode profile --variant v3 --rows 111525 --warmup 3 \
    > "${output_root}/deform_mlp_tensor_tail_v3.log" 2>&1

"${common[@]}" \
    --kernel-name regex:tacker_deform_mlp_tensor_tail_padded_v4 \
    --export "${output_root}/deform_mlp_tensor_tail_padded_v4" \
    "$python_bin" "$profile_script" \
        --mode profile --variant v4 --rows 111525 --warmup 3 \
    > "${output_root}/deform_mlp_tensor_tail_padded_v4.log" 2>&1

for label in deform_mlp_tensor_tail_v3 deform_mlp_tensor_tail_padded_v4; do
    "$ncu_bin" --import "${output_root}/${label}.ncu-rep" \
        --page details --csv > "${output_root}/${label}.details.csv" 2>&1
    "$ncu_bin" --import "${output_root}/${label}.ncu-rep" \
        --page raw --csv > "${output_root}/${label}.raw.csv" 2>&1
done

timeout 2 tegrastats --interval 1000 > "${output_root}/tegrastats_after.txt" 2>&1 || true
date --iso-8601=seconds > "${output_root}/completed_at.txt"
chmod -R a+rX "$output_root"
echo "Jetson Tensor-tail padding NCU comparison complete: $output_root"
