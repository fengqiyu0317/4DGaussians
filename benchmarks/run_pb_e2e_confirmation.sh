#!/usr/bin/env bash
set -euo pipefail

project_root=/home/qyfeng/full_mlp_cta_profile_20260920_v1
python_executable=/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python
profile_dir=/data/qyfeng/full_mlp_partition_validation/20260921_pb_e2e_validation_v1/profiles
output_dir=/data/qyfeng/full_mlp_partition_validation/20260921_pb_e2e_validation_v1/confirmation
cuda_compat=/usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06
nvml_compat=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06

mkdir -p "${output_dir}"
echo "GPU 2 preflight:"
env LD_PRELOAD="${nvml_compat}" /usr/bin/nvidia-smi -i 2 \
    --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader
compute_processes="$({
    env LD_PRELOAD="${nvml_compat}" /usr/bin/nvidia-smi -i 2 \
        --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
} || true)"
if [[ -n "${compute_processes}" ]]; then
    echo "Refusing to benchmark occupied GPU 2:" >&2
    echo "${compute_processes}" >&2
    exit 2
fi
echo "GPU 2 has no compute processes."

order=(4480 4800 4800 4480)
for index in "${!order[@]}"; do
    pb="${order[${index}]}"
    run_number=$((index + 1))
    output="${output_dir}/confirm-$(printf '%02d' "${run_number}")-pb${pb}.json"
    echo "START run=${run_number} pb=${pb} trials=5"
    (
        cd "${project_root}"
        env \
            CUDA_VISIBLE_DEVICES=2 \
            LD_PRELOAD="${cuda_compat}" \
            PYTHONPATH="${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}" \
            "${python_executable}" scripts/run_profile_render_sealed.py \
            --model_path /data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak \
            --source_path /data/qyfeng/datasets/n3dv/flame_steak \
            --configs "${project_root}/arguments/dynerf/flame_steak.py" \
            --iteration 14000 \
            --split test \
            --warmup 10 \
            --frames 50 \
            --trials 5 \
            --execution-mode tacker \
            --qualification-mode \
            --qualification-profile "${profile_dir}/full_mlp_pb${pb}.json" \
            --workload-name flame_steak \
            --metadata "${output}" \
            --quiet
    )
    "${python_executable}" - "${output}" "${pb}" <<'PY'
import json
import sys

path, expected_pb = sys.argv[1], int(sys.argv[2])
document = json.load(open(path, encoding="utf-8"))
trials = document.get("trials", [])
checks = (
    document.get("passed") is True,
    document.get("actual_execution_mode") == "tacker",
    document.get("persistent_blocks") == expected_pb,
    len(trials) == 5,
    all(trial.get("actual_execution_mode") == "tacker" for trial in trials),
    all(trial.get("fallback_reason") is None for trial in trials),
)
if not all(checks):
    raise SystemExit("measurement validation failed: {}".format(path))
print(
    "DONE pb={} median_fps={:.6f} trial_fps={}".format(
        expected_pb,
        document["median_throughput_fps"],
        ",".join("{:.6f}".format(trial["throughput_fps"]) for trial in trials),
    )
)
PY
done
