#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: PROJECT_ROOT OUTPUT_ROOT pilot|confirm' >&2; exit 2; }
project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
phase=$3
[[ "$phase" == pilot || "$phase" == confirm ]] || exit 2
[[ ! -e "$output_root" ]] || exit 2
mkdir -p "$output_root"
qualification=/data/qyfeng/4DGaussians-tacker-v12-split-20261001/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json
export PYTHONPATH="$project_root/_python_deps_compat:$project_root/tacker_ext:$project_root/submodules/depth-diff-gaussian-rasterization:$project_root/submodules/simple-knn:$project_root:/home/jetson/.local/lib/python3.10/site-packages"
export CUDA_VISIBLE_DEVICES=0 TACKER_GPU_NAME=Orin TACKER_CUDA_ARCH=sm_87
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99 FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
export TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1 TACKER_EXPERIMENTAL_RASTER_HEXPLANE=0 TACKER_HEXPLANE_LOWREG=0
unset LD_PRELOAD TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11
unset TACKER_MIXED_V12_SPLIT_TILE TACKER_MIXED_V12_SPLIT_FRACTION TACKER_MIXED_V12_PREPASS_PHYSICAL_BLOCKS TACKER_RASTER_PREPASS_MAX_CTAS
unset TACKER_PROFILE_MIXED_PIPELINE_JSONL TACKER_PROFILE_MIXED_V12_SPLIT_JSONL TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE
sha256sum "$project_root/scripts/profile_sampling_kernel_activity.py" "$project_root/scripts/sampling_kernel_activity.cpp" \
    "$project_root/scripts/sampling_kernel_activity.so" "$project_root/scripts/run_jetson_sampling_kernel_activity.sh" \
    "$project_root/scripts/run_profile_render_sealed.py" "$project_root/profile_render.py" \
    "$project_root/gaussian_renderer/tacker_pipeline.py" "$project_root/scene/hexplane.py" \
    "$project_root/tacker_ext/tacker_4dgs_head/_C.cpython-310-aarch64-linux-gnu.so" \
    "$project_root/submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-310-aarch64-linux-gnu.so" \
    "$qualification" > "$output_root/runtime.sha256"
check_idle() {
    local stem=$1
    sha256sum --check "$output_root/runtime.sha256" > "$output_root/$stem.source_check.txt"
    ps -eo pid,user,comm,args > "$output_root/$stem.processes.txt"
    free -m > "$output_root/$stem.memory.txt"
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > "$output_root/$stem.gpu_processes.csv"
    /usr/bin/python3 -c 'import csv,sys; rows=list(csv.reader(open(sys.argv[1])))[1:]; assert not any(row and row[0].strip().isdigit() for row in rows),rows' "$output_root/$stem.gpu_processes.csv"
    timeout 3 tegrastats --interval 1000 > "$output_root/$stem.tegrastats.txt" 2>&1 || [[ $? == 124 ]]
    /usr/bin/python3 -c 'import sys; s=open(sys.argv[1]).read(); assert "GR3D_FREQ 0%" in s,s' "$output_root/$stem.tegrastats.txt"
    local freq=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
    [[ "$(< "$freq/min_freq")" == 1300500000 && "$(< "$freq/max_freq")" == 1300500000 && "$(< "$freq/cur_freq")" == 1300500000 ]] || exit 3
}
if [[ "$phase" == pilot ]]; then
    modes=(off on)
    frames=10
    trials=2
else
    modes=(off on on off off on on off)
    frames=50
    trials=4
fi
for index in "${!modes[@]}"; do
    mode=${modes[$index]}
    stem=$(printf '%02d-%s-f%s' "$index" "$mode" "$frames")
    check_idle "$stem"
    echo "START $stem $(date --iso-8601=seconds)"
    # Read-only host audit catches a second CUDA process starting mid-trial.
    # Record the full process list rather than relying on integrated NVML's N/A.
    (
        while true; do
            date --iso-8601=ns
            ps -eo pid,user,comm,args
            sleep 1
        done
    ) > "$output_root/$stem.during_processes.txt" &
    audit_pid=$!
    trap 'kill "$audit_pid" 2>/dev/null || true' EXIT
    timeout 1200 sudo -n env -u LD_PRELOAD PYTHONPATH="$PYTHONPATH" \
        CUDA_VISIBLE_DEVICES=0 TACKER_GPU_NAME=Orin TACKER_CUDA_ARCH=sm_87 \
        FOURDGS_SOURCE_COMMIT="$FOURDGS_SOURCE_COMMIT" FOURDGS_SOURCE_DIRTY=true \
        FOURDGS_RASTERIZER_COMMIT="$FOURDGS_RASTERIZER_COMMIT" FOURDGS_SIMPLE_KNN_COMMIT="$FOURDGS_SIMPLE_KNN_COMMIT" \
        TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1 TACKER_EXPERIMENTAL_RASTER_HEXPLANE=0 TACKER_HEXPLANE_LOWREG=0 \
        /usr/bin/python3 "$project_root/scripts/profile_sampling_kernel_activity.py" \
        --kernel-mode "$mode" --kernel-library "$project_root/scripts/sampling_kernel_activity.so" \
        --kernel-output "$output_root/$stem.activity.json" \
        --model_path /data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak \
        --source_path /data/qyfeng/datasets/n3dv/flame_steak_4dgs_min \
        --configs "$project_root/arguments/dynerf/flame_steak.py" --iteration 14000 \
        --split test --warmup 20 --frames "$frames" --trials "$trials" --workload-name flame_steak \
        --execution-mode tacker --qualification-mode --qualification-profile "$qualification" \
        --quiet --metadata "$output_root/$stem.json" > "$output_root/$stem.log" 2>&1
    # Only stop the read-only audit process spawned immediately above.
    kill "$audit_pid"
    wait "$audit_pid" || true
    trap - EXIT
    sudo -n /usr/bin/python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"] and d["selected_variant_id"]=="full_mlp_overlap_graph_pb0" and d["post_run"]["byte_stability"]["verified"]; print("FPS",d["throughput_fps"])' "$output_root/$stem.json"
done
check_idle final
date --iso-8601=seconds > "$output_root/completed.txt"
echo "DONE $output_root"
