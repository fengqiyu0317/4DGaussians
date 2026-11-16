#!/usr/bin/env bash
# Profile the existing, sealed v12 + fused reduction runtime; never rebuild it.
# Kernel replay is isolated per-launch measurement, NOT concurrent attribution.
set -euo pipefail
umask 022
[[ $# -ge 3 && $# -le 5 ]] || { echo 'usage: PROJECT_ROOT OUTPUT_ROOT inventory|kernels|range|baseline [SKIP=0] [COUNT=200]'; exit 2; }
[[ $(id -u) == 0 ]] || { echo 'root required for performance counters'; exit 2; }
project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
phase=$3
skip=${4:-0}
count=${5:-200}
[[ "$phase" == inventory || "$phase" == kernels || "$phase" == range || "$phase" == baseline ]] || exit 2
[[ "$skip" =~ ^[0-9]+$ && "$count" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ ! -e "$output_root" ]] || { echo 'refusing to overwrite output'; exit 2; }
mkdir -p "$output_root"
qualification=/data/qyfeng/4DGaussians-tacker-v12-split-20261001/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json
export PYTHONPATH="$project_root/_python_deps_compat:$project_root/tacker_ext:$project_root/submodules/depth-diff-gaussian-rasterization:$project_root/submodules/simple-knn:$project_root:/home/jetson/.local/lib/python3.10/site-packages"
export CUDA_VISIBLE_DEVICES=0 TACKER_GPU_NAME=Orin TACKER_CUDA_ARCH=sm_87
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99 FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
export TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1 TACKER_HEXPLANE_LOWREG=0
export TACKER_HEXPLANE_FUSED_REDUCTION=1 TACKER_HEXPLANE_REDUCTION_THREADS=256
export TACKER_EXPERIMENTAL_RASTER_HEXPLANE=0
export TACKER_EXPERIMENTAL_GREEN_CONTEXTS=0 TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF=0
unset LD_PRELOAD TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11
unset TACKER_MIXED_V12_SPLIT_TILE TACKER_MIXED_V12_SPLIT_FRACTION TACKER_MIXED_V12_PREPASS_PHYSICAL_BLOCKS TACKER_RASTER_PREPASS_MAX_CTAS
unset TACKER_PROFILE_MIXED_PIPELINE_JSONL TACKER_PROFILE_MIXED_V12_SPLIT_JSONL TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE
unset TACKER_EXPERIMENTAL_RASTER_HEXPLANE_MODE TACKER_HEXPLANE_SAMPLE_THREADS

sha256sum "$project_root/profile_render.py" "$project_root/scripts/run_profile_render_sealed.py" \
    "$project_root/gaussian_renderer/tacker_pipeline.py" "$project_root/scene/hexplane.py" \
    "$project_root/tacker_ext/csrc/hexplane_reduce.cu" \
    "$project_root/tacker_ext/tacker_4dgs_head/_C.cpython-310-aarch64-linux-gnu.so" \
    "$project_root/tacker_ext/tacker_4dgs_head/_hexplane.cpython-310-aarch64-linux-gnu.so" \
    "$project_root/submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-310-aarch64-linux-gnu.so" \
    "$qualification" > "$output_root/runtime.sha256"
env | grep -E '^(TACKER_[A-Z0-9_]*|CUDA_VISIBLE_DEVICES|FOURDGS_[A-Z0-9_]*|PYTHONPATH)=' | sort > "$output_root/environment.txt"

check_idle() {
    local stem=$1
    sha256sum --check "$output_root/runtime.sha256" > "$output_root/$stem.source_check.txt"
    ps -eo pid,user,comm,args > "$output_root/$stem.processes.txt"
    free -m > "$output_root/$stem.memory.txt"
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > "$output_root/$stem.gpu_processes.csv"
    /usr/bin/python3 -c 'import csv,sys; rows=list(csv.reader(open(sys.argv[1])))[1:]; assert not any(r and r[0].strip().isdigit() for r in rows),rows' "$output_root/$stem.gpu_processes.csv"
    timeout 3 tegrastats --interval 1000 > "$output_root/$stem.tegrastats.txt" 2>&1 || [[ $? == 124 ]]
    /usr/bin/python3 -c 'import re,sys; s=open(sys.argv[1]).read(); a=re.findall(r"GR3D_FREQ (\d+)%",s); assert a and all(x=="0" for x in a),s' "$output_root/$stem.tegrastats.txt"
    /usr/bin/python3 -c 'import sys; sys.path.insert(0,sys.argv[1]); from analyze_sampling_kernel_activity import foreign_python_processes; bad=foreign_python_processes(open(sys.argv[2]).read()); assert not bad,bad' "$project_root/scripts" "$output_root/$stem.processes.txt"
    local freq=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
    [[ "$(< "$freq/min_freq")" == 1300500000 && "$(< "$freq/max_freq")" == 1300500000 && "$(< "$freq/cur_freq")" == 1300500000 ]] || exit 3
}

check_idle before
date --iso-8601=seconds > "$output_root/started.txt"
/usr/local/cuda/bin/ncu --version > "$output_root/ncu_version.txt"
jetson_clocks --show > "$output_root/clocks.txt"
command=(/usr/bin/python3 "$project_root/scripts/run_profile_render_sealed.py"
    --model_path /data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak
    --source_path /data/qyfeng/datasets/n3dv/flame_steak_4dgs_min
    --configs "$project_root/arguments/dynerf/flame_steak.py" --iteration 14000
    --split test --warmup 20 --frames 3 --trials 1 --workload-name flame_steak
    --execution-mode tacker --qualification-mode --qualification-profile "$qualification"
    --quiet --metadata "$output_root/renderer.metadata.json")
ncu=(/usr/local/cuda/bin/ncu --target-processes all
    --clock-control none --export "$output_root/report")
[[ "$phase" == range ]] || ncu+=(--profile-from-start off)
if [[ "$phase" == inventory ]]; then
    ncu+=(--replay-mode kernel --graph-profiling node --section LaunchStats --launch-count "$count")
elif [[ "$phase" == kernels ]]; then
    ncu+=(--replay-mode kernel --graph-profiling node
        --launch-skip-before-match "$skip" --launch-count "$count"
        --section LaunchStats --section Occupancy --section SpeedOfLight
        --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis
        --section SchedulerStats --section WarpStateStats)
elif [[ "$phase" == range ]]; then
    # Graph API capture is unsupported by range replay; app-range preserves it.
    ncu+=(--replay-mode app-range --launch-count 1 --cache-control none
        --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__cycles_active.avg,sm__cycles_elapsed.avg,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,l1tex__throughput.avg.pct_of_peak_sustained_active,lts__throughput.avg.pct_of_peak_sustained_elapsed)
    # The standard profiler provides one start/stop range for the whole trial.
    for i in "${!command[@]}"; do
        [[ "${command[$i]}" != --frames ]] || command[$((i+1))]=20
    done
else
    for i in "${!command[@]}"; do
        [[ "${command[$i]}" != --frames ]] || command[$((i+1))]=${5:-50}
        [[ "${command[$i]}" != --trials ]] || command[$((i+1))]=2
    done
fi
printf '%q ' "${ncu[@]}" "${command[@]}" > "$output_root/command.txt"
(
    while true; do date --iso-8601=ns; ps -eo pid,user,comm,args; sleep 1; done
) > "$output_root/during_processes.txt" &
audit_pid=$!
trap 'kill "$audit_pid" 2>/dev/null || true' EXIT
cd "$project_root"
if [[ "$phase" == baseline ]]; then
    timeout 900 "${command[@]}" > "$output_root/run.log" 2>&1
else
    timeout 900 "${ncu[@]}" "${command[@]}" > "$output_root/run.log" 2>&1
fi
kill "$audit_pid"
wait "$audit_pid" || true
trap - EXIT
/usr/bin/python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"] and d["actual_execution_mode"]=="tacker" and d["qualification_mode_executed"] and d["selected_variant_id"]=="full_mlp_overlap_graph_pb0"; assert d["tacker_fallback_reason"] is None and d["post_run"]["byte_stability"]["verified"] and not d["green_contexts"]["enabled"]; assert (d["gaussian_count"],d["image_width"],d["image_height"])==(111525,1352,1014); print("GRAPH V12 UNPARTITIONED VERIFIED",d["throughput_fps"])' "$output_root/renderer.metadata.json"
if [[ "$phase" != baseline ]]; then
    /usr/local/cuda/bin/ncu --import "$output_root/report.ncu-rep" --page raw --csv > "$output_root/report.raw.csv" 2>&1
    /usr/local/cuda/bin/ncu --import "$output_root/report.ncu-rep" --page details --csv > "$output_root/report.details.csv" 2>&1
fi
check_idle after
date --iso-8601=seconds > "$output_root/completed.txt"
chmod -R a+rX "$output_root"
echo "DONE $phase $output_root"
