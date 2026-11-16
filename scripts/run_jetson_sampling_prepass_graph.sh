#!/usr/bin/env bash
# All new scripts, results, temporary files and caches stay on /data/qyfeng.
set -euo pipefail
[[ $# -ge 3 && $# -le 6 ]] || { echo 'usage: RUNTIME_ROOT DRIVER_PATH OUTPUT_ROOT [default|default-node|high] [ABBA_BLOCKS] [REPEATS]'; exit 2; }
runtime_root=$(realpath "$1")
driver=$(realpath "$2")
output_root=$(realpath -m "$3")
priority=${4:-default}
abba_blocks=${5:-4}
repeats=${6:-40}
[[ "$priority" == default || "$priority" == default-node || "$priority" == high ]] || { echo 'invalid priority policy'; exit 2; }
benchmark_priority=$priority
backend_args=()
if [[ "$priority" == default-node ]]; then
    benchmark_priority=default
    backend_args=(--node-priority-backend)
fi
[[ "$abba_blocks" =~ ^[1-9][0-9]*$ && "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo 'invalid positive count'; exit 2; }
[[ "$driver" == /data/qyfeng/* && "$output_root" == /data/qyfeng/* ]] || { echo 'new experiment files must be on /data/qyfeng'; exit 2; }
priority_helper="$(dirname "$driver")/sampling_prepass_priority.py"
[[ -f "$priority_helper" ]] || { echo 'priority helper missing'; exit 2; }
[[ ! -e "$output_root" ]] || { echo 'refusing to overwrite results'; exit 2; }
mkdir -p "$output_root"
# Sealed CUDA binary snapshots can be large; keep only this process's temporary
# files on the data disk, never purge shared /tmp or alter global settings.
task_tmp=$(mktemp -d /data/qyfeng/sampling-prepass-tmp-XXXXXX)
export TMPDIR="$task_tmp"
export XDG_CACHE_HOME="$task_tmp/cache" CUDA_CACHE_PATH="$task_tmp/cuda-cache"
export TORCH_HOME="$task_tmp/torch" TORCH_EXTENSIONS_DIR="$task_tmp/torch-extensions"
export MPLCONFIGDIR="$task_tmp/matplotlib"
printf '%s\n' "$task_tmp" > "$output_root/temporary_root.txt"
qualification=/data/qyfeng/4DGaussians-tacker-v12-split-20261001/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json
collector="$runtime_root/scripts/sampling_kernel_activity.so"
export PYTHONPATH="$runtime_root/_python_deps_compat:$runtime_root/tacker_ext:$runtime_root/submodules/depth-diff-gaussian-rasterization:$runtime_root/submodules/simple-knn:$runtime_root:/home/jetson/.local/lib/python3.10/site-packages"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 TACKER_GPU_NAME=Orin TACKER_CUDA_ARCH=sm_87
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99 FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
export TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1 TACKER_HEXPLANE_LOWREG=0
export TACKER_HEXPLANE_FUSED_REDUCTION=1 TACKER_HEXPLANE_REDUCTION_THREADS=256
export TACKER_EXPERIMENTAL_RASTER_HEXPLANE=0 TACKER_EXPERIMENTAL_GREEN_CONTEXTS=0 TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF=0
unset LD_PRELOAD TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11
unset TACKER_MIXED_V12_SPLIT_TILE TACKER_MIXED_V12_SPLIT_FRACTION TACKER_MIXED_V12_PREPASS_PHYSICAL_BLOCKS TACKER_RASTER_PREPASS_MAX_CTAS
unset TACKER_PROFILE_MIXED_PIPELINE_JSONL TACKER_PROFILE_MIXED_V12_SPLIT_JSONL TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE
unset TACKER_EXPERIMENTAL_RASTER_HEXPLANE_MODE TACKER_HEXPLANE_SAMPLE_THREADS
sha256sum "$driver" "$priority_helper" "$0" "$runtime_root/profile_render.py" "$runtime_root/scripts/run_profile_render_sealed.py" \
    "$runtime_root/gaussian_renderer/tacker_pipeline.py" "$runtime_root/scene/hexplane.py" \
    "$runtime_root/tacker_ext/tacker_4dgs_head/_C.cpython-310-aarch64-linux-gnu.so" \
    "$runtime_root/tacker_ext/tacker_4dgs_head/_hexplane.cpython-310-aarch64-linux-gnu.so" \
    "$runtime_root/submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-310-aarch64-linux-gnu.so" \
    "$qualification" "$collector" > "$output_root/runtime.sha256"
env | grep -E '^(TACKER_[A-Z0-9_]*|CUDA_VISIBLE_DEVICES|FOURDGS_[A-Z0-9_]*|PYTHONPATH|TMPDIR|XDG_CACHE_HOME|CUDA_CACHE_PATH|TORCH_HOME|TORCH_EXTENSIONS_DIR|MPLCONFIGDIR)=' | sort > "$output_root/environment.txt"
check_idle() {
    local stem=$1
    sha256sum --check "$output_root/runtime.sha256" > "$output_root/$stem.source_check.txt"
    ps -eo pid,user,comm,args > "$output_root/$stem.processes.txt"
    free -m > "$output_root/$stem.memory.txt"
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > "$output_root/$stem.gpu_processes.csv"
    /usr/bin/python3 -c 'import csv,sys; rows=list(csv.reader(open(sys.argv[1])))[1:]; assert not any(r and r[0].strip().isdigit() for r in rows),rows' "$output_root/$stem.gpu_processes.csv"
    timeout 3 tegrastats --interval 1000 > "$output_root/$stem.tegrastats.txt" 2>&1 || [[ $? == 124 ]]
    /usr/bin/python3 -c 'import re,sys; s=open(sys.argv[1]).read(); a=re.findall(r"GR3D_FREQ (\d+)%",s); assert a and all(x=="0" for x in a),s' "$output_root/$stem.tegrastats.txt"
    /usr/bin/python3 -c 'import sys; sys.path.insert(0,sys.argv[1]); from analyze_sampling_kernel_activity import foreign_python_processes; bad=foreign_python_processes(open(sys.argv[2]).read()); assert not bad,bad' "$runtime_root/scripts" "$output_root/$stem.processes.txt"
    local freq=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
    [[ "$(< "$freq/min_freq")" == 1300500000 && "$(< "$freq/max_freq")" == 1300500000 && "$(< "$freq/cur_freq")" == 1300500000 ]]
}
check_idle before
date --iso-8601=seconds > "$output_root/started.txt"
sudo -n jetson_clocks --show > "$output_root/clocks.txt"
command=(/usr/bin/python3 "$driver" --runtime-root "$runtime_root"
    --benchmark-output "$output_root/benchmark.json" --abba-blocks "$abba_blocks" --repeats "$repeats" --bench-warmup 10
    --sampling-priority "$benchmark_priority" "${backend_args[@]}"
    --trace-library "$collector" --trace-repeats 8
    --model_path /data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak
    --source_path /data/qyfeng/datasets/n3dv/flame_steak_4dgs_min
    --configs "$runtime_root/arguments/dynerf/flame_steak.py" --iteration 14000
    --split test --warmup 20 --frames 5 --trials 1 --workload-name flame_steak
    --execution-mode tacker --qualification-mode --qualification-profile "$qualification"
    --quiet --metadata "$output_root/renderer.metadata.json")
printf '%q ' "${command[@]}" > "$output_root/command.txt"
(
    while true; do date --iso-8601=ns; ps -eo pid,user,comm,args; sleep 1; done
) > "$output_root/during_processes.txt" &
audit_pid=$!
trap 'kill "$audit_pid" 2>/dev/null || true' EXIT
cd "$runtime_root"
timeout 900 "${command[@]}" > "$output_root/run.log" 2>&1
kill "$audit_pid"
wait "$audit_pid" || true
trap - EXIT
/usr/bin/python3 -c 'import sys; sys.path.insert(0,sys.argv[1]); from analyze_sampling_kernel_activity import foreign_python_processes; bad=[s for s in foreign_python_processes(open(sys.argv[2]).read()) if sys.argv[3] not in s]; assert not bad,bad' "$runtime_root/scripts" "$output_root/during_processes.txt" "$driver"
check_idle after
date --iso-8601=seconds > "$output_root/completed.txt"
echo "DONE $output_root"
