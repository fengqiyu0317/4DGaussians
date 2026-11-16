#!/usr/bin/env bash
# Read-only existing runtime; dedicated matched Deform library lives in the new experiment directory.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: EXPERIMENT_ROOT PROCESS_INDEX'; exit 2; }
experiment=$(realpath "$1")
index=$2
runtime=$(realpath "$experiment/runtime")
output="$experiment/run$index"
[[ "$experiment" == /data/qyfeng/independent-matched-graph-* && "$index" =~ ^[0-9]+$ ]] || exit 2
[[ ! -e "$output" ]] || { echo 'output already exists'; exit 2; }
mkdir -p "$output" "$output/tmp"
export TMPDIR="$output/tmp" XDG_CACHE_HOME="$output/tmp/cache" CUDA_CACHE_PATH="$output/tmp/cuda-cache"
export TORCH_HOME="$output/tmp/torch" TORCH_EXTENSIONS_DIR="$output/tmp/extensions" MPLCONFIGDIR="$output/tmp/matplotlib"
export PYTHONPATH="$runtime/_python_deps_compat:$runtime/tacker_ext:$runtime/submodules/depth-diff-gaussian-rasterization:$runtime/submodules/simple-knn:$runtime:/home/jetson/.local/lib/python3.10/site-packages"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 TACKER_GPU_NAME=Orin TACKER_CUDA_ARCH=sm_87
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99 FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
export TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1 TACKER_HEXPLANE_LOWREG=0
export TACKER_HEXPLANE_FUSED_REDUCTION=1 TACKER_HEXPLANE_REDUCTION_THREADS=256
export TACKER_EXPERIMENTAL_RASTER_HEXPLANE=0 TACKER_EXPERIMENTAL_GREEN_CONTEXTS=0 TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF=0
unset LD_PRELOAD TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11
unset TACKER_MIXED_V12_SPLIT_TILE TACKER_MIXED_V12_SPLIT_FRACTION TACKER_MIXED_V12_PREPASS_PHYSICAL_BLOCKS
unset TACKER_PROFILE_MIXED_PIPELINE_JSONL TACKER_PROFILE_MIXED_V12_SPLIT_JSONL TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE
unset TACKER_PROFILE_MIXED_BALANCED_SUBGROUP_MODE TACKER_PROFILE_MIXED_BALANCED_SUBGROUP_JSONL TACKER_BENCHMARK_MIXED_MODE
qualification=/data/qyfeng/4DGaussians-tacker-v12-split-20261001/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json
sha256sum "$runtime/gaussian_renderer/tacker_pipeline.py" "$runtime/scene/hexplane.py" "$runtime/profile_render.py" \
  "$runtime/submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-310-aarch64-linux-gnu.so" \
  "$experiment/benchmark_independent_matched_graph.py" "$experiment/libdeform_matched.so" "$runtime/tacker_ext/include/deform_mlp_device.cuh" "$qualification" > "$output/runtime.sha256"
check_idle() {
  local stem=$1
  ps -eo pid,user,comm,args > "$output/$stem.processes.txt"
  free -m > "$output/$stem.memory.txt"
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > "$output/$stem.gpu.csv"
  /usr/bin/python3 -c 'import csv,sys; rows=list(csv.reader(open(sys.argv[1])))[1:]; assert not any(r and r[0].strip().isdigit() for r in rows), rows' "$output/$stem.gpu.csv"
  timeout 3 tegrastats --interval 1000 > "$output/$stem.tegrastats.txt" 2>&1 || [[ $? == 124 ]]
  /usr/bin/python3 -c 'import re,sys; s=open(sys.argv[1]).read(); a=re.findall(r"GR3D_FREQ (\d+)%",s); assert a and all(x=="0" for x in a),s' "$output/$stem.tegrastats.txt"
  /usr/bin/python3 -c 'import sys; sys.path.insert(0,sys.argv[1]); from analyze_sampling_kernel_activity import foreign_python_processes; bad=foreign_python_processes(open(sys.argv[2]).read()); assert not bad,bad' "$runtime/scripts" "$output/$stem.processes.txt"
  local freq=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
  [[ "$(< "$freq/min_freq")" == 1300500000 && "$(< "$freq/max_freq")" == 1300500000 && "$(< "$freq/cur_freq")" == 1300500000 ]]
}
check_idle before
date --iso-8601=seconds > "$output/started.txt"
(
  while true; do date --iso-8601=ns; cat /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq; ps -eo pid,user,comm,args; sleep 1; done
) > "$output/during_processes_and_frequency.txt" &
audit_pid=$!
trap 'kill "$audit_pid" 2>/dev/null || true' EXIT
cd "$runtime"
timeout 600 /usr/bin/python3 "$experiment/benchmark_independent_matched_graph.py" \
  --runtime-root "$runtime" --benchmark-output "$output/benchmark.json" --rounds 6 --repeats 30 --bench-warmup 10 \
  --model_path /data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak \
  --source_path /data/qyfeng/datasets/n3dv/flame_steak_4dgs_min \
  --configs "$runtime/arguments/dynerf/flame_steak.py" --iteration 14000 --split test --warmup 10 --frames 5 --trials 1 \
  --workload-name flame_steak --execution-mode tacker --qualification-mode --qualification-profile "$qualification" \
  --quiet --metadata "$output/renderer.metadata.json" > "$output/run.log" 2>&1
kill "$audit_pid"
wait "$audit_pid" || true
trap - EXIT
sha256sum --check "$output/runtime.sha256" > "$output/runtime_after_check.txt"
/usr/bin/python3 -c 'import sys; sys.path.insert(0,sys.argv[1]); from analyze_sampling_kernel_activity import foreign_python_processes; bad=[s for s in foreign_python_processes(open(sys.argv[2]).read()) if "benchmark_independent_matched_graph.py" not in s]; assert not bad,bad' "$runtime/scripts" "$output/during_processes_and_frequency.txt"
check_idle after
date --iso-8601=seconds > "$output/completed.txt"
echo "RUN_DONE $output"
