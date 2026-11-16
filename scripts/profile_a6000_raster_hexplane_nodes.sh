#!/usr/bin/env bash
# Node traces are diagnostics, never the source of unprofiled FPS claims.
set -euo pipefail
[[ $# -eq 4 ]] || { echo 'usage: PROJECT_ROOT OUTPUT_ROOT PHYSICAL_GPU RATIO' >&2; exit 2; }
project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
physical_gpu=$3
ratio=$4
[[ "$physical_gpu" =~ ^[0-9]+$ && "$ratio" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ ! -e "$output_root" ]] || exit 2
mkdir -p "$output_root"
python_bin=/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python
export CUDA_VISIBLE_DEVICES="$physical_gpu" TACKER_CUDA_ARCH=sm_86 TACKER_GPU_NAME='NVIDIA RTX A6000'
export PYTHONPATH="$project_root/tacker_ext:$project_root/submodules/depth-diff-gaussian-rasterization:$project_root/submodules/simple-knn:$project_root"
export TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1 TACKER_RASTER_HEXPLANE_TASKS_PER_CTA="$ratio" TACKER_HEXPLANE_LOWREG=0
unset LD_PRELOAD TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11
unset TACKER_MIXED_V12_SPLIT_TILE TACKER_MIXED_V12_SPLIT_FRACTION TACKER_MIXED_V12_PREPASS_PHYSICAL_BLOCKS
unset TACKER_PROFILE_MIXED_PIPELINE_JSONL TACKER_PROFILE_MIXED_V12_SPLIT_JSONL TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE
for mode in baseline lowreg separate fused; do
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$output_root/$mode.gpu_processes.csv"
    nvidia-smi -i "$physical_gpu" --query-gpu=uuid,memory.used,utilization.gpu --format=csv > "$output_root/$mode.gpu.csv"
    "$python_bin" -c '
import csv,sys
r,m=sys.argv[1:]; gpu=list(csv.reader(open(r+"/"+m+".gpu.csv")))[1]; rows=list(csv.reader(open(r+"/"+m+".gpu_processes.csv")))[1:]
assert not any(row and row[0].strip()==gpu[0].strip() for row in rows),rows
assert int(gpu[1].split()[0])<256 and int(gpu[2].split()[0])<=5,gpu
' "$output_root" "$mode"
    export TACKER_EXPERIMENTAL_RASTER_HEXPLANE="$mode" TACKER_HEXPLANE_LOWREG=0
    [[ "$mode" != baseline && "$mode" != lowreg ]] || export TACKER_EXPERIMENTAL_RASTER_HEXPLANE=0
    [[ "$mode" != lowreg ]] || export TACKER_HEXPLANE_LOWREG=1
    echo "NSYS $mode"
    /usr/local/cuda/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
        --export=sqlite --output="$output_root/$mode" \
        "$python_bin" "$project_root/scripts/run_profile_render_sealed.py" \
        --model_path /data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak \
        --source_path /data/qyfeng/datasets/n3dv/flame_steak \
        --configs "$project_root/arguments/dynerf/flame_steak.py" \
        --iteration 14000 --split test --warmup 20 --frames 10 --trials 1 \
        --workload-name flame_steak --execution-mode tacker --qualification-mode \
        --qualification-profile /data/qyfeng/4DGaussians-flame-steak-full/benchmarks/results/full_mlp_graph_formal_20260925/full_mlp_overlap_graph_pb0.qualification.json \
        --quiet --metadata "$output_root/$mode.metadata.json" > "$output_root/$mode.log" 2>&1
    "$python_bin" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"] and d["actual_execution_mode"]=="tacker" and d["qualification_mode_executed"]; assert d["selected_variant_id"]=="full_mlp_overlap_graph_pb0"; assert d["pipeline_execution_counts"]["mixed_launches"]==9' "$output_root/$mode.metadata.json"
done
echo "NSYS DONE $output_root"
