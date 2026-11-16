#!/usr/bin/env bash
# Event-only original-v12 Graph diagnostic. Never changes clocks or services.
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: PROJECT_ROOT OUTPUT_ROOT cuda|pilot|confirm|coverage' >&2; exit 2; }
project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
phase=$3
[[ "$phase" == cuda || "$phase" == pilot || "$phase" == confirm || "$phase" == coverage ]] || exit 2
[[ ! -e "$output_root" ]] || { echo "output already exists: $output_root" >&2; exit 2; }
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
sha256sum "$project_root/scripts/profile_hexplane_overlap_events.py" "$project_root/scripts/run_profile_render_sealed.py" \
    "$project_root/scripts/run_jetson_sampling_overlap_events.sh" "$project_root/profile_render.py" \
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

if [[ "$phase" == cuda ]]; then
    check_idle cuda
    /usr/bin/python3 "$project_root/tacker_ext/tests/test_hexplane_overlap_events_cuda.py" > "$output_root/cuda.log" 2>&1
    check_idle final
    echo "CUDA EVENT PASS $output_root"
    exit 0
fi
if [[ "$phase" == pilot ]]; then
    modes=(off last all)
    frames=(50 50 50)
    trials=3
elif [[ "$phase" == confirm ]]; then
    modes=(off last all all last off off last all all last off)
    frames=(50 50 50 50 50 50 50 50 50 50 50 50)
    trials=8
else
    modes=(off all all off off all)
    frames=(12 12 24 24 36 36)
    trials=6
fi
for index in "${!modes[@]}"; do
    mode=${modes[$index]}
    count=${frames[$index]}
    stem=$(printf '%02d-%s-f%s' "$index" "$mode" "$count")
    check_idle "$stem"
    echo "START $stem $(date --iso-8601=seconds)"
    timeout 1200 /usr/bin/python3 "$project_root/scripts/profile_hexplane_overlap_events.py" \
        --event-mode "$mode" --event-output "$output_root/$stem.events.json" \
        --model_path /data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak \
        --source_path /data/qyfeng/datasets/n3dv/flame_steak_4dgs_min \
        --configs "$project_root/arguments/dynerf/flame_steak.py" --iteration 14000 \
        --split test --warmup 20 --frames "$count" --trials "$trials" --workload-name flame_steak \
        --execution-mode tacker --qualification-mode --qualification-profile "$qualification" \
        --quiet --metadata "$output_root/$stem.json" > "$output_root/$stem.log" 2>&1
    /usr/bin/python3 -c '
import json,sys
path,frames=sys.argv[1:]; d=json.load(open(path)); frames=int(frames)
assert d["passed"] and d["actual_execution_mode"]=="tacker" and d["qualification_mode_executed"]
assert d["selected_variant_id"]=="full_mlp_overlap_graph_pb0" and d["tacker_fallback_reason"] is None
assert d["post_run"]["byte_stability"]["verified"]
c=d["pipeline_execution_counts"]; assert c["mixed_launches"]==frames-1 and c["raster_prepasses"]==frames-1 and c["outputs"]==frames,c
print("FPS",d["throughput_fps"])
' "$output_root/$stem.json" "$count"
done
check_idle final
echo "DONE $output_root"
