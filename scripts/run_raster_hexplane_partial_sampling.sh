#!/usr/bin/env bash
# Isolated, Graph-qualified partial-sampling experiment on A6000 or Jetson.
set -euo pipefail
[[ $# -eq 7 ]] || { echo 'usage: PROJECT_ROOT OUTPUT_ROOT a6000|jetson GPU RATIO cuda|quality|sweep|confirm|diagnostic|trace MIXED_PLANES' >&2; exit 2; }
project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
target=$3
physical_gpu=$4
ratio=$5
phase=$6
mixed_planes=$7
[[ "$physical_gpu" =~ ^[0-9]+$ && "$ratio" =~ ^[1-9][0-9]*$ && "$mixed_planes" =~ ^[1-9][0-9]*$ ]] || exit 2
((ratio <= 2147483647 && mixed_planes <= 12)) || exit 2
[[ "$phase" == cuda || "$phase" == quality || "$phase" == sweep || "$phase" == confirm || "$phase" == diagnostic || "$phase" == trace ]] || exit 2
[[ ! -e "$output_root" ]] || { echo "output already exists: $output_root" >&2; exit 2; }
if [[ "$target" == a6000 ]]; then
    python_bin=/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python
    qualification=/data/qyfeng/4DGaussians-flame-steak-full/benchmarks/results/full_mlp_graph_formal_20260925/full_mlp_overlap_graph_pb0.qualification.json
    model=/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak
    source=/data/qyfeng/datasets/n3dv/flame_steak
    architecture=sm_86
    export TACKER_GPU_NAME='NVIDIA RTX A6000'
    nsys_bin=/usr/local/cuda/bin/nsys
    export PYTHONPATH="$project_root/tacker_ext:$project_root/submodules/depth-diff-gaussian-rasterization:$project_root/submodules/simple-knn:$project_root"
    binary_tag=cpython-310-x86_64-linux-gnu
elif [[ "$target" == jetson ]]; then
    [[ "$physical_gpu" == 0 ]] || exit 2
    python_bin=/usr/bin/python3
    qualification=/data/qyfeng/4DGaussians-tacker-v12-split-20261001/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json
    model=/data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak
    source=/data/qyfeng/datasets/n3dv/flame_steak_4dgs_min
    architecture=sm_87
    export TACKER_GPU_NAME=Orin
    nsys_bin=/usr/local/bin/nsys
    export PYTHONPATH="$project_root/_python_deps_compat:$project_root/tacker_ext:$project_root/submodules/depth-diff-gaussian-rasterization:$project_root/submodules/simple-knn:$project_root:/home/jetson/.local/lib/python3.10/site-packages"
    binary_tag=cpython-310-aarch64-linux-gnu
else
    exit 2
fi
mkdir -p "$output_root"
export CUDA_VISIBLE_DEVICES="$physical_gpu" TACKER_CUDA_ARCH="$architecture"
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99 FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
export TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1 TACKER_RASTER_HEXPLANE_TASKS_PER_CTA="$ratio"
unset LD_PRELOAD TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11
unset TACKER_MIXED_V12_SPLIT_TILE TACKER_MIXED_V12_SPLIT_FRACTION TACKER_MIXED_V12_PREPASS_PHYSICAL_BLOCKS TACKER_RASTER_PREPASS_MAX_CTAS
unset TACKER_PROFILE_MIXED_PIPELINE_JSONL TACKER_PROFILE_MIXED_V12_SPLIT_JSONL TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE
sha256sum "$project_root/gaussian_renderer/tacker_pipeline.py" "$project_root/gaussian_renderer/raster_hexplane_graph.py" \
    "$project_root/profile_render.py" "$project_root/scripts/run_profile_render_sealed.py" "$project_root/scripts/validate_tacker_modes.py" \
    "$project_root/scene/hexplane.py" "$project_root/tacker_ext/tacker_4dgs_head/hexplane.py" \
    "$project_root/tacker_ext/tacker_4dgs_head/_C.$binary_tag.so" "$project_root/tacker_ext/tacker_4dgs_head/_hexplane.$binary_tag.so" \
    "$project_root/submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization/_C.$binary_tag.so" "$qualification" > "$output_root/runtime.sha256"

check_gpu_idle() {
    local stem=$1
    sha256sum --check "$output_root/runtime.sha256" > "$output_root/$stem.source_check.txt"
    ps -eo pid,user,comm,args > "$output_root/$stem.processes.txt"
    free -m > "$output_root/$stem.memory.txt"
    if [[ "$target" == a6000 ]]; then
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$output_root/$stem.gpu_processes.csv"
        nvidia-smi -i "$physical_gpu" --query-gpu=uuid,memory.used,utilization.gpu,pstate,clocks.sm,clocks.mem,temperature.gpu --format=csv > "$output_root/$stem.gpu.csv"
        "$python_bin" -c 'import csv,sys; r,s=sys.argv[1:]; g=list(csv.reader(open(r+"/"+s+".gpu.csv")))[1]; p=list(csv.reader(open(r+"/"+s+".gpu_processes.csv")))[1:]; assert not any(x and x[0].strip()==g[0].strip() for x in p),p; assert int(g[1].split()[0])<256 and int(g[2].split()[0])<=5,g' "$output_root" "$stem"
    else
        nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > "$output_root/$stem.gpu_processes.csv"
        "$python_bin" -c 'import csv,sys; p=list(csv.reader(open(sys.argv[1])))[1:]; assert not any(x and x[0].strip().isdigit() for x in p),p' "$output_root/$stem.gpu_processes.csv"
        timeout 2 tegrastats --interval 1000 > "$output_root/$stem.tegrastats.txt" 2>&1 || [[ $? -eq 124 ]]
        "$python_bin" -c 'import sys; assert "GR3D_FREQ 0%" in open(sys.argv[1]).read()' "$output_root/$stem.tegrastats.txt"
        local freq=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
        [[ "$(< "$freq/min_freq")" == 1300500000 && "$(< "$freq/max_freq")" == 1300500000 && "$(< "$freq/cur_freq")" == 1300500000 ]] || exit 3
    fi
}

if [[ "$phase" == cuda ]]; then
    check_gpu_idle cuda
    "$python_bin" "$project_root/tacker_ext/tests/test_raster_hexplane_mixed_cuda.py" > "$output_root/mixed.log" 2>&1
    check_gpu_idle final
    echo "CUDA PASS $output_root"
    exit 0
fi
if [[ "$phase" == quality ]]; then
    schedule=(fused separate)
elif [[ "$phase" == sweep ]]; then
    schedule=(baseline lowreg fused fused fused fused fused fused fused fused fused fused lowreg baseline)
    planes=(12 12 1 3 6 9 12 12 9 6 3 1 12 12)
elif [[ "$phase" == confirm ]]; then
    # Bracket the selected partial split with an all-sampling-in-mixed control
    # as well as the equally split but unfused separate control.
    schedule=(baseline lowreg fused separate fused fused separate fused lowreg baseline baseline lowreg fused separate fused fused separate fused lowreg baseline)
    planes=(12 12 12 "$mixed_planes" "$mixed_planes" "$mixed_planes" "$mixed_planes" 12 12 12 12 12 12 "$mixed_planes" "$mixed_planes" "$mixed_planes" "$mixed_planes" 12 12 12)
elif [[ "$phase" == diagnostic ]]; then
    schedule=(baseline lowreg fused separate fused)
    planes=(12 12 12 "$mixed_planes" "$mixed_planes")
else
    schedule=(baseline fused fused)
    planes=(12 12 "$mixed_planes")
fi
for index in "${!schedule[@]}"; do
    mode=${schedule[$index]}
    current_planes=$mixed_planes
    [[ "$phase" != sweep && "$phase" != confirm && "$phase" != diagnostic && "$phase" != trace ]] || current_planes=${planes[$index]}
    export TACKER_HEXPLANE_LOWREG=0 TACKER_EXPERIMENTAL_RASTER_HEXPLANE="$mode" TACKER_RASTER_HEXPLANE_MIXED_PLANES="$current_planes"
    [[ "$mode" != baseline && "$mode" != lowreg ]] || export TACKER_EXPERIMENTAL_RASTER_HEXPLANE=0
    [[ "$mode" != lowreg ]] || export TACKER_HEXPLANE_LOWREG=1
    stem=$(printf '%02d-%s-mp%s-q%s' "$index" "$mode" "$current_planes" "$ratio")
    check_gpu_idle "$stem"
    echo "START $target $stem $(date --iso-8601=seconds)"
    if [[ "$phase" == quality ]]; then
        timeout 1200 "$python_bin" "$project_root/scripts/validate_tacker_modes.py" -m "$model" -s "$source" \
            --configs "$project_root/arguments/dynerf/flame_steak.py" --iteration 14000 --scene-name flame_steak \
            --expected-image-width 1352 --expected-image-height 1014 --expected-gaussian-count 111525 \
            --split test --frames 10 --modes serial tacker --qualification-mode --qualification-profile "$qualification" \
            --lpips-net vgg --gpu 0 --quiet --output "$output_root/$stem.json" > "$output_root/$stem.log" 2>&1
        "$python_bin" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"],d["errors"]; print("QUALITY PASS",d["deltas"])' "$output_root/$stem.json"
    else
        frames=50
        launch=(timeout 1200)
        if [[ "$phase" == diagnostic ]]; then
            export TACKER_PROFILE_MIXED_PIPELINE_JSONL="$output_root/$stem.timeline.jsonl"
        elif [[ "$phase" == trace ]]; then
            frames=10
            launch=("$nsys_bin" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite --output="$output_root/$stem")
        fi
        "${launch[@]}" "$python_bin" "$project_root/scripts/run_profile_render_sealed.py" \
            --model_path "$model" --source_path "$source" --configs "$project_root/arguments/dynerf/flame_steak.py" \
            --iteration 14000 --split test --warmup 20 --frames "$frames" --trials 1 --workload-name flame_steak \
            --execution-mode tacker --qualification-mode --qualification-profile "$qualification" --quiet \
            --metadata "$output_root/$stem.json" > "$output_root/$stem.log" 2>&1
        "$python_bin" -c '
import json,sys
path,mode,planes,frames=sys.argv[1:]; d=json.load(open(path)); steady=int(frames)-1
assert d["passed"] and d["actual_execution_mode"]=="tacker" and d["tacker_fallback_reason"] is None
assert d["qualification_mode_executed"] and d["selected_variant_id"]=="full_mlp_overlap_graph_pb0",d
c=d["pipeline_execution_counts"]; assert c["mixed_launches"]==steady and c["raster_prepasses"]==steady and c["outputs"]==int(frames),c
if mode in ("fused","separate"):
    assert c["raster_hexplane_fused_launches"]==(steady if mode=="fused" else 0),c
    assert c["raster_hexplane_sampling_graph_replays"]==(steady if mode=="separate" else 0),c
    assert c.get("raster_hexplane_early_sampling_graph_replays",0)==(steady if int(planes)<12 else 0),c
print("FPS",mode,"mixed_planes",planes,d["throughput_fps"])
' "$output_root/$stem.json" "$mode" "$current_planes" "$frames"
    fi
done
check_gpu_idle final
echo "DONE $output_root"
