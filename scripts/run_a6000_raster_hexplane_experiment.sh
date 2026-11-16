#!/usr/bin/env bash
# Graph-qualified A6000 experiment; all artifacts remain in an isolated tree.
set -euo pipefail
if [[ $# -ne 5 ]]; then
    echo 'usage: PROJECT_ROOT OUTPUT_ROOT cuda|quality|abba|diagnostic|sweep|refine|confirm PHYSICAL_GPU HEX_TASKS_PER_CTA' >&2
    exit 2
fi
project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
phase=$3
physical_gpu=$4
ratio=$5
[[ "$physical_gpu" =~ ^[0-9]+$ ]] || exit 2
[[ "$ratio" =~ ^[1-9][0-9]*$ ]] && ((ratio <= 2147483647)) || exit 2
[[ "$phase" == cuda || "$phase" == quality || "$phase" == abba || "$phase" == diagnostic || "$phase" == sweep || "$phase" == refine || "$phase" == confirm ]] || exit 2
[[ ! -e "$output_root" ]] || { echo "output already exists: $output_root" >&2; exit 2; }
mkdir -p "$output_root"
python_bin=/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python
qualification=/data/qyfeng/4DGaussians-flame-steak-full/benchmarks/results/full_mlp_graph_formal_20260925/full_mlp_overlap_graph_pb0.qualification.json
model=/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak
source=/data/qyfeng/datasets/n3dv/flame_steak
export CUDA_VISIBLE_DEVICES="$physical_gpu" TACKER_CUDA_ARCH=sm_86 TACKER_GPU_NAME='NVIDIA RTX A6000'
export PYTHONPATH="${project_root}/tacker_ext:${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}/submodules/simple-knn:${project_root}"
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99 FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d
export FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
export TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1
export TACKER_RASTER_HEXPLANE_TASKS_PER_CTA="$ratio" TACKER_HEXPLANE_LOWREG=0
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11
unset TACKER_EXPERIMENTAL_RASTER_HEXPLANE
unset TACKER_MIXED_V12_SPLIT_TILE TACKER_MIXED_V12_SPLIT_FRACTION TACKER_MIXED_V12_PREPASS_PHYSICAL_BLOCKS
unset TACKER_PROFILE_MIXED_PIPELINE_JSONL TACKER_PROFILE_MIXED_V12_SPLIT_JSONL TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE
# The server's libcuda/NVML symlinks were verified to match its 570 kernel
# module before this experiment. Do not retain the old mismatch workaround.
unset LD_PRELOAD
sha256sum "$project_root/gaussian_renderer/tacker_pipeline.py" \
    "$project_root/profile_render.py" \
    "$project_root/scripts/run_profile_render_sealed.py" \
    "$project_root/gaussian_renderer/raster_hexplane_graph.py" \
    "$project_root/scene/hexplane.py" \
    "$project_root/tacker_ext/tacker_4dgs_head/hexplane.py" \
    "$project_root/tacker_ext/tacker_4dgs_head/_C.cpython-310-x86_64-linux-gnu.so" \
    "$project_root/tacker_ext/tacker_4dgs_head/_hexplane.cpython-310-x86_64-linux-gnu.so" \
    "$project_root/submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so" \
    "$qualification" > "$output_root/runtime.sha256"
gpu_uuid=$(nvidia-smi -i "$physical_gpu" --query-gpu=uuid --format=csv,noheader)

check_gpu_idle() {
    local stem=$1
    sha256sum --check "$output_root/runtime.sha256" > "$output_root/$stem.source_check.txt"
    ps -eo pid,user,comm,args > "$output_root/$stem.processes.txt"
    free -m > "$output_root/$stem.memory.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv \
        > "$output_root/$stem.gpu_processes.csv"
    nvidia-smi -i "$physical_gpu" --query-gpu=uuid,memory.used,utilization.gpu,pstate,clocks.sm,clocks.mem,temperature.gpu \
        --format=csv > "$output_root/$stem.gpu.csv"
    "$python_bin" -c '
import csv,sys
root,stem,uuid=sys.argv[1:]
rows=list(csv.reader(open(root+"/"+stem+".gpu_processes.csv")))[1:]
assert not any(row and row[0].strip()==uuid.strip() for row in rows),rows
gpu=list(csv.reader(open(root+"/"+stem+".gpu.csv")))[1]
assert int(gpu[1].split()[0])<256 and int(gpu[2].split()[0])<=5,gpu
' "$output_root" "$stem" "$gpu_uuid"
}

if [[ "$phase" == cuda ]]; then
    check_gpu_idle cuda
    "$python_bin" "$project_root/tacker_ext/tests/test_hexplane_cuda.py" > "$output_root/hexplane.log" 2>&1
    check_gpu_idle mixed_cuda
    "$python_bin" "$project_root/tacker_ext/tests/test_raster_hexplane_mixed_cuda.py" > "$output_root/mixed.log" 2>&1
    check_gpu_idle resources
    "$python_bin" -c 'import json; from diff_gaussian_rasterization import GaussianRasterizer; from tacker_4dgs_head.hexplane import resources; print(json.dumps({"mixed":GaussianRasterizer.raster_hexplane_resources(),"hexplane":resources()},indent=2))' > "$output_root/resources.json"
    check_gpu_idle final
    echo "CUDA PASS $output_root"
    exit 0
fi
if [[ "$phase" == quality ]]; then
    schedule=(fused separate)
elif [[ "$phase" == diagnostic ]]; then
    schedule=(baseline lowreg separate fused)
elif [[ "$phase" == sweep ]]; then
    # Two reversed passes; sweep results are exploratory, not an optimum claim.
    schedule=(fused fused fused fused fused fused fused fused)
    ratios=(4 8 17 32 32 17 8 4)
elif [[ "$phase" == refine ]]; then
    schedule=(baseline lowreg fused fused fused fused fused fused fused fused lowreg baseline)
    ratios=(17 17 1 2 3 4 4 3 2 1 17 17)
elif [[ "$phase" == confirm ]]; then
    schedule=(baseline lowreg fused fused lowreg baseline baseline lowreg fused fused lowreg baseline)
else
    # Three ABBA repetitions plus references with original scheduling. Lowreg
    # keeps the original Graph scheduler but swaps only the sampling kernel.
    schedule=(baseline lowreg separate fused fused separate lowreg baseline separate fused fused separate baseline lowreg separate fused fused separate lowreg baseline)
fi
for index in "${!schedule[@]}"; do
    mode=${schedule[$index]}
    current_ratio=$ratio
    [[ "$phase" != sweep && "$phase" != refine ]] || current_ratio=${ratios[$index]}
    export TACKER_RASTER_HEXPLANE_TASKS_PER_CTA="$current_ratio"
    export TACKER_HEXPLANE_LOWREG=0
    export TACKER_EXPERIMENTAL_RASTER_HEXPLANE="$mode"
    if [[ "$mode" == baseline || "$mode" == lowreg ]]; then
        export TACKER_EXPERIMENTAL_RASTER_HEXPLANE=0
    fi
    [[ "$mode" != lowreg ]] || export TACKER_HEXPLANE_LOWREG=1
    stem=$(printf '%02d-%s-q%s' "$index" "$mode" "$current_ratio")
    check_gpu_idle "$stem"
    echo "START $stem $(date --iso-8601=seconds)"
    if [[ "$phase" == quality ]]; then
        timeout 1200 "$python_bin" "$project_root/scripts/validate_tacker_modes.py" \
            -m "$model" -s "$source" --configs "$project_root/arguments/dynerf/flame_steak.py" \
            --iteration 14000 --scene-name flame_steak --expected-image-width 1352 \
            --expected-image-height 1014 --expected-gaussian-count 111525 \
            --split test --frames 10 --modes serial tacker --qualification-mode \
            --qualification-profile "$qualification" --lpips-net vgg --gpu 0 --quiet \
            --output "$output_root/$stem.json" > "$output_root/$stem.log" 2>&1
        "$python_bin" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"],d["errors"]; print("QUALITY PASS",d["deltas"])' "$output_root/$stem.json"
    else
        if [[ "$phase" == diagnostic ]]; then
            export TACKER_PROFILE_MIXED_PIPELINE_JSONL="$output_root/$stem.timeline.jsonl"
        fi
        timeout 1200 "$python_bin" "$project_root/scripts/run_profile_render_sealed.py" \
            --model_path "$model" --source_path "$source" \
            --configs "$project_root/arguments/dynerf/flame_steak.py" \
            --iteration 14000 --split test --warmup 20 --frames 50 --trials 1 \
            --workload-name flame_steak --execution-mode tacker --qualification-mode \
            --qualification-profile "$qualification" --quiet --metadata "$output_root/$stem.json" \
            > "$output_root/$stem.log" 2>&1
        "$python_bin" -c '
import json,sys
path,mode=sys.argv[1:]; d=json.load(open(path))
assert d["passed"] and d["actual_execution_mode"]=="tacker" and d["tacker_fallback_reason"] is None
assert d["qualification_mode_executed"] and d["selected_variant_id"]=="full_mlp_overlap_graph_pb0",d
c=d["pipeline_execution_counts"]
assert c["mixed_launches"]==49 and c["raster_prepasses"]==49 and c["outputs"]==50,c
if mode in ("fused","separate"):
    assert c["raster_hexplane_fused_launches"]==(49 if mode=="fused" else 0),c
    assert c["raster_hexplane_sampling_graph_replays"]==(49 if mode=="separate" else 0),c
print("FPS", mode, d["throughput_fps"])
' "$output_root/$stem.json" "$mode"
    fi
done
sha256sum --check "$output_root/runtime.sha256" > "$output_root/source_check_after.txt"
check_gpu_idle final
echo "DONE $output_root"
