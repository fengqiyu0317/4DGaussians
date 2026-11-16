#!/usr/bin/env bash
# Opt-in split-Graph experiment. Never updates the reference Jetson checkout.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo 'usage: PROJECT_ROOT OUTPUT_ROOT quality|abba|diagnostic HEX_TASKS_PER_CTA' >&2
    exit 2
fi
project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
phase=$3
ratio=$4
[[ "$ratio" =~ ^[1-9][0-9]*$ ]] && ((ratio <= 2147483647)) || exit 2
[[ "$phase" == quality || "$phase" == abba || "$phase" == diagnostic ]] || exit 2
[[ ! -e "$output_root" ]] || { echo "output already exists: $output_root" >&2; exit 2; }
mkdir -p "$output_root"
qualification=/data/qyfeng/4DGaussians-tacker-v12-split-20261001/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json
model=/data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak
source=/data/qyfeng/datasets/n3dv/flame_steak_4dgs_min
gpu_freq=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
export CUDA_VISIBLE_DEVICES=0 TACKER_CUDA_ARCH=sm_87 TACKER_GPU_NAME=Orin
export PYTHONPATH="${project_root}/_python_deps_compat:${project_root}/tacker_ext:${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}/submodules/simple-knn:${project_root}:/home/jetson/.local/lib/python3.10/site-packages"
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99 FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d
export FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
export TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1
export TACKER_RASTER_HEXPLANE_TASKS_PER_CTA="$ratio"
export TACKER_HEXPLANE_LOWREG=0
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11
unset TACKER_MIXED_V12_SPLIT_TILE TACKER_MIXED_V12_SPLIT_FRACTION TACKER_RASTER_PREPASS_MAX_CTAS
unset TACKER_MIXED_V12_PREPASS_PHYSICAL_BLOCKS
unset TACKER_PROFILE_MIXED_PIPELINE_JSONL TACKER_PROFILE_MIXED_V12_SPLIT_JSONL
unset TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE
sha256sum "$project_root/gaussian_renderer/tacker_pipeline.py" \
    "$project_root/gaussian_renderer/raster_hexplane_graph.py" \
    "$project_root/scene/hexplane.py" \
    "$project_root/tacker_ext/tacker_4dgs_head/hexplane.py" \
    "$project_root/tacker_ext/tacker_4dgs_head/_C.cpython-310-aarch64-linux-gnu.so" \
    "$project_root/tacker_ext/tacker_4dgs_head/_hexplane.cpython-310-aarch64-linux-gnu.so" \
    "$project_root/submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-310-aarch64-linux-gnu.so" \
    "$qualification" > "$output_root/runtime.sha256"

check_gpu_idle() {
    local stem=$1
    sha256sum --check "$output_root/runtime.sha256" > "$output_root/$stem.source_check.txt"
    ps -eo pid,user,comm,args > "$output_root/$stem.processes.txt"
    free -m > "$output_root/$stem.memory.txt"
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv \
        > "$output_root/$stem.gpu_processes.txt"
    timeout 2 tegrastats --interval 1000 > "$output_root/$stem.tegrastats.txt" 2>&1 || [[ $? -eq 124 ]]
    grep -q 'GR3D_FREQ 0%' "$output_root/$stem.tegrastats.txt" || exit 3
    [[ "$(cat "$gpu_freq/min_freq")" == 1300500000 && \
       "$(cat "$gpu_freq/max_freq")" == 1300500000 && \
       "$(cat "$gpu_freq/cur_freq")" == 1300500000 ]] || exit 3
}

if [[ "$phase" == quality ]]; then
    schedule=(fused separate)
elif [[ "$phase" == diagnostic ]]; then
    schedule=(baseline separate fused)
else
    # Three independent ABBA repetitions, plus original v12 Graph references.
    schedule=(baseline separate fused fused separate baseline separate fused fused separate baseline separate fused fused separate baseline)
fi
for index in "${!schedule[@]}"; do
    mode=${schedule[$index]}
    stem=$(printf '%02d-%s-q%s' "$index" "$mode" "$ratio")
    check_gpu_idle "$stem"
    export TACKER_EXPERIMENTAL_RASTER_HEXPLANE="$mode"
    [[ "$mode" != baseline ]] || export TACKER_EXPERIMENTAL_RASTER_HEXPLANE=0
    echo "START $stem $(date --iso-8601=seconds)"
    if [[ "$phase" == quality ]]; then
        timeout 1200 /usr/bin/python3 "$project_root/scripts/validate_tacker_modes.py" \
            -m "$model" -s "$source" --configs "$project_root/arguments/dynerf/flame_steak.py" \
            --iteration 14000 --scene-name flame_steak --expected-image-width 1352 \
            --expected-image-height 1014 --expected-gaussian-count 111525 \
            --split test --frames 10 --modes serial tacker --qualification-mode \
            --qualification-profile "$qualification" --lpips-net vgg --gpu 0 --quiet \
            --output "$output_root/$stem.json" > "$output_root/$stem.log" 2>&1
        /usr/bin/python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"],d["errors"]; print("QUALITY PASS",d["deltas"])' "$output_root/$stem.json"
    else
        if [[ "$phase" == diagnostic ]]; then
            export TACKER_PROFILE_MIXED_PIPELINE_JSONL="$output_root/$stem.timeline.jsonl"
        fi
        timeout 1200 /usr/bin/python3 "$project_root/scripts/run_profile_render_sealed.py" \
            --model_path "$model" --source_path "$source" \
            --configs "$project_root/arguments/dynerf/flame_steak.py" \
            --iteration 14000 --split test --warmup 10 --frames 50 --trials 1 \
            --workload-name flame_steak --execution-mode tacker --qualification-mode \
            --qualification-profile "$qualification" --quiet --metadata "$output_root/$stem.json" \
            > "$output_root/$stem.log" 2>&1
        /usr/bin/python3 -c '
import json,sys
path,mode=sys.argv[1:]; d=json.load(open(path))
assert d["passed"] and d["actual_execution_mode"]=="tacker" and d["tacker_fallback_reason"] is None
assert d["qualification_mode_executed"] and d["selected_variant_id"]=="full_mlp_overlap_graph_pb0",d
c=d["pipeline_execution_counts"]
assert c["mixed_launches"]==49 and c["raster_prepasses"]==49 and c["outputs"]==50,c
if mode!="baseline":
    assert c["raster_hexplane_fused_launches"]==(49 if mode=="fused" else 0),c
    assert c["raster_hexplane_sampling_graph_replays"]==(49 if mode=="separate" else 0),c
print("FPS", mode, d["throughput_fps"])
' "$output_root/$stem.json" "$mode"
    fi
done
sha256sum --check "$output_root/runtime.sha256" > "$output_root/source_check_after.txt"
check_gpu_idle final
echo "DONE $output_root"
