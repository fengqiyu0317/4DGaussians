#!/usr/bin/env bash
# Run only in an isolated runtime copy, never rebuild or alter the reference.
set -euo pipefail
[[ $# == 3 || ( $# == 4 && $3 == trace ) ]] || { echo 'usage: PROJECT_ROOT OUTPUT_ROOT cuda|memcheck|quality|formal|trace [TRACE_CONFIG]'; exit 2; }
[[ $(id -u) == 0 ]] || { echo 'root required for matched CUPTI permission'; exit 2; }
project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
phase=$3
trace_config=${4:-}
if [[ -n "$trace_config" ]]; then
    case "$trace_config" in original-r0|original-r128|original-r256|mixed-r0|mixed-r128|mixed-r256) ;; *) exit 2 ;; esac
fi
[[ "$phase" == cuda || "$phase" == memcheck || "$phase" == quality || "$phase" == formal || "$phase" == trace ]] || exit 2
[[ ! -e "$output_root" ]] || exit 2
mkdir -p "$output_root"
qualification=/data/qyfeng/4DGaussians-tacker-v12-split-20261001/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json
export PYTHONPATH="$project_root/_python_deps_compat:$project_root/tacker_ext:$project_root/submodules/depth-diff-gaussian-rasterization:$project_root/submodules/simple-knn:$project_root:/home/jetson/.local/lib/python3.10/site-packages"
export CUDA_VISIBLE_DEVICES=0 TACKER_GPU_NAME=Orin TACKER_CUDA_ARCH=sm_87
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99 FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
export TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1 TACKER_HEXPLANE_LOWREG=0
export TACKER_RASTER_HEXPLANE_TASKS_PER_CTA=3 TACKER_RASTER_HEXPLANE_MIXED_PLANES=12
unset LD_PRELOAD TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9
unset TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V10 TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V11
unset TACKER_MIXED_V12_SPLIT_TILE TACKER_MIXED_V12_SPLIT_FRACTION TACKER_MIXED_V12_PREPASS_PHYSICAL_BLOCKS TACKER_RASTER_PREPASS_MAX_CTAS
unset TACKER_PROFILE_MIXED_PIPELINE_JSONL TACKER_PROFILE_MIXED_V12_SPLIT_JSONL TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE
sha256sum "$project_root/scripts/run_jetson_hexplane_reduction.sh" \
    "$project_root/scripts/profile_hexplane_reduction_activity.py" "$project_root/scripts/sampling_kernel_activity.so" \
    "$project_root/scripts/run_profile_render_sealed.py" "$project_root/scripts/validate_tacker_modes.py" \
    "$project_root/profile_render.py" "$project_root/scene/hexplane.py" \
    "$project_root/gaussian_renderer/tacker_pipeline.py" "$project_root/gaussian_renderer/raster_hexplane_graph.py" \
    "$project_root/tacker_ext/tacker_4dgs_head/hexplane.py" "$project_root/tacker_ext/csrc/hexplane_reduce.cu" \
    "$project_root/tacker_ext/tacker_4dgs_head/_C.cpython-310-aarch64-linux-gnu.so" \
    "$project_root/tacker_ext/tacker_4dgs_head/_hexplane.cpython-310-aarch64-linux-gnu.so" \
    "$project_root/submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-310-aarch64-linux-gnu.so" \
    "$qualification" > "$output_root/runtime.sha256"

check_idle() {
    local stem=$1
    sha256sum --check "$output_root/runtime.sha256" > "$output_root/$stem.source_check.txt"
    ps -eo pid,user,comm,args > "$output_root/$stem.processes.txt"
    free -m > "$output_root/$stem.memory.txt"
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > "$output_root/$stem.gpu_processes.csv"
    /usr/bin/python3 -c 'import csv,sys; rows=list(csv.reader(open(sys.argv[1])))[1:]; assert not any(r and r[0].strip().isdigit() for r in rows),rows' "$output_root/$stem.gpu_processes.csv"
    timeout 3 tegrastats --interval 1000 > "$output_root/$stem.tegrastats.txt" 2>&1 || [[ $? == 124 ]]
    /usr/bin/python3 -c 'import re,sys; s=open(sys.argv[1]).read(); a=re.findall(r"GR3D_FREQ (\d+)%",s); assert a and all(v=="0" for v in a),s' "$output_root/$stem.tegrastats.txt"
    local freq=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
    [[ "$(< "$freq/min_freq")" == 1300500000 && "$(< "$freq/max_freq")" == 1300500000 && "$(< "$freq/cur_freq")" == 1300500000 ]] || exit 3
    /usr/bin/python3 -c 'import sys; sys.path.insert(0,sys.argv[1]); from analyze_sampling_kernel_activity import foreign_python_processes; bad=foreign_python_processes(open(sys.argv[2]).read()); assert not bad,bad' "$project_root/scripts" "$output_root/$stem.processes.txt"
}

if [[ "$phase" == cuda || "$phase" == memcheck ]]; then
    check_idle initial
    command=(/usr/bin/python3 "$project_root/tacker_ext/tests/test_hexplane_reduction_cuda.py" -v)
    if [[ "$phase" == memcheck ]]; then
        command=(/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 "${command[@]}")
    fi
    "${command[@]}" > "$output_root/reduction.log" 2>&1
    if [[ "$phase" == cuda ]]; then
        /usr/bin/python3 "$project_root/tacker_ext/tests/test_hexplane_cuda.py" -v > "$output_root/sampling.log" 2>&1
        /usr/bin/python3 "$project_root/tacker_ext/tests/test_raster_hexplane_mixed_cuda.py" -v > "$output_root/mixed.log" 2>&1
    fi
    check_idle final
    date --iso-8601=seconds > "$output_root/completed.txt"
    tail -6 "$output_root/reduction.log"
    exit 0
fi

if [[ "$phase" == quality ]]; then
    schedule=(original-r128 original-r256 mixed-r128 mixed-r256)
    modes=(off off off off)
elif [[ "$phase" == formal ]]; then
    # Two mirrored complete blocks: 4 independent processes/configuration.
    schedule=(original-r0 original-r128 original-r256 mixed-r0 mixed-r128 mixed-r256 mixed-r256 mixed-r128 mixed-r0 original-r256 original-r128 original-r0
              original-r0 original-r128 original-r256 mixed-r0 mixed-r128 mixed-r256 mixed-r256 mixed-r128 mixed-r0 original-r256 original-r128 original-r0)
else
    # Per-configuration ABBA collector calibration, unchanged GPU work.
    schedule=()
    modes=()
    for config in original-r0 original-r128 original-r256 mixed-r0 mixed-r128 mixed-r256; do
        [[ -z "$trace_config" || "$config" == "$trace_config" ]] || continue
        schedule+=("$config" "$config" "$config" "$config")
        modes+=(off on on off)
    done
fi
for index in "${!schedule[@]}"; do
    config=${schedule[$index]}
    sampling=${config%-r*}
    threads=${config##*-r}
    export TACKER_EXPERIMENTAL_RASTER_HEXPLANE=0 TACKER_HEXPLANE_FUSED_REDUCTION=0 TACKER_HEXPLANE_REDUCTION_THREADS=256
    [[ "$sampling" != mixed ]] || export TACKER_EXPERIMENTAL_RASTER_HEXPLANE=fused
    [[ "$threads" == 0 ]] || export TACKER_HEXPLANE_FUSED_REDUCTION=1 TACKER_HEXPLANE_REDUCTION_THREADS="$threads"
    mode=off
    [[ "$phase" != trace ]] || mode=${modes[$index]}
    stem=$(printf '%02d-%s-%s' "$index" "$config" "$mode")
    check_idle "$stem"
    echo "START $phase $stem $(date --iso-8601=seconds)"
    (
        while true; do
            date --iso-8601=ns
            ps -eo pid,user,comm,args
            sleep 1
        done
    ) > "$output_root/$stem.during_processes.txt" &
    audit_pid=$!
    trap 'kill "$audit_pid" 2>/dev/null || true' EXIT
    if [[ "$phase" == quality ]]; then
        timeout 1200 /usr/bin/python3 "$project_root/scripts/validate_tacker_modes.py" \
            -m /data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak \
            -s /data/qyfeng/datasets/n3dv/flame_steak_4dgs_min \
            --configs "$project_root/arguments/dynerf/flame_steak.py" --iteration 14000 --scene-name flame_steak \
            --expected-image-width 1352 --expected-image-height 1014 --expected-gaussian-count 111525 \
            --split test --frames 10 --modes serial tacker --qualification-mode --qualification-profile "$qualification" \
            --lpips-net vgg --gpu 0 --quiet --output "$output_root/$stem.json" > "$output_root/$stem.log" 2>&1
        /usr/bin/python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"],d["errors"]; print("QUALITY PASS",d["deltas"])' "$output_root/$stem.json"
    else
        trials=4
        [[ "$phase" != trace ]] || trials=2
        timeout 1200 /usr/bin/python3 "$project_root/scripts/profile_hexplane_reduction_activity.py" \
            --kernel-mode "$mode" --kernel-library "$project_root/scripts/sampling_kernel_activity.so" \
            --kernel-output "$output_root/$stem.activity.json" \
            --model_path /data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak \
            --source_path /data/qyfeng/datasets/n3dv/flame_steak_4dgs_min \
            --configs "$project_root/arguments/dynerf/flame_steak.py" --iteration 14000 \
            --split test --warmup 20 --frames 50 --trials "$trials" --workload-name flame_steak \
            --execution-mode tacker --qualification-mode --qualification-profile "$qualification" \
            --quiet --metadata "$output_root/$stem.json" > "$output_root/$stem.log" 2>&1
        /usr/bin/python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"] and d["selected_variant_id"]=="full_mlp_overlap_graph_pb0" and d["post_run"]["byte_stability"]["verified"]; print("FPS",d["throughput_fps"])' "$output_root/$stem.json"
    fi
    kill "$audit_pid"
    wait "$audit_pid" || true
    trap - EXIT
done
check_idle final
date --iso-8601=seconds > "$output_root/completed.txt"
echo "DONE $phase $output_root"
