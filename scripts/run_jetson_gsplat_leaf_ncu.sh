#!/usr/bin/env bash
# Profile sealed binaries; all new files stay below the experiment's data root.
set -euo pipefail
root=/data/qyfeng/gsplat-leaf-ncu-20261028
phase=${1:?usage: validation|ncu|raster_memory}
[[ "$phase" == validation || "$phase" == ncu || "$phase" == raster_memory ]]
[[ "$phase" == validation || $(id -u) == 0 ]]
project=/data/qyfeng/gsplat-ca-20261027/project
legacy=/data/qyfeng/independent-matched-graph-20261015-1348/runtime
output="$root/results/$phase"
[[ ! -f "$output/completed.txt" ]]
mkdir -p "$output" "$root/tmp/$phase" "$root/cache" "$root/config" "$root/extensions"
export TMPDIR="$root/tmp/$phase" TMP="$root/tmp/$phase" TEMP="$root/tmp/$phase"
export XDG_CACHE_HOME="$root/cache" XDG_CONFIG_HOME="$root/config" XDG_DATA_HOME="$root/data"
export CUDA_CACHE_PATH="$root/cache/cuda" TORCH_EXTENSIONS_DIR="$root/extensions"
export TORCH_HOME=/data/qyfeng/cache/torch MPLCONFIGDIR="$root/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.6
export GSPLAT_REFERENCE_PROJECT="$project"
export GSPLAT_NATIVE_BINARY=/data/qyfeng/gsplat-ca-20261027/extensions/gsplat_cuda/gsplat_cuda.so
export GSPLAT_MIXED_BINARY=/data/qyfeng/gsplat-ca-20261027/extensions/gsplat_mixed_tensor_v7_ca/gsplat_mixed_tensor_v7_ca.so
export GSPLAT_LEGACY_V7_LIBRARY=/data/qyfeng/independent-matched-graph-20261015-1348/libdeform_matched.so
export PYTHONPATH="/data/qyfeng/gsplat-v7-20261021/deps:$legacy/submodules/depth-diff-gaussian-rasterization:$project/_python_deps_compat:$project/tacker_ext:$project/submodules/simple-knn:$project:$project/scripts:/home/jetson/.local/lib/python3.10/site-packages"
while IFS='=' read -r variable _; do
    case "$variable" in TACKER_EXPERIMENTAL_*|TACKER_PROFILE_*|TACKER_MIXED_*|TACKER_HEXPLANE_*|TACKER_RASTER_HEXPLANE_*|TACKER_BENCHMARK_*) unset "$variable" ;; esac
done < <(env)
export TACKER_GPU_NAME=Orin TACKER_CUDA_ARCH=sm_87
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99 FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
qualification=/data/qyfeng/4DGaussians-tacker-v12-split-20261001/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json
sha256sum "$root/profile_gsplat_leaf_resources.py" "$root/run_jetson_gsplat_leaf_ncu.sh" \
    "$GSPLAT_NATIVE_BINARY" "$GSPLAT_MIXED_BINARY" "$GSPLAT_LEGACY_V7_LIBRARY" \
    "$legacy/submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-310-aarch64-linux-gnu.so" \
    "$qualification" > "$output/before.sha256"
nvidia-smi > "$output/gpu_before.txt"
ps -eo pid,user,comm,args > "$output/processes_before.txt"
free -m > "$output/memory_before.txt"
timeout 3 tegrastats --interval 1000 > "$output/tegrastats_before.txt" 2>&1 || [[ $? == 124 ]]
/usr/bin/python3 - "$output" "$project/scripts" <<'PY'
from pathlib import Path
import re,sys
sys.path.insert(0,sys.argv[2])
from analyze_sampling_kernel_activity import foreign_python_processes
p=Path(sys.argv[1]);bad=foreign_python_processes((p/'processes_before.txt').read_text())
assert not bad,bad
activity=re.findall(r'GR3D_FREQ (\d+)%',(p/'tegrastats_before.txt').read_text())
assert activity and set(activity)=={'0'},activity
PY
freq=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cat "$freq/min_freq" "$freq/max_freq" > "$output/frequency_bounds.txt"
[[ "$(< "$freq/min_freq")" == 1300500000 && "$(< "$freq/max_freq")" == 1300500000 ]]
(
    while true; do
        date --iso-8601=ns
        cat "$freq/cur_freq"
        ps -eo pid,user,comm,args
        sleep 1
    done
) > "$output/during_processes_and_frequency.txt" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
command=(/usr/bin/python3 "$root/profile_gsplat_leaf_resources.py"
    --model_path /data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak
    --source_path /data/qyfeng/datasets/n3dv/flame_steak_4dgs_min
    --configs "$project/arguments/dynerf/flame_steak.py" --iteration 14000
    --qualification-profile "$qualification" --stage sweep --output "$output")
cd "$root"
if [[ "$phase" != validation ]]; then
    [[ -f "$root/results/validation/completed.txt" ]]
    export GSPLAT_NCU_CAPTURE=1
    /usr/local/cuda/bin/ncu --version > "$output/ncu_version.txt"
    ncu=(/usr/local/cuda/bin/ncu --config-file off
        --section-folder /opt/nvidia/nsight-compute/2024.3.1/sections
        --target-processes all --replay-mode kernel --graph-profiling node
        --profile-from-start off --clock-control none --cache-control all
        --export "$output/report" --section LaunchStats)
    if [[ "$phase" == raster_memory ]]; then
        ncu+=(--launch-count 2 --metrics gpu__time_duration.sum,l1tex__data_pipe_lsu_wavefronts.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sector_hit_rate.pct)
    else
        ncu+=(--launch-count 4 --section Occupancy --section SpeedOfLight
            --section SchedulerStats --section WarpStateStats
            --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis
            --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum)
    fi
    printf '%q ' "${ncu[@]}" "${command[@]}" > "$output/command.txt"
    timeout 900 "${ncu[@]}" "${command[@]}" > "$output/run.log" 2>&1
    /usr/local/cuda/bin/ncu --config-file off --import "$output/report.ncu-rep" --page raw --csv > "$output/report.raw.csv"
    /usr/local/cuda/bin/ncu --config-file off --import "$output/report.ncu-rep" --page details --csv > "$output/report.details.csv"
else
    unset GSPLAT_NCU_CAPTURE
    timeout 600 "${command[@]}" > "$output/run.log" 2>&1
fi
kill "$monitor"
wait "$monitor" || true
trap - EXIT
sha256sum --check "$output/before.sha256" > "$output/after_check.txt"
nvidia-smi > "$output/gpu_after.txt"
date --iso-8601=seconds > "$output/completed.txt"
chmod -R a+rX "$output"
echo "DONE $phase"
