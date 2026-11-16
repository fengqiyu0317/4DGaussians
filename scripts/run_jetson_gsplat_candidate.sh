#!/usr/bin/env bash
# All artifacts stay on the Jetson data disk; existing runtimes remain read-only.
set -euo pipefail
root=/data/qyfeng/gsplat-ca-20261027
stage=${1:?usage: run_jetson_gsplat_candidate.sh build|cuda|leaf|formal}
project="$root/project"
mkdir -p "$root/tmp" "$root/cache" "$root/extensions" "$root/results/$stage"
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache" CUDA_CACHE_PATH="$root/cache/cuda"
export TORCH_HOME=/data/qyfeng/cache/torch TORCH_EXTENSIONS_DIR="$root/extensions"
export MPLCONFIGDIR="$root/cache/matplotlib" PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=2 CUDA_VISIBLE_DEVICES=0
export PATH="$root/deps/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="/data/qyfeng/gsplat-v7-20261021/deps:$project/_python_deps_compat:$project/tacker_ext:$project/submodules/depth-diff-gaussian-rasterization:$project/submodules/simple-knn:$project:/home/jetson/.local/lib/python3.10/site-packages"
export TACKER_GPU_NAME=Orin TACKER_CUDA_ARCH=sm_87
export FOURDGS_SOURCE_COMMIT=60018d41f4f57b693b0e0dc4052f85b0c9d31a99 FOURDGS_SOURCE_DIRTY=true
export FOURDGS_RASTERIZER_COMMIT=464096373b869fff713265bcdc34b1822277980d FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
while IFS='=' read -r variable _; do
    case "$variable" in TACKER_EXPERIMENTAL_*|TACKER_PROFILE_*|TACKER_MIXED_*|TACKER_HEXPLANE_*|TACKER_RASTER_HEXPLANE_*) unset "$variable" ;; esac
done < <(env)
qualification=/data/qyfeng/4DGaussians-tacker-v12-split-20261001/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json
nvidia-smi > "$root/results/$stage/gpu_before.txt"
ps -eo pid,user,comm,args > "$root/results/$stage/processes_before.txt"
free -m > "$root/results/$stage/memory_before.txt"
timeout 3 tegrastats --interval 1000 > "$root/results/$stage/tegrastats_before.txt" 2>&1 || [[ $? == 124 ]]
/usr/bin/python3 - "$root/results/$stage/tegrastats_before.txt" "$project/scripts" "$root/results/$stage/processes_before.txt" <<'PY'
import re,sys
sys.path.insert(0,sys.argv[2])
from analyze_sampling_kernel_activity import foreign_python_processes
assert not foreign_python_processes(open(sys.argv[3]).read()), 'foreign Python workload'
text=open(sys.argv[1]).read(); activity=re.findall(r'GR3D_FREQ (\d+)%',text)
assert activity and all(x=='0' for x in activity), text
PY
freq=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cat "$freq/cur_freq" "$freq/min_freq" "$freq/max_freq" > "$root/results/$stage/frequency.txt"
[[ "$(< "$freq/min_freq")" == 1300500000 && "$(< "$freq/max_freq")" == 1300500000 ]]
(
    while true; do
        date --iso-8601=ns
        cat "$freq/cur_freq"
        ps -eo pid,user,comm,args
        sleep 2
    done
) > "$root/results/$stage/during_processes_and_frequency.txt" &
audit_pid=$!
trap 'kill "$audit_pid" 2>/dev/null || true' EXIT
export GSPLAT_LEGACY_V7_LIBRARY=/data/qyfeng/independent-matched-graph-20261015-1348/libdeform_matched.so
cd "$project"
case "$stage" in
build)
    export GSPLAT_BUILD_STAGE="$stage"
    /usr/bin/python3 - <<'PY'
import torch,os
from gsplat.cuda import _backend
from gsplat_mixed_ext import load
for variant in ('tensor_v7_ca','tensor_v7'):
    module=load(variant)
    print(variant,module.resources(),module.standalone_resources(),flush=True)
PY
    ;;
cuda)
    /usr/bin/python3 -m unittest tests.test_gsplat_candidate_cuda -v
    ;;
leaf|formal)
    /usr/bin/python3 scripts/benchmark_gsplat_candidate.py \
        --model_path /data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak \
        --source_path /data/qyfeng/datasets/n3dv/flame_steak_4dgs_min \
        --configs "$project/arguments/dynerf/flame_steak.py" --iteration 14000 \
        --qualification-profile "$qualification" --stage "$([[ "$stage" == leaf ]] && echo sweep || echo formal)" --output "$root/results/$stage"
    ;;
*) exit 2 ;;
esac
kill "$audit_pid"
wait "$audit_pid" || true
trap - EXIT
nvidia-smi > "$root/results/$stage/gpu_after.txt"
date --iso-8601=seconds > "$root/results/$stage/completed.txt"
