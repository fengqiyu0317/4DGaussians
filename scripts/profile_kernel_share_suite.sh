#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 PROJECT_ROOT OUTPUT_ROOT PHYSICAL_GPU" >&2
    exit 2
fi

project_root=$(realpath "$1")
output_root=$(realpath -m "$2")
physical_gpu="$3"

python_bin=${FOURDGS_PYTHON:-/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python}
nsys_bin=${NSYS_BIN:-/usr/local/cuda-12.8/bin/nsys}
cuda_graph_trace=${NSYS_CUDA_GRAPH_TRACE:-node}
model_path=${FOURDGS_MODEL_PATH:-/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak}
source_path=${FOURDGS_SOURCE_PATH:-/data/qyfeng/datasets/n3dv/flame_steak}
config_path=${FOURDGS_CONFIG_PATH:-${project_root}/arguments/dynerf/flame_steak.py}
profile_root=${FOURDGS_PROFILE_ROOT:-/data/qyfeng/4DGaussians-flame-steak-full/benchmarks/results/full_mlp_graph_formal_20260925}

resume=${FOURDGS_RESUME:-0}
if [[ -e "$output_root" && "$resume" != 1 ]]; then
    echo "refusing to overwrite existing output root: $output_root" >&2
    exit 2
fi

for required in \
    "$python_bin" \
    "$nsys_bin" \
    "$model_path" \
    "$source_path" \
    "$config_path" \
    "${project_root}/scripts/run_profile_render_sealed.py" \
    "${project_root}/scripts/summarize_nsight_stats.py"; do
    if [[ ! -e "$required" ]]; then
        echo "required path is missing: $required" >&2
        exit 2
    fi
done

mkdir -p "$output_root"
if [[ ! -e "${output_root}/collector.sh" ]]; then
    cp "$0" "${output_root}/collector.sh"
fi

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.570.124.06:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06
export FOURDGS_NVTX=1
export NSYS_TMPDIR=${NSYS_TMPDIR:-/tmp}
export PYTHONPATH="${project_root}/submodules/depth-diff-gaussian-rasterization:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export FOURDGS_SOURCE_COMMIT=${FOURDGS_SOURCE_COMMIT:-60018d41f4f57b693b0e0dc4052f85b0c9d31a99}
export FOURDGS_SOURCE_DIRTY=${FOURDGS_SOURCE_DIRTY:-true}
export FOURDGS_RASTERIZER_COMMIT=${FOURDGS_RASTERIZER_COMMIT:-464096373b869fff713265bcdc34b1822277980d}
export FOURDGS_SIMPLE_KNN_COMMIT=${FOURDGS_SIMPLE_KNN_COMMIT:-44f764299fa305faf6ec5ebd99939e0508331503}

labels=(
    serial
    two_stream
    full_mlp
    full_mlp_overlap
    full_mlp_overlap_graph
)
modes=(
    serial
    two_stream
    tacker
    tacker
    tacker
)
profiles=(
    ""
    ""
    "${profile_root}/full_mlp_legacy_pb0.qualification.json"
    "${profile_root}/full_mlp_overlap_pb0.qualification.json"
    "${profile_root}/full_mlp_overlap_graph_pb0.qualification.json"
)

cd "$project_root"
for index in "${!labels[@]}"; do
    label=${labels[$index]}
    mode=${modes[$index]}
    qualification_profile=${profiles[$index]}
    run_dir="${output_root}/${label}"
    report_base="${run_dir}/4dgs_render_${label}"
    metadata_path="${run_dir}/${label}_profile_metadata.json"
    summary_path="${run_dir}/${label}_summary.json"
    if [[ -e "$run_dir" && "$resume" != 1 ]]; then
        echo "refusing to overwrite existing run directory: $run_dir" >&2
        exit 2
    fi
    mkdir -p "$run_dir"

    profile_args=(
        --model_path "$model_path"
        --source_path "$source_path"
        --configs "$config_path"
        --iteration 14000
        --split test
        --warmup 10
        --frames 50
        --execution-mode "$mode"
        --metadata "$metadata_path"
    )
    if [[ "$mode" == tacker ]]; then
        if [[ ! -f "$qualification_profile" ]]; then
            echo "qualification profile is missing: $qualification_profile" >&2
            exit 2
        fi
        profile_args+=(
            --workload-name flame_steak
            --qualification-mode
            --qualification-profile "$qualification_profile"
        )
    fi

    if [[ ! -f "${report_base}.nsys-rep" ]]; then
        echo "[$(date --iso-8601=seconds)] profiling ${label} on physical GPU ${physical_gpu}"
        "$nsys_bin" profile \
            --trace=cuda,nvtx,osrt \
            --cuda-graph-trace="$cuda_graph_trace" \
            --sample=none \
            --cpuctxsw=none \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --force-overwrite=false \
            --output="$report_base" \
            "$python_bin" "${project_root}/scripts/run_profile_render_sealed.py" "${profile_args[@]}"
    else
        echo "[$(date --iso-8601=seconds)] reusing ${label} trace"
    fi

    if [[ ! -f "${report_base}_stats.csv" ]]; then
        "$nsys_bin" stats \
            --report nvtx_sum,nvtx_gpu_proj_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
            --format csv \
            --force-export=true \
            "${report_base}.nsys-rep" > "${report_base}_stats.csv"
    fi

    if ! "$python_bin" "${project_root}/scripts/summarize_nsight_stats.py" \
        "${report_base}_stats.csv" \
        --metadata "$metadata_path" \
        --output "$summary_path"; then
        rm -f "$summary_path"
        echo "warning: legacy summary failed for ${label}; raw statistics are preserved" >&2
    fi
    echo "[$(date --iso-8601=seconds)] completed ${label}"
done

echo "kernel-share suite complete: $output_root"
