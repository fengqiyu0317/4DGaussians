#!/usr/bin/env bash
set -euo pipefail

PHASE31_CODE_ROOT=/home/qyfeng/tacker_phase31_code/20260913-codex-phase31
PHASE31_EVIDENCE_ROOT=/data/qyfeng/tacker_phase31_validation/20260913-codex-phase31
PHASE31_PYTHON=/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python3.10
PHASE31_RASTER_ROOT="${PHASE31_CODE_ROOT}/submodules/depth-diff-gaussian-rasterization"

export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="${PHASE31_CODE_ROOT}/.qualification-bin:/usr/local/cuda-12.4/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PHASE31_CODE_ROOT}/tacker_ext:${PHASE31_RASTER_ROOT}:${PHASE31_CODE_ROOT}/submodules/simple-knn:${PHASE31_CODE_ROOT}"
export FOURDGS_SOURCE_COMMIT=ad71abe0f90d606bc9a5f3955d05cef1ce781e3d
export FOURDGS_RASTERIZER_COMMIT=7d8fb35515521c11e75f2a3d20b5764fa2279790
export FOURDGS_SIMPLE_KNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
export NSYS_BIN=/usr/local/cuda-12.4/bin/nsys

exec "${PHASE31_PYTHON}" "${PHASE31_CODE_ROOT}/scripts/run_tacker_phase31.py" \
  --model-path /data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak \
  --source-path /data/qyfeng/datasets/n3dv/flame_steak \
  --config "${PHASE31_CODE_ROOT}/arguments/dynerf/flame_steak.py" \
  --current-tacker-profile "${PHASE31_CODE_ROOT}/tacker_profiles/incumbents/current-v1.json" \
  --template-profile "${PHASE31_CODE_ROOT}/tacker_profiles/raster_head_sm86.json" \
  --cuda-suite-report "${PHASE31_EVIDENCE_ROOT}/cuda-suite-final.json" \
  --output "${PHASE31_EVIDENCE_ROOT}/run-complete-v3" \
  --python-executable "${PHASE31_PYTHON}" \
  --nvidia-smi "${PHASE31_CODE_ROOT}/.qualification-bin/nvidia-smi" \
  --profile-nsight-script "${PHASE31_CODE_ROOT}/scripts/profile_nsight.sh" \
  --gpu 0 \
  --workload-name flame_steak \
  --iteration 14000 \
  --split test \
  --image-width 1352 \
  --image-height 1014 \
  --gaussian-count 111525 \
  --head-rows 111525 \
  --c3-top-k 4 \
  --c4-top-k-per-family 2 \
  --top-k 5 \
  --screen-batch-size 30 \
  --screen-frames 10 \
  --screen-warmup 2 \
  --screen-trials 2 \
  --leaf-views 2 \
  --leaf-warmup 5 \
  --leaf-repetitions 50 \
  --seed 0 \
  --timeout-seconds 900 \
  "$@"
