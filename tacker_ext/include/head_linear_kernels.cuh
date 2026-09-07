#pragma once

#include <cuda_fp16.h>

// Stable CUDA C entrypoints for direct launch or cuModuleGetFunction lookup.
extern "C" __global__ void tacker_head_linear_solo_v1(
    const half* input,
    const half* weight,
    const float* bias,
    float* output,
    int rows,
    int logical_grid_x);

extern "C" __global__ void tacker_head_linear_gptb_v1(
    const half* input,
    const half* weight,
    const float* bias,
    float* output,
    int rows,
    int logical_grid_x,
    int logical_grid_y,
    int ptb_start_block_pos,
    int ptb_iter_block_step,
    int ptb_end_block_pos);
