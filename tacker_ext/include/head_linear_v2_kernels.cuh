#pragma once

#include "head_linear_v2_device.cuh"

// Stable CUDA C entrypoints.  HeadLinearTaskV2 is a by-value POD kernel
// parameter whose byte layout is frozen in abi/head_linear_v2.json.
extern "C" __global__ void tacker_head_linear_multi_solo_v2(
    tacker_4dgs::HeadLinearTaskV2 task0,
    tacker_4dgs::HeadLinearTaskV2 task1,
    tacker_4dgs::HeadLinearTaskV2 task2,
    tacker_4dgs::HeadLinearTaskV2 task3,
    tacker_4dgs::HeadLinearTaskV2 task4,
    int task_count,
    int worker_groups,
    int logical_end);

extern "C" __global__ void tacker_head_linear_multi_gptb_v2(
    tacker_4dgs::HeadLinearTaskV2 task0,
    tacker_4dgs::HeadLinearTaskV2 task1,
    tacker_4dgs::HeadLinearTaskV2 task2,
    tacker_4dgs::HeadLinearTaskV2 task3,
    tacker_4dgs::HeadLinearTaskV2 task4,
    int task_count,
    int worker_groups,
    int ptb_start_block_pos,
    int ptb_iter_block_step,
    int ptb_end_block_pos);

extern "C" __global__ void tacker_head_linear_packed_gptb_v2(
    const half* input,
    const half* packed_weights,
    const float* packed_biases,
    float* packed_outputs,
    int rows,
    int head_count,
    int worker_groups,
    int ptb_start_block_pos,
    int ptb_iter_block_step,
    int ptb_end_block_pos);

extern "C" __global__ void tacker_whole_head_gptb_v2(
    tacker_4dgs::WholeHeadTaskV2 task,
    int ptb_start_row,
    int ptb_iter_row_step,
    int ptb_end_row);
