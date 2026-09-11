#pragma once

#include <cuda_fp16.h>
#include <stddef.h>
#include <math.h>

#include "head_linear_device.cuh"

namespace tacker_4dgs {

// ABI v2 deliberately keeps task descriptors as pointer-only PODs.  A mixed
// wrapper can construct up to five descriptors from its kernel arguments and
// call the adapters below without allocating device-side metadata storage.
constexpr int kHeadLinearAbiV2 = 2;
constexpr int kMaxHeadTasksV2 = 5;
constexpr int kMaxTailFeaturesV2 = 128;
constexpr int kWholeHeadBarrierParticipantsV2 = kHeadThreads;
constexpr int kWholeHeadScratchFloatsPerGroupV2 = kHeadFeatures;

struct HeadLinearTaskV2 {
    const half* input;
    const half* weight;
    const float* bias;
    float* output;
    int rows;
};

struct WholeHeadTaskV2 {
    const half* input;
    const half* first_weight;
    const float* first_bias;
    const float* tail_weight;
    const float* tail_bias;
    float* output;
    int rows;
    int tail_features;
};

// A worker group is one contiguous, warp-aligned 128-thread range.  Group g
// owns tasks g, g + worker_groups, ...; consequently one group serializes a
// bundle while up to five groups execute one head apiece.  Every task clips
// the common GPTB end position to its own logical grid, which makes differing
// row counts and empty heads safe.  No CTA-wide or named barrier is used.
__device__ __forceinline__ void head_linear_multi_gptb_device(
    const HeadLinearTaskV2* tasks,
    int task_count,
    int worker_groups,
    int ptb_start_block_pos,
    int ptb_iter_block_step,
    int ptb_end_block_pos,
    int thread_base) {
    const int backend_thread = static_cast<int>(threadIdx.x) - thread_base;
    if (backend_thread < 0 || task_count <= 0 || task_count > kMaxHeadTasksV2 ||
        worker_groups <= 0 || worker_groups > task_count ||
        backend_thread >= worker_groups * kHeadThreads ||
        ptb_iter_block_step <= 0) {
        return;
    }

    const int worker_group = backend_thread / kHeadThreads;
    const int local_thread = backend_thread % kHeadThreads;
    for (int task_index = worker_group; task_index < task_count;
         task_index += worker_groups) {
        const HeadLinearTaskV2& task = tasks[task_index];
        if (task.rows <= 0) {
            continue;
        }
        const int logical_grid_x =
            (task.rows + kRowsPerTile - 1) / kRowsPerTile;
        const int logical_grid_y = kHeadFeatures / kColumnsPerBlock;
        const int logical_blocks = logical_grid_x * logical_grid_y;
        const int logical_end =
            ptb_end_block_pos < logical_blocks
                ? ptb_end_block_pos
                : logical_blocks;
        for (int block_position =
                 static_cast<int>(blockIdx.x) + ptb_start_block_pos;
             block_position < logical_end;
             block_position += ptb_iter_block_step) {
            head_linear_logical_block(
                task.input,
                task.weight,
                task.bias,
                task.output,
                task.rows,
                logical_grid_x,
                block_position,
                local_thread);
        }
    }
}

// Packed C3 layout shares one input across heads and uses contiguous
// [head_count,128,128] weights, [head_count,128] biases and
// [head_count,rows,128] outputs.  It has the same scheduling semantics and no
// synchronization footprint as the descriptor-based adapter.
__device__ __forceinline__ void head_linear_packed_gptb_device(
    const half* input,
    const half* packed_weights,
    const float* packed_biases,
    float* packed_outputs,
    int rows,
    int head_count,
    int worker_groups,
    int ptb_start_block_pos,
    int ptb_iter_block_step,
    int ptb_end_block_pos,
    int thread_base) {
    const int backend_thread = static_cast<int>(threadIdx.x) - thread_base;
    if (backend_thread < 0 || rows <= 0 || head_count <= 0 ||
        head_count > kMaxHeadTasksV2 || worker_groups <= 0 ||
        worker_groups > head_count ||
        backend_thread >= worker_groups * kHeadThreads ||
        ptb_iter_block_step <= 0) {
        return;
    }

    const int worker_group = backend_thread / kHeadThreads;
    const int local_thread = backend_thread % kHeadThreads;
    const int logical_grid_x = (rows + kRowsPerTile - 1) / kRowsPerTile;
    const int logical_grid_y = kHeadFeatures / kColumnsPerBlock;
    const int logical_blocks = logical_grid_x * logical_grid_y;
    const int logical_end =
        ptb_end_block_pos < logical_blocks
            ? ptb_end_block_pos
            : logical_blocks;
    const size_t weight_stride =
        static_cast<size_t>(kHeadFeatures) * kHeadFeatures;
    const size_t bias_stride = kHeadFeatures;
    const size_t output_stride =
        static_cast<size_t>(rows) * kHeadFeatures;

    for (int head_index = worker_group; head_index < head_count;
         head_index += worker_groups) {
        const half* weight =
            packed_weights + static_cast<size_t>(head_index) * weight_stride;
        const float* bias =
            packed_biases + static_cast<size_t>(head_index) * bias_stride;
        float* output =
            packed_outputs + static_cast<size_t>(head_index) * output_stride;
        for (int block_position =
                 static_cast<int>(blockIdx.x) + ptb_start_block_pos;
             block_position < logical_end;
             block_position += ptb_iter_block_step) {
            head_linear_logical_block(
                input,
                weight,
                bias,
                output,
                rows,
                logical_grid_x,
                block_position,
                local_thread);
        }
    }
}

__device__ __forceinline__ void head_linear_v2_subgroup_sync(
    int barrier_id) {
    asm volatile(
        "bar.sync %0, %1;"
        :
        : "r"(barrier_id), "r"(kWholeHeadBarrierParticipantsV2)
        : "memory");
}

// C4 source adapter for Linear(128,128) -> ReLU -> Linear(128,O).  Tail
// weights/biases stay FP32, matching the unfused PyTorch head after the first
// FP16-qualified layer.  The caller supplies block-shared scratch containing
// worker_groups*128 floats and a disjoint range of worker_groups named barrier
// IDs.  Raster currently owns barrier 1, so mixed wrappers normally use 2..6.
__device__ __forceinline__ void whole_head_multi_gptb_device(
    const WholeHeadTaskV2* tasks,
    int task_count,
    int worker_groups,
    int ptb_start_row,
    int ptb_iter_row_step,
    int ptb_end_row,
    int thread_base,
    float* shared_hidden,
    int barrier_base_id) {
    const int backend_thread = static_cast<int>(threadIdx.x) - thread_base;
    if (backend_thread < 0 || task_count <= 0 ||
        task_count > kMaxHeadTasksV2 || worker_groups <= 0 ||
        worker_groups > task_count ||
        backend_thread >= worker_groups * kHeadThreads ||
        ptb_iter_row_step <= 0) {
        return;
    }

    const int worker_group = backend_thread / kHeadThreads;
    const int local_thread = backend_thread % kHeadThreads;
    float* hidden =
        shared_hidden + worker_group * kWholeHeadScratchFloatsPerGroupV2;
    const int barrier_id = barrier_base_id + worker_group;

    for (int task_index = worker_group; task_index < task_count;
         task_index += worker_groups) {
        const WholeHeadTaskV2& task = tasks[task_index];
        if (task.rows <= 0 || task.tail_features <= 0 ||
            task.tail_features > kMaxTailFeaturesV2) {
            continue;
        }
        const int row_end =
            ptb_end_row < task.rows ? ptb_end_row : task.rows;
        for (int row = static_cast<int>(blockIdx.x) + ptb_start_row;
             row < row_end;
             row += ptb_iter_row_step) {
            float accumulator = task.first_bias[local_thread];
            const half* input_row =
                task.input + static_cast<size_t>(row) * kHeadFeatures;
            const half* weight_row =
                task.first_weight +
                static_cast<size_t>(local_thread) * kHeadFeatures;
#pragma unroll 4
            for (int feature = 0; feature < kHeadFeatures; ++feature) {
                accumulator = fmaf(
                    __half2float(input_row[feature]),
                    __half2float(weight_row[feature]),
                    accumulator);
            }
            hidden[local_thread] = accumulator > 0.0f ? accumulator : 0.0f;
            head_linear_v2_subgroup_sync(barrier_id);

            if (local_thread < task.tail_features) {
                float tail_accumulator = task.tail_bias[local_thread];
                const float* tail_weight_row =
                    task.tail_weight +
                    static_cast<size_t>(local_thread) * kHeadFeatures;
#pragma unroll 4
                for (int feature = 0; feature < kHeadFeatures; ++feature) {
                    tail_accumulator = fmaf(
                        hidden[feature],
                        tail_weight_row[feature],
                        tail_accumulator);
                }
                task.output[
                    static_cast<size_t>(row) * task.tail_features +
                    local_thread] = tail_accumulator;
            }
            // Prevent a fast warp from overwriting shared hidden values while
            // another warp is still consuming the current row.
            head_linear_v2_subgroup_sync(barrier_id);
        }
    }
}

}  // namespace tacker_4dgs
