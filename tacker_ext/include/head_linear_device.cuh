#pragma once

#include <cuda_fp16.h>
#include <stddef.h>
#include <math.h>
#include <mma.h>

namespace tacker_4dgs {

namespace wmma = nvcuda::wmma;

constexpr int kHeadFeatures = 128;
constexpr int kHeadThreads = 128;
constexpr int kRowsPerTile = 16;
constexpr int kColumnsPerBlock = 64;

// Pointer ABI:
//   input  : row-major half  [rows, 128]
//   weight : row-major half  [128 out, 128 in] (PyTorch Linear layout)
//   bias   : float           [128]
//   output : row-major float [rows, 128]
// The caller owns all storage and must keep it alive until its CUDA event has
// completed.  The adapter allocates no memory and touches only these pointers.

__device__ __forceinline__ void head_linear_scalar_tail(
    const half* input,
    const half* weight,
    const float* bias,
    float* output,
    int rows,
    int row_begin,
    int column_begin,
    int local_thread) {
    const int tail_rows = rows - row_begin;
    for (int row_offset = 0; row_offset < tail_rows; ++row_offset) {
        const int column = column_begin + local_thread;
        if (local_thread < kColumnsPerBlock && column < kHeadFeatures) {
            float accumulator = bias[column];
            const size_t input_offset =
                static_cast<size_t>(row_begin + row_offset) * kHeadFeatures;
            const size_t weight_offset =
                static_cast<size_t>(column) * kHeadFeatures;
            const half* input_row = input + input_offset;
            const half* weight_row = weight + weight_offset;
#pragma unroll 4
            for (int k = 0; k < kHeadFeatures; ++k) {
                accumulator = fmaf(
                    __half2float(input_row[k]),
                    __half2float(weight_row[k]),
                    accumulator);
            }
            const size_t output_offset =
                static_cast<size_t>(row_begin + row_offset) * kHeadFeatures +
                static_cast<size_t>(column);
            output[output_offset] = accumulator;
        }
    }
}

__device__ __forceinline__ void head_linear_logical_block(
    const half* input,
    const half* weight,
    const float* bias,
    float* output,
    int rows,
    int logical_grid_x,
    int block_position,
    int local_thread) {
    const int logical_block_x = block_position % logical_grid_x;
    const int logical_block_y = block_position / logical_grid_x;
    const int row_begin = logical_block_x * kRowsPerTile;
    const int column_begin = logical_block_y * kColumnsPerBlock;

    if (row_begin >= rows) {
        return;
    }
    if (rows - row_begin < kRowsPerTile) {
        head_linear_scalar_tail(
            input, weight, bias, output, rows, row_begin, column_begin, local_thread);
        return;
    }

#if __CUDA_ARCH__ >= 700
    const int warp = local_thread >> 5;
    const int lane = local_thread & 31;
    const int column_tile = column_begin + warp * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

#pragma unroll
    for (int k = 0; k < kHeadFeatures; k += 16) {
        const size_t input_offset =
            static_cast<size_t>(row_begin) * kHeadFeatures +
            static_cast<size_t>(k);
        const size_t weight_offset =
            static_cast<size_t>(column_tile) * kHeadFeatures +
            static_cast<size_t>(k);
        wmma::load_matrix_sync(
            a, input + input_offset, kHeadFeatures);
        // Treat row-major [out, in] weight as column-major [in, out], which
        // implements input @ weight.T without repacking the real model weight.
        wmma::load_matrix_sync(
            b, weight + weight_offset, kHeadFeatures);
        wmma::mma_sync(accumulator, a, b, accumulator);
    }
    const size_t output_tile_offset =
        static_cast<size_t>(row_begin) * kHeadFeatures +
        static_cast<size_t>(column_tile);
    wmma::store_matrix_sync(
        output + output_tile_offset,
        accumulator,
        kHeadFeatures,
        wmma::mem_row_major);
    __syncwarp();

    // Each warp owns a disjoint 16x16 output tile.  No CTA-wide barrier is
    // needed, which makes this safe inside a 128-thread mixed-kernel subgroup.
    for (int element = lane; element < 16 * 16; element += 32) {
        const int row_offset = element / 16;
        const int column_offset = element % 16;
        const size_t output_offset =
            static_cast<size_t>(row_begin + row_offset) * kHeadFeatures +
            static_cast<size_t>(column_tile + column_offset);
        output[output_offset] += bias[column_tile + column_offset];
    }
#endif
}

// Tacker/GPTB-compatible adapter.  Its traversal matches Tacker's convention:
// block_position starts at physical blockIdx.x + ptb_start_block_pos and then
// advances by ptb_iter_block_step.  A mixed wrapper assigns exactly 128
// contiguous threads at a warp-aligned thread_base to this adapter.  For the
// first 4DGS Raster+GEMM wrapper, the intended ABI is a 384-thread CTA with
// Raster on [0, 256) and this adapter on [256, 384).  This implementation does
// not consume a named CTA barrier, leaving Raster's barrier IDs isolated.
__device__ __forceinline__ void head_linear_gptb_device(
    const half* input,
    const half* weight,
    const float* bias,
    float* output,
    int rows,
    int logical_grid_x,
    int logical_grid_y,
    int ptb_start_block_pos,
    int ptb_iter_block_step,
    int ptb_end_block_pos,
    int thread_base) {
    const int local_thread = static_cast<int>(threadIdx.x) - thread_base;
    if (local_thread < 0 || local_thread >= kHeadThreads) {
        return;
    }
    if (logical_grid_x <= 0 || logical_grid_y <= 0 || ptb_iter_block_step <= 0) {
        return;
    }
    const int logical_blocks = logical_grid_x * logical_grid_y;
    const int end =
        ptb_end_block_pos < logical_blocks ? ptb_end_block_pos : logical_blocks;
    for (int block_position = static_cast<int>(blockIdx.x) + ptb_start_block_pos;
         block_position < end;
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

}  // namespace tacker_4dgs
