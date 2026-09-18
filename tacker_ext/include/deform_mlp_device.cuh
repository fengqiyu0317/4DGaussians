#pragma once

#include <cuda_fp16.h>
#include <mma.h>
#include <stddef.h>

namespace tacker_4dgs {

constexpr int kDeformRows = 16;
constexpr int kDeformInput = 32;
constexpr int kDeformWidth = 128;
// Offset consecutive half rows by one 16-byte WMMA alignment unit so they do
// not all begin on the same shared-memory bank pattern.
constexpr int kDeformActivatedStride = kDeformWidth + 8;
constexpr int kDeformHeads = 5;
constexpr int kDeformThreads = 128;

// Fixed order: position, scale, rotation, opacity, SH. Outputs are deltas.
struct DeformMlpParamsV1 {
    const float* features;          // [N,32]
    const float* backbone_weight;   // [32,128], transposed once on the host
    const float* backbone_bias;     // [128]
    const half* first_weights;      // [5,128,128], [head,out,in]
    const float* first_biases;      // [5,128]
    const float* tail_weights;      // [59,128], concatenated output rows
    const float* tail_biases;       // [59]
    float* outputs[kDeformHeads];   // [N,3], [N,3], [N,4], [N,1], [N,48]
    int rows;
};

struct __align__(32) DeformMlpScratchV1 {
    float features[kDeformRows * kDeformInput];
    half activated[kDeformRows * kDeformActivatedStride];
    float hidden[kDeformRows * kDeformWidth];
};
static_assert(sizeof(DeformMlpScratchV1) == 14592, "unexpected scratch layout");

struct DeformMlpCtaSyncV1 {
    __device__ __forceinline__ void sync() const {
        __syncthreads();
    }
};

struct DeformMlpNamedSyncV1 {
    int barrier_id;

    __device__ __forceinline__ void sync() const {
        asm volatile(
            "bar.sync %0, %1;"
            :
            : "r"(barrier_id), "r"(kDeformThreads)
            : "memory");
    }
};

__device__ __forceinline__ float deform_relu(float value) {
    // Unlike fmaxf(value, 0), preserve NaNs rather than hiding invalid results.
    return value < 0.0f ? 0.0f : value;
}

__device__ __forceinline__ void deform_backbone_linear_v1(
    const DeformMlpParamsV1& p, DeformMlpScratchV1& s, int row_begin, int tid) {
    // Adjacent threads own adjacent output columns. Weight[k, tid] is coalesced.
    for (int r = 0; r < kDeformRows; ++r) {
        float sum = 0.0f;
        if (row_begin + r < p.rows) {
#pragma unroll
            for (int k = 0; k < kDeformInput; ++k) {
                sum = fmaf(s.features[r * kDeformInput + k],
                           p.backbone_weight[k * kDeformWidth + tid], sum);
            }
            sum += p.backbone_bias[tid];
        }
        s.hidden[r * kDeformWidth + tid] = sum;
    }
}

__device__ __forceinline__ void deform_backbone_relu_v1(
    DeformMlpScratchV1& s, int tid) {
    for (int e = tid; e < kDeformRows * kDeformWidth; e += kDeformThreads) {
        // Explicit quantization boundary, matching existing packed-head inputs.
        const int row = e / kDeformWidth;
        const int column = e % kDeformWidth;
        s.activated[row * kDeformActivatedStride + column] =
            __float2half_rn(deform_relu(s.hidden[e]));
    }
}

__device__ __forceinline__ void deform_head_linear_v1(
    const DeformMlpParamsV1& p, DeformMlpScratchV1& s, int head, int tid) {
    namespace wmma = nvcuda::wmma;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const half* weight = p.first_weights + head * kDeformWidth * kDeformWidth;
    const float* bias = p.first_biases + head * kDeformWidth;

    // Four warps each own two disjoint 16x16 output tiles, covering all 128 cols.
#pragma unroll 1
    for (int q = 0; q < 2; ++q) {
        const int col = warp * 16 + q * 64;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int k = 0; k < kDeformWidth; k += 16) {
            wmma::load_matrix_sync(a, s.activated + k,
                                   kDeformActivatedStride);
            // [out,in] row-major weights are [in,out] column-major for A @ W.T.
            wmma::load_matrix_sync(b, weight + col * kDeformWidth + k,
                                   kDeformWidth);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(s.hidden + col, acc, kDeformWidth,
                               wmma::mem_row_major);
        __syncwarp();
        // Access known shared-memory coordinates, not opaque fragment indices.
        for (int e = lane; e < 16 * 16; e += 32) {
            const int row = e / 16;
            const int column = col + e % 16;
            s.hidden[row * kDeformWidth + column] += bias[column];
        }
    }
}

__device__ __forceinline__ void deform_head_relu_v1(
    DeformMlpScratchV1& s, int tid) {
    for (int e = tid; e < kDeformRows * kDeformWidth; e += kDeformThreads) {
        s.hidden[e] = deform_relu(s.hidden[e]);
    }
}

__device__ __forceinline__ void deform_tail_linear_v1(
    const DeformMlpParamsV1& p, const DeformMlpScratchV1& s,
    int head, int row_begin, int tid) {
    const int width = head < 2 ? 3 : (head == 2 ? 4 : (head == 3 ? 1 : 48));
    const int offset = head == 0 ? 0 : (head == 1 ? 3 :
                       (head == 2 ? 6 : (head == 3 ? 10 : 11)));
    const int warp = tid / 32;
    const int lane = tid % 32;
    // Each entire warp computes one output element, including opacity O=1.
    for (int u = warp; u < kDeformRows * width; u += kDeformThreads / 32) {
        const int row = u / width;
        const int col = u % width;
        float sum = 0.0f;
#pragma unroll
        for (int k = lane; k < kDeformWidth; k += 32) {
            sum = fmaf(s.hidden[row * kDeformWidth + k],
                       p.tail_weights[(offset + col) * kDeformWidth + k], sum);
        }
#pragma unroll
        for (int distance = 16; distance > 0; distance /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, distance);
        }
        if (lane == 0 && row_begin + row < p.rows) {
            p.outputs[head][static_cast<size_t>(row_begin + row) * width + col] =
                sum + p.tail_biases[offset + col];
        }
    }
}

// Every thread in one 128-thread logical worker must call this function.  The
// caller supplies both the logical thread id and synchronization policy so the
// same math can execute either as a standalone CTA or as a named-barrier
// subgroup inside a larger mixed Raster CTA.
template <typename SyncPolicy>
__device__ __forceinline__ void deform_mlp_device_impl_v1(
    const DeformMlpParamsV1& p,
    DeformMlpScratchV1& s,
    int tid,
    int tile_start,
    int tile_step,
    const SyncPolicy& worker_sync) {
    const int tiles = static_cast<int>((static_cast<long long>(p.rows) + 15) / 16);
    for (int tile = tile_start; tile < tiles; tile += tile_step) {
        const int row_begin = tile * kDeformRows;
        for (int e = tid; e < kDeformRows * kDeformInput; e += kDeformThreads) {
            const int row = row_begin + e / kDeformInput;
            s.features[e] = row < p.rows
                ? p.features[static_cast<size_t>(row) * kDeformInput + e % kDeformInput]
                : 0.0f;
        }
        worker_sync.sync();

        // The five requested stages execute in order, inside ONE kernel.
        deform_backbone_linear_v1(p, s, row_begin, tid);
        worker_sync.sync();
        deform_backbone_relu_v1(s, tid);
        worker_sync.sync();
        for (int head = 0; head < kDeformHeads; ++head) {
            deform_head_linear_v1(p, s, head, tid);
            worker_sync.sync();
            deform_head_relu_v1(s, tid);
            worker_sync.sync();
            deform_tail_linear_v1(p, s, head, row_begin, tid);
            worker_sync.sync();  // Readers finish before shared storage is reused.
        }
    }
}

// Standalone compatibility wrapper used by tacker_deform_mlp_delta_v1.
// All 128 physical CTA threads must call it.
__device__ __forceinline__ void deform_mlp_device_v1(
    const DeformMlpParamsV1& p, DeformMlpScratchV1& s) {
    deform_mlp_device_impl_v1(
        p,
        s,
        static_cast<int>(threadIdx.x),
        static_cast<int>(blockIdx.x),
        static_cast<int>(gridDim.x),
        DeformMlpCtaSyncV1{});
}

// Mixed-kernel adapter.  Raster lanes are rejected before any named barrier;
// exactly the contiguous 128-thread deform range participates in barrier_id.
__device__ __forceinline__ void deform_mlp_gptb_device_v1(
    const DeformMlpParamsV1& p,
    DeformMlpScratchV1& s,
    int tile_start,
    int tile_step,
    int thread_base,
    int barrier_id) {
    const int tid = static_cast<int>(threadIdx.x) - thread_base;
    if (tid < 0 || tid >= kDeformThreads || tile_step <= 0) {
        return;
    }
    deform_mlp_device_impl_v1(
        p,
        s,
        tid,
        tile_start,
        tile_step,
        DeformMlpNamedSyncV1{barrier_id});
}

}  // namespace tacker_4dgs
