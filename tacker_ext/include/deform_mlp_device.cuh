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
// The v11 mixed-kernel path gives the FP32 first-head output the same kind of
// row skew as the FP16 activation tile.  A 16-byte offset per row changes the
// bank mapping seen by WMMA's shared-memory store without violating the
// four-float leading-dimension requirement of store_matrix_sync.
constexpr int kDeformFirstHiddenStride = kDeformWidth + 4;
constexpr int kDeformHeads = 5;
constexpr int kDeformThreads = 128;
// Tensor-Core tail v3 pads each head independently to 16 output columns.
// Fixed order: position 16, scale 16, rotation 16, opacity 16, SH 48.
constexpr int kDeformTensorTailRows = 112;
// Tensor-tail v4 offsets each compact FP16 activation row by one 16-byte
// alignment unit.  The padded 4,352-byte tile plus the 3,072-byte FP32 output
// slab still fits inside the existing 8-KiB hidden buffer.
constexpr int kDeformTensorTailPaddedStride = kDeformWidth + 8;
constexpr int kDeformTensorTailOutputOffsetFloats =
    kDeformRows * kDeformTensorTailPaddedStride / 2;
static_assert(
    kDeformTensorTailOutputOffsetFloats + kDeformRows * 48 <=
        kDeformRows * kDeformWidth,
    "padded tensor-tail scratch must fit in the existing hidden buffer");

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

// Experimental Tensor-Core-tail descriptor used by standalone v4 and the
// opt-in mixed ABI v6.  Keep it separate from DeformMlpParamsV1 so the
// qualified scalar-tail mixed ABI v5 remains source- and binary-stable.
struct DeformMlpTensorTailParamsV2 {
    const float* features;          // [N,32]
    const float* backbone_weight;   // [32,128]
    const float* backbone_bias;     // [128]
    const half* first_weights;      // [5,128,128]
    const float* first_biases;      // [5,128]
    const half* packed_tail_weights;  // [112,128]
    const float* tail_biases;       // [59]
    float* outputs[kDeformHeads];
    int rows;
};

struct __align__(32) DeformMlpScratchV1 {
    float features[kDeformRows * kDeformInput];
    half activated[kDeformRows * kDeformActivatedStride];
    float hidden[kDeformRows * kDeformWidth];
};
static_assert(sizeof(DeformMlpScratchV1) == 14592, "unexpected scratch layout");

// v11-only scratch layout.  Keep V1 intact so the already-qualified v6-v10
// kernels retain their original shared-memory footprint and code generation.
struct __align__(32) DeformMlpScratchV2 {
    float features[kDeformRows * kDeformInput];
    half activated[kDeformRows * kDeformActivatedStride];
    float hidden[kDeformRows * kDeformFirstHiddenStride];
};
static_assert(sizeof(DeformMlpScratchV2) == 14848,
              "unexpected padded scratch layout");

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

// Pointer-only forms keep the larger v3 body from taking the address of the
// by-value kernel parameter struct.  On CUDA 12.4, taking that address caused
// the complete 104-byte parameter block to be copied to per-thread local
// memory even though ptxas reported no register spills.
__device__ __forceinline__ void deform_backbone_linear_v3(
    const float* features,
    const float* backbone_weight,
    const float* backbone_bias,
    int rows,
    DeformMlpScratchV1& s,
    int row_begin,
    int tid) {
    for (int r = 0; r < kDeformRows; ++r) {
        float sum = 0.0f;
        if (row_begin + r < rows) {
#pragma unroll
            for (int k = 0; k < kDeformInput; ++k) {
                sum = fmaf(
                    s.features[r * kDeformInput + k],
                    backbone_weight[k * kDeformWidth + tid], sum);
            }
            sum += backbone_bias[tid];
        }
        s.hidden[r * kDeformWidth + tid] = sum;
    }
}

__device__ __forceinline__ void deform_head_linear_v3(
    const half* first_weights,
    const float* first_biases,
    DeformMlpScratchV1& s,
    int head,
    int tid) {
    namespace wmma = nvcuda::wmma;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const half* weight =
        first_weights + head * kDeformWidth * kDeformWidth;
    const float* bias = first_biases + head * kDeformWidth;
#pragma unroll 1
    for (int q = 0; q < 2; ++q) {
        const int col = warp * 16 + q * 64;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int k = 0; k < kDeformWidth; k += 16) {
            wmma::load_matrix_sync(
                a, s.activated + k, kDeformActivatedStride);
            wmma::load_matrix_sync(
                b, weight + col * kDeformWidth + k, kDeformWidth);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(
            s.hidden + col, acc, kDeformWidth, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 16 * 16; e += 32) {
            const int row = e / 16;
            const int column = col + e % 16;
            s.hidden[row * kDeformWidth + column] += bias[column];
        }
    }
}

// v11 Deform stages.  The first-head WMMA result is written with a skewed
// shared-memory row stride, and bias is deliberately not applied here.  The
// following ReLU/compaction stage consumes each FP32 result once and folds in
// the bias, eliminating the old shared-memory read-modify-write pass.
template <int Unroll = kDeformInput>
__device__ __forceinline__ void deform_backbone_linear_v4(
    const float* backbone_weight,
    const float* backbone_bias,
    int rows,
    DeformMlpScratchV2& s,
    int row_begin,
    int tid) {
    for (int r = 0; r < kDeformRows; ++r) {
        float sum = 0.0f;
        if (row_begin + r < rows) {
#pragma unroll Unroll
            for (int k = 0; k < kDeformInput; ++k) {
                sum = fmaf(
                    s.features[r * kDeformInput + k],
                    backbone_weight[k * kDeformWidth + tid], sum);
            }
            sum += backbone_bias[tid];
        }
        s.hidden[r * kDeformFirstHiddenStride + tid] = sum;
    }
}

__device__ __forceinline__ void deform_backbone_relu_v2(
    DeformMlpScratchV2& s, int tid) {
    for (int e = tid; e < kDeformRows * kDeformWidth;
         e += kDeformThreads) {
        const int row = e / kDeformWidth;
        const int column = e % kDeformWidth;
        s.activated[row * kDeformActivatedStride + column] =
            __float2half_rn(deform_relu(
                s.hidden[row * kDeformFirstHiddenStride + column]));
    }
}

// Keep four rows in flight so a weight loaded by a lane serves four FMAs.
// Each row still accumulates in k order and can write its final FP16 activation.
__device__ __forceinline__ void deform_backbone_reuse_weights(
    const float* weights, const float* biases, int rows,
    DeformMlpScratchV2& s, int row_begin, int tid) {
    constexpr int rows_per_weight = 4;
    const float bias = biases[tid];
#pragma unroll 1
    for (int row = 0; row < kDeformRows; row += rows_per_weight) {
        float sum[rows_per_weight] = {};
#pragma unroll 1
        for (int k = 0; k < kDeformInput; ++k) {
            const float weight = weights[k * kDeformWidth + tid];
#pragma unroll
            for (int i = 0; i < rows_per_weight; ++i) {
                sum[i] = fmaf(s.features[(row + i) * kDeformInput + k], weight, sum[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < rows_per_weight; ++i) {
            const float value = row_begin + row + i < rows ? sum[i] + bias : 0.0f;
            s.activated[(row + i) * kDeformActivatedStride + tid] =
                __float2half_rn(deform_relu(value));
        }
    }
}

__device__ __forceinline__ void deform_head_linear_v4(
    const half* first_weights,
    DeformMlpScratchV2& s,
    int head,
    int tid) {
    namespace wmma = nvcuda::wmma;
    const int warp = tid / 32;
    const half* weight =
        first_weights + head * kDeformWidth * kDeformWidth;
#pragma unroll 1
    for (int q = 0; q < 2; ++q) {
        const int col = warp * 16 + q * 64;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                       wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                       wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int k = 0; k < kDeformWidth; k += 16) {
            wmma::load_matrix_sync(
                a, s.activated + k, kDeformActivatedStride);
            wmma::load_matrix_sync(
                b, weight + col * kDeformWidth + k, kDeformWidth);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(
            s.hidden + col, acc, kDeformFirstHiddenStride,
            wmma::mem_row_major);
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ float4 deform_mma_streamed(
    const half* input, const half* weights, int lane) {
    static_assert(kDeformActivatedStride == kDeformTensorTailPaddedStride);
    const int group = lane / 4;
    const int pair = (lane % 4) * 2;
    const half* a = input + (lane % 16) * kDeformActivatedStride + (lane / 16) * 8;
    const auto a_base = static_cast<unsigned int>(__cvta_generic_to_shared(a));
    const half* b = weights + group * kDeformWidth + pair;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
#pragma unroll 1
    for (int k = 0; k < kDeformWidth; k += 16) {
        unsigned int a0, a1, a2, a3;
        const unsigned int address = a_base + k * sizeof(half);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(address) : "memory");
        const auto b0 = *reinterpret_cast<const unsigned int*>(b + k);
        const auto b1 = *reinterpret_cast<const unsigned int*>(b + k + 8);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    return make_float4(c0, c1, c2, c3);
}

// The first projection reads the common backbone activation and writes to a
// different scratch slab, so each warp can pack its accumulator directly.
__device__ __forceinline__ void deform_head_linear_relu_streamed(
    const half* weights, const float* bias, DeformMlpScratchV2& s, int head, int tid) {
    const int lane = tid % 32;
    const int group = lane / 4;
    const int pair = (lane % 4) * 2;
    half* output = reinterpret_cast<half*>(s.hidden);
    weights += head * kDeformWidth * kDeformWidth;
    bias += head * kDeformWidth;
#pragma unroll 1
    for (int n = 0; n < 4; ++n) {
        const int col = (tid / 32) * 16 + (n / 2) * 64 + (n % 2) * 8;
        const float4 v = deform_mma_streamed(s.activated, weights + col * kDeformWidth, lane);
        const float b0 = bias[col + pair], b1 = bias[col + pair + 1];
        half* dst = output + group * kDeformTensorTailPaddedStride + col + pair;
        *reinterpret_cast<half2*>(dst) =
            __floats2half2_rn(deform_relu(v.x + b0), deform_relu(v.y + b1));
        *reinterpret_cast<half2*>(dst + 8 * kDeformTensorTailPaddedStride) =
            __floats2half2_rn(deform_relu(v.z + b0), deform_relu(v.w + b1));
    }
}

__device__ __forceinline__ void deform_tail_streamed(
    const half* weights, DeformMlpScratchV2& s, int head, int tid) {
    const int warp_col = (tid / 32) * 16;
    const int lane = tid % 32;
    const int group = lane / 4;
    const int pair = (lane % 4) * 2;
    weights += head * 16 * kDeformWidth;
    const half* input = reinterpret_cast<const half*>(s.hidden);
    float* out = s.hidden + kDeformTensorTailOutputOffsetFloats;
#pragma unroll 1
    for (int col = warp_col; col < min(warp_col + 16, head == 4 ? 48 : 8); col += 8) {
        const float4 v = deform_mma_streamed(input, weights + col * kDeformWidth, lane);
        *reinterpret_cast<float2*>(out + group * 48 + col + pair) = make_float2(v.x, v.y);
        *reinterpret_cast<float2*>(out + (group + 8) * 48 + col + pair) = make_float2(v.z, v.w);
    }
}
#endif

__device__ __forceinline__ void deform_head_relu_v1(
    DeformMlpScratchV1& s, int tid) {
    for (int e = tid; e < kDeformRows * kDeformWidth; e += kDeformThreads) {
        s.hidden[e] = deform_relu(s.hidden[e]);
    }
}

__device__ __forceinline__ void deform_head_relu_half_v3(
    DeformMlpScratchV1& s, int tid) {
    // Preserve s.activated: all five first-layer heads still need the common
    // backbone activation.  Instead compact the 8-KiB FP32 hidden buffer into
    // its own lower 4 KiB as FP16.  Every source value in a 1024-element half
    // is first captured in registers; the barrier then makes the overlapping
    // forward compaction race-free.  Two halves keep register pressure below
    // the six-block occupancy boundary on sm_86.
    half* compact = reinterpret_cast<half*>(s.hidden);
#pragma unroll
    for (int group = 0; group < 2; ++group) {
        float values[8];
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int e = group * 1024 + item * kDeformThreads + tid;
            values[item] = deform_relu(s.hidden[e]);
        }
        __syncthreads();
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int e = group * 1024 + item * kDeformThreads + tid;
            // This is the additional v3 precision boundary.  It is explicit
            // so image-quality admission can measure its effect.
            compact[e] = __float2half_rn(values[item]);
        }
    }
}

template <typename SyncPolicy>
__device__ __forceinline__ void deform_head_relu_half_v4(
    DeformMlpScratchV1& s, int tid, const SyncPolicy& worker_sync) {
    // Preserve v3 and write an independently padded experimental layout.
    // As in v3, capture one 1024-element source half before the overlapping
    // in-place FP32 -> FP16 write.  Padding does not cross into the unread
    // second source half, and the group barrier makes both writes race-free.
    half* compact = reinterpret_cast<half*>(s.hidden);
#pragma unroll
    for (int group = 0; group < 2; ++group) {
        float values[8];
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int e = group * 1024 + item * kDeformThreads + tid;
            values[item] = deform_relu(s.hidden[e]);
        }
        worker_sync.sync();
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int e = group * 1024 + item * kDeformThreads + tid;
            const int row = e / kDeformWidth;
            const int column = e % kDeformWidth;
            compact[row * kDeformTensorTailPaddedStride + column] =
                __float2half_rn(values[item]);
        }
    }
}

template <typename SyncPolicy>
__device__ __forceinline__ void deform_head_relu_half_v5(
    DeformMlpScratchV1& s, int tid, const SyncPolicy& worker_sync) {
    // Low-register mixed candidate.  A 512-element source group covers four
    // rows, so every thread keeps four FP32 values live across the barrier
    // instead of v4's eight.  The padded destination of each completed group
    // ends before the next group's unread FP32 source begins, which preserves
    // the race-free in-place compaction contract.
    half* compact = reinterpret_cast<half*>(s.hidden);
#pragma unroll
    for (int group = 0; group < 4; ++group) {
        float values[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int e = group * 512 + item * kDeformThreads + tid;
            values[item] = deform_relu(s.hidden[e]);
        }
        worker_sync.sync();
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int e = group * 512 + item * kDeformThreads + tid;
            const int row = e / kDeformWidth;
            const int column = e % kDeformWidth;
            compact[row * kDeformTensorTailPaddedStride + column] =
                __float2half_rn(values[item]);
        }
    }
}

template <typename SyncPolicy>
__device__ __forceinline__ void deform_head_relu_half_v6(
    DeformMlpScratchV1& s, int tid, const SyncPolicy& worker_sync) {
    // Four-CTA experiment: retain only two FP32 source values per thread
    // across each named barrier.  Each 256-element group covers two rows.
    // Its padded destination ends before the next unread FP32 source group,
    // preserving the race-free in-place conversion used by v4/v5.
    half* compact = reinterpret_cast<half*>(s.hidden);
#pragma unroll 1
    for (int group = 0; group < 8; ++group) {
        float values[2];
#pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int e = group * 256 + item * kDeformThreads + tid;
            values[item] = deform_relu(s.hidden[e]);
        }
        worker_sync.sync();
#pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int e = group * 256 + item * kDeformThreads + tid;
            const int row = e / kDeformWidth;
            const int column = e % kDeformWidth;
            compact[row * kDeformTensorTailPaddedStride + column] =
                __float2half_rn(values[item]);
        }
    }
}

template <typename SyncPolicy>
__device__ __forceinline__ void deform_head_relu_half_v7(
    const float* first_biases,
    DeformMlpScratchV2& s,
    int head,
    int tid,
    const SyncPolicy& worker_sync) {
    // Every thread owns one output column, so one cached bias value serves all
    // 16 rows.  Two source rows are captured before the overlapping FP32 ->
    // FP16 write, matching v6's two-register compaction envelope.
    const float bias = first_biases[head * kDeformWidth + tid];
    half* compact = reinterpret_cast<half*>(s.hidden);
#pragma unroll 1
    for (int group = 0; group < 8; ++group) {
        float values[2];
#pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int row = group * 2 + item;
            values[item] = deform_relu(
                s.hidden[row * kDeformFirstHiddenStride + tid] + bias);
        }
        worker_sync.sync();
#pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int row = group * 2 + item;
            compact[row * kDeformTensorTailPaddedStride + tid] =
                __float2half_rn(values[item]);
        }
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

__device__ __forceinline__ void deform_tail_tensor_v3(
    const half* packed_tail_weights,
    const float* tail_biases,
    float* output,
    int rows,
    DeformMlpScratchV1& s,
    int head,
    int row_begin,
    int tid) {
    namespace wmma = nvcuda::wmma;
    const int width = head < 2 ? 3 : (head == 2 ? 4 : (head == 3 ? 1 : 48));
    const int bias_offset = head == 0 ? 0 : (head == 1 ? 3 :
                            (head == 2 ? 6 : (head == 3 ? 10 : 11)));
    const int packed_offset = head * 16;
    const int output_tiles = (width + 15) / 16;
    const int warp = tid / 32;
    const int lane = tid % 32;

    if (warp >= output_tiles) return;
    const int col = warp * 16;
    const half* weight = packed_tail_weights +
        (packed_offset + col) * kDeformWidth;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
#pragma unroll
    for (int k = 0; k < kDeformWidth; k += 16) {
        wmma::load_matrix_sync(
            a, reinterpret_cast<const half*>(s.hidden) + k, kDeformWidth);
        // Packed [out,in] row-major is [in,out] column-major for A @ W.T.
        wmma::load_matrix_sync(b, weight + k, kDeformWidth);
        wmma::mma_sync(acc, a, b, acc);
    }
    // The compact FP16 input occupies bytes [0,4096) of s.hidden.  Use the
    // disjoint upper 4 KiB for a 16x48 FP32 tail-output slab.
    float* tail_output = s.hidden + 1024;
    wmma::store_matrix_sync(
        tail_output + col, acc, 48, wmma::mem_row_major);
    __syncwarp();

    for (int e = lane; e < kDeformRows * 16; e += 32) {
        const int row = e / 16;
        const int column = col + e % 16;
        if (column < width && row_begin + row < rows) {
            output[
                static_cast<size_t>(row_begin + row) * width + column] =
                tail_output[row * 48 + column] +
                tail_biases[bias_offset + column];
        }
    }
}

__device__ __forceinline__ void deform_tail_tensor_v4(
    const half* packed_tail_weights,
    const float* tail_biases,
    float* output,
    int rows,
    DeformMlpScratchV1& s,
    int head,
    int row_begin,
    int tid) {
    namespace wmma = nvcuda::wmma;
    const int width = head < 2 ? 3 : (head == 2 ? 4 : (head == 3 ? 1 : 48));
    const int bias_offset = head == 0 ? 0 : (head == 1 ? 3 :
                            (head == 2 ? 6 : (head == 3 ? 10 : 11)));
    const int packed_offset = head * 16;
    const int output_tiles = (width + 15) / 16;
    const int warp = tid / 32;
    const int lane = tid % 32;

    if (warp >= output_tiles) return;
    const int col = warp * 16;
    const half* weight = packed_tail_weights +
        (packed_offset + col) * kDeformWidth;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
#pragma unroll
    for (int k = 0; k < kDeformWidth; k += 16) {
        wmma::load_matrix_sync(
            a, reinterpret_cast<const half*>(s.hidden) + k,
            kDeformTensorTailPaddedStride);
        wmma::load_matrix_sync(b, weight + k, kDeformWidth);
        wmma::mma_sync(acc, a, b, acc);
    }
    float* tail_output = s.hidden + kDeformTensorTailOutputOffsetFloats;
    wmma::store_matrix_sync(
        tail_output + col, acc, 48, wmma::mem_row_major);
    __syncwarp();

    for (int e = lane; e < kDeformRows * 16; e += 32) {
        const int row = e / 16;
        const int column = col + e % 16;
        if (column < width && row_begin + row < rows) {
            output[
                static_cast<size_t>(row_begin + row) * width + column] =
                tail_output[row * 48 + column] +
                tail_biases[bias_offset + column];
        }
    }
}

// ABI-v8 splits the Tensor-Core matmul and scalar output epilogue at a named
// subgroup barrier.  The WMMA phase therefore does not keep output pointers,
// row bounds, and bias state live alongside its A/B/accumulator fragments.
__device__ __forceinline__ void deform_tail_tensor_matmul_v5(
    const half* packed_tail_weights,
    DeformMlpScratchV1& s,
    int head,
    int tid) {
    namespace wmma = nvcuda::wmma;
    const int output_tiles = head == 4 ? 3 : 1;
    const int warp = tid / 32;
    if (warp >= output_tiles) return;

    const int col = warp * 16;
    const half* weight = packed_tail_weights +
        (head * 16 + col) * kDeformWidth;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
#pragma unroll
    for (int k = 0; k < kDeformWidth; k += 16) {
        wmma::load_matrix_sync(
            a, reinterpret_cast<const half*>(s.hidden) + k,
            kDeformTensorTailPaddedStride);
        wmma::load_matrix_sync(b, weight + k, kDeformWidth);
        wmma::mma_sync(acc, a, b, acc);
    }
    float* tail_output = s.hidden + kDeformTensorTailOutputOffsetFloats;
    wmma::store_matrix_sync(
        tail_output + col, acc, 48, wmma::mem_row_major);
}

// Register-minimal sm_80+ Tensor Core path for ABI-v8.  Two m16n8k16 MMA
// operations replace one opaque WMMA m16n16k16 fragment.  Computing the two
// eight-column halves sequentially keeps only four A, two B, and four FP32
// accumulator registers live at once.  Fragment coordinates follow the PTX
// mma.m16n8k16 row/column layout.
template <typename Scratch>
__device__ __forceinline__ void deform_tail_tensor_matmul_v6(
    const half* packed_tail_weights,
    Scratch& s,
    int head,
    int tid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int warp = tid / 32;
    if (warp >= (head == 4 ? 3 : 1)) return;

    const int lane = tid % 32;
    const int group = lane / 4;
    const int thread_in_group = lane % 4;
    const half* activation = reinterpret_cast<const half*>(s.hidden);
    float* tail_output = s.hidden + kDeformTensorTailOutputOffsetFloats;
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        float c0 = 0.0f;
        float c1 = 0.0f;
        float c2 = 0.0f;
        float c3 = 0.0f;
#pragma unroll
        for (int k = 0; k < kDeformWidth; k += 16) {
            const int activation_col = k + thread_in_group * 2;
            const unsigned int a0 = *reinterpret_cast<const unsigned int*>(
                activation + group * kDeformTensorTailPaddedStride +
                activation_col);
            const unsigned int a1 = *reinterpret_cast<const unsigned int*>(
                activation + (group + 8) * kDeformTensorTailPaddedStride +
                activation_col);
            const unsigned int a2 = *reinterpret_cast<const unsigned int*>(
                activation + group * kDeformTensorTailPaddedStride +
                activation_col + 8);
            const unsigned int a3 = *reinterpret_cast<const unsigned int*>(
                activation + (group + 8) * kDeformTensorTailPaddedStride +
                activation_col + 8);
            const int weight_col = warp * 16 + n * 8 + group;
            const half* weight = packed_tail_weights +
                (head * 16 + weight_col) * kDeformWidth + k +
                thread_in_group * 2;
            const unsigned int b0 =
                *reinterpret_cast<const unsigned int*>(weight);
            const unsigned int b1 =
                *reinterpret_cast<const unsigned int*>(weight + 8);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1));
        }
        const int output_col = warp * 16 + n * 8 + thread_in_group * 2;
        *reinterpret_cast<float2*>(
            tail_output + group * 48 + output_col) = make_float2(c0, c1);
        *reinterpret_cast<float2*>(
            tail_output + (group + 8) * 48 + output_col) =
            make_float2(c2, c3);
    }
#else
    deform_tail_tensor_matmul_v5(packed_tail_weights, s, head, tid);
#endif
}

template <typename Scratch>
__device__ __forceinline__ void deform_tail_tensor_epilogue_v5(
    const float* tail_biases,
    float* output,
    int rows,
    Scratch& s,
    int head,
    int row_begin,
    int tid) {
    const int width = head < 2 ? 3 : (head == 2 ? 4 : (head == 3 ? 1 : 48));
    const int bias_offset = head == 0 ? 0 : (head == 1 ? 3 :
                            (head == 2 ? 6 : (head == 3 ? 10 : 11)));
    const int output_tiles = (width + 15) / 16;
    const int warp = tid / 32;
    const int lane = tid % 32;
    if (warp >= output_tiles) return;

    const int col = warp * 16;
    const float* tail_output =
        s.hidden + kDeformTensorTailOutputOffsetFloats;
    for (int e = lane; e < kDeformRows * 16; e += 32) {
        const int row = e / 16;
        const int column = col + e % 16;
        if (column < width && row_begin + row < rows) {
            output[
                static_cast<size_t>(row_begin + row) * width + column] =
                tail_output[row * 48 + column] +
                tail_biases[bias_offset + column];
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

// Standalone experimental Tensor-Core-tail path.  Deliberately not wired into
// deform_mlp_gptb_device_v1: the qualified mixed ABI remains byte-for-byte v5.
__device__ __forceinline__ void deform_mlp_tensor_tail_device_v3(
    const float* features,
    const float* backbone_weight,
    const float* backbone_bias,
    const half* first_weights,
    const float* first_biases,
    const half* packed_tail_weights,
    const float* tail_biases,
    float* output0,
    float* output1,
    float* output2,
    float* output3,
    float* output4,
    int rows,
    DeformMlpScratchV1& s) {
    const int tid = static_cast<int>(threadIdx.x);
    const int tiles = static_cast<int>(
        (static_cast<long long>(rows) + kDeformRows - 1) /
        kDeformRows);
    for (int tile = static_cast<int>(blockIdx.x); tile < tiles;
         tile += static_cast<int>(gridDim.x)) {
        const int row_begin = tile * kDeformRows;
        for (int e = tid; e < kDeformRows * kDeformInput;
             e += kDeformThreads) {
            const int row = row_begin + e / kDeformInput;
            s.features[e] = row < rows
                ? features[
                      static_cast<size_t>(row) * kDeformInput +
                      e % kDeformInput]
                : 0.0f;
        }
        __syncthreads();

        deform_backbone_linear_v3(
            features, backbone_weight, backbone_bias, rows,
            s, row_begin, tid);
        __syncthreads();
        deform_backbone_relu_v1(s, tid);
        __syncthreads();
        for (int head = 0; head < kDeformHeads; ++head) {
            deform_head_linear_v3(
                first_weights, first_biases, s, head, tid);
            __syncthreads();
            deform_head_relu_half_v3(s, tid);
            __syncthreads();
            float* output = head == 0 ? output0 : (head == 1 ? output1 :
                            (head == 2 ? output2 :
                            (head == 3 ? output3 : output4)));
            deform_tail_tensor_v3(
                packed_tail_weights, tail_biases, output, rows,
                s, head, row_begin, tid);
            __syncthreads();
        }
    }
}

// Shared implementation for standalone v4 and mixed ABI v6.  Synchronization
// is explicitly scoped by the caller: a 128-thread CTA for standalone, or a
// 128-participant named barrier for the Deform subgroup of a mixed CTA.
template <typename SyncPolicy>
__device__ __forceinline__ void deform_mlp_tensor_tail_device_impl_v4(
    const float* features,
    const float* backbone_weight,
    const float* backbone_bias,
    const half* first_weights,
    const float* first_biases,
    const half* packed_tail_weights,
    const float* tail_biases,
    float* output0,
    float* output1,
    float* output2,
    float* output3,
    float* output4,
    int rows,
    DeformMlpScratchV1& s,
    int tid,
    int tile_start,
    int tile_step,
    const SyncPolicy& worker_sync) {
    const int tiles = static_cast<int>(
        (static_cast<long long>(rows) + kDeformRows - 1) /
        kDeformRows);
    for (int tile = tile_start; tile < tiles; tile += tile_step) {
        const int row_begin = tile * kDeformRows;
        for (int e = tid; e < kDeformRows * kDeformInput;
             e += kDeformThreads) {
            const int row = row_begin + e / kDeformInput;
            s.features[e] = row < rows
                ? features[
                      static_cast<size_t>(row) * kDeformInput +
                      e % kDeformInput]
                : 0.0f;
        }
        worker_sync.sync();

        deform_backbone_linear_v3(
            features, backbone_weight, backbone_bias, rows,
            s, row_begin, tid);
        worker_sync.sync();
        deform_backbone_relu_v1(s, tid);
        worker_sync.sync();
        for (int head = 0; head < kDeformHeads; ++head) {
            deform_head_linear_v3(
                first_weights, first_biases, s, head, tid);
            worker_sync.sync();
            deform_head_relu_half_v4(s, tid, worker_sync);
            worker_sync.sync();
            float* output = head == 0 ? output0 : (head == 1 ? output1 :
                            (head == 2 ? output2 :
                            (head == 3 ? output3 : output4)));
            deform_tail_tensor_v4(
                packed_tail_weights, tail_biases, output, rows,
                s, head, row_begin, tid);
            worker_sync.sync();
        }
    }
}

// Standalone v4 keeps v3 intact and changes only the compact tail-activation
// layout.
__device__ __forceinline__ void deform_mlp_tensor_tail_device_v4(
    const float* features,
    const float* backbone_weight,
    const float* backbone_bias,
    const half* first_weights,
    const float* first_biases,
    const half* packed_tail_weights,
    const float* tail_biases,
    float* output0,
    float* output1,
    float* output2,
    float* output3,
    float* output4,
    int rows,
    DeformMlpScratchV1& s) {
    deform_mlp_tensor_tail_device_impl_v4(
        features, backbone_weight, backbone_bias, first_weights,
        first_biases, packed_tail_weights, tail_biases,
        output0, output1, output2, output3, output4, rows, s,
        static_cast<int>(threadIdx.x), static_cast<int>(blockIdx.x),
        static_cast<int>(gridDim.x), DeformMlpCtaSyncV1{});
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

// Experimental mixed ABI-v6 adapter.  Raster lanes return before the first
// barrier; only the contiguous 128-thread Deform subgroup participates.
__device__ __forceinline__ void deform_mlp_tensor_tail_gptb_device_v4(
    const DeformMlpTensorTailParamsV2& p,
    DeformMlpScratchV1& s,
    int tile_start,
    int tile_step,
    int thread_base,
    int barrier_id) {
    const int tid = static_cast<int>(threadIdx.x) - thread_base;
    if (tid < 0 || tid >= kDeformThreads || tile_step <= 0) {
        return;
    }
    deform_mlp_tensor_tail_device_impl_v4(
        p.features, p.backbone_weight, p.backbone_bias, p.first_weights,
        p.first_biases, p.packed_tail_weights, p.tail_biases,
        p.outputs[0], p.outputs[1], p.outputs[2], p.outputs[3], p.outputs[4],
        p.rows, s, tid, tile_start, tile_step,
        DeformMlpNamedSyncV1{barrier_id});
}

// ABI-v7 keeps the Tensor-tail descriptor in shared memory and reads its
// fields at the phase that consumes them.  Unlike v4's generic standalone
// implementation, it does not expand seven input and five output pointers
// into long-lived top-level arguments.  The four-value compaction further
// reduces the state that must survive a named barrier.
template <typename SyncPolicy>
__device__ __forceinline__ void deform_mlp_tensor_tail_device_impl_v5(
    const DeformMlpTensorTailParamsV2& p,
    DeformMlpScratchV1& s,
    int tid,
    int tile_start,
    int tile_step,
    const SyncPolicy& worker_sync) {
    const int tiles = static_cast<int>(
        (static_cast<long long>(p.rows) + kDeformRows - 1) /
        kDeformRows);
    for (int tile = tile_start; tile < tiles; tile += tile_step) {
        const int row_begin = tile * kDeformRows;
        for (int e = tid; e < kDeformRows * kDeformInput;
             e += kDeformThreads) {
            const int row = row_begin + e / kDeformInput;
            s.features[e] = row < p.rows
                ? p.features[
                      static_cast<size_t>(row) * kDeformInput +
                      e % kDeformInput]
                : 0.0f;
        }
        worker_sync.sync();

        deform_backbone_linear_v3(
            p.features, p.backbone_weight, p.backbone_bias, p.rows,
            s, row_begin, tid);
        worker_sync.sync();
        deform_backbone_relu_v1(s, tid);
        worker_sync.sync();
        for (int head = 0; head < kDeformHeads; ++head) {
            deform_head_linear_v3(
                p.first_weights, p.first_biases, s, head, tid);
            worker_sync.sync();
            deform_head_relu_half_v5(s, tid, worker_sync);
            worker_sync.sync();
            deform_tail_tensor_v4(
                p.packed_tail_weights,
                p.tail_biases,
                p.outputs[head],
                p.rows,
                s,
                head,
                row_begin,
                tid);
            worker_sync.sync();
        }
    }
}

__device__ __forceinline__ void deform_mlp_tensor_tail_gptb_device_v5(
    const DeformMlpTensorTailParamsV2& p,
    DeformMlpScratchV1& s,
    int tile_start,
    int tile_step,
    int thread_base,
    int barrier_id) {
    const int tid = static_cast<int>(threadIdx.x) - thread_base;
    if (tid < 0 || tid >= kDeformThreads || tile_step <= 0) {
        return;
    }
    deform_mlp_tensor_tail_device_impl_v5(
        p,
        s,
        tid,
        tile_start,
        tile_step,
        DeformMlpNamedSyncV1{barrier_id});
}

// ABI-v8 four-CTA experiment.  Volatile shared descriptor reads are copied
// into phase-local pointer variables so ptxas cannot hoist all model/output
// pointers across the complete MLP and its named barriers.
template <typename SyncPolicy>
__device__ __forceinline__ void deform_mlp_tensor_tail_device_impl_v6(
    const volatile DeformMlpTensorTailParamsV2& p,
    DeformMlpScratchV1& s,
    int tid,
    int tile_start,
    int tile_step,
    const SyncPolicy& worker_sync) {
    const int rows = p.rows;
    // Iterate directly in row space so no separate ceil-div tile count must
    // remain live across every MLP phase.
    for (int row_begin = tile_start * kDeformRows;
         row_begin < rows;
         row_begin += tile_step * kDeformRows) {
        {
            const float* phase_features = p.features;
            for (int e = tid; e < kDeformRows * kDeformInput;
                 e += kDeformThreads) {
                const int row = row_begin + e / kDeformInput;
                s.features[e] = row < rows
                    ? phase_features[
                          static_cast<size_t>(row) * kDeformInput +
                          e % kDeformInput]
                    : 0.0f;
            }
        }
        worker_sync.sync();

        {
            const float* phase_features = p.features;
            const float* phase_weight = p.backbone_weight;
            const float* phase_bias = p.backbone_bias;
            deform_backbone_linear_v3(
                phase_features, phase_weight, phase_bias, rows,
                s, row_begin, tid);
        }
        worker_sync.sync();
        deform_backbone_relu_v1(s, tid);
        worker_sync.sync();
        for (int head = 0; head < kDeformHeads; ++head) {
            {
                const half* phase_weight = p.first_weights;
                const float* phase_bias = p.first_biases;
                deform_head_linear_v3(
                    phase_weight, phase_bias, s, head, tid);
            }
            worker_sync.sync();
            deform_head_relu_half_v6(s, tid, worker_sync);
            worker_sync.sync();
            deform_tail_tensor_matmul_v6(
                p.packed_tail_weights, s, head, tid);
            worker_sync.sync();
            deform_tail_tensor_epilogue_v5(
                p.tail_biases,
                p.outputs[head],
                rows,
                s,
                head,
                row_begin,
                tid);
            worker_sync.sync();
        }
    }
}

__device__ __forceinline__ void deform_mlp_tensor_tail_gptb_device_v6(
    const DeformMlpTensorTailParamsV2& p,
    DeformMlpScratchV1& s,
    int tile_start,
    int tile_step,
    int thread_base,
    int barrier_id) {
    const int tid = static_cast<int>(threadIdx.x) - thread_base;
    if (tid < 0 || tid >= kDeformThreads || tile_step <= 0) {
        return;
    }
    deform_mlp_tensor_tail_device_impl_v6(
        reinterpret_cast<const volatile DeformMlpTensorTailParamsV2&>(p),
        s,
        tid,
        tile_start,
        tile_step,
        DeformMlpNamedSyncV1{barrier_id});
}

// v11 mixed-kernel implementation: padded first-head output plus fused
// bias/ReLU/compaction.  Other Deform phases remain identical to v6.
__device__ __forceinline__ void deform_tail_write_strided(
    const float* bias, float* output, int rows,
    const DeformMlpScratchV2& s, int head, int row_begin, int tid) {
    const int width = head < 2 ? 3 : (head == 2 ? 4 : (head == 3 ? 1 : 48));
    const int offset = head < 2 ? head * 3 : (head == 2 ? 6 : (head == 3 ? 10 : 11));
    const int lane = tid % 32;
    const int col = (tid / 32) * 16 + lane % 16;
    const int first = lane / 16;
    if (col >= width || row_begin + first >= rows) return;
    const float b = bias[offset + col];
    const float* src = s.hidden + kDeformTensorTailOutputOffsetFloats + col;
    const size_t base = static_cast<size_t>(row_begin) * width + col;
#pragma unroll 1
    for (int r = first; r < kDeformRows && row_begin + r < rows; r += 2)
        output[base + r * width] = src[r * 48] + b;
}

template <int BackboneUnroll = kDeformInput, bool ReuseWeights = false,
          bool PairedTile = false, typename Params, typename SyncPolicy>
__device__ __forceinline__ void deform_mlp_tensor_tail_device_impl_v7(
    const Params& p,
    DeformMlpScratchV2& s,
    int tid,
    int tile_start,
    int tile_step,
    const SyncPolicy& worker_sync) {
    const int rows = p.rows;
    for (int row_begin = tile_start * kDeformRows;
         row_begin < rows;
         row_begin += tile_step * kDeformRows) {
        {
            const float* phase_features = p.features;
            for (int e = tid; e < kDeformRows * kDeformInput;
                 e += kDeformThreads) {
                const int row = row_begin + e / kDeformInput;
                s.features[e] = row < rows
                    ? phase_features[
                          static_cast<size_t>(row) * kDeformInput +
                          e % kDeformInput]
                    : 0.0f;
            }
        }
        worker_sync.sync();

        if constexpr (ReuseWeights) {
            deform_backbone_reuse_weights(p.backbone_weight, p.backbone_bias, rows, s, row_begin, tid);
        } else {
            deform_backbone_linear_v4<BackboneUnroll>(
                p.backbone_weight, p.backbone_bias, rows, s, row_begin, tid);
            worker_sync.sync();
            deform_backbone_relu_v2(s, tid);
        }
        worker_sync.sync();
        for (int head = 0; head < kDeformHeads; ++head) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (PairedTile) {
                deform_head_linear_relu_streamed(p.first_weights, p.first_biases, s, head, tid);
            } else
#endif
            {
                deform_head_linear_v4(p.first_weights, s, head, tid);
                worker_sync.sync();
                deform_head_relu_half_v7(p.first_biases, s, head, tid, worker_sync);
            }
            worker_sync.sync();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (PairedTile) {
                deform_tail_streamed(p.packed_tail_weights, s, head, tid);
                worker_sync.sync();
                deform_tail_write_strided(p.tail_biases, p.outputs[head], rows, s, head, row_begin, tid);
            } else
#endif
            {
                deform_tail_tensor_matmul_v6(p.packed_tail_weights, s, head, tid);
                worker_sync.sync();
                deform_tail_tensor_epilogue_v5(p.tail_biases, p.outputs[head], rows, s, head, row_begin, tid);
            }
            worker_sync.sync();
        }
        if constexpr (PairedTile) break;
    }
}

template <int BackboneUnroll = kDeformInput, bool ReuseWeights = false>
__device__ __forceinline__ void deform_mlp_tensor_tail_gptb_device_v7(
    const DeformMlpTensorTailParamsV2& p,
    DeformMlpScratchV2& s,
    int tile_start,
    int tile_step,
    int thread_base,
    int barrier_id) {
    const int tid = static_cast<int>(threadIdx.x) - thread_base;
    if (tid < 0 || tid >= kDeformThreads || tile_step <= 0) {
        return;
    }
    deform_mlp_tensor_tail_device_impl_v7<BackboneUnroll, ReuseWeights>(
        reinterpret_cast<const volatile DeformMlpTensorTailParamsV2&>(p),
        s,
        tid,
        tile_start,
        tile_step,
        DeformMlpNamedSyncV1{barrier_id});
}

}  // namespace tacker_4dgs
