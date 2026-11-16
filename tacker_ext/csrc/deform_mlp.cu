#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>

#include "deform_mlp.h"

#ifndef TACKER_CC_MAJOR
#define TACKER_CC_MAJOR 8
#endif
#ifndef TACKER_CC_MINOR
#define TACKER_CC_MINOR 6
#endif
#include "deform_mlp_device.cuh"

namespace {

constexpr int kOutputWidths[] = {3, 3, 4, 1, 48};
constexpr int kSlicedOutputWidth = 59;
constexpr int kSlicedCount = 8;
constexpr int kSlicedRowsPerGroup = 128;
constexpr int kSlicedMicroTiles =
    kSlicedRowsPerGroup / tacker_4dgs::kDeformRows;
static_assert(kSlicedRowsPerGroup % tacker_4dgs::kDeformRows == 0,
              "sliced row group must contain complete WMMA row tiles");

void check_tensor(const torch::Tensor& value, const char* name,
                  at::ScalarType dtype, const c10::Device& device,
                  std::initializer_list<int64_t> shape) {
    TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(value.device() == device, name, " must be on the same CUDA device");
    TORCH_CHECK(value.scalar_type() == dtype, name, " has incorrect dtype");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(value.sizes() == at::IntArrayRef(shape), name, " has incorrect shape");
}

void check_inputs(const torch::Tensor& features,
                  const torch::Tensor& backbone_weight,
                  const torch::Tensor& backbone_bias,
                  const torch::Tensor& first_weights,
                  const torch::Tensor& first_biases,
                  const torch::Tensor& tail_weights,
                  const torch::Tensor& tail_biases,
                  int64_t persistent_blocks) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "deform_mlp is inference-only; call it under torch.no_grad()");
    TORCH_CHECK(features.is_cuda(), "features must be a CUDA tensor");
    TORCH_CHECK(features.dim() == 2 && features.size(1) == 32,
                "features must have shape [N,32]");
    TORCH_CHECK(features.size(0) <= INT32_MAX, "row count exceeds int32 ABI");
    TORCH_CHECK(persistent_blocks >= 0 && persistent_blocks <= INT32_MAX,
                "persistent_blocks must be in [0, INT32_MAX]");
    const auto device = features.device();
    check_tensor(features, "features", at::kFloat, device, {features.size(0), 32});
    check_tensor(backbone_weight, "backbone_weight_kh", at::kFloat, device, {32, 128});
    check_tensor(backbone_bias, "backbone_bias", at::kFloat, device, {128});
    check_tensor(first_weights, "first_weights", at::kHalf, device, {5, 128, 128});
    check_tensor(first_biases, "first_biases", at::kFloat, device, {5, 128});
    check_tensor(tail_weights, "tail_weights", at::kFloat, device, {59, 128});
    check_tensor(tail_biases, "tail_biases", at::kFloat, device, {59});
    TORCH_CHECK(reinterpret_cast<uintptr_t>(first_weights.data_ptr()) % 32 == 0,
                "first_weights must be 32-byte aligned for WMMA");
}

void check_tensor_tail_inputs(const torch::Tensor& features,
                              const torch::Tensor& backbone_weight,
                              const torch::Tensor& backbone_bias,
                              const torch::Tensor& first_weights,
                              const torch::Tensor& first_biases,
                              const torch::Tensor& packed_tail_weights,
                              const torch::Tensor& tail_biases,
                              int64_t persistent_blocks) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "deform_mlp_tensor_tail is inference-only; call it under "
                "torch.no_grad()");
    TORCH_CHECK(features.is_cuda(), "features must be a CUDA tensor");
    TORCH_CHECK(features.dim() == 2 && features.size(1) == 32,
                "features must have shape [N,32]");
    TORCH_CHECK(features.size(0) <= INT32_MAX, "row count exceeds int32 ABI");
    TORCH_CHECK(persistent_blocks >= 0 && persistent_blocks <= INT32_MAX,
                "persistent_blocks must be in [0, INT32_MAX]");
    const auto device = features.device();
    check_tensor(features, "features", at::kFloat, device,
                 {features.size(0), 32});
    check_tensor(backbone_weight, "backbone_weight_kh", at::kFloat, device,
                 {32, 128});
    check_tensor(backbone_bias, "backbone_bias", at::kFloat, device, {128});
    check_tensor(first_weights, "first_weights", at::kHalf, device,
                 {5, 128, 128});
    check_tensor(first_biases, "first_biases", at::kFloat, device,
                 {5, 128});
    check_tensor(packed_tail_weights, "packed_tail_weights", at::kHalf,
                 device, {tacker_4dgs::kDeformTensorTailRows, 128});
    check_tensor(tail_biases, "tail_biases", at::kFloat, device, {59});
    TORCH_CHECK(reinterpret_cast<uintptr_t>(first_weights.data_ptr()) % 32 == 0,
                "first_weights must be 32-byte aligned for WMMA");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(packed_tail_weights.data_ptr()) % 32 == 0,
        "packed_tail_weights must be 32-byte aligned for WMMA");
}

void check_sliced_inputs(const torch::Tensor& features,
                         const torch::Tensor& backbone_weight,
                         const torch::Tensor& backbone_bias,
                         const torch::Tensor& first_weights,
                         const torch::Tensor& first_biases,
                         const torch::Tensor& tail_weights) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "deform_mlp_sliced_v2 is inference-only; call it under "
                "torch.no_grad()");
    TORCH_CHECK(features.is_cuda(), "features must be a CUDA tensor");
    TORCH_CHECK(features.dim() == 2 && features.size(1) == 32,
                "features must have shape [N,32]");
    TORCH_CHECK(features.size(0) <= INT32_MAX, "row count exceeds int32 ABI");
    const auto device = features.device();
    check_tensor(features, "features", at::kFloat, device, {features.size(0), 32});
    check_tensor(backbone_weight, "backbone_weight_kh", at::kFloat, device,
                 {32, 128});
    check_tensor(backbone_bias, "backbone_bias", at::kFloat, device, {128});
    check_tensor(first_weights, "first_weights", at::kHalf, device,
                 {5, 128, 128});
    check_tensor(first_biases, "first_biases", at::kFloat, device, {5, 128});
    check_tensor(tail_weights, "tail_weights", at::kFloat, device, {59, 128});
    TORCH_CHECK(reinterpret_cast<uintptr_t>(first_weights.data_ptr()) % 32 == 0,
                "first_weights must be 32-byte aligned for WMMA");
}

bool overlaps(const torch::Tensor& a, const torch::Tensor& b) {
    if (a.numel() == 0 || b.numel() == 0) return false;
    const uintptr_t pa = reinterpret_cast<uintptr_t>(a.data_ptr());
    const uintptr_t pb = reinterpret_cast<uintptr_t>(b.data_ptr());
    // All tensors have already passed fixed-shape/row-bound validation.
    const uint64_t na = static_cast<uint64_t>(a.numel()) * a.element_size();
    const uint64_t nb = static_cast<uint64_t>(b.numel()) * b.element_size();
    return pa <= pb ? static_cast<uint64_t>(pb - pa) < na
                    : static_cast<uint64_t>(pa - pb) < nb;
}

struct __align__(32) DeformBackboneScratchV2 {
    float features[tacker_4dgs::kDeformRows * tacker_4dgs::kDeformInput];
};

struct __align__(32) DeformHeadScratchV2 {
    half activated[
        tacker_4dgs::kDeformRows * tacker_4dgs::kDeformActivatedStride];
    float hidden[tacker_4dgs::kDeformRows * 16];
};

static_assert(sizeof(DeformBackboneScratchV2) == 2048,
              "unexpected sliced backbone scratch layout");
static_assert(sizeof(DeformHeadScratchV2) == 5376,
              "unexpected per-head scratch layout");

template <int Head>
__device__ __forceinline__ void compute_sliced_head_tile_v2(
    const half* activated,
    const half* first_weights,
    const float* first_biases,
    int slice,
    float* hidden) {
    namespace wmma = nvcuda::wmma;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int hidden_begin = slice * 16;
    const half* weight = first_weights +
        Head * tacker_4dgs::kDeformWidth * tacker_4dgs::kDeformWidth;
    const float* bias = first_biases + Head * tacker_4dgs::kDeformWidth;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
#pragma unroll
    for (int k = 0; k < tacker_4dgs::kDeformWidth; k += 16) {
        wmma::load_matrix_sync(
            a, activated + k, tacker_4dgs::kDeformActivatedStride);
        wmma::load_matrix_sync(
            b, weight + hidden_begin * tacker_4dgs::kDeformWidth + k,
            tacker_4dgs::kDeformWidth);
        wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(hidden, acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < tacker_4dgs::kDeformRows * 16; e += 32) {
        const int column = e % 16;
        hidden[e] = tacker_4dgs::deform_relu(
            hidden[e] + bias[hidden_begin + column]);
    }
    __syncwarp();
}

template <int Width, int Offset>
__device__ __forceinline__ void store_sliced_tail_partial_v2(
    const float* hidden,
    const float* tail_weights,
    float* partials,
    int rows,
    int row_begin,
    int slice,
    int local_tid,
    int local_step) {
    const int hidden_begin = slice * 16;
    for (int u = local_tid; u < tacker_4dgs::kDeformRows * Width;
         u += local_step) {
        const int row = u / Width;
        const int column = u % Width;
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum = fmaf(
                hidden[row * 16 + k],
                tail_weights[
                    (Offset + column) * tacker_4dgs::kDeformWidth +
                    hidden_begin + k],
                sum);
        }
        const int global_row = row_begin + row;
        if (global_row < rows) {
            partials[
                (static_cast<size_t>(slice) * rows + global_row) *
                    kSlicedOutputWidth +
                Offset + column] = sum;
        }
    }
}

template <int Head, int Width, int Offset>
__device__ __forceinline__ void run_sliced_head_v2(
    const half* activated,
    const half* first_weights,
    const float* first_biases,
    const float* tail_weights,
    float* partials,
    int rows,
    DeformHeadScratchV2& scratch) {
    const int tid = static_cast<int>(threadIdx.x);
    const int row_group_begin = static_cast<int>(blockIdx.x) *
        kSlicedRowsPerGroup;
    const int slice = static_cast<int>(blockIdx.y);
    int micro_tiles =
        (rows - row_group_begin + tacker_4dgs::kDeformRows - 1) /
        tacker_4dgs::kDeformRows;
    if (micro_tiles > kSlicedMicroTiles) micro_tiles = kSlicedMicroTiles;

    for (int micro = 0; micro < micro_tiles; ++micro) {
        const int row_begin = row_group_begin +
            micro * tacker_4dgs::kDeformRows;
        for (int e = tid;
             e < tacker_4dgs::kDeformRows * tacker_4dgs::kDeformWidth;
             e += blockDim.x) {
            const int row = e / tacker_4dgs::kDeformWidth;
            const int column = e % tacker_4dgs::kDeformWidth;
            const int global_row = row_begin + row;
            scratch.activated[
                row * tacker_4dgs::kDeformActivatedStride + column] =
                global_row < rows
                    ? activated[
                          static_cast<size_t>(global_row) *
                              tacker_4dgs::kDeformWidth +
                          column]
                    : __float2half_rn(0.0f);
        }
        __syncthreads();

        // A single warp owns the 16x16 WMMA slab for this head.  Tail work is
        // then distributed across the head-specific block size.
        if (tid < 32) {
            compute_sliced_head_tile_v2<Head>(
                scratch.activated, first_weights, first_biases,
                slice, scratch.hidden);
        }
        __syncthreads();
        store_sliced_tail_partial_v2<Width, Offset>(
            scratch.hidden, tail_weights, partials,
            rows, row_begin, slice, tid, blockDim.x);
        __syncthreads();
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(128)
tacker_deform_mlp_delta_v1(tacker_4dgs::DeformMlpParamsV1 params) {
    __shared__ tacker_4dgs::DeformMlpScratchV1 scratch;
    tacker_4dgs::deform_mlp_device_v1(params, scratch);
}

extern "C" __global__ void __launch_bounds__(128)
tacker_deform_mlp_tensor_tail_v3(
    tacker_4dgs::DeformMlpParamsV1 params,
    const half* packed_tail_weights) {
    __shared__ tacker_4dgs::DeformMlpScratchV1 scratch;
    tacker_4dgs::deform_mlp_tensor_tail_device_v3(
        params.features, params.backbone_weight, params.backbone_bias,
        params.first_weights, params.first_biases, packed_tail_weights,
        params.tail_biases,
        params.outputs[0], params.outputs[1], params.outputs[2],
        params.outputs[3], params.outputs[4], params.rows, scratch);
}

extern "C" __global__ void __launch_bounds__(128)
tacker_deform_mlp_tensor_tail_padded_v4(
    tacker_4dgs::DeformMlpParamsV1 params,
    const half* packed_tail_weights) {
    __shared__ tacker_4dgs::DeformMlpScratchV1 scratch;
    tacker_4dgs::deform_mlp_tensor_tail_device_v4(
        params.features, params.backbone_weight, params.backbone_bias,
        params.first_weights, params.first_biases, packed_tail_weights,
        params.tail_biases,
        params.outputs[0], params.outputs[1], params.outputs[2],
        params.outputs[3], params.outputs[4], params.rows, scratch);
}

extern "C" __global__ void __launch_bounds__(128)
tacker_deform_mlp_backbone_v2(
    const float* features,
    const float* backbone_weight,
    const float* backbone_bias,
    half* activated,
    int rows) {
    __shared__ DeformBackboneScratchV2 scratch;
    const int tid = static_cast<int>(threadIdx.x);
    const int row_group_begin = static_cast<int>(blockIdx.x) *
        kSlicedRowsPerGroup;
    int micro_tiles =
        (rows - row_group_begin + tacker_4dgs::kDeformRows - 1) /
        tacker_4dgs::kDeformRows;
    if (micro_tiles > kSlicedMicroTiles) micro_tiles = kSlicedMicroTiles;

    // One CTA owns up to 128 rows, but retains the proven 16-row shared-memory
    // micro-tile.  This cuts the grid by 8x without increasing shared memory.
    for (int micro = 0; micro < micro_tiles; ++micro) {
        const int row_begin = row_group_begin +
            micro * tacker_4dgs::kDeformRows;
        for (int e = tid;
             e < tacker_4dgs::kDeformRows * tacker_4dgs::kDeformInput;
             e += blockDim.x) {
            const int row = row_begin + e / tacker_4dgs::kDeformInput;
            scratch.features[e] = row < rows
                ? features[
                      static_cast<size_t>(row) * tacker_4dgs::kDeformInput +
                      e % tacker_4dgs::kDeformInput]
                : 0.0f;
        }
        __syncthreads();

        if (tid < tacker_4dgs::kDeformWidth) {
#pragma unroll
            for (int row = 0; row < tacker_4dgs::kDeformRows; ++row) {
                float sum = 0.0f;
                if (row_begin + row < rows) {
#pragma unroll
                    for (int k = 0; k < tacker_4dgs::kDeformInput; ++k) {
                        sum = fmaf(
                            scratch.features[
                                row * tacker_4dgs::kDeformInput + k],
                            backbone_weight[
                                k * tacker_4dgs::kDeformWidth + tid],
                            sum);
                    }
                    sum += backbone_bias[tid];
                    activated[
                        static_cast<size_t>(row_begin + row) *
                            tacker_4dgs::kDeformWidth +
                        tid] = __float2half_rn(
                            tacker_4dgs::deform_relu(sum));
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(32)
tacker_deform_mlp_sliced_head0_v2(
    const half* activated,
    const half* first_weights,
    const float* first_biases,
    const float* tail_weights,
    float* partials,
    int rows) {
    __shared__ DeformHeadScratchV2 scratch;
    run_sliced_head_v2<0, 3, 0>(
        activated, first_weights, first_biases, tail_weights,
        partials, rows, scratch);
}

extern "C" __global__ void __launch_bounds__(32)
tacker_deform_mlp_sliced_head1_v2(
    const half* activated,
    const half* first_weights,
    const float* first_biases,
    const float* tail_weights,
    float* partials,
    int rows) {
    __shared__ DeformHeadScratchV2 scratch;
    run_sliced_head_v2<1, 3, 3>(
        activated, first_weights, first_biases, tail_weights,
        partials, rows, scratch);
}

extern "C" __global__ void __launch_bounds__(32)
tacker_deform_mlp_sliced_head2_v2(
    const half* activated,
    const half* first_weights,
    const float* first_biases,
    const float* tail_weights,
    float* partials,
    int rows) {
    __shared__ DeformHeadScratchV2 scratch;
    run_sliced_head_v2<2, 4, 6>(
        activated, first_weights, first_biases, tail_weights,
        partials, rows, scratch);
}

extern "C" __global__ void __launch_bounds__(32)
tacker_deform_mlp_sliced_head3_v2(
    const half* activated,
    const half* first_weights,
    const float* first_biases,
    const float* tail_weights,
    float* partials,
    int rows) {
    __shared__ DeformHeadScratchV2 scratch;
    run_sliced_head_v2<3, 1, 10>(
        activated, first_weights, first_biases, tail_weights,
        partials, rows, scratch);
}

extern "C" __global__ void __launch_bounds__(128)
tacker_deform_mlp_sliced_head4_v2(
    const half* activated,
    const half* first_weights,
    const float* first_biases,
    const float* tail_weights,
    float* partials,
    int rows) {
    __shared__ DeformHeadScratchV2 scratch;
    run_sliced_head_v2<4, 48, 11>(
        activated, first_weights, first_biases, tail_weights,
        partials, rows, scratch);
}

std::vector<torch::Tensor> deform_mlp_delta_out_v1_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& tail_weights,
    const torch::Tensor& tail_biases,
    const std::vector<torch::Tensor>& outputs,
    int64_t persistent_blocks) {
    check_inputs(features, backbone_weight_kh, backbone_bias, first_weights,
                 first_biases, tail_weights, tail_biases, persistent_blocks);
    TORCH_CHECK(outputs.size() == 5, "outputs must contain exactly five tensors");
    const std::array<torch::Tensor, 7> inputs = {
        features, backbone_weight_kh, backbone_bias, first_weights,
        first_biases, tail_weights, tail_biases};
    for (int i = 0; i < 5; ++i) {
        check_tensor(outputs[i], "output", at::kFloat, features.device(),
                     {features.size(0), kOutputWidths[i]});
        TORCH_CHECK(!outputs[i].requires_grad(), "outputs must not require gradients");
        for (const auto& input : inputs) {
            TORCH_CHECK(!overlaps(outputs[i], input),
                        "outputs must not overlap any input storage");
        }
        for (int j = 0; j < i; ++j) {
            TORCH_CHECK(!overlaps(outputs[i], outputs[j]),
                        "outputs must not overlap each other");
        }
    }
    const c10::cuda::CUDAGuard device_guard(features.device());
    const int rows = static_cast<int>(features.size(0));
    if (rows == 0) return outputs;
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        properties->major == TACKER_CC_MAJOR &&
            properties->minor == TACKER_CC_MINOR,
        "deform_mlp_v1 GPU capability does not match its build target");
    const int tiles = static_cast<int>((static_cast<int64_t>(rows) + 15) / 16);
    // The old automatic policy launched one block per SM.  This kernel has
    // only four warps per block and several CTA-wide barriers, so that policy
    // achieved just one resident block (8.33% occupancy) per SM on A6000.
    // A full logical grid lets the hardware keep the occupancy-limited six
    // blocks resident per SM and was 2.48x faster at the real 111,525-row
    // workload.  Positive values retain the explicit persistent-grid mode.
    const int blocks = persistent_blocks == 0
        ? tiles
        : std::min(tiles, static_cast<int>(persistent_blocks));
    tacker_4dgs::DeformMlpParamsV1 p{};
    p.features = features.data_ptr<float>();
    p.backbone_weight = backbone_weight_kh.data_ptr<float>();
    p.backbone_bias = backbone_bias.data_ptr<float>();
    p.first_weights = reinterpret_cast<const half*>(first_weights.data_ptr<at::Half>());
    p.first_biases = first_biases.data_ptr<float>();
    p.tail_weights = tail_weights.data_ptr<float>();
    p.tail_biases = tail_biases.data_ptr<float>();
    p.rows = rows;
    for (int i = 0; i < 5; ++i) p.outputs[i] = outputs[i].data_ptr<float>();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    tacker_deform_mlp_delta_v1<<<blocks, tacker_4dgs::kDeformThreads, 0, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return outputs;
}

std::vector<torch::Tensor> deform_mlp_delta_v1_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& tail_weights,
    const torch::Tensor& tail_biases,
    int64_t persistent_blocks) {
    check_inputs(features, backbone_weight_kh, backbone_bias, first_weights,
                 first_biases, tail_weights, tail_biases, persistent_blocks);
    std::vector<torch::Tensor> outputs;
    for (const int width : kOutputWidths) {
        outputs.push_back(torch::empty({features.size(0), width},
                                      features.options().requires_grad(false)));
    }
    return deform_mlp_delta_out_v1_cuda(features, backbone_weight_kh, backbone_bias,
        first_weights, first_biases, tail_weights, tail_biases, outputs,
        persistent_blocks);
}

namespace {

std::vector<torch::Tensor> deform_mlp_tensor_tail_out_impl(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& packed_tail_weights,
    const torch::Tensor& tail_biases,
    const std::vector<torch::Tensor>& outputs,
    int64_t persistent_blocks,
    bool padded_tail_input) {
    check_tensor_tail_inputs(
        features, backbone_weight_kh, backbone_bias, first_weights,
        first_biases, packed_tail_weights, tail_biases, persistent_blocks);
    TORCH_CHECK(outputs.size() == 5,
                "outputs must contain exactly five tensors");
    const std::array<torch::Tensor, 7> inputs = {
        features, backbone_weight_kh, backbone_bias, first_weights,
        first_biases, packed_tail_weights, tail_biases};
    for (int i = 0; i < 5; ++i) {
        check_tensor(outputs[i], "output", at::kFloat, features.device(),
                     {features.size(0), kOutputWidths[i]});
        TORCH_CHECK(!outputs[i].requires_grad(),
                    "outputs must not require gradients");
        for (const auto& input : inputs) {
            TORCH_CHECK(!overlaps(outputs[i], input),
                        "outputs must not overlap any input storage");
        }
        for (int j = 0; j < i; ++j) {
            TORCH_CHECK(!overlaps(outputs[i], outputs[j]),
                        "outputs must not overlap each other");
        }
    }

    const c10::cuda::CUDAGuard device_guard(features.device());
    const int rows = static_cast<int>(features.size(0));
    if (rows == 0) return outputs;
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        properties->major == TACKER_CC_MAJOR &&
            properties->minor == TACKER_CC_MINOR,
        "deform_mlp_tensor_tail GPU capability does not match its build "
        "target");
    const int tiles = static_cast<int>(
        (static_cast<int64_t>(rows) + tacker_4dgs::kDeformRows - 1) /
        tacker_4dgs::kDeformRows);
    const int blocks = persistent_blocks == 0
        ? tiles
        : std::min(tiles, static_cast<int>(persistent_blocks));

    tacker_4dgs::DeformMlpParamsV1 p{};
    p.features = features.data_ptr<float>();
    p.backbone_weight = backbone_weight_kh.data_ptr<float>();
    p.backbone_bias = backbone_bias.data_ptr<float>();
    p.first_weights = reinterpret_cast<const half*>(
        first_weights.data_ptr<at::Half>());
    p.first_biases = first_biases.data_ptr<float>();
    p.tail_weights = nullptr;
    p.tail_biases = tail_biases.data_ptr<float>();
    p.rows = rows;
    for (int i = 0; i < 5; ++i) {
        p.outputs[i] = outputs[i].data_ptr<float>();
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const half* packed_tail = reinterpret_cast<const half*>(
        packed_tail_weights.data_ptr<at::Half>());
    if (padded_tail_input) {
        tacker_deform_mlp_tensor_tail_padded_v4<<<
            blocks, tacker_4dgs::kDeformThreads, 0, stream>>>(p, packed_tail);
    } else {
        tacker_deform_mlp_tensor_tail_v3<<<
            blocks, tacker_4dgs::kDeformThreads, 0, stream>>>(p, packed_tail);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return outputs;
}

}  // namespace

std::vector<torch::Tensor> deform_mlp_tensor_tail_out_v3_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& packed_tail_weights,
    const torch::Tensor& tail_biases,
    const std::vector<torch::Tensor>& outputs,
    int64_t persistent_blocks) {
    return deform_mlp_tensor_tail_out_impl(
        features, backbone_weight_kh, backbone_bias, first_weights,
        first_biases, packed_tail_weights, tail_biases, outputs,
        persistent_blocks, false);
}

std::vector<torch::Tensor> deform_mlp_tensor_tail_out_v4_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& packed_tail_weights,
    const torch::Tensor& tail_biases,
    const std::vector<torch::Tensor>& outputs,
    int64_t persistent_blocks) {
    return deform_mlp_tensor_tail_out_impl(
        features, backbone_weight_kh, backbone_bias, first_weights,
        first_biases, packed_tail_weights, tail_biases, outputs,
        persistent_blocks, true);
}

std::vector<torch::Tensor> deform_mlp_tensor_tail_v3_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& packed_tail_weights,
    const torch::Tensor& tail_biases,
    int64_t persistent_blocks) {
    check_tensor_tail_inputs(
        features, backbone_weight_kh, backbone_bias, first_weights,
        first_biases, packed_tail_weights, tail_biases, persistent_blocks);
    std::vector<torch::Tensor> outputs;
    for (const int width : kOutputWidths) {
        outputs.push_back(torch::empty(
            {features.size(0), width},
            features.options().requires_grad(false)));
    }
    return deform_mlp_tensor_tail_out_v3_cuda(
        features, backbone_weight_kh, backbone_bias, first_weights,
        first_biases, packed_tail_weights, tail_biases, outputs,
        persistent_blocks);
}

std::vector<torch::Tensor> deform_mlp_tensor_tail_v4_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& packed_tail_weights,
    const torch::Tensor& tail_biases,
    int64_t persistent_blocks) {
    check_tensor_tail_inputs(
        features, backbone_weight_kh, backbone_bias, first_weights,
        first_biases, packed_tail_weights, tail_biases, persistent_blocks);
    std::vector<torch::Tensor> outputs;
    for (const int width : kOutputWidths) {
        outputs.push_back(torch::empty(
            {features.size(0), width},
            features.options().requires_grad(false)));
    }
    return deform_mlp_tensor_tail_out_v4_cuda(
        features, backbone_weight_kh, backbone_bias, first_weights,
        first_biases, packed_tail_weights, tail_biases, outputs,
        persistent_blocks);
}

std::vector<torch::Tensor> deform_mlp_sliced_out_v2_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& tail_weights,
    const torch::Tensor& activated,
    const torch::Tensor& partials) {
    check_sliced_inputs(features, backbone_weight_kh, backbone_bias,
                        first_weights, first_biases, tail_weights);
    const auto device = features.device();
    check_tensor(activated, "activated", at::kHalf, device,
                 {features.size(0), tacker_4dgs::kDeformWidth});
    check_tensor(partials, "partials", at::kFloat, device,
                 {kSlicedCount, features.size(0), kSlicedOutputWidth});
    TORCH_CHECK(!activated.requires_grad(),
                "activated must not require gradients");
    TORCH_CHECK(!partials.requires_grad(),
                "partials must not require gradients");

    const std::array<torch::Tensor, 6> inputs = {
        features, backbone_weight_kh, backbone_bias,
        first_weights, first_biases, tail_weights};
    for (const auto& input : inputs) {
        TORCH_CHECK(!overlaps(activated, input),
                    "activated must not overlap any input storage");
        TORCH_CHECK(!overlaps(partials, input),
                    "partials must not overlap any input storage");
    }
    TORCH_CHECK(!overlaps(activated, partials),
                "activated and partials must not overlap");

    const c10::cuda::CUDAGuard device_guard(device);
    const int rows = static_cast<int>(features.size(0));
    if (rows == 0) return {activated, partials};
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        properties->major == TACKER_CC_MAJOR &&
            properties->minor == TACKER_CC_MINOR,
        "deform_mlp_sliced_v2 GPU capability does not match its build target");
    const int groups = static_cast<int>(
        (static_cast<int64_t>(rows) + kSlicedRowsPerGroup - 1) /
        kSlicedRowsPerGroup);
    const auto current_stream = at::cuda::getCurrentCUDAStream();
    const cudaStream_t stream = current_stream.stream();
    tacker_deform_mlp_backbone_v2<<<
        groups, tacker_4dgs::kDeformThreads, 0, stream>>>(
        features.data_ptr<float>(),
        backbone_weight_kh.data_ptr<float>(),
        backbone_bias.data_ptr<float>(),
        reinterpret_cast<half*>(activated.data_ptr<at::Half>()),
        rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 head_grid(
        static_cast<unsigned int>(groups), kSlicedCount, 1);
    const half* activated_ptr =
        reinterpret_cast<const half*>(activated.data_ptr<at::Half>());
    const half* first_weights_ptr =
        reinterpret_cast<const half*>(first_weights.data_ptr<at::Half>());
    const float* first_biases_ptr = first_biases.data_ptr<float>();
    const float* tail_weights_ptr = tail_weights.data_ptr<float>();
    float* partials_ptr = partials.data_ptr<float>();

    // Publish the backbone activation to five independent streams.  Joining
    // all five streams back into the caller's current stream before returning
    // preserves normal PyTorch stream semantics: an immediately following
    // partials.sum() cannot race any head kernel.
    at::cuda::CUDAEvent backbone_ready;
    backbone_ready.record(current_stream);
    std::array<c10::cuda::CUDAStream, 5> head_streams = {
        c10::cuda::getStreamFromPool(false, device.index()),
        c10::cuda::getStreamFromPool(false, device.index()),
        c10::cuda::getStreamFromPool(false, device.index()),
        c10::cuda::getStreamFromPool(false, device.index()),
        c10::cuda::getStreamFromPool(false, device.index()),
    };
    std::array<at::cuda::CUDAEvent, 5> head_done;
    for (const auto& head_stream : head_streams) {
        backbone_ready.block(head_stream);
    }

#define TACKER_LAUNCH_SLICED_HEAD_V2(index, kernel, threads)                  \
    kernel<<<head_grid, threads, 0, head_streams[index].stream()>>>(          \
        activated_ptr, first_weights_ptr, first_biases_ptr,                  \
        tail_weights_ptr, partials_ptr, rows);                               \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    head_done[index].record(head_streams[index])
    TACKER_LAUNCH_SLICED_HEAD_V2(
        0, tacker_deform_mlp_sliced_head0_v2, 32);
    TACKER_LAUNCH_SLICED_HEAD_V2(
        1, tacker_deform_mlp_sliced_head1_v2, 32);
    TACKER_LAUNCH_SLICED_HEAD_V2(
        2, tacker_deform_mlp_sliced_head2_v2, 32);
    TACKER_LAUNCH_SLICED_HEAD_V2(
        3, tacker_deform_mlp_sliced_head3_v2, 32);
    TACKER_LAUNCH_SLICED_HEAD_V2(
        4, tacker_deform_mlp_sliced_head4_v2,
        tacker_4dgs::kDeformThreads);
#undef TACKER_LAUNCH_SLICED_HEAD_V2
    for (auto& done : head_done) {
        done.block(current_stream);
    }
    return {activated, partials};
}

std::vector<torch::Tensor> deform_mlp_sliced_v2_cuda(
    const torch::Tensor& features,
    const torch::Tensor& backbone_weight_kh,
    const torch::Tensor& backbone_bias,
    const torch::Tensor& first_weights,
    const torch::Tensor& first_biases,
    const torch::Tensor& tail_weights) {
    check_sliced_inputs(features, backbone_weight_kh, backbone_bias,
                        first_weights, first_biases, tail_weights);
    const auto rows = features.size(0);
    torch::Tensor activated = torch::empty(
        {rows, tacker_4dgs::kDeformWidth},
        features.options().dtype(torch::kFloat16).requires_grad(false));
    torch::Tensor partials = torch::empty(
        {kSlicedCount, rows, kSlicedOutputWidth},
        features.options().dtype(torch::kFloat32).requires_grad(false));
    return deform_mlp_sliced_out_v2_cuda(
        features, backbone_weight_kh, backbone_bias,
        first_weights, first_biases, tail_weights,
        activated, partials);
}
