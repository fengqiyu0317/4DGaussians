#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>

#include "deform_mlp.h"
#include "deform_mlp_device.cuh"

namespace {

constexpr int kOutputWidths[] = {3, 3, 4, 1, 48};
constexpr int kSlicedOutputWidth = 59;
constexpr int kSlicedCount = 8;

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

struct __align__(32) DeformSliceScratchV2 {
    half activated[
        tacker_4dgs::kDeformRows * tacker_4dgs::kDeformActivatedStride];
    // Heads 0--3 execute concurrently, one per warp.  Head 4 reuses slot 0.
    float hidden[4][tacker_4dgs::kDeformRows * 16];
};

static_assert(sizeof(DeformBackboneScratchV2) == 2048,
              "unexpected sliced backbone scratch layout");
static_assert(sizeof(DeformSliceScratchV2) == 8448,
              "unexpected sliced head scratch layout");

__device__ __forceinline__ void compute_sliced_head_tile_v2(
    const half* activated,
    const half* first_weights,
    const float* first_biases,
    int head,
    int slice,
    float* hidden) {
    namespace wmma = nvcuda::wmma;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int hidden_begin = slice * 16;
    const half* weight = first_weights +
        head * tacker_4dgs::kDeformWidth * tacker_4dgs::kDeformWidth;
    const float* bias = first_biases + head * tacker_4dgs::kDeformWidth;

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

__device__ __forceinline__ void store_sliced_tail_partial_v2(
    const float* hidden,
    const float* tail_weights,
    float* partials,
    int rows,
    int row_begin,
    int head,
    int slice,
    int local_tid,
    int local_step) {
    const int width = head < 2 ? 3 : (head == 2 ? 4 : (head == 3 ? 1 : 48));
    const int offset = head == 0 ? 0 : (head == 1 ? 3 :
                       (head == 2 ? 6 : (head == 3 ? 10 : 11)));
    const int hidden_begin = slice * 16;
    for (int u = local_tid; u < tacker_4dgs::kDeformRows * width;
         u += local_step) {
        const int row = u / width;
        const int column = u % width;
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum = fmaf(
                hidden[row * 16 + k],
                tail_weights[
                    (offset + column) * tacker_4dgs::kDeformWidth +
                    hidden_begin + k],
                sum);
        }
        const int global_row = row_begin + row;
        if (global_row < rows) {
            partials[
                (static_cast<size_t>(slice) * rows + global_row) *
                    kSlicedOutputWidth +
                offset + column] = sum;
        }
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(128)
tacker_deform_mlp_delta_v1(tacker_4dgs::DeformMlpParamsV1 params) {
    __shared__ tacker_4dgs::DeformMlpScratchV1 scratch;
    tacker_4dgs::deform_mlp_device_v1(params, scratch);
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
    const int row_begin = static_cast<int>(blockIdx.x) *
        tacker_4dgs::kDeformRows;
    for (int e = tid;
         e < tacker_4dgs::kDeformRows * tacker_4dgs::kDeformInput;
         e += blockDim.x) {
        const int row = row_begin + e / tacker_4dgs::kDeformInput;
        scratch.features[e] = row < rows
            ? features[static_cast<size_t>(row) * tacker_4dgs::kDeformInput +
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
                        scratch.features[row * tacker_4dgs::kDeformInput + k],
                        backbone_weight[k * tacker_4dgs::kDeformWidth + tid],
                        sum);
                }
                sum += backbone_bias[tid];
                activated[
                    static_cast<size_t>(row_begin + row) *
                        tacker_4dgs::kDeformWidth +
                    tid] = __float2half_rn(tacker_4dgs::deform_relu(sum));
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(128)
tacker_deform_mlp_sliced_heads_v2(
    const half* activated,
    const half* first_weights,
    const float* first_biases,
    const float* tail_weights,
    float* partials,
    int rows) {
    __shared__ DeformSliceScratchV2 scratch;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int row_begin = static_cast<int>(blockIdx.x) *
        tacker_4dgs::kDeformRows;
    const int slice = static_cast<int>(blockIdx.y);

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

    // The first four heads execute concurrently, one complete 128x16 WMMA
    // slab per warp.  Their tail partials are warp-private and need no CTA
    // barrier beyond the activation staging above.
    if (warp < 4) {
        compute_sliced_head_tile_v2(
            scratch.activated, first_weights, first_biases,
            warp, slice, scratch.hidden[warp]);
        store_sliced_tail_partial_v2(
            scratch.hidden[warp], tail_weights, partials,
            rows, row_begin, warp, slice, lane, 32);
    }
    __syncthreads();

    // SH is the fifth head.  Warp 0 produces its 16 hidden channels, then all
    // four warps share the 48-wide tail work.
    if (warp == 0) {
        compute_sliced_head_tile_v2(
            scratch.activated, first_weights, first_biases,
            4, slice, scratch.hidden[0]);
    }
    __syncthreads();
    store_sliced_tail_partial_v2(
        scratch.hidden[0], tail_weights, partials,
        rows, row_begin, 4, slice, tid, blockDim.x);
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
    TORCH_CHECK(properties->major == 8 && properties->minor == 6,
                "deform_mlp_v1 requires the sm_86 build target");
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
    TORCH_CHECK(properties->major == 8 && properties->minor == 6,
                "deform_mlp_sliced_v2 requires the sm_86 build target");
    const int tiles = static_cast<int>(
        (static_cast<int64_t>(rows) + tacker_4dgs::kDeformRows - 1) /
        tacker_4dgs::kDeformRows);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    tacker_deform_mlp_backbone_v2<<<
        tiles, tacker_4dgs::kDeformThreads, 0, stream>>>(
        features.data_ptr<float>(),
        backbone_weight_kh.data_ptr<float>(),
        backbone_bias.data_ptr<float>(),
        reinterpret_cast<half*>(activated.data_ptr<at::Half>()),
        rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    tacker_deform_mlp_sliced_heads_v2<<<
        dim3(static_cast<unsigned int>(tiles), kSlicedCount, 1),
        tacker_4dgs::kDeformThreads, 0, stream>>>(
        reinterpret_cast<const half*>(activated.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(first_weights.data_ptr<at::Half>()),
        first_biases.data_ptr<float>(),
        tail_weights.data_ptr<float>(),
        partials.data_ptr<float>(),
        rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
