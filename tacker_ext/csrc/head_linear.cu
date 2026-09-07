#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>

#include "head_linear.h"
#include "head_linear_device.cuh"
#include "head_linear_kernels.cuh"

namespace {

using tacker_4dgs::kHeadFeatures;
using tacker_4dgs::kHeadThreads;

constexpr uintptr_t kWmmaAlignment = 32;

bool is_aligned(const torch::Tensor& tensor, uintptr_t alignment) {
    return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0;
}

uint64_t tensor_byte_count(const torch::Tensor& tensor) {
    const uint64_t elements = static_cast<uint64_t>(tensor.numel());
    const uint64_t element_size = static_cast<uint64_t>(tensor.element_size());
    TORCH_CHECK(
        element_size == 0 ||
            elements <= std::numeric_limits<uint64_t>::max() / element_size,
        "tensor byte size overflows uint64 range");
    return elements * element_size;
}

bool byte_ranges_overlap(
    const torch::Tensor& left,
    const torch::Tensor& right) {
    if (left.numel() == 0 || right.numel() == 0) {
        return false;
    }

    const uintptr_t left_begin =
        reinterpret_cast<uintptr_t>(left.data_ptr());
    const uintptr_t right_begin =
        reinterpret_cast<uintptr_t>(right.data_ptr());
    const uint64_t left_bytes = tensor_byte_count(left);
    const uint64_t right_bytes = tensor_byte_count(right);

    // Compare offsets instead of forming end pointers, which avoids unsigned
    // address wraparound for malformed/extreme tensor metadata.
    if (left_begin <= right_begin) {
        return static_cast<uint64_t>(right_begin - left_begin) < left_bytes;
    }
    return static_cast<uint64_t>(left_begin - right_begin) < right_bytes;
}

void check_arguments(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
    TORCH_CHECK(
        !at::GradMode::is_enabled(),
        "head_linear is inference-only; call it under torch.no_grad()");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be float16");
    TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be float16");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(input.dim() == 2, "input must have shape [N, 128]");
    TORCH_CHECK(
        input.size(1) == kHeadFeatures, "input must have shape [N, 128]");
    TORCH_CHECK(
        weight.dim() == 2 && weight.size(0) == kHeadFeatures &&
            weight.size(1) == kHeadFeatures,
        "weight must have shape [128, 128]");
    TORCH_CHECK(
        bias.dim() == 1 && bias.size(0) == kHeadFeatures,
        "bias must have shape [128]");
    TORCH_CHECK(
        input.device() == weight.device() && input.device() == bias.device(),
        "input, weight, and bias must be on the same CUDA device");
    TORCH_CHECK(
        input.size(0) <= static_cast<int64_t>(INT32_MAX),
        "input row count exceeds the int32 kernel ABI");
    TORCH_CHECK(
        is_aligned(input, kWmmaAlignment),
        "input data pointer must be natively 32-byte aligned for WMMA; "
        "misaligned contiguous views are not supported");
    TORCH_CHECK(
        is_aligned(weight, kWmmaAlignment),
        "weight data pointer must be natively 32-byte aligned for WMMA; "
        "misaligned contiguous views are not supported");
}

void check_output(const torch::Tensor& input, const torch::Tensor& output) {
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(output.scalar_type() == at::kFloat, "output must be float32");
    TORCH_CHECK(
        output.dim() == 2 && output.size(0) == input.size(0) &&
            output.size(1) == kHeadFeatures,
        "output must have shape [N, 128]");
    TORCH_CHECK(
        output.device() == input.device(),
        "output must be on the same CUDA device as input");
    TORCH_CHECK(
        is_aligned(output, kWmmaAlignment),
        "output data pointer must be natively 32-byte aligned for WMMA; "
        "misaligned contiguous views are not supported");
}

void check_no_output_alias(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& output) {
    TORCH_CHECK(
        !byte_ranges_overlap(input, output),
        "output storage must not overlap input storage");
    TORCH_CHECK(
        !byte_ranges_overlap(weight, output),
        "output storage must not overlap weight storage");
    TORCH_CHECK(
        !byte_ranges_overlap(bias, output),
        "output storage must not overlap bias storage");
}

}  // namespace

extern "C" __global__ void tacker_head_linear_solo_v1(
    const half* input,
    const half* weight,
    const float* bias,
    float* output,
    int rows,
    int logical_grid_x) {
    const int block_position = static_cast<int>(blockIdx.x);
    tacker_4dgs::head_linear_logical_block(
        input,
        weight,
        bias,
        output,
        rows,
        logical_grid_x,
        block_position,
        static_cast<int>(threadIdx.x));
}

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
    int ptb_end_block_pos) {
    tacker_4dgs::head_linear_gptb_device(
        input,
        weight,
        bias,
        output,
        rows,
        logical_grid_x,
        logical_grid_y,
        ptb_start_block_pos,
        ptb_iter_block_step,
        ptb_end_block_pos,
        0);
}

namespace {

torch::Tensor allocate_output(const torch::Tensor& input) {
    return torch::empty(
        {input.size(0), kHeadFeatures}, input.options().dtype(torch::kFloat32));
}

}  // namespace

torch::Tensor head_linear_solo_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
    check_arguments(input, weight, bias);
    auto output = allocate_output(input);
    return head_linear_solo_out_cuda(input, weight, bias, output);
}

torch::Tensor head_linear_solo_out_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& output) {
    check_arguments(input, weight, bias);
    check_output(input, output);
    check_no_output_alias(input, weight, bias, output);
    const c10::cuda::CUDAGuard device_guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    if (rows == 0) {
        return output;
    }

    const int logical_grid_x = static_cast<int>(
        (static_cast<int64_t>(rows) + tacker_4dgs::kRowsPerTile - 1) /
        tacker_4dgs::kRowsPerTile);
    const int logical_grid_y = 2;
    const int logical_blocks = logical_grid_x * logical_grid_y;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    tacker_head_linear_solo_v1<<<logical_blocks, kHeadThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        logical_grid_x);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor head_linear_gptb_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t persistent_blocks) {
    check_arguments(input, weight, bias);
    auto output = allocate_output(input);
    return head_linear_gptb_out_cuda(
        input, weight, bias, output, persistent_blocks);
}

torch::Tensor head_linear_gptb_out_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& output,
    int64_t persistent_blocks) {
    check_arguments(input, weight, bias);
    check_output(input, output);
    check_no_output_alias(input, weight, bias, output);
    TORCH_CHECK(persistent_blocks >= 0, "persistent_blocks must be >= 0");
    TORCH_CHECK(
        persistent_blocks <= INT32_MAX, "persistent_blocks exceeds int32 range");
    const c10::cuda::CUDAGuard device_guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    if (rows == 0) {
        return output;
    }

    const int logical_grid_x = static_cast<int>(
        (static_cast<int64_t>(rows) + tacker_4dgs::kRowsPerTile - 1) /
        tacker_4dgs::kRowsPerTile);
    const int logical_grid_y = 2;
    const int logical_blocks = logical_grid_x * logical_grid_y;
    int physical_blocks = static_cast<int>(persistent_blocks);
    if (physical_blocks == 0) {
        const auto* properties = at::cuda::getCurrentDeviceProperties();
        physical_blocks = properties->multiProcessorCount;
    }
    physical_blocks = std::max(1, std::min(physical_blocks, logical_blocks));

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    tacker_head_linear_gptb_v1<<<physical_blocks, kHeadThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        logical_grid_x,
        logical_grid_y,
        0,
        physical_blocks,
        logical_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
