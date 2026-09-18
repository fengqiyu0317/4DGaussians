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

}  // namespace

extern "C" __global__ void __launch_bounds__(128)
tacker_deform_mlp_delta_v1(tacker_4dgs::DeformMlpParamsV1 params) {
    __shared__ tacker_4dgs::DeformMlpScratchV1 scratch;
    tacker_4dgs::deform_mlp_device_v1(params, scratch);
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
