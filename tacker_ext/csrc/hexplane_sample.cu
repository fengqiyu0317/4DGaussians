#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>

#include "hexplane_sample.h"
#include "hexplane_sample_device.cuh"

extern "C" __global__ __launch_bounds__(256)
void hexplane_sample_planes_batched_v1(
    const __grid_constant__ tacker_4dgs::HexPlaneSampleArgs args) {
    tacker_4dgs::hexplane_sample_task_device(
        args, int(blockIdx.x), int(threadIdx.x));
}

// No register cap on the solo kernel: report the compiler's natural usage.
__global__ __launch_bounds__(256) void hexplane_sample_plane_solo_v1(
    const float* coords, const float* plane, float* output,
    int rows, int height, int width, int axis_x, int axis_y) {
    tacker_4dgs::hexplane_sample_plane_device(
        coords, plane, output, rows, height, width, axis_x, axis_y,
        int(blockIdx.x), int(threadIdx.x));
}

// Resource/mapping probe only; the first 256 threads contain no Raster body.
// This verifies the exact logical IDs intended for Tacker spatial fusion,
// but cannot predict the registers of the future Raster+HexPlane kernel.
__global__ __launch_bounds__(512, 3) void hexplane_sample_plane_512_probe_v1(
    const float* coords, const float* plane, float* output,
    int rows, int height, int width, int axis_x, int axis_y) {
    if (threadIdx.x >= 256) {
        tacker_4dgs::hexplane_sample_plane_device(
            coords, plane, output, rows, height, width, axis_x, axis_y,
            int(blockIdx.x), int(threadIdx.x) - 256);
    }
}

namespace {
void check_inputs(const torch::Tensor& coords, const torch::Tensor& plane,
                  int64_t axis_x, int64_t axis_y) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "HexPlane low-register sampling is inference-only; use no_grad()");
    TORCH_CHECK(coords.is_cuda() && coords.scalar_type() == at::kFloat &&
                coords.is_contiguous() && coords.dim() == 2 && coords.size(1) == 4,
                "coords must be contiguous CUDA FP32 [N,4]");
    TORCH_CHECK(plane.is_cuda() && plane.device() == coords.device() &&
                plane.scalar_type() == at::kFloat && plane.is_contiguous() &&
                plane.dim() == 4 && plane.size(0) == 1 && plane.size(1) == 16 &&
                plane.size(2) >= 1 && plane.size(3) >= 1,
                "plane must be contiguous CUDA FP32 [1,16,H,W] on coords device");
    // Bound every element index (not just the number of points).
    TORCH_CHECK(coords.size(0) <= INT_MAX / 16 && plane.numel() <= INT_MAX &&
                plane.size(2) <= (1 << 24) && plane.size(3) <= (1 << 24),
                "HexPlane sampler requires int32-safe element indexing");
    TORCH_CHECK(axis_x >= 0 && axis_x < 4 && axis_y >= 0 && axis_y < 4 &&
                axis_x != axis_y, "axes must be distinct and in [0,4)");
}
}  // namespace

torch::Tensor hexplane_sample_plane_out_cuda(
    const torch::Tensor& coords, const torch::Tensor& plane,
    const torch::Tensor& output, int64_t axis_x, int64_t axis_y, bool probe_512) {
    check_inputs(coords, plane, axis_x, axis_y);
    TORCH_CHECK(output.is_cuda() && output.device() == coords.device() &&
                output.scalar_type() == at::kFloat && output.dim() == 2 &&
                output.size(0) == coords.size(0) && output.size(1) == 16 &&
                (coords.size(0) == 0 ||
                 (output.stride(0) == 1 && output.stride(1) == coords.size(0))),
                "output must be FP32 [N,16] with strides (1,N); use empty(16,N).t()");
    TORCH_CHECK(!output.is_alias_of(coords) && !output.is_alias_of(plane),
                "output must not alias inputs");
    const c10::cuda::CUDAGuard guard(coords.device());
    const int rows = int(coords.size(0));
    if (rows == 0) return output;
    const int blocks = (rows + 255) / 256;
    const auto stream = at::cuda::getCurrentCUDAStream(coords.get_device());
    if (probe_512) {
        hexplane_sample_plane_512_probe_v1<<<blocks, 512, 0, stream>>>(
            coords.data_ptr<float>(), plane.data_ptr<float>(), output.data_ptr<float>(),
            rows, int(plane.size(2)), int(plane.size(3)), int(axis_x), int(axis_y));
    } else {
        hexplane_sample_plane_solo_v1<<<blocks, 256, 0, stream>>>(
            coords.data_ptr<float>(), plane.data_ptr<float>(), output.data_ptr<float>(),
            rows, int(plane.size(2)), int(plane.size(3)), int(axis_x), int(axis_y));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor hexplane_sample_plane_cuda(
    const torch::Tensor& coords, const torch::Tensor& plane,
    int64_t axis_x, int64_t axis_y, bool probe_512) {
    check_inputs(coords, plane, axis_x, axis_y);
    const auto output = torch::empty({16, coords.size(0)}, coords.options()).t();
    return hexplane_sample_plane_out_cuda(coords, plane, output, axis_x, axis_y, probe_512);
}

std::vector<std::vector<int64_t>> hexplane_sample_resources_cuda() {
    std::vector<std::vector<int64_t>> result;
    for (int threads : {256, 512}) {
        const void* function = threads == 256
            ? reinterpret_cast<const void*>(hexplane_sample_plane_solo_v1)
            : reinterpret_cast<const void*>(hexplane_sample_plane_512_probe_v1);
        cudaFuncAttributes attributes;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, function));
        int active_blocks = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, function, threads, 0));
        result.push_back({threads, attributes.numRegs, int64_t(attributes.localSizeBytes),
                          int64_t(attributes.sharedSizeBytes), active_blocks,
                          attributes.binaryVersion});
    }
    return result;
}

std::vector<torch::Tensor> hexplane_sample_planes_out_cuda(
    const torch::Tensor& coords, const std::vector<torch::Tensor>& planes,
    const std::vector<std::vector<int64_t>>& axes,
    const std::vector<torch::Tensor>& outputs) {
    TORCH_CHECK(!planes.empty() && planes.size() <= tacker_4dgs::kMaxHexPlaneSamples &&
                axes.size() == planes.size() && outputs.size() == planes.size(),
                "provide 1..12 planes, one axis pair and one output per plane");
    tacker_4dgs::HexPlaneSampleArgs args = {};
    for (size_t i = 0; i < planes.size(); ++i) {
        TORCH_CHECK(axes[i].size() == 2, "each plane requires an axis pair");
        check_inputs(coords, planes[i], axes[i][0], axes[i][1]);
        const auto& output = outputs[i];
        TORCH_CHECK(output.is_cuda() && output.device() == coords.device() &&
                    output.scalar_type() == at::kFloat && output.dim() == 2 &&
                    output.size(0) == coords.size(0) && output.size(1) == 16 &&
                    (coords.size(0) == 0 ||
                     (output.stride(0) == 1 && output.stride(1) == coords.size(0))),
                    "output must be FP32 [N,16] with strides (1,N)");
        TORCH_CHECK(!output.is_alias_of(coords), "outputs must not alias coords");
        for (const auto& plane : planes)
            TORCH_CHECK(!output.is_alias_of(plane), "outputs must not alias any plane");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(!output.is_alias_of(outputs[j]), "outputs must not alias each other");
        args.planes[i] = {planes[i].data_ptr<float>(), output.data_ptr<float>(),
                          int(planes[i].size(2)), int(planes[i].size(3)),
                          int(axes[i][0]), int(axes[i][1])};
    }
    const c10::cuda::CUDAGuard guard(coords.device());
    args.coords = coords.data_ptr<float>();
    args.rows = int(coords.size(0));
    args.plane_count = int(planes.size());
    args.blocks_per_plane = (args.rows + 255) / 256;
    args.tasks_per_cta = 1;
    const int blocks = args.plane_count * args.blocks_per_plane;
    if (blocks > 0) {
        const auto stream = at::cuda::getCurrentCUDAStream(coords.get_device());
        hexplane_sample_planes_batched_v1<<<blocks, 256, 0, stream>>>(args);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return outputs;
}

std::vector<torch::Tensor> hexplane_sample_planes_cuda(
    const torch::Tensor& coords, const std::vector<torch::Tensor>& planes,
    const std::vector<std::vector<int64_t>>& axes) {
    TORCH_CHECK(!planes.empty() && planes.size() <= tacker_4dgs::kMaxHexPlaneSamples &&
                axes.size() == planes.size(), "provide 1..12 planes and matching axes");
    for (size_t i = 0; i < planes.size(); ++i) {
        TORCH_CHECK(axes[i].size() == 2, "each plane requires an axis pair");
        check_inputs(coords, planes[i], axes[i][0], axes[i][1]);
    }
    const c10::cuda::CUDAGuard guard(coords.device());
    std::vector<torch::Tensor> outputs;
    for (size_t i = 0; i < planes.size(); ++i)
        outputs.push_back(torch::empty({16, coords.size(0)}, coords.options()).t());
    return hexplane_sample_planes_out_cuda(coords, planes, axes, outputs);
}

std::vector<int64_t> hexplane_sample_batch_resources_cuda() {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, hexplane_sample_planes_batched_v1));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, hexplane_sample_planes_batched_v1, 256, 0));
    return {256, attributes.numRegs, int64_t(attributes.localSizeBytes),
            int64_t(attributes.sharedSizeBytes), blocks, attributes.binaryVersion};
}
