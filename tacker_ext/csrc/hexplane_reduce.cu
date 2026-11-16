#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>

#include "hexplane_sample.h"

namespace {
struct ReduceArgs {
    const float* samples[12];
    float* output;
    int rows;
};

// Each CTA owns 32 points and all 32 output channels. Adjacent lanes read
// adjacent points; the padded shared tile transposes to coalesced row-major
// stores. No intermediate products or separate concatenation are written.
template<int Threads>
__global__ __launch_bounds__(Threads) void hexplane_reduce_features_tiled_v1(
    const __grid_constant__ ReduceArgs args) {
    __shared__ float tile[32][33];
    const int base = int(blockIdx.x) * 32;
    for (int index = int(threadIdx.x); index < 32 * 32; index += Threads) {
        const int point = base + (index & 31);
        const int channel = index >> 5;
        float product = 1.0f;
        if (point < args.rows) {
            const int offset = (channel & 15) * args.rows + point;
            const int begin = (channel >> 4) * 6;
            // Preserve the reference's left fold INCLUDING its first 1*x.
            // Explicit RN operations must not become a reassociated tree.
            #pragma unroll
            for (int plane = 0; plane < 6; ++plane)
                product = __fmul_rn(product, args.samples[begin + plane][offset]);
        }
        tile[channel][index & 31] = product;
    }
    // Tail points still participate: never return before the CTA barrier.
    __syncthreads();
    for (int index = int(threadIdx.x); index < 32 * 32; index += Threads) {
        const int point = base + (index >> 5);
        if (point < args.rows)
            args.output[point * 32 + (index & 31)] = tile[index & 31][index >> 5];
    }
}

void check_samples(const std::vector<torch::Tensor>& samples, int64_t threads) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "HexPlane fused reduction is inference-only; use no_grad()");
    TORCH_CHECK(threads == 128 || threads == 256, "threads must be 128 or 256");
    TORCH_CHECK(samples.size() == 12, "samples must contain exactly 12 planes");
    const auto& first = samples[0];
    TORCH_CHECK(first.is_cuda() && first.scalar_type() == at::kFloat &&
                first.dim() == 2 && first.size(1) == 16,
                "samples must be CUDA FP32 [N,16]");
    TORCH_CHECK(first.size(0) <= INT_MAX / 32,
                "samples exceed int32-safe [N,32] output indexing");
    for (const auto& sample : samples) {
        TORCH_CHECK(sample.is_cuda() && sample.device() == first.device() &&
                    sample.scalar_type() == at::kFloat && sample.dim() == 2 &&
                    sample.size(0) == first.size(0) && sample.size(1) == 16 &&
                    (sample.size(0) == 0 ||
                     ((sample.size(0) == 1 || sample.stride(0) == 1) &&
                      sample.stride(1) == sample.size(0))),
                    "samples must be same-device CUDA FP32 [N,16] with strides (1,N)");
    }
}
}  // namespace

torch::Tensor hexplane_reduce_features_out_cuda(
    const std::vector<torch::Tensor>& samples, const torch::Tensor& output,
    int64_t threads) {
    check_samples(samples, threads);
    const auto& first = samples[0];
    TORCH_CHECK(output.is_cuda() && output.device() == first.device() &&
                output.scalar_type() == at::kFloat && output.dim() == 2 &&
                output.size(0) == first.size(0) && output.size(1) == 32 &&
                output.is_contiguous(), "output must be contiguous CUDA FP32 [N,32]");
    for (const auto& sample : samples)
        TORCH_CHECK(!output.is_alias_of(sample), "output must not alias samples");
    const c10::cuda::CUDAGuard guard(first.device());
    const int rows = int(first.size(0));
    if (rows == 0) return output;
    ReduceArgs args{};
    args.rows = rows;
    args.output = output.data_ptr<float>();
    for (int plane = 0; plane < 12; ++plane)
        args.samples[plane] = samples[plane].data_ptr<float>();
    const auto stream = at::cuda::getCurrentCUDAStream(first.get_device());
    const int blocks = (rows + 31) / 32;
    if (threads == 128)
        hexplane_reduce_features_tiled_v1<128><<<blocks, 128, 0, stream>>>(args);
    else
        hexplane_reduce_features_tiled_v1<256><<<blocks, 256, 0, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor hexplane_reduce_features_cuda(
    const std::vector<torch::Tensor>& samples, int64_t threads) {
    check_samples(samples, threads);
    auto output = torch::empty({samples[0].size(0), 32}, samples[0].options());
    return hexplane_reduce_features_out_cuda(samples, output, threads);
}

std::vector<std::vector<int64_t>> hexplane_reduce_resources_cuda() {
    std::vector<std::vector<int64_t>> result;
    for (int threads : {128, 256}) {
        const void* function = threads == 128
            ? reinterpret_cast<const void*>(hexplane_reduce_features_tiled_v1<128>)
            : reinterpret_cast<const void*>(hexplane_reduce_features_tiled_v1<256>);
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
