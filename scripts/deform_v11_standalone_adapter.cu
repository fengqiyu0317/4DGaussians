// Test adapter only: use the exact device-v7 implementation selected by v11.
// No Raster, HexPlane, residual update, or renderer is executed here.
#include <cuda_runtime.h>
#include <cstdint>
#include "deform_mlp_device.cuh"

using namespace tacker_4dgs;
constexpr int kDescriptorBytes = 128;
constexpr int kMixedSharedBytes = 25216;
static_assert(sizeof(DeformMlpTensorTailParamsV2) <= kDescriptorBytes, "descriptor");

template <int THREAD_BASE>
__device__ __forceinline__ void run_deform(
    const DeformMlpTensorTailParamsV2& p, unsigned char* storage) {
    const int tid = int(threadIdx.x) - THREAD_BASE;
    if (tid < 0 || tid >= kDeformThreads) return;
    auto& shared_p = *reinterpret_cast<DeformMlpTensorTailParamsV2*>(storage);
    auto& scratch = *reinterpret_cast<DeformMlpScratchV2*>(storage + kDescriptorBytes);
    for (int i = tid; i < int(sizeof(p) / sizeof(uint32_t)); i += kDeformThreads)
        reinterpret_cast<uint32_t*>(&shared_p)[i] =
            reinterpret_cast<const uint32_t*>(&p)[i];
    DeformMlpNamedSyncV1{2}.sync();
    deform_mlp_tensor_tail_gptb_device_v7(
        shared_p, scratch, int(blockIdx.x), int(gridDim.x), THREAD_BASE, 2);
}

__global__ __launch_bounds__(128, 4)
void v11_standalone_128(DeformMlpTensorTailParamsV2 p) {
    __shared__ __align__(32) unsigned char storage[kDescriptorBytes + sizeof(DeformMlpScratchV2)];
    run_deform<0>(p, storage);
}

// Resource-envelope control, not the production mixed kernel: the 256 Raster
// threads do no work, and each CTA handles one 16-row Deform tile. Candidate-
// aware Raster/Deform pairing is intentionally outside this standalone test.
__global__ __launch_bounds__(384, 4)
void v11_deform_only_384(DeformMlpTensorTailParamsV2 p) {
    __shared__ __align__(32) unsigned char storage[kMixedSharedBytes];
    run_deform<256>(p, storage);
}

extern "C" int deform_v11_launch(
    const float* features, const float* backbone_weight, const float* backbone_bias,
    const half* first_weights, const float* first_biases,
    const half* packed_tail_weights, const float* tail_biases,
    float* out0, float* out1, float* out2, float* out3, float* out4,
    int rows, int resource_control, uintptr_t stream_value) {
    DeformMlpTensorTailParamsV2 p{};
    p.features = features;
    p.backbone_weight = backbone_weight;
    p.backbone_bias = backbone_bias;
    p.first_weights = first_weights;
    p.first_biases = first_biases;
    p.packed_tail_weights = packed_tail_weights;
    p.tail_biases = tail_biases;
    p.outputs[0] = out0; p.outputs[1] = out1; p.outputs[2] = out2;
    p.outputs[3] = out3; p.outputs[4] = out4; p.rows = rows;
    const int blocks = (rows + kDeformRows - 1) / kDeformRows;
    const auto stream = reinterpret_cast<cudaStream_t>(stream_value);
    if (resource_control)
        v11_deform_only_384<<<blocks, 384, 0, stream>>>(p);
    else
        v11_standalone_128<<<blocks, 128, 0, stream>>>(p);
    return int(cudaPeekAtLastError());
}

// Returns registers, static shared bytes, local bytes, max threads, and
// predicted active CTAs per SM. Does not claim production mixed occupancy.
extern "C" int deform_v11_resources(int control, int* values) {
    cudaFuncAttributes attributes{};
    const void* fn = control ? reinterpret_cast<const void*>(v11_deform_only_384)
                            : reinterpret_cast<const void*>(v11_standalone_128);
    auto status = cudaFuncGetAttributes(&attributes, fn);
    if (status != cudaSuccess) return int(status);
    int active = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, fn, control ? 384 : 128, 0);
    values[0] = attributes.numRegs;
    values[1] = int(attributes.sharedSizeBytes);
    values[2] = int(attributes.localSizeBytes);
    values[3] = attributes.maxThreadsPerBlock;
    values[4] = active;
    return int(status);
}
