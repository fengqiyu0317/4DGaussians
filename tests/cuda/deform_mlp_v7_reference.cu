// Dedicated Deform kernel: exact device-v7 math used by mixed v11/v12.
// No Raster code, idle warp group, or mixed resource envelope.
#include <cuda_runtime.h>
#include <cstdint>
#include "deform_mlp_device.cuh"

using namespace tacker_4dgs;

extern "C" __global__ __launch_bounds__(128)
void tacker_deform_mlp_v7_standalone_matched(DeformMlpTensorTailParamsV2 params) {
    __shared__ DeformMlpTensorTailParamsV2 shared_params;
    __shared__ DeformMlpScratchV2 scratch;
    if (threadIdx.x == 0) shared_params = params;
    DeformMlpNamedSyncV1{2}.sync();
    deform_mlp_tensor_tail_gptb_device_v7(
        shared_params, scratch, int(blockIdx.x), int(gridDim.x), 0, 2);
}

extern "C" int deform_matched_launch(
    const float* features, const float* backbone_weight, const float* backbone_bias,
    const half* first_weights, const float* first_biases,
    const half* packed_tail_weights, const float* tail_biases,
    float* out0, float* out1, float* out2, float* out3, float* out4,
    int rows, uintptr_t stream_value) {
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
    tacker_deform_mlp_v7_standalone_matched<<<
        (rows + kDeformRows - 1) / kDeformRows, 128, 0,
        reinterpret_cast<cudaStream_t>(stream_value)>>>(p);
    return int(cudaPeekAtLastError());
}
