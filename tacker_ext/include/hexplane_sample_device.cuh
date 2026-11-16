#pragma once

#include <cuda_runtime.h>
#include "hexplane_sample_args.h"

namespace tacker_4dgs {

// Matches grid_sample(..., mode="bilinear", padding_mode="border",
// align_corners=True). NaN coordinates select the negative border, as in
// ATen's CUDA clip_coordinates. Keep FMA contraction disabled for the weights
// and the four-term interpolation. The result is FP32-close to ATen, whose
// compiled kernel may contract multiply-adds; validation uses a tolerance.
__device__ __forceinline__ float hexplane_border_coordinate(
    float coordinate, int size) {
    const float unnormalized = __fmul_rn(
        __fmul_rn(__fadd_rn(coordinate, 1.0f), 0.5f), float(size - 1));
    return fminf(float(size - 1), fmaxf(unnormalized, 0.0f));
}

// Logical IDs are explicit: this body can run in threads 256..511 of a
// 512-thread Tacker CTA with local_thread_id=threadIdx.x-256. No physical
// block dimensions, whole-CTA barriers, shared memory, or 64-bit indexing.
// coords is contiguous [N,4], plane is [1,16,H,W], output storage is [16,N]
// (returned to Python as a transposed [N,16] view, just like grid_sample).
__device__ __forceinline__ void hexplane_sample_plane_device(
    const float* coords, const float* plane, float* output,
    int rows, int height, int width, int axis_x, int axis_y,
    int logical_block_id, int local_thread_id) {
    const int point = logical_block_id * 256 + local_thread_id;
    if (point >= rows) return;
    const float x = hexplane_border_coordinate(coords[point * 4 + axis_x], width);
    const float y = hexplane_border_coordinate(coords[point * 4 + axis_y], height);
    const int x0 = int(x);
    const int y0 = int(y);
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float dx = __fsub_rn(x, float(x0));
    const float dy = __fsub_rn(y, float(y0));
    const float w00 = __fmul_rn(__fsub_rn(1.0f, dx), __fsub_rn(1.0f, dy));
    const float w10 = __fmul_rn(dx, __fsub_rn(1.0f, dy));
    const float w01 = __fmul_rn(__fsub_rn(1.0f, dx), dy);
    const float w11 = __fmul_rn(dx, dy);
    const int stride = height * width;
    const int i00 = y0 * width + x0;
    const int i10 = y0 * width + x1;
    const int i01 = y1 * width + x0;
    const int i11 = y1 * width + x1;

    // Serial channel traversal bounds live state; do not unroll 16 channels.
    #pragma unroll 1
    for (int channel = 0; channel < 16; ++channel) {
        const float* values = plane + channel * stride;
        float value = __fmul_rn(values[i00], w00);
        value = __fadd_rn(value, __fmul_rn(values[i10], w10));
        value = __fadd_rn(value, __fmul_rn(values[i01], w01));
        value = __fadd_rn(value, __fmul_rn(values[i11], w11));
        output[channel * rows + point] = value;
    }
}

// Shared by B (all planes in one launch) and C (whole-block Raster/HexPlane).
// Adjacent tasks cycle through planes for the same 256-point tile.
__device__ __forceinline__ void hexplane_sample_task_device(
    const HexPlaneSampleArgs& args, int task, int local_thread_id) {
    const int plane_id = task % args.plane_count;
    const int point_block = task / args.plane_count;
    const auto& plane = args.planes[plane_id];
    hexplane_sample_plane_device(
        args.coords, plane.plane, plane.output, args.rows,
        plane.height, plane.width, plane.axis_x, plane.axis_y,
        point_block, local_thread_id);
}

}  // namespace tacker_4dgs
