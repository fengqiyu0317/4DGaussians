#include <algorithm>
#include <climits>
#include <cstdint>

#include "tacker_raster_hexplane.h"
#include "tacker_forward.cuh"
#include "hexplane_sample_device.cuh"

using CudaRasterizer::Tacker::RasterHexPlaneRasterArgs;
using CudaRasterizer::Tacker::RasterHexPlaneSampleArgs;

// Exactly the requested Tacker mapping: the two halves share blockIdx.x but
// virtualize threadIdx.x. The Raster barriers use ID 1 with 256 participants;
// HexPlane needs no barriers. NEVER put __syncthreads() in either branch.
// grid_constant prevents dynamically indexing the plane descriptor array
// from creating a private copy in each thread's local memory (CUDA >= 11.7).
extern "C" __global__ __launch_bounds__(512, 3)
void tacker_mix_raster_hexplane_512_v1(
    const __grid_constant__ RasterHexPlaneRasterArgs raster,
    const __grid_constant__ RasterHexPlaneSampleArgs hexplane)
{
    if (threadIdx.x < 256) {
        CudaRasterizer::Tacker::general_ptb_render_raster_optimized<3, true>(
            raster.ranges, raster.point_list, raster.width, raster.height,
            raster.points_xy, raster.features, raster.depths,
            raster.conic_opacity, raster.final_T, raster.n_contrib,
            raster.background, raster.color, raster.depth,
            raster.grid_x, raster.grid_y, raster.start, int(gridDim.x),
            raster.end, 0, 1);
    } else {
        const int total = hexplane.plane_count * hexplane.blocks_per_plane;
        // Also guard before multiplying: Raster-only tail CTAs can have much
        // larger IDs than the HexPlane task domain when tasks_per_cta is large.
        const int hex_ctas = total == 0 ? 0 :
            1 + (total - 1) / hexplane.tasks_per_cta;
        if (int(blockIdx.x) >= hex_ctas) return;
        const int begin = int(blockIdx.x) * hexplane.tasks_per_cta;
        const int end = begin + min(hexplane.tasks_per_cta, total - begin);
        const int local_thread = int(threadIdx.x) - 256;
        #pragma unroll 1
        for (int task = begin; task < end; ++task) {
            const int plane_id = task / hexplane.blocks_per_plane;
            const int point_block = task % hexplane.blocks_per_plane;
            const auto& plane = hexplane.planes[plane_id];
            tacker_4dgs::hexplane_sample_plane_device(
                hexplane.coords, plane.plane, plane.output, hexplane.rows,
                plane.height, plane.width, plane.axis_x, plane.axis_y,
                point_block, local_thread);
        }
    }
}

namespace CudaRasterizer {
namespace Tacker {

namespace {
cudaError_t checkRasterHexPlaneArguments(
    const RasterHexPlaneRasterArgs& raster,
    const RasterHexPlaneSampleArgs& hexplane)
{
    // Keep the raw C++ entry point safe as well as the checked Torch bridge.
    // All products/round-up expressions below are int32 on the device.
    if (raster.width <= 0 || raster.height <= 0 ||
        int64_t(raster.width) * raster.height > INT_MAX / 3 ||
        raster.grid_x != (raster.width + BLOCK_X - 1) / BLOCK_X ||
        raster.grid_y != (raster.height + BLOCK_Y - 1) / BLOCK_Y ||
        raster.start < 0 || raster.end < raster.start ||
        raster.end > int64_t(raster.grid_x) * raster.grid_y ||
        hexplane.rows < 0 || hexplane.rows > INT_MAX / 16 ||
        hexplane.plane_count < 1 ||
        hexplane.plane_count > kMaxMixedHexPlanes ||
        hexplane.tasks_per_cta < 1 ||
        hexplane.blocks_per_plane != (hexplane.rows + 255) / 256)
        return cudaErrorInvalidValue;
    if (raster.start < raster.end &&
        (!raster.ranges || !raster.point_list || !raster.points_xy ||
         !raster.features || !raster.depths || !raster.conic_opacity ||
         !raster.background || !raster.final_T || !raster.n_contrib ||
         !raster.color || !raster.depth))
        return cudaErrorInvalidValue;
    if (hexplane.rows > 0 && !hexplane.coords)
        return cudaErrorInvalidValue;
    for (int i = 0; i < hexplane.plane_count; ++i) {
        const auto& plane = hexplane.planes[i];
        if (plane.height <= 0 || plane.width <= 0 ||
            plane.height > (1 << 24) || plane.width > (1 << 24) ||
            int64_t(plane.height) * plane.width > INT_MAX / 16 ||
            plane.axis_x < 0 || plane.axis_x >= 4 ||
            plane.axis_y < 0 || plane.axis_y >= 4 ||
            plane.axis_x == plane.axis_y ||
            (hexplane.rows > 0 && (!plane.plane || !plane.output)))
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}
}  // namespace

cudaError_t launchRasterHexPlaneMixed(
    const RasterHexPlaneRasterArgs& raster,
    const RasterHexPlaneSampleArgs& hexplane, cudaStream_t stream)
{
    const auto error = checkRasterHexPlaneArguments(raster, hexplane);
    if (error != cudaSuccess) return error;
    const int total = hexplane.plane_count * hexplane.blocks_per_plane;
    const int hex_ctas = total == 0 ? 0 :
        1 + (total - 1) / hexplane.tasks_per_cta;
    const int grid = std::max(raster.end - raster.start, hex_ctas);
    if (grid == 0) return cudaSuccess;
    tacker_mix_raster_hexplane_512_v1<<<grid, 512, 0, stream>>>(raster, hexplane);
    return cudaGetLastError();
}

cudaError_t queryRasterHexPlaneMixedResources(
    cudaFuncAttributes* attributes, int* active_blocks_per_sm)
{
    auto error = cudaFuncGetAttributes(attributes, tacker_mix_raster_hexplane_512_v1);
    if (error != cudaSuccess) return error;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        active_blocks_per_sm, tacker_mix_raster_hexplane_512_v1, 512, 0);
}

extern "C" __global__ __launch_bounds__(256)
void tacker_mix_raster_hexplane_blocks_v1(
    const __grid_constant__ RasterHexPlaneRasterArgs raster,
    const __grid_constant__ RasterHexPlaneSampleArgs hexplane)
{
    const int samples = hexplane.plane_count * hexplane.blocks_per_plane;
    const int tiles = raster.end - raster.start;
    const auto task = tacker_4dgs::hexplane_block_task(int(blockIdx.x), samples, tiles);
    // Uniform for every thread in this CTA. There are no cross-CTA waits.
    if (task.sampling) {
        tacker_4dgs::hexplane_sample_task_device(hexplane, task.index, int(threadIdx.x));
    } else {
        general_ptb_render_raster_optimized<3, true>(
            raster.ranges, raster.point_list, raster.width, raster.height,
            raster.points_xy, raster.features, raster.depths,
            raster.conic_opacity, raster.final_T, raster.n_contrib,
            raster.background, raster.color, raster.depth,
            raster.grid_x, raster.grid_y, raster.start, tiles,
            raster.end, 0, 1, task.index);
    }
}

cudaError_t launchRasterHexPlaneBlocks(
    const RasterHexPlaneRasterArgs& raster,
    const RasterHexPlaneSampleArgs& hexplane, cudaStream_t stream)
{
    const auto error = checkRasterHexPlaneArguments(raster, hexplane);
    if (error != cudaSuccess) return error;
    if (hexplane.tasks_per_cta != 1) return cudaErrorInvalidValue;
    const int64_t grid = int64_t(hexplane.plane_count) * hexplane.blocks_per_plane +
                         raster.end - raster.start;
    if (grid > INT_MAX) return cudaErrorInvalidValue;
    if (grid == 0) return cudaSuccess;
    tacker_mix_raster_hexplane_blocks_v1<<<int(grid), 256, 0, stream>>>(raster, hexplane);
    return cudaGetLastError();
}

cudaError_t queryRasterHexPlaneBlocksResources(
    cudaFuncAttributes* attributes, int* active_blocks_per_sm)
{
    auto error = cudaFuncGetAttributes(attributes, tacker_mix_raster_hexplane_blocks_v1);
    if (error != cudaSuccess) return error;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        active_blocks_per_sm, tacker_mix_raster_hexplane_blocks_v1, 256, 0);
}

}  // namespace Tacker
}  // namespace CudaRasterizer
