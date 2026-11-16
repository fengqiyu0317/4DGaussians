#pragma once

#include <cuda_runtime.h>
#include "hexplane_sample_args.h"

namespace CudaRasterizer {
namespace Tacker {

// Independent experimental family, not a change to the qualified v11/v12 ABI.
constexpr int kRasterHexPlaneThreads = 512;
constexpr int kMaxMixedHexPlanes = 12;

struct RasterHexPlaneRasterArgs {
    const uint2* ranges;
    const uint32_t* point_list;
    const float2* points_xy;
    const float* features;
    const float* depths;
    const float4* conic_opacity;
    const float* background;
    float* final_T;
    uint32_t* n_contrib;
    float* color;
    float* depth;
    int width, height, grid_x, grid_y, start, end;
};

using RasterHexPlanePlaneArgs = tacker_4dgs::HexPlanePlaneArgs;
using RasterHexPlaneSampleArgs = tacker_4dgs::HexPlaneSampleArgs;

// grid = max(number of Raster tiles, ceil(Hex logical tasks/tasks_per_cta)).
// Each CTA renders AT MOST one tile, while its other half executes a sequence
// of single-plane, 256-point tasks. No global counters or device-side waits.
cudaError_t launchRasterHexPlaneMixed(
    const RasterHexPlaneRasterArgs& raster,
    const RasterHexPlaneSampleArgs& hexplane,
    cudaStream_t stream);

cudaError_t queryRasterHexPlaneMixedResources(
    cudaFuncAttributes* attributes, int* active_blocks_per_sm);

// C: grid = Hex tasks + Raster tiles; each 256-thread CTA owns one task.
cudaError_t launchRasterHexPlaneBlocks(
    const RasterHexPlaneRasterArgs& raster,
    const RasterHexPlaneSampleArgs& hexplane, cudaStream_t stream);
cudaError_t queryRasterHexPlaneBlocksResources(
    cudaFuncAttributes* attributes, int* active_blocks_per_sm);

}  // namespace Tacker
}  // namespace CudaRasterizer
