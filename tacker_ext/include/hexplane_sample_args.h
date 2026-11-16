#pragma once

#include <cstdint>

namespace tacker_4dgs {

constexpr int kMaxHexPlaneSamples = 12;

struct HexPlanePlaneArgs {
    const float* plane;
    float* output;  // [16,N] backing storage for the [N,16] view.
    int height, width, axis_x, axis_y;
};

struct HexPlaneSampleArgs {
    const float* coords;  // Contiguous normalized FP32 [N,4].
    int rows, plane_count, blocks_per_plane, tasks_per_cta;
    HexPlanePlaneArgs planes[kMaxHexPlaneSamples];
};

struct HexPlaneBlockTask {
    bool sampling;
    int index;  // Sampling task rank or Raster tile rank, not physical block ID.
};

// Spread both task domains across the grid, including very unequal domains.
// This is a work mapping only: CUDA does not guarantee block execution order.
#ifdef __CUDACC__
__host__ __device__
#endif
inline HexPlaneBlockTask hexplane_block_task(int block, int samples, int raster) {
    const int64_t total = int64_t(samples) + raster;  // Caller launches total > 0.
    const int before = int(int64_t(block) * samples / total);
    const int after = int((int64_t(block) + 1) * samples / total);
    return {after != before, after != before ? before : block - before};
}

}  // namespace tacker_4dgs
