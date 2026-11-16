// SPDX-License-Identifier: Apache-2.0
// The raster device adapter is derived from gsplat 1.5.3
// gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu.
// Copyright 2025 the Regents of the University of California,
// Nerfstudio Team and contributors. See LICENSE.gsplat.
// Changes: single camera, FP32 RGB+D, named 256-thread barriers, grid-stride
// logical tiles, and physical fusion with the project's ABI-v5 Deform MLP.
#include <torch/extension.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <climits>
#include <cuda_runtime.h>
#include "deform_mlp_device.cuh"

using torch::Tensor;
using namespace tacker_4dgs;

#ifndef GSPLAT_MIXED_MIN_BLOCKS
#define GSPLAT_MIXED_MIN_BLOCKS 1
#endif

struct RasterParams {
    const float2* means;
    const float3* conics;
    const float* colors;
    const float* opacities;
    const int* offsets;
    const int* ids;
    const float* background;
    float* image;
    float* alpha;
    int* last;
    int width, height, tile_width, tile_height, intersections;
};
struct RasterScratch {
    int ids[256];
    float3 xy_opacity[256];
    float3 conics[256];
};
static_assert(sizeof(RasterScratch) == 7168, "raster scratch ABI");

__device__ __forceinline__ int raster_done_count(bool done) {
    int count;
    // CTA-wide __syncthreads_count would deadlock: the Deform warps do not
    // participate. Barrier 1 belongs exclusively to the 256 raster threads.
    asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 0; "
                 "bar.red.popc.u32 %0, 1, 256, p; }"
                 : "=r"(count) : "r"(int(done)) : "memory");
    return count;
}
__device__ __forceinline__ void raster_sync() {
    asm volatile("bar.sync 1, 256;" ::: "memory");
}

__device__ __forceinline__ void gsplat_raster_device(
    const RasterParams& p, RasterScratch& s) {
    const int tid = int(threadIdx.x);
    const int tiles = p.tile_width * p.tile_height;
    for (int tile = int(blockIdx.x); tile < tiles; tile += int(gridDim.x)) {
        const int i = (tile / p.tile_width) * 16 + tid / 16;
        const int j = (tile % p.tile_width) * 16 + tid % 16;
        const bool inside = i < p.height && j < p.width;
        bool done = !inside;
        const float px = float(j) + 0.5f, py = float(i) + 0.5f;
        const int begin = p.offsets[tile];
        const int end = tile == tiles - 1 ? p.intersections : p.offsets[tile + 1];
        const int batches = (end - begin + 255) / 256;
        float T = 1.f;
        float out[4] = {0.f};
        int last = 0;
        for (int b = 0; b < batches; ++b) {
            if (raster_done_count(done) >= 256) break;
            const int batch_begin = begin + b * 256;
            const int idx = batch_begin + tid;
            if (idx < end) {
                const int g = p.ids[idx];
                const float2 xy = p.means[g];
                s.ids[tid] = g;
                s.xy_opacity[tid] = make_float3(xy.x, xy.y, p.opacities[g]);
                s.conics[tid] = p.conics[g];
            }
            raster_sync();
            const int size = min(256, end - batch_begin);
            for (int t = 0; t < size && !done; ++t) {
                const float3 conic = s.conics[t];
                const float3 xy_opac = s.xy_opacity[t];
                const float dx = xy_opac.x - px, dy = xy_opac.y - py;
                const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy)
                                    + conic.y * dx * dy;
                const float alpha = min(0.999f, xy_opac.z * __expf(-sigma));
                if (sigma < 0.f || alpha < (1.f / 255.f)) continue;
                const float next_T = T * (1.f - alpha);
                if (next_T <= 1e-4f) { done = true; break; }
                const float vis = alpha * T;
                const float* color = p.colors + s.ids[t] * 4;
#pragma unroll
                for (int k = 0; k < 4; ++k) out[k] += color[k] * vis;
                last = batch_begin + t;
                T = next_T;
            }
        }
        if (inside) {
            const int pixel = i * p.width + j;
            p.alpha[pixel] = 1.f - T;
#pragma unroll
            for (int k = 0; k < 4; ++k)
                p.image[pixel * 4 + k] = out[k] + T * p.background[k];
            p.last[pixel] = last;
        }
        // A grid-stride next tile can overwrite scratch even when the last
        // batch finished without another done-count barrier.
        raster_sync();
    }
}

#if GSPLAT_MIXED_MIN_BLOCKS == 1
extern "C" __global__ __launch_bounds__(384)
#else
extern "C" __global__ __launch_bounds__(384, GSPLAT_MIXED_MIN_BLOCKS)
#endif
void gsplat_mix_render_deform_v1(RasterParams raster, DeformMlpParamsV1 deform,
                                int mode, unsigned long long* timing) {
    __shared__ RasterScratch raster_scratch;
    __shared__ DeformMlpScratchV1 deform_scratch;
    __shared__ DeformMlpParamsV1 shared_deform;
    const int tid = int(threadIdx.x);
    const int block = int(blockIdx.x);
    if (tid < 256) {
        const bool active = mode != 1 && block < raster.tile_width * raster.tile_height;
        if (timing && tid == 0) {
            timing[block * 4] = active ? clock64() : 0;
            timing[block * 4 + 1] = 0;
        }
        if (active) gsplat_raster_device(raster, raster_scratch);
        if (timing && tid == 0 && active) timing[block * 4 + 1] = clock64();
    } else {
        const bool active = mode != 0 && block < (deform.rows + 15) / 16;
        if (timing && tid == 256) {
            timing[block * 4 + 2] = active ? clock64() : 0;
            timing[block * 4 + 3] = 0;
        }
        if (active) {
            if (tid == 256) shared_deform = deform;
            DeformMlpNamedSyncV1{2}.sync();
            deform_mlp_gptb_device_v1(shared_deform, deform_scratch,
                                     block, int(gridDim.x), 256, 2);
        }
        if (timing && tid == 256 && active) timing[block * 4 + 3] = clock64();
    }
}

// Attribution controls: identical device arithmetic with separate resource
// envelopes. Modes 0/1 above instead retain the mixed CTA's full allocation.
extern "C" __global__ __launch_bounds__(256)
void gsplat_raster_standalone(RasterParams raster) {
    __shared__ RasterScratch scratch;
    gsplat_raster_device(raster, scratch);
}

extern "C" __global__ __launch_bounds__(128)
void gsplat_deform_standalone(DeformMlpParamsV1 deform) {
    __shared__ DeformMlpScratchV1 scratch;
    __shared__ DeformMlpParamsV1 shared_deform;
    if (threadIdx.x == 0) shared_deform = deform;
    __syncthreads();
    deform_mlp_gptb_device_v1(shared_deform, scratch, int(blockIdx.x),
                             int(gridDim.x), 0, 2);
}

static void check(const Tensor& t, at::ScalarType dtype, c10::Device device,
                  at::IntArrayRef shape, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device() == device, name, " has wrong device");
    TORCH_CHECK(t.scalar_type() == dtype && t.is_contiguous(), name, " has wrong dtype/layout");
    TORCH_CHECK(t.sizes() == shape, name, " has wrong shape");
}

std::vector<Tensor> gsplat_mixed_forward(
    const std::vector<Tensor>& r, const std::vector<Tensor>& d,
    int64_t width, int64_t height, int64_t blocks, int64_t mode, bool profile) {
    TORCH_CHECK(!at::GradMode::is_enabled(), "gsplat mixed is inference-only");
    TORCH_CHECK(r.size() == 7 && d.size() == 7, "expected seven raster and seven MLP tensors");
    TORCH_CHECK(width > 0 && height > 0 && width * height <= INT_MAX / 4,
                "unsupported image dimensions");
    TORCH_CHECK(mode >= 0 && mode <= 5 && blocks >= 0 && blocks <= INT_MAX,
                "invalid mode or physical block count");
    TORCH_CHECK(r[0].is_cuda() && r[0].dim() == 3 && r[0].size(0) == 1,
                "only dense single-camera raster inputs are supported");
    const auto device = r[0].device();
    c10::cuda::CUDAGuard guard(device);
    const int64_t n = r[0].size(1), rows = d[0].size(0);
    const int tw = (width + 15) / 16, th = (height + 15) / 16;
    TORCH_CHECK(n <= INT_MAX && rows <= INT_MAX - 15 && r[5].numel() <= INT_MAX,
                "tensor size exceeds int32 ABI");
    check(r[0], at::kFloat, device, {1,n,2}, "means2d");
    check(r[1], at::kFloat, device, {1,n,3}, "conics");
    check(r[2], at::kFloat, device, {1,n,4}, "colors");
    check(r[3], at::kFloat, device, {1,n}, "opacities");
    check(r[4], at::kInt, device, {1,th,tw}, "tile_offsets");
    check(r[5], at::kInt, device, {r[5].numel()}, "flatten_ids");
    check(r[6], at::kFloat, device, {1,4}, "background");
    check(d[0], at::kFloat, device, {rows,32}, "features");
    check(d[1], at::kFloat, device, {32,128}, "backbone_weight");
    check(d[2], at::kFloat, device, {128}, "backbone_bias");
    check(d[3], at::kHalf, device, {5,128,128}, "first_weights");
    check(d[4], at::kFloat, device, {5,128}, "first_biases");
    check(d[5], at::kFloat, device, {59,128}, "tail_weights");
    check(d[6], at::kFloat, device, {59}, "tail_biases");
    const int logical = std::max(tw * th, int((rows + 15) / 16));
    const int physical = blocks == 0 ? logical : int(blocks);
    const auto options = r[0].options();
    std::vector<Tensor> out = {torch::empty({1,height,width,4}, options),
                               torch::empty({1,height,width,1}, options),
                               torch::empty({1,height,width}, options.dtype(at::kInt))};
    for (int cols : {3,3,4,1,48}) out.push_back(torch::empty({rows,cols}, options));
    out.push_back(torch::empty({profile ? physical : 0, 4}, options.dtype(at::kLong)));
    RasterParams rp{reinterpret_cast<const float2*>(r[0].data_ptr<float>()),
                    reinterpret_cast<const float3*>(r[1].data_ptr<float>()),
                    r[2].data_ptr<float>(), r[3].data_ptr<float>(),
                    r[4].data_ptr<int>(), r[5].data_ptr<int>(), r[6].data_ptr<float>(),
                    out[0].data_ptr<float>(), out[1].data_ptr<float>(), out[2].data_ptr<int>(),
                    int(width), int(height), tw, th, int(r[5].numel())};
    DeformMlpParamsV1 dp{};
    dp.features=d[0].data_ptr<float>(); dp.backbone_weight=d[1].data_ptr<float>();
    dp.backbone_bias=d[2].data_ptr<float>(); dp.first_weights=reinterpret_cast<const half*>(d[3].data_ptr());
    dp.first_biases=d[4].data_ptr<float>(); dp.tail_weights=d[5].data_ptr<float>();
    dp.tail_biases=d[6].data_ptr<float>(); dp.rows=int(rows);
    for (int i=0; i<5; ++i) dp.outputs[i]=out[3+i].data_ptr<float>();
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (mode <= 2) {
        gsplat_mix_render_deform_v1<<<physical,384,0,stream>>>(
            rp, dp, int(mode), profile ? reinterpret_cast<unsigned long long*>(out[8].data_ptr<int64_t>()) : nullptr);
    } else {
        TORCH_CHECK(!profile, "subgroup profiling requires a mixed-envelope mode (0/1/2)");
        // 3: serial separate, 4: raster only, 5: MLP only. Default grids match
        // each leaf's own logical tile count instead of reserving empty work.
        if (mode != 5) gsplat_raster_standalone<<<blocks ? physical : tw*th,256,0,stream>>>(rp);
        if (mode != 4 && rows) gsplat_deform_standalone<<<blocks ? physical : (rows+15)/16,128,0,stream>>>(dp);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<int64_t> gsplat_mixed_resources() {
    cudaFuncAttributes a{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, gsplat_mix_render_deform_v1));
    int active=0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
        gsplat_mix_render_deform_v1,384,0));
    return {a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),
            a.maxThreadsPerBlock,active};
}

std::vector<std::vector<int64_t>> gsplat_standalone_resources() {
    std::vector<std::vector<int64_t>> result;
    for (auto item : {std::make_pair((const void*)gsplat_raster_standalone,256),
                      std::make_pair((const void*)gsplat_deform_standalone,128)}) {
        cudaFuncAttributes a{};
        C10_CUDA_CHECK(cudaFuncGetAttributes(&a, item.first));
        int active=0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,item.first,item.second,0));
        result.push_back({a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),
                          a.maxThreadsPerBlock,active});
    }
    return result;
}
