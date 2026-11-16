// SPDX-License-Identifier: Apache-2.0
// The raster device adapter is derived from gsplat 1.5.3
// gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu.
// Copyright 2025 the Regents of the University of California,
// Nerfstudio Team and contributors. See LICENSE.gsplat.
// Candidate-aware v11 pairing port. Baseline arithmetic stays unchanged.
// Changes: dense FP32 RGB+D plus the unchanged device-v7 Tensor Core tail
// MLP used by the legacy v11/v13 kernels. Separate MLP calls the same device
// function with a 128-thread CTA. Mixed can run two such Deform groups.
#include <torch/extension.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cuda_runtime.h>
#include "deform_mlp_device.cuh"

using torch::Tensor;
using namespace tacker_4dgs;

#ifndef GSPLAT_MIXED_MIN_BLOCKS
#define GSPLAT_MIXED_MIN_BLOCKS 4
#endif
#ifndef GSPLAT_MIXED_RASTER_WARPS
#define GSPLAT_MIXED_RASTER_WARPS 8
#endif
#ifndef GSPLAT_MIXED_DEFORM_GROUPS
#define GSPLAT_MIXED_DEFORM_GROUPS 1
#endif
#ifndef GSPLAT_MIXED_BACKBONE_UNROLL
#define GSPLAT_MIXED_BACKBONE_UNROLL kDeformInput
#endif
#ifndef GSPLAT_MIXED_REUSE_WEIGHTS
#define GSPLAT_MIXED_REUSE_WEIGHTS 0
#endif
#ifndef GSPLAT_MIXED_PAIRED_LEAN
#define GSPLAT_MIXED_PAIRED_LEAN 0
#endif

constexpr int kRasterTileSize = 16;
constexpr int kRasterTilePixels = kRasterTileSize * kRasterTileSize;
constexpr int kMixedRasterThreads = GSPLAT_MIXED_RASTER_WARPS * 32;
constexpr int kMixedDeformGroups = GSPLAT_MIXED_DEFORM_GROUPS;
constexpr int kMixedThreads = kMixedRasterThreads + kMixedDeformGroups * kDeformThreads;
static_assert(kMixedDeformGroups == 1 || kMixedDeformGroups == 2);

#ifdef GSPLAT_MIXED_WIDE_RASTER
#if GSPLAT_MIXED_DEFORM_GROUPS == 2
#if GSPLAT_MIXED_RASTER_WARPS == 4
#define GSPLAT_CA_MIXED_KERNEL gsplat_mix_render_deform_tensor_v7_ca_wide_r4_d8
#elif GSPLAT_MIXED_RASTER_WARPS == 8
#if GSPLAT_MIXED_REUSE_WEIGHTS
#define GSPLAT_CA_MIXED_KERNEL gsplat_mix_render_deform_tensor_v7_ca_wide_r8_d8_reuse
#elif GSPLAT_MIXED_MIN_BLOCKS == 3
#define GSPLAT_CA_MIXED_KERNEL gsplat_mix_render_deform_tensor_v7_ca_wide_r8_d8_3cta
#else
#define GSPLAT_CA_MIXED_KERNEL gsplat_mix_render_deform_tensor_v7_ca_wide_r8_d8
#endif
#else
#error "Two Deform groups require 4 or 8 Raster warps"
#endif
#elif GSPLAT_MIXED_RASTER_WARPS == 4
#define GSPLAT_CA_MIXED_KERNEL gsplat_mix_render_deform_tensor_v7_ca_wide_r4
#elif GSPLAT_MIXED_RASTER_WARPS == 6
#define GSPLAT_CA_MIXED_KERNEL gsplat_mix_render_deform_tensor_v7_ca_wide_r6
#elif GSPLAT_MIXED_RASTER_WARPS == 8
#define GSPLAT_CA_MIXED_KERNEL gsplat_mix_render_deform_tensor_v7_ca_wide
#else
#error "Wide Raster requires 4, 6, or 8 Raster warps"
#endif
#define GSPLAT_CA_RASTER_KERNEL gsplat_raster_standalone_wide
#else
static_assert(kMixedRasterThreads == kRasterTilePixels && kMixedDeformGroups == 1);
#define GSPLAT_CA_MIXED_KERNEL gsplat_mix_render_deform_tensor_v7_ca
#define GSPLAT_CA_RASTER_KERNEL gsplat_raster_standalone
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
#ifdef GSPLAT_MIXED_WIDE_RASTER
    float2 xy[256];
    float4 conic_opacity[256];
    float4 rgbd[256];
#else
    int ids[256];
    float3 xy_opacity[256];
    float3 conics[256];
#endif
};
#ifdef GSPLAT_MIXED_WIDE_RASTER
static_assert(sizeof(RasterScratch) == 10240, "wide raster scratch ABI");
#else
static_assert(sizeof(RasterScratch) == 7168, "raster scratch ABI");
#endif

template <int RasterThreads>
__device__ __forceinline__ int raster_done_count(bool done) {
    int count;
    // Barrier 1 covers the Raster subgroup.
    asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 0; "
                 "bar.red.popc.u32 %0, 1, %2, p; }"
                 : "=r"(count) : "r"(int(done)), "n"(RasterThreads) : "memory");
    return count;
}

template <int RasterThreads>
__device__ __forceinline__ void raster_sync() {
    asm volatile("bar.sync 1, %0;" :: "n"(RasterThreads) : "memory");
}

struct RasterPixel {
    int row, col;
    bool inside, done;
    float transmittance;
    float color[4];
    int last;
};

template <int RasterThreads, bool SingleTile = false>
__device__ __forceinline__ void gsplat_raster_device(
    const RasterParams& p, RasterScratch& s) {
    constexpr int pixels_per_thread = (kRasterTilePixels + RasterThreads - 1) / RasterThreads;
    const int tid = int(threadIdx.x);
    const int tiles = p.tile_width * p.tile_height;
#ifdef GSPLAT_MIXED_WIDE_RASTER
    const bool aligned_rgbd = (reinterpret_cast<uintptr_t>(p.colors) & 15) == 0;
#endif
    for (int tile = int(blockIdx.x); tile < tiles; tile += int(gridDim.x)) {
        RasterPixel pixels[pixels_per_thread];
#pragma unroll
        for (int slot = 0; slot < pixels_per_thread; ++slot) {
            const int index = tid + slot * RasterThreads;
            const int row = (tile / p.tile_width) * kRasterTileSize + index / kRasterTileSize;
            const int col = (tile % p.tile_width) * kRasterTileSize + index % kRasterTileSize;
            const bool inside = index < kRasterTilePixels && row < p.height && col < p.width;
            pixels[slot] = {row, col, inside, !inside, 1.f, {0.f}, 0};
        }
        const int begin = p.offsets[tile];
        const int end = tile == tiles - 1 ? p.intersections : p.offsets[tile + 1];
        for (int batch_begin = begin; batch_begin < end;) {
            bool done = true;
#pragma unroll
            for (int slot = 0; slot < pixels_per_thread; ++slot) done &= pixels[slot].done;
            if (raster_done_count<RasterThreads>(done) == RasterThreads) break;
            const int size = min(kRasterTilePixels, end - batch_begin);
#pragma unroll
            for (int slot = 0; slot < pixels_per_thread; ++slot) {
                const int index = tid + slot * RasterThreads;
                if (index >= size) continue;
                const int idx = batch_begin + index;
                const int g = p.ids[idx];
                const float2 xy = p.means[g];
#ifdef GSPLAT_MIXED_WIDE_RASTER
                const float3 conic = p.conics[g];
                s.xy[index] = xy;
                s.conic_opacity[index] = make_float4(conic.x, conic.y, conic.z, p.opacities[g]);
                const float* color = p.colors + g * 4;
                s.rgbd[index] = aligned_rgbd ? *reinterpret_cast<const float4*>(color) :
                    make_float4(color[0], color[1], color[2], color[3]);
#else
                s.ids[index] = g;
                s.xy_opacity[index] = make_float3(xy.x, xy.y, p.opacities[g]);
                s.conics[index] = p.conics[g];
#endif
            }
            raster_sync<RasterThreads>();
#pragma unroll
            for (int slot = 0; slot < pixels_per_thread; ++slot) {
                RasterPixel& pixel = pixels[slot];
                const float px = float(pixel.col) + 0.5f, py = float(pixel.row) + 0.5f;
                for (int t = 0; t < size && !pixel.done; ++t) {
#ifdef GSPLAT_MIXED_WIDE_RASTER
                    const float4 conic = s.conic_opacity[t];
                    const float2 xy = s.xy[t];
                    const float opacity = conic.w;
#else
                    const float3 conic = s.conics[t];
                    const float3 xy = s.xy_opacity[t];
                    const float opacity = xy.z;
#endif
                    const float dx = xy.x - px, dy = xy.y - py;
                    const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy)
                                        + conic.y * dx * dy;
                    const float alpha = min(0.999f, opacity * __expf(-sigma));
                    if (sigma < 0.f || alpha < (1.f / 255.f)) continue;
                    const float next_T = pixel.transmittance * (1.f - alpha);
                    if (next_T <= 1e-4f) { pixel.done = true; break; }
                    const float vis = alpha * pixel.transmittance;
#ifdef GSPLAT_MIXED_WIDE_RASTER
                    const float4 rgbd = s.rgbd[t];
                    const float* color = reinterpret_cast<const float*>(&rgbd);
#else
                    const float* color = p.colors + s.ids[t] * 4;
#endif
#pragma unroll
                    for (int k = 0; k < 4; ++k) pixel.color[k] += color[k] * vis;
                    pixel.last = batch_begin + t;
                    pixel.transmittance = next_T;
                }
            }
            batch_begin += size;
        }
#pragma unroll
        for (int slot = 0; slot < pixels_per_thread; ++slot) {
            const RasterPixel& pixel = pixels[slot];
            if (!pixel.inside) continue;
            const int index = pixel.row * p.width + pixel.col;
            p.alpha[index] = 1.f - pixel.transmittance;
#pragma unroll
            for (int k = 0; k < 4; ++k)
                p.image[index * 4 + k] = pixel.color[k] + pixel.transmittance * p.background[k];
            p.last[index] = pixel.last;
        }
        if constexpr (SingleTile) break;
        // A grid-stride next tile can overwrite scratch even when the last
        // batch finished without another done-count barrier.
        raster_sync<RasterThreads>();
    }
}

// Same group boundaries, strict-greater tie rule and extra-tile ranks as
// tackerCandidateAwareExtraDeformRank. gsplat offsets encode the ranges.
__device__ __forceinline__ int tile_candidates(const RasterParams& p,int tile) {
    const int tiles=p.tile_width*p.tile_height;
    return (tile==tiles-1 ? p.intersections : p.offsets[tile+1])-p.offsets[tile];
}

template <typename Product = long long>
__device__ __forceinline__ int candidate_extra_rank(const RasterParams& p,int extra,int block) {
    const int tiles=p.tile_width*p.tile_height;
    if (extra<=0 || extra>=tiles || block<0 || block>=tiles) return -1;
    const Product group=((static_cast<Product>(block)+1)*extra-1)/tiles;
    const int begin=int(group*tiles/extra),end=int((group+1)*tiles/extra);
    int best=begin,best_count=tile_candidates(p,begin);
    for (int t=begin+1;t<end;++t) {
        const int count=tile_candidates(p,t);
        if (count>best_count) { best=t; best_count=count; }
    }
    return best==block ? int(group) : -1;
}

#if GSPLAT_MIXED_PAIRED_LEAN
static_assert(kMixedRasterThreads == 256 && kMixedDeformGroups == 2);
extern "C" __global__ __launch_bounds__(kMixedThreads)
void gsplat_mix_render_deform_tensor_v7_ca_wide_r8_d8_lean(
    const __grid_constant__ RasterParams raster,
    const __grid_constant__ DeformMlpTensorTailParamsV2 deform) {
    __shared__ RasterScratch raster_scratch;
    __shared__ DeformMlpScratchV2 deform_scratch[2];
    __shared__ int extra_rank;
    const int tid = int(threadIdx.x);
    if (tid < kMixedRasterThreads) {
        gsplat_raster_device<kMixedRasterThreads, true>(raster, raster_scratch);
        return;
    }
    const int block = int(blockIdx.x);
    const int raster_tiles = int(gridDim.x);
    if (tid == kMixedRasterThreads) {
        extra_rank = candidate_extra_rank<unsigned int>(
            raster, (deform.rows + 15) / 16 - raster_tiles, block);
    }
    asm volatile("bar.sync 4, 256;" ::: "memory");
    const int worker = (tid - kMixedRasterThreads) / kDeformThreads;
    if (worker != 0 && extra_rank < 0) return;
    const int tile = worker == 0 ? block : raster_tiles + extra_rank;
    deform_mlp_tensor_tail_device_impl_v7<kDeformInput, true, true>(
        deform, deform_scratch[worker], tid % kDeformThreads, tile, 0,
        DeformMlpNamedSyncV1{2 + worker});
}
#endif

#if GSPLAT_MIXED_MIN_BLOCKS == 1
extern "C" __global__ __launch_bounds__(kMixedThreads)
#else
extern "C" __global__ __launch_bounds__(kMixedThreads, GSPLAT_MIXED_MIN_BLOCKS)
#endif
void GSPLAT_CA_MIXED_KERNEL(RasterParams raster, DeformMlpTensorTailParamsV2 deform,
                                int mode, unsigned long long* timing, bool candidate_aware, int* assignments) {
    __shared__ RasterScratch raster_scratch;
    __shared__ DeformMlpScratchV2 deform_scratch[kMixedDeformGroups];
    __shared__ DeformMlpTensorTailParamsV2 shared_deform;
    __shared__ int shared_extra_rank;
    const int tid = int(threadIdx.x);
    const int block = int(blockIdx.x);
    if (tid < kMixedRasterThreads) {
        const bool active = mode != 1 && block < raster.tile_width * raster.tile_height;
        if (timing && tid == 0) {
            timing[block * 4] = active ? clock64() : 0;
            timing[block * 4 + 1] = 0;
        }
        if (active) gsplat_raster_device<kMixedRasterThreads>(raster, raster_scratch);
        if (timing && tid == 0 && active) timing[block * 4 + 1] = clock64();
    } else {
        const bool active = mode != 0 && block < (deform.rows + 15) / 16;
        if (timing && tid == kMixedRasterThreads) {
            timing[block * 4 + 2] = active ? clock64() : 0;
            timing[block * 4 + 3] = 0;
        }
        const int raster_tiles=raster.tile_width*raster.tile_height;
        const int deform_tiles=(deform.rows+15)/16;
        const bool paired=candidate_aware && int(gridDim.x)==raster_tiles &&
                          deform_tiles>raster_tiles && deform_tiles-raster_tiles<raster_tiles;
        if (assignments && tid == kMixedRasterThreads) {
            assignments[block*4]=0; assignments[block*4+1]=-1;
            assignments[block*4+2]=paired ? 1 : 0;
            assignments[block*4+3]=block<raster_tiles ? tile_candidates(raster,block) : 0;
        }
        if (active) {
            if (tid == kMixedRasterThreads) {
                shared_deform = deform;
                shared_extra_rank=paired ? candidate_extra_rank(raster,deform_tiles-raster_tiles,block) : -1;
            }
            if constexpr (kMixedDeformGroups == 2) {
                // Publish the descriptor and assignment to both Deform groups.
                asm volatile("bar.sync 4, 256;" ::: "memory");
            } else {
                DeformMlpNamedSyncV1{2}.sync();
            }
            const int worker = kMixedDeformGroups == 1 ? 0 :
                (tid - kMixedRasterThreads) / kDeformThreads;
            const int thread_base = kMixedRasterThreads + worker * kDeformThreads;
            const int barrier = 2 + worker;
            if constexpr (GSPLAT_MIXED_REUSE_WEIGHTS) {
                const int grid = int(gridDim.x);
                const int tile = paired
                    ? (worker == 0 ? block : shared_extra_rank >= 0
                        ? raster_tiles + shared_extra_rank : deform_tiles)
                    : block + worker * min(grid, deform_tiles);
                const int step = paired ? deform_tiles :
                    min(2LL * grid, static_cast<long long>(deform_tiles));
                if (tile < deform_tiles)
                    deform_mlp_tensor_tail_gptb_device_v7<kDeformInput, true>(
                        shared_deform, deform_scratch[worker], tile, step, thread_base, barrier);
            } else if (paired) {
                if (worker == 0 || shared_extra_rank >= 0) {
                    const int tile = worker == 0 ? block : raster_tiles + shared_extra_rank;
                    deform_mlp_tensor_tail_gptb_device_v7<GSPLAT_MIXED_BACKBONE_UNROLL>(
                        shared_deform, deform_scratch[worker],
                        tile, deform_tiles, thread_base, barrier);
                }
                if constexpr (kMixedDeformGroups == 1) {
                    if (shared_extra_rank >= 0)
                        deform_mlp_tensor_tail_gptb_device_v7(shared_deform, deform_scratch[0],
                            raster_tiles + shared_extra_rank, deform_tiles, thread_base, barrier);
                }
            } else {
                const int grid = int(gridDim.x);
                const int tile = block + worker * min(grid, deform_tiles);
                const int step = kMixedDeformGroups == 1 ? grid :
                    min(2LL * grid, static_cast<long long>(deform_tiles));
                if (tile < deform_tiles)
                    deform_mlp_tensor_tail_gptb_device_v7<GSPLAT_MIXED_BACKBONE_UNROLL>(
                        shared_deform, deform_scratch[worker],
                        tile, step, thread_base, barrier);
            }
            if constexpr (kMixedDeformGroups == 2) {
                // The recorded Deform interval ends when both workers finish.
                if (timing) asm volatile("bar.sync 4, 256;" ::: "memory");
            }
            if (assignments && tid == kMixedRasterThreads) {
                assignments[block*4]=paired ? 1+(shared_extra_rank>=0) :
                    1+(deform_tiles-1-block)/int(gridDim.x);
                assignments[block*4+1]=shared_extra_rank;
            }
        }
        if (timing && tid == kMixedRasterThreads && active) timing[block * 4 + 3] = clock64();
    }
}

// Attribution controls: identical device arithmetic with separate resource
// envelopes. Modes 0/1 above instead retain the mixed CTA's full allocation.
extern "C" __global__ __launch_bounds__(256)
void GSPLAT_CA_RASTER_KERNEL(RasterParams raster) {
    __shared__ RasterScratch scratch;
    gsplat_raster_device<kRasterTilePixels>(raster, scratch);
}

extern "C" __global__ __launch_bounds__(128)
void gsplat_deform_tensor_v7_standalone(DeformMlpTensorTailParamsV2 deform) {
    __shared__ DeformMlpScratchV2 scratch;
    __shared__ DeformMlpTensorTailParamsV2 shared_deform;
    if (threadIdx.x == 0) shared_deform = deform;
    DeformMlpNamedSyncV1{2}.sync();
    deform_mlp_tensor_tail_gptb_device_v7(shared_deform, scratch, int(blockIdx.x),
                             int(gridDim.x), 0, 2);
}

static void check(const Tensor& t, at::ScalarType dtype, c10::Device device,
                  at::IntArrayRef shape, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device() == device, name, " has wrong device");
    TORCH_CHECK(t.scalar_type() == dtype && t.is_contiguous(), name, " has wrong dtype/layout");
    TORCH_CHECK(t.sizes() == shape, name, " has wrong shape");
}

static RasterParams prepare_raster(const std::vector<Tensor>& r, int64_t width,
                                   int64_t height, std::vector<Tensor>& out) {
    TORCH_CHECK(width > 0 && height > 0 && width <= INT_MAX / 4 / height,
                "unsupported image dimensions");
    TORCH_CHECK(r[0].dim() == 3 && r[0].size(0) == 1,
                "only dense single-camera raster inputs are supported");
    const auto device = r[0].device();
    const int64_t n = r[0].size(1);
    const int tw = (width + 15) / 16, th = (height + 15) / 16;
    TORCH_CHECK(n <= INT_MAX && r[5].numel() <= INT_MAX, "raster size exceeds int32 ABI");
    check(r[0], at::kFloat, device, {1,n,2}, "means2d");
    check(r[1], at::kFloat, device, {1,n,3}, "conics");
    check(r[2], at::kFloat, device, {1,n,4}, "colors");
    check(r[3], at::kFloat, device, {1,n}, "opacities");
    check(r[4], at::kInt, device, {1,th,tw}, "tile_offsets");
    check(r[5], at::kInt, device, {r[5].numel()}, "flatten_ids");
    check(r[6], at::kFloat, device, {1,4}, "background");
    const auto options = r[0].options();
    out = {torch::empty({1,height,width,4}, options), torch::empty({1,height,width,1}, options),
           torch::empty({1,height,width}, options.dtype(at::kInt))};
    return {reinterpret_cast<const float2*>(r[0].data_ptr<float>()),
            reinterpret_cast<const float3*>(r[1].data_ptr<float>()),
            r[2].data_ptr<float>(), r[3].data_ptr<float>(),
            r[4].data_ptr<int>(), r[5].data_ptr<int>(), r[6].data_ptr<float>(),
            out[0].data_ptr<float>(), out[1].data_ptr<float>(), out[2].data_ptr<int>(),
            int(width), int(height), tw, th, int(r[5].numel())};
}

std::vector<Tensor> gsplat_raster_forward(const std::vector<Tensor>& r, int64_t width, int64_t height) {
    TORCH_CHECK(!at::GradMode::is_enabled(), "gsplat raster is inference-only");
    TORCH_CHECK(r.size() == 7 && r[0].is_cuda(), "expected seven CUDA raster tensors");
    c10::cuda::CUDAGuard guard(r[0].device());
    std::vector<Tensor> out;
    const auto raster = prepare_raster(r, width, height, out);
    GSPLAT_CA_RASTER_KERNEL<<<raster.tile_width * raster.tile_height,256,0,
        c10::cuda::getCurrentCUDAStream()>>>(raster);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<Tensor> gsplat_mixed_forward(
    const std::vector<Tensor>& r, const std::vector<Tensor>& d,
    int64_t width, int64_t height, int64_t blocks, int64_t mode, bool profile, bool candidate_aware) {
    TORCH_CHECK(!at::GradMode::is_enabled(), "gsplat mixed is inference-only");
    TORCH_CHECK(r.size() == 7 && d.size() == 7, "expected seven raster and seven MLP tensors");
    TORCH_CHECK(mode >= 0 && mode <= 5 && blocks >= 0 && blocks <= INT_MAX,
                "invalid mode or physical block count");
    TORCH_CHECK(r[0].is_cuda(), "expected CUDA raster inputs");
    const auto device = r[0].device();
    c10::cuda::CUDAGuard guard(device);
    const int64_t rows = d[0].size(0);
    TORCH_CHECK(rows <= INT_MAX - 15, "MLP size exceeds int32 ABI");
    check(d[0], at::kFloat, device, {rows,32}, "features");
    check(d[1], at::kFloat, device, {32,128}, "backbone_weight");
    check(d[2], at::kFloat, device, {128}, "backbone_bias");
    check(d[3], at::kHalf, device, {5,128,128}, "first_weights");
    check(d[4], at::kFloat, device, {5,128}, "first_biases");
    check(d[5], at::kHalf, device, {112,128}, "packed_tail_weights");
    check(d[6], at::kFloat, device, {59}, "tail_biases");
    std::vector<Tensor> out;
    const auto rp = prepare_raster(r, width, height, out);
    const int raster_tiles = rp.tile_width * rp.tile_height;
    const int physical = blocks == 0 ? raster_tiles : int(blocks);
    const auto options = r[0].options();
    for (int cols : {3,3,4,1,48}) out.push_back(torch::empty({rows,cols}, options));
    out.push_back(torch::empty({profile ? physical : 0, 4}, options.dtype(at::kLong)));
    out.push_back(torch::empty({profile ? physical : 0, 4}, options.dtype(at::kInt)));
    DeformMlpTensorTailParamsV2 dp{};
    dp.features=d[0].data_ptr<float>(); dp.backbone_weight=d[1].data_ptr<float>();
    dp.backbone_bias=d[2].data_ptr<float>(); dp.first_weights=reinterpret_cast<const half*>(d[3].data_ptr());
    dp.first_biases=d[4].data_ptr<float>(); dp.packed_tail_weights=reinterpret_cast<const half*>(d[5].data_ptr());
    dp.tail_biases=d[6].data_ptr<float>(); dp.rows=int(rows);
    for (int i=0; i<5; ++i) dp.outputs[i]=out[3+i].data_ptr<float>();
    auto stream = c10::cuda::getCurrentCUDAStream();
#if GSPLAT_MIXED_PAIRED_LEAN
    const int deform_tiles = (dp.rows + 15) / 16;
    // With these bounds every numerator in the paired rank fits in uint32.
    if (mode == 2 && !profile && candidate_aware && physical == raster_tiles &&
        raster_tiles <= 65536 && deform_tiles > raster_tiles && deform_tiles - raster_tiles < raster_tiles) {
        gsplat_mix_render_deform_tensor_v7_ca_wide_r8_d8_lean<<<physical,kMixedThreads,0,stream>>>(rp, dp);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
#endif
    if (mode <= 2) {
        GSPLAT_CA_MIXED_KERNEL<<<physical,kMixedThreads,0,stream>>>(
            rp, dp, int(mode), profile ? reinterpret_cast<unsigned long long*>(out[8].data_ptr<int64_t>()) : nullptr,
            candidate_aware,profile ? out[9].data_ptr<int>() : nullptr);
    } else {
        TORCH_CHECK(!profile, "subgroup profiling requires a mixed-envelope mode (0/1/2)");
        // 3: serial separate, 4: raster only, 5: MLP only. Default grids match
        // each leaf's own logical tile count instead of reserving empty work.
        if (mode != 5) GSPLAT_CA_RASTER_KERNEL<<<physical,256,0,stream>>>(rp);
        if (mode != 4 && rows) gsplat_deform_tensor_v7_standalone<<<blocks ? physical : (rows+15)/16,128,0,stream>>>(dp);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<int64_t> gsplat_mixed_resources() {
#if GSPLAT_MIXED_PAIRED_LEAN
    const void* kernel = (const void*)gsplat_mix_render_deform_tensor_v7_ca_wide_r8_d8_lean;
#else
    const void* kernel = (const void*)GSPLAT_CA_MIXED_KERNEL;
#endif
    cudaFuncAttributes a{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
    int active=0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
        kernel,kMixedThreads,0));
    return {a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),
            a.maxThreadsPerBlock,active};
}

std::vector<std::vector<int64_t>> gsplat_standalone_resources() {
    std::vector<std::vector<int64_t>> result;
    for (auto item : {std::make_pair((const void*)GSPLAT_CA_RASTER_KERNEL,256),
                      std::make_pair((const void*)gsplat_deform_tensor_v7_standalone,128)}) {
        cudaFuncAttributes a{};
        C10_CUDA_CHECK(cudaFuncGetAttributes(&a, item.first));
        int active=0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,item.first,item.second,0));
        result.push_back({a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),
                          a.maxThreadsPerBlock,active});
    }
    return result;
}

// Production independent leaf: allocate only the five MLP outputs, retaining
// exactly the same device-v7 arithmetic and named synchronization as mixed.
std::vector<Tensor> gsplat_tensor_mlp_forward(const std::vector<Tensor>& d, int64_t blocks) {
    TORCH_CHECK(!at::GradMode::is_enabled(), "gsplat tensor MLP is inference-only");
    TORCH_CHECK(d.size()==7 && d[0].is_cuda() && d[0].dim()==2, "expected seven CUDA MLP operands");
    TORCH_CHECK(blocks>=0 && blocks<=INT_MAX, "invalid MLP block count");
    const auto device=d[0].device();
    c10::cuda::CUDAGuard guard(device);
    const int64_t rows=d[0].size(0);
    TORCH_CHECK(rows<=INT_MAX-15, "row count exceeds int32 ABI");
    check(d[0],at::kFloat,device,{rows,32},"features");
    check(d[1],at::kFloat,device,{32,128},"backbone_weight");
    check(d[2],at::kFloat,device,{128},"backbone_bias");
    check(d[3],at::kHalf,device,{5,128,128},"first_weights");
    check(d[4],at::kFloat,device,{5,128},"first_biases");
    check(d[5],at::kHalf,device,{112,128},"packed_tail_weights");
    check(d[6],at::kFloat,device,{59},"tail_biases");
    std::vector<Tensor> out;
    for (int width : {3,3,4,1,48}) out.push_back(torch::empty({rows,width},d[0].options()));
    DeformMlpTensorTailParamsV2 p{};
    p.features=d[0].data_ptr<float>(); p.backbone_weight=d[1].data_ptr<float>();
    p.backbone_bias=d[2].data_ptr<float>(); p.first_weights=reinterpret_cast<const half*>(d[3].data_ptr());
    p.first_biases=d[4].data_ptr<float>(); p.packed_tail_weights=reinterpret_cast<const half*>(d[5].data_ptr());
    p.tail_biases=d[6].data_ptr<float>(); p.rows=int(rows);
    for (int i=0;i<5;++i) p.outputs[i]=out[i].data_ptr<float>();
    if (rows) gsplat_deform_tensor_v7_standalone<<<blocks ? int(blocks) : int((rows+15)/16),128,0,
        c10::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
