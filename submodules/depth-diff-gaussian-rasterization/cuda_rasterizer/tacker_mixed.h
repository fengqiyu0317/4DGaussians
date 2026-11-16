#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#include "rasterizer.h"

namespace CudaRasterizer
{
namespace Tacker
{

// Stable source-level ABI for the first physical Raster+deformation-head
// mixed leaf.  A physical CTA is split into two independent contiguous
// subgroups; neither subgroup is allowed to execute a CTA-wide barrier.
static const int kMixedAbiVersion = 1;
static const int kMixedThreads = 384;
static const int kRasterThreads = 256;
static const int kHeadThreads = 128;
static const int kHeadThreadBase = 256;
static const int kRasterBarrierId = 1;
static const int kDeformRowsPerTile = 16;

// Mixed ABI v2 keeps the exact Raster subgroup and adds 1--5 independent
// 128-thread worker groups.  The worker groups execute 1--5 first-linear head
// tasks using tacker_ext's stable multi-task device adapter.
static const int kMixedMultiAbiVersion = 2;
static const int kMixedPackedAbiVersion = 3;
static const int kMixedWholeHeadAbiVersion = 4;
static const int kMixedDeformMlpAbiVersion = 5;
static const int kMixedDeformMlpTensorTailAbiVersion = 6;
static const int kMixedDeformMlpTensorTailLowRegisterAbiVersion = 7;
static const int kMixedDeformMlpTensorTailFourCtaAbiVersion = 8;
static const int kMixedDeformMlpTensorTailBalancedAbiVersion = 9;
static const int kMixedDeformMlpTensorTailRasterOptimizedAbiVersion = 10;
static const int kMixedDeformMlpTensorTailRasterFastAbiVersion = 11;
static const int kMixedDeformMlpTensorTailSplitRasterAbiVersion = 12;
static const int kMaxHeadTasksV2 = 5;
static const int kMinWorkerGroupsV2 = 1;
static const int kMaxWorkerGroupsV2 = 5;
static const int kWorkerGroupThreadsV2 = 128;
static const int kHeadDescriptorBarrierIdV2 = 2;
static const int kWholeHeadBarrierBaseIdV2 = 2;
static const int kWholeHeadDescriptorBarrierIdV2 = 7;
static const int kDeformMlpBarrierIdV1 = 2;
static const int kMaxTailFeaturesV2 = 128;
static const int kMaxMixedThreadsV2 =
	kRasterThreads + kMaxWorkerGroupsV2 * kWorkerGroupThreadsV2;

enum class MixedBackendFamily
{
	FirstLinear,
	PackedFirstLinear,
	WholeHead,
	FullDeformMlp,
	FullDeformMlpTensorTail,
	FullDeformMlpTensorTailLowRegister,
	FullDeformMlpTensorTailFourCta,
	FullDeformMlpTensorTailBalanced,
	FullDeformMlpTensorTailRasterOptimized,
	FullDeformMlpTensorTailRasterFast
};

struct MixedKernelResources
{
	int abi_version;
	MixedBackendFamily backend_family;
	int worker_groups;
	int physical_threads;
	int device_ordinal;
	int compute_capability_major;
	int compute_capability_minor;
	int multiprocessor_count;
	int device_max_threads_per_block;
	int device_max_threads_per_multiprocessor;
	int warp_size;
	int kernel_max_threads_per_block;
	int registers_per_thread;
	std::size_t static_shared_bytes;
	std::size_t local_bytes_per_thread;
	std::size_t max_dynamic_shared_bytes;
	int active_blocks_per_multiprocessor;
	int active_warps_per_multiprocessor;
	int max_warps_per_multiprocessor;
	double occupancy;
	bool launch_supported;
};

// Launch a real 384-thread mixed kernel.  raster_grid may be empty, in which
// case the Raster subgroup performs no memory access and the head subgroup is
// still executed.  persistent_blocks==0 is resolved from the active device's
// multiProcessorCount; no GPU model-specific SM count is embedded here.
void launchMixedRenderHead(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedHeadTask& head,
	cudaStream_t stream);

// Launch mixed ABI v2.  task_count is in [1, 5] and worker_groups is in
// [1, task_count].  For group g, the tacker_ext adapter evaluates tasks
// g, g + worker_groups, ... exactly once.  Empty-row tasks are legal.
void launchMixedRenderHeads(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedHeadBundleV2& heads,
	cudaStream_t stream);

// C3 packed shared-input first-linear backend.  One output tensor stores
// [head_count, rows, 128] contiguously and every logical head executes once.
void launchMixedRenderPackedHeads(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedPackedHeadBundleV2& heads,
	cudaStream_t stream);

// C4 single/multi whole-head backend.  task_count==1 is the physical
// single-head configuration; larger bundles use 1--5 independent groups.
void launchMixedRenderWholeHeads(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedWholeHeadBundleV2& heads,
	cudaStream_t stream);

// ABI v5: direct, non-persistent-by-default fusion of the Raster render leaf
// with the complete deformation MLP.  persistent_blocks==0 launches one
// physical CTA per logical Raster/deform tile; positive values enable the
// grid-stride persistent form for later tuning.
void launchMixedRenderDeformMlp(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedDeformMlpTaskV1& deform,
	cudaStream_t stream);

// Experimental ABI v6: the v4 padded Tensor-Core tail inside the same
// 256-Raster + 128-Deform physical CTA.  ABI v5 remains available unchanged.
void launchMixedRenderDeformMlpTensorTail(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedDeformMlpTensorTailTaskV2& deform,
	cudaStream_t stream);

// Experimental ABI v7: preserve ABI-v6 while reducing Tensor-tail live state
// and requesting at least three resident 384-thread CTAs per SM.
void launchMixedRenderDeformMlpTensorTailLowRegister(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedDeformMlpTensorTailTaskV2& deform,
	cudaStream_t stream);

// Experimental ABI v8: two-value compaction, phase-local descriptor loads,
// and a four-resident-CTA launch-bounds target.  ABI-v7 remains unchanged.
void launchMixedRenderDeformMlpTensorTailFourCta(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedDeformMlpTensorTailTaskV2& deform,
	cudaStream_t stream);

// Experimental ABI v9: preserve the ABI-v8 warp split and math path, but
// launch one physical CTA per Raster tile by default.  When Deform has more
// tiles than Raster, the 128-thread Deform subgroup evaluates a uniformly
// distributed second tile without involving the Raster subgroup.
void launchMixedRenderDeformMlpTensorTailBalanced(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedDeformMlpTensorTailTaskV2& deform,
	cudaStream_t stream);

// Experimental ABI v10: ABI-v9 balanced scheduling plus an exact Raster
// adapter that removes hot-loop contributor increments, performs a
// conservative pre-exp alpha rejection, and caches RGB+depth per batch.
void launchMixedRenderDeformMlpTensorTailRasterOptimized(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedDeformMlpTensorTailTaskV2& deform,
	cudaStream_t stream);

// Experimental ABI v11: ABI-v10 plus __expf and explicit FMA composition.
// It is intentionally separate because its Raster output is approximate.
void launchMixedRenderDeformMlpTensorTailRasterFast(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedDeformMlpTensorTailTaskV2& deform,
	cudaStream_t stream);

// Experimental ABI v12 schedule, first leaf: render exactly the half-open
// logical Raster tile range [raster_start, raster_end) with the ABI-v11 fast
// math path.  This is a dedicated 256-thread Raster-only kernel so no idle
// Deform subgroup is carried by the prefix launch.
void launchRasterFastRange(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	int raster_start,
	int raster_end,
	cudaStream_t stream);

// Experimental ABI v12 schedule, second leaf: fuse the remaining Raster
// range [raster_start, raster_grid.x * raster_grid.y) with every Deform tile.
// The default physical grid contains one CTA per remaining Raster tile, so
// the Deform subgroup naturally evaluates two or three tiles per CTA.
void launchMixedRenderDeformMlpTensorTailSplitRaster(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	int raster_start,
	const MixedDeformMlpTensorTailTaskV2& deform,
	cudaStream_t stream);

// Profiling-only ABI-v8 companion.  diagnostic_mode selects 0=both,
// 1=Raster only, or 2=Deform only while retaining the same 384-thread CTA.
// timing_cycles is [timing_rows,4] int64 with columns
// [raster_start, raster_end, deform_start, deform_end].
void launchMixedRenderDeformMlpTensorTailFourCtaDiagnostic(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedDeformMlpTensorTailTaskV2& deform,
	int diagnostic_mode,
	int64_t* timing_cycles,
	int timing_rows,
	cudaStream_t stream);

// Profiling-only ABI-v10/v11 companion with the exact production balanced
// work mapping.  abi_version must be 10 or 11; diagnostic_mode selects
// 0=both, 1=Raster only, or 2=Deform only.  timing_cycles is
// [timing_rows,4] int64 with Raster start/end and Deform start/end columns.
void launchMixedRenderDeformMlpTensorTailRasterDiagnostic(
	const dim3 raster_grid,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* points_xy_image,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* background,
	float* out_color,
	float* out_depth,
	const MixedDeformMlpTensorTailTaskV2& deform,
	int abi_version,
	int diagnostic_mode,
	int64_t* timing_cycles,
	int timing_rows,
	cudaStream_t stream);

// Runtime resource/capability query used by offline candidate filtering.
// ABI v1/v5/v6/v7/v8 require worker_groups==1; ABI v2/v3/v4 accept [1, 5].  This call
// queries the active device and does not launch or synchronize a kernel.
MixedKernelResources queryMixedKernelResources(
	int abi_version,
	int worker_groups);

MixedKernelResources queryMixedKernelResources(
	int abi_version,
	int worker_groups,
	MixedBackendFamily family);

const char* mixedBackendFamilyName(MixedBackendFamily family);

}  // namespace Tacker
}  // namespace CudaRasterizer
