/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	// Non-owning, inference-only description of the deformation-head work that
	// may share the final raster CTA.  Keeping the operand pointers opaque here
	// prevents the core rasterizer interface from depending on CUDA half types.
	// The mixed launcher interprets input/weight as row-major FP16 tensors and
	// bias/output as FP32 tensors.
	struct MixedHeadTask
	{
		const void* input;
		const void* weight;
		const float* bias;
		float* output;
		int rows;
		int persistent_blocks;
	};

	// Source-level host contract for the mixed ABI v2.  The corresponding
	// tacker_ext device descriptor deliberately has the same field order, but
	// CUDA half types remain outside the core Rasterizer header.  Read-only
	// operands may alias across tasks (the five deformation heads normally
	// share one hidden activation); every output must be distinct.
	static const int kMaxMixedHeadTasksV2 = 5;
	struct MixedHeadTaskV2
	{
		const void* input;
		const void* weight;
		const float* bias;
		float* output;
		int rows;
	};

	struct MixedHeadBundleV2
	{
		MixedHeadTaskV2 tasks[kMaxMixedHeadTasksV2];
		int task_count;
		int worker_groups;
		int persistent_blocks;
	};

	// C3 uses tacker_ext's packed shared-input adapter.  All heads have the
	// same row count and their weights/biases/outputs are contiguous along the
	// leading head dimension.
	struct MixedPackedHeadBundleV2
	{
		const void* input;
		const void* weights;
		const float* biases;
		float* output;
		int rows;
		int head_count;
		int worker_groups;
		int persistent_blocks;
	};

	// C4 evaluates the complete Linear(128,128) -> ReLU -> Linear(128,O)
	// head.  Tail widths may differ by task, so C4 retains task descriptors
	// while sharing one physical Raster+backend CTA.
	struct MixedWholeHeadTaskV2
	{
		const void* input;
		const void* first_weight;
		const float* first_bias;
		const float* tail_weight;
		const float* tail_bias;
		float* output;
		int rows;
		int tail_features;
	};

	struct MixedWholeHeadBundleV2
	{
		MixedWholeHeadTaskV2 tasks[kMaxMixedHeadTasksV2];
		int task_count;
		int worker_groups;
		int persistent_blocks;
	};

	// Full deformation MLP backend.  The task consumes HexPlane features and
	// evaluates backbone + all five heads, returning task-ordered deltas.  CUDA
	// half stays opaque in this core header just like the existing head ABIs.
	static const int kDeformMlpOutputCountV1 = 5;
	struct MixedDeformMlpTaskV1
	{
		const float* features;
		const float* backbone_weight;
		const float* backbone_bias;
		const void* first_weights;
		const float* first_biases;
		const float* tail_weights;
		const float* tail_biases;
		float* outputs[kDeformMlpOutputCountV1];
		int rows;
		int persistent_blocks;
	};

	// Experimental ABI-v6 descriptor.  It preserves the v5 task and replaces
	// only the scalar FP32 tail matrix with the cached padded FP16 Tensor-Core
	// matrix used by tacker_deform_mlp_tensor_tail_padded_v4.
	struct MixedDeformMlpTensorTailTaskV2
	{
		const float* features;
		const float* backbone_weight;
		const float* backbone_bias;
		const void* first_weights;
		const float* first_biases;
		const void* packed_tail_weights;
		const float* tail_biases;
		float* outputs[kDeformMlpOutputCountV1];
		int rows;
		int persistent_blocks;
	};

	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present,
			cudaStream_t stream = nullptr);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_depth,
			int* radii = nullptr,
			bool debug = false,
			cudaStream_t stream = nullptr);

		// Inference-only overload.  All preprocessing, scan, sort, and range
		// construction are identical to forward(); only the final render leaf is
		// replaced by one physical Raster+head mixed kernel.
		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_depth,
			int* radii,
			bool debug,
			cudaStream_t stream,
			const MixedHeadTask* mixed_head);

		// Internal v2 overload used by the multi-head Python binding.  Exactly
		// one of mixed_head_v1 and mixed_heads_v2 may be non-null.
		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_depth,
			int* radii,
			bool debug,
			cudaStream_t stream,
			const MixedHeadTask* mixed_head_v1,
			const MixedHeadBundleV2* mixed_heads_v2);

		// Family-aware mixed overload.  At most one backend descriptor may be
		// non-null; the preceding overload remains source-compatible and routes
		// to this one with C3/C4 disabled.
		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_depth,
			int* radii,
			bool debug,
			cudaStream_t stream,
			const MixedHeadTask* mixed_head_v1,
			const MixedHeadBundleV2* mixed_heads_v2,
			const MixedPackedHeadBundleV2* mixed_packed_heads_v2,
			const MixedWholeHeadBundleV2* mixed_whole_heads_v2,
			const MixedDeformMlpTaskV1* mixed_deform_mlp_v1 = nullptr);

		// Opt-in ABI-v5 scheduling path. The legacy forward() overloads above
		// remain monolithic; these entry points split only at the boundary
		// immediately before the final render leaf.
		static int prepareForwardForDeformMlpOverlap(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			int* radii,
			bool debug,
			cudaStream_t stream,
			cudaStream_t handoff_stream = nullptr,
			cudaEvent_t deform_done = nullptr,
			cudaEvent_t duplicate_done = nullptr);

		static void renderPreparedDeformMlpOverlap(
			const int P,
			const int num_rendered,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			char* geometry_buffer,
			char* binning_buffer,
			char* image_buffer,
			float* out_color,
			float* out_depth,
			bool debug,
			cudaStream_t stream,
			const MixedDeformMlpTaskV1* mixed_deform_mlp_v1 = nullptr);

		// Opt-in experimental ABI-v6 leaf.  Kept separate from the v5 entry point
		// so production callers and the qualified v5 symbol remain unchanged.
		static void renderPreparedDeformMlpTensorTailOverlap(
			const int P,
			const int num_rendered,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			char* geometry_buffer,
			char* binning_buffer,
			char* image_buffer,
			float* out_color,
			float* out_depth,
			bool debug,
			cudaStream_t stream,
			const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp_v2);

		// Opt-in ABI-v7 leaf with the low-register Tensor-tail implementation.
		static void renderPreparedDeformMlpTensorTailLowRegisterOverlap(
			const int P,
			const int num_rendered,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			char* geometry_buffer,
			char* binning_buffer,
			char* image_buffer,
			float* out_color,
			float* out_depth,
			bool debug,
			cudaStream_t stream,
			const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp_v2);

		// Opt-in ABI-v8 leaf targeting four resident 384-thread CTAs per SM.
		static void renderPreparedDeformMlpTensorTailFourCtaOverlap(
			const int P,
			const int num_rendered,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			char* geometry_buffer,
			char* binning_buffer,
			char* image_buffer,
			float* out_color,
			float* out_depth,
			bool debug,
			cudaStream_t stream,
			const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp_v2);

		// Opt-in ABI-v9 leaf that balances extra Deform tiles across Raster CTAs.
		static void renderPreparedDeformMlpTensorTailBalancedOverlap(
			const int P,
			const int num_rendered,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			char* geometry_buffer,
			char* binning_buffer,
			char* image_buffer,
			float* out_color,
			float* out_depth,
			bool debug,
			cudaStream_t stream,
			const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp_v2);

		// Opt-in ABI-v10 leaf with the exact Raster hot-loop optimizations.
		static void renderPreparedDeformMlpTensorTailRasterOptimizedOverlap(
			const int P,
			const int num_rendered,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			char* geometry_buffer,
			char* binning_buffer,
			char* image_buffer,
			float* out_color,
			float* out_depth,
			bool debug,
			cudaStream_t stream,
			const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp_v2);

		// Opt-in ABI-v11 leaf with approximate exp/FMA Raster math.
		static void renderPreparedDeformMlpTensorTailRasterFastOverlap(
			const int P,
			const int num_rendered,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			char* geometry_buffer,
			char* binning_buffer,
			char* image_buffer,
			float* out_color,
			float* out_depth,
			bool debug,
			cudaStream_t stream,
			const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp_v2);

		// ABI-v12 first leaf: render a disjoint fast-Raster tile prefix.
		static void renderPreparedRasterFastRange(
			const int P,
			const int num_rendered,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			char* geometry_buffer,
			char* binning_buffer,
			char* image_buffer,
			float* out_color,
			float* out_depth,
			bool debug,
			cudaStream_t stream,
			int raster_start,
			int raster_end);

		// ABI-v12 second leaf: remaining fast Raster tiles plus full Deform.
		static void renderPreparedDeformMlpTensorTailSplitRasterOverlap(
			const int P,
			const int num_rendered,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			char* geometry_buffer,
			char* binning_buffer,
			char* image_buffer,
			float* out_color,
			float* out_depth,
			bool debug,
			cudaStream_t stream,
			int raster_start,
			const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp_v2);

		// Profiling-only ABI-v8 leaf with per-CTA subgroup clock64 records.
		static void renderPreparedDeformMlpTensorTailFourCtaDiagnostic(
			const int P,
			const int num_rendered,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			char* geometry_buffer,
			char* binning_buffer,
			char* image_buffer,
			float* out_color,
			float* out_depth,
			bool debug,
			cudaStream_t stream,
			const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp_v2,
			int diagnostic_mode,
			int64_t* timing_cycles,
			int timing_rows);

		// Profiling-only ABI-v10/v11 leaf using the production balanced grid.
		static void renderPreparedDeformMlpTensorTailRasterDiagnostic(
			const int P,
			const int num_rendered,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			char* geometry_buffer,
			char* binning_buffer,
			char* image_buffer,
			float* out_color,
			float* out_depth,
			bool debug,
			cudaStream_t stream,
			const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp_v2,
			int abi_version,
			int diagnostic_mode,
			int64_t* timing_cycles,
			int timing_rows);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_dpix_depth,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_ddepth,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool debug,
			cudaStream_t stream = nullptr);
	};
};

#endif
