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

#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <string>
#include <vector>

// Profiling-only readback of the image buffer produced by a forward pass.
// The returned tensors are independent CUDA clones: [tile_count, 2] uint32
// range endpoints represented as int32, followed by [H, W] n_contrib.
std::vector<torch::Tensor> TackerImageStateCUDA(
	const torch::Tensor& image_buffer,
	const int image_height,
	const int image_width);
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);

// Inference-only Raster+deformation-head mixed path.  The first seven return
// values have exactly the same order and meaning as RasterizeGaussiansCUDA;
// the eighth is the FP32 [N, 128] head output.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansWithHeadCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& head_input,
	const torch::Tensor& head_weight,
	const torch::Tensor& head_bias,
	const int64_t persistent_blocks);

// Mixed ABI v2.  Raster return values retain their legacy order; the eighth
// value is a task-ordered vector of distinct FP32 [N_i, 128] outputs.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
RasterizeGaussiansWithHeadsCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const std::vector<torch::Tensor>& head_inputs,
	const std::vector<torch::Tensor>& head_weights,
	const std::vector<torch::Tensor>& head_biases,
	const int64_t worker_groups,
	const int64_t persistent_blocks);

// C3 packed shared-input first-linear backend.  The eighth value is one
// contiguous FP32 [H, N, 128] tensor.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansWithPackedHeadsCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& head_input,
	const torch::Tensor& head_weights,
	const torch::Tensor& head_biases,
	const int64_t worker_groups,
	const int64_t persistent_blocks);

// C4 whole-head backend.  The plural form supports 1--5 tasks and returns
// task-ordered FP32 [N_i, O_i] outputs; the singular form is a strict
// convenience binding for the one-task physical configuration.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
RasterizeGaussiansWithWholeHeadsCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const std::vector<torch::Tensor>& head_inputs,
	const std::vector<torch::Tensor>& first_weights,
	const std::vector<torch::Tensor>& first_biases,
	const std::vector<torch::Tensor>& tail_weights,
	const std::vector<torch::Tensor>& tail_biases,
	const int64_t worker_groups,
	const int64_t persistent_blocks);

// ABI v5 full deformation MLP backend.  Returns five FP32 delta tensors in
// position/scale/rotation/opacity/SH order alongside the Raster outputs.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
RasterizeGaussiansWithDeformMlpCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& tail_weights,
	const torch::Tensor& tail_biases,
	const int64_t persistent_blocks);

// Opt-in split scheduling for ABI v5. The prepare call owns the ordinary
// Raster preprocess/scan/sort/range construction. The two final calls consume
// those buffers either with the same physical v5 mixed leaf or Raster alone.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor>
PrepareGaussiansForDeformMlpOverlapCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor>
PrepareGaussiansForDeformMlpOverlapHandoffCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const uint64_t handoff_stream_handle,
	const uint64_t deform_done_handle,
	const uint64_t duplicate_done_handle);

std::vector<torch::Tensor> RenderPreparedGaussiansWithDeformMlpOverlapCUDA(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	const int64_t num_rendered,
	const int image_height,
	const int image_width,
	const bool debug,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& tail_weights,
	const torch::Tensor& tail_biases,
	const int64_t persistent_blocks);

// Experimental ABI-v6 prepared leaf.  packed_tail_weights is the cached
// FP16 [112,128] layout produced by pack_deform_tail_weights_v3.
std::vector<torch::Tensor>
RenderPreparedGaussiansWithDeformMlpTensorTailOverlapCUDA(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	const int64_t num_rendered,
	const int image_height,
	const int image_width,
	const bool debug,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& packed_tail_weights,
	const torch::Tensor& tail_biases,
	const int64_t persistent_blocks);

// Experimental ABI-v7 prepared leaf.  It shares the ABI-v6 tensor contract
// while selecting the low-register kernel symbol.
std::vector<torch::Tensor>
RenderPreparedGaussiansWithDeformMlpTensorTailLowRegisterOverlapCUDA(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	const int64_t num_rendered,
	const int image_height,
	const int image_width,
	const bool debug,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& packed_tail_weights,
	const torch::Tensor& tail_biases,
	const int64_t persistent_blocks);

// Experimental ABI-v8 prepared leaf.  It preserves the ABI-v7 entry point
// while selecting the four-CTA launch-bounds kernel.
std::vector<torch::Tensor>
RenderPreparedGaussiansWithDeformMlpTensorTailFourCtaOverlapCUDA(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	const int64_t num_rendered,
	const int image_height,
	const int image_width,
	const bool debug,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& packed_tail_weights,
	const torch::Tensor& tail_biases,
	const int64_t persistent_blocks);

// Experimental ABI-v9 prepared leaf.  It preserves ABI-v8 and balances a
// second Deform tile across the Raster CTA grid when the Deform grid is larger.
std::vector<torch::Tensor>
RenderPreparedGaussiansWithDeformMlpTensorTailBalancedOverlapCUDA(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	const int64_t num_rendered,
	const int image_height,
	const int image_width,
	const bool debug,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& packed_tail_weights,
	const torch::Tensor& tail_biases,
	const int64_t persistent_blocks);

// Experimental ABI-v10 prepared leaf with exact Raster hot-loop changes.
std::vector<torch::Tensor>
RenderPreparedGaussiansWithDeformMlpTensorTailRasterOptimizedOverlapCUDA(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	const int64_t num_rendered,
	const int image_height,
	const int image_width,
	const bool debug,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& packed_tail_weights,
	const torch::Tensor& tail_biases,
	const int64_t persistent_blocks);

// Experimental ABI-v11 prepared leaf with approximate Raster math.
std::vector<torch::Tensor>
RenderPreparedGaussiansWithDeformMlpTensorTailRasterFastOverlapCUDA(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	const int64_t num_rendered,
	const int image_height,
	const int image_width,
	const bool debug,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& packed_tail_weights,
	const torch::Tensor& tail_biases,
	const int64_t persistent_blocks);

// Experimental ABI-v12 Raster-only prefix range.
void RenderPreparedGaussiansRasterFastRangeCUDA(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	const int64_t num_rendered,
	const int image_height,
	const int image_width,
	const bool debug,
	const int64_t raster_start,
	const int64_t raster_end);

// Experimental 512-thread spatial Raster + HexPlane sampling leaf.
void RenderPreparedGaussiansWithHexPlaneSamplingCUDA(
    const torch::Tensor& background,
    const torch::Tensor& colors,
    const torch::Tensor& out_color,
    const torch::Tensor& out_depth,
    const torch::Tensor& radii,
    const torch::Tensor& geom_buffer,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& image_buffer,
    int64_t num_rendered, int image_height, int image_width,
    int64_t raster_start, int64_t raster_end,
    const torch::Tensor& coords,
    const std::vector<torch::Tensor>& planes,
    const std::vector<std::vector<int64_t>>& axes,
    const std::vector<torch::Tensor>& outputs,
    int64_t hex_tasks_per_cta);

std::vector<int64_t> TackerRasterHexPlaneResourcesCUDA();
void RenderPreparedGaussiansWithHexPlaneBlocksCUDA(
    const torch::Tensor& background, const torch::Tensor& colors,
    const torch::Tensor& out_color, const torch::Tensor& out_depth,
    const torch::Tensor& radii, const torch::Tensor& geom_buffer,
    const torch::Tensor& binning_buffer, const torch::Tensor& image_buffer,
    int64_t num_rendered, int image_height, int image_width,
    int64_t raster_start, int64_t raster_end, const torch::Tensor& coords,
    const std::vector<torch::Tensor>& planes,
    const std::vector<std::vector<int64_t>>& axes,
    const std::vector<torch::Tensor>& outputs, int64_t hex_tasks_per_cta);
std::vector<int64_t> TackerRasterHexPlaneBlockResourcesCUDA();

// Experimental ABI-v12 remaining Raster range + complete Deform leaf.
std::vector<torch::Tensor>
RenderPreparedGaussiansWithDeformMlpTensorTailSplitRasterOverlapCUDA(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	const int64_t num_rendered,
	const int image_height,
	const int image_width,
	const bool debug,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& packed_tail_weights,
	const torch::Tensor& tail_biases,
	const int64_t persistent_blocks,
	const int64_t raster_start);

// Profiling-only ABI-v8 entry.  Returns the five Deform outputs followed by
// an int64 [physical_ctas,4] clock64 tensor ordered as Raster start/end and
// Deform start/end.  diagnostic_mode: 0=both, 1=Raster only, 2=Deform only.
std::vector<torch::Tensor>
RenderPreparedGaussiansWithDeformMlpTensorTailFourCtaDiagnosticCUDA(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	const int64_t num_rendered,
	const int image_height,
	const int image_width,
	const bool debug,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& packed_tail_weights,
	const torch::Tensor& tail_biases,
	const int64_t persistent_blocks,
	const int64_t diagnostic_mode);

// Profiling-only ABI-v10/v11 entry.  Returns five Deform outputs followed by
// an int64 [physical_ctas,4] clock64 tensor.  abi_version is 10 or 11;
// diagnostic_mode is 0=both, 1=Raster only, or 2=Deform only.
std::vector<torch::Tensor>
RenderPreparedGaussiansWithDeformMlpTensorTailRasterDiagnosticCUDA(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	const int64_t num_rendered,
	const int image_height,
	const int image_width,
	const bool debug,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& packed_tail_weights,
	const torch::Tensor& tail_biases,
	const int64_t persistent_blocks,
	const int64_t abi_version,
	const int64_t diagnostic_mode);

void RenderPreparedGaussiansOverlapCUDA(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	const int64_t num_rendered,
	const int image_height,
	const int image_width,
	const bool debug);

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansWithWholeHeadCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& head_input,
	const torch::Tensor& first_weight,
	const torch::Tensor& first_bias,
	const torch::Tensor& tail_weight,
	const torch::Tensor& tail_bias,
	const int64_t persistent_blocks);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);
