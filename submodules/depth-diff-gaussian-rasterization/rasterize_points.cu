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

#include <math.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <utility>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "rasterize_points.h"
#include <memory>
#include <limits>
#include <vector>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "cuda_rasterizer/tacker_mixed.h"
#include "cuda_rasterizer/tacker_raster_hexplane.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::vector<torch::Tensor> TackerImageStateCUDA(
	const torch::Tensor& image_buffer,
	const int image_height,
	const int image_width)
{
	TORCH_CHECK(image_buffer.is_cuda(), "image_buffer must be a CUDA tensor");
	TORCH_CHECK(
		image_buffer.scalar_type() == at::kByte,
		"image_buffer must have dtype uint8");
	TORCH_CHECK(image_buffer.is_contiguous(), "image_buffer must be contiguous");
	TORCH_CHECK(image_height > 0, "image_height must be > 0");
	TORCH_CHECK(image_width > 0, "image_width must be > 0");
	const int64_t pixels = static_cast<int64_t>(image_height) * image_width;
	TORCH_CHECK(pixels <= INT_MAX, "image dimensions exceed int32 indexing");
	const size_t required_bytes =
		CudaRasterizer::required<CudaRasterizer::ImageState>(pixels);
	TORCH_CHECK(
		static_cast<size_t>(image_buffer.numel()) >= required_bytes,
		"image_buffer is smaller than the Raster ImageState layout");

	const c10::cuda::CUDAGuard device_guard(image_buffer.device());
	char* chunk = reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>());
	const CudaRasterizer::ImageState state =
		CudaRasterizer::ImageState::fromChunk(chunk, pixels);
	const int64_t grid_x = (image_width + BLOCK_X - 1) / BLOCK_X;
	const int64_t grid_y = (image_height + BLOCK_Y - 1) / BLOCK_Y;
	const int64_t tile_count = grid_x * grid_y;
	const auto int_options = image_buffer.options().dtype(torch::kInt32);
	auto range_view = torch::from_blob(
		reinterpret_cast<int32_t*>(state.ranges),
		{tile_count, 2},
		int_options);
	auto contributor_view = torch::from_blob(
		reinterpret_cast<int32_t*>(state.n_contrib),
		{image_height, image_width},
		int_options);
	return {range_view.clone(), contributor_view.clone()};
}

namespace
{

constexpr uintptr_t kWmmaAlignment = 32;

bool isPlaceholder(const torch::Tensor& tensor)
{
	return tensor.dim() == 1 && tensor.size(0) == 0;
}

bool isNativeWmmaAligned(const torch::Tensor& tensor)
{
	return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kWmmaAlignment == 0;
}

bool tensorByteRangesOverlap(
	const torch::Tensor& left,
	const torch::Tensor& right)
{
	if (left.numel() == 0 || right.numel() == 0)
		return false;
	const uintptr_t left_begin =
		reinterpret_cast<uintptr_t>(left.data_ptr());
	const uintptr_t right_begin =
		reinterpret_cast<uintptr_t>(right.data_ptr());
	const std::size_t left_bytes = static_cast<std::size_t>(left.numel()) *
		static_cast<std::size_t>(left.element_size());
	const std::size_t right_bytes = static_cast<std::size_t>(right.numel()) *
		static_cast<std::size_t>(right.element_size());
	TORCH_CHECK(
		left_begin <= std::numeric_limits<uintptr_t>::max() - left_bytes &&
			right_begin <= std::numeric_limits<uintptr_t>::max() - right_bytes,
		"tensor byte range overflows uintptr_t");
	const uintptr_t left_end = left_begin + left_bytes;
	const uintptr_t right_end = right_begin + right_bytes;
	return left_begin < right_end && right_begin < left_end;
}

void checkInt32RowCount(const char* name, const torch::Tensor& tensor)
{
	TORCH_CHECK(
		tensor.size(0) <= static_cast<int64_t>(INT_MAX),
		name,
		" row count exceeds the int32 Raster ABI");
}

void checkRasterDimensions(int image_height, int image_width)
{
	TORCH_CHECK(image_height > 0, "image_height must be > 0");
	TORCH_CHECK(image_width > 0, "image_width must be > 0");
	const int64_t pixels = static_cast<int64_t>(image_height) * image_width;
	TORCH_CHECK(
		pixels <= static_cast<int64_t>(INT_MAX) / NUM_CHANNELS,
		"image dimensions exceed the int32 Raster color-indexing ABI");
}

int legacyShCoefficientCount(const torch::Tensor& sh)
{
	if (sh.size(0) == 0)
		return 0;
	TORCH_CHECK(sh.dim() >= 2, "non-empty sh tensor must have a coefficient axis");
	TORCH_CHECK(
		sh.size(1) <= static_cast<int64_t>(INT_MAX),
		"sh coefficient count exceeds the int32 Raster ABI");
	return static_cast<int>(sh.size(1));
}

void checkCudaFloatOnRasterDevice(
	const char* name,
	const torch::Tensor& tensor,
	const torch::Tensor& means3D)
{
	TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
	TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
	TORCH_CHECK(
		tensor.device() == means3D.device(),
		name,
		" must be on the same CUDA device as means3D");
}

struct MixedRasterChoices
{
	bool use_sh;
	bool use_cov3d;
};

MixedRasterChoices checkMixedRasterArguments(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& sh,
	const torch::Tensor& campos,
	int degree)
{
	TORCH_CHECK(means3D.is_cuda(), "means3D must be a CUDA tensor");
	TORCH_CHECK(means3D.scalar_type() == at::kFloat, "means3D must be float32");
	TORCH_CHECK(
		means3D.dim() == 2 && means3D.size(1) == 3,
		"means3D must have shape [P, 3]");

	checkCudaFloatOnRasterDevice("background", background, means3D);
	checkCudaFloatOnRasterDevice("colors", colors, means3D);
	checkCudaFloatOnRasterDevice("opacity", opacity, means3D);
	checkCudaFloatOnRasterDevice("scales", scales, means3D);
	checkCudaFloatOnRasterDevice("rotations", rotations, means3D);
	checkCudaFloatOnRasterDevice("cov3D_precomp", cov3D_precomp, means3D);
	checkCudaFloatOnRasterDevice("viewmatrix", viewmatrix, means3D);
	checkCudaFloatOnRasterDevice("projmatrix", projmatrix, means3D);
	checkCudaFloatOnRasterDevice("sh", sh, means3D);
	checkCudaFloatOnRasterDevice("campos", campos, means3D);

	const int64_t rows = means3D.size(0);
	TORCH_CHECK(
		background.dim() == 1 && background.size(0) == NUM_CHANNELS,
		"background must have shape [3]");
	TORCH_CHECK(
		opacity.dim() == 2 && opacity.size(0) == rows && opacity.size(1) == 1,
		"opacity must have shape [P, 1]");
	TORCH_CHECK(
		viewmatrix.dim() == 2 && viewmatrix.size(0) == 4 &&
			viewmatrix.size(1) == 4,
		"viewmatrix must have shape [4, 4]");
	TORCH_CHECK(
		projmatrix.dim() == 2 && projmatrix.size(0) == 4 &&
			projmatrix.size(1) == 4,
		"projmatrix must have shape [4, 4]");
	TORCH_CHECK(
		campos.dim() == 1 && campos.size(0) == 3,
		"campos must have shape [3]");

	// Rank/shape, not numel(), identifies an omitted alternative. This is
	// important at P==0: [0, M, 3] is a real SH tensor while [0] is a
	// placeholder, even though both have zero elements.
	const bool has_sh = sh.dim() == 3 && sh.size(0) == rows &&
		sh.size(1) > 0 && sh.size(2) == 3;
	const bool has_colors = colors.dim() == 2 && colors.size(0) == rows &&
		colors.size(1) == NUM_CHANNELS;
	TORCH_CHECK(degree >= 0 && degree <= 3, "degree must be in [0, 3]");
	if (has_sh)
	{
		const int64_t required_coefficients =
			static_cast<int64_t>(degree + 1) * (degree + 1);
		TORCH_CHECK(
			sh.size(1) >= required_coefficients,
			"sh coefficient count is too small for degree");
		TORCH_CHECK(
			sh.size(1) <= static_cast<int64_t>(INT_MAX),
			"sh coefficient count exceeds the int32 Raster ABI");
	}
	TORCH_CHECK(
		(has_sh && isPlaceholder(colors)) ||
			(has_colors && isPlaceholder(sh)),
		"provide exactly one of sh [P, M, 3] or colors [P, 3]; "
		"the omitted tensor must be a [0] placeholder");

	const bool has_scales = scales.dim() == 2 && scales.size(0) == rows &&
		scales.size(1) == 3;
	const bool has_rotations = rotations.dim() == 2 &&
		rotations.size(0) == rows && rotations.size(1) == 4;
	const bool has_cov3d = cov3D_precomp.dim() == 2 &&
		cov3D_precomp.size(0) == rows && cov3D_precomp.size(1) == 6;
	TORCH_CHECK(
		(has_cov3d && isPlaceholder(scales) && isPlaceholder(rotations)) ||
			(has_scales && has_rotations && isPlaceholder(cov3D_precomp)),
		"provide exactly one of cov3D_precomp [P, 6] or the pair "
		"scales [P, 3] and rotations [P, 4]; omitted tensors must be "
		"[0] placeholders");

	return MixedRasterChoices{has_sh, has_cov3d};
}

void checkMixedHeadArguments(
	const torch::Tensor& means3D,
	const torch::Tensor& input,
	const torch::Tensor& weight,
	const torch::Tensor& bias,
	int64_t persistent_blocks)
{
	TORCH_CHECK(
		!at::GradMode::is_enabled(),
		"rasterize_gaussians_with_head is inference-only; call it under torch.no_grad()");
	TORCH_CHECK(means3D.is_cuda(), "means3D must be a CUDA tensor");
	TORCH_CHECK(input.is_cuda(), "head_input must be a CUDA tensor");
	TORCH_CHECK(weight.is_cuda(), "head_weight must be a CUDA tensor");
	TORCH_CHECK(bias.is_cuda(), "head_bias must be a CUDA tensor");
	TORCH_CHECK(input.is_contiguous(), "head_input must be contiguous");
	TORCH_CHECK(weight.is_contiguous(), "head_weight must be contiguous");
	TORCH_CHECK(bias.is_contiguous(), "head_bias must be contiguous");
	TORCH_CHECK(input.scalar_type() == at::kHalf, "head_input must be float16");
	TORCH_CHECK(weight.scalar_type() == at::kHalf, "head_weight must be float16");
	TORCH_CHECK(bias.scalar_type() == at::kFloat, "head_bias must be float32");
	TORCH_CHECK(
		input.dim() == 2 && input.size(1) == 128,
		"head_input must have shape [N, 128]");
	TORCH_CHECK(
		weight.dim() == 2 && weight.size(0) == 128 && weight.size(1) == 128,
		"head_weight must have shape [128, 128]");
	TORCH_CHECK(
		bias.dim() == 1 && bias.size(0) == 128,
		"head_bias must have shape [128]");
	TORCH_CHECK(
		means3D.device() == input.device() && input.device() == weight.device() &&
			input.device() == bias.device(),
		"raster and head tensors must be on the same CUDA device");
	TORCH_CHECK(
		means3D.size(0) <= static_cast<int64_t>(INT_MAX),
		"Gaussian row count exceeds the int32 raster ABI");
	TORCH_CHECK(
		input.size(0) <= static_cast<int64_t>(INT_MAX),
		"head_input row count exceeds the int32 mixed ABI");
	TORCH_CHECK(
		isNativeWmmaAligned(input),
		"head_input data pointer must be natively 32-byte aligned for WMMA; "
		"misaligned contiguous views are not supported");
	TORCH_CHECK(
		isNativeWmmaAligned(weight),
		"head_weight data pointer must be natively 32-byte aligned for WMMA; "
		"misaligned contiguous views are not supported");
	TORCH_CHECK(persistent_blocks >= 0, "persistent_blocks must be >= 0");
	TORCH_CHECK(
		persistent_blocks <= static_cast<int64_t>(INT_MAX),
		"persistent_blocks exceeds the int32 mixed ABI");
}

void checkMixedHeadsArguments(
	const torch::Tensor& means3D,
	const std::vector<torch::Tensor>& inputs,
	const std::vector<torch::Tensor>& weights,
	const std::vector<torch::Tensor>& biases,
	int64_t worker_groups,
	int64_t persistent_blocks)
{
	TORCH_CHECK(
		!at::GradMode::is_enabled(),
		"rasterize_gaussians_with_heads is inference-only; "
		"call it under torch.no_grad()");
	TORCH_CHECK(
		inputs.size() == weights.size() && inputs.size() == biases.size(),
		"head_inputs, head_weights, and head_biases must have equal lengths");
	TORCH_CHECK(
		!inputs.empty() &&
			inputs.size() <=
				static_cast<std::size_t>(CudaRasterizer::Tacker::kMaxHeadTasksV2),
		"mixed ABI v2 requires between 1 and 5 head tasks");
	TORCH_CHECK(
		worker_groups >= CudaRasterizer::Tacker::kMinWorkerGroupsV2 &&
			worker_groups <= static_cast<int64_t>(inputs.size()),
		"worker_groups must be in [1, task_count]");
	TORCH_CHECK(persistent_blocks >= 0, "persistent_blocks must be >= 0");
	TORCH_CHECK(
		persistent_blocks <= static_cast<int64_t>(INT_MAX),
		"persistent_blocks exceeds the int32 mixed ABI v2");
	TORCH_CHECK(
		means3D.size(0) <= static_cast<int64_t>(INT_MAX),
		"Gaussian row count exceeds the int32 raster ABI");

	for (std::size_t task_index = 0; task_index < inputs.size(); ++task_index)
	{
		const torch::Tensor& input = inputs[task_index];
		const torch::Tensor& weight = weights[task_index];
		const torch::Tensor& bias = biases[task_index];
		TORCH_CHECK(
			input.is_cuda(), "head_inputs[", task_index, "] must be a CUDA tensor");
		TORCH_CHECK(
			weight.is_cuda(), "head_weights[", task_index, "] must be a CUDA tensor");
		TORCH_CHECK(
			bias.is_cuda(), "head_biases[", task_index, "] must be a CUDA tensor");
		TORCH_CHECK(
			input.is_contiguous(), "head_inputs[", task_index, "] must be contiguous");
		TORCH_CHECK(
			weight.is_contiguous(), "head_weights[", task_index, "] must be contiguous");
		TORCH_CHECK(
			bias.is_contiguous(), "head_biases[", task_index, "] must be contiguous");
		TORCH_CHECK(
			input.scalar_type() == at::kHalf,
			"head_inputs[", task_index, "] must be float16");
		TORCH_CHECK(
			weight.scalar_type() == at::kHalf,
			"head_weights[", task_index, "] must be float16");
		TORCH_CHECK(
			bias.scalar_type() == at::kFloat,
			"head_biases[", task_index, "] must be float32");
		TORCH_CHECK(
			input.dim() == 2 && input.size(1) == 128,
			"head_inputs[", task_index, "] must have shape [N, 128]");
		TORCH_CHECK(
			weight.dim() == 2 && weight.size(0) == 128 &&
				weight.size(1) == 128,
			"head_weights[", task_index, "] must have shape [128, 128]");
		TORCH_CHECK(
			bias.dim() == 1 && bias.size(0) == 128,
			"head_biases[", task_index, "] must have shape [128]");
		TORCH_CHECK(
			means3D.device() == input.device() &&
				input.device() == weight.device() &&
				input.device() == bias.device(),
			"raster and head task ", task_index,
			" tensors must be on the same CUDA device");
		TORCH_CHECK(
			input.size(0) <= static_cast<int64_t>(INT_MAX),
			"head_inputs[", task_index,
			"] row count exceeds the int32 mixed ABI v2");
		TORCH_CHECK(
			isNativeWmmaAligned(input),
			"head_inputs[", task_index,
			"] data pointer must be natively 32-byte aligned for WMMA; "
			"misaligned contiguous views are not supported");
		TORCH_CHECK(
			isNativeWmmaAligned(weight),
			"head_weights[", task_index,
			"] data pointer must be natively 32-byte aligned for WMMA; "
			"misaligned contiguous views are not supported");
	}
}

void checkMixedPackedHeadsArguments(
	const torch::Tensor& means3D,
	const torch::Tensor& input,
	const torch::Tensor& weights,
	const torch::Tensor& biases,
	int64_t worker_groups,
	int64_t persistent_blocks)
{
	TORCH_CHECK(
		!at::GradMode::is_enabled(),
		"rasterize_gaussians_with_packed_heads is inference-only; "
		"call it under torch.no_grad()");
	TORCH_CHECK(weights.dim() == 3,
		"packed head_weights must have shape [H, 128, 128]");
	const int64_t head_count = weights.size(0);
	TORCH_CHECK(
		head_count >= 1 &&
			head_count <= CudaRasterizer::Tacker::kMaxHeadTasksV2,
		"packed head count must be in [1, 5]");
	TORCH_CHECK(
		worker_groups >= CudaRasterizer::Tacker::kMinWorkerGroupsV2 &&
			worker_groups <= head_count,
		"packed worker_groups must be in [1, head_count]");
	TORCH_CHECK(persistent_blocks >= 0, "persistent_blocks must be >= 0");
	TORCH_CHECK(
		persistent_blocks <= static_cast<int64_t>(INT_MAX),
		"persistent_blocks exceeds the int32 mixed packed ABI v3");
	TORCH_CHECK(
		means3D.size(0) <= static_cast<int64_t>(INT_MAX),
		"Gaussian row count exceeds the int32 raster ABI");

	TORCH_CHECK(input.is_cuda(), "packed head_input must be a CUDA tensor");
	TORCH_CHECK(weights.is_cuda(), "packed head_weights must be a CUDA tensor");
	TORCH_CHECK(biases.is_cuda(), "packed head_biases must be a CUDA tensor");
	TORCH_CHECK(input.is_contiguous(), "packed head_input must be contiguous");
	TORCH_CHECK(weights.is_contiguous(), "packed head_weights must be contiguous");
	TORCH_CHECK(biases.is_contiguous(), "packed head_biases must be contiguous");
	TORCH_CHECK(input.scalar_type() == at::kHalf,
		"packed head_input must be float16");
	TORCH_CHECK(weights.scalar_type() == at::kHalf,
		"packed head_weights must be float16");
	TORCH_CHECK(biases.scalar_type() == at::kFloat,
		"packed head_biases must be float32");
	TORCH_CHECK(
		input.dim() == 2 && input.size(1) == 128,
		"packed head_input must have shape [N, 128]");
	TORCH_CHECK(
		weights.size(1) == 128 && weights.size(2) == 128,
		"packed head_weights must have shape [H, 128, 128]");
	TORCH_CHECK(
		biases.dim() == 2 && biases.size(0) == head_count &&
			biases.size(1) == 128,
		"packed head_biases must have shape [H, 128]");
	TORCH_CHECK(
		means3D.device() == input.device() &&
			input.device() == weights.device() &&
			input.device() == biases.device(),
		"raster and packed head tensors must be on the same CUDA device");
	TORCH_CHECK(
		input.size(0) <= static_cast<int64_t>(INT_MAX),
		"packed head row count exceeds the int32 mixed ABI v3");
	TORCH_CHECK(
		isNativeWmmaAligned(input),
		"packed head_input data pointer must be natively 32-byte aligned");
	TORCH_CHECK(
		isNativeWmmaAligned(weights),
		"packed head_weights data pointer must be natively 32-byte aligned");
	TORCH_CHECK(
		isNativeWmmaAligned(biases),
		"packed head_biases data pointer must be natively 32-byte aligned");
}

void checkMixedWholeHeadsArguments(
	const torch::Tensor& means3D,
	const std::vector<torch::Tensor>& inputs,
	const std::vector<torch::Tensor>& first_weights,
	const std::vector<torch::Tensor>& first_biases,
	const std::vector<torch::Tensor>& tail_weights,
	const std::vector<torch::Tensor>& tail_biases,
	int64_t worker_groups,
	int64_t persistent_blocks)
{
	TORCH_CHECK(
		!at::GradMode::is_enabled(),
		"rasterize_gaussians_with_whole_heads is inference-only; "
		"call it under torch.no_grad()");
	TORCH_CHECK(
		inputs.size() == first_weights.size() &&
			inputs.size() == first_biases.size() &&
			inputs.size() == tail_weights.size() &&
			inputs.size() == tail_biases.size(),
		"whole-head inputs and parameter lists must have equal lengths");
	TORCH_CHECK(
		!inputs.empty() &&
			inputs.size() <=
				static_cast<std::size_t>(CudaRasterizer::Tacker::kMaxHeadTasksV2),
		"mixed whole-head ABI v4 requires between 1 and 5 tasks");
	TORCH_CHECK(
		worker_groups >= CudaRasterizer::Tacker::kMinWorkerGroupsV2 &&
			worker_groups <= static_cast<int64_t>(inputs.size()),
		"whole-head worker_groups must be in [1, task_count]");
	TORCH_CHECK(persistent_blocks >= 0, "persistent_blocks must be >= 0");
	TORCH_CHECK(
		persistent_blocks <= static_cast<int64_t>(INT_MAX),
		"persistent_blocks exceeds the int32 mixed whole-head ABI v4");
	TORCH_CHECK(
		means3D.size(0) <= static_cast<int64_t>(INT_MAX),
		"Gaussian row count exceeds the int32 raster ABI");

	for (std::size_t task_index = 0; task_index < inputs.size(); ++task_index)
	{
		const torch::Tensor& input = inputs[task_index];
		const torch::Tensor& first_weight = first_weights[task_index];
		const torch::Tensor& first_bias = first_biases[task_index];
		const torch::Tensor& tail_weight = tail_weights[task_index];
		const torch::Tensor& tail_bias = tail_biases[task_index];
		for (const torch::Tensor* tensor : {
				&input, &first_weight, &first_bias, &tail_weight, &tail_bias})
		{
			TORCH_CHECK(tensor->is_cuda(),
				"whole-head task ", task_index, " tensors must be CUDA tensors");
			TORCH_CHECK(tensor->is_contiguous(),
				"whole-head task ", task_index, " tensors must be contiguous");
			TORCH_CHECK(isNativeWmmaAligned(*tensor),
				"whole-head task ", task_index,
				" tensor data pointers must be natively 32-byte aligned");
		}
		TORCH_CHECK(input.scalar_type() == at::kHalf,
			"whole-head inputs[", task_index, "] must be float16");
		TORCH_CHECK(first_weight.scalar_type() == at::kHalf,
			"whole-head first_weights[", task_index, "] must be float16");
		TORCH_CHECK(first_bias.scalar_type() == at::kFloat,
			"whole-head first_biases[", task_index, "] must be float32");
		TORCH_CHECK(tail_weight.scalar_type() == at::kFloat,
			"whole-head tail_weights[", task_index, "] must be float32");
		TORCH_CHECK(tail_bias.scalar_type() == at::kFloat,
			"whole-head tail_biases[", task_index, "] must be float32");
		TORCH_CHECK(
			input.dim() == 2 && input.size(1) == 128,
			"whole-head inputs[", task_index, "] must have shape [N, 128]");
		TORCH_CHECK(
			first_weight.dim() == 2 && first_weight.size(0) == 128 &&
				first_weight.size(1) == 128,
			"whole-head first_weights[", task_index,
			"] must have shape [128, 128]");
		TORCH_CHECK(
			first_bias.dim() == 1 && first_bias.size(0) == 128,
			"whole-head first_biases[", task_index,
			"] must have shape [128]");
		TORCH_CHECK(
			tail_weight.dim() == 2 && tail_weight.size(1) == 128 &&
				tail_weight.size(0) >= 1 &&
				tail_weight.size(0) <= CudaRasterizer::Tacker::kMaxTailFeaturesV2,
			"whole-head tail_weights[", task_index,
			"] must have shape [O, 128] with 1 <= O <= 128");
		TORCH_CHECK(
			tail_bias.dim() == 1 && tail_bias.size(0) == tail_weight.size(0),
			"whole-head tail_biases[", task_index, "] must have shape [O]");
		TORCH_CHECK(
			means3D.device() == input.device() &&
				input.device() == first_weight.device() &&
				input.device() == first_bias.device() &&
				input.device() == tail_weight.device() &&
				input.device() == tail_bias.device(),
			"raster and whole-head task ", task_index,
			" tensors must be on the same CUDA device");
		TORCH_CHECK(
			input.size(0) <= static_cast<int64_t>(INT_MAX),
			"whole-head inputs[", task_index,
			"] row count exceeds the int32 mixed ABI v4");
	}
}

void checkMixedDeformMlpArguments(
	const torch::Tensor& means3D,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& tail_weights,
	const torch::Tensor& tail_biases,
	int64_t persistent_blocks)
{
	TORCH_CHECK(
		!at::GradMode::is_enabled(),
		"rasterize_gaussians_with_deform_mlp is inference-only; "
		"call it under torch.no_grad()");
	const std::pair<const char*, const torch::Tensor*> tensors[] = {
		{"deform_features", &deform_features},
		{"backbone_weight", &backbone_weight},
		{"backbone_bias", &backbone_bias},
		{"first_weights", &first_weights},
		{"first_biases", &first_biases},
		{"tail_weights", &tail_weights},
		{"tail_biases", &tail_biases},
	};
	for (const auto& named : tensors)
	{
		TORCH_CHECK(named.second->is_cuda(), named.first, " must be a CUDA tensor");
		TORCH_CHECK(named.second->is_contiguous(), named.first, " must be contiguous");
		TORCH_CHECK(
			named.second->device() == means3D.device(), named.first,
			" must be on the same CUDA device as means3D");
	}
	TORCH_CHECK(
		deform_features.scalar_type() == at::kFloat &&
		backbone_weight.scalar_type() == at::kFloat &&
		backbone_bias.scalar_type() == at::kFloat &&
		first_biases.scalar_type() == at::kFloat &&
		tail_weights.scalar_type() == at::kFloat &&
		tail_biases.scalar_type() == at::kFloat,
		"deform features, backbone, biases, and tail parameters must be float32");
	TORCH_CHECK(
		first_weights.scalar_type() == at::kHalf,
		"first_weights must be float16");
	TORCH_CHECK(
		deform_features.dim() == 2 && deform_features.size(1) == 32,
		"deform_features must have shape [N, 32]");
	TORCH_CHECK(
		backbone_weight.dim() == 2 && backbone_weight.size(0) == 32 &&
			backbone_weight.size(1) == 128,
		"backbone_weight must have shape [32, 128]");
	TORCH_CHECK(
		backbone_bias.dim() == 1 && backbone_bias.size(0) == 128,
		"backbone_bias must have shape [128]");
	TORCH_CHECK(
		first_weights.dim() == 3 && first_weights.size(0) == 5 &&
			first_weights.size(1) == 128 && first_weights.size(2) == 128,
		"first_weights must have shape [5, 128, 128]");
	TORCH_CHECK(
		first_biases.dim() == 2 && first_biases.size(0) == 5 &&
			first_biases.size(1) == 128,
		"first_biases must have shape [5, 128]");
	TORCH_CHECK(
		tail_weights.dim() == 2 && tail_weights.size(0) == 59 &&
			tail_weights.size(1) == 128,
		"tail_weights must have shape [59, 128]");
	TORCH_CHECK(
		tail_biases.dim() == 1 && tail_biases.size(0) == 59,
		"tail_biases must have shape [59]");
	TORCH_CHECK(
		deform_features.size(0) <= static_cast<int64_t>(INT_MAX),
		"deform row count exceeds the int32 mixed ABI v5");
	TORCH_CHECK(
		means3D.size(0) <= static_cast<int64_t>(INT_MAX),
		"Gaussian row count exceeds the int32 raster ABI");
	TORCH_CHECK(
		isNativeWmmaAligned(first_weights),
		"first_weights data pointer must be natively 32-byte aligned for WMMA");
	TORCH_CHECK(persistent_blocks >= 0, "persistent_blocks must be >= 0");
	TORCH_CHECK(
		persistent_blocks <= static_cast<int64_t>(INT_MAX),
		"persistent_blocks exceeds the int32 mixed ABI v5");
}

void checkMixedDeformMlpTensorTailArguments(
	const torch::Tensor& raster_device_tensor,
	const torch::Tensor& deform_features,
	const torch::Tensor& backbone_weight,
	const torch::Tensor& backbone_bias,
	const torch::Tensor& first_weights,
	const torch::Tensor& first_biases,
	const torch::Tensor& packed_tail_weights,
	const torch::Tensor& tail_biases,
	int64_t persistent_blocks)
{
	TORCH_CHECK(
		!at::GradMode::is_enabled(),
		"mixed Tensor-tail deform MLP is inference-only; call it under torch.no_grad()");
	const std::pair<const char*, const torch::Tensor*> tensors[] = {
		{"deform_features", &deform_features},
		{"backbone_weight", &backbone_weight},
		{"backbone_bias", &backbone_bias},
		{"first_weights", &first_weights},
		{"first_biases", &first_biases},
		{"packed_tail_weights", &packed_tail_weights},
		{"tail_biases", &tail_biases},
	};
	for (const auto& named : tensors)
	{
		TORCH_CHECK(named.second->is_cuda(), named.first, " must be a CUDA tensor");
		TORCH_CHECK(named.second->is_contiguous(), named.first, " must be contiguous");
		TORCH_CHECK(
			named.second->device() == raster_device_tensor.device(), named.first,
			" must be on the same CUDA device as the Raster state");
	}
	TORCH_CHECK(
		deform_features.scalar_type() == at::kFloat &&
		backbone_weight.scalar_type() == at::kFloat &&
		backbone_bias.scalar_type() == at::kFloat &&
		first_biases.scalar_type() == at::kFloat &&
		tail_biases.scalar_type() == at::kFloat,
		"deform features, backbone, first biases, and tail biases must be float32");
	TORCH_CHECK(
		first_weights.scalar_type() == at::kHalf &&
		packed_tail_weights.scalar_type() == at::kHalf,
		"first_weights and packed_tail_weights must be float16");
	TORCH_CHECK(
		deform_features.dim() == 2 && deform_features.size(1) == 32,
		"deform_features must have shape [N, 32]");
	TORCH_CHECK(
		backbone_weight.dim() == 2 && backbone_weight.size(0) == 32 &&
			backbone_weight.size(1) == 128,
		"backbone_weight must have shape [32, 128]");
	TORCH_CHECK(
		backbone_bias.dim() == 1 && backbone_bias.size(0) == 128,
		"backbone_bias must have shape [128]");
	TORCH_CHECK(
		first_weights.dim() == 3 && first_weights.size(0) == 5 &&
			first_weights.size(1) == 128 && first_weights.size(2) == 128,
		"first_weights must have shape [5, 128, 128]");
	TORCH_CHECK(
		first_biases.dim() == 2 && first_biases.size(0) == 5 &&
			first_biases.size(1) == 128,
		"first_biases must have shape [5, 128]");
	TORCH_CHECK(
		packed_tail_weights.dim() == 2 &&
			packed_tail_weights.size(0) == 112 &&
			packed_tail_weights.size(1) == 128,
		"packed_tail_weights must have shape [112, 128]");
	TORCH_CHECK(
		tail_biases.dim() == 1 && tail_biases.size(0) == 59,
		"tail_biases must have shape [59]");
	TORCH_CHECK(
		deform_features.size(0) <= static_cast<int64_t>(INT_MAX),
		"deform row count exceeds the int32 mixed ABI v6");
	TORCH_CHECK(
		isNativeWmmaAligned(first_weights) &&
		isNativeWmmaAligned(packed_tail_weights),
		"first_weights and packed_tail_weights must be 32-byte aligned for WMMA");
	TORCH_CHECK(persistent_blocks >= 0, "persistent_blocks must be >= 0");
	TORCH_CHECK(
		persistent_blocks <= static_cast<int64_t>(INT_MAX),
		"persistent_blocks exceeds the int32 mixed ABI v6");
}

void checkPreparedRasterArguments(
	const torch::Tensor& background,
	const torch::Tensor& colors,
	const torch::Tensor& out_color,
	const torch::Tensor& out_depth,
	const torch::Tensor& radii,
	const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& image_buffer,
	int64_t num_rendered,
	int image_height,
	int image_width)
{
	TORCH_CHECK(
		!at::GradMode::is_enabled(),
		"prepared Raster overlap is inference-only; call it under torch.no_grad()");
	checkRasterDimensions(image_height, image_width);
	TORCH_CHECK(num_rendered >= 0, "num_rendered must be >= 0");
	TORCH_CHECK(
		num_rendered <= static_cast<int64_t>(INT_MAX),
		"num_rendered exceeds the int32 Raster ABI");
	TORCH_CHECK(radii.is_cuda(), "radii must be a CUDA tensor");
	TORCH_CHECK(radii.is_contiguous(), "radii must be contiguous");
	TORCH_CHECK(radii.scalar_type() == at::kInt, "radii must be int32");
	TORCH_CHECK(radii.dim() == 1, "radii must have shape [P]");
	TORCH_CHECK(
		radii.size(0) <= static_cast<int64_t>(INT_MAX),
		"Gaussian row count exceeds the int32 Raster ABI");
	const int64_t P = radii.size(0);

	for (const auto& named : {
		std::make_pair("background", &background),
		std::make_pair("colors", &colors),
		std::make_pair("out_color", &out_color),
		std::make_pair("out_depth", &out_depth),
		std::make_pair("geom_buffer", &geom_buffer),
		std::make_pair("binning_buffer", &binning_buffer),
		std::make_pair("image_buffer", &image_buffer)})
	{
		TORCH_CHECK(named.second->is_cuda(), named.first, " must be a CUDA tensor");
		TORCH_CHECK(
			named.second->is_contiguous(), named.first, " must be contiguous");
		TORCH_CHECK(
			named.second->device() == radii.device(), named.first,
			" must be on the same CUDA device as radii");
	}
	TORCH_CHECK(
		background.scalar_type() == at::kFloat &&
		colors.scalar_type() == at::kFloat &&
		out_color.scalar_type() == at::kFloat &&
		out_depth.scalar_type() == at::kFloat,
		"background, colors, out_color, and out_depth must be float32");
	TORCH_CHECK(
		geom_buffer.scalar_type() == at::kByte &&
		binning_buffer.scalar_type() == at::kByte &&
		image_buffer.scalar_type() == at::kByte,
		"prepared Raster buffers must have dtype uint8");
	TORCH_CHECK(
		background.dim() == 1 && background.size(0) == NUM_CHANNELS,
		"background must have shape [3]");
	TORCH_CHECK(
		isPlaceholder(colors) ||
			(colors.dim() == 2 && colors.size(0) == P &&
			 colors.size(1) == NUM_CHANNELS),
		"colors must be a [0] placeholder or have shape [P, 3]");
	TORCH_CHECK(
		out_color.dim() == 3 && out_color.size(0) == NUM_CHANNELS &&
		out_color.size(1) == image_height && out_color.size(2) == image_width,
		"out_color must have shape [3, H, W]");
	TORCH_CHECK(
		out_depth.dim() == 3 && out_depth.size(0) == 1 &&
		out_depth.size(1) == image_height && out_depth.size(2) == image_width,
		"out_depth must have shape [1, H, W]");

	if (P > 0)
	{
		TORCH_CHECK(
			static_cast<size_t>(geom_buffer.numel()) >=
				CudaRasterizer::required<CudaRasterizer::GeometryState>(P),
			"geom_buffer is smaller than the prepared GeometryState");
		TORCH_CHECK(
			static_cast<size_t>(binning_buffer.numel()) >=
				CudaRasterizer::required<CudaRasterizer::BinningState>(num_rendered),
			"binning_buffer is smaller than the prepared BinningState");
		const int64_t pixels =
			static_cast<int64_t>(image_height) * image_width;
		TORCH_CHECK(
			static_cast<size_t>(image_buffer.numel()) >=
				CudaRasterizer::required<CudaRasterizer::ImageState>(pixels),
			"image_buffer is smaller than the prepared ImageState");
	}
}

using MixedRasterResult = std::tuple<int, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

MixedRasterResult rasterizeWithNewMixedBackend(
	const MixedRasterChoices& raster_choices,
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
	const CudaRasterizer::MixedPackedHeadBundleV2* packed_heads,
	const CudaRasterizer::MixedWholeHeadBundleV2* whole_heads,
	const CudaRasterizer::MixedDeformMlpTaskV1* deform_mlp)
{
	const int selected_backends =
		(packed_heads != nullptr ? 1 : 0) +
		(whole_heads != nullptr ? 1 : 0) +
		(deform_mlp != nullptr ? 1 : 0);
	TORCH_CHECK(
		selected_backends == 1,
		"exactly one C3/C4/full-deform mixed backend must be selected");
	const c10::cuda::CUDAGuard device_guard(means3D.device());
	const int P = static_cast<int>(means3D.size(0));
	const int H = image_height;
	const int W = image_width;
	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full(
		{P}, 0, means3D.options().dtype(torch::kInt32));
	const torch::TensorOptions byte_options =
		means3D.options().dtype(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, byte_options);
	torch::Tensor binningBuffer = torch::empty({0}, byte_options);
	torch::Tensor imgBuffer = torch::empty({0}, byte_options);
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	const torch::Tensor background_contiguous = background.contiguous();
	int rendered = 0;
	if (P != 0)
	{
		const int M = raster_choices.use_sh ? static_cast<int>(sh.size(1)) : 0;
		const torch::Tensor means3D_contiguous = means3D.contiguous();
		const torch::Tensor sh_contiguous = sh.contiguous();
		const torch::Tensor colors_contiguous = colors.contiguous();
		const torch::Tensor opacity_contiguous = opacity.contiguous();
		const torch::Tensor scales_contiguous = scales.contiguous();
		const torch::Tensor rotations_contiguous = rotations.contiguous();
		const torch::Tensor cov3D_contiguous = cov3D_precomp.contiguous();
		const torch::Tensor viewmatrix_contiguous = viewmatrix.contiguous();
		const torch::Tensor projmatrix_contiguous = projmatrix.contiguous();
		const torch::Tensor campos_contiguous = campos.contiguous();
		const float* sh_ptr = raster_choices.use_sh
			? sh_contiguous.data_ptr<float>() : nullptr;
		const float* colors_ptr = raster_choices.use_sh
			? nullptr : colors_contiguous.data_ptr<float>();
		const float* scales_ptr = raster_choices.use_cov3d
			? nullptr : scales_contiguous.data_ptr<float>();
		const float* rotations_ptr = raster_choices.use_cov3d
			? nullptr : rotations_contiguous.data_ptr<float>();
		const float* cov3D_ptr = raster_choices.use_cov3d
			? cov3D_contiguous.data_ptr<float>() : nullptr;

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc, binningFunc, imgFunc,
			P, degree, M,
			background_contiguous.data_ptr<float>(),
			W, H,
			means3D_contiguous.data_ptr<float>(),
			sh_ptr,
			colors_ptr,
			opacity_contiguous.data_ptr<float>(),
			scales_ptr,
			scale_modifier,
			rotations_ptr,
			cov3D_ptr,
			viewmatrix_contiguous.data_ptr<float>(),
			projmatrix_contiguous.data_ptr<float>(),
			campos_contiguous.data_ptr<float>(),
			tan_fovx, tan_fovy,
			prefiltered,
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			radii.data_ptr<int>(),
			debug,
			stream,
				nullptr,
				nullptr,
				packed_heads,
				whole_heads,
				deform_mlp);
	}
	else if (packed_heads != nullptr)
	{
		CudaRasterizer::Tacker::launchMixedRenderPackedHeads(
			dim3(0, 0, 1), nullptr, nullptr, W, H, nullptr, nullptr,
			nullptr, nullptr, nullptr, nullptr,
			background_contiguous.data_ptr<float>(),
			out_color.data_ptr<float>(), out_depth.data_ptr<float>(),
			*packed_heads, stream);
	}
	else if (whole_heads != nullptr)
	{
		CudaRasterizer::Tacker::launchMixedRenderWholeHeads(
			dim3(0, 0, 1), nullptr, nullptr, W, H, nullptr, nullptr,
			nullptr, nullptr, nullptr, nullptr,
			background_contiguous.data_ptr<float>(),
			out_color.data_ptr<float>(), out_depth.data_ptr<float>(),
				*whole_heads, stream);
	}
	else
	{
		CudaRasterizer::Tacker::launchMixedRenderDeformMlp(
			dim3(0, 0, 1), nullptr, nullptr, W, H, nullptr, nullptr,
			nullptr, nullptr, nullptr, nullptr,
			background_contiguous.data_ptr<float>(),
			out_color.data_ptr<float>(), out_depth.data_ptr<float>(),
			*deform_mlp, stream);
	}

	bool backend_has_rows = packed_heads != nullptr
		? packed_heads->rows > 0
		: false;
	if (whole_heads != nullptr)
	{
		for (int task_index = 0; task_index < whole_heads->task_count; ++task_index)
			backend_has_rows = backend_has_rows ||
				whole_heads->tasks[task_index].rows > 0;
	}
	if (deform_mlp != nullptr)
		backend_has_rows = deform_mlp->rows > 0;
	if (P != 0 || backend_has_rows)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
				"mixed Raster launch failed: ", cudaGetErrorString(status));
	}
	return std::make_tuple(
		rendered, out_color, out_depth, radii,
		geomBuffer, binningBuffer, imgBuffer);
}

}  // namespace

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
	const bool debug)
{
	return PrepareGaussiansForDeformMlpOverlapHandoffCUDA(
		background, means3D, colors, opacity, scales, rotations, scale_modifier,
		cov3D_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy,
		image_height, image_width, sh, degree, campos, prefiltered, debug,
		0, 0, 0);
}

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
	const uint64_t duplicate_done_handle)
{
	const bool handoff = handoff_stream_handle != 0;
	TORCH_CHECK((!handoff && deform_done_handle == 0 && duplicate_done_handle == 0) ||
		(handoff && deform_done_handle != 0 && duplicate_done_handle != 0),
		"handoff requires a nonzero target stream and both producer events");
	TORCH_CHECK(
		!at::GradMode::is_enabled(),
		"prepare_gaussians_for_deform_mlp_overlap is inference-only; "
		"call it under torch.no_grad()");
	const MixedRasterChoices raster_choices = checkMixedRasterArguments(
		background, means3D, colors, opacity, scales, rotations,
		cov3D_precomp, viewmatrix, projmatrix, sh, campos, degree);
	checkRasterDimensions(image_height, image_width);

	const c10::cuda::CUDAGuard device_guard(means3D.device());
	const int P = static_cast<int>(means3D.size(0));
	const int H = image_height;
	const int W = image_width;
	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor out_color =
		torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full(
		{P}, 0, means3D.options().dtype(torch::kInt32));
	const torch::TensorOptions byte_options =
		means3D.options().dtype(torch::kByte);
	torch::Tensor geom_buffer = torch::empty({0}, byte_options);
	torch::Tensor binning_buffer = torch::empty({0}, byte_options);
	torch::Tensor image_buffer = torch::empty({0}, byte_options);
	std::function<char*(size_t)> geom_func = resizeFunctional(geom_buffer);
	std::function<char*(size_t)> binning_func = resizeFunctional(binning_buffer);
	std::function<char*(size_t)> image_func = resizeFunctional(image_buffer);

	int rendered = 0;
	if (P != 0)
	{
		const int M = raster_choices.use_sh ? static_cast<int>(sh.size(1)) : 0;
		const torch::Tensor means3D_contiguous = means3D.contiguous();
		const torch::Tensor sh_contiguous = sh.contiguous();
		const torch::Tensor colors_contiguous = colors.contiguous();
		const torch::Tensor opacity_contiguous = opacity.contiguous();
		const torch::Tensor scales_contiguous = scales.contiguous();
		const torch::Tensor rotations_contiguous = rotations.contiguous();
		const torch::Tensor cov3D_contiguous = cov3D_precomp.contiguous();
		const torch::Tensor viewmatrix_contiguous = viewmatrix.contiguous();
		const torch::Tensor projmatrix_contiguous = projmatrix.contiguous();
		const torch::Tensor campos_contiguous = campos.contiguous();
		const float* sh_ptr = raster_choices.use_sh
			? sh_contiguous.data_ptr<float>() : nullptr;
		const float* colors_ptr = raster_choices.use_sh
			? nullptr : colors_contiguous.data_ptr<float>();
		const float* scales_ptr = raster_choices.use_cov3d
			? nullptr : scales_contiguous.data_ptr<float>();
		const float* rotations_ptr = raster_choices.use_cov3d
			? nullptr : rotations_contiguous.data_ptr<float>();
		const float* cov3D_ptr = raster_choices.use_cov3d
			? cov3D_contiguous.data_ptr<float>() : nullptr;

		rendered =
			CudaRasterizer::Rasterizer::prepareForwardForDeformMlpOverlap(
				geom_func, binning_func, image_func,
				P, degree, M,
				W, H,
				means3D_contiguous.data_ptr<float>(),
				sh_ptr,
				colors_ptr,
				opacity_contiguous.data_ptr<float>(),
				scales_ptr,
				scale_modifier,
				rotations_ptr,
				cov3D_ptr,
				viewmatrix_contiguous.data_ptr<float>(),
				projmatrix_contiguous.data_ptr<float>(),
				campos_contiguous.data_ptr<float>(),
				tan_fovx, tan_fovy,
				prefiltered,
				radii.data_ptr<int>(),
				debug,
				at::cuda::getCurrentCUDAStream().stream(),
				reinterpret_cast<cudaStream_t>(handoff_stream_handle),
				reinterpret_cast<cudaEvent_t>(deform_done_handle),
				reinterpret_cast<cudaEvent_t>(duplicate_done_handle));
	}

	return std::make_tuple(
		rendered, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer);
}

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
	const int64_t persistent_blocks)
{
	checkPreparedRasterArguments(
		background, colors, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer,
		num_rendered, image_height, image_width);
	checkMixedDeformMlpArguments(
		radii, deform_features, backbone_weight, backbone_bias,
		first_weights, first_biases, tail_weights, tail_biases,
		persistent_blocks);

	const c10::cuda::CUDAGuard device_guard(radii.device());
	constexpr int output_widths[CudaRasterizer::kDeformMlpOutputCountV1] = {
		3, 3, 4, 1, 48};
	std::vector<torch::Tensor> deform_outputs;
	deform_outputs.reserve(CudaRasterizer::kDeformMlpOutputCountV1);
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
	{
		deform_outputs.push_back(torch::empty(
			{deform_features.size(0), output_widths[output]},
			deform_features.options().dtype(torch::kFloat32).requires_grad(false)));
	}

	CudaRasterizer::MixedDeformMlpTaskV1 deform_task = {};
	deform_task.features = deform_features.data_ptr<float>();
	deform_task.backbone_weight = backbone_weight.data_ptr<float>();
	deform_task.backbone_bias = backbone_bias.data_ptr<float>();
	deform_task.first_weights = reinterpret_cast<const void*>(
		first_weights.data_ptr<at::Half>());
	deform_task.first_biases = first_biases.data_ptr<float>();
	deform_task.tail_weights = tail_weights.data_ptr<float>();
	deform_task.tail_biases = tail_biases.data_ptr<float>();
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
		deform_task.outputs[output] = deform_outputs[output].data_ptr<float>();
	deform_task.rows = static_cast<int>(deform_features.size(0));
	deform_task.persistent_blocks = static_cast<int>(persistent_blocks);

	const int P = static_cast<int>(radii.size(0));
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	if (P != 0)
	{
		CudaRasterizer::Rasterizer::renderPreparedDeformMlpOverlap(
			P,
			static_cast<int>(num_rendered),
			background.data_ptr<float>(),
			image_width, image_height,
			isPlaceholder(colors) ? nullptr : colors.data_ptr<float>(),
			reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>()),
			reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>()),
			reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>()),
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			debug,
			stream,
			&deform_task);
	}
	else
	{
		CudaRasterizer::Tacker::launchMixedRenderDeformMlp(
			dim3(0, 0, 1), nullptr, nullptr,
			image_width, image_height,
			nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
			background.data_ptr<float>(),
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			deform_task,
			stream);
	}

	if (P != 0 || deform_task.rows > 0)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"prepared mixed Raster launch failed: ", cudaGetErrorString(status));
	}
	return deform_outputs;
}

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
	const int64_t persistent_blocks)
{
	checkPreparedRasterArguments(
		background, colors, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer,
		num_rendered, image_height, image_width);
	checkMixedDeformMlpTensorTailArguments(
		radii, deform_features, backbone_weight, backbone_bias,
		first_weights, first_biases, packed_tail_weights, tail_biases,
		persistent_blocks);

	const c10::cuda::CUDAGuard device_guard(radii.device());
	constexpr int output_widths[CudaRasterizer::kDeformMlpOutputCountV1] = {
		3, 3, 4, 1, 48};
	std::vector<torch::Tensor> deform_outputs;
	deform_outputs.reserve(CudaRasterizer::kDeformMlpOutputCountV1);
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
	{
		deform_outputs.push_back(torch::empty(
			{deform_features.size(0), output_widths[output]},
			deform_features.options().dtype(torch::kFloat32).requires_grad(false)));
	}

	CudaRasterizer::MixedDeformMlpTensorTailTaskV2 deform_task = {};
	deform_task.features = deform_features.data_ptr<float>();
	deform_task.backbone_weight = backbone_weight.data_ptr<float>();
	deform_task.backbone_bias = backbone_bias.data_ptr<float>();
	deform_task.first_weights = reinterpret_cast<const void*>(
		first_weights.data_ptr<at::Half>());
	deform_task.first_biases = first_biases.data_ptr<float>();
	deform_task.packed_tail_weights = reinterpret_cast<const void*>(
		packed_tail_weights.data_ptr<at::Half>());
	deform_task.tail_biases = tail_biases.data_ptr<float>();
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
		deform_task.outputs[output] = deform_outputs[output].data_ptr<float>();
	deform_task.rows = static_cast<int>(deform_features.size(0));
	deform_task.persistent_blocks = static_cast<int>(persistent_blocks);

	const int P = static_cast<int>(radii.size(0));
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	if (P != 0)
	{
		CudaRasterizer::Rasterizer::renderPreparedDeformMlpTensorTailOverlap(
			P,
			static_cast<int>(num_rendered),
			background.data_ptr<float>(),
			image_width, image_height,
			isPlaceholder(colors) ? nullptr : colors.data_ptr<float>(),
			reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>()),
			reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>()),
			reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>()),
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			debug,
			stream,
			deform_task);
	}
	else
	{
		CudaRasterizer::Tacker::launchMixedRenderDeformMlpTensorTail(
			dim3(0, 0, 1), nullptr, nullptr,
			image_width, image_height,
			nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
			background.data_ptr<float>(),
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			deform_task,
			stream);
	}

	if (P != 0 || deform_task.rows > 0)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"prepared mixed Tensor-tail Raster launch failed: ",
			cudaGetErrorString(status));
	}
	return deform_outputs;
}

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
	const int64_t persistent_blocks)
{
	checkPreparedRasterArguments(
		background, colors, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer,
		num_rendered, image_height, image_width);
	checkMixedDeformMlpTensorTailArguments(
		radii, deform_features, backbone_weight, backbone_bias,
		first_weights, first_biases, packed_tail_weights, tail_biases,
		persistent_blocks);

	const c10::cuda::CUDAGuard device_guard(radii.device());
	constexpr int output_widths[CudaRasterizer::kDeformMlpOutputCountV1] = {
		3, 3, 4, 1, 48};
	std::vector<torch::Tensor> deform_outputs;
	deform_outputs.reserve(CudaRasterizer::kDeformMlpOutputCountV1);
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
	{
		deform_outputs.push_back(torch::empty(
			{deform_features.size(0), output_widths[output]},
			deform_features.options().dtype(torch::kFloat32).requires_grad(false)));
	}

	CudaRasterizer::MixedDeformMlpTensorTailTaskV2 deform_task = {};
	deform_task.features = deform_features.data_ptr<float>();
	deform_task.backbone_weight = backbone_weight.data_ptr<float>();
	deform_task.backbone_bias = backbone_bias.data_ptr<float>();
	deform_task.first_weights = reinterpret_cast<const void*>(
		first_weights.data_ptr<at::Half>());
	deform_task.first_biases = first_biases.data_ptr<float>();
	deform_task.packed_tail_weights = reinterpret_cast<const void*>(
		packed_tail_weights.data_ptr<at::Half>());
	deform_task.tail_biases = tail_biases.data_ptr<float>();
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
		deform_task.outputs[output] = deform_outputs[output].data_ptr<float>();
	deform_task.rows = static_cast<int>(deform_features.size(0));
	deform_task.persistent_blocks = static_cast<int>(persistent_blocks);

	const int P = static_cast<int>(radii.size(0));
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	if (P != 0)
	{
		CudaRasterizer::Rasterizer::
			renderPreparedDeformMlpTensorTailLowRegisterOverlap(
				P,
				static_cast<int>(num_rendered),
				background.data_ptr<float>(),
				image_width, image_height,
				isPlaceholder(colors) ? nullptr : colors.data_ptr<float>(),
				reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>()),
				out_color.data_ptr<float>(),
				out_depth.data_ptr<float>(),
				debug,
				stream,
				deform_task);
	}
	else
	{
		CudaRasterizer::Tacker::launchMixedRenderDeformMlpTensorTailLowRegister(
			dim3(0, 0, 1), nullptr, nullptr,
			image_width, image_height,
			nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
			background.data_ptr<float>(),
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			deform_task,
			stream);
	}

	if (P != 0 || deform_task.rows > 0)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"prepared mixed low-register Tensor-tail Raster launch failed: ",
			cudaGetErrorString(status));
	}
	return deform_outputs;
}

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
	const int64_t persistent_blocks)
{
	checkPreparedRasterArguments(
		background, colors, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer,
		num_rendered, image_height, image_width);
	checkMixedDeformMlpTensorTailArguments(
		radii, deform_features, backbone_weight, backbone_bias,
		first_weights, first_biases, packed_tail_weights, tail_biases,
		persistent_blocks);

	const c10::cuda::CUDAGuard device_guard(radii.device());
	constexpr int output_widths[CudaRasterizer::kDeformMlpOutputCountV1] = {
		3, 3, 4, 1, 48};
	std::vector<torch::Tensor> deform_outputs;
	deform_outputs.reserve(CudaRasterizer::kDeformMlpOutputCountV1);
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
	{
		deform_outputs.push_back(torch::empty(
			{deform_features.size(0), output_widths[output]},
			deform_features.options().dtype(torch::kFloat32).requires_grad(false)));
	}

	CudaRasterizer::MixedDeformMlpTensorTailTaskV2 deform_task = {};
	deform_task.features = deform_features.data_ptr<float>();
	deform_task.backbone_weight = backbone_weight.data_ptr<float>();
	deform_task.backbone_bias = backbone_bias.data_ptr<float>();
	deform_task.first_weights = reinterpret_cast<const void*>(
		first_weights.data_ptr<at::Half>());
	deform_task.first_biases = first_biases.data_ptr<float>();
	deform_task.packed_tail_weights = reinterpret_cast<const void*>(
		packed_tail_weights.data_ptr<at::Half>());
	deform_task.tail_biases = tail_biases.data_ptr<float>();
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
		deform_task.outputs[output] = deform_outputs[output].data_ptr<float>();
	deform_task.rows = static_cast<int>(deform_features.size(0));
	deform_task.persistent_blocks = static_cast<int>(persistent_blocks);

	const int P = static_cast<int>(radii.size(0));
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	if (P != 0)
	{
		CudaRasterizer::Rasterizer::
			renderPreparedDeformMlpTensorTailFourCtaOverlap(
				P,
				static_cast<int>(num_rendered),
				background.data_ptr<float>(),
				image_width, image_height,
				isPlaceholder(colors) ? nullptr : colors.data_ptr<float>(),
				reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>()),
				out_color.data_ptr<float>(),
				out_depth.data_ptr<float>(),
				debug,
				stream,
				deform_task);
	}
	else
	{
		CudaRasterizer::Tacker::launchMixedRenderDeformMlpTensorTailFourCta(
			dim3(0, 0, 1), nullptr, nullptr,
			image_width, image_height,
			nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
			background.data_ptr<float>(),
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			deform_task,
			stream);
	}

	if (P != 0 || deform_task.rows > 0)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"prepared mixed four-CTA Tensor-tail Raster launch failed: ",
			cudaGetErrorString(status));
	}
	return deform_outputs;
}

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
	const int64_t persistent_blocks)
{
	checkPreparedRasterArguments(
		background, colors, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer,
		num_rendered, image_height, image_width);
	checkMixedDeformMlpTensorTailArguments(
		radii, deform_features, backbone_weight, backbone_bias,
		first_weights, first_biases, packed_tail_weights, tail_biases,
		persistent_blocks);

	const c10::cuda::CUDAGuard device_guard(radii.device());
	constexpr int output_widths[CudaRasterizer::kDeformMlpOutputCountV1] = {
		3, 3, 4, 1, 48};
	std::vector<torch::Tensor> deform_outputs;
	deform_outputs.reserve(CudaRasterizer::kDeformMlpOutputCountV1);
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
	{
		deform_outputs.push_back(torch::empty(
			{deform_features.size(0), output_widths[output]},
			deform_features.options().dtype(torch::kFloat32).requires_grad(false)));
	}

	CudaRasterizer::MixedDeformMlpTensorTailTaskV2 deform_task = {};
	deform_task.features = deform_features.data_ptr<float>();
	deform_task.backbone_weight = backbone_weight.data_ptr<float>();
	deform_task.backbone_bias = backbone_bias.data_ptr<float>();
	deform_task.first_weights = reinterpret_cast<const void*>(
		first_weights.data_ptr<at::Half>());
	deform_task.first_biases = first_biases.data_ptr<float>();
	deform_task.packed_tail_weights = reinterpret_cast<const void*>(
		packed_tail_weights.data_ptr<at::Half>());
	deform_task.tail_biases = tail_biases.data_ptr<float>();
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
		deform_task.outputs[output] = deform_outputs[output].data_ptr<float>();
	deform_task.rows = static_cast<int>(deform_features.size(0));
	deform_task.persistent_blocks = static_cast<int>(persistent_blocks);

	const int P = static_cast<int>(radii.size(0));
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	if (P != 0)
	{
		CudaRasterizer::Rasterizer::
			renderPreparedDeformMlpTensorTailBalancedOverlap(
				P,
				static_cast<int>(num_rendered),
				background.data_ptr<float>(),
				image_width, image_height,
				isPlaceholder(colors) ? nullptr : colors.data_ptr<float>(),
				reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>()),
				out_color.data_ptr<float>(),
				out_depth.data_ptr<float>(),
				debug,
				stream,
				deform_task);
	}
	else
	{
		CudaRasterizer::Tacker::launchMixedRenderDeformMlpTensorTailBalanced(
			dim3(0, 0, 1), nullptr, nullptr,
			image_width, image_height,
			nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
			background.data_ptr<float>(),
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			deform_task,
			stream);
	}

	if (P != 0 || deform_task.rows > 0)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"prepared mixed balanced Tensor-tail Raster launch failed: ",
			cudaGetErrorString(status));
	}
	return deform_outputs;
}

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
	const int64_t persistent_blocks)
{
	checkPreparedRasterArguments(
		background, colors, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer,
		num_rendered, image_height, image_width);
	checkMixedDeformMlpTensorTailArguments(
		radii, deform_features, backbone_weight, backbone_bias,
		first_weights, first_biases, packed_tail_weights, tail_biases,
		persistent_blocks);

	const c10::cuda::CUDAGuard device_guard(radii.device());
	constexpr int output_widths[CudaRasterizer::kDeformMlpOutputCountV1] = {
		3, 3, 4, 1, 48};
	std::vector<torch::Tensor> deform_outputs;
	deform_outputs.reserve(CudaRasterizer::kDeformMlpOutputCountV1);
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
	{
		deform_outputs.push_back(torch::empty(
			{deform_features.size(0), output_widths[output]},
			deform_features.options().dtype(torch::kFloat32).requires_grad(false)));
	}

	CudaRasterizer::MixedDeformMlpTensorTailTaskV2 deform_task = {};
	deform_task.features = deform_features.data_ptr<float>();
	deform_task.backbone_weight = backbone_weight.data_ptr<float>();
	deform_task.backbone_bias = backbone_bias.data_ptr<float>();
	deform_task.first_weights = reinterpret_cast<const void*>(
		first_weights.data_ptr<at::Half>());
	deform_task.first_biases = first_biases.data_ptr<float>();
	deform_task.packed_tail_weights = reinterpret_cast<const void*>(
		packed_tail_weights.data_ptr<at::Half>());
	deform_task.tail_biases = tail_biases.data_ptr<float>();
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
		deform_task.outputs[output] = deform_outputs[output].data_ptr<float>();
	deform_task.rows = static_cast<int>(deform_features.size(0));
	deform_task.persistent_blocks = static_cast<int>(persistent_blocks);

	const int P = static_cast<int>(radii.size(0));
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	if (P != 0)
	{
		CudaRasterizer::Rasterizer::
			renderPreparedDeformMlpTensorTailRasterOptimizedOverlap(
				P,
				static_cast<int>(num_rendered),
				background.data_ptr<float>(),
				image_width, image_height,
				isPlaceholder(colors) ? nullptr : colors.data_ptr<float>(),
				reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>()),
				out_color.data_ptr<float>(),
				out_depth.data_ptr<float>(),
				debug,
				stream,
				deform_task);
	}
	else
	{
		CudaRasterizer::Tacker::
			launchMixedRenderDeformMlpTensorTailRasterOptimized(
				dim3(0, 0, 1), nullptr, nullptr,
				image_width, image_height,
				nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
				background.data_ptr<float>(),
				out_color.data_ptr<float>(),
				out_depth.data_ptr<float>(),
				deform_task,
				stream);
	}

	if (P != 0 || deform_task.rows > 0)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"prepared mixed Raster-optimized Tensor-tail launch failed: ",
			cudaGetErrorString(status));
	}
	return deform_outputs;
}

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
	const int64_t persistent_blocks)
{
	checkPreparedRasterArguments(
		background, colors, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer,
		num_rendered, image_height, image_width);
	checkMixedDeformMlpTensorTailArguments(
		radii, deform_features, backbone_weight, backbone_bias,
		first_weights, first_biases, packed_tail_weights, tail_biases,
		persistent_blocks);

	const c10::cuda::CUDAGuard device_guard(radii.device());
	constexpr int output_widths[CudaRasterizer::kDeformMlpOutputCountV1] = {
		3, 3, 4, 1, 48};
	std::vector<torch::Tensor> deform_outputs;
	deform_outputs.reserve(CudaRasterizer::kDeformMlpOutputCountV1);
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
	{
		deform_outputs.push_back(torch::empty(
			{deform_features.size(0), output_widths[output]},
			deform_features.options().dtype(torch::kFloat32).requires_grad(false)));
	}

	CudaRasterizer::MixedDeformMlpTensorTailTaskV2 deform_task = {};
	deform_task.features = deform_features.data_ptr<float>();
	deform_task.backbone_weight = backbone_weight.data_ptr<float>();
	deform_task.backbone_bias = backbone_bias.data_ptr<float>();
	deform_task.first_weights = reinterpret_cast<const void*>(
		first_weights.data_ptr<at::Half>());
	deform_task.first_biases = first_biases.data_ptr<float>();
	deform_task.packed_tail_weights = reinterpret_cast<const void*>(
		packed_tail_weights.data_ptr<at::Half>());
	deform_task.tail_biases = tail_biases.data_ptr<float>();
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
		deform_task.outputs[output] = deform_outputs[output].data_ptr<float>();
	deform_task.rows = static_cast<int>(deform_features.size(0));
	deform_task.persistent_blocks = static_cast<int>(persistent_blocks);

	const int P = static_cast<int>(radii.size(0));
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	if (P != 0)
	{
		CudaRasterizer::Rasterizer::
			renderPreparedDeformMlpTensorTailRasterFastOverlap(
				P,
				static_cast<int>(num_rendered),
				background.data_ptr<float>(),
				image_width, image_height,
				isPlaceholder(colors) ? nullptr : colors.data_ptr<float>(),
				reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>()),
				out_color.data_ptr<float>(),
				out_depth.data_ptr<float>(),
				debug,
				stream,
				deform_task);
	}
	else
	{
		CudaRasterizer::Tacker::launchMixedRenderDeformMlpTensorTailRasterFast(
			dim3(0, 0, 1), nullptr, nullptr,
			image_width, image_height,
			nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
			background.data_ptr<float>(),
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			deform_task,
			stream);
	}

	if (P != 0 || deform_task.rows > 0)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"prepared mixed fast-Raster Tensor-tail launch failed: ",
			cudaGetErrorString(status));
	}
	return deform_outputs;
}

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
	const int64_t raster_end)
{
	checkPreparedRasterArguments(
		background, colors, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer,
		num_rendered, image_height, image_width);
	const int64_t raster_blocks =
		static_cast<int64_t>((image_width + 15) / 16) *
		static_cast<int64_t>((image_height + 15) / 16);
	TORCH_CHECK(
		raster_start >= 0 && raster_start <= raster_end &&
			raster_end <= raster_blocks,
		"split Raster range must satisfy 0 <= start <= end <= ",
		raster_blocks);
	TORCH_CHECK(
		raster_end <= INT_MAX,
		"split Raster tile range exceeds int32 ABI-v12");
	const int P = static_cast<int>(radii.size(0));
	if (P == 0 || raster_start == raster_end)
		return;

	const c10::cuda::CUDAGuard device_guard(radii.device());
	CudaRasterizer::Rasterizer::renderPreparedRasterFastRange(
		P,
		static_cast<int>(num_rendered),
		background.data_ptr<float>(),
		image_width, image_height,
		isPlaceholder(colors) ? nullptr : colors.data_ptr<float>(),
		reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>()),
		reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>()),
		reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>()),
		out_color.data_ptr<float>(),
		out_depth.data_ptr<float>(),
		debug,
		at::cuda::getCurrentCUDAStream().stream(),
		static_cast<int>(raster_start),
		static_cast<int>(raster_end));
	const cudaError_t status = cudaGetLastError();
	TORCH_CHECK(
		status == cudaSuccess,
		"prepared split Raster prefix launch failed: ",
		cudaGetErrorString(status));
}

static void RenderPreparedGaussiansWithHexPlaneSamplingImpl(
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
    int64_t hex_tasks_per_cta, bool block_fusion)
{
    checkPreparedRasterArguments(
        background, colors, out_color, out_depth, radii,
        geom_buffer, binning_buffer, image_buffer,
        num_rendered, image_height, image_width);
    const int grid_x = (image_width + BLOCK_X - 1) / BLOCK_X;
    const int grid_y = (image_height + BLOCK_Y - 1) / BLOCK_Y;
    TORCH_CHECK(raster_start >= 0 && raster_start <= raster_end &&
                raster_end <= int64_t(grid_x) * grid_y,
                "Raster range must satisfy 0 <= start <= end <= tile count");
    TORCH_CHECK(hex_tasks_per_cta > 0 && hex_tasks_per_cta <= INT_MAX,
                "hex_tasks_per_cta must be positive and int32-safe");
    TORCH_CHECK(!block_fusion || hex_tasks_per_cta == 1,
                "block fusion requires exactly one sampling task per CTA");
    TORCH_CHECK(coords.is_cuda() && coords.device() == radii.device() &&
                coords.scalar_type() == at::kFloat && coords.is_contiguous() &&
                coords.dim() == 2 && coords.size(1) == 4 &&
                coords.size(0) <= INT_MAX / 16,
                "coords must be contiguous CUDA FP32 [N,4] on Raster device "
                "with int32-safe element indexing");
    TORCH_CHECK(!planes.empty() &&
                planes.size() <= CudaRasterizer::Tacker::kMaxMixedHexPlanes &&
                axes.size() == planes.size() && outputs.size() == planes.size(),
                "provide 1..12 planes, one axis pair and one output per plane");

    CudaRasterizer::Tacker::RasterHexPlaneSampleArgs hex = {};
    hex.coords = coords.data_ptr<float>();
    hex.rows = int(coords.size(0));
    hex.plane_count = int(planes.size());
    hex.blocks_per_plane = (hex.rows + 255) / 256;
    hex.tasks_per_cta = int(hex_tasks_per_cta);
    for (size_t i = 0; i < planes.size(); ++i) {
        const auto& plane = planes[i];
        const auto& output = outputs[i];
        TORCH_CHECK(plane.is_cuda() && plane.device() == radii.device() &&
                    plane.scalar_type() == at::kFloat && plane.is_contiguous() &&
                    plane.dim() == 4 && plane.size(0) == 1 && plane.size(1) == 16 &&
                    plane.size(2) > 0 && plane.size(3) > 0 &&
                    plane.size(2) <= (1 << 24) && plane.size(3) <= (1 << 24) &&
                    plane.numel() <= INT_MAX,
                    "each plane must be contiguous CUDA FP32 [1,16,H,W] "
                    "on Raster device with int32-safe indexing");
        TORCH_CHECK(axes[i].size() == 2 && axes[i][0] >= 0 && axes[i][0] < 4 &&
                    axes[i][1] >= 0 && axes[i][1] < 4 && axes[i][0] != axes[i][1],
                    "each axis pair must contain two distinct values in [0,4)");
        TORCH_CHECK(output.is_cuda() && output.device() == radii.device() &&
                    output.scalar_type() == at::kFloat && output.dim() == 2 &&
                    output.size(0) == hex.rows && output.size(1) == 16 &&
                    (hex.rows == 0 ||
                     (output.stride(0) == 1 && output.stride(1) == hex.rows)),
                    "HexPlane output must be FP32 [N,16] with strides (1,N); "
                    "allocate with empty(16,N).t()");
        hex.planes[i] = {plane.data_ptr<float>(), output.data_ptr<float>(),
                         int(plane.size(2)), int(plane.size(3)),
                         int(axes[i][0]), int(axes[i][1])};
    }

    // Do not let either subgroup overwrite anything read by the other half.
    // Overlap checks also cover distinct tensor views of a shared allocation.
    std::vector<const torch::Tensor*> inputs = {
        &background, &colors, &radii, &geom_buffer, &binning_buffer,
        &image_buffer, &coords};
    for (const auto& plane : planes) inputs.push_back(&plane);
    std::vector<const torch::Tensor*> writes = {&out_color, &out_depth};
    for (const auto& output : outputs) writes.push_back(&output);
    for (size_t i = 0; i < writes.size(); ++i) {
        for (const auto* input : inputs)
            TORCH_CHECK(!tensorByteRangesOverlap(*writes[i], *input),
                        "mixed Raster/HexPlane outputs must not overlap inputs");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(!tensorByteRangesOverlap(*writes[i], *writes[j]),
                        "mixed Raster/HexPlane outputs must not overlap each other");
    }
    // ImageState contains mutable final_T and n_contrib as well as ranges.
    for (const auto* input : {&background, &colors, &radii, &geom_buffer,
                              &binning_buffer, &coords})
        TORCH_CHECK(!tensorByteRangesOverlap(image_buffer, *input),
                    "mutable ImageState must not overlap mixed inputs");
    for (const auto& plane : planes)
        TORCH_CHECK(!tensorByteRangesOverlap(image_buffer, plane),
                    "mutable ImageState must not overlap HexPlane planes");

    const c10::cuda::CUDAGuard device_guard(radii.device());
    CudaRasterizer::Tacker::RasterHexPlaneRasterArgs raster = {};
    raster.width = image_width;
    raster.height = image_height;
    raster.grid_x = grid_x;
    raster.grid_y = grid_y;
    raster.start = int(raster_start);
    // Empty Gaussian scenes keep prepare's initialized image, but still
    // execute all requested HexPlane tasks (including on a different N).
    raster.end = radii.numel() == 0 ? raster.start : int(raster_end);
    if (raster.start < raster.end) {
        char* geom_chunk = reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>());
        char* binning_chunk = reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>());
        char* image_chunk = reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>());
        const auto geom = CudaRasterizer::GeometryState::fromChunk(
            geom_chunk, radii.numel());
        const auto binning = CudaRasterizer::BinningState::fromChunk(
            binning_chunk, num_rendered);
        const auto image = CudaRasterizer::ImageState::fromChunk(
            image_chunk, int64_t(image_width) * image_height);
        raster.ranges = image.ranges;
        raster.point_list = binning.point_list;
        raster.points_xy = geom.means2D;
        raster.features = isPlaceholder(colors) ? geom.rgb : colors.data_ptr<float>();
        raster.depths = geom.depths;
        raster.conic_opacity = geom.conic_opacity;
        raster.background = background.data_ptr<float>();
        raster.final_T = image.accum_alpha;
        raster.n_contrib = image.n_contrib;
        raster.color = out_color.data_ptr<float>();
        raster.depth = out_depth.data_ptr<float>();
    }
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto status = block_fusion
        ? CudaRasterizer::Tacker::launchRasterHexPlaneBlocks(raster, hex, stream)
        : CudaRasterizer::Tacker::launchRasterHexPlaneMixed(raster, hex, stream);
    TORCH_CHECK(status == cudaSuccess,
                "Raster/HexPlane mixed launch failed: ",
                cudaGetErrorString(status));
}

void RenderPreparedGaussiansWithHexPlaneSamplingCUDA(
    const torch::Tensor& background, const torch::Tensor& colors,
    const torch::Tensor& out_color, const torch::Tensor& out_depth,
    const torch::Tensor& radii, const torch::Tensor& geom_buffer,
    const torch::Tensor& binning_buffer, const torch::Tensor& image_buffer,
    int64_t num_rendered, int image_height, int image_width,
    int64_t raster_start, int64_t raster_end, const torch::Tensor& coords,
    const std::vector<torch::Tensor>& planes,
    const std::vector<std::vector<int64_t>>& axes,
    const std::vector<torch::Tensor>& outputs, int64_t hex_tasks_per_cta)
{
    RenderPreparedGaussiansWithHexPlaneSamplingImpl(
        background, colors, out_color, out_depth, radii, geom_buffer,
        binning_buffer, image_buffer, num_rendered, image_height, image_width,
        raster_start, raster_end, coords, planes, axes, outputs, hex_tasks_per_cta, false);
}

void RenderPreparedGaussiansWithHexPlaneBlocksCUDA(
    const torch::Tensor& background, const torch::Tensor& colors,
    const torch::Tensor& out_color, const torch::Tensor& out_depth,
    const torch::Tensor& radii, const torch::Tensor& geom_buffer,
    const torch::Tensor& binning_buffer, const torch::Tensor& image_buffer,
    int64_t num_rendered, int image_height, int image_width,
    int64_t raster_start, int64_t raster_end, const torch::Tensor& coords,
    const std::vector<torch::Tensor>& planes,
    const std::vector<std::vector<int64_t>>& axes,
    const std::vector<torch::Tensor>& outputs, int64_t hex_tasks_per_cta)
{
    RenderPreparedGaussiansWithHexPlaneSamplingImpl(
        background, colors, out_color, out_depth, radii, geom_buffer,
        binning_buffer, image_buffer, num_rendered, image_height, image_width,
        raster_start, raster_end, coords, planes, axes, outputs, hex_tasks_per_cta, true);
}

std::vector<int64_t> TackerRasterHexPlaneResourcesCUDA()
{
    cudaFuncAttributes attributes;
    int active_blocks = 0;
    const auto status = CudaRasterizer::Tacker::queryRasterHexPlaneMixedResources(
        &attributes, &active_blocks);
    TORCH_CHECK(status == cudaSuccess, "mixed resource query failed: ",
                cudaGetErrorString(status));
    return {512, attributes.numRegs, int64_t(attributes.localSizeBytes),
            int64_t(attributes.sharedSizeBytes), active_blocks,
            attributes.binaryVersion};
}

std::vector<int64_t> TackerRasterHexPlaneBlockResourcesCUDA()
{
    cudaFuncAttributes attributes;
    int active_blocks = 0;
    const auto status = CudaRasterizer::Tacker::queryRasterHexPlaneBlocksResources(
        &attributes, &active_blocks);
    TORCH_CHECK(status == cudaSuccess, "block mixed resource query failed: ",
                cudaGetErrorString(status));
    return {256, attributes.numRegs, int64_t(attributes.localSizeBytes),
            int64_t(attributes.sharedSizeBytes), active_blocks, attributes.binaryVersion};
}

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
	const int64_t raster_start)
{
	checkPreparedRasterArguments(
		background, colors, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer,
		num_rendered, image_height, image_width);
	checkMixedDeformMlpTensorTailArguments(
		radii, deform_features, backbone_weight, backbone_bias,
		first_weights, first_biases, packed_tail_weights, tail_biases,
		persistent_blocks);
	const int64_t raster_blocks =
		static_cast<int64_t>((image_width + 15) / 16) *
		static_cast<int64_t>((image_height + 15) / 16);
	TORCH_CHECK(
		raster_start >= 0 && raster_start <= raster_blocks,
		"split mixed Raster start must be within [0, ", raster_blocks, "]");
	TORCH_CHECK(
		raster_start <= INT_MAX,
		"split mixed Raster start exceeds int32 ABI-v12");

	const c10::cuda::CUDAGuard device_guard(radii.device());
	constexpr int output_widths[CudaRasterizer::kDeformMlpOutputCountV1] = {
		3, 3, 4, 1, 48};
	std::vector<torch::Tensor> deform_outputs;
	deform_outputs.reserve(CudaRasterizer::kDeformMlpOutputCountV1);
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
	{
		deform_outputs.push_back(torch::empty(
			{deform_features.size(0), output_widths[output]},
			deform_features.options().dtype(torch::kFloat32).requires_grad(false)));
	}

	CudaRasterizer::MixedDeformMlpTensorTailTaskV2 deform_task = {};
	deform_task.features = deform_features.data_ptr<float>();
	deform_task.backbone_weight = backbone_weight.data_ptr<float>();
	deform_task.backbone_bias = backbone_bias.data_ptr<float>();
	deform_task.first_weights = reinterpret_cast<const void*>(
		first_weights.data_ptr<at::Half>());
	deform_task.first_biases = first_biases.data_ptr<float>();
	deform_task.packed_tail_weights = reinterpret_cast<const void*>(
		packed_tail_weights.data_ptr<at::Half>());
	deform_task.tail_biases = tail_biases.data_ptr<float>();
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
		deform_task.outputs[output] = deform_outputs[output].data_ptr<float>();
	deform_task.rows = static_cast<int>(deform_features.size(0));
	deform_task.persistent_blocks = static_cast<int>(persistent_blocks);

	const int P = static_cast<int>(radii.size(0));
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	if (P != 0)
	{
		CudaRasterizer::Rasterizer::
			renderPreparedDeformMlpTensorTailSplitRasterOverlap(
				P,
				static_cast<int>(num_rendered),
				background.data_ptr<float>(),
				image_width, image_height,
				isPlaceholder(colors) ? nullptr : colors.data_ptr<float>(),
				reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>()),
				out_color.data_ptr<float>(),
				out_depth.data_ptr<float>(),
				debug,
				stream,
				static_cast<int>(raster_start),
				deform_task);
	}
	else
	{
		CudaRasterizer::Tacker::
			launchMixedRenderDeformMlpTensorTailSplitRaster(
				dim3(0, 0, 1), nullptr, nullptr,
				image_width, image_height,
				nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
				background.data_ptr<float>(),
				out_color.data_ptr<float>(),
				out_depth.data_ptr<float>(),
				0,
				deform_task,
				stream);
	}

	if (P != 0 || deform_task.rows > 0)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"prepared split Raster Tensor-tail launch failed: ",
			cudaGetErrorString(status));
	}
	return deform_outputs;
}

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
	const int64_t diagnostic_mode)
{
	checkPreparedRasterArguments(
		background, colors, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer,
		num_rendered, image_height, image_width);
	checkMixedDeformMlpTensorTailArguments(
		radii, deform_features, backbone_weight, backbone_bias,
		first_weights, first_biases, packed_tail_weights, tail_biases,
		persistent_blocks);
	TORCH_CHECK(
		abi_version == 10 || abi_version == 11,
		"Raster diagnostic abi_version must be 10 or 11");
	TORCH_CHECK(
		diagnostic_mode >= 0 && diagnostic_mode <= 2,
		"diagnostic_mode must be 0 (both), 1 (Raster only), or 2 (Deform only)");

	const c10::cuda::CUDAGuard device_guard(radii.device());
	constexpr int output_widths[CudaRasterizer::kDeformMlpOutputCountV1] = {
		3, 3, 4, 1, 48};
	std::vector<torch::Tensor> results;
	results.reserve(CudaRasterizer::kDeformMlpOutputCountV1 + 1);
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
	{
		results.push_back(torch::empty(
			{deform_features.size(0), output_widths[output]},
			deform_features.options().dtype(torch::kFloat32).requires_grad(false)));
	}

	CudaRasterizer::MixedDeformMlpTensorTailTaskV2 deform_task = {};
	deform_task.features = deform_features.data_ptr<float>();
	deform_task.backbone_weight = backbone_weight.data_ptr<float>();
	deform_task.backbone_bias = backbone_bias.data_ptr<float>();
	deform_task.first_weights = reinterpret_cast<const void*>(
		first_weights.data_ptr<at::Half>());
	deform_task.first_biases = first_biases.data_ptr<float>();
	deform_task.packed_tail_weights = reinterpret_cast<const void*>(
		packed_tail_weights.data_ptr<at::Half>());
	deform_task.tail_biases = tail_biases.data_ptr<float>();
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
		deform_task.outputs[output] = results[output].data_ptr<float>();
	deform_task.rows = static_cast<int>(deform_features.size(0));
	deform_task.persistent_blocks = static_cast<int>(persistent_blocks);

	const int64_t raster_blocks = radii.size(0) == 0 ? 0 :
		static_cast<int64_t>((image_width + BLOCK_X - 1) / BLOCK_X) *
		((image_height + BLOCK_Y - 1) / BLOCK_Y);
	const int64_t deform_tiles =
		(deform_features.size(0) + CudaRasterizer::Tacker::kDeformRowsPerTile - 1) /
		CudaRasterizer::Tacker::kDeformRowsPerTile;
	// Match the production ABI-v10/v11 default launch exactly: one physical
	// CTA per Raster tile, with extra Deform tiles distributed as a second pass.
	const int64_t logical_blocks = raster_blocks > 0 ? raster_blocks : deform_tiles;
	const int64_t physical_blocks = logical_blocks == 0 ? 0 :
		(persistent_blocks == 0
			? logical_blocks : std::min(persistent_blocks, logical_blocks));
	TORCH_CHECK(
		physical_blocks <= INT_MAX,
		"Raster diagnostic physical block count exceeds int32 indexing");
	torch::Tensor timing_cycles = torch::empty(
		{physical_blocks, 4},
		radii.options().dtype(torch::kInt64).requires_grad(false));
	results.push_back(timing_cycles);

	const int P = static_cast<int>(radii.size(0));
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	if (P != 0)
	{
		CudaRasterizer::Rasterizer::
			renderPreparedDeformMlpTensorTailRasterDiagnostic(
				P,
				static_cast<int>(num_rendered),
				background.data_ptr<float>(),
				image_width, image_height,
				isPlaceholder(colors) ? nullptr : colors.data_ptr<float>(),
				reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>()),
				out_color.data_ptr<float>(),
				out_depth.data_ptr<float>(),
				debug,
				stream,
				deform_task,
				static_cast<int>(abi_version),
				static_cast<int>(diagnostic_mode),
				timing_cycles.data_ptr<int64_t>(),
				static_cast<int>(physical_blocks));
	}
	else
	{
		CudaRasterizer::Tacker::
			launchMixedRenderDeformMlpTensorTailRasterDiagnostic(
				dim3(0, 0, 1), nullptr, nullptr,
				image_width, image_height,
				nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
				background.data_ptr<float>(),
				out_color.data_ptr<float>(),
				out_depth.data_ptr<float>(),
				deform_task,
				static_cast<int>(abi_version),
				static_cast<int>(diagnostic_mode),
				timing_cycles.data_ptr<int64_t>(),
				static_cast<int>(physical_blocks),
				stream);
	}

	if (P != 0 || deform_task.rows > 0)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"prepared mixed Raster diagnostic launch failed: ",
			cudaGetErrorString(status));
	}
	return results;
}

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
	const int64_t diagnostic_mode)
{
	checkPreparedRasterArguments(
		background, colors, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer,
		num_rendered, image_height, image_width);
	checkMixedDeformMlpTensorTailArguments(
		radii, deform_features, backbone_weight, backbone_bias,
		first_weights, first_biases, packed_tail_weights, tail_biases,
		persistent_blocks);
	TORCH_CHECK(
		diagnostic_mode >= 0 && diagnostic_mode <= 2,
		"diagnostic_mode must be 0 (both), 1 (Raster only), or 2 (Deform only)");

	const c10::cuda::CUDAGuard device_guard(radii.device());
	constexpr int output_widths[CudaRasterizer::kDeformMlpOutputCountV1] = {
		3, 3, 4, 1, 48};
	std::vector<torch::Tensor> results;
	results.reserve(CudaRasterizer::kDeformMlpOutputCountV1 + 1);
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
	{
		results.push_back(torch::empty(
			{deform_features.size(0), output_widths[output]},
			deform_features.options().dtype(torch::kFloat32).requires_grad(false)));
	}

	CudaRasterizer::MixedDeformMlpTensorTailTaskV2 deform_task = {};
	deform_task.features = deform_features.data_ptr<float>();
	deform_task.backbone_weight = backbone_weight.data_ptr<float>();
	deform_task.backbone_bias = backbone_bias.data_ptr<float>();
	deform_task.first_weights = reinterpret_cast<const void*>(
		first_weights.data_ptr<at::Half>());
	deform_task.first_biases = first_biases.data_ptr<float>();
	deform_task.packed_tail_weights = reinterpret_cast<const void*>(
		packed_tail_weights.data_ptr<at::Half>());
	deform_task.tail_biases = tail_biases.data_ptr<float>();
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
		deform_task.outputs[output] = results[output].data_ptr<float>();
	deform_task.rows = static_cast<int>(deform_features.size(0));
	deform_task.persistent_blocks = static_cast<int>(persistent_blocks);

	const int64_t raster_blocks = radii.size(0) == 0 ? 0 :
		static_cast<int64_t>((image_width + BLOCK_X - 1) / BLOCK_X) *
		((image_height + BLOCK_Y - 1) / BLOCK_Y);
	const int64_t deform_tiles =
		(deform_features.size(0) + CudaRasterizer::Tacker::kDeformRowsPerTile - 1) /
		CudaRasterizer::Tacker::kDeformRowsPerTile;
	const int64_t logical_blocks = std::max(raster_blocks, deform_tiles);
	const int64_t physical_blocks = logical_blocks == 0 ? 0 :
		(persistent_blocks == 0
			? logical_blocks : std::min(persistent_blocks, logical_blocks));
	TORCH_CHECK(
		physical_blocks <= INT_MAX,
		"diagnostic physical block count exceeds int32 indexing");
	torch::Tensor timing_cycles = torch::empty(
		{physical_blocks, 4},
		radii.options().dtype(torch::kInt64).requires_grad(false));
	results.push_back(timing_cycles);

	const int P = static_cast<int>(radii.size(0));
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	if (P != 0)
	{
		CudaRasterizer::Rasterizer::
			renderPreparedDeformMlpTensorTailFourCtaDiagnostic(
				P,
				static_cast<int>(num_rendered),
				background.data_ptr<float>(),
				image_width, image_height,
				isPlaceholder(colors) ? nullptr : colors.data_ptr<float>(),
				reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>()),
				reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>()),
				out_color.data_ptr<float>(),
				out_depth.data_ptr<float>(),
				debug,
				stream,
				deform_task,
				static_cast<int>(diagnostic_mode),
				timing_cycles.data_ptr<int64_t>(),
				static_cast<int>(physical_blocks));
	}
	else
	{
		CudaRasterizer::Tacker::
			launchMixedRenderDeformMlpTensorTailFourCtaDiagnostic(
				dim3(0, 0, 1), nullptr, nullptr,
				image_width, image_height,
				nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
				background.data_ptr<float>(),
				out_color.data_ptr<float>(),
				out_depth.data_ptr<float>(),
				deform_task,
				static_cast<int>(diagnostic_mode),
				timing_cycles.data_ptr<int64_t>(),
				static_cast<int>(physical_blocks),
				stream);
	}

	if (P != 0 || deform_task.rows > 0)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"prepared mixed four-CTA diagnostic launch failed: ",
			cudaGetErrorString(status));
	}
	return results;
}

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
	const bool debug)
{
	checkPreparedRasterArguments(
		background, colors, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer,
		num_rendered, image_height, image_width);
	const int P = static_cast<int>(radii.size(0));
	if (P == 0)
		return;

	const c10::cuda::CUDAGuard device_guard(radii.device());
	CudaRasterizer::Rasterizer::renderPreparedDeformMlpOverlap(
		P,
		static_cast<int>(num_rendered),
		background.data_ptr<float>(),
		image_width, image_height,
		isPlaceholder(colors) ? nullptr : colors.data_ptr<float>(),
		reinterpret_cast<char*>(geom_buffer.data_ptr<uint8_t>()),
		reinterpret_cast<char*>(binning_buffer.data_ptr<uint8_t>()),
		reinterpret_cast<char*>(image_buffer.data_ptr<uint8_t>()),
		out_color.data_ptr<float>(),
		out_depth.data_ptr<float>(),
		debug,
		at::cuda::getCurrentCUDAStream().stream(),
		nullptr);
	const cudaError_t status = cudaGetLastError();
	TORCH_CHECK(
		status == cudaSuccess,
		"prepared Raster launch failed: ", cudaGetErrorString(status));
}

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
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

	TORCH_CHECK(means3D.is_cuda(), "means3D must be a CUDA tensor");
	const c10::cuda::CUDAGuard device_guard(means3D.device());
	checkInt32RowCount("means3D", means3D);
	checkRasterDimensions(image_height, image_width);
  
	const int P = static_cast<int>(means3D.size(0));
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
	int rendered = 0;
	if(P != 0)
	{
	  const int M = legacyShCoefficientCount(sh);
	  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_depth.contiguous().data<float>(),
		radii.contiguous().data<int>(),
			debug,
			stream);
  }
  return std::make_tuple(rendered, out_color, out_depth, radii, geomBuffer, binningBuffer, imgBuffer);
}

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
	const int64_t persistent_blocks)
{
	const MixedRasterChoices raster_choices = checkMixedRasterArguments(
		background,
		means3D,
		colors,
		opacity,
		scales,
		rotations,
		cov3D_precomp,
		viewmatrix,
		projmatrix,
		sh,
		campos,
		degree);
	checkMixedHeadArguments(
		means3D, head_input, head_weight, head_bias, persistent_blocks);
	checkRasterDimensions(image_height, image_width);

	const c10::cuda::CUDAGuard device_guard(means3D.device());
	const int P = static_cast<int>(means3D.size(0));
	const int H = image_height;
	const int W = image_width;
	const int head_rows = static_cast<int>(head_input.size(0));

	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full(
		{P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor head_output = torch::empty(
		{head_input.size(0), 128}, head_input.options().dtype(torch::kFloat32));
	TORCH_CHECK(
		isNativeWmmaAligned(head_output),
		"head_output data pointer must be natively 32-byte aligned for WMMA");

	const torch::TensorOptions byte_options =
		means3D.options().dtype(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, byte_options);
	torch::Tensor binningBuffer = torch::empty({0}, byte_options);
	torch::Tensor imgBuffer = torch::empty({0}, byte_options);
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	CudaRasterizer::MixedHeadTask head_task = {
		reinterpret_cast<const void*>(head_input.data_ptr<at::Half>()),
		reinterpret_cast<const void*>(head_weight.data_ptr<at::Half>()),
		head_bias.data_ptr<float>(),
		head_output.data_ptr<float>(),
		head_rows,
		static_cast<int>(persistent_blocks)};

	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	const torch::Tensor background_contiguous = background.contiguous();
	int rendered = 0;
	if (P != 0)
	{
		const int M = raster_choices.use_sh ? static_cast<int>(sh.size(1)) : 0;

		// Keep any materialized contiguous views alive through launch and pass
		// literal null pointers for omitted alternatives.  The rasterizer uses
		// pointer nullness to choose SH/colors and covariance/scale-rotation.
		const torch::Tensor means3D_contiguous = means3D.contiguous();
		const torch::Tensor sh_contiguous = sh.contiguous();
		const torch::Tensor colors_contiguous = colors.contiguous();
		const torch::Tensor opacity_contiguous = opacity.contiguous();
		const torch::Tensor scales_contiguous = scales.contiguous();
		const torch::Tensor rotations_contiguous = rotations.contiguous();
		const torch::Tensor cov3D_contiguous = cov3D_precomp.contiguous();
		const torch::Tensor viewmatrix_contiguous = viewmatrix.contiguous();
		const torch::Tensor projmatrix_contiguous = projmatrix.contiguous();
		const torch::Tensor campos_contiguous = campos.contiguous();
		const float* sh_ptr = raster_choices.use_sh
			? sh_contiguous.data_ptr<float>()
			: nullptr;
		const float* colors_ptr = raster_choices.use_sh
			? nullptr
			: colors_contiguous.data_ptr<float>();
		const float* scales_ptr = raster_choices.use_cov3d
			? nullptr
			: scales_contiguous.data_ptr<float>();
		const float* rotations_ptr = raster_choices.use_cov3d
			? nullptr
			: rotations_contiguous.data_ptr<float>();
		const float* cov3D_ptr = raster_choices.use_cov3d
			? cov3D_contiguous.data_ptr<float>()
			: nullptr;

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background_contiguous.data_ptr<float>(),
			W, H,
			means3D_contiguous.data_ptr<float>(),
			sh_ptr,
			colors_ptr,
			opacity_contiguous.data_ptr<float>(),
			scales_ptr,
			scale_modifier,
			rotations_ptr,
			cov3D_ptr,
			viewmatrix_contiguous.data_ptr<float>(),
			projmatrix_contiguous.data_ptr<float>(),
			campos_contiguous.data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			out_color.contiguous().data<float>(),
			out_depth.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			debug,
			stream,
			&head_task);
	}
	else
	{
		// Preserve the legacy P==0 raster outputs/buffers while still running
		// the independent head task exactly once when N>0.
		CudaRasterizer::Tacker::launchMixedRenderHead(
			dim3(0, 0, 1),
			nullptr,
			nullptr,
			W, H,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			background_contiguous.data_ptr<float>(),
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			head_task,
			stream);
	}

	if (P != 0 || head_rows != 0)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"rasterize_gaussians_with_head launch failed: ",
			cudaGetErrorString(status));
	}

	return std::make_tuple(
		rendered,
		out_color,
		out_depth,
		radii,
		geomBuffer,
		binningBuffer,
		imgBuffer,
		head_output);
}

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
	const int64_t persistent_blocks)
{
	const MixedRasterChoices raster_choices = checkMixedRasterArguments(
		background,
		means3D,
		colors,
		opacity,
		scales,
		rotations,
		cov3D_precomp,
		viewmatrix,
		projmatrix,
		sh,
		campos,
		degree);
	checkMixedHeadsArguments(
		means3D,
		head_inputs,
		head_weights,
		head_biases,
		worker_groups,
		persistent_blocks);
	checkRasterDimensions(image_height, image_width);

	const c10::cuda::CUDAGuard device_guard(means3D.device());
	const int P = static_cast<int>(means3D.size(0));
	const int H = image_height;
	const int W = image_width;

	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full(
		{P}, 0, means3D.options().dtype(torch::kInt32));
	std::vector<torch::Tensor> head_outputs;
	head_outputs.reserve(head_inputs.size());
	for (std::size_t task_index = 0; task_index < head_inputs.size(); ++task_index)
	{
		head_outputs.push_back(torch::empty(
			{head_inputs[task_index].size(0), 128},
			head_inputs[task_index].options().dtype(torch::kFloat32)));
		TORCH_CHECK(
			isNativeWmmaAligned(head_outputs.back()),
			"head_outputs[", task_index,
			"] data pointer must be natively 32-byte aligned for WMMA");
		if (head_outputs.back().numel() != 0)
		{
			for (std::size_t other = 0; other < task_index; ++other)
			{
				TORCH_CHECK(
					head_outputs[other].numel() == 0 ||
						head_outputs[other].data_ptr() !=
							head_outputs.back().data_ptr(),
					"mixed ABI v2 head outputs must not alias each other");
			}
		}
	}

	const torch::TensorOptions byte_options =
		means3D.options().dtype(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, byte_options);
	torch::Tensor binningBuffer = torch::empty({0}, byte_options);
	torch::Tensor imgBuffer = torch::empty({0}, byte_options);
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	CudaRasterizer::MixedHeadBundleV2 head_bundle = {};
	head_bundle.task_count = static_cast<int>(head_inputs.size());
	head_bundle.worker_groups = static_cast<int>(worker_groups);
	head_bundle.persistent_blocks = static_cast<int>(persistent_blocks);
	bool any_head_rows = false;
	for (std::size_t task_index = 0; task_index < head_inputs.size(); ++task_index)
	{
		const int rows = static_cast<int>(head_inputs[task_index].size(0));
		head_bundle.tasks[task_index] = {
			reinterpret_cast<const void*>(
				head_inputs[task_index].data_ptr<at::Half>()),
			reinterpret_cast<const void*>(
				head_weights[task_index].data_ptr<at::Half>()),
			head_biases[task_index].data_ptr<float>(),
			head_outputs[task_index].data_ptr<float>(),
			rows};
		any_head_rows = any_head_rows || rows != 0;
	}

	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	const torch::Tensor background_contiguous = background.contiguous();
	int rendered = 0;
	if (P != 0)
	{
		const int M = raster_choices.use_sh ? static_cast<int>(sh.size(1)) : 0;

		// Materialized Raster views and the three task vectors stay alive until
		// after the single v2 mixed kernel has been enqueued on this stream.
		const torch::Tensor means3D_contiguous = means3D.contiguous();
		const torch::Tensor sh_contiguous = sh.contiguous();
		const torch::Tensor colors_contiguous = colors.contiguous();
		const torch::Tensor opacity_contiguous = opacity.contiguous();
		const torch::Tensor scales_contiguous = scales.contiguous();
		const torch::Tensor rotations_contiguous = rotations.contiguous();
		const torch::Tensor cov3D_contiguous = cov3D_precomp.contiguous();
		const torch::Tensor viewmatrix_contiguous = viewmatrix.contiguous();
		const torch::Tensor projmatrix_contiguous = projmatrix.contiguous();
		const torch::Tensor campos_contiguous = campos.contiguous();
		const float* sh_ptr = raster_choices.use_sh
			? sh_contiguous.data_ptr<float>()
			: nullptr;
		const float* colors_ptr = raster_choices.use_sh
			? nullptr
			: colors_contiguous.data_ptr<float>();
		const float* scales_ptr = raster_choices.use_cov3d
			? nullptr
			: scales_contiguous.data_ptr<float>();
		const float* rotations_ptr = raster_choices.use_cov3d
			? nullptr
			: rotations_contiguous.data_ptr<float>();
		const float* cov3D_ptr = raster_choices.use_cov3d
			? cov3D_contiguous.data_ptr<float>()
			: nullptr;

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background_contiguous.data_ptr<float>(),
			W, H,
			means3D_contiguous.data_ptr<float>(),
			sh_ptr,
			colors_ptr,
			opacity_contiguous.data_ptr<float>(),
			scales_ptr,
			scale_modifier,
			rotations_ptr,
			cov3D_ptr,
			viewmatrix_contiguous.data_ptr<float>(),
			projmatrix_contiguous.data_ptr<float>(),
			campos_contiguous.data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			out_color.contiguous().data<float>(),
			out_depth.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			debug,
			stream,
			nullptr,
			&head_bundle);
	}
	else
	{
		// Empty Raster input does not suppress independent head work.  Conversely,
		// all-zero-row heads plus an empty Raster result in no physical launch.
		CudaRasterizer::Tacker::launchMixedRenderHeads(
			dim3(0, 0, 1),
			nullptr,
			nullptr,
			W, H,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			background_contiguous.data_ptr<float>(),
			out_color.data_ptr<float>(),
			out_depth.data_ptr<float>(),
			head_bundle,
			stream);
	}

	if (P != 0 || any_head_rows)
	{
		const cudaError_t status = cudaGetLastError();
		TORCH_CHECK(
			status == cudaSuccess,
			"rasterize_gaussians_with_heads launch failed: ",
			cudaGetErrorString(status));
	}

	return std::make_tuple(
		rendered,
		out_color,
		out_depth,
		radii,
		geomBuffer,
		binningBuffer,
		imgBuffer,
		head_outputs);
}

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
	const int64_t persistent_blocks)
{
	const MixedRasterChoices raster_choices = checkMixedRasterArguments(
		background, means3D, colors, opacity, scales, rotations,
		cov3D_precomp, viewmatrix, projmatrix, sh, campos, degree);
	checkMixedPackedHeadsArguments(
		means3D, head_input, head_weights, head_biases,
		worker_groups, persistent_blocks);
	checkRasterDimensions(image_height, image_width);

	const c10::cuda::CUDAGuard device_guard(means3D.device());
	torch::Tensor head_output = torch::empty(
		{head_weights.size(0), head_input.size(0), 128},
		head_input.options().dtype(torch::kFloat32));
	TORCH_CHECK(
		isNativeWmmaAligned(head_output),
		"packed head_output data pointer must be natively 32-byte aligned");
	TORCH_CHECK(
		!tensorByteRangesOverlap(head_output, head_input) &&
			!tensorByteRangesOverlap(head_output, head_weights) &&
			!tensorByteRangesOverlap(head_output, head_biases),
		"packed head_output must not alias input, weights, or biases");

	CudaRasterizer::MixedPackedHeadBundleV2 head_bundle = {
		reinterpret_cast<const void*>(head_input.data_ptr<at::Half>()),
		reinterpret_cast<const void*>(head_weights.data_ptr<at::Half>()),
		head_biases.data_ptr<float>(),
		head_output.data_ptr<float>(),
		static_cast<int>(head_input.size(0)),
		static_cast<int>(head_weights.size(0)),
		static_cast<int>(worker_groups),
		static_cast<int>(persistent_blocks)};

	int rendered;
	torch::Tensor out_color;
	torch::Tensor out_depth;
	torch::Tensor radii;
	torch::Tensor geom_buffer;
	torch::Tensor binning_buffer;
	torch::Tensor image_buffer;
	std::tie(
		rendered, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer) =
		rasterizeWithNewMixedBackend(
			raster_choices,
			background, means3D, colors, opacity, scales, rotations,
			scale_modifier, cov3D_precomp, viewmatrix, projmatrix,
			tan_fovx, tan_fovy, image_height, image_width,
			sh, degree, campos, prefiltered, debug,
			&head_bundle, nullptr, nullptr);
	return std::make_tuple(
		rendered, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer, head_output);
}

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
	const int64_t persistent_blocks)
{
	const MixedRasterChoices raster_choices = checkMixedRasterArguments(
		background, means3D, colors, opacity, scales, rotations,
		cov3D_precomp, viewmatrix, projmatrix, sh, campos, degree);
	checkMixedWholeHeadsArguments(
		means3D, head_inputs, first_weights, first_biases,
		tail_weights, tail_biases, worker_groups, persistent_blocks);
	checkRasterDimensions(image_height, image_width);

	const c10::cuda::CUDAGuard device_guard(means3D.device());
	std::vector<torch::Tensor> head_outputs;
	head_outputs.reserve(head_inputs.size());
	for (std::size_t task_index = 0; task_index < head_inputs.size(); ++task_index)
	{
		head_outputs.push_back(torch::empty(
			{head_inputs[task_index].size(0), tail_weights[task_index].size(0)},
			head_inputs[task_index].options().dtype(torch::kFloat32)));
		TORCH_CHECK(
			isNativeWmmaAligned(head_outputs.back()),
			"whole-head outputs[", task_index,
			"] data pointer must be natively 32-byte aligned");
		for (std::size_t parameter_task = 0;
			 parameter_task < head_inputs.size(); ++parameter_task)
		{
			TORCH_CHECK(
				!tensorByteRangesOverlap(
					head_outputs.back(), head_inputs[parameter_task]) &&
				!tensorByteRangesOverlap(
					head_outputs.back(), first_weights[parameter_task]) &&
				!tensorByteRangesOverlap(
					head_outputs.back(), first_biases[parameter_task]) &&
				!tensorByteRangesOverlap(
					head_outputs.back(), tail_weights[parameter_task]) &&
				!tensorByteRangesOverlap(
					head_outputs.back(), tail_biases[parameter_task]),
				"whole-head output ", task_index,
				" must not alias any input or parameter");
		}
		for (std::size_t other = 0; other < task_index; ++other)
		{
			TORCH_CHECK(
				!tensorByteRangesOverlap(head_outputs.back(), head_outputs[other]),
				"whole-head outputs must not alias each other");
		}
	}

	CudaRasterizer::MixedWholeHeadBundleV2 head_bundle = {};
	head_bundle.task_count = static_cast<int>(head_inputs.size());
	head_bundle.worker_groups = static_cast<int>(worker_groups);
	head_bundle.persistent_blocks = static_cast<int>(persistent_blocks);
	for (std::size_t task_index = 0; task_index < head_inputs.size(); ++task_index)
	{
		head_bundle.tasks[task_index] = {
			reinterpret_cast<const void*>(
				head_inputs[task_index].data_ptr<at::Half>()),
			reinterpret_cast<const void*>(
				first_weights[task_index].data_ptr<at::Half>()),
			first_biases[task_index].data_ptr<float>(),
			tail_weights[task_index].data_ptr<float>(),
			tail_biases[task_index].data_ptr<float>(),
			head_outputs[task_index].data_ptr<float>(),
			static_cast<int>(head_inputs[task_index].size(0)),
			static_cast<int>(tail_weights[task_index].size(0))};
	}

	int rendered;
	torch::Tensor out_color;
	torch::Tensor out_depth;
	torch::Tensor radii;
	torch::Tensor geom_buffer;
	torch::Tensor binning_buffer;
	torch::Tensor image_buffer;
	std::tie(
		rendered, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer) =
		rasterizeWithNewMixedBackend(
			raster_choices,
			background, means3D, colors, opacity, scales, rotations,
			scale_modifier, cov3D_precomp, viewmatrix, projmatrix,
			tan_fovx, tan_fovy, image_height, image_width,
			sh, degree, campos, prefiltered, debug,
			nullptr, &head_bundle, nullptr);
	return std::make_tuple(
		rendered, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer, head_outputs);
}

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
	const int64_t persistent_blocks)
{
	const MixedRasterChoices raster_choices = checkMixedRasterArguments(
		background, means3D, colors, opacity, scales, rotations,
		cov3D_precomp, viewmatrix, projmatrix, sh, campos, degree);
	checkMixedDeformMlpArguments(
		means3D, deform_features, backbone_weight, backbone_bias,
		first_weights, first_biases, tail_weights, tail_biases,
		persistent_blocks);
	checkRasterDimensions(image_height, image_width);

	const c10::cuda::CUDAGuard device_guard(means3D.device());
	constexpr int output_widths[CudaRasterizer::kDeformMlpOutputCountV1] = {
		3, 3, 4, 1, 48};
	std::vector<torch::Tensor> deform_outputs;
	deform_outputs.reserve(CudaRasterizer::kDeformMlpOutputCountV1);
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
	{
		deform_outputs.push_back(torch::empty(
			{deform_features.size(0), output_widths[output]},
			deform_features.options().dtype(torch::kFloat32).requires_grad(false)));
	}

	CudaRasterizer::MixedDeformMlpTaskV1 deform_task = {};
	deform_task.features = deform_features.data_ptr<float>();
	deform_task.backbone_weight = backbone_weight.data_ptr<float>();
	deform_task.backbone_bias = backbone_bias.data_ptr<float>();
	deform_task.first_weights = reinterpret_cast<const void*>(
		first_weights.data_ptr<at::Half>());
	deform_task.first_biases = first_biases.data_ptr<float>();
	deform_task.tail_weights = tail_weights.data_ptr<float>();
	deform_task.tail_biases = tail_biases.data_ptr<float>();
	for (int output = 0; output < CudaRasterizer::kDeformMlpOutputCountV1;
		 ++output)
		deform_task.outputs[output] = deform_outputs[output].data_ptr<float>();
	deform_task.rows = static_cast<int>(deform_features.size(0));
	deform_task.persistent_blocks = static_cast<int>(persistent_blocks);

	int rendered;
	torch::Tensor out_color;
	torch::Tensor out_depth;
	torch::Tensor radii;
	torch::Tensor geom_buffer;
	torch::Tensor binning_buffer;
	torch::Tensor image_buffer;
	std::tie(
		rendered, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer) =
		rasterizeWithNewMixedBackend(
			raster_choices,
			background, means3D, colors, opacity, scales, rotations,
			scale_modifier, cov3D_precomp, viewmatrix, projmatrix,
			tan_fovx, tan_fovy, image_height, image_width,
			sh, degree, campos, prefiltered, debug,
			nullptr, nullptr, &deform_task);
	return std::make_tuple(
		rendered, out_color, out_depth, radii,
		geom_buffer, binning_buffer, image_buffer, deform_outputs);
}

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
	const int64_t persistent_blocks)
{
	auto result = RasterizeGaussiansWithWholeHeadsCUDA(
		background, means3D, colors, opacity, scales, rotations,
		scale_modifier, cov3D_precomp, viewmatrix, projmatrix,
		tan_fovx, tan_fovy, image_height, image_width,
		sh, degree, campos, prefiltered, debug,
		std::vector<torch::Tensor>{head_input},
		std::vector<torch::Tensor>{first_weight},
		std::vector<torch::Tensor>{first_bias},
		std::vector<torch::Tensor>{tail_weight},
		std::vector<torch::Tensor>{tail_bias},
		1,
		persistent_blocks);
	return std::make_tuple(
		std::get<0>(result), std::get<1>(result), std::get<2>(result),
		std::get<3>(result), std::get<4>(result), std::get<5>(result),
		std::get<6>(result), std::get<7>(result).front());
}

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
	const bool debug) 
{
	TORCH_CHECK(means3D.is_cuda(), "means3D must be a CUDA tensor");
	const c10::cuda::CUDAGuard device_guard(means3D.device());
	const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	TORCH_CHECK(
		dL_dout_color.dim() == 3,
		"dL_dout_color must have shape [channels, H, W]");
	TORCH_CHECK(
		dL_dout_color.size(1) <= static_cast<int64_t>(INT_MAX) &&
			dL_dout_color.size(2) <= static_cast<int64_t>(INT_MAX),
		"gradient image dimensions exceed the int32 Raster ABI");
	checkInt32RowCount("means3D", means3D);
	const int P = static_cast<int>(means3D.size(0));
	const int H = static_cast<int>(dL_dout_color.size(1));
	const int W = static_cast<int>(dL_dout_color.size(2));
	checkRasterDimensions(H, W);
  
	const int M = legacyShCoefficientCount(sh);

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_depth.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_ddepths.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug,
	  stream);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
	TORCH_CHECK(means3D.is_cuda(), "means3D must be a CUDA tensor");
	const c10::cuda::CUDAGuard device_guard(means3D.device());
	checkInt32RowCount("means3D", means3D);
	const int P = static_cast<int>(means3D.size(0));
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
	if(P != 0)
	{
		const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
		CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
			present.contiguous().data<bool>(),
			stream);
  }
  
  return present;
}
