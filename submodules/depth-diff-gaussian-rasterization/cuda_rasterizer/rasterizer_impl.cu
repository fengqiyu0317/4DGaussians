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

#include "rasterizer_impl.h"
#include <climits>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

namespace
{
	// The rasterizer needs the final scan value on the host to size its binning
	// workspace.  Keep a tiny per-host-thread pinned buffer so the copy can be
	// ordered on the caller's stream without introducing a device-wide fence.
	struct PinnedRenderedCount
	{
		uint32_t* value = nullptr;
		cudaError_t status = cudaSuccess;

		PinnedRenderedCount()
		{
			status = cudaMallocHost(
				reinterpret_cast<void**>(&value), sizeof(uint32_t));
		}

		~PinnedRenderedCount()
		{
			if (value != nullptr)
				cudaFreeHost(value);
		}
	};

	thread_local PinnedRenderedCount rendered_count;
}

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"
#include "tacker_mixed.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present,
	cudaStream_t stream)
{
	checkFrustum << <(P + 255) / 256, 256, 0, stream >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
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
	cudaStream_t stream)
{
	return forward(
		geometryBuffer,
		binningBuffer,
		imageBuffer,
		P, D, M,
		background,
		width, height,
		means3D,
		shs,
		colors_precomp,
		opacities,
		scales,
		scale_modifier,
		rotations,
		cov3D_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		tan_fovx, tan_fovy,
		prefiltered,
		out_color,
		out_depth,
		radii,
		debug,
		stream,
		nullptr);
}

int CudaRasterizer::Rasterizer::forward(
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
	const MixedHeadTask* mixed_head)
{
	return forward(
		geometryBuffer,
		binningBuffer,
		imageBuffer,
		P, D, M,
		background,
		width, height,
		means3D,
		shs,
		colors_precomp,
		opacities,
		scales,
		scale_modifier,
		rotations,
		cov3D_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		tan_fovx, tan_fovy,
		prefiltered,
		out_color,
		out_depth,
		radii,
		debug,
		stream,
		mixed_head,
		nullptr);
}

int CudaRasterizer::Rasterizer::forward(
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
	const MixedHeadTask* mixed_head,
	const MixedHeadBundleV2* mixed_heads)
{
	return forward(
		geometryBuffer,
		binningBuffer,
		imageBuffer,
		P, D, M,
		background,
		width, height,
		means3D,
		shs,
		colors_precomp,
		opacities,
		scales,
		scale_modifier,
		rotations,
		cov3D_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		tan_fovx, tan_fovy,
		prefiltered,
		out_color,
		out_depth,
		radii,
		debug,
		stream,
		mixed_head,
		mixed_heads,
		nullptr,
		nullptr,
		nullptr);
}

int CudaRasterizer::Rasterizer::forward(
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
	const MixedHeadTask* mixed_head,
	const MixedHeadBundleV2* mixed_heads,
	const MixedPackedHeadBundleV2* mixed_packed_heads,
	const MixedWholeHeadBundleV2* mixed_whole_heads,
	const MixedDeformMlpTaskV1* mixed_deform_mlp)
{
	const int selected_backends =
		(mixed_head != nullptr ? 1 : 0) +
		(mixed_heads != nullptr ? 1 : 0) +
		(mixed_packed_heads != nullptr ? 1 : 0) +
		(mixed_whole_heads != nullptr ? 1 : 0) +
		(mixed_deform_mlp != nullptr ? 1 : 0);
	if (selected_backends > 1)
		throw std::invalid_argument(
			"mixed Raster launch must select at most one backend family");
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		stream
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P, stream), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	if (rendered_count.status != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(rendered_count.status));
	const cudaError_t rendered_copy_status = cudaMemcpyAsync(
		rendered_count.value,
		geomState.point_offsets + P - 1,
		sizeof(uint32_t),
		cudaMemcpyDeviceToHost,
		stream);
	if (rendered_copy_status != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(rendered_copy_status));
	const cudaError_t rendered_sync_status = cudaStreamSynchronize(stream);
	if (rendered_sync_status != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(rendered_sync_status));
	const uint32_t num_rendered_u32 = *rendered_count.value;
	if (num_rendered_u32 > static_cast<uint32_t>(INT_MAX))
		throw std::overflow_error(
			"rendered Gaussian instance count exceeds the int32 Raster ABI");
	const int num_rendered = static_cast<int>(num_rendered_u32);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256, 0, stream >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit, stream), debug)

	CHECK_CUDA(cudaMemsetAsync(imgState.ranges, 0,
		tile_grid.x * tile_grid.y * sizeof(uint2), stream), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256, 0, stream >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	if (mixed_deform_mlp != nullptr)
	{
		CHECK_CUDA(Tacker::launchMixedRenderDeformMlp(
			tile_grid,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			feature_ptr,
			geomState.depths,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color,
			out_depth,
			*mixed_deform_mlp,
			stream), debug)
	}
	else if (mixed_whole_heads != nullptr)
	{
		CHECK_CUDA(Tacker::launchMixedRenderWholeHeads(
			tile_grid,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			feature_ptr,
			geomState.depths,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color,
			out_depth,
			*mixed_whole_heads,
			stream), debug)
	}
	else if (mixed_packed_heads != nullptr)
	{
		CHECK_CUDA(Tacker::launchMixedRenderPackedHeads(
			tile_grid,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			feature_ptr,
			geomState.depths,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color,
			out_depth,
			*mixed_packed_heads,
			stream), debug)
	}
	else if (mixed_heads != nullptr)
	{
		CHECK_CUDA(Tacker::launchMixedRenderHeads(
			tile_grid,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			feature_ptr,
			geomState.depths,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color,
			out_depth,
			*mixed_heads,
			stream), debug)
	}
	else if (mixed_head != nullptr)
	{
		CHECK_CUDA(Tacker::launchMixedRenderHead(
			tile_grid,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			feature_ptr,
			geomState.depths,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color,
			out_depth,
			*mixed_head,
			stream), debug)
	}
	else
	{
		CHECK_CUDA(FORWARD::render(
			tile_grid, block,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			feature_ptr,
			geomState.depths,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color,
			out_depth,
			stream), debug)
	}

	return num_rendered;
}

int CudaRasterizer::Rasterizer::prepareForwardForDeformMlpOverlap(
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
	cudaStream_t handoff_stream,
	cudaEvent_t deform_done,
	cudaEvent_t duplicate_done)
{
	if (P <= 0)
		throw std::invalid_argument(
			"prepared Raster prefix requires at least one Gaussian");

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);
	if (radii == nullptr)
		radii = geomState.internal_radii;

	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);

	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
		throw std::runtime_error(
			"For non-RGB, provide precomputed Gaussian colors!");

	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		stream), debug)

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		geomState.scanning_space,
		geomState.scan_size,
		geomState.tiles_touched,
		geomState.point_offsets,
		P,
		stream), debug)

	if (rendered_count.status != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(rendered_count.status));
	const cudaError_t rendered_copy_status = cudaMemcpyAsync(
		rendered_count.value,
		geomState.point_offsets + P - 1,
		sizeof(uint32_t),
		cudaMemcpyDeviceToHost,
		stream);
	if (rendered_copy_status != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(rendered_copy_status));
	const cudaError_t rendered_sync_status = cudaStreamSynchronize(stream);
	if (rendered_sync_status != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(rendered_sync_status));
	const uint32_t num_rendered_u32 = *rendered_count.value;
	if (num_rendered_u32 > static_cast<uint32_t>(INT_MAX))
		throw std::overflow_error(
			"rendered Gaussian instance count exceeds the int32 Raster ABI");
	const int num_rendered = static_cast<int>(num_rendered_u32);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState =
		BinningState::fromChunk(binning_chunkptr, num_rendered);

	duplicateWithKeys << <(P + 255) / 256, 256, 0, stream >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	// Only future kernels change resource scope. Keep the duplicate launch
	// on the Green stream and join both producers on the full-device stream.
	if (handoff_stream != nullptr)
	{
		auto checked_handoff = [](cudaError_t status) {
			if (status != cudaSuccess)
				throw std::runtime_error(cudaGetErrorString(status));
		};
		checked_handoff(cudaEventRecord(duplicate_done, stream));
		checked_handoff(cudaStreamWaitEvent(handoff_stream, duplicate_done, 0));
		checked_handoff(cudaStreamWaitEvent(handoff_stream, deform_done, 0));
		stream = handoff_stream;
	}

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted,
		binningState.point_list_keys,
		binningState.point_list_unsorted,
		binningState.point_list,
		num_rendered, 0, 32 + bit, stream), debug)

	CHECK_CUDA(cudaMemsetAsync(
		imgState.ranges,
		0,
		tile_grid.x * tile_grid.y * sizeof(uint2),
		stream), debug);

	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256, 0, stream >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	return num_rendered;
}

void CudaRasterizer::Rasterizer::renderPreparedDeformMlpOverlap(
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
	const MixedDeformMlpTaskV1* mixed_deform_mlp)
{
	if (P <= 0)
		throw std::invalid_argument(
			"prepared Raster render requires at least one Gaussian");
	if (num_rendered < 0)
		throw std::invalid_argument(
			"prepared Raster render count must be non-negative");

	char* geom_chunk = geometry_buffer;
	char* binning_chunk = binning_buffer;
	char* image_chunk = image_buffer;
	GeometryState geomState = GeometryState::fromChunk(geom_chunk, P);
	BinningState binningState =
		BinningState::fromChunk(binning_chunk, num_rendered);
	ImageState imgState = ImageState::fromChunk(image_chunk, width * height);

	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	const float* feature_ptr =
		colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	if (mixed_deform_mlp != nullptr)
	{
		CHECK_CUDA(Tacker::launchMixedRenderDeformMlp(
			tile_grid,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			feature_ptr,
			geomState.depths,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color,
			out_depth,
			*mixed_deform_mlp,
			stream), debug)
	}
	else
	{
		CHECK_CUDA(FORWARD::render(
			tile_grid, block,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			feature_ptr,
			geomState.depths,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color,
			out_depth,
			stream), debug)
	}
}

void CudaRasterizer::Rasterizer::renderPreparedDeformMlpTensorTailOverlap(
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
	const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp)
{
	if (P <= 0)
		throw std::invalid_argument(
			"prepared Raster render requires at least one Gaussian");
	if (num_rendered < 0)
		throw std::invalid_argument(
			"prepared Raster render count must be non-negative");

	char* geom_chunk = geometry_buffer;
	char* binning_chunk = binning_buffer;
	char* image_chunk = image_buffer;
	GeometryState geomState = GeometryState::fromChunk(geom_chunk, P);
	BinningState binningState =
		BinningState::fromChunk(binning_chunk, num_rendered);
	ImageState imgState = ImageState::fromChunk(image_chunk, width * height);
	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const float* feature_ptr =
		colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(Tacker::launchMixedRenderDeformMlpTensorTail(
		tile_grid,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		mixed_deform_mlp,
		stream), debug)
}

void CudaRasterizer::Rasterizer::renderPreparedDeformMlpTensorTailLowRegisterOverlap(
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
	const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp)
{
	if (P <= 0)
		throw std::invalid_argument(
			"prepared Raster render requires at least one Gaussian");
	if (num_rendered < 0)
		throw std::invalid_argument(
			"prepared Raster render count must be non-negative");

	char* geom_chunk = geometry_buffer;
	char* binning_chunk = binning_buffer;
	char* image_chunk = image_buffer;
	GeometryState geomState = GeometryState::fromChunk(geom_chunk, P);
	BinningState binningState =
		BinningState::fromChunk(binning_chunk, num_rendered);
	ImageState imgState = ImageState::fromChunk(image_chunk, width * height);
	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const float* feature_ptr =
		colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(Tacker::launchMixedRenderDeformMlpTensorTailLowRegister(
		tile_grid,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		mixed_deform_mlp,
		stream), debug)
}

void CudaRasterizer::Rasterizer::renderPreparedDeformMlpTensorTailFourCtaOverlap(
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
	const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp)
{
	if (P <= 0)
		throw std::invalid_argument(
			"prepared Raster render requires at least one Gaussian");
	if (num_rendered < 0)
		throw std::invalid_argument(
			"prepared Raster render count must be non-negative");

	char* geom_chunk = geometry_buffer;
	char* binning_chunk = binning_buffer;
	char* image_chunk = image_buffer;
	GeometryState geomState = GeometryState::fromChunk(geom_chunk, P);
	BinningState binningState =
		BinningState::fromChunk(binning_chunk, num_rendered);
	ImageState imgState = ImageState::fromChunk(image_chunk, width * height);
	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const float* feature_ptr =
		colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(Tacker::launchMixedRenderDeformMlpTensorTailFourCta(
		tile_grid,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		mixed_deform_mlp,
		stream), debug)
}

void CudaRasterizer::Rasterizer::renderPreparedDeformMlpTensorTailBalancedOverlap(
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
	const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp)
{
	if (P <= 0)
		throw std::invalid_argument(
			"prepared Raster render requires at least one Gaussian");
	if (num_rendered < 0)
		throw std::invalid_argument(
			"prepared Raster render count must be non-negative");

	char* geom_chunk = geometry_buffer;
	char* binning_chunk = binning_buffer;
	char* image_chunk = image_buffer;
	GeometryState geomState = GeometryState::fromChunk(geom_chunk, P);
	BinningState binningState =
		BinningState::fromChunk(binning_chunk, num_rendered);
	ImageState imgState = ImageState::fromChunk(image_chunk, width * height);
	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const float* feature_ptr =
		colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(Tacker::launchMixedRenderDeformMlpTensorTailBalanced(
		tile_grid,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		mixed_deform_mlp,
		stream), debug)
}

void CudaRasterizer::Rasterizer::renderPreparedDeformMlpTensorTailRasterOptimizedOverlap(
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
	const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp)
{
	if (P <= 0)
		throw std::invalid_argument(
			"prepared Raster render requires at least one Gaussian");
	if (num_rendered < 0)
		throw std::invalid_argument(
			"prepared Raster render count must be non-negative");

	char* geom_chunk = geometry_buffer;
	char* binning_chunk = binning_buffer;
	char* image_chunk = image_buffer;
	GeometryState geomState = GeometryState::fromChunk(geom_chunk, P);
	BinningState binningState =
		BinningState::fromChunk(binning_chunk, num_rendered);
	ImageState imgState = ImageState::fromChunk(image_chunk, width * height);
	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const float* feature_ptr =
		colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(Tacker::launchMixedRenderDeformMlpTensorTailRasterOptimized(
		tile_grid,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		mixed_deform_mlp,
		stream), debug)
}

void CudaRasterizer::Rasterizer::renderPreparedDeformMlpTensorTailRasterFastOverlap(
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
	const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp)
{
	if (P <= 0)
		throw std::invalid_argument(
			"prepared Raster render requires at least one Gaussian");
	if (num_rendered < 0)
		throw std::invalid_argument(
			"prepared Raster render count must be non-negative");

	char* geom_chunk = geometry_buffer;
	char* binning_chunk = binning_buffer;
	char* image_chunk = image_buffer;
	GeometryState geomState = GeometryState::fromChunk(geom_chunk, P);
	BinningState binningState =
		BinningState::fromChunk(binning_chunk, num_rendered);
	ImageState imgState = ImageState::fromChunk(image_chunk, width * height);
	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const float* feature_ptr =
		colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(Tacker::launchMixedRenderDeformMlpTensorTailRasterFast(
		tile_grid,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		mixed_deform_mlp,
		stream), debug)
}

void CudaRasterizer::Rasterizer::renderPreparedRasterFastRange(
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
	int raster_end)
{
	if (P <= 0)
		throw std::invalid_argument(
			"prepared Raster render requires at least one Gaussian");
	if (num_rendered < 0)
		throw std::invalid_argument(
			"prepared Raster render count must be non-negative");

	char* geom_chunk = geometry_buffer;
	char* binning_chunk = binning_buffer;
	char* image_chunk = image_buffer;
	GeometryState geomState = GeometryState::fromChunk(geom_chunk, P);
	BinningState binningState =
		BinningState::fromChunk(binning_chunk, num_rendered);
	ImageState imgState = ImageState::fromChunk(image_chunk, width * height);
	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const float* feature_ptr =
		colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(Tacker::launchRasterFastRange(
		tile_grid,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		raster_start,
		raster_end,
		stream), debug)
}

void CudaRasterizer::Rasterizer::renderPreparedDeformMlpTensorTailSplitRasterOverlap(
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
	const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp)
{
	if (P <= 0)
		throw std::invalid_argument(
			"prepared Raster render requires at least one Gaussian");
	if (num_rendered < 0)
		throw std::invalid_argument(
			"prepared Raster render count must be non-negative");

	char* geom_chunk = geometry_buffer;
	char* binning_chunk = binning_buffer;
	char* image_chunk = image_buffer;
	GeometryState geomState = GeometryState::fromChunk(geom_chunk, P);
	BinningState binningState =
		BinningState::fromChunk(binning_chunk, num_rendered);
	ImageState imgState = ImageState::fromChunk(image_chunk, width * height);
	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const float* feature_ptr =
		colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(Tacker::launchMixedRenderDeformMlpTensorTailSplitRaster(
		tile_grid,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		raster_start,
		mixed_deform_mlp,
		stream), debug)
}

void CudaRasterizer::Rasterizer::renderPreparedDeformMlpTensorTailRasterDiagnostic(
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
	const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp,
	int abi_version,
	int diagnostic_mode,
	int64_t* timing_cycles,
	int timing_rows)
{
	if (P <= 0)
		throw std::invalid_argument(
			"prepared Raster render requires at least one Gaussian");
	if (num_rendered < 0)
		throw std::invalid_argument(
			"prepared Raster render count must be non-negative");

	char* geom_chunk = geometry_buffer;
	char* binning_chunk = binning_buffer;
	char* image_chunk = image_buffer;
	GeometryState geomState = GeometryState::fromChunk(geom_chunk, P);
	BinningState binningState =
		BinningState::fromChunk(binning_chunk, num_rendered);
	ImageState imgState = ImageState::fromChunk(image_chunk, width * height);
	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const float* feature_ptr =
		colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(Tacker::launchMixedRenderDeformMlpTensorTailRasterDiagnostic(
		tile_grid,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		mixed_deform_mlp,
		abi_version,
		diagnostic_mode,
		timing_cycles,
		timing_rows,
		stream), debug)
}

void CudaRasterizer::Rasterizer::renderPreparedDeformMlpTensorTailFourCtaDiagnostic(
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
	const MixedDeformMlpTensorTailTaskV2& mixed_deform_mlp,
	int diagnostic_mode,
	int64_t* timing_cycles,
	int timing_rows)
{
	if (P <= 0)
		throw std::invalid_argument(
			"prepared Raster render requires at least one Gaussian");
	if (num_rendered < 0)
		throw std::invalid_argument(
			"prepared Raster render count must be non-negative");

	char* geom_chunk = geometry_buffer;
	char* binning_chunk = binning_buffer;
	char* image_chunk = image_buffer;
	GeometryState geomState = GeometryState::fromChunk(geom_chunk, P);
	BinningState binningState =
		BinningState::fromChunk(binning_chunk, num_rendered);
	ImageState imgState = ImageState::fromChunk(image_chunk, width * height);
	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const float* feature_ptr =
		colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(Tacker::launchMixedRenderDeformMlpTensorTailFourCtaDiagnostic(
		tile_grid,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		mixed_deform_mlp,
		diagnostic_mode,
		timing_cycles,
		timing_rows,
		stream), debug)
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	char* img_buffer,
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
	cudaStream_t stream)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		depth_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_dpix_depth,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_ddepth,
		stream), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		stream), debug)
}
