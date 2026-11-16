/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group
 *
 * Tacker-compatible forward adapters derived from the original forward
 * kernels in this directory.  They preserve the original math while
 * virtualizing the logical block/thread indices so two independent tasks can
 * share one physical CTA.
 */

#pragma once

#include <math.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "auxiliary.h"
#include "config.h"

namespace CudaRasterizer
{
namespace Tacker
{

__device__ __forceinline__ unsigned int subgroup_count(
	bool predicate,
	int barrier_id,
	int participant_count)
{
	unsigned int result;
	const unsigned int value = predicate ? 1u : 0u;
	asm volatile(
		"{ .reg .pred p;\n\t"
		"  setp.ne.u32 p, %3, 0;\n\t"
		"  bar.red.popc.u32 %0, %1, %2, p;\n\t"
		"}"
		: "=r"(result)
		: "r"(barrier_id), "r"(participant_count), "r"(value)
		: "memory");
	return result;
}

__device__ __forceinline__ void subgroup_sync(
	int barrier_id,
	int participant_count)
{
	asm volatile("bar.sync %0, %1;"
		:
		: "r"(barrier_id), "r"(participant_count)
		: "memory");
}

__device__ __forceinline__ void general_ptb_duplicate_with_keys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const int* radii,
	dim3 tile_grid,
	int logical_block_size,
	int ptb_start_block_pos,
	int ptb_iter_block_step,
	int ptb_end_block_pos,
	int thread_base)
{
	const int local_thread = static_cast<int>(threadIdx.x) - thread_base;
	if (local_thread < 0 || local_thread >= logical_block_size)
		return;

	for (int block_pos = static_cast<int>(blockIdx.x) + ptb_start_block_pos;
		 block_pos < ptb_end_block_pos;
		 block_pos += ptb_iter_block_step)
	{
		const int idx = block_pos * logical_block_size + local_thread;
		if (idx >= P || radii[idx] <= 0)
			continue;

		uint32_t off = idx == 0 ? 0 : offsets[idx - 1];
		uint2 rect_min;
		uint2 rect_max;
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, tile_grid);

		for (int y = rect_min.y; y < rect_max.y; ++y)
		{
			for (int x = rect_min.x; x < rect_max.x; ++x)
			{
				uint64_t key = static_cast<uint64_t>(y * tile_grid.x + x) << 32;
				key |= *reinterpret_cast<const uint32_t*>(&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = static_cast<uint32_t>(idx);
				++off;
			}
		}
	}
}

__device__ __forceinline__ void general_ptb_identify_tile_ranges(
	int L,
	const uint64_t* point_list_keys,
	uint2* ranges,
	int logical_block_size,
	int ptb_start_block_pos,
	int ptb_iter_block_step,
	int ptb_end_block_pos,
	int thread_base)
{
	const int local_thread = static_cast<int>(threadIdx.x) - thread_base;
	if (local_thread < 0 || local_thread >= logical_block_size)
		return;

	for (int block_pos = static_cast<int>(blockIdx.x) + ptb_start_block_pos;
		 block_pos < ptb_end_block_pos;
		 block_pos += ptb_iter_block_step)
	{
		const int idx = block_pos * logical_block_size + local_thread;
		if (idx >= L)
			continue;

		const uint64_t key = point_list_keys[idx];
		const uint32_t currtile = static_cast<uint32_t>(key >> 32);
		if (idx == 0)
			ranges[currtile].x = 0;
		else
		{
			const uint32_t prevtile = static_cast<uint32_t>(point_list_keys[idx - 1] >> 32);
			if (currtile != prevtile)
			{
				ranges[prevtile].y = idx;
				ranges[currtile].x = idx;
			}
		}
		if (idx == L - 1)
			ranges[currtile].y = L;
	}
}

template <uint32_t CHANNELS>
__device__ __forceinline__ void general_ptb_render(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W,
	int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_depth,
	int logical_grid_x,
	int logical_grid_y,
	int ptb_start_block_pos,
	int ptb_iter_block_step,
	int ptb_end_block_pos,
	int thread_base,
	int barrier_id)
{
	constexpr int kRenderThreads = BLOCK_X * BLOCK_Y;
	const int local_thread = static_cast<int>(threadIdx.x) - thread_base;
	if (local_thread < 0 || local_thread >= kRenderThreads)
		return;

	// These arrays are block-scoped, but only the contiguous render subgroup
	// accesses them.  Its named barrier prevents unrelated GEMM threads from
	// participating in render synchronization.
	__shared__ int collected_id[kRenderThreads];
	__shared__ float2 collected_xy[kRenderThreads];
	__shared__ float4 collected_conic_opacity[kRenderThreads];

	const int logical_grid_size = logical_grid_x * logical_grid_y;
	const int logical_end = min(ptb_end_block_pos, logical_grid_size);
	for (int block_pos = static_cast<int>(blockIdx.x) + ptb_start_block_pos;
		 block_pos < logical_end;
		 block_pos += ptb_iter_block_step)
	{
		const int logical_block_x = block_pos % logical_grid_x;
		const int logical_block_y = block_pos / logical_grid_x;
		const int logical_thread_x = local_thread % BLOCK_X;
		const int logical_thread_y = local_thread / BLOCK_X;

		const uint2 pix_min = {
			static_cast<uint32_t>(logical_block_x * BLOCK_X),
			static_cast<uint32_t>(logical_block_y * BLOCK_Y)};
		const uint2 pix = {
			pix_min.x + static_cast<uint32_t>(logical_thread_x),
			pix_min.y + static_cast<uint32_t>(logical_thread_y)};
		const uint32_t pix_id = static_cast<uint32_t>(W) * pix.y + pix.x;
		const float2 pixf = {static_cast<float>(pix.x), static_cast<float>(pix.y)};

		const bool inside = pix.x < static_cast<uint32_t>(W) &&
			pix.y < static_cast<uint32_t>(H);
		bool done = !inside;
		const uint2 range = ranges[block_pos];
		const int rounds = (static_cast<int>(range.y - range.x) + kRenderThreads - 1) /
			kRenderThreads;
		int to_do = static_cast<int>(range.y - range.x);

		float transmittance = 1.0f;
		uint32_t contributor = 0;
		uint32_t last_contributor = 0;
		float color[CHANNELS] = {0};
		float depth = 0.0f;

		for (int round = 0; round < rounds; ++round, to_do -= kRenderThreads)
		{
			const unsigned int num_done = subgroup_count(done, barrier_id, kRenderThreads);
			if (num_done == kRenderThreads)
				break;

			const int progress = round * kRenderThreads + local_thread;
			if (range.x + progress < range.y)
			{
				const int gaussian_id = point_list[range.x + progress];
				collected_id[local_thread] = gaussian_id;
				collected_xy[local_thread] = points_xy_image[gaussian_id];
				collected_conic_opacity[local_thread] = conic_opacity[gaussian_id];
			}
			subgroup_sync(barrier_id, kRenderThreads);

			for (int j = 0; !done && j < min(kRenderThreads, to_do); ++j)
			{
				++contributor;
				const float2 xy = collected_xy[j];
				const float2 delta = {xy.x - pixf.x, xy.y - pixf.y};
				const float4 conic = collected_conic_opacity[j];
				const float power = -0.5f *
					(conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) -
					conic.y * delta.x * delta.y;
				if (power > 0.0f)
					continue;

				const float alpha = min(0.99f, conic.w * exp(power));
				if (alpha < 1.0f / 255.0f)
					continue;
				const float next_transmittance = transmittance * (1.0f - alpha);
				if (next_transmittance < 0.0001f)
				{
					done = true;
					continue;
				}

				const int gaussian_id = collected_id[j];
				for (int channel = 0; channel < CHANNELS; ++channel)
					color[channel] += features[gaussian_id * CHANNELS + channel] *
						alpha * transmittance;
				depth += depths[gaussian_id] * alpha * transmittance;
				transmittance = next_transmittance;
				last_contributor = contributor;
			}
		}

		if (inside)
		{
			final_T[pix_id] = transmittance;
			n_contrib[pix_id] = last_contributor;
			for (int channel = 0; channel < CHANNELS; ++channel)
				out_color[channel * H * W + pix_id] =
					color[channel] + transmittance * bg_color[channel];
			out_depth[pix_id] = depth;
		}

		// Keep every render-subgroup thread at the same persistent-loop
		// boundary before any thread reuses shared storage for the next tile.
		subgroup_sync(barrier_id, kRenderThreads);
	}
}

// Raster-optimized adapter used only by post-ABI-v9 experimental kernels.
// The original adapter above remains unchanged so all qualified mixed ABIs
// retain their exact code generation and resource envelope.
//
// The exact path makes three source-level changes:
//   1. derive n_contrib from (round, j) instead of incrementing an integer in
//      every pixel/Gaussian iteration;
//   2. reject powers below a conservative alpha cutoff before evaluating exp;
//   3. cooperatively cache RGB+depth once per Gaussian batch and remove the
//      repeated indirect global reads from the per-pixel blend path.
//
// FAST_MATH is a separate accuracy experiment.  It additionally uses __expf
// and explicit FMA composition; FAST_MATH=false preserves the original
// floating-point expression order for all surviving contributions.
template <uint32_t CHANNELS, bool FAST_MATH>
__device__ __forceinline__ void general_ptb_render_raster_optimized(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W,
	int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_depth,
	int logical_grid_x,
	int logical_grid_y,
	int ptb_start_block_pos,
	int ptb_iter_block_step,
	int ptb_end_block_pos,
	int thread_base,
	int barrier_id,
	int logical_block_id = -1)
{
	static_assert(CHANNELS == 3,
		"the Raster-optimized mixed adapter requires RGB features");
	constexpr int kRenderThreads = BLOCK_X * BLOCK_Y;
	// exp(-5.55) is safely below 1/255.  The opacity guard keeps this
	// early-out conservative even if a caller supplies a nonstandard value.
	constexpr float kConservativePowerCutoff = -5.55f;
	const int local_thread = static_cast<int>(threadIdx.x) - thread_base;
	if (local_thread < 0 || local_thread >= kRenderThreads)
		return;

	__shared__ float2 collected_xy[kRenderThreads];
	__shared__ float4 collected_conic_opacity[kRenderThreads];
	__shared__ float4 collected_rgb_depth[kRenderThreads];

	const int logical_grid_size = logical_grid_x * logical_grid_y;
	const int logical_end = min(ptb_end_block_pos, logical_grid_size);
	const int first_block = logical_block_id < 0 ? static_cast<int>(blockIdx.x) : logical_block_id;
	for (int block_pos = first_block + ptb_start_block_pos;
		 block_pos < logical_end;
		 block_pos += ptb_iter_block_step)
	{
		const int logical_block_x = block_pos % logical_grid_x;
		const int logical_block_y = block_pos / logical_grid_x;
		const int logical_thread_x = local_thread % BLOCK_X;
		const int logical_thread_y = local_thread / BLOCK_X;

		const uint2 pix_min = {
			static_cast<uint32_t>(logical_block_x * BLOCK_X),
			static_cast<uint32_t>(logical_block_y * BLOCK_Y)};
		const uint2 pix = {
			pix_min.x + static_cast<uint32_t>(logical_thread_x),
			pix_min.y + static_cast<uint32_t>(logical_thread_y)};
		const uint32_t pix_id = static_cast<uint32_t>(W) * pix.y + pix.x;
		const float2 pixf = {static_cast<float>(pix.x), static_cast<float>(pix.y)};

		const bool inside = pix.x < static_cast<uint32_t>(W) &&
			pix.y < static_cast<uint32_t>(H);
		bool done = !inside;
		const uint2 range = ranges[block_pos];
		const int range_size = static_cast<int>(range.y - range.x);
		const int rounds = (range_size + kRenderThreads - 1) / kRenderThreads;
		int to_do = range_size;

		float transmittance = 1.0f;
		uint32_t last_contributor = 0;
		float color[CHANNELS] = {0};
		float depth = 0.0f;

		for (int round = 0; round < rounds; ++round, to_do -= kRenderThreads)
		{
			const unsigned int num_done = subgroup_count(done, barrier_id, kRenderThreads);
			if (num_done == kRenderThreads)
				break;

			const int progress = round * kRenderThreads + local_thread;
			if (progress < range_size)
			{
				const int gaussian_id = point_list[range.x + progress];
				const int feature_offset = gaussian_id * CHANNELS;
				collected_xy[local_thread] = points_xy_image[gaussian_id];
				collected_conic_opacity[local_thread] = conic_opacity[gaussian_id];
				collected_rgb_depth[local_thread] = make_float4(
					features[feature_offset],
					features[feature_offset + 1],
					features[feature_offset + 2],
					depths[gaussian_id]);
			}
			subgroup_sync(barrier_id, kRenderThreads);

			const int batch_size = min(kRenderThreads, to_do);
			for (int j = 0; !done && j < batch_size; ++j)
			{
				// Start both independent geometry loads before consuming either.
				// In the old order delta immediately depended on xy, then the
				// quadratic immediately depended on a later conic load, producing
				// two serial short-scoreboard windows.
				const float4 conic = collected_conic_opacity[j];
				const float2 xy = collected_xy[j];
				const float2 delta = {xy.x - pixf.x, xy.y - pixf.y};
				float power;
				if (FAST_MATH)
				{
					const float dx2 = delta.x * delta.x;
					const float dy2 = delta.y * delta.y;
					const float quadratic = fmaf(
						conic.x, dx2, conic.z * dy2);
					power = fmaf(
						-conic.y, delta.x * delta.y, -0.5f * quadratic);
				}
				else
				{
					power = -0.5f *
						(conic.x * delta.x * delta.x +
						 conic.z * delta.y * delta.y) -
						conic.y * delta.x * delta.y;
				}
				if (power > 0.0f)
					continue;
				if (conic.w <= 1.0f && power < kConservativePowerCutoff)
					continue;

				const float falloff = FAST_MATH ? __expf(power) : exp(power);
				const float alpha = min(0.99f, conic.w * falloff);
				if (alpha < 1.0f / 255.0f)
					continue;
				const float next_transmittance = FAST_MATH
					? fmaf(-alpha, transmittance, transmittance)
					: transmittance * (1.0f - alpha);
				if (next_transmittance < 0.0001f)
				{
					done = true;
					continue;
				}

				const float4 rgb_depth = collected_rgb_depth[j];
				if (FAST_MATH)
				{
					const float weight = alpha * transmittance;
					color[0] = fmaf(rgb_depth.x, weight, color[0]);
					color[1] = fmaf(rgb_depth.y, weight, color[1]);
					color[2] = fmaf(rgb_depth.z, weight, color[2]);
					depth = fmaf(rgb_depth.w, weight, depth);
				}
				else
				{
					color[0] += rgb_depth.x * alpha * transmittance;
					color[1] += rgb_depth.y * alpha * transmittance;
					color[2] += rgb_depth.z * alpha * transmittance;
					depth += rgb_depth.w * alpha * transmittance;
				}
				transmittance = next_transmittance;
				last_contributor = static_cast<uint32_t>(
					round * kRenderThreads + j + 1);
			}
		}

		if (inside)
		{
			final_T[pix_id] = transmittance;
			n_contrib[pix_id] = last_contributor;
			for (int channel = 0; channel < CHANNELS; ++channel)
				out_color[channel * H * W + pix_id] =
					color[channel] + transmittance * bg_color[channel];
			out_depth[pix_id] = depth;
		}

		subgroup_sync(barrier_id, kRenderThreads);
	}
}

}  // namespace Tacker
}  // namespace CudaRasterizer
