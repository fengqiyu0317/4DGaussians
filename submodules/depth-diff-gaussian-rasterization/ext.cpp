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

#include <torch/extension.h>
#include <stdexcept>
#include <string>
#include "rasterize_points.h"
#include "cuda_rasterizer/tacker_mixed.h"

namespace py = pybind11;

#ifndef TACKER_SM_TARGET
#define TACKER_SM_TARGET sm_86
#endif
#define TACKER_STRINGIFY_IMPL(value) #value
#define TACKER_STRINGIFY(value) TACKER_STRINGIFY_IMPL(value)

py::dict TackerCapabilities() {
  py::dict capabilities;
  capabilities["stream_aware"] = true;
  capabilities["mixed_render_head_abi"] = CudaRasterizer::Tacker::kMixedAbiVersion;
  capabilities["mixed_render_head"] = true;
  capabilities["mixed_symbol"] = "tacker_mix_render_head_v1";
  capabilities["mixed_threads"] = CudaRasterizer::Tacker::kMixedThreads;
  capabilities["raster_threads"] = CudaRasterizer::Tacker::kRasterThreads;
  capabilities["head_threads"] = CudaRasterizer::Tacker::kHeadThreads;
  capabilities["head_thread_base"] = CudaRasterizer::Tacker::kHeadThreadBase;
  capabilities["raster_named_barrier_id"] = CudaRasterizer::Tacker::kRasterBarrierId;
  capabilities["head_features"] = 128;
  capabilities["rasterizer_commit"] = "e49506654e8e11ed8a62d22bcb693e943fdecacf";
  capabilities["sm_target"] = TACKER_STRINGIFY(TACKER_SM_TARGET);
  capabilities["mixed_manifest_sha256"] =
      "231c90c429321b2673b88ecd09efb40b6aedda7a23f3e061a2bcedec06d44426";
  capabilities["head_manifest_sha256"] =
      "24570aa6e67e8b9b10fa94524fec4dc03a4eb3fdc3bf822af34c2c52ce4937ac";

  // ABI v1 fields above are intentionally unchanged.  ABI v2 describes a
  // generic 1--5 first-linear bundle; semantic head identities live in the
  // caller's candidate descriptor, not in this physical wrapper.
  capabilities["mixed_render_heads_abi"] =
      CudaRasterizer::Tacker::kMixedMultiAbiVersion;
  capabilities["mixed_render_heads"] = true;
  capabilities["mixed_multi_symbol"] = "tacker_mix_render_heads_v2";
  capabilities["mixed_multi_manifest"] =
      "abi/tacker_mixed_render_heads_v2.json";
  capabilities["mixed_multi_manifest_sha256"] =
      "310b15957c5920773bb03a61a37c5771f6d4570393061ece4e1805fd20989056";
  capabilities["head_multi_manifest_sha256"] =
      "9d6a1558acd6b642b975bcabe22abcbe3fd7242e4c9e0d635636ef4d2eb5da7f";
  capabilities["max_head_tasks"] = CudaRasterizer::Tacker::kMaxHeadTasksV2;
  capabilities["max_mixed_heads"] = CudaRasterizer::Tacker::kMaxHeadTasksV2;
  capabilities["min_worker_groups"] =
      CudaRasterizer::Tacker::kMinWorkerGroupsV2;
  capabilities["max_worker_groups"] =
      CudaRasterizer::Tacker::kMaxWorkerGroupsV2;
  capabilities["worker_group_threads"] =
      CudaRasterizer::Tacker::kWorkerGroupThreadsV2;
  capabilities["head_descriptor_named_barrier_id"] =
      CudaRasterizer::Tacker::kHeadDescriptorBarrierIdV2;
  py::list supported_worker_groups;
  py::dict threads_by_worker_groups;
  for (int worker_groups = CudaRasterizer::Tacker::kMinWorkerGroupsV2;
       worker_groups <= CudaRasterizer::Tacker::kMaxWorkerGroupsV2;
       ++worker_groups) {
    supported_worker_groups.append(worker_groups);
    threads_by_worker_groups[py::int_(worker_groups)] =
        CudaRasterizer::Tacker::kRasterThreads +
        worker_groups * CudaRasterizer::Tacker::kWorkerGroupThreadsV2;
  }
  capabilities["supported_worker_groups"] = supported_worker_groups;
  capabilities["mixed_threads_by_worker_groups"] = threads_by_worker_groups;
  capabilities["resource_query"] = "tacker_resource_requirements";
  capabilities["resource_query_family_aware"] = true;

  capabilities["mixed_render_packed_heads_abi"] =
      CudaRasterizer::Tacker::kMixedPackedAbiVersion;
  capabilities["mixed_render_packed_heads"] = true;
  capabilities["mixed_packed_family"] = "packed_first_linear_v3";
  capabilities["mixed_packed_symbol"] =
      "tacker_mix_render_packed_heads_v3";
  capabilities["mixed_packed_manifest"] =
      "abi/tacker_mixed_render_packed_heads_v3.json";
  capabilities["mixed_packed_manifest_sha256"] =
      "c98ed90853308179443146d3022e5da072c4f507975193f7a01f4fbe4400cf40";
  capabilities["mixed_packed_head_manifest_sha256"] =
      "9d6a1558acd6b642b975bcabe22abcbe3fd7242e4c9e0d635636ef4d2eb5da7f";

  capabilities["mixed_render_whole_heads_abi"] =
      CudaRasterizer::Tacker::kMixedWholeHeadAbiVersion;
  capabilities["mixed_render_whole_heads"] = true;
  capabilities["mixed_whole_family"] = "whole_heads_v4";
  capabilities["mixed_whole_symbol"] =
      "tacker_mix_render_whole_heads_v4";
  capabilities["mixed_whole_manifest"] =
      "abi/tacker_mixed_render_whole_heads_v4.json";
  capabilities["mixed_whole_manifest_sha256"] =
      "293b8471fc9397070f1d1ebbe1297420f24f49e6882369e2e6cf8dcd9d49b7a1";
  capabilities["mixed_whole_head_manifest_sha256"] =
      "9d6a1558acd6b642b975bcabe22abcbe3fd7242e4c9e0d635636ef4d2eb5da7f";
  capabilities["whole_head_tail_features_min"] = 1;
  capabilities["whole_head_tail_features_max"] =
      CudaRasterizer::Tacker::kMaxTailFeaturesV2;
  capabilities["whole_head_scratch_bytes_per_worker_group"] =
      128 * static_cast<int>(sizeof(float));
  capabilities["whole_head_named_barriers_per_worker_group"] = 1;
  capabilities["whole_head_barrier_base_id"] =
      CudaRasterizer::Tacker::kWholeHeadBarrierBaseIdV2;
  capabilities["whole_head_descriptor_named_barrier_id"] =
      CudaRasterizer::Tacker::kWholeHeadDescriptorBarrierIdV2;
  capabilities["mixed_render_deform_mlp_abi"] =
      CudaRasterizer::Tacker::kMixedDeformMlpAbiVersion;
  capabilities["mixed_render_deform_mlp"] = true;
  capabilities["mixed_deform_mlp_family"] = "full_deform_mlp_v5";
  capabilities["mixed_deform_mlp_symbol"] =
      "tacker_mix_render_deform_mlp_v5";
  capabilities["mixed_deform_mlp_manifest"] =
      "abi/tacker_mixed_render_deform_mlp_v5.json";
  capabilities["mixed_deform_mlp_manifest_sha256"] =
      "0472fca3de2ff8c6c37a70220a3e751918603203b0367bf321fb67f35c09c531";
  capabilities["deform_mlp_manifest_sha256"] =
      "1c8112a30ce8c5075689d3bd5ed03b8aaadbdcdf29f86cf92ee0a725af3b4900";
  capabilities["deform_mlp_threads"] = 128;
  capabilities["deform_mlp_named_barrier_id"] =
      CudaRasterizer::Tacker::kDeformMlpBarrierIdV1;
  capabilities["deform_mlp_direct_grid_default"] = true;
  capabilities["prepared_raster_deform_mlp_overlap"] = true;
  capabilities["mixed_render_deform_mlp_tensor_tail_abi"] =
      CudaRasterizer::Tacker::kMixedDeformMlpTensorTailAbiVersion;
  capabilities["mixed_render_deform_mlp_tensor_tail"] = true;
  capabilities["mixed_deform_mlp_tensor_tail_family"] =
      "full_deform_mlp_tensor_tail_v6";
  capabilities["mixed_deform_mlp_tensor_tail_symbol"] =
      "tacker_mix_render_deform_mlp_tensor_tail_v6";
  capabilities["mixed_deform_mlp_tensor_tail_manifest"] =
      "abi/tacker_mixed_render_deform_mlp_tensor_tail_v6.json";
  capabilities["mixed_deform_mlp_tensor_tail_manifest_sha256"] =
      "10ab03f30e42ce9f0d0466decd1e0fa902b65286bd9e4dc80ee28043a334c829";
  capabilities["prepared_raster_deform_mlp_tensor_tail_overlap"] = true;
  capabilities["mixed_render_deform_mlp_tensor_tail_low_register_abi"] =
      CudaRasterizer::Tacker::
          kMixedDeformMlpTensorTailLowRegisterAbiVersion;
  capabilities["mixed_render_deform_mlp_tensor_tail_low_register"] = true;
  capabilities["mixed_deform_mlp_tensor_tail_low_register_family"] =
      "full_deform_mlp_tensor_tail_low_register_v7";
  capabilities["mixed_deform_mlp_tensor_tail_low_register_symbol"] =
      "tacker_mix_render_deform_mlp_tensor_tail_low_register_v7";
  capabilities["mixed_deform_mlp_tensor_tail_low_register_manifest"] =
      "abi/tacker_mixed_render_deform_mlp_tensor_tail_low_register_v7.json";
  capabilities["mixed_deform_mlp_tensor_tail_low_register_manifest_sha256"] =
      "40de26212cb6b3f1671df77860989815b6f91510a952a5d0799d45fdf60d5842";
  capabilities[
      "prepared_raster_deform_mlp_tensor_tail_low_register_overlap"] = true;
  capabilities["mixed_render_deform_mlp_tensor_tail_four_cta_abi"] =
      CudaRasterizer::Tacker::kMixedDeformMlpTensorTailFourCtaAbiVersion;
  capabilities["mixed_render_deform_mlp_tensor_tail_four_cta"] = true;
  capabilities["mixed_deform_mlp_tensor_tail_four_cta_family"] =
      "full_deform_mlp_tensor_tail_four_cta_v8";
  capabilities["mixed_deform_mlp_tensor_tail_four_cta_symbol"] =
      "tacker_mix_render_deform_mlp_tensor_tail_four_cta_v8";
  capabilities["mixed_deform_mlp_tensor_tail_four_cta_manifest"] =
      "abi/tacker_mixed_render_deform_mlp_tensor_tail_four_cta_v8.json";
  capabilities["mixed_deform_mlp_tensor_tail_four_cta_manifest_sha256"] =
      "c91b092d6bc342676f3698395539f2eca2583009170d7fef2a5c0eb5a25f38dd";
  capabilities[
      "prepared_raster_deform_mlp_tensor_tail_four_cta_overlap"] = true;
  capabilities[
      "prepared_raster_deform_mlp_tensor_tail_four_cta_diagnostic"] = true;
  capabilities["mixed_render_deform_mlp_tensor_tail_balanced_abi"] =
      CudaRasterizer::Tacker::kMixedDeformMlpTensorTailBalancedAbiVersion;
  capabilities["mixed_render_deform_mlp_tensor_tail_balanced"] = true;
  capabilities["mixed_deform_mlp_tensor_tail_balanced_family"] =
      "full_deform_mlp_tensor_tail_balanced_v9";
  capabilities["mixed_deform_mlp_tensor_tail_balanced_symbol"] =
      "tacker_mix_render_deform_mlp_tensor_tail_balanced_v9";
  capabilities["mixed_deform_mlp_tensor_tail_balanced_manifest"] =
      "abi/tacker_mixed_render_deform_mlp_tensor_tail_balanced_v9.json";
  capabilities["mixed_deform_mlp_tensor_tail_balanced_manifest_sha256"] =
      "cc22fdee5f770059e1ec6c119a8e0c1ed43e5982ce75b91cd9c9e3a650cf6907";
  capabilities[
      "prepared_raster_deform_mlp_tensor_tail_balanced_overlap"] = true;
  capabilities["mixed_render_deform_mlp_tensor_tail_raster_optimized_abi"] =
      CudaRasterizer::Tacker::
          kMixedDeformMlpTensorTailRasterOptimizedAbiVersion;
  capabilities["mixed_render_deform_mlp_tensor_tail_raster_optimized"] = true;
  capabilities["mixed_deform_mlp_tensor_tail_raster_optimized_family"] =
      "full_deform_mlp_tensor_tail_raster_optimized_v10";
  capabilities["mixed_deform_mlp_tensor_tail_raster_optimized_symbol"] =
      "tacker_mix_render_deform_mlp_tensor_tail_raster_optimized_v10";
  capabilities["mixed_deform_mlp_tensor_tail_raster_optimized_manifest"] =
      "abi/tacker_mixed_render_deform_mlp_tensor_tail_raster_optimized_v10.json";
  capabilities["mixed_deform_mlp_tensor_tail_raster_optimized_manifest_sha256"] =
      "874a60dfa04786d2282b144e02c1c9930721b9cc2e55eeb3846bf626000f013f";
  capabilities[
      "prepared_raster_deform_mlp_tensor_tail_raster_optimized_overlap"] = true;
  capabilities["mixed_render_deform_mlp_tensor_tail_raster_fast_abi"] =
      CudaRasterizer::Tacker::kMixedDeformMlpTensorTailRasterFastAbiVersion;
  capabilities["mixed_render_deform_mlp_tensor_tail_raster_fast"] = true;
  capabilities["mixed_deform_mlp_tensor_tail_raster_fast_family"] =
      "full_deform_mlp_tensor_tail_raster_fast_v11";
  capabilities["mixed_deform_mlp_tensor_tail_raster_fast_symbol"] =
      "tacker_mix_render_deform_mlp_tensor_tail_raster_fast_v11";
  capabilities["mixed_deform_mlp_tensor_tail_raster_fast_manifest"] =
      "abi/tacker_mixed_render_deform_mlp_tensor_tail_raster_fast_v11.json";
  capabilities["mixed_deform_mlp_tensor_tail_raster_fast_manifest_sha256"] =
      "8990266ab60b1bf4f5ffeb3c8008da6a4400df78fce03f396f3da2f42c6548ee";
  capabilities[
      "prepared_raster_deform_mlp_tensor_tail_raster_fast_overlap"] = true;
  capabilities["mixed_render_deform_mlp_tensor_tail_split_raster_abi"] =
      CudaRasterizer::Tacker::kMixedDeformMlpTensorTailSplitRasterAbiVersion;
  capabilities["mixed_render_deform_mlp_tensor_tail_split_raster"] = true;
  capabilities["mixed_deform_mlp_tensor_tail_split_raster_symbol"] =
      "tacker_mix_render_deform_mlp_tensor_tail_split_raster_v12";
  capabilities["raster_fast_range_symbol"] =
      "tacker_render_raster_fast_range_v12";
  capabilities["mixed_deform_mlp_tensor_tail_split_raster_manifest"] =
      "abi/tacker_mixed_render_deform_mlp_tensor_tail_split_raster_v12.json";
  capabilities["mixed_deform_mlp_tensor_tail_split_raster_manifest_sha256"] =
      "1b34996b07bc4116d5cffff6c183e53ea79045c4d304de7de8966132ac09e42d";
  capabilities[
      "prepared_raster_deform_mlp_tensor_tail_split_raster_overlap"] = true;
  capabilities[
      "prepared_raster_deform_mlp_tensor_tail_raster_diagnostic"] = true;
  py::list supported_backend_families;
  supported_backend_families.append("first_linear_heads_v2");
  supported_backend_families.append("packed_first_linear_v3");
  supported_backend_families.append("whole_heads_v4");
  supported_backend_families.append("full_deform_mlp_v5");
  capabilities["supported_backend_families"] = supported_backend_families;
  py::list supported_mixed_abis;
  for (int abi_version = CudaRasterizer::Tacker::kMixedAbiVersion;
       abi_version <= CudaRasterizer::Tacker::kMixedDeformMlpAbiVersion;
       ++abi_version) {
    supported_mixed_abis.append(abi_version);
  }
  capabilities["supported_mixed_abis"] = supported_mixed_abis;
  capabilities["prepared_raster_hexplane_sampling"] = true;
  capabilities["raster_hexplane_symbol"] = "tacker_mix_raster_hexplane_512_v1";
  capabilities["raster_hexplane_threads"] = 512;
  capabilities["raster_hexplane_max_planes"] = 12;
  capabilities["prepared_raster_hexplane_blocks"] = true;
  capabilities["raster_hexplane_blocks_symbol"] = "tacker_mix_raster_hexplane_blocks_v1";
  capabilities["raster_hexplane_blocks_threads"] = 256;
  return capabilities;
}

CudaRasterizer::Tacker::MixedBackendFamily ParseBackendFamily(
    int abi_version,
    const std::string& family) {
  if (family.empty()) {
    if (abi_version == CudaRasterizer::Tacker::kMixedAbiVersion ||
        abi_version == CudaRasterizer::Tacker::kMixedMultiAbiVersion) {
      return CudaRasterizer::Tacker::MixedBackendFamily::FirstLinear;
    }
    if (abi_version == CudaRasterizer::Tacker::kMixedPackedAbiVersion) {
      return CudaRasterizer::Tacker::MixedBackendFamily::PackedFirstLinear;
    }
    if (abi_version == CudaRasterizer::Tacker::kMixedWholeHeadAbiVersion) {
      return CudaRasterizer::Tacker::MixedBackendFamily::WholeHead;
    }
    if (abi_version == CudaRasterizer::Tacker::kMixedDeformMlpAbiVersion) {
      return CudaRasterizer::Tacker::MixedBackendFamily::FullDeformMlp;
    }
    if (abi_version ==
        CudaRasterizer::Tacker::kMixedDeformMlpTensorTailAbiVersion) {
      return CudaRasterizer::Tacker::MixedBackendFamily::FullDeformMlpTensorTail;
    }
    if (abi_version == CudaRasterizer::Tacker::
                           kMixedDeformMlpTensorTailLowRegisterAbiVersion) {
      return CudaRasterizer::Tacker::MixedBackendFamily::
          FullDeformMlpTensorTailLowRegister;
    }
    if (abi_version == CudaRasterizer::Tacker::
                           kMixedDeformMlpTensorTailFourCtaAbiVersion) {
      return CudaRasterizer::Tacker::MixedBackendFamily::
          FullDeformMlpTensorTailFourCta;
    }
    if (abi_version == CudaRasterizer::Tacker::
                           kMixedDeformMlpTensorTailBalancedAbiVersion) {
      return CudaRasterizer::Tacker::MixedBackendFamily::
          FullDeformMlpTensorTailBalanced;
    }
    throw std::invalid_argument("unsupported mixed ABI version");
  }
  if ((family == "first_linear_head_v1" &&
       abi_version == CudaRasterizer::Tacker::kMixedAbiVersion) ||
      (family == "first_linear_heads_v2" &&
       abi_version == CudaRasterizer::Tacker::kMixedMultiAbiVersion)) {
    return CudaRasterizer::Tacker::MixedBackendFamily::FirstLinear;
  }
  if (family == "packed_first_linear_v3" &&
      abi_version == CudaRasterizer::Tacker::kMixedPackedAbiVersion) {
    return CudaRasterizer::Tacker::MixedBackendFamily::PackedFirstLinear;
  }
  if (family == "whole_heads_v4" &&
      abi_version == CudaRasterizer::Tacker::kMixedWholeHeadAbiVersion) {
    return CudaRasterizer::Tacker::MixedBackendFamily::WholeHead;
  }
  if (family == "full_deform_mlp_v5" &&
      abi_version == CudaRasterizer::Tacker::kMixedDeformMlpAbiVersion) {
    return CudaRasterizer::Tacker::MixedBackendFamily::FullDeformMlp;
  }
  if (family == "full_deform_mlp_tensor_tail_v6" &&
      abi_version ==
          CudaRasterizer::Tacker::kMixedDeformMlpTensorTailAbiVersion) {
    return CudaRasterizer::Tacker::MixedBackendFamily::FullDeformMlpTensorTail;
  }
  if (family == "full_deform_mlp_tensor_tail_low_register_v7" &&
      abi_version == CudaRasterizer::Tacker::
                         kMixedDeformMlpTensorTailLowRegisterAbiVersion) {
    return CudaRasterizer::Tacker::MixedBackendFamily::
        FullDeformMlpTensorTailLowRegister;
  }
  if (family == "full_deform_mlp_tensor_tail_four_cta_v8" &&
      abi_version == CudaRasterizer::Tacker::
                         kMixedDeformMlpTensorTailFourCtaAbiVersion) {
    return CudaRasterizer::Tacker::MixedBackendFamily::
        FullDeformMlpTensorTailFourCta;
  }
  if (family == "full_deform_mlp_tensor_tail_balanced_v9" &&
      abi_version == CudaRasterizer::Tacker::
                         kMixedDeformMlpTensorTailBalancedAbiVersion) {
    return CudaRasterizer::Tacker::MixedBackendFamily::
        FullDeformMlpTensorTailBalanced;
  }
  if (family == "full_deform_mlp_tensor_tail_raster_optimized_v10" &&
      abi_version == CudaRasterizer::Tacker::
                         kMixedDeformMlpTensorTailRasterOptimizedAbiVersion) {
    return CudaRasterizer::Tacker::MixedBackendFamily::
        FullDeformMlpTensorTailRasterOptimized;
  }
  if (family == "full_deform_mlp_tensor_tail_raster_fast_v11" &&
      abi_version == CudaRasterizer::Tacker::
                         kMixedDeformMlpTensorTailRasterFastAbiVersion) {
    return CudaRasterizer::Tacker::MixedBackendFamily::
        FullDeformMlpTensorTailRasterFast;
  }
  throw std::invalid_argument(
      "backend family does not match the requested mixed ABI version");
}

const char* ResourceBackendFamily(
    int abi_version,
    CudaRasterizer::Tacker::MixedBackendFamily family) {
  if (abi_version == CudaRasterizer::Tacker::kMixedAbiVersion &&
      family == CudaRasterizer::Tacker::MixedBackendFamily::FirstLinear) {
    return "first_linear_head_v1";
  }
  return CudaRasterizer::Tacker::mixedBackendFamilyName(family);
}

py::dict TackerResourceRequirements(
    int abi_version,
    int worker_groups,
    const std::string& family) {
  const CudaRasterizer::Tacker::MixedBackendFamily parsed_family =
      ParseBackendFamily(abi_version, family);
  const CudaRasterizer::Tacker::MixedKernelResources resources =
      CudaRasterizer::Tacker::queryMixedKernelResources(
          abi_version, worker_groups, parsed_family);
  py::dict result;
  result["abi_version"] = resources.abi_version;
  result["backend_abi_version"] = resources.abi_version;
  result["backend_family"] =
      ResourceBackendFamily(resources.abi_version, resources.backend_family);
  result["family"] =
      ResourceBackendFamily(resources.abi_version, resources.backend_family);
  result["worker_groups"] = resources.worker_groups;
  result["physical_threads"] = resources.physical_threads;
  result["device_ordinal"] = resources.device_ordinal;
  result["compute_capability_major"] = resources.compute_capability_major;
  result["compute_capability_minor"] = resources.compute_capability_minor;
  result["multiprocessor_count"] = resources.multiprocessor_count;
  result["device_max_threads_per_block"] =
      resources.device_max_threads_per_block;
  result["device_max_threads_per_multiprocessor"] =
      resources.device_max_threads_per_multiprocessor;
  result["warp_size"] = resources.warp_size;
  result["kernel_max_threads_per_block"] =
      resources.kernel_max_threads_per_block;
  result["registers_per_thread"] = resources.registers_per_thread;
  result["static_shared_bytes"] = resources.static_shared_bytes;
  result["local_bytes_per_thread"] = resources.local_bytes_per_thread;
  result["max_dynamic_shared_bytes"] = resources.max_dynamic_shared_bytes;
  result["active_blocks_per_multiprocessor"] =
      resources.active_blocks_per_multiprocessor;
  result["active_warps_per_multiprocessor"] =
      resources.active_warps_per_multiprocessor;
  result["max_warps_per_multiprocessor"] =
      resources.max_warps_per_multiprocessor;
  result["occupancy"] = resources.occupancy;
  result["launch_supported"] = resources.launch_supported;
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def(
      "tacker_image_state",
      &TackerImageStateCUDA,
      "Profiling-only clone of tile ranges and per-pixel contributor indices");
  m.def(
      "rasterize_gaussians_with_head",
      &RasterizeGaussiansWithHeadCUDA,
      "Inference-only physical Raster+head mixed kernel");
  m.def(
      "rasterize_gaussians_with_heads",
      &RasterizeGaussiansWithHeadsCUDA,
      "Inference-only physical Raster+1--5 first-linear heads mixed kernel");
  m.def(
      "rasterize_gaussians_with_packed_heads",
      &RasterizeGaussiansWithPackedHeadsCUDA,
      "Inference-only physical Raster+packed first-linear heads mixed kernel");
  m.def(
      "rasterize_gaussians_with_whole_head",
      &RasterizeGaussiansWithWholeHeadCUDA,
      "Inference-only physical Raster+one complete head mixed kernel");
  m.def(
      "rasterize_gaussians_with_whole_heads",
      &RasterizeGaussiansWithWholeHeadsCUDA,
      "Inference-only physical Raster+1--5 complete heads mixed kernel");
  m.def(
      "rasterize_gaussians_with_deform_mlp",
      &RasterizeGaussiansWithDeformMlpCUDA,
      "Inference-only physical Raster+full deformation MLP mixed kernel");
  m.def(
      "prepare_gaussians_for_deform_mlp_overlap",
      &PrepareGaussiansForDeformMlpOverlapCUDA,
      "Prepare Raster state without launching the final render leaf");
  m.def(
      "prepare_gaussians_for_deform_mlp_overlap_handoff",
      &PrepareGaussiansForDeformMlpOverlapHandoffCUDA,
      "Opt-in: switch streams after duplicateWithKeys, joining both producer events");
  m.def(
      "render_prepared_gaussians_with_deform_mlp_overlap",
      &RenderPreparedGaussiansWithDeformMlpOverlapCUDA,
      "Launch ABI-v5 against an already prepared Raster state");
  m.def(
      "render_prepared_gaussians_with_deform_mlp_tensor_tail_overlap",
      &RenderPreparedGaussiansWithDeformMlpTensorTailOverlapCUDA,
      "Launch experimental Tensor-tail ABI-v6 against prepared Raster state");
  m.def(
      "render_prepared_gaussians_with_deform_mlp_tensor_tail_low_register_overlap",
      &RenderPreparedGaussiansWithDeformMlpTensorTailLowRegisterOverlapCUDA,
      "Launch experimental low-register Tensor-tail ABI-v7 against prepared "
      "Raster state");
  m.def(
      "render_prepared_gaussians_with_deform_mlp_tensor_tail_four_cta_overlap",
      &RenderPreparedGaussiansWithDeformMlpTensorTailFourCtaOverlapCUDA,
      "Launch experimental four-CTA Tensor-tail ABI-v8 against prepared Raster "
      "state");
  m.def(
      "render_prepared_gaussians_with_deform_mlp_tensor_tail_four_cta_diagnostic",
      &RenderPreparedGaussiansWithDeformMlpTensorTailFourCtaDiagnosticCUDA,
      "Profile ABI-v8 Raster/Deform subgroup lifetimes in both or solo mode");
  m.def(
      "render_prepared_gaussians_with_deform_mlp_tensor_tail_raster_diagnostic",
      &RenderPreparedGaussiansWithDeformMlpTensorTailRasterDiagnosticCUDA,
      "Profile ABI-v10/v11 balanced Raster/Deform subgroup lifetimes in both "
      "or solo mode");
  m.def(
      "render_prepared_gaussians_with_deform_mlp_tensor_tail_balanced_overlap",
      &RenderPreparedGaussiansWithDeformMlpTensorTailBalancedOverlapCUDA,
      "Launch experimental balanced-work Tensor-tail ABI-v9 against prepared "
      "Raster state");
  m.def(
      "render_prepared_gaussians_with_deform_mlp_tensor_tail_raster_optimized_overlap",
      &RenderPreparedGaussiansWithDeformMlpTensorTailRasterOptimizedOverlapCUDA,
      "Launch exact Raster-optimized Tensor-tail ABI-v10 against prepared "
      "Raster state");
  m.def(
      "render_prepared_gaussians_with_deform_mlp_tensor_tail_raster_fast_overlap",
      &RenderPreparedGaussiansWithDeformMlpTensorTailRasterFastOverlapCUDA,
      "Launch approximate fast-Raster Tensor-tail ABI-v11 against prepared "
      "Raster state");
  m.def(
      "render_prepared_gaussians_raster_fast_range",
      &RenderPreparedGaussiansRasterFastRangeCUDA,
      "Launch the ABI-v12 fast-Raster-only half-open tile range");
  m.def(
      "render_prepared_gaussians_with_deform_mlp_tensor_tail_split_raster_overlap",
      &RenderPreparedGaussiansWithDeformMlpTensorTailSplitRasterOverlapCUDA,
      "Launch ABI-v12 remaining fast Raster tiles plus complete Deform");
  m.def(
      "render_prepared_gaussians_with_hexplane_sampling",
      &RenderPreparedGaussiansWithHexPlaneSamplingCUDA,
      "Launch 512-thread Tacker spatial fusion: Raster 0..255, HexPlane 256..511");
  m.def("tacker_raster_hexplane_resources", &TackerRasterHexPlaneResourcesCUDA);
  m.def("render_prepared_gaussians_with_hexplane_blocks",
        &RenderPreparedGaussiansWithHexPlaneBlocksCUDA);
  m.def("tacker_raster_hexplane_block_resources", &TackerRasterHexPlaneBlockResourcesCUDA);
  m.def(
      "render_prepared_gaussians_overlap",
      &RenderPreparedGaussiansOverlapCUDA,
      "Launch the ordinary render leaf against prepared Raster state");
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("tacker_capabilities", &TackerCapabilities);
  m.def(
      "tacker_resource_requirements",
      &TackerResourceRequirements,
      py::arg("abi_version") = CudaRasterizer::Tacker::kMixedMultiAbiVersion,
      py::arg("worker_groups") = 1,
      py::arg("family") = "",
      "Query active-device family-specific registers, shared memory, and occupancy");
}
