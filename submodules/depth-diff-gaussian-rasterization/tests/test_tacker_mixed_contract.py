"""Torch/CUDA-free source contract tests for the mixed Raster+head ABI."""

import json
import hashlib
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def manifest_sha256(relative_path):
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def capability_sha256(extension, key):
    match = re.search(
        rf'capabilities\["{re.escape(key)}"\]\s*=\s*"([0-9a-f]{{64}})";',
        extension,
    )
    if match is None:
        raise AssertionError(f"missing SHA-256 capability {key!r}")
    return match.group(1)


class MixedKernelContractTest(unittest.TestCase):
    def test_wrapper_is_one_real_384_thread_kernel(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        self.assertIn("kMixedThreads = 384", header)
        self.assertIn("tacker_mix_render_head_v1", cuda)
        self.assertIn("<<<\n\t\tphysical_blocks, kMixedThreads", cuda)

    def test_wrapper_calls_both_real_device_adapters(self):
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        adapter = source("cuda_rasterizer/tacker_forward.cuh")
        self.assertIn("general_ptb_render<NUM_CHANNELS>", cuda)
        self.assertIn("tacker_4dgs::head_linear_gptb_device", cuda)
        self.assertIn("#include \"head_linear_device.cuh\"", cuda)
        self.assertNotIn("#define GLM_FORCE_CUDA", adapter)

    def test_thread_ranges_and_barrier_are_disjoint(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        self.assertIn("kRasterThreads = 256", header)
        self.assertIn("kHeadThreads = 128", header)
        self.assertIn("kHeadThreadBase = 256", header)
        self.assertIn("kRasterBarrierId = 1", header)
        self.assertIn("static_assert(", cuda)
        self.assertNotIn("__syncthreads(", cuda)

    def test_zero_policy_queries_device_and_launch_check_is_async(self):
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        self.assertIn("cudaGetDeviceProperties", cuda)
        self.assertIn("cached_properties", cuda)
        self.assertIn("cudaPeekAtLastError", cuda)
        self.assertIn("(void)cudaGetLastError()", cuda)
        self.assertNotIn("cudaDeviceSynchronize", cuda)
        self.assertNotIn("physical_blocks = 68", cuda)
        self.assertNotIn("physical_blocks = 142", cuda)

    def test_manifest_matches_source_contract(self):
        manifest = json.loads(
            source("abi/tacker_mixed_render_head_v1.json")
        )
        self.assertEqual(manifest["abi_version"], 1)
        self.assertEqual(manifest["cuda_arch"], "sm_86")
        self.assertEqual(manifest["global_kernel_symbol"], "tacker_mix_render_head_v1")
        self.assertEqual(
            manifest["subgroups"]["raster"]["thread_range"], [0, 255]
        )
        self.assertEqual(
            manifest["subgroups"]["head"]["thread_range"], [256, 383]
        )
        self.assertEqual(manifest["subgroups"]["raster"]["named_barrier_id"], 1)

    def test_v2_wrapper_supports_one_to_five_heads_and_dynamic_ctas(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        adapter = source("../../tacker_ext/include/head_linear_v2_device.cuh")
        self.assertIn("kMixedMultiAbiVersion = 2", header)
        self.assertIn("kMaxHeadTasksV2 = 5", header)
        self.assertIn("kMaxMixedThreadsV2", header)
        self.assertIn("tacker_mix_render_heads_v2", cuda)
        self.assertIn("head_linear_multi_gptb_device", cuda)
        self.assertIn("head_linear_multi_gptb_device", adapter)
        self.assertIn(
            "heads.worker_groups * kWorkerGroupThreadsV2", cuda
        )
        self.assertIn("heads.tasks[4]", cuda)

    def test_v2_named_barriers_are_disjoint_and_never_cta_wide(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        self.assertIn("kRasterBarrierId = 1", header)
        self.assertIn("kHeadDescriptorBarrierIdV2 = 2", header)
        self.assertIn("backend_threads", cuda)
        self.assertIn("shared_head_tasks", cuda)
        self.assertNotIn("__syncthreads(", cuda)

    def test_v2_manifest_matches_source_and_covers_c1_c2(self):
        manifest_text = source("abi/tacker_mixed_render_heads_v2.json")
        manifest = json.loads(manifest_text)
        self.assertEqual(manifest["abi_version"], 2)
        self.assertEqual(
            manifest["global_kernel_symbol"], "tacker_mix_render_heads_v2"
        )
        self.assertEqual(manifest["limits"]["task_count"], [1, 5])
        self.assertEqual(
            manifest["physical_launch"]["thread_counts_by_worker_groups"],
            {"1": 384, "2": 512, "3": 640, "4": 768, "5": 896},
        )
        self.assertEqual(
            manifest["candidate_coverage"]["C1_head_names"],
            ["pos", "scales", "rotations", "opacity", "shs"],
        )
        self.assertEqual(
            manifest["candidate_coverage"]["required_dual_head_variant"][
                "supported_worker_groups"
            ],
            [1, 2],
        )
        self.assertEqual(
            manifest["subgroups"]["head_workers"][
                "descriptor_named_barrier_id"
            ],
            2,
        )
        # Make the byte-level artifact consumed by the parent profile easy to
        # reproduce without importing torch or normalizing JSON.
        self.assertEqual(
            len(hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()), 64
        )

    def test_c3_packed_wrapper_uses_adapter_without_backend_barrier(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        manifest = json.loads(
            source("abi/tacker_mixed_render_packed_heads_v3.json")
        )
        self.assertIn("kMixedPackedAbiVersion = 3", header)
        self.assertIn("tacker_mix_render_packed_heads_v3", cuda)
        self.assertIn("head_linear_packed_gptb_device", cuda)
        self.assertEqual(manifest["abi_version"], 3)
        self.assertEqual(manifest["backend_family"], "packed_first_linear_v3")
        self.assertEqual(
            manifest["global_kernel_symbol"],
            "tacker_mix_render_packed_heads_v3",
        )
        self.assertEqual(
            manifest["subgroups"]["head_workers"]["named_barrier_ids"], []
        )
        self.assertFalse(
            manifest["subgroups"]["head_workers"]["cta_wide_barriers"]
        )

    def test_c4_whole_wrapper_has_exact_disjoint_barriers_and_scratch(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        manifest = json.loads(
            source("abi/tacker_mixed_render_whole_heads_v4.json")
        )
        self.assertIn("kMixedWholeHeadAbiVersion = 4", header)
        self.assertIn("kWholeHeadBarrierBaseIdV2 = 2", header)
        self.assertIn("kWholeHeadDescriptorBarrierIdV2 = 7", header)
        self.assertIn("tacker_mix_render_whole_heads_v4", cuda)
        self.assertIn("whole_head_multi_gptb_device", cuda)
        self.assertIn("backend_threads", cuda)
        self.assertEqual(manifest["abi_version"], 4)
        workers = manifest["subgroups"]["head_workers"]
        self.assertEqual(workers["worker_named_barrier_ids"], "[2, 1 + worker_groups]")
        self.assertEqual(workers["worker_barrier_participants"], 128)
        self.assertEqual(workers["descriptor_named_barrier_id"], 7)
        self.assertEqual(
            workers["descriptor_barrier_participants"], "worker_groups * 128"
        )
        self.assertEqual(
            manifest["physical_launch"][
                "static_shared_scratch_bytes_per_worker_group"
            ],
            512,
        )
        self.assertNotIn("__syncthreads(", cuda)

    def test_c3_c4_manifests_preserve_old_abi_artifacts(self):
        expected = [
            "abi/tacker_mixed_render_head_v1.json",
            "abi/tacker_mixed_render_heads_v2.json",
        ]
        for path in (
            "abi/tacker_mixed_render_packed_heads_v3.json",
            "abi/tacker_mixed_render_whole_heads_v4.json",
        ):
            with self.subTest(path=path):
                manifest = json.loads(source(path))
                self.assertEqual(manifest["preserved_abis"], expected)
                self.assertEqual(
                    manifest["tacker_ext_dependency"]["manifest_sha256"],
                    manifest_sha256("../../tacker_ext/abi/head_linear_v2.json"),
                )

    def test_v5_direct_full_deform_wrapper_contract(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        adapter = source("../../tacker_ext/include/deform_mlp_device.cuh")
        manifest = json.loads(
            source("abi/tacker_mixed_render_deform_mlp_v5.json")
        )
        self.assertIn("kMixedDeformMlpAbiVersion = 5", header)
        self.assertIn("tacker_mix_render_deform_mlp_v5", cuda)
        self.assertIn("deform_mlp_gptb_device_v1", cuda)
        self.assertIn("DeformMlpNamedSyncV1", adapter)
        self.assertEqual(manifest["physical_launch"]["threads"], 384)
        self.assertEqual(
            manifest["physical_launch"]["persistent_blocks_zero_policy"],
            "direct logical grid",
        )
        self.assertEqual(
            manifest["subgroups"]["deform_mlp"]["thread_range"], [256, 383]
        )
        self.assertEqual(
            manifest["subgroups"]["deform_mlp"]["named_barrier_id"], 2
        )
        self.assertNotIn("__syncthreads(", cuda)

    def test_v6_tensor_tail_preserves_v5_and_uses_named_sync(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        adapter = source("../../tacker_ext/include/deform_mlp_device.cuh")
        manifest = json.loads(source(
            "abi/tacker_mixed_render_deform_mlp_tensor_tail_v6.json"
        ))
        self.assertIn("kMixedDeformMlpAbiVersion = 5", header)
        self.assertIn("kMixedDeformMlpTensorTailAbiVersion = 6", header)
        self.assertIn("tacker_mix_render_deform_mlp_v5", cuda)
        self.assertIn("tacker_mix_render_deform_mlp_tensor_tail_v6", cuda)
        self.assertIn("deform_mlp_tensor_tail_gptb_device_v4", adapter)
        self.assertIn("DeformMlpNamedSyncV1{barrier_id}", adapter)
        self.assertEqual(manifest["abi_version"], 6)
        self.assertEqual(
            manifest["preserved_abis"][-1],
            "abi/tacker_mixed_render_deform_mlp_v5.json",
        )
        self.assertEqual(manifest["physical_launch"]["threads"], 384)
        self.assertEqual(
            manifest["tensor_contract"]["tail_activation"][
                "shared_row_stride"
            ],
            136,
        )

    def test_v7_low_register_tensor_tail_preserves_v6_contract(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        adapter = source("../../tacker_ext/include/deform_mlp_device.cuh")
        manifest = json.loads(source(
            "abi/tacker_mixed_render_deform_mlp_tensor_tail_low_register_v7.json"
        ))
        self.assertIn("kMixedDeformMlpTensorTailAbiVersion = 6", header)
        self.assertIn(
            "kMixedDeformMlpTensorTailLowRegisterAbiVersion = 7", header
        )
        self.assertIn("tacker_mix_render_deform_mlp_tensor_tail_v6", cuda)
        self.assertIn(
            "tacker_mix_render_deform_mlp_tensor_tail_low_register_v7", cuda
        )
        self.assertIn("deform_mlp_tensor_tail_gptb_device_v5", adapter)
        self.assertIn("float values[4]", adapter)
        self.assertIn("kMixedThreads, 3", cuda)
        self.assertEqual(manifest["abi_version"], 7)
        self.assertEqual(
            manifest["preserved_abis"][-1],
            "abi/tacker_mixed_render_deform_mlp_tensor_tail_v6.json",
        )
        self.assertEqual(
            manifest["physical_launch"]["launch_bounds_min_blocks_per_sm"],
            3,
        )
        self.assertFalse(
            manifest["subgroups"]["deform_mlp"]["cta_wide_barriers"]
        )
        self.assertNotIn("__syncthreads(", cuda)

    def test_v8_four_cta_tensor_tail_preserves_v7_contract(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        adapter = source("../../tacker_ext/include/deform_mlp_device.cuh")
        manifest = json.loads(source(
            "abi/tacker_mixed_render_deform_mlp_tensor_tail_four_cta_v8.json"
        ))
        self.assertIn(
            "kMixedDeformMlpTensorTailLowRegisterAbiVersion = 7", header
        )
        self.assertIn(
            "kMixedDeformMlpTensorTailFourCtaAbiVersion = 8", header
        )
        self.assertIn(
            "tacker_mix_render_deform_mlp_tensor_tail_low_register_v7", cuda
        )
        self.assertIn(
            "tacker_mix_render_deform_mlp_tensor_tail_four_cta_v8", cuda
        )
        self.assertIn("deform_mlp_tensor_tail_gptb_device_v6", adapter)
        self.assertIn("float values[2]", adapter)
        self.assertIn("kMixedThreads, 4", cuda)
        self.assertEqual(manifest["abi_version"], 8)
        self.assertEqual(
            manifest["preserved_abis"][-1],
            "abi/tacker_mixed_render_deform_mlp_tensor_tail_low_register_v7.json",
        )
        self.assertEqual(
            manifest["physical_launch"]["launch_bounds_min_blocks_per_sm"],
            4,
        )
        self.assertEqual(
            manifest["subgroups"]["deform_mlp"]["compaction_values_per_thread"],
            2,
        )
        self.assertEqual(
            manifest["subgroups"]["deform_mlp"]["descriptor_access"],
            "volatile_shared_phase_local",
        )

    def test_v9_balanced_tensor_tail_preserves_v8_contract(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        manifest = json.loads(source(
            "abi/tacker_mixed_render_deform_mlp_tensor_tail_balanced_v9.json"
        ))
        self.assertIn(
            "kMixedDeformMlpTensorTailFourCtaAbiVersion = 8", header
        )
        self.assertIn(
            "kMixedDeformMlpTensorTailBalancedAbiVersion = 9", header
        )
        self.assertIn(
            "tacker_mix_render_deform_mlp_tensor_tail_balanced_v9", cuda
        )
        self.assertIn("extra_before", cuda)
        self.assertIn("second_tile", cuda)
        self.assertEqual(manifest["abi_version"], 9)
        self.assertEqual(
            manifest["preserved_abis"][-1],
            "abi/tacker_mixed_render_deform_mlp_tensor_tail_four_cta_v8.json",
        )
        self.assertEqual(
            manifest["physical_launch"]["launch_bounds_min_blocks_per_sm"],
            4,
        )
        self.assertFalse(
            manifest["subgroups"]["deform_mlp"]["cta_wide_barriers"]
        )
        self.assertNotIn("__syncthreads(", cuda)

    def test_v10_and_v11_preserve_v9_and_isolate_fast_raster_math(self):
        header = source("cuda_rasterizer/tacker_mixed.h")
        adapter = source("cuda_rasterizer/tacker_forward.cuh")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        v10 = json.loads(source(
            "abi/tacker_mixed_render_deform_mlp_tensor_tail_"
            "raster_optimized_v10.json"
        ))
        v11 = json.loads(source(
            "abi/tacker_mixed_render_deform_mlp_tensor_tail_"
            "raster_fast_v11.json"
        ))
        self.assertIn(
            "kMixedDeformMlpTensorTailRasterOptimizedAbiVersion = 10",
            header,
        )
        self.assertIn(
            "kMixedDeformMlpTensorTailRasterFastAbiVersion = 11",
            header,
        )
        self.assertIn("general_ptb_render_raster_optimized", adapter)
        self.assertIn("kConservativePowerCutoff = -5.55f", adapter)
        self.assertIn("collected_rgb_depth", adapter)
        self.assertIn("round * kRenderThreads + j + 1", adapter)
        self.assertIn(
            "tacker_mix_render_deform_mlp_tensor_tail_raster_optimized_v10",
            cuda,
        )
        self.assertIn(
            "tacker_mix_render_deform_mlp_tensor_tail_raster_fast_v11",
            cuda,
        )
        self.assertEqual(v10["abi_version"], 10)
        self.assertEqual(v11["abi_version"], 11)
        self.assertFalse(v10["subgroups"]["raster"]["fast_math"])
        self.assertEqual(
            v11["subgroups"]["raster"]["fast_math"],
            "__expf_and_explicit_fma",
        )
        self.assertEqual(
            v11["subgroups"]["raster"]["geometry_load_order"],
            "conic_then_xy_before_first_consumer",
        )
        self.assertEqual(
            v11["subgroups"]["deform_mlp"]["device_adapter"],
            "tacker_4dgs::deform_mlp_tensor_tail_gptb_device_v7",
        )
        self.assertEqual(
            v11["subgroups"]["deform_mlp"]["static_shared_scratch_bytes"],
            14848,
        )
        self.assertIn("DeformMlpScratchV2", cuda)
        self.assertIn("deform_mlp_tensor_tail_gptb_device_v7", cuda)
        self.assertNotIn("__syncthreads(", cuda)


class ExtensionApiContractTest(unittest.TestCase):
    def test_old_binding_and_forward_path_remain(self):
        extension = source("ext.cpp")
        implementation = source("cuda_rasterizer/rasterizer_impl.cu")
        self.assertIn('m.def("rasterize_gaussians", &RasterizeGaussiansCUDA)', extension)
        self.assertIn("if (mixed_head != nullptr)", implementation)
        self.assertIn("FORWARD::render(", implementation)
        self.assertIn("nullptr);", implementation)

    def test_new_binding_and_capability_query_are_exposed(self):
        extension = source("ext.cpp")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        self.assertIn("rasterize_gaussians_with_head", extension)
        self.assertIn("tacker_capabilities", extension)
        self.assertIn('capabilities["stream_aware"] = true', extension)
        self.assertIn("def tacker_capabilities():", python_api)
        self.assertIn("def forward_with_head(", python_api)

    def test_v2_binding_capabilities_and_resource_query_are_exposed(self):
        extension = source("ext.cpp")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        header = source("rasterize_points.h")
        for required in (
            "rasterize_gaussians_with_heads",
            "mixed_render_heads_abi",
            "max_mixed_heads",
            "supported_worker_groups",
            "tacker_resource_requirements",
        ):
            self.assertIn(required, extension)
        self.assertIn("RasterizeGaussiansWithHeadsCUDA", header)
        self.assertIn("def forward_with_heads(", python_api)
        self.assertIn("def tacker_variant_resources(", python_api)
        for resource_key in (
            "block_threads",
            "registers_per_thread",
            "static_shared_memory_bytes",
            "max_threads_per_block",
            "active_blocks_per_sm",
        ):
            self.assertIn(resource_key, python_api)

    def test_capability_manifest_hashes_match_exact_artifact_bytes(self):
        extension = source("ext.cpp")
        manifest_by_capability = {
            "mixed_manifest_sha256": "abi/tacker_mixed_render_head_v1.json",
            "head_manifest_sha256": "../../tacker_ext/abi/head_linear_v1.json",
            "mixed_multi_manifest_sha256": (
                "abi/tacker_mixed_render_heads_v2.json"
            ),
            "head_multi_manifest_sha256": (
                "../../tacker_ext/abi/head_linear_v2.json"
            ),
            "mixed_packed_manifest_sha256": (
                "abi/tacker_mixed_render_packed_heads_v3.json"
            ),
            "mixed_packed_head_manifest_sha256": (
                "../../tacker_ext/abi/head_linear_v2.json"
            ),
            "mixed_whole_manifest_sha256": (
                "abi/tacker_mixed_render_whole_heads_v4.json"
            ),
            "mixed_whole_head_manifest_sha256": (
                "../../tacker_ext/abi/head_linear_v2.json"
            ),
            "mixed_deform_mlp_manifest_sha256": (
                "abi/tacker_mixed_render_deform_mlp_v5.json"
            ),
            "deform_mlp_manifest_sha256": (
                "../../tacker_ext/abi/deform_mlp_v1.json"
            ),
            "mixed_deform_mlp_tensor_tail_manifest_sha256": (
                "abi/tacker_mixed_render_deform_mlp_tensor_tail_v6.json"
            ),
            "mixed_deform_mlp_tensor_tail_low_register_manifest_sha256": (
                "abi/tacker_mixed_render_deform_mlp_tensor_tail_low_register_v7.json"
            ),
            "mixed_deform_mlp_tensor_tail_four_cta_manifest_sha256": (
                "abi/tacker_mixed_render_deform_mlp_tensor_tail_four_cta_v8.json"
            ),
            "mixed_deform_mlp_tensor_tail_balanced_manifest_sha256": (
                "abi/tacker_mixed_render_deform_mlp_tensor_tail_balanced_v9.json"
            ),
            "mixed_deform_mlp_tensor_tail_raster_optimized_manifest_sha256": (
                "abi/tacker_mixed_render_deform_mlp_tensor_tail_"
                "raster_optimized_v10.json"
            ),
            "mixed_deform_mlp_tensor_tail_raster_fast_manifest_sha256": (
                "abi/tacker_mixed_render_deform_mlp_tensor_tail_"
                "raster_fast_v11.json"
            ),
        }
        for capability, manifest_path in manifest_by_capability.items():
            with self.subTest(capability=capability):
                self.assertEqual(
                    capability_sha256(extension, capability),
                    manifest_sha256(manifest_path),
                )

    def test_v2_cpp_validation_is_fail_closed(self):
        binding = source("rasterize_points.cu")
        for required in (
            "head_inputs, head_weights, and head_biases must have equal lengths",
            "mixed ABI v2 requires between 1 and 5 head tasks",
            "worker_groups must be in [1, task_count]",
            "must have shape [N, 128]",
            "must have shape [128, 128]",
            "must have shape [128]",
            "row count exceeds the int32 mixed ABI v2",
            "must be natively 32-byte aligned for WMMA",
            "mixed ABI v2 head outputs must not alias each other",
        ):
            self.assertIn(required, binding)
        self.assertIn("MixedHeadBundleV2 head_bundle = {}", binding)
        self.assertIn("any_head_rows", binding)

    def test_c3_c4_bindings_methods_and_capabilities_are_exposed(self):
        extension = source("ext.cpp")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        header = source("rasterize_points.h")
        for name in (
            "rasterize_gaussians_with_packed_heads",
            "rasterize_gaussians_with_whole_head",
            "rasterize_gaussians_with_whole_heads",
        ):
            self.assertIn(name, extension)
        for name in (
            "RasterizeGaussiansWithPackedHeadsCUDA",
            "RasterizeGaussiansWithWholeHeadCUDA",
            "RasterizeGaussiansWithWholeHeadsCUDA",
        ):
            self.assertIn(name, header)
        self.assertIn("def forward_with_packed_heads(", python_api)
        self.assertIn("packed_head_weights", python_api)
        self.assertIn("packed_head_biases", python_api)
        self.assertIn("def forward_with_whole_head(", python_api)
        self.assertIn("def forward_with_whole_heads(", python_api)
        self.assertIn("output_widths", python_api)
        for key in (
            "mixed_render_packed_heads_abi",
            "mixed_render_packed_heads",
            "mixed_packed_symbol",
            "mixed_packed_manifest_sha256",
            "mixed_render_whole_heads_abi",
            "mixed_render_whole_heads",
            "mixed_whole_symbol",
            "mixed_whole_manifest_sha256",
            "supported_backend_families",
            "resource_query_family_aware",
        ):
            self.assertIn(key, extension)

    def test_c3_c4_cpp_validation_is_fail_closed(self):
        binding = source("rasterize_points.cu")
        for required in (
            "packed head count must be in [1, 5]",
            "packed worker_groups must be in [1, head_count]",
            "packed head_input must have shape [N, 128]",
            "packed head_weights must have shape [H, 128, 128]",
            "packed head_biases must have shape [H, 128]",
            "packed head_output must not alias input, weights, or biases",
            "whole-head inputs and parameter lists must have equal lengths",
            "mixed whole-head ABI v4 requires between 1 and 5 tasks",
            "must have shape [O, 128] with 1 <= O <= 128",
            "whole-head outputs must not alias each other",
            "must not alias any input or parameter",
            "persistent_blocks exceeds the int32 mixed whole-head ABI v4",
        ):
            self.assertIn(required, binding)
        self.assertGreaterEqual(binding.count("natively 32-byte aligned"), 8)

    def test_v5_binding_method_and_validation_are_exposed(self):
        extension = source("ext.cpp")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        binding = source("rasterize_points.cu")
        header = source("rasterize_points.h")
        self.assertIn("rasterize_gaussians_with_deform_mlp", extension)
        self.assertIn("RasterizeGaussiansWithDeformMlpCUDA", header)
        self.assertIn("def forward_with_deform_mlp(", python_api)
        for required in (
            "deform_features must have shape [N, 32]",
            "backbone_weight must have shape [32, 128]",
            "first_weights must have shape [5, 128, 128]",
            "tail_weights must have shape [59, 128]",
            "persistent_blocks exceeds the int32 mixed ABI v5",
        ):
            self.assertIn(required, binding)

    def test_resource_query_is_abi_and_family_specific(self):
        extension = source("ext.cpp")
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        for family in (
            "first_linear_heads_v2",
            "packed_first_linear_v3",
            "whole_heads_v4",
            "full_deform_mlp_v5",
            "full_deform_mlp_tensor_tail_v6",
            "full_deform_mlp_tensor_tail_low_register_v7",
            "full_deform_mlp_tensor_tail_four_cta_v8",
            "full_deform_mlp_tensor_tail_balanced_v9",
        ):
            self.assertIn(family, extension + cuda + python_api)
        self.assertIn('result["backend_abi_version"]', extension)
        self.assertIn('result["backend_family"]', extension)
        self.assertIn('result["family"]', extension)
        self.assertIn("backend family does not match", extension)
        self.assertIn("family=family", python_api)

    def test_v7_binding_capability_and_opt_in_route_are_exposed(self):
        extension = source("ext.cpp")
        binding_header = source("rasterize_points.h")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        pipeline = source("../../gaussian_renderer/tacker_pipeline.py")
        for required in (
            "render_prepared_gaussians_with_deform_mlp_tensor_tail_"
            "low_register_overlap",
            "mixed_render_deform_mlp_tensor_tail_low_register_abi",
            "prepared_raster_deform_mlp_tensor_tail_low_register_overlap",
        ):
            self.assertIn(required, extension)
        self.assertIn(
            "RenderPreparedGaussiansWithDeformMlpTensorTailLowRegisterOverlapCUDA",
            binding_header,
        )
        self.assertIn(
            "def render_prepared_with_deform_mlp_tensor_tail_low_register_overlap(",
            python_api,
        )
        self.assertIn("TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V7", pipeline)

    def test_v8_binding_capability_and_opt_in_route_are_exposed(self):
        extension = source("ext.cpp")
        binding_header = source("rasterize_points.h")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        pipeline = source("../../gaussian_renderer/tacker_pipeline.py")
        for required in (
            "render_prepared_gaussians_with_deform_mlp_tensor_tail_"
            "four_cta_overlap",
            "mixed_render_deform_mlp_tensor_tail_four_cta_abi",
            "prepared_raster_deform_mlp_tensor_tail_four_cta_overlap",
        ):
            self.assertIn(required, extension)
        self.assertIn(
            "RenderPreparedGaussiansWithDeformMlpTensorTailFourCtaOverlapCUDA",
            binding_header,
        )
        self.assertIn(
            "def render_prepared_with_deform_mlp_tensor_tail_four_cta_overlap(",
            python_api,
        )
        self.assertIn("TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V8", pipeline)

    def test_v9_binding_capability_and_opt_in_route_are_exposed(self):
        extension = source("ext.cpp")
        binding_header = source("rasterize_points.h")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        pipeline = source("../../gaussian_renderer/tacker_pipeline.py")
        for required in (
            "render_prepared_gaussians_with_deform_mlp_tensor_tail_"
            "balanced_overlap",
            "mixed_render_deform_mlp_tensor_tail_balanced_abi",
            "prepared_raster_deform_mlp_tensor_tail_balanced_overlap",
        ):
            self.assertIn(required, extension)
        self.assertIn(
            "RenderPreparedGaussiansWithDeformMlpTensorTailBalancedOverlapCUDA",
            binding_header,
        )
        self.assertIn(
            "def render_prepared_with_deform_mlp_tensor_tail_balanced_overlap(",
            python_api,
        )
        self.assertIn("TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V9", pipeline)

    def test_v10_v11_bindings_and_opt_in_routes_are_exposed(self):
        extension = source("ext.cpp")
        binding_header = source("rasterize_points.h")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        pipeline = source("../../gaussian_renderer/tacker_pipeline.py")
        for suffix, abi in (
            ("raster_optimized", "V10"),
            ("raster_fast", "V11"),
        ):
            self.assertIn(
                "render_prepared_gaussians_with_deform_mlp_tensor_tail_"
                f"{suffix}_overlap",
                extension,
            )
            self.assertIn(
                f"TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_{abi}", pipeline
            )
        self.assertIn(
            "RenderPreparedGaussiansWithDeformMlpTensorTailRasterOptimizedOverlapCUDA",
            binding_header,
        )
        self.assertIn(
            "RenderPreparedGaussiansWithDeformMlpTensorTailRasterFastOverlapCUDA",
            binding_header,
        )
        self.assertIn(
            "def render_prepared_with_deform_mlp_tensor_tail_raster_optimized_overlap(",
            python_api,
        )
        self.assertIn(
            "def render_prepared_with_deform_mlp_tensor_tail_raster_fast_overlap(",
            python_api,
        )

    def test_v8_subgroup_diagnostic_is_separate_and_mode_selectable(self):
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        header = source("cuda_rasterizer/tacker_mixed.h")
        extension = source("ext.cpp")
        binding_header = source("rasterize_points.h")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        symbol = (
            "tacker_mix_render_deform_mlp_tensor_tail_"
            "four_cta_v8_diagnostic"
        )
        self.assertIn(symbol, cuda)
        self.assertIn("diagnostic_mode is 0=both", cuda)
        self.assertIn("launchMixedRenderDeformMlpTensorTailFourCtaDiagnostic", header)
        self.assertIn(
            "RenderPreparedGaussiansWithDeformMlpTensorTailFourCtaDiagnosticCUDA",
            binding_header,
        )
        self.assertIn(
            "prepared_raster_deform_mlp_tensor_tail_four_cta_diagnostic",
            extension,
        )
        self.assertIn(
            "def profile_prepared_deform_mlp_tensor_tail_four_cta_overlap(",
            python_api,
        )
        for mode in ('"both"', '"raster_only"', '"deform_only"'):
            self.assertIn(mode, python_api)

    def test_v10_v11_balanced_subgroup_diagnostic_is_separate(self):
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        header = source("cuda_rasterizer/tacker_mixed.h")
        extension = source("ext.cpp")
        binding_header = source("rasterize_points.h")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        pipeline = source("../../gaussian_renderer/tacker_pipeline.py")
        for symbol in (
            "tacker_mix_render_deform_mlp_tensor_tail_"
            "raster_optimized_v10_diagnostic",
            "tacker_mix_render_deform_mlp_tensor_tail_"
            "raster_fast_v11_diagnostic",
        ):
            self.assertIn(symbol, cuda)
        self.assertIn(
            "launchMixedRenderDeformMlpTensorTailRasterDiagnostic", header
        )
        self.assertIn(
            "RenderPreparedGaussiansWithDeformMlpTensorTailRasterDiagnosticCUDA",
            binding_header,
        )
        self.assertIn(
            "prepared_raster_deform_mlp_tensor_tail_raster_diagnostic",
            extension,
        )
        self.assertIn(
            "def profile_prepared_deform_mlp_tensor_tail_raster_overlap(",
            python_api,
        )
        self.assertIn(
            "TACKER_PROFILE_MIXED_BALANCED_SUBGROUP_MODE", pipeline
        )
        self.assertIn(
            "TACKER_PROFILE_MIXED_BALANCED_SUBGROUP_JSONL", pipeline
        )
        self.assertIn(
            "deform_tiles <= raster_blocks", cuda
        )

    def test_v12_split_raster_schedule_has_disjoint_tile_ranges(self):
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        header = source("cuda_rasterizer/tacker_mixed.h")
        extension = source("ext.cpp")
        binding_header = source("rasterize_points.h")
        python_api = source("diff_gaussian_rasterization/__init__.py")
        pipeline = source("../../gaussian_renderer/tacker_pipeline.py")
        manifest = json.loads(
            source(
                "abi/"
                "tacker_mixed_render_deform_mlp_tensor_tail_split_raster_v12.json"
            )
        )
        self.assertIn("kMixedDeformMlpTensorTailSplitRasterAbiVersion = 12", header)
        self.assertIn("tacker_render_raster_fast_range_v12", cuda)
        self.assertIn(
            "TACKER_MIXED_V12_PREPASS_PHYSICAL_BLOCKS", cuda
        )
        self.assertIn("splitRasterPrefixPhysicalBlocks", cuda)
        self.assertIn(
            "tacker_mix_render_deform_mlp_tensor_tail_split_raster_v12", cuda
        )
        mixed_helper = cuda[
            cuda.index("tackerMixedBalancedRasterOptimizedDevice") :
            cuda.index("// ABI-v10 preserves")
        ]
        self.assertIn(
            "raster_grid_x,\n\t\traster_grid_y,\n\t\traster_start,\n"
            "\t\tphysical_blocks,\n\t\traster_end,",
            mixed_helper,
        )
        self.assertIn("launchRasterFastRange", header)
        self.assertIn(
            "launchMixedRenderDeformMlpTensorTailSplitRaster", header
        )
        self.assertIn(
            "RenderPreparedGaussiansRasterFastRangeCUDA", binding_header
        )
        self.assertIn(
            "RenderPreparedGaussiansWithDeformMlpTensorTailSplitRasterOverlapCUDA",
            binding_header,
        )
        self.assertIn(
            '"render_prepared_gaussians_raster_fast_range"', extension
        )
        self.assertIn(
            "def render_prepared_raster_fast_range(", python_api
        )
        self.assertIn(
            "TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12", pipeline
        )
        self.assertIn(
            "launch_prepared_raster_prefix", pipeline
        )
        self.assertLess(
            pipeline.index("launch_prepared_raster_prefix", pipeline.index("def _enqueue_prepared_mixed")),
            pipeline.index("wait_event(next_slot.prefix_ready)", pipeline.index("def _enqueue_prepared_mixed")),
        )
        self.assertEqual(manifest["abi_version"], 12)
        self.assertEqual(
            manifest["work_mapping"]["tile_coverage"],
            "each_raster_tile_exactly_once",
        )
        self.assertEqual(
            manifest["subgroups"]["raster_prefix"]["threads"], 256
        )
        self.assertEqual(
            manifest["subgroups"]["deform_mlp"]["threads"], 128
        )
        self.assertEqual(
            capability_sha256(
                extension,
                "mixed_deform_mlp_tensor_tail_split_raster_manifest_sha256",
            ),
            manifest_sha256(
                "abi/"
                "tacker_mixed_render_deform_mlp_tensor_tail_split_raster_v12.json"
            ),
        )

    def test_v2_resources_include_register_smem_and_occupancy(self):
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        header = source("cuda_rasterizer/tacker_mixed.h")
        for required in (
            "cudaFuncGetAttributes",
            "attributes.numRegs",
            "attributes.sharedSizeBytes",
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
            "launch_supported",
        ):
            self.assertIn(required, cuda + header)
        self.assertIn("cudaPeekAtLastError", cuda)
        self.assertIn("(void)cudaGetLastError()", cuda)
        self.assertNotIn("cudaDeviceSynchronize", cuda)

    def test_v1_symbol_binding_and_manifest_are_preserved_with_v2(self):
        cuda = source("cuda_rasterizer/tacker_mixed.cu")
        extension = source("ext.cpp")
        manifest = json.loads(source("abi/tacker_mixed_render_head_v1.json"))
        self.assertIn("tacker_mix_render_head_v1", cuda)
        self.assertIn('"rasterize_gaussians_with_head"', extension)
        self.assertEqual(manifest["abi_version"], 1)
        self.assertEqual(
            manifest["global_kernel_symbol"], "tacker_mix_render_head_v1"
        )

    def test_inference_guard_and_no_backward_registration(self):
        python_api = source("diff_gaussian_rasterization/__init__.py")
        extension = source("ext.cpp")
        self.assertIn("torch.is_grad_enabled()", python_api)
        self.assertIn("call it under torch.no_grad()", python_api)
        self.assertNotIn(
            'm.def("rasterize_gaussians_with_head_backward"', extension
        )

    def test_cpp_checks_head_contract_and_int32_rows(self):
        binding = source("rasterize_points.cu")
        for required in (
            "head_input must be contiguous",
            "head_weight must be contiguous",
            "head_bias must be contiguous",
            "head_input must be float16",
            "head_weight must be float16",
            "head_bias must be float32",
            "head_input must have shape [N, 128]",
            "head_weight must have shape [128, 128]",
            "head_input row count exceeds the int32 mixed ABI",
            "head_input data pointer must be natively 32-byte aligned",
            "head_weight data pointer must be natively 32-byte aligned",
            "head_output data pointer must be natively 32-byte aligned",
        ):
            self.assertIn(required, binding)

    def test_cpp_validates_complete_raster_contract_and_empty_shape_tags(self):
        binding = source("rasterize_points.cu")
        for required in (
            "background must have shape [3]",
            "opacity must have shape [P, 1]",
            "viewmatrix must have shape [4, 4]",
            "projmatrix must have shape [4, 4]",
            "campos must have shape [3]",
            "provide exactly one of sh [P, M, 3] or colors [P, 3]",
            "provide exactly one of cov3D_precomp [P, 6]",
            "image_height must be > 0",
            "image_width must be > 0",
            "image dimensions exceed the int32 Raster color-indexing ABI",
            "degree must be in [0, 3]",
            "sh coefficient count is too small for degree",
        ):
            self.assertIn(required, binding)
        self.assertIn("bool isPlaceholder", binding)
        self.assertIn("sh.dim() == 3", binding)
        self.assertIn("colors.dim() == 2", binding)
        self.assertNotIn("sh_contiguous.numel() == 0", binding)

    def test_legacy_forward_backward_and_visibility_use_current_stream(self):
        binding = source("rasterize_points.cu")
        rasterizer_header = source("cuda_rasterizer/rasterizer.h")
        rasterizer_impl = source("cuda_rasterizer/rasterizer_impl.cu")
        backward_header = source("cuda_rasterizer/backward.h")
        backward_cuda = source("cuda_rasterizer/backward.cu")
        self.assertGreaterEqual(binding.count("c10::cuda::CUDAGuard"), 4)
        self.assertGreaterEqual(
            binding.count("at::cuda::getCurrentCUDAStream().stream()"), 4
        )
        self.assertIn("cudaStream_t stream = nullptr", rasterizer_header)
        self.assertIn("dL_ddepth,\n\t\tstream), debug)", rasterizer_impl)
        self.assertIn("cudaStream_t stream = nullptr", backward_header)
        preprocess_kernel_section = backward_cuda.split(
            "void BACKWARD::preprocess", 1
        )[0]
        self.assertNotIn("cudaStream_t stream", preprocess_kernel_section)
        self.assertIn("glm::vec4* dL_drot,\n\tcudaStream_t stream)", backward_cuda)
        self.assertIn("256, 0, stream", backward_cuda)
        self.assertIn("grid, block, 0, stream", backward_cuda)

    def test_rendered_count_cuda_api_errors_are_checked_without_debug(self):
        implementation = source("cuda_rasterizer/rasterizer_impl.cu")
        self.assertIn("rendered_copy_status = cudaMemcpyAsync", implementation)
        self.assertIn("rendered_copy_status != cudaSuccess", implementation)
        self.assertIn("rendered_sync_status = cudaStreamSynchronize", implementation)
        self.assertIn("rendered_sync_status != cudaSuccess", implementation)
        self.assertIn("uint32_t* value", implementation)
        self.assertIn("num_rendered_u32", implementation)
        self.assertIn(
            "rendered Gaussian instance count exceeds the int32 Raster ABI",
            implementation,
        )

    def test_zero_gaussian_path_still_launches_head_wrapper(self):
        binding = source("rasterize_points.cu")
        self.assertIn("if (P != 0)", binding)
        self.assertIn("dim3(0, 0, 1)", binding)
        self.assertIn("Preserve the legacy P==0 raster outputs", binding)

    def test_setup_uses_absolute_paths_and_compiles_mixed_source(self):
        setup = source("setup.py")
        self.assertIn("Path(__file__).resolve().parent", setup)
        self.assertIn('"tacker_mixed.cu"', setup)
        self.assertIn("TACKER_4DGS_HEAD_INCLUDE", setup)
        self.assertIn("head_linear_v2_device.cuh", setup)
        self.assertIn('os.environ.get("TACKER_CUDA_ARCH", "sm_86")', setup)
        self.assertIn('os.environ["TORCH_CUDA_ARCH_LIST"] = "{}.{}"', setup)
        self.assertIn('"-Xptxas=-v"', setup)


if __name__ == "__main__":
    unittest.main()
