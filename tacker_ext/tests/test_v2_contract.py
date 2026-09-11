import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ExtensionV2ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = json.loads(
            (ROOT / "abi" / "head_linear_v1.json").read_text(encoding="utf-8")
        )
        cls.v2 = json.loads(
            (ROOT / "abi" / "head_linear_v2.json").read_text(encoding="utf-8")
        )
        cls.device = (ROOT / "include" / "head_linear_v2_device.cuh").read_text(
            encoding="utf-8"
        )
        cls.kernels = (ROOT / "include" / "head_linear_v2_kernels.cuh").read_text(
            encoding="utf-8"
        )
        cls.cuda = (ROOT / "csrc" / "head_linear_v2.cu").read_text(
            encoding="utf-8"
        )
        cls.bindings = (ROOT / "csrc" / "bindings.cpp").read_text(
            encoding="utf-8"
        )

    def test_v1_manifest_and_symbols_remain_independent(self):
        self.assertEqual(self.v1["abi_version"], 1)
        self.assertEqual(self.v2["abi_version"], 2)
        self.assertEqual(self.v2["legacy_abi"]["manifest"], "abi/head_linear_v1.json")
        for symbol in (
            "tacker_head_linear_solo_v1",
            "tacker_head_linear_gptb_v1",
        ):
            self.assertIn(symbol, self.v2["legacy_abi"]["preserved_symbols"])
            self.assertIn(symbol, (ROOT / "csrc" / "head_linear.cu").read_text())
        self.assertEqual(self.v2["capability_query"], "tacker_capabilities_v2")
        self.assertIn('"tacker_capabilities"', self.bindings)
        self.assertIn('"tacker_capabilities_v2"', self.bindings)

    def test_descriptor_layout_is_frozen_in_manifest_and_static_asserts(self):
        descriptor = self.v2["head_linear_task_v2"]
        self.assertEqual(descriptor["size_bytes"], 40)
        self.assertEqual(
            [(field["name"], field["offset_bytes"]) for field in descriptor["fields"]],
            [("input", 0), ("weight", 8), ("bias", 16), ("output", 24), ("rows", 32)],
        )
        self.assertIn("struct HeadLinearTaskV2", self.device)
        self.assertIn("sizeof(HeadLinearTaskV2) == 40", self.cuda)
        self.assertIn("offsetof(HeadLinearTaskV2, rows) == 32", self.cuda)

    def test_multi_adapter_supports_one_to_five_and_worker_groups(self):
        limits = self.v2["limits"]
        self.assertEqual(limits["max_head_tasks"], 5)
        self.assertEqual(limits["worker_group_threads"], 128)
        self.assertEqual(limits["max_backend_threads"], 640)
        adapter = self.v2["multi_first_linear"]["device_adapter"]
        self.assertEqual(
            adapter["symbol"], "tacker_4dgs::head_linear_multi_gptb_device"
        )
        self.assertEqual(
            adapter["argument_order"],
            [
                "tasks",
                "task_count",
                "worker_groups",
                "ptb_start_block_pos",
                "ptb_iter_block_step",
                "ptb_end_block_pos",
                "thread_base",
            ],
        )
        self.assertIn("head_linear_multi_gptb_device(", self.device)
        self.assertIn("task_index += worker_groups", self.device)
        self.assertIn("worker_groups * kHeadThreads", self.device)
        multi_section = self.device.split("head_linear_multi_gptb_device(", 1)[1]
        multi_section = multi_section.split("head_linear_packed_gptb_device(", 1)[0]
        self.assertNotIn("__syncthreads", multi_section)
        self.assertNotIn("bar.sync", multi_section)

    def test_all_c1_roles_and_one_dual_head_c2_are_manifested(self):
        variants = self.v2["canonical_variants"]
        c1 = [item for item in variants if item["family"] == "C1"]
        self.assertEqual(
            [item["head_roles"][0] for item in c1],
            ["position", "scale", "rotation", "opacity", "sh"],
        )
        c2 = [item for item in variants if item["family"] == "C2"]
        self.assertEqual(len(c2), 1)
        self.assertEqual(c2[0]["head_roles"], ["position", "scale"])
        self.assertEqual(c2[0]["worker_groups"], 2)
        for role in ("position", "scale", "rotation", "opacity", "sh"):
            self.assertIn('std::string("c1_") + role', self.bindings)

    def test_global_symbols_are_declared_defined_and_capability_reported(self):
        for record in self.v2["global_kernel_symbols"].values():
            if not isinstance(record, dict):
                continue
            symbol = record["symbol"]
            self.assertIn("void " + symbol, self.kernels)
            self.assertIn(symbol + "(", self.cuda)
            self.assertIn('"' + symbol + '"', self.bindings)

    def test_packed_and_whole_head_adapters_have_explicit_sync_contracts(self):
        packed = self.v2["packed_first_linear"]["device_adapter"]
        self.assertEqual(
            packed["symbol"], "tacker_4dgs::head_linear_packed_gptb_device"
        )
        self.assertEqual(packed["named_barrier_ids"], [])
        self.assertIn("head_linear_packed_gptb_device(", self.device)
        whole = self.v2["whole_head"]["device_adapter"]
        self.assertEqual(whole["shared_scratch_bytes_per_worker_group"], 512)
        self.assertEqual(whole["recommended_mixed_barrier_ids"], [2, 3, 4, 5, 6])
        self.assertIn("whole_head_multi_gptb_device(", self.device)
        self.assertIn('"bar.sync %0, %1;"', self.device)
        self.assertNotIn("__syncthreads", self.device)

    def test_resource_query_exposes_launch_filter_inputs(self):
        self.assertEqual(self.v2["resource_query"], "tacker_resources_v2")
        self.assertTrue(self.v2["resource_contract"]["query_before_launch"])
        self.assertIn("cudaFuncGetAttributes", self.cuda)
        self.assertIn("cudaOccupancyMaxActiveBlocksPerMultiprocessor", self.cuda)
        self.assertIn("registers_per_thread", self.bindings)
        self.assertIn("static_shared_memory_bytes", self.bindings)
        self.assertIn("max_threads_per_block", self.bindings)
        self.assertIn("active_blocks_per_sm", self.bindings)
        self.assertIn('"tacker_resources_v2"', self.bindings)

    def test_validation_and_native_guards_fail_closed(self):
        for text in (
            "supports at most 5 heads",
            "worker_groups must not exceed head task count",
            "all head tasks must be on the same CUDA device",
            "head outputs must not overlap any input storage",
            "head output storages must not overlap each other",
            "data pointer must be 32-byte aligned",
            "C10_CUDA_KERNEL_LAUNCH_CHECK",
        ):
            self.assertIn(text, self.cuda)

    def test_build_and_python_api_include_v2_without_breaking_python37(self):
        setup = (ROOT / "setup.py").read_text(encoding="utf-8")
        python_api = (ROOT / "tacker_4dgs_head" / "__init__.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('str(ROOT / "csrc" / "head_linear_v2.cu")', setup)
        for name in (
            "head_linear_multi_solo",
            "head_linear_multi_gptb",
            "head_linear_packed_gptb",
            "whole_head_gptb",
            "tacker_capabilities_v2",
            "tacker_resources_v2",
        ):
            self.assertIn('"' + name + '"', self.bindings)
            self.assertIn("def " + name + "(", python_api)
        for path in list((ROOT / "tacker_4dgs_head").glob("*.py")) + list(
            (ROOT / "tests").glob("*.py")
        ):
            ast.parse(path.read_text(encoding="utf-8"), str(path), feature_version=7)


if __name__ == "__main__":
    unittest.main()
