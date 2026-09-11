"""CPU tests for deterministic Phase-2 qualification profile generation."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from tests.test_tacker_pipeline import _load_module


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_tacker_phase2_profiles",
    ROOT / "scripts" / "generate_tacker_phase2_profiles.py",
)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class Phase2ProfileGeneratorTest(unittest.TestCase):
    def test_emits_c0_all_c1_and_one_c2_with_runtime_resources(self):
        module = _load_module()
        template = module.load_tacker_profile()
        calls = []

        def resources(abi_version, worker_groups):
            calls.append((abi_version, worker_groups))
            return {
                "physical_threads": 256 + 128 * worker_groups,
                "registers_per_thread": 32 + worker_groups,
                "static_shared_bytes": 80,
                "kernel_max_threads_per_block": 896,
                "active_blocks_per_multiprocessor": 1,
                "occupancy": 0.25,
                "launch_supported": True,
            }

        profiles = GENERATOR.build_profiles(
            module, template, persistent_blocks=80, resource_provider=resources
        )

        self.assertEqual(len(profiles), 7)
        self.assertEqual(calls, [(1, 1), (2, 1), (2, 2)])
        self.assertTrue(any(name.startswith("c0_") for name in profiles))
        self.assertEqual(
            len([name for name in profiles if name.startswith("c1_")]), 5
        )
        self.assertEqual(
            len([name for name in profiles if name.startswith("c2_")]), 1
        )
        for variant_id, profile in profiles.items():
            self.assertEqual(profile["selected_variant_id"], variant_id)
            self.assertEqual(profile["manifest"]["persistent_blocks"], 80)
            self.assertEqual(profile["deployment"], {"enabled": False, "valid": False})
            selected = next(
                candidate
                for candidate in profile["candidates"]
                if candidate["variant_id"] == variant_id
            )
            self.assertNotIn("launch_supported", selected["resources"])
            self.assertIs(module.validate_tacker_profile(profile), profile)

        c0 = next(
            profile for name, profile in profiles.items() if name.startswith("c0_")
        )
        self.assertEqual(
            c0["provenance"]["input_sha256"],
            {
                "mixed_abi": module.MIXED_ABI_SHA256,
                "head_abi": GENERATOR.LEGACY_HEAD_ABI_SHA256,
            },
        )
        for name, profile in profiles.items():
            if name.startswith(("c1_", "c2_")):
                self.assertEqual(
                    profile["provenance"]["input_sha256"],
                    {
                        "mixed_abi": module.MIXED_MULTI_ABI_SHA256,
                        "head_abi": module.HEAD_MULTI_ABI_SHA256,
                    },
                )

    def test_resource_query_must_explicitly_report_launch_support(self):
        base = {
            "physical_threads": 384,
            "registers_per_thread": 32,
            "static_shared_bytes": 0,
            "kernel_max_threads_per_block": 1024,
            "active_blocks_per_multiprocessor": 1,
        }
        for value in (None, False):
            resources = dict(base)
            if value is not None:
                resources["launch_supported"] = value
            with self.subTest(launch_supported=value), self.assertRaisesRegex(
                ValueError, "launch-supported"
            ):
                GENERATOR._normalise_resources(resources)

    def test_direct_script_bootstrap_adds_project_root(self):
        source = (ROOT / "scripts" / "generate_tacker_phase2_profiles.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("sys.path.insert(0, str(PROJECT_ROOT))", source)

    def test_atomic_writer_replaces_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            GENERATOR._atomic_write_json(path, {"variant": "c1_pos"})
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{\n  "variant": "c1_pos"\n}\n',
            )


if __name__ == "__main__":
    unittest.main()
