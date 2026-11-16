"""CPU tests for workload-specific qualification profile materialization."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "materialize_tacker_workload_profiles.py"
SPEC = importlib.util.spec_from_file_location("materialize_profiles", MODULE_PATH)
MATERIALIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATERIALIZE)


class WorkloadProfileTests(unittest.TestCase):
    def test_rebinds_only_workload_identity_and_reseals_hashes(self):
        source = {
            "schema_version": 2,
            "workload_key": "old:14000:1:1x1:sm_86",
            "selection_objective": "median_throughput_fps",
            "selected_variant_id": "candidate",
            "manifest": {
                "workload": "old",
                "workload_key": "old:14000:1:1x1:sm_86",
                "iteration": 14000,
                "gaussian_count": 1,
                "resolution": [1, 1],
            },
            "manifest_sha256": "unused-source-digest",
            "correctness_thresholds": {},
            "candidates": [{"variant_id": "candidate"}],
            "selection": None,
            "deployment": {"enabled": False, "valid": False},
            "provenance": {"template": True},
            "profile_sha256": "unused-source-digest",
        }
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            result = MATERIALIZE.materialize_profile(
                source,
                source_path,
                "Coffee-Martini",
                14000,
                98765,
                1352,
                1014,
            )
        self.assertEqual(result["manifest"]["workload"], "coffee_martini")
        self.assertEqual(result["manifest"]["gaussian_count"], 98765)
        self.assertEqual(
            result["workload_key"],
            "coffee_martini:14000:98765:1352x1014:sm_86",
        )
        self.assertEqual(
            result["manifest_sha256"],
            MATERIALIZE.manifest_sha256(result["manifest"]),
        )
        self.assertEqual(
            result["profile_sha256"], MATERIALIZE.profile_sha256(result)
        )
        self.assertEqual(result["candidates"], source["candidates"])
        self.assertEqual(result["deployment"], source["deployment"])

    def test_atomic_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            MATERIALIZE.atomic_write_new(path, {"value": 1})
            with self.assertRaises(MATERIALIZE.MaterializationError):
                MATERIALIZE.atomic_write_new(path, {"value": 2})


if __name__ == "__main__":
    unittest.main()
