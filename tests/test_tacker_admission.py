"""CPU-only tests for the fail-closed Tacker admission generator."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "benchmark_tacker_admission.py"
SPEC = importlib.util.spec_from_file_location("benchmark_tacker_admission", MODULE_PATH)
ADMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADMISSION)


def _read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


MODEL_PATH = "/data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak"
SOURCE_PATH = "/data/qyfeng/datasets/n3dv/flame_steak_4dgs_min"
VIEW_INDICES = [0, 1, 2, 3]
TEMPLATE_PROFILE = _read_json(
    PROJECT_ROOT / "tacker_profiles" / "raster_head_sm86.json"
)
PROFILE_PERSISTENT_BLOCKS = TEMPLATE_PROFILE["manifest"]["persistent_blocks"]


def _workload():
    return {
        "scene": "flame_steak",
        "iteration": 14000,
        "resolution": [1352, 1014],
        "gaussian_count": 111525,
        "split": "test",
        "model_path": MODEL_PATH,
        "source_path": SOURCE_PATH,
        "current_view_indices": [0, 1],
        "next_view_indices": [1, 2],
    }


def _leaf_context(kind, measurements):
    return {
        "schema_version": 1,
        "kind": kind,
        "passed": True,
        "workload": _workload(),
        "device": {"name": "NVIDIA RTX A6000"},
        "measurements": measurements,
        "measurement_config": {
            "persistent_blocks": PROFILE_PERSISTENT_BLOCKS
        },
        "numerics": {"passed": True},
    }


def _render_context(mode, frame_ms):
    return {
        "schema_version": 1,
        "kind": "4dgaussians_tacker_render_profile",
        "passed": True,
        "workload_name": "flame_steak",
        "iteration": 14000,
        "image_width": 1352,
        "image_height": 1014,
        "gaussian_count": 111525,
        "device": {"name": "NVIDIA RTX A6000"},
        "measurements": {"p50_frame_ms": frame_ms},
        "actual_execution_mode": mode,
        "split": "test",
        "warmup_frames": 10,
        "profile_frames": len(VIEW_INDICES),
        "view_indices": list(VIEW_INDICES),
        "model_path": MODEL_PATH,
        "source_path": SOURCE_PATH,
        "timing_method": "perf_counter_with_cuda_synchronize",
        "frame_timing_method": "cuda_event_consumer_completion_intervals",
        "io_in_timed_region": False,
    }


def passing_inputs():
    template = copy.deepcopy(TEMPLATE_PROFILE)
    tacker = _render_context("tacker", 23.5)
    tacker["persistent_blocks"] = PROFILE_PERSISTENT_BLOCKS
    tacker["profile_manifest_sha256"] = template["manifest_sha256"]
    return {
        "device": {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_device",
            "passed": True,
            "workload": _workload(),
            "device": {
                "name": "NVIDIA RTX A6000",
                "compute_capability": [8, 6],
                "cuda_arch": "sm_86",
            },
            "extensions": {
                "rasterizer": {
                    "capabilities": dict(ADMISSION.EXPECTED_RASTER_CAPABILITIES),
                    "cuda_global_symbols": [ADMISSION.EXPECTED_MIXED_SYMBOL],
                },
                "head": {
                    "capabilities": dict(ADMISSION.EXPECTED_HEAD_CAPABILITIES),
                    "cuda_global_symbols": [
                        ADMISSION.EXPECTED_HEAD_SOLO_SYMBOL,
                        ADMISSION.EXPECTED_HEAD_GPTB_SYMBOL,
                    ],
                },
            },
        },
        "quality": {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_quality_validation",
            "passed": True,
            "workload": dict(
                _workload(),
                split="test",
                frames=len(VIEW_INDICES),
                view_indices=list(VIEW_INDICES),
                model_path=MODEL_PATH,
                source_path=SOURCE_PATH,
            ),
            "device": {"name": "NVIDIA RTX A6000"},
            "modes": {
                "serial": {"actual_mode": "serial"},
                "two_stream": {"actual_mode": "two_stream"},
                "tacker": {"actual_mode": "tacker"},
            },
            "deltas": {
                "tacker": {
                    "psnr_drop_db": 0.01,
                    "ssim_drop": 0.00001,
                    "lpips_increase": 0.00001,
                }
            },
        },
        "raster": _leaf_context(
            "4dgaussians_tacker_raster_profile",
            {
                "solo_raster_p50_ms": 8.0,
                "mixed_raster_p50_ms": 8.2,
            }
        ),
        "leaf": _leaf_context(
            "4dgaussians_tacker_leaf_profile",
            {
                "mixed_p50_ms": 8.6,
                "solo_raster_p50_ms": 8.0,
                "solo_head_p50_ms": 0.8,
            }
        ),
        "two_stream": _render_context("two_stream", 24.0),
        "tacker": tacker,
        "mixed_abi": _read_json(
            PROJECT_ROOT
            / "submodules"
            / "depth-diff-gaussian-rasterization"
            / "abi"
            / "tacker_mixed_render_head_v1.json"
        ),
        "head_abi": _read_json(
            PROJECT_ROOT / "tacker_ext" / "abi" / "head_linear_v1.json"
        ),
        "template": template,
    }


class AdmissionTests(unittest.TestCase):
    def test_all_boundaries_pass_and_output_runtime_schema(self):
        inputs = passing_inputs()
        inputs["quality"]["deltas"]["tacker"] = {
            "psnr_drop_db": 0.05,
            "ssim_drop": 0.0001,
            "lpips_increase": 0.0001,
        }
        inputs["raster"]["measurements"] = {
            "solo_raster_p50_ms": 20.0,
            "mixed_raster_p50_ms": 21.0,
        }
        inputs["leaf"]["measurements"]["solo_raster_p50_ms"] = 20.0
        inputs["two_stream"]["measurements"]["p50_frame_ms"] = 24.0
        inputs["tacker"]["measurements"]["p50_frame_ms"] = 24.0

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertTrue(report["passed"])
        self.assertIsNotNone(profile)
        self.assertEqual(profile["schema_version"], 1)
        self.assertEqual(profile["admission"], {"enabled": True, "valid": True})
        self.assertEqual(profile["measurements"]["raster_slowdown_pct"], 5.0)
        self.assertEqual(
            profile["manifest_sha256"],
            ADMISSION.manifest_sha256(profile["manifest"]),
        )
        self.assertEqual(profile["manifest"]["workload"], "flame_steak")
        self.assertEqual(profile["manifest"], inputs["template"]["manifest"])
        self.assertEqual(
            profile["provenance"]["validated_abi"]["mixed_abi_version"], 1
        )
        self.assertEqual(profile["manifest"]["physical_cta_threads"], 384)
        self.assertEqual(profile["manifest"]["raster_named_barrier_id"], 1)
        self.assertEqual(
            set(profile["measurements"]),
            {
                "raster_slowdown_pct",
                "mixed_p50_ms",
                "solo_raster_p50_ms",
                "solo_head_p50_ms",
                "tacker_end_to_end_p50_ms",
                "two_stream_end_to_end_p50_ms",
                "psnr_drop_db",
                "ssim_drop",
                "lpips_increase",
            },
        )
        # Cross-check the produced schema against the actual runtime validator,
        # loaded with the repository's CPU-only CUDA stubs.
        from tests.test_tacker_pipeline import _load_module

        runtime = _load_module()
        self.assertIsNone(runtime.tacker_profile_admission_reason(profile))

    def test_missing_field_fails_closed(self):
        inputs = passing_inputs()
        del inputs["leaf"]["measurements"]["solo_head_p50_ms"]

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("solo_head_p50_ms", report["errors"][0])

    def test_nan_fails_closed(self):
        inputs = passing_inputs()
        inputs["quality"]["deltas"]["tacker"]["ssim_drop"] = float("nan")

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("finite", report["errors"][0])

    def test_wrong_device_fails_closed(self):
        inputs = passing_inputs()
        inputs["device"]["device"]["name"] = "NVIDIA RTX 4090"
        inputs["device"]["device"]["compute_capability"] = [8, 9]

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("RTX A6000", report["errors"][0])

    def test_device_requires_compiled_extension_provenance(self):
        inputs = passing_inputs()
        del inputs["device"]["extensions"]["head"]["capabilities"]

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("head capabilities", report["errors"][0])

    def test_every_measurement_requires_complete_fixed_workload(self):
        for document_name in (
            "quality",
            "raster",
            "leaf",
            "two_stream",
            "tacker",
        ):
            for field in ("resolution", "gaussian_count"):
                inputs = passing_inputs()
                document = inputs[document_name]
                if document_name in ("two_stream", "tacker"):
                    target_field = (
                        "image_width" if field == "resolution" else field
                    )
                    del document[target_field]
                else:
                    del document["workload"][field]

                report, profile = ADMISSION.evaluate_admission(inputs)

                with self.subTest(document=document_name, field=field):
                    self.assertFalse(report["passed"])
                    self.assertIsNone(profile)

    def test_leaf_and_raster_must_be_successful_profiler_documents(self):
        for document_name in ("raster", "leaf"):
            for mutation in (
                {"passed": False},
                {"schema_version": 999},
                {"kind": "wrong_kind"},
                {"numerics": {"passed": False}},
            ):
                inputs = passing_inputs()
                inputs[document_name].update(mutation)

                report, profile = ADMISSION.evaluate_admission(inputs)

                with self.subTest(document=document_name, mutation=mutation):
                    self.assertFalse(report["passed"])
                    self.assertIsNone(profile)

    def test_end_to_end_sampling_contract_must_match_quality(self):
        mutations = (
            ("split", "train"),
            ("warmup_frames", 0),
            ("profile_frames", 3),
            ("view_indices", [0, 2, 3, 4]),
            ("model_path", "/different/model"),
            ("source_path", "/different/source"),
        )
        for field, value in mutations:
            inputs = passing_inputs()
            inputs["tacker"][field] = value
            if field == "profile_frames":
                inputs["tacker"]["view_indices"] = [0, 1, 2]

            report, profile = ADMISSION.evaluate_admission(inputs)

            with self.subTest(field=field):
                self.assertFalse(report["passed"])
                self.assertIsNone(profile)

        inputs = passing_inputs()
        for document_name in ("two_stream", "tacker"):
            inputs[document_name]["split"] = "train"
        report, profile = ADMISSION.evaluate_admission(inputs)
        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("quality and end-to-end split", report["errors"][0])

    def test_persistent_blocks_and_profile_hash_are_bound_to_measurements(self):
        for document_name, field, value in (
            ("raster", "measurement_config", {"persistent_blocks": 17}),
            ("leaf", "measurement_config", {"persistent_blocks": 17}),
            ("tacker", "persistent_blocks", 17),
            ("tacker", "profile_manifest_sha256", "0" * 64),
        ):
            inputs = passing_inputs()
            inputs[document_name][field] = value

            report, profile = ADMISSION.evaluate_admission(inputs)

            with self.subTest(document=document_name, field=field):
                self.assertFalse(report["passed"])
                self.assertIsNone(profile)

    def test_leaf_workload_provenance_must_match_quality_and_device(self):
        inputs = passing_inputs()
        for document_name in ("device", "raster", "leaf"):
            inputs[document_name]["workload"].update(
                {
                    "split": "video",
                    "model_path": "/different/model",
                    "source_path": "/different/source",
                    "current_view_indices": [70, 71],
                    "next_view_indices": [71, 72],
                }
            )

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("leaf and quality split", report["errors"][0])

    def test_leaf_view_pair_provenance_is_structurally_validated(self):
        inputs = passing_inputs()
        inputs["leaf"]["workload"]["next_view_indices"] = [0, 2]

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("ordered non-negative view pairs", report["errors"][0])

    def test_missing_head_capability_query_fails_closed(self):
        inputs = passing_inputs()
        del inputs["head_abi"]["capability_query"]

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("capability_query", report["errors"][0])

    def test_inconsistent_solo_raster_measurement_fails_closed(self):
        inputs = passing_inputs()
        inputs["leaf"]["measurements"]["solo_raster_p50_ms"] = 7.9

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("same solo Raster", report["errors"][0])

    def test_mixed_leaf_is_strict(self):
        inputs = passing_inputs()
        inputs["leaf"]["measurements"].update(
            {
                "mixed_p50_ms": 8.8,
                "solo_raster_p50_ms": 8.0,
                "solo_head_p50_ms": 0.8,
            }
        )

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        gate = next(
            item for item in report["gates"] if item["name"] == "mixed_leaf_speedup"
        )
        self.assertFalse(gate["passed"])

    def test_failure_writes_report_without_touching_profile_target(self):
        inputs = passing_inputs()
        inputs["tacker"]["measurements"]["p50_frame_ms"] = 25.0
        report, profile = ADMISSION.evaluate_admission(inputs)
        self.assertIsNone(profile)

        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary = Path(temporary_dir)
            report_path = temporary / "report.json"
            profile_path = temporary / "enabled.json"
            sentinel = {"admission": {"enabled": False, "valid": False}}
            profile_path.write_text(json.dumps(sentinel), encoding="utf-8")

            written = ADMISSION.write_admission_outputs(
                report, profile, report_path, profile_path
            )

            self.assertFalse(written["passed"])
            self.assertFalse(written["enabled_profile_written"])
            self.assertEqual(_read_json(profile_path), sentinel)
            self.assertFalse(_read_json(report_path)["passed"])

    def test_passing_outputs_are_atomic_and_machine_readable(self):
        report, profile = ADMISSION.evaluate_admission(passing_inputs())
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary = Path(temporary_dir)
            report_path = temporary / "report.json"
            profile_path = temporary / "profiles" / "enabled.json"

            written = ADMISSION.write_admission_outputs(
                report, profile, report_path, profile_path
            )

            report_json = _read_json(report_path)
            profile_json = _read_json(profile_path)
            self.assertTrue(written["enabled_profile_written"])
            self.assertTrue(report_json["enabled_profile_written"])
            self.assertTrue(profile_json["admission"]["enabled"])
            self.assertEqual(
                profile_json["manifest_sha256"],
                ADMISSION.manifest_sha256(profile_json["manifest"]),
            )

    def test_bad_barrier_manifest_fails_closed(self):
        inputs = passing_inputs()
        inputs["mixed_abi"] = copy.deepcopy(inputs["mixed_abi"])
        inputs["mixed_abi"]["subgroups"]["raster"]["named_barrier_id"] = 2

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("barrier", report["errors"][0])

    def test_explicit_qualification_evidence_is_traceable(self):
        inputs = passing_inputs()
        inputs["quality"]["qualification"] = {
            "enabled": True,
            "admission_claimed": False,
        }
        inputs["quality"]["modes"]["tacker"][
            "qualification_executed"
        ] = True

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertTrue(report["passed"])
        self.assertTrue(report["derived"]["quality_was_qualification_run"])
        self.assertTrue(
            profile["provenance"]["quality_was_qualification_run"]
        )

    def test_qualification_claim_without_execution_fails_closed(self):
        inputs = passing_inputs()
        inputs["quality"]["qualification"] = {
            "enabled": True,
            "admission_claimed": False,
        }
        inputs["quality"]["modes"]["tacker"][
            "qualification_executed"
        ] = False

        report, profile = ADMISSION.evaluate_admission(inputs)

        self.assertFalse(report["passed"])
        self.assertIsNone(profile)
        self.assertIn("physical qualification", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
