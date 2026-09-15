"""CPU-only contract for the one-click sealed Phase-4 qualification."""

from pathlib import Path
import stat
import subprocess
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_tacker_qualification.sh"


class TackerQualificationScriptContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT_PATH.read_text(encoding="utf-8")

    def test_script_is_executable_and_has_valid_bash_syntax(self):
        self.assertTrue(SCRIPT_PATH.stat().st_mode & stat.S_IXUSR)
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT_PATH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrapper_is_fail_closed_and_delegates_to_phase4_runner(self):
        self.assertIn("set -euo pipefail", self.source)
        self.assertIn(
            'exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_tacker_phase4.py"',
            self.source,
        )
        self.assertNotRegex(self.source, r"(?m)^\s*ssh(?:\s|$)")
        self.assertNotIn("benchmark_tacker_admission.py", self.source)
        self.assertNotIn("profile_render.py", self.source)
        self.assertNotIn("validate_tacker_modes.py", self.source)

    def test_canonical_sealed_inputs_and_a6000_environment_are_explicit(self):
        required = (
            "/data/qyfeng/tacker_phase31_validation/20260913-codex-phase31/run-complete-v3",
            "/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak",
            "/data/qyfeng/datasets/n3dv/flame_steak",
            "/usr/local/cuda-12.4",
            "/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python3.10",
            'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"',
            "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
            "export TORCH_CUDA_ARCH_LIST=8.6",
            '--expected-gpu-name "NVIDIA RTX A6000"',
            "--expected-python 3.10",
            "--expected-torch 2.4.1",
            "--expected-cuda 12.4",
            "--gpu 0",
            '--physical-gpu "${CUDA_VISIBLE_DEVICES}"',
        )
        for value in required:
            self.assertIn(value, self.source)

    def test_primary_and_sequence_contracts_are_explicit(self):
        for value in (
            "--iteration 14000",
            "--image-width 1352",
            "--image-height 1014",
            "--gaussian-count 111525",
            "--leaf-views 2",
            "--leaf-warmup 5",
            "--leaf-repetitions 50",
            "--sequence-trials 3",
            "--generalization-trials 3",
            "--long-frames 500",
        ):
            self.assertIn(value, self.source)

    def test_exactly_two_distinct_generalization_workloads_are_bound(self):
        self.assertEqual(self.source.count('--generalization-workload "${'), 2)
        for value in (
            "flame_steak_iteration_3000_native",
            '"iteration":3000',
            '"image_width":1352,"image_height":1014',
            '"gaussian_count":92999',
            '"raster_deformation_mix":"raster_heavy"',
            "flame_steak_iteration_14000_scale4",
            '"iteration":14000',
            '"image_width":338,"image_height":254',
            '"gaussian_count":111525',
            '"raster_deformation_mix":"deformation_heavy"',
            '["--resolution","4"]',
        ):
            self.assertIn(value, self.source)

    def test_wrapper_does_not_generate_or_overwrite_profiles(self):
        for forbidden in (
            "tacker_autotune.py",
            "run_tacker_phase3.py",
            "run_tacker_phase31.py",
            "generate_tacker_phase2_profiles.py",
            "ADMITTED_PROFILE=",
        ):
            self.assertNotIn(forbidden, self.source)
        self.assertNotRegex(self.source, r"(?m)^\s*(?:cp|mv|rm)\s")


if __name__ == "__main__":
    unittest.main()
