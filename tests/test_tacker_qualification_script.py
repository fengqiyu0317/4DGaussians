"""CPU-only source contract for the one-click remote Tacker qualification."""

import ast
from pathlib import Path
import re
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

    def test_fail_closed_shell_and_remote_path_defaults(self):
        self.assertIn("set -euo pipefail", self.source)
        self.assertIn(
            "MODEL_PATH=\"${MODEL_PATH:-/data/qyfeng/"
            "4DGaussians-flame-steak-full/outputs/n3dv_flame_steak}\"",
            self.source,
        )
        self.assertIn(
            "SOURCE_PATH=\"${SOURCE_PATH:-/data/qyfeng/datasets/n3dv/"
            "flame_steak}\"",
            self.source,
        )
        self.assertIn(
            "OUTPUT_DIR=\"${OUTPUT_DIR:-/data/qyfeng/tacker_admission}\"",
            self.source,
        )
        self.assertIn(
            "TORCH_HOME=\"${TORCH_HOME:-/data/qyfeng/cache/torch}\"",
            self.source,
        )
        self.assertIn("export TORCH_HOME", self.source)
        self.assertIn('mkdir -p "${TORCH_HOME}"', self.source)
        self.assertIn(
            "ADMITTED_PROFILE=\"${ADMITTED_PROFILE:-${PROJECT_ROOT}/"
            "tacker_profiles/raster_head_sm86.admitted.json}\"",
            self.source,
        )
        self.assertIn(
            '[[ ! -e "${ADMITTED_PROFILE}" ]]',
            self.source,
        )
        self.assertIn(
            "the admitted profile must not overwrite the disabled template",
            self.source,
        )
        self.assertNotRegex(self.source, r"(?m)^\s*ssh(?:\s|$)")

    def test_preflight_selects_one_exact_a6000_as_logical_device_zero(self):
        for required in (
            'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"',
            '[[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]',
            "torch.cuda.device_count() != 1",
            "torch.cuda.set_device(0)",
            "torch.cuda.get_device_name(0)",
            'gpu_name != "NVIDIA RTX A6000"',
            "torch.cuda.get_device_capability(0)",
            "compute_capability != (8, 6)",
            'sys.version_info[:2] != (3, 10)',
            'torch.version.cuda != "12.4"',
            'torch_public_version != "2.4.1"',
            '[[ "${NVCC_RELEASE}" == "12.4" ]]',
            'CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"',
            'readonly PERSISTENT_BLOCKS="${PERSISTENT_BLOCKS:-7000}"',
            '[[ "${PERSISTENT_BLOCKS}" =~ ^[1-9][0-9]*$ ]]',
            'template["manifest"]["persistent_blocks"]',
            "Tacker-4DGS-runtime",
            '[[ "${#SMI_ALL_GPU_NAMES[@]}" -eq 4 ]]',
            'all four physical GPUs must be NVIDIA RTX A6000',
        ):
            self.assertIn(required, self.source)
        self.assertIn("hostname -f 2>/dev/null || hostname", self.source)
        self.assertNotRegex(self.source, r"hostname[^\n]*(?:==|!=).*4A6000")
        self.assertIn("--gpu 0", self.source)

    def test_builds_runtime_and_runs_all_cuda_test_suites(self):
        for required in (
            '-S "${TACKER_ROOT}/src"',
            "-DTACKER_BUILD_LEGACY=OFF",
            "-DTACKER_ENABLE_CUDA_BACKEND=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=86",
            '-DCMAKE_CUDA_COMPILER="${NVCC_PATH}"',
            'grep -Fq "CudaExecutionBackend.cc"',
            '"${CTEST_BIN}" --output-on-failure',
            'cd "${SIMPLE_KNN_DIR}"',
            '--build-temp "${SIMPLE_KNN_BUILD_TEMP}"',
            'tee "${SIMPLE_KNN_BUILD_LOG}"',
            'cd "${HEAD_EXTENSION_DIR}"',
            '"${PYTHON_BIN}" setup.py build_ext --inplace --force',
            '--build-temp "${HEAD_BUILD_TEMP}"',
            '--build-temp "${RASTER_BUILD_TEMP}"',
            'tee "${HEAD_BUILD_LOG}"',
            'tee "${RASTER_BUILD_LOG}"',
            'grep -Fq "ptxas info" "${HEAD_BUILD_LOG}"',
            'grep -Fq "ptxas info" "${RASTER_BUILD_LOG}"',
            'cd "${RASTER_EXTENSION_DIR}"',
            'TACKER_4DGS_HEAD_INCLUDE="${HEAD_EXTENSION_DIR}/include"',
            "tests.test_head_linear_cuda",
            "tests.test_tacker_mixed_cuda",
            "tests.test_stream_aware_legacy_cuda",
            'hasattr(raster_backend, "rasterize_gaussians_with_head")',
            "from simple_knn._C import distCUDA2",
        ):
            self.assertIn(required, self.source)

    def test_runs_real_measurement_quality_e2e_and_admission_entry_points(self):
        for required in (
            '"${PYTHON_BIN}" profile_tacker_leaves.py',
            '--persistent-blocks "${PERSISTENT_BLOCKS}"',
            '--device-output "${DEVICE_JSON}"',
            '--raster-output "${RASTER_JSON}"',
            '--leaf-output "${LEAF_JSON}"',
            '"${PYTHON_BIN}" scripts/validate_tacker_modes.py',
            "--modes serial two_stream tacker",
            '--qualification-profile "${TEMPLATE_PROFILE}"',
            '"${PYTHON_BIN}" profile_render.py',
            "--execution-mode two_stream",
            "--execution-mode tacker",
            '"${PYTHON_BIN}" scripts/benchmark_tacker_admission.py',
            '--enabled-profile "${ADMITTED_PROFILE}"',
        ):
            self.assertIn(required, self.source)

        # These tools, not shell redirection or literal JSON, own every
        # qualification/admission result artifact.
        self.assertNotIn('> "${DEVICE_JSON}"', self.source)
        self.assertNotIn('> "${QUALITY_JSON}"', self.source)
        self.assertNotIn('> "${ADMITTED_PROFILE}"', self.source)

    def test_required_stages_are_in_strict_order(self):
        markers = (
            "1/10 exact device and environment preflight",
            "2/10 build and test the reusable Tacker CUDA runtime",
            "3/10 build simple-knn, head, and mixed Raster extensions",
            "4/10 run the head CUDA tests",
            "5/10 run the mixed and legacy stream-aware Raster CUDA tests",
            "6/10 collect canonical device, Raster, and mixed-leaf measurements",
            "7/10 validate serial, two_stream, and qualification Tacker quality",
            "8/10 collect comparable two_stream and qualification Tacker E2E timings",
            "9/10 run fail-closed admission",
            "10/10 verify normal Tacker execution with the admitted profile",
        )
        positions = [self.source.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))

    def test_final_run_uses_admitted_profile_without_qualification_override(self):
        final_section = self.source.split(
            'step "10/10 verify normal Tacker execution with the admitted profile"',
            1,
        )[1].split("# profile_render.py records fallbacks", 1)[0]
        self.assertIn("--execution-mode tacker", final_section)
        self.assertIn('--tacker-profile "${ADMITTED_PROFILE}"', final_section)
        self.assertNotIn("--qualification-mode", final_section)
        self.assertNotIn("--qualification-profile", final_section)

        # profile_render exits successfully on fallback, so the shell must
        # inspect the generated metadata and reject anything but real Tacker.
        self.assertIn(
            'metadata.get("actual_execution_mode") != "tacker"', self.source
        )
        self.assertIn(
            'metadata.get("profile_manifest_sha256") != '
            'profile.get("manifest_sha256")',
            self.source,
        )
        self.assertIn(
            'profile.get("admission") != {"enabled": True, "valid": True}',
            self.source,
        )

    def test_embedded_python_is_valid_under_python_37_grammar(self):
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", self.source, re.DOTALL)
        self.assertEqual(len(blocks), 3)
        for index, block in enumerate(blocks):
            ast.parse(block, "embedded-python-{}".format(index), feature_version=7)


if __name__ == "__main__":
    unittest.main()
