import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ExtensionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.abi = json.loads(
            (ROOT / "abi" / "head_linear_v1.json").read_text(encoding="utf-8")
        )

    def test_versioned_global_symbols_are_declared_and_defined(self):
        declarations = (ROOT / "include" / "head_linear_kernels.cuh").read_text(
            encoding="utf-8"
        )
        definitions = (ROOT / "csrc" / "head_linear.cu").read_text(
            encoding="utf-8"
        )
        expected = {
            "solo": "tacker_head_linear_solo_v1",
            "gptb": "tacker_head_linear_gptb_v1",
        }
        for kind, symbol in expected.items():
            self.assertEqual(
                self.abi["global_kernel_symbols"][kind]["symbol"], symbol
            )
            self.assertIn('extern "C" __global__ void ' + symbol, declarations)
            self.assertIn('extern "C" __global__ void ' + symbol, definitions)

    def test_out_api_argument_order_is_stable(self):
        self.assertEqual(
            self.abi["public_api"]["python_out"]["head_linear_solo_out"],
            ["input", "weight", "bias", "output"],
        )
        self.assertEqual(
            self.abi["public_api"]["python_out"]["head_linear_gptb_out"],
            ["input", "weight", "bias", "output", "persistent_blocks"],
        )
        bindings = (ROOT / "csrc" / "bindings.cpp").read_text(encoding="utf-8")
        self.assertIn('"head_linear_solo_out"', bindings)
        self.assertIn('"head_linear_gptb_out"', bindings)

    def test_compiled_capability_query_locks_kernel_symbols(self):
        bindings = (ROOT / "csrc" / "bindings.cpp").read_text(encoding="utf-8")
        python_api = (ROOT / "tacker_4dgs_head" / "__init__.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(self.abi["capability_query"], "tacker_capabilities")
        self.assertIn('"tacker_capabilities"', bindings)
        self.assertIn('"tacker_head_linear_solo_v1"', bindings)
        self.assertIn('"tacker_head_linear_gptb_v1"', bindings)
        self.assertIn("def _extension():", python_api)
        self.assertNotIn("def _C():", python_api)

    def test_python_37_is_declared_and_sources_parse_with_37_grammar(self):
        setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('python_requires=">=3.7"', setup_source)
        for path in list((ROOT / "tacker_4dgs_head").glob("*.py")) + list(
            (ROOT / "tests").glob("*.py")
        ):
            ast.parse(path.read_text(encoding="utf-8"), str(path), feature_version=7)

    def test_extension_does_not_link_libtacker_runtime(self):
        setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertNotIn("libtacker_runtime", setup_source)
        self.assertNotIn("libraries=", setup_source)

    def test_extension_keeps_python_abi_suffix(self):
        setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertNotIn("no_python_abi_suffix=True", setup_source)
        self.assertIn('cmdclass={"build_ext": BuildExtension}', setup_source)
        self.assertIn('"-Xptxas=-v"', setup_source)

    def test_native_alignment_grad_and_alias_guards_are_in_cpp(self):
        implementation = (ROOT / "csrc" / "head_linear.cu").read_text(
            encoding="utf-8"
        )
        self.assertIn("kWmmaAlignment = 32", implementation)
        self.assertIn("input data pointer must be natively 32-byte aligned", implementation)
        self.assertIn("weight data pointer must be natively 32-byte aligned", implementation)
        self.assertIn("output data pointer must be natively 32-byte aligned", implementation)
        self.assertIn("output storage must not overlap input storage", implementation)
        self.assertIn("!at::GradMode::is_enabled()", implementation)

    def test_large_row_addressing_uses_size_t_and_safe_ceil_div(self):
        implementation = (ROOT / "csrc" / "head_linear.cu").read_text(
            encoding="utf-8"
        )
        device_header = (ROOT / "include" / "head_linear_device.cuh").read_text(
            encoding="utf-8"
        )
        self.assertIn("static_cast<int64_t>(rows)", implementation)
        self.assertIn("const size_t input_offset", device_header)
        self.assertIn("const size_t output_offset", device_header)


if __name__ == "__main__":
    unittest.main()
