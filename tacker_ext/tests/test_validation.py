import unittest

from tacker_4dgs_head.validation import (
    TensorSpec,
    launch_metadata,
    validate_head_linear_output_spec,
    validate_head_linear_specs,
)


def valid_specs(rows=111_525):
    return (
        TensorSpec((rows, 128), "float16", "cuda:0"),
        TensorSpec((128, 128), "float16", "cuda:0"),
        TensorSpec((128,), "float32", "cuda:0"),
    )


class ValidationTest(unittest.TestCase):
    def test_accepts_real_shape_and_tail(self):
        result = validate_head_linear_specs(*valid_specs())
        self.assertEqual(result.rows, 111_525)

    def test_accepts_zero_rows(self):
        result = validate_head_linear_specs(*valid_specs(0))
        self.assertEqual(result.rows, 0)

    def test_launch_shape(self):
        metadata = launch_metadata(111_525)
        self.assertEqual(metadata["logical_grid"], (6_971, 2, 1))
        self.assertEqual(metadata["logical_blocks"], 13_942)
        self.assertEqual(metadata["threads"], 128)

    def test_rejects_wrong_input_width(self):
        _, weight, bias = valid_specs()
        with self.assertRaisesRegex(ValueError, r"\[N, 128\]"):
            validate_head_linear_specs(
                TensorSpec((17, 127), "float16", "cuda:0"), weight, bias
            )

    def test_rejects_noncontiguous_weight(self):
        input, _, bias = valid_specs()
        weight = TensorSpec((128, 128), "float16", "cuda:0", contiguous=False)
        with self.assertRaisesRegex(ValueError, "weight must be contiguous"):
            validate_head_linear_specs(input, weight, bias)

    def test_rejects_wrong_dtypes(self):
        _, weight, bias = valid_specs()
        with self.assertRaisesRegex(TypeError, "input must be float16"):
            validate_head_linear_specs(
                TensorSpec((17, 128), "float32", "cuda:0"), weight, bias
            )

    def test_rejects_mixed_devices(self):
        input, weight, _ = valid_specs()
        with self.assertRaisesRegex(ValueError, "same CUDA device"):
            validate_head_linear_specs(
                input, weight, TensorSpec((128,), "float32", "cuda:1")
            )

    def test_rejects_cpu(self):
        _, weight, bias = valid_specs()
        with self.assertRaisesRegex(ValueError, "input must be on CUDA"):
            validate_head_linear_specs(
                TensorSpec((17, 128), "float16", "cpu"), weight, bias
            )

    def test_accepts_caller_owned_output(self):
        input, _, _ = valid_specs(37)
        validate_head_linear_output_spec(
            input, TensorSpec((37, 128), "float32", "cuda:0")
        )

    def test_rejects_wrong_output_shape(self):
        input, _, _ = valid_specs(37)
        with self.assertRaisesRegex(ValueError, r"\[N, 128\]"):
            validate_head_linear_output_spec(
                input, TensorSpec((36, 128), "float32", "cuda:0")
            )

    def test_rejects_wrong_output_dtype(self):
        input, _, _ = valid_specs(37)
        with self.assertRaisesRegex(TypeError, "output must be float32"):
            validate_head_linear_output_spec(
                input, TensorSpec((37, 128), "float16", "cuda:0")
            )

    def test_rejects_output_on_different_device(self):
        input, _, _ = valid_specs(37)
        with self.assertRaisesRegex(ValueError, "same CUDA device"):
            validate_head_linear_output_spec(
                input, TensorSpec((37, 128), "float32", "cuda:1")
            )


if __name__ == "__main__":
    unittest.main()
