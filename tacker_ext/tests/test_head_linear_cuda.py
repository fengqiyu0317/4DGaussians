"""GPU numerical tests; run only on the configured remote A6000 machine."""

import json
import unittest

try:
    import torch
    from tacker_4dgs_head import (
        head_linear_gptb,
        head_linear_gptb_out,
        head_linear_solo,
        head_linear_solo_out,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA PyTorch required")
class HeadLinearCudaTest(unittest.TestCase):
    @staticmethod
    def _linear_fp32_no_tf32(input, weight, bias):
        """Reference with FP32 operands and TF32 explicitly disabled."""

        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return torch.mm(input.float(), weight.float().t()) + bias.float()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    @staticmethod
    def _error_metrics(actual, reference):
        difference = actual.float() - reference.float()
        reference_norm = torch.linalg.vector_norm(reference.float())
        return {
            "max_abs": torch.max(torch.abs(difference)).item(),
            "mean_abs": torch.mean(torch.abs(difference)).item(),
            "rmse": torch.sqrt(torch.mean(difference * difference)).item(),
            "relative_l2": (
                torch.linalg.vector_norm(difference)
                / torch.clamp(reference_norm, min=1e-12)
            ).item(),
        }

    def _case(self, rows, persistent_blocks=0):
        generator = torch.Generator(device="cuda").manual_seed(20260831 + rows)
        # Keep the original FP32 hidden state/weight as the quality baseline;
        # quantization is a separate error source from kernel implementation.
        input_fp32 = torch.randn(
            rows, 128, device="cuda", dtype=torch.float32, generator=generator
        )
        weight_fp32 = torch.randn(
            128, 128, device="cuda", dtype=torch.float32, generator=generator
        )
        bias = torch.randn(
            128, device="cuda", dtype=torch.float32, generator=generator
        )
        input_fp16 = input_fp32.half()
        weight_fp16 = weight_fp32.half()

        original_fp32 = self._linear_fp32_no_tf32(input_fp32, weight_fp32, bias)
        quantized_fp32_reference = self._linear_fp32_no_tf32(
            input_fp16, weight_fp16, bias
        )

        with torch.no_grad():
            solo = head_linear_solo(input_fp16, weight_fp16, bias)
            gptb = head_linear_gptb(
                input_fp16, weight_fp16, bias, persistent_blocks
            )
            solo_storage = torch.empty_like(quantized_fp32_reference)
            gptb_storage = torch.empty_like(quantized_fp32_reference)
            solo_out = head_linear_solo_out(
                input_fp16, weight_fp16, bias, solo_storage
            )
            gptb_out = head_linear_gptb_out(
                input_fp16,
                weight_fp16,
                bias,
                gptb_storage,
                persistent_blocks,
            )
        torch.cuda.synchronize()

        self.assertEqual(solo_out.data_ptr(), solo_storage.data_ptr())
        self.assertEqual(gptb_out.data_ptr(), gptb_storage.data_ptr())
        torch.testing.assert_close(
            solo, quantized_fp32_reference, rtol=2e-3, atol=2e-3
        )
        torch.testing.assert_close(
            gptb, quantized_fp32_reference, rtol=2e-3, atol=2e-3
        )
        torch.testing.assert_close(solo_storage, solo, rtol=0, atol=0)
        torch.testing.assert_close(gptb_storage, gptb, rtol=0, atol=0)

        quantization = self._error_metrics(solo, original_fp32)
        report = {
            "rows": rows,
            "kernel_reference": "FP16 operands converted to FP32, TF32 disabled",
            "quality_reference": "original FP32 hidden/weight, TF32 disabled",
            "fp16_quantization": quantization,
            "gates": {
                "rmse_lte": 1e-2,
                "max_abs_lte": 5e-2,
                "relative_l2_lte": 1e-3,
            },
        }
        print("HEAD_LINEAR_NUMERICS " + json.dumps(report, sort_keys=True), flush=True)
        self.assertLessEqual(quantization["rmse"], 1e-2)
        self.assertLessEqual(quantization["max_abs"], 5e-2)
        self.assertLessEqual(quantization["relative_l2"], 1e-3)

    def test_exact_tile(self):
        self._case(16)

    def test_non_tile_tail(self):
        self._case(37, persistent_blocks=3)

    def test_real_row_count(self):
        self._case(111_525)

    def test_empty_and_out_storage_identity(self):
        input = torch.empty((0, 128), device="cuda", dtype=torch.float16)
        weight = torch.empty((128, 128), device="cuda", dtype=torch.float16)
        bias = torch.empty((128,), device="cuda", dtype=torch.float32)
        solo_output = torch.empty((0, 128), device="cuda", dtype=torch.float32)
        gptb_output = torch.empty((0, 128), device="cuda", dtype=torch.float32)
        with torch.no_grad():
            self.assertEqual(
                tuple(head_linear_solo(input, weight, bias).shape), (0, 128)
            )
            self.assertEqual(
                tuple(head_linear_gptb(input, weight, bias).shape), (0, 128)
            )
            self.assertEqual(
                head_linear_solo_out(input, weight, bias, solo_output).data_ptr(),
                solo_output.data_ptr(),
            )
            self.assertEqual(
                head_linear_gptb_out(input, weight, bias, gptb_output).data_ptr(),
                gptb_output.data_ptr(),
            )

    def test_grad_mode_is_rejected(self):
        input = torch.empty((16, 128), device="cuda", dtype=torch.float16)
        weight = torch.empty((128, 128), device="cuda", dtype=torch.float16)
        bias = torch.empty((128,), device="cuda", dtype=torch.float32)
        self.assertTrue(torch.is_grad_enabled())
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            head_linear_solo(input, weight, bias)

    def test_misaligned_contiguous_views_are_rejected(self):
        rows = 16
        input_storage = torch.empty(
            rows * 128 + 1, device="cuda", dtype=torch.float16
        )
        weight_storage = torch.empty(
            128 * 128 + 1, device="cuda", dtype=torch.float16
        )
        output_storage = torch.empty(
            rows * 128 + 1, device="cuda", dtype=torch.float32
        )
        aligned_input = torch.empty(
            (rows, 128), device="cuda", dtype=torch.float16
        )
        aligned_weight = torch.empty(
            (128, 128), device="cuda", dtype=torch.float16
        )
        bias = torch.empty((128,), device="cuda", dtype=torch.float32)
        misaligned_input = input_storage[1:].view(rows, 128)
        misaligned_weight = weight_storage[1:].view(128, 128)
        misaligned_output = output_storage[1:].view(rows, 128)
        self.assertTrue(misaligned_input.is_contiguous())
        self.assertTrue(misaligned_weight.is_contiguous())
        self.assertTrue(misaligned_output.is_contiguous())

        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "input data pointer.*32-byte"):
                head_linear_solo(misaligned_input, aligned_weight, bias)
            with self.assertRaisesRegex(RuntimeError, "weight data pointer.*32-byte"):
                head_linear_solo(aligned_input, misaligned_weight, bias)
            with self.assertRaisesRegex(RuntimeError, "output data pointer.*32-byte"):
                head_linear_solo_out(
                    aligned_input, aligned_weight, bias, misaligned_output
                )

    def test_out_storage_must_not_overlap_input(self):
        rows = 16
        backing = torch.empty(
            rows * 128, device="cuda", dtype=torch.float32
        )
        input = backing.view(torch.float16)[: rows * 128].view(rows, 128)
        output = backing.view(rows, 128)
        weight = torch.empty((128, 128), device="cuda", dtype=torch.float16)
        bias = torch.empty((128,), device="cuda", dtype=torch.float32)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "must not overlap input"):
                head_linear_solo_out(input, weight, bias, output)


if __name__ == "__main__":
    unittest.main()
