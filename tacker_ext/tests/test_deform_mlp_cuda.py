"""Standalone full-MLP CUDA tests. Run on 4A6000 after building the extension."""

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

from tacker_4dgs_head import (
    deform_mlp_delta_v1,
    deform_mlp_delta_out_v1,
    deform_mlp_sliced_out_v2,
    deform_mlp_sliced_v2,
)


WIDTHS = (3, 3, 4, 1, 48)
OFFSETS = (0, 3, 6, 10, 11, 59)


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA required")
class DeformMlpCudaTest(unittest.TestCase):
    def setUp(self):
        self.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.previous_tf32

    @staticmethod
    def case(rows):
        generator = torch.Generator(device="cuda").manual_seed(20260916 + rows)

        def randn(*shape):
            return torch.randn(*shape, device="cuda", generator=generator)

        return [randn(rows, 32), randn(32, 128) / math.sqrt(32),
                randn(128) * 0.1, (randn(5, 128, 128) / math.sqrt(128)).half(),
                randn(5, 128) * 0.1, randn(59, 128) / math.sqrt(128),
                randn(59) * 0.1]

    @staticmethod
    def reference(args):
        features, w0, b0, weights, biases, tails, tail_bias = args
        activated = torch.relu(features @ w0 + b0).half().float()
        outputs = []
        for head in range(5):
            hidden = torch.relu(activated @ weights[head].float().t() + biases[head])
            lo, hi = OFFSETS[head:head + 2]
            outputs.append(hidden @ tails[lo:hi].t() + tail_bias[lo:hi])
        return outputs

    @staticmethod
    def sliced_reference(args):
        features, w0, b0, weights, biases, tails, _tail_bias = args
        activated = torch.relu(features @ w0 + b0).half()
        partials = []
        for slice_index in range(8):
            lo_hidden = slice_index * 16
            hi_hidden = lo_hidden + 16
            packed = torch.empty(
                (features.shape[0], 59), device=features.device,
                dtype=torch.float32)
            for head in range(5):
                hidden = torch.relu(
                    activated.float()
                    @ weights[head, lo_hidden:hi_hidden].float().t()
                    + biases[head, lo_hidden:hi_hidden]
                )
                lo_output, hi_output = OFFSETS[head:head + 2]
                packed[:, lo_output:hi_output] = (
                    hidden
                    @ tails[
                        lo_output:hi_output, lo_hidden:hi_hidden
                    ].t()
                )
            partials.append(packed)
        return activated, torch.stack(partials, dim=0)

    def assert_outputs(self, actual, expected):
        self.assertEqual(len(actual), 5)
        for value, ref in zip(actual, expected):
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(value.is_contiguous())
            self.assertTrue(torch.isfinite(value).all().item())
            # FP32 backbone reduction order can move an activation across an
            # FP16 rounding boundary. Test a matched-precision model, not bits.
            torch.testing.assert_close(value, ref, rtol=2e-3, atol=5e-4)

    def test_empty_partial_and_full_tiles(self):
        with torch.no_grad():
            for rows in (0, 1, 5, 15, 16, 17, 31, 32, 33):
                with self.subTest(rows=rows):
                    args = self.case(rows)
                    self.assert_outputs(deform_mlp_delta_v1(*args), self.reference(args))

    def test_real_row_count_and_persistent_grid(self):
        with torch.no_grad():
            args = self.case(111525)
            reference = self.reference(args)
            for blocks in (0, 127, (111525 + 15) // 16):
                with self.subTest(blocks=blocks):
                    self.assert_outputs(deform_mlp_delta_v1(
                        *args, persistent_blocks=blocks), reference)

    def test_both_relu_stages_biases_and_head_order(self):
        with torch.no_grad():
            args = self.case(2)
            for value in args:
                value.zero_()
            args[0][:, 0] = torch.tensor([-2.0, 2.0], device="cuda")
            args[1][0, 0] = 1.0
            args[2][0] = 1.0
            args[3][:, 0, 0] = 2.0
            args[4][:, 0] = -1.0
            args[5][:, 0] = torch.arange(1, 60, device="cuda")
            args[6].copy_(torch.arange(59, device="cuda"))
            actual = deform_mlp_delta_v1(*args)
            for h, value in enumerate(actual):
                lo, hi = OFFSETS[h:h + 2]
                expected = torch.stack((args[6][lo:hi],
                                        5 * args[5][lo:hi, 0] + args[6][lo:hi]))
                torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_out_identity_guards_and_repeat_overwrite(self):
        with torch.no_grad():
            args = self.case(33)
            buffers = [torch.full((33 * width + 8,), -12345.0, device="cuda")
                       for width in WIDTHS]
            outputs = [value[4:-4].view(33, width)
                       for value, width in zip(buffers, WIDTHS)]
            for blocks in (1, 3):
                args[0].mul_(0.5)
                returned = deform_mlp_delta_out_v1(*args, outputs, blocks)
                self.assert_outputs(returned, self.reference(args))
                for value, output, buffer in zip(returned, outputs, buffers):
                    self.assertEqual(value.data_ptr(), output.data_ptr())
                    self.assertTrue((buffer[:4] == -12345).all().item())
                    self.assertTrue((buffer[-4:] == -12345).all().item())

    def test_validation_rejects_invalid_inputs_and_aliases(self):
        args = self.case(17)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            deform_mlp_delta_v1(*args)
        with torch.no_grad():
            with self.assertRaisesRegex(TypeError, "persistent_blocks"):
                deform_mlp_delta_v1(*args, persistent_blocks=True)
            with self.assertRaisesRegex(ValueError, "persistent_blocks"):
                deform_mlp_delta_v1(*args, persistent_blocks=-1)
            wrong = list(args)
            wrong[0] = args[0].half()
            with self.assertRaisesRegex(RuntimeError, "dtype"):
                deform_mlp_delta_v1(*wrong)
            wrong = list(args)
            wrong[1] = torch.empty(128, 32, device="cuda").t()
            with self.assertRaisesRegex(RuntimeError, "contiguous"):
                deform_mlp_delta_v1(*wrong)
            wrong = list(args)
            wrong[3] = torch.empty(5 * 128 * 128 + 1, device="cuda",
                                   dtype=torch.float16)[1:].view(5, 128, 128)
            with self.assertRaisesRegex(RuntimeError, "32-byte aligned"):
                deform_mlp_delta_v1(*wrong)
            outputs = [torch.empty(17, width, device="cuda") for width in WIDTHS]
            outputs[1] = outputs[0]
            with self.assertRaisesRegex(RuntimeError, "overlap each other"):
                deform_mlp_delta_out_v1(*args, outputs)
            outputs[1] = torch.empty(17, 3, device="cuda")
            outputs[0] = args[0].flatten()[:17 * 3].view(17, 3)
            with self.assertRaisesRegex(RuntimeError, "overlap any input"):
                deform_mlp_delta_out_v1(*args, outputs)

    def test_non_default_stream(self):
        with torch.no_grad():
            args = self.case(37)
            reference = self.reference(args)
            ready = torch.cuda.Event()
            ready.record()
            stream = torch.cuda.Stream()
            done = torch.cuda.Event()
            with torch.cuda.stream(stream):
                stream.wait_event(ready)
                for value in args:
                    value.record_stream(stream)
                outputs = deform_mlp_delta_v1(*args, persistent_blocks=2)
                done.record()
            torch.cuda.current_stream().wait_event(done)
            for value in outputs:
                value.record_stream(torch.cuda.current_stream())
            self.assert_outputs(outputs, reference)

    def test_exactly_one_cuda_kernel(self):
        with torch.no_grad():
            args = self.case(33)
            outputs = deform_mlp_delta_v1(*args)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                deform_mlp_delta_out_v1(*args, outputs)
                torch.cuda.synchronize()
            kernels = [event.name for event in prof.events()
                       if event.device_type == torch.autograd.DeviceType.CUDA]
            self.assertEqual(len(kernels), 1, kernels)
            self.assertIn("tacker_deform_mlp_delta_v1", kernels[0])

    def test_sliced_v2_partial_tiles_and_reduction(self):
        with torch.no_grad():
            for rows in (0, 1, 15, 16, 17, 33, 127, 128, 129):
                with self.subTest(rows=rows):
                    args = self.case(rows)
                    activated, partials = deform_mlp_sliced_v2(*args[:-1])
                    expected_activated, expected_partials = (
                        self.sliced_reference(args)
                    )
                    self.assertEqual(activated.shape, (rows, 128))
                    self.assertEqual(activated.dtype, torch.float16)
                    self.assertEqual(partials.shape, (8, rows, 59))
                    self.assertEqual(partials.dtype, torch.float32)
                    torch.testing.assert_close(
                        activated, expected_activated, rtol=2e-3, atol=5e-4)
                    torch.testing.assert_close(
                        partials, expected_partials, rtol=2e-3, atol=5e-4)

                    packed = partials.sum(dim=0) + args[-1]
                    outputs = tuple(
                        packed[:, lo:hi].contiguous()
                        for lo, hi in zip(OFFSETS[:-1], OFFSETS[1:])
                    )
                    self.assert_outputs(outputs, self.reference(args))

    def test_sliced_v2_out_buffers_and_parallel_six_kernel_contract(self):
        with torch.no_grad():
            args = self.case(33)
            activated = torch.empty(
                (33, 128), device="cuda", dtype=torch.float16)
            partials = torch.empty(
                (8, 33, 59), device="cuda", dtype=torch.float32)
            returned = deform_mlp_sliced_out_v2(
                *args[:-1], activated, partials)
            self.assertEqual(returned[0].data_ptr(), activated.data_ptr())
            self.assertEqual(returned[1].data_ptr(), partials.data_ptr())
            torch.cuda.synchronize()

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                deform_mlp_sliced_out_v2(
                    *args[:-1], activated, partials)
                torch.cuda.synchronize()
            kernels = [
                event.name
                for event in prof.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            ]
            expected_names = ["tacker_deform_mlp_backbone_v2"] + [
                "tacker_deform_mlp_sliced_head{}_v2".format(head)
                for head in range(5)
            ]
            self.assertEqual(len(kernels), len(expected_names), kernels)
            for expected in expected_names:
                self.assertEqual(
                    sum(expected in actual for actual in kernels), 1, kernels)

    def test_sliced_v2_parallel_heads_join_non_default_stream(self):
        with torch.no_grad():
            args = self.case(129)
            expected_activated, expected_partials = self.sliced_reference(args)
            ready = torch.cuda.Event()
            ready.record()
            stream = torch.cuda.Stream()
            done = torch.cuda.Event()
            with torch.cuda.stream(stream):
                stream.wait_event(ready)
                for value in args:
                    value.record_stream(stream)
                activated, partials = deform_mlp_sliced_v2(*args[:-1])
                # This reduction is deliberately queued immediately.  It is
                # correct only if the extension rejoins all five head streams.
                packed = partials.sum(dim=0) + args[-1]
                done.record()
            torch.cuda.current_stream().wait_event(done)
            for value in (activated, partials, packed):
                value.record_stream(torch.cuda.current_stream())
            torch.testing.assert_close(
                activated, expected_activated, rtol=2e-3, atol=5e-4)
            torch.testing.assert_close(
                partials, expected_partials, rtol=2e-3, atol=5e-4)
            torch.testing.assert_close(
                packed, expected_partials.sum(dim=0) + args[-1],
                rtol=2e-3, atol=5e-4)

    def test_sliced_v2_validation_rejects_bad_outputs(self):
        with torch.no_grad():
            args = self.case(17)
            activated = torch.empty(
                (17, 128), device="cuda", dtype=torch.float16)
            partials = torch.empty(
                (8, 17, 59), device="cuda", dtype=torch.float32)
            with self.assertRaisesRegex(RuntimeError, "activated.*shape"):
                deform_mlp_sliced_out_v2(
                    *args[:-1],
                    torch.empty(
                        (17, 127), device="cuda", dtype=torch.float16),
                    partials)
            with self.assertRaisesRegex(RuntimeError, "partials.*shape"):
                deform_mlp_sliced_out_v2(
                    *args[:-1], activated,
                    torch.empty(
                        (8, 17, 58), device="cuda", dtype=torch.float32))

            storage = torch.empty(8 * 17 * 59, device="cuda")
            aliased_args = list(args[:-1])
            aliased_args[0] = storage[:17 * 32].view(17, 32)
            partial_alias = storage.view(8, 17, 59)
            with self.assertRaisesRegex(RuntimeError, "partials.*overlap"):
                deform_mlp_sliced_out_v2(
                    *aliased_args, activated, partial_alias)


if __name__ == "__main__":
    unittest.main()
