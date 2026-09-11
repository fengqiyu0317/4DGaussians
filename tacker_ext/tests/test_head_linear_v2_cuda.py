"""ABI v2 CUDA numerics/boundaries; run after building on the RTX A6000."""

import unittest

try:
    import torch
    from tacker_4dgs_head import (
        head_linear_multi_gptb,
        head_linear_multi_gptb_out,
        head_linear_multi_solo,
        head_linear_packed_gptb,
        head_linear_packed_gptb_out,
        tacker_capabilities_v2,
        tacker_resources_v2,
        whole_head_gptb,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA PyTorch required")
class HeadLinearV2CudaTest(unittest.TestCase):
    @staticmethod
    def _reference(input_tensor, weight, bias):
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return torch.mm(input_tensor.float(), weight.float().t()) + bias
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    def _tasks(self, rows):
        inputs = []
        weights = []
        biases = []
        for index, row_count in enumerate(rows):
            generator = torch.Generator(device="cuda").manual_seed(
                20260910 + index * 101 + row_count
            )
            inputs.append(
                torch.randn(
                    row_count,
                    128,
                    dtype=torch.float16,
                    device="cuda",
                    generator=generator,
                )
            )
            weights.append(
                torch.randn(
                    128,
                    128,
                    dtype=torch.float16,
                    device="cuda",
                    generator=generator,
                )
            )
            biases.append(
                torch.randn(
                    128,
                    dtype=torch.float32,
                    device="cuda",
                    generator=generator,
                )
            )
        return inputs, weights, biases

    def _assert_multi(self, rows, worker_groups, persistent_blocks):
        inputs, weights, biases = self._tasks(rows)
        references = [
            self._reference(input_tensor, weight, bias)
            for input_tensor, weight, bias in zip(inputs, weights, biases)
        ]
        with torch.no_grad():
            solo = head_linear_multi_solo(
                inputs, weights, biases, worker_groups=worker_groups
            )
            gptb = head_linear_multi_gptb(
                inputs,
                weights,
                biases,
                worker_groups=worker_groups,
                persistent_blocks=persistent_blocks,
            )
        torch.cuda.synchronize()
        self.assertEqual(len(gptb), len(rows))
        for solo_value, gptb_value, reference in zip(solo, gptb, references):
            torch.testing.assert_close(solo_value, reference, rtol=2e-3, atol=2e-3)
            torch.testing.assert_close(gptb_value, reference, rtol=2e-3, atol=2e-3)

    def test_one_to_five_heads_serial_worker(self):
        for count in range(1, 6):
            self._assert_multi(
                [0, 1, 17, 32, 37][:count],
                worker_groups=1,
                persistent_blocks=3,
            )

    def test_dual_head_c2_serial_and_parallel_workers(self):
        self._assert_multi([37, 18], worker_groups=1, persistent_blocks=2)
        self._assert_multi([37, 18], worker_groups=2, persistent_blocks=2)

    def test_five_parallel_workers_and_real_tail_count(self):
        self._assert_multi(
            [111_525, 33, 17, 1, 0], worker_groups=5, persistent_blocks=0
        )

    def test_multi_out_storage_identity_and_repeat_overwrite(self):
        inputs, weights, biases = self._tasks([17, 37])
        outputs = [
            torch.full((17, 128), float("nan"), device="cuda"),
            torch.full((37, 128), float("nan"), device="cuda"),
        ]
        with torch.no_grad():
            first = head_linear_multi_gptb_out(
                inputs, weights, biases, outputs, 2, 3
            )
            snapshots = [value.clone() for value in outputs]
            second = head_linear_multi_gptb_out(
                inputs, weights, biases, outputs, 2, 3
            )
        torch.cuda.synchronize()
        for index in range(2):
            self.assertEqual(first[index].data_ptr(), outputs[index].data_ptr())
            self.assertEqual(second[index].data_ptr(), outputs[index].data_ptr())
            torch.testing.assert_close(outputs[index], snapshots[index], rtol=0, atol=0)

    def test_packed_two_and_five_heads(self):
        for head_count, worker_groups in ((2, 1), (2, 2), (5, 5)):
            inputs, weights, biases = self._tasks([37] * head_count)
            shared_input = inputs[0]
            packed_weights = torch.stack(weights)
            packed_biases = torch.stack(biases)
            references = torch.stack(
                [
                    self._reference(shared_input, weight, bias)
                    for weight, bias in zip(weights, biases)
                ]
            )
            output = torch.empty_like(references)
            with torch.no_grad():
                value = head_linear_packed_gptb(
                    shared_input,
                    packed_weights,
                    packed_biases,
                    worker_groups,
                    3,
                )
                out_value = head_linear_packed_gptb_out(
                    shared_input,
                    packed_weights,
                    packed_biases,
                    output,
                    worker_groups,
                    3,
                )
            torch.cuda.synchronize()
            self.assertEqual(out_value.data_ptr(), output.data_ptr())
            torch.testing.assert_close(value, references, rtol=2e-3, atol=2e-3)
            torch.testing.assert_close(output, references, rtol=2e-3, atol=2e-3)

    def test_whole_head_tail_widths_and_tail_rows(self):
        input_tensor, first_weight, first_bias = self._tasks([17])
        input_tensor = input_tensor[0]
        first_weight = first_weight[0]
        first_bias = first_bias[0]
        generator = torch.Generator(device="cuda").manual_seed(20260911)
        for tail_features in (1, 3, 4, 48):
            tail_weight = torch.randn(
                tail_features,
                128,
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            tail_bias = torch.randn(
                tail_features,
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            hidden = torch.relu(
                self._reference(input_tensor, first_weight, first_bias)
            )
            reference = self._reference(hidden, tail_weight, tail_bias)
            with torch.no_grad():
                value = whole_head_gptb(
                    input_tensor,
                    first_weight,
                    first_bias,
                    tail_weight,
                    tail_bias,
                    persistent_blocks=3,
                )
            torch.cuda.synchronize()
            torch.testing.assert_close(value, reference, rtol=2e-3, atol=2e-3)

    def test_invalid_counts_worker_groups_dtype_shape_and_alias_fail_closed(self):
        inputs, weights, biases = self._tasks([17] * 2)
        with torch.no_grad():
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                head_linear_multi_gptb([], [], [])
            with self.assertRaisesRegex(ValueError, "must not exceed"):
                head_linear_multi_gptb(inputs, weights, biases, worker_groups=3)
            with self.assertRaisesRegex(TypeError, "input must be float16"):
                head_linear_multi_gptb(
                    [inputs[0].float()], [weights[0]], [biases[0]]
                )
            with self.assertRaisesRegex(RuntimeError, "must not overlap each other"):
                shared_output = torch.empty((17, 128), device="cuda")
                head_linear_multi_gptb_out(
                    inputs,
                    weights,
                    biases,
                    [shared_output, shared_output],
                    worker_groups=2,
                )

    def test_misaligned_multi_input_is_rejected(self):
        rows = 17
        storage = torch.empty(rows * 128 + 1, device="cuda", dtype=torch.float16)
        misaligned = storage[1:].view(rows, 128)
        _, weights, biases = self._tasks([rows])
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "32-byte aligned"):
                head_linear_multi_gptb([misaligned], weights, biases)

    def test_non_default_stream_lifetime_and_capability_resources(self):
        inputs, weights, biases = self._tasks([37, 18])
        stream = torch.cuda.Stream()
        with torch.no_grad(), torch.cuda.stream(stream):
            outputs = head_linear_multi_gptb(
                inputs, weights, biases, worker_groups=2, persistent_blocks=3
            )
            completion = torch.cuda.Event()
            completion.record(stream)
        completion.synchronize()
        for output, input_tensor, weight, bias in zip(
            outputs, inputs, weights, biases
        ):
            torch.testing.assert_close(
                output,
                self._reference(input_tensor, weight, bias),
                rtol=2e-3,
                atol=2e-3,
            )

        capabilities = tacker_capabilities_v2()
        resources = tacker_resources_v2()
        self.assertEqual(capabilities["abi_version"], 2)
        self.assertEqual(capabilities["max_head_tasks"], 5)
        self.assertEqual(resources["abi_version"], 2)
        for symbol in capabilities["global_kernel_symbols"].values():
            self.assertIn(symbol, resources["kernels"])
            self.assertGreater(resources["kernels"][symbol]["max_threads_per_block"], 0)
            self.assertTrue(resources["kernels"][symbol]["active_blocks_per_sm"])
            self.assertGreater(
                resources["kernels"][symbol]["active_blocks_per_sm"][0], 0
            )


if __name__ == "__main__":
    unittest.main()
