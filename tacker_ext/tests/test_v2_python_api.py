import unittest
from unittest import mock

import tacker_4dgs_head as api


class FakeTensor:
    def __init__(self, shape, dtype, device="cuda:0", contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


def task(rows=37):
    return (
        FakeTensor((rows, 128), "float16"),
        FakeTensor((128, 128), "float16"),
        FakeTensor((128,), "float32"),
        FakeTensor((rows, 128), "float32"),
    )


class FakeExtension:
    def __init__(self):
        self.calls = []

    def head_linear_multi_gptb(
        self, inputs, weights, biases, worker_groups, persistent_blocks
    ):
        self.calls.append(
            (
                "multi_gptb",
                inputs,
                weights,
                biases,
                worker_groups,
                persistent_blocks,
            )
        )
        return ["first", "second"]

    def head_linear_multi_gptb_out(
        self, inputs, weights, biases, outputs, worker_groups, persistent_blocks
    ):
        self.calls.append(
            (
                "multi_gptb_out",
                inputs,
                weights,
                biases,
                outputs,
                worker_groups,
                persistent_blocks,
            )
        )
        return outputs

    def head_linear_packed_gptb(
        self, input_tensor, weights, biases, worker_groups, persistent_blocks
    ):
        self.calls.append(
            (
                "packed",
                input_tensor,
                weights,
                biases,
                worker_groups,
                persistent_blocks,
            )
        )
        return "packed-output"

    def whole_head_gptb(
        self,
        input_tensor,
        first_weight,
        first_bias,
        tail_weight,
        tail_bias,
        persistent_blocks,
    ):
        self.calls.append(
            (
                "whole",
                input_tensor,
                first_weight,
                first_bias,
                tail_weight,
                tail_bias,
                persistent_blocks,
            )
        )
        return "whole-output"

    @staticmethod
    def tacker_capabilities_v2():
        return {"abi_version": 2, "max_head_tasks": 5}

    @staticmethod
    def tacker_resources_v2():
        return {"abi_version": 2, "kernels": {}}


class PythonV2ApiTest(unittest.TestCase):
    def setUp(self):
        self.extension = FakeExtension()
        self.patch = mock.patch.object(api, "_extension", return_value=self.extension)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_multi_sequence_argument_order_and_tuple_result(self):
        tasks = [task(17), task(37)]
        inputs = [value[0] for value in tasks]
        weights = [value[1] for value in tasks]
        biases = [value[2] for value in tasks]
        outputs = [value[3] for value in tasks]

        result = api.head_linear_multi_gptb(
            inputs, weights, biases, worker_groups=2, persistent_blocks=7
        )
        self.assertEqual(result, ("first", "second"))
        self.assertEqual(self.extension.calls[-1][0], "multi_gptb")
        self.assertEqual(self.extension.calls[-1][-2:], (2, 7))

        result = api.head_linear_multi_gptb_out(
            inputs, weights, biases, outputs, worker_groups=1, persistent_blocks=3
        )
        self.assertEqual(result, tuple(outputs))
        self.assertEqual(self.extension.calls[-1][0], "multi_gptb_out")
        self.assertEqual(self.extension.calls[-1][-2:], (1, 3))

    def test_packed_and_whole_argument_order(self):
        first = task(17)
        packed_weights = FakeTensor((2, 128, 128), "float16")
        packed_biases = FakeTensor((2, 128), "float32")
        self.assertEqual(
            api.head_linear_packed_gptb(
                first[0], packed_weights, packed_biases, 2, 5
            ),
            "packed-output",
        )
        self.assertEqual(self.extension.calls[-1][-2:], (2, 5))

        tail_weight = FakeTensor((3, 128), "float32")
        tail_bias = FakeTensor((3,), "float32")
        self.assertEqual(
            api.whole_head_gptb(
                first[0], first[1], first[2], tail_weight, tail_bias, 11
            ),
            "whole-output",
        )
        self.assertEqual(self.extension.calls[-1][-1], 11)

    def test_capability_queries_are_independently_versioned(self):
        self.assertEqual(api.tacker_capabilities_v2()["abi_version"], 2)
        self.assertEqual(api.tacker_resources_v2()["abi_version"], 2)

    def test_validation_happens_before_extension_dispatch(self):
        first = task(17)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            api.head_linear_multi_gptb(
                [first[0]], [first[1]], [first[2]], worker_groups=2
            )
        with self.assertRaisesRegex(TypeError, "must be an int"):
            api.head_linear_multi_gptb(
                [first[0]],
                [first[1]],
                [first[2]],
                persistent_blocks=True,
            )
        self.assertEqual(self.extension.calls, [])


if __name__ == "__main__":
    unittest.main()
