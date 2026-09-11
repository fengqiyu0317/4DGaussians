import unittest

from tacker_4dgs_head.validation import (
    TensorSpec,
    multi_launch_metadata,
    validate_multi_head_linear_specs,
    validate_multi_head_output_specs,
    validate_packed_head_linear_specs,
    validate_packed_head_output_spec,
    validate_whole_head_specs,
)


def first_linear_specs(rows=37, device="cuda:0"):
    return (
        TensorSpec((rows, 128), "float16", device),
        TensorSpec((128, 128), "float16", device),
        TensorSpec((128,), "float32", device),
    )


class MultiHeadValidationTest(unittest.TestCase):
    def test_accepts_every_supported_task_count(self):
        for count in range(1, 6):
            tasks = [first_linear_specs(17 + index) for index in range(count)]
            spec = validate_multi_head_linear_specs(
                [task[0] for task in tasks],
                [task[1] for task in tasks],
                [task[2] for task in tasks],
                worker_groups=count,
            )
            self.assertEqual(spec.task_count, count)
            self.assertEqual(spec.rows, tuple(17 + index for index in range(count)))
            self.assertEqual(spec.backend_threads, count * 128)

    def test_accepts_empty_and_differing_row_tasks(self):
        rows = (0, 1, 16, 17, 111_525)
        tasks = [first_linear_specs(value) for value in rows]
        spec = validate_multi_head_linear_specs(
            [task[0] for task in tasks],
            [task[1] for task in tasks],
            [task[2] for task in tasks],
            worker_groups=2,
        )
        self.assertEqual(spec.rows, rows)

    def test_dual_head_c2_serial_and_parallel_metadata(self):
        serial = multi_launch_metadata((37, 18), worker_groups=1)
        parallel = multi_launch_metadata((37, 18), worker_groups=2)
        self.assertEqual(serial["logical_blocks_by_task"], (6, 4))
        self.assertEqual(serial["common_logical_end"], 6)
        self.assertEqual(serial["backend_threads"], 128)
        self.assertEqual(parallel["backend_threads"], 256)
        self.assertEqual(parallel["named_barrier_ids"], ())

    def test_rejects_empty_more_than_five_and_mismatched_sequences(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            validate_multi_head_linear_specs([], [], [])
        tasks = [first_linear_specs() for _ in range(6)]
        with self.assertRaisesRegex(ValueError, "at most 5"):
            validate_multi_head_linear_specs(
                [task[0] for task in tasks],
                [task[1] for task in tasks],
                [task[2] for task in tasks],
            )
        one = first_linear_specs()
        with self.assertRaisesRegex(ValueError, "equal sequence lengths"):
            validate_multi_head_linear_specs([one[0]], [one[1], one[1]], [one[2]])

    def test_rejects_invalid_worker_groups(self):
        task = first_linear_specs()
        for value in (0, -1):
            with self.assertRaisesRegex(ValueError, "must be >= 1"):
                validate_multi_head_linear_specs([task[0]], [task[1]], [task[2]], value)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            validate_multi_head_linear_specs([task[0]], [task[1]], [task[2]], 2)
        with self.assertRaisesRegex(TypeError, "must be an int"):
            validate_multi_head_linear_specs([task[0]], [task[1]], [task[2]], True)

    def test_rejects_cross_task_device_mismatch(self):
        first = first_linear_specs(device="cuda:0")
        second = first_linear_specs(device="cuda:1")
        with self.assertRaisesRegex(ValueError, "all head tasks"):
            validate_multi_head_linear_specs(
                [first[0], second[0]],
                [first[1], second[1]],
                [first[2], second[2]],
            )

    def test_validates_one_output_per_task(self):
        inputs = [first_linear_specs(0)[0], first_linear_specs(37)[0]]
        outputs = [
            TensorSpec((0, 128), "float32", "cuda:0"),
            TensorSpec((37, 128), "float32", "cuda:0"),
        ]
        validate_multi_head_output_specs(inputs, outputs)
        with self.assertRaisesRegex(ValueError, "same sequence length"):
            validate_multi_head_output_specs(inputs, outputs[:1])
        with self.assertRaisesRegex(ValueError, r"head task 1.*\[N, 128\]"):
            validate_multi_head_output_specs(
                inputs,
                [outputs[0], TensorSpec((36, 128), "float32", "cuda:0")],
            )


class PackedAndWholeHeadValidationTest(unittest.TestCase):
    def test_packed_accepts_one_to_five_heads(self):
        input_spec = TensorSpec((37, 128), "float16", "cuda:0")
        for count in range(1, 6):
            weights = TensorSpec((count, 128, 128), "float16", "cuda:0")
            biases = TensorSpec((count, 128), "float32", "cuda:0")
            result = validate_packed_head_linear_specs(
                input_spec, weights, biases, worker_groups=count
            )
            self.assertEqual(result.head_count, count)
            output = TensorSpec((count, 37, 128), "float32", "cuda:0")
            validate_packed_head_output_spec(input_spec, weights, output)

    def test_packed_rejects_bad_count_shape_and_dtype(self):
        input_spec = TensorSpec((37, 128), "float16", "cuda:0")
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            validate_packed_head_linear_specs(
                input_spec,
                TensorSpec((6, 128, 128), "float16", "cuda:0"),
                TensorSpec((6, 128), "float32", "cuda:0"),
            )
        with self.assertRaisesRegex(ValueError, r"\[H, 128\]"):
            validate_packed_head_linear_specs(
                input_spec,
                TensorSpec((2, 128, 128), "float16", "cuda:0"),
                TensorSpec((2, 127), "float32", "cuda:0"),
            )
        with self.assertRaisesRegex(TypeError, "weights must be float16"):
            validate_packed_head_linear_specs(
                input_spec,
                TensorSpec((2, 128, 128), "float32", "cuda:0"),
                TensorSpec((2, 128), "float32", "cuda:0"),
            )

    def test_whole_head_accepts_all_4dgs_tail_widths(self):
        first = first_linear_specs(37)
        for tail_features in (1, 3, 4, 48):
            result = validate_whole_head_specs(
                first[0],
                first[1],
                first[2],
                TensorSpec((tail_features, 128), "float32", "cuda:0"),
                TensorSpec((tail_features,), "float32", "cuda:0"),
            )
            self.assertEqual(result.tail_features, tail_features)

    def test_whole_head_rejects_invalid_tail(self):
        first = first_linear_specs(37)
        with self.assertRaisesRegex(ValueError, "1 <= O <= 128"):
            validate_whole_head_specs(
                first[0],
                first[1],
                first[2],
                TensorSpec((129, 128), "float32", "cuda:0"),
                TensorSpec((129,), "float32", "cuda:0"),
            )
        with self.assertRaisesRegex(TypeError, "tail_weight must be float32"):
            validate_whole_head_specs(
                first[0],
                first[1],
                first[2],
                TensorSpec((3, 128), "float16", "cuda:0"),
                TensorSpec((3,), "float32", "cuda:0"),
            )


if __name__ == "__main__":
    unittest.main()
