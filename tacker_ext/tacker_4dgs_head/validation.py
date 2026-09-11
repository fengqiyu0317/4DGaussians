"""Torch-independent validation for the v1 and v2 head-kernel ABIs."""

from dataclasses import dataclass
from typing import Sequence, Tuple


HEAD_FEATURES = 128
MAX_HEAD_TASKS = 5
WORKER_GROUP_THREADS = 128
_FP16_NAMES = frozenset(("float16", "half"))
_FP32_NAMES = frozenset(("float32", "float"))


@dataclass(frozen=True)
class TensorSpec:
    shape: Tuple[int, ...]
    dtype: str
    device: str
    contiguous: bool = True


@dataclass(frozen=True)
class HeadLinearSpec:
    rows: int
    in_features: int = HEAD_FEATURES
    out_features: int = HEAD_FEATURES


@dataclass(frozen=True)
class MultiHeadLinearSpec:
    rows: Tuple[int, ...]
    task_count: int
    worker_groups: int
    backend_threads: int


@dataclass(frozen=True)
class PackedHeadLinearSpec:
    rows: int
    head_count: int
    worker_groups: int


@dataclass(frozen=True)
class WholeHeadSpec:
    rows: int
    tail_features: int


def _validate_shape_values(name: str, spec: TensorSpec) -> None:
    if any(not isinstance(size, int) or isinstance(size, bool) for size in spec.shape):
        raise TypeError(f"{name}.shape must contain integers")
    if any(size < 0 for size in spec.shape):
        raise ValueError(f"{name}.shape cannot contain negative dimensions")


def validate_head_linear_specs(
    input: TensorSpec, weight: TensorSpec, bias: TensorSpec
) -> HeadLinearSpec:
    """Validate the public pointer ABI without importing PyTorch."""

    for name, spec in (("input", input), ("weight", weight), ("bias", bias)):
        if not isinstance(spec, TensorSpec):
            raise TypeError(f"{name} must be a TensorSpec")
        _validate_shape_values(name, spec)
        if not spec.contiguous:
            raise ValueError(f"{name} must be contiguous")
        if not spec.device.startswith("cuda"):
            raise ValueError(f"{name} must be on CUDA")

    if len(input.shape) != 2 or input.shape[1] != HEAD_FEATURES:
        raise ValueError(f"input must have shape [N, {HEAD_FEATURES}]")
    if weight.shape != (HEAD_FEATURES, HEAD_FEATURES):
        raise ValueError(
            f"weight must have shape [{HEAD_FEATURES}, {HEAD_FEATURES}]"
        )
    if bias.shape != (HEAD_FEATURES,):
        raise ValueError(f"bias must have shape [{HEAD_FEATURES}]")
    if input.dtype not in _FP16_NAMES:
        raise TypeError("input must be float16")
    if weight.dtype not in _FP16_NAMES:
        raise TypeError("weight must be float16")
    if bias.dtype not in _FP32_NAMES:
        raise TypeError("bias must be float32")
    if input.device != weight.device or input.device != bias.device:
        raise ValueError("input, weight, and bias must be on the same CUDA device")

    return HeadLinearSpec(rows=input.shape[0])


def validate_head_linear_output_spec(input: TensorSpec, output: TensorSpec) -> None:
    """Validate caller-owned output storage used by the out variants."""

    if not isinstance(input, TensorSpec):
        raise TypeError("input must be a TensorSpec")
    if not isinstance(output, TensorSpec):
        raise TypeError("output must be a TensorSpec")
    _validate_shape_values("output", output)
    if not output.contiguous:
        raise ValueError("output must be contiguous")
    if not output.device.startswith("cuda"):
        raise ValueError("output must be on CUDA")
    if output.dtype not in _FP32_NAMES:
        raise TypeError("output must be float32")
    if output.shape != (input.shape[0], HEAD_FEATURES):
        raise ValueError(f"output must have shape [N, {HEAD_FEATURES}]")
    if output.device != input.device:
        raise ValueError("output must be on the same CUDA device as input")


def launch_metadata(rows: int) -> dict:
    """Return the logical launch shape consumed by the GPTB adapter."""

    if not isinstance(rows, int) or isinstance(rows, bool):
        raise TypeError("rows must be an int")
    if rows < 0:
        raise ValueError("rows must be >= 0")
    grid_x = (rows + 15) // 16
    grid_y = 2
    return {
        "logical_grid": (grid_x, grid_y, 1),
        "logical_blocks": grid_x * grid_y,
        "threads": 128,
        "input_features": HEAD_FEATURES,
        "output_features": HEAD_FEATURES,
    }


def _validate_worker_groups(task_count: int, worker_groups: int) -> None:
    if not isinstance(worker_groups, int) or isinstance(worker_groups, bool):
        raise TypeError("worker_groups must be an int")
    if worker_groups < 1:
        raise ValueError("worker_groups must be >= 1")
    if worker_groups > task_count:
        raise ValueError("worker_groups must not exceed head task count")


def _validate_tensor_spec_sequence(name: str, specs: Sequence[TensorSpec]) -> None:
    if not isinstance(specs, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of TensorSpec values")
    if not specs:
        raise ValueError("head task sequence must not be empty")
    if len(specs) > MAX_HEAD_TASKS:
        raise ValueError(f"head task sequence supports at most {MAX_HEAD_TASKS} heads")


def validate_multi_head_linear_specs(
    inputs: Sequence[TensorSpec],
    weights: Sequence[TensorSpec],
    biases: Sequence[TensorSpec],
    worker_groups: int = 1,
) -> MultiHeadLinearSpec:
    """Validate one to five independent first-linear tasks for ABI v2."""

    _validate_tensor_spec_sequence("inputs", inputs)
    _validate_tensor_spec_sequence("weights", weights)
    _validate_tensor_spec_sequence("biases", biases)
    if len(inputs) != len(weights) or len(inputs) != len(biases):
        raise ValueError("inputs, weights, and biases must have equal sequence lengths")
    _validate_worker_groups(len(inputs), worker_groups)

    rows = []
    device = inputs[0].device if isinstance(inputs[0], TensorSpec) else None
    for index, (input_spec, weight_spec, bias_spec) in enumerate(
        zip(inputs, weights, biases)
    ):
        try:
            result = validate_head_linear_specs(input_spec, weight_spec, bias_spec)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"head task {index}: {exc}")
        if input_spec.device != device:
            raise ValueError("all head tasks must be on the same CUDA device")
        rows.append(result.rows)
    return MultiHeadLinearSpec(
        rows=tuple(rows),
        task_count=len(rows),
        worker_groups=worker_groups,
        backend_threads=worker_groups * WORKER_GROUP_THREADS,
    )


def validate_multi_head_output_specs(
    inputs: Sequence[TensorSpec], outputs: Sequence[TensorSpec]
) -> None:
    """Validate the one-output-per-task storage contract for ABI v2."""

    _validate_tensor_spec_sequence("inputs", inputs)
    _validate_tensor_spec_sequence("outputs", outputs)
    if len(inputs) != len(outputs):
        raise ValueError("outputs must have the same sequence length as inputs")
    for index, (input_spec, output_spec) in enumerate(zip(inputs, outputs)):
        try:
            validate_head_linear_output_spec(input_spec, output_spec)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"head task {index}: {exc}")


def multi_launch_metadata(rows: Sequence[int], worker_groups: int = 1) -> dict:
    """Return the common GPTB traversal for a multi-head launch."""

    if not isinstance(rows, (list, tuple)):
        raise TypeError("rows must be a list or tuple of ints")
    if not rows:
        raise ValueError("rows must not be empty")
    if len(rows) > MAX_HEAD_TASKS:
        raise ValueError(f"rows supports at most {MAX_HEAD_TASKS} heads")
    for value in rows:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("rows must contain ints")
        if value < 0:
            raise ValueError("rows must contain values >= 0")
    _validate_worker_groups(len(rows), worker_groups)
    logical_blocks = tuple(((value + 15) // 16) * 2 for value in rows)
    return {
        "task_count": len(rows),
        "worker_groups": worker_groups,
        "backend_threads": worker_groups * WORKER_GROUP_THREADS,
        "logical_blocks_by_task": logical_blocks,
        "common_logical_end": max(logical_blocks),
        "named_barrier_ids": (),
        "dynamic_shared_memory_bytes": 0,
    }


def validate_packed_head_linear_specs(
    input: TensorSpec,
    weights: TensorSpec,
    biases: TensorSpec,
    worker_groups: int = 1,
) -> PackedHeadLinearSpec:
    """Validate shared-input packed layout [H,...] used by the C3 adapter."""

    for name, spec in (("input", input), ("weights", weights), ("biases", biases)):
        if not isinstance(spec, TensorSpec):
            raise TypeError(f"{name} must be a TensorSpec")
        _validate_shape_values(name, spec)
        if not spec.contiguous:
            raise ValueError(f"{name} must be contiguous")
        if not spec.device.startswith("cuda"):
            raise ValueError(f"{name} must be on CUDA")
    if len(input.shape) != 2 or input.shape[1] != HEAD_FEATURES:
        raise ValueError(f"input must have shape [N, {HEAD_FEATURES}]")
    if (
        len(weights.shape) != 3
        or weights.shape[1:] != (HEAD_FEATURES, HEAD_FEATURES)
    ):
        raise ValueError(f"weights must have shape [H, {HEAD_FEATURES}, {HEAD_FEATURES}]")
    head_count = weights.shape[0]
    if not 1 <= head_count <= MAX_HEAD_TASKS:
        raise ValueError(f"packed head count must be between 1 and {MAX_HEAD_TASKS}")
    if biases.shape != (head_count, HEAD_FEATURES):
        raise ValueError(f"biases must have shape [H, {HEAD_FEATURES}]")
    if input.dtype not in _FP16_NAMES:
        raise TypeError("input must be float16")
    if weights.dtype not in _FP16_NAMES:
        raise TypeError("weights must be float16")
    if biases.dtype not in _FP32_NAMES:
        raise TypeError("biases must be float32")
    if input.device != weights.device or input.device != biases.device:
        raise ValueError("packed tensors must be on the same CUDA device")
    _validate_worker_groups(head_count, worker_groups)
    return PackedHeadLinearSpec(input.shape[0], head_count, worker_groups)


def validate_packed_head_output_spec(
    input: TensorSpec, weights: TensorSpec, output: TensorSpec
) -> None:
    if not isinstance(output, TensorSpec):
        raise TypeError("output must be a TensorSpec")
    _validate_shape_values("output", output)
    if not output.contiguous:
        raise ValueError("output must be contiguous")
    if not output.device.startswith("cuda"):
        raise ValueError("output must be on CUDA")
    if output.dtype not in _FP32_NAMES:
        raise TypeError("output must be float32")
    expected = (weights.shape[0], input.shape[0], HEAD_FEATURES)
    if output.shape != expected:
        raise ValueError("output must have shape [H, N, 128]")
    if output.device != input.device:
        raise ValueError("output must be on the same CUDA device as input")


def validate_whole_head_specs(
    input: TensorSpec,
    first_weight: TensorSpec,
    first_bias: TensorSpec,
    tail_weight: TensorSpec,
    tail_bias: TensorSpec,
) -> WholeHeadSpec:
    """Validate Linear(128,128)->ReLU->Linear(128,O) C4 inputs."""

    first = validate_head_linear_specs(input, first_weight, first_bias)
    for name, spec in (("tail_weight", tail_weight), ("tail_bias", tail_bias)):
        if not isinstance(spec, TensorSpec):
            raise TypeError(f"{name} must be a TensorSpec")
        _validate_shape_values(name, spec)
        if not spec.contiguous:
            raise ValueError(f"{name} must be contiguous")
        if not spec.device.startswith("cuda"):
            raise ValueError(f"{name} must be on CUDA")
        if spec.dtype not in _FP32_NAMES:
            raise TypeError(f"{name} must be float32")
    if (
        len(tail_weight.shape) != 2
        or tail_weight.shape[1] != HEAD_FEATURES
        or not 1 <= tail_weight.shape[0] <= HEAD_FEATURES
    ):
        raise ValueError("tail_weight must have shape [O, 128] with 1 <= O <= 128")
    if tail_bias.shape != (tail_weight.shape[0],):
        raise ValueError("tail_bias must have shape [O]")
    if input.device != tail_weight.device or input.device != tail_bias.device:
        raise ValueError("whole-head tensors must be on the same CUDA device")
    return WholeHeadSpec(first.rows, tail_weight.shape[0])
