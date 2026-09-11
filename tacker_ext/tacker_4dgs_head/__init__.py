"""PyTorch API for the functional 4DGaussians deformation-head kernels.

The compiled extension is imported lazily so parameter validation and its CPU
tests can run on machines that do not have PyTorch or CUDA installed.
"""

from importlib import import_module
from typing import Any

from .validation import (
    HEAD_FEATURES,
    MAX_HEAD_TASKS,
    WORKER_GROUP_THREADS,
    HeadLinearSpec,
    MultiHeadLinearSpec,
    PackedHeadLinearSpec,
    TensorSpec,
    WholeHeadSpec,
    launch_metadata,
    multi_launch_metadata,
    validate_head_linear_output_spec,
    validate_head_linear_specs,
    validate_multi_head_linear_specs,
    validate_multi_head_output_specs,
    validate_packed_head_linear_specs,
    validate_packed_head_output_spec,
    validate_whole_head_specs,
)

__all__ = [
    "HEAD_FEATURES",
    "MAX_HEAD_TASKS",
    "WORKER_GROUP_THREADS",
    "HeadLinearSpec",
    "MultiHeadLinearSpec",
    "PackedHeadLinearSpec",
    "TensorSpec",
    "WholeHeadSpec",
    "head_linear_gptb",
    "head_linear_gptb_out",
    "head_linear_multi_gptb",
    "head_linear_multi_gptb_out",
    "head_linear_multi_solo",
    "head_linear_multi_solo_out",
    "head_linear_packed_gptb",
    "head_linear_packed_gptb_out",
    "head_linear_solo",
    "head_linear_solo_out",
    "launch_metadata",
    "multi_launch_metadata",
    "tacker_capabilities",
    "tacker_capabilities_v2",
    "tacker_resources_v2",
    "validate_head_linear_specs",
    "validate_head_linear_output_spec",
    "validate_multi_head_linear_specs",
    "validate_multi_head_output_specs",
    "validate_packed_head_linear_specs",
    "validate_packed_head_output_spec",
    "validate_whole_head_specs",
    "whole_head_gptb",
]


def _tensor_spec(tensor: Any) -> TensorSpec:
    dtype = str(tensor.dtype)
    if dtype.startswith("torch."):
        dtype = dtype[len("torch.") :]
    return TensorSpec(
        shape=tuple(tensor.shape),
        dtype=dtype,
        device=str(tensor.device),
        contiguous=bool(tensor.is_contiguous()),
    )


def _validate_tensors(input: Any, weight: Any, bias: Any) -> HeadLinearSpec:
    return validate_head_linear_specs(
        _tensor_spec(input), _tensor_spec(weight), _tensor_spec(bias)
    )


def _extension():
    try:
        return import_module("tacker_4dgs_head._C")
    except ImportError as exc:
        raise RuntimeError(
            "tacker_4dgs_head CUDA extension is not built; run "
            "`python setup.py build_ext --inplace` from 4DGaussians/tacker_ext"
        ) from exc


def tacker_capabilities():
    """Return the ABI facts reported by the compiled CUDA extension."""

    provider = getattr(_extension(), "tacker_capabilities", None)
    if not callable(provider):
        raise RuntimeError(
            "tacker_4dgs_head CUDA extension does not expose ABI capabilities"
        )
    return dict(provider())


def tacker_capabilities_v2():
    """Return the independently versioned multi/packed/whole-head contract."""

    provider = getattr(_extension(), "tacker_capabilities_v2", None)
    if not callable(provider):
        raise RuntimeError(
            "tacker_4dgs_head CUDA extension does not expose ABI v2 capabilities"
        )
    return dict(provider())


def tacker_resources_v2():
    """Query CUDA function attributes used for runtime resource filtering."""

    provider = getattr(_extension(), "tacker_resources_v2", None)
    if not callable(provider):
        raise RuntimeError(
            "tacker_4dgs_head CUDA extension does not expose ABI v2 resources"
        )
    return dict(provider())


def _tensor_sequence(name: str, tensors: Any):
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of tensors")
    return tuple(tensors)


def _validate_multi_tensors(
    inputs: Any, weights: Any, biases: Any, worker_groups: int
):
    input_tensors = _tensor_sequence("inputs", inputs)
    weight_tensors = _tensor_sequence("weights", weights)
    bias_tensors = _tensor_sequence("biases", biases)
    spec = validate_multi_head_linear_specs(
        tuple(_tensor_spec(value) for value in input_tensors),
        tuple(_tensor_spec(value) for value in weight_tensors),
        tuple(_tensor_spec(value) for value in bias_tensors),
        worker_groups,
    )
    return input_tensors, weight_tensors, bias_tensors, spec


def _validate_persistent_blocks(persistent_blocks: int) -> None:
    if not isinstance(persistent_blocks, int) or isinstance(persistent_blocks, bool):
        raise TypeError("persistent_blocks must be an int")
    if persistent_blocks < 0:
        raise ValueError("persistent_blocks must be >= 0")


def head_linear_solo(input: Any, weight: Any, bias: Any):
    """Run ``input @ weight.T + bias`` with one CUDA block per logical tile."""

    _validate_tensors(input, weight, bias)
    return _extension().head_linear_solo(input, weight, bias)


def head_linear_solo_out(input: Any, weight: Any, bias: Any, output: Any):
    """Write the solo result into caller-owned contiguous FP32 storage."""

    input_spec = _validate_tensors(input, weight, bias)
    validate_head_linear_output_spec(_tensor_spec(input), _tensor_spec(output))
    if tuple(output.shape) != (input_spec.rows, HEAD_FEATURES):
        raise ValueError("output must have shape [N, 128]")
    return _extension().head_linear_solo_out(input, weight, bias, output)


def head_linear_gptb(
    input: Any, weight: Any, bias: Any, persistent_blocks: int = 0
):
    """Run the same Linear using a grid of persistent GPTB blocks.

    ``persistent_blocks=0`` asks the extension to use the current device's SM
    count.  A positive value is useful for resource/QoS profiling.
    """

    _validate_tensors(input, weight, bias)
    _validate_persistent_blocks(persistent_blocks)
    return _extension().head_linear_gptb(input, weight, bias, persistent_blocks)


def head_linear_gptb_out(
    input: Any,
    weight: Any,
    bias: Any,
    output: Any,
    persistent_blocks: int = 0,
):
    """Write the persistent result into caller-owned contiguous FP32 storage."""

    _validate_tensors(input, weight, bias)
    validate_head_linear_output_spec(_tensor_spec(input), _tensor_spec(output))
    _validate_persistent_blocks(persistent_blocks)
    return _extension().head_linear_gptb_out(
        input, weight, bias, output, persistent_blocks
    )


def head_linear_multi_solo(
    inputs: Any, weights: Any, biases: Any, worker_groups: int = 1
):
    """Run one to five independent first-linear heads in one CUDA launch."""

    values = _validate_multi_tensors(inputs, weights, biases, worker_groups)
    return tuple(
        _extension().head_linear_multi_solo(
            values[0], values[1], values[2], worker_groups
        )
    )


def head_linear_multi_solo_out(
    inputs: Any,
    weights: Any,
    biases: Any,
    outputs: Any,
    worker_groups: int = 1,
):
    """Write a multi-head solo launch into one caller-owned output per task."""

    values = _validate_multi_tensors(inputs, weights, biases, worker_groups)
    output_tensors = _tensor_sequence("outputs", outputs)
    validate_multi_head_output_specs(
        tuple(_tensor_spec(value) for value in values[0]),
        tuple(_tensor_spec(value) for value in output_tensors),
    )
    return tuple(
        _extension().head_linear_multi_solo_out(
            values[0], values[1], values[2], output_tensors, worker_groups
        )
    )


def head_linear_multi_gptb(
    inputs: Any,
    weights: Any,
    biases: Any,
    worker_groups: int = 1,
    persistent_blocks: int = 0,
):
    """Run one to five first-linear tasks with persistent GPTB traversal."""

    values = _validate_multi_tensors(inputs, weights, biases, worker_groups)
    _validate_persistent_blocks(persistent_blocks)
    return tuple(
        _extension().head_linear_multi_gptb(
            values[0],
            values[1],
            values[2],
            worker_groups,
            persistent_blocks,
        )
    )


def head_linear_multi_gptb_out(
    inputs: Any,
    weights: Any,
    biases: Any,
    outputs: Any,
    worker_groups: int = 1,
    persistent_blocks: int = 0,
):
    """Persistent multi-head launch into one caller-owned output per task."""

    values = _validate_multi_tensors(inputs, weights, biases, worker_groups)
    output_tensors = _tensor_sequence("outputs", outputs)
    validate_multi_head_output_specs(
        tuple(_tensor_spec(value) for value in values[0]),
        tuple(_tensor_spec(value) for value in output_tensors),
    )
    _validate_persistent_blocks(persistent_blocks)
    return tuple(
        _extension().head_linear_multi_gptb_out(
            values[0],
            values[1],
            values[2],
            output_tensors,
            worker_groups,
            persistent_blocks,
        )
    )


def head_linear_packed_gptb(
    input: Any,
    weights: Any,
    biases: Any,
    worker_groups: int = 1,
    persistent_blocks: int = 0,
):
    """Run shared-input packed first-linear heads with layout ``[H,...]``."""

    validate_packed_head_linear_specs(
        _tensor_spec(input),
        _tensor_spec(weights),
        _tensor_spec(biases),
        worker_groups,
    )
    _validate_persistent_blocks(persistent_blocks)
    return _extension().head_linear_packed_gptb(
        input, weights, biases, worker_groups, persistent_blocks
    )


def head_linear_packed_gptb_out(
    input: Any,
    weights: Any,
    biases: Any,
    output: Any,
    worker_groups: int = 1,
    persistent_blocks: int = 0,
):
    """Write packed first-linear heads to caller-owned ``[H,N,128]`` storage."""

    input_spec = _tensor_spec(input)
    weight_spec = _tensor_spec(weights)
    validate_packed_head_linear_specs(
        input_spec, weight_spec, _tensor_spec(biases), worker_groups
    )
    validate_packed_head_output_spec(input_spec, weight_spec, _tensor_spec(output))
    _validate_persistent_blocks(persistent_blocks)
    return _extension().head_linear_packed_gptb_out(
        input, weights, biases, output, worker_groups, persistent_blocks
    )


def whole_head_gptb(
    input: Any,
    first_weight: Any,
    first_bias: Any,
    tail_weight: Any,
    tail_bias: Any,
    persistent_blocks: int = 0,
):
    """Run ``Linear(128,128) -> ReLU -> Linear(128,O)`` as one head."""

    validate_whole_head_specs(
        _tensor_spec(input),
        _tensor_spec(first_weight),
        _tensor_spec(first_bias),
        _tensor_spec(tail_weight),
        _tensor_spec(tail_bias),
    )
    _validate_persistent_blocks(persistent_blocks)
    return _extension().whole_head_gptb(
        input,
        first_weight,
        first_bias,
        tail_weight,
        tail_bias,
        persistent_blocks,
    )
