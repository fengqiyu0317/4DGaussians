"""PyTorch API for the functional 4DGaussians deformation-head kernels.

The compiled extension is imported lazily so parameter validation and its CPU
tests can run on machines that do not have PyTorch or CUDA installed.
"""

from importlib import import_module
from typing import Any

from .validation import (
    HEAD_FEATURES,
    HeadLinearSpec,
    TensorSpec,
    launch_metadata,
    validate_head_linear_output_spec,
    validate_head_linear_specs,
)

__all__ = [
    "HEAD_FEATURES",
    "HeadLinearSpec",
    "TensorSpec",
    "head_linear_gptb",
    "head_linear_gptb_out",
    "head_linear_solo",
    "head_linear_solo_out",
    "launch_metadata",
    "tacker_capabilities",
    "validate_head_linear_specs",
    "validate_head_linear_output_spec",
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
    if not isinstance(persistent_blocks, int) or isinstance(persistent_blocks, bool):
        raise TypeError("persistent_blocks must be an int")
    if persistent_blocks < 0:
        raise ValueError("persistent_blocks must be >= 0")
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
    if not isinstance(persistent_blocks, int) or isinstance(persistent_blocks, bool):
        raise TypeError("persistent_blocks must be an int")
    if persistent_blocks < 0:
        raise ValueError("persistent_blocks must be >= 0")
    return _extension().head_linear_gptb_out(
        input, weight, bias, output, persistent_blocks
    )
