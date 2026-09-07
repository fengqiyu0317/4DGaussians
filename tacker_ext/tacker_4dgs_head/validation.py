"""Torch-independent validation for the fixed 128x128 head Linear ABI."""

from dataclasses import dataclass
from typing import Tuple


HEAD_FEATURES = 128
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
