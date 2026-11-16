"""Inference-only FP32/16-channel sampler; CUDA import is lazy."""

from importlib import import_module
import os


def _extension():
    try:
        return import_module("tacker_4dgs_head._hexplane")
    except ImportError as exc:
        raise RuntimeError(
            "Build the HexPlane extension in tacker_ext with "
            "TACKER_BUILD_HEXPLANE_ONLY=1 TACKER_CUDA_ARCH=sm_87 "
            "python setup.py build_ext --inplace"
        ) from exc


def sample_plane(coords, plane, axis_x, axis_y, *, probe_512=False):
    return _extension().sample_plane(coords, plane, axis_x, axis_y, probe_512)


def sample_plane_out(coords, plane, output, axis_x, axis_y, *, probe_512=False):
    return _extension().sample_plane_out(
        coords, plane, output, axis_x, axis_y, probe_512)


def resources():
    keys = ("threads", "registers_per_thread", "local_bytes_per_thread",
            "static_shared_bytes", "active_blocks_per_sm", "binary_version")
    return {name: dict(zip(keys, row)) for name, row in zip(
        ("solo", "probe_512"), _extension().resources())}


def sample_planes(coords, planes, axes, *, outputs=None):
    """B: sample all 1..12 planes in one 256-thread/block kernel launch.

    Each block owns one plane and 256 points. Outputs keep the original
    grid_sample column-major [N,16] layout. Preallocate for CUDA Graph capture.
    """
    extension = _extension()
    if outputs is None:
        return tuple(extension.sample_planes(coords, tuple(planes), tuple(axes)))
    return tuple(extension.sample_planes_out(
        coords, tuple(planes), tuple(axes), tuple(outputs)))


def batch_resources():
    keys = ("threads", "registers_per_thread", "local_bytes_per_thread",
            "static_shared_bytes", "active_blocks_per_sm", "binary_version")
    return dict(zip(keys, _extension().batch_resources()))


def fused_reduction_threads():
    """Default-off inference experiment; call once while constructing a Graph."""
    enabled = os.environ.get("TACKER_HEXPLANE_FUSED_REDUCTION", "0")
    if enabled not in ("0", "1"):
        raise ValueError("TACKER_HEXPLANE_FUSED_REDUCTION must be 0 or 1")
    if enabled == "0":
        return None
    value = os.environ.get("TACKER_HEXPLANE_REDUCTION_THREADS", "256")
    if value not in ("128", "256"):
        raise ValueError("TACKER_HEXPLANE_REDUCTION_THREADS must be 128 or 256")
    return int(value)


def reduce_features(samples, *, threads=256, output=None):
    extension = _extension()
    if output is None:
        return extension.reduce_features(tuple(samples), threads)
    return extension.reduce_features_out(tuple(samples), output, threads)


def reduction_resources():
    keys = ("threads", "registers_per_thread", "local_bytes_per_thread",
            "static_shared_bytes", "active_blocks_per_sm", "binary_version")
    return {row[0]: dict(zip(keys, row)) for row in _extension().reduction_resources()}
