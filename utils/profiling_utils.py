"""Low-overhead profiling helpers shared by the render path."""

import os
from contextlib import contextmanager

import torch


_NVTX_ENABLED = os.environ.get("FOURDGS_NVTX", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@contextmanager
def nvtx_range(message):
    """Emit an NVTX range only for explicitly profiled runs."""
    if not _NVTX_ENABLED:
        yield
        return

    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
