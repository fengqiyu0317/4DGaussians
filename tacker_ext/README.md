# Functional deformation-head CUDA extension

This directory is an independent, ahead-of-time PyTorch CUDA extension for the
large `Linear(128, 128)` layers in the 4DGaussians deformation heads.  It never
allocates scratch A/B/C buffers: every kernel consumes the real tensor pointers
passed by PyTorch.

## Build on the A6000 server

```bash
cd /home/qyfeng/4DGaussians/tacker_ext
python setup.py build_ext --inplace
```

The build is fixed to `sm_86`. CUDA 11.1 is the minimum supported toolkit and
CUDA 11.6 is recommended for the target PyTorch 1.13/A6000 environment. PyTorch,
its matching CUDA toolkit, and a C++17 compiler must already be installed.

## API and layout

```python
from tacker_4dgs_head import (
    head_linear_gptb,
    head_linear_gptb_out,
    head_linear_solo,
    head_linear_solo_out,
    tacker_capabilities,
)

output = head_linear_solo(input, weight, bias)
output = head_linear_gptb(input, weight, bias, persistent_blocks=0)

# Reuse storage owned by a frame slot / TaskGraph node.
head_linear_solo_out(input, weight, bias, output)
head_linear_gptb_out(input, weight, bias, output, persistent_blocks=0)

# Inspect the ABI facts compiled into this extension.
abi = tacker_capabilities()
```

Both functions implement `input @ weight.T + bias`:

- `input`: contiguous CUDA FP16 `[N, 128]`
- `weight`: contiguous CUDA FP16 `[128, 128]`, standard PyTorch `[out, in]`
- `bias`: contiguous CUDA FP32 `[128]`
- output: contiguous CUDA FP32 `[N, 128]`

The WMMA operands and output must have a native 32-byte-aligned data pointer.
Normal base tensors returned by the CUDA allocator satisfy this without a copy;
a contiguous slice with a misaligned storage offset is rejected explicitly.

The `_out` variants strictly require caller-owned contiguous CUDA FP32
`output[N, 128]` on the input device and return the same Tensor/storage. The
allocating APIs remain compatible but delegate to these out implementations.
Caller-owned output storage must not overlap input, weight, or bias storage.
All entry points are inference-only and therefore require `torch.no_grad()`.

`head_linear_solo` launches one 128-thread block per logical `16x64` output
tile. `head_linear_gptb` launches a persistent grid, defaulting to one physical
block per SM. Both use FP16 WMMA operands and FP32 accumulation. A final partial
row tile is handled by an FP32-accumulating scalar path and is not rounded down.

The reusable mixed-wrapper adapter is defined entirely in
`include/head_linear_device.cuh`:

```cpp
tacker_4dgs::head_linear_gptb_device(
    input, weight, bias, output, rows,
    logical_grid_x, logical_grid_y,
    ptb_start_block_pos, ptb_iter_block_step, ptb_end_block_pos,
    thread_base);
```

A mixed CTA must assign exactly 128 contiguous threads at a warp-aligned base.
The intended Raster+GEMM layout is Raster `[0, 256)` and GEMM `[256, 384)` in a
384-thread CTA. The adapter uses warp-local synchronization only, no named CTA
barrier, and no dynamic shared memory, so Raster retains its barrier namespace.
The machine-readable argument order is in `abi/head_linear_v1.json`. Storage
lifetime and stream/event ordering remain the caller's responsibility.

The CUDA entrypoints use stable versioned C symbols `tacker_head_linear_solo_v1`
and `tacker_head_linear_gptb_v1`; their pointer/argument ABI and adapter order are
recorded in `abi/head_linear_v1.json` for the future Tacker registry. The
compiled `tacker_capabilities()` query reports those symbols, dtypes, launch
width, and `sm_86` target without relying on ELF dynamic-symbol visibility.

## Python/PyTorch ABI compatibility

The Python sources and package declaration target Python 3.7+. Build this
extension inside the exact PyTorch environment used by 4DGaussians. In
particular, PyTorch 1.13 propagates its `_GLIBCXX_USE_CXX11_ABI` choice into
`CUDAExtension`; a Tacker library compiled with the opposite libstdc++ ABI is
not link-compatible. This scaffold deliberately does **not** link an independent
`libtacker_runtime`. Runtime integration should register the versioned CUDA
symbol/driver handle or compile a mixed wrapper from the public device header.

## Tests

CPU-only ABI validation does not import PyTorch:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest \
    tests.test_validation tests.test_contract -v
```

After building on `4A6000`, run numerical coverage, including the real row count,
non-tile tails, out-storage identity, and separate FP16 quantization metrics:

```bash
PYTHONPATH=. python -m unittest tests.test_head_linear_cuda -v
```

The GPU test disables TF32 for both FP32 references. It compares kernel output
against FP32 accumulation of the already-quantized FP16 operands, then separately
reports FP16-vs-original-FP32 RMSE, maximum absolute error, and relative L2 error.
