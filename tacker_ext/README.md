# Functional deformation-head CUDA extension

This directory is an independent, ahead-of-time PyTorch CUDA extension for the
large `Linear(128, 128)` layers in the 4DGaussians deformation heads. ABI v1 is
preserved for the original position-head leaf. The independent ABI v2 adds
one-to-five first-linear tasks, shared-input packed heads, and a source adapter
for a complete `Linear -> ReLU -> tail Linear` head.

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
    head_linear_multi_gptb,
    head_linear_multi_gptb_out,
    head_linear_packed_gptb,
    whole_head_gptb,
    tacker_capabilities_v2,
    tacker_resources_v2,
)

output = head_linear_solo(input, weight, bias)
output = head_linear_gptb(input, weight, bias, persistent_blocks=0)

# Reuse storage owned by a frame slot / TaskGraph node.
head_linear_solo_out(input, weight, bias, output)
head_linear_gptb_out(input, weight, bias, output, persistent_blocks=0)

# Inspect the ABI facts compiled into this extension.
abi = tacker_capabilities()

# ABI v2: 1-5 independent first-linear tasks. One 128-thread worker group
# serializes all tasks; up to one group per task executes them in parallel.
outputs = head_linear_multi_gptb(
    head_inputs,
    head_weights,
    head_biases,
    worker_groups=2,
    persistent_blocks=0,
)
head_linear_multi_gptb_out(
    head_inputs,
    head_weights,
    head_biases,
    output_slots,
    worker_groups=2,
    persistent_blocks=0,
)

# C3 packed layout: weights [H,128,128], biases [H,128], output [H,N,128].
packed = head_linear_packed_gptb(
    shared_input, packed_weights, packed_biases, worker_groups=2
)

# C4 source/kernel path. Tail weights and bias remain FP32.
delta = whole_head_gptb(
    head_input, first_weight, first_bias, tail_weight, tail_bias
)

abi_v2 = tacker_capabilities_v2()
ptx_resources = tacker_resources_v2()
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

## ABI v2 device adapters

The v2 machine-readable contract is `abi/head_linear_v2.json`; it does not
replace or reinterpret `abi/head_linear_v1.json`. Its main mixed-wrapper entry
point is declared in `include/head_linear_v2_device.cuh`:

```cpp
tacker_4dgs::head_linear_multi_gptb_device(
    const tacker_4dgs::HeadLinearTaskV2* tasks,
    int task_count,                 // 1..5
    int worker_groups,              // 1..task_count
    int ptb_start_block_pos,
    int ptb_iter_block_step,
    int ptb_end_block_pos,
    int thread_base);
```

`HeadLinearTaskV2` contains `input`, `weight`, `bias`, `output`, and `rows` in
a frozen 40-byte POD layout. Worker group `g` owns task indices
`g, g + worker_groups, ...`. Each group is exactly 128 contiguous threads, and
`thread_base` must be warp-aligned. Thus the Raster wrapper can use 256 Raster
threads plus 1-5 backend groups for physical CTA widths 384, 512, 640, 768, or
896. Empty-row tasks are no-ops. Non-empty pointers, shapes, alignment, aliases,
resources, and launch status must be checked by the host wrapper before it calls
the source adapter.

The first-linear multi and packed adapters use no named barrier, CTA-wide
barrier, dynamic shared memory, or allocation. The whole-head adapter has a
separate explicit contract: the caller supplies 512 bytes of block-shared
scratch and one named barrier per worker group. For a Raster mixed wrapper the
recommended isolated barrier IDs are 2-6; Raster retains barrier 1.

ABI v2 covers all five C1 roles (`position`, `scale`, `rotation`, `opacity`,
`sh`) through the same generic symbol and manifests a canonical two-worker
`position+scale` C2 variant. `tacker_resources_v2()` returns the compiled CUDA
function attributes needed to combine ptxas data with device thread/register/
shared-memory and occupancy gates. These values are diagnostic/correctness
requirements and must be collected on the target binary rather than copied
from another build.

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
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover \
    -s tests -p 'test_*.py' -v
```

After building on `4A6000`, run numerical coverage, including the real row count,
non-tile tails, out-storage identity, and separate FP16 quantization metrics:

```bash
PYTHONPATH=. python -m unittest \
    tests.test_head_linear_cuda tests.test_head_linear_v2_cuda -v
```

The GPU test disables TF32 for both FP32 references. It compares kernel output
against FP32 accumulation of the already-quantized FP16 operands, then separately
reports FP16-vs-original-FP32 RMSE, maximum absolute error, and relative L2 error.
ABI v2 coverage additionally exercises 1-5 heads, serial and parallel worker
groups, the dual-head C2, packed heads, whole-head tail widths 1/3/4/48, empty
and non-tile rows, caller-owned output identity, repeat overwrite, pointer
alignment, alias rejection, non-default stream completion, and resource queries.
