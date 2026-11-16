# Differential Gaussian Rasterization

Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

## Tacker Raster+head inference ABI

This checkout also provides an inference-only physical mixed leaf for the
4DGaussians integration. `tacker_mix_render_head_v1` launches one 384-thread
CTA split into Raster `[0, 256)` and deformation-head GEMM `[256, 384)`.
Raster uses named barrier 1 with exactly 256 participants; the head adapter
uses only warp-local synchronization. `persistent_blocks=0` queries and caches
the active GPU's SM count.

CUDA 11.1 is the minimum supported toolkit for this `sm_86` path; CUDA 11.6 is
recommended for the target PyTorch 1.13/A6000 environment. Mixed head input,
weight, and output pointers are required to be natively 32-byte aligned for
WMMA. Normal CUDA allocator base tensors satisfy this without a copy, while a
misaligned contiguous storage view is rejected.

The Python entry point is
`GaussianRasterizer.forward_with_head(...)`, which must be called under
`torch.no_grad()`. It returns `(color, radii, depth, head_output)`, where the
last tensor is FP32 and computes `FP16 input @ FP16 weight.T + FP32 bias` with
FP32 accumulation. `tacker_capabilities()` reports the compiled ABI. See
`abi/tacker_mixed_render_head_v1.json` for the machine-readable contract.
The mixed C++ entry validates all Raster operands (including `[0]` placeholders)
as same-device CUDA FP32 tensors and uses rank/shape rather than `numel()` to
select SH versus precomputed colors and covariance versus scale/rotation. Thus
real zero-row tensors such as `[0, M, 3]` remain distinguishable from omitted
`[0]` placeholders.

Mixed ABI v2 adds
`GaussianRasterizer.forward_with_heads(..., head_inputs, head_weights,
head_biases, worker_groups, persistent_blocks)`. It accepts 1--5 independent
first-linear tasks and returns `(color, radii, depth, head_outputs)`, preserving
task order. The physical CTA has `256 + worker_groups * 128` threads (384,
512, 640, 768, or 896). Worker group `g` owns tasks `g, g+worker_groups, ...`,
so a two-head C2 candidate may use either one serialized backend group or two
parallel groups. Read-only inputs may alias across heads; outputs are distinct.
Zero rows and non-multiples of 16 are valid, while shape, dtype, device,
contiguity, 32-byte WMMA alignment, int32 overflow, worker-group, occupancy,
and launch errors fail closed.

`tacker_capabilities()` enumerates v1 and v2 without touching the device.
`tacker_variant_resources(worker_groups)` queries the active device and reports
the CTA thread count, ptxas register/static-shared/local-memory attributes, and
CUDA occupancy used by runtime admission. The raw query is
`tacker_resource_requirements(abi_version=2, worker_groups=1)`. The exact v2
contract is in `abi/tacker_mixed_render_heads_v2.json`; v1 remains callable and
byte-independent.

Mixed ABI v3 is the production C3 packed backend.  Call
`GaussianRasterizer.forward_with_packed_heads(...)` with one shared contiguous
FP16 `head_input [N,128]`, contiguous FP16
`packed_head_weights [H,128,128]`, and contiguous FP32
`packed_head_biases [H,128]`.  It returns one FP32 `[H,N,128]` tensor in the
selected-head order.  `H` is 1--5, `worker_groups` is in `[1,H]`, and every
packed head is evaluated exactly once by
`tacker_mix_render_packed_heads_v3`.  The backend adapter uses no named or
CTA-wide barrier; Raster retains barrier 1 with exactly 256 participants.  The
machine-readable contract is
`abi/tacker_mixed_render_packed_heads_v3.json`.

Mixed ABI v4 is the production C4 complete-head backend.  The plural method
`GaussianRasterizer.forward_with_whole_heads(...)` accepts 1--5 task-ordered
input/first-weight/first-bias/tail-weight/tail-bias sequences and a matching
`output_widths` sequence.  Each output computes
`Linear_fp32(ReLU(Linear_fp16(input)))` and has shape `[N_i,O_i]`, where
`1 <= O_i <= 128`; the 4DGaussians widths are 1, 3, 4, and 48.  The singular
`forward_with_whole_head(...)` method uses the same physical
`tacker_mix_render_whole_heads_v4` kernel with one worker group.  Raster uses
barrier 1/256 participants, whole-head groups use disjoint barriers 2--6/128
participants, and descriptor broadcast uses barrier 7 with exactly
`worker_groups*128` participants.  Static hidden scratch is 512 bytes per
worker group.  See `abi/tacker_mixed_render_whole_heads_v4.json`.

All C3/C4 operands and outputs require same-device CUDA storage, exact dtypes
and shapes, contiguity, and native 32-byte pointer alignment.  Output/input,
output/parameter, and output/output overlap is rejected before launch.  Empty
row sets and row tails are safe; invalid tail widths, `persistent_blocks`,
resource configurations, and launch errors fail closed.  An empty Raster does
not suppress non-empty C3/C4 work, while an entirely empty call launches no
kernel.

The family-aware resource APIs are
`tacker_resource_requirements(abi_version, worker_groups, family)` and
`tacker_variant_resources(worker_groups, family=...)`.  Families are
`first_linear_heads_v2`, `packed_first_linear_v3`, and `whole_heads_v4`.
Reports include both `backend_abi_version` and `backend_family`, preventing a
candidate from reusing resource evidence captured from another physical
kernel family.  The old two-argument query remains equivalent to the v1/v2
first-linear query.

The legacy forward, visibility, and autograd backward paths enqueue every CUDA
kernel on PyTorch's current stream. The small rendered-count host readback is a
stream-local synchronization required to size the binning buffers; it is never
a device-wide synchronization.

CPU-only source contract tests do not import PyTorch:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_tacker_mixed_contract -v
```

After building the extension on `4A6000`, the CUDA boundary cases (native
alignment and zero-row ranked alternatives) can be run with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_tacker_mixed_cuda -v
```

The CUDA suite covers 1--5 heads, every legal worker-group count, a dual-head
C2 variant, C3 packed and C4 single/multi complete-head configurations,
zero/tail rows, shared-input aliasing, variable output widths, fail-closed
argument errors, non-default streams, and per-family runtime resource queries.
Numerical head tolerance is
`atol=rtol=2e-3`; full Raster color/depth/radii equivalence still requires the
configured A6000 qualification run.

The legacy training API has a separate GPU smoke test that renders one real
Gaussian and runs autograd forward/backward on a non-default PyTorch stream:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  tests.test_stream_aware_legacy_cuda -v
```

CUDA compilation, numerical comparison against the legacy Raster path, head
reference checks, Raster QoS, and performance qualification must be run on the
configured RTX A6000 server.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>
