# Phase 2 A6000 targeted qualification

This directory freezes the verified facts from the Phase 2 targeted A6000 run.
It is qualification evidence for physical execution, ABI/resource gating, and
CUDA correctness. It is not a Phase 3 interleaved benchmark, an admission
report, or a winner profile.

## Execution environment

- Connection: Windows OpenSSH client, host alias `4A6000`
- Isolated snapshot: `/data/qyfeng/tacker_phase2_validation/20260910-codex-phase2`
- Runtime: PyTorch `2.4.1+cu124`, CUDA 12.4
- Device: physical GPU 1, exposed as CUDA logical device 0 through
  `CUDA_VISIBLE_DEVICES=1`
- Workload: `flame_steak`, iteration 14000, test views 0–49
- Shape: 111,525 Gaussians at 1352 × 1014

`nvidia-smi` could not initialize NVML because of a driver/library version
mismatch involving version `580.173`. PyTorch CUDA compilation, extension
loading, resource queries, and kernel execution nevertheless completed on the
device. This targeted run therefore does not claim that the full Phase 4
qualification preflight passed.

The snapshot is separate from the existing remote repository and its dirty
worktree. Build logs and raw per-candidate metadata remain under the snapshot's
`logs/` directory.

## Sealed ABIs and CUDA tests

- Head v1 manifest SHA-256:
  `24570aa6e67e8b9b10fa94524fec4dc03a4eb3fdc3bf822af34c2c52ce4937ac`
- Raster v1 manifest SHA-256:
  `231c90c429321b2673b88ecd09efb40b6aedda7a23f3e061a2bcedec06d44426`
- Head v2 manifest SHA-256:
  `9d6a1558acd6b642b975bcabe22abcbe3fd7242e4c9e0d635636ef4d2eb5da7f`
- Raster v2 manifest SHA-256:
  `310b15957c5920773bb03a61a37c5771f6d4570393061ece4e1805fd20989056`
- Head CUDA suite: 16/16 passed
- Raster v2 plus legacy CUDA suite: 12/12 passed
- Complete remote suites after the final rebuild: head 60/60, Raster 34/34

The ptxas results were:

| Kernel family | Registers/thread | Static shared memory | Stack | Spills |
|---|---:|---:|---:|---:|
| mixed v2 | 68 | 7376 B | — | 0 |
| whole head | 32 | 512 B | — | 0 |
| packed head | 48 | — | — | 0 |
| multi GPTB / solo | 48 | — | 200 B | 0 |

The runtime resource query reported a 896-thread kernel limit and
`launch_supported=true` for every queried worker-group count:

| Worker groups | Block threads | Registers/thread | Static shared memory | Active blocks/SM | Occupancy |
|---:|---:|---:|---:|---:|---:|
| 1 | 384 | 68 | 7376 B | 2 | 0.500000 |
| 2 | 512 | 68 | 7376 B | 1 | 0.333333 |
| 3 | 640 | 68 | 7376 B | 1 | 0.416667 |
| 4 | 768 | 68 | 7376 B | 1 | 0.500000 |
| 5 | 896 | 68 | 7376 B | 1 | 0.583333 |

## Physical 50-view runs

All seven generated profiles remained disabled and were run explicitly in
qualification mode. Each run used 10 warmup views followed by one 50-view
trial. Both the initial run and the final sealed-gate execution-count replay
reported `actual_execution_mode=tacker` and `tacker_fallback_reason=null` for
every candidate.

| Candidate | Partition | Worker groups | Initial FPS | Final sealed-gate FPS |
|---|---|---:|---:|---:|
| `c0_pos_l1_pb7000` | C0 pos | 1 | 87.629535 | 87.983156 |
| `c1_opacity_l1_pb7000` | C1 opacity | 1 | 85.339025 | 85.842899 |
| `c1_pos_l1_pb7000` | C1 pos | 1 | 85.846198 | 85.779940 |
| `c1_rotations_l1_pb7000` | C1 rotations | 1 | 85.068259 | 85.229702 |
| `c1_scales_l1_pb7000` | C1 scales | 1 | 85.128992 | 84.982640 |
| `c1_shs_l1_pb7000` | C1 SH | 1 | 84.745141 | 85.338641 |
| `c2_pos_scales_l1_wg2_pb7000` | C2 pos + scales | 2 | 67.083068 | 66.852933 |

These single-trial FPS values are diagnostics only. They were not interleaved,
do not satisfy Phase 3's repeated-trial statistics, and must not be used to
declare a winner.

## Execution-count replay

Every counted replay produced the same scheduler invariant:

| Counter | Observed value |
|---|---:|
| input frames | 50 |
| full deformation | 1 |
| prefix | 49 |
| mixed launches | 49 |
| suffix | 49 |
| solo raster | 1 |
| outputs | 50 |
| selected-head evaluations per selected head | 50 |

The final run also compared the profile-declared ABI hashes, symbols, resource
contract, worker-group table, and named barriers against capabilities exported
by the loaded Raster binary. The runtime checks the execution-count invariant
before publishing the final output. Together with the unit test that keeps
selected Python `Linear` calls at zero after prefill, this establishes the
Phase 2 no-duplicate condition. The seven raw
replay documents are named `*-final-50view.json` under the remote snapshot's
`logs/` directory. Phase 2's targeted exit condition is complete.

This does not promote any candidate. Full image-quality gates, interleaved
multi-trial ranking, and deployment admission remain Phase 3/4 work.

`resource-report.json` is the machine-readable summary of the same facts.
