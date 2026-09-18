# A6000 automatic full-grid optimization

Date: 2026-09-17

## Change

For `deform_mlp_delta_v1`, `persistent_blocks=0` now launches one block per
16-row logical tile. Positive values retain the previous explicit
persistent-grid behavior and remain capped by the logical tile count.

The previous automatic policy used one block per SM. On the target RTX A6000
that provided only four active warps per SM for this 128-thread,
barrier-heavy kernel.

## Isolated remote build

- Source: `/home/qyfeng/deform_mlp_opt_20260917_auto_fullgrid_v1`
- Validation: `/data/qyfeng/deform_mlp_opt_validation/20260917-auto-fullgrid-v1`
- GPU: physical GPU 1, NVIDIA RTX A6000, SM 8.6, 84 SMs
- PyTorch: 2.4.1+cu124
- CUDA toolkit: 12.4
- Workload: 111,525 rows, 6,971 logical tiles

## Correctness

All seven existing CUDA unit tests passed. The 111,525-row matched-precision
reference check also passed with the same maximum absolute errors as the
pre-optimization binary:

```text
9.536743e-7
1.192093e-6
1.311302e-6
9.536743e-7
1.668930e-6
```

## CUDA Event timing

Each entry is the median of seven rounds, with 100 launches per round and 20
warm-up launches per configuration.

| Requested persistent blocks | Median ms |
|---:|---:|
| 0 (new automatic full-grid) | 3.030723 |
| 84 | 7.558073 |
| 168 | 4.404396 |
| 336 | 3.114056 |
| 672 | 3.554171 |
| 6,971 (explicit full-grid) | 3.074375 |

Compared with the pre-change automatic median of 7.540038 ms, the new default
is 2.488x faster and reduces kernel time by 59.80%. The small difference
between automatic and explicit full-grid measurements is run-order noise; both
launch the same grid.

## Nsight Compute verification

The optimized default launch was directly verified as:

```text
Grid Size                       6,971
Block Size                        128
Achieved Occupancy              49.31%
Achieved Active Warps Per SM    23.67
Waves Per SM                    13.83
L2 Cache Throughput             57.03%
DRAM Throughput                  1.48%
```

Report:
`/data/qyfeng/deform_mlp_opt_validation/20260917-auto-fullgrid-v1/overview-default-auto-fullgrid.ncu-rep`

The production renderer default is unchanged because this standalone MLP is
not yet wired into the renderer default path; this change only updates the
standalone extension API's automatic grid policy.
