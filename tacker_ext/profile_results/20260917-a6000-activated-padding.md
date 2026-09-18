# A6000 `s.activated` padding experiment

Date: 2026-09-17

## Change

The logical activation matrix remains `[16, 128]`, but its shared-memory row
stride is padded from 128 to 136 half values:

```cpp
constexpr int kDeformActivatedStride = kDeformWidth + 8;
half activated[kDeformRows * kDeformActivatedStride];
```

The backbone ReLU writes logical `(row, column)` coordinates using the padded
stride, and the WMMA A load receives 136 as its leading dimension. The eight
extra half values add 16 bytes between rows and preserve WMMA's alignment and
leading-dimension requirements.

## Isolated A6000 build

- Source: `/home/qyfeng/deform_mlp_opt_20260917_activated_pad136_v1`
- Results: `/data/qyfeng/deform_mlp_opt_validation/20260917-activated-pad136-v1`
- GPU: physical GPU 1, NVIDIA RTX A6000, SM 8.6
- Workload: 111,525 rows, 6,971 logical tiles

PTXAS resource comparison:

| Variant | Registers/thread | Static shared memory/block |
|---|---:|---:|
| Unpadded | 73 | 14,336 bytes |
| Activated stride 136 | 74 | 14,592 bytes |

There are no spill loads or stores in either variant.

## Correctness

All seven CUDA unit tests passed. The 111,525-row matched-precision reference
check also passed with maximum absolute errors:

```text
9.536743e-7
1.192093e-6
1.311302e-6
9.536743e-7
1.668930e-6
```

These are identical to the unpadded result.

## CUDA Event A/B

Both variants were rebuilt from their corresponding source and measured on
the same physical GPU. Each value is the median of seven rounds, with 100
launches per round and 20 warm-up launches. Both use the automatic 6,971-block
full grid.

| Variant | Median milliseconds | Range |
|---|---:|---:|
| Unpadded | 3.029708 | 3.009116-3.065344 |
| Activated stride 136 | 2.833019 | 2.815610-2.865858 |

Padding reduces the median kernel time by **6.49%**, a **1.069x speedup**.

## Nsight Compute result

Report:
`/data/qyfeng/deform_mlp_opt_validation/20260917-activated-pad136-v1/source-detail.ncu-rep`

The WMMA A shared-memory load changed from eight wavefronts per ideal
wavefront to exactly the ideal count:

| WMMA A load | Actual shared wavefronts | Ideal shared wavefronts | Excess |
|---|---:|---:|---:|
| Unpadded stride 128 | 71,383,040 | 8,922,880 | 62,460,160 |
| Padded stride 136 | 8,922,880 | 8,922,880 | 0 |

Whole-kernel shared-memory totals:

| Variant | Actual wavefronts | Ideal wavefronts | Excess |
|---|---:|---:|---:|
| Unpadded | 128,600,304 | 54,986,544 | 73,613,760 |
| Padded | 66,140,144 | 54,986,544 | 11,153,600 |

Thus the padding removes all 62,460,160 excessive wavefronts previously
attributed to WMMA A. It reduces total shared wavefronts by 48.57% and
excessive shared wavefronts by 84.85%.

The remaining 11,153,600 excessive wavefronts are unchanged and come from:

- WMMA accumulator store to `s.hidden`: 6,692,160;
- `s.hidden += bias` read/modify/write: 4,461,440.

Global-sector counts are unchanged (`250,837,314` actual versus
`215,145,794` ideal), as expected because this experiment changes only the
shared-memory A operand layout. PC-sampling totals moved from 160,927 to
165,036 long-scoreboard samples and from 90,102 to 70,330 short-scoreboard
samples. These samples are not timing units and can redistribute when the
instruction schedule changes; the CUDA Event A/B is the performance result.

## Conclusion

Stride-136 padding is successful and should be retained. It completely
eliminates the dominant WMMA A bank-conflict pattern without changing the
numerical result, and it improves the real workload by 6.49%. The next shared
layout target is `s.hidden`, whose WMMA store and two-row bias update account
for every remaining excessive shared-memory wavefront.
