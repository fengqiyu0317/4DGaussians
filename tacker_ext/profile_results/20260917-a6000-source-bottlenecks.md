# A6000 deform-MLP source-level bottleneck attribution

Date: 2026-09-17

## Scope

- GPU: NVIDIA RTX A6000, SM 8.6
- Workload: 111,525 rows, 6,971 logical tiles
- Launch: 6,971 blocks x 128 threads
- NCU report:
  `/data/qyfeng/deform_mlp_ncu/20260916-codex-v1/detail-warm-pb6971.ncu-rep`
- Kernel: `tacker_deform_mlp_delta_v1`

The report was captured from the pre-default-grid-change binary using an
explicit 6,971-block launch. The device function is unchanged, so its source
and SASS attribution applies to the current automatic full-grid kernel.

## Shared-memory conflicts

NCU reported 73,613,760 excessive shared-memory wavefronts. The source view
accounts for every one of them:

| Source operation | Actual wavefronts | Ideal wavefronts | Excess | Share of all excess |
|---|---:|---:|---:|---:|
| WMMA A load from `s.activated` (line 83; intrinsic line 91) | 71,383,040 | 8,922,880 | 62,460,160 | 84.85% |
| WMMA accumulator store to `s.hidden` (line 89; intrinsic line 474) | 8,922,880 | 2,230,720 | 6,692,160 | 9.09% |
| `s.hidden += bias` read/modify/write (line 96) | 8,922,880 | 4,461,440 | 4,461,440 | 6.06% |
| **Total** | | | **73,613,760** | **100.00%** |

The dominant conflict is therefore the shared-memory WMMA A load:

```cpp
wmma::load_matrix_sync(a, s.activated + k, kDeformWidth);
```

Its compiled `__hmma_m16n16k16_ld_a` path needs eight times the ideal number
of shared-memory wavefronts. The accumulator store needs four times ideal,
and the following bias update needs twice ideal. Other major shared-memory
accesses, including the backbone and tail reads, had actual wavefront counts
equal to their ideal counts.

## Excessive global sectors

The WMMA B load is the entire source of the kernel's 35,691,520 excessive
global-memory sectors:

| Source operation | Actual sectors | Ideal sectors | Excess |
|---|---:|---:|---:|
| WMMA B load from `first_weights` (line 85; intrinsic line 103) | 71,383,040 | 35,691,520 | 35,691,520 |

The compiled `__hmma_m16n16k16_ld_b` path therefore transfers twice the ideal
number of sectors. The other important global reads are coalesced according
to this metric:

| Global read | Actual sectors | Ideal sectors |
|---|---:|---:|
| `backbone_weight` (line 49) | 57,100,800 | 57,100,800 |
| `tail_weights` (line 124) | 105,289,984 | 105,289,984 |
| `first_biases` on line 96 | 1,673,040 | 1,673,040 |
| `tail_biases` on line 132 | 6,579,975 | 6,579,975 |

Coalescing removes unnecessary sectors, but it does not remove the load
latency seen by a dependent instruction.

## Long-scoreboard attribution

The kernel has 160,927 long-scoreboard PC samples. NCU records the stall on
the instruction waiting for a result, not necessarily on the producer LDG.
The SASS register dependencies allow the waiting instructions to be mapped
back to their global loads:

| Global producer | Sampled consumer/source | Samples | Share |
|---|---|---:|---:|
| `tail_weights` LDGs (line 124) | tail FFMA, line 123 | 80,959 | 50.31% |
| `tail_biases` LDG | final FADD, line 132 | 29,178 | 18.13% |
| `backbone_weight` LDGs (line 49) | backbone FFMAs, line 48 | 24,714 | 15.36% |
| WMMA B / `first_weights` load | WMMA MMA consumers, line 87 and intrinsic line 626 | 20,436 | 12.70% |
| `first_biases` LDGs | bias add, line 96 | 4,314 | 2.68% |
| input `features` load | feature staging, line 148 | 645 | 0.40% |
| Other sampled instructions | | 681 | 0.42% |
| **Total** | | **160,927** | **100.00%** |

For example, the SASS for the tail path issues `LDG.E` into registers
`R10/R19/R21/R23/R22`. The long-scoreboard samples appear on the later
`FFMA` instructions that consume those registers. Similarly, the tail-bias
path is an `LDG.E R14` followed by a sampled `FADD` consuming `R14`.

The 75,096 short-scoreboard samples on line 128 are a separate issue: they
belong to the warp-shuffle reduction and are not global-memory latency.

## Conclusion and next targets

This is not a DRAM-bandwidth bottleneck. The full-grid profile has a 99.73%
L2 hit rate and only 1.47% DRAM throughput. It is a combination of:

1. a severe shared-memory layout conflict in the WMMA A load;
2. a two-times-sector WMMA B global load;
3. dependent global reads, especially `tail_weights`, whose cache-hit latency
   is exposed as long-scoreboard stalls; and
4. the tail shuffle reduction's short-scoreboard dependency chain.

The first layout experiment pads `s.activated` so consecutive WMMA rows do not
begin on the same bank pattern. It removes all 62,460,160 excessive WMMA A
wavefronts and improves CUDA Event time by 6.49%; see
`20260917-a6000-activated-padding.md`. A corresponding padded `s.hidden`
layout can address the accumulator-store and two-row bias conflicts. After
that, the WMMA B weight layout and the tail path should be benchmarked
separately; the source data does not support treating general L2 traffic as
the sole root cause.
