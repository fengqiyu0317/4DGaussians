# Phase 0 A6000 FPS baseline

This directory freezes the Phase 0 whole-sequence FPS baseline for the first
Tacker workload.

- Workload: `flame_steak`, iteration 14000, test views 0–49
- Shape: 111,525 Gaussians at 1352 × 1014
- Device: GPU 1, NVIDIA RTX A6000 (`GPU-cd6fce20-09cc-52ba-6968-28e8f796d7c8`)
- Schedule: deterministic ABBA, seed 0
- Sampling: 10 trials per mode, 50 measured frames and 10 warmup frames per trial
- Bootstrap: 10,000 paired-round resamples, seed 0
- Primary metric: `median_throughput_fps`, higher is better
- Aggregate report SHA-256: `ef0fec9020f81a68cf0f7fe6ec7c2e059d6afd7766bfb78bc40c3755f14a509e`

| Rank | Mode | Median FPS | Median total render time | Trial FPS range |
|---:|---|---:|---:|---:|
| 1 | current Tacker | 86.979719 | 574.846646 ms | 86.597897–87.099374 |
| 2 | two-stream | 86.254250 | 579.681585 ms | 85.856130–86.436900 |
| 3 | serial | 82.423385 | 606.623962 ms | 81.264402–82.890268 |

Current Tacker / two-stream has a ratio-of-medians of `1.0084108268`.
The paired bootstrap 95% confidence interval is
`[1.0061090994, 1.0097120318]`.

`baseline-report.json` is the aggregate, fail-closed report. The `raw/`
directory contains the 30 independent `profile_render.py` metadata documents
used as its inputs. Every Tacker document reports physical `tacker` execution,
and no document reports a Tacker or two-stream fallback.

The remote source tree did not include `.git` metadata. Consequently, the main
commit and submodule commits in the report are explicitly marked as
environment-provided provenance. Exact deployed content remains bound by the
recorded SHA-256 values for `profile_render.py`, the interleaved driver, the
workload config, and the admitted profile.
