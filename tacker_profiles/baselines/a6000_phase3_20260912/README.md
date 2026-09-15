# RTX A6000 Phase 3 evidence — 2026-09-12

This directory is the compact, hash-addressed evidence bundle for the completed
offline Phase 3 Tacker search on `flame_steak`. The full run remains at
`/data/qyfeng/tacker_phase3_validation/20260911-codex-phase3/run-complete` on
the `4A6000` host; large per-trial outputs and raw `.nsys-rep` files are not
duplicated here.

## Sealed identity and protocol

- Isolated source commit: `f31762793604b66e7e4330d3b6254f053d73ab5b`
- Run identity SHA-256: `b8f04b8fce967e4fd6a479b48265a59ea80a9cea9b05d546c7cf183a3cc0e8d8`
- Device: physical GPU 1, NVIDIA RTX A6000 (`sm_86`)
- Workload: `flame_steak`, iteration 14000, 111,525 Gaussians,
  1352 × 1014, test split
- Formal protocol: warmup 10, 50 frames, 10 trials per entry, ABBA schedule,
  seed 0
- Final matrix: 366 candidates; sealed matrix SHA-256
  `8956a212b0ee99d9cf17fb2de3a43e793745eadfcc0651924fb135e2603c4295`

The hierarchical search completed 78/78, 246/246, 222/222, 150/150, and
66/66 short E2E child runs for H1 through H5: 762/762 in total, with no child
errors. All five selected candidates passed the variant leaf gate and the
50-view PSNR/SSIM/LPIPS gate. The formal run completed 80/80 whole-sequence
measurements (five candidates plus three baselines, ten trials each).

## Result

| Final rank | Entry | Median FPS |
|---:|---|---:|
| 1 | `c2h5_pos_scales_rotations_opacity_shs_l1_wg1_pb5440` | 89.7762405051 |
| 2 | `c2h5_pos_scales_rotations_opacity_shs_l1_wg1_pb7000` | 88.8638958963 |
| 3 | `c2h3_scales_rotations_opacity_l1_wg1_pb7000` | 88.4582866269 |
| 4 | `c2h3_scales_opacity_shs_l1_wg1_pb5440` | 87.8672991961 |
| 5 | `c2h4_pos_scales_opacity_shs_l1_wg1_pb5440` | 87.6867304997 |
| 6 | `current_tacker` | 87.5359422893 |
| 7 | `two_stream` | 86.5772121031 |
| 8 | `serial` | 82.9829919708 |

The experimental argmax and deployment winner are both the rank-1 entry. Its
ratio-of-medians versus `current_tacker` is 1.0255928954, with paired bootstrap
95% CI `[1.0215977858, 1.0297240659]`; versus `two_stream` it is 1.0369500048,
with CI `[1.0334421818, 1.0396602793]`. The recorded decision is
`promote_challenger` with reason code `promotion_gates_passed`.

This is a Phase 3 selection result, not a deployed profile. The archived winner
is deliberately a qualification profile with `deployment.enabled=false` and
`deployment.valid=false`; Phase 4 owns admission and rollout.

## Nsight interpretation

The formal top three each passed a 50-frame Nsight Systems run. The winner's
diagnostic render FPS is 82.6741330624; it records 121 launches/frame, 17 stream
synchronizations/frame, a `renderer/setup` critical-path proxy, and 1.87753406
ms/frame of idle/gap proxy. Its dominant category is
`tacker_mixed_render_head` at 3.5707962 ms/frame.

The first Nsight attempt failed closed when the 2023.4.4.54 qdstrm importer
reported `Wrong event order`. Resuming with the same source, input hashes,
`/usr/local/cuda-12.4/bin/nsys`, and run identity succeeded on attempt 2. The
checkpoint preserved the completed 762 screening runs, five correctness gates,
and 80 formal measurements; none were repeated.

## Reproducibility and scope

Postflight validation reopened and checked the SQLite database, rebuilt H3–H5,
regenerated all 366 qualification profiles, and recomputed the formal plan.
The final matrix is byte-identical, normalized formal selection records and
correctness qualifications are identical, and all 366 profile file hashes
match. See `reproducibility.json` for the machine-readable checks and
`artifact-manifest.json` for every local artifact hash and remote source path.

The measured candidate space is the C0–C2 first-linear family already connected
to the production Raster execution path, covering one through five selected
heads and the worker-group/persistent-block grid. Phase 2's C3 packed and C4
whole-head adapters are not yet wired into that runtime search path, so this
bundle does not claim that they participated in the ranking.

Key files:

- `phase3-report.json`: completed orchestrator report
- `formal-fps.json`: formal measurements, comparisons, confidence intervals,
  and promotion decision
- `top3-nsight.json`: Top-3 diagnostics and hashes of retained remote raw data
- `candidate-matrix.json` and `screening-ranking.json`: sealed search output
- `winner-qualification-profile.json`: disabled winner profile for Phase 4
- `correctness-db.json`, `baseline-quality.json`, and
  `formal-correctness.json`: qualification evidence
- `dry-run-plan.json`, `resource-query.json`, and the three test logs: preflight
  and test evidence
- `reproducibility.json` and `db-validate.json`: postflight checks

The final isolated source passed 274/274 main tests, 60/60 head-extension CUDA
tests, and 34/34 Raster CUDA tests. The original remote working tree was not
modified.
