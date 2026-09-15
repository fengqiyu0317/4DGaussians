# RTX A6000 Phase 4 准入、回归与发布证据

本目录是 2026-09-15 在远端 `4A6000` 上完成的 Phase 4 紧凑证据镜像。最终运行根为：

```text
/data/qyfeng/tacker_phase4_validation/20260914-codex-phase4/run-complete-v5
```

最终 `phase4-state.json` 为 `status=succeeded`，`phase4-report.json` 为 `passed=true`，13 个 journal stage 全部成功（`13/13`）。运行开始于 `2026-09-15T02:46:58.939353Z`，完成于 `2026-09-15T03:28:39.870675Z`。

## 发布边界

本轮的准确结论是：

- 新 challenger profile 已通过准入、显式-profile canary、回归和回滚演练。
- release 状态是 `ready_for_explicit_promotion`。
- `automatic_default_replacement_performed=false`，canary 也记录 `default_profile_replaced=false` 和 `scope=explicit_profile_only`。
- 因此，这些证据不表示默认生产 profile 已切换；若要切换，仍需一次显式 promotion 操作。

## 身份与封印

| 对象 | SHA-256 / commit |
| --- | --- |
| Phase 4 identity | `66cab463bd2682f3f330095f74afc37df4eeb257ba98083825172c40c150f39f` |
| `phase4-report.json` 文件 | `51fd1d70dd276e6508250cc9761dff99819ef116ff72c5793dc1f348f2373bd7` |
| `phase4-state.json` 文件 | `6fe406bcf39377642963cfdab159d20ae45a6208b85c98230fa72e783211d7fe` |
| Phase 3.1 identity | `0f16b7250611891db52cc59dada0f8bd77c40d735c26f98aed7c81047c14d8a2` |
| Phase 3.1 report 文件 | `b24699941748fff134830bf59e0a0001bc64ac419ed119d10481bb894b956ea9` |
| Phase 3.1 matrix | `aa45c9dce881a5a71c9d86130fcd6b12070ee2773b83b44a538f2147bbde8577` |
| Phase 3.1 formal set | `7c592726cb2edfeb9ea7c030b569d213b179c0917d31a9d1b4a56007abc54102` |
| Phase 3.1 selection | `1a91a02c2605ae442817bb060b7956ead5d3b940146d274b94a85e00849ec126` |

Phase 4 只重放 Phase 3.1 封印的 9 个 finalist 与 `serial` / `two_stream` / `current_tacker`，`candidate_generation_allowed=false`。Phase 3.1 重放核验了 49 个成功 stage、7354 个唯一 stage artifact 和 661 个 screening terminal record。

## 源码与二进制身份

远程 v8 源码归档为：

```text
/data/qyfeng/tacker_phase4_validation/20260914-codex-phase4/tacker-phase4-final-20260914-v8.tar.gz
SHA-256: 9d7520ec75449681a9d9a3f9522d4111674987ce7e5877ae53de4c462b273a33
size: 7,283,749 bytes
```

| 源码对象 | commit / SHA-256 |
| --- | --- |
| v8 项目 Git（clean） | `a6c475ee737341c28f88a8fda5fa04479e211592` |
| Rasterizer 子仓库 Git（clean） | `79975a092b027cfb374caa2651959942d9aae4f0` |
| Rasterizer ABI manifest 兼容 commit | `e49506654e8e11ed8a62d22bcb693e943fdecacf` |
| simple-knn 子仓库 Git（clean） | `b3554e0fee8a51b4f9201644577ab23c5bb10507` |
| Tacker runtime Git | `a6e84eef97b315424c9587cd534792583b609101` |
| `scripts/run_tacker_phase4.py` | `93d3b829120043ce6dec23f1b6329dde722e08234bd8eccea273d275cbde4a26` |
| `scripts/run_tacker_qualification.sh` | `f3a1804f3ddda5032cb1d2e30a0431a5a5b1664c1c80ff236ae3a29fd28b12ba` |
| `scripts/benchmark_tacker_admission.py` | `90b1ea538aedf853bf8b01de5aa502a5d850eb3260225a78ca2015e00e2db77b` |
| `scripts/benchmark_tacker_fps.py` | `c0eb33b50301334f072cdb4dc6004d9c000300d7fe0722b1268587812b3e667e` |
| `scripts/run_profile_render_sealed.py` | `bbafbedbecab9cc7abfb83911f286923cf54e873b5cf01572692d9b40f361e31` |
| `profile_render.py` | `d7006d8b5fcce5b92abfb1257d01ad616c80cb54f7892f7c5e7310fa4f33d6e6` |
| `profile_tacker_leaves.py` | `1fbfc48cef7a5d59c1383e5bca636796579192a7ff7f3dbe382115fab10159ff` |

Runtime Git 在身份采集时的 `dirty=true` 仅来自既有、可过滤的 `build-runtime-a6000/` 未跟踪构建输出；runtime 源文件仍全部逐文件封印。完整的 project/runtime 源文件 SHA-256 表位于 `phase4-report.json` 的 `identity.payload.project_sources` 和 `identity.payload.runtime.sources`。

| 构建产物 | SHA-256 | 大小 |
| --- | --- | ---: |
| `libtacker_runtime.so.0.4.0` | `78f4d2b1f85eb91dccaed07fc71597a27babcafda93413f96dfc47bd3f8e0671` | 164,640 B |
| Rasterizer `_C.cpython-310-x86_64-linux-gnu.so` | `cd76862fee530e24a3a96be571d6decf479205caa3c128ed8e4ca84591af263c` | 2,844,464 B |
| Head `_C.cpython-310-x86_64-linux-gnu.so` | `f99bc3134ef9997d3a88f4772476a6641b219db275a522d67327048d3f2db3a5` | 468,608 B |
| simple-knn `_C.cpython-310-x86_64-linux-gnu.so` | `c04852b5c0d0db5cd2c76bdba106c81fca79f3502896dec2ed33e87ea48fdae8` | 1,804,824 B |

构建针对 `sm_86`，`ptxas` 日志存在，且源码与二进制在构建后和整轮运行后均完成了字节级不变性核验。

## 13 个成功 stage

`phase4-state.json` 中的 13 个 stage 均为 `attempt=1` 且 `status=succeeded`：

1. `preflight`
2. `build-and-cuda`
3. `verify-phase31-seal`
4. `finalist-resources-numerics`
5. `quality-50-view`
6. `formal-10x50-abba`
7. `selection-admission`
8. `sequence-1-2-50-long`
9. `enabled-profile-rerun`
10. `fallback-smoke`
11. `generalization-workload-1`
12. `generalization-workload-2`
13. `canary-release-rollback`

环境为 NVIDIA RTX A6000（SM 8.6，84 SM，50,908,823,552 B 显存）、Python 3.10、PyTorch 2.4.1+cu124、CUDA 12.4，仅暴露一张物理 GPU 1。

## CPU / CUDA 回归

| 套件 | 通过数 |
| --- | ---: |
| Phase 4 Python CPU contracts | 241 |
| Head Python CPU contracts | 60 |
| Rasterizer Python CPU contracts | 44 |
| Tacker runtime CTest | 1 |
| Head CUDA tests | 16 |
| Rasterizer ABI 1–4 CUDA tests | 16 |

即 Python CPU contract `345/345`、CUDA test `32/32`，另有 runtime CTest `1/1`。

## 资源、数值与画质

- finalist 资源/数值核验 `10/10` 有效，无 invalid record。winner 的实测资源为 70 registers/thread、7,168 B static shared memory、2 active blocks/SM、0.5 occupancy。
- 50-view 画质核验 `12/12` candidate 有效，无 invalid candidate。
- winner 相对 serial 的画质差异为 PSNR drop `0.00020996093750369482 dB`、SSIM drop `0.0000013160705566450659`、LPIPS increase `6.729364395163806e-7`，数值合同也通过。

## Formal 10×50 ABBA

Formal benchmark 为 12 个 candidate × 10 轮 × 50 帧，`completed_execution_count=120` 与 `expected_execution_count=120`，无 error。experimental winner 与 deployment winner 都是：

```text
c3_packed_first_linear_pos_scales_rotations_opacity_shs_wg1_pb5440
```

| 模式 | median FPS | 50-frame median total | max peak allocated | max peak reserved |
| --- | ---: | ---: | ---: | ---: |
| winner | 100.508256 | 497.471737 ms | 1,280,936,960 B (1221.597 MiB) | 2,065,694,720 B (1970 MiB) |
| `current_tacker` | 87.398407 | 572.092808 ms | 844,475,392 B (805.354 MiB) | 1,340,080,128 B (1278 MiB) |
| `two_stream` | 86.341664 | 579.094713 ms | 401,748,480 B (383.137 MiB) | 1,015,021,568 B (968 MiB) |
| `serial` | 82.588306 | 605.412622 ms | 310,938,112 B | 530,579,456 B |

winner 相对 `current_tacker` 的 median FPS ratio 为 `1.1500010032962709`，成对轮次 median ratio 为 `1.15044519305058`，10,000 次 paired bootstrap 95% CI 为 `[1.1446137379277883, 1.1542292551006885]`。相对 `two_stream` 的 median FPS ratio 为 `1.1640759715928337`，成对轮次 median ratio 为 `1.1657503700218983`，95% CI 为 `[1.1596655822923547, 1.1687732973780072]`。promotion decision 是 `promote_challenger`，表示通过准入门槛，不等于已自动切换默认 profile。

准入生成的 challenger profile 文件 SHA-256 为 `43ae401fc607b1ca7611f04d1e12a30a788fad5d647d3e37cd1d645789667d8e`；其 manifest SHA-256 为 `74ebfa63c3be7d8f3283b098fb9fd6155cf8e9d39ebdf0197ef473b587a4ea52`，selection SHA-256 为 `22e27f5ee9cf5f98b53cf4e790bb165163c1a61f9970775da4faabea0d9f22d6`。

## 1 / 2 / 50 / 500 帧序列回归

下表的计数顺序为 `full_deformation / prefix / mixed_launches / suffix / solo_raster / outputs / selected_head_evaluations_per_head`。

| 帧数 | p50 | p95 | max | peak allocated | peak reserved | 执行计数 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 202.915833 ms | 202.915833 ms | 202.915833 ms | 779.890 MiB | 1970 MiB | `1 / 0 / 0 / 0 / 1 / 1 / 1` |
| 2 | 108.794884 ms | 201.154864 ms | 211.417084 ms | 1221.597 MiB | 1970 MiB | `1 / 1 / 1 / 1 / 1 / 2 / 2` |
| 50 | 9.813019 ms | 10.097307 ms | 208.758789 ms | 1221.597 MiB | 1970 MiB | `1 / 49 / 49 / 49 / 1 / 50 / 50` |
| 500 | 9.840576 ms | 9.891193 ms | 199.469055 ms | 1223.327 MiB | 2026 MiB | `1 / 499 / 499 / 499 / 1 / 500 / 500` |

1、2、50 帧输出顺序都与 legacy 结果对比；500 帧用于 steady-state 与显存回归。整个 sequence stage 为 `passed=true`。

## Enabled-profile 正常部署重跑

准入文件在非 qualification 的正常部署路径上重跑 10×50：

- `execution_mode=tacker`、`actual_execution_mode=tacker`，`qualification_mode_requested=false`、`qualification_mode_executed=false`，无 fallback。
- median throughput `72.79804249116366 FPS`，median total `686.831955332309 ms`，median frame p50/p95/max 分别为 `9.78814697265625 / 9.834360885620118 / 208.30916595458984 ms`。
- median peak allocated/reserved 为 `1,282,388,736 / 2,124,414,976 B`；全轮 max peak allocated/reserved 为 `1,284,494,848 / 2,124,414,976 B`。
- 每轮计数为 `50 input / 1 full deformation / 49 prefix / 49 mixed / 49 suffix / 1 solo raster / 50 outputs / 50 per-head evaluations`。
- post-run 对 70 个源/profile/输入文件和 3 个已加载二进制的字节稳定性核验全部通过。

## Fallback smoke

3 个请求用例全部执行且全部带明确原因回退到 `two_stream`（`3/3`）：

| 用例 | 实际模式 | 原因 |
| --- | --- | --- |
| `missing` | `two_stream` | profile 在 sealed pre-import snapshot 中不存在 |
| `hash_mismatch` | `two_stream` | profile manifest SHA-256 mismatch |
| `stale_workload` | `two_stream` | workload name 与 admitted profile 不匹配 |

## 两个 baseline-only generalization 工作负载

这两个工作负载只比较安全 baseline，没有评估 Phase 3.1 的主工作负载 profile，因为 profile 绑定 workload key、分辨率与 Gaussian 数。全程 `candidate_generation_allowed=false`、`cross_workload_profile_reuse_allowed=false`、`global_variant_claimed=false`。balance proxy 是整运行 overlap sensitivity proxy，不是 Raster/变形 kernel 时间比。

| 工作负载 | `serial` | `split_serial` | `two_stream` | winner | split/serial | two/split |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| iter 3000，1352×1014，92,999 Gaussian（raster-heavy） | 101.053316 FPS | 101.206854 FPS | 105.618013 FPS | `two_stream` | 1.001519 | 1.043586 |
| iter 14000，338×254，111,525 Gaussian（deformation-heavy） | 113.757326 FPS | 113.507657 FPS | 121.374622 FPS | `two_stream` | 0.997805 | 1.069308 |

两组 proxy 可测地不同，且两个 generalization stage 均通过；这只支持「不同工作负载需单独选择」，不支持一个全局 Tacker variant 结论。

## Canary / release / rollback 三件套

| 文件 | 文件 SHA-256 | 内部 payload seal | 结果 |
| --- | --- | --- | --- |
| `canary.json` | `f24e2d450cfeb77d3874c6b62103a918049f85bc1450b3e6c6a2a79c436f7e7c` | `c8a8d272a76550a833db707df98566abdf6ec27f4ccf76e35b577bc243b8941f` | `passed=true`，仅显式 profile |
| `release-selection.json` | `8886c50839549eaaa14bc6bd318500ed99218fd02296954fa149d21df5c7ee41` | `e7dd8d081d445b51dc92cccc74704690e9187863163ae2dd26b6fb43d5ba88d8` | `ready_for_explicit_promotion` |
| `rollback.json` | `bd3b135bec857f1dfdcc7d76941063b16c5c564fb4abab23e596e99adabcdf99` | `9d83ca5216a8de2738c8f73a30da521e34653353c0e9ae42a6194fdaa8dcd529` | `passed=true` |

Rollback 同时对 `current_tacker` 和 `two_stream` 进行了 3×50 帧实跑，两个 drill 都 `eligible=true`、`executed=true`、`passed=true`。回滚策略是只替换 deployment selection/profile，不修改已封印源码或二进制。

## 外部干扰与恢复记录

Formal 成功后、`selection-admission` 执行前（约 `2026-09-15 03:20 UTC`），一次辅助 admission probe 在封印源码树中生成了单个 `benchmark_tacker_fps.cpython-310.pyc`。随后的全局身份检查按设计中断进程，错误原文为：

```text
Phase-4 qualification failed: Phase-4 source, runtime, workload, environment, or sealed input changed during the run
```

这是外部 probe 造成的源树污染，不是 journal stage 失败，也不是 Phase 4 代码失败。该单一 `.pyc` 已移入：

```text
/data/qyfeng/tacker_phase4_validation/20260914-codex-phase4/run-complete-v5/diagnostics/external-interference/benchmark_tacker_fps.cpython-310.pyc
SHA-256: 9bb98d49fcb554e84596bfb87431a3b46c778f8263c586a989053d4c6f7082fc
```

清理源树后，以同一 Phase 4 identity 执行 `--resume`。Journal 在复用任何既有 stage 之前都逐个重新校验其 artifact 哈希，然后完成剩余 stage。因为该中断不属于 stage，最终 state 中 13 个 stage 仍全部是 `attempt=1`。

## 紧凑镜像边界

本地 `run-complete-v5/` 约 15 MiB，包含 130 个文件（79 JSON、50 log、1 个 runtime `.so`）。它保留了：

- 最终 report/state、13 个 `stage-result.json`、封印和命令日志；
- 资源/画质/formal/序列/enabled-profile/fallback/generalization 的汇总证据；
- admitted challenger profile、formal checkpoint、canary/release/rollback 三件套；
- runtime library 和 `compile_commands.json`，以及 winner、C4 边界样本、`current_tacker` 的详细资源/数值报告。

`artifact-manifest.json` 对上述 130 个远端镜像文件和本地 `README.md` 共 131 个文件逐项记录相对路径、大小、SHA-256 与远端来源路径；manifest 自身不递归列入。原始紧凑归档为：

```text
/data/qyfeng/tacker_phase4_validation/20260914-codex-phase4/tacker-phase4-run-complete-v5-compact.tar.gz
SHA-256: e4001628392c787fce7bbbf6265df833726491b56e815b772c9d10a859138f16
size: 1,225,058 bytes
```

为避免把几百 MB 的短期中间产物纳入仓库，镜像没有包含完整的 `head-build/`、`raster-build/`、`simple-knn-build/` 与 runtime 中间对象，也没有复制 formal 的 120 个单次 metadata 和所有 finalist 的大型原始 leaf/Raster trace。完整证据仍位于上述远程 run root。因此本目录用于审阅与结论复核，不是脱离远程模型、数据集、Phase 3.1 封印和 v8 源树的自包含重跑包。

## 远程重现

在 `4A6000` 上使用一个新、空的输出目录：

```bash
export TACKER_ROOT=/home/qyfeng/tacker_phase4_runtime/20260914-codex-phase4
export PHASE31_RUN_ROOT=/data/qyfeng/tacker_phase31_validation/20260913-codex-phase31/run-complete-v3
export OUTPUT_DIR=/data/qyfeng/tacker_phase4_validation/20260914-codex-phase4/run-reproduction-v8
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-12.4
export TORCH_HOME=/data/qyfeng/cache/torch
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06

bash /home/qyfeng/tacker_phase4_code/20260914-codex-phase4-v8/scripts/run_tacker_qualification.sh
```

若同一输出目录中已有与当前 identity 完全一致的 journal，可在最后加 `--resume`；不同 identity 不得混用或强行复用证据。
