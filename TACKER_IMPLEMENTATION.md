# 4DGaussians × Tacker 实现说明

Tacker 集成仅用于推理，默认执行模式仍为 `serial`。`two_stream` 和
`tacker` 都必须显式选择，不会改变训练路径。Tacker 使用 fail-closed
策略：任一 workload、设备、ABI、资源、哈希、数值或画质契约不匹配，
都不得执行未准入的物理融合路径。

## 工作负载与物理路径

首个部署契约限定为：

- `flame_steak`，dynerf loader，iteration 14000；
- 1352 × 1014，111,525 Gaussians，test 视角 0–49；
- 单张 NVIDIA RTX A6000，compute capability 8.6（`sm_86`）；
- Rasterizer 基准 commit
  `e49506654e8e11ed8a62d22bcb693e943fdecacf`。

对应的 `workload_key` 是
`flame_steak:14000:111525:1352x1014:sm_86`。profile 不是可跨 workload 复用的
“通用优化开关”；每个新 workload 都需要独立的 correctness 证据和 whole-run
FPS 选择。

逻辑任务图仍为：

```text
Raster LC:      setup -> preprocess -> scan -> duplicate -> sort -> ranges -> render
Deformation BE: prefix -> selected heads -> selected suffix -> activation -> state
```

双槽位及 `ready`、`prefix_ready`、`mixed_done`、`raster_done` 事件保护依赖、
分配器生命期与槽位复用。`D(0)` 是唯一的完整形变 prefill；之后已选
head 不由 PyTorch 重复计算，并在最后一帧 drain。C1/C2 使用 first-linear
partition，C3 使用 packed first-linear，C4 覆盖 whole-head；具体线程划分、
worker group、`persistent_blocks` 和 barrier 合同全部由选中 profile 的 ABI
描述，不再在运行时猜测。

## 组件

- `gaussian_renderer/__init__.py`：串行 API 及 stream-aware 双槽位 renderer。
- `gaussian_renderer/tacker_pipeline.py`：profile v1/v2 契约、物理 dispatch、工作负载
  门控、字节快照加载与可见 fallback 原因。
- `tacker_ext/`：solo/GPTB FP16 head 扩展及 head ABI。
- `submodules/depth-diff-gaussian-rasterization/`：Raster 当前 stream 路径、C1–C4
  mixed kernels、资源/占用率查询及 Raster ABI。
- `profile_render.py`：whole-run 试验、p50/p95/max、峰值显存和加载二进制
  provenance。
- `profile_tacker_leaves.py`：数值、Raster/leaf 时间与资源诊断；leaf 结果
  不作为性能准入门禁。
- `scripts/benchmark_tacker_fps.py`：交错的完整序列 FPS 测量和 paired
  bootstrap。
- `scripts/benchmark_tacker_admission.py`：重算 correctness、selection 和 promotion
  契约，条件成立时生成独立 enabled profile。
- `scripts/validate_tacker_modes.py`：逐视角质量、输出顺序及 1/2/N 次数
  契约。
- `scripts/run_tacker_phase4.py` 与 `scripts/run_tacker_qualification.sh`：只消费
  Phase 3.1 sealed finalists 的 Phase 4 fail-closed 编排。
- `../Tacker/src/runtime/`：独立 `libtacker_runtime` 控制平面，与 Python
  生产路径同步 QoS-free、多 mixed registration 和 whole-run FPS 语义。

PyTorch 路径直接使用由 PyTorch 管理的 stream/event 并调用 CUDA ABI，不将
独立 C++ runtime 链接进 Torch 扩展，以避免额外的 libstdc++/Torch C++ ABI
边界。

## schema v2 与 QoS-free 选择

schema v2 将一个 workload 的候选集、correctness、资源诊断、whole-run FPS
trials、排名、promotion 证据和唯一 `selected_variant_id` 密封到 profile。
`manifest_sha256`、`profile_sha256`、候选 ABI SHA 和输入 provenance 使运行
证据可与精确字节绑定。仓库内 disabled profile 只用于 qualification，绝不得
原地改为 enabled。

决策分为两层：

1. correctness qualification 验证真实物理执行、无 fallback、数值、50-view
   画质、workload、ABI、capability 和 resource 契约。
2. performance selection 仅在 correctness-valid 候选中，按多次完整序列的
   `median(throughput_fps)` 排名。`serial`、`two_stream` 和 current Tacker 都是
   正式候选。

画质门禁仍为 PSNR drop `<= 0.05 dB`、SSIM drop `<= 1e-4`、LPIPS
increase `<= 1e-4`。Raster slowdown、leaf-sum savings 和单次 Tacker/two-stream
p50 比值仅是 diagnostics，即使 Raster slowdown 超过 5% 也不会单独否决候选。
正式选择使用 10 次、50-frame、warmup 10、ABBA/seed 0 的 whole-run 试验，
并以共享 round 为单位做 10,000 次 paired bootstrap。替换 correctness-valid
incumbent 要求 FPS ratio `>= 1.01` 且 95% CI 下界 `> 1.0`；候选还不得
慢于有效 two-stream 下限。

## 构建与 CPU 契约

远程 A6000 资格验证环境为 Python 3.10、PyTorch 2.4.1+cu124 和 CUDA
toolkit 12.4，必须生成 `sm_86` 代码。扩展构建顺序为：

```bash
cd /home/qyfeng/4DGaussians/tacker_ext
python setup.py build_ext --inplace

cd /home/qyfeng/4DGaussians/submodules/depth-diff-gaussian-rasterization
TACKER_4DGS_HEAD_INCLUDE=/home/qyfeng/4DGaussians/tacker_ext/include \
  python setup.py build_ext --inplace
```

独立 runtime 的构建与 CPU 测试：

```bash
cmake -S /home/qyfeng/Tacker-4DGS-runtime/src \
  -B /home/qyfeng/Tacker-4DGS-runtime/build-runtime \
  -DTACKER_BUILD_LEGACY=OFF -DTACKER_BUILD_TESTS=ON
cmake --build /home/qyfeng/Tacker-4DGS-runtime/build-runtime --parallel
ctest --test-dir /home/qyfeng/Tacker-4DGS-runtime/build-runtime --output-on-failure

cd /home/qyfeng/4DGaussians
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=tacker_ext PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tacker_ext/tests -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s submodules/depth-diff-gaussian-rasterization/tests -p 'test_*.py' -v
```

## 常规执行与 fallback

只有独立的 enabled/valid admitted profile 可用于常规 Tacker 渲染：

```bash
python render.py ... \
  --execution-mode tacker \
  --workload-name flame_steak \
  --tacker-profile /absolute/releases/<release-id>/winner.admitted.json
```

`--execution-mode two_stream` 选择双 stream 基线；省略时为 `serial`。disabled
qualification profile 只能由验证/性能分析入口以显式 qualification mode
使用。profile 缺失、过期、字节哈希不匹配或工作负载不匹配时，报告必须保留
`requested_execution_mode`、`actual_execution_mode` 和非空 fallback reason。

profile 使用单个 regular-file descriptor 读取，加载前后核对 `fstat/lstat` 及字节
SHA，并拒绝 symlink。正式 Phase 4 在 heavy import/配置解析前捕获源码、
config 和候选二进制快照，import 后将快照绑定到实际加载的 Raster 二进制与
`libtacker_runtime.so`，在发布报告前再次验哈希。任一字节变化都使整次
run fail closed，防止 TOCTOU 证据混用。

## 验证状态与证据边界

2026-08-31 的 v1/`persistent_blocks=7000` 运行只是 **QoS-era 历史记录**。
当时记录的 Raster 5% QoS、leaf-sum 和端到端 p50 门禁已从当前准入
契约移除；这些历史数字不得用来启用 v2 profile。2026-09-10 的
Phase 0 whole-run 基线另行封存于
`tacker_profiles/baselines/a6000_flame_steak_phase0_20260910/`。

Phase 3.1 已在 2026-09-13–14 完成 A6000 封存选择。实验 argmax 与
deployment selector 均选中
`c3_packed_first_linear_pos_scales_rotations_opacity_shs_wg1_pb5440`，median FPS
为 `100.9417269038`；current Tacker 与 two-stream 分别为 `87.2573058329` 和
`86.3680297778` FPS。winner/current ratio 为 `1.1568283703`，paired-bootstrap
95% CI 为 `[1.1544964934, 1.1592246266]`。selection SHA-256 为
`1a91a02c2605ae442817bb060b7956ead5d3b940146d274b94a85e00849ec126`。

这只是 Phase 3.1 selector 证据，不是发布状态。封存 winner profile 仍为
`deployment.enabled=false, valid=false`。紧凑证据位于
`tacker_profiles/baselines/a6000_phase31_20260913/`。

## Phase 4 准入、灰度、发布与回滚

Phase 4 是 sealed Phase 3.1 的消费者，不生成、扩展或重排候选矩阵。编排器
依次执行：预检；扩展/runtime 构建与 CPU/CUDA 回归；Phase 3.1 seal
复核；每个 sealed finalist 的资源和数值独立复验；50-view 画质；
10 × 50 ABBA whole-run 复测与 selection/admission 重算；1/2/50 与长序列
prefill/steady-state/drain；enabled profile 常规复跑；missing/stale/hash-mismatch
fallback smoke；两个声明为不同 Raster/deformation mix、且 whole-run overlap
proxy 可测地区分的泛化 workload；最后是
canary/release/rollback 检查。每个阶段只在前置证据哈希不变时可恢复。

最终 A6000 实机 run 为
`/data/qyfeng/tacker_phase4_validation/20260914-codex-phase4/run-complete-v5`；
13/13 个 hash-bound stage 全部 `succeeded`。紧凑证据镜像位于
`tacker_profiles/baselines/a6000_phase4_20260914/run-complete-v5/`，完整参数与
本次实际使用的单张 A6000 命令见 `tacker_profiles/README.md`。

正式 12-entry benchmark 在 warmup 10、50 frames、10 trials、ABBA/seed 0
下完成 120/120 次执行。实验与部署 winner 均为
`c3_packed_first_linear_pos_scales_rotations_opacity_shs_wg1_pb5440`，
median FPS 为 `100.5082562657`；current Tacker 与 two-stream 为
`87.3984074601` / `86.3416638762` FPS。winner/current ratio-of-medians 为
`1.1500010033`，paired-bootstrap 95% CI 为
`[1.1446137379, 1.1542292551]`；winner/two-stream ratio 为
`1.1640759716`，95% CI 为 `[1.1596655823, 1.1687732974]`。最终决策是
`promote_challenger / promotion_gates_passed`。

资源、正确性与序列证据如下：

- 10/10 份 Tacker 资源/数值记录有效，12/12 个 formal entry 的 50-view
  correctness 有效；39 条 ptxas 记录均为 0 spill。winner 使用
  70 registers/thread、7168 B static shared memory、0 local bytes/thread、
  2 active blocks/SM，occupancy 为 `0.5`。其 Raster slowdown `31.154%` 仅是
  diagnostic，不是 QoS 否决项。
- winner 的 50-view 平均差异为 PSNR drop `0.000209961 dB`、SSIM drop
  `1.3161e-6`、LPIPS increase `6.7294e-7`，均低于固定门禁。
- 1/2/50/500 帧的 p50 为 `202.915833` / `108.794884` / `9.813019` /
  `9.840576 ms`，p95 为 `202.915833` / `201.154864` / `10.097307` /
  `9.891193 ms`。对应 mixed launch 为 0/1/49/499，输出与每个 selected
  head 评估次数均精确等于 N；500 帧记录的峰值 allocated/reserved
  是 `1,282,750,976` / `2,124,414,976` B。
- admission 生成的 enabled/valid profile 在常规非 qualification 路径复跑
  10 × 50，`actual_execution_mode=tacker`、无 fallback，整个 Phase 4 观测到的
  最大 allocated/reserved 为 `1,284,494,848` / `2,124,414,976` B。
- missing、stale-workload 和 hash-mismatch 3/3 种负向情况都可见地回退到
  `two_stream`，并保留非空 fallback reason。

两个额外 workload 只做 baseline-only、evaluation-only 泛化检查，没有生成
新候选、复用主 workload profile 或声称全局 winner。iteration 3000 native
（1352×1014，92,999 Gaussians）中 serial/split-serial/two-stream 为
`101.0533156930` / `101.2068541033` / `105.6180133902` FPS；iteration 14000
scale-4（338×254，111,525 Gaussians）为 `113.7573262077` /
`113.5076567124` / `121.3746216960` FPS。两者都只在各自 workload key 内
选中 `two_stream`；two-stream/split-serial overlap proxy 分别为
`1.0435855785` 与 `1.0693077913`，证明两类负载比例可测地不同。

本次 identity 严格绑定环境和字节 provenance：物理 GPU 1 为
RTX A6000（`sm_86`、84 SM），driver `570.124.06`、CUDA 12.4、PyTorch
`2.4.1+cu124`、Python 3.10。Phase 4 identity 为
`66cab463bd2682f3f330095f74afc37df4eeb257ba98083825172c40c150f39f`；
主仓/Raster/simple-knn commit 为
`a6c475ee737341c28f88a8fda5fa04479e211592` /
`79975a092b027cfb374caa2651959942d9aae4f0` /
`b3554e0fee8a51b4f9201644577ab23c5bb10507`，Tacker runtime commit 为
`a6e84eef97b315424c9587cd534792583b609101`。实际 runtime/head/Raster/simple-knn
二进制 SHA-256 为 `78f4d2b1f85eb91dccaed07fc71597a27babcafda93413f96dfc47bd3f8e0671` /
`f99bc3134ef9997d3a88f4772476a6641b219db275a522d67327048d3f2db3a5` /
`cd76862fee530e24a3a96be571d6decf479205caa3c128ed8e4ca84591af263c` /
`c04852b5c0d0db5cd2c76bdba106c81fca79f3502896dec2ed33e87ea48fdae8`，
且前后字节稳定性检查通过。顶层 report/state 文件 SHA-256 为
`51fd1d70dd276e6508250cc9761dff99819ef116ff72c5793dc1f348f2373bd7` /
`6fe406bcf39377642963cfdab159d20ae45a6208b85c98230fa72e783211d7fe`；formal
report 文件 SHA-256 为
`8cc23567062e7eb0d43d420ba6d8f9128b80844e1c96d2f3b44769f065982777`。

admission 生成 profile 的文件/canonical SHA-256 为
`43ae401fc607b1ca7611f04d1e12a30a788fad5d647d3e37cd1d645789667d8e` /
`22e27f5ee9cf5f98b53cf4e790bb165163c1a61f9970775da4faabea0d9f22d6`，manifest
SHA-256 为 `74ebfa63c3be7d8f3283b098fb9fd6155cf8e9d39ebdf0197ef473b587a4ea52`。
显式 profile canary 、current-Tacker/two-stream 两项 rollback drill 和不可变
release-selection artifact 都已通过；它们的文件 SHA-256 分别为
`f24e2d450cfeb77d3874c6b62103a918049f85bc1450b3e6c6a2a79c436f7e7c`、
`bd3b135bec857f1dfdcc7d76941063b16c5c564fb4abab23e596e99adabcdf99` 和
`8886c50839549eaaa14bc6bd318500ed99218fd02296954fa149d21df5c7ee41`，release canonical
SHA-256 为 `e7dd8d081d445b51dc92cccc74704690e9187863163ae2dd26b6fb43d5ba88d8`。

发布边界仍然是 fail closed：本轮 release 状态只是
`ready_for_explicit_promotion`，且
`automatic_default_replacement_performed=false`。canary 明确限定为
`explicit_profile_only`，`default_profile_replaced=false`；因此资格链完成不表示
默认生产 deployment selection/profile 已切换。只有获得明确运维授权后，
才可将已完整写入并 `fsync` 的新 profile/selection 通过 `os.replace`
原子替换生产 selection；不得原地改写已加载 profile 或使用 symlink。
授权后的回滚仍只需原子指回封存 current Tacker profile 或 `two_stream`，
重启/重载后复验实际模式和 fallback。
