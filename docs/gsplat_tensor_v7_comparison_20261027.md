# gsplat 接入旧版 v11/v13 优化 MLP：Graph 对照

2026-10-27 完成。旧版 **device-v7 Tensor Core tail MLP** 已接入 gsplat 的独立与 mixed 两条路径。
同 checkpoint、同 Graph 前缀下，mixed 为 **54.25 FPS**，独立调用为 **51.90 FPS**，FPS 提升 **4.52%**。
两条路径的 50 帧 RGB、累计深度全部逐位一致。原始记录见[汇总](../benchmarks/results/gsplat_v7_20261027/summary.json)。

## 实现与对照边界

- 两组直接复用 `deform_mlp_tensor_tail_gptb_device_v7`，即原 v11/v13 的 device 函数；保留 padded shared scratch、融合 bias/ReLU/compaction、FP16 Tensor Core tail 及 FP32 累加/输出。
- 从同一 checkpoint 提取参数，使用原 `pack_deform_tail_weights_v3` 将 tail 打包为 FP16 `[112,128]`，只在初始化时打包。两组缓存参数逐位相同。
- 独立组：原生 gsplat RGB+D pixels kernel，随后单独调用同一 device-v7 MLP（128-thread CTA）。mixed 组：同一 384-thread kernel 内，256 线程做 RGB+D pixels，128 线程做下一帧完整 MLP。
- mixed 使用 **3,486 个物理 CTA**，通过 grid-stride 完成全部 **5,440 Raster tile＋6,971 MLP tile / 111,525 rows**。没有独立 MLP 尾部，也没有省略 Raster 工作。
- 本次迁移的是旧版优化 MLP。Graph 前缀沿用上一轮 gsplat 的 native HexPlane 双 slot 配置，**没有启用旧 v13 的 Raster prepass、HexPlane fused reduction 或 lowreg sampling**；这两组的配置完全一致。
- 未修改 checkpoint。正式运行前后源码和 checkpoint 指纹一致，checkpoint 四个文件与此前 A6000/Orin 实验的 SHA256 均一致。

新增/接入入口：

- [device-v7 mixed 和独立 kernel](../gsplat_mixed_ext/mixed_tensor_v7.cu)
- [binding](../gsplat_mixed_ext/bindings_tensor_v7.cpp) · [JIT loader](../gsplat_mixed_ext/__init__.py)
- [共享 Graph 管线](../gaussian_renderer/gsplat_mixed.py)
- [CUDA 测试及旧版库 oracle](../tests/test_gsplat_tensor_v7_cuda.py)
- [benchmark](../scripts/benchmark_gsplat_v7.py) · [启动脚本](../scripts/run_jetson_gsplat_v7.sh)

## 环境、Graph 与协议

Jetson AGX Orin、16 SM、GPU 固定 **1.3005 GHz**，PyTorch 2.8.0、CUDA 12.6、sm_87、gsplat 1.5.3。
`flame_steak` iteration 14000，111,525 Gaussians、1352×1014，test views 0–49，dense/classic RGB+D。

实际 parent profile 为 `full_mlp_overlap_graph.qualification.json`，selected variant 为 **`full_mlp_overlap_graph_pb0`**，SHA256：

```text
a13d87f6c4a22a24b20f83cb3c66a009e759b739b498c9ff55f8c35c9b82d23a
```

它验证复用的 workload 和 Graph 前缀，不自动为新 gsplat leaf 提供生产 qualification。新 leaf 的输出、资源与实际 CUDA 节点在本实验单独验证；没有混入 eager `full_mlp_overlap` 数据。

- **2026-10-21 leaf 初筛：** 三组真实输入 Raster→MLP 帧配对 `0→1 / 24→25 / 48→49`。各 arm 6 轮、正逆序交替，每轮预热 5 次、测 20 次 CUDA Graph replay；取各组轮间中位数，再平均三组。分配、前缀和 gsplat prepare 不计时。
- **2026-10-27 正式续跑：** 复核原始源码/checkpoint/旧 MLP oracle 无变化，重新通过 9 项 CUDA 测试。各组先初始化并预热，10 轮正逆序交替，每轮完整 50 帧。使用预先选中的 3,486 CTA 配置，没有按正式 FPS 重新筛选。
- **端到端 wall time：** 包含 Python 提交、context setup、HexPlane Graph、MLP、残差/激活、gsplat prepare/pixels、首帧 prefill、末帧 drain 和最终同步；排除读取、编译、预热、画质计算及 I/O。
- **渲染控制：** 使用两组共享的、预先计算好的 device-v7 Gaussian 状态，计 gsplat prepare＋pixels＋输出布局转换。`mixed_envelope` 关闭 MLP，只测 mixed 资源配置中的 Raster，属于诊断控制。

正式两组均有 2 个 slot，50 帧实际调用 **49 次 Graph replay**，逐 slot 实测 `[24,25]`。独立组执行 49 对独立 leaf，mixed 组执行 49 次物理 mixed；两组均为 1 次首帧完整 deformation、50 次 Raster prepare、1 次末帧 solo raster、50 个输出，无 fallback。

## 性能

### 输入已就绪的 leaf

这里的 mixed leaf **同时包括当前帧像素渲染和下一帧 MLP**，不能称为纯渲染时间。

| 项目 | CUDA 时间 |
| --- | ---: |
| 原生 gsplat pixels | 5.312 ms |
| 封存的旧版 v11 独立 MLP oracle | 6.598 ms |
| 迁入后的独立 device-v7 MLP | 6.598 ms |
| pixels＋device-v7 MLP，独立串行实测 | **11.908 ms** |
| mixed，默认 6,971 CTA | 11.142 ms |
| mixed，5,440 CTA | 11.047 ms |
| mixed，3,486 CTA | **11.047 ms** |
| mixed，2,720 CTA | 11.073 ms |

3,486 CTA 为三组输入平均的名义最低值，与 5,440 CTA 几乎持平，不能解释为普遍最优。
选中 mixed 相对匹配独立组，leaf 耗时降低 **7.23%**，加速 **1.0779×**。
其仪器化验证中 3,486 个配对 CTA 均观察到 subgroup 执行窗口重叠；仪器化不参与正式计时，窗口重叠比例不等于吞吐收益。

### 50 帧端到端

| 同 checkpoint、同 Graph、同 device-v7 MLP | 中位 wall ms/frame | 中位 CUDA span ms/frame | 中位 FPS |
| --- | ---: | ---: | ---: |
| gsplat＋独立调用 | 19.268 | 19.268 | **51.90** |
| gsplat mixed，3,486 CTA | **18.434** | **18.434** | **54.25** |

mixed 的 FPS **+4.52%**，wall time **−4.33%**。10 轮 FPS 的 CV：独立 0.0128%、mixed 0.0240%。
配对轮次 FPS 比的 bootstrap 95% 区间为 `[1.04507,1.04555]`；这只描述本次同进程轮次变化，不代表跨进程或跨场景置信区间。

### 完整渲染控制（不含 deformation）

| 同一组预缓存 Gaussian 状态，含 prepare＋RGB+D pixels | 中位 wall ms/frame |
| --- | ---: |
| 原生 gsplat 渲染 | **8.686** |
| mixed 资源配置，仅启用 Raster | 9.086 |

两组控制的 50 帧输出也逐位一致。9.086 ms **不是实际 mixed 内 Raster subgroup 的独占耗时**，也不能用它与 mixed 总时间相减计算 MLP 时间。它说明资源配置本身仍有成本，而完整 mixed 的重叠收益足以抵消该成本。

## 画质

指标均为同一 50 帧、相同 GT，PSNR/SSIM/LPIPS 对 RGB clamp 到 `[0,1]` 后计算；LPIPS 为 VGG 0.1。

| 实现 | 平均 PSNR ↑ | 平均 SSIM ↑ | 平均 LPIPS ↓ |
| --- | ---: | ---: | ---: |
| 原生 gsplat＋原 PyTorch MLP | 32.570320 | 0.95746630 | 0.14144589 |
| gsplat＋独立 device-v7 MLP | 32.570166 | 0.95746544 | 0.14144845 |
| gsplat mixed＋device-v7 MLP | **32.570166** | **0.95746544** | **0.14144845** |

- **融合增量：** mixed 相对同 MLP 独立组的 50 帧 RGB、累计深度全部逐位相同，最大绝对差均为 **0**。
- **优化 MLP 管线相对原生 gsplat/PyTorch 管线：** PSNR drop `0.0001540 dB`、SSIM drop `8.5592e-7`、LPIPS increase `2.5567e-6`，均通过预设门槛 `0.05 dB / 1e-4 / 1e-4`。
- 通过平均指标门槛**不代表相对原生管线逐像素一致**。未 clamp RGB 的逐帧 MAE 均值为 `7.6095e-6`，全局最大绝对差为 `0.093549`（view 45）；累计深度最大绝对差为 `1.779582`（view 5），该值是累计深度输出差异，不是几何 GT 深度误差。独立和 mixed 两组具有完全相同的这些差异。

上述门槛仅相对原生 gsplat/PyTorch 管线。本轮没有重新比较旧 diff-gaussian 后端，不表示此前后端替换带来的 SSIM/LPIPS 差异已被消除。

## 资源与验证

| 本扩展 kernel | threads/CTA | registers/thread | shared/CTA | local/thread | 可驻留 CTA/SM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 独立 device-v7 MLP | 128 | 51 | 14,952 B | 0 B | 9 |
| gsplat＋device-v7 mixed | 384 | 40 | 22,120 B | 0 B | 4 |

- 迁入的独立与 mixed MLP 在随机/边界和三组真实输入上，与封存旧版 v11 matched library 的五个输出全部逐位一致；像素输出与原生 gsplat 一致。
- 2026-10-27 **9 项 CUDA 测试通过**，覆盖尺寸、partial tile、grid-stride、early termination、空 Gaussian/零行、Graph replay 和 dtype/grad guard。此前本地 CPU suite 共 88 项，14 项因无 CUDA 跳过，代码未变更。
- 完成批次的 GPU 频率全程为 1.3005 GHz，未观察到外部 Python 工作负载；源码/checkpoint 前后指纹与源码快照核对通过。本地新增 kernel、gsplat 管线、测试与 CLI 均与正式运行快照一致。公共 `tacker_pipeline.py`、`hexplane.py` 复用此前隔离 runtime，具体版本以归档快照为准。
- 本次未重新运行 memcheck；此前 Orin 的 Compute Sanitizer 报 GPU debugging features disabled，不记作通过。

这次证明了 **gsplat 与旧优化 MLP 的组合可以取得物理融合净收益**。它不否定先前 ABI-v5 scalar-tail 组合的回退结果，也不将旧 Graph、MLP 优化或换后端的全部收益归入物理融合。
结论限于本次 Orin、场景、checkpoint 和配置；本轮没有在 A6000 上运行。

## 使用与原始记录

入口保持显式选择：

```text
--raster_backend gsplat --gsplat_mixed_kernel tensor_v7 --gsplat_mixed_blocks 3486
--execution-mode gsplat_mixed
```

以 `--execution-mode gsplat_matched_graph` 运行同 MLP 的独立组。两者都需要相应硬件的 `--tacker-profile` 和 `--workload-name flame_steak`。默认后端/默认 kernel 未切换。

远端代码与缓存复用 `/data/qyfeng/gsplat-v7-20261021`，续跑结果单独存入 `results_20261027`；暂停记录保留，没有覆盖。所有远端新文件均在 `/data/qyfeng`。

- [结果汇总](../benchmarks/results/gsplat_v7_20261027/summary.json)
- [三组 leaf 初筛](../benchmarks/results/gsplat_v7_20261027/results_20261027/sweep/report.json)
- [正式 50 帧、逐帧指标与源码指纹](../benchmarks/results/gsplat_v7_20261027/results_20261027/formal/report.json)
- [本次 CUDA 复核](../benchmarks/results/gsplat_v7_20261027/cuda_tests_20261027.log) · [运行日志](../benchmarks/results/gsplat_v7_20261027/formal_20261027.log)
- [续跑启动器](../benchmarks/results/gsplat_v7_20261027/resume_20261027.sh) · [原始数据与源码快照归档](../benchmarks/results/gsplat_v7_20261027/raw_results.tar.gz)
