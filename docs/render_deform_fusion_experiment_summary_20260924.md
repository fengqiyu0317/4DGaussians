# RenderCUDA 与 Deformation MLP 融合实验总结

更新日期：2026-09-27

## 1. 结论摘要

当前画质验证完整、可作为正式结果引用的最快实现是 **Full Deformation MLP ABI-v5 mixed kernel + split Raster prepare + 双 slot HexPlane CUDA Graph，persistent blocks = 0**。

在 NVIDIA RTX A6000 上，对 `flame_steak` iteration 14000、111,525 个 Gaussians、1352×1014 分辨率的 50 帧 test 序列进行正式 10×50 ABBA 测试，结果为：

| 实现 | 中位 FPS | 中位 ms/frame | 相对 Serial |
|---|---:|---:|---:|
| Serial | 82.578 | 12.110 | 1.000× |
| Two-stream | 85.884 | 11.644 | 1.040× |
| Full MLP ABI-v5 mixed, PB=0 | 113.982 | 8.773 | 1.380× |
| Full MLP + split Raster prepare | 114.827 | 8.709 | 1.391× |
| Full MLP + split prepare + HexPlane Graph | **129.741** | **7.708** | **1.571×** |

Graph 版本相对 Two-stream 提升 **51.07%**，相对 Serial 提升 **57.11%**，相对同轮旧 `full_mlp_overlap` 提升 **12.99%**。它的整帧画质相对 Serial 只变化了 PSNR `-0.000211 dB`、SSIM `-1.33e-6`、LPIPS `+5.89e-7`，远低于预设的质量准入阈值。

这个实现的核心不是让同一帧的 Raster 等待 Deform，而是做跨帧稳态重叠：

```text
Mixed(t) = Raster(t) + FullDeformMLP(t+1)
```

一个 384-thread CUDA CTA 内，前 256 个线程执行当前帧 Raster 的最终 render leaf，后 128 个线程执行下一帧完整 deformation MLP 的 backbone 和五个 head。Python 层的双 slot 调度器处理首帧 prefill、稳态 mixed 以及末帧 drain。

2026-09-24 新增的显式 opt-in `full_mlp_overlap` 调度把 Raster 的 preprocess / scan / sort / range construction 拆成可延期消费的 prepared state，使 `RasterPrepare(t)` 可以与 `HexPlanePrefix(t+1)` 在两个 stream 上并行。但是原始 HexPlane 前缀需要 Python/PyTorch 逐个提交多个小 kernel，CPU 在提交完前缀后才能提交 Raster prepare，实测 GPU 上几乎没有形成并行区间。

2026-09-25 又新增第三个显式 opt-in 版本 `full_mlp_overlap_graph`：每个 slot 捕获一个固定形状的 HexPlane CUDA Graph，稳态帧只提交 xyz/time 的 device-to-device copy 和一次 graph replay，然后 CPU 立即继续提交 Raster prepare。旧 `full_mlp` 和 `full_mlp_overlap` 都保留不变。该 graph 版本已通过完整 50-view 画质准入和 5 候选正式 10×50 ABBA；50/50 个独立 child 全部通过，正式中位数为 **129.741 FPS / 7.708 ms**，因此现已取代 115.333 FPS 成为新的正式最佳结果。

## 2. 测试对象和口径

### 2.1 主工作负载

| 项目 | 设置 |
|---|---|
| GPU | NVIDIA RTX A6000, SM 8.6 |
| CUDA / PyTorch | CUDA 12.4 / PyTorch 2.4.1+cu124（新 graph 正式报告） |
| 场景 | N3DV `flame_steak` |
| Checkpoint | iteration 14000 |
| Gaussian 数 | 111,525 |
| 分辨率 | 1352×1014 |
| 测量视角 | test views 0–49，共 50 帧 |
| 预热 | 每个 child run 10 帧 |
| 正式速度协议 | 每个候选 10 个 trial，候选按 ABBA 交错，seed 0 |
| 主计时 | `perf_counter` 包围 50 帧完整序列，末尾 `cudaDeviceSynchronize` |

加载、renderer prepare 和 warmup 不计入速度区间。每次 GPU 实验前均重新检查设备使用状态，不与其他任务共用实验 GPU。

### 2.2 各名称的含义

- **Serial**：完成当前帧 deformation，再完成 rasterization。
- **Two-stream**：Deform 和 Raster 仍是独立 kernel，使用 CUDA stream/event 做跨帧重叠。
- **Current Tacker**：早期已有 Tacker 路径。
- **C0/C1/C2**：把选定 head 的第一层 linear 与 Raster 结合的渐进方案。
- **C3 packed**：共享输入并打包多个 head 的第一层 linear，是 ABI-v3。
- **C4 whole-head**：尝试把完整 head 放入 mixed kernel，是 ABI-v4。
- **Full MLP ABI-v5**：把 deformation backbone 和 position/scale/rotation/opacity/SH 五个完整 head 一次性放入 mixed kernel。

## 3. 端到端 FPS 结果

### 3.1 Graph 版本同场正式对比

下表五项来自同一次 10×50 ABBA 速度实验：每个候选 10 个独立 child process，每个 child 先 warmup 10 帧，再计时完整 50 帧。50/50 个 child 全部通过。

| 模式 | 实际执行模式 | 中位 FPS | 中位 ms/frame | min–max FPS | CV | Peak allocated | Peak reserved |
|---|---|---:|---:|---:|---:|---:|---:|
| Serial | `serial` | 82.578 | 12.110 | 80.835–83.198 | 0.749% | 296.5 MiB | 506 MiB |
| Two-stream | `two_stream` | 85.884 | 11.644 | 84.474–86.389 | 0.769% | 383.1 MiB | 968 MiB |
| Full MLP PB=0 | `tacker` | 113.982 | 8.773 | 113.392–115.145 | 0.386% | 458.0 MiB | 940 MiB |
| Full MLP overlap PB=0 | `tacker` | 114.827 | 8.709 | 114.434–115.219 | 0.232% | 861.1 MiB | 1240 MiB |
| Full MLP overlap graph PB=0 | `tacker` | **129.741** | **7.708** | 129.011–130.752 | 0.451% | 867.4 MiB | 1222 MiB |

比较结果：

- Graph / Serial 的中位 FPS ratio 为 1.5711，10,000 次 paired bootstrap 95% CI 为 `[1.5647, 1.5788]`。
- Graph / Two-stream 的中位 FPS ratio 为 1.5107，95% CI 为 `[1.5027, 1.5315]`。
- Graph / 旧 Full MLP 的中位 FPS ratio 为 1.1383，95% CI 为 `[1.1338, 1.1442]`。
- Graph / Full MLP overlap 的中位 FPS ratio 为 1.1299，95% CI 为 `[1.1247, 1.1350]`。
- 候选排名为 `full_mlp_overlap_graph > full_mlp_overlap > full_mlp > two_stream > serial`。

本轮没有对 `current_tacker` 重做同一画质准入，因此它被正式调度器以 `correctness_invalid` 排除。上述同场正式结论不包含“Graph 相对 Current Tacker”的直接声明。驱动选择结果为 `experimental_winner=deployment_winner=full_mlp_overlap_graph`，但本次测试没有改写实际 deployment profile。

### 3.2 历史版本演进

下表的 C0–C4 数据来自 Phase 4 正式 10×50 ABBA；ABI-v5 和 Graph 行来自后续各自的独立正式实验。各批基线 FPS 接近，适合展示技术演进，但不应当作同一轮 ABBA 排名；同轮直接比较以 3.1 节为准。

| 阶段 | 代表实现 | 中位 FPS | 相对当轮 Serial |
|---|---|---:|---:|
| Baseline | Serial | 82.588 | 0.00% |
| Stream overlap | Two-stream | 86.342 | +4.54% |
| Existing scheduler | Current Tacker | 87.398 | +5.82% |
| C0 | Legacy position first-linear | 87.556 | +6.02% |
| C1 | Opacity first-linear | 85.285 | +3.27% |
| C2 | Five independent first-linear heads | 89.589 | +8.48% |
| C3 / ABI-v3 | Packed five-head first-linear, PB=5440 | **100.508** | **+21.70%** |
| C4 / ABI-v4 | Whole opacity head, PB=5440 | 44.698 | -45.88% |
| ABI-v5，独立后续实验 | Full deformation MLP, PB=0 | **115.333** | **+38.57%** |
| ABI-v5 + Graph，新正式实验 | Split prepare + HexPlane Graph, PB=0 | **129.741** | **+57.11%** |

C3 在 Phase 3.1 中为 100.942 FPS，Phase 4 重放为 100.508 FPS，两次差约 0.43%。ABI-v5 的绝对 FPS 比 Phase 4 C3 高 14.75%；为减少跨实验基线漂移的影响，按各自 Serial 归一化后，ABI-v5 的加速幅度比 C3 高约 13.87%。这些是跨实验趋势，不是配对置信区间结论。

C4 的 44.698 FPS 是一个负面实验结果：它只是单 opacity whole-head representative，不应解读为“版本越新越慢”。ABI-v5 改变了计算组织、参数打包和 CTA 内并行形式，不是 C4 的简单扩展。

### 3.3 Persistent-block 参数结果

正式实验前的 5×50 筛查结果：

| ABI-v5 配置 | FPS | 说明 |
|---|---:|---|
| PB=336 | 109.23 | Persistent grid |
| PB=0 | **115.47** | Direct logical grid，快 5.71% |

因此正式 ABBA 选择了 PB=0。后续进行的单 trial、50 帧 PB refinement 中，每个候选有四次重复：

| PB | 4 次单-trial FPS 的中位数 |
|---:|---:|
| 4096 | 116.701 |
| 4480 | **117.274** |
| 4800 | 117.043 |
| 4864 | 117.154 |

PB=4480 是旧 Full MLP 路径上最有希望的调参候选，但这些是单 trial 调参数据，没有完成与 PB=0 同规格的 10-trial ABBA 和 50-view 画质复验。当前新的正式最佳结果是 graph PB=0 的 **129.741 FPS**，不应把该历史调参数据当作新记录。

## 4. 精度和图像质量

### 4.1 ABI-v5 整帧 50-view 画质

所有模式都实际进入请求路径，没有 fallback。

| 模式 | Mean PSNR | Mean SSIM | Mean LPIPS | 相对 Serial | 结果 |
|---|---:|---:|---:|---|---|
| Serial | 32.574433 | 0.95773631 | 0.140365163 | Baseline | PASS |
| Two-stream | 32.574433 | 0.95773631 | 0.140365163 | 三项一致 | PASS |
| Full MLP PB=0 | 32.574222 | 0.95773498 | 0.140365753 | PSNR -0.000211 dB；SSIM -1.33e-6；LPIPS +5.89e-7 | **PASS** |
| Full MLP overlap graph PB=0 | 32.574222 | 0.95773498 | 0.140365753 | PSNR -0.000211 dB；SSIM -1.33e-6；LPIPS +5.89e-7 | **PASS** |

准入阈值为：

- PSNR drop `<= 0.05 dB`
- SSIM drop `<= 1e-4`
- LPIPS increase `<= 1e-4`

ABI-v5 的实际变化比阈值小两个数量级以上。就本次 50-view 评估而言，mixed kernel 生成的图像与 Serial 在视觉指标上可视为等价。

### 4.2 Kernel 数值精度

Standalone `deform_mlp_device_v1` 在 111,525 行实际负载上，与匹配混合精度 reference 按 `rtol=2e-3` / `atol=5e-4` 对比通过。按 position/scale/rotation/opacity/SH 顺序，最大绝对误差为：

```text
9.536743e-7
1.192093e-6
1.311302e-6
9.536743e-7
1.668930e-6
```

融合路径是显式混合精度推理：HexPlane feature、backbone、bias、tail 及最终输出使用 FP32；backbone ReLU 后的 activation 显式转为 FP16，五个 `128×128` head 第一层权重也使用 FP16，WMMA 用 FP32 accumulator。因此它不追求与原 FP32 PyTorch 路径逐 bit 一致，而是通过 kernel 数值容差和整帧 PSNR/SSIM/LPIPS 双重验证。

### 4.3 历史 C3 画质

Phase 4 C3 winner 相对 Serial 的画质差异为：

- PSNR drop `0.000209961 dB`
- SSIM drop `1.3161e-6`
- LPIPS increase `6.7294e-7`

这些结果同样通过画质准入。C3 和 ABI-v5 的精度变化都处于极小数值差异范围，没有发现可归因于融合的图像质量下降。

## 5. RenderCUDA 与 Deform 是如何融合的

### 5.1 融合边界

ABI-v5 没有把整个 4DGaussians 渲染图都放进一个 kernel。它选择的边界是：

```text
Deform prefix (PyTorch/CUDA)
  HexPlane grid query -> deform_features [N,32]
                    |
                    v
Mixed CUDA kernel
  current-frame Raster render leaf
  + next-frame backbone and all five MLP heads
                    |
                    v
Deform suffix (PyTorch/CUDA)
  apply mask/residual -> scaling/rotation/opacity activation -> render state
```

也就是说：

- HexPlane feature 查询仍在前缀中执行。
- Deformation `Linear(32,128)` backbone、ReLU、五个完整 head 在 ABI-v5 CUDA kernel 中执行。
- Mask、residual 组合和 Gaussian 属性 activation 仍在后缀中执行。
- Raster 的几何预处理、binning 等既有阶段保持不变，融合的是最终 `general_ptb_render` leaf。

这个边界避免了重写 HexPlane 和整个 rasterizer，同时把原来由多个 PyTorch linear/ReLU kernel 组成的主要 Deform MLP 计算合并进一次物理 mixed launch。

### 5.2 同一 CTA 的线程分区

全局 kernel 符号为 `tacker_mix_render_deform_mlp_v5`，每个 block 共 384 个线程：

| 线程范围 | 数量 | 任务 | 同步域 |
|---|---:|---|---|
| `[0,255]` | 256 | `general_ptb_render<NUM_CHANNELS>` | named barrier 1，256 participants |
| `[256,383]` | 128 | `deform_mlp_gptb_device_v1` | named barrier 2，128 participants |

两个 subgroup 不使用 CTA-wide `__syncthreads()`。Raster 线程只进入 barrier 1，Deform 线程只进入 barrier 2；因此一侧的 grid 已经没有逻辑任务时，可以安全 no-op，不会把另一侧卡在全 CTA barrier 上。GPU warp scheduler 在同一 CTA 内交错发射 Raster 和 Deform 的 warp，从而在一次 kernel launch 内实现物理共驻留。

### 5.3 Deformation MLP 的 device adapter

`deform_mlp_gptb_device_v1` 是 standalone `deform_mlp_device_v1` 的 mixed-kernel adapter。两者共用同一个 `deform_mlp_device_impl_v1`，区别只是线程编号和同步策略：

- Standalone：128-thread CTA，用 `__syncthreads()`。
- Mixed：从 `threadIdx.x` 减去 256 得到局部 tid，只允许 128 个 Deform 线程进入 named barrier 2。

每个 Deform logical tile 处理 16 个 Gaussian row，计算顺序为：

1. 把 `[16,32]` FP32 HexPlane features 载入 shared memory，tail row 用 0 padding。
2. 执行 FP32 backbone `Linear(32,128)` 和 bias。
3. 执行 ReLU，并显式转换为 FP16 activation。
4. 按 position、scale、rotation、opacity、SH 顺序执行五个 head。
5. 每个 head 的 `Linear(128,128)` 使用 WMMA/Tensor Core，FP16 输入和权重、FP32 accumulator。
6. 执行 ReLU 和 FP32 tail linear，分别写出 3、3、4、1、48 维 delta。

输出固定为：

```text
position [N,3]
scale    [N,3]
rotation [N,4]
opacity  [N,1]
SH       [N,48]
```

### 5.4 Logical grid 与 persistent blocks

Raster 和 Deform 的 logical task 数通常不同：

```text
raster_logical_blocks = raster_grid_x * raster_grid_y
deform_logical_blocks = ceil(N / 16)
logical_blocks = max(raster_logical_blocks, deform_logical_blocks)
```

当 `persistent_blocks=0` 时，ABI-v5 启动 `logical_blocks` 个物理 CTA，即 direct logical grid。当设置正数 PB 时，物理 block 数被 clamp 到 `[1, logical_blocks]`，每个 block 用 grid-stride 方式处理后续 logical task。

在任一 logical index 上：

- 如果 Raster 仍有 tile，前 256 线程处理它；否则 Raster subgroup no-op。
- 如果 Deform 仍有 16-row tile，后 128 线程处理它；否则 Deform subgroup no-op。

所以这种设计不要求 Raster tile 数和 Gaussian/16 完全相等。PB=0 在当前工作负载上比 PB=336 快，说明直接保留更多 CTA 并行度比限制成小型 persistent grid 更合适。

### 5.5 旧正式版本中 Raster 其它阶段在什么时候运行

每一次稳态 mixed 调用并不是一进入 `Rasterizer::forward` 就同时运行 Raster 和 Deform MLP。真实顺序是：先在 `raster_stream` 上完成当前帧 Raster 的前处理、binning 和 sorting，到最后的像素 blending 阶段才启动 mixed kernel，并在该 kernel 中同时计算下一帧的 Deform MLP。

对一个稳态帧对 `(t, t+1)`，时间线如下：

```text
deform_stream:  HexPlane prefix(t+1) --record prefix_ready-- [mask work]
                                             |
                                             v
raster_stream: wait state_ready(t) + wait prefix_ready(t+1)
               -> preprocessCUDA(t)
               -> CUB InclusiveSum(t)
               -> copy num_rendered to CPU + cudaStreamSynchronize
               -> duplicateWithKeys(t)
               -> CUB RadixSort(t)
               -> identifyTileRanges(t)
               -> mixed kernel {
                    general_ptb_render(t)
                    + FullDeformMLP(t+1)
                  }
                                             |
                                             v
deform_stream:                         suffix(t+1)
```

因此，Raster 各部分的运行时机是：

| Raster 阶段 | 在每帧中的运行时机 | 运行位置 |
|---|---|---|
| 投影、协方差、SH 转颜色、半径与 tile 覆盖计算 | mixed kernel 之前 | `preprocessCUDA`，`raster_stream` |
| tile 数前缀和 | mixed kernel 之前 | CUB `InclusiveSum`，`raster_stream` |
| 获取展开后的 Gaussian instance 数 | scan 之后、分配 binning buffer 之前 | D2H copy + `cudaStreamSynchronize` |
| Gaussian 按覆盖 tile 展开 | mixed kernel 之前 | `duplicateWithKeys`，`raster_stream` |
| 按 tile/depth 排序 | mixed kernel 之前 | CUB `DeviceRadixSort`，`raster_stream` |
| 生成每个 tile 的起止 range | mixed kernel 紧前 | `identifyTileRanges`，`raster_stream` |
| 颜色、透明度和深度 blending | mixed kernel 内 | `general_ptb_render(t)` |

关键点是：**FullDeformMLP(t+1) 只与 Raster(t) 的最后 blending leaf 物理融合，不与 Raster(t) 的 preprocess、scan、duplicate 或 sort 同时执行。** 这些前置阶段必须先生成排序后的 `point_list` 和每个 tile 的 `ranges`，`general_ptb_render` 才能开始 blending。

对 50 帧序列，上述 Raster 前置阶段每帧都会执行一次：前 49 帧的最后 blending 使用 mixed kernel，第 50 帧在 drain 阶段走普通 solo Raster，其最后 blending 使用独立 `renderCUDA`。

### 5.6 新 `full_mlp_overlap` 版本的并行区间

新版本不改变 mixed leaf，只改变它之前的排队位置。调度器先在 `deform_stream` 上排下一帧 HexPlane prefix，然后在 `raster_stream` 上等待当前帧 render state 并执行 Raster prepare。只有即将启动最终 leaf 时，`raster_stream` 才等待 `prefix_ready(t+1)`：

```text
deform_stream: wait reusable slot
               -> HexPlanePrefix(t+1)
               -> record prefix_ready(t+1)
               -> optional mask work(t+1)

raster_stream: wait state_ready(t)
               -> preprocessCUDA(t)
               -> CUB InclusiveSum(t)
               -> D2H num_rendered + stream-local synchronize
               -> duplicateWithKeys(t)
               -> CUB RadixSort(t)
               -> identifyTileRanges(t)
               -> wait prefix_ready(t+1)
               -> mixed leaf {
                    general_ptb_render(t)
                    + FullDeformMLP(t+1)
                  }
```

这里的 D2H count 仍会阻塞提交 Raster prepare 的 CPU 线程，但它只同步 `raster_stream`；已经提交到 `deform_stream` 的 HexPlane prefix 可以继续在 GPU 上运行。因此新版本扩大的是两个 prefix 的 GPU 重叠区间，而不是消除 Raster 内部为动态分配 binning buffer 所需的 host readback。

末帧同样分成 `prepare_raster` 和普通 prepared render leaf；不会为了 drain 再跑一遍 preprocess/sort。实现作为新 partition kind `full_mlp_overlap` 注册，旧 `FullMlpPartition`、旧 `forward_with_deform_mlp`、旧 pybind 和旧 profile 均未替换。

当前验证状态：

- CUDA 扩展在 RTX A6000 / CUDA 12.4 环境编译通过。
- 非空 Raster 单元负载中，split mixed 的 color/radii/depth 与旧 ABI-v5 逐元素一致；五个 deform 输出通过原有 `atol=5e-4, rtol=2e-3` 容差。
- prepared solo leaf 与普通 Raster 的 color/radii/depth 逐元素一致。
- 真实 `flame_steak` 50 帧执行计数为 `1/49/49/49/1/50`，实际模式为 `tacker` 且无 fallback。
- 同环境单 trial 中，旧 `full_mlp` PB=0 为 115.989 FPS，新 `full_mlp_overlap` PB=0 为 116.680 FPS，初步提升 0.596%。这只是工程回归，不能替代 ABBA 多 trial 结论。

### 5.7 `full_mlp_overlap_graph` 如何压缩 HexPlane 提交时间

原 `full_mlp_overlap` 虽然先调用 `_enqueue_prefix`，再调用 `_enqueue_raster_prepare`，但这两个 Python 调用不等于 GPU 已开始并行。HexPlane 查询包含 normalize/concat、多个 plane 坐标构造、`grid_sample`、plane product 和 multi-scale concat。PyTorch eager 需要 CPU 依次发布这些操作。上一轮仪器化测量中，HexPlane 的 host prepare 约为 1.618 ms，而它在 GPU 上的 prefix 约为 1.541 ms；即 CPU 还没有走到 Raster prepare 的提交点，GPU 已经基本排空 HexPlane 队列。

graph 版本把这个问题改成：

```text
第一次使用 slot 0/1（warmup，不计时）：
  3 次 HexPlane warmup
  -> capture HexPlane CUDA Graph
  -> 保留 static xyz/time/output tensor

稳态每帧：
  deform_stream: copy xyz -> static_xyz
                 copy time -> static_time
                 graph.replay()
                 record prefix_ready

  CPU:           立即转去提交 RasterPrepare(t)

  raster_stream: RasterPrepare(t)
                 wait prefix_ready(t+1)
                 mixed leaf
```

两个 pipeline slot 各自拥有 graph 和 static output，只在该 slot 的 `raster_done` 后才复用，因此 graph replay 不会覆盖仍被 mixed kernel 消费的 feature。第一次 capture 在非计时 warmup 中完成，不进入 FPS 区间。

为了让真实 HexPlane 可被当前 PyTorch 2.4.1 CUDA Graph 捕获，`scene/hexplane.py` 新增 `forward_cuda_graph`。它只替换一处不可 capture 的高级索引：`pts[..., coo_comb]` 会在 capture 期间隐式创建 CUDA index tensor，graph 路径改用逐轴整数 view 加 `torch.stack`。`grid_sample`、坐标顺序和乘积公式均不变。

前期同一进程 5×50 工程测量为 131.009 FPS，随后的独立 child 正式 ABBA 结果为：

| opt-in partition | 中位 FPS | 中位 ms/frame | min–max FPS |
|---|---:|---:|---:|
| `full_mlp_overlap` | 114.827 | 8.709 | 114.434–115.219 |
| `full_mlp_overlap_graph` | **129.741** | **7.708** | 129.011–130.752 |

graph 版本相对旧 overlap 的正式中位 FPS 提升 12.99%；10,000 次 paired bootstrap ratio 的 95% CI 为 `[1.1247, 1.1350]`。两个候选的每个 50 帧 child 执行计数均为 `1/49/49/49/1/50`，实际模式为 `tacker`，无 fallback。

完整 50-view 画质对比也通过：Serial 的 mean PSNR/SSIM/LPIPS 为 `32.574433 / 0.95773631 / 0.140365163`，graph 版本为 `32.574222 / 0.95773498 / 0.140365753`；变化分别为 PSNR `-0.000211 dB`、SSIM `-1.33e-6`、LPIPS `+5.89e-7`，与旧 ABI-v5 的已知混合精度误差一致，没有新增 graph 数值损失。

该路径的准入条件是：Gaussian 数、xyz/time 形状、dtype 和 device 必须与 capture 时一致；模型必须提供 `forward_cuda_graph`；仍只支持 inference。任一条不满足时，不会误用这个 profile。

### 5.8 跨帧双 slot 流水

Mixed kernel 同时消费当前帧已完成的 render state，以及下一帧已生成的 HexPlane features。两个 slot 轮换使用，CUDA event 建立依赖：

```text
D(0) full prefill
      |
      v
prefix(1) -> [ Raster(0) + MLP(1) ] -> suffix(1)
prefix(2) -> [ Raster(1) + MLP(2) ] -> suffix(2)
    ...
prefix(49) -> [ Raster(48) + MLP(49) ] -> suffix(49)
                                              |
                                              v
                                      solo Raster(49)
```

对 50 帧序列，实际执行计数必须为：

| 事件 | 次数 |
|---|---:|
| 首帧 full deformation prefill | 1 |
| HexPlane prefix | 49 |
| Mixed launch | 49 |
| Deform suffix | 49 |
| 末帧 solo Raster drain | 1 |
| 输出帧 | 50 |
| 每个 deformation head 的求值次数 | 50 |

正式验证的计数与上表完全一致。这项检查很重要，因为它能防止“通过少算 head 或丢帧获得更高 FPS”。

### 5.9 Python/C++/CUDA 调用链

```text
TackerRenderer.render_sequence
  -> FullMlpPartition.prepare
     -> prepare_full_mlp_task                  # HexPlane prefix
  -> FullMlpPartition.launch_mixed
     -> _forward_with_deform_mlp
        -> GaussianRasterizer.forward_with_deform_mlp
           -> rasterize_gaussians_with_deform_mlp
              -> RasterizeGaussiansWithDeformMlpCUDA
                 -> Rasterizer::forward
                    -> launchMixedRenderDeformMlp
                       -> tacker_mix_render_deform_mlp_v5
                          |- general_ptb_render
                          `- deform_mlp_gptb_device_v1
  -> FullMlpPartition.finish
     -> finish_full_mlp_task                   # residual/activation suffix
```

`full_mlp_overlap` 复用同一 `prepare_full_mlp_task` 和 `finish_full_mlp_task`，中间调用链改为：

```text
TackerRenderer.render_sequence
  -> FullMlpOverlapPartition.prepare_raster
     -> GaussianRasterizer.prepare_for_deform_mlp_overlap
        -> PrepareGaussiansForDeformMlpOverlapCUDA
           -> Rasterizer::prepareForwardForDeformMlpOverlap
  -> wait prefix_ready
  -> FullMlpOverlapPartition.launch_prepared_mixed
     -> GaussianRasterizer.render_prepared_with_deform_mlp_overlap
        -> RenderPreparedGaussiansWithDeformMlpOverlapCUDA
           -> Rasterizer::renderPreparedDeformMlpOverlap
              -> launchMixedRenderDeformMlp
```

graph opt-in 在上述 split-prefix 路径上只替换 HexPlane 提交步骤：

```text
TackerRenderer._enqueue_prefix
  -> FullMlpOverlapGraphPartition.create_prefix_graph   # 每 slot 一次
     -> _HexPlaneCudaGraphSlot
        -> HexPlaneField.forward_cuda_graph
        -> torch.cuda.CUDAGraph capture
  -> FullMlpOverlapGraphPartition.prepare_graph         # 每帧
     -> prepare_full_mlp_graph_task
        -> static xyz/time copy
        -> CUDAGraph.replay
        -> record prefix_ready
```

C++ binding 在 launch 前检查：必须在 `torch.no_grad()` 下运行；所有 tensor 必须在同一 CUDA device 上且 contiguous；feature 必须是 `[N,32]` FP32；打包第一层权重必须是 `[5,128,128]` FP16 且满足 WMMA 对齐；输出宽度固定为 3/3/4/1/48。

## 6. 为什么能够加速

主要收益来自三个方面：

1. **减少 MLP kernel 与 framework 调度开销。** 原版 Deform 的 backbone、多个 ReLU 和五个 head 由多个 PyTorch CUDA kernel 组成；ABI-v5 把 backbone 和所有 head 收敛到一个 device function 内。
2. **做真正的物理共驻留。** Raster 和 Deform 不再只是两个独立 kernel 交给 GPU 尝试 overlap，而是同一 CTA 内两组独立 warp，共用一次 block 调度和驻留机会。
3. **跨帧填补资源空洞。** `Raster(t)` 与 `MLP(t+1)` 没有同帧数据依赖，可以同时执行；双 slot 和 event 让这种重叠在长序列中稳定发生。

真实串行渲染上下文中的 Deform-only 测量也支持这一解释：

| 测量项 | ms/frame |
|---|---:|
| 原版完整 Deform | 7.664655 |
| 使用 `deform_mlp_device_v1` 后的完整 Deform | 4.991628 |
| 其中单个 fused MLP kernel | 2.985140 |

仅替换 Deform MLP 就使完整 Deform 延迟降低 34.87%，加速 1.54×。这里的 2.985140 ms 是真实 `deform -> activation -> rasterize` 序列中的 kernel 时间；连续热启动 standalone 中位数为 2.839593 ms，会略微高估实际集成收益。

## 7. Kernel 资源和剩余瓶颈

ABI-v5 mixed kernel 的资源查询结果为：

| 指标 | 结果 |
|---|---:|
| Threads/block | 384 |
| Registers/thread | 40 |
| Static shared memory | 21,888 B |
| Active blocks/SM | 4 |
| 理论 occupancy | 1.0 |

对 standalone `deform_mlp_device_v1` 的 NCU 和 kernel 内部插桩显示：

- 早期实现的 WMMA A shared-memory load 存在严重 bank conflict；对 activated shared-memory row 增加 8 个 half padding 后，该类过量 wavefront 被消除，CUDA Event 时间改善 6.49%。
- 当时 full-grid NCU 的 L2 hit rate 为 99.73%，DRAM throughput 仅 1.47%，因此不是简单的 DRAM 带宽饱和问题。
- 当前 kernel 内部插桩中，全部 tail 合计约占 79.94%，其中 48 维 SH tail 单独占 66.90%，折算约 1.900 ms。
- Backbone linear/ReLU/转 FP16 约占 9.56%，五个 head 的 WMMA first-linear 合计约占 8.53%。

因此下一阶段的主要 kernel 优化对象是 SH tail 的 48 维 FP32 reduction/读权重路径，而不是继续只围绕 WMMA first-linear 做微调。

上述 A6000 源码级归因来自共用同一 `deform_mlp_device_impl_v1` 的 standalone kernel。2026-09-27 又在 Jetson AGX Orin 上完成了 standalone Deform、solo `renderCUDA` 和 ABI-v5 mixed 的成套 full-set NCU 采集；见第 10 节。两台 GPU 的指标不能直接混合比较，但 Orin 数据补齐了 mixed kernel 自身的正式微架构证据。

## 8. 安全性、回退与限制

- ABI-v5 binding 只支持 inference，必须在 `torch.no_grad()` 下运行，没有融合 backward。
- `static_mlp` mask 当前不支持；`empty_voxel` mask 由 prefix/suffix 路径处理。
- Profile 与 workload name、分辨率、Gaussian 数和实现 manifest 绑定。不符合准入条件时回退到 Two-stream，Two-stream 不可用时再回退到 Serial。
- 混合 kernel 第一次 launch 若在输出任何帧前同步失败，序列可安全重放到 fallback；已经向调用方发布帧后则 fail closed，避免重复或乱序输出。
- 当前正式结论主要针对 `flame_steak` iteration 14000、1352×1014、111,525 Gaussians，不应直接外推为所有场景的全局最佳 PB 或最佳 variant。
- 50 帧序列有 49 次 mixed overlap；单帧请求无法获得稳态跨帧重叠收益。
- `full_mlp_overlap_graph` 的 graph capture 有额外 static tensor/graph 内存，并依赖固定形状。它已通过 50-view 画质准入和 10×50 ABBA，正式中位数为 129.741 FPS。
- 正式驱动选出 `full_mlp_overlap_graph`，但实验没有自动改写生产 deployment profile；部署仍应保留现有回退链。

## 9. 实现与证据索引

### 正式结果

- [Graph 版本正式 10×50 ABBA 与 50-view 画质摘要](../benchmarks/results/full_mlp_graph_formal_20260925/README.md)
- [Graph 版本正式聚合 JSON](../benchmarks/results/full_mlp_graph_formal_20260925/report.json)
- [Graph 与四个对照的 paired-bootstrap 比较](../benchmarks/results/full_mlp_graph_formal_20260925/comparisons.json)
- [Graph 版本 50-view 画质报告](../benchmarks/results/full_mlp_graph_formal_20260925/graph_quality_50f.json)
- [ABI-v5 端到端画质与速度报告](full_mlp_e2e_benchmark_20260920.md)
- [Phase 4 C0–C4 准入、画质和 Formal FPS](../tacker_profiles/baselines/a6000_phase4_20260914/README.md)
- [Phase 3.1 候选搜索、Formal FPS 与 Nsight](../tacker_profiles/baselines/a6000_phase31_20260913/README.md)
- [PB 端到端调参原始结果](../benchmarks/results/pb_e2e_validation_v1/)

### 融合实现

- [Python 双 slot 流水、FullMlpPartition 和 prefix/suffix](../gaussian_renderer/tacker_pipeline.py)
- [HexPlane eager 与 CUDA-Graph capture-safe 入口](../scene/hexplane.py)
- [PyTorch/C++ binding 与 tensor 合同](../submodules/depth-diff-gaussian-rasterization/rasterize_points.cu)
- [384-thread mixed kernel 与 launch 策略](../submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/tacker_mixed.cu)
- [Mixed task 结构与 C++ launcher 接口](../submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/tacker_mixed.h)
- [ABI-v5 manifest](../submodules/depth-diff-gaussian-rasterization/abi/tacker_mixed_render_deform_mlp_v5.json)
- [Deformation MLP device adapter](../tacker_ext/include/deform_mlp_device.cuh)
- [Deformation MLP ABI manifest](../tacker_ext/abi/deform_mlp_v1.json)

## 10. Jetson AGX Orin 复验（2026-09-27）

相同 `flame_steak` checkpoint、111,525 Gaussians、1352×1014 和 test views 0–49 已在 Jetson AGX Orin（SM 8.7、16 SM、CUDA 12.6、PyTorch 2.8.0）上完成独立编译、50-view 画质门禁和正式 10×50 ABBA。50/50 个正式 child 均成功，三个融合版本均实际执行 `tacker`，没有 fallback。

| 版本 | 中位 FPS | 中位 ms/frame | 相对 Serial | 相对 Two-stream |
|---|---:|---:|---:|---:|
| Serial | 21.017 | 47.581 | 1.000× | 0.995× |
| Two-stream | 21.130 | 47.326 | 1.005× | 1.000× |
| Full MLP | 23.023 | 43.435 | 1.095× | 1.090× |
| Full MLP + split Raster prepare | 23.066 | 43.354 | 1.098× | 1.092× |
| Full MLP + split prepare + HexPlane Graph | **24.287** | **41.174** | **1.156×** | **1.149×** |

Graph 相对 Two-stream 的中位 FPS ratio 为 `1.1494`，10,000 次 paired bootstrap 95% CI 为 `[1.1473, 1.1520]`。Graph 相对 overlap 再提升 5.29%，而 overlap 相对未拆分的 Full MLP 只提升 0.19%。

画质方面，Serial 为 PSNR `32.574433`、SSIM `0.957736305`、LPIPS `0.140365341`；三个融合版本均为 PSNR `32.574222`、SSIM `0.957734996`、LPIPS `0.140365921`。相对变化为 PSNR `-0.000211 dB`、SSIM `-1.309e-6`、LPIPS `+5.803e-7`，全部远低于准入阈值。

Orin 上的提升小于 A6000，主要因为瓶颈组成已经改变。此前同场景 Jetson Nsight 基线中，Rasterization 为 `23.999 ms/frame`、Deformation 为 `23.594 ms/frame`，两者接近各占一半；其中 `renderCUDA` 单项约 `17.452 ms/frame`。A6000 基线中 Deformation `8.742 ms/frame`、Rasterization `4.139 ms/frame`，能够由 ABI-v5 消除或跨帧隐藏的 Deformation 占比更高。融合不会消除 Raster preprocess、scan、sort、range construction、首帧 prefill 和末帧 drain，因此 Orin 的剩余 Raster 下限更高。

Orin 本机 NCU 2024.3.1 的 full-set kernel replay 进一步解释了这个差异：

| 目标 kernel | Duration | Compute | Memory hierarchy | Achieved occupancy | No eligible | Instructions |
|---|---:|---:|---:|---:|---:|---:|
| standalone `tacker_deform_mlp_delta_v1` | 12.442 ms | 59.70% | 93.95% | 66.36% | 51.62% | 498.6 M |
| solo `renderCUDA` | 17.489 ms | 90.03% | 33.36% | 98.74% | 9.60% | 1306.3 M |
| mixed `tacker_mix_render_deform_mlp_v5` | 29.892 ms | 78.69% | 60.92% | 83.14% | 21.28% | 1951.4 M |

Standalone Deform 偏 L1/TEX/shared-memory 与调度器受限，solo Raster 偏 compute/issue 受限；它们虽有互补性，但融合后仍竞争 issue slot、L1/TEX、shared-memory 路径和浮点执行管线。两个 standalone 的 NCU duration 之和为 `29.932 ms`，mixed 为 `29.892 ms`，仅低 `0.13%`，远未接近完全 overlap 时由 `max(12.442, 17.489) = 17.489 ms` 给出的理想下限。Mixed 还比两个独立 kernel 的指令数之和多 8.12%，achieved occupancy 只有 83.14%。因此物理共驻留成立，但 Orin 上 kernel 内部的净关键路径重叠很小；端到端约 9% 的 Full MLP 收益主要还来自消除原 PyTorch MLP 的多个 launch 和中间数据流量，Graph 的额外 5.29%则来自减少 host launch/bubble。

三个 NCU 目标由独立进程采集，且 full metric set 使用 kernel replay，上述 duration 只用于瓶颈归因，正式速度仍以 10×50 ABBA 为准。Jetson 的 DRAM byte 百分比计数在本次报告中为零，不能据此断言没有 DRAM 流量；表中的 Memory 是 NCU 整体 memory hierarchy throughput。

完整 Orin 报告与逐 trial 证据见 [Jetson AGX Orin 复验摘要](../benchmarks/results/jetson_orin_20260927/flame_steak/README.md)。

### 性能分析

- [五版本 Nsight Systems kernel 耗时占比与 Graph node 级 trace](../nsight_reports/flame_steak_versions_kernel_share_node_20260927/README.md)
- [真实渲染上下文中的 Deform 与 fused MLP 测量](../tacker_ext/profile_results/20260923-a6000-deform-render-context.md)
- [Standalone Deform MLP NCU 源码级瓶颈归因](../tacker_ext/profile_results/20260917-a6000-source-bottlenecks.md)
- [Activated shared-memory padding 实验](../tacker_ext/profile_results/20260917-a6000-activated-padding.md)

## 11. 可用于报告的简短表述

> 我们将当前帧 Gaussian Rasterization 的最终 render leaf 与下一帧完整 deformation MLP 融合到同一个 384-thread CUDA kernel 中，并用双 slot CUDA Graph 压缩 HexPlane 前缀的 host 提交时间，使它能与 Raster prepare 实际重叠。在 RTX A6000 的 `flame_steak` 50 帧正式 10×50 ABBA 实验中，该 `full_mlp_overlap_graph` 实现达到 129.741 FPS，相对 Serial 提升 57.11%，相对 Two-stream 提升 51.07%，相对旧 `full_mlp_overlap` 提升 12.99%。其相对 Serial 的 PSNR、SSIM 和 LPIPS 变化分别为 -0.000211 dB、-1.33e-6 和 +5.89e-7，通过整帧画质准入。

## 12. N3DV 多场景扩展验证

2026-09-25 又在 `coffee_martini`、`cook_spinach` 和 `flame_salmon_1` 上重新训练 iteration 14000 checkpoint，并对 Serial、Two-stream、`full_mlp`、`full_mlp_overlap` 和 `full_mlp_overlap_graph` 做了同样的 50-view 画质门控与 10×50 ABBA。

| 场景 | Serial FPS | Two-stream FPS | Full MLP FPS | overlap FPS | Graph FPS | Graph vs Two-stream |
|---|---:|---:|---:|---:|---:|---:|
| coffee_martini | 66.802 | 69.056 | 88.442 | 88.521 | **97.421** | +41.08% |
| cook_spinach | 74.278 | 76.959 | 103.558 | 103.590 | **116.280** | +51.09% |
| flame_salmon_1 | 60.713 | 62.593 | 79.789 | 79.942 | **87.302** | +39.47% |

三个场景的实验赢家均是 Graph 版本。三场景几何平均上，Graph 相对 Serial、Two-stream 和旧 overlap 分别提升 48.62%、43.79% 和 10.50%。三个融合版本在每个场景上的画质指标相同并全部通过；最大观测差异是 coffee 的 0.000340 dB PSNR drop、7.794e-6 SSIM drop 和 2.055e-6 LPIPS increase。

详细协议、逐场景稳定性、质量数值和 150 个正式 child-run artifact 见 [N3DV 多场景正式验证](../benchmarks/results/multiscene_formal_20260925/README.md)。

## 13. 五版本 kernel 耗时占比

2026-09-27 使用 Nsight Systems 2024.6.2 和 `--cuda-graph-trace=node`，在同一 `flame_steak` 50-view workload 上统一重采了 Serial、Two-stream、Full MLP、Full MLP overlap 和 Full MLP overlap graph。Graph replay 内的 child kernel 已按 node 展开；各 kernel duration 之和用于表示 GPU 工作量占比，并不等同于并行版本的关键路径时间。

| 版本 | Nsight 墙钟 ms/frame | GPU 投影跨度 ms/frame | kernel 累计 ms/frame | 主要 kernel family |
|---|---:|---:|---:|---|
| Serial | 12.681 | 12.752 | 11.095 | GEMM 35.40%，Raster render 21.38%，activation 15.26% |
| Two-stream | 12.526 | 12.585 | 12.984 | GEMM 32.71%，Raster render 19.55%，activation 13.73% |
| Full MLP | 9.594 | 9.619 | 7.362 | mixed v5 59.76%，radix sort 14.06%，elementwise 10.31% |
| Full MLP overlap | 9.547 | 9.573 | 7.375 | mixed v5 59.80%，radix sort 14.03%，elementwise 10.30% |
| Full MLP overlap graph | **7.950** | **7.975** | 7.732 | mixed v5 58.25%，radix sort 13.43%，elementwise 9.66% |

Two-stream 相对 Serial 的 kernel 累计时间增加 17.03%，其中 `duplicateWithKeys` 从 0.465 增至 1.250 ms/frame，说明两个 stream 的资源竞争抵消了大部分 overlap 收益。Full MLP 则把 Serial 中主要的 GEMM、activation 和 Raster render leaf 收进 4.400 ms/frame 的 `tacker_mix_render_deform_mlp_v5`，使 kernel 累计时间减少 33.65%、GPU kernel 数减少 27.54%。

Graph 版本相对普通 overlap 的 kernel 累计时间反而增加 4.85%，但 host 提交从 87.66 次普通 launch/frame 降为 46.50 次普通 launch + 0.98 次 graph launch，减少 45.84%；GPU 投影跨度缩短 16.70%。这进一步验证了 Graph 的主要收益是减少 host launch/bubble 并让 Raster prepare 与 HexPlane prefix 真正重叠，而不是减少 GPU 算术量。

完整 family 表、每个版本最重的具体 kernel、原始 CSV、`.nsys-rep`、SQLite 和结构化 JSON 见 [五版本 kernel 耗时占比报告](../nsight_reports/flame_steak_versions_kernel_share_node_20260927/README.md)。

## 14. Deform tail Tensor Core 化可行性实验

为降低 Orin mixed kernel 中 Deform warp 对 issue slot、ALU 和 FP32 FMA 的压力，
新增了不覆盖 v1 的 standalone `tacker_deform_mlp_tensor_tail_v3`。它将五个
head final-linear 的 FP32 标量 reduction 改为 FP16 WMMA，并保留 FP32 accumulator。

Jetson AGX Orin 上 111,525 rows 的 5-cycle repeated ABBA 结果为：v1
`12.414621 ms`，v3 `7.599733 ms`，加速 `1.6336×`，时间降低
`38.7840%`。整体 relative L2 误差为 `2.9512e-4`，五个 head 中最大
absolute error 为 `1.0391e-3`。这些是 kernel 输出精度，不是端到端画质指标。

NCU full-set 中，v3 相对 v1 将 executed instructions 从 498.588 M 降到
129.133 M（-74.10%），`duration × issue-active` 从 6.007 ms 降到 1.558 ms
（-74.07%），证明 issue-side 目标已经达到。但 v3 的 L1/TEX throughput 升到
99.41%，shared-load bank conflicts 从 2.529 M 增到 13.187 M，long-scoreboard
stall/issue 从 4.05 增到 21.81。因此它只获得 1.63× 而不是接近 3.86×：
瓶颈已从标量指令发射转移到 shared-memory/L1 供给和同步。

A6000 上同一 v3 的 NCU 加速为 1.920×，v3 L1/TEX 只有 70.96%，这解释了
为什么 Tensor-Core tail 在 A6000 上转化出更大的 wall-time 收益。当前 v3 尚未接入
mixed ABI-v5；下一步应先优化 FP16 activation tile 的 shared-memory bank conflict，再做
mixed-only NCU，不应直接外推端到端 FPS。

完整的精度、ABBA sample、NCU 指标、A6000/Orin 对比和 `.ncu-rep` 索引见
[Jetson Deform Tensor-Core tail 实验报告](../benchmarks/results/jetson_20260927/tensor_tail_v3/analysis.md)。

### 14.1 Tensor-tail shared-memory padding

上述 v3 的主要 shared-load conflict 已用不覆盖旧 symbol 的 v4 实验验证。v4 将
tail WMMA A tile 的 shared-memory leading dimension 从 128 half 改为 136 half，并在
原有 8 KiB hidden buffer 内重新安排 output slab，所以静态 shared memory 仍为
14.592 KB/block。

Jetson 5-cycle repeated ABBA 中，v3 `7.598928 ms`，v4 `7.100757 ms`，v4 额外
加速 `1.07016×`，时间降低 `6.5558%`；五个 head 全部 bitwise identical。
NCU 中 shared-load bank conflicts 从 13.188 M 降到 2.257 M（-82.887%），
shared-load wavefronts 降低 30.304%，eligible warps/scheduler 提高 7.788%，
duration 降低 5.953%。代价是 registers/thread 从 61 增至 64、executed
instructions 增加 1.992%，但 occupancy 仍为 66.3%且 local memory 仍为零。

该结果确认 v4 是更好的 mixed 集成候选，但当前仍是 standalone kernel，没有
改动 mixed ABI-v5，也没有运行端到端测试。完整报告和 NCU 证据见
[Jetson Tensor-tail v4 padding 实验](../benchmarks/results/jetson_20260927/tensor_tail_v4_padding/analysis.md)。

### 14.2 Tensor-tail 接入 mixed ABI-v6

在保留生产 `tacker_mix_render_deform_mlp_v5` 的前提下，新增了实验 symbol
`tacker_mix_render_deform_mlp_tensor_tail_v6`。它保持原 8 Raster warps + 4 Deform
warps 的 384-thread CTA 布局，用 named barrier 驱动 Deform subgroup，并把 v4 的
FP16 WMMA padded tail 接入四个 Deform warps。v6 只在显式设置
`TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V6=1` 时启用，尚未加入生产 ABI 清单。

Jetson 上 111,525 rows 的 mixed-only 数值门槛中，v6 相对 v5 scalar tail 的 overall
relative L2 为 `1.27094e-4`，五个 head 最大 absolute error 的最大值为
`3.77018e-5`。运行时资源显示 registers/thread 从 40 增至 69，使 active blocks/SM
从 4 降至 2、理论 occupancy 从 100% 降至 50%，没有 local-memory spill。

真实 prepared-overlap workload 的 NCU 定向采集中，v6 相对 v5 将 executed
instructions 减少 27.819%、shared-load wavefronts 减少 29.774%，单次 replay duration
从 29.889952 ms 降至 28.576064 ms（-4.396%，1.046x）。但 achieved occupancy 从
83.16% 降至 39.94%，eligible warps/scheduler 从 3.379 降至 1.011。也就是说，
Tensor-tail 的算术减负是真实的，但 69-register CTA 让 Raster 和 Deform 一起失去
一半驻留并发度，抵消了大部分收益。

本轮按要求没有做端到端 FPS 测试；NCU profile runner 的 metadata 时间不作为 FPS
证据。v6 暂不替换生产 v5，下一步应优先把寄存器压到约 56 以下以争取 3 CTA/SM，
再重新做 mixed-only NCU。完整指标、精度和 `.ncu-rep` 见
[Jetson mixed Tensor-tail ABI-v6 实验](../benchmarks/results/jetson_20260928/mixed_tensor_tail_v6/analysis.md)。

### 14.3 降低 mixed Tensor-tail 寄存器压力

在继续保留 v5/v6 的前提下，新增实验 ABI-v7 symbol
`tacker_mix_render_deform_mlp_tensor_tail_low_register_v7`。它把原地 FP32 -> FP16
compaction 从每线程 `values[8]` 改为 `values[4]`，让 mixed device implementation
按阶段读取 shared descriptor，并用 `__launch_bounds__(384, 3)` 约束寄存器目标。

Jetson ptxas 与运行时资源查询均显示 registers/thread 从 v6 的 69 降至 v7 的 56，
没有 stack、spill 或 local memory；active blocks/SM 从 2 增至 3，理论 occupancy
从 50% 增至 75%。v7 与 v6 的五个输出全部 bitwise identical，因此没有新增数值
误差。

同一真实 prepared-overlap workload 的 mixed-only NCU 中，v7 相对 v6：duration
从 28.621344 ms 降至 23.270016 ms（-18.697%，1.22997x）；executed instructions
仅增加 0.039%；achieved occupancy 从 40.01% 增至 57.27%；eligible warps/scheduler
从 1.000 增至 1.802；issue slots busy 从 59.28% 增至 72.90%。这确认 v6 的主要损失
来自寄存器导致的低驻留，恢复第三个 CTA 后 latency hiding 显著改善。

四值分组使 barrier stall/issue 增加 12.57%，但代价远小于 occupancy 收益。本轮仍未
进行端到端 FPS 或图像质量测试，v7 保持 opt-in 实验状态，不替换生产 v5。完整报告
和 `.ncu-rep` 见
[Jetson low-register mixed Tensor-tail ABI-v7](../benchmarks/results/jetson_20260928/mixed_tensor_tail_v7_lowreg/analysis.md)。

### 14.4 恢复 4 CTA/SM 的 mixed Tensor-tail ABI-v8

在保留 v5/v6/v7 的前提下，新增实验 ABI-v8 symbol
`tacker_mix_render_deform_mlp_tensor_tail_four_cta_v8`。v8 使用
`__launch_bounds__(384, 4)`，把 compaction 缩为每线程 2 值并按 8 组顺序执行，
采用 phase-local descriptor load、原生 `mma.sync.m16n8k16` 和 row-space tile loop。
这组修改将 Jetson 上的 registers/thread 从 v7 的 56 降至 40，保持零 stack/spill，
active CTA/SM 从 3 恢复为 4，理论 occupancy 从 75% 恢复为 100%。

111,525 rows 的 mixed-only 精度门禁中，v8 与 v7 的五个 head 全部 bitwise
identical；相对 v5 的 overall relative L2 为 `1.12552e-4`，五个 head 的最大
absolute error 均不超过 `3.56362e-5`。

同一真实 prepared-overlap workload 的 mixed-only NCU 中，v8 相对 v7：duration
从 23.31 ms 降至 21.98 ms（-5.71%，1.061×）；achieved occupancy 从 57.27%
升至 75.08%；eligible warps/scheduler 从 1.81 升至 2.90；issue slots busy 从
72.99% 升至 80.12%。第四个 CTA 的 latency-hiding 收益是真实的，但顺序
compaction 与同步使 executed instructions 增加 3.64%，ALU 和 math-pipe 压力也随
发射率上升，因此收益没有随理论 occupancy 线性增长。

后续完成的正式 10×50 ABBA 中，v5、v7、v8 的中位 FPS 分别为 23.022、27.024、
27.986。v8 相对 v7 提升 3.560%，paired bootstrap 95% CI 为
`[1.03438, 1.03684]`；相对 v5 提升 21.565%。NCU 每次 mixed launch 预计节省
1.321696 ms，按 49/50 帧进入 steady-state 计算，预计每帧节省 1.295262 ms；
端到端实际节省 1.271986 ms，兑现约 98.2%。

50-view 画质中，v8 的 PSNR/SSIM/LPIPS 为 32.574282、0.957735574、
0.140365895，与 v7 完全相同；相对 Serial 的差异为 -0.000151 dB、-7.308e-7、
+5.543e-7，全部通过门禁。30/30 个 FPS child 和三组质量报告均实际执行 `tacker`、
无 fallback。v8 仍为 opt-in 实验版本。完整实现说明、精度、资源、
scheduler/pipeline、端到端结果和 `.ncu-rep` 见
[Jetson four-CTA mixed Tensor-tail ABI-v8](../benchmarks/results/jetson_20260928/mixed_tensor_tail_v8_fourcta/analysis.md)。

### 14.5 A6000 与四场景验证

2026-09-28 在物理 GPU 2 的 RTX A6000（`sm_86`）上完成了 v8 的补充
验证。`sm_86` ptxas 报告 v5/v7/v8 分别为 40/56/40 registers/thread，
三者都是 0 stack、0 spill；运行时理论驻留分别为 4/3/4 CTA/SM。
111,525 rows 数值门禁中，v8 与 v7 的五个输出张量全部 bitwise
identical，v8 相对 v5 的 overall relative L2 为 `1.36549e-4`。

A6000 mixed-only NCU 的 v5/v7/v8 duration 分别为 `5.766080 ms`、
`4.297536 ms`、`4.059552 ms`，v8 相对 v7 加速 `1.05862×`。v8 使
achieved occupancy 从 57.29% 升至 74.68%、eligible warps/scheduler 从
1.720 升至 2.722、issue active 从 69.72% 升至 76.35%。代价是 executed
instructions 增加 3.64%，ALU/math-pipe 压力和 shared-store bank conflict 也上升，
所以性能收益不会与理论 occupancy 线性增长。

`flame_steak`、`coffee_martini`、`cook_spinach` 和 `flame_salmon_1`
每个场景均完成 v5/v7/v8 各10个独立 child 的 10×50 ABBA。端到端中位
FPS 如下：

| 场景 | v5 | v7 | v8 | v8/v7 | v8/v5 |
|---|---:|---:|---:|---:|---:|
| flame_steak | 115.242 | 134.651 | **136.804** | +1.599% | +18.710% |
| coffee_martini | 88.962 | 100.430 | **102.175** | +1.738% | +14.853% |
| cook_spinach | 104.931 | 123.420 | **125.582** | +1.752% | +19.680% |
| flame_salmon_1 | 79.538 | 90.287 | **91.501** | +1.345% | +15.041% |

v8/v7 的四场景等权几何平均提升为 1.608%；每个场景的 10,000 次
paired-round bootstrap 95% CI 下界都大于 1。v8/v5 的四场景几何
平均提升为 17.051%。120/120 个 FPS child 均实际执行 `tacker`、无 fallback。

12/12 份 50-view 画质报告通过 PSNR/SSIM/LPIPS 门限；每个场景的 v8
和 v7 逐-view 三项指标完全相同。因此第 14.4 节要求的 A6000 和多场景
验证已完成，但本轮仍未自动切换生产 profile。完整协议、配对 CI、NCU 指标和
原始证据见
[A6000 mixed Tensor-tail ABI-v8 多场景验证](../benchmarks/results/a6000_20260928/mixed_tensor_tail_v8/analysis.md)。

### 14.6 Raster leaf 优化 ABI-v10/v11

在继续保留 v9 及全部旧 ABI 的前提下，新增 v10 exact Raster adapter 和独立的
v11 fast-math adapter。v10 从 `(round, j)` 推导 contributor index、在 `exp` 前做
保守 alpha cutoff，并把 RGB+depth cooperative-load 到 shared memory；v11 仅额外
启用 `__expf` 和显式 FMA。两者在 Jetson 上均为 40 registers/thread、零 local
memory、4 CTA/SM 和 100% 理论 occupancy。RGB+depth cache 将 static shared
memory 从 v9 的 21,888 B 增至 24,960 B，但没有降低驻留。

真实 `flame_steak` prepared mixed leaf 使用镜像顺序
`v9,v10,v11,v11,v10,v9`，每版两个独立 child；剔除 warmup 后每版 18 个 CUDA
Event 样本。v9/v10/v11 的中位时间分别为 20.7163/17.7469/16.9633 ms。
因此 v10 相对 v9 加速 1.16732×（时间降低 14.3337%）；v11 相对 v9 加速
1.22124×（降低 18.1162%），并相对 v10 再加速 1.04619×。2,581 个 GPU
频率样本均为 1.3005 GHz。

随后完成 Serial、Two-stream、v9、v10、v11 的正式 10×50 端到端 ABBA。中位
FPS 分别为 20.998、21.131、29.029、31.690、32.511。v10 相对 v9 提升
9.169%，paired bootstrap 95% CI 为 `[1.090475, 1.095027]`；v11 相对 v9
提升 11.995%，CI 为 `[1.118598, 1.122530]`；v11 相对 v10 再提升 2.588%，
CI 为 `[1.023639, 1.026934]`。v10/v11 相对 Two-stream 分别提升 49.972% 和
53.854%。50/50 个 child 成功，30/30 个 Tacker child 均为 49 次 mixed launch、
无 fallback，100 个运行前后频率读数均为 1.3005 GHz。

50-view 画质验证中，v10 的逐视角 PSNR/SSIM/LPIPS 与 v9 完全相同；v11
相对 v9 的最大单视角差异仅为 `3.8147e-6 dB`、`1.7881e-7` 和
`8.7917e-7`，全部通过画质、输出顺序和 pipeline-count gate。完整 mixed-leaf
证据见 [Jetson Raster leaf ABI-v9/v10/v11](../benchmarks/results/jetson_20260929/mixed_v9_v10_v11_leaf_timing_v1/analysis.md)，
正式端到端结果见 [Jetson v9/v10/v11 E2E 与基线](../benchmarks/results/jetson_20260929/mixed_v9_v10_v11_e2e_baselines_v1/analysis.md)。

### 14.7 v10/v11 Raster-only、Deform-only、Both 同包络诊断

为分开“时间轴上重叠”和“真正隐藏时间”，为 v10/v11 各新增了
profiling-only 诊断 symbol。Raster-only、Deform-only 和 Both 均保留生产版的
384 threads/block、40 registers/thread、24,960 B static shared memory、3 named
barriers、`__launch_bounds__(384, 4)` 以及 5,440 个物理 CTA。三种模式只通过
运行时开关关闭其中一组 warp，所以避免了“solo kernel 资源包络更小”带来的
不公平对比。ptxas 证实两个诊断 symbol 都是零 stack/spill。

`flame_steak` 上每个版本/模式运行 2 个独立 child，剔除每个 child 的 2 个
warm-up mixed launch 后各保留 18 个测量样本。CUDA Event 均值结果为：

| 版本 | Raster-only | Deform-only | Both | R+D 串行和 | 融合效率 | 相对串行加速 |
|---|---:|---:|---:|---:|---:|---:|
| v10 | 14.7610 ms | 7.6471 ms | 18.3799 ms | 22.4081 ms | 52.68% | 1.2192× |
| v11 | 13.6683 ms | 7.6437 ms | 17.5182 ms | 21.3120 ms | 49.63% | 1.2166× |

融合效率定义为 `(R + D - Both) / min(R, D)`。v10 隐藏了理论可隐藏
7.6471 ms 中的 4.0282 ms；v11 隐藏了 7.6437 ms 中的 3.7938 ms。
因此两版都只兑现了约一半的理想 overlap，Both 相对理想下界仍分别慢
24.52% 和 28.17%。

CTA 内 `clock64()` 显示，Both 中短分支的时间轴重叠率实际高达
99.9651%/99.9639%，但同时活跃时子组本身变慢：v10 的 Raster/Deform
分别膨胀 1.03%/14.69%，v11 分别膨胀 3.85%/11.05%。在 Deform
内部，只处理一个 tile 的 CTA 膨胀 21.98%/17.49%，处理两个 tile 的
CTA 膨胀 5.06%/2.55%。这证明限制因素不是两组 warp 没有同时运行，
而是它们同时运行时共享 issue/ALU/内存供给，使 Deform 尤其是单-tile CTA
的有效吞吐降低，再加上多轮 CTA wave 的全局调度，最终只隐藏约一半的
Deform-only wall time。

v11 相对 v10 将 Raster-only 缩短 7.40%，Deform-only 仅变化 -0.045%，
符合两版 Deform 路径完全相同的预期；Both 因此缩短 4.69%。诊断
kernel 含 `clock64` 写回和 mode 分支，Both 中位时间比先前生产 leaf 高
约 3.63%/3.32%，所以绝对值不取代生产 kernel 时间；同包络的三模式差值才是
本诊断的核心证据。12/12 个 child 都实际执行 `tacker`、无 fallback，
9,877 个运行期频率样本均为 1.3005 GHz。完整结果见
[Jetson v10/v11 同包络子组诊断](../benchmarks/results/jetson_20260929/mixed_v10_v11_subgroup_timing_v2/analysis.md)。
