# gsplat mixed 回退分析与优化

2026-10-21。完成三版 mixed 的对照、资源诊断和 50 帧 Graph 管线验证。

结论：**主要损失来自 mixed 的资源配置及其 MLP 执行成本，同时执行还有额外竞争。**
v3 优化了寄存器上限和只读参数传递，端到端比 v1 快 **5.47%**，但仍比同配置独立调用慢 **9.64%**。
当前速度优选仍是独立 Graph 管线，v3 保留为显式实验入口。

## 环境与可比性

- 按实验室约定，本轮在 **Jetson AGX Orin，GPU 固定 1.3005 GHz** 上运行；PyTorch 2.8.0、CUDA 12.6、sm_87、gsplat 1.5.3。
- flame_steak iteration 14000，111,525 Gaussians、1352×1014；checkpoint 的四个文件与此前 A6000 实验 **SHA256 全部一致**。
- 端到端使用 test views 0–49，双 slot HexPlane CUDA Graph，10 帧预热后 10 轮正/逆序交替；所有候选的前处理、MLP 精度、残差和激活一致。
- 实际 parent qualification 为 `full_mlp_overlap_graph.qualification.json`，selected variant 为 **`full_mlp_overlap_graph_pb0`**，SHA256 为 `a13d87f6c4a22a24b20f83cb3c66a009e759b739b498c9ff55f8c35c9b82d23a`。每条 50 帧管线实际 replay 49 次 Graph，mixed 两版各执行 49 次物理融合，独立对照执行 49 对 leaf，无 fallback。
- 此 profile 验证复用的工作负载及 Graph 前缀，**不代表新 gsplat kernel 已获得原生产 leaf 的 qualification**。本轮单独核验新 leaf 的输出与资源；未混入 eager `full_mlp_overlap` 数据。
- leaf 使用 Raster(view 0)＋MLP(view 1) 的固定真实输入，CUDA Graph 内只捕获计算；每 arm 6 轮，每轮预热 5 次、计 20 次 replay 均值，再取轮间中位数。编译、分配、前缀、前处理不在 leaf 计时内。
- 端到端 wall time 包含 Python 提交、context、HexPlane、MLP、残差/激活、gsplat prepare/leaf、首帧 prefill、末帧 drain 和最终同步；排除读取、编译、预热、画质计算及 I/O。

此前 A6000 的 168.25/172.04 FPS 不能与本表直接相除。**本轮没有在 A6000 上复测优化版**；下述动态诊断与优化收益只对本轮 Orin 条件成立。

## 为什么融合反而变慢

### 1. 资源按整块 CTA 分配，空闲 subgroup 仍占资源

实际编译资源（按 Graph 节点查询，非源码估算）：

| kernel | threads/CTA | registers/thread | shared/CTA | local/thread | 可驻留 CTA/SM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原生 gsplat Raster | 256 | 40 | 7,168 B | 0 B | 6 |
| 原生 ABI-v5 MLP | 128 | 60 | 14,592 B | 0 B | 8 |
| 同 device 计算函数的独立 MLP 诊断版 | 128 | 40 | 14,696 B | 0 B | 10 |
| mixed v1 | 384 | 56 | 21,864 B | 0 B | 3 |
| mixed v2：四 CTA 寄存器约束 | 384 | 40 | 21,864 B | 8 B | 4 |
| mixed v3：再改用 grid-constant 参数 | 384 | 40 | 21,760 B | 16 B | 4 |

v1 同时容纳 256 个 Raster 线程和 128 个 MLP 线程。寄存器数按整块 kernel 的分配结果计算，两组都受此配置约束；一个 subgroup 提前结束后，CTA 的资源仍要等整块结束才能释放。

每个 SM 的理论 MLP 驻留线程容量从原生的 8×128 降到 mixed 的 3×128，Raster 则从 6×256 降到 3×256。这是容量变化，不能直接当作耗时倍率；实际损失由下面的控制实验确认。

| v1 控制 | leaf 中位耗时 |
| --- | ---: |
| 原生 MLP，独立 128-thread CTA | 12.354 ms |
| 同 device 计算函数，独立 128-thread CTA | 13.627 ms |
| 同 mixed 二进制，只启用 MLP、Raster 不执行 | **19.090 ms** |
| 原生 Raster＋原生 MLP 独立串行 | **17.668 ms** |
| 完整 mixed v1 | **21.574 ms** |

**即使完全关掉 Raster 的计算，mixed 配置中的 MLP 也已比原生两核串行的总时间长。** 因此，缺少重叠并非主因；该配置中的 MLP 成本本身已经过高。

独立诊断版与 mixed 共用 device 计算函数，但编译上下文、参数传递和同步方式不同，**不能把两者全部差值量化为纯 occupancy 损失**。原生 MLP 反而比更高驻留率的独立诊断版更快，也说明寄存器越少、理论驻留率越高，并不自动意味着执行越快。

### 2. 同时执行还有额外损失

在**相同 mixed 二进制、相同 CTA 配置**中，分别启用 Raster、MLP、Both，以单独的 `clock64` 仪器化调用比较各 CTA 的 subgroup 执行窗口：v1 的 Raster 窗口中位数增长约 **14.3%**，MLP 增长约 **9.2%**。正式计时关闭仪器化。

这说明动态共同执行确实会拉长各自的窗口。两者仍共享 SM 执行、共享内存通路及缓存等资源；本轮未采 NCU，**不能进一步指定为某一类 scoreboard、带宽或 Tensor Core stall，也不能把窗口增长比例当成整帧耗时比例**。

### 3. 尾部空线程存在，但单独调整 grid 收益很小

逻辑任务数为 5,440 Raster tile＋6,971 MLP tile。v1 默认启动 6,971 个 CTA，最后 1,531 个只执行 MLP，却仍使用 mixed 的资源配置。

| v1 物理 CTA 数 | leaf 中位耗时 |
| --- | ---: |
| 默认 6,971 | 21.574 ms |
| 5,440 | 21.611 ms |
| 3,486 | 21.554 ms |
| 2,720 | 21.506 ms |
| 768 | 22.739 ms |
| 192 | 25.543 ms |

只把额外 MLP tile 折回已有 CTA，不能消除主要损失；过少的 CTA 反而变慢。Raster-only 使用同 grid 的控制也几乎不受那 1,531 个空 CTA 影响（5.68417 vs 5.68403 ms），不支持“多启动几个空 CTA 是主要原因”的解释。

默认独立双流 leaf Graph 为 17.664 ms，独立串行为 17.668 ms，本次配置未观察到净收益；没有使用它替代正式端到端的原匹配对照。

## 为什么旧 v11/v13 仍能加速

384-thread CTA 有资源代价，但不意味着融合必然变慢。旧 v11 已有真实的同计算独立对照：Raster 14.499 ms、MLP 6.599 ms，独立串行实测 21.095 ms，完整 mixed 为 16.619 ms，耗时降低 21.22%。这组组合的重叠收益足以覆盖融合代价。旧 mixed 实际为 40 registers/thread、0 B local memory，也不能把它视为本轮 v1 的 56-register 配置。

还有一个必须对齐的变量：旧 v11/v13 使用 **device-v7 Tensor Core tail MLP**，本轮 gsplat 为保持 ABI-v5 基准采用 **FP32 scalar tail MLP**。本轮每个 gsplat 融合/独立配对内部的计算与精度一致，但与旧 v11/v13 相比，并非只替换了 Raster 后端。Raster 与 MLP 的成本比例也从旧组合的 14.499/6.599 变成约 5.311/12.354 ms；不能将两次实验的全部差异归因于 gsplat。

旧结果为三个独立进程的样本均值，本轮为轮间中位数；这些数据解释组合差异，不能替代统一协议的后端因果对照。进一步应测试 **gsplat＋旧 device-v7 MLP** 的融合与独立 Graph 配对，才能判断迁入既有优化后是否仍有融合收益。

[旧 v11/v13 的同计算独立对照](../benchmarks/results/jetson_20261015/independent_matched_graph/summary.json)。此前 A6000 表中从原始串行到旧 Graph mixed 的 FPS 提升，还同时包含自定义 MLP 与 Graph 调度的影响，不能全部记为物理融合收益。

## 已实现的优化

- **v2**：`__launch_bounds__(384, 4)` 约束编译器，使实际寄存器数从 56 降到 40，可驻留 CTA 从 3 提到 4。该版本存在 8 B/thread local memory，不能只凭驻留率判断成功。
- **v3**：保留四 CTA 约束，使用 `const __grid_constant__` kernel 参数，移除 MLP 参数 descriptor 的共享内存复制和起始 subgroup barrier。MLP 数据运算、FP16 量化边界、FP32 累加与 FP32 scalar tail 均保持原实现。该版本有 16 B/thread local memory，因此也通过实测而非资源表决定保留。
- 独立筛选选中 **v3、5,440 physical CTAs**。全部 Raster tile 和 MLP tile 仍在同一个 mixed kernel 内完成，部分 CTA 通过 grid-stride 执行第二个 MLP tile，无独立 MLP 尾部。

第二轮同批交替测得：

| leaf | 中位耗时 |
| --- | ---: |
| 原生独立串行 | **17.665 ms** |
| mixed v1 默认 | 21.575 ms |
| mixed v2、5,440 CTA | 20.847 ms |
| mixed v3、5,440 CTA | **19.937 ms** |

v3 相对 v1 leaf 耗时降低 **7.59%**，仍未超过原生独立串行。v3 MLP-only 控制为 18.286 ms，主要成本仍未消除。

## 端到端与画质

| 同 checkpoint、50 帧 Graph 管线 | 中位 ms/frame | 中位 FPS |
| --- | ---: | ---: |
| gsplat＋ABI-v5 MLP，独立 leaf | **24.993** | **40.01** |
| mixed v1 默认 | 29.172 | 34.28 |
| mixed v3、5,440 CTA | **27.660** | **36.15** |

- v3 相对 v1：FPS **+5.47%**，wall time **−5.18%**。10 对轮次 FPS 比的中位数为 1.05507，配对轮次 bootstrap 95% 区间为 [1.05435, 1.05529]；这只描述同一进程内的轮次变化，不代表跨机器或跨进程置信区间。
- 相对独立 Graph：v3 的 FPS 仍低 **9.64%**，因此当前建议继续使用独立 Graph 作为速度优选。
- v1、v3 的 **全部 50 帧 RGB 和深度都与独立对照逐位一致，最大误差均为 0**；真实输入 leaf 的 RGB、alpha 和全部五个 MLP 输出也一致。本轮直接检查图像/深度，没有重复计算 PSNR/SSIM/LPIPS；它不消除此前 gsplat 相对旧渲染器的画质差异。
- 8 项 CUDA 测试通过，覆盖三版的奇数尺寸、grid-stride 和 early termination 等。CPU suite 共 82 项，8 项 CUDA 因本地无 GPU 跳过。
- Jetson 的 Compute Sanitizer 报 `GPU debugging features are disabled`，退出码 3；**没有获得本轮 memcheck 通过结论**。未改变设备调试或安全配置。
- 运行前检查空闲和内存，运行期间频率始终为 1.3005 GHz、无外部 Python 工作负载；源码/checkpoint 前后 hash 一致。正式运行源码快照已归档，复用的 `tacker_pipeline.py`、`hexplane.py` 以该快照为准。

## 使用与记录

默认渲染后端及默认 mixed v1 入口未切换。显式使用实验候选：

```python
renderer = GsplatMixedRenderer(
    pc, pipe, background, qualification_profile,
    kernel_variant="v3", physical_blocks=5440,
)
```

`5440` 来自本工作负载的筛选，不是适合任意场景的通用设置。v3 使用独立 CUDA 源文件和独立 JIT 模块，保留 v1/v2 二进制作为对照。

远端隔离目录为 `/data/qyfeng/gsplat-mixed-opt-20261021`，所有新代码、缓存、编译产物和结果均在数据盘。启动与复核脚本：

- [分析/Graph benchmark](../scripts/analyze_gsplat_mixed.py)
- [Jetson 启动脚本](../scripts/run_jetson_gsplat_mixed_opt.sh)
- [v3 kernel](../gsplat_mixed_ext/mixed_grid_constant.cu)
- [CUDA 测试](../tests/test_gsplat_mixed_cuda.py)

原始结果：

- [汇总](../benchmarks/results/gsplat_mixed_opt_20261021/summary.json)
- [第一轮资源控制与完整 CTA 扫描](../benchmarks/results/gsplat_mixed_opt_20261021/results/sweep/report.json)
- [第二轮 v3 配对验证](../benchmarks/results/gsplat_mixed_opt_20261021/results/sweep_grid/report.json)
- [正式 50 帧与源码快照目录](../benchmarks/results/gsplat_mixed_opt_20261021/results/formal/report.json)
- [CUDA 测试日志](../benchmarks/results/gsplat_mixed_opt_20261021/cuda_tests_v3.log) · [memcheck 不可用日志](../benchmarks/results/gsplat_mixed_opt_20261021/memcheck.log)

进一步若要求反超独立调用，应优先重新设计 MLP subgroup 的资源/任务比例和参数访问，或研究按工作负载分段融合；单纯压缩 grid 已不足以解决。上述方向尚未验证，不能作为已获得的收益。
