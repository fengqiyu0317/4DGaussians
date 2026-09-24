# A6000 上原版 deform 与 `deform_mlp_device_v1` 的真实渲染上下文性能分析

日期：2026-09-23

## 结论摘要

在本次测试的 `flame_steak` 模型上，4DGaussians 原版完整 deform 的耗时约为 **7.65 ms/帧**。将其中的 backbone 和五个 PyTorch MLP head 替换为 `deform_mlp_device_v1` 后，在真实串行推理流程中，完整 deform 的耗时下降到约 **4.99 ms/帧**。

对融合 kernel 的内部阶段插桩进一步表明，所有 Tail 合计约占 **79.94%**；其中 48 维 SH Tail 单独占 **66.90% / 1.900 ms**，是当前实现最主要的瓶颈。

测试采用的真实执行顺序为：

```text
deform → activation → rasterize → 下一帧 deform → ...
```

CUDA Event 只包围各表格中明确标注的部分。因此，虽然相邻两次 deform 之间实际执行了 activation 和 rasterize，但报告中的 deform 耗时并不包含这两部分。

## 测试环境与工作负载

- GPU：NVIDIA RTX A6000，SM 8.6（物理 GPU 3）
- PyTorch：2.4.1+cu124
- PyTorch 报告的 CUDA 版本：12.4
- 数据集：N3DV `flame_steak`
- 模型 checkpoint：fine iteration 14,000
- Gaussian 数量：111,525
- 每轮测试视角数：50 个真实 test camera
- 预热：20 帧
- 重复次数：7 轮，共测量 350 帧
- 主要统计口径：7 个“每轮平均值”的中位数
- 原版执行入口：4DGaussians 串行 `render()` 路径
- 融合 kernel 启动配置：6,971 blocks，每 16 行对应一个逻辑 block
- 融合 kernel 输出：五个由调用方预分配的 FP32 buffer

每次 GPU 实验前都重新检查了 GPU 使用情况。每次检查时，GPU 3 仅使用 4 MiB 显存，GPU utilization 为 0。测试进程仅通过 `LD_PRELOAD` 使用与已加载内核模块匹配的 NVIDIA 570 版本运行库，没有修改系统驱动软链接或软件包。

## 原版 deform：总耗时

原版 deform 在最小插桩条件下的结果如下：

| 统计项 | 耗时（ms/帧） |
|---|---:|
| 每轮平均值的中位数 | **7.647726** |
| 单次调用中位数 | 7.641088 |
| 平均值 | 7.645285 |
| P25 | 7.632928 |
| P75 | 7.648256 |
| 最小值 | 7.604224 |
| 最大值 | 7.924576 |

详细模块插桩会使外层 deform 耗时增加约 1.64%。因此，下面的各部分占比来自详细插桩结果，但耗时统一缩放回最小插桩得到的 7.647726 ms 总耗时。

## 原版 deform：各部分耗时

| 部分 | 耗时（ms/帧） | 占比 |
|---|---:|---:|
| point/scale/rotation positional encoding | 0.342084 | 4.47% |
| HexPlane 特征查询 | 1.388302 | 18.15% |
| HexPlane 输出到 backbone 的单输入拼接 | 0.057227 | 0.75% |
| Backbone `Linear(32,128)` | 0.170325 | 2.23% |
| Position head 与 residual | 1.057053 | 13.82% |
| Scale head 与 residual | 1.071336 | 14.01% |
| Rotation head 与 residual | 1.054723 | 13.79% |
| Opacity head 与 residual | 0.986677 | 12.90% |
| SH head 与 residual | 1.514724 | 19.81% |
| Mask 构造和 wrapper 边界 | 0.005275 | 0.07% |
| **合计** | **7.647726** | **100.00%** |

五个 head module 合计耗时 5.426993 ms，相关 residual 更新另需 0.257519 ms。整个 head 路径共计 **5.684513 ms/帧**，占完整原版 deform 的 **74.33%**。

### 五个 head 的内部耗时

下表均为缩放后的估算值，单位为 ms/帧。`层间间隔` 表示 head 内连续模块之间由 CUDA Event 测得的小段间隔。

| Head | ReLU 0 | `Linear(128,128)` | ReLU 1 | 输出 Linear | 层间间隔 | Residual | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Position | 0.176992 | 0.537860 | 0.164210 | 0.150444 | 0.007390 | 0.020157 | 1.057053 |
| Scale | 0.165193 | 0.560377 | 0.166518 | 0.153181 | 0.006917 | 0.019149 | 1.071336 |
| Rotation | 0.165693 | 0.544991 | 0.166691 | 0.153715 | 0.006987 | 0.016645 | 1.054723 |
| Opacity | 0.164597 | 0.555640 | 0.166592 | 0.084792 | 0.005985 | 0.009071 | 0.986677 |
| SH | 0.166028 | 0.566097 | 0.167029 | 0.416071 | 0.007001 | 0.192498 | 1.514724 |

将五个 head 按操作类型聚合后：

| 操作组 | 耗时（ms/帧） | 占完整 deform 的比例 |
|---|---:|---:|
| 五个 `Linear(128,128)` | 2.764965 | 36.15% |
| 十个 ReLU | 1.669544 | 21.83% |
| 五个输出 Linear | 0.958204 | 12.53% |
| 层间间隔 | 0.034281 | 0.45% |
| Residual 更新 | 0.257519 | 3.37% |

SH head 是耗时最高的单个 head，原因是它的输出层需要为每个 Gaussian 生成 48 个值，而且 residual 需要更新完整的 `[N,16,3]` SH 张量。

### Positional encoding

将详细插桩结果缩放回粗粒度 positional encoding 总耗时后：

| Encoding | 耗时（ms/帧） | encoding 内部占比 |
|---|---:|---:|
| Point | 0.220629 | 64.50% |
| Scale | 0.054977 | 16.07% |
| Rotation | 0.066478 | 19.43% |
| **合计** | **0.342084** | **100.00%** |

### HexPlane 补充说明

在 CUDA Event 分解中，HexPlane 共占用 1.388302 ms/帧。针对同一工作负载的另一份 Nsight Systems 结果显示，12 次 `grid_sample` CUDA kernel 合计约为 0.246498 ms/帧。

这两个数来自不同的 profiling 方法，不能当作严格的同一次事件分解直接相减。不过结果表明，HexPlane 的主要开销并不只是 `grid_sample` kernel，还包括坐标归一化、索引、平面特征相乘、多尺度拼接以及大量小算子的调度开销。

## 真实渲染序列中的融合 MLP

对比实验首先在同一次运行中重新测量原版 forward，随后临时将其替换为融合版 forward。融合版保留了相同的 positional encoding、HexPlane 查询、mask、residual 更新、renderer activation 和 rasterizer；只有原来的 backbone 和五个 PyTorch head 被替换为单次 `tacker_deform_mlp_delta_v1` kernel 启动，其 device 函数主体为 `deform_mlp_device_v1`。

| 测量区间 | 每轮平均值的中位数（ms） | 单次调用中位数（ms） | P25–P75（ms） |
|---|---:|---:|---:|
| 同一次对比实验中的原版完整 deform | **7.664655** | 7.659520 | 7.652352–7.666688 |
| 融合后的完整 deform | **4.991628** | 4.988736 | 4.980736–5.000192 |
| 仅融合 MLP kernel | **2.985140** | 2.981888 | 2.969600–2.990080 |

同一次运行中重测的原版结果与前面详细分解使用的原版基准仅相差 0.016929 ms，约为 0.22%。

### 直接对比

| 对比项 | 结果 |
|---|---:|
| 完整 deform 节省的时间 | **2.673027 ms/帧** |
| 完整 deform 延迟降幅 | **34.87%** |
| 完整 deform 加速比 | **1.54×** |
| 融合 kernel 占完整 fused deform 的比例 | **59.80%** |

融合版完整 deform 中除 kernel 之外的约 2.006488 ms，来自 positional encoding、HexPlane 查询、特征准备、mask/residual 操作和 wrapper 开销。

## `deform_mlp_device_v1` 的 kernel 内部阶段分解

下面的结果来自 stride-136 版本在 standalone 模式下的 kernel 内部插桩。标准负载仍为 111,525 行，启动配置为 6,971 blocks × 128 threads。未插桩时，`deform_mlp_device_v1` 总时间中位数为 **2.8396 ms**，与仓库历史结果 **2.8330 ms** 基本一致。

每个 CTA 处理 16 行。表中的“单 CTA 中位延迟”来自 CTA 内部阶段边界的 `clock64` 读数；“折算到总 kernel”则按各阶段占比，将未插桩的 2.8396 ms 总时间进行等效分摊。

| 阶段 | 单个 16 行 CTA 中位延迟（µs） | 阶段占比 | 折算到总 kernel（ms） |
|---|---:|---:|---:|
| 输入搬入 shared memory | 0.534 | 0.25% | 0.007 |
| Backbone Linear | 19.343 | 9.00% | 0.255 |
| Backbone ReLU 与 FP16 转换 | 1.218 | 0.57% | 0.016 |
| Head 0 Linear | 3.173 | 1.48% | 0.042 |
| Head 0 ReLU | 0.977 | 0.45% | 0.013 |
| Head 0 Tail（位置，3 维） | 9.874 | 4.59% | 0.130 |
| Head 1 Linear | 4.040 | 1.88% | 0.053 |
| Head 1 ReLU | 0.559 | 0.26% | 0.007 |
| Head 1 Tail（缩放，3 维） | 6.610 | 3.07% | 0.087 |
| Head 2 Linear | 3.970 | 1.85% | 0.052 |
| Head 2 ReLU | 0.428 | 0.20% | 0.006 |
| Head 2 Tail（旋转，4 维） | 9.252 | 4.30% | 0.122 |
| Head 3 Linear | 3.947 | 1.84% | 0.052 |
| Head 3 ReLU | 0.737 | 0.34% | 0.010 |
| Head 3 Tail（透明度，1 维） | 2.299 | 1.07% | 0.030 |
| Head 4 Linear | 3.211 | 1.49% | 0.042 |
| Head 4 ReLU | 1.005 | 0.47% | 0.013 |
| Head 4 Tail（SH，48 维） | 143.868 | 66.90% | 1.900 |

### Head 第一层的 WMMA 权重加载路径

五个 head 的第一层权重连续存放为 FP16 `first_weights[5,128,128]`，单个 head 内仍保持 PyTorch Linear 的 `[out,in]` row-major 布局。权重没有先整体搬入 shared memory，而是由每个 warp 执行 `wmma::load_matrix_sync`，将当前 `16×16` 的 FP16 权重 tile 从 global memory 经由硬件 cache 层级加载到分布在 32 个 lane 寄存器中的 WMMA fragment：

```text
global FP16 weight → L2/L1 cache → warp WMMA fragment 寄存器
                                      → Tensor Core MMA
                                      → FP32 accumulator fragment
```

对于当前输出列起点 `col` 和 K 维起点 `k`，加载起始地址为 `weight + col * 128 + k`，leading dimension 为 128。虽然物理存储是 `W[out,in]` row-major，B fragment 被声明为 `col_major`，因此 WMMA 观察到的逻辑元素为：

```text
B[i,j] = W[out = col+j, in = k+i]
```

这恰好对应 `A @ W.T`，不需要在 kernel 中显式转置权重。一个 `16×16` FP16 B fragment 为 512 B；计算一个 `16×16` 输出块需沿 128 维 K 轴加载 8 个 fragment。四个 warp 各负责两个不重叠的输出列块，所以每个 CTA 对每个 head 逻辑上读取完整的 32 KiB 权重，五个 head 合计 160 KiB。

同一 CTA 内各 warp 的权重区间互不重叠，因此主要的权重复用发生在不同 CTA 之间：每个 CTA 处理不同的 16 行 Gaussian，但使用同一套 160 KiB 权重。后续 CTA 能否从 L1/L2 命中取决于当时的硬件缓存状态，不是源码显式保证的复用。相比之下，A fragment 所使用的 backbone activation 已在 shared memory 中，各输出列块对其的重复加载不会再产生 global-memory 读取。

### kernel 内部的主要瓶颈

| 操作组 | 占比 | 等效时间（ms） |
|---|---:|---:|
| 所有 Tail | **79.94%** | **2.270** |
| 其中：SH Tail | **66.90%** | **1.900** |
| Backbone Linear、ReLU 与 FP16 转换 | 9.56% | 0.272 |
| 五个 head 第一层 WMMA Linear | 8.53% | 0.242 |
| 所有 ReLU | 2.29% | 0.065 |

最主要的结论是：融合 kernel 已经不再受五个 head 第一层 Linear 主导，所有 Tail 合计占约 79.94%。其中，仅 48 维 SH Tail 就占 **66.90% / 1.900 ms**，是当前 `deform_mlp_device_v1` 的绝对瓶颈。

### 插桩方法与解释限制

- 计时在现有 barrier 边界读取 `clock64`，并使用 CTA 的全局时间进行校准。
- 每个阶段收集了 48,797 个样本。
- 阶段插桩使整个 kernel 从约 2.84 ms 增加到约 3.21 ms。
- 因此，表中的“折算到总 kernel”是按照阶段占比分摊到未插桩总时间的等效值，不是能够分别独立测量并串行相加的子 kernel 墙钟时间。
- Full-grid 配置下，不同 CTA 会在 GPU 上并行和重叠执行；单 CTA 阶段延迟不等于整个 grid 在该阶段独占 GPU 的持续时间。
- 该阶段占比来自 standalone 插桩实验。真实渲染上下文中的 kernel 总时间为 2.985140 ms，但不能在没有重新插桩的情况下假定其内部比例完全不变。

## 实验性 sliced v2 head kernel 的当前数据流

本节记录 `deform_mlp_sliced_v2` 当前实验实现的数据流。它与前文真实渲染对比中使用的单 kernel `deform_mlp_device_v1` 不同，目前也没有替换报告 headline 所对应的渲染路径。记录该实现是为了明确纵向切片、五个 head stream、全局中间量和最终 slice 累加之间的关系。

### 总体数据流与 stream 依赖

```text
features FP32 [N,32]
        │
        ▼
backbone kernel（调用方 current stream）
        │
        ▼
activated FP16 [N,128]（global memory）
        │
        ├── head0 kernel（独立 pooled stream）
        ├── head1 kernel（独立 pooled stream）
        ├── head2 kernel（独立 pooled stream）
        ├── head3 kernel（独立 pooled stream）
        └── head4 kernel（独立 pooled stream）
                         │
                         ▼
              partials FP32 [8,N,59]
                         │
                         ▼
            sum(dim=0) + tail_bias [59]
                         │
                         ▼
                  packed FP32 [N,59]
```

backbone 完成后在 current stream 上记录一个 CUDA event。五条 head stream 均先等待该 event，保证能够观察到完整的 `activated`。每个 head 完成后记录自己的完成 event；调用方 current stream 等待全部五个 event，随后才允许执行 `partials.sum(dim=0)`。这些依赖均为 stream/event 排队，不包含 CPU 侧的 `cudaDeviceSynchronize`。

### Backbone 的数据路径

backbone 输入为 FP32 `features[N,32]`、`backbone_weight[32,128]` 和 `backbone_bias[128]`。grid 的 X 维为 `ceil(N/128)`；一个 CTA 负责最多 128 行，内部串行遍历最多八个 16-row micro-tile：

```text
global features FP32 [16,32]
        │
        ▼
shared features FP32 [16,32]
        │
        ▼
FP32 FMA： [16,32] × [32,128]
        │
        ▼
bias + ReLU + FP16 转换
        │
        ▼
global activated FP16 [N,128]
```

`activated` 必须在 global memory 中物化，因为后续 head 位于不同 kernel 和不同 CUDA stream。backbone CTA 的 shared memory 在该 CTA 结束后不可由其他 CTA 或其他 kernel 访问。

### 五个 head 的模板特化与输出区间

五个 global kernel 共用 `run_sliced_head_v2<Head,Width,Offset>` 模板。模板参数均为编译期常量，因此最终生成五个独立的特化 kernel，不需要在设备端运行时判断 head 类型：

| Head | 语义 | 模板参数 | packed 输出列 | 每 block 线程数 |
|---|---|---|---|---:|
| head0 | position | `<0,3,0>` | `[0,3)` | 32 |
| head1 | scale | `<1,3,3>` | `[3,6)` | 32 |
| head2 | rotation | `<2,4,6>` | `[6,10)` | 32 |
| head3 | opacity | `<3,1,10>` | `[10,11)` | 32 |
| head4 | SH | `<4,48,11>` | `[11,59)` | 128 |

每个 head 的 grid 均为：

```text
grid.x = ceil(N / 128)   // 128-row group
grid.y = 8               // 16-channel hidden slice
```

所以 `(blockIdx.x, blockIdx.y)` 唯一确定“哪一组 128 行”和“哪一个 16 维 hidden slice”。每个 CTA 再把自己的 128 行串行拆成最多八个 16-row micro-tile。

### 单个 head block 内部的数据流

对每个 16-row micro-tile，执行顺序如下：

```text
global activated FP16 [16,128]
        │
        ▼
shared activated FP16 [16,136]（128 个有效元素 + 8 个 padding）
        │
        ▼
WMMA： [16,128] × first_weight_slice.T [128,16]
        │
        ▼
WMMA FP32 accumulator fragment（寄存器）
        │
        ▼
shared hidden FP32 [16,16]
        │
        ▼
first_bias + ReLU
        │
        ▼
FP32 FMA： [16,16] × tail_weight_slice.T [16,Width]
        │
        ▼
线程私有 FP32 accumulator（寄存器）
        │
        ▼
global partials[slice,row,Offset:Offset+Width]
```

具体分为四步：

1. CTA 将当前 16 行的完整 128 维 activation 搬入 padded shared memory。head0--3 使用一个 warp 搬运；head4 使用四个 warp 搬运。
2. 第一个 warp 沿 K=128 依次执行八次 `16×16×16` WMMA，计算当前 head、当前 slice 的 `[16,16]` hidden。head4 的其余三个 warp 在该阶段不参与 WMMA。
3. 第一个 warp 对 shared hidden 加当前 16 维 `first_bias` 并执行 ReLU。
4. 所有可用线程用普通 FP32 `fmaf` 计算当前 slice 对最终输出的 tail contribution，然后写入自己的 partial plane。该阶段不使用 WMMA。

第一层权重逻辑上选择当前 head 的一个输出通道 slab：

```text
first_weights[head, slice*16:(slice+1)*16, 0:128]
```

tail 阶段选择相同的 hidden slice：

```text
tail_weights[Offset:Offset+Width, slice*16:(slice+1)*16]
```

### Partial 布局与最终汇合

不同 head 写入 `[8,N,59]` 的不同输出列，不同 slice 写入不同的第一维 plane：

```text
head0 → partials[slice,:, 0:3]
head1 → partials[slice,:, 3:6]
head2 → partials[slice,:, 6:10]
head3 → partials[slice,:,10:11]
head4 → partials[slice,:,11:59]
```

因此 head 之间和 slice 之间不存在写地址冲突，不需要 atomic。head kernel 内部不进行跨 slice 累加，也不添加 `tail_bias`。五个 head 全部完成并汇合到 current stream 后，调用方执行：

```python
packed = partials.sum(dim=0) + tail_bias
```

该操作将八个 16-channel slice contribution 合并为最终 FP32 `[N,59]`，再按 `[0,3,6,10,11,59]` 切回 position、scale、rotation、opacity 和 SH。

### Activation 的复用边界

当前切分的是第一层 GEMM 的输出通道，而不是输入 K 维。计算任意一个 16 维输出 slice 仍然需要完整的 128 维 activation：

```text
X[16,128] × W_slice.T[128,16] → hidden_slice[16,16]
```

同一 row group 对应五个 head、每个 head 八个 slice block，所以源码层面会发生 `5×8=40` 次对同一 activation tile 的逻辑加载。CUDA kernel 参数只传递 global memory 指针；一个 CTA 不能直接读取另一个 CTA 的 shared memory。因此，一个 slice block 将 activation 搬入自己的 shared memory 后，不能把该 shared 数据直接交给其余七个 slice block。

“逻辑加载 40 次”不等价于 40 次都访问 DRAM。后续读取可能命中 L2，甚至在调度到同一 SM 时受益于 cache 状态，但每个 CTA 仍会独立发出 load 指令、占用 L1/TEX 管线，并重新填充自己的 shared memory。若要在源码结构上只加载一次，需要让同一 CTA 在加载 activation 后处理多个或全部 slice，例如由四个 warp 分两轮处理八个 slice；这将改变当前“一 slice 一个 block”的并行划分。

## 单独连续运行与真实渲染上下文的差别

此前的 standalone benchmark 会连续调用融合 kernel，相邻调用之间没有 rasterization，测得 2.839593 ms/次。在真实串行渲染序列中，同一个 full-grid kernel 的耗时为 2.985140 ms/帧。

| 执行上下文 | 融合 kernel 耗时（ms） |
|---|---:|
| Standalone 连续调用 | 2.839593 |
| 真实 `deform → activation → rasterize` 序列 | **2.985140** |
| 真实渲染上下文增加的耗时 | **0.145547 ms / 5.13%** |

因此，在普通串行推理中，**2.985140 ms** 是更有代表性的融合 kernel 数据。2.839593 ms 仍可用于衡量 kernel 在连续热启动条件下的性能，但会略微高估实际集成后的收益。

## 结果解释与限制

1. 公平的完整 deform 对比是 7.664655 ms 对 4.991628 ms。不能直接拿 2.985140 ms 的 kernel-only 数据与原版完整 deform 对比，因为该数据不包含 positional encoding、HexPlane 和 residual。
2. Activation 和 rasterization 确实在实验中执行，但位于 CUDA Event 计时区间之外。因此，本报告没有给出完整帧耗时或最终 FPS 提升。
3. 融合 kernel 是显式的混合精度推理路径：共享 backbone activation 和 head 第一层权重在约定边界使用 FP16，累加与输出使用 FP32。它通过误差容限验证，并不保证与全 FP32 PyTorch 路径逐 bit 相同。
4. 原版各子模块的详细数值使用嵌套 CUDA Event 并经过缩放；原版和融合版的 headline 总耗时则只在每次完整 deform 外层放置一对 CUDA Event。

## 相关源码位置

- 原版网络与 residual 路径：`scene/deformation.py`
- HexPlane 查询：`scene/hexplane.py`
- 融合 Python API：`tacker_ext/tacker_4dgs_head/__init__.py`
- 融合 kernel 启动：`tacker_ext/csrc/deform_mlp.cu`
- Device 实现：`tacker_ext/include/deform_mlp_device.cuh`
- 融合 ABI 清单：`tacker_ext/abi/deform_mlp_v1.json`

本次 profiling 使用的 driver 脚本均为临时文件，结果记录完成后已经删除。测试没有修改正式的 deformation 或 renderer 源码。
