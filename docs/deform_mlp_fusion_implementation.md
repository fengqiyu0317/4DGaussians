# Deform MLP 融合实现方案

更新日期：2026-09-16。状态：**已实现 standalone full-MLP 推理 kernel 和 Python 接口，并通过 A6000 GPU 测试**。当前输出五组 delta；residual/activation、渲染接入和 Tacker mixed 仍为后续设计，尚未测量本实现的渲染画质和性能收益。

已实现入口：

- [设备主函数 `deform_mlp_device_v1`](/home/huawei/大二春/实验室/4DGaussians/tacker_ext/include/deform_mlp_device.cuh:139)：一个函数内依次调用五个阶段，head 第一层使用 WMMA。
- [CUDA kernel 与 host 检查](/home/huawei/大二春/实验室/4DGaussians/tacker_ext/csrc/deform_mlp.cu)：一次 launch，支持当前 stream、空输入及调用者提供的输出缓冲。
- [Python 调用示例](/home/huawei/大二春/实验室/4DGaussians/tacker_ext/README.md)：参数打包及 `deform_mlp_delta_v1` / `deform_mlp_delta_out_v1`。
- [GPU 测试](/home/huawei/大二春/实验室/4DGaussians/tacker_ext/tests/test_deform_mlp_cuda.py)：7 项通过，包含单 kernel profiler 断言；memcheck、racecheck、synccheck 均通过。具体范围见第 11 节。

阅读导航：[数学与语义](#3-数学结构与不可改变的语义) · [参数与接口](#5-参数布局与-api) · [CUDA 调度](#6-cuda-block-的具体执行过程) · [资源预算](#7-存储预算kernel-骨架与调优方向) · [Tacker 接入](#9-接入渲染和-tacker) · [验证计划](#11-验证与测量计划)。

## 1. 方案摘要与适用范围

将 HexPlane 采样后的共享 backbone 和五个输出 head 融合成一个推理 kernel。每个 CUDA block 负责一批 Gaussian，完整计算这些 Gaussian 的隐藏通道；多个 block 沿 Gaussian 维度并行。

首个完整 MLP 版本采用：

| 项目 | 首版选择 | 原因 |
|---|---|---|
| 平台 | 远端 `4A6000` 的单张 RTX A6000，`sm_86` | 遵循项目计算资源约定；单卡即可完成实验 |
| 融合边界 | HexPlane 特征 → backbone → 五个 head → 五组 delta | 首先单独验证 MLP；residual 和渲染激活分阶段加入 |
| Gaussian tile | 每 block 16 个 Gaussian | 对应 WMMA 的 16 行矩阵 tile |
| 线程 | 128 threads，即 4 warps | 一个 worker group 拥有完整 128 维隐藏通道 |
| head 调度 | 五个 head 依次执行，复用中间缓冲 | 限制共享内存和寄存器占用 |
| backbone | FP32 输入、权重和累加 | 控制新增量化误差 |
| head 第一层 | FP16 输入/权重，WMMA，FP32 累加 | 延续现有 C3/C4 的量化边界，并利用 Tensor Core |
| head 尾层 | FP32，warp 协作点积 | 保留原尾层精度，适配 1/3/4/48 维输出 |
| 输出 | 五个独立、连续的 FP32 tensor | 直接适配后续属性处理，避免拆分 packed 输出的复制 |

这里的“融合”保留所有 Linear、bias 和 ReLU 的数学语义。它减少 kernel 启动和中间 tensor 的显存读写；不会把带 ReLU 的网络化简成一个 Linear，也不意味着整帧只启动一个 CUDA block。

### 1.1 首版支持条件

以当前 DyNeRF 配置为起点，并在加载真实 checkpoint 后检查实际模块和 tensor：

- `no_grid=False`、`grid_pe=0`，HexPlane 输出维度 `F=32`。
- `net_width=128`；`feature_out` 恰好只有一个 `Linear(32,128)`。
- 五个 head 均启用：`no_dx/no_ds/no_dr/no_do/no_dshs=False`。
- `static_mlp=False`、`empty_voxel=False`、`apply_rotation=False`。
- SH 属性为 `[N,16,3]`，五个 head 的尾层输出依次为 `3,3,4,1,48`。
- fine 阶段推理，在关闭 autograd 的上下文执行；模型参数在 renderer 生命周期内固定。
- 接入渲染时使用普通 scales/rotations 路径。首版对 `compute_cov3D_python=True` 或 `convert_SHs_python=True` 使用现有回退路径，避免额外分支混入首次验证。

`defor_depth=0` 不代表没有 backbone：`create_net()` 总会创建第一个 Linear。通用默认配置、HyperNeRF 配置和不同 checkpoint 可能具有不同输入维度或启用的 head；不应只凭配置名称放行。

禁用 head、多层 backbone、动态 mask 等情况后续增加专门变体。首版在 dispatch 前发现不支持的条件，就选择现有渲染实现。

## 2. 源码依据与已有实现

| 已有位置 | 需要核对的行为 |
|---|---|
| [DyNeRF 配置](/home/huawei/大二春/实验室/4DGaussians/arguments/dynerf/default.py) | 每尺度 16 维、两个尺度、宽度 128、五个 head 启用 |
| [deformation.py](/home/huawei/大二春/实验室/4DGaussians/scene/deformation.py) 的 `create_net/query_time/forward_dynamic` | backbone、head、mask、residual 的真实顺序 |
| [hexplane.py](/home/huawei/大二春/实验室/4DGaussians/scene/hexplane.py) 的 `interpolate_ms_features` | 每尺度的六个平面特征相乘，不同尺度拼接 |
| [渲染入口](/home/huawei/大二春/实验室/4DGaussians/gaussian_renderer/__init__.py) 的 `deform_for_render` | deform 输出后再执行 scale/rotation/opacity 激活 |
| [gaussian_model.py](/home/huawei/大二春/实验室/4DGaussians/scene/gaussian_model.py) 的 `setup_functions` | scale 用 `exp`，opacity 用 `sigmoid`，rotation 用 `F.normalize` |
| [Tacker 分区](/home/huawei/大二春/实验室/4DGaussians/gaussian_renderer/tacker_pipeline.py) | C3/C4 参数缓存、prefix/suffix、两槽事件与生命周期 |
| [现有 WMMA 第一层](/home/huawei/大二春/实验室/4DGaussians/tacker_ext/include/head_linear_device.cuh) | 每个逻辑 block 输出 `16×64`，FP16 乘法、FP32 累加 |
| [现有 whole-head adapter](/home/huawei/大二春/实验室/4DGaussians/tacker_ext/include/head_linear_v2_device.cuh) | `Linear→ReLU→tail Linear`；逐行标量 FMA，512 B scratch/group |

### 2.1 C3/C4 给出的经验

在封存的 Phase 3.1 formal 实验中，五 head 第一层 C3 packed 候选为 `100.9417 FPS`；入围的 C4 单 opacity whole-head 代表为 `44.6504 FPS`。这是同一历史评测中的不同候选，并非相同融合范围下的算法对照，不能由此推出全 MLP 融合必然慢。[历史结果与范围说明](/home/huawei/大二春/实验室/4DGaussians/tacker_profiles/baselines/a6000_phase31_20260913/README.md)

从源码可以确认：现有 C4 的第一层使用标量 `fmaf`，并且尾层只有 `tail_features` 个线程实际输出。这些是新实现应重点改善的机制；它们各自对历史耗时的贡献还需要 profiling 确认。

另外，现有 C3 的 packed 布局和单次 launch 不等于把五个第一层实现成了一次 `128→640` 的通用 GEMM，也不保证输入只从物理显存读取一次。本文把数学拼接和实际 CUDA 调度分别描述。

## 3. 数学结构与不可改变的语义

对一个 Gaussian，使用列向量记法：

```text
f ∈ R^32
h = W0 f + b0                              W0: [128,32]
a = ReLU(h)
zi = ReLU(Wi a + bi)                        Wi: [128,128]
Δi = Vi zi + ci                            Vi: [Oi,128]
Oi = 3, 3, 4, 1, 48
```

五个 head 的首个 ReLU 输入相同，因此共享 `a` 即可。两个带 ReLU 的 Linear 通常不能预乘权重：例如 `V·ReLU(Wa+b)` 不能对所有输入写成一个固定仿射变换。

### 3.1 横向拼接与分组尾层

按输出通道堆叠五个 `Wi` 得到 `[640,128]`，可以表示一次 `128→640` 运算；得到的五段 128 维激活再分别送入各自的 `Vi`。

在数学表达上，尾层相当于 `[59,640]` 的块对角矩阵。实际应使用分组运算：

```text
五个独立尾层的 MAC 数：128 × (3+3+4+1+48) = 7,552
稠密 640→59 的 MAC 数：640 × 59 = 37,760
```

直接把块对角矩阵补零后交给稠密 GEMM，会执行约 5 倍尾层乘加。本方案依次完成每个 head，既保留上述数学结构，又不产生整个 `[N,640]` 中间 tensor。

每个 Gaussian 的 Linear MAC 总数为：

```text
32×128 + 5×128×128 + 128×59 = 93,568 MAC
```

按一次乘加算 2 FLOPs，111,525 个 Gaussian 约为 20.87 GFLOPs。这里未计 bias、ReLU、采样和渲染；该算术量不能直接换算成预计运行时间。

### 3.2 residual 与渲染激活

设输入的原始属性为 `x, s, q, o, sh`，mask 为 `m`。原始代码的启用分支是：

```text
x_raw  = x  * m + Δpos
s_raw  = s  * m + Δscales
q_raw  = q      + Δrotations               # apply_rotation=False；此处没有乘 mask
o_raw  = o  * m + Δopacity
sh_raw = sh * m + reshape(Δshs, [16,3])
```

首版 `m=1`。注意 mask 乘在原始属性上，不是乘在 delta 上。head 禁用时直接保留对应原始属性，也不能仍然执行上述带 mask 的公式。

随后 renderer 执行：

```text
scale_out   = exp(s_raw)
rotation_out = normalize(q_raw, p=2, dim=1, eps=1e-12)
opacity_out = sigmoid(o_raw)
```

`s_raw` 是激活前的尺度参数，`o_raw` 是 opacity logit；不能先对原属性激活再相加。rotation 应在完整四维 residual 得到后再归一化。若后续支持 `apply_rotation=True`，必须复用并验证当前四元数乘法约定。

## 4. 分阶段的融合边界

以下名字仅用于本文规划，不是当前已注册的 backend：

| 阶段 | kernel 输入 → 输出 | 目的 |
|---|---|---|
| M0 | 已有 `ReLU(backbone)` → 五组 delta | 首先验证 Tensor Core whole-head，隔离 backbone 误差 |
| M1 | HexPlane `[N,32]` → 五组 delta | 本文的完整 MLP 首版；包含 backbone |
| M2 | HexPlane + 原属性 → 五组 raw 属性 | 将 residual 融入尾层写回 |
| M3 | HexPlane + 原属性 → renderer 激活后的属性 | 再减少 exp/sigmoid/normalize 的启动和访存 |
| M4 | xyz/time + 平面参数 → 属性 | 后续独立评估 HexPlane 与 MLP 联合融合 |

先完成 M0/M1 standalone，再验证 M2/M3，再考虑 Tacker mixed wrapper。每扩大一次边界，都保留上一个通过验证的候选参与 E2E 比较。

## 5. 参数布局与 API

### 5.1 连续布局

固定 head 顺序为 `pos, scales, rotations, opacity, shs`，尾层通道前缀和为 `[0,3,6,10,11,59]`。

| 名称 | dtype / shape | 布局与用途 |
|---|---|---|
| `features` | FP32 `[N,32]` | HexPlane 采样结果，按 Gaussian 连续 |
| `backbone_weight_kh` | FP32 `[32,128]` | 原 `[out,in]` 权重转置后的缓存，便于相邻线程读取相邻输出通道 |
| `backbone_bias` | FP32 `[128]` | 原 backbone bias |
| `first_weights` | FP16 `[5,128,128]` | `[head,out,in]`，保留 PyTorch Linear 的每-head 排列 |
| `first_biases` | FP32 `[5,128]` | 不量化 bias |
| `tail_weights` | FP32 `[59,128]` | 依 head 顺序拼接尾层的输出行，不补零 |
| `tail_biases` | FP32 `[59]` | 对应尾层拼接顺序 |
| `outputs` | 五个 FP32 `[N,Oi]` | 每个独立连续；SH 可零复制 reshape 为 `[N,16,3]` |

不要默认把 `[N,59]` 的列切片作为连续 `[N,Oi]` 输出：它们的 row stride 仍为 59。若以后采用这种布局，需明确让消费者支持 stride，并测量可能发生的 `.contiguous()` 复制。

### 5.2 一次性参数缓存示意

以下参数打包示例可用于当前 standalone 接口；模型结构检查及 renderer 缓存管理尚未自动接入：

```python
net = pc._deformation.deformation_net
backbone = net.feature_out[0]
heads = [net.pos_deform, net.scales_deform, net.rotations_deform,
         net.opacity_deform, net.shs_deform]

with torch.no_grad():
    backbone_weight_kh = backbone.weight.detach().float().t().contiguous()
    backbone_bias = backbone.bias.detach().float().clone().contiguous()
    first_weights = torch.stack([
        head[1].weight.detach().half() for head in heads
    ]).contiguous()
    first_biases = torch.stack([
        head[1].bias.detach().float() for head in heads
    ]).contiguous()
    tail_weights = torch.cat([
        head[3].weight.detach().float() for head in heads
    ], dim=0).contiguous()
    tail_biases = torch.cat([
        head[3].bias.detach().float() for head in heads
    ], dim=0).contiguous()
```

缓存只在模型加载后建立。新 checkpoint、参数原地修改、device/dtype/结构变化时必须重建或拒绝继续使用；不能只以旧 tensor 地址判断权重未改变。若训练或 densification 后继续渲染，应重新创建 renderer 及其参数/slot 缓存。

### 5.3 已实现的 M1 Python 接口

```python
from tacker_4dgs_head import deform_mlp_delta_v1, deform_mlp_delta_out_v1

# 自动分配五个输出；在 torch.no_grad() 或 inference_mode() 中调用。
outputs = deform_mlp_delta_v1(
    features,
    backbone_weight_kh, backbone_bias,
    first_weights, first_biases,
    tail_weights, tail_biases,
)

# 使用已分配的五组输出。
deform_mlp_delta_out_v1(
    features,
    backbone_weight_kh, backbone_bias,
    first_weights, first_biases,
    tail_weights, tail_biases,
    outputs,                    # 调用者预分配的五组 delta
    persistent_blocks=0,
)
```

当前约定：`persistent_blocks=0` 取 `logical_tiles=ceil(N/16)`，即默认 full-grid；2026-09-16 的 A6000 实测中，旧的 `min(SM_count, logical_tiles)` 策略只达到 8.33% occupancy 和 4 active warps/SM，而 full-grid 达到 49.23% occupancy 和 23.63 active warps/SM，真实行数 `N=111525` 的 CUDA Event 中位时延从 7.540 ms 降到 3.045 ms。显式正数仍表示 persistent grid 大小，并裁剪到逻辑 tile 数。`N=0` 在 host 直接返回空输出，避免零大小 launch。

Host wrapper 已检查 CUDA device 一致、dtype/shape、tensor 连续性、参数数目、输出和所有输入的存储不重叠，以及多个输出互不重叠。接口要求关闭 autograd，无论输入是否需要梯度；调用者提供的输出也不能需要梯度。WMMA 权重基地址检查 32 B 对齐，内部 shared memory 布局满足 WMMA 对齐要求。目前布局由 C++ 结构和接口固定；未来 mixed ABI 还需显式记录 row count、tile shape、精度模式和输出含义。

使用当前 PyTorch CUDA stream 和匹配的 device guard，执行 launch error 检查；正常入口不调用 `cudaDeviceSynchronize()`。首次分配由上层 slot/初始化负责，计时内优先使用 `_out` API。M2/M3 用不同接口或显式输出模式，避免把 raw 属性误当作 delta 或已激活属性。

## 6. CUDA block 的具体执行过程

### 6.1 工作划分

令 `B=16, H=128, K=32`，`t=threadIdx.x`，`warp=t/32`，`lane=t%32`。

```text
logical_tiles = ceil(N / 16)
tile = blockIdx.x; tile < logical_tiles; tile += gridDim.x
row_begin = tile * 16
```

111,525 个 Gaussian 对应 6,971 个逻辑 tile，最后一个 tile 有 5 个有效行。

现有第一层 adapter 把 128 个输出通道拆成两个 `16×64` 的逻辑 block，其工作单元数量为 13,942。完整 MLP 的下一层需要同一行的全部 128 维，故需重写该划分，不能直接沿用旧 block grid 后在 kernel 内追加尾层。

### 6.2 加载特征与计算 backbone

1. 128 个线程合作把 `16×32` 个 FP32 特征读入 `sF`；越过 `N` 的行补零。
2. 同步后，每个线程负责输出通道 `c=t`，依次遍历 16 行。
3. 对每个有效行进行长度 32 的 FP32 点积，最后加 `backbone_bias[c]`。
4. 应用 ReLU，再使用 round-to-nearest 转为 FP16，写入 `sA[r,c]`。
5. 无效行将 `sA` 直接置零。所有线程同步后进入 head 计算。

```cpp
// 示意：sF 已加载，且全部 128 个 worker threads 已完成同步。
for (int r = 0; r < 16; ++r) {
    float value = 0.0f;
    if (row_begin + r < rows) {
        for (int k = 0; k < 32; ++k)
            value = fmaf(sF[r * 32 + k], W0_KH[k * 128 + t], value);
        value = relu_fp32(value + b0[t]);
    }
    sA[r * 128 + t] = __float2half_rn(value);
}
worker_sync();
```

这里 `W0_KH` 的转置在初始化时完成，因此固定 `k` 下同一 warp 读取连续的 32 个输出通道权重。该 baseline 易于验证，但 FP32 backbone 仍可能限制性能；它的 FLOPs 占比小不能证明它的时间占比小。

### 6.3 一个 head 的第一层：WMMA

一个 head 要计算 `sA[16,128] × Wi.T[128,128]`。每个 warp 负责两个 `16×16` 输出 tile：

```text
warp 0：输出列 0..15   和 64..79
warp 1：输出列 16..31  和 80..95
warp 2：输出列 32..47  和 96..111
warp 3：输出列 48..63  和 112..127
```

对 `q=0,1`，`col=16*warp+64*q`：

1. 创建 `wmma::fragment<wmma::accumulator,16,16,16,float>`，置零。
2. `k=0,16,...,112`，共八次 K tile。
3. A 从 `sA+k` 读取，`row_major`、`ldm=128`。
4. B 从 `Wi+col*128+k` 读取，解释成 `col_major`、`ldm=128`。原 `[out,in]` row-major 存储恰好对应这里的转置矩阵，不需要逐帧重新排布。
5. 执行 `wmma::mma_sync`，最后写入 FP32 `sZ+col`，`mem_row_major`、`ldm=128`。

首版逐个 `q` 计算，限制同时存活的 accumulator 数量；同时保留两个 accumulator 以复用 A fragment 可作为后续调优候选。

四个 warp 写完各自区域后执行 worker barrier，然后线程合作对 `sZ[e]` 加 `bi[e%128]` 并应用 ReLU，再执行 worker barrier，供尾层读取。

不要假设 WMMA fragment 的内部元素与具体输出列一一对应后直接添加不同通道的 bias。先 store 到布局明确的 shared memory，再按列索引 bias，是本方案的基准实现。[WMMA 布局与对齐要求](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#warp-matrix-functions)

### 6.4 尾层：warp 协作完成 FP32 点积

head 输出宽度为 `O`，将本 tile 的输出元素编号为 `u∈[0,16*O)`：

```text
每个 warp 处理 u = warp, warp+4, warp+8, ...
r = u / O
c = u % O
每个 lane 累加 k = lane, lane+32, lane+64, lane+96
warp shuffle reduction 得到 128 项的和
lane 0 加 tail bias，再写一个有效输出元素
```

这样 opacity 的单输出也由整个 warp 合作完成，避免把 128 线程中的 127 个都闲置在尾层。对于 SH 的 48 维输出，同一调度处理更多 `u`；先保持 FP32，后续再比较专门的尾层实现。

所有 warp 在本 head 尾层结束后执行 worker barrier，然后下一 head 才能覆盖 `sZ`。`sA` 在五个 head 之间保持不变。M2 可以把原属性读取、residual 和输出写回放在上述 lane 0 的写出位置。

### 6.5 同步、尾行与精度要求

- standalone 的 `worker_sync()` 使用全体 128 线程参与的 `__syncthreads()`；Tacker mixed 版本使用第 9 节的独立 named barrier。
- 所有 32 个 warp lanes 必须执行 WMMA 和 shuffle；仅在实际输出 store 处判断有效行。禁止按 `row>=N` 提前退出一部分线程。
- 线程必须走完一致的 head 循环和 worker barrier 序列；多余 tile 的跳过只能由整个 worker group 一致决定。
- 不使用 `fmaxf(x,0)` 悄悄掩盖 NaN。定义并测试与目标 ReLU 一致的处理；输入/中间值出现非有限数时不能被误报为数值通过。
- 首版不启用 `--use_fast_math`。FP32 累加次序和 warp reduction 仍可能与 PyTorch/cuBLAS 不同，因此不承诺逐 bit 相等。

## 7. 存储预算、kernel 骨架与调优方向

### 7.1 首版 shared memory 预算

| 缓冲 | 内容 | 大小 |
|---|---|---:|
| `sF` | FP32 `[16,32]` 输入特征 | 2,048 B |
| `sA` | FP16 `[16,128]` backbone 的 ReLU 输出 | 4,096 B |
| `sZ` | FP32 `[16,128]` 单个 head 的 GEMM/ReLU 结果 | 8,192 B |
| 合计 | 五个 head 复用同一 `sZ` | **14,336 B，即 14 KiB** |

以上不含未来的权重 staging、descriptor、额外 padding、调试输出或 mixed wrapper 的 Raster shared memory。三个缓冲均按 32 B 对齐声明；WMMA 的 half stride 采用 128、float stride 采用 128，满足对应对齐要求。

五个 head 的第一层 FP16 权重合计 `5×128×128×2=163,840 B`，即 160 KiB。不能假设把这些权重同时装进一个 block 的 shared memory：`sm_86` 的每 block 共享内存上限为 99 KiB，且超过静态分配限制的动态共享内存需要显式 opt-in。[Ampere 资源限制](https://docs.nvidia.com/cuda/archive/12.6.0/ampere-tuning-guide/index.html#occupancy)

首版 WMMA 从全局权重缓存读取，由硬件缓存复用。后续可对正在使用的 K tile 做 shared-memory staging 或异步复制；必须把 staging 与双缓冲计入总预算，并验证同步。

### 7.2 已实现的设备调用顺序

下面是实际设备函数在每个 16 行 tile 加载并同步后的主体。各 helper 均已实现在 `deform_mlp_device.cuh`，不是 cuBLAS host API，也没有启动子 kernel：

```cpp
deform_backbone_linear_v1(p, s, row_begin, tid);
__syncthreads();
deform_backbone_relu_v1(s, tid);
__syncthreads();
for (int head = 0; head < kDeformHeads; ++head) {
    deform_head_linear_v1(p, s, head, tid);       // WMMA + bias
    __syncthreads();
    deform_head_relu_v1(s, tid);
    __syncthreads();
    deform_tail_linear_v1(p, s, head, row_begin, tid);
    __syncthreads();                            // scratch 可安全复用
}
```

`tacker_deform_mlp_delta_v1` 以 128 threads 启动，声明 shared scratch 并调用 `deform_mlp_device_v1(params, scratch)`。设备主体使用 `tid=threadIdx.x`，从 `blockIdx.x` 开始以 `gridDim.x` 为步长遍历 `ceil(N/16)` 个 tile。矩阵层之间的数据依赖全部在本 block 内完成。

本版使用覆盖整个 block 的 `__syncthreads()`，必须由全部 128 个线程共同执行。第 9 节的 mixed worker 需要单独适配同步策略；当前函数不能原样放进只有部分 Raster/MLP 线程执行的分支。

### 7.3 能省多少中间数据？

当前 C3 的五组第一层输出为 FP32 `[5,N,128]`。在 `N=111,525` 时，单份大小为：

```text
5 × 111,525 × 128 × 4 = 285,504,000 B ≈ 272.28 MiB
```

如果把写出和后续读取各计一次，这条边对应约 544.56 MiB 的逻辑 tensor 流量。融合可以消除该完整中间 tensor 的物化；实际减少多少 DRAM 流量取决于缓存、原实现的 ReLU 中间结果和编译行为，应以 Nsight 测量为准。

MLP 仍需读权重、特征并写最终结果；shared memory 也存在访问和同步成本。不能用上述字节数除以峰值显存带宽，直接宣称预计加速倍数。

### 7.4 有顺序地调优

1. 固定 `B=16, threads=128`，比较 full-grid 与 persistent 调度。候选 `P` 可取 `S,2S,4S,8S,L`，其中 `S=SM_count`、`L=ceil(N/16)`；裁剪到 `[1,L]` 并去重。
2. 只在首版正确后比较 `B=32`、256 线程或同一 CTA 的多个行 tile；重新核算 shared memory、同步参与者和输出覆盖。
3. 根据 profiling 分别优化 backbone、WMMA 权重读取和尾层。尾层 FP16/TF32、backbone Tensor Core 都必须作为新的精度模式验证。
4. 对每个编译变体记录 registers/thread、static/dynamic shared memory、local memory、spill 和实际 active blocks/SM；使用实际 kernel 调用 occupancy API。
5. 根据完整序列 FPS 决定是否扩大融合边界。不要以最高 occupancy 或最少 launch 单独决定赢家。

## 8. 精度合同与数值参考

### 8.1 三种参考必须分开

| 参考 | 定义 | 回答的问题 |
|---|---|---|
| R0 原模型 FP32 | 原 checkpoint，所有 Linear 使用 FP32 参考计算 | 相对原模型的最终数值和画质差异 |
| R1 匹配量化边界 | backbone FP32；`a` 和 head 第一层权重转 FP16 后再以 FP32 计算；bias/tail FP32 | 量化造成的误差与 CUDA 实现误差分别有多少 |
| R2 当前候选 | 在同一 workload 下真实执行当前 C3 或当前有效 renderer | 新实现是否改善当前实际系统 |

R1 的概念参考如下，运行时显式关闭 TF32 并在测试后恢复原设置：

```python
h = features.float() @ backbone_weight_kh.float() + backbone_bias
a = torch.relu(h).half().float()
deltas = []
for i in range(5):
    z = torch.relu(a @ first_weights[i].float().t() + first_biases[i])
    lo, hi = tail_offsets[i], tail_offsets[i + 1]
    delta = z @ tail_weights[lo:hi].float().t() + tail_biases[lo:hi]
    deltas.append(delta)
```

R1 与实际 M1 kernel 的 backbone 累加顺序可能不同，导致 `a.half()` 舍入在边界上不同。调试版本应允许导出 `a`，必要时向 head-only M0 注入同一份 `a`，把 backbone 误差、第一层误差和尾层误差拆开定位。

FP16 存储不代表整个网络以 FP16 累加；首版 tail 不量化。数学重排的等价性也不意味着浮点逐 bit 等价。

### 8.2 kernel-level 指标

每个 head 分别记录 max absolute error、RMSE、relative L2、非有限值数量；同时检查最终 raw/activated 属性。测试数据应包含真实 checkpoint 的 feature/属性分布，而不只是独立随机正态数。

现有扩展随机测试使用 `rtol=2e-3, atol=2e-3`，可以作为第一轮 head 运算 smoke test 的起点，但不能直接当成新增完整 MLP 的已验证误差合同。正式阈值须在 qualification 前声明并封存，分别约束 R1 的实现误差和 R0 的模型误差。

应覆盖大/小数、ReLU 零点附近、接近 FP16 舍入边界、零向量四元数、极端 opacity logits。发生 FP16 溢出或非有限输出时，标记该 precision/workload 组合无效；不能依靠放宽 `allclose` 让它通过。

### 8.3 渲染级指标

继续采用仓库当前记录的 50-view 平均画质门限：PSNR 下降不超过 0.05 dB、SSIM 下降不超过 `1e-4`、LPIPS 上升不超过 `1e-4`。[现有门限](/home/huawei/大二春/实验室/4DGaussians/tacker_profiles/raster_head_sm86.json)

同时记录逐视角最大变化，避免仅看平均值掩盖个别视角错误。若测试 mixed wrapper，固定相同 Raster 输入时，其 color/depth/radii 应与 standalone Raster 一致；若比较 R0 与量化 MLP 的最终渲染，则不能要求所有 radii 逐项相等，因为输入 Gaussian 已有微小数值变化。

## 9. 接入渲染和 Tacker

### 9.1 standalone 接入

新增推理专用 backend，在 `deform_for_render` 对应的 fine 路径中选择：

```text
network.grid(xyz, time)
  → features.contiguous()（如需复制，计入运行时间）
  → deform_mlp_delta_out_v1(...)
  → 原 residual 规则
  → 原 renderer 激活
  → GaussianRenderState
```

不能先调用 `network.query_time()` 再把结果作为 M1 输入：该函数已经执行了 backbone，而且返回 `[N,128]`，不是 `[N,32]` 的 HexPlane 特征。

当前有效配置下，位置/scale/rotation 的正余弦编码最终未被这些 MLP 分支消费，实际使用的是前 3/3/4 个原始分量；时间 MLP 调用也处于注释状态。M1 专用路径可以直接用原始输入采样并构造 residual，省去无用编码。资格检查必须限制在本文支持配置，并把这部分调度变化计入 E2E 结果。

M1 输出 delta 时可复用 residual 逻辑；M2 输出 raw 属性时只执行 activation；M3 输出已激活属性时直接构造 state。应有清晰的输出类型/模式，确保 residual、sigmoid、exp 或 normalize 不会执行两遍。

### 9.2 新增 Tacker partition

现有 `_prepare_deformation_prefix()` 已运行位置编码、`query_time()` 和 mask，所以 M1 需要新的 feature-only prepare。拟议 `FullMlpPartition` 应提供：

| 方法 | 拟议职责 |
|---|---|
| `cache_parameters` | 缓存 backbone 转置、head 首层 FP16、FP32 尾层及布局元数据 |
| `prepare` | 计算 HexPlane 特征；准备输入/输出 slot；记录 `prefix_ready` |
| `launch_mixed` | 当前帧 Raster render leaf + 下一帧 full MLP |
| `result_from_mixed` | 分离当前帧图像结果与下一帧五组 MLP 输出 |
| `finish` | 按 delta/raw/activated 模式执行剩余步骤，构造下一帧 state |

还需新增 partition kind、backend/ABI family、candidate builder、注册和 validation。`fused_nodes/skipped_python_nodes` 必须包含 backbone 和五个 head 的对应运算；首版五 head 全部由 kernel 执行，不能再调用 `_run_parallel_deformation_heads()` 计算它们。

新的 task 需要显式 `features` 和 `rows`；现有部分校验从 `task.hidden.shape[0]` 推断行数，不能为兼容接口而制造一个无用的 `[N,128]` hidden tensor。扩展 shape 校验和 `record_stream` 遍历即可。

新 ABI 输入为 `[N,32]` FP32，逻辑单位是 16 行全 MLP，输出为五组 delta；与现有 C3/C4 合同不同，必须使用新 symbol/manifest/version。旧 C3/C4 不能因名字相似被视为兼容。

### 9.3 mixed block 的线程与 barrier

首版 mixed CTA 采用：

```text
物理线程 [0,256)   ：Raster，使用自己的 barrier 1
物理线程 [256,384) ：full MLP，local_tid=threadIdx.x-256
full MLP barrier 2：恰好 128 个参与者
full MLP scratch  ：独立的 sF/sA/sZ，共 14 KiB，另计 Raster 资源
```

MLP 的 `worker_sync()` 在 mixed 版本替换为与当前 adapter 同类的 `bar.sync`，显式参与人数 128；可通过模板 `SyncPolicy` 复用主体。不能在仅 MLP 线程执行的分支内调用覆盖全部 384 线程的 `__syncthreads()`，否则可能等待永远不会到达的 Raster 线程。

参数 descriptor 若由一个 warp 写入 shared memory，必须在读取前进行对应 worker barrier；也可以使用明确的按值参数，避免额外 descriptor 广播。所有 named barrier 必须登记，且与 Raster 保持隔离。

本方案一个 MLP worker 依次处理五个 head。若以后增加多个 worker group，优先分配不同 Gaussian tile，各自拥有完整 backbone 和 head 链；若选择按 head 切分并共享 backbone，则需要重新设计跨 group 的数据可见性和 barrier，不属于首版。

实际 mixed kernel 的寄存器和共享内存占用必须在编译后重新获取，不能直接把旧 C3/C4 的 occupancy 写进新 manifest。特别是额外 14 KiB scratch 可能降低 active blocks/SM，抵消节省的访存。

### 9.4 两槽事件和存储生命周期

沿用当前框架中的依赖方向：

```text
deform stream：HexPlane(t+1) → prefix_ready(t+1)
                                  ↓ raster stream 等待
raster stream：等待 ready(t)、prefix_ready(t+1)
               → Raster(t) + MLP(t+1)
               → mixed_done(t+1)、raster_done(t)
                                  ↓ deform stream 等待
deform stream：finish(t+1) → ready(t+1)
```

- 参数缓存完成的事件必须先于任意使用它的 stream；不能依赖默认 stream 的隐式同步。
- `features`、参数、原属性、输出及 camera/Raster 输入在相应 consumer 完成前保持有效。跨 stream 的 allocator 使用需要 `record_stream` 或等价的显式持有策略；它不代替 producer/consumer event。
- `_out` 缓冲每 slot 独立，只有旧 slot 的 Raster/consumer 读完后才能覆写。保留当前 `raster_done` 复用约束和调用者 consumer stream 规则。
- 首帧 prefill 和末帧 drain 没有成对的跨帧 mixed 工作。可暂时沿用现有首帧完整 deformation/末帧 solo Raster；报告应注明首帧是否使用原始精度。若改为新 standalone MLP prefill，也需单独验证并纳入计时。
- 提交前的配置、ABI 或资源不匹配使用既有回退机制。若 GPU launch 已失败，须确认已提交工作和 CUDA 上下文状态，不能吞掉异步错误后继续复用 slot。

## 10. 实现文件与阶段交付

### 10.1 已实现与后续代码位置

已创建并接通 Python 构建和绑定：

```text
tacker_ext/include/deform_mlp_device.cuh       # 设备主体、scratch、CTA/named-barrier 同步策略
tacker_ext/include/deform_mlp.h                # host API 声明
tacker_ext/csrc/deform_mlp.cu                  # standalone kernel、host 检查
tacker_ext/tests/test_deform_mlp_cuda.py       # GPU 数值、边界、stream、覆盖检查
tacker_ext/abi/deform_mlp_v1.json              # full MLP 设备 ABI
submodules/depth-diff-gaussian-rasterization/
  cuda_rasterizer/tacker_mixed.cu              # ABI v5 直接融合 kernel
  abi/tacker_mixed_render_deform_mlp_v5.json   # Raster+full MLP 合同
```

已修改的接线位置：`tacker_ext/csrc/bindings.cpp`、`tacker_ext/setup.py`、`tacker_ext/tacker_4dgs_head/__init__.py`。

ABI v5 已接通 rasterizer 的 CUDA/C++/Python 直接调用入口；`persistent_blocks=0` 表示完整 logical grid，尚未缩减为 persistent grid。后续仍需要新增 `tests/test_full_mlp_partition.py`，并修改 `gaussian_renderer/__init__.py`、`gaussian_renderer/tacker_pipeline.py` 以接入两槽跨帧调度。离线枚举/资格验证入口仍需在 `scripts/tacker_autotune.py`、`profile_tacker_leaves.py` 及相应 benchmark 工具中扩展。

### 10.2 每阶段的完成条件

1. **提取参考与参数打包**：对真实 checkpoint 验证 shape、head 顺序、权重转置和尾层 offsets，得到 R0/R1。
2. **M0 Tensor Core whole-head**：同一份 `a` 下五个输出通过数值/尾行/并发测试；确认第一层使用矩阵指令。
3. **M1 全 MLP standalone**：backbone 不在 Python 重复执行，中间 `[N,128]` 和 `[5,N,128]` 不再全局物化，通过完整画质检查。
4. **M2/M3 属性写回**：验证 delta/raw/activated 三种含义，以及只执行一次 residual/activation；量化吞吐收益。
5. **Tacker mixed**：验证新 partition、named barrier、两槽事件、首尾帧和无重复计算，随后与 standalone/two-stream/C3 比较 E2E。
6. **选择候选**：封存源码、二进制、参数和测量报告；只有通过当前正确性和性能准入规则的结果才成为可选 profile。

## 11. 验证与测量计划

### 11.0 本次实际验证结果（2026-09-16）

在远端 `4A6000` 的单张 RTX A6000（物理 GPU 1）编译并执行，环境为 PyTorch `2.4.1+cu124`、CUDA Toolkit `12.4`，编译目标 `sm_86`。本次使用合成权重/特征，包含实际场景规模的行数 `N=111525`；尚未加载真实 checkpoint 验证完整渲染画质。

| 已执行检查 | 结果与范围 |
|---|---|
| GPU unittest | **7 项全部通过**：空/尾 tile、111525 行及不同 persistent grid、解析 bias/ReLU/head 顺序、输出复用与 canary、非法参数、非默认 stream、profiler |
| 单 launch 验证 | profiler 对一次非空调用断言恰好 **1 个 CUDA kernel**，名称包含 `tacker_deform_mlp_delta_v1` |
| 数值比较 | 对齐 FP16 量化边界的 PyTorch 参考，关闭 TF32；`rtol=2e-3, atol=5e-4`，并检查结果有限 |
| Compute Sanitizer | 对空/尾 tile 和输出缓冲复用测试分别运行 memcheck、racecheck、synccheck；**0 错误、0 数据竞争隐患** |
| 编译资源 | ptxas：**73 registers/thread、14,336 B shared memory、0 spill、0 stack frame** |
| 本地 CPU 回归 | 67 项发现，44 项通过、23 项 GPU 测试跳过；GPU 工作全部在远端执行 |

远端独立源码目录：`/home/qyfeng/deform_mlp_20260916_01a0_v1`。编译产物和验证日志：`/data/qyfeng/deform_mlp_validation/20260916_01a0_v1`，包含 `build.log`、`tests.log`、`memcheck.log`、`racecheck.log`、`synccheck.log`。没有覆盖现有运行环境中的扩展。

上述结果证明 standalone 接口在这些测试下能够正确执行，并消除了各阶段之间的 kernel launch；不等于全 FP32 逐位一致、真实场景画质准入或性能胜出。下面保留后续接入与正式测量的完整验收计划，未列入上表的项目尚未全部执行。

2026-09-17 新增的 ABI v5 直接融合版当前仅通过无 Torch/CUDA 的源码合同与 Python/JSON 检查；由于本地环境没有 PyTorch/CUDA，且本次无法连通 `4A6000`，它尚未列入上表的 GPU 通过结果。

### 11.1 数值与执行测试

| 类别 | 最小覆盖 |
|---|---|
| 行数 | `N=0,1,5,15,16,17,31,32,33,111525` |
| 输出 | 五 head 分别与 R1/R0 比较；SH reshape 和输出顺序正确 |
| 边界 | 尾 tile 零填充、不越界写；canary 缓冲与 Compute Sanitizer memcheck |
| 同步 | Compute Sanitizer racecheck/synccheck；不同 persistent grid 下无漏算/重复写 |
| 输出复用 | `_out` 返回原存储、重复调用覆盖完整；两 slot 不互相污染 |
| 参数/ABI | 错误 dtype、形状、stride、misalignment、alias、device、旧缓存被拒绝 |
| stream | 非默认 stream、输入生产事件、输出消费事件、allocator 压力下生命周期 |
| 数值极端 | ReLU 边界、FP16 舍入/溢出、归一化零向量、极端 logits、非有限值 |
| partition | backbone/head/residual 的每帧计数、首尾帧、提前结束消费、错误回退 |

非 GPU 的分区测试用可观察的模块调用和事件顺序证明“没有重复运算”和“依赖正确”，不要只比较新增函数名是否出现在源文件中。

### 11.2 三层性能测量

**第一层：设备主体。** 用相同实际 tensor 测 M0/M1 和既有 C3/C4 leaf，CUDA event 记录耗时；同时记录是否包含 feature 连续化、输入转换、输出分配。单 kernel 数据只用于定位瓶颈。

**第二层：完整 deformation。** 从 xyz/time/原属性开始，到 renderer 所需属性就绪为止，包含 HexPlane、MLP、必要复制、residual 和 activation。对比是否只是把时间从 MLP 移到了 prefix/suffix。

**第三层：完整渲染序列。** 固定 checkpoint、Gaussian 数、分辨率、test 视角 0–49，warmup 10 帧，至少 10 轮交错运行。保留原 serial、two-stream、有效 C3/incumbent、新 standalone MLP、新 mixed MLP 等实际可执行基线。

```text
throughput_fps = measured_frames / measured_elapsed_seconds
```

计时从提交第一个 measured frame 前开始，到最后一个结果在 consumer stream 完成并同步为止；包含 Python 调度、prefill/drain、事件等待和实际 tensor 转换。不包含加载数据/模型、编译、一次性权重打包和图像文件 I/O。

正式比较使用每轮完整序列 FPS，给出 median、paired ratio 和置信区间，并核对 `actual_execution_mode`、fallback 和执行计数。Nsight 采样单独运行，不把带 profiler 开销的耗时混入正式排名。

沿用项目已有的候选选择规则：报告封存候选集合中的实测最佳项；替换有效 incumbent 时检查至少 1% 提升及 paired bootstrap 95% CI 下界大于 1.0。画质和执行正确性仍是前提。[既有选择与计时规则](/home/huawei/大二春/实验室/4DGaussians/TACKER_LATENCY_FIRST_FUSION_PLAN.md)

### 11.3 必须封存的诊断数据

- 精确 workload、checkpoint/config/hash、源码和子模块版本、已加载二进制、CUDA/PyTorch/driver、GPU 状态。
- `B/threads/P`、head 调度方式、precision mode、融合边界、barrier 和输出布局。
- kernel launches、Tensor Core 指令使用、显存/缓存流量、shared memory 冲突、寄存器和 spill。
- MLP、完整 deformation、完整渲染三层耗时及逐轮原始数据；所有输出误差与逐视角画质变化。

所有 GPU 工作在已连接的 `4A6000` 执行，不在本地运行。远端代码和小配置放在 `/home/qyfeng` 下；checkpoint、数据集及大体积 Nsight/测量输出放在 `/data/qyfeng` 下。四张卡是服务器配置，不要求一个实验同时占用四张卡。

## 12. 后续扩展：HexPlane 联合融合

只有在 M1/M2 的性能已明确后，再考虑把采样放进同一个 block。每个 Gaussian 独立采样，理论上不需要跨 Gaussian 同步；但必须保持当前 `grid_sample` 的坐标归一化、`align_corners=True`、`padding_mode='border'`、平面轴顺序、双线性插值和乘积顺序。

当前四维输入生成 `xy,xz,xt,yz,yt,zt` 六个平面；在每个尺度内把六个采样特征逐元素相乘，最后拼接两个尺度，得到 32 维。不要误写成只把三个空间/时间平面对相乘后求和。

联合融合还可能消除 feature tensor 的物化，但增加不规则访存、寄存器寿命和共享内存压力。它是新的候选，需要重新验证收益。训练反向传播则需额外处理权重梯度归约、激活保存或重算，以及平面梯度散射；本文的 inference kernel 不自动提供训练支持。

## 13. 实施结论

当前已实现并通过 standalone GPU 测试：**一个 128-thread block 每次处理 16 个 Gaussian，在 block 内完成 FP32 backbone、五个 head 的 Tensor Core 第一层和 FP32 尾层，依次复用 14 KiB scratch，输出五组 delta。**

这条路径具有明确的输入输出、量化边界和同步范围，后续可扩大到 residual/activation 和 Tacker mixed。是否接入默认渲染路径，由真实 checkpoint 的画质验证与目标 A6000 workload 的完整渲染 FPS 决定；本次尚未进行这些测量。
