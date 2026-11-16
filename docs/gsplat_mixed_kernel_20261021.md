# gsplat 像素混合 + Deform MLP 物理 mixed kernel v1

日期：2026-10-21。已实现、编译并在 RTX A6000 上运行。
原始报告：[report.json](../benchmarks/results/gsplat_mixed_20261021/formal50/report.json)。

## 结论

第一版确实把 **gsplat 的 RGB+D 像素混合(t)** 与 **完整 Deform MLP(t+1)**
放进同一个 CUDA kernel，并接入双 slot HexPlane CUDA Graph 流水线。

固定 checkpoint 的 50 帧、10 轮测试中，新 mixed Graph 管线为
**168.25 FPS**，原有 Raster Graph mixed 为 **130.14 FPS**。
但是，相同 gsplat 前处理、相同精度 MLP、相同 Graph 前缀的非融合对照
为 **172.04 FPS**。因此第一版物理融合本身没有带来净收益，反而让
FPS 下降 **2.20%**；不能把相对旧管线的全部收益归因于 mixed kernel。

默认后端和默认调度均未切换，新实现作为显式实验入口保留。

## 融合边界与实现

```text
deform_stream: HexPlaneGraph(t+1) ────────────────┐
                                               ↓ prefix_ready
raster_stream: gsplat 投影/SH/tile/排序(t) → mixed kernel
                                              ├─ 前 256 线程：gsplat blending(t)
                                              └─ 后 128 线程：完整 MLP(t+1)
                                                        ↓ mixed_done
deform_stream:                                residual + activation(t+1)
```

`gsplat_mix_render_deform_v1` 是一个真实的 384-thread CUDA kernel：

- Raster 部分保留 gsplat 1.5.3 的像素中心、alpha 上限 0.999、
  alpha threshold 1/255、exclusive early termination 及 RGB+D 计算顺序。
- 原生 `__syncthreads_count` 和 CTA 同步改为 **barrier 1 / 256 threads**。
  如果直接搬用 CTA-wide barrier，会等待没有参加 Raster 的 MLP 线程。
- MLP 部分复用项目已验证的 `deform_mlp_gptb_device_v1`，使用
  **barrier 2 / 128 threads**；保留 ABI-v5 的 FP32 backbone、FP16
  first-head 权重/激活与 FP32 累加、FP32 scalar tail。
- 一个 subgroup 不访问另一个 subgroup 的 shared scratch。
- Raster 以 16×16 tile 工作，MLP 以 16 Gaussian rows 工作。
  默认物理 CTA 数取两者逻辑 tile 数的最大值，也支持显式 grid-stride。
- 投影、SH、tile 求交与排序仍由原生 gsplat 算子实现。
- 首帧使用原 PyTorch deformation prefill；末帧使用 gsplat solo leaf drain。
- 双 slot 独立持有 Graph 输出，使用 CUDA events 控制复用与跨 stream
  数据依赖，并记录 tensor 的 stream 生命周期。

实现文件：

- `gsplat_mixed_ext/mixed.cu`：物理 mixed kernel、参数检查与资源查询。
- `gsplat_mixed_ext/bindings.cpp`、`__init__.py`：PyTorch binding 与独立 JIT。
- `gaussian_renderer/gsplat_mixed.py`：gsplat prepare/leaf 拆分与 Graph 调度器。
- `tests/test_gsplat_mixed_cuda.py`：真实 CUDA 输出、边界与 grid-stride 测试。

Raster device 代码基于 gsplat 1.5.3 源码改写，保留 Apache-2.0 来源说明及
`gsplat_mixed_ext/LICENSE.gsplat`。没有修改用户安装的 gsplat 包。

## 工作负载与协议

RTX A6000 物理 GPU 2、PyTorch 2.4.1+cu124、CUDA Toolkit 12.4、gsplat
1.5.3，`flame_steak` iteration 14000，111,525 Gaussians，1352×1014。
所有候选读取同一 checkpoint，使用 test views 0–49。

每个候选先完成 lazy 初始化及两个 slot 的 capture，再预热 10 帧。
在同一个进程内执行 10 轮，每轮所有候选各跑完整 50 帧，正/逆序交替。
首帧 prefill、末帧 drain、context setup、HexPlane、MLP、激活和 Raster
均计入端到端 wall time；读取、编译、预热、图片写盘、质量指标不计时。

原有 Graph mixed 对照使用 `full_mlp_overlap_graph_pb0` qualification，
SHA256 `844669df6303460fd6c40b15d9719bb7abc7fe06df4cb843af440939cf4f7da5`。
新实现复用该 profile 验证 MLP 架构/GPU/workload 与 Graph 前缀配置，
**该旧 qualification 不会自动给新 gsplat leaf 准入**；新 leaf 的输出、
资源和物理重叠在本轮单独验证。未混用 eager `full_mlp_overlap` 数据。

## 物理融合与数值验证

对真实场景固定输入进行 leaf 核验：

| 项目 | 结果 |
|---|---:|
| Raster RGB+D 相对原生 gsplat 最大绝对误差 | **0** |
| Alpha 相对原生 gsplat 最大绝对误差 | **0** |
| 五个 MLP delta 相对原 ABI-v5 standalone 最大绝对误差 | **全部为 0** |
| 配对 Raster/MLP CTA | 5,440 |
| clock64 观察到两组执行窗口重叠的 CTA 比例 | **100%** |
| 重叠窗口 / 较短 subgroup 窗口，中位数 | **99.978%** |

clock64 数据来自单独的仪器化调用，正式计时关闭该仪器化。
窗口重叠说明两组任务在同一 CTA 内交错执行，并不代表每个周期两组都
能同时发射，也不保证总耗时下降。

每个 50 帧新 mixed sequence 的执行计数为：

```text
full_deformation=1
prefix_graph_replays=49
mixed_launches=49
raster_prepares=50
solo_raster=1
outputs=50
```

匹配非融合对照为 `mixed_launches=0, separate_leaf_pairs=49`，其余
调度相同，无 fallback。两个路径均实际使用两个 HexPlane CUDA Graph slot。

## 性能

| 实现 | 中位 ms/frame | 中位 FPS |
|---|---:|---:|
| 原后端串行 | 12.050 | 82.99 |
| gsplat dense 串行，原 PyTorch MLP | 10.111 | 98.91 |
| gsplat dense 双流，原 PyTorch MLP | 9.473 | 105.56 |
| 原 Raster + ABI-v5 MLP，现有 Graph mixed | 7.684 | 130.14 |
| gsplat + ABI-v5 MLP + Graph，独立 leaf 对照 | **5.813** | **172.04** |
| gsplat + ABI-v5 MLP + Graph，物理 mixed v1 | 5.943 | 168.25 |

10 轮 FPS 样本 CV：mixed 为 0.452%，匹配非融合为 0.420%。
相对原 Graph mixed，新 mixed 的 FPS 提升 29.28%；相对 gsplat 原双流
提升 59.39%。但相对匹配非融合对照下降 2.20%，后者才是判断物理融合
增量收益的直接对照。

固定输入、排除整个前缀后的 leaf wall time 中位数：

- gsplat native blending + standalone ABI-v5 MLP：**3.736 ms**。
- 单个物理 mixed leaf：**3.959 ms**，慢约 5.98%。

新 mixed 编译资源为每线程 **56 registers**、每 CTA **21,864 bytes
shared memory**、**0 bytes local memory**，理论可驻留 **3 CTA/SM**。
资源共享和调度占用可能抵消重叠收益；这是根据资源数据作出的推断，
本轮尚未用 NCU 将回退归因到某个具体 stall 或吞吐瓶颈。

## 整帧画质

新 mixed 与匹配非融合路径的 **50 帧逐帧指标全部相同**。
相对原生 gsplat dense：

| 指标 | 变化 |
|---|---:|
| PSNR drop | 0.000210876 dB |
| SSIM drop | 0.000001332 |
| LPIPS(VGG) increase | 0.000001607 |

三项均通过沿用项目阈值的增量检查。该微小变化来自复用 ABI-v5 MLP
精度路径；单独 mixed leaf 对原生 gsplat 与该 MLP 的输出都是逐元素一致。

gsplat 相对旧 rasterizer 的既有画质差异仍然存在，新 mixed 相对旧串行
PSNR 为 32.570109 vs 32.574433，SSIM 为 0.95746498 vs 0.95773631，
LPIPS 为 0.14144727 vs 0.14036516，因此仍不满足旧渲染器的严格 SSIM/
LPIPS 等价替换门槛。没有因为 leaf 通过就自动切换默认后端或部署。

## 测试与复现

4 项 CUDA 测试通过，覆盖奇数尺寸、非整 16 行、grid-stride、opaque
早停、无 intersection、0 MLP rows、dtype/layout 与 inference guard。
CUDA Compute Sanitizer memcheck 为 **0 errors**。
关键源码、checkpoint 的运行前后 SHA256 核验通过；报告记录 JIT 扩展
及旧编译扩展的 SHA256。依赖与实验副本均放在独立目录。

在准备好的远端环境复现：

```bash
export TORCH_EXTENSIONS_DIR=/data/qyfeng/gsplat-mixed-20261021/torch_extensions
bash /data/qyfeng/gsplat-mixed-20261021/project/scripts/run_a6000_gsplat_mixed.sh \
  /data/qyfeng/gsplat-mixed-20261021/project \
  /data/qyfeng/gsplat-backend-20261021/deps \
  /data/qyfeng/gsplat-mixed-20261021/results/new_run 2
```

脚本先重新验证 GPU 空闲，输出目录必须不存在。
`GSPLAT_MIXED_BLOCKS=0` 为本次配置；正整数用于独立的 grid-stride 实验。

普通 render.py 的显式实验入口：

```bash
python render.py --model_path MODEL --source_path DATA \
  --configs arguments/dynerf/flame_steak.py --iteration 14000 \
  --skip_train --skip_video --raster_backend gsplat \
  --execution-mode gsplat_mixed --workload-name flame_steak \
  --tacker-profile FULL_MLP_OVERLAP_GRAPH_QUALIFICATION.json
```

`--execution-mode gsplat_matched_graph` 运行匹配非融合对照。
两个入口都要求 gsplat 1.5.3、dense layout、标准 FoV 相机和合适的父
Graph qualification。它们没有通过原 Tacker profile 冒充旧 mixed kernel。
