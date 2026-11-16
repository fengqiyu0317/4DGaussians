# gsplat 可选渲染后端与固定 checkpoint 对比

日期：2026-10-21。原始数据保存在
[report.json](../benchmarks/results/gsplat_backend_20261021/formal50/report.json)。

本页记录首次后端接入实验。随后已实现并测试物理 mixed kernel，见
[gsplat mixed kernel 实验](gsplat_mixed_kernel_20261021.md)；下文数据保留为首次接入对照。

## 结论

gsplat 1.5.3 dense 的 Raster 调用耗时从 **4.198 ms 降至 2.120 ms**，
约 **1.98×** 加速。相同串行调度下，端到端速度从 **82.89 FPS 提升至
98.85 FPS（+19.25%）**。gsplat 双流为 **105.55 FPS**，现有
`full_mlp_overlap_graph_pb0` Graph 融合对照为 **130.33 FPS**。

gsplat 的画质与原实现非常接近，但不完全相同：平均 PSNR 下降
`0.004113 dB`，SSIM 下降 `0.0002700`，LPIPS 增加 `0.0010805`。
**SSIM 和 LPIPS 超过项目现有严格质量门槛，不能把本次结果当成已经通过
等价替换准入。默认后端仍为原实现。**

这些结果适用于以下单场景、单卡、固定版本条件，不代表所有 gsplat
版本或其他场景的性能。

## 实现

- `PipelineParams.raster_backend`：`diff_gaussian`（默认）或 `gsplat`。
- `PipelineParams.gsplat_packed`：默认 False；显式开启 packed。
- 在 `prepare_render_context()` 选择 rasterizer，复用已有的
  `deform_for_render()` 和 `rasterize_state()` 接口。
- HexPlane、Deform MLP、参数激活和 checkpoint 均不转换。
- 适配器返回 CHW RGB、每高斯半径、`[1,H,W]` 累计深度。
  深度对应 `sum(alpha_i * T_i * z_i)`，不是按 alpha 归一化的期望深度。
- 相机使用转置后的 world-to-camera 矩阵，主点为 `(W/2,H/2)`；
  gsplat 的半像素中心约定与原 rasterizer 的整数像素中心相对应。
- 设置 `near_plane=0.2`、`eps2d=0.3`、`radius_clip=0`、
  `rasterize_mode=classic`、`render_mode=RGB+D`。
- gsplat 1.5.3 的 packed 路径在背景维度检查上存在不一致。
  适配器在 packed 模式使用零背景渲染，再合成 `RGB + (1-alpha)*background`。
  这部分开销已包含在 packed 计时中，未修改第三方库。
- 对两条 CUDA stream 显式记录相机 tensor 的使用周期。
- 当前适配器仅支持推理，训练会显式报错，因为原训练代码依赖 dummy
  `means2D` 的 densification 梯度，不能直接映射到 gsplat 的梯度接口。
- `TackerRenderer` 要求 `diff_gaussian`；本轮首次后端比较没有包含 gsplat
  物理 mixed kernel。后续融合实验使用单独的 `GsplatMixedRenderer`。
  普通 `TwoStreamRenderer` 已实测支持 gsplat。
- PanopticSports 的 opaque camera 路径暂不支持此适配器。

## 工作负载与协议

| 项目 | 本次设置 |
|---|---|
| GPU | 4A6000 的物理 GPU 2，RTX A6000 / SM 8.6 |
| 环境 | Python 3.10.20，PyTorch 2.4.1+cu124，CUDA Toolkit 12.4 |
| gsplat | 1.5.3，独立 `--target` 依赖目录，独立 JIT 缓存 |
| 驱动 | 570.124.06，普通 nvidia-smi 正常，未使用 LD_PRELOAD |
| 场景 | N3DV `flame_steak` |
| checkpoint | iteration 14000，111,525 Gaussians |
| 分辨率 | 1352×1014 |
| 视角 | test views 0–49，所有候选顺序一致 |
| 预热 | 每个候选先做 lazy 初始化，再预热 10 帧 |
| 计时 | 单进程、10 轮，每轮每候选完整执行 50 帧 |
| 候选顺序 | 正序、逆序交替；所有表格使用 10 轮中位数 |
| 画质 | 50 帧全部计算，使用现有仓库 PSNR / SSIM / LPIPS(VGG v0.1) 实现 |

端到端包含 context setup、HexPlane、MLP、激活、RGB/深度光栅化和末尾
CUDA 同步，主指标为 `perf_counter` 测得的 50 帧总耗时。
checkpoint/相机读取、首次编译、预热、图像写盘和质量指标不计时。
首帧 prefill 与末帧 drain 包含在序列计时内。

Raster-only 先缓存同一批已激活的 GaussianRenderState，各后端逐帧消费
完全相同的 state；相机 context 在计时前准备。CUDA event span 也记录在
JSON 中；它包含 host 提交造成的 GPU 空闲间隙，不是 Nsight 的 kernel
active time 求和。下表 Raster 使用 wall time，CUDA span 与其几乎相同。

每次 GPU 实验前重新检查进程、显存与利用率。checkpoint 与关键源码的
SHA256 在运行前后核验一致。完整结果记录实际加载的扩展二进制 SHA256。
这次比较使用独立实验副本，保留远端已编译基线及其对应 Python 实现；
适配入口、gsplat 后端和比较脚本与本地提交内容一致。原模型和原环境未改写。

## 性能

| 候选 | Raster ms/frame | 端到端 ms/frame | 端到端 FPS | 相对原串行 FPS |
|---|---:|---:|---:|---:|
| 原后端串行 | 4.198 | 12.064 | 82.89 | 1.000× |
| gsplat dense 串行 | 2.120 | 10.117 | 98.85 | 1.192× |
| gsplat packed 串行 | 2.452 | 10.494 | 95.29 | 1.150× |
| 原后端双流 | — | 11.580 | 86.35 | 1.042× |
| gsplat dense 双流 | — | 9.474 | 105.55 | 1.273× |
| 现有 Graph mixed 融合 | — | 7.673 | 130.33 | 1.572× |

端到端 FPS 的 10 轮样本变异系数依次为 0.194%、0.261%、0.438%、
0.247%、0.371%、0.325%。

gsplat dense 比相同双流调度的原后端快约 22.23%。它尚未叠加现有
物理 mixed kernel 优化，因此该双流结果与 Graph mixed 是不同的调度。
本场景 dense 比 packed 更快，dense 为本次建议的试验配置。

Graph 对照实际使用
`/data/qyfeng/4DGaussians-flame-steak-full/benchmarks/results/full_mlp_graph_formal_20260925/full_mlp_overlap_graph_pb0.qualification.json`，
SHA256 为 `844669df6303460fd6c40b15d9719bb7abc7fe06df4cb843af440939cf4f7da5`。
报告确认 `qualification_executed=true`，实际选中
`full_mlp_overlap_graph_pb0`，每 50 帧执行 49 次 mixed、49 次 prefix、
1 次 full deformation、1 次 solo raster，输出 50 帧，无 fallback。
该 profile 对应 HexPlane prefix Graph replay。实验开关没有启用 v11/v12
等其他 mixed 变体；这里的 Graph 数据不能解读成这些变体之间的比较。

## 画质与差异

| 候选 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 相对原串行质量门槛 |
|---|---:|---:|---:|---|
| 原后端串行 / 双流 | 32.574433 | 0.95773631 | 0.14036516 | 参考 / 通过 |
| gsplat dense 串行 / 双流 | 32.570320 | 0.95746631 | 0.14144566 | 未通过 SSIM、LPIPS |
| gsplat packed 串行 | 32.570320 | 0.95746632 | 0.14144559 | 未通过 SSIM、LPIPS |
| 现有 Graph mixed | 32.574222 | 0.95773498 | 0.14036575 | 通过 |

沿用项目门槛：平均 PSNR drop ≤0.05 dB、SSIM drop ≤1e-4、
LPIPS increase ≤1e-4。LPIPS 的输入及预处理沿用
`scripts/validate_tacker_modes.py` 的仓库实现和 `[0,1]` 图像输入。

gsplat dense 与原串行输出的平均 RGB MAE 为 `5.579e-5`，平均 RGB
RMSE 为 `7.865e-4`。50 帧中逐帧 PSNR drop 范围为
`0.003605–0.004929 dB`。累计深度的平均 MAE 为 `0.001775`
（场景单位），所有输出 RGB 与深度均有限。

`radius_clip=0` 不意味着两个实现逐像素等价。gsplat 投影源码使用
opacity-aware、逐轴的 bounding box；旧实现使用基于最大特征值的圆形
半径扩展 tile 范围。源码已确认存在这些工作范围差异，但本次没有用
消融实验把全部像素误差或全部速度收益归因到某一项策略。
两后端的 visibility/radii 定义也会因此不同，本次平均 visibility mask
差异约 8.67%，不应直接把它解释成相同数量的可见图像内容被丢弃。

已经检查相机矩阵转置、半像素约定、旋转/非等方差高斯、尺度修正、
预计算协方差、RGB/累计深度及 packed/dense 输出。适配器通过小场景
CUDA 测试；真实场景仍未达到项目的严格等价替换门槛。

对照图从左到右：**GT、原后端、gsplat dense、绝对 RGB 差异×10**。

![view 25](../benchmarks/results/gsplat_backend_20261021/formal50/comparison_view_025.png)

另有 view 0、49 对照图，逐帧指标与逐轮耗时均在 report.json 中。

## 使用与复现

在兼容 CUDA/PyTorch 环境安装 `requirements-gsplat.txt`；首次调用会编译
CUDA 扩展。当前实验依赖安装在独立目录，没有升级原有 PyTorch。

```bash
python render.py \
  --model_path /data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak \
  --source_path /data/qyfeng/datasets/n3dv/flame_steak \
  --configs arguments/dynerf/flame_steak.py --iteration 14000 \
  --skip_train --skip_video --raster_backend gsplat
```

增加 `--execution-mode two_stream` 使用双流，增加 `--gsplat_packed`
使用 packed；默认仍是原渲染后端。普通 render.py 会将渲染图片写入
模型目录，正式比较脚本则将所有产物写到单独的 `--output` 目录。

本次远端复现入口：

```bash
export TORCH_EXTENSIONS_DIR=/data/qyfeng/gsplat-backend-20261021/torch_extensions
bash /data/qyfeng/gsplat-backend-20261021/project/scripts/run_a6000_gsplat_backend.sh \
  /data/qyfeng/gsplat-backend-20261021/project \
  /data/qyfeng/gsplat-backend-20261021/deps \
  /data/qyfeng/gsplat-backend-20261021/results/new_run \
  2
```

输出目录必须不存在。脚本会在运行前检查指定 GPU 是否空闲；依赖、
原有 CUDA 扩展和 Graph qualification 需提前准备好。其他模型/数据位置
可通过脚本中的 `FOURDGS_MODEL_PATH`、`FOURDGS_DATASET_PATH`、
`FOURDGS_GRAPH_QUALIFICATION` 环境变量覆盖。

代码入口：`gaussian_renderer/gsplat_backend.py`、
`scripts/benchmark_gsplat_backend.py`、
`scripts/run_a6000_gsplat_backend.sh`。
