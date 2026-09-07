# 4DGaussians flame_steak：Jetson Nsight 推理分析

## 摘要

- 平台：NVIDIA Jetson AGX Orin，JetPack/L4T R36.5，PyTorch 2.8.0，CUDA 12.6，Nsight Systems 2024.5.4。
- 功耗模式：`MAXN`；采样前 GPU 约 44.3°C。未取得 root 权限锁定 `jetson_clocks`，GPU 仍采用动态时钟。
- 模型：fine iteration 14000，111,525 个 Gaussian。
- 输入：n3dv `flame_steak` test split，1352×1014；与 A6000 使用完全相同的 `poses_bounds.npy`、点云和前 50 张测试图像。
- 方法：10 帧 warmup 后捕获 50 帧；CUDA、NVTX、OS runtime；应用内 `cudaProfilerStart/Stop`。
- 端到端：48.1902 ms/帧，20.75 FPS；相对 A6000 的 13.1551 ms/帧慢 3.66×。
- kernel duration 总和：45.4416 ms/帧，50 帧共 6,050 次，即 121 次/帧；相对 A6000 慢 4.04×。
- 最大瓶颈：`renderCUDA` 17.4524 ms/帧，占全部 kernel duration 的 38.41%，相对 A6000 慢 7.23×。

## 数据与运行链路验收

Jetson 上使用独立的 4DGaussians DyNeRF 精简数据目录，而不是直接复用 E-D3DGS/Colmap 布局：

- 21/21 个 `camXX/images` 链接有效；每个相机保留 300 个时间槽。
- `cam00` 的前 50 帧均为非空真实图像，`cam01/0000.png` 用于训练相机集合初始化。
- `poses_bounds.npy` 形状为 `(21, 17)`；关键文件和图像 SHA-256 与 A6000 一致。
- `cam00/0000.png`、`cam00/0049.png`、`cam01/0000.png` 均为 1352×1014。
- ARM64 `depth-diff-gaussian-rasterization` 与 `simple-knn` 均在 Orin 上编译，并从新项目目录加载；最小 rasterizer 前向返回 `image, radii, depth` 三项。
- 单帧 smoke test 退出码为 0，确认 iteration 14000、111,525 Gaussians、1352×1014、`dynerf` 和实际一帧 render 全部成功。

## 端到端与阶段时间

GPU 投影时间用于判断 GPU 阶段延迟；CPU/NVTX 区间包含 CUDA API 提交与等待，不能与 GPU 投影时间直接相加。

| 指标 | Jetson ms/帧 | A6000 ms/帧 | Jetson/A6000 |
|---|---:|---:|---:|
| 端到端 render loop | 48.1902 | 13.1551 | 3.66× |
| GPU 投影 render loop | 48.6449 | 13.2263 | 3.68× |
| kernel duration 总和 | 45.4416 | 11.2588 | 4.04× |

| 主阶段 | Jetson GPU ms/帧 | Jetson 占比 | A6000 GPU ms/帧 | 倍率 |
|---|---:|---:|---:|---:|
| setup | 0.9388 | 1.93% | 0.3113 | 3.02× |
| deformation | 23.5935 | 48.50% | 8.7419 | 2.70× |
| activation | 0.0993 | 0.20% | 0.0252 | 3.94× |
| rasterization | 23.9988 | 49.33% | 4.1385 | 5.80× |

A6000 上 deformation 占 GPU 投影区间 66.09%、rasterization 占 31.29%；Jetson 上两者变为 48.50% 与 49.33%。瓶颈因此从“以 deformation 为主”转变为“rasterization 与 deformation 基本各占一半”。

| deformation 子阶段 | Jetson GPU ms/帧 | A6000 GPU ms/帧 | 倍率 |
|---|---:|---:|---:|
| positional encoding | 1.9284 | 0.3643 | 5.29× |
| HexPlane feature sampling | 5.3382 | 2.2728 | 2.35× |
| backbone MLP | 0.7653 | 0.1303 | 5.87× |
| heads and residuals | 15.3893 | 5.8661 | 2.62× |

50 个帧区间的中位数为 48.5757 ms，标准差 3.0750 ms，最大值 49.7145 ms。`frame_0000` 为 26.6828 ms，是 capture 边界后的首帧；其余帧集中在约 48.45–49.71 ms。为保持与 A6000 相同口径，汇总没有剔除该帧。

## CUDA kernel 构成

| 类别 | Jetson ms/帧 | kernel 占比 | A6000 ms/帧 | 倍率 |
|---|---:|---:|---:|---:|
| raster render | 17.4524 | 38.41% | 2.4149 | 7.23× |
| GEMM | 8.6877 | 19.12% | 3.9468 | 2.20× |
| activation / clamp | 6.0377 | 13.29% | 1.6873 | 3.58× |
| radix sort | 3.8713 | 8.52% | 1.0331 | 3.75× |
| elementwise | 3.5610 | 7.84% | 0.9272 | 3.84× |
| concat copy | 1.9454 | 4.28% | 0.3243 | 6.00× |
| duplicateWithKeys | 1.8849 | 4.15% | 0.4709 | 4.00× |
| HexPlane grid_sample | 1.1162 | 2.46% | 0.2465 | 4.53× |
| other | 0.6701 | 1.47% | 0.1520 | 4.41× |
| raster preprocess | 0.2149 | 0.47% | 0.0559 | 3.85× |

Rasterizer 直接相关的 `renderCUDA + radix sort + duplicateWithKeys + preprocessCUDA` 合计 23.4235 ms/帧，占 kernel duration 的 51.55%。GEMM、activation、elementwise、concat 与 grid sample 合计 21.3480 ms/帧，占 46.98%。

Top kernel：

| 排名 | kernel | ms/帧 | kernel 占比 |
|---:|---|---:|---:|
| 1 | `renderCUDA<3>` | 17.4524 | 38.41% |
| 2 | `ampere_sgemm_128x64_tn` | 6.0721 | 13.36% |
| 3 | clamp/ReLU vectorized elementwise | 6.0377 | 13.29% |
| 4 | CUB radix sort onesweep | 3.6595 | 8.05% |
| 5 | `duplicateWithKeys` | 1.8849 | 4.15% |
| 6 | 2-D concat copy | 1.2667 | 2.79% |
| 7 | `ampere_sgemm_32x128_tn` | 1.2061 | 2.65% |
| 8 | vectorized multiply | 1.1539 | 2.54% |
| 9 | `grid_sampler_2d_kernel` | 1.1162 | 2.46% |
| 10 | CUTLASS backbone GEMM | 0.7653 | 1.68% |

38 个唯一 kernel 的完整名称、实例数、总时长、每帧时长和占比见 `kernel-breakdown.csv`。

## CUDA API 与同步

`cuda_api_sum` 是 CPU 调 CUDA Runtime API 的时间，不是 GPU kernel duration，不能直接加到端到端时间上。

- `cudaMemcpy`：50 次，恰好 1 次/帧，平均 12.7538 ms/API；A6000 为 4.4292 ms/API，Jetson 慢约 2.88×。
- 与之对应的实际 Device-to-Host GPU 搬运仅 0.001224 ms/帧。因此 12.7538 ms 主要是等待前序 GPU 工作，而不是复制 4 字节本身。
- 该同步来自 rasterizer 读取 `num_rendered` 的 Device-to-Host `cudaMemcpy`，会阻塞 CPU，随后才能调整 binning buffer。
- `cudaStreamSynchronize`：16 次/帧，API 总时间折合 23.7241 ms/帧；A6000 为 3.6003 ms/帧。
- `cudaLaunchKernel`：120 次/帧，API 总时间折合 2.0952 ms/帧；A6000 为 0.8515 ms/帧。
- GPU memory operations 总和为 0.2338 ms/帧，说明主要瓶颈仍是 kernel 计算和同步等待，而非显式数据搬运。

## 结论与优化优先级

1. **优先优化 rasterizer tile render。** `renderCUDA` 单项达到 17.4524 ms/帧、38.41%，并且相对 A6000 慢 7.23×，是 Jetson 上最突出的架构敏感瓶颈。
2. **同时控制 tile sort/binning 成本。** radix sort、`duplicateWithKeys` 与 preprocess 合计 5.9707 ms/帧；连同 render 后，rasterizer 直接 kernel 已占 51.55%。可评估 Gaussian 裁剪、LOD、tile 覆盖控制和更适合 Orin 的 block/occupancy 配置。
3. **优化 deformation heads。** heads and residuals 的 GPU 投影为 15.3893 ms/帧；GEMM 与 activation 合计占 kernel duration 32.41%。共享 head trunk、合并输出 GEMM、减少全量 mask/zeros_like 与算子融合仍有价值。
4. **消除每帧 4 字节 D2H 同步。** `cudaMemcpy` 平均阻塞 12.7538 ms；可研究复用/预分配 binning 上界或 device 侧大小管理。该修改涉及内存安全，必须验证数值与峰值内存。
5. **降低小 kernel 与 concat 开销。** 121 次 kernel/帧不变，但 Jetson 的 concat copy 相对 A6000 慢 6.00×、positional encoding 慢 5.29×。减少临时张量、合并 elementwise/concat、固定形状后评估 CUDA Graph 更有意义。

## 适用范围

对比保持了相同 checkpoint、Gaussian 数、分辨率、前 50 个测试时间戳、warmup 和 capture 范围。两端软件栈不同：Jetson 为 PyTorch 2.8/CUDA 12.6，A6000 为 PyTorch 2.4.1/CUDA 12.4；因此倍率同时包含 GPU 架构、内存系统和软件版本差异。Jetson 处于 MAXN，但没有锁定最高 GPU 时钟。
