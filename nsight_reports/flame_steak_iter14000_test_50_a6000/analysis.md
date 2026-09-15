# n3dv flame_steak：4DGaussians 训练与 Nsight 推理分析

## 摘要

- 平台：4× NVIDIA RTX A6000；训练和分析使用 GPU 2。
- 最终模型：fine iteration 14000，111,525 个 Gaussian。
- 训练指标：test PSNR 32.7299 dB，train PSNR 36.3147 dB。
- Nsight：test split，1352×1014，warmup 10 帧后采集 50 帧。
- 端到端：13.1551 ms/帧，76.02 FPS；中位数 12.9766 ms，标准差 0.6472 ms，最大值 16.7860 ms。
- GPU 投影：deformation 8.7419 ms/帧（66.09%），rasterization 4.1385 ms/帧（31.29%）。首要瓶颈是形变网络。

## 部署与版本

完整项目：`/data/qyfeng/4DGaussians-flame-steak-full`

| 组件 | commit |
|---|---|
| 4DGaussians | `843d5ac636c37e4b611242287754f3d4ed150144` |
| depth-diff-gaussian-rasterization | `e49506654e8e11ed8a62d22bcb693e943fdecacf` |
| GLM | `5c46b9c07008ae65cb81ab79cd677ecc1934b903` |
| simple-knn | `44f764299fa305faf6ec5ebd99939e0508331503` |

独立环境：`/data/qyfeng/conda-envs/4dgaussians-flame-steak`，Python 3.10.20，PyTorch 2.4.1+cu124，CUDA 12.4，Nsight Systems 2023.4.4。

rasterizer 与 simple-knn 均从上述固定子模块 editable 编译；实际加载的 Python 包和共享库都位于新仓库的 `submodules` 下。最小 CUDA 前向已验证 rasterizer 返回 image、radii、depth，不再使用共享环境中的同名扩展。

## Checkpoints

目录：`/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak`

| checkpoint | iteration | Gaussian 数 | 大小 | SHA-256 |
|---|---:|---:|---:|---|
| `chkpnt_coarse_3000.pth` | 3000 | 74,514 | 63,600,890 B | `c97d8763b14563c024ecd464c1ffbbc9fa534d8688b181ce77ba0fc99802f3cc` |
| `chkpnt_fine_3000.pth` | 3000 | 93,403 | 96,927,953 B | `7a0632c3dbb2808515f895ddb1748796b7430a03317acbe912359e04175bbcae` |
| `chkpnt_fine_14000.pth` | 14000 | 111,525 | 109,994,370 B | `9f5676d8b4329026c454dacc93c4d6268307796be9b16b3a88a740141be68e6d` |

三个 checkpoint 均已反序列化并核对内部 iteration；对应 point cloud、deformation、deformation_table 和 deformation_accum 文件齐全。

## 方法

- 模型：fine iteration 14000
- split：test
- warmup：10 帧，不捕获
- profile：50 帧
- trace：CUDA、NVTX、OS runtime
- capture：应用内 `cudaProfilerStart/Stop`
- Python SH 转换：关闭，SH→RGB 在 rasterizer 中完成
- Python covariance：关闭，covariance 在 rasterizer preprocess 中完成

报告目录：`/data/qyfeng/4DGaussians-flame-steak-full/nsight_reports/flame_steak_iter14000_test_50`

## 推理渲染流程

```text
测试相机、时间戳
  └─ renderer/setup
       ├─ screen-space tensor
       ├─ 相机矩阵与 raster settings
       └─ 展开每个 Gaussian 的时间输入
  └─ renderer/deformation
       ├─ positional encoding
       ├─ HexPlane feature sampling
       ├─ backbone MLP
       └─ position / scale / rotation / opacity / SH heads + residuals
  └─ renderer/activation
       ├─ scale activation
       ├─ quaternion normalization
       └─ opacity sigmoid
  └─ renderer/rasterization
       ├─ preprocessCUDA：投影、协方差、SH→RGB、tile 覆盖
       ├─ prefix scan
       ├─ 读取 num_rendered，调整 binning buffer
       ├─ duplicateWithKeys
       ├─ CUB radix sort
       ├─ identifyTileRanges
       └─ renderCUDA：tile alpha compositing，输出 RGB 与 depth
```

## 阶段时间

GPU 投影用于判断 GPU 阶段延迟；CPU 区间用于观察 Python/CUDA API 提交和阻塞，二者不能互相替代。

| 主阶段 | CPU ms/帧 | CPU 占比 | GPU 投影 ms/帧 | GPU 投影占比 | GPU ops/帧 |
|---|---:|---:|---:|---:|---:|
| setup | 3.9225 | 29.82% | 0.3113 | 2.35% | 8 |
| deformation | 4.1276 | 31.38% | 8.7419 | 66.09% | 106 |
| activation | 0.1561 | 1.19% | 0.0252 | 0.19% | 5 |
| rasterization | 4.8396 | 36.79% | 4.1385 | 31.29% | 28 |
| 未标记 | 0.1092 | 0.83% | 0.0094 | 0.07% | — |

| deformation 子阶段 | CPU ms/帧 | GPU 投影 ms/帧 | GPU 投影总区间占比 |
|---|---:|---:|---:|
| positional encoding | 0.3179 | 0.3643 | 2.75% |
| HexPlane feature sampling | 2.3449 | 2.2728 | 17.18% |
| backbone MLP | 0.0817 | 0.1303 | 0.99% |
| heads and residuals | 1.2720 | 5.8661 | 44.35% |

每个主阶段和子阶段均恰有 50 个 NVTX 实例，与采样帧数一致。

## GPU kernel 构成

50 帧共 6,050 个 GPU kernel，即 121 次/帧；kernel duration 总和为 11.2588 ms/帧。duration 总和用于分析计算构成，在存在并发时不能替代端到端墙钟时间。

| 类别 | ms/帧 | kernel 时间占比 |
|---|---:|---:|
| GEMM | 3.9468 | 35.06% |
| raster render | 2.4149 | 21.45% |
| activation / clamp / ReLU | 1.6873 | 14.99% |
| radix sort | 1.0331 | 9.18% |
| elementwise | 0.9272 | 8.23% |
| duplicateWithKeys | 0.4709 | 4.18% |
| concat copy | 0.3243 | 2.88% |
| HexPlane grid_sample | 0.2465 | 2.19% |
| raster preprocess | 0.0559 | 0.50% |

最大单 kernel 是 `ampere_sgemm_128x64_tn`，2.9028 ms/帧；其次是 `renderCUDA`，2.4149 ms/帧；第三是 ReLU/clamp 的 vectorized elementwise kernel，1.6873 ms/帧。

## CUDA API 时间口径

`cuda_api_sum` 是 CPU 调 CUDA Runtime API 的时间，不是 GPU kernel 执行时间。

- `cudaLaunchKernel`：6,000 次，42.5763 ms 总计，即 120 次/帧、0.8515 ms/帧；只代表 CPU 提交开销。
- `cuda_gpu_kern_sum`：11.2588 ms/帧；表示 GPU kernel duration 构成。
- NVTX GPU projection：13.2263 ms/帧；表示阶段在 GPU 时间线上的投影。
- `cudaStreamSynchronize`：800 次，180.0154 ms 总计，即 16 次/帧；API 时间可能包含 CPU 等待 GPU。
- 同步 `cudaMemcpy`：50 次，221.4610 ms 总计，恰好 1 次/帧，平均 4.4292 ms/API。

固定 rasterizer 的 `cuda_rasterizer/rasterizer_impl.cu:282` 每帧有一次：

```cpp
cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1,
           sizeof(int), cudaMemcpyDeviceToHost)
```

它只复制 4 字节，但要等待前面的 prefix scan，随后 CPU 才能按 `num_rendered` 调整 binning buffer。全部 GPU memory operations 只有 0.0681 ms/帧，因此 4.4292 ms 的 API 时间主要是等待/同步，不是搬运 4 字节本身。API 时间会与其它区间重叠，不能直接加到 13.1551 ms 端到端时间上。

## 瓶颈与优化优先级

1. **优先优化 deformation heads。** 五个 head 的 heads_and_residuals 为 5.8661 ms/帧；GEMM 与 activation 合计约占 kernel 时间 50%。可评估共享 head trunk、合并输出 GEMM、推理 fused kernel 或 `torch.compile`，并重新验证画质。
2. **删除推理路径的无效张量创建。** 当前多次 `zeros_like` 后立即覆盖，并在本配置下创建全 1 mask 再乘加。推理专用 residual 分支可减少 fill、elementwise 和 allocator 压力。
3. **消除 rasterizer 的 4 字节 D2H 同步。** 可研究上界/复用 binning buffer 或 device 侧大小管理。该改动涉及内存安全，必须做数值和峰值显存验证。
4. **优化 tile render/sort。** raster render、radix sort、duplicate 合计约 3.9188 ms/帧。可评估预裁剪、LOD、Gaussian 数控制或新版 rasterizer，并验证 PSNR/SSIM/LPIPS。
5. **融合 HexPlane 周边算子。** HexPlane 阶段 2.2728 ms/帧，而 grid_sample 本身仅 0.2465 ms/帧，其余主要是 index、concat、elementwise 与小 kernel 调度。
6. **缓存 setup 对象。** setup 的 GPU 投影仅 0.3113 ms/帧，但 CPU 区间 3.9225 ms/帧。可缓存相机 device tensor、时间展开 tensor、raster settings，并避免推理时创建带梯度的 screen-space tensor。
7. **减少约 120 次 runtime launch/帧。** 消除动态内存和 host sync 后，再评估 CUDA Graph、固定形状 replay 和算子融合。

## 适用范围

这些数据仅适用于当前 checkpoint、111,525 个 Gaussian、1352×1014 测试视角和 RTX A6000。更高分辨率、不同时间分布、Gaussian 数或 rasterizer 版本都需重新采样。优化前后必须保持相同 warmup、50 帧集合、分辨率和 capture range。
