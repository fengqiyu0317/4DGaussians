# FullMlpPartition A6000 端到端速度筛查

日期：2026-09-20

## 结论

在 `flame_steak` iteration 14000、111,525 Gaussians、1352x1014 的 50 帧 test 序列上，`FullMlpPartition` 的 ABI-v5 mixed kernel 在 `persistent_blocks=0` 时达到 **115.47 FPS / 8.661 ms per frame**。

与同一 checkpoint 和视角序列相比：

- 相对 serial 的 83.52 FPS，吞吐提升 **38.24%**，平均帧时间降低 **27.66%**。
- 相对 two-stream 的 87.32 FPS，吞吐提升 **32.23%**，平均帧时间降低 **24.37%**。
- PB=0 比 PB=336 的 109.23 FPS 快 **5.71%**，因此当前 direct logical grid 优于这个 persistent-grid 对照点。

## 测试合同

- GPU: NVIDIA RTX A6000, SM 8.6，物理 GPU 2。
- PyTorch: 2.4.1+cu124；CUDA runtime: 12.4；driver module: 570.124.06。
- 模型：`/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak`。
- 数据：`/data/qyfeng/datasets/n3dv/flame_steak`。
- 工作负载：test views 0--49，10 帧 warmup，每个模式 5 个 50-frame whole-sequence trials。
- 主指标：`perf_counter` 包围整个序列，trial 结束时 `cudaDeviceSynchronize`；同时用 consumer-stream CUDA events 记录逐帧 completion。
- 加载、renderer prepare 和 warmup 不在计时区间内。
- 测试顺序：serial -> two-stream -> PB=0 -> PB=336。各组内是 5 次连续 trial，不是正式 10x50 ABBA 交错准入协议。
- 每次 GPU 实验前均重新检查 NVML；GPU 2 无计算进程。GPU 0 上他人的 GR00T 服务未被使用或干预。

## 结果

| 模式 | 实际执行模式 | 中位 FPS | 中位 ms/frame | p50 ms | p95 ms | CUDA peak allocated |
|---|---|---:|---:|---:|---:|---:|
| serial | serial | 83.52 | 11.973 | 11.941 | 12.007 | 296.5 MiB |
| two-stream | two_stream | 87.32 | 11.451 | 11.438 | 12.514 | 383.1 MiB |
| Full MLP PB=0 | tacker | **115.47** | **8.661** | **8.561** | **8.638** | 459.3 MiB |
| Full MLP PB=336 | tacker | 109.23 | 9.155 | 9.068 | 9.124 | 459.3 MiB |

五次 whole-sequence FPS 原始值：

- serial: 84.018, 83.523, 83.477, 83.641, 83.469（CV 0.275%）
- two-stream: 86.934, 87.351, 87.349, 87.006, 87.325（CV 0.235%）
- Full MLP PB=0: 115.466, 114.994, 115.333, 115.475, 115.602（CV 0.202%）
- Full MLP PB=336: 108.415, 109.230, 109.324, 108.168, 109.332（CV 0.513%）

PB=0 的每个 50-frame trial 均实际执行：1 次 full-deformation prefill、49 次 prefix、49 次 mixed launch、49 次 suffix、1 次 solo-raster drain，共输出 50 帧，且每个 deformation head 每帧只计算一次。`actual_execution_mode=tacker`，两类 fallback reason 均为 null。

## 可复现性

- 顶层实现 commit: `2edb0452e937e12e4260f2f53d2e9dc0ec99c189`
- rasterizer submodule commit: `464096373b869fff713265bcdc34b1822277980d`
- `gaussian_renderer/tacker_pipeline.py` SHA-256: `22da9d1f7351240d097eeb31ceb2c6d406c4df89b4ebcdc93064028fe0de5220`
- `profile_render.py` SHA-256: `d7006d8b5fcce5b92abfb1257d01ad616c80cb54f7892f7c5e7310fa4f33d6e6`
- loaded rasterizer binary SHA-256: `daad1cec0065b596423046dfff9db533ea69d2561167e5924a6c5d9d5ab692d0`
- PB=0 qualification profile SHA-256: `8c981b15a1d18f49d4da3793566bf57a66e7904bf096c3ed9edd81cb728ebdab`
- 远端原始 metadata: `/data/qyfeng/full_mlp_partition_validation/20260920_e2e`

## 限制

这是端到端速度筛查，不是正式 deployment admission。qualification profile 保持 `deployment.enabled=false` 和 `correctness.valid=false`。ABI-v5 CUDA 数值与跨 stream 路径已由单元/集成测试覆盖，但候选项在启用为默认路径前，仍需执行 serial 对照的整帧 PSNR/SSIM/LPIPS 质量测试和正式交错 benchmark。
