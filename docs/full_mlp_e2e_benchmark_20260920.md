# FullMlpPartition A6000 端到端质量与正式速度验证

日期：2026-09-20

## 结论

在 `flame_steak` iteration 14000、111,525 Gaussians、1352x1014 的 50 帧 test 序列上，`FullMlpPartition` 的 ABI-v5 mixed kernel（`persistent_blocks=0`）同时通过了整帧画质门槛和正式 `10x50` ABBA 交错速度测试。

- 正式速度中位数为 **115.333 FPS / 8.671 ms per frame**。
- 相对 two-stream 的 86.703 FPS，吞吐提升 **33.02%**；配对 FPS ratio 中位数为 1.3284，10,000 次 paired bootstrap 的 95% CI 为 **[1.3269, 1.3346]**。
- 相对 serial 的 83.229 FPS，吞吐提升 **38.57%**。
- 相对 serial 的画质变化为：PSNR `-0.000211 dB`、SSIM `-0.00000133`、LPIPS `+0.000000589`，均显著小于准入阈值。
- 30/30 个正式 child run 完成，`phase0_exit_condition.met=true`；三个合格候选中的排名为 `full_mlp_pb0 > two_stream > serial`。

本结论只比较本轮完成画质资格验证的 `serial`、`two_stream` 和 `full_mlp_pb0`。既有 `current_tacker` 没有在本轮做相同画质验证，因此被正式调度器以 `correctness_invalid` 排除；不能据此声称 PB=0 已经快于既有 Tacker 实现。

## 测试合同

- GPU：NVIDIA RTX A6000，SM 8.6，物理 GPU 2。
- CUDA runtime：12.4；driver module：570.124.06。
- 模型：`/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak`。
- 数据：`/data/qyfeng/datasets/n3dv/flame_steak`。
- 工作负载：test views 0--49；每个 child run 先执行 10 帧 warmup，再计时完整 50 帧序列。
- 正式速度协议：3 个合格候选，各 10 个独立 trial，共 30 个 child run；候选按 ABBA 规则交错，随机种子为 0。
- 主指标：`perf_counter` 包围整个序列，trial 结束时 `cudaDeviceSynchronize`；同时由 consumer-stream CUDA events 记录逐帧 completion。
- 置信区间：对同一轮次的候选结果做 10,000 次 paired bootstrap。
- 加载、renderer prepare 和 warmup 不在计时区间内。
- 每次 GPU 实验前均重新检查 NVML；GPU 2 无计算进程。GPU 0 上他人的 GR00T 服务未被使用或干预。

## 整帧画质验证

画质验证对 test views 0--49 全部 50 帧计算 PSNR、SSIM 和 LPIPS（VGG）。三个路径都实际进入请求模式，未发生 fallback。

| 模式 | 实际执行模式 | Mean PSNR | Mean SSIM | Mean LPIPS | 相对 serial | 结果 |
|---|---|---:|---:|---:|---|---|
| serial | serial | 32.574433 | 0.95773631 | 0.140365163 | baseline | PASS |
| two-stream | two_stream | 32.574433 | 0.95773631 | 0.140365163 | 三项与 serial 一致 | PASS |
| Full MLP PB=0 | tacker | 32.574222 | 0.95773498 | 0.140365753 | PSNR -0.000211 dB；SSIM -1.33e-6；LPIPS +5.89e-7 | **PASS** |

准入阈值为 PSNR drop `<= 0.05 dB`、SSIM drop `<= 1e-4`、LPIPS increase `<= 1e-4`。PB=0 同时通过实际模式、qualification、输出顺序、执行计数和画质五类检查。

PB=0 的 50 帧实际执行计数为：1 次 full-deformation prefill、49 次 prefix、49 次 mixed launch、49 次 suffix、1 次 solo-raster drain，共输出 50 帧；每个 deformation head 在每帧只计算一次。

## 正式 10x50 ABBA 结果

| 模式 | 实际执行模式 | 中位 FPS | 中位 ms/frame | min--max FPS | CV | CUDA peak allocated | CUDA peak reserved |
|---|---|---:|---:|---:|---:|---:|---:|
| serial | serial | 83.229 | 12.015 | 82.033--83.741 | 0.555% | 296.5 MiB | 506.0 MiB |
| two-stream | two_stream | 86.703 | 11.534 | 86.287--87.098 | 0.313% | 383.1 MiB | 968.0 MiB |
| Full MLP PB=0 | tacker | **115.333** | **8.671** | 114.342--116.320 | 0.472% | 458.0 MiB | 940.0 MiB |

十次 whole-sequence FPS 原始值：

- serial：83.741, 83.287, 83.272, 83.375, 83.186, 82.994, 83.346, 82.033, 82.863, 82.795
- two-stream：86.367, 87.037, 86.644, 87.098, 86.841, 86.519, 86.657, 86.750, 86.928, 86.287
- Full MLP PB=0：116.320, 116.037, 115.023, 115.397, 115.324, 114.342, 115.234, 115.054, 115.510, 115.342

PB=0 相对 two-stream 的十个配对 FPS ratio 为：1.3468, 1.3332, 1.3275, 1.3249, 1.3280, 1.3216, 1.3298, 1.3263, 1.3288, 1.3367。

## 前置筛查与参数选择

正式测试前执行了 5x50 速度筛查。PB=0 为 115.47 FPS，PB=336 为 109.23 FPS；PB=0 快 5.71%，因此正式 ABBA 只保留 direct logical grid 的 PB=0。该筛查不是正式准入结果。

当前 ABI-v5 mixed kernel 的查询资源为 384 threads/block、40 registers/thread、21,888 bytes static shared memory、4 active blocks/SM，理论 occupancy 为 1.0。

## 可复现性

- 顶层实现 commit：`2edb0452e937e12e4260f2f53d2e9dc0ec99c189`
- rasterizer submodule commit：`464096373b869fff713265bcdc34b1822277980d`
- 隔离测试快照 commit：`23205a292867db2867af39afc2ad0ec0f15720c9`（只用于远端测试快照的不可变 provenance，未推送且不影响原项目）
- `gaussian_renderer/tacker_pipeline.py` SHA-256：`22da9d1f7351240d097eeb31ceb2c6d406c4df89b4ebcdc93064028fe0de5220`
- `profile_render.py` SHA-256：`d7006d8b5fcce5b92abfb1257d01ad616c80cb54f7892f7c5e7310fa4f33d6e6`
- loaded rasterizer binary SHA-256：`daad1cec0065b596423046dfff9db533ea69d2561167e5924a6c5d9d5ab692d0`
- PB=0 qualification profile SHA-256：`8c981b15a1d18f49d4da3793566bf57a66e7904bf096c3ed9edd81cb728ebdab`
- 正式速度报告：`/data/qyfeng/full_mlp_partition_validation/20260920_e2e/formal_10x50_abba.json`，SHA-256 `f0760001ca3a6ecf6e255a4ee4f5e8336a5f8cf2d4754657dec3f33f1f85778e`
- 整帧画质报告：`/data/qyfeng/full_mlp_partition_validation/20260920_e2e/quality_50view_pb0.json`，SHA-256 `49a5e19aa157af2012e648cf68f11a8900b7bea9dd54c89118e7cad94c06203f`
- correctness manifest：`/data/qyfeng/full_mlp_partition_validation/20260920_e2e/full_mlp_formal_correctness.json`，SHA-256 `4a3cef914b57209d94f85ad51715662044d4f7b0cedef36d01ceae07f7c5ac2d`
- 正式 child-run 目录：`/data/qyfeng/full_mlp_partition_validation/20260920_e2e/formal-runs/full-mlp-10x50-abba-v1`

## 部署边界

测试期间 qualification profile 保持 `deployment.enabled=false`，避免验证过程自动改变生产选择。正式 driver 已在三个合格候选中选择 `full_mlp_pb0`；若要把它设为默认生产路径，仍应单独更新 deployment/admission 配置并保留回退开关。
