# C3 最快 Tacker 配置：RTX A6000 flame_steak Nsight 逐 kernel 分析

本报告重新分析 2026-09-14 的既有 Nsight Systems 采集；本次未重新运行 GPU 实验。原始 CSV、metadata、summary 和 stdout 的 SHA-256 均与 Phase 3.1 封存 manifest 一致。

## 配置与执行核验

- 候选：`c3_packed_first_linear_pos_scales_rotations_opacity_shs_wg1_pb5440`。
- mixed kernel：`tacker_mix_render_packed_heads_v3`，worker_groups=1，persistent_blocks=5440。
- 主机：4A6000；使用单张物理 GPU 1，NVIDIA RTX A6000，CUDA 12.4、PyTorch 2.4.1+cu124、Nsight Systems 2023.4.4。
- 数据：flame_steak，iteration 14000，111,525 Gaussians，1352×1014，test 视角 0–49；预热 10 帧，捕获 50 帧。
- actual_execution_mode=tacker；无 fallback；50 输入和输出，1 次 full deformation、49 次 prefix/mixed/suffix、1 次 solo raster。

## 总体结果

- 39 个不同 kernel 符号，共 5,658 次执行，平均 113.16 次/帧。
- 全部 kernel 执行总和：454.566573 ms / 50 帧 = 9.09133146 ms/帧。
- mixed kernel：49 次，平均 3.64651182 ms/次；分摊到 50 帧为 3.57358158 ms/帧，占 39.3076%。
- 采集期间应用内完整序列计时：571.677700 ms，11.433554 ms/帧，87.4619 FPS。
- NVTX render_loop 区间：11.360064 ms/帧，88.0277 FPS；其区间边界与应用完整序列计时不同。
- Phase 4 独立 formal benchmark 为 100.508256 FPS；该值来自另一轮正式吞吐测量，不能作为本次 Nsight 采集的 FPS。

## 计时口径

- 次/帧 = 50 帧总调用次数 ÷ 50；平均 μs/次 = GPU duration 总和 ÷ 调用次数；累计 ms/帧 = GPU duration 总和 ÷ 50。
- 同一符号可能被不同算子位置或张量形状调用；单次平均值不是每次调用固定的耗时。
- GPU kernel duration 不包括 CPU launch API 时间、GPU memcpy/memset；在流水线并发时，duration 相加不等于墙钟延迟。
- mixed kernel 包含 Raster 与 head 工作，Nsight 符号汇总无法再把其内部两部分耗时独立拆开。
- 下表只缩短名称，不合并符号；完整名称和 min/median/max/stddev 在后文及 CSV。

## 所有 kernel，按每帧累计耗时降序

| # | kernel 简写 | 作用 | 50 帧调用数 | 次/帧 | 平均 μs/次 | 累计 ms/帧 | 占比 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `tacker_mix_render_packed_heads_v3` | 跨帧融合 Raster 与下一帧五个 head 的第一层 Linear | 49 | 0.98 | 3646.512 | 3.573582 | 39.308% |
| 2 | `vectorized_elementwise / launch_clamp_scalar<float>` | clamp/ReLU 类操作 | 354 | 7.08 | 143.292 | 1.014510 | 11.159% |
| 3 | `DeviceRadixSortOnesweepKernel` | 基数排序主 kernel | 300 | 6.00 | 162.361 | 0.974168 | 10.715% |
| 4 | `ampere_sgemm_32x128_tn` | FP32 矩阵乘法 | 200 | 4.00 | 228.267 | 0.913067 | 10.043% |
| 5 | `duplicateWithKeys` | 生成 Gaussian–tile 实例及排序键 | 50 | 1.00 | 472.926 | 0.472926 | 5.202% |
| 6 | `vectorized_elementwise / BinaryFunctor<Mul<float>>` | 张量逐元素乘法，向量化实现 | 550 | 11.00 | 29.728 | 0.327009 | 3.597% |
| 7 | `grid_sampler_2d_kernel<float,int>` | 二维特征平面采样 | 600 | 12.00 | 20.563 | 0.246750 | 2.714% |
| 8 | `CatArrayBatchedCopy_aligned16_contig / ndim=2` | 二维连续张量拼接 | 200 | 4.00 | 49.276 | 0.197102 | 2.168% |
| 9 | `elementwise<128,2> / BinaryFunctor<Mul<float>>` | 张量逐元素乘法，通用实现 | 350 | 7.00 | 19.874 | 0.139121 | 1.530% |
| 10 | `cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>` | FP32 矩阵乘法；原报告的 backbone MLP kernel | 50 | 1.00 | 131.149 | 0.131149 | 1.443% |
| 11 | `unrolled_elementwise / direct_copy<c10::Half,cast>` | 带 dtype 转换的拷贝，对应 packed head 输入 FP16 转换 | 49 | 0.98 | 126.784 | 0.124248 | 1.367% |
| 12 | `vectorized_elementwise / CUDAFunctor_add<float>` | 加法，向量化实现 | 200 | 4.00 | 26.959 | 0.107835 | 1.186% |
| 13 | `internal::gemvx::kernel<…float…>` | 矩阵–向量乘法 GEMV | 50 | 1.00 | 82.454 | 0.082454 | 0.907% |
| 14 | `index_elementwise_kernel / index_kernel_impl` | 张量索引、取子集 | 600 | 12.00 | 6.739 | 0.080865 | 0.889% |
| 15 | `CatArrayBatchedCopy_aligned16_contig / ndim=3` | 三维连续张量拼接 | 50 | 1.00 | 78.683 | 0.078683 | 0.865% |
| 16 | `vectorized_elementwise / FillFunctor<float>` | 浮点张量填充 | 450 | 9.00 | 8.130 | 0.073169 | 0.805% |
| 17 | `ampere_sgemm_128x64_tn` | FP32 矩阵乘法 | 5 | 0.10 | 605.693 | 0.060569 | 0.666% |
| 18 | `vectorized_elementwise / cos_kernel_cuda` | 逐元素余弦 | 150 | 3.00 | 19.804 | 0.059413 | 0.654% |
| 19 | `DeviceRadixSortHistogramKernel` | 基数排序直方图 | 50 | 1.00 | 57.151 | 0.057151 | 0.629% |
| 20 | `preprocessCUDA<3>` | Gaussian 投影、协方差、SH→RGB、tile 覆盖预处理 | 50 | 1.00 | 55.926 | 0.055926 | 0.615% |
| 21 | `identifyTileRanges` | 确定各 tile 在排序结果中的范围 | 50 | 1.00 | 55.244 | 0.055244 | 0.608% |
| 22 | `vectorized_elementwise / sin_kernel_cuda` | 逐元素正弦 | 150 | 3.00 | 16.681 | 0.050044 | 0.550% |
| 23 | `CatArrayBatchedCopy / ndim=2` | 二维张量拼接，通用实现 | 50 | 1.00 | 49.339 | 0.049339 | 0.543% |
| 24 | `renderCUDA<3>` | tile 光栅化、透明度混合，输出 RGB/depth | 1 | 0.02 | 2395.272 | 0.047905 | 0.527% |
| 25 | `vectorized_elementwise / AUnaryFunctor<Mul<float>>` | 张量与标量相乘 | 150 | 3.00 | 14.605 | 0.043815 | 0.482% |
| 26 | `elementwise<128,2> / CUDAFunctor_add<float>` | 加法，通用实现 | 100 | 2.00 | 13.363 | 0.026726 | 0.294% |
| 27 | `reduce_kernel / NormTwoOps<float>` | L2 范数归约 | 50 | 1.00 | 8.616 | 0.008616 | 0.095% |
| 28 | `elementwise<128,2> / direct_copy_kernel_cuda<float>` | 张量逐元素拷贝 | 150 | 3.00 | 2.390 | 0.007169 | 0.079% |
| 29 | `vectorized_elementwise / exp_kernel_cuda` | 逐元素指数 | 50 | 1.00 | 5.620 | 0.005620 | 0.062% |
| 30 | `DeviceScanKernel` | 前缀和扫描 | 50 | 1.00 | 4.643 | 0.004643 | 0.051% |
| 31 | `elementwise<128,2> / BinaryFunctor<Div<float>>` | 逐元素除法 | 50 | 1.00 | 4.150 | 0.004150 | 0.046% |
| 32 | `vectorized_elementwise / CUDAFunctorOnSelf_add<float>` | 标量加法 | 100 | 2.00 | 2.045 | 0.004091 | 0.045% |
| 33 | `vectorized_elementwise / compare_scalar_kernel<int>` | 整数与标量比较 | 50 | 1.00 | 2.763 | 0.002763 | 0.030% |
| 34 | `vectorized_elementwise / sigmoid_kernel_cuda` | 逐元素 sigmoid | 50 | 1.00 | 2.630 | 0.002630 | 0.029% |
| 35 | `DeviceRadixSortExclusiveSumKernel` | 排序内部的排他前缀和 | 50 | 1.00 | 2.054 | 0.002054 | 0.023% |
| 36 | `vectorized_elementwise / reciprocal_kernel_cuda` | 逐元素倒数 | 50 | 1.00 | 1.973 | 0.001973 | 0.022% |
| 37 | `unrolled_elementwise / CUDAFunctor_add<float>` | 加法，展开实现 | 50 | 1.00 | 1.757 | 0.001757 | 0.019% |
| 38 | `vectorized_elementwise / FillFunctor<int>` | 整数张量填充 | 50 | 1.00 | 1.548 | 0.001548 | 0.017% |
| 39 | `DeviceScanInitKernel` | 初始化前缀扫描状态 | 50 | 1.00 | 1.546 | 0.001546 | 0.017% |

## 与原始串行采集比较

串行历史采集使用同一主机的物理 GPU 2，C3 使用 GPU 1；模型路径、iteration、Gaussian 数、分辨率及前 50 个 test 视角一致。采集日期和源码快照不同，因此下面用于说明 kernel 构成变化，不作为同轮受控加速比。

- 全部 kernel duration：11.25882512 → 9.09133146 ms/帧，减少 19.2515%。
- SGEMM128：250 → 5 次/50帧；第一帧保留完整 deformation，之后五个 head 的第一层 Linear 进入 mixed kernel。
- renderCUDA：50 → 1 次/50帧；49 次渲染由 mixed 承担，独立 render 用于流水线收尾。
- clamp/ReLU：550 → 354 次/50帧；C3 将五个 first-linear 输入的 ReLU 共享，减少 4×49=196 次独立调用。
- 新增 mixed 49 次和带类型转换的 copy 49 次。相较原始 6,050 次总调用，减少 196+245+49-49-49=392 次，结果为 5,658 次。
- 首层之外的 Linear、GEMV、ReLU、HexPlane 采样以及 Raster 投影、排序等仍独立执行。

## GPU memory operations（另列，不计入 kernel 合计）

| operation | 50 帧次数 | ms/帧 |
|---|---:|---:|
| [CUDA memcpy Device-to-Device] | 50 | 0.043827 |
| [CUDA memset] | 450 | 0.016654 |
| [CUDA memcpy Host-to-Device] | 800 | 0.007017 |
| [CUDA memcpy Device-to-Host] | 50 | 0.001185 |

## 完整 kernel 符号及单次分布

下列分布统计来自 Nsight 原始摘要；单位 μs。

### 1. tacker_mix_render_packed_heads_v3

```text
tacker_mix_render_packed_heads_v3
```

min / median / max / stddev = 3531.5790 / 3584.5070 / 3801.8230 / 98.5008 μs。

### 2. vectorized_elementwise / launch_clamp_scalar<float>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

min / median / max / stddev = 1.4720 / 165.2830 / 169.8270 / 57.5807 μs。

### 3. DeviceRadixSortOnesweepKernel

```text
void cub::CUB_200301_860_NS::DeviceRadixSortOnesweepKernel<cub::CUB_200301_860_NS::DeviceRadixSortPolicy<unsigned long, unsigned int, unsigned int>::Policy900, (bool)0, unsigned long, unsigned int, unsigned int, int, int, cub::CUB_200301_860_NS::detail::identity_decomposer_t>(T7 *, T7 *, T5 *, const T5 *, T3 *, const T3 *, T4 *, const T4 *, T6, int, int, T8)
```

min / median / max / stddev = 159.8110 / 161.5865 / 168.9310 / 2.1183 μs。

### 4. ampere_sgemm_32x128_tn

```text
ampere_sgemm_32x128_tn
```

min / median / max / stddev = 150.4990 / 159.0750 / 456.2320 / 122.6511 μs。

### 5. duplicateWithKeys

```text
duplicateWithKeys(int, const float2 *, const float *, const unsigned int *, unsigned long *, unsigned int *, int *, dim3)
```

min / median / max / stddev = 456.9040 / 469.3520 / 488.6480 / 10.1335 μs。

### 6. vectorized_elementwise / BinaryFunctor<Mul<float>>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, at::detail::Array<char *, (int)3>>(int, T2, T3)
```

min / median / max / stddev = 3.0720 / 32.3200 / 33.7280 / 8.3602 μs。

### 7. grid_sampler_2d_kernel<float,int>

```text
void at::native::<unnamed>::grid_sampler_2d_kernel<float, int>(T2, at::cuda::detail::TensorInfo<const T1, T2>, at::cuda::detail::TensorInfo<const T1, T2>, at::cuda::detail::TensorInfo<T1, T2>, at::native::detail::GridSamplerInterpolation, at::native::detail::GridSamplerPadding, bool)
```

min / median / max / stddev = 14.9760 / 19.4880 / 33.3440 / 4.4227 μs。

### 8. CatArrayBatchedCopy_aligned16_contig / ndim=2

```text
void at::native::<unnamed>::CatArrayBatchedCopy_aligned16_contig<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)2, (int)128, (int)1>(T1 *, at::native::<unnamed>::CatArrInputTensorMetadata<T1, T2, T4, T5>, at::native::<unnamed>::TensorSizeStride<T2, (unsigned int)4>, int, T2)
```

min / median / max / stddev = 5.5370 / 39.8245 / 112.3220 / 38.3322 μs。

### 9. elementwise<128,2> / BinaryFunctor<Mul<float>>

```text
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
```

min / median / max / stddev = 3.3600 / 12.1600 / 67.9040 / 20.0033 μs。

### 10. cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>

```text
void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(T1::Params)
```

min / median / max / stddev = 127.5540 / 129.4260 / 135.6190 / 2.8592 μs。

### 11. unrolled_elementwise / direct_copy<c10::Half,cast>

```text
void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 10)]::operator ()() const::[lambda(c10::Half) (instance 1)], at::detail::Array<char *, (int)2>, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T3, T4, T5, T6)
```

min / median / max / stddev = 125.8900 / 126.6260 / 128.4510 / 0.4913 μs。

### 12. vectorized_elementwise / CUDAFunctor_add<float>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>>(int, T2, T3)
```

min / median / max / stddev = 1.8560 / 4.6720 / 97.3770 / 39.9627 μs。

### 13. internal::gemvx::kernel<…float…>

```text
std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, float, float, float, float, (bool)0, (bool)1, (bool)1, (bool)0, (int)5, (bool)0, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<float>, float>>(T13)
```

min / median / max / stddev = 82.0490 / 82.4330 / 83.1700 / 0.2573 μs。

### 14. index_elementwise_kernel / index_kernel_impl

```text
void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &)::[lambda(int) (instance 1)]>(long, T3)
```

min / median / max / stddev = 4.6390 / 6.8480 / 7.8720 / 0.6389 μs。

### 15. CatArrayBatchedCopy_aligned16_contig / ndim=3

```text
void at::native::<unnamed>::CatArrayBatchedCopy_aligned16_contig<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)128, (int)1>(T1 *, at::native::<unnamed>::CatArrInputTensorMetadata<T1, T2, T4, T5>, at::native::<unnamed>::TensorSizeStride<T2, (unsigned int)4>, int, T2)
```

min / median / max / stddev = 77.7930 / 78.5935 / 80.0010 / 0.4993 μs。

### 16. vectorized_elementwise / FillFunctor<float>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)
```

min / median / max / stddev = 1.1510 / 2.0320 / 30.8810 / 10.3558 μs。

### 17. ampere_sgemm_128x64_tn

```text
ampere_sgemm_128x64_tn
```

min / median / max / stddev = 593.3220 / 611.0180 / 616.2020 / 11.3892 μs。

### 18. vectorized_elementwise / cos_kernel_cuda

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::cos_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

min / median / max / stddev = 5.6000 / 12.0000 / 41.9210 / 15.3965 μs。

### 19. DeviceRadixSortHistogramKernel

```text
void cub::CUB_200301_860_NS::DeviceRadixSortHistogramKernel<cub::CUB_200301_860_NS::DeviceRadixSortPolicy<unsigned long, unsigned int, unsigned int>::Policy900, (bool)0, unsigned long, unsigned int, cub::CUB_200301_860_NS::detail::identity_decomposer_t>(T4 *, const T3 *, T4, int, int, T5)
```

min / median / max / stddev = 56.5450 / 57.0250 / 58.7540 / 0.4729 μs。

### 20. preprocessCUDA<3>

```text
void preprocessCUDA<(int)3>(int, int, int, const float *, const glm::vec<(int)3, float, (glm::qualifier)0> *, float, const glm::vec<(int)4, float, (glm::qualifier)0> *, const float *, const float *, bool *, const float *, const float *, const float *, const float *, const glm::vec<(int)3, float, (glm::qualifier)0> *, int, int, float, float, float, float, int *, float2 *, float *, float *, float *, float4 *, dim3, unsigned int *, bool)
```

min / median / max / stddev = 53.9530 / 55.6010 / 58.4010 / 1.0863 μs。

### 21. identifyTileRanges

```text
identifyTileRanges(int, unsigned long *, uint2 *)
```

min / median / max / stddev = 54.9450 / 55.2335 / 55.6480 / 0.1537 μs。

### 22. vectorized_elementwise / sin_kernel_cuda

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::sin_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

min / median / max / stddev = 3.6800 / 5.0240 / 41.4090 / 17.1568 μs。

### 23. CatArrayBatchedCopy / ndim=2

```text
void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)2, (int)64, (int)64>(T1 *, at::native::<unnamed>::CatArrInputTensorMetadata<T1, T2, T4, T5>, at::native::<unnamed>::TensorSizeStride<T2, (unsigned int)4>, int, T2)
```

min / median / max / stddev = 48.7370 / 49.3120 / 49.9210 / 0.2866 μs。

### 24. renderCUDA<3>

```text
void renderCUDA<(unsigned int)3>(const uint2 *, const unsigned int *, int, int, const float2 *, const float *, const float *, const float4 *, float *, unsigned int *, const float *, float *, float *)
```

min / median / max / stddev = 2395.2720 / 2395.2720 / 2395.2720 / 0.0000 μs。

### 25. vectorized_elementwise / AUnaryFunctor<Mul<float>>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, at::detail::Array<char *, (int)2>>(int, T2, T3)
```

min / median / max / stddev = 1.2800 / 20.9920 / 21.8570 / 9.3934 μs。

### 26. elementwise<128,2> / CUDAFunctor_add<float>

```text
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
```

min / median / max / stddev = 11.9360 / 12.9285 / 14.6240 / 0.8257 μs。

### 27. reduce_kernel / NormTwoOps<float>

```text
void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::NormTwoOps<float, float, float>, unsigned int, float, (int)4>>(T3)
```

min / median / max / stddev = 8.3200 / 8.5280 / 8.9610 / 0.2015 μs。

### 28. elementwise<128,2> / direct_copy_kernel_cuda<float>

```text
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
```

min / median / max / stddev = 1.7920 / 2.1760 / 3.1680 / 0.4749 μs。

### 29. vectorized_elementwise / exp_kernel_cuda

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::exp_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

min / median / max / stddev = 5.0880 / 5.6160 / 6.0480 / 0.2319 μs。

### 30. DeviceScanKernel

```text
void cub::CUB_200301_860_NS::DeviceScanKernel<cub::CUB_200301_860_NS::DeviceScanPolicy<unsigned int, cuda::std::__4::plus<void>>::Policy900, unsigned int *, unsigned int *, cub::CUB_200301_860_NS::ScanTileState<unsigned int, (bool)1>, cuda::std::__4::plus<void>, cub::CUB_200301_860_NS::NullType, int, unsigned int>(T2, T3, T4, int, T5, T6, T7)
```

min / median / max / stddev = 4.4480 / 4.6240 / 4.9280 / 0.1317 μs。

### 31. elementwise<128,2> / BinaryFunctor<Div<float>>

```text
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
```

min / median / max / stddev = 3.8710 / 4.1440 / 4.5120 / 0.1387 μs。

### 32. vectorized_elementwise / CUDAFunctorOnSelf_add<float>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, at::detail::Array<char *, (int)2>>(int, T2, T3)
```

min / median / max / stddev = 1.8560 / 2.0480 / 2.5280 / 0.1181 μs。

### 33. vectorized_elementwise / compare_scalar_kernel<int>

```text
void at::native::vectorized_elementwise_kernel<(int)4, void at::native::compare_scalar_kernel<int>(at::TensorIteratorBase &, at::native::<unnamed>::OpType, T1)::[lambda(int) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

min / median / max / stddev = 2.5920 / 2.7525 / 3.0080 / 0.0966 μs。

### 34. vectorized_elementwise / sigmoid_kernel_cuda

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

min / median / max / stddev = 2.4960 / 2.6240 / 2.8170 / 0.0904 μs。

### 35. DeviceRadixSortExclusiveSumKernel

```text
void cub::CUB_200301_860_NS::DeviceRadixSortExclusiveSumKernel<cub::CUB_200301_860_NS::DeviceRadixSortPolicy<unsigned long, unsigned int, unsigned int>::Policy900, unsigned int>(T2 *)
```

min / median / max / stddev = 1.8880 / 2.0480 / 2.2390 / 0.0804 μs。

### 36. vectorized_elementwise / reciprocal_kernel_cuda

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::reciprocal_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

min / median / max / stddev = 1.8230 / 1.9525 / 2.0800 / 0.0566 μs。

### 37. unrolled_elementwise / CUDAFunctor_add<float>

```text
void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>, TrivialOffsetCalculator<(int)2, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, T1, T2, T3, T4, T5, T6)
```

min / median / max / stddev = 1.4720 / 1.7920 / 1.9520 / 0.1154 μs。

### 38. vectorized_elementwise / FillFunctor<int>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, at::detail::Array<char *, (int)1>>(int, T2, T3)
```

min / median / max / stddev = 1.4720 / 1.5360 / 1.6640 / 0.0586 μs。

### 39. DeviceScanInitKernel

```text
void cub::CUB_200301_860_NS::DeviceScanInitKernel<cub::CUB_200301_860_NS::ScanTileState<unsigned int, (bool)1>>(T1, int)
```

min / median / max / stddev = 1.4400 / 1.5360 / 1.6960 / 0.0667 μs。

## 原始数据与重分析文件

- `4dgs_render_tacker_stats.csv`：原始 Nsight CSV。
- `tacker_profile_metadata.json`：配置、执行次数与应用计时。
- `tacker_summary.json`：封存的原始聚合结果。
- `profile.stdout.txt`：原始运行日志。
- `kernel-details.csv`：39 个 kernel 的完整耗时和分布。
- `serial-comparison.csv`：按完整 kernel 符号对齐的串行/C3 对比。
- `analysis-summary.json`：重分析结果、核验哈希和口径。
