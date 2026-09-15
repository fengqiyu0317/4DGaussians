# flame_steak：RTX A6000 逐 kernel 耗时明细

来源：已有 Nsight Systems `cuda_gpu_kern_sum` 原始数据；本次仅展开统计，没有重新采样。
原始串行路径；4A6000 的单张 GPU 2；iteration 14000；111,525 Gaussians；1352×1014；test split；预热 10 帧，捕获 50 帧。

共 37 个不同 kernel 符号，6,050 次执行，即 121 次/帧；总执行时间 562.941256 ms，即 11.25882512 ms/帧。

## 计时口径

- 单次平均耗时 = 该符号所有调用的 GPU duration 总和 ÷ 调用次数。
- 每帧累计耗时 = 该符号所有调用的 GPU duration 总和 ÷ 50。
- 同一符号可以被不同算子位置或不同张量形状复用；平均值不能视作每一次调用的固定延迟。
- 表中名称仅缩短命名空间和模板参数，37 行与原始 37 个符号逐一对应。完整符号见下方附录。
- 这些是按符号汇总的数据，不是 6,050 次 launch 的逐条时间线；不含 CPU 提交时间和 CUDA memcpy/memset。
- 上一版类别统计把 GEMV（0.08218844 ms/帧）归入 other；此处单独列出。

## 按每帧累计耗时降序

| # | kernel 简写 | 作用 | 次/帧 | 平均 μs/次 | 累计 ms/帧 | kernel 占比 |
|---:|---|---|---:|---:|---:|---:|
| 1 | `ampere_sgemm_128x64_tn` | FP32 矩阵乘法 | 5 | 580.556 | 2.902778 | 25.782% |
| 2 | `renderCUDA<3>` | tile 光栅化、透明度混合，输出 RGB/depth | 1 | 2414.867 | 2.414867 | 21.449% |
| 3 | `vectorized_elementwise / launch_clamp_scalar<float>` | clamp/ReLU 类操作 | 11 | 153.395 | 1.687349 | 14.987% |
| 4 | `DeviceRadixSortOnesweepKernel` | 基数排序主 kernel | 6 | 162.303 | 0.973817 | 8.649% |
| 5 | `ampere_sgemm_32x128_tn` | FP32 矩阵乘法 | 4 | 228.438 | 0.913754 | 8.116% |
| 6 | `duplicateWithKeys` | 生成 Gaussian–tile 实例及排序键 | 1 | 470.900 | 0.470900 | 4.182% |
| 7 | `vectorized_elementwise / BinaryFunctor<Mul<float>>` | 张量逐元素乘法，向量化实现 | 11 | 29.582 | 0.325406 | 2.890% |
| 8 | `grid_sampler_2d_kernel<float,int>` | 二维特征平面采样 | 12 | 20.541 | 0.246498 | 2.189% |
| 9 | `CatArrayBatchedCopy_aligned16_contig / ndim=2` | 二维连续张量拼接 | 4 | 49.163 | 0.196650 | 1.747% |
| 10 | `elementwise<128,2> / BinaryFunctor<Mul<float>>` | 张量逐元素乘法，通用实现 | 7 | 19.212 | 0.134485 | 1.194% |
| 11 | `cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>` | FP32 矩阵乘法；原报告的 backbone MLP kernel | 1 | 130.304 | 0.130304 | 1.157% |
| 12 | `vectorized_elementwise / CUDAFunctor_add<float>` | 加法，向量化实现 | 4 | 26.590 | 0.106362 | 0.945% |
| 13 | `internal::gemvx::kernel<…float…>` | 矩阵–向量乘法 GEMV | 1 | 82.188 | 0.082188 | 0.730% |
| 14 | `index_elementwise_kernel / index_kernel_impl` | 张量索引、取子集 | 12 | 6.550 | 0.078596 | 0.698% |
| 15 | `CatArrayBatchedCopy_aligned16_contig / ndim=3` | 三维连续张量拼接 | 1 | 78.508 | 0.078508 | 0.697% |
| 16 | `vectorized_elementwise / FillFunctor<float>` | 浮点张量填充 | 9 | 8.216 | 0.073945 | 0.657% |
| 17 | `vectorized_elementwise / cos_kernel_cuda` | 逐元素余弦 | 3 | 19.659 | 0.058978 | 0.524% |
| 18 | `DeviceRadixSortHistogramKernel` | 基数排序直方图 | 1 | 57.229 | 0.057229 | 0.508% |
| 19 | `preprocessCUDA<3>` | Gaussian 投影、协方差、SH→RGB、tile 覆盖预处理 | 1 | 55.851 | 0.055851 | 0.496% |
| 20 | `identifyTileRanges` | 确定各 tile 在排序结果中的范围 | 1 | 55.216 | 0.055216 | 0.490% |
| 21 | `vectorized_elementwise / sin_kernel_cuda` | 逐元素正弦 | 3 | 16.659 | 0.049976 | 0.444% |
| 22 | `CatArrayBatchedCopy / ndim=2` | 二维张量拼接，通用实现 | 1 | 49.149 | 0.049149 | 0.437% |
| 23 | `vectorized_elementwise / AUnaryFunctor<Mul<float>>` | 张量与标量相乘 | 3 | 14.572 | 0.043715 | 0.388% |
| 24 | `elementwise<128,2> / CUDAFunctor_add<float>` | 加法，通用实现 | 2 | 12.402 | 0.024803 | 0.220% |
| 25 | `reduce_kernel / NormTwoOps<float>` | L2 范数归约 | 1 | 8.509 | 0.008509 | 0.076% |
| 26 | `elementwise<128,2> / direct_copy_kernel_cuda<float>` | 张量逐元素拷贝 | 3 | 2.294 | 0.006883 | 0.061% |
| 27 | `vectorized_elementwise / exp_kernel_cuda` | 逐元素指数 | 1 | 5.576 | 0.005576 | 0.050% |
| 28 | `DeviceScanKernel` | 前缀和扫描 | 1 | 4.584 | 0.004584 | 0.041% |
| 29 | `vectorized_elementwise / CUDAFunctorOnSelf_add<float>` | 标量加法 | 2 | 2.027 | 0.004053 | 0.036% |
| 30 | `elementwise<128,2> / BinaryFunctor<Div<float>>` | 逐元素除法 | 1 | 4.002 | 0.004002 | 0.036% |
| 31 | `vectorized_elementwise / compare_scalar_kernel<int>` | 整数与标量比较 | 1 | 2.737 | 0.002737 | 0.024% |
| 32 | `vectorized_elementwise / sigmoid_kernel_cuda` | 逐元素 sigmoid | 1 | 2.584 | 0.002584 | 0.023% |
| 33 | `DeviceRadixSortExclusiveSumKernel` | 排序内部的排他前缀和 | 1 | 2.015 | 0.002015 | 0.018% |
| 34 | `vectorized_elementwise / reciprocal_kernel_cuda` | 逐元素倒数 | 1 | 1.835 | 0.001835 | 0.016% |
| 35 | `unrolled_elementwise / CUDAFunctor_add<float>` | 加法，展开实现 | 1 | 1.708 | 0.001708 | 0.015% |
| 36 | `vectorized_elementwise / FillFunctor<int>` | 整数张量填充 | 1 | 1.510 | 0.001510 | 0.013% |
| 37 | `DeviceScanInitKernel` | 初始化前缀扫描状态 | 1 | 1.507 | 0.001507 | 0.013% |

## 完整 kernel 符号与单次分布

min / median / max / stddev 均来自 Nsight 原始摘要，单位 μs。

### 1. ampere_sgemm_128x64_tn

```text
ampere_sgemm_128x64_tn
```

调用次数：250；单次 min / median / max / stddev：560.3820 / 575.8220 / 616.4140 / 13.1056 μs。

### 2. renderCUDA<3>

```text
void renderCUDA<(unsigned int)3>(const uint2 *, const unsigned int *, int, int, const float2 *, const float *, const float *, const float4 *, float *, unsigned int *, const float *, float *, float *)
```

调用次数：50；单次 min / median / max / stddev：2373.6870 / 2406.5350 / 2493.2390 / 29.3110 μs。

### 3. vectorized_elementwise / launch_clamp_scalar<float>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

调用次数：550；单次 min / median / max / stddev：1.5670 / 168.5750 / 171.4240 / 48.0403 μs。

### 4. DeviceRadixSortOnesweepKernel

```text
void cub::CUB_200301_860_NS::DeviceRadixSortOnesweepKernel<cub::CUB_200301_860_NS::DeviceRadixSortPolicy<unsigned long, unsigned int, unsigned int>::Policy900, (bool)0, unsigned long, unsigned int, unsigned int, int, int, cub::CUB_200301_860_NS::detail::identity_decomposer_t>(T7 *, T7 *, T5 *, const T5 *, T3 *, const T3 *, T4 *, const T4 *, T6, int, int, T8)
```

调用次数：300；单次 min / median / max / stddev：159.6790 / 161.4555 / 170.0160 / 2.2367 μs。

### 5. ampere_sgemm_32x128_tn

```text
ampere_sgemm_32x128_tn
```

调用次数：200；单次 min / median / max / stddev：150.7830 / 158.6560 / 460.4780 / 122.8504 μs。

### 6. duplicateWithKeys

```text
duplicateWithKeys(int, const float2 *, const float *, const unsigned int *, unsigned long *, unsigned int *, int *, dim3)
```

调用次数：50；单次 min / median / max / stddev：460.4460 / 469.1665 / 490.8460 / 7.0354 μs。

### 7. vectorized_elementwise / BinaryFunctor<Mul<float>>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, at::detail::Array<char *, (int)3>>(int, T2, T3)
```

调用次数：550；单次 min / median / max / stddev：2.9770 / 32.1910 / 33.5040 / 8.3709 μs。

### 8. grid_sampler_2d_kernel<float,int>

```text
void at::native::<unnamed>::grid_sampler_2d_kernel<float, int>(T2, at::cuda::detail::TensorInfo<const T1, T2>, at::cuda::detail::TensorInfo<const T1, T2>, at::cuda::detail::TensorInfo<T1, T2>, at::native::detail::GridSamplerInterpolation, at::native::detail::GridSamplerPadding, bool)
```

调用次数：600；单次 min / median / max / stddev：14.9760 / 19.1680 / 33.6960 / 4.3486 μs。

### 9. CatArrayBatchedCopy_aligned16_contig / ndim=2

```text
void at::native::<unnamed>::CatArrayBatchedCopy_aligned16_contig<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)2, (int)128, (int)1>(T1 *, at::native::<unnamed>::CatArrInputTensorMetadata<T1, T2, T4, T5>, at::native::<unnamed>::TensorSizeStride<T2, (unsigned int)4>, int, T2)
```

调用次数：200；单次 min / median / max / stddev：5.6320 / 40.0320 / 112.7990 / 38.2374 μs。

### 10. elementwise<128,2> / BinaryFunctor<Mul<float>>

```text
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
```

调用次数：350；单次 min / median / max / stddev：3.2310 / 10.7200 / 67.1040 / 20.2729 μs。

### 11. cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>

```text
void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(T1::Params)
```

调用次数：50；单次 min / median / max / stddev：127.4240 / 129.7435 / 135.8400 / 1.9620 μs。

### 12. vectorized_elementwise / CUDAFunctor_add<float>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>>(int, T2, T3)
```

调用次数：200；单次 min / median / max / stddev：1.6960 / 4.3200 / 97.2790 / 40.1527 μs。

### 13. internal::gemvx::kernel<…float…>

```text
std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, float, float, float, float, (bool)0, (bool)1, (bool)1, (bool)0, (int)5, (bool)0, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<float>, float>>(T13)
```

调用次数：50；单次 min / median / max / stddev：81.8550 / 82.1920 / 82.6240 / 0.1865 μs。

### 14. index_elementwise_kernel / index_kernel_impl

```text
void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)4>>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>)::[lambda(char *, const char *, long) (instance 1)]>(at::TensorIteratorBase &, c10::ArrayRef<long>, c10::ArrayRef<long>, const T1 &)::[lambda(int) (instance 1)]>(long, T3)
```

调用次数：600；单次 min / median / max / stddev：4.6400 / 6.6560 / 7.4880 / 0.5667 μs。

### 15. CatArrayBatchedCopy_aligned16_contig / ndim=3

```text
void at::native::<unnamed>::CatArrayBatchedCopy_aligned16_contig<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)3, (int)128, (int)1>(T1 *, at::native::<unnamed>::CatArrInputTensorMetadata<T1, T2, T4, T5>, at::native::<unnamed>::TensorSizeStride<T2, (unsigned int)4>, int, T2)
```

调用次数：50；单次 min / median / max / stddev：77.6640 / 78.5275 / 79.7440 / 0.4103 μs。

### 16. vectorized_elementwise / FillFunctor<float>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)
```

调用次数：450；单次 min / median / max / stddev：1.1510 / 2.0480 / 30.8800 / 10.4511 μs。

### 17. vectorized_elementwise / cos_kernel_cuda

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::cos_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

调用次数：150；单次 min / median / max / stddev：5.8240 / 11.8400 / 41.8880 / 15.3509 μs。

### 18. DeviceRadixSortHistogramKernel

```text
void cub::CUB_200301_860_NS::DeviceRadixSortHistogramKernel<cub::CUB_200301_860_NS::DeviceRadixSortPolicy<unsigned long, unsigned int, unsigned int>::Policy900, (bool)0, unsigned long, unsigned int, cub::CUB_200301_860_NS::detail::identity_decomposer_t>(T4 *, const T3 *, T4, int, int, T5)
```

调用次数：50；单次 min / median / max / stddev：56.4800 / 57.0240 / 59.2000 / 0.6520 μs。

### 19. preprocessCUDA<3>

```text
void preprocessCUDA<(int)3>(int, int, int, const float *, const glm::vec<(int)3, float, (glm::qualifier)0> *, float, const glm::vec<(int)4, float, (glm::qualifier)0> *, const float *, const float *, bool *, const float *, const float *, const float *, const float *, const glm::vec<(int)3, float, (glm::qualifier)0> *, int, int, float, float, float, float, int *, float2 *, float *, float *, float *, float4 *, dim3, unsigned int *, bool)
```

调用次数：50；单次 min / median / max / stddev：54.2080 / 55.7440 / 58.0800 / 1.0684 μs。

### 20. identifyTileRanges

```text
identifyTileRanges(int, unsigned long *, uint2 *)
```

调用次数：50；单次 min / median / max / stddev：54.9750 / 55.2000 / 55.4240 / 0.1091 μs。

### 21. vectorized_elementwise / sin_kernel_cuda

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::sin_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

调用次数：150；单次 min / median / max / stddev：3.5520 / 4.8320 / 41.6640 / 17.2687 μs。

### 22. CatArrayBatchedCopy / ndim=2

```text
void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>, unsigned int, (int)2, (int)64, (int)64>(T1 *, at::native::<unnamed>::CatArrInputTensorMetadata<T1, T2, T4, T5>, at::native::<unnamed>::TensorSizeStride<T2, (unsigned int)4>, int, T2)
```

调用次数：50；单次 min / median / max / stddev：48.6400 / 49.1360 / 49.7590 / 0.2505 μs。

### 23. vectorized_elementwise / AUnaryFunctor<Mul<float>>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, at::detail::Array<char *, (int)2>>(int, T2, T3)
```

调用次数：150；单次 min / median / max / stddev：1.3110 / 20.9920 / 21.6960 / 9.3892 μs。

### 24. elementwise<128,2> / CUDAFunctor_add<float>

```text
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
```

调用次数：100；单次 min / median / max / stddev：11.7120 / 12.4480 / 12.9920 / 0.2422 μs。

### 25. reduce_kernel / NormTwoOps<float>

```text
void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::NormTwoOps<float, float, float>, unsigned int, float, (int)4>>(T3)
```

调用次数：50；单次 min / median / max / stddev：8.2230 / 8.4640 / 8.8960 / 0.1643 μs。

### 26. elementwise<128,2> / direct_copy_kernel_cuda<float>

```text
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
```

调用次数：150；单次 min / median / max / stddev：1.7600 / 2.2400 / 2.9770 / 0.4221 μs。

### 27. vectorized_elementwise / exp_kernel_cuda

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::exp_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

调用次数：50；单次 min / median / max / stddev：5.1840 / 5.5365 / 5.8870 / 0.1408 μs。

### 28. DeviceScanKernel

```text
void cub::CUB_200301_860_NS::DeviceScanKernel<cub::CUB_200301_860_NS::DeviceScanPolicy<unsigned int, cuda::std::__4::plus<void>>::Policy900, unsigned int *, unsigned int *, cub::CUB_200301_860_NS::ScanTileState<unsigned int, (bool)1>, cuda::std::__4::plus<void>, cub::CUB_200301_860_NS::NullType, int, unsigned int>(T2, T3, T4, int, T5, T6, T7)
```

调用次数：50；单次 min / median / max / stddev：4.3510 / 4.5760 / 4.8640 / 0.1028 μs。

### 29. vectorized_elementwise / CUDAFunctorOnSelf_add<float>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, at::detail::Array<char *, (int)2>>(int, T2, T3)
```

调用次数：100；单次 min / median / max / stddev：1.8880 / 2.0320 / 2.2720 / 0.0984 μs。

### 30. elementwise<128,2> / BinaryFunctor<Div<float>>

```text
void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
```

调用次数：50；单次 min / median / max / stddev：3.8710 / 3.9680 / 4.2250 / 0.0969 μs。

### 31. vectorized_elementwise / compare_scalar_kernel<int>

```text
void at::native::vectorized_elementwise_kernel<(int)4, void at::native::compare_scalar_kernel<int>(at::TensorIteratorBase &, at::native::<unnamed>::OpType, T1)::[lambda(int) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

调用次数：50；单次 min / median / max / stddev：2.5920 / 2.7200 / 3.0400 / 0.0878 μs。

### 32. vectorized_elementwise / sigmoid_kernel_cuda

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

调用次数：50；单次 min / median / max / stddev：2.4000 / 2.5600 / 2.8160 / 0.1038 μs。

### 33. DeviceRadixSortExclusiveSumKernel

```text
void cub::CUB_200301_860_NS::DeviceRadixSortExclusiveSumKernel<cub::CUB_200301_860_NS::DeviceRadixSortPolicy<unsigned long, unsigned int, unsigned int>::Policy900, unsigned int>(T2 *)
```

调用次数：50；单次 min / median / max / stddev：1.8880 / 2.0155 / 2.2720 / 0.0640 μs。

### 34. vectorized_elementwise / reciprocal_kernel_cuda

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::reciprocal_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)
```

调用次数：50；单次 min / median / max / stddev：1.6320 / 1.8240 / 1.9850 / 0.0509 μs。

### 35. unrolled_elementwise / CUDAFunctor_add<float>

```text
void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>, TrivialOffsetCalculator<(int)2, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, T1, T2, T3, T4, T5, T6)
```

调用次数：50；单次 min / median / max / stddev：1.5680 / 1.7120 / 1.9200 / 0.0844 μs。

### 36. vectorized_elementwise / FillFunctor<int>

```text
void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, at::detail::Array<char *, (int)1>>(int, T2, T3)
```

调用次数：50；单次 min / median / max / stddev：1.4710 / 1.5040 / 1.6000 / 0.0399 μs。

### 37. DeviceScanInitKernel

```text
void cub::CUB_200301_860_NS::DeviceScanInitKernel<cub::CUB_200301_860_NS::ScanTileState<unsigned int, (bool)1>>(T1, int)
```

调用次数：50；单次 min / median / max / stddev：1.4080 / 1.5040 / 1.6960 / 0.0537 μs。
