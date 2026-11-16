# v13 sampling / raster prepass：B 与 C 实验实现

默认 v13 配置保持原样。新增路径通过实验开关启用；B/C 共用
`hexplane_sample_task_device`，均采用 256 threads/block。

| 方案 | Sampling | Raster prepass | 阶段 kernel 数 |
| --- | --- | --- | --- |
| A | 原始 12 次 PyTorch FP32 `grid_sample` | 独立，双 stream 并发 | 13 |
| B | 一个 kernel 覆盖 12 planes | 独立，双 stream 并发 | 2 |
| C | sampling blocks | 同一 kernel 内的 raster blocks | 1 |

每个 sampling block 负责一个 plane 的最多 256 个点。任务编号先遍历
planes，再推进点区间。C 按任务总量比例交错映射两类 blocks；映射只分配
工作，不保证 block 执行顺序或 SM 位置。Raster 显式使用逻辑 tile 编号，
不会将混合 grid 的物理 block 编号当成 tile 编号。

输出维持 `[16,N]` 存储上的 `[N,16]` 转置 view，支持预分配。输入要求为
contiguous CUDA FP32 `[N,4]` 坐标与 `[1,16,H,W]` planes，采样语义为
bilinear / border / align_corners=True。B/C 采样使用相同 FP32 运算；与
原始 ATen 的 FMA 差异使用容差验证，B/C 之间要求逐位一致。Feature
reduction 和 MLP 保持独立。

## 构建

在独立 checkout 中分别构建 sampler 和 raster 扩展。例如 A6000：

```bash
cd tacker_ext
TACKER_BUILD_HEXPLANE_ONLY=1 TACKER_CUDA_ARCH=sm_86 python setup.py build_ext --inplace
cd ../submodules/depth-diff-gaussian-rasterization
TACKER_CUDA_ARCH=sm_86 python setup.py build_ext --inplace
```

Jetson 使用 `sm_87`。设置 `PYTHONPATH` 以优先导入该 checkout 的扩展。
共享服务器上测试前重新查询 GPU 显存和计算进程，并显式设置空闲的
`CUDA_VISIBLE_DEVICES`。按照项目 AGENTS.md 检查是否需要匹配驱动的预加载库。

## Python 入口

```python
from tacker_4dgs_head.hexplane import sample_planes, batch_resources

# B：一个 sampling kernel；prepass 在另一个 stream 独立执行。
samples = sample_planes(coords, planes, axes, outputs=preallocated_samples)

# C：一个混合 kernel 完成两类任务。
samples = rasterizer.render_prepared_with_hexplane_blocks(
    prepared, coords, planes, axes,
    raster_start=0, raster_end=split_tile,
    outputs=preallocated_samples,
)
```

通用 leaf 支持 1..12 planes。完整 renderer 的 B/C 模式固定全部 12 planes，
每 CTA 一个 sampling task，不与历史 partial-plane/512-thread 实验混用。

## 完整 renderer 实验开关

使用 `--execution-mode tacker --qualification-mode` 和
`full_mlp_overlap_graph` 对应 qualification/profile，同时启用 ABI-v12：

```bash
export TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12=1
export TACKER_HEXPLANE_FUSED_REDUCTION=1
export TACKER_HEXPLANE_REDUCTION_THREADS=256
export TACKER_RASTER_HEXPLANE_TASKS_PER_CTA=1
export TACKER_RASTER_HEXPLANE_MIXED_PLANES=12
export TACKER_EXPERIMENTAL_RASTER_HEXPLANE=batched       # B
# 或 block_fused                                      # C
```

B 在 coordinate Graph 后立即重放 sampling 和 reduction Graph，保留与
Raster prepare 的重叠窗口。C 等待 prepared raster 与 coords 就绪，完成
整个 prepass 和 sampling 后再重放 reduction Graph。现有两槽 renderer
仍在捕获的 coordinate/reduction Graph 之间提交 prepared mixed leaf，
不声称整条 renderer 流水线已捕获为单个 Graph。

执行次数沿用严格审计：B 每稳态帧一次 sampling Graph、一次独立 prepass；
C 每稳态帧一次 fused leaf、无额外 sampling Graph。默认配置与历史
`fused` / `separate` 两种 512-thread 实验的含义不变。

## 相同就绪边界的 A/B/C 阶段基准

`scripts/benchmark_sampling_prepass_bc.py` 从通过 qualification 的原始
`full_mlp_overlap_graph_pb0` 路径提取两个真实槽的最终输入。基准不使用
renderer 实验开关；提取时要求 `TACKER_EXPERIMENTAL_RASTER_HEXPLANE=0`。

```bash
python scripts/benchmark_sampling_prepass_bc.py \
  --runtime-root /absolute/path/to/isolated/checkout \
  --benchmark-output /absolute/path/to/new-results/benchmark.json \
  --abba-blocks 4 --repeats 40 --bench-warmup 10 \
  --execution-mode tacker --qualification-mode \
  --qualification-profile /absolute/path/to/full_mlp_overlap_graph.qualification.json \
  --metadata /absolute/path/to/new-results/extraction.json \
  --model_path /absolute/path/to/model --source_path /absolute/path/to/data \
  --configs arguments/dynerf/flame_steak.py \
  --iteration 14000 --split test --warmup 10 --frames 10 --trials 1 --quiet
```

三组统一使用 `torch.cuda.CUDAGraph` 重放、priority 0、无 SM 分区，测量从
两边输入均已就绪至两种任务均完成。该批次应独立报告，不与历史 v13 的
Driver `flags=8` 批次混合统计。

添加 `--node-priority-backend` 可让三组统一使用 v13 的 Driver
`USE_NODE_PRIORITY / flags=8` 后端，检查捕获节点的默认优先级、启动几何、
节点数量，以及实例化 flags 的直接读回或行为验证。2026-10-14 Jetson
正式批次采用该选项。
A/B 捕获双 stream fork/join，C 捕获单个 mixed kernel。B/C 共享同一
`[N,4]` 坐标地址，A 使用相同数值的原始双轴坐标。按预先固定的
`A,B,C,C,B,A` 对称顺序测量，分别报告 B/A、C/B、C/A。

所有主计时结束后才启用 profiler，按 `cudaGraphLaunch` correlation ID
严格检查 Graph 内为 13/2/1 个 kernels，拒绝额外 copy/gather。某些 PyTorch
版本在 Graph 外执行重放辅助 kernels；报告同时保存 Graph 内、Graph 外及
总 kernel 数，主计时包含这些辅助开销。输出先写 primary 记录，再写含
kernel 审计、资源和正确性结果的报告。原生采样与 B/C 输出采用容差，
raster 输出逐位一致；B/C 采样输出逐位一致。

阶段基准排除 coordinate/raster prepare、feature reduction 和 MLP。
需要另外测完整 renderer 才能判断提前采样窗口变化对总帧耗时的影响。
实现或 leaf 测试通过不代表性能改善或正式 profile 准入。

## 验证

```bash
python -m unittest discover -s tests -p 'test_hexplane_block_mapping.py'
python -m unittest discover -s tests -p 'test_raster_hexplane_graph.py'
python -m unittest discover -s tests -p 'test_sampling_prepass_bc.py'
python tacker_ext/tests/test_hexplane_batch_blocks_cuda.py -v
```

CUDA 测试覆盖 1/6/12 planes、边界/NaN/Inf、非整 tile 点数、空任务域、
guard 区、输出别名拒绝、非零 Raster 起点、不可见 Gaussian，以及
修改输入后的重复 Graph 重放和实际 13/2/1 kernel 数。

## 2026-10-13 实现验证

在独立 A6000 目录、sm_86、CUDA 12.4、PyTorch 2.4.1+cu124 上完成构建。
73 项针对性 CPU 测试和 18 项 CUDA 测试通过。真实 flame_steak 场景的
B/C 各三帧检查均通过 `full_mlp_overlap_graph_pb0` qualification 质量门槛；
两个真实槽的 A/B/C 阶段脚本也完成短流程验证，B/C sampling 逐位一致。

B/C 均为 36 registers/thread、0 local bytes/thread、occupancy API 容量
6 blocks/SM。B 使用 0 shared bytes，C 使用 10240 shared bytes。
Graph 内 kernels 为 13/2/1；该 PyTorch 版本的重放辅助 kernels 另行记录。

完整日志、原始 traces、源码/产物哈希和资源数据保存在
[验证摘要](../benchmarks/results/a6000_20261013/sampling_prepass_bc_implementation/summary.json)。
短流程仅使用一个对称计时块、每批两次重放，不形成正式性能结论。
Jetson sm_87 尚未构建或执行。

## 2026-10-14 Jetson 正式结果

Jetson AGX Orin，sm_87，PyTorch 2.8.0，CUDA 12.6，GPU 固定 1300.5 MHz。
场景为 flame_steak，111525 points，1352×1014；全部使用
`full_mlp_overlap_graph_pb0`，qualification SHA 为
`a13d87f6c4a22a24b20f83cb3c66a009e759b739b498c9ff55f8c35c9b82d23a`。

| 方案 | sampling+prepass 阶段 ms | 序列平均帧耗时 ms | FPS |
| --- | ---: | ---: | ---: |
| A：原始采样 | 6.1681 | 28.1052 | 35.5806 |
| B：统一采样 kernel | 6.1466 | 28.0403 | 35.6630 |
| C：block 级混合 | 5.8972 | 27.9935 | 35.7226 |

B/A 阶段耗时降低 0.35%，C/B 进一步降低 4.06%，C/A 总计降低 4.39%。
整帧序列平均耗时分别降低 0.23%、0.17%、0.40%，整帧收益较小。

辅助独立采样对照为 A=1.0429 ms、B=0.9120 ms，B 的 sampling 本身降低
12.55%。计时之后的诊断 trace 显示 A 与 prepass 的 kernel 生命周期
重叠均值为 0.3318 ms，B 为 0.0087 ms；B 的独立采样加速未全部转化为
并发阶段收益。生命周期区间含 stall，不代表指令活跃时间。

阶段正式测试为 3 个独立进程、每进程两个真实槽、每槽 6 轮 ABCCBA，
每批 80 次计时、20 次 warmup，共每组 5760 次计时重放。阶段主计时之后
才启用 profiler。整帧为两轮 ABCCBA，每组 4 个独立进程，各 warmup 20 帧、
计时 50 帧。两轮整帧 C/A 耗时分别降低 0.407% 和 0.387%。

18 项 CUDA 测试、三组各 10 帧 serial/tacker 质量检查，以及连续进程、
频率、源码/二进制哈希、实际 Graph 节点/启动几何审计均通过。
未将预检短运行或历史批次计入结果，默认 v13 配置保持原样。

[正式结果与原始记录](../benchmarks/results/jetson_20261014/v13_abc/summary.json)、
[阶段批次 CSV](../benchmarks/results/jetson_20261014/v13_abc/stage_batches.csv)、
[整帧独立运行 CSV](../benchmarks/results/jetson_20261014/v13_abc/renderer_runs.csv)。
完整固定协议见 `benchmarks/protocols/jetson_v13_abc_20261014.json`；复现入口为
`scripts/run_jetson_sampling_prepass_bc.py`，分析入口为
`scripts/analyze_jetson_sampling_prepass_bc.py`。
