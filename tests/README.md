# 核心测试

从仓库根目录运行：

```bash
python3 tests/run_core_tests.py
python3 tests/run_core_tests.py --list
```

默认只运行 CPU 测试，不需要 PyTorch 或 GPU，覆盖运行入口、参数、渲染调度、
Graph 生命周期、候选任务配对、HexPlane 映射、ABI 与参数验证。
训练回归覆盖所有场景配置、最终模型保存、DataLoader 新批次选择、checkpoint
阶段识别与 coarse/fine 调度；使用替身验证控制逻辑，不执行训练或 GPU 工作。
`test_render_entry.py` 还覆盖通用 `scripts/render.py` 的自定义模型／场景参数、后端分发、
融合 profile 要求、帧数检查与图片写入失败处理，使用替身而非 GPU 渲染。
入口导入路径还会在隔离子进程中从项目外目录检查，不依赖现有 `PYTHONPATH`。
HexPlane 映射测试需要 C++ 编译器；缺失时 unittest 会明确标记跳过。
测试入口显式列出发布文件，避免加载原工作区仍保留但已取消跟踪的实验测试。
`test_tacker_phase2.py` 同时提供 full-MLP 测试使用的公共 fixture，因此保留。

## CUDA 数值测试

先按[运行指南](../docs/INFERENCE_VERSIONS.md)安装环境并编译当前设备的扩展。
gsplat 的核心数值测试还需要独立 device-v7 MLP 对照库。源码来自原实验中的
独立调用适配器，保留在 `cuda/deform_mlp_v7_reference.cu`，调用共用的
`tacker_ext/include/deform_mlp_device.cuh`。这是当前源码对照，不代表旧实验
封存二进制的身份，也不构成性能证据。

```bash
# 从仓库根目录执行；A6000 使用 -arch=sm_86。
nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC -arch=sm_87 \
  -Itacker_ext/include tests/cuda/deform_mlp_v7_reference.cu \
  -o tests/cuda/libdeform_v7_reference.so

# 仅在已确认空闲的目标设备上执行，0 替换为该设备编号。
CUDA_VISIBLE_DEVICES=0 python3 tests/run_core_tests.py --cuda
```

Jetson 上代码、缓存、构建与输出应放在有空间的可写数据盘；可用
`TMPDIR`、`TORCH_EXTENSIONS_DIR`、`CUDA_CACHE_PATH`、`XDG_CACHE_HOME`
指定缓存路径。不要在不同 CUDA/设备架构间混用已编译扩展。

参考库默认读取上面的本地路径，也可用 `GSPLAT_V7_REFERENCE_LIBRARY` 指定。
兼容旧变量 `GSPLAT_LEGACY_V7_LIBRARY`，仅供显式选用已有参考库。
测试保留 gsplat `tensor_v7` 及早期实现作为数值对照；两版运行入口始终选择
普通 v13 或 `tensor_v7_ca`。CUDA 测试验证数值、边界输入、流和 Graph 行为，
不输出正式性能结论。缺少 CUDA 或参考库时不会当作数值验证成功。

本次精简验证以 CPU 回归、干净文件集和入口 dry-run 为范围；GPU 编译、CUDA
数值测试、画质与端到端性能需要在目标设备上单独执行。
