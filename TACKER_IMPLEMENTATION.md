# 4DGaussians × Tacker 实现说明

此集成仅用于推理，并采用 fail-closed（任何条件不满足时均不启用）策略。仓库的默认执行模式仍为 `serial`；`two_stream` 和 `tacker` 都不会改变训练行为，也不会被隐式启用。

## 首个固定工作负载

首个物理配置有意限定为：

- `flame_steak`，使用 dynerf 加载器，迭代次数为 14000；
- 在 1352 × 1014 分辨率下使用 111,525 个高斯点；
- 一张 NVIDIA RTX A6000（`sm_86`）；
- 光栅化器基准提交必须严格为 `e49506654e8e11ed8a62d22bcb693e943fdecacf`；
- 首个物理配置将 Raster 的最终渲染任务与 `pos_deform[1]`（即 `Linear(128, 128)` 层）融合执行；该层的输入与权重采用 FP16，偏置、累加过程及输出采用 FP32。

任何条件不匹配时，都会将完整序列交给 `TwoStreamRenderer`。如果该后端同样不受支持，则继续回退到未经修改的串行渲染器。

## 任务图与物理配对

两个逻辑任务图如下：

```text
Raster LC:      setup -> preprocess -> scan -> duplicate -> sort -> ranges -> render
Deformation BE: prefix -> selected pos head -> selected suffix -> activation -> state
```

Raster 内部不透明的准备流程保持不变。首个物理候选方案仅将 Raster 最终的 render 叶子任务，与 `pos_deform` 中真实的第一个 `Linear(128,128)` 层配对：

```text
D(0) 完整形变
    |
前缀 D(1) -- prefix_ready
    |                 \
其余四个 head          R(0) + 被选中的 head D(1)（一个 384 线程 CTA）
    |                 /
    +---- mixed_done -> 被选中的后缀 D(1) -> ready D(1)
                                             |
                                      下一次稳态迭代
```

Raster 占用线程 `[0,256)`，并使用具有 256 个参与者的命名屏障 1。head 占用 `[256,384)`，并采用 warp 局部的 WMMA 同步。两个槽位和四种事件角色（`ready`、`prefix_ready`、`mixed_done`、`raster_done`）共同保护依赖关系、分配器生命周期和槽位复用。`D(0)` 是唯一一次完整的形变预填充；之后被选中的 head 不会再由 PyTorch 重复计算。

## 组件

- `gaussian_renderer/__init__.py`：拆分后的串行 API，以及可感知 stream 的双槽位渲染器。
- `gaussian_renderer/tacker_pipeline.py`：经准入的物理流水线、严格的模型/工作负载门控、配置验证、资格认证引导和回退报告。
- `tacker_ext/`：独立的 solo/GPTB FP16 head 扩展和稳定的内核 ABI。
- `submodules/depth-diff-gaussian-rasterization/`：基于当前 stream 的 Raster 路径、Render+head 混合内核、能力查询和 ABI 清单。
- `../Tacker/src/runtime/`：可复用的 `libtacker_runtime` 控制平面库；远程部署时使用 `../Tacker-4DGS-runtime`，以保留现有的 Tacker 脏工作区，其中包含 TaskGraph、注册表、配置/QoS 门控和双图调度。
- `profile_render.py` 和 `render.py`：显式集成执行模式；两者默认均为串行模式。
- `profile_tacker_leaves.py`：使用真实输入对叶子任务/Raster 进行资格认证测量。
- `scripts/validate_tacker_modes.py`：验证同一视角下的图像质量。
- `scripts/benchmark_tacker_admission.py`：生成 fail-closed 报告和经准入的配置。
- `scripts/run_tacker_qualification.sh`：一条命令即可完成远程预检、构建、CUDA 测试、测量、准入和常规配置验证。

PyTorch 路径使用由 PyTorch 管理的 stream/event，并直接调用物理 CUDA ABI。该路径有意不把独立的 C++ runtime 链接进 Torch 扩展，从而避免第二个 libstdc++/Torch C++ ABI 边界。独立 runtime 仍可供原生启动器复用，并强制执行相同的双图调度概念。

## 构建与 CPU 约定

在已配置的远程主机上，源代码应保存在 `/home/qyfeng` 下，大型数据集、检查点、Nsight 文件和报告应保存在 `/data/qyfeng` 下。受支持的工具链必须能够构建 `sm_86`。该主机上锁定的资格认证环境为 Python 3.10、PyTorch 2.4.1+cu124 和 CUDA toolkit 12.4（上游项目 README 记录的原始基线为 PyTorch 1.13.1+cu116）。请按以下顺序构建扩展：

```bash
cd /home/qyfeng/4DGaussians/tacker_ext
python setup.py build_ext --inplace

cd /home/qyfeng/4DGaussians/submodules/depth-diff-gaussian-rasterization
TACKER_4DGS_HEAD_INCLUDE=/home/qyfeng/4DGaussians/tacker_ext/include \
  python setup.py build_ext --inplace
```

原生 runtime 是独立的：

```bash
cmake -S /home/qyfeng/Tacker-4DGS-runtime/src \
  -B /home/qyfeng/Tacker-4DGS-runtime/build-runtime \
  -DTACKER_BUILD_LEGACY=OFF -DTACKER_BUILD_TESTS=ON
cmake --build /home/qyfeng/Tacker-4DGS-runtime/build-runtime --parallel
ctest --test-dir /home/qyfeng/Tacker-4DGS-runtime/build-runtime --output-on-failure
```

在注明支持的情况下，无需 Torch 或 CUDA 即可运行仅 CPU 的约定测试：

```bash
cd /home/qyfeng/4DGaussians
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=tacker_ext PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tacker_ext/tests -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s submodules/depth-diff-gaussian-rasterization/tests -p 'test_*.py' -v
```

## 执行模式

使用已准入配置进行常规渲染：

```bash
python render.py ... \
  --execution-mode tacker \
  --workload-name flame_steak \
  --tacker-profile /home/qyfeng/4DGaussians/tacker_profiles/raster_head_sm86.admitted.json
```

使用 `--execution-mode two_stream` 可选择 stream 基线模式。省略该选项时使用串行模式。资格认证模式仅通过验证/性能分析入口开放，需要显式提供内存中的配置覆盖，且不会启用签入仓库的模板。

有关单张 A6000 的完整命令和输入 schema，请参阅 `tacker_profiles/README.md`。准入要求必须同时满足以下全部条件：

- PSNR 平均下降不超过 0.05 dB；
- SSIM 平均下降不超过 `1e-4`；
- LPIPS 平均上升不超过 `1e-4`；
- Raster LC 减速不超过 5%；
- 混合叶子任务必须严格快于单独执行 Raster 和 head 的耗时之和；
- Tacker 端到端 p50 不得慢于 two-stream p50。

## 验证状态

截至 2026-08-31，完整的 1--10 资格认证脚本已在配置了四张 A6000 的主机上通过，运行工作负载时仅向其暴露 GPU 0。封存的运行使用 Python 3.10.20、PyTorch 2.4.1+cu124、CUDA 12.4、迭代次数 14000、111,525 个高斯点、50 个实测测试视角，并设置 `persistent_blocks=7000`。原生 runtime CTest、强制构建 CUDA 扩展、ptxas 检查、全部 CUDA 测试、图像质量、Raster QoS、混合叶子任务加速、端到端速度、准入，以及一次常规的非资格认证 Tacker 渲染均已通过。

准入测量结果如下：

- Raster p50 为 4.177920 ms；混合 p50 为 4.365824 ms；head p50 为 0.261120 ms；
- Raster 减速 4.497552%（上限为 5%），且混合叶子任务快于 4.439040 ms 的独立执行耗时之和；
- Tacker 端到端 p50 为 11.489281 ms，two-stream 则为 11.648499 ms，比值为 0.986331；
- PSNR 下降 0.000139 dB，SSIM 下降 1.232624e-6，LPIPS 上升 1.055002e-7。

最终报告位于 `/data/qyfeng/tacker_admission_full_20260831/admission-report.json`；启用的配置位于 `/home/qyfeng/4DGaussians/tacker_profiles/raster_head_sm86.admitted.full_20260831.json`；常规模式验证结果位于 `/data/qyfeng/tacker_admission_full_20260831/tacker-admitted-verification.json`。常规渲染报告显示 `actual_execution_mode=tacker`，未发生回退，也没有请求或执行资格认证模式。

最终的本地 CPU 回归测试包含 72 个根目录测试、27 个 head 扩展测试（本地跳过其中 7 个仅限 GPU 的测试），以及 20 个 Raster 扩展测试（本地跳过其中 6 个仅限 GPU 的测试）。签入仓库的配置有意保持 `enabled: false, valid: false`：它是一个 fail-closed 资格认证模板；只有生成的、与测量结果绑定且通过准入的配置才会启用。
