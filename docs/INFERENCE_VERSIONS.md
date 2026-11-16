# 普通 v13 与 gsplat tensor_v7_ca

这两条推理路线共享场景、HexPlane、Deform MLP 和 profile 校验代码。
普通 v13 是 ABI-v12 Raster prepass + mixed remainder、原生 HexPlane Graph
与 fused feature reduction 的组合；gsplat 采用 `tensor_v7_ca`，原生 HexPlane
双槽 Graph、device-v7 Tensor Core MLP 和 candidate-aware 任务配对。
gsplat 入口不启用普通路线的 Raster prepass 或 fused feature reduction。

本页描述固定配置的融合入口。需要用自己的兼容场景和 checkpoint 进行基础渲染时，
请使用 [scripts/render.py 通用渲染指南](GENERAL_RENDERING.md)；其串行／双流模式不要求
本页的固定场景 profile。通用入口与两版融合入口同时保留。

## 目录与范围

| 目录 / 文件 | 内容 |
| --- | --- |
| `gaussian_renderer/tacker_pipeline.py` | 普通路线、Graph、双槽调度与准入 |
| `gaussian_renderer/gsplat_backend.py`, `gsplat_mixed.py` | gsplat 接口与融合调度 |
| `scene/`, `arguments/`, `utils/` | 场景、模型、配置和公共工具 |
| `tacker_ext/` | MLP、HexPlane、头文件、Python 绑定及构建 |
| `submodules/depth-diff-gaussian-rasterization/` | 已纳入主仓库的光栅化及 mixed CUDA 源码 |
| `submodules/simple-knn/` | 已纳入主仓库的近邻扩展 |
| `gsplat_mixed_ext/` | gsplat mixed kernel、JIT 构建及核心测试使用的对照实现 |
| `profiles/` | 两份原始 disabled Graph qualification 配置 |
| `scripts/render_versions.py` | 两版共用的可移植图像导出入口 |
| `scripts/render.py` | 通用场景的串行／双流渲染及图片、视频导出 |
| `scripts/build_inference_extensions.sh` | 两种设备架构的扩展构建入口 |
| `tests/run_core_tests.py` | 明确选择核心测试，避免运行本地留存的旧实验测试 |

本文介绍现成数据与 checkpoint 的两版推理；训练和必要的数据准备工具已恢复，
使用方法见[训练指南](TRAINING.md)。benchmark、实验协议与启动器、NCU 分析和
旧报告仍取消 Git 跟踪；原工作区保留这些文件，并通过 `.gitignore` 防止再次误提交。
运行入口不依赖服务器用户名、绝对实验路径、Git 子模块元数据或旧实验的 `.so`。
共用 CUDA 扩展中的历史 ABI、兼容符号和测试对照实现保留，未通过删函数改变内核。

## 安装

需要 Linux / WSL2、Python 3.10 或兼容版本、CUDA toolkit（含 `nvcc`）、C++
编译器，以及与设备/驱动匹配的 CUDA PyTorch 和 torchvision。已记录的历史
环境包括 RTX A6000 的 PyTorch 2.4.1 + CUDA 12.4，以及 Jetson Orin 的
PyTorch 2.8.0 + CUDA 12.6；这是历史环境说明，不是本次重新验证的安装结果。

先为你的平台安装匹配的 PyTorch/torchvision，尤其 Jetson 应使用匹配其
JetPack 的构建。然后在仓库根目录运行：

```bash
python3 -m pip install -r requirements-inference.txt
python3 -m pip install --no-build-isolation -r requirements-gsplat.txt

# 二选一；不要在同一 Python 环境混用不同架构的已编译扩展。
bash scripts/build_inference_extensions.sh sm_87  # Jetson AGX Orin
# bash scripts/build_inference_extensions.sh sm_86  # RTX A6000
```

不需要 `git submodule update`。目录名 `submodules` 保留以兼容现有 include
和构建路径，Git 中这些目录已是普通文件。GLM 仅保留编译必需的头文件与许可。
使用上述 Python 扩展构建脚本；历史 CMake 目标未作为本次发布构建入口验证。

`requirements-inference.txt` 不替换已有 PyTorch；仅使用这里的依赖文件，
原工作区留存的旧训练 `requirements.txt` 不属于当前发布版本。
`MAX_JOBS` 默认为 2，可按机器内存调整。运行前确认选中的 GPU 空闲；脚本
不会自动抢占进程、修改驱动或锁定 GPU 频率。系统驱动/用户库需已兼容。

## 数据、checkpoint 与 qualification

本仓库不包含数据集或模型。需要准备 DyNeRF/Neu3D 格式数据，以及同一次训练的：

```text
MODEL/
  point_cloud/iteration_14000/
    point_cloud.ply
    deformation.pth
    deformation_table.pth     # 如果原 checkpoint 存在，请保留
    deformation_accum.pth     # 如果原 checkpoint 存在，请保留
```

新入口通过显式 `--config` 读取网络/场景参数，不读取旧 `cfg_args` 中的机器路径。
默认配置是 `arguments/dynerf/flame_steak.py`。请使用与 checkpoint 一致的配置。

| 配置 | 设备与工作负载 |
| --- | --- |
| `profiles/flame_steak_sm87.qualification.json` | Orin / sm_87 |
| `profiles/flame_steak_sm86.qualification.json` | NVIDIA RTX A6000 / sm_86 |

两份配置都绑定 `flame_steak`、iteration 14000、111525 个高斯、1352×1014、
`full_mlp_overlap_graph_pb0`。配置保留原始字节/哈希与资源约束，始终 disabled，
仅用于显式 qualification 执行。它们是必需的运行配置，不是性能结果归档。
模型数量、图像分辨率、GPU 名称/架构或重新编译后的资源不匹配时会报错；
不要为了启动而手改哈希、开启 deployment 或移除检查。其他场景/设备需要
重新生成和验证其自己的 qualification；本精简版不包含实验配置生成与准入工具。
`tacker_profiles/raster_head_sm86.json` 是共用 API 及核心测试引用的旧 disabled
默认模板，不替代上述两份显式运行配置。

## 运行两条路线

以下路径是示例，替换为实际目录。共享机器先检查负载并选定空闲 GPU。
Jetson 的代码、输出、临时目录和缓存应放在可写数据盘；入口把运行缓存置于
`--output/cache`。`--output` 必须是新的目录，且位于模型与数据目录之外。

```bash
# 普通 v13：显式 ABI-v12 + fused reduction + full_mlp_overlap_graph。
python3 scripts/render_versions.py \
  --variant v13 \
  --profile profiles/flame_steak_sm87.qualification.json \
  --model /data/models/flame_steak \
  --source /data/datasets/flame_steak \
  --config arguments/dynerf/flame_steak.py \
  --device 0 --frames 50 --output /data/runs/v13

# gsplat：显式 tensor_v7_ca，physical_blocks=0，candidate pairing 开启。
python3 scripts/render_versions.py \
  --variant gsplat_ca \
  --profile profiles/flame_steak_sm87.qualification.json \
  --model /data/models/flame_steak \
  --source /data/datasets/flame_steak \
  --config arguments/dynerf/flame_steak.py \
  --device 0 --frames 50 --output /data/runs/gsplat_ca
```

A6000 使用 sm_86 配置及其对应构建。`--device` 接受一个 CUDA 设备编号或 UUID。
命令加 `--dry-run` 可离线检查版本、profile、环境开关和输出路径，不需要
PyTorch，不创建输出，不执行 GPU 工作。至少渲染两帧才能覆盖 mixed 阶段。
正式执行前会清理继承的 `TACKER_*` 实验开关，以固定这两种配置。

输出包括 `renders/*.png`、`plan.json` 和成功后才生成的 `execution.json`。
后者记录 qualification 文件哈希、版本、环境、Graph/mixed 执行次数。
发生 fallback、Graph 未执行或帧数不符时直接报错；失败目录可能含部分图片，
没有 `execution.json` 就不是完整成功。入口不计算 FPS，逐帧导出带有 CPU
传输与编码开销，不能用其墙钟耗时评价融合性能。

两版均使用上面的显式 qualification 入口，不能把 disabled profile 当作
enabled 部署配置使用。通用 `scripts/render.py` 显示单次循环 FPS，但没有正式性能实验的
预热、重复和配对控制；本发布版本不包含 benchmark 工具。如另行开展性能实验，
仍需确认实际命中 `full_mlp_overlap_graph` 与对应 profile，并区分 Graph / eager。

## 验证

不使用 GPU 的核心回归（入口、参数、渲染调度、Graph、任务配对、ABI 与验证）：

```bash
python3 tests/run_core_tests.py
```

不要在留有旧文件的工作区对整个测试目录做无筛选的 discovery；显式测试入口
只选取随仓库发布的测试，在原目录和新克隆中使用同一组文件。
CUDA 数值测试保留于根目录及扩展测试目录；gsplat 对照用的 device-v7 独立源码
放在 `tests/cuda/`，可在当前设备编译，不要求旧服务器上的封存库。
构建参考库和 GPU 测试方法见 [核心测试说明](../tests/README.md)。
CPU 检查不代表完成 GPU 编译、数值/画质验证或性能复测。

## Git 与本地数据

`.gitignore` 排除历史实验代码与报告、实验结果、NSight 文件、模型、数据、缓存、
编译产物和 notebook 自动备份。取消跟踪的文件仍保留在原工作区，核心测试采用
明确文件清单以隔离它们。普通 Git 提交不会删除历史提交里的大文件；本次不重写历史。
许可证和原始文件头随源码保留，详见根目录 `THIRD_PARTY_NOTICES.md`。
