# 普通 v13 与 gsplat tensor_v7_ca

这两条推理路线共享场景、HexPlane、Deform MLP 和 profile 校验代码。
普通 v13 是 ABI-v12 Raster prepass + mixed remainder、原生 HexPlane Graph
与 fused feature reduction 的组合；gsplat 采用 `tensor_v7_ca`，原生 HexPlane
双槽 Graph、device-v7 Tensor Core MLP 和 candidate-aware 任务配对。
gsplat 入口不启用普通路线的 Raster prepass 或 fused feature reduction。

## 目录与范围

| 目录 / 文件 | 内容 |
| --- | --- |
| `gaussian_renderer/tacker_pipeline.py` | 普通路线、Graph、双槽调度与准入 |
| `gaussian_renderer/gsplat_backend.py`, `gsplat_mixed.py` | gsplat 接口与融合调度 |
| `scene/`, `arguments/`, `utils/` | 场景、模型、配置和公共工具 |
| `tacker_ext/` | MLP、HexPlane、头文件、Python 绑定及构建 |
| `submodules/depth-diff-gaussian-rasterization/` | 已纳入主仓库的光栅化及 mixed CUDA 源码 |
| `submodules/simple-knn/` | 已纳入主仓库的近邻扩展 |
| `gsplat_mixed_ext/` | gsplat mixed kernel，保留历史与 wide 实验变体 |
| `profiles/` | 两份原始 disabled Graph qualification 配置 |
| `scripts/render_versions.py` | 本次新增的可移植图像导出入口 |

保留历史训练、验证、分析脚本供研究查阅，但它们不一定是可直接移植的入口。
特别是 `run_jetson_*` / `run_a6000_*` 中的旧机器路径以及旧报告的结果链接，
不代表这些数据随仓库发布。新入口不依赖服务器用户名、绝对实验路径、Git
子模块元数据或旧实验的已编译 `.so`。

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

`requirements-inference.txt` 不替换已有 PyTorch；原始 `requirements.txt`
中的 PyTorch 1.13.1 是上游旧训练环境，不应用于这两条现代推理路线。
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
重新生成和验证其自己的 qualification。现有 `materialize_tacker_workload_profiles.py`
可辅助重绑定工作负载，但不等同于完成数值、画质和性能验证。

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

历史 `render.py --execution-mode tacker` 走部署准入，不能把这里的 disabled
profile 当作 enabled 部署配置使用。两版本演示优先使用上面的新入口。
正式性能测试继续遵循 `full_mlp_overlap_graph` 与对应 profile，Graph 和 eager
数据分别报告。历史 sealed profiler 和实验脚本的 Git provenance 仍按原协议设计，
单仓库发布后使用它们前需另行适配和验证，不能伪造旧 Git 提交身份。

## 验证

不使用 GPU 的入口与任务分配检查：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_render_versions.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_gsplat_candidate_pairing.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_hexplane_block_mapping.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_raster_hexplane_graph.py
```

CUDA 数值测试仍在 `tacker_ext/tests` 与 `tests/test_*cuda.py`。部分历史 gsplat
对照测试需要单独编译的 legacy-v7 参考库；请读取相应测试的环境变量要求。
本次源码整理不声称重新完成 GPU 编译、数值/画质验证或性能复测。
依赖旧 Phase 0 / Phase 3.1 结果归档的三项测试在归档缺失时明确跳过；其他
CPU provenance 测试使用临时合成 fixture，不要求仓库中存在编译好的 CUDA 库。

## Git 与本地数据

`.gitignore` 排除实验结果、NSight 文件、模型、数据、缓存、编译产物和 notebook
自动备份。之前已跟踪的结果在本次提交中取消跟踪，原磁盘文件保留。普通 Git
提交不会删除历史提交里的结果；这里没有重写历史，也不需要 force push。
许可证和原始文件头随源码保留，详见根目录 `THIRD_PARTY_NOTICES.md`。
