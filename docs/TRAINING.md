# 训练与训练后推理

`train.py` 恢复原有 4DGaussians 的 coarse / fine 两阶段训练，使用可微
`diff_gaussian` 光栅化，包含损失反传、优化器、增密／剪枝、模型保存、checkpoint
恢复、训练中 L1 / PSNR 检查和 TensorBoard 日志。无需 inference qualification。
普通 v13 与 gsplat 融合版仍用于推理；训练入口不使用融合内核或 gsplat 后端。

## 环境

先安装与 GPU、CUDA toolkit 匹配的 CUDA PyTorch 和 torchvision，从仓库根目录执行：

```bash
python3 -m pip install -r requirements-training.txt
bash scripts/build_inference_extensions.sh sm_87  # Orin；A6000 使用 sm_86
# 可选：训练后使用 gsplat 推理。
python3 -m pip install --no-build-isolation -r requirements-gsplat.txt
python3 train.py --help
```

构建脚本虽然名为 `build_inference_extensions.sh`，其中的光栅化扩展保留
反向传播，可供训练复用。无需 Git 子模块操作。训练依赖复用
`requirements-inference.txt`，另加 `lpips` 和 `tensorboard`；没有重新引入
旧 `requirements.txt` 的 PyTorch 1.13.1 / MMCV 固定依赖，不覆盖已有 PyTorch。
配置使用仓库自身的 `_base_` 加载器。原训练循环的 LPIPS 损失仍未启用；
当前训练损失为 L1、可选 SSIM 及网格正则项。

共享设备先确认 GPU 空闲，再用 `CUDA_VISIBLE_DEVICES` 指定设备。Jetson 的
代码、数据、模型、构建与缓存都放在 `/data/qyfeng`，例如：

```bash
mkdir -p /data/qyfeng/cache/4dgs/{tmp,torch,extensions,cuda,xdg,matplotlib}
export TMPDIR=/data/qyfeng/cache/4dgs/tmp
export TORCH_HOME=/data/qyfeng/cache/4dgs/torch
export TORCH_EXTENSIONS_DIR=/data/qyfeng/cache/4dgs/extensions
export CUDA_CACHE_PATH=/data/qyfeng/cache/4dgs/cuda
export XDG_CACHE_HOME=/data/qyfeng/cache/4dgs/xdg
export MPLCONFIGDIR=/data/qyfeng/cache/4dgs/matplotlib
```

## 数据与配置

数据集、相机标定和预训练模型不随仓库提供。训练需要与读取器兼容的图像、
相机与时间信息，真实场景还需要初始点云。

| 格式 | 主要输入 | 配置目录 |
| --- | --- | --- |
| D-NeRF / Blender | 图像、`transforms_train.json`、`transforms_test.json`；无点云时读取器可生成初始点云 | `arguments/dnerf/` |
| DyNeRF / Neu3D | `poses_bounds.npy`、`camXX/images/*.png`、`points3D_downsample2.ply` | `arguments/dynerf/` |
| HyperNeRF / Nerfies | `dataset.json`、`scene.json`、`metadata.json`、`camera/`、`rgb/2x/`、`points3D_downsample2.ply` | `arguments/hypernerf/` |
| DyCheck | 需准备成当前 Nerfies 读取器兼容的相机／图像布局 | `arguments/dycheck/` |
| MultipleView | 多视角帧、`sparse_/`、`poses_bounds_multipleview.npy`、`points3D_multipleview.ply` | `arguments/multipleview/` |

23 份场景配置及其 `_base_` 文件均已恢复。更换数据时选择对应配置；配置存在
不代表任意原始数据布局都能直接读取。

DyNeRF 的抽帧、已有姿态转 COLMAP、点云下采样工具已恢复。已有完整数据时跳过：

```bash
# Open3D 仅用于下采样；已有点云的训练不需要它。
python3 -m pip install -r requirements-preprocess.txt
python3 scripts/preprocess_dynerf.py --datadir /data/qyfeng/datasets/flame_steak
# 另需安装 COLMAP 命令；继承所选 GPU，不强制选择 GPU 0。
CUDA_VISIBLE_DEVICES=0 bash colmap.sh /data/qyfeng/datasets/flame_steak llff
python3 scripts/downsample_point.py \
  /data/qyfeng/datasets/flame_steak/colmap/dense/workspace/fused.ply \
  /data/qyfeng/datasets/flame_steak/points3D_downsample2.ply
```

`colmap.sh` 也接受 `blender` / `hypernerf`，调用对应姿态转换脚本。
它拒绝覆盖已有 `sparse_`、`image_colmap` 或 `colmap` 输出，不删除旧重建数据。
COLMAP 的 GPU 支持与命令选项需与本机安装相符；此次恢复未重新执行重建流程。

## 开始训练

以 DyNeRF 为例，路径替换为实际数据和一个新的模型输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  -s /data/qyfeng/datasets/flame_steak \
  -m /data/qyfeng/models/flame_steak_new \
  --configs arguments/dynerf/flame_steak.py \
  --disable_gui --num_workers 0 \
  --test_iterations 3000 7000 14000 \
  --save_iterations 3000 7000 14000 \
  --checkpoint_iterations 3000 7000 14000
```

D-NeRF 可将数据与配置改为 `arguments/dnerf/bouncingballs.py`；HyperNeRF
使用对应的 `arguments/hypernerf/*.py`。不传 `-m` 时必须提供 `--expname`，
模型写入 `output/<expname>`。已有模型目录会被复用，同名结果可能覆盖，
新实验应使用新的输出目录。

保留原有优先级：`--configs` 覆盖命令行对应参数组的值。例如改变训练轮数应
修改配置中的 `OptimizationParams.iterations`，或建立继承原配置的自定义配置；
只传 `--iterations` 可能被配置覆盖。`--num_workers` 默认 0，可按资源增加。
`--disable_gui` 避免开启查看器端口；需要兼容查看器连接时去掉此项，用
`--ip` / `--port` 指定监听地址。查看器程序本身不包含在仓库中。

## 输出与恢复

```text
MODEL/
  cfg_args
  point_cloud/coarse_iteration_<N>/...
  point_cloud/iteration_<N>/
    point_cloud.ply
    deformation.pth
    deformation_table.pth
    deformation_accum.pth
  chkpnt_coarse_<N>.pth
  chkpnt_fine_<N>.pth
  events.out.tfevents.*
```

两阶段结束时均保存模型；fine 最终迭代数在配置合并后确定。
`--checkpoint_iterations` 控制包含优化器状态的恢复文件，与用于推理的模型目录
用途不同。`--test_iterations` 控制训练中评估，`tensorboard --logdir MODEL`
可查看日志；`--render_process` 还可导出训练过程预览。

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  -s /data/qyfeng/datasets/flame_steak \
  -m /data/qyfeng/models/flame_steak_new \
  --configs arguments/dynerf/flame_steak.py --disable_gui \
  --start_checkpoint /data/qyfeng/models/flame_steak_new/chkpnt_fine_7000.pth \
  --checkpoint_iterations 10000 14000
```

恢复时使用同一数据和模型结构配置，文件名保留 `coarse` 或 `fine` 阶段标识。
fine checkpoint 跳过 coarse 阶段。恢复模型和优化器状态不等于逐位复现中断前
的数据采样随机序列。

## 训练后渲染与检查

新模型按[通用渲染指南](GENERAL_RENDERING.md)使用 `scripts/render.py`，选择
`diff_gaussian` 或 `gsplat` 后端，无需固定 profile。
[两版融合入口](INFERENCE_VERSIONS.md)保持原样：qualification 仍绑定特定模型
规模、分辨率和设备。即使重训同名 `flame_steak`，也不能假定高斯数量与资源条件
仍匹配，更不能手改哈希绕过校验。

`python3 tests/run_core_tests.py` 包含训练参数／配置、阶段调度、最终保存、
DataLoader 重启回归及已有两版推理核心测试。CPU 检查不会训练模型；CUDA
反传测试保留在扩展测试目录，见[测试说明](../tests/README.md)。此次恢复未完成
目标 GPU 上的完整训练、收敛验证或性能复测。
