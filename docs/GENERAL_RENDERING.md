# 通用渲染入口 scripts/render.py

`scripts/render.py` 用于加载已有的 4DGaussians checkpoint 和准备好的数据集，渲染训练集、
测试集或数据读取器生成的视频视角。它保留上游的模型格式和参数读取方式。
`serial` / `two_stream` 可选择 `diff_gaussian` 或 `gsplat` 光栅化后端，不要求
`flame_steak` qualification，也不固定高斯数量、图像尺寸或训练迭代数。
模型结构、相机和数据格式仍须与当前 4DGaussians 读取器兼容。

| 入口 / 模式 | 用途与限制 |
| --- | --- |
| `scripts/render.py --execution-mode serial` | 通用逐帧渲染，默认模式，不需要 profile |
| `scripts/render.py --execution-mode two_stream` | 通用双流调度，不需要 profile；不满足条件时会报告回退 |
| `scripts/render_versions.py --variant v13` | 固定普通 v13 融合配置，必须匹配 qualification |
| `scripts/render_versions.py --variant gsplat_ca` | 固定 gsplat `tensor_v7_ca` 融合配置，必须匹配 qualification |

`scripts/render.py` 的历史 `tacker`、`gsplat_mixed`、`gsplat_matched_graph` 选项仍保留，
但它们继续执行 profile 校验。通用串行／双流渲染不等于启用了 v13 或
`tensor_v7_ca` 融合内核；运行这两个固定融合版本应使用[两版融合指南](INFERENCE_VERSIONS.md)。
尤其 `scripts/render.py --execution-mode tacker` 走部署准入，不能用随仓库提供的 disabled
qualification 文件替代已验证的部署 profile。

## 环境与输入

先按[安装指南](INFERENCE_VERSIONS.md#安装)准备 CUDA PyTorch、torchvision 和
扩展，安装 `requirements-inference.txt`。该依赖文件包含 `imageio[ffmpeg]`，
用于视频编码。选择 gsplat 后端还需要 `requirements-gsplat.txt`。
当前构建脚本提供 A6000 / sm_86 和 Orin / sm_87 目标，通用场景入口不自动解决
其他 GPU 的编译兼容问题。

模型目录至少包含训练时保存的 `cfg_args` 和所选迭代的 checkpoint：

```text
MODEL/
  cfg_args
  point_cloud/iteration_7000/
    point_cloud.ply
    deformation.pth
    deformation_table.pth   # 原 checkpoint 中存在时保留
    deformation_accum.pth   # 原 checkpoint 中存在时保留
```

`--configs` 应指向训练该模型时使用的网络／场景配置；其 `_base_` 引用的文件也需
存在。不应给其他结构的模型套用仓库里的 `flame_steak.py`。`cfg_args` 保留已有模型
参数，显式 `-m` / `-s` 可覆盖旧机器路径；`--configs` 随后合并，因此其中若定义了
`ModelParams` 路径，也应同步更新。此入口没有训练、原始视频预处理或任意 PLY 导入功能。

## 使用示例

两个入口统一位于 `scripts/`。以下命令从仓库根目录执行；也可以从其他目录通过
脚本的绝对路径启动，源码导入仍按脚本位置定位。相对数据、模型和配置路径依然
相对于当前工作目录解析。需要训练自己的模型时使用根目录 `train.py`，详见
[训练指南](TRAINING.md)。

以下路径为示例，替换为自己的数据、模型和训练配置。先确认目标 GPU 空闲；
Jetson 上模型、输出、临时目录及缓存均应位于 `/data/qyfeng` 数据盘。

```bash
# 原始光栅化后端，渲染测试集；-1 表示自动选择最新 checkpoint。
CUDA_VISIBLE_DEVICES=0 python3 scripts/render.py \
  -m /data/qyfeng/models/my_scene \
  -s /data/qyfeng/datasets/my_scene \
  --configs /data/qyfeng/configs/my_scene.py \
  --iteration -1 --raster_backend diff_gaussian \
  --execution-mode serial --skip_train --skip_video

# gsplat 后端，渲染指定迭代的测试集。
CUDA_VISIBLE_DEVICES=0 python3 scripts/render.py \
  -m /data/qyfeng/models/my_scene \
  -s /data/qyfeng/datasets/my_scene \
  --configs /data/qyfeng/configs/my_scene.py \
  --iteration 7000 --raster_backend gsplat \
  --execution-mode serial --skip_train --skip_video
```

将 `serial` 改为 `two_stream` 可使用双流调度；gsplat 默认 dense 布局，
`--gsplat_packed` 可选 packed 布局。默认运行 train、test、video 三组视角；
`--skip_train` / `--skip_test` / `--skip_video` 分别跳过一组。
`--skip_video` 只跳过 video 视角组，train/test 的图片序列仍会生成 MP4。

## 输出与计时

输出写入模型目录中的 `train|test|video/ours_<iteration>/`：

- `renders/*.png`：渲染 RGB 图片。
- `gt/*.png`：train/test 对应的真值图片。
- `video_rgb.mp4`：该组渲染帧组成的 30 FPS 视频。

**同一模型、视角组和迭代重复运行会复用目录并覆盖同名输出，后端名称不进入目录名。**
比较两种后端时应先将上一轮输出移到单独的结果目录，避免覆盖或混入旧帧。
入口会跳过空视角组，图片写入失败会报错。

控制台 `FPS` 是本次渲染循环的墙钟吞吐，图片传回 CPU 和编码在计时区间之外；
它没有正式性能实验的预热、重复轮次和配对控制，不能替代融合性能评测。
它也不计算 PSNR / SSIM / LPIPS，不导出深度图。

## 验证

```bash
python3 tests/run_core_tests.py
```

CPU 核心测试包括通用入口的配置合并、两种后端的串行分发、双流入口、融合
profile 要求、输出数量与图片写入错误处理。它们使用替身，不执行 GPU 渲染；
实际模型的画质、CUDA 数值和视频编码需在安装完整环境后验证。
