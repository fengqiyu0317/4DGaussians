# 4DGaussians：普通 v13 与 gsplat 推理

本仓库提供通用渲染入口，以及两条共享模型、场景和 HexPlane / Deform MLP 的融合推理路线：

| 版本 | 实现 |
| --- | --- |
| 普通 v13 | mixed ABI-v12、Raster prepass、原生 HexPlane Graph、fused feature reduction |
| gsplat | gsplat 1.5.3、`tensor_v7_ca`、双槽 Graph、candidate-aware 任务配对 |

**[通用渲染指南](docs/GENERAL_RENDERING.md)** 介绍如何使用自己的兼容场景与模型，
选择原始光栅化或 gsplat 后端，导出图片和视频；串行／双流模式不需要固定场景 profile。

**[两版融合安装与运行指南](docs/INFERENCE_VERSIONS.md)** 包含环境、数据与 checkpoint
要求、两版命令和 qualification 限制。随仓库提供的融合配置为 `flame_steak` iteration 14000、
111525 个高斯、1352×1014，设备分别为 Orin / sm_87 和 A6000 / sm_86。
其他工作负载需要另行生成并验证配置。

```bash
# 先安装适合设备的 CUDA PyTorch 和 torchvision。
python3 -m pip install -r requirements-inference.txt
python3 -m pip install --no-build-isolation -r requirements-gsplat.txt
bash scripts/build_inference_extensions.sh sm_87  # A6000 改为 sm_86

python3 scripts/render_versions.py --help
python3 scripts/render.py --help  # 安装完整依赖后查看通用入口参数
python3 tests/run_core_tests.py  # 默认只运行 CPU 核心测试
```

通用入口为 `scripts/render.py`，支持 `diff_gaussian` / `gsplat` 后端的串行和双流渲染。
模型需有训练时的 `cfg_args`，并使用匹配 checkpoint 的配置与已准备好的数据。

固定融合入口为 `scripts/render_versions.py`，通过 `--variant v13` 或
`--variant gsplat_ca` 选择版本，必须显式传入对应 `profiles/` 配置。
入口检查实际 Graph / mixed 执行并导出图片，不报告性能 FPS。

## 发布范围

- `scripts/render.py`：通用图片／视频导出入口；视频编码依赖 `imageio[ffmpeg]`。
- `gaussian_renderer/`、`scene/`、`utils/`、`arguments/`：两版共用的 Python 源码与配置。
- `tacker_ext/`、`gsplat_mixed_ext/`、`submodules/`：CUDA 源码、绑定、ABI、头文件与构建文件。
- `profiles/`：必需的 disabled qualification 配置；不是实验结果归档。
- `tests/` 和扩展各自的 `tests/`：核心回归测试，见 [测试说明](tests/README.md)。

CUDA 依赖和 GLM 头文件以普通源码目录纳入单仓库，无需 `git submodule update`。
共用扩展中的兼容实现与测试对照内核仍保留，以维持构建、ABI 和数值验证。
模型、数据集、编译产物、benchmark、实验启动器、旧报告、训练和预处理入口不随
当前推理版本发布。原工作区中的这些历史文件仅取消 Git 跟踪并加入忽略规则，
本地内容保留；旧版本也可从 Git 历史查看。

## 来源与许可

本项目基于 [HUST 4DGaussians](https://github.com/hustvl/4DGaussians)，对应
CVPR 2024 论文 *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*。
沿用上游对 [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)、
[K-planes](https://github.com/Giodiro/kplanes_nerfstudio)、
[HexPlane](https://github.com/Caoang327/HexPlane)、
[TiNeuVox](https://github.com/hustvl/TiNeuVox) 和
[Depth-Rasterization](https://github.com/ingra14m/depth-diff-gaussian-rasterization)
的来源致谢。各组件许可与修改来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
和 [LICENSE.md](LICENSE.md)。

```bibtex
@InProceedings{Wu_2024_CVPR,
    author    = {Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
    title     = {4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {20310-20320}
}

@inproceedings{TiNeuVox,
  author = {Fang, Jiemin and Yi, Taoran and Wang, Xinggang and Xie, Lingxi and Zhang, Xiaopeng and Liu, Wenyu and Nie\ss{}ner, Matthias and Tian, Qi},
  title = {Fast Dynamic Radiance Fields with Time-Aware Neural Voxels},
  year = {2022},
  booktitle = {SIGGRAPH Asia 2022 Conference Papers}
}
```
