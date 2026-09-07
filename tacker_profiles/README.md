# Tacker 资格验证与准入

仓库中已提交的 `raster_head_sm86.json` 被有意设置为禁用状态。它是一份密封的契约模板，并不能证明 CUDA 路径已经通过质量或性能资格验证。只有在所有必需的 JSON 输入均通过全部门禁后，准入工具才会写入启用的配置文件。

## 固定的首个工作负载

首个配置文件仅对以下精确契约有效：

- 场景：`flame_steak`（使用 `dynerf` 加载器）
- 检查点迭代次数：`14000`
- 高斯数量：`111525`
- 图像尺寸：`1352 x 1014`
- GPU：NVIDIA RTX A6000，计算能力 8.6（`sm_86`）
- 光栅化器上游提交：`e49506654e8e11ed8a62d22bcb693e943fdecacf`
- 物理 CTA：384 个线程；Raster 使用 `[0, 255]`，head 使用 `[256, 383]`
- Raster 命名屏障：ID 为 1，共 256 个参与者；head 不使用命名屏障
- 选定任务：`pos_deform[1]`，输入/权重使用 FP16，偏置/累加/输出使用 FP32

仅在已配置的 `4A6000` 机器上运行 GPU 工作。除非后续实验明确需要更多 GPU，否则本次资格验证只使用一张 A6000。源代码、脚本、配置和小型配置文件存放在 `/home/qyfeng` 下；数据集、检查点、Nsight 文件、渲染输出和报告存放在 `/data/qyfeng` 下。

## 一条命令完成远程资格验证

仓库中已提交的入口脚本会执行完整的失败即拒绝（fail-closed）流程：检查主机是否恰好配备四张 A6000 且仅有一个设备可见、构建基于 CUDA 的 Tacker 运行时并运行 CTest、强制重新构建 simple-knn/head/Raster 的 sm_86 版本、运行 head 和 Raster CUDA 测试、执行 leaf/Raster 测量、进行三种模式的质量验证、采集可比较的端到端计时、执行准入，并进行一次常规的（非资格验证）Tacker 验证：

```bash
cd /home/qyfeng/4DGaussians
CUDA_VISIBLE_DEVICES=0 ./scripts/run_tacker_qualification.sh
```

数据和报告默认写入 `/data/qyfeng/tacker_admission`。唯一生成的小型配置文件默认写入 `/home/qyfeng/4DGaussians/tacker_profiles/raster_head_sm86.admitted.json`。每次运行都会使用全新的扩展对象目录，并将两份 ptxas 资源日志一并保存在数据输出目录下；系统绝不会接受旧对象缓存中的扩展。脚本会拒绝复用该文件，也不会覆盖已禁用的模板。因此，重新运行前必须先保留或移动之前已准入的配置文件，或者选择一个全新的路径，例如 `ADMITTED_PROFILE=/home/qyfeng/4DGaussians/tacker_profiles/raster_head_sm86.admitted.run2.json`。任何阶段都不会把跳过的测试、回退元数据或旧 JSON 当作通过结果。下面列出的各条命令仍可用于问题诊断。

## 1. 质量验证

`scripts/validate_tacker_modes.py` 会在每种请求的模式下渲染完全相同且顺序一致的视图批次，并使用同一组真值评估每张图像。它复用仓库中的 PSNR、SSIM 和 LPIPS 实现，并以原子方式写入一份 JSON 报告。该脚本不执行带计时的性能测量。

默认模式集合仅为 `serial two_stream`。物理 Tacker 执行绝不会被隐式启用。第一次测量运行使用有意限制的资格验证覆盖方式：必须同时提供 `--qualification-mode` 和显式的 `--qualification-profile`。这条路径会保留对清单、工作负载、模型、ABI、布局和 sm_86 的全部检查，仅跳过“准入测量结果必须已经存在”这一循环依赖要求。

```bash
cd /home/qyfeng/4DGaussians
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/submodules/depth-diff-gaussian-rasterization:$PWD/submodules/simple-knn${PYTHONPATH:+:$PYTHONPATH}"

python scripts/validate_tacker_modes.py \
  --model_path /data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak \
  --source_path /data/qyfeng/datasets/n3dv/flame_steak \
  --configs arguments/dynerf/flame_steak.py \
  --iteration 14000 \
  --split test \
  --frames 50 \
  --modes serial two_stream tacker \
  --qualification-mode \
  --qualification-profile /home/qyfeng/4DGaussians/tacker_profiles/raster_head_sm86.json \
  --output /data/qyfeng/tacker_admission/quality.json
```

请将示例中的数据集和模型位置替换为远程机器上的实际路径。发生回退即视为验证失败：报告会记录 `actual_mode` 和 `fallback_reason`，且只有物理 Tacker 模式确实运行时，准入工具才会接受质量数据。

这里有一个有意设置的引导边界。禁用的模板通常不能执行物理路径，但最终准入又需要该路径产生的质量数据。资格验证模式只会将其作为显式的内存中 `profile_override` 接受，绝不会重写或启用模板。报告会标记 `qualification.enabled: true` 和 `admission_claimed: false`。下文生成的最终配置文件是唯一能够代表全部门禁均已通过的产物。准入完成后，去掉资格验证参数，并通过 `--tacker-profile` 传入已准入的配置文件，即可执行常规验证复跑。

质量报告包含：

- `workload`、`device`、选定的 `view_indices` 以及精确路径；
- 每种模式的 `requested_mode`、`actual_mode`、回退原因、平均指标和逐视图指标；
- 相对于 serial 的有符号差值 `deltas.<mode>.psnr_drop_db`、`ssim_drop` 和 `lpips_increase`；
- 每个非 serial 模式对应一个门禁，以及一个总的 `passed` 标志。

## 2. 测量 JSON 输入

`scripts/benchmark_tacker_admission.py` 负责验证测量结果；它不会启动内核，也不会声称已在本机完成基准测试。请提供以下相互独立的输入：

首先采集规范的设备、完整 Raster、真实 head 和物理混合测量结果。该分析器会同时验证已编译的能力查询和 ABI 清单，在计时前检查数值等价性，并且只在本次运行通过后写入全部三项准入输入：

```bash
cd /home/qyfeng/4DGaussians
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/tacker_ext:$PWD/submodules/depth-diff-gaussian-rasterization:$PWD/submodules/simple-knn${PYTHONPATH:+:$PYTHONPATH}"

python profile_tacker_leaves.py \
  --model_path /data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak \
  --source_path /data/qyfeng/datasets/n3dv/flame_steak \
  --configs arguments/dynerf/flame_steak.py \
  --scene-name flame_steak --iteration 14000 \
  --split test --views 2 --warmup 5 --repetitions 50 \
  --persistent-blocks 7000 \
  --device-output /data/qyfeng/tacker_admission/device.json \
  --raster-output /data/qyfeng/tacker_admission/raster.json \
  --leaf-output /data/qyfeng/tacker_admission/leaf.json \
  --report /data/qyfeng/tacker_admission/leaf-profile-report.json
```

密封的首个工作负载配置文件使用 `persistent_blocks=7000`。对于固定的 A6000 工作负载，这会让每个 1352x1014 Raster 图块拥有自己的物理混合 CTA，同时在同一次启动中分配 13,942 个 positional-head 逻辑块。资格验证必须测量这个精确值；`0`（每个 SM 一个物理块）只是用于诊断的默认值，并非获准使用的调度方案。

可接受的源目录基本名称包括规范名称 `flame_steak`，以及已知的最小化数据集目录 `flame_steak_4dgs_min`；显式的场景、加载器、检查点、高斯数量和分辨率门禁仍为必需项。

请使用相同的数据划分、预热次数和帧数采集可比较的端到端元数据。第一次 Tacker 计时运行使用与质量验证相同的显式资格验证覆盖方式：

```bash
cd /home/qyfeng/4DGaussians
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/submodules/depth-diff-gaussian-rasterization:$PWD/submodules/simple-knn${PYTHONPATH:+:$PYTHONPATH}"

python profile_render.py \
  --model_path /data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak \
  --source_path /data/qyfeng/datasets/n3dv/flame_steak \
  --configs arguments/dynerf/flame_steak.py \
  --iteration 14000 --split test --warmup 10 --frames 50 \
  --execution-mode two_stream --workload-name flame_steak \
  --metadata /data/qyfeng/tacker_admission/two-stream.json

python profile_render.py \
  --model_path /data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak \
  --source_path /data/qyfeng/datasets/n3dv/flame_steak \
  --configs arguments/dynerf/flame_steak.py \
  --iteration 14000 --split test --warmup 10 --frames 50 \
  --execution-mode tacker --workload-name flame_steak \
  --qualification-mode \
  --qualification-profile /home/qyfeng/4DGaussians/tacker_profiles/raster_head_sm86.json \
  --metadata /data/qyfeng/tacker_admission/tacker.json
```

准入完成后，将第二条命令中的两个资格验证参数替换为 `--tacker-profile /home/qyfeng/4DGaussians/tacker_profiles/raster_head_sm86.admitted.json`，即可执行常规验证。Tacker 回退结果绝不会被接受为 Tacker 测量结果。

1. 只有成功的 `4dgaussians_tacker_device` schema-v1 文档才会被接受为 `device.json`。除了精确的工作负载和 A6000/sm_86 标识外，它还必须包含已编译的 Raster 与 head 能力字典，以及 `profile_tacker_leaves.py` 输出的三个带版本 CUDA 符号。有意拒绝仅手工填写设备名称的文档。

2. `raster.json` 必须是成功的 `4dgaussians_tacker_raster_profile` schema-v1 文档，其中包含 `solo_raster_p50_ms`、`mixed_raster_p50_ms`、通过检查的数值证据，以及 `measurement_config.persistent_blocks`。

3. `leaf.json` 必须是成功的 `4dgaussians_tacker_leaf_profile` schema-v1 文档，其中包含 `mixed_p50_ms`、`solo_raster_p50_ms`、`solo_head_p50_ms`、同一份通过检查的数值证据，以及相同的 persistent-block 配置。`raster.json` 与 `leaf.json` 中的 `solo_raster_p50_ms` 必须是完全相同的规范测量值。设备、Raster 和 leaf 文档还必须在数据划分、解析后的模型/源路径，以及结构有效且有序的视图对方面保持一致；它们的模型/源/数据划分来源信息必须与质量和端到端运行相匹配。

4. 分别提供一份 `two_stream` 和一份 `tacker` 的端到端 JSON。推荐使用的聚合字段为 `p50_frame_ms`：

   ```json
   {
     "schema_version": 1,
     "kind": "4dgaussians_tacker_render_profile",
     "passed": true,
     "workload_name": "flame_steak",
     "model_path": "/data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak",
     "source_path": "/data/qyfeng/datasets/n3dv/flame_steak",
     "iteration": 14000,
     "split": "test",
     "warmup_frames": 10,
     "profile_frames": 4,
     "view_indices": [0, 1, 2, 3],
     "image_width": 1352,
     "image_height": 1014,
     "gaussian_count": 111525,
     "gpu_name": "NVIDIA RTX A6000",
     "actual_execution_mode": "two_stream",
     "p50_frame_ms": 1.0,
     "timing_method": "perf_counter_with_cuda_synchronize",
     "frame_timing_method": "cuda_event_consumer_completion_intervals",
     "io_in_timed_region": false
   }
   ```

   `tacker` 文档还必须将 `persistent_blocks` 和 `profile_manifest_sha256` 绑定到运行时实际使用的精确模板。如果两种模式都使用同一统计量，也接受包含 `mean_frame_ms` 的 `profile_render.py` 元数据。两份计时文档必须在数据划分、有序视图列表、帧数、模型和源方面彼此一致，并与 `quality.json` 一致；两者的预热次数也必须相同。绝不能将一种模式的平均值与另一种模式的 p50 进行比较。

5. 第 1 步生成的 `quality.json`。该工具要求它是成功的 schema-v1 验证文档，要求 `modes.tacker.actual_mode == "tacker"`，并读取 `deltas.tacker.{psnr_drop_db,ssim_drop,lpips_increase}`。

混合 Raster+head ABI、独立 head ABI 和禁用的配置模板默认使用仓库中的路径。可以通过 `--mixed-abi-json`、`--head-abi-json` 和 `--template-profile` 显式覆盖。

## 3. 失败即拒绝的准入流程

在一张 A6000 上采集真实测量结果后，运行：

```bash
cd /home/qyfeng/4DGaussians

python scripts/benchmark_tacker_admission.py \
  --device-json /data/qyfeng/tacker_admission/device.json \
  --quality-json /data/qyfeng/tacker_admission/quality.json \
  --raster-json /data/qyfeng/tacker_admission/raster.json \
  --leaf-json /data/qyfeng/tacker_admission/leaf.json \
  --two-stream-json /data/qyfeng/tacker_admission/two-stream.json \
  --tacker-json /data/qyfeng/tacker_admission/tacker.json \
  --report /data/qyfeng/tacker_admission/admission-report.json \
  --enabled-profile /home/qyfeng/4DGaussians/tacker_profiles/raster_head_sm86.admitted.json
```

所有门禁均执行精确检查，并遵循失败即拒绝原则：

- Raster LC 减速幅度：最多 5%；
- 完整混合 leaf：必须严格快于 `solo_raster_p50_ms + solo_head_p50_ms`；
- Tacker 端到端耗时 / two-stream 端到端耗时：最多 1.0；
- 平均 PSNR 降幅：最多 0.05 dB；
- 平均 SSIM 降幅：最多 `1e-4`；
- 平均 LPIPS 增幅：最多 `1e-4`；
- A6000/sm_86、工作负载、迭代次数、分辨率、高斯数量、光栅化器提交、ABI v1 符号、CTA 布局、数据类型和命名屏障必须精确匹配；
- 规范分析器输出的 schema/kind/数值证据必须成功；
- 两次端到端测量的模型、源、数据划分、有序视图列表、帧数、预热次数和计时方法必须完全相同；
- `persistent_blocks` 和配置清单哈希必须与测量期间运行时使用的模板完全一致。

字段缺失、将布尔值用作数值、计时为零或负数、`NaN` 以及无穷值都会导致失败。报告始终以原子方式替换。只有当每项输入契约和门禁都通过时，启用的配置文件才会以原子方式写入；失败时不会创建或修改目标配置文件。请使用新的已准入配置文件名，避免把旧的已准入文件误认为失败复跑的结果。

启用后的配置文件使用运行时 schema：

- `schema_version: 1`；
- `manifest`，以及基于规范化排序并最小化后的 JSON 计算出的 `manifest_sha256`；
- 锁定的 `thresholds`；
- `admission: {"enabled": true, "valid": true}`；
- 九项有限值运行时 `measurements`；
- `provenance`，其中包含输入文档哈希和计时统计量。

独立报告会包含每项门禁的测量值、限制值、比较方式、通过/失败结果、派生比率、输入哈希、错误信息，以及是否已写入启用的配置文件。

## CPU 契约检查

准入单元测试无需 CUDA 或 PyTorch：

```bash
cd /home/qyfeng/4DGaussians
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_tacker_admission -v
```
