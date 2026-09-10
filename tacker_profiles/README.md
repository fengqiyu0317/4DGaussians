# Tacker correctness qualification 与 FPS 选择

当前 profile contract 已升级为 schema v2。仓库中的
`raster_head_sm86.json` 始终是 disabled qualification template；它可以用来验证
物理 Tacker 路径，但不是部署凭证，也不得被原地启用。新的准入流程只会
在 Tacker 获得部署资格时写出一份独立、带哈希的 winner profile。

## 固定工作负载

首个 v2 profile 只对下列精确契约有效：

- `flame_steak`，iteration 14000，111,525 Gaussians；
- 1352 × 1014，test 视角 0–49；
- NVIDIA RTX A6000，compute capability 8.6（`sm_86`）；
- Rasterizer commit `e49506654e8e11ed8a62d22bcb693e943fdecacf`；
- 当前物理 ABI 为 384-thread CTA：Raster `[0,255]`，pos-L1 `[256,383]`；
- Raster named barrier ID 1，256 participants；head 子组不使用 named barrier。

`workload_key` 为
`flame_steak:14000:111525:1352x1014:sm_86`。任何 workload、设备、ABI 或
profile 哈希不匹配都会 fail closed。

## Phase 1 决策契约

Phase 1 把两件事显式拆开：

1. **Correctness qualification** 决定候选是否有资格参与测量和排名。
   Tacker 候选必须实际执行 `tacker`、无 fallback，并通过数值、画质、
   workload、ABI 和 capability 契约。
2. **Performance selection** 只在 correctness-valid 候选中，按多次完整序列
   `median(throughput_fps)` 排名。`serial` 和 `two_stream` 也是正式候选，
   因此“不融合”是合法结果。

`raster_slowdown_pct`、`mixed_p50_ms < solo_raster + solo_head` 和历史的
Tacker/two-stream 单次 p50 比值现在只是 diagnostics，不再否决候选。画质上限
仍为 PSNR drop `<= 0.05 dB`、SSIM drop `<= 1e-4`、LPIPS increase
`<= 1e-4`。

选择器保留两个结果：

- `experimental_winner`：所有有效候选中 median FPS 的精确全局 argmax；
- `deployment_winner`：考虑 0.5% 等价区间的稳定资源 tie-break，以及替换
  incumbent 时 `FPS ratio >= 1.01` 且 paired bootstrap 95% CI 下界
  `> 1.0` 的最终部署结果。部署 Tacker 还不得慢于有效的
  `two_stream` 和 incumbent。

正式证据的统计协议不可由报告自由降级：bootstrap 固定为
`confidence=0.95`、10,000 次、seed 0，并以共享的 whole-run round 为
配对重采样单位。Admission 会重算区间，同时重建 seed 0 的 ABBA 或
round-robin 调度，并要求每个有效候选在每个 round 恰好出现一次。

如果 incumbent 已 correctness-invalid，它不会被调度或排名，1%/CI 替换门槛
无法应用；系统会从顶部 0.5% 等价集中选择不慢于 two-stream 的
稳定优选候选。如果基线获胜，
准入报告仍会成功写出，但不生成 enabled Tacker profile。

## Profile schema v2

v2 的顶层密封字段包括：

- `workload_key` 和 `selection_objective: "median_throughput_fps"`；
- `candidates[]` 中的物理模式、variant/ABI/task-graph contract、correctness、
  whole-run FPS trials、资源 tie-break 数据和 diagnostics；
- 唯一的 `selected_variant_id`、完整排名、等价集和 promotion 证据；
- `manifest_sha256`、`profile_sha256` 和输入 provenance。

`profile_sha256` 对选择相关字段做 canonical JSON 哈希；
`provenance.generated_at_utc` 不参与该哈希，所以同一输入可重现稳定的
selection hash。`profile_render.py` 会对实际加载的同一份字节快照记录
file SHA、manifest SHA、selection SHA 和 selected candidate ABI SHA，避免加载后
再读文件引入 provenance 竞态。交错 benchmark 还会锁定 config、
`gaussian_renderer` 两条执行路径、Raster Python wrapper 与实际加载的
`diff_gaussian_rasterization._C` 二进制 SHA，并要求 repository dirty
状态和子模块状态在所有 trial 间一致。这些源码、config 和已加载
二进制的哈希在 warmup/计时前采集，后续结果只复用该快照。
Admission 还会将每个 Tacker trial 的 manifest/selection SHA、selected
variant、ABI SHA 和 `persistent_blocks` 与该文件 SHA 对应的已加载
profile 逐项比对；`qualification_mode` 必须与 profile 的 deployment
状态互补。可空的 fallback/profile 证据也必须显式出现，缺字段不等于
`null`。

Phase 1 绑定 model/source 路径、iteration、shape 和 Gaussian 数量，但尚未对
checkpoint 文件内容做 fingerprint；PLY/PTH 内容哈希属于 Phase 4 的完整
发布验收，不应把本阶段报告解读为已绑定权重字节。

schema v1 仅作 `legacy_pos_l1` 兼容读取。运行时仍会验证它的结构、
ABI 和画质证据，但不再重新执行旧 Raster QoS/leaf/E2E 性能否决。
新的准入工具拒绝生成 v1 profile。缺少 Phase 1 correctness/selection
字段的兼容路径只对仓库中封存的 Phase 0 报告开放，admission 会校验
其完整 canonical SHA-256；任何修改或从新报告删字段都不能获得
legacy 默认值。

## 1. 准备 correctness 和 tie-break 输入

FPS 驱动器的 `--correctness-json` 必须精确覆盖本次所有候选：

```json
{
  "serial": {"valid": true},
  "two_stream": {"valid": true},
  "current_tacker": {"valid": true},
  "pos_l1_pb80": {"valid": true}
}
```

这份文件是 benchmark 的调度前置过滤：invalid 候选不会被运行，也不会进入
summary 或排名。最终 admission 仍会从质量报告和候选 correctness 证据中
重新验证物理模式、fallback 和画质数值，不会只信任这个布尔值。
对新报告，selector 的完整结果和顶层镜像字段也是必填证据；admission
会按同一固定参数重算并逐字段比对，不允许删除、改写或只保留 winner。

可选的 `--selection-metadata-json` 用于 0.5% 等价集，所有值都是越小越好：

```json
{
  "pos_l1_pb80": {
    "abi_complexity": 2,
    "peak_memory_bytes": 123456,
    "registers_per_thread": 64,
    "shared_memory_bytes": 0
  }
}
```

缺省时 `serial`/`two_stream`/Tacker 的 ABI complexity 分别为 0/1/2；显式资源值
会被密封到报告和 winner profile。只接受示例中的四个字段；可选值
应当直接省略，不应写为 `null`。未知字段、非有限数或负数会在任何
GPU child 启动前 fail closed，避免 producer 生成 admission 无法消费的报告。

额外 Tacker 候选在最终准入时还需要完整的 `--candidate-correctness-json`
证据，例如：

```json
{
  "pos_l1_pb80": {
    "valid": true,
    "actual_execution_mode": "tacker",
    "fallback_reason": null,
    "psnr_drop_db": 0.01,
    "ssim_drop": 0.00001,
    "lpips_increase": 0.00001,
    "numerics": {"passed": true}
  }
}
```

该文档的候选名必须与 FPS 报告和 `--candidate-profile` 中的名字一致。

## 2. 交错测量 whole-run FPS

`scripts/benchmark_tacker_fps.py` 每轮以 ABBA 或 round-robin 顺序交错启动
`serial`、`two_stream`、`current_tacker` 和额外候选。关闭的 v2 candidate
profile 会自动通过显式 qualification mode 执行；`current_tacker` 如果是 v2，
则必须是已启用的部署 profile。

```bash
cd /home/qyfeng/4DGaussians
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD/submodules/depth-diff-gaussian-rasterization:$PWD/submodules/simple-knn${PYTHONPATH:+:$PYTHONPATH}"

python scripts/benchmark_tacker_fps.py \
  --output /data/qyfeng/tacker_phase1/fps-report.json \
  --runs-dir /data/qyfeng/tacker_phase1/runs \
  --run-id a6000-gpu1-phase1 \
  --current-tacker-profile /home/qyfeng/4DGaussians/tacker_profiles/raster_head_sm86.admitted.full_20260831.json \
  --candidate pos_l1_pb80=/home/qyfeng/4DGaussians/tacker_profiles/pos_l1_pb80.disabled.json \
  --correctness-json /data/qyfeng/tacker_phase1/correctness.json \
  --selection-metadata-json /data/qyfeng/tacker_phase1/selection-metadata.json \
  --model-path /data/qyfeng/4DGaussians-flame-steak-full/outputs/n3dv_flame_steak \
  --source-path /data/qyfeng/datasets/n3dv/flame_steak \
  --configs arguments/dynerf/flame_steak.py \
  --workload-name flame_steak --iteration 14000 --split test \
  --frames 50 --warmup 10 --trials 10 --schedule abba --seed 0 \
  --bootstrap-resamples 10000 --timeout-seconds 300 \
  --expected-image-width 1352 --expected-image-height 1014 \
  --expected-gaussian-count 111525
```

报告和 run directory 均拒绝覆盖；重跑必须使用新路径或新 `run-id`。每个子进程
必须返回请求/实际模式、无 fallback、精确 workload 和同一 profile 字节哈希，
否则整份报告 fail closed。正式 admission 会直接复核每个有效候选至少 10 个
paired whole-run trial，且每个 trial 必须恰好包含 50 帧；不依赖报告中可删除的
声明性字段放行。

## 3. 生成准入报告和 winner profile

`scripts/benchmark_tacker_admission.py` 不启动 GPU 内核；它校验已存的设备、ABI、
质量、leaf/Raster diagnostics 和 whole-run FPS 证据。当 benchmark 含额外
Tacker 候选时，使用可重复的 `--candidate-profile NAME=PATH` 传入候选的
物理 v2 descriptor；该文件的完整 file SHA 必须与 FPS 报告中的实际运行
输入一致。

新的 prefiltered FPS 报告始终要求显式传入
`--candidate-profile current_tacker=/absolute/current-profile.json`。首次从严格
验证的 enabled/valid schema v1 incumbent 迁移时，它会被 SHA 绑定并映射为
`legacy_pos_l1`；后续则直接从已部署 v2 winner 提取真实 variant 和
`persistent_blocks`。因此每一轮 1% promotion 比较都继续绑定实际 incumbent，
缺失 descriptor 不会默认回退到 legacy。只有没有 Phase 1 correctness 字段的
封存 Phase 0 报告可使用兼容映射。

```bash
python scripts/benchmark_tacker_admission.py \
  --device-json /data/qyfeng/tacker_phase1/device.json \
  --quality-json /data/qyfeng/tacker_phase1/quality.json \
  --raster-json /data/qyfeng/tacker_phase1/raster.json \
  --leaf-json /data/qyfeng/tacker_phase1/leaf.json \
  --fps-benchmark-json /data/qyfeng/tacker_phase1/fps-report.json \
  --candidate-profile current_tacker=/home/qyfeng/4DGaussians/tacker_profiles/raster_head_sm86.admitted.full_20260831.json \
  --candidate-profile pos_l1_pb80=/home/qyfeng/4DGaussians/tacker_profiles/pos_l1_pb80.disabled.json \
  --candidate-correctness-json /data/qyfeng/tacker_phase1/candidate-correctness.json \
  --report /data/qyfeng/tacker_phase1/admission-report.json \
  --enabled-profile /home/qyfeng/4DGaussians/tacker_profiles/raster_head_sm86.admitted.phase1.json
```

`device.json`、`raster.json`、`leaf.json` 仍可由 `profile_tacker_leaves.py` 采集；
Raster slowdown 和 leaf-sum 结果会原样保存在 diagnostics，但无论数值多高都不会
转换为性能否决。结构、数值证据或 provenance 无效时，对应的物理
Tacker 候选会被标记为 correctness-invalid。

输出使用同目录临时文件、finite canonical JSON、`fsync` 和 `os.replace`。
报告/profile 路径不得相同，也不得覆盖 disabled template。建议每次运行使用新的
winner 文件名，以便保留完整回滚记录。

## Phase 0 封存基线

Phase 0 于 2026-09-10 在独占 RTX A6000 GPU 1 上完成。三种模式以 ABBA、
seed 0 各执行 10 个 50-frame trial，warmup 10；30/30 次均通过物理模式、
fallback、workload、计时边界、profile 哈希和 provenance 校验。

| 排名 | 模式 | median FPS | median total | FPS 范围 |
|---:|---|---:|---:|---:|
| 1 | current Tacker | 86.979719 | 574.846646 ms | 86.597897–87.099374 |
| 2 | two-stream | 86.254250 | 579.681585 ms | 85.856130–86.436900 |
| 3 | serial | 82.423385 | 606.623962 ms | 81.264402–82.890268 |

current Tacker / two-stream ratio-of-medians 为 `1.0084108268`，10,000 次配对
bootstrap 95% CI 为 `[1.0061090994, 1.0097120318]`。聚合报告、30 份原始
JSON、哈希和复现说明位于
`tacker_profiles/baselines/a6000_flame_steak_phase0_20260910/`。

## CPU contract 回归

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  tests.test_tacker_pipeline \
  tests.test_profile_render_modes \
  tests.test_benchmark_tacker_fps \
  tests.test_tacker_admission -v
```

`scripts/run_tacker_qualification.sh` 仍是 schema-v1 历史编排入口，不代表新的
Phase 1 准入契约；将它串接到多候选 correctness/FPS 流程属于 Phase 4。
