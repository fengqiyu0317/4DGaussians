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
- 当前已部署的 v1 ABI 为 384-thread CTA：Raster `[0,255]`，pos-L1
  `[256,383]`；Phase 2/3.1 qualification 同时覆盖 C1/C2 first-linear、
  C3 packed first-linear 和 C4 whole-head，精确的 CTA、worker-group 与
  `persistent_blocks` 从候选 ABI/profile 读取；
- Raster named barrier ID 1，256 participants；head device adapter 内部不使用
  named barrier。v2 mixed wrapper 仍以 barrier ID 2、`128 * worker_groups`
  participants 广播 task descriptors。

`workload_key` 为
`flame_steak:14000:111525:1352x1014:sm_86`。任何 workload、设备、ABI 或
profile 哈希不匹配都会 fail closed。

## 当前决策契约

从 Phase 1 起，准入把两件事显式拆开：

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

Phase 1–3.1 封存证据绑定 model/source 路径、iteration、shape 和
Gaussian 数量，但不得把这解读为已完成所有 checkpoint 字节的 Phase 4
发布验收。Phase 4 的 workload file inventory、源码/config/候选二进制预快照、
实际加载二进制绑定和 post-run 字节稳定性检查必须全部通过，否则不可
发布。

## Phase 2/3.1 候选 profile

Phase 2 的通用 first-linear partition 使用同一套候选描述覆盖五个
C1（`pos`、`scales`、`rotations`、`opacity`、`shs`）和一个
`pos+scales` C2。每个候选都锁定两层 ABI：

- `tacker_ext/abi/head_linear_v2.json`：`9d6a1558acd6b642b975bcabe22abcbe3fd7242e4c9e0d635636ef4d2eb5da7f`；
- `submodules/depth-diff-gaussian-rasterization/abi/tacker_mixed_render_heads_v2.json`：`310b15957c5920773bb03a61a37c5771f6d4570393061ece4e1805fd20989056`。

Phase 3.1 另加入两个 Raster ABI：

- C3 packed v3：`c98ed90853308179443146d3022e5da072c4f507975193f7a01f4fbe4400cf40`；
- C4 whole-head v4：`293b8471fc9397070f1d1ebbe1297420f24f49e6882369e2e6cf8dcd9d49b7a1`。

在 A6000 上构建两个扩展后，以编译后的函数属性和 occupancy 生成
C0 + 全部 C1 + 一个 C2 的 disabled qualification profile：

```bash
cd /home/qyfeng/4DGaussians
export PYTHONPATH="$PWD/submodules/depth-diff-gaussian-rasterization${PYTHONPATH:+:$PYTHONPATH}"
python scripts/generate_tacker_phase2_profiles.py \
  --output-dir /data/qyfeng/tacker_phase2/candidates \
  --persistent-blocks 7000
```

生成器要求 resource query 显式返回 `launch_supported=true`、非零
`active_blocks_per_sm`、寄存器、static shared memory 和 kernel thread limit；
任一缺失或不匹配都 fail closed。生成的文件始终保持
`deployment.enabled=false` 和 `deployment.valid=false`，只是真实 CUDA 数值、
50-view 画质和 whole-run FPS 验证的输入，不是可部署凭证。

schema v1 仅作 `legacy_pos_l1` 兼容读取。运行时仍会验证它的结构、
ABI 和画质证据，但不再重新执行旧 Raster QoS/leaf/E2E 性能否决。
新的准入工具拒绝生成 v1 profile。缺少 Phase 1 correctness/selection
字段的兼容路径只对仓库中封存的 Phase 0 报告开放，admission 会校验
其完整 canonical SHA-256；任何修改或从新报告删字段都不能获得
legacy 默认值。

## 手工复现构建块

下列 1–3 节是 selector/admission 的底层手工构建块，便于独立审计中间
证据。它们不会自动执行 Phase 4 的 seal 校验、TOCTOU、1/2/50+长序列、
fallback、泛化评估或发布检查。完整 Phase 4 应使用后文的唯一编排入口。

### 1. 准备 correctness 和 tie-break 输入

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

### 2. 交错测量 whole-run FPS

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

### 3. 生成准入报告和 winner profile

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

## Phase 0 封存历史基线

Phase 0 于 2026-09-10 在独占 RTX A6000 GPU 1 上完成。三种模式以 ABBA、
seed 0 各执行 10 个 50-frame trial，warmup 10；30/30 次均通过物理模式、
fallback、workload、计时边界、profile 哈希和 provenance 校验。
这份证据只是历史 incumbent 参考，不包含 C3/C4，也不能代替 Phase 4
准入或发布决定。

| 排名 | 模式 | median FPS | median total | FPS 范围 |
|---:|---|---:|---:|---:|
| 1 | current Tacker | 86.979719 | 574.846646 ms | 86.597897–87.099374 |
| 2 | two-stream | 86.254250 | 579.681585 ms | 85.856130–86.436900 |
| 3 | serial | 82.423385 | 606.623962 ms | 81.264402–82.890268 |

current Tacker / two-stream ratio-of-medians 为 `1.0084108268`，10,000 次配对
bootstrap 95% CI 为 `[1.0061090994, 1.0097120318]`。聚合报告、30 份原始
JSON、哈希和复现说明位于
`tacker_profiles/baselines/a6000_flame_steak_phase0_20260910/`。

## Phase 3.1 封存输入

Phase 3.1 的最终有效 `run-complete-v3` 已紧凑镜像到
`tacker_profiles/baselines/a6000_phase31_20260913/`，完整 canonical run 保留在：

```text
/data/qyfeng/tacker_phase31_validation/20260913-codex-phase31/run-complete-v3
```

它封存 661 个 C0–C4 候选、9 个 generated finalists 与 3 个 baseline。
formal 协议为同一 GPU 1、视角 0–49、warmup 10、50 frames、10 trials、
ABBA/seed 0，120/120 次成功。封存选择为：

- winner：`c3_packed_first_linear_pos_scales_rotations_opacity_shs_wg1_pb5440`；
- median FPS：`100.9417269038`；
- matrix SHA-256：`aa45c9dce881a5a71c9d86130fcd6b12070ee2773b83b44a538f2147bbde8577`；
- formal-set SHA-256：`7c592726cb2edfeb9ea7c030b569d213b179c0917d31a9d1b4a56007abc54102`；
- selection SHA-256：`1a91a02c2605ae442817bb060b7956ead5d3b940146d274b94a85e00849ec126`；
- winner profile file SHA-256：`5c3c2031aabe8018d11f41a90e324f375b1a7a14cdfaaef2db730514ec027a51`；
- winner canonical profile SHA-256：`f91e5f129708ee7612e4a45e95b00e23fe15a92eb1f1be376c2a85095379599e`。

Phase 3.1 的 `promote_challenger` 只是 selector 结论。winner profile 仍为
`deployment.enabled=false, valid=false`；它是 Phase 4 输入，不是已发布凭证。

## Phase 4 fail-closed 编排

`scripts/run_tacker_qualification.sh` 现为 Phase 4 入口，转发给
`scripts/run_tacker_phase4.py`。编排器只消费上述 sealed finalists，并明确禁止
候选生成、matrix 扩展或重排。它的 hash-bound/checkpointed stages 为：

1. preflight 和扩展/runtime 构建、CPU/CUDA 回归；
2. Phase 3.1 identity/matrix/formal-set/selection 的字节 seal 复核；
3. 每个 sealed finalist 的资源、occupancy 和数值独立复验；
4. 50-view 画质和 10 × 50 ABBA/seed 0 whole-run 复测；
5. selector/admission 重算，及条件性 enabled profile 的常规非 qualification 复跑；
6. 1/2/50 与长序列的输出顺序、prefill/steady-state/drain 计数；
7. missing、stale-workload、hash-mismatch 三种可见 fallback smoke；
8. 两个声明为不同 Raster/deformation mix、且 whole-run overlap proxy
   可测地区分的泛化 workload，各自以独立 `workload_key` 报告结果，不宣称
   全局 winner；
9. 显式 canary、原子 release artifact 和 current-Tacker/two-stream rollback 复验。

运行前可在相同环境向命令末尾添加 `--dry-run`，检查密封哈希、必填
参数和完整 stage plan；dry-run 不调用任何子进程。

### 最终 A6000 run 与精确命令

2026-09-15 的唯一最终结论来自
`/data/qyfeng/tacker_phase4_validation/20260914-codex-phase4/run-complete-v5`。
当次从隔离的 v8 源码树实际使用了以下命令：

```bash
cd /home/qyfeng/tacker_phase4_code/20260914-codex-phase4-v8

export TACKER_ROOT=/home/qyfeng/tacker_phase4_runtime/20260914-codex-phase4
export PHASE31_RUN_ROOT=/data/qyfeng/tacker_phase31_validation/20260913-codex-phase31/run-complete-v3
export OUTPUT_DIR=/data/qyfeng/tacker_phase4_validation/20260914-codex-phase4/run-complete-v5
export CUDA_VISIBLE_DEVICES=1
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.124.06
export CUDA_HOME=/usr/local/cuda-12.4
export TORCH_HOME=/data/qyfeng/cache/torch
export PYTHON_BIN=/data/qyfeng/conda-envs/4dgaussians-flame-steak/bin/python3.10

bash scripts/run_tacker_qualification.sh
```

这是 canonical run 的原始命令记录，不应就地覆盖或改写该证据目录。做独立
复现时，只将 `OUTPUT_DIR` 改成一个新的、空的 run root；不得把 Phase 3.1
canonical root 用作可写输出目录，也不得暴露多张 GPU。

Phase 4 报告必须同时绑定主仓源文件、Raster 子模块、Tacker runtime
源文件和实际 `libtacker_runtime.so`、CUDA/PyTorch/GPU、workload files、ABI、
config/profile 与实际加载 Raster 二进制。预快照在 heavy import/配置解析前
建立，import 后绑定实际二进制，并在最终报告发布前复核所有字节未变。
任一缺失、不匹配或运行期改写都 fail closed。

### 最终资格结果

13/13 个 hash-bound stage 均为 `succeeded`。正式测量将 9 个 sealed
finalist 与 serial、two-stream、current Tacker 组成 12-entry 集合，在
warmup 10、50 frames、10 trials、ABBA/seed 0 下完成 120/120 次执行。

| entry | median FPS | 相对 winner |
|---|---:|---:|
| `c3_packed_first_linear_pos_scales_rotations_opacity_shs_wg1_pb5440` | `100.5082562657` | `1.0` |
| current Tacker | `87.3984074601` | winner/current `1.1500010033` |
| two-stream | `86.3416638762` | winner/two-stream `1.1640759716` |

winner/current 的 paired-bootstrap 95% CI 为
`[1.1446137379, 1.1542292551]`，winner/two-stream 的 CI 为
`[1.1596655823, 1.1687732974]`。实验 argmax 和 deployment selector 一致，
决策为 `promote_challenger / promotion_gates_passed`。

资源与画质证据为：

- 10/10 份 Tacker 资源/数值记录有效，12/12 个 formal entry 的 50-view
  correctness 有效，39 条 ptxas 记录均为 0 spill。
- winner 的 mixed/solo-Raster/solo-head p50 为 `5.474816` / `4.174336` /
  `1.435648 ms`；Raster slowdown `31.154%` 仅作 diagnostic，不是 QoS gate。
- winner 的 Raster 物理配置是 70 registers/thread、7168 B static shared memory、
  0 local bytes/thread、2 active blocks/SM、occupancy `0.5`；packed head 为
  48 registers/thread、0 static shared memory、occupancy `0.8333333333`。
- 50-view 平均差异为 PSNR drop `0.000209961 dB`、SSIM drop `1.3161e-6`、
  LPIPS increase `6.7294e-7`，低于 `0.05 dB` / `1e-4` / `1e-4` 门禁。

1/2/50/500 帧回归都完成；下表中 counts 顺序是
`full/prefix/mixed/suffix/solo-raster/outputs/selected-head-per-head`：

| 帧数 | p50 / p95 / max（ms） | allocated / reserved 峰值（B） | counts |
|---:|---:|---:|---:|
| 1 | `202.915833 / 202.915833 / 202.915833` | `817,773,568 / 2,065,694,720` | `1/0/0/0/1/1/1` |
| 2 | `108.794884 / 201.154864 / 211.417084` | `1,280,936,960 / 2,065,694,720` | `1/1/1/1/1/2/2` |
| 50 | `9.813019 / 10.097307 / 208.758789` | `1,280,936,960 / 2,065,694,720` | `1/49/49/49/1/50/50` |
| 500 | `9.840576 / 9.891193 / 199.469055` | `1,282,750,976 / 2,124,414,976` | `1/499/499/499/1/500/500` |

admission 生成的 enabled/valid profile 又在常规、非 qualification 路径完成
10 × 50 复跑：`actual_execution_mode=tacker`、无 fallback，每次 50 帧的
counts 为 `1/49/49/49/1/50/50`。该路径的 median throughput 为
`72.7980424912 FPS`、median p50/p95 为 `9.788147/9.834361 ms`；它用于验证
常规加载路径，不替代前述交错 formal selector 统计。本次整个 Phase 4
观测到的最大 allocated/reserved 为 `1,284,494,848` / `2,124,414,976` B。

missing、stale-workload、hash-mismatch 三个负向用例完成 3/3：它们都从
请求的 `tacker` 可见地回退到 `two_stream`，并分别记录“预快照中 profile
缺失”、“workload name 不匹配”与“manifest SHA-256 不匹配”的非空原因。

泛化阶段严格限定为 baseline-only/evaluation-only：

| workload | serial / split-serial / two-stream FPS | overlap proxy（split/serial，two/split） | 局部 winner |
|---|---:|---:|---|
| iteration 3000 native，1352×1014，92,999 Gaussians | `101.053316 / 101.206854 / 105.618013` | `1.001519 / 1.043586` | `two_stream` |
| iteration 14000 scale-4，338×254，111,525 Gaussians | `113.757326 / 113.507657 / 121.374622` | `0.997805 / 1.069308` | `two_stream` |

两个 workload 的 key 不同，proxy 可测地不同；本轮没有对它们评估主 workload
Tacker variant，也没有复用 profile、生成候选或声称跨 workload winner。

### provenance 与关键 artifact 哈希

- 物理 GPU 1 为 NVIDIA RTX A6000（`sm_86`、84 SM），driver
  `570.124.06`、CUDA 12.4、PyTorch `2.4.1+cu124`、Python 3.10。
- Phase 4 identity：`66cab463bd2682f3f330095f74afc37df4eeb257ba98083825172c40c150f39f`；
  顶层 report/state 文件 SHA-256：
  `51fd1d70dd276e6508250cc9761dff99819ef116ff72c5793dc1f348f2373bd7` /
  `6fe406bcf39377642963cfdab159d20ae45a6208b85c98230fa72e783211d7fe`。
- 主仓/Raster/simple-knn commit：
  `a6c475ee737341c28f88a8fda5fa04479e211592` /
  `79975a092b027cfb374caa2651959942d9aae4f0` /
  `b3554e0fee8a51b4f9201644577ab23c5bb10507`；Tacker runtime commit：
  `a6e84eef97b315424c9587cd534792583b609101`。runtime 的源码集合和稳定
  dirty-status 同样进入 identity，不会被解读为未记录的 clean tree。
- 实际 runtime/head/Raster/simple-knn 二进制 SHA-256：
  `78f4d2b1f85eb91dccaed07fc71597a27babcafda93413f96dfc47bd3f8e0671` /
  `f99bc3134ef9997d3a88f4772476a6641b219db275a522d67327048d3f2db3a5` /
  `cd76862fee530e24a3a96be571d6decf479205caa3c128ed8e4ca84591af263c` /
  `c04852b5c0d0db5cd2c76bdba106c81fca79f3502896dec2ed33e87ea48fdae8`。
  所有实际执行都绑定同一 Raster binary，前后字节稳定性验证通过。
- formal report 文件 SHA-256：
  `8cc23567062e7eb0d43d420ba6d8f9128b80844e1c96d2f3b44769f065982777`；
  admission report 文件 SHA-256：
  `877d7a2af7983cc9bbbd2d0626693ca6e06766ff2ce7ac74df305cc6293289f6`。
- enabled profile 文件/canonical/manifest SHA-256：
  `43ae401fc607b1ca7611f04d1e12a30a788fad5d647d3e37cd1d645789667d8e` /
  `22e27f5ee9cf5f98b53cf4e790bb165163c1a61f9970775da4faabea0d9f22d6` /
  `74ebfa63c3be7d8f3283b098fb9fd6155cf8e9d39ebdf0197ef473b587a4ea52`；
  selected mixed ABI SHA-256：
  `c98ed90853308179443146d3022e5da072c4f507975193f7a01f4fbe4400cf40`。
- canary/rollback/release-selection 文件 SHA-256：
  `f24e2d450cfeb77d3874c6b62103a918049f85bc1450b3e6c6a2a79c436f7e7c` /
  `bd3b135bec857f1dfdcc7d76941063b16c5c564fb4abab23e596e99adabcdf99` /
  `8886c50839549eaaa14bc6bd318500ed99218fd02296954fa149d21df5c7ee41`；
  release canonical SHA-256：
  `e7dd8d081d445b51dc92cccc74704690e9187863163ae2dd26b6fb43d5ba88d8`。

紧凑镜像位于
`tacker_profiles/baselines/a6000_phase4_20260914/run-complete-v5/`；所有上述结论应以
该目录内的 JSON 和顶层 canonical run 为准。

## canary、发布与回滚

发布单位是一对哈希绑定的 `deployment selection + profile`，而不是对
disabled template 的原地改写。

1. admission 仅在 Tacker 候选真正获胜且所有门禁通过时生成新
   enabled/valid profile。若 `serial` 或 `two_stream` 获胜，只发布 baseline
   selection，不生成空 Tacker profile。
2. 将新 profile 作为不可变 regular file 放入唯一 release 目录，用绝对路径
   显式运行 canary；不更改默认执行模式。
3. canary 通过后仍不自动切换生产默认项。只有获得明确运维授权后，
   才先写完并 `fsync` profile/selection 新文件，再用 `os.replace` 原子
   替换 deployment selection。不修改已加载 profile，不使用 symlink；新进程
   重新加载 regular file 并验哈希。
4. 回滚只需原子将 selection 指回封存的 current Tacker enabled profile，或
   切到 `two_stream`。重启/重载后再验证 `actual_execution_mode`、无非预期
   fallback 和 50-view 输出顺序。

本次已完成显式 profile canary、current Tacker 与 two-stream 两项 rollback drill，
并生成不可变 release-selection artifact。但它的状态是
`ready_for_explicit_promotion`，`automatic_default_replacement_performed=false`；canary 也记录
`scope=explicit_profile_only`、`default_profile_replaced=false`。因此当前默认生产
selection/profile 尚未切换。

## CPU contract 回归

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  tests.test_tacker_pipeline \
  tests.test_profile_render_modes \
  tests.test_benchmark_tacker_fps \
  tests.test_tacker_admission \
  tests.test_validate_tacker_modes \
  tests.test_tacker_qualification_script \
  tests.test_run_tacker_phase4 -v
```

`run-complete-v5` 中的远端回归计数为：Phase 4 主 CPU contract 241/241、
head CPU 60/60、Raster CPU 44/44、head CUDA 16/16、Raster CUDA 16/16，
Tacker runtime CTest 1/1。最终本地根目录完整回归为 412/412 tests passed。
