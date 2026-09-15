# Tacker 去 QoS、FPS 优先的融合计划

- 状态：Phase 0–2 complete；Phase 3（C0–C2 beam 范围）complete；Phase 3.1 complete；Phase 4 qualification、显式 canary、release artifact 与 rollback drill complete；生产默认项的显式 promotion 未执行
- 首期目标平台：NVIDIA RTX A6000（`sm_86`）
- 首期目标负载：`flame_steak`，iteration 14000，111,525 Gaussians，1352 × 1014，test 前 50 个视角

## 1. 决策摘要

本次改造不再保护 Raster 子任务的独立 QoS。`raster_slowdown_pct <= 5%` 和 `mixed_p50 < solo_raster + solo_head` 从准入硬门禁降为诊断数据，融合方案只以完整推理渲染序列的端到端 FPS 排序和晋级，FPS 越高越好。

但仅删除 `5.0` 阈值并不能实现自由选择。当前 4DGaussians 推理路径没有候选选择器，而是先把唯一候选写死为 `render leaf + pos_deform[1] Linear(128,128)`，再用 QoS 判断整条 Tacker 路径是否准入。因此需要同时完成：

1. 把单一 pair/profile 升级为多候选 contract；
2. 把固定 `pos_deform[1]` 的 prefix/mixed/suffix 拆分泛化为候选驱动的任务分区；
3. 提供不同融合边界、CTA 配比和 `persistent_blocks` 的 CUDA 变体；
4. 离线短筛所有有效候选，再对封存的 formal finalist 执行完整序列 FPS 测量并选出赢家；
5. 运行时只加载已经验证的赢家，不在线试跑或搜索。

图像质量、数值正确性、DAG 依赖、stream/allocator 生命周期、ABI、同步隔离、设备资源检查和 fail-closed 回退继续保留。这些是正确性与安全约束，不属于本次要取消的 Raster QoS。

## 2. 目标、非目标与度量边界

### 2.1 目标

- 允许 Raster render leaf 与一个或多个 deformation head kernel、完整 head 或更大的可融合子图组成候选。
- 在固定模型、设备和视角序列上，使 N 帧推理渲染的 `throughput_fps` 最大。
- 若新候选没有稳定优于当前最优实现，自动保留当前实现；“不融合”也是合法候选。
- 所有选择结果都可由 workload、代码、ABI、候选参数和测量数据的哈希复现。

### 2.2 非目标

- 不修改训练路径、模型权重或渲染语义。
- 首期不追求跨 GPU、跨分辨率或跨场景的单一通用赢家。
- 不在正式推理期间自动编译、试跑或重新调优。
- 不把 Nsight 的 kernel duration 求和、单个 leaf 耗时或 Raster slowdown 当作最终优化目标。
- 不在首轮同时改造 Raster 的 opaque preprocess/sort 链；它作为独立优化轨处理。

### 2.3 主指标：FPS

主指标定义为：

```text
throughput_fps = measured_frames / measured_elapsed_seconds
```

`measured_elapsed_seconds` 从提交第一个 measured frame 前开始，到最后一个 measured frame 在 consumer stream 上完成并同步为止。该区间包括 Python 调度、prefill、steady state、drain、候选 dispatch 和全部 GPU 完成时间；不包括模型/数据加载、扩展编译、一次性参数缓存、图片编码和文件 I/O。

正式排名使用多次完整序列运行的 `median(throughput_fps)`，按降序选择，不能使用相邻输出完成间隔的 p50 代替。所有进入同一测量阶段的候选必须使用相同帧数、视角顺序和计时边界。

辅助指标包括 `total_render_ms`、mean ms/frame、completion interval p50/p95/max、Raster slowdown、mixed leaf savings、kernel launches、occupancy、寄存器/共享内存和峰值显存。它们用于解释结果，不重新成为 Raster QoS 否决条件。

## 3. 当前状态与限制来源

| 位置 | 当前行为 | 对本次目标的影响 |
|---|---|---|
| `gaussian_renderer/tacker_pipeline.py:50-77` | 固定 `PAIR_KEY`、5% Raster slowdown、单一 measurement schema | 无法描述或比较多个候选 |
| `gaussian_renderer/tacker_pipeline.py:142-188` | 锁死 pair、384 线程、256+128 子组、dtype/barrier 和 QoS 阈值 | 即使删除一个 `if`，物理 ABI 仍只能运行当前 pos linear |
| `gaussian_renderer/tacker_pipeline.py:250-283` | 运行时再次检查 Raster QoS、leaf sum 和 E2E | QoS 是唯一候选的 yes/no admission，不是候选选择 |
| `gaussian_renderer/tacker_pipeline.py:540-697` | prefix 只跳过 `pos_deform[1]`，mixed 只返回一个 head output，suffix 固定执行 `pos_deform[2:]` | 任务分区必须泛化，且必须避免选中算子被 PyTorch 重复执行 |
| `gaussian_renderer/tacker_pipeline.py:934-1087` | 固定两槽和 `prefix_ready -> mixed_done -> ready` 事件链 | 可以复用流水框架，但事件应绑定候选的真实输入/输出依赖 |
| `scene/deformation.py:61-65` | 五个 head 都是 `ReLU -> Linear(128,128) -> ReLU -> tail Linear` | 五个共享 hidden 的 head 是首批自然候选 |
| `profile_tacker_leaves.py:315-335` | `mixed_raster_p50_ms` 实际是完整 `forward_with_head`，含 Raster opaque prefix 和整个 mixed leaf | backend 工作越大越容易被记成 Raster slowdown；该值只应作诊断 |
| `scripts/benchmark_tacker_admission.py:71-78,934-981` | Raster QoS、leaf sum、E2E 和画质均为硬门禁 | 应拆成 correctness qualification 与 E2E selection |
| `tacker_profiles/raster_head_sm86.json:1-39` | 单候选、`persistent_blocks=7000`、禁用模板 | 需要可列举候选和记录赢家的 schema v2 |
| `submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/tacker_mixed.cu:15-90` | 唯一 384-thread wrapper：Raster 256 + head 128 | 需要多种 backend bundle/线程配比的显式 ABI |
| `../Tacker/src/runtime/Scheduler.h:16-25` | 通用 runtime 默认 `max_primary_slowdown_percent=5` | 后续语义需与应用层一致，但修改这里不会自动改变 Python 推理路径 |
| `../Tacker/src/runtime/Scheduler.cc:126-195` | 每个 ready pair 只取一个 mixed registration，按局部 `solo_sum-mixed_p50` 贪心 | 与最大化整段 pipeline FPS 不等价 |

当前封存结果已经接近 QoS 边界：solo Raster 4.177920 ms、mixed 4.365824 ms、solo head 0.261120 ms，记录的 Raster slowdown 为 4.497552%；当前 Tacker E2E p50 为 11.489281 ms，two-stream 为 11.648499 ms。更大的 backend 工作很容易越过 5%，但是否值得应由完整序列 FPS 决定。现有 p50 数据仅作为历史基线，Phase 0 需要重新采集同口径 FPS。

另一个必须写清的边界是：正式 4DGS 路径由 Python `TackerRenderer` 直接调用固定混合 ABI，并不经过 `../Tacker/src/runtime/Scheduler.cc`。因此本计划先改真实推理路径，再同步通用 runtime 的语义。

## 4. 新的准入与选择规则

### 4.1 删除的硬门禁

- `raster_slowdown_pct_max`；
- `mixed_p50_strictly_less_than_solo_sum`；
- 通用 runtime 中 `primary_slowdown_percent > max_primary_slowdown_percent` 的候选过滤；
- 基于 `SubmissionRole` 排除 `BestEffort-left/LatencyCritical-right` 候选的 QoS 方向限制。ABI 声明的 left/right 参数顺序仍然保留。

这些字段可以继续记录在报告中，用于分析为何某个候选更快或更慢，但不影响 correctness-valid 状态。

### 4.2 保留的硬门禁

- 对 Tacker 候选，真实物理路径必须被执行，`actual_execution_mode == "tacker"`，且没有 fallback；
- Raster color/depth/radii 与候选各输出通过 kernel-level 数值检查；
- 50-view PSNR 平均下降不超过 0.05 dB、SSIM 平均下降不超过 `1e-4`、LPIPS 平均上升不超过 `1e-4`；
- model/workload/shape/dtype/alignment、ABI hash、CUDA arch、CTA 最大线程数、寄存器/共享内存和命名 barrier contract 匹配；
- 跨帧 DAG、事件顺序、slot 复用、consumer stream 和 allocator lifetime 正确；
- 候选输出只计算一次，选中的 PyTorch 算子必须被准确跳过；
- launch 或能力检查失败时 fail closed 到 two-stream，再按现有规则回退 serial。

### 4.3 性能选择与部署规则

候选集合必须包含 `serial`、`two_stream` 和当前 `pos_deform[1]` Tacker，实现不会为了“必须融合”而选择更慢方案。

1. 所有通过 ABI、资源和 kernel-level 数值门禁的候选先执行同口径的短 E2E screening；screening 只用于分层扩展与晋级，不作为最终性能分数；
2. formal finalist 集合由全局 screening top-K、每个已生成 C0–C4 family 的最佳 screening-successful 代表（若存在）及 serial/two-stream/current Tacker 按 `variant_id` 去重后，再通过完整 correctness 与 50-view 画质门禁得到；
3. 所有 formal-qualified finalist 按严格交错多 trial 的 `median(throughput_fps)` 降序排列，实验赢家是该封存 finalist 集合的 argmax；报告必须同时保留全量 screening 排行榜和未晋级候选的 terminal status，不得把 finalist argmax 称为未穷举搜索空间的数学全局最优；
4. 部署赢家的 FPS 还必须同时不低于 formal-qualified 的 two-stream 和当前已部署 Tacker；无效基线不参与性能下限比较，而是按 fail-closed 规则单独处置；
5. 为避免噪声导致频繁切换，仅当 incumbent correctness-valid、`candidate_fps/incumbent_fps >= 1.01`，且配对 bootstrap 95% 置信区间下界大于 1.0 时才替换 incumbent；有效 incumbent 未达到晋级条件时予以保留，invalid incumbent 则直接按既有替换/回退规则处置。报告始终记录实验 argmax；
6. 确定部署 winner 时，若多个候选落在 0.5% 的等价区间，选择 ABI 更简单、峰值显存更低、寄存器/共享内存更少的候选；这不改变实验 argmax 的记录。

这里的 1% 是部署稳定性门槛，不是新的 Raster QoS；优化搜索仍报告全量 screening 结果及封存 formal finalist 集合内实测的最高 FPS。

## 5. 目标架构

### 5.1 候选驱动的任务分区

将固定 `PosHeadTask` 升级为通用 `FusionTask`，候选描述符明确列出 prefix、mixed、parallel 和 suffix 中各自负责的节点及输出：

```text
D(t+1) prefix ── prefix_ready ─┬─ selected nodes ─┐
                               │  in mixed kernel │
                               └─ parallel nodes ─┤
R(t) opaque prefix ── render leaf + selected bundle ─ mixed_done
                                                   │
                                  join + suffix ───┴─ ready(t+1)
```

每个 candidate 必须提供：

- `prepare(task)`：产生 mixed 所需输入，并在最后一个真实依赖完成后记录 `prefix_ready`；
- `launch_mixed(...)`：运行 Raster render leaf 与候选 bundle；
- `finish(task, outputs)`：消费全部 mixed/parallel 输出，构造唯一的 `GaussianRenderState`；
- `skipped_python_nodes`：用于测试没有重复计算；
- `required_outputs` 和 stream-lifetime 列表：用于事件与 allocator 安全检查。

两槽 prefill/steady/drain 框架保留，但不再知道 `pos_deform[1]` 的具体含义。

### 5.2 Profile/ABI schema v2

新 profile 用 `workload_key + candidates[] + selected_variant_id` 取代单一 `pair_key`。每个候选至少包含：

- `variant_id`、`fused_nodes[]`、`parallel_nodes[]`、`suffix_nodes[]`；
- CUDA symbol、ABI manifest hash、输入/输出 tensor contract；
- `raster_threads`、每个 backend subgroup 的 thread range、barrier IDs；
- `persistent_blocks`、tile shape、编译时资源数据和 capability requirements；
- correctness 结果、screening/terminal status；若进入 formal，还须记录完整序列各 trial、汇总统计和环境 provenance；
- Raster/leaf/Nsight 诊断，不含 Raster QoS 通过/失败字段；
- `selection_objective: "median_throughput_fps"` 和被选原因。

schema v1 作为 `legacy_pos_l1` 兼容读取；只有 schema v2 能承载多候选并由新 selector 生成。模板仍保持 disabled，调优工具总是写新的 hashed selection manifest，并仅在新 C0–C4 challenger 获胜时写 hashed winner profile；若基线获胜则复用其既有部署 artifact，不生成新的 Tacker profile。禁止原地启用模板。

### 5.3 CUDA 候选实现

Raster 子组继续使用 256 线程和独立命名 barrier。backend 子组不得调用覆盖整个物理 CTA 的 `__syncthreads()`；需要跨 warp 同步时，必须分配与 Raster 隔离的 named barrier，并精确声明参与者。

候选实现按风险递增：

| 层级 | 候选 | 目的 |
|---|---|---|
| B0 | serial、two-stream、当前 384-thread pos-L1 | 稳定基线与回退 |
| C0 | 当前 pos-L1，扫描 `persistent_blocks` | 先验证仅调整持久块数量能否提高 FPS |
| C1 | Raster + 任一单 head 的首个 `Linear(128,128)` | 验证关键路径应迁移哪个 head，而非预设 pos |
| C2 | Raster + 2–5 个 head 的首层 Linear 子集 | 逐步把 deformation 关键工作移入 mixed leaf |
| C3 | 合并多个首层 Linear 为 packed/batched GEMM，共享 `ReLU(hidden)` | 减少重复读取、激活和 launch |
| C4 | Raster + 单个或多个 whole-head 子图 | 探索 first GEMM、ReLU、tail Linear 及 residual 的更大边界 |

对 C1–C4 同时搜索 backend 执行方式和 CTA 配比：单个 128-thread worker 串行处理多个逻辑 bundle，或使用 2/3/4/5 个 128-thread 子组形成 512/640/768/896-thread CTA。所有变体都必须经过 ptxas 资源过滤和真实 occupancy 检查，不能只依赖 `threads <= 1024`。

`persistent_blocks` 至少覆盖：SM 数、2×/4×SM、接近 Raster tile 数、当前 7000、接近 backend logical-block 数。具体值由设备查询和 workload shape 生成，不再由模板写死单值。

为控制组合爆炸，C1 全量生成；C2 在 Phase 3.1 补齐所有 2–5 head set，并对每个 head set 覆盖全部合法 worker-group/`persistent_blocks` 组合。C3 基于 C2 screening top-K head set 生成；C4 基于 C1–C3 各层保留的 top-K head set 生成，以同时覆盖 single-head 和 multi-head whole-head。leaf 数据只用于早期剪枝和解释；screening 只决定候选是否晋级为 finalist，最终 winner 和部署选择必须由严格的完整 E2E formal 结果决定。

### 5.4 离线调优与运行时 dispatch

新增离线 tuner，流程固定为：

1. 枚举候选及 launch 参数；
2. 构建扩展并收集 symbol、ptxas、寄存器、shared memory 和 occupancy；
3. 运行 kernel-level 数值检查，失败候选立即隔离；
4. 执行短 E2E screening，保留全量 screening 排行榜与 terminal status；
5. 将全局 screening top-K、每个已生成 family 的最佳 screening-successful 代表和三种基线合并去重，封存 formal 候选集合；
6. 对全部 formal 候选（包括三种基线）运行或独立重验完整 correctness 与 50-view 画质门禁；不得复用 workload/environment/artifact hash 不匹配的旧资格。challenger 失败时隔离并按同 family 的封存 screening 顺序补位，基线失败时标记 invalid 并按 fail-closed 规则处置；
7. 对全部 formal-qualified finalist 执行严格的交错多 trial 完整序列测量；
8. 对 formal top-3 做 Nsight 分析；
9. 选择赢家并原子写入 hashed selection manifest、完整报告和排行榜；仅在新 C0–C4 challenger 获胜时写 disabled hashed profile，基线获胜时复用其既有部署 artifact。

正式运行仅校验 workload/capability/profile hash，然后通过 `selected_variant_id` 一次 dispatch 到固定 variant。调优时间和搜索复杂度不会进入推理关键路径。

## 6. 分阶段实施

### Phase 0：冻结基线并修正计量口径

- [x] 扩展 `profile_render.py`，增加 whole-run `--trials`、p95/max、环境信息和每 trial 的 `throughput_fps`；同时保留 `total_render_ms` 与原有单次字段供诊断和兼容。
- [x] 新增交错运行驱动，按轮转/ABBA 顺序测 `serial`、`two_stream`、current Tacker 和候选，避免固定“先 two-stream 后 Tacker”的温度/时钟偏差。
- [x] 记录 GPU clocks、temperature、power mode、driver/CUDA/PyTorch、代码 commit、子模块 commit 和 profile hash。
- [x] 在 A6000 上复现当前结果，并把 raw JSON 作为新调优报告的 baseline 输入。
- [x] 明确 `throughput_fps = frames / elapsed_seconds` 是正式主指标，completion interval p50 和总时长只作辅助诊断。

退出条件：同一实现的 10 次 50-frame trial 可重复，报告能计算 paired ratio 和 bootstrap 95% CI。

Phase 0 已于 2026-09-10 在独占的 RTX A6000 GPU 1 上完成。三种模式按 ABBA、seed 0 各运行 10 次，每次 warmup 10 帧并测量 test 视角 0–49；30/30 次均通过实际执行模式、无 fallback、workload、计时边界、profile 内容哈希和 provenance 校验。

| 排名 | 模式 | `median(throughput_fps)` | `median(total_render_ms)` | FPS 范围 |
|---:|---|---:|---:|---:|
| 1 | current Tacker | 86.979719 | 574.846646 ms | 86.597897–87.099374 |
| 2 | two-stream | 86.254250 | 579.681585 ms | 85.856130–86.436900 |
| 3 | serial | 82.423385 | 606.623962 ms | 81.264402–82.890268 |

current Tacker / two-stream 的 FPS ratio-of-medians 为 `1.0084108268`，按 paired round 做 10,000 次固定 seed bootstrap 得到 95% CI `[1.0061090994, 1.0097120318]`。聚合报告、30 份原始 trial JSON、哈希和复现说明封存在 `tacker_profiles/baselines/a6000_flame_steak_phase0_20260910/`。

### Phase 1：拆分 correctness qualification 与性能选择

- [x] 将 profile 升为 schema v2；v1 以只读 legacy candidate 继续支持。
- [x] 从 `gaussian_renderer/tacker_pipeline.py` 和 `scripts/benchmark_tacker_admission.py` 移除 Raster QoS、mixed leaf sum 的否决逻辑，保留原始测量为 diagnostics。
- [x] 把 E2E 比较从单个 `tacker/two_stream p50` 改为多 candidate 的 whole-run 排名和 incumbent promotion。
- [x] 更新 disabled template、profile hash、provenance 和原子写入 contract。
- [x] 增加明确测试：Raster slowdown 大于 5% 的候选，只要 correctness 通过且 E2E 为赢家，就可以被选择。

退出条件：使用合成数据时，selector 能在多个候选和“不融合”基线间稳定选择全局最高 FPS，并拒绝任何 correctness-invalid 候选。

**完成状态：2026-09-10**

- **CPU contract**：覆盖多候选全局 FPS 排名、`correctness-invalid` 预过滤、0.5% 等价集的稳定资源优先级、1% + paired bootstrap CI promotion、invalid incumbent 替换、two-stream 性能下限、baseline 获胜时不生成 Tacker profile，以及生成 profile 后的运行时交叉验证。
- **兼容性**：Phase 0 的真实 A6000 数据可被新的 admission parser 作为兼容输入。
- **GPU 结论**：Phase 1 本身不新增 CUDA variant，因此未伪造新的 GPU 性能结论。

**发布收口要求**

- **统计证据**：锁定为 10,000 次、seed 0 的 paired bootstrap。
- **执行轨迹**：重建并校验 ABBA/round-robin 的每轮执行轨迹。
- **选择一致性**：对 selector 完整输出与 admission 独立重算做一致性检查。
- **Legacy 兼容**：仅限封存 Phase 0 报告的精确 canonical SHA-256。
- **Provenance 采集**：源码、config 和已加载 Raster 二进制的 provenance 哈希在 warmup/计时前采集，以避免后采样竞态。
- **Trial 绑定**：每个 Tacker trial 的 manifest/selection SHA、variant、ABI 和 `persistent_blocks` 均与 SHA-bound source profile 逐项绑定。
- **Fail-closed**：qualification 状态和所有可空证据均使用显式字段，并按 fail-closed 原则处理。

### Phase 2：泛化 Python 任务分区与混合 ABI

- [x] 将 `PosHeadTask`、`prepare_pos_head_task`、`finish_pos_head_task` 和 `_forward_with_head` 抽象为候选接口。
- [x] 保留两槽事件链，针对每种 partition 验证 prefix/mixed/parallel/suffix 的依赖和输出生命周期。
- [x] 在 `tacker_ext` 增加多 head/packed/whole-head device adapter 和独立 ABI manifest。
- [x] 在 Raster 子模块增加多 variant mixed wrapper、Python binding 和 capability query。
- [x] 对每个 variant 收集 ptxas 资源；补上通用 runtime 目前缺失的寄存器/occupancy 过滤。（A6000 实测 mixed v2 为 68 registers/thread、7376 B static shared memory、0 spill；1–5 个 worker group 的 runtime query 均取得非零 occupancy，并由 profile 生成与运行时路径 fail closed 校验。）
- [x] 增加重复执行检测、空输入/尾块、非对齐、错误 dtype/shape、alias、barrier 和 launch-failure fallback 测试。

退出条件：至少完成 C0、全部 C1 和一个 C2 双-head variant，且它们均能在真实 50-view 流水中无重复计算、无 fallback 地执行。

**实现状态：2026-09-10，真实 A6000 Phase 2 退出验收通过**

- **候选覆盖**：通用 `FusionTask`/`FusionPartition` 已覆盖 C0、五个 C1 和 `pos+scales` C2；生成器仅写出 disabled qualification profile，不伪造资源或性能数据。
- **ABI 封存**：legacy head/Raster SHA-256 分别为 `24570aa6e67e8b9b10fa94524fec4dc03a4eb3fdc3bf822af34c2c52ce4937ac` / `231c90c429321b2673b88ecd09efb40b6aedda7a23f3e061a2bcedec06d44426`；head v2 为 `9d6a1558acd6b642b975bcabe22abcbe3fd7242e4c9e0d635636ef4d2eb5da7f`，Raster v2 显式锁定该依赖，自身为 `310b15957c5920773bb03a61a37c5771f6d4570393061ece4e1805fd20989056`。四个哈希均由已加载 Raster binary 的 capability 回报并与 profile 核对。
- **本机验证**：175 项 Phase 2 CPU/源码契约通过；28 项 CUDA 测试因本机无 PyTorch/CUDA 按预期跳过；另有 18 项 `profile_render` 元数据契约测试通过。Python 3.7 语法、shell 语法和两个工作树的 `diff --check` 通过。
- **远端路径**：通过 Windows OpenSSH 客户端连接 `4A6000` alias，在隔离快照 `/data/qyfeng/tacker_phase2_validation/20260910-codex-phase2` 内构建和测试，没有修改原有脏工作树。使用 PyTorch `2.4.1+cu124`、CUDA 12.4 和物理 GPU 1；`nvidia-smi` 因 NVML driver/library `580.173` 不匹配而不可用，但 PyTorch CUDA 编译、加载和执行正常，因此这里不声称完整 Phase 4 qualification preflight 已通过。
- **NVML 后续处置（2026-09-11）**：确认故障由已加载且磁盘上仍在使用的 `570.124.06` 内核模块与系统默认 `580.173.02` NVML 用户态库混装导致；服务器有其他用户的长期 GPU 进程，且当前账号无免密 sudo，因此没有热卸载模块或擅自重启。隔离快照内新增 `.qualification-bin/nvidia-smi` 和 `.qualification-bin/with-driver-nvml`，仅对验收命令预加载现存、与内核模块精确匹配的 `libnvidia-ml.so.570.124.06`。在后者包裹下，4 张 RTX A6000 的 NVML 查询、物理 GPU 1 映射、PyTorch `2.4.1+cu124`、CUDA 12.4、CC 8.6 与实际 CUDA 张量执行组成的 exact device/environment preflight 已通过且无 NVML 警告；这不等于完整 Phase 4 已通过。系统级根治仍需管理员统一 NVIDIA 内核/用户态版本并在维护窗口重启。
- **CUDA correctness**：重编后的 head 完整套件 60/60、Raster v2/legacy 完整套件 34/34 通过，其中 CUDA 定向测试分别为 16/16 与 12/12；覆盖真实 binding、尾块/非对齐输入、错误 contract、barrier 以及 legacy 路径。
- **编译资源**：ptxas 报告 mixed v2 为 68 registers/thread、7376 B static shared memory、0 spill；whole-head 为 32 registers/thread、512 B static shared memory、0 spill；packed 为 48 registers/thread、0 spill；multi GPTB/solo 为 48 registers/thread、200 B stack、0 spill。runtime query 对 1–5 个 worker group 分别返回 384/512/640/768/896 threads，active blocks/SM 为 2/1/1/1/1，occupancy 为 0.5/0.333333/0.416667/0.5/0.583333，且全部 `launch_supported=true`、kernel max threads 为 896。
- **50-view 物理路径**：7 份 disabled qualification profile（C0、全部五个 C1、一个 `pos+scales` C2）各以 warmup 10、50 measured views、111,525 Gaussians、1352 × 1014 跑通；最终 metadata 均通过已加载 binary manifest SHA 与实时资源门控，且为 `actual_execution_mode=tacker`、`tacker_fallback_reason=null`。最终单 trial diagnostic FPS 依次为 C0 87.983156、C1 opacity 85.842899、C1 pos 85.779940、C1 rotations 85.229702、C1 scales 84.982640、C1 shs 85.338641、C2 pos+scales WG2 66.852933。
- **无重复执行证据**：每个候选的 50-view 复跑均记录 `input_frames=50`、`full_deformation=1`、`prefix=49`、`mixed_launches=49`、`suffix=49`、`solo_raster=1`、`outputs=50`、`selected_head_evaluations_per_head=50`，并在最后一个输出前由运行时不变量 fail closed 校验。结合 selected Linear 的 Python 跳过/重复调用测试，Phase 2 的无重复计算、无 fallback 退出条件已满足。
- **证据边界**：上述 FPS 只证明当时接入的 C0–C2 候选能在真实流水执行，不是 Phase 3/3.1 的交错多 trial 排名，也没有选出 winner；C3/C4 的 adapter 数值与资源证据不等于它们已接入生产 dispatch。完整 Phase 4 preflight、50-view 画质门槛和正式 admission 仍未声称完成。定向验证摘要保存在 `tacker_profiles/baselines/a6000_phase2_20260910/`。

### Phase 3：实现 E2E autotuner 与候选选择（C0–C2 beam 范围）

- [x] 新增候选矩阵生成和 profile DB 工具，支持 top-K/beam search、参数去重与断点续跑。
- [x] 泛化 `profile_tacker_leaves.py`，按 variant 输出数值、leaf、资源和 workload provenance。
- [x] 对全部 366 个 beam-generated 候选执行短 E2E screening，再对 correctness-valid top-5 执行完整 E2E 交错 benchmark，并生成稳定的 screening/formal 排行榜。
- [x] 在封存的 C0–C2 beam-generated formal finalist 集合内自动选择实验 argmax 和部署 winner；写出选择理由、置信区间、与 current/two-stream 的 FPS paired ratio。
- [x] 对 top-3 运行 Nsight，确认关键路径变化、空转、straggler、同步和 kernel launch 数。

退出条件：给定同一候选集合和测量输入，selector 结果确定且 profile hash 可复现；正式运行没有在线搜索开销。

**Phase 3（C0–C2）实机完成记录（2026-09-12）**：

- 在物理 GPU 1 的 RTX A6000 上完成分层搜索：H1/H2/H3/H4/H5 分别执行 78/246/222/150/66 次短 E2E 测量，共 762/762 成功；最终矩阵包含 366 个去重候选，sealed matrix SHA-256 为 `8956a212b0ee99d9cf17fb2de3a43e793745eadfcc0651924fb135e2603c4295`。
- 排名前五的候选全部通过 variant leaf gate 与 50-view PSNR/SSIM/LPIPS gate；正式阶段按 warmup 10、50 frames、10 trials、ABBA 交错执行 80/80 次，无运行错误。
- 在本轮 C0–C2 候选集合内，实验 argmax 与部署 winner 均为 `c2h5_pos_scales_rotations_opacity_shs_l1_wg1_pb5440`，median FPS 为 `89.7762405051`。相对 current Tacker 的 ratio-of-medians 为 `1.0255928954`，paired bootstrap 95% CI 为 `[1.0215977858, 1.0297240659]`；相对 two-stream 为 `1.0369500048`，95% CI 为 `[1.0334421818, 1.0396602793]`，因此 selector 给出 `promote_challenger / promotion_gates_passed`。
- Top-3 Nsight 均完成 50-frame 诊断。winner 的 Nsight render FPS 为 `82.6741330624`，121 launches/frame、17 stream synchronizations/frame，关键路径代理为 `renderer/setup`；首轮 qdstrm importer 的 `Wrong event order` 使编排 fail closed，随后在同一代码、输入、Nsight 2023.4.4.54 和运行身份下从 checkpoint 恢复成功，已有 762 次筛选、5 个 correctness gate 和 80 次正式测量均未重跑。
- postflight 重新验证 SQLite、重算 H3–H5 矩阵和 formal selection，并重生成全部 366 份 qualification profile：矩阵字节一致、formal 选择记录一致、profile hash mismatch 为 0。证据摘要保存在 `tacker_profiles/baselines/a6000_phase3_20260912/`。
- **范围边界**：本轮实测覆盖已经接入生产 Raster 路径的 C0–C2 first-linear family，但 C2 按 `beam_width=3` 分层生成，不是穷举搜索：H2 覆盖 10/10 个 head set，H3 覆盖 6/10，H4 覆盖 3/5，H5 覆盖 1/1；每个被保留的 head set 均覆盖全部合法 worker-group 与 6 个去重后的 effective `persistent_blocks`。共330 个 C2 候选完成短 screening，其中只有 top-5 完成严格 10-trial formal 测量；因此本节的 argmax/winner 仅限 beam-generated formal finalist 集合。Phase 2 已实现的 C3 packed / C4 whole-head adapter 尚未接入该运行时搜索路径，它们的生产接入与重排见 Phase 3.1。本节选出的 profile 仍是 `enabled=false, valid=false` 的 qualification profile；Phase 3.1 若选出新 C0–C4 challenger，其最终 profile 正式启用留给 Phase 4。

### Phase 3.1：补齐 C2 候选矩阵、接入 C3/C4 并重开 E2E 排名

- [x] 将 Phase 2 已实现的 C3 packed 与 C4 whole-head adapter 接入 `TackerRenderer` 的生产 Raster mixed 路径、候选 registry/matrix generator、`selected_variant_id` dispatch 与 fail-closed fallback，不使用只能单独调用的诊断旁路代替真实运行时接入。
- [x] 先补齐被原 beam 剪掉的 4 个 H3 和 2 个 H4 head set；按当前 A6000 的合法 worker-group × 6 个 effective `persistent_blocks` 网格，新增并短筛 120 个去重候选，使 C2 的 2–5 head-set/launch 矩阵达到 450/450 结构性覆盖。这一项只声称全量 screening 覆盖，不声称 450 个候选均已完成 10-trial formal 测量。
- [x] 按第 5.3 节的分层规则，以补齐后的 C2 screening top-K head set 生成 C3；C4 从 C1–C3 各层保留的 top-K head set 生成，必须同时覆盖 single-head 和 multi-head whole-head。同时搜索 backend 执行方式、worker-group/CTA 配比与 `persistent_blocks`，并在 E2E 前执行 symbol、ptxas、register/shared-memory、occupancy 和 launch-support 过滤。
- [x] 对 C3/C4 补齐 partition、选中算子只执行一次、prefix/mixed/parallel/suffix 依赖、event/stream/slot 生命周期、尾块/非对齐输入、ABI/shape/dtype/alignment 与 launch-failure fallback 测试；至少各有一个 C3 和 C4 候选在真实 50-view 流水中无重复计算、无 fallback 地完成。
- [x] 以新的候选集重新执行 kernel-level 数值门禁、短 E2E screening 和 top-K/beam search；不向 Phase 3 已 sealed 的 366 候选矩阵直接追加零散测量。
- [x] 在同一设备、workload、视角顺序和计时边界下，以“全局 screening top-K + 每个已生成 C0–C4 family 的最佳 screening-successful 代表 + serial/current Tacker/two-stream”去重得到 formal 候选集合；对其重新执行完整 correctness 与 50-view PSNR/SSIM/LPIPS gate，再对 formal-qualified finalist 重跑交错多 trial 完整序列测量、formal selection 和 top-3 Nsight。
- [x] 封存新的 candidate-matrix hash、全量 screening 排行榜与 terminal status、formal finalist 排行榜、选择理由、paired bootstrap CI 和 selection manifest。仅当部署选择为新 C0–C4 challenger 时生成 `enabled=false, valid=false` 的 qualification profile；若三种基线之一获胜，则复用其既有部署 artifact，不生成新的 Tacker profile。

退出条件：C2 达到 450/450 结构性 screening 覆盖；至少一个 C3 和一个 C4 通过真实生产 dispatch 参加 formal 交错多 trial 比较；给定同一候选集和测量输入，新 selector 结果确定且 matrix/selection/profile（如有）hash 可复现，正式运行仍无在线搜索开销。

**Phase 3.1 实机完成记录（2026-09-14）**：

- **隔离身份与环境**：最终有效证据只来自新的 `run-complete-v3` 链。主仓验证快照 commit 为 `ad71abe0f90d606bc9a5f3955d05cef1ce781e3d`，Raster/simple-knn 子模块 commit 分别为 `7d8fb35515521c11e75f2a3d20b5764fa2279790` / `44f764299fa305faf6ec5ebd99939e0508331503`，run identity 为 `0f16b7250611891db52cc59dada0f8bd77c40d735c26f98aed7c81047c14d8a2`。实测设备为物理 GPU 1 的 RTX A6000（`sm_86`、84 SM），driver `570.124.06`、CUDA 12.4、PyTorch `2.4.1+cu124`。
- **ABI/CUDA 退出门禁**：head/Raster 已加载二进制 SHA-256 分别为 `cf75c80f737169b6c9c96610cbde78ff7b2359691178b08d31e3ddb2c541f0d9` / `ef1478ffd323df4d9707044f01af4249d2f940d54b37b01600c354155fce9fa7`；Raster ABI v1/v2/v3/v4 manifest 分别为 `231c90c429321b2673b88ecd09efb40b6aedda7a23f3e061a2bcedec06d44426` / `310b15957c5920773bb03a61a37c5771f6d4570393061ece4e1805fd20989056` / `c98ed90853308179443146d3022e5da072c4f507975193f7a01f4fbe4400cf40` / `293b8471fc9397070f1d1ebbe1297420f24f49e6882369e2e6cf8dcd9d49b7a1`。A6000 上 head CUDA 60/60、Raster ABI1–4 CUDA 44/44、C3/C4 runtime/fallback 48/48 和 Phase 3.1 CPU contract 271/271 通过；C3 packed profiler 修复后的远端专项测试为 42/42。所有 8 个封存 kernel 的 ptxas 资源门禁通过且 spill 为 0；C3/C4 mixed kernel 分别为 70/36 registers/thread、7168/10016 B static shared memory。
- **完整新矩阵**：最终矩阵为 661 个去重候选：C0/C1/C2/C3/C4 分别 6/30/450/84/91。C2 的 H2/H3/H4/H5 分别为 120/180/120/30，head set 覆盖 10/10、10/10、5/5、1/1；这里仅声明 450/450 已完成短 screening，不声明其全部进入 formal。matrix SHA-256 为 `aa45c9dce881a5a71c9d86130fcd6b12070ee2773b83b44a538f2147bbde8577`。
- **全量 screening 与 qualification**：661/661 候选均达到 terminal/successful，失败数为 0；screening ranking SHA-256 为 `32e64e73703ce5a2e0ea07e1113139e2eec16ce8b3b87810b4d59268b5e770b7`。全局 top-5 与各 family 最佳代表去重后得到 9 个 generated finalist，再加入 serial/two-stream/current Tacker；9 个 challenger 的 kernel leaf 与独立 50-view 画质门禁全部通过，C0–C4 family 均有有效代表且无需 backfill。formal candidate-set/qualification-plan SHA-256 分别为 `7c592726cb2edfeb9ea7c030b569d213b179c0917d31a9d1b4a56007abc54102` / `48f68059dea3a877d34275abb784cf0779e2a5b180c5977363f408ec02bc6b3d`。
- **真实 C3/C4 路径**：五个 C3 和一个 C4 challenger 进入 formal。每个 challenger 的 10 个 50-frame 子进程均为 `actual_execution_mode=tacker`、无 fallback；90/90 challenger 序列逐个验证 `input_frames=50`、`full_deformation=1`、`prefix/mixed/suffix=49`、`solo_raster=1`、`outputs=50`、每个 selected head 恰好计算 50 次。C3 winner 的 50-view 画质 delta 为 PSNR drop `0.000209961 dB`、SSIM drop `1.3161e-6`、LPIPS increase `6.7294e-7`；C4 opacity whole-head 代表也通过相同阈值。
- **正式 E2E 结果**：12 个 formal-qualified entry 在相同的 test 视角 0–49 上按 warmup 10、50 measured frames、10 trials、ABBA、seed 0 运行 120/120 次。实验 argmax 与部署 winner 均为 `c3_packed_first_linear_pos_scales_rotations_opacity_shs_wg1_pb5440`，median FPS `100.9417269038`（median 50-frame elapsed `495.335342 ms`）；current Tacker/two-stream 分别为 `87.2573058329` / `86.3680297778` FPS。winner 相对 current Tacker 的 ratio-of-medians 为 `1.1568283703`，10,000 次 paired bootstrap 95% CI `[1.1544964934, 1.1592246266]`；相对 two-stream 为 `1.1687394880`，95% CI `[1.1640678357, 1.1715347439]`，因此决策为 `promote_challenger / promotion_gates_passed`。formal report SHA-256 为 `d56c0385f8e86a3e8aa2927fad449d950c83073025a55f37d1e8e1831b545d98`。
- **选择与部署边界**：selection SHA-256 为 `1a91a02c2605ae442817bb060b7956ead5d3b940146d274b94a85e00849ec126`。按计划只生成 winner qualification profile；其文件 SHA-256 为 `5c3c2031aabe8018d11f41a90e324f375b1a7a14cdfaaef2db730514ec027a51`，内部 canonical profile SHA-256 为 `f91e5f129708ee7612e4a45e95b00e23fe15a92eb1f1be376c2a85095379599e`，且仍保持 `deployment.enabled=false, valid=false`，正式 admission/灰度留给 Phase 4。
- **Top-3 Nsight**：formal top-3 均为 C3 五头 packed 的 `pb5440/pb7000/pb13942`；rank-1 Nsight diagnostic 为 `88.027670 FPS`、`113.16` launches/frame、`17` stream synchronizations/frame，关键路径代理为 `renderer/deformation`。第一次 rank-3 qdstrm importer 因 Nsight `Wrong event order` fail closed；同一 identity 从 checkpoint 恢复，复用前两份结果并仅重跑 rank-3 后完成，最终 Nsight report SHA-256 为 `6ccfe718480aa03e779928403efa9e7c35d21e7ec63f36e6f5cc868ea7ff659e`。
- **独立重放**：仓库外 postflight 对 49 个成功 stage 的 7354 个唯一产物逐项验哈希，独立重建矩阵、661 份 profile、formal 原始 trial 聚合、selection 与三份 Nsight CSV 摘要；profile hash mismatch 为 0。最终 run report/postflight 文件 SHA-256 分别为 `b24699941748fff134830bf59e0a0001bc64ac419ed119d10481bb894b956ea9` / `4949cb3206df127e5c205cbcb1eb47aa32c942da636947680a698b0a424650eb`。紧凑证据封存在 `tacker_profiles/baselines/a6000_phase31_20260913/`。
- **历史与复现边界**：`run-complete-v2` 只用于发现 C3 ABI3 packed 输出被 leaf profiler 错当成单 head 的解码问题；修复改变了 identity，因此 v2 的排名、资格结果和 hash 均未混入最终结论。v2 首次 LPIPS 运行还因外部 VGG 权重获取停滞；v3 使用环境中已有的官方 LPIPS v0.1 VGG 权重缓存完成全部质量门禁。Phase 4 的 1/2/50+长 steady-state、完整峰值显存、独立 `../Tacker` runtime/TOCTOU 和真实 profile-missing fallback smoke 仍未在本阶段声称完成。

**证据边界**：Phase 3 的 matrix、screening/formal 排行榜和 C2 winner 作为 C0–C2 beam 搜索的封存历史保留，不作为 Phase 3.1 的最终选择证据。Phase 4 必须消费 Phase 3.1 新生成的 selection report/manifest 及与选中模式对应的部署 artifact，不得仅将 C2 补测或 C3/C4 的零散结果拼接到旧排行榜后直接启用原 profile。

### Phase 4：准入、回归与发布

入口条件：Phase 3.1 退出条件已满足，新的 candidate matrix、selection report/manifest 和选中模式对应的部署 artifact 已封存；仅在新 C0–C4 challenger 获胜时要求 disabled qualification profile。

- [x] 将 source/config/已加载二进制 provenance 前移到 import/parse 之前，并增加前后字节不变校验，关闭并发原地改写的 TOCTOU 窗口。
- [x] 更新 `scripts/run_tacker_qualification.sh`，串起构建、资源检查、数值、对 sealed matrix/finalist 的候选 E2E 独立复验、画质、selection 重算、常规 profile 复跑和 fallback smoke test；Phase 4 不再扩展候选集。
- [x] 更新 CPU contract、CUDA 数值测试和真实 GPU 性能回归。
- [x] 在 1、2、50 帧序列验证 prefill/drain，在 500 帧循环序列验证 steady state。
- [x] 对两个声明为不同 Raster/deformation mix、且 whole-run overlap proxy 可测地区分的额外 workload 完成 baseline-only 泛化评估；仍按 workload-key 独立报告，不复用主负载 Tacker profile，也不声称全局 variant。
- [x] 更新 `TACKER_IMPLEMENTATION.md` 和 `tacker_profiles/README.md`，说明 QoS-free selection、profile v2、复现命令和回滚流程。
- [x] 新 C3 challenger 获胜后生成 run-local 独立 enabled/valid admission profile artifact，并完成显式 profile canary、不可变 release-selection artifact 以及 current Tacker/two-stream rollback drill。
- [ ] 运维侧显式 promotion：最终 release 状态是 `ready_for_explicit_promotion`，且 `automatic_default_replacement_performed=false`；本轮未替换默认 deployment selection/profile。

**Phase 4 实机完成记录（2026-09-15）**：

- **最终身份与证据链**：唯一最终结论来自 `/data/qyfeng/tacker_phase4_validation/20260914-codex-phase4/run-complete-v5`，13/13 个 stage 均为 `succeeded`，Phase 4 identity 为 `66cab463bd2682f3f330095f74afc37df4eeb257ba98083825172c40c150f39f`。顶层 report/state 文件 SHA-256 分别为 `51fd1d70dd276e6508250cc9761dff99819ef116ff72c5793dc1f348f2373bd7` / `6fe406bcf39377642963cfdab159d20ae45a6208b85c98230fa72e783211d7fe`；紧凑证据镜像位于 `tacker_profiles/baselines/a6000_phase4_20260914/`。
- **环境、源码与二进制 provenance**：物理 GPU 1 为 RTX A6000（`sm_86`、84 SM），driver `570.124.06`、CUDA 12.4、PyTorch `2.4.1+cu124`、Python 3.10。隔离主仓/Raster/simple-knn commit 为 `a6c475ee737341c28f88a8fda5fa04479e211592` / `79975a092b027cfb374caa2651959942d9aae4f0` / `b3554e0fee8a51b4f9201644577ab23c5bb10507`；Tacker runtime commit 为 `a6e84eef97b315424c9587cd534792583b609101`，其源码集合和稳定 dirty-status 哈希均进入 identity。实际 runtime/head/Raster/simple-knn 二进制 SHA-256 为 `78f4d2b1f85eb91dccaed07fc71597a27babcafda93413f96dfc47bd3f8e0671` / `f99bc3134ef9997d3a88f4772476a6641b219db275a522d67327048d3f2db3a5` / `cd76862fee530e24a3a96be571d6decf479205caa3c128ed8e4ca84591af263c` / `c04852b5c0d0db5cd2c76bdba106c81fca79f3502896dec2ed33e87ea48fdae8`，所有执行都绑定同一 Raster binary。
- **构建、资源和正确性**：远端 Phase 4/head/Raster CPU contract 分别 241/241、60/60、44/44，通过 head/Raster CUDA 16/16、16/16 和 runtime CTest 1/1；39 条 ptxas 记录均为 0 spill。10/10 份 Tacker 资源/数值记录有效，12/12 个 formal entry 的 50-view correctness 有效。winner 使用 70 registers/thread、7168 B static shared memory、occupancy `0.5`；其 Raster slowdown `31.154%` 只作为诊断，不触发 QoS 否决。50-view delta 为 PSNR drop `0.000209961 dB`、SSIM drop `1.3161e-6`、LPIPS increase `6.7294e-7`。
- **正式性能与 admission**：封存的 12 个 entry 按 warmup 10、50 measured frames、10 trials、ABBA/seed 0 完成 120/120 次执行。实验 argmax 和 deployment selector winner 均为 `c3_packed_first_linear_pos_scales_rotations_opacity_shs_wg1_pb5440`，median FPS `100.5082562657`；current Tacker/two-stream 为 `87.3984074601` / `86.3416638762` FPS。winner/current ratio-of-medians 为 `1.1500010033`，paired bootstrap 95% CI `[1.1446137379, 1.1542292551]`；winner/two-stream 为 `1.1640759716`，CI `[1.1596655823, 1.1687732974]`，formal/admission selector 决策为 `promote_challenger / promotion_gates_passed`。formal report 文件 SHA-256 为 `8cc23567062e7eb0d43d420ba6d8f9128b80844e1c96d2f3b44769f065982777`。
- **序列、显存与负向回退**：1/2/50/500 帧均验证输出数、prefill/mixed/suffix/drain 和每个 selected head 恰执行 N 次；500 帧 p50/p95 为 `9.840576/9.891193 ms`，峰值 allocated/reserved 为 `1,282,750,976` / `2,124,414,976` B。独立 enabled-profile 10 × 50 常规复跑为 `actual_execution_mode=tacker` 且无 fallback。missing、stale-workload、hash-mismatch 3/3 均可见地回退到 `two_stream` 并给出非空原因。
- **泛化与发布边界**：iteration 3000 native 与 iteration 14000 scale-4 两个 evaluation-only workload 的 overlap proxy 可测地区分，二者 baseline winner 都是 `two_stream`（`105.6180133902` / `121.3746216960` FPS）；没有复用主 workload profile 或声称跨 workload winner。admission 生成 profile 的文件/canonical SHA-256 为 `43ae401fc607b1ca7611f04d1e12a30a788fad5d647d3e37cd1d645789667d8e` / `22e27f5ee9cf5f98b53cf4e790bb165163c1a61f9970775da4faabea0d9f22d6`。显式 canary、两项 rollback drill 与 release artifact 均通过，其文件 SHA-256 分别为 `f24e2d450cfeb77d3874c6b62103a918049f85bc1450b3e6c6a2a79c436f7e7c` / `bd3b135bec857f1dfdcc7d76941063b16c5c564fb4abab23e596e99adabcdf99` / `8886c50839549eaaa14bc6bd318500ed99218fd02296954fa149d21df5c7ee41`；release 明确为 `ready_for_explicit_promotion`、`automatic_default_replacement_performed=false`，资格链完成不代表默认生产指针已切换。

退出条件：满足第 8 节全部验收标准，并能够只替换 deployment selection/profile 回滚到 current Tacker 或 two-stream。

### 独立并行轨：移除 Raster rendered-count D2H 同步

Raster 当前每帧为了获取 `num_rendered` 有一次 4-byte Device-to-Host 同步，它会阻塞后续 buffer sizing。该问题不属于 QoS 放宽，但可能显著影响 FPS。应独立评估预分配上界、buffer 复用或 device-side sizing；由于涉及内存安全，不与首轮 fusion 改动混在同一个变更中，也必须单独做峰值显存与越界验证。

## 7. 预计修改面

| 文件/组件 | 计划修改 |
|---|---|
| `gaussian_renderer/tacker_pipeline.py` | profile v2、通用 FusionTask/variant registry、候选 dispatch、QoS-free admission、事件/回退 |
| `profile_render.py` | whole-run multi-trial、p95、环境和 paired measurement 元数据 |
| `profile_tacker_leaves.py` | 从单 pos-L1 profiler 泛化为 variant profiler；leaf 指标降为诊断 |
| `scripts/benchmark_tacker_admission.py` | 拆分 correctness qualification、candidate ranking 和 incumbent promotion |
| `scripts/run_tacker_qualification.sh` | 构建候选、交错 E2E、选择、质量、常规复跑 |
| `scripts/validate_tacker_modes.py` | 支持按 `variant_id` 输出逐候选质量与真实执行信息 |
| `tacker_profiles/*.json`、`tacker_profiles/README.md` | schema v2 disabled template、selection manifest、条件性 winner profile 和复现说明 |
| `tacker_ext/include`、`tacker_ext/csrc`、`tacker_ext/abi` | multi-head/packed/whole-head GPTB adapter 与 ABI |
| Raster 子模块 `cuda_rasterizer/tacker_mixed.*`、binding、ABI | 多 mixed wrapper、资源/capability 描述、variant dispatch |
| `tests/test_tacker_*.py`、`tacker_ext/tests`、Raster CUDA tests | 新 schema、选择器、依赖、数值、fallback 和 GPU 回归 |
| `../Tacker/src/runtime/{Scheduler,Registry}.*` | QoS-free 选项、多 mixed registration、E2E/FPS score、资源检查；作为第二阶段语义同步 |

Raster 子模块和 `../Tacker` 是独立版本边界，实施时应分别提交并在主仓 profile provenance 中锁定精确 commit。

## 8. 验收标准

### 8.1 功能与正确性

- [x] profile 能同时描述多个 correctness-valid candidate，并唯一记录 `selected_variant_id`。
- [x] 一个 Raster slowdown 明显高于 5% 的合成或真实候选不会因 Raster QoS 被拒绝。
- [x] 所有被选节点只执行一次；1/2/50 帧的输出顺序和 legacy renderer 一致，500 帧另行验证 steady-state 执行计数与显存。
- [x] 事件、stream 和 slot 复用测试覆盖 prefix、parallel、mixed、suffix、drain 和异常回退。
- [x] 任意 ABI、shape、dtype、alignment、resource、manifest 或 launch 错误都 fail closed。
- [x] kernel 数值阈值保持：color/depth max abs `1e-5`、radii mismatch 为 0、head 输出 atol/rtol `2e-3`，或为新输出定义不弱于等价 FP32/legacy 参考的明确阈值。
- [x] 50-view 画质阈值保持：PSNR drop `<= 0.05 dB`、SSIM drop `<= 1e-4`、LPIPS increase `<= 1e-4`。

### 8.2 性能

- [x] Phase 4 在同一 A6000、同一 50-view 顺序、warmup=10 下独立重测 Phase 3.1 封存的 12 个 formal-qualified entry；每个 entry 完成 10 次完整序列 trial，并按 ABBA 交错执行。
- [x] Phase 4 正式 winner 是封存的 formal-qualified finalist 集合中 `median(throughput_fps)` 最高者；报告保留 Phase 3.1 全量 screening 边界并将其与 Phase 4 formal 重测区分。
- [x] Phase 4 deployment selector winner 的 FPS 同时不低于 formal-qualified 的 current Tacker 和 two-stream；仅在 incumbent correctness-valid 时应用 FPS paired ratio `>= 1.01` 且 bootstrap 95% CI 下界 `> 1.0`，invalid incumbent 按 fail-closed 替换/回退规则处置。
- [x] 若新候选未达到部署条件，系统保留合格 incumbent；若 incumbent 低于有效 two-stream 下限，则部署 two-stream，不启用更慢的融合。
- [x] 选择/dispatch 的运行时开销被包含在 measured region；正式推理不发生在线编译或搜索。
- [x] p50/p95/max、Raster slowdown、leaf time、occupancy、资源和峰值显存全部报告，但不作为 Raster QoS 否决项。

### 8.3 可复现与运维

- [x] 报告绑定主仓、Raster 子模块、Tacker runtime、CUDA/PyTorch、GPU、workload、ABI 和候选参数哈希。
- [x] Phase 3.1 selection seal 可独立重放；Phase 4 admission 重算并生成字节稳定的 selected manifest/profile canonical hash。
- [x] profile 缺失、过期或不匹配时回退路径与原因清晰可见。
- [x] current v1 profile 在迁移期仍可运行；回滚只需要切回旧 profile 或 `two_stream`。

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 更大的 backend subgroup 拉长 `raster_done`，整体 FPS 反而降低 | 不再凭 leaf savings 推断；所有通过前置门禁的候选先经过短 E2E screening，所有 formal finalist 必须经过完整序列 FPS 排名 |
| 512–896 线程 CTA 降低 occupancy 或触发 register/smem 限制 | ptxas + runtime occupancy 双重过滤；保留 384-thread/不融合候选 |
| 多 head FP16 影响 scale、rotation、opacity 或 SH | 每种输出做 kernel reference，再做 50-view PSNR/SSIM/LPIPS gate |
| barrier 或跨 warp 同步与 Raster 冲突导致死锁 | 子组独立 named barrier、精确 participant count、timeout CUDA tests；禁止 subgroup 内全 CTA barrier |
| 候选组合和编译数量爆炸 | 分层搜索、top-K/beam search、哈希缓存、断点续跑 |
| 只对两个 leaf view 或一次 50 帧运行过拟合 | leaf 只筛错；多 trial、完整视角序列、交错顺序和 bootstrap CI 决策 |
| 单场景赢家在其他 workload 退化 | profile 以 workload key 区分；第二阶段增加不同负载比例的场景，不强制共享赢家 |
| 只改通用 Tacker runtime、实际 Python 路径不生效 | 以 `TackerRenderer` + Raster binding 为主交付，runtime 改动作为语义同步单独验收 |

## 10. 交付物与完成定义

最终交付物包括：

- profile/ABI schema v2 及 v1 兼容读取；
- C0、全部 C1、C2 的 450/450 结构性 screening 矩阵、至少一个可运行的 C3 和 C4，以及对应的安全 fallback；
- C0–C4 生成候选的 terminal status、whole-run FPS autotuner、全量 screening 排行榜、formal finalist 排行榜和可复现的 selection manifest/profile（如有）；
- CPU contract、CUDA 数值、50-view 质量和真实 A6000 性能回归；
- 更新后的实现说明、资格验证命令和回滚说明；
- 一份包含 current Tacker、two-stream、所有生成候选的 screening/terminal status、全部 formal finalist 及 top-3 Nsight 解释的最终报告。

满足以下条件才算完成：Raster 5% QoS 与局部 leaf-sum 已不再影响候选准入；C2 的 2–5 head-set/launch 矩阵已全量 screening；至少一个 C3 packed 和一个 C4 whole-head 候选已通过真实生产路径验证，并与 C0–C2 参与 formal 同口径比较；系统按完整序列 FPS 选择封存 formal finalist 集合的有效赢家，且赢家通过全部正确性门禁。incumbent 有效时，challenger 必须通过稳定晋级门槛，否则明确保留 incumbent；incumbent 无效时，按 fail-closed 规则选择有效候选或基线。Phase 4 的资格完成定义止于显式 canary、不可变 release artifact 和 rollback drill；默认生产 selection/profile 的切换仍是需要单独授权的运维动作。
