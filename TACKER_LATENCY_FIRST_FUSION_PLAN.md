# Tacker 去 QoS、FPS 优先的融合计划

- 状态：Phase 0–2 complete；Phase 3–4 proposed
- 首期目标平台：NVIDIA RTX A6000（`sm_86`）
- 首期目标负载：`flame_steak`，iteration 14000，111,525 Gaussians，1352 × 1014，test 前 50 个视角

## 1. 决策摘要

本次改造不再保护 Raster 子任务的独立 QoS。`raster_slowdown_pct <= 5%` 和 `mixed_p50 < solo_raster + solo_head` 从准入硬门禁降为诊断数据，融合方案只以完整推理渲染序列的端到端 FPS 排序和晋级，FPS 越高越好。

但仅删除 `5.0` 阈值并不能实现自由选择。当前 4DGaussians 推理路径没有候选选择器，而是先把唯一候选写死为 `render leaf + pos_deform[1] Linear(128,128)`，再用 QoS 判断整条 Tacker 路径是否准入。因此需要同时完成：

1. 把单一 pair/profile 升级为多候选 contract；
2. 把固定 `pos_deform[1]` 的 prefix/mixed/suffix 拆分泛化为候选驱动的任务分区；
3. 提供不同融合边界、CTA 配比和 `persistent_blocks` 的 CUDA 变体；
4. 离线测量所有有效候选的完整序列 FPS 并选出赢家；
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

正式排名使用多次完整序列运行的 `median(throughput_fps)`，按降序选择，不能使用相邻输出完成间隔的 p50 代替。所有候选必须使用相同帧数、视角顺序和计时边界。

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

1. 所有 correctness-valid 候选按 `median(throughput_fps)` 降序排列；
2. 实验赢家是全局 argmax；
3. 部署赢家的 FPS 还必须同时不低于 two-stream 和当前已部署 Tacker；
4. 为避免噪声导致频繁切换，只有当 `candidate_fps/incumbent_fps >= 1.01`，且配对 bootstrap 95% 置信区间下界大于 1.0 时才替换 incumbent；否则保留 incumbent，并仍在报告中记录实验 argmax；
5. 若多个候选落在 0.5% 的等价区间，选择 ABI 更简单、峰值显存更低、寄存器/共享内存更少的候选。

这里的 1% 是部署稳定性门槛，不是新的 Raster QoS；优化搜索本身仍报告真正的最高 FPS。

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
- correctness 结果、完整序列各 trial、汇总统计和环境 provenance；
- Raster/leaf/Nsight 诊断，不含 Raster QoS 通过/失败字段；
- `selection_objective: "median_throughput_fps"` 和被选原因。

schema v1 作为 `legacy_pos_l1` 兼容读取；只有 schema v2 能承载多候选并由新 selector 生成。模板仍保持 disabled，调优工具写新的 hashed winner profile，禁止原地启用模板。

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

为控制组合爆炸，C1 全量测量；C2 先测所有双 head，再采用 beam search 扩展 top-K 到 3–5 head；C3/C4 只基于前一层的 top-K 组合生成。leaf 数据只用于早期剪枝和解释，最终淘汰必须由完整 E2E 结果决定。

### 5.4 离线调优与运行时 dispatch

新增离线 tuner，流程固定为：

1. 枚举候选及 launch 参数；
2. 构建扩展并收集 symbol、ptxas、寄存器、shared memory 和 occupancy；
3. 运行 kernel-level 数值检查，失败候选立即隔离；
4. 执行短 E2E screening，保留 top-K；
5. 对 top-K 和三种基线执行严格的交错多 trial 完整序列测量；
6. 对最终 top-3 做 Nsight 分析；
7. 运行 50-view 画质验证；
8. 选择赢家并原子写入 hashed profile、完整报告和候选排行榜。

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
- **CUDA correctness**：重编后的 head 完整套件 60/60、Raster v2/legacy 完整套件 34/34 通过，其中 CUDA 定向测试分别为 16/16 与 12/12；覆盖真实 binding、尾块/非对齐输入、错误 contract、barrier 以及 legacy 路径。
- **编译资源**：ptxas 报告 mixed v2 为 68 registers/thread、7376 B static shared memory、0 spill；whole-head 为 32 registers/thread、512 B static shared memory、0 spill；packed 为 48 registers/thread、0 spill；multi GPTB/solo 为 48 registers/thread、200 B stack、0 spill。runtime query 对 1–5 个 worker group 分别返回 384/512/640/768/896 threads，active blocks/SM 为 2/1/1/1/1，occupancy 为 0.5/0.333333/0.416667/0.5/0.583333，且全部 `launch_supported=true`、kernel max threads 为 896。
- **50-view 物理路径**：7 份 disabled qualification profile（C0、全部五个 C1、一个 `pos+scales` C2）各以 warmup 10、50 measured views、111,525 Gaussians、1352 × 1014 跑通；最终 metadata 均通过已加载 binary manifest SHA 与实时资源门控，且为 `actual_execution_mode=tacker`、`tacker_fallback_reason=null`。最终单 trial diagnostic FPS 依次为 C0 87.983156、C1 opacity 85.842899、C1 pos 85.779940、C1 rotations 85.229702、C1 scales 84.982640、C1 shs 85.338641、C2 pos+scales WG2 66.852933。
- **无重复执行证据**：每个候选的 50-view 复跑均记录 `input_frames=50`、`full_deformation=1`、`prefix=49`、`mixed_launches=49`、`suffix=49`、`solo_raster=1`、`outputs=50`、`selected_head_evaluations_per_head=50`，并在最后一个输出前由运行时不变量 fail closed 校验。结合 selected Linear 的 Python 跳过/重复调用测试，Phase 2 的无重复计算、无 fallback 退出条件已满足。
- **证据边界**：上述 FPS 只证明候选能在真实流水执行，不是 Phase 3 的交错多 trial 排名，也没有选出 winner；完整 Phase 4 preflight、50-view 画质门槛和正式 admission 仍未声称完成。定向验证摘要保存在 `tacker_profiles/baselines/a6000_phase2_20260910/`。

### Phase 3：实现 E2E autotuner 与候选选择

- [ ] 新增候选矩阵生成和 profile DB 工具，支持 top-K/beam search、参数去重与断点续跑。
- [ ] 泛化 `profile_tacker_leaves.py`，按 variant 输出数值、leaf、资源和 workload provenance。
- [ ] 对 correctness-valid 候选执行完整 E2E 交错 benchmark，并生成稳定排行榜。
- [ ] 自动选择实验 argmax 和部署 winner；写出选择理由、置信区间、与 current/two-stream 的 FPS paired ratio。
- [ ] 对 top-3 运行 Nsight，确认关键路径变化、空转、straggler、同步和 kernel launch 数。

退出条件：给定同一候选集合和测量输入，selector 结果确定且 profile hash 可复现；正式运行没有在线搜索开销。

### Phase 4：准入、回归与发布

- [ ] 将 source/config/已加载二进制 provenance 前移到 import/parse 之前，并增加前后字节不变校验，关闭并发原地改写的 TOCTOU 窗口。
- [ ] 更新 `scripts/run_tacker_qualification.sh`，串起构建、资源检查、数值、候选 E2E、画质、选择、常规 profile 复跑和 fallback smoke test。
- [ ] 更新 CPU contract、CUDA 数值测试和真实 GPU 性能回归。
- [ ] 在 1、2、50 帧序列验证 prefill/drain，在更长循环序列验证 steady state。
- [ ] 对至少两个额外 Raster/deformation 比例不同的 workload 只做泛化评估；首期仍按 workload-key 选择各自赢家，不强求一个全局 variant。
- [ ] 更新 `TACKER_IMPLEMENTATION.md` 和 `tacker_profiles/README.md`，说明 QoS-free selection、profile v2、复现命令和回滚流程。
- [ ] 先以显式 profile 灰度运行；验证稳定后再替换当前 admitted profile。

退出条件：满足第 8 节全部验收标准，并能够只替换 profile 回滚到 current Tacker 或 two-stream。

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
| `tacker_profiles/*.json`、`tacker_profiles/README.md` | schema v2 disabled template、winner profile 和复现说明 |
| `tacker_ext/include`、`tacker_ext/csrc`、`tacker_ext/abi` | multi-head/packed/whole-head GPTB adapter 与 ABI |
| Raster 子模块 `cuda_rasterizer/tacker_mixed.*`、binding、ABI | 多 mixed wrapper、资源/capability 描述、variant dispatch |
| `tests/test_tacker_*.py`、`tacker_ext/tests`、Raster CUDA tests | 新 schema、选择器、依赖、数值、fallback 和 GPU 回归 |
| `../Tacker/src/runtime/{Scheduler,Registry}.*` | QoS-free 选项、多 mixed registration、E2E/FPS score、资源检查；作为第二阶段语义同步 |

Raster 子模块和 `../Tacker` 是独立版本边界，实施时应分别提交并在主仓 profile provenance 中锁定精确 commit。

## 8. 验收标准

### 8.1 功能与正确性

- [x] profile 能同时描述多个 correctness-valid candidate，并唯一记录 `selected_variant_id`。
- [x] 一个 Raster slowdown 明显高于 5% 的合成或真实候选不会因 Raster QoS 被拒绝。
- [ ] 所有被选节点只执行一次；1/2/N 帧的输出顺序和 legacy renderer 一致。
- [ ] 事件、stream 和 slot 复用测试覆盖 prefix、parallel、mixed、suffix、drain 和异常回退。
- [ ] 任意 ABI、shape、dtype、alignment、resource、manifest 或 launch 错误都 fail closed。
- [ ] kernel 数值阈值保持：color/depth max abs `1e-5`、radii mismatch 为 0、head 输出 atol/rtol `2e-3`，或为新输出定义不弱于等价 FP32/legacy 参考的明确阈值。
- [ ] 50-view 画质阈值保持：PSNR drop `<= 0.05 dB`、SSIM drop `<= 1e-4`、LPIPS increase `<= 1e-4`。

### 8.2 性能

- [ ] 同一 A6000、同一 50-view 顺序、warmup=10；每个 baseline/candidate 至少 10 次完整序列 trial，并交错执行。
- [x] 正式 winner 是所有 correctness-valid 方案中 `median(throughput_fps)` 最高者。
- [x] 部署的 Tacker winner FPS 同时不低于 current Tacker 和 two-stream；替换 incumbent 时 FPS paired ratio `>= 1.01` 且 bootstrap 95% CI 下界 `> 1.0`。
- [x] 若新候选未达到部署条件，系统保留合格 incumbent；若 incumbent 低于有效 two-stream 下限，则部署 two-stream，不启用更慢的融合。
- [x] 选择/dispatch 的运行时开销被包含在 measured region；正式推理不发生在线编译或搜索。
- [ ] p50/p95/max、Raster slowdown、leaf time、occupancy、资源和峰值显存全部报告，但不作为 Raster QoS 否决项。

### 8.3 可复现与运维

- [ ] 报告绑定主仓、Raster 子模块、Tacker runtime、CUDA/PyTorch、GPU、workload、ABI 和候选参数哈希。
- [x] 同一报告可重新生成字节稳定的 selected manifest hash。
- [ ] profile 缺失、过期或不匹配时回退路径与原因清晰可见。
- [x] current v1 profile 在迁移期仍可运行；回滚只需要切回旧 profile 或 `two_stream`。

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 更大的 backend subgroup 拉长 `raster_done`，整体 FPS 反而降低 | 不再凭 leaf savings 推断；所有候选必须经过完整序列 FPS 排名 |
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
- 至少 C0、全部 C1、一个 C2 和对应的安全 fallback；
- whole-run FPS autotuner、完整候选排行榜和可复现 winner profile；
- CPU contract、CUDA 数值、50-view 质量和真实 A6000 性能回归；
- 更新后的实现说明、资格验证命令和回滚说明；
- 一份包含 current Tacker、two-stream、所有有效候选及 top-3 Nsight 解释的最终报告。

满足以下条件才算完成：Raster 5% QoS 与局部 leaf-sum 已不再影响候选准入；多个融合边界确实可运行并参与同口径比较；系统按完整序列 FPS 选择全局有效赢家；赢家通过全部正确性门禁且稳定优于 incumbent，或者在没有可靠收益时明确保留 incumbent。
