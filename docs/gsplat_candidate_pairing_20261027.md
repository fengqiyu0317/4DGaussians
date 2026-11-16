# gsplat mixed：迁入 v11 候选感知配对（2026-10-27）

**迁移已完成。** 候选感知版本相对同 MLP 独立调用，leaf 耗时降低 **6.23%**，端到端 FPS 变化 **+3.68%**；相对当前 3,486 CTA mixed，leaf 耗时变化 **+1.07%**、端到端 FPS 变化 **-0.87%**。未超过当前 mixed，保留为显式实验选项。

## 实现范围

保留 gsplat 1.5.3 的 projection／SH／tile 排序与像素合成，以及此前迁入的旧 v11/v13 device-v7 Tensor Core tail MLP。本轮新增 `tensor_v7_ca`，移植旧 `tackerCandidateAwareExtraDeformRank` 的任务分配：每个 Raster CTA 先做一个 MLP tile，再将额外任务分成连续组，每组由候选 Gaussian 数最多的 Raster CTA 接收第二个 MLP tile；候选数并列时取首项。

本场景为 5,440 个 Raster tile、6,971 个 MLP tile、1,531 个额外任务，每组 3–4 个 CTA。GPU 直接读取 gsplat offsets 计算候选数，无全局排序或额外预处理 launch。physical grid 不等于 Raster tile 数，或 `D<=R`／`D>=2R` 时回退原 grid-stride，保证完整覆盖。MLP 数学、权重精度、参数打包、激活和 residual 规则均与当前版相同。HexPlane 仍使用原生双 slot CUDA Graph；本轮没有引入 v13 Raster prepass 或 HexPlane fused reduction。

## 固定条件与口径

- Jetson AGX Orin，16 SM，GPU 固定 1.3005 GHz；PyTorch 2.8.0、CUDA 12.6。
- flame_steak iteration 14000，111,525 Gaussians，1352×1014，test views 0–49。四个 checkpoint 文件及 parent qualification 的 SHA256 与此前 v7 实验一致。
- Parent qualification 为 **`full_mlp_overlap_graph_pb0`**（`full_mlp_overlap_graph`），SHA256 `a13d87f6c4a22a24b20f83cb3c66a009e759b739b498c9ff55f8c35c9b82d23a`；它约束工作负载与 prefix，新增 leaf 单独验证。实际每组 50 帧执行 49 次 prefix Graph replay，质量阶段的两 slot 真实 replay 计数为 24／25，无 eager fallback。
- leaf：固定帧对 `0→1 / 24→25 / 48→49`，6 轮正逆序交替，每轮预热 5 次、测 20 次 Graph replay；先取各帧对轮间中位数，再平均三组。仅含输入就绪后的像素核＋下一帧 MLP，排除 prefix、prepare、分配、编译和预热。独立组为两个核，Both 为一个核。
- 端到端：同一进程内五组正逆序交替，50 帧、10 轮中位数，计正常 Python 推理、prefill／drain、context、prefix、MLP、激活、prepare、RGBD 与同步，排除加载／JIT／预热／指标／I/O。每轮核验 50 outputs、49 prefix、49 mixed 或 separate pairs、1 solo Raster、无 fallback。

## 最终同批次结果

| 实现 | leaf／Both（ms） | 端到端 ms/帧 | FPS |
| --- | ---: | ---: | ---: |
| 同 MLP 独立调用 | 11.910000 | 19.262535 | 51.9142 |
| 当前 mixed，3,486 CTA | 11.049938 | 18.416171 | 54.3001 |
| 旧 mixed，5,440 CTA（grid 控制） | 11.051075 | 18.414003 | 54.3065 |
| 新 kernel 关闭配对，5,440 CTA | 11.233630 | 18.631490 | 53.6726 |
| 候选感知 mixed，5,440 CTA | 11.168033 | 18.578372 | 53.8260 |

两侧 standalone：**Raster 5.313153 ms**（原生 gsplat 256-thread RGBD pixels）、**MLP 6.598730 ms**（device-v7 128-thread）。二者之和 **11.911883 ms**；融合收益以实测独立串行 **11.910000 ms** 为基准。

同一新二进制开启／关闭候选配对时，leaf 耗时降低 **0.584%**，端到端 FPS 变化 **+0.286%**。但同 grid 的新二进制关闭配对相对旧二进制，leaf 反而增加 **1.652%**；旧 kernel 只将 3,486 CTA 改为 5,440 CTA 的 leaf 变化为 **+0.010%**。因此本轮主要问题是新增实现的总成本超过配对本身的局部收益，grid 数量变化影响很小。配对开／关的 10 轮 FPS 变化范围为 **-0.166%～+0.335%**，其微小收益不作为跨场景稳定加速结论。

新候选相对当前版的 10 轮配对 FPS 变化范围为 **-1.265%～-0.824%**。统计只覆盖本次 Orin、场景与 checkpoint，未据此推广到其他 GPU 或场景。

## 渲染耗时控制

| 关闭 MLP 的渲染控制 | prepare＋RGBD pixels＋layout（ms/帧） |
| --- | ---: |
| native | 8.666965 |
| current_3486 | 9.066764 |
| ca_off_5440 | 9.003190 |
| ca_on_5440 | 9.001877 |


共享相同的 50 帧预缓存、已激活 Gaussian 状态，不计 MLP／HexPlane。mixed 控制通过 mode 0 关闭 MLP，保留对应 kernel 的静态资源分配；这些是 prepare＋像素阶段的诊断控制，**不是 Both 中 Raster 子组的独占耗时**。

## 画质：平均差与最大差

| 实现 | 平均 PSNR ↑ | 平均 SSIM ↑ | 平均 LPIPS ↓ |
| --- | ---: | ---: | ---: |
| 原生 gsplat＋PyTorch MLP | 32.570319595 | 0.9574662971 | 0.1414458913 |
| 独立 device-v7 MLP | 32.570165558 | 0.9574654412 | 0.1414484480 |
| 当前 mixed | 32.570165558 | 0.9574654412 | 0.1414484480 |
| 候选感知 mixed | 32.570165558 | 0.9574654412 | 0.1414484480 |

各组使用相同 GT，RGB clamp 到 `[0,1]` 后算 PSNR／SSIM／LPIPS-VGG 0.1。当前 mixed、grid 控制、新 kernel 配对开／关与独立组的 50 帧原始 RGB、累计深度全部逐位相同，因此迁移新增的平均差与最大差均为 0。

| 指标 | 相对独立组平均绝对差 | 最大单帧绝对差 | 相对原生 GS/PyTorch 平均绝对差 | 最大单帧绝对差（view） |
| --- | ---: | ---: | ---: | ---: |
| psnr_db | 0 | 0 | 0.00044303894 | 0.00303649902（24） |
| ssim | 0 | 0 | 2.6011467e-06 | 2.10404396e-05（24） |
| lpips | 0 | 0 | 6.95079565e-06 | 6.88582659e-05（49） |


上表差值逐帧取绝对值，再统计平均和单帧最大，不能用“整体均值相减的绝对值”替代平均绝对差。

| 输出／参照 | 平均绝对差 MAE | 全局最大绝对差（view） |
| --- | ---: | ---: |
| rgb／independent | 0 | 0（0） |
| depth／independent | 0 | 0（0） |
| rgb／native | 7.609515451e-06 | 0.09354905784（45） |
| depth／native | 0.0001203666582 | 1.779582024（5） |


像素 MAE 是全部帧／像素／通道的平均绝对差；全局最大取全部帧／像素／通道的最大值，不是逐帧最大值的平均。深度为累计深度输出差，不是 GT 几何深度误差。相对原生 gsplat/PyTorch 的差异来自已存在的优化 MLP 管线，候选配对没有增加差异；本轮没有重新比较旧 diff-gaussian 后端，也没有消除此前换后端的画质差。

## 配对、资源与优化尝试

| Raster→MLP 帧 | 普通映射：双 tile CTA 平均候选数 | 候选感知：双 tile CTA 平均候选数 | 额外任务数 |
| --- | ---: | ---: | ---: |
| 0→1 | 226.007 | 284.784 | 1531 |
| 24→25 | 226.043 | 284.502 | 1531 |
| 48→49 | 226.201 | 284.827 | 1531 |


运行时记录每个 CTA 的任务数、选中 rank 和候选数，与独立 CPU 分组参考逐项核对，重建后的 6,971 个 MLP tile 均恰好覆盖一次，三组均选择 1,531 个额外任务接收者。全部五个 MLP 输出对封存旧 v11 matched library 逐位一致，像素对原生 gsplat 逐位一致。clock 窗口确认两子组重叠；窗口重叠比例不作为有效加速比。

| kernel | threads/CTA | registers/thread | shared/CTA（B） | local/thread（B） | CTA/SM |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_3486 | 384 | 40 | 22120 | 0 | 4 |
| ca_on_5440 | 384 | 40 | 22124 | 32 | 4 |


保持 40 registers、4 CTA/SM，但新增配对实现的 local memory 属性为 32 B/thread，旧版为 0；同二进制开关对照衡量候选分配的收益，旧／新关闭配对对照反映实现总成本。仅凭静态 local 属性不将全部差额归因为 spill，未新增 NCU 指令级归因。

迁移后尝试两种单调用点改写，未取得更低 leaf 耗时，最终保留直接 base／extra 调用：

| 写法（各自独立批次） | 候选开启 leaf（ms） | local/thread（B） |
| --- | ---: | ---: |
| 直接迁移首轮 | 11.170016 | 32 |
| 统一外层循环 | 11.273621 | 136 |
| v7 内层自定义步长 | 11.218097 | 32 |

上述为实现筛选，不与最终同批次结果混算。首轮完整 formal 的连续审计发现外部 `seal_v1_j50_wait_comparison.py`，整批性能不纳入主结论；两次改写的 formal 主动中断，不纳入性能结论。原始记录与中断说明均保留。最终实现指纹及五组测试在正式复测前固定，最终使用全部 10 轮，不按 FPS 挑选样本。

## 验证、使用与记录

- 本地 suite 共 83 项：78 项通过，5 项 CUDA 测试在 Jetson 补测通过，覆盖配对分支、并列／空候选、partial tile、空行、fallback grids、同二进制开关与 Graph replay。
- 最终 CUDA／leaf／formal 批次的启动前与连续进程、频率审计通过，采样频率均为 1.3005 GHz，未观察到其他 Python 工作负载。源码／checkpoint 前后指纹一致，leaf／formal 快照一致；四个 checkpoint 文件和 parent qualification 与之前 v7 实验一致。公共 `tacker_pipeline.py`／`hexplane.py` 使用前次隔离 runtime 的版本，以归档快照为准。
- 默认后端和默认 kernel 未切换；显式使用 `--raster_backend gsplat --execution-mode gsplat_mixed --gsplat_mixed_kernel tensor_v7_ca --gsplat_mixed_blocks 5440`，并传匹配的 `--tacker-profile`。`--execution-mode gsplat_matched_graph --gsplat_mixed_kernel tensor_v7` 为同 MLP 独立对照。配对关闭控制通过 `GsplatMixedRenderer(..., candidate_pairing=False)` 使用。

完整复现入口为 `scripts/run_jetson_gsplat_candidate.sh build|cuda|leaf|formal`，远端根目录 `/data/qyfeng/gsplat-ca-20261027`，全部新增文件与缓存位于数据盘。离线汇总：

```bash
python3 scripts/analyze_gsplat_candidate.py benchmarks/results/gsplat_candidate_20261027 --previous-report benchmarks/results/gsplat_v7_20261027/results_20261027/formal/report.json
```

- [固定协议与实现选择](../benchmarks/protocols/gsplat_candidate_pairing_20261027.json)
- [最终汇总与审计](../benchmarks/results/gsplat_candidate_20261027/summary.json) · [leaf 原始记录](../benchmarks/results/gsplat_candidate_20261027/results/leaf/report.json) · [50 帧正式记录](../benchmarks/results/gsplat_candidate_20261027/results/formal/report.json)
- [CUDA 测试日志](../benchmarks/results/gsplat_candidate_20261027/cuda_suite.log) · [正式运行日志](../benchmarks/results/gsplat_candidate_20261027/formal_suite.log) · [原始数据与源码快照](../benchmarks/results/gsplat_candidate_20261027/raw_results.tar.gz)
- [首轮及外部作业审计](../benchmarks/results/gsplat_candidate_20261027/initial_port/audit_rejection.json) · [首轮归档](../benchmarks/results/gsplat_candidate_20261027/initial_port.tar.gz) · [外层循环尝试](../benchmarks/results/gsplat_candidate_20261027/outer_loop_attempt.tar.gz) · [步长尝试](../benchmarks/results/gsplat_candidate_20261027/single_stride_attempt.tar.gz)
