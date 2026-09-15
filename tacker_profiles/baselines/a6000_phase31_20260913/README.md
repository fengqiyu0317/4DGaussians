# Tacker Phase 3.1 A6000 实机证据

本目录是 Phase 3.1 最终有效 `run-complete-v3` 证据链的**紧凑、哈希绑定镜像**。运行于 2026-09-13 启动，最终报告生成于 2026-09-14 09:18（Asia/Shanghai）。完整 canonical run 保留在远端：

```text
/data/qyfeng/tacker_phase31_validation/20260913-codex-phase31/run-complete-v3
```

Phase 3.1 已完成候选生成、全量短筛、formal qualification、正式交错 E2E 排名、选择和 Top-3 Nsight；**未执行 Phase 4 admission 或部署**。选出的 qualification profile 仍为 `deployment.enabled=false, valid=false`。

## 结论

- 最终矩阵共 661 个候选：C0/C1/C2/C3/C4 分别为 6/30/450/84/91。
- C2 的 H2/H3/H4/H5 分别为 120/180/120/30，覆盖 10/10、10/10、5/5、1/1 个 head set；450/450 均完成短 screening，不代表 450 个候选都做了 formal benchmark。
- 661/661 个候选达到 screening terminal/successful，0 个失败。
- 全局 screening top-5 与 C0–C4 各 family 最佳成功代表去重后得到 9 个 generated finalist，再加入 `serial`、`two_stream`、`current_tacker`，共 12 个 formal entry。
- 9/9 个 challenger 均通过 kernel leaf 和独立 50-view PSNR/SSIM/LPIPS 门禁；其中 5 个 C3 和 1 个 C4 进入 formal，无 backfill。
- 正式协议为同一物理 GPU 1、相同 test 视角 0–49、warmup 10、50 measured frames、每项 10 trials、ABBA、seed 0；120/120 次执行成功。
- 实验 argmax 与部署选择均为 `c3_packed_first_linear_pos_scales_rotations_opacity_shs_wg1_pb5440`，median FPS 为 `100.9417269037555`。
- 相对 `current_tacker` 的 ratio-of-medians 为 `1.1568283703038962`，paired bootstrap 95% CI 为 `[1.1544964933882491, 1.159224626587274]`。
- 相对 `two_stream` 的 ratio-of-medians 为 `1.168739488018986`，paired bootstrap 95% CI 为 `[1.1640678356817016, 1.171534743864701]`。
- 决策为 `promote_challenger / promotion_gates_passed`，但 promotion 只是 Phase 3.1 selector 结论；实际启用仍属于 Phase 4。

## Formal 排名

| Rank | Entry | Median FPS |
|---:|---|---:|
| 1 | `c3_packed_first_linear_pos_scales_rotations_opacity_shs_wg1_pb5440` | 100.9417269038 |
| 2 | `c3_packed_first_linear_pos_scales_rotations_opacity_shs_wg1_pb7000` | 99.8578549363 |
| 3 | `c3_packed_first_linear_pos_scales_rotations_opacity_shs_wg1_pb13942` | 99.3050482317 |
| 4 | `c3_packed_first_linear_pos_scales_opacity_wg1_pb7000` | 93.6198930598 |
| 5 | `c3_packed_first_linear_scales_rotations_opacity_wg1_pb5440` | 93.5544870326 |
| 6 | `c2h5_pos_scales_rotations_opacity_shs_l1_wg1_pb5440` | 89.5695714899 |
| 7 | `c0_legacy_pos_l1_pb5440` | 87.4223044976 |
| 8 | `current_tacker` | 87.2573058329 |
| 9 | `two_stream` | 86.3680297778 |
| 10 | `c1_opacity_l1_wg1_pb13942` | 85.2458969087 |
| 11 | `serial` | 82.6454439817 |
| 12 | `c4_whole_heads_opacity_wg1_pb5440` | 44.6503939710 |

Formal 排名只对封存、完整 qualification 后的 finalist 集合声明 argmax。C3/C4 是按计划分层生成的 top-K 搜索，不是对其理论组合空间的穷举；formal 中的 C4 是 single-head opacity whole-head 代表。

## 正确性、画质和执行完整性

- 9 个 challenger 的 90 个正式 50-frame 序列逐一验证：`input_frames=50`、`full_deformation=1`、`prefix=49`、`mixed_launches=49`、`suffix=49`、`solo_raster=1`、`outputs=50`，每个 selected head 恰好求值 50 次。
- 全部 challenger 的实际模式为 `tacker` 且无 fallback。
- Winner 的 50-view 画质变化：PSNR drop `0.00020996093750369482 dB`、SSIM drop `1.3160705566450659e-6`、LPIPS increase `6.729364395163806e-7`。
- C4 representative 的 50-view 画质变化：PSNR drop `-1.594543456917563e-5 dB`、SSIM drop `-3.576278717609682e-9`、LPIPS increase `-9.14931297391064e-8`。
- C3/C4 raster color/depth max abs 均为 0、radii mismatch 为 0；head 对 quantized FP32 的 max abs 分别为 `4.76837158203125e-6` / `5.7220458984375e-6`，standalone-vs-mixed 为 0。
- 8 个封存 kernel 的 ptxas 检查均为 0 spill；C3/C4 mixed kernel 分别使用 70/36 registers per thread、7168/10016 B static shared memory。

实机测试结果：

| Suite | Result |
|---|---:|
| head extension CUDA | 60/60 passed |
| Raster ABI v1–v4 CUDA | 44/44 passed |
| C3/C4 runtime + fallback | 48/48 passed |
| Phase 3.1 CPU contract | 271/271 passed |
| C3 packed profiler 定向回归 | 42/42 passed（与 CPU contract 有重叠，不另行累加） |

## Top-3 Nsight

Formal top-3 均为 C3 five-head packed，`persistent_blocks` 为 5440/7000/13942。Rank 1 的 Nsight diagnostic 为 `88.0276702939 FPS`、`113.16` launches/frame、`17` stream synchronizations/frame，critical-path proxy 为 `renderer/deformation`，idle/gap proxy 为 `2.2687329 ms/frame`。

第一次 rank-3 qdstrm 导入遇到 Nsight `Wrong event order`，runner 按 fail-closed 退出。因此 `phase31-v3-driver.log` 只记录首轮失败，不代表最终状态。随后在**同一 run identity** 下从 checkpoint 恢复，复用 rank 1/2，仅重跑 rank 3；screening、qualification 和 120 次 formal 执行均未重跑。最终 `phase31-state.json` 记录 `top3-nsight` attempt 2 succeeded，`top3-nsight.json` 记录 `resume.enabled=true`、`recovered_count=2`，三份 CSV 摘要均被 postflight 独立重放。

## 身份与哈希

| 项目 | SHA-256 / 标识 |
|---|---|
| run identity | `0f16b7250611891db52cc59dada0f8bd77c40d735c26f98aed7c81047c14d8a2` |
| validation snapshot: main | `ad71abe0f90d606bc9a5f3955d05cef1ce781e3d` |
| validation snapshot: Raster | `7d8fb35515521c11e75f2a3d20b5764fa2279790` |
| validation snapshot: simple-knn | `44f764299fa305faf6ec5ebd99939e0508331503` |
| loaded head binary | `cf75c80f737169b6c9c96610cbde78ff7b2359691178b08d31e3ddb2c541f0d9` |
| loaded Raster binary | `ef1478ffd323df4d9707044f01af4249d2f940d54b37b01600c354155fce9fa7` |
| head ABI v2 | `9d6a1558acd6b642b975bcabe22abcbe3fd7242e4c9e0d635636ef4d2eb5da7f` |
| Raster ABI v1 | `231c90c429321b2673b88ecd09efb40b6aedda7a23f3e061a2bcedec06d44426` |
| Raster ABI v2 | `310b15957c5920773bb03a61a37c5771f6d4570393061ece4e1805fd20989056` |
| Raster ABI v3 | `c98ed90853308179443146d3022e5da072c4f507975193f7a01f4fbe4400cf40` |
| Raster ABI v4 | `293b8471fc9397070f1d1ebbe1297420f24f49e6882369e2e6cf8dcd9d49b7a1` |
| final matrix, portable | `aa45c9dce881a5a71c9d86130fcd6b12070ee2773b83b44a538f2147bbde8577` |
| screening input | `f93abaa677993311ef65eefe402745597503aa8a9da1867545906f4e58d450a7` |
| screening ranking, portable | `32e64e73703ce5a2e0ea07e1113139e2eec16ce8b3b87810b4d59268b5e770b7` |
| generated candidate set | `3cebae7663eb0ac1d1cb0994ad1de7432c82076458957cb3b35330250679cfc0` |
| formal set | `7c592726cb2edfeb9ea7c030b569d213b179c0917d31a9d1b4a56007abc54102` |
| qualification plan | `48f68059dea3a877d34275abb784cf0779e2a5b180c5977363f408ec02bc6b3d` |
| formal FPS report file | `d56c0385f8e86a3e8aa2927fad449d950c83073025a55f37d1e8e1831b545d98` |
| selection, portable | `1a91a02c2605ae442817bb060b7956ead5d3b940146d274b94a85e00849ec126` |
| winner profile file | `5c3c2031aabe8018d11f41a90e324f375b1a7a14cdfaaef2db730514ec027a51` |
| winner canonical profile | `f91e5f129708ee7612e4a45e95b00e23fe15a92eb1f1be376c2a85095379599e` |
| Top-3 Nsight report | `6ccfe718480aa03e779928403efa9e7c35d21e7ec63f36e6f5cc868ea7ff659e` |
| final run report file | `b24699941748fff134830bf59e0a0001bc64ac419ed119d10481bb894b956ea9` |
| external postflight file | `4949cb3206df127e5c205cbcb1eb47aa32c942da636947680a698b0a424650eb` |
| source compact archive | `06e20059514a325d7a68e5d00700ab58a37e80892d4b02aa8d363dbfd66a2d75` |

这些 main/Raster/simple-knn commit 是远端隔离验证快照标识，不表示当前本地 checkout 已指向这些 commits。本地 Raster 子模块仍是独立、可能 dirty 的版本边界；后续若提交，应先提交 Raster，再更新主仓 submodule pointer。

Winner profile 中的 `manifest.rasterizer_commit=e49506654e8e11ed8a62d22bcb693e943fdecacf` 是沿用模板的静态 runtime compatibility 常量，不是本次实测源码版本。实际验证身份绑定的是 Raster snapshot `7d8fb35515521c11e75f2a3d20b5764fa2279790` 和已加载 binary SHA `ef1478ffd323df4d9707044f01af4249d2f940d54b37b01600c354155fce9fa7`。

## 证据入口

- `postflight-v3.json`：仓库外独立 verifier 的最终结论。
- `run-complete-v3/phase31-report.json`：最终 run 摘要和 sealed identity。
- `run-complete-v3/phase31-state.json`：stage ledger、attempt 与 checkpoint 状态。
- `run-complete-v3/attempts/c4-matrix/0001/matrix.json`：最终 661-candidate matrix。
- `run-complete-v3/attempts/screening-ranking/0001/screening-ranking.json`：全量 screening 排名与 terminal status。
- `run-complete-v3/attempts/formal-candidate-set/0001/formal-candidate-set.json`：formal 集合封存记录。
- `run-complete-v3/sessions/formal/formal-fps.json`：120 次 formal 执行、排名、paired CI 与 promotion decision。
- `run-complete-v3/attempts/formal-benchmark/0001/execution-integrity.json`：90 个 challenger sequence 的逐项计数验证。
- `run-complete-v3/attempts/selection/0001/selection.json`：稳定 selection manifest。
- `run-complete-v3/attempts/selection/0001/winner-qualification-profile.json`：仍 disabled/invalid 的 winner qualification profile。
- `run-complete-v3/sessions/nsight/top3-nsight.json`：Top-3 诊断与 checkpoint-resume 记录。
- `artifact-manifest.json`：本地镜像中除 manifest 自身外每个文件的相对路径、大小、SHA-256 和原始远端路径。

## 紧凑镜像边界

本目录不是完整、自包含、可离线重放的 run archive。`postflight-v3.json` 是在完整远端树上生成的，它验证了 49 个成功 stage 和 7354/7354 个唯一 ledger artifact；紧凑镜像仅保留其中 42 个 ledger artifact，以及日志、runner 和 verifier 等共 59 个原始文件。README 和 manifest 是下载后的本地索引元数据。

为控制仓库体积，镜像省略了：

- 大量 screening child artifacts；
- 660 个非 winner final profile；
- 7/9 个 finalist 的独立 qualification 详证，仅保留 C3 winner 与 C4 representative；
- 120 组 formal 原始 metadata/stdout/stderr；
- Nsight `.nsys-rep`、SQLite 和其他大体积原始文件；
- 编译二进制、模型、数据集、Python 环境和 NVML wrapper。

JSON 中的远端绝对路径为保持原始字节与已封存哈希而没有重写。因此，仅凭本地镜像不能直接运行 verifier；重放需要原始远端目录、同一源码快照、workload 和 Python/CUDA 环境。原始 compact tarball 大小为 1,266,947 bytes，其 SHA-256 见上表。

`ptxas-evidence.json` 中引用的 Phase 2 build log 仅用于未变化二进制的编译资源溯源。最终矩阵、screening、qualification、formal、selection 和 Nsight 结论全部来自 v3；旧 `run-complete-v2` 的矩阵、排名、qualification 和选择哈希没有混入本目录。

## 远端复现命令

以下脚本记录了原始绝对路径和环境，需在原始 A6000 验证主机上运行：

```bash
bash /data/qyfeng/tacker_phase31_validation/20260913-codex-phase31/run-phase31-v3.sh
bash /data/qyfeng/tacker_phase31_validation/20260913-codex-phase31/run-postflight-v3.sh
```

若某 stage fail closed，可对第一条命令追加 `--resume`；恢复必须保持相同 identity。postflight 输出采用 create-once 语义，重新验证前需选择新的输出路径，不能覆盖原始 `postflight-v3.json`。

## Phase 4 仍待完成

- 真实 1/2/50 帧 prefill/drain 顺序和更长 steady-state；
- 完整 p50/p95/max、Raster slowdown、leaf、occupancy、资源和峰值显存报告；
- 独立 `../Tacker` runtime commit 与完整 TOCTOU provenance；
- 真实 A6000 上 profile missing/stale/mismatch fallback smoke；
- 额外 workload 泛化、显式 profile 灰度、正式 admission、发布与回滚演练。

本轮 formal selection metadata 未提供完整 peak-memory/register/shared-memory tie-break 字段。此次 0.5% equivalence set 只有 rank-1 winner 一项，所以不影响本次选择；不能据此声称通用资源 tie-break 已完成实机覆盖。
