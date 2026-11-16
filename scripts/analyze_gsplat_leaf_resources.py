#!/usr/bin/env python3
"""Validate and explain the frozen v11/gsplat NCU leaf comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

from analyze_sampling_kernel_activity import foreign_python_processes

ARMS = ('old_raster', 'gsplat_raster', 'old_mixed_v11', 'gsplat_mixed_ca')
NAMES = ('tacker_render_raster_fast_range_v12', 'rasterize_to_pixels_3dgs_fwd_kernel<4',
         'tacker_mix_render_deform_mlp_tensor_tail_raster_fast_v11', 'gsplat_mix_render_deform_tensor_v7_ca')
METRICS = {
    'theoretical_occupancy_percent': 'sm__maximum_warps_per_active_cycle_pct',
    'achieved_warp_occupancy_percent': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'active_warps_per_sm': 'sm__warps_active.avg.per_cycle_active',
    'issue_active_percent': 'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'eligible_warps_per_scheduler': 'smsp__warps_eligible.avg.per_cycle_active',
    'sm_throughput_percent_elapsed': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'l1tex_throughput_percent_active': 'l1tex__throughput.avg.pct_of_peak_sustained_active',
    'l2_throughput_percent_elapsed': 'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'l1_sector_hit_percent': 'l1tex__t_sector_hit_rate.pct',
    'l2_sector_hit_percent': 'lts__t_sector_hit_rate.pct',
    'local_load_sectors': 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
    'local_store_sectors': 'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
}


def number(value):
    assert value not in ('', None, 'n/a', 'N/A'), value
    return float(value.replace(',', ''))


def byte_value(row, units, key):
    factor = {'byte/block': 1, 'Kbyte/block': 1000, 'Kbyte': 1000, 'byte': 1}[units[key]]
    return round(number(row[key]) * factor)


def audit(directory):
    assert (directory/'completed.txt').exists(), directory
    lines = (directory/'after_check.txt').read_text().splitlines()
    assert lines and all(line.endswith(': OK') for line in lines)
    assert not foreign_python_processes((directory/'processes_before.txt').read_text())
    text = (directory/'during_processes_and_frequency.txt').read_text()
    foreign = foreign_python_processes(text)
    command = '/usr/bin/python3 /data/qyfeng/gsplat-leaf-ncu-20261028/profile_gsplat_leaf_resources.py '
    foreign = sorted({line for line in foreign if not line.split(None, 3)[3].startswith(command)})
    assert not foreign, foreign
    frequencies = [int(x) for x in text.splitlines() if re.fullmatch(r'\d{8,10}', x)]
    assert frequencies and set(frequencies) == {1300500000}, set(frequencies)
    return {'hash_checks_passed': True, 'no_foreign_python_workload_observed': True,
            'frequency_hz': 1300500000, 'frequency_samples': len(frequencies)}


def analyze(root):
    read = lambda p: json.loads(p.read_text())
    validation = read(root/'results/validation/report.json')
    ncu = read(root/'results/ncu/report.json')
    for key in ('workload', 'parent_qualification_sha256', 'binary_sha256',
                'source_and_checkpoint_sha256', 'graph', 'intersections'):
        assert validation[key] == ncu[key], key
    assert validation['completed'] and ncu['completed']
    assert validation['inputs_unchanged'] and ncu['inputs_unchanged']
    assert ncu['graph']['parent'] == 'full_mlp_overlap_graph_pb0' and ncu['graph']['actual_prefix_replays'] == 1
    audits = {phase: audit(root/'results'/phase) for phase in ('validation', 'ncu')}
    with (root/'results/ncu/report.raw.csv').open() as handle:
        reader = csv.DictReader(handle); units = next(reader)
        rows = [row for row in reader if row.get('ID', '').isdigit()]
    assert len(rows) == 4
    result = {}
    for arm, name, row in zip(ARMS, NAMES, rows):
        assert name in row['Kernel Name'], row['Kernel Name']
        data = {'ncu_id': int(row['ID']), 'kernel': row['Kernel Name'],
                **ncu['arms'][arm]['resources'][0]}
        assert ncu['arms'][arm] == validation['arms'][arm]
        assert ncu['arms'][arm]['output_checks_passed']
        assert int(number(row['launch__grid_size'])) == 5440
        assert int(number(row['launch__block_size'])) == data['threads']
        assert int(number(row['launch__registers_per_thread'])) == data['registers']
        for suffix, field in (('static', 'static_shared_bytes'), ('dynamic', 'dynamic_shared_bytes')):
            assert byte_value(row, units, 'launch__shared_mem_per_block_'+suffix) == data[field]
        data['user_shared_bytes'] = data['static_shared_bytes'] + data['dynamic_shared_bytes']
        data['allocated_shared_bytes_including_driver'] = byte_value(row, units, 'launch__shared_mem_per_block_allocated')
        data['driver_shared_bytes'] = byte_value(row, units, 'launch__shared_mem_per_block_driver')
        data['sm_shared_config_bytes'] = byte_value(row, units, 'launch__shared_mem_config_size')
        data['occupancy_limits_ctas_per_sm'] = {k: int(number(row['launch__occupancy_limit_'+k]))
                                               for k in ('blocks', 'registers', 'shared_mem', 'warps')}
        assert min(data['occupancy_limits_ctas_per_sm'].values()) == data['active_ctas_per_sm']
        data['ncu_duration_ms'] = number(row['gpu__time_duration.sum']) * {'ns': 1e-6, 'us': 1e-3, 'ms': 1}[units['gpu__time_duration.sum']]
        data.update({label: number(row[metric]) for label, metric in METRICS.items()})
        data['unprofiled_graph_median_ms'] = validation['unprofiled_median_ms'][arm]
        data['stall_ratios_not_time_percent'] = {
            reason: number(row['smsp__average_warps_issue_stalled_'+reason+'_per_issue_active.ratio'])
            for reason in ('short_scoreboard', 'long_scoreboard', 'mio_throttle', 'barrier', 'math_pipe_throttle')}
        result[arm] = data
    summary = {'completed': True, 'date': '2026-10-28', 'environment': ncu['environment'],
               'workload': ncu['workload'], 'graph': ncu['graph'], 'audit': audits,
               'qualification_sha256': ncu['parent_qualification_sha256'], 'binary_sha256': ncu['binary_sha256'],
               'intersections': ncu['intersections'], 'candidate_pairing': ncu['candidate_pairing'],
               'backend_rgb_difference': ncu['backend_rgb_difference'], 'arms': result,
               'scope': ncu['scope']}
    summary['raw_sha256'] = {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in (
        'results/ncu/report.ncu-rep', 'results/ncu/report.raw.csv', 'results/ncu/report.details.csv',
        'results/ncu/report.json', 'results/validation/report.json')}
    (root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


def report(root, s):
    a = s['arms']; old, new = a['old_mixed_v11'], a['gsplat_mixed_ca']
    names = ('旧 v11 独立 Raster', '原生 gsplat RGBD pixels', '旧 v11 mixed', '最新 gsplat 候选感知 mixed')
    table = '| 核 | threads/CTA | registers/thread | 用户 shared/CTA | local/thread | CTA/SM 上限 | 实际 warp occupancy |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for key, name in zip(ARMS, names):
        d = a[key]
        table += f'| {name} | {d["threads"]} | {d["registers"]} | {d["user_shared_bytes"]:,} B | {d["local_bytes"]} B | {d["active_ctas_per_sm"]} | {d["achieved_warp_occupancy_percent"]:.2f}% |\n'
    limits = '| 核 | Block limit | Registers limit | Shared limit | Warps limit | SM shared 配置 | 含 driver 的实际 shared 分配/CTA |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    performance = '| 核 | 未插桩 Graph 中位耗时 | NCU replay 耗时 | Issue active | Eligible warps/scheduler | L1/TEX throughput（active） |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for key, name in zip(ARMS, names):
        d = a[key]; lim = d['occupancy_limits_ctas_per_sm']
        limits += f'| {name} | {lim["blocks"]} | {lim["registers"]} | {lim["shared_mem"]} | {lim["warps"]} | {d["sm_shared_config_bytes"]:,} B | {d["allocated_shared_bytes_including_driver"]:,} B |\n'
        performance += f'| {name} | {d["unprofiled_graph_median_ms"]:.6f} ms | {d["ncu_duration_ms"]:.6f} ms | {d["issue_active_percent"]:.2f}% | {d["eligible_warps_per_scheduler"]:.3f} | {d["l1tex_throughput_percent_active"]:.2f}% |\n'
    text = f'''# 原 v11 与最新 gsplat leaf：NCU 资源分析（2026-10-28）

**结论：shared memory 减少，但没有增加每 SM 可驻留 CTA。** 独立 Raster 仍为 6 CTA/SM，完整 mixed 仍为 4 CTA/SM；mixed 实际 warp occupancy 仅从 {old['achieved_warp_occupancy_percent']:.2f}% 变为 {new['achieved_warp_occupancy_percent']:.2f}%。新版 mixed 新增的 32 B/thread local memory 在本次输入上确实有访问；L1/TEX 压力也更高。因此提速并非来自更多 CTA 驻留，资源变化也不是全面减少。

## 同输入、同设备与实际 Graph

- Jetson AGX Orin，sm_87、16 SM，GPU min/max 均为 1.3005 GHz。Nsight Compute 2024.3.1；PyTorch {s['environment']['torch']}、CUDA {s['environment']['cuda']}。
- 同一 flame_steak iteration 14000 checkpoint，111,525 Gaussians、1352×1014。Raster 使用 view 0 完全相同的已激活 Gaussian 状态，MLP 使用 view 1 完全相同的 native HexPlane Graph-prefix 特征与同一组打包权重。
- 显式使用 **full_mlp_overlap_graph_pb0** parent qualification，SHA256 `{s['qualification_sha256']}`；两 slot Graph 准备成功，实际核验一次 prefix Graph replay。四个 leaf 均为实际捕获的单 kernel Graph 节点，每个覆盖 5,440 个 Raster tile；没有把 eager full_mlp_overlap 记录作为 Graph 对照。
- 直接加载此前封存的旧 Raster/v11、gsplat 1.5.3 和 `tensor_v7_ca` 二进制，没有重编译或更改生产 kernel。旧独立 Raster 为 v11 相同 fast-Raster device 计算的 `tacker_render_raster_fast_range_v12` 全量范围适配器，不是 v12/v13 的部分 tile prepass。
- NCU 采四个节点，各 16 passes；kernel replay、graph-profiling node、cache-control all、clock-control none。时钟由实验前已固定的上下限约束，连续采样全部为 1.3005 GHz。NCU 的未修改时钟提示与本设置一致。

## 静态资源与 occupancy

{table}

用户 shared 不含 driver 预留，原生 gsplat 的 7,168 B 是 dynamic shared，其余为 static shared。local 属性来自原始 CUDA function attributes，不把它自动解释成全部线程都发生同等寄存器 spill。

{limits}

NCU 显示，两独立 Raster 的寄存器驻留限制均为 6、warp 限制均为 6；两 mixed 的寄存器和 warp 限制均为 4。新版每 CTA 用户 shared 少了 **{100*(1-new['user_shared_bytes']/old['user_shared_bytes']):.2f}%**，driver 选择的 SM shared 配置也从 135,168 B 改为 102,400 B，因此新版 shared limit 是 4，最终仍只有 4 CTA/SM。共享内存减少不能直接推导为 occupancy 增加。

实际 occupancy 是 `sm__warps_active.avg.pct_of_peak_sustained_active`，表示活跃 warp 比例，不是 CTA 驻留数量的直接测量；不能用它除以每 CTA warp 数，推算实际平均驻留 CTA。四个核的理论 warp occupancy 均为 100%。

## 实际访问与调度压力

{performance}

同输入候选交点数为旧后端 **{s['intersections']['old']:,}**、gsplat **{s['intersections']['gsplat']:,}**，减少 **{100*(1-s['intersections']['gsplat']/s['intersections']['old']):.2f}%**。这不是减少 Gaussian 数，而是 Gaussian 与 tile 的交点列表不同。独立 Raster 的实际 occupancy 甚至略低，却仍明显更快；这些结果支持收益主要来自候选工作量和渲染实现变化，而非提高驻留数量。不能将候选数差直接等同于时间差。

最新 mixed 的实际 local load／store sector 数分别为 **{new['local_load_sectors']:,.0f}／{new['local_store_sectors']:,.0f}**，其他三个核均为 0。按 32 B/sector 换算，请求粒度对应约 **{new['local_load_sectors']*32/1048576:.3f}／{new['local_store_sectors']*32/1048576:.3f} MiB**；这是 L1/TEX local 请求量，不是独立 DRAM 流量，也不是峰值显存占用。现有数据证明存在 local 访问，尚不能把全部时差归因于 spill。

旧／新 mixed 的 L1/TEX throughput 为 **{old['l1tex_throughput_percent_active']:.2f}%／{new['l1tex_throughput_percent_active']:.2f}%**；issue active 从 **{old['issue_active_percent']:.2f}%** 降至 **{new['issue_active_percent']:.2f}%**，eligible warps/scheduler 从 **{old['eligible_warps_per_scheduler']:.3f}** 降至 **{new['eligible_warps_per_scheduler']:.3f}**。这提示当前组合更受 L1/TEX/LSU 数据通路和依赖、同步供给影响，而不是可驻留 CTA 数不足。

NCU 的 MIO throttle ratio 为旧 **{old['stall_ratios_not_time_percent']['mio_throttle']:.3f}**、新 **{new['stall_ratios_not_time_percent']['mio_throttle']:.3f}**；long scoreboard ratio 为旧 **{old['stall_ratios_not_time_percent']['long_scoreboard']:.3f}**、新 **{new['stall_ratios_not_time_percent']['long_scoreboard']:.3f}**。这些 ratio 不是耗时百分比，不能相加分账为毫秒；本轮也不据此宣称 shared bank conflict 的具体变化。

## 验证与范围

两阶段的源码、checkpoint、实际加载库指纹一致并通过前后检查；连续进程记录未发现其他 Python 工作负载。四个节点均通过采集前后输出检查：每个 mixed 与各自独立 Raster 的图像逐位一致，五个 MLP 输出与同一封存 device-v7 oracle 逐位一致；跨后端本身仍保留既有 RGB 差异，单帧 MAE/max 为 {s['backend_rgb_difference']['mae']:.9g}/{s['backend_rgb_difference']['max']:.9g}。新候选分配实际开启，1,531 个 CTA 接收额外任务，覆盖全部 6,971 个 MLP tile。

未插桩验证为同进程、单个固定真实帧对、6 轮正逆序交替，预热 5 次、每轮 20 次 Graph replay 的中位数；用于本次诊断校验，不冒充 50 帧端到端 FPS。NCU 数字为隔离 kernel replay，下游并发、缓存状态与完整管线时长不能据此直接外推。没有修改 log.md 或生产 kernel。

## 原始记录与复现

- [结构化汇总](summary.json) · [NCU 报告](results/ncu/report.ncu-rep) · [raw CSV](results/ncu/report.raw.csv) · [details CSV](results/ncu/report.details.csv)
- [未插桩验证](results/validation/report.json) · [NCU 输出验证及源码指纹](results/ncu/report.json) · [实际 NCU 命令](results/ncu/command.txt) · [采集日志](results/ncu/run.log)
- [全部原始记录及启动器归档](raw_results.tar.gz) · [协议](../../protocols/gsplat_leaf_ncu_20261028.json)

远端根目录为 `/data/qyfeng/gsplat-leaf-ncu-20261028`，所有新增文件和缓存位于数据盘；启动器为 `scripts/run_jetson_gsplat_leaf_ncu.sh validation|ncu`（NCU 阶段使用 sudo -n）。本地重建汇总：

```bash
python3 scripts/analyze_gsplat_leaf_resources.py benchmarks/results/gsplat_leaf_ncu_20261028
```
'''
    (root/'analysis.md').write_text(text)


def raster_memory_report(root, baseline):
    directory = root/'results/raster_memory'
    if not (directory/'completed.txt').exists():
        return
    metadata = json.loads((directory/'report.json').read_text())
    original = json.loads((root/'results/ncu/report.json').read_text())
    for key in ('workload', 'binary_sha256', 'source_and_checkpoint_sha256', 'graph', 'intersections'):
        assert metadata[key] == original[key], key
    with (directory/'report.raw.csv').open() as handle:
        reader = csv.DictReader(handle); units = next(reader)
        rows = [row for row in reader if row.get('ID', '').isdigit()]
    assert len(rows) == 2
    fields = {
        'duration_ms': 'gpu__time_duration.sum',
        'lsu_wavefronts': 'l1tex__data_pipe_lsu_wavefronts.sum',
        'shared_load_wavefronts': 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
        'shared_store_wavefronts': 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
        'global_load_requests': 'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
        'global_load_sectors': 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
        'global_store_sectors': 'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum',
        'l1_sector_hit_percent': 'l1tex__t_sector_hit_rate.pct',
    }
    data = {}
    for key, name, row in zip(ARMS[:2], NAMES[:2], rows):
        assert name in row['Kernel Name']
        assert units['gpu__time_duration.sum'] == 'ms'
        data[key] = {label: number(row[metric]) for label, metric in fields.items()}
    old, new = data['old_raster'], data['gsplat_raster']
    ratios = {key: new[key]/old[key] for key in fields if key != 'l1_sector_hit_percent'}
    result = {'completed': True, 'audit': audit(directory), 'arms': data,
              'gsplat_over_old_ratios': ratios,
              'relative_wavefront_service_rate': ratios['lsu_wavefronts']/ratios['duration_ms'],
              'main_collection_l1tex_percent': {key: baseline['arms'][key]['l1tex_throughput_percent_active'] for key in ARMS[:2]},
              'scope': 'same frozen input/binaries as main NCU collection; two Raster nodes, four passes each, cold-cache kernel replay; counters are not DRAM bytes or end-to-end FPS'}
    (root/'raster_memory_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    table = '| 指标 | 旧 v11 独立 Raster | gsplat 独立 Raster |\n| --- | ---: | ---: |\n'
    labels = {'duration_ms': 'NCU 耗时（ms）', 'lsu_wavefronts': 'LSU wavefront 总数',
              'shared_load_wavefronts': 'shared load wavefront', 'shared_store_wavefronts': 'shared store wavefront',
              'global_load_requests': 'global load 请求', 'global_load_sectors': 'global load sector',
              'global_store_sectors': 'global store sector', 'l1_sector_hit_percent': 'L1 sector 命中率（%）'}
    for key, label in labels.items():
        table += f'| {label} | {old[key]:,.6f} | {new[key]:,.6f} |\n'
    text = f'''# 独立 gsplat Raster 的 L1/TEX throughput 为何更高

2026-10-28，在原 NCU 对照的同一封存二进制、checkpoint、view 0 输入和 full_mlp_overlap_graph_pb0 配置下补采两个独立 Raster，均为真实 Graph 单节点。每个核 4 passes，cache-control all、clock-control none；连续审计未见其他 Python 作业，频率均为 1.3005 GHz，输入和库指纹与主采一致。生产 kernel 与 log.md 均未修改。

## 指标口径

主采 L1/TEX throughput 为旧 **38.85%**、gsplat **72.74%**。这个 throughput 是相关子通路相对峰值利用率的最大值，并非总字节量；本次对应主导项为 LSU wavefront 数据通路。它反映单位时间内的服务密度。定义见 [NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-structure)。

## 定向计数

{table}

throughput 百分比来自原 16-pass 主采，表内计数来自独立的 4-pass 定向补采；两次均使用相同冻结输入和二进制，不混合成正式端到端计时。

## 原因一：访存组织不同

旧 v11 的 `general_ptb_render_raster_optimized` 在每批 Gaussian 的协作加载阶段，把 xy、conic/opacity、RGBD 都放入 shared；像素循环从 shared 读取 RGBD。原生 gsplat 只将 ID、xy/opacity、conic 放入 shared，像素通过 alpha/透射率检查后，直接读取 `colors + g * CDIM`。这将颜色复用更多地交给 L1 缓存路径，shared 用量也从 10,240 B 降至 7,168 B。

计数与源码相符：global load 请求从 **{old['global_load_requests']:,.0f}** 增至 **{new['global_load_requests']:,.0f}**，约 **{ratios['global_load_requests']:.2f} 倍**；shared load wavefront 则减少 **{100*(1-ratios['shared_load_wavefronts']):.2f}%**。但 global load sector 只从 **{old['global_load_sectors']:,.0f}** 变为 **{new['global_load_sectors']:,.0f}**，减少 **{100*(1-ratios['global_load_sectors']):.2f}%**；每请求对应的 sector 数下降，说明不能把请求增加直接等同于传输字节增加。

L1 sector 命中率由 **{old['l1_sector_hit_percent']:.2f}%** 升至 **{new['l1_sector_hit_percent']:.2f}%**，更多请求命中缓存。命中率同时受访问模式、工作集、候选列表及 shared/L1 配置变化影响；本轮没有固定 carveout 做独立消融，不将其单独归因于缓存容量或某一行加载代码。两独立 Raster 的 local load/store 均为 0，因此这里不引入 mixed 的 local 访问解释。

## 原因二：核时间下降更快，单位时间服务率更高

累计 LSU wavefront 从 **{old['lsu_wavefronts']:,.0f}** 降至 **{new['lsu_wavefronts']:,.0f}**，减少 **{100*(1-ratios['lsu_wavefronts']):.2f}%**；核耗时则减少 **{100*(1-ratios['duration_ms']):.2f}%**。累计服务工作较少，但被压缩在更短时间内：

```text
相对平均服务率 ≈ wavefront 总数之比 / 时间之比
               = {ratios['lsu_wavefronts']:.4f} / {ratios['duration_ms']:.4f}
               = {result['relative_wavefront_service_rate']:.3f} 倍
```

与主采 throughput 的约 1.87 倍上升一致。wavefront 是硬件服务工作单位，不是字节数；以上推导用于解释相同硬件和固定频率下的利用率变化。

独立 Raster 的 issue active 仍约 86.93%，相比旧核 86.16% 没有下降，且核显著更快。高 L1/TEX throughput 在这里表现为更密集的硬件利用，不能单凭这一项判定性能退化或认定 L1/TEX 已成为唯一瓶颈。完整 mixed 另有 MLP 和 local 访问，需单独分析。

[计数汇总](raster_memory_summary.json) · [原始 NCU](results/raster_memory/report.ncu-rep) · [raw CSV](results/raster_memory/report.raw.csv) · [命令](results/raster_memory/command.txt) · [补采归档](raster_memory_raw.tar.gz) · [主资源报告](analysis.md)
'''
    (root/'raster_memory_analysis.md').write_text(text)
    with (root/'analysis.md').open('a') as handle:
        handle.write('\n[独立 Raster L1/TEX throughput 原因与定向补采](raster_memory_analysis.md)\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    summary = analyze(args.root)
    report(args.root, summary)
    raster_memory_report(args.root, summary)
    print(json.dumps({'audit': summary['audit'], 'arms': summary['arms']}, indent=2))
