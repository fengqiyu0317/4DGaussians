#!/usr/bin/env python3
"""Check the Jetson run and summarize matched Graph timings, counters and pixels."""
import csv
import json
from pathlib import Path
import re
from statistics import mean
import sys

from analyze_gsplat_leaf_resources import byte_value, number
from analyze_sampling_kernel_activity import foreign_python_processes

ARMS = ('on_demand', 'prefetch', 'padded')
KERNELS = ('rasterize_to_pixels_3dgs_fwd_kernel', 'rasterize_rgbd_prefetch_kernel',
           'rasterize_rgbd_padded_kernel')
METRICS = {
    'l1tex_percent_active': 'l1tex__throughput.avg.pct_of_peak_sustained_active',
    'l1_hit_percent': 'l1tex__t_sector_hit_rate.pct',
    'issue_percent_active': 'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'lsu_wavefronts': 'l1tex__data_pipe_lsu_wavefronts.sum',
    'shared_load_wavefronts': 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
    'shared_store_wavefronts': 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
    'global_load_requests': 'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
    'global_load_sectors': 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
}


def audit(path):
    assert (path / 'completed.txt').is_file()
    assert all(line.endswith(': OK') for line in (path / 'after_check.txt').read_text().splitlines())
    assert not foreign_python_processes((path / 'processes_before.txt').read_text())
    samples = (path / 'during_processes_and_frequency.txt').read_text()
    expected = '/usr/bin/python3 /data/qyfeng/gsplat-rgbd-ablation-20261103/profile_gsplat_rgbd_ablation.py '
    foreign = foreign_python_processes(samples)
    assert all(line.split(None, 3)[3].startswith(expected) for line in foreign), foreign
    clocks = re.findall(r'^\d{10}$', samples, re.MULTILINE)
    assert set(clocks) == {'1300500000'}, clocks
    return {'clock_hz': 1300500000, 'samples': len(clocks), 'passed': True}


def main(root, reference):
    validation_dir, ncu_dir = (root / 'results' / name for name in ('validation', 'ncu'))
    validation, ncu, previous = [json.loads(p.read_text()) for p in
        (validation_dir / 'report.json', ncu_dir / 'report.json', reference)]
    for key in ('workload', 'graph', 'arms', 'parent_qualification_sha256',
                'source_and_checkpoint_sha256', 'ablation_sha256'):
        assert validation[key] == ncu[key], key
    for key in ('workload', 'parent_qualification_sha256'):
        assert ncu[key] == previous[key], key
    for path, digest in previous['source_and_checkpoint_sha256'].items():
        if '/iteration_14000/' in path:
            assert ncu['source_and_checkpoint_sha256'][path] == digest, path
    for path, digest in previous['binary_sha256'].items():
        if path.endswith(('gsplat_cuda.so', 'gsplat_mixed_tensor_v7_ca.so')):
            assert ncu['ablation_sha256'][path] == digest, path
    assert ncu['frames'][0]['intersections'] == previous['intersections']['gsplat']
    assert ncu['frames'][0]['differences'] == validation['frames'][0]['differences']
    assert [f['view'] for f in validation['frames']] == list(range(50))
    frames = [f for f in validation['frames'] if 'unprofiled_median_ms' in f]
    assert [f['view'] for f in frames] == [0, 10, 20, 30, 40]
    with (ncu_dir / 'report.raw.csv').open() as handle:
        reader = csv.DictReader(handle)
        units = next(reader)
        rows = [row for row in reader if row.get('ID', '').isdigit()]
    assert len(rows) == len(ARMS)
    summary = {k: ncu[k] for k in ('workload', 'graph', 'parent_qualification_sha256', 'ablation_sha256')}
    summary['audits'] = {p.name: audit(p) for p in (validation_dir, ncu_dir)}
    summary['view0_intersections'] = ncu['frames'][0]['intersections']
    summary['arms'] = {}
    for arm, kernel, row in zip(ARMS, KERNELS, rows):
        assert kernel in row['Kernel Name'], row['Kernel Name']
        data = {**ncu['arms'][arm]['resources']}
        assert int(number(row['launch__grid_size'])) == 5440
        assert int(number(row['launch__block_size'])) == data['threads']
        assert int(number(row['launch__registers_per_thread'])) == data['registers']
        assert byte_value(row, units, 'launch__shared_mem_per_block_dynamic') == data['dynamic_shared_bytes']
        data['sm_shared_config_bytes'] = byte_value(row, units, 'launch__shared_mem_config_size')
        data.update({k: number(row[v]) for k, v in METRICS.items()})
        duration = 'gpu__time_duration.sum'
        data['ncu_duration_ms'] = number(row[duration]) * {'ns': 1e-6, 'us': 1e-3, 'ms': 1}[units[duration]]
        data['graph_ms_by_view'] = {f['view']: f['unprofiled_median_ms'][arm] for f in frames}
        data['graph_ms_mean_of_view_medians'] = mean(data['graph_ms_by_view'].values())
        if arm != 'on_demand':
            diffs = [f['differences'][arm] for f in validation['frames']]
            data['quality_vs_on_demand'] = {channel: {
                'mae': mean(d[channel]['mae'] for d in diffs),
                'max': max(d[channel]['max'] for d in diffs),
                'bitwise_equal_frames': sum(d[channel]['bitwise_equal'] for d in diffs),
            } for channel in ('rgb', 'depth', 'alpha')}
            data['quality_vs_on_demand']['last_ids_equal_frames'] = sum(d['last_ids_equal'] for d in diffs)
        summary['arms'][arm] = data
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
