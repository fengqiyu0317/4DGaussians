#!/usr/bin/env python3
"""Validate candidate identity and summarize the wide-load comparison."""
import csv
import json
from pathlib import Path
import re
from statistics import mean
import sys

from analyze_gsplat_leaf_resources import byte_value, number
from analyze_gsplat_rgbd_ablation import METRICS
from analyze_sampling_kernel_activity import foreign_python_processes

LISTS = ('legacy_tiles', 'gsplat_tiles')
LAYOUTS = ('prefetch', 'wide')
ARMS = tuple((candidates, layout) for candidates in LISTS for layout in LAYOUTS)


def audit(path):
    assert (path / 'completed.txt').is_file()
    hashes = (path / 'after_check.txt').read_text().splitlines()
    assert hashes and all(line.endswith(': OK') for line in hashes)
    assert not foreign_python_processes((path / 'processes_before.txt').read_text())
    samples = (path / 'during_processes_and_frequency.txt').read_text()
    expected = '/usr/bin/python3 /data/qyfeng/gsplat-wide-load-20261103/profile_gsplat_wide_load_ablation.py '
    foreign = foreign_python_processes(samples)
    assert all(line.split(None, 3)[3].startswith(expected) for line in foreign), foreign
    clocks = re.findall(r'^\d{10}$', samples, re.MULTILINE)
    assert set(clocks) == {'1300500000'}, clocks
    return {'clock_hz': 1300500000, 'samples': len(clocks), 'passed': True}


def sass_loads(path):
    kernels = []
    with path.open() as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if row[0] == 'Kernel Name':
                kernels.append([])
            elif row[0] == 'Address':
                columns = {key: index for index, key in enumerate(row)}
            elif row[0].startswith('0x') and re.search(r'\bLDS(?:\.\d+)?\s', row[1]):
                kernels[-1].append({
                    'instruction': row[1].strip(),
                    'executed': int(row[columns['Instructions Executed']]),
                    'wavefronts': int(row[columns['L1 Wavefronts Shared']]),
                    'excessive': int(row[columns['L1 Wavefronts Shared Excessive']]),
                })
    return kernels


def main(root, reference_root):
    validation_dir, ncu_dir = (root / 'results' / name for name in ('validation', 'ncu'))
    validation, ncu, previous = [json.loads(p.read_text()) for p in
        (validation_dir / 'report.json', ncu_dir / 'report.json',
         reference_root / 'results/validation/report.json')]
    assert validation['completed'] and ncu['completed']
    for key in ('workload', 'graph', 'arms', 'parent_qualification_sha256',
                'source_and_checkpoint_sha256', 'ablation_sha256'):
        assert validation[key] == ncu[key], key
    for key in ('workload', 'graph', 'parent_qualification_sha256', 'source_and_checkpoint_sha256'):
        assert ncu[key] == previous[key], key
    for path, digest in previous['ablation_sha256'].items():
        if path.endswith(('.so', 'project.cu')):
            assert ncu['ablation_sha256'][path] == digest, path
    for frame, ref in zip(validation['frames'], previous['frames'], strict=True):
        for key in ('view', 'intersections', 'supplemented_gaussians'):
            assert frame[key] == ref[key], key
    first = validation['frames'][0]
    for key, value in ncu['frames'][0].items():
        assert first[key] == value, key
    assert first['membership_hashes'] == previous['frames'][0]['membership_hashes']
    assert [f['view'] for f in validation['frames']] == list(range(50))
    frames = [f for f in validation['frames'] if 'unprofiled_median_ms' in f]
    assert [f['view'] for f in frames] == [0, 10, 20, 30, 40]
    with (ncu_dir / 'report.raw.csv').open() as handle:
        reader = csv.DictReader(handle)
        units = next(reader)
        rows = [row for row in reader if row.get('ID', '').isdigit()]
    loads = sass_loads(ncu_dir / 'report.sass.csv')
    assert len(rows) == len(loads) == len(ARMS)
    summary = {k: ncu[k] for k in ('workload', 'graph', 'parent_qualification_sha256', 'ablation_sha256')}
    summary['audits'] = {p.name: audit(p) for p in (validation_dir, ncu_dir)}
    summary['view0_intersections'] = first['intersections']
    summary['view0_membership_hashes'] = first['membership_hashes']
    summary['arms'] = {}
    extra_metrics = {
        'shared_load_instructions': 'smsp__sass_inst_executed_op_shared_ld.sum',
        'shared_load_bank_conflicts': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
        'shared_store_bank_conflicts': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    }
    for (candidates, layout), row, sass in zip(ARMS, rows, loads):
        arm = f'{candidates}_{layout}'
        assert f'rasterize_rgbd_{layout}_kernel' in row['Kernel Name'], row['Kernel Name']
        expected_widths = [32] * 6 + [128] if layout == 'prefetch' else [64, 128, 128]
        widths = [int(re.search(r'LDS(?:\.(\d+))?', ld['instruction'])[1] or 32) for ld in sass]
        assert widths == expected_widths, widths
        data = {**ncu['arms'][arm]['resources']}
        assert int(number(row['launch__grid_size'])) == 5440
        assert int(number(row['launch__block_size'])) == data['threads']
        assert int(number(row['launch__registers_per_thread'])) == data['registers']
        assert byte_value(row, units, 'launch__shared_mem_per_block_dynamic') == data['dynamic_shared_bytes']
        data['sm_shared_config_bytes'] = byte_value(row, units, 'launch__shared_mem_config_size')
        data.update({k: number(row[v]) for k, v in (METRICS | extra_metrics).items()})
        duration = 'gpu__time_duration.sum'
        data['ncu_duration_ms'] = number(row[duration]) * {'ns': 1e-6, 'us': 1e-3, 'ms': 1}[units[duration]]
        data['graph_ms_by_view'] = {f['view']: f['unprofiled_median_ms'][arm] for f in frames}
        data['graph_ms_mean_of_view_medians'] = mean(data['graph_ms_by_view'].values())
        data['geometry_loads'] = sass[:-1]
        data['rgbd_load'] = sass[-1]
        summary['arms'][arm] = data
    summary['quality'] = {}
    for candidates in LISTS:
        quality = [f['quality'][candidates] for f in validation['frames']]
        summary['quality'][candidates] = {channel: {
            'mae': mean(q[channel]['mae'] for q in quality),
            'max': max(q[channel]['max'] for q in quality),
            'bitwise_equal_frames': sum(q[channel]['bitwise_equal'] for q in quality),
        } for channel in ('rgb', 'depth', 'alpha')}
        summary['quality'][candidates]['last_ids_equal_frames'] = sum(q['last_ids_equal'] for q in quality)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
