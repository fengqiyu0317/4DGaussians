#!/usr/bin/env python3
"""Verify and summarize the same-kernel tile membership experiment."""
import csv
import json
from pathlib import Path
import re
from statistics import mean
import sys

from analyze_gsplat_leaf_resources import byte_value, number
from analyze_gsplat_rgbd_ablation import METRICS
from analyze_sampling_kernel_activity import foreign_python_processes

ARMS = ('gsplat_tiles', 'legacy_tiles')


def audit(path):
    assert (path / 'completed.txt').is_file()
    hashes = (path / 'after_check.txt').read_text().splitlines()
    assert hashes and all(line.endswith(': OK') for line in hashes)
    assert not foreign_python_processes((path / 'processes_before.txt').read_text())
    samples = (path / 'during_processes_and_frequency.txt').read_text()
    expected = '/usr/bin/python3 /data/qyfeng/gsplat-tile-ablation-20261103/profile_gsplat_tile_ablation.py '
    foreign = foreign_python_processes(samples)
    assert all(line.split(None, 3)[3].startswith(expected) for line in foreign), foreign
    clocks = re.findall(r'^\d{10}$', samples, re.MULTILINE)
    assert set(clocks) == {'1300500000'}, clocks
    return {'clock_hz': 1300500000, 'samples': len(clocks), 'passed': True}


def main(root, reference_root):
    validation_dir, ncu_dir = (root / 'results' / name for name in ('validation', 'ncu'))
    validation, ncu, previous = [json.loads(p.read_text()) for p in
        (validation_dir / 'report.json', ncu_dir / 'report.json',
         reference_root / 'results/validation/report.json')]
    assert validation['completed'] and ncu['completed']
    for key in ('workload', 'graph', 'arms', 'parent_qualification_sha256',
                'source_and_checkpoint_sha256', 'ablation_sha256'):
        assert validation[key] == ncu[key], key
    for key in ('workload', 'graph', 'parent_qualification_sha256'):
        assert ncu[key] == previous[key], key
    for path, digest in previous['source_and_checkpoint_sha256'].items():
        if '/iteration_14000/' in path:
            assert ncu['source_and_checkpoint_sha256'][path] == digest, path
    for path, digest in previous['ablation_sha256'].items():
        if path.endswith(('gsplat_cuda.so', 'gsplat_mixed_tensor_v7_ca.so', 'prefetch.so')):
            assert ncu['ablation_sha256'][path] == digest, path
    first = validation['frames'][0]
    for key, value in ncu['frames'][0].items():
        assert first[key] == value, key
    assert first['intersections']['gsplat_tiles'] == previous['frames'][0]['intersections']
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
    summary['view0_intersections'] = first['intersections']
    summary['view0_supplemented_gaussians'] = first['supplemented_gaussians']
    summary['arms'] = {}
    assert ncu['arms'][ARMS[0]] == ncu['arms'][ARMS[1]]
    for arm, row in zip(ARMS, rows):
        assert 'rasterize_rgbd_prefetch_kernel' in row['Kernel Name'], row['Kernel Name']
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
        summary['arms'][arm] = data
    summary['quality'] = {}
    for channel in ('rgb', 'depth', 'alpha'):
        quality = [(f['view'], f['quality'][channel]) for f in validation['frames']]
        view, largest = max(quality, key=lambda q: q[1]['max'])
        summary['quality'][channel] = {
            'mae': mean(q['mae'] for _, q in quality), 'max': largest['max'], 'max_view': view,
            'view0': quality[0][1],
            'different_pixels': sum(q['different_pixels'] for _, q in quality),
            'bitwise_equal_frames': sum(q['bitwise_equal'] for _, q in quality),
        }
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
