#!/usr/bin/env python3
"""Check the paired Jetson benchmark and summarize its three timing scopes."""
import csv
import json
from pathlib import Path
import re
from statistics import mean
import sys

from analyze_gsplat_leaf_resources import byte_value, number
from analyze_gsplat_rgbd_ablation import METRICS
from analyze_sampling_kernel_activity import foreign_python_processes

PHASES = ('leaf', 'formal', 'ncu')
NCU_ARMS = {
    'raster': 'rasterize_to_pixels_3dgs_fwd_kernel',
    'wide_raster': 'gsplat_raster_standalone_wide',
    'mixed': 'gsplat_mix_render_deform_tensor_v7_ca',
    'wide_mixed': 'gsplat_mix_render_deform_tensor_v7_ca_wide',
}


def audit(path):
    assert (path / 'completed.txt').is_file()
    hashes = (path / 'after_check.txt').read_text().splitlines()
    assert hashes and all(line.endswith(': OK') for line in hashes)
    assert not foreign_python_processes((path / 'processes_before.txt').read_text())
    samples = (path / 'during_processes_and_frequency.txt').read_text()
    expected = ('/usr/bin/python3 /data/qyfeng/gsplat-wide-mixed-20261103/project/'
                'scripts/benchmark_gsplat_wide_mixed.py ')
    foreign = foreign_python_processes(samples)
    assert all(line.split(None, 3)[3].startswith(expected) for line in foreign), foreign
    clocks = re.findall(r'^\d{10}$', samples, re.MULTILINE)
    assert set(clocks) == {'1300500000'}, clocks
    return {'clock_hz': 1300500000, 'samples': len(clocks), 'passed': True}


def main(root, reference):
    paths = {phase: root / 'results' / phase for phase in PHASES}
    leaf, formal, ncu = [json.loads((paths[p] / 'report.json').read_text()) for p in PHASES]
    previous = json.loads(reference.read_text())
    for phase in (leaf, formal, ncu):
        assert phase['completed']
        for key in ('workload', 'profile', 'configurations', 'parent_qualification_sha256',
                    'source_and_checkpoint_sha256', 'binary_sha256'):
            assert phase[key] == leaf[key], key
    for key in ('workload', 'parent_qualification_sha256'):
        assert leaf[key] == previous[key], key
    for path, digest in previous['source_and_checkpoint_sha256'].items():
        if '/iteration_14000/' in path:
            assert leaf['source_and_checkpoint_sha256'][path] == digest, path
    for path, digest in previous['binary_sha256'].items():
        if not path.endswith('.so') or 'diff_gaussian_rasterization' in path:
            continue
        assert leaf['binary_sha256'][path] == digest, path
    assert [c['raster_view'] for c in leaf['leaf_cases']] == [0, 24, 48]
    first, profiled = leaf['leaf_cases'][0], ncu['leaf_cases'][0]
    assert first['intersections'] == profiled['intersections'] == previous['intersections']['gsplat']
    for key in ('candidate_assignment_sha256', 'candidate_extra_workers', 'deform_tiles',
                'actual_prefix_replays'):
        assert first[key] == profiled[key], key
    assert all(row['bitwise_equal'] for case in leaf['leaf_cases'] for row in case['arms'].values())
    summary = {k: leaf[k] for k in ('workload', 'profile', 'parent_qualification_sha256', 'binary_sha256',
                                    'configurations', 'leaf_ms')}
    summary['audits'] = {phase: audit(path) for phase, path in paths.items()}
    summary['mapping'] = [{k: c[k] for k in ('raster_view', 'deform_view', 'intersections',
        'candidate_assignment_sha256', 'candidate_extra_workers', 'deform_tiles')} for c in leaf['leaf_cases']]
    summary['end_to_end'] = {}
    summary['prepare_pixels_layout'] = {}
    summary['quality_vs_independent'] = {}
    for name, (_, fused) in leaf['configurations'].items():
        result = formal['end_to_end'][name]
        assert len(result['trials']) == 10
        assert len(formal['prepare_pixels_layout'][name]['trials']) == 6
        counts = result['execution_counts']
        assert counts['outputs'] == counts['raster_prepares'] == 50 and counts['prefix_graph_replays'] == 49
        assert counts['mixed_launches'] == (49 if fused else 0) and counts['solo_raster'] == 1
        assert counts['separate_leaf_pairs'] == (0 if fused else 49)
        summary['end_to_end'][name] = {**result['median'], 'execution_counts': counts}
        summary['prepare_pixels_layout'][name] = formal['prepare_pixels_layout'][name]['median']
        quality = formal['quality'][name]
        assert sum(quality['actual_prefix_replays_per_slot']) == 49
        assert [row['view'] for row in quality['rows']] == list(range(50))
        summary['quality_vs_independent'][name] = {channel: {
            'mae': mean(row[channel]['mae'] for row in quality['rows']),
            'max': max(row[channel]['max'] for row in quality['rows']),
            'bitwise_equal_frames': sum(row[channel]['bitwise_equal'] for row in quality['rows']),
        } for channel in ('rgb', 'depth')}
    summary['ncu'] = {}
    with (paths['ncu'] / 'report.raw.csv').open() as handle:
        reader = csv.DictReader(handle)
        units = next(reader)
        rows = [row for row in reader if row.get('ID', '').isdigit()]
    assert len(rows) == len(NCU_ARMS)
    additional = {
        'achieved_warp_occupancy': 'sm__warps_active.avg.pct_of_peak_sustained_active',
        'eligible_warps_per_scheduler': 'smsp__warps_eligible.avg.per_cycle_active',
        'local_load_sectors': 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
        'local_store_sectors': 'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
        'shared_load_warp_instructions': 'smsp__sass_inst_executed_op_shared_ld.sum',
    }
    for (arm, kernel), row in zip(NCU_ARMS.items(), rows):
        assert kernel in row['Kernel Name'], row['Kernel Name']
        assert first['arms'][arm]['resources'] == profiled['arms'][arm]['resources']
        assert profiled['arms'][arm]['bitwise_equal']
        data = {**profiled['arms'][arm]['resources'][0]}
        assert int(number(row['launch__grid_size'])) == 5440
        assert int(number(row['launch__block_size'])) == data['threads']
        assert int(number(row['launch__registers_per_thread'])) == data['registers']
        data['sm_shared_config_bytes'] = byte_value(row, units, 'launch__shared_mem_config_size')
        data.update({k: number(row[v]) for k, v in (METRICS | additional).items()})
        duration = 'gpu__time_duration.sum'
        data['ncu_duration_ms'] = number(row[duration]) * {'ns': 1e-6, 'us': 1e-3, 'ms': 1}[units[duration]]
        summary['ncu'][arm] = data
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
