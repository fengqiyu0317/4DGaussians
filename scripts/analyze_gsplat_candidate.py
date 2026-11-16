#!/usr/bin/env python3
"""Summarize the fixed candidate-pairing batch and validate its host audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from analyze_sampling_kernel_activity import foreign_python_processes


def read(path):
    return json.loads(path.read_text())


def host_audit(directory, stage):
    assert (directory / 'completed.txt').exists(), (stage, 'incomplete')
    before = foreign_python_processes((directory / 'processes_before.txt').read_text())
    assert not before, (stage, before)
    during = (directory / 'during_processes_and_frequency.txt').read_text()
    foreign = foreign_python_processes(during)
    own = ('/usr/bin/python3 -m unittest tests.test_gsplat_candidate_cuda -v'
           if stage == 'cuda' else '/usr/bin/python3 scripts/benchmark_gsplat_candidate.py ')
    foreign = sorted({line for line in foreign if not line.split(None, 3)[3].startswith(own)})
    assert not foreign, (stage, foreign)
    frequencies = [int(line) for line in during.splitlines() if re.fullmatch(r'\d{8,10}', line)]
    assert frequencies and set(frequencies) == {1300500000}, (stage, set(frequencies))
    return {'completed': True, 'foreign_workload_observed': False,
            'frequency_samples': len(frequencies), 'frequency_hz': sorted(set(frequencies))}


def compare_quality(value, reference):
    rows, refs = value['per_view'], reference['per_view']
    assert [r['view'] for r in rows] == [r['view'] for r in refs] == list(range(50))
    result = {}
    for metric in ('psnr_db', 'ssim', 'lpips'):
        diffs = [(abs(a[metric] - b[metric]), a['view']) for a, b in zip(rows, refs)]
        maximum, view = max(diffs, key=lambda x: x[0])
        result[metric] = {'mean_abs_difference': statistics.mean(x for x, _ in diffs),
                          'max_abs_difference': maximum, 'max_view': view}
    return result


def summarize(root, previous_report=None):
    leaf = read(root / 'results/leaf/report.json')
    formal = read(root / 'results/formal/report.json')
    assert leaf['completed'] and formal['completed'] and leaf['inputs_unchanged'] and formal['inputs_unchanged']
    for key in ('source_and_checkpoint_sha256', 'parent_qualification_sha256', 'workload',
                'predeclared_protocol', 'graph_configuration', 'configs'):
        assert leaf[key] == formal[key], key
    assert formal['parent_qualification_sha256'] == formal['predeclared_protocol']['qualification']['sha256']
    selected = formal['predeclared_protocol']['implementation_selection']['selected_cuda_source_sha256']
    assert formal['source_and_checkpoint_sha256'][
        '/data/qyfeng/gsplat-ca-20261027/project/gsplat_mixed_ext/mixed_tensor_v7_ca.cu'] == selected
    for case in leaf['leaf_cases']:
        assert all(arm['bitwise_equal'] for arm in case['arms'].values())
        mapping = case['mapping']['ca_on_5440']
        assert mapping['active'] and mapping['selected_extra_workers'] == 1531
        assert mapping['mlp_tiles_executed'] == 6971 and mapping['all_tiles_exactly_once']
    audit = {stage: host_audit(root / 'results' / stage, stage) for stage in ('cuda', 'leaf', 'formal')}
    test_log = (root / 'cuda_suite.log').read_text()
    assert 'Ran 5 tests' in test_log and '\nOK\n' in test_log, 'CUDA test suite incomplete'
    quality = formal['quality']
    quality_summary = {}
    for label, value in quality.items():
        row = {'means': {k: value['mean'][k] for k in ('psnr_db', 'ssim', 'lpips')},
               'metric_diff_vs_native': compare_quality(value, quality['gsplat_native'])}
        if label != 'gsplat_native':
            row['metric_diff_vs_independent'] = compare_quality(value, quality['independent'])
            row['actual_graph_replays_per_slot'] = value['actual_graph_replays_per_slot']
            assert sorted(row['actual_graph_replays_per_slot']) == [24, 25]
            row['quality_gate_passed'] = value['quality_gate_passed']
            assert value['quality_gate_passed']
            for ref in ('native', 'independent'):
                for output in ('rgb', 'depth'):
                    rows = value['per_view']
                    max_row = max(rows, key=lambda r: r[f'{output}_max_abs_vs_{ref}'])
                    row[f'{output}_diff_vs_{ref}'] = {
                        'mean_absolute': value['mean'][f'{output}_mae_vs_{ref}'],
                        'global_max_absolute': max_row[f'{output}_max_abs_vs_{ref}'],
                        'max_view': max_row['view']}
                    if ref == 'independent':
                        assert row[f'{output}_diff_vs_{ref}']['global_max_absolute'] == 0
        quality_summary[label] = row
    e2e = {label: value['median'] for label, value in formal['end_to_end'].items()}
    comparisons = {}
    for target, baseline in formal['predeclared_protocol']['primary_comparisons'] + formal['predeclared_protocol']['controls']:
        t, b = e2e[target], e2e[baseline]
        paired = [100 * (a['fps'] / z['fps'] - 1) for a, z in zip(
            formal['end_to_end'][target]['trials'], formal['end_to_end'][baseline]['trials'])]
        comparisons[f'{target}_vs_{baseline}'] = {
            'leaf_time_reduction_percent': 100 * (1 - leaf['leaf_average_ms'][target] / leaf['leaf_average_ms'][baseline]),
            'e2e_wall_time_reduction_percent': 100 * (1 - t['wall_ms'] / b['wall_ms']),
            'e2e_fps_increase_percent': 100 * (t['fps'] / b['fps'] - 1),
            'paired_round_fps_change_percent': {'min': min(paired), 'max': max(paired),
                                              'median': statistics.median(paired)}}
    files = ('results/leaf/report.json', 'results/formal/report.json', 'cuda_suite.log')
    checkpoint = {k: v for k, v in formal['source_and_checkpoint_sha256'].items()
                  if '/point_cloud/iteration_14000/' in k}
    assert len(checkpoint) == 4
    if previous_report:
        previous = read(previous_report)
        assert checkpoint == {k: v for k, v in previous['source_and_checkpoint_sha256'].items()
                              if '/point_cloud/iteration_14000/' in k}
        assert previous['parent_qualification_sha256'] == formal['parent_qualification_sha256']
    return {'completed': True, 'environment': formal['environment'], 'workload': formal['workload'],
            'checkpoint_sha256': checkpoint,
            'checkpoint_and_qualification_match_previous_experiment': bool(previous_report),
            'parent_qualification_sha256': formal['parent_qualification_sha256'],
            'graph_configuration': formal['graph_configuration'], 'host_audit': audit,
            'leaf_average_ms': leaf['leaf_average_ms'], 'end_to_end': e2e, 'comparisons': comparisons,
            'mapping': [{'raster_view': case['raster_view'], **case['mapping']} for case in leaf['leaf_cases']],
            'raster_only_controls': {k: v['median'] for k, v in formal['raster_only_controls'].items()},
            'quality': quality_summary, 'resources': formal['resources'],
            'raw_sha256': {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--previous-report', type=Path)
    args = parser.parse_args()
    result = summarize(args.root, args.previous_report)
    (args.root / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('leaf_average_ms', 'end_to_end', 'comparisons')}, indent=2))
