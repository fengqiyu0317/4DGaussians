"""Validate and summarize the complete predeclared Jetson A/B/C batch."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from benchmark_sampling_prepass_bc import summarize_abc, validate_abc_nodes
from sampling_prepass_trace import split_graph_kernels


def read(path):
    return json.loads(path.read_text())


def comparisons(times):
    return {label: {'ratio': times[new] / times[old],
                    'saved_ms': times[old] - times[new],
                    'time_saved_percent': 100 * (1 - times[new] / times[old]),
                    'speedup': times[old] / times[new]}
            for label, new, old in [('B_vs_A', 'B', 'A'), ('C_vs_B', 'C', 'B'), ('C_vs_A', 'C', 'A')]}


def validate_phase(path):
    if not (path / 'completed.txt').exists():
        raise ValueError('incomplete/rejected phase: ' + str(path))
    protocol = read(path / 'protocol.json')
    for check in path.rglob('*.source_check.json'):
        if not read(check)['passed']:
            raise ValueError('source check failed: ' + str(check))
    for job in sorted(p for p in path.iterdir() if p.is_dir() and (p / 'command.json').exists()):
        if not (job / 'completed.txt').exists() or not read(job / 'audit_summary.json')['passed']:
            raise ValueError('incomplete/rejected job: ' + str(job))
        samples = 0
        with (job / 'during.audit.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                if row['foreign'] or any(value != protocol['hardware']['GPU_clock_hz'] for value in row['clocks'].values()):
                    raise ValueError('during-run audit failed: ' + str(job))
                samples += 1
        if samples < 1:
            raise ValueError('missing continuous audit: ' + str(job))
    return protocol, read(path / 'runtime.sha256.json')


def qualified(data, protocol):
    if not (data['passed'] and data['actual_execution_mode'] == 'tacker'
            and data['qualification_mode_executed'] and data['tacker_fallback_reason'] is None
            and data['selected_variant_id'] == protocol['actual_variant']
            and data['qualification_profile_sha256'] == protocol['qualification_sha256']
            and data['post_run']['byte_stability']['verified']):
        raise ValueError('execution did not satisfy the required Graph qualification')


def analyze(batch, output):
    protocols, manifests = [], []
    for name in ('cuda', 'quality', 'smoke', 'stage', 'renderer'):
        protocol, manifest = validate_phase(batch / name)
        protocols.append(protocol)
        manifests.append(manifest)
    if any(value != protocols[0] for value in protocols[1:]) or any(value != manifests[0] for value in manifests[1:]):
        raise ValueError('protocol/runtime differs across admitted phases')
    protocol = protocols[0]
    stage_rows, stage_processes, stage_cases, diagnostic_rows = [], [], [], []
    fingerprints = {}
    resources = None
    for index in range(protocol['stage']['independent_processes']):
        directory = batch / 'stage' / f'run{index}'
        data = read(directory / 'benchmark.json')
        qualified(read(directory / 'extraction.json'), protocol)
        if not (data['correctness_passed'] and data['policy']['graph_backend'] == 'cuda_driver_use_node_priority'
                and data['policy']['requested_graph_instantiate_flags'] == 8
                and data['actual_variant'] == protocol['actual_variant']
                and data['qualification_sha256'] == protocol['qualification_sha256']
                and len(data['cases']) == protocol['stage']['real_input_slots_per_process']):
            raise ValueError('invalid stage backend or qualification')
        if resources is not None and data['resources'] != resources:
            raise ValueError('CUDA resources changed across stage processes')
        resources = data['resources']
        process_cases = []
        for case in data['cases']:
            if not case['B_C_sampling_bitwise_equal']:
                raise ValueError('B/C sampler mismatch')
            summary = summarize_abc(case['records'])
            if summary != case['summary'] or summary['block_count'] != protocol['stage']['symmetric_blocks_per_slot']:
                raise ValueError('stage symmetric-block summary mismatch')
            for row in case['records']:
                if len(row['samples_ms']) != protocol['stage']['replays_per_batch'] or row['mean_ms'] != statistics.mean(row['samples_ms']):
                    raise ValueError('raw event timing mismatch')
                stage_rows.append({'process': index, 'case': case['case'], 'block': row['block'],
                                   'position': row['position'], 'arm': row['arm'], 'mean_ms': row['mean_ms']})
            for arm in 'ABC':
                work = case['workloads'][arm]
                if (work['points'], work['resolution'], work['raster_tile_range']) != (111525, [1352, 1014], [0, 1954]):
                    raise ValueError('workload geometry changed')
                if work['stream_priorities'] != {'control': 0, 'sampling': 0, 'prepass': 0}:
                    raise ValueError('stream priorities changed')
                key = case['case']
                fingerprint = work['workload_fingerprints']
                if key in fingerprints and fingerprint != fingerprints[key]:
                    raise ValueError('input/reference fingerprints changed across processes or arms')
                fingerprints[key] = fingerprint
                if len(work['graph_priority_audits']) != 5:
                    raise ValueError('missing Graph instantiation audits')
                for mode, audit in work['graph_priority_audits'].items():
                    counts = validate_abc_nodes(audit['nodes'], arm, mode, 111525, 1954)
                    if not audit['verified'] or audit['requested_instantiation_flags'] != 8 or counts != audit['kernel_counts']:
                        raise ValueError('Graph instantiation/node audit failed')
                    if audit['instantiate_flags'] != 8:
                        proof = audit.get('behavioral_control', {})
                        if (proof.get('default_update_status'), proof.get('default_update_result'),
                            proof.get('priority_update_status'), proof.get('priority_update_result')) != (0, 0, 910, 8):
                            raise ValueError('unverified Driver flag readback quirk')
                audit = case['kernel_audits'][arm]
                body, outside = split_graph_kernels(read(directory / f"case{case['case']}.{arm}.trace.json"),
                                                   driver_audit=work['graph_priority_audits']['both'])
                if not (len(body) == audit['graph_kernel_count'] == {'A': 13, 'B': 2, 'C': 1}[arm]
                        and len(outside) == audit['outside_graph_kernel_count']
                        and [row['name'] for row in body] == audit['symbols']):
                    raise ValueError('trace kernel count mismatch')
                if arm in 'AB':
                    raster = next(row for row in body if row['name'] == 'tacker_render_raster_fast_range_v12')
                    sampling = [row for row in body if row is not raster]
                    overlap = sum(max(0, min(row['ts'] + row['dur'], raster['ts'] + raster['dur'])
                                      - max(row['ts'], raster['ts'])) for row in sampling) / 1000
                    diagnostic_rows.append({'process': index, 'case': case['case'], 'arm': arm,
                                            'sampling_kernel_sum_ms': sum(row['dur'] for row in sampling) / 1000,
                                            'prepass_ms': raster['dur'] / 1000,
                                            'kernel_lifetime_overlap_ms': overlap})
            times = {arm: summary['median'][arm + '_ms'] for arm in 'ABC'}
            entry = {'process': index, 'case': case['case'], 'times_ms': times,
                     'comparisons': comparisons(times),
                     'isolated_controls_ms': {key: value['mean_ms'] for key, value in case['isolated_controls'].items()}}
            process_cases.append(entry)
            stage_cases.append(entry)
        means = {arm: statistics.mean(case['times_ms'][arm] for case in process_cases) for arm in 'ABC'}
        stage_processes.append({'process': index, 'times_ms': means, 'comparisons': comparisons(means)})
    stage_times = {arm: statistics.mean(row['times_ms'][arm] for row in stage_processes) for arm in 'ABC'}
    controls = {key: statistics.mean(row['isolated_controls_ms'][key] for row in stage_cases)
                for key in stage_cases[0]['isolated_controls_ms']}
    renderer_rows = []
    views = None
    schedule = protocol['renderer']['order'] * protocol['renderer']['symmetric_blocks']
    for index, arm in enumerate(schedule):
        path = batch / 'renderer' / f'{index:02d}_{arm}' / 'metadata.json'
        data = read(path)
        qualified(data, protocol)
        if (data['warmup_frames'], data['profile_frames'], data['trial_count']) != (20, 50, 1):
            raise ValueError('renderer timing protocol changed')
        if views is not None and data['view_indices'] != views:
            raise ValueError('renderer views changed')
        views = data['view_indices']
        counts = data['pipeline_execution_counts']
        expected = {'mixed_launches': 49, 'raster_prepasses': 49, 'prefix': 49, 'outputs': 50}
        if arm in 'BC':
            expected.update(raster_hexplane_fused_launches=49 if arm == 'C' else 0,
                            raster_hexplane_sampling_graph_replays=49 if arm == 'B' else 0)
        if any(counts.get(key) != value for key, value in expected.items()):
            raise ValueError('renderer A/B/C dispatch counts changed')
        if not abs(data['throughput_fps'] * data['mean_frame_ms'] - 1000) < 1e-6:
            raise ValueError('renderer throughput/frame-time mismatch')
        renderer_rows.append({'index': index, 'block': index // 6, 'position': index % 6,
                              'arm': arm, 'mean_frame_ms': data['mean_frame_ms'],
                              'throughput_fps': data['throughput_fps'],
                              'cuda_event_mean_frame_ms': data['cuda_event_mean_frame_ms']})
    renderer_blocks = []
    for begin in range(0, len(renderer_rows), 6):
        records = [{'arm': row['arm'], 'mean_ms': row['mean_frame_ms']} for row in renderer_rows[begin:begin + 6]]
        times = {arm: summarize_abc(records)['median'][arm + '_ms'] for arm in 'ABC'}
        renderer_blocks.append({'block': begin // 6, 'times_ms': times, 'comparisons': comparisons(times)})
    renderer_times = {arm: statistics.mean(row['mean_frame_ms'] for row in renderer_rows if row['arm'] == arm) for arm in 'ABC'}
    quality = {arm: read(batch / 'quality' / arm / 'quality.json') for arm in 'ABC'}
    if not all(value['passed'] for value in quality.values()):
        raise ValueError('quality gate failed')
    summary = {
        'status': 'complete_and_validated', 'date': protocol['date'], 'timezone': protocol['timezone'],
        'qualification_profile': 'full_mlp_overlap_graph', 'actual_variant': protocol['actual_variant'],
        'qualification_sha256': protocol['qualification_sha256'], 'protocol': protocol,
        'stage': {'times_ms': stage_times, 'comparisons': comparisons(stage_times),
                  'aggregation': 'mean of three process means; each process averages the two slot medians of symmetric-block means',
                  'processes': stage_processes, 'cases': stage_cases,
                  'timed_replays_per_arm': protocol['stage']['independent_processes'] * protocol['stage']['real_input_slots_per_process']
                      * protocol['stage']['symmetric_blocks_per_slot'] * 2 * protocol['stage']['replays_per_batch'],
                  'standalone_controls_ms': controls,
                  'standalone_sampling_time_saved_percent': 100 * (1 - controls['B_sampling'] / controls['A_sampling']),
                  'process_mean_ranges_ms': {arm: [min(row['times_ms'][arm] for row in stage_processes),
                                                  max(row['times_ms'][arm] for row in stage_processes)] for arm in 'ABC'}},
        'renderer': {'times_ms': renderer_times,
                     'aggregate_throughput_fps': {arm: 1000 / value for arm, value in renderer_times.items()},
                     'comparisons': comparisons(renderer_times), 'blocks': renderer_blocks, 'runs': renderer_rows,
                     'aggregation': 'mean whole-sequence frame time across four independent processes per arm',
                     'run_ranges_ms': {arm: [min(row['mean_frame_ms'] for row in renderer_rows if row['arm'] == arm),
                                             max(row['mean_frame_ms'] for row in renderer_rows if row['arm'] == arm)] for arm in 'ABC'}},
        'resources': resources, 'quality': {arm: {'passed': value['passed'], 'deltas': value['deltas']} for arm, value in quality.items()},
        'post_timing_trace_diagnostics': {
            'means': {arm: {key: statistics.mean(row[key] for row in diagnostic_rows if row['arm'] == arm)
                           for key in ('sampling_kernel_sum_ms', 'prepass_ms', 'kernel_lifetime_overlap_ms')} for arm in 'AB'},
            'records': diagnostic_rows,
            'warning': 'Kernel lifetime overlap includes stalls, not instruction-active overlap. These profiler diagnostics were collected after primary event timings.'},
        'workload_fingerprints': fingerprints, 'runtime_sha256': manifests[0],
        'remote_project_root': str(next(Path(name).parents[1] for name in manifests[0]
                                       if name.endswith('/scripts/run_jetson_sampling_prepass_bc.py'))),
        'rejected_preflight_attempts': {
            'results': 'CUDA tests passed; audit falsely classified a positively identified terminating task PID after argv disappeared. No formal timings.',
            'results_v2': 'Preflight smoke only; Jetson Kineto omitted direct Driver API callback. Fixed by requiring one positive correlation group plus independently verified Driver nodes and matching trace geometry. No formal stage/renderer data pooled.'},
        'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limits': ['Stage microbenchmark excludes prepare, reduction and MLP; renderer measurements include the full sequence.',
                   'B versus A includes custom sampler arithmetic/implementation as well as launch consolidation.',
                   'Renderer B/C comparisons include their current Graph submission/dependency schedules.',
                   'Ranges are observed replication ranges, not confidence intervals; historical batches are not pooled.']}
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'raw_artifacts.tar.gz'
    if archive.exists():
        summary['raw_artifacts_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    for name, rows in [('stage_batches.csv', stage_rows), ('renderer_runs.csv', renderer_rows)]:
        with (output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': summary['status'],
                      'stage': {key: summary['stage'][key] for key in ('times_ms', 'comparisons', 'process_mean_ranges_ms')},
                      'renderer': {key: summary['renderer'][key] for key in ('times_ms', 'aggregate_throughput_fps', 'comparisons', 'run_ranges_ms')},
                      'resources': resources}, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    options = parser.parse_args()
    analyze(options.batch_root, options.output_root)
