"""Run the predeclared Jetson A/B/C qualification, stage and renderer batch."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from analyze_sampling_kernel_activity import foreign_python_processes


GPU_FREQ = Path('/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu')
QUALIFICATION = Path('/data/qyfeng/4DGaussians-tacker-v12-split-20261001/benchmarks/results/jetson_orin_20260927/flame_steak/profiles_sm87_resources/full_mlp_overlap_graph.qualification.json')
MODEL = '/data/qyfeng/4DGaussians-flame-steak/outputs/n3dv_flame_steak'
DATA = '/data/qyfeng/datasets/n3dv/flame_steak_4dgs_min'
MODES = {'A': '0', 'B': 'batched', 'C': 'block_fused'}


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clock_state():
    return {name: int((GPU_FREQ / name).read_text()) for name in ('min_freq', 'max_freq', 'cur_freq')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--phase', choices=('cuda', 'quality', 'smoke', 'stage', 'renderer'), required=True)
    options = parser.parse_args()
    root, output = options.runtime_root.resolve(), options.output_root.resolve()
    if not str(root).startswith('/data/qyfeng/') or not str(output).startswith('/data/qyfeng/'):
        parser.error('all new runtime/results/cache files must stay on /data/qyfeng')
    phase = output / options.phase
    phase.mkdir(parents=True, exist_ok=False)
    protocol_path = root / 'benchmarks/protocols/jetson_v13_abc_20261014.json'
    protocol = read_json(protocol_path)
    (phase / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    if sha(QUALIFICATION) != protocol['qualification_sha256']:
        raise RuntimeError('unexpected Graph qualification profile bytes')
    env = dict(os.environ)
    for name in list(env):
        if name.startswith('TACKER_') or name == 'LD_PRELOAD':
            env.pop(name)
    env.update(
        CUDA_VISIBLE_DEVICES='0', TACKER_CUDA_ARCH='sm_87', TACKER_GPU_NAME='Orin',
        PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(root / 'tmp'),
        XDG_CACHE_HOME=str(root / 'cache'), CUDA_CACHE_PATH=str(root / 'cache/cuda'),
        TORCH_HOME=str(root / 'cache/torch'), TORCH_EXTENSIONS_DIR=str(root / 'cache/extensions'),
        MPLCONFIGDIR=str(root / 'cache/matplotlib'),
        PYTHONPATH=':'.join(str(root / name) for name in (
            '_python_deps_compat', 'tacker_ext', 'submodules/depth-diff-gaussian-rasterization',
            'submodules/simple-knn', '')) + ':/home/jetson/.local/lib/python3.10/site-packages',
        FOURDGS_SOURCE_COMMIT='60018d41f4f57b693b0e0dc4052f85b0c9d31a99', FOURDGS_SOURCE_DIRTY='true',
        FOURDGS_RASTERIZER_COMMIT='464096373b869fff713265bcdc34b1822277980d',
        FOURDGS_SIMPLE_KNN_COMMIT='44f764299fa305faf6ec5ebd99939e0508331503',
        TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12='1', TACKER_HEXPLANE_LOWREG='0',
        TACKER_HEXPLANE_FUSED_REDUCTION='1', TACKER_HEXPLANE_REDUCTION_THREADS='256',
        TACKER_EXPERIMENTAL_RASTER_HEXPLANE='0', TACKER_RASTER_HEXPLANE_TASKS_PER_CTA='1',
        TACKER_RASTER_HEXPLANE_MIXED_PLANES='12', TACKER_EXPERIMENTAL_GREEN_CONTEXTS='0',
        TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF='0')
    for name in ('TMPDIR', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'MPLCONFIGDIR'):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    (phase / 'environment.json').write_text(json.dumps({key: value for key, value in env.items()
        if key.startswith(('TACKER_', 'CUDA_', 'FOURDGS_')) or key in ('PYTHONPATH', 'TMPDIR', 'TORCH_HOME')}, indent=2) + '\n')
    sources = [protocol_path, QUALIFICATION]
    for directory in ('gaussian_renderer', 'scene', 'utils', 'arguments', 'scripts',
                      'tacker_ext/tacker_4dgs_head', 'submodules/depth-diff-gaussian-rasterization'):
        sources += [path for path in (root / directory).rglob('*') if path.is_file()
                    and path.suffix in ('.py', '.so', '.cu', '.cuh', '.h')
                    and not any(part in ('build', '__pycache__', 'third_party') for part in path.relative_to(root).parts)]
    sources += [root / 'profile_render.py']
    manifest = {str(path): sha(path) for path in sorted(set(sources))}
    (phase / 'runtime.sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def verify_sources(path):
        changed = [name for name, digest in manifest.items() if sha(Path(name)) != digest]
        path.write_text(json.dumps({'passed': not changed, 'changed': changed}) + '\n')
        if changed:
            raise RuntimeError('runtime source/binary changed: ' + str(changed))

    owned_python_pids = {os.getpid()}

    def foreign(text, script=None):
        # Retain positively identified task PIDs: a terminating Python process
        # may briefly appear as [python3] after its argv has disappeared.
        if script:
            for line in text.splitlines():
                parts = line.split(None, 3)
                if (len(parts) == 4 and parts[0].isdigit() and parts[2].startswith('python')
                        and script in parts[3] and str(root) in parts[3]):
                    owned_python_pids.add(int(parts[0]))
        found = []
        for line in foreign_python_processes(text):
            pid = int(line.split(None, 1)[0])
            if pid in owned_python_pids:
                continue
            found.append(line)
        return found

    def check_idle(directory, stem):
        verify_sources(directory / (stem + '.source_check.json'))
        processes = subprocess.check_output(['ps', '-eo', 'pid,user,comm,args'], text=True)
        (directory / (stem + '.processes.txt')).write_text(processes)
        bad = foreign(processes)
        if bad:
            raise RuntimeError('foreign Python compute processes: ' + str(bad))
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv'], text=True)
        (directory / (stem + '.gpu_processes.csv')).write_text(gpu)
        if any(row and row[0].strip().isdigit() for row in list(csv.reader(gpu.splitlines()))[1:]):
            raise RuntimeError('GPU process query reports active compute processes')
        snapshot = subprocess.run(['timeout', '3', 'tegrastats', '--interval', '1000'],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (directory / (stem + '.tegrastats.txt')).write_text(snapshot.stdout)
        values = re.findall(r'GR3D_FREQ (\d+)%', snapshot.stdout)
        if not values or any(value != '0' for value in values):
            raise RuntimeError('GPU not idle in tegrastats: ' + snapshot.stdout)
        clocks = clock_state()
        (directory / (stem + '.clocks.json')).write_text(json.dumps(clocks) + '\n')
        if any(value != protocol['hardware']['GPU_clock_hz'] for value in clocks.values()):
            raise RuntimeError('GPU clocks changed: ' + str(clocks))
        (directory / (stem + '.memory.txt')).write_text(subprocess.check_output(['free', '-m'], text=True))

    def run(name, arguments, mode='0', root_process=False, extra_env=None):
        directory = phase / name
        directory.mkdir(exist_ok=False)
        check_idle(directory, 'before')
        case_env = {**env, 'TACKER_EXPERIMENTAL_RASTER_HEXPLANE': mode, **(extra_env or {})}
        # Only task-local process environments are passed to sudo; no driver,
        # clock, service, or other shared system settings are modified.
        command = ['/usr/bin/python3', *map(str, arguments)]
        if root_process:
            keys = ('CUDA_VISIBLE_DEVICES', 'PYTHONDONTWRITEBYTECODE', 'TMPDIR', 'XDG_CACHE_HOME',
                    'CUDA_CACHE_PATH', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'MPLCONFIGDIR', 'PYTHONPATH')
            settings = [key + '=' + value for key, value in case_env.items()
                        if key.startswith(('TACKER_', 'FOURDGS_')) or key in keys]
            command = ['sudo', '-n', 'env', *settings, 'timeout', '1200', *command]
        else:
            command = ['timeout', '1200', *command]
        (directory / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        stop = threading.Event()
        violations = []
        script = Path(arguments[0]).name
        def monitor():
            with (directory / 'during.audit.jsonl').open('w') as handle:
                while not stop.is_set():
                    try:
                        text = subprocess.check_output(['ps', '-eo', 'pid,user,comm,args'], text=True)
                        bad = foreign(text, script)
                        clocks = clock_state()
                        record = {'time_unix': time.time(), 'foreign': bad, 'clocks': clocks,
                                  'owned_task_pids': sorted(owned_python_pids), 'processes': text}
                        handle.write(json.dumps(record) + '\n')
                        handle.flush()
                        if bad or any(value != protocol['hardware']['GPU_clock_hz'] for value in clocks.values()):
                            violations.append(record)
                    except BaseException as error:
                        violations.append({'monitor_error': str(error)})
                    stop.wait(1)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        print('START', options.phase, name, flush=True)
        try:
            with (directory / 'run.log').open('w') as log:
                result = subprocess.run(command, cwd=root, env=case_env, stdout=log, stderr=subprocess.STDOUT)
        finally:
            stop.set()
            thread.join()
        (directory / 'audit_summary.json').write_text(json.dumps({'passed': not violations, 'violations': violations}, indent=2) + '\n')
        if result.returncode or violations:
            raise RuntimeError(f'{name}: exit={result.returncode}; audit_violations={len(violations)}; see {directory}')
        # tegrastats utilization can retain the final sample briefly.
        time.sleep(2)
        check_idle(directory, 'after')
        (directory / 'completed.txt').write_text('complete\n')
        print('PASS', options.phase, name, flush=True)
        return directory

    renderer_args = ['--model_path', MODEL, '--source_path', DATA,
                     '--configs', str(root / 'arguments/dynerf/flame_steak.py'), '--iteration', '14000',
                     '--split', 'test', '--trials', '1', '--workload-name', 'flame_steak',
                     '--execution-mode', 'tacker', '--qualification-mode',
                     '--qualification-profile', str(QUALIFICATION), '--quiet']
    if options.phase == 'cuda':
        for name, file in [('bc', 'test_hexplane_batch_blocks_cuda.py'),
                           ('solo', 'test_hexplane_cuda.py'), ('old_mixed', 'test_raster_hexplane_mixed_cuda.py')]:
            run(name, [root / 'tacker_ext/tests' / file, '-v'], root_process=True,
                extra_env={'TACKER_BC_TEST_TRACE_DIR': str(phase / 'graph_traces')})
    elif options.phase == 'quality':
        for arm in 'ABC':
            path = phase / arm / 'quality.json'
            run(arm, [root / 'scripts/validate_tacker_modes.py', '-m', MODEL, '-s', DATA,
                      '--configs', root / 'arguments/dynerf/flame_steak.py', '--iteration', '14000',
                      '--scene-name', 'flame_steak', '--expected-image-width', '1352',
                      '--expected-image-height', '1014', '--expected-gaussian-count', '111525',
                      '--split', 'test', '--frames', '10', '--modes', 'serial', 'tacker',
                      '--qualification-mode', '--qualification-profile', QUALIFICATION,
                      '--lpips-net', 'vgg', '--gpu', '0', '--quiet', '--output', path], mode=MODES[arm])
            if not read_json(path)['passed']:
                raise RuntimeError('quality failed: ' + arm)
    elif options.phase in ('stage', 'smoke'):
        smoke = options.phase == 'smoke'
        for index in range(1 if smoke else protocol['stage']['independent_processes']):
            name = f'run{index}'
            run(name, [root / 'scripts/benchmark_sampling_prepass_bc.py', '--runtime-root', root,
                       '--benchmark-output', phase / name / 'benchmark.json', '--node-priority-backend',
                       '--abba-blocks', str(1 if smoke else protocol['stage']['symmetric_blocks_per_slot']),
                       '--repeats', str(2 if smoke else protocol['stage']['replays_per_batch']),
                       '--bench-warmup', str(1 if smoke else protocol['stage']['warmup_replays']),
                       *renderer_args, '--warmup', '20', '--frames', '5',
                       '--metadata', str(phase / name / 'extraction.json')], root_process=True)
    else:
        schedule = protocol['renderer']['order'] * protocol['renderer']['symmetric_blocks']
        for index, arm in enumerate(schedule):
            name = f'{index:02d}_{arm}'
            path = phase / name / 'metadata.json'
            run(name, [root / 'scripts/run_profile_render_sealed.py', *renderer_args,
                       '--warmup', str(protocol['renderer']['warmup_frames']),
                       '--frames', str(protocol['renderer']['measured_frames']),
                       '--metadata', path], mode=MODES[arm])
            data = read_json(path)
            if not (data['passed'] and data['actual_execution_mode'] == 'tacker'
                    and data['qualification_mode_executed'] and data['tacker_fallback_reason'] is None
                    and data['selected_variant_id'] == protocol['actual_variant']
                    and data['qualification_profile_sha256'] == protocol['qualification_sha256']
                    and data['post_run']['byte_stability']['verified']
                    and data['pipeline_execution_counts']['mixed_launches'] == 49
                    and data['pipeline_execution_counts']['outputs'] == 50):
                raise RuntimeError('renderer qualification/dispatch validation failed: ' + name)
    verify_sources(phase / 'final.source_check.json')
    (phase / 'completed.txt').write_text('complete\n')
    print('PHASE_COMPLETE', options.phase, phase, flush=True)


if __name__ == '__main__':
    main()
