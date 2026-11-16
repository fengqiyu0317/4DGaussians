"""Run the shipped core suites without discovering local archived experiments."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

CPU_SUITES = (
    ("tests", (
        "test_arguments", "test_render_entry", "test_render_versions", "test_render_pipeline",
        "test_tacker_pipeline", "test_tacker_phase2", "test_full_mlp_partition",
        "test_gsplat_candidate_pairing", "test_hexplane_block_mapping",
        "test_hexplane_fused_reduction", "test_raster_hexplane_graph",
    )),
    ("tacker_ext/tests", (
        "test_contract", "test_validation", "test_v2_contract",
        "test_v2_python_api", "test_v2_validation",
    )),
    ("submodules/depth-diff-gaussian-rasterization/tests", (
        "test_tacker_mixed_contract",
    )),
)

CUDA_SUITES = (
    ("tests", (
        "test_full_mlp_partition_cuda", "test_gsplat_backend_cuda",
        "test_gsplat_mixed_cuda", "test_gsplat_tensor_v7_cuda",
        "test_gsplat_candidate_cuda",
    )),
    ("tacker_ext/tests", (
        "test_deform_mlp_cuda", "test_head_linear_cuda", "test_head_linear_v2_cuda",
        "test_hexplane_cuda", "test_hexplane_reduction_cuda",
        "test_raster_hexplane_mixed_cuda",
    )),
    ("submodules/depth-diff-gaussian-rasterization/tests", (
        "test_tacker_mixed_cuda", "test_stream_aware_legacy_cuda",
    )),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true', help='Also run CUDA suites on the selected GPU')
    parser.add_argument('--list', action='store_true', help='List selected test files without executing them')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    suites = CPU_SUITES + (CUDA_SUITES if args.cuda else ())
    for directory, names in suites:
        for name in names:
            path = ROOT / directory / (name + '.py')
            if not path.is_file():
                parser.error('missing core test: ' + str(path))
            if args.list:
                print(path.relative_to(ROOT))
    if args.list:
        return 0

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    if args.cuda:
        subprocess.run([sys.executable, '-c',
                        'import torch; assert torch.cuda.is_available(), "CUDA is required"'],
                       env=env, check=True, cwd=ROOT)
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    for directory, names in suites:
        paths = [ROOT, ROOT / 'tacker_ext',
                 ROOT / 'submodules/depth-diff-gaussian-rasterization', ROOT / directory]
        suite_env = dict(env, PYTHONPATH=os.pathsep.join(map(str, paths)))
        modules = ['tests.' + name if directory == 'tests' else name for name in names]
        command = [sys.executable, '-m', 'unittest', *modules]
        if args.verbose:
            command.append('-v')
        print('Running: ' + directory + ' (' + ', '.join(names) + ')', flush=True)
        result = subprocess.run(command, cwd=ROOT, env=suite_env)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
