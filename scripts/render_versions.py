"""Portable, explicit qualification runner for ordinary v13 and gsplat tensor_v7_ca.

This exports images and execution counts, not benchmark FPS. It does not require
Git metadata, enable deployment profiles, or alter CUDA implementations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--variant', choices=('v13', 'gsplat_ca'), required=True)
    result.add_argument('--profile', type=Path, required=True)
    result.add_argument('--model', type=Path, required=True, help='Checkpoint model directory')
    result.add_argument('--source', type=Path, required=True, help='Prepared dataset directory')
    result.add_argument('--config', type=Path, default=ROOT / 'arguments/dynerf/flame_steak.py')
    result.add_argument('--output', type=Path, required=True, help='New output directory')
    result.add_argument('--frames', type=int, default=50)
    result.add_argument('--device', default='0', help='One CUDA_VISIBLE_DEVICES index or UUID')
    result.add_argument('--dry-run', action='store_true', help='Inspect the plan without importing PyTorch or writing files')
    return result


def make_plan(args):
    if args.frames < 2:
        raise ValueError('at least two frames are required to exercise the mixed kernel')
    if not args.device.strip() or ',' in args.device:
        raise ValueError('--device must select exactly one GPU')
    raw = args.profile.read_bytes()
    profile = json.loads(raw)
    candidate = next(c for c in profile['candidates']
                     if c['variant_id'] == profile['selected_variant_id'])
    if candidate.get('partition', {}).get('kind') != 'full_mlp_overlap_graph':
        raise ValueError('a full_mlp_overlap_graph qualification profile is required')
    if profile.get('deployment', {}).get('enabled'):
        raise ValueError('use a disabled qualification profile for this research runner')
    manifest = profile['manifest']
    arch = manifest['cuda_arch']
    if arch not in ('sm_86', 'sm_87'):
        raise ValueError('this release documents only sm_86 and sm_87')
    config = args.config.expanduser().resolve(strict=True)
    model = args.model.expanduser().resolve()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for protected in (model, source):
        if output == protected or protected in output.parents:
            raise ValueError('--output must be outside the model and dataset directories')
    env = {
        'CUDA_VISIBLE_DEVICES': args.device,
        'TACKER_CUDA_ARCH': arch,
        'TACKER_GPU_NAME': manifest['gpu_name'],
        'TORCH_CUDA_ARCH_LIST': '{}.{}'.format(*manifest['compute_capability']),
        'TACKER_EXPERIMENTAL_RASTER_HEXPLANE': '0',
        'TACKER_HEXPLANE_LOWREG': '0',
        'TACKER_HEXPLANE_FUSED_REDUCTION': '1' if args.variant == 'v13' else '0',
        'TACKER_HEXPLANE_REDUCTION_THREADS': '256',
    }
    if args.variant == 'v13':
        env['TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12'] = '1'
    return {
        'variant': args.variant, 'qualification_mode': True,
        'qualification_partition': 'full_mlp_overlap_graph',
        'selected_variant_id': profile['selected_variant_id'],
        'profile': str(args.profile.expanduser().resolve()),
        'profile_sha256': hashlib.sha256(raw).hexdigest(),
        'model': str(model), 'source': str(source), 'config': str(config),
        'output': str(output), 'frames': args.frames,
        'manifest': manifest, 'environment': env,
        'kernel': 'mixed_abi_v12' if args.variant == 'v13' else 'tensor_v7_ca',
        'gsplat_physical_blocks': 0,
    }


def configure_environment(plan):
    # Experimental settings from a previous run must not silently change this version.
    for name in list(os.environ):
        if name.startswith('TACKER_'):
            del os.environ[name]
    os.environ.update(plan['environment'])
    cache = Path(plan['output']) / 'cache'
    for name, folder in (('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'),
                         ('CUDA_CACHE_PATH', 'cuda'), ('TORCH_HOME', 'torch'),
                         ('TORCH_EXTENSIONS_DIR', 'extensions'), ('MPLCONFIGDIR', 'matplotlib')):
        path = cache / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def audit_execution(renderer, variant, count):
    reason = renderer.last_fallback_reason
    if reason is not None:
        raise RuntimeError('requested version fell back: ' + str(reason))
    counts = renderer.last_execution_counts
    for key, value in {'input_frames': count, 'outputs': count,
                       'mixed_launches': count - 1, 'full_deformation': 1}.items():
        if counts.get(key) != value:
            raise RuntimeError('unexpected {}: {}'.format(key, counts.get(key)))
    if variant == 'v13':
        if not renderer.last_used_tacker or not renderer.last_qualification_mode:
            raise RuntimeError('physical Tacker qualification path was not used')
        if counts.get('raster_prepasses') != count - 1:
            raise RuntimeError('v12 Raster prepass was not executed')
        if (not renderer.slots or
                sum(slot.hexplane_graph is not None for slot in renderer.slots) < min(count - 1, 2)):
            raise RuntimeError('HexPlane Graph slots were not populated')
    elif counts.get('prefix_graph_replays') != count - 1:
        raise RuntimeError('gsplat HexPlane Graph path was not executed')
    return counts


def run(plan):
    model, source, output = (Path(plan[name]) for name in ('model', 'source', 'output'))
    if not model.is_dir() or not source.is_dir():
        raise ValueError('model and prepared dataset directories must exist')
    checkpoint = model / 'point_cloud' / ('iteration_' + str(plan['manifest']['iteration']))
    for filename in ('point_cloud.ply', 'deformation.pth'):
        if not (checkpoint / filename).is_file():
            raise ValueError('missing checkpoint file: ' + str(checkpoint / filename))
    output.mkdir(parents=True, exist_ok=False)
    configure_environment(plan)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    sys.path.insert(0, str(ROOT))
    import torch
    from torchvision.utils import save_image
    from arguments import ModelParams, PipelineParams, ModelHiddenParams
    from utils.params_utils import load_config, merge_hparams
    from utils.general_utils import safe_state
    from scene import Scene, GaussianModel
    from gaussian_renderer.tacker_pipeline import TackerRenderer, load_tacker_profile
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    options_parser = argparse.ArgumentParser()
    model_group = ModelParams(options_parser)
    pipeline_group = PipelineParams(options_parser)
    hidden_group = ModelHiddenParams(options_parser)
    options = merge_hparams(options_parser.parse_args([]), load_config(plan['config']))
    options.model_path, options.source_path = str(model), str(source)
    options.raster_backend = 'diff_gaussian' if plan['variant'] == 'v13' else 'gsplat'
    options.gsplat_packed = False
    options.gsplat_mixed_kernel = 'tensor_v7_ca'
    pipeline = pipeline_group.extract(options)
    safe_state(False)
    with torch.no_grad():
        # Retain all normal profile/hash, workload, resource and device checks.
        profile = load_tacker_profile(plan['profile'])
        if hashlib.sha256(Path(plan['profile']).read_bytes()).hexdigest() != plan['profile_sha256']:
            raise RuntimeError('profile changed after planning')
        pc = GaussianModel(options.sh_degree, hidden_group.extract(options))
        scene = Scene(model_group.extract(options), pc,
                      load_iteration=plan['manifest']['iteration'], shuffle=False)
        pc._deformation.eval()
        cameras = scene.getTestCameras()
        if len(cameras) < plan['frames']:
            raise ValueError('requested more test views than the dataset contains')
        views = [cameras[index] for index in range(plan['frames'])]
        for view in views:
            if [view.image_width, view.image_height] != profile['manifest']['resolution']:
                raise ValueError('view resolution does not match the qualification profile')
        bg = torch.tensor([1, 1, 1] if options.white_background else [0, 0, 0],
                          dtype=torch.float32, device='cuda')
        common = dict(workload_name=profile['manifest']['workload'],
                      iteration=profile['manifest']['iteration'], cam_type=scene.dataset_type)
        if plan['variant'] == 'v13':
            renderer = TackerRenderer(pc, pipeline, bg, profile_override=profile,
                                      qualification_mode=True, **common)
            if renderer.fallback_reason is not None:
                raise RuntimeError(renderer.fallback_reason)
        else:
            from gaussian_renderer.gsplat_mixed import GsplatMixedRenderer
            renderer = GsplatMixedRenderer(pc, pipeline, bg, plan['profile'],
                                          kernel_variant='tensor_v7_ca', physical_blocks=0,
                                          fused=True, candidate_pairing=True, **common)
        frames = renderer.render_sequence(views)
        images = output / 'renders'
        images.mkdir()
        produced = 0
        try:
            for index, result in enumerate(frames):
                save_image(result['render'].clamp(0, 1), images / ('%05d.png' % index))
                produced += 1
            renderer.synchronize()
            if produced != plan['frames']:
                raise RuntimeError('incorrect number of returned images')
            counts = audit_execution(renderer, plan['variant'], produced)
        finally:
            frames.close()
            renderer.synchronize()
            close = getattr(renderer, 'close', None)
            if close:
                close()
        metadata = dict(plan, completed=True, execution_counts=counts,
                        torch_version=torch.__version__, cuda_version=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(0),
                        timing_scope='image export only; not a throughput benchmark')
        (output / 'execution.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Completed:', output)


def main():
    args = parser().parse_args()
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        run(plan)


if __name__ == '__main__':
    main()
