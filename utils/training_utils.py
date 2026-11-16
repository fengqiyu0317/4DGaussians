"""Training argument handling without importing PyTorch or initializing CUDA."""

from argparse import ArgumentParser
from pathlib import Path

from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.params_utils import load_config, merge_hparams


def checkpoint_stage(path):
    """Accept the stage markers used by the original training checkpoints."""
    name = Path(path).name
    stages = [stage for stage in ('coarse', 'fine') if stage in name]
    if len(stages) != 1:
        raise ValueError('checkpoint filename must contain exactly one of coarse or fine')
    return stages[0]


def parse_training_args(argv=None):
    parser = ArgumentParser(description='Train 4DGaussians with the differentiable rasterizer')
    groups = (ModelParams(parser), ModelHiddenParams(parser),
              OptimizationParams(parser), PipelineParams(parser))
    parser.add_argument('--ip', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--disable_gui', action='store_true', help='Train without opening a viewer socket')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers; 0 avoids worker subprocesses')
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[3000, 7000, 14000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[14000, 20000, 30000, 45000, 60000])
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--start_checkpoint')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--expname', default='')
    parser.add_argument('--configs', default='')
    args = parser.parse_args(argv)
    if args.configs:
        # Preserve the original precedence: config values override group CLI values.
        args = merge_hparams(args, load_config(args.configs))
    if not args.source_path:
        parser.error('--source_path/-s is required')
    if not args.model_path and not args.expname:
        parser.error('provide --model_path/-m or --expname for the training output')
    if args.iterations < 1 or args.coarse_iterations < 0 or args.batch_size < 1:
        parser.error('iterations and batch_size must be positive; coarse_iterations must be nonnegative')
    if args.num_workers < 0:
        parser.error('--num_workers must be nonnegative')
    if args.raster_backend != 'diff_gaussian':
        parser.error('training uses diff_gaussian; gsplat is available through the inference entry points')
    if args.start_checkpoint:
        try:
            checkpoint_stage(args.start_checkpoint)
        except ValueError as error:
            parser.error(str(error))
        if not Path(args.start_checkpoint).is_file():
            parser.error('checkpoint file does not exist: ' + args.start_checkpoint)
    args.source_path = str(Path(args.source_path).expanduser().resolve())
    args.model_path = str(Path(args.model_path or Path('output') / args.expname).expanduser().resolve())
    if args.model_path == args.source_path or Path(args.source_path) in Path(args.model_path).parents:
        parser.error('training output must be outside the source dataset directory')
    # Configs can change iterations; the final fine-stage model must always be saved.
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    return args, groups
