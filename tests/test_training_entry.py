"""CPU regression tests for restored training configuration and orchestration."""

import ast
from contextlib import redirect_stderr
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from utils.params_utils import load_config
from utils.training_utils import checkpoint_stage, parse_training_args

ROOT = Path(__file__).resolve().parents[1]


class TrainingArgumentsTests(unittest.TestCase):
    def parse(self, *extra):
        return parse_training_args(['-s', '/tmp/training-data', '-m', '/tmp/training-model', *extra])[0]

    def test_help_does_not_import_runtime_or_touch_cuda(self):
        result = subprocess.run([sys.executable, str(ROOT / 'train.py'), '--help'],
                                capture_output=True, text=True, cwd='/tmp')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--start_checkpoint', result.stdout)
        self.assertIn('--disable_gui', result.stdout)

    def test_final_save_uses_configured_iteration_and_preserves_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.py'
            config.write_text('OptimizationParams = dict(iterations=17, coarse_iterations=3, batch_size=2)\n')
            args = self.parse('--iterations', '22', '--configs', str(config), '--save_iterations', '2')
        self.assertEqual(args.iterations, 17)
        self.assertEqual(args.save_iterations, [2, 17])
        self.assertEqual(args.batch_size, 2)

    def test_expname_sets_model_directory(self):
        args, _ = parse_training_args(['-s', '/tmp/training-data', '--expname', 'unit-test'])
        self.assertEqual(Path(args.model_path), (Path.cwd() / 'output/unit-test').resolve())

    def test_training_rejects_inference_backend_and_invalid_sizes(self):
        for options in [('--raster_backend', 'gsplat'), ('--iterations', '0'),
                        ('--coarse_iterations', '-1'), ('--batch_size', '0'), ('--num_workers', '-1')]:
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(*options)

    def test_output_cannot_overwrite_source_data(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_training_args(['-s', '/tmp/data', '-m', '/tmp/data/model'])

    def test_checkpoint_stage_uses_filename_not_parent_path(self):
        self.assertEqual(checkpoint_stage('/tmp/coarse/runs/chkpnt_fine_7.pth'), 'fine')
        self.assertEqual(checkpoint_stage('/tmp/fine/runs/chkpnt_coarse_3.pth'), 'coarse')
        for name in ('checkpoint.pth', 'coarse_fine.pth'):
            with self.assertRaises(ValueError):
                checkpoint_stage(name)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chkpnt_fine_7.pth'
            path.touch()
            args = self.parse('--start_checkpoint', str(path))
            self.assertEqual(args.start_checkpoint, str(path))

    def test_every_shipped_scene_config_loads_its_base(self):
        configs = list((ROOT / 'arguments').glob('*/*.py'))
        self.assertGreaterEqual(len(configs), 23)
        for config in configs:
            with self.subTest(config=config.name):
                self.assertIsInstance(load_config(config), dict)
                args = self.parse('--configs', str(config))
                self.assertIn(args.iterations, args.save_iterations)

    def test_headless_and_worker_settings(self):
        args = self.parse('--disable_gui', '--num_workers', '2')
        self.assertTrue(args.disable_gui)
        self.assertEqual(args.num_workers, 2)


class TrainingStagesTests(unittest.TestCase):
    def test_exhausted_dataloader_fetches_the_new_epoch_batch(self):
        tree = ast.parse((ROOT / 'train.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'scene_reconstruction')
        branch = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == 'opt.dataloader and (not load_in_memory)')
        batch = [object()]
        namespace = dict(opt=SimpleNamespace(dataloader=True), load_in_memory=False,
                         loader=iter(()), random_loader=True, viewpoint_stack_loader=[batch],
                         viewpoint_cams=['stale batch'], print=lambda *args: None)
        loop = ast.For(target=ast.Name(id='_step', ctx=ast.Store()),
                       iter=ast.Tuple(elts=[ast.Constant(value=0)], ctx=ast.Load()),
                       body=[branch], orelse=[])
        module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
        exec(compile(module, 'training batch selection', 'exec'), namespace)
        self.assertIs(namespace['viewpoint_cams'], batch)

    def test_last_iteration_saves_each_stage_even_without_a_scheduled_save(self):
        tree = ast.parse((ROOT / 'train.py').read_text())
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == 'iteration in saving_iterations or iteration == final_iter')
        for stage in ('coarse', 'fine'):
            scene = Mock()
            namespace = dict(iteration=17, final_iter=17, saving_iterations=[], stage=stage,
                             scene=scene, print=lambda *args: None)
            exec(compile(ast.Module(body=[branch], type_ignores=[]), 'training final save', 'exec'), namespace)
            scene.save.assert_called_once_with(17, stage)

    def test_training_runs_coarse_then_fine_with_same_scene_and_output(self):
        # Execute the actual orchestration function with lightweight model/scene doubles.
        tree = ast.parse((ROOT / 'train.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'training')
        model, scene, writer = object(), object(), object()
        timer = Mock()
        reconstruct = Mock()
        namespace = dict(prepare_output_and_logger=Mock(return_value=writer),
                         GaussianModel=Mock(return_value=model), Timer=Mock(return_value=timer),
                         Scene=Mock(return_value=scene), scene_reconstruction=reconstruct,
                         args=SimpleNamespace(model_path='/tmp/restored-training'))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / 'train.py'), 'exec'), namespace)
        dataset = SimpleNamespace(sh_degree=3, model_path='')
        opt = SimpleNamespace(coarse_iterations=3, iterations=17)
        namespace['training'](dataset, object(), opt, object(), [], [17], [7], None, -1, 'run')
        self.assertEqual(dataset.model_path, '/tmp/restored-training')
        self.assertEqual([call.args[-4] for call in reconstruct.call_args_list], ['coarse', 'fine'])
        self.assertEqual([call.args[-2] for call in reconstruct.call_args_list], [3, 17])
        for call in reconstruct.call_args_list:
            self.assertIs(call.args[9], model)
            self.assertIs(call.args[10], scene)
        timer.start.assert_called_once()


if __name__ == '__main__':
    unittest.main()
