"""CPU contracts for the general entry point, without importing CUDA packages."""

from contextlib import contextmanager, nullcontext
import importlib.util
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'render.py'


@contextmanager
def load_entry():
    def module(name, **values):
        result = ModuleType(name)
        result.__dict__.update(values)
        return result

    gaussian_model = mock.Mock()
    modules = {
        'imageio': module('imageio', mimwrite=mock.Mock()),
        'numpy': module('numpy'),
        'torch': module('torch', no_grad=nullcontext, tensor=mock.Mock(),
                        float32='float32', cuda=SimpleNamespace(synchronize=mock.Mock())),
        'scene': module('scene', Scene=mock.Mock(), GaussianModel=gaussian_model),
        'cv2': module('cv2'),
        'tqdm': module('tqdm', tqdm=lambda values, **kwargs: values),
        'gaussian_renderer': module('gaussian_renderer', render=mock.Mock(),
                                   TackerRenderer=mock.Mock(), TwoStreamRenderer=mock.Mock(),
                                   GaussianModel=gaussian_model),
        'gaussian_renderer.gsplat_mixed': module('gaussian_renderer.gsplat_mixed',
                                               GsplatMixedRenderer=mock.Mock()),
        'torchvision': module('torchvision', utils=SimpleNamespace(save_image=mock.Mock())),
        'utils.general_utils': module('utils.general_utils', safe_state=mock.Mock()),
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('general_render_test_entry', ENTRY)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        yield entry, modules


class GeneralRenderEntryTest(unittest.TestCase):
    def test_serial_preserves_view_order_without_a_profile_for_both_backends(self):
        for backend in ('diff_gaussian', 'gsplat'):
            with self.subTest(backend=backend), load_entry() as (entry, _):
                pipe = SimpleNamespace(raster_backend=backend)
                entry.render.side_effect = lambda view, *args, **kwargs: {'render': view}
                self.assertIsNone(entry._create_sequence_renderer(
                    'model', pipe, 'bg', 'dynerf', 'serial', None, None, 7000))
                outputs = list(entry._render_outputs(
                    [3, 1, 2], 'model', pipe, 'bg', 'dynerf', 'serial', None, None, 7000))
                self.assertEqual([item['render'] for item in outputs], [3, 1, 2])
                self.assertTrue(all(call.args[2] is pipe for call in entry.render.call_args_list))
                entry.TackerRenderer.assert_not_called()

    def test_two_stream_does_not_require_a_profile(self):
        with load_entry() as (entry, _):
            result = entry._create_sequence_renderer(
                'model', 'pipe', 'bg', 'dynerf', 'two_stream', None, None, 7000)
            self.assertIs(result, entry.TwoStreamRenderer.return_value)
            entry.TwoStreamRenderer.assert_called_once_with(
                'model', 'pipe', 'bg', stage='fine', cam_type='dynerf')

    def test_fusion_modes_keep_explicit_profile_requirements(self):
        with load_entry() as (entry, modules):
            for mode in ('gsplat_mixed', 'gsplat_matched_graph'):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'tacker-profile'):
                    entry._create_sequence_renderer(
                        None, None, None, 'dynerf', mode, None, None, 7000)
            with self.assertRaisesRegex(ValueError, 'tacker-profile'):
                entry.render_sets(None, None, 7000, None, True, True, True, execution_mode='tacker')
            factory = modules['gaussian_renderer.gsplat_mixed'].GsplatMixedRenderer
            for mode, fused in (('gsplat_mixed', True), ('gsplat_matched_graph', False)):
                entry._create_sequence_renderer(
                    'model', SimpleNamespace(), 'bg', 'dynerf', mode, 'profile.json', 'scene', 7000)
                self.assertEqual(factory.call_args.args[3], 'profile.json')
                self.assertEqual(factory.call_args.kwargs['fused'], fused)
                self.assertEqual(factory.call_args.kwargs['kernel_variant'], 'tensor_v7_ca')

    def test_sequence_frame_count_mismatches_fail_and_synchronize(self):
        with load_entry() as (entry, _):
            for count, message in ((1, 'fewer'), (3, 'more')):
                renderer = mock.Mock()
                renderer.render_sequence.return_value = (i for i in range(count))
                with self.subTest(count=count), self.assertRaisesRegex(RuntimeError, message):
                    list(entry._render_outputs([1, 2], None, None, None, 'dynerf',
                                              'two_stream', None, None, 7000, renderer))
                renderer.synchronize.assert_called_once()

    def test_image_writer_keeps_frame_filenames(self):
        with load_entry() as (entry, _):
            entry.multithread_write(['first', 'second'], '/destination')
            entry.torchvision.utils.save_image.assert_has_calls([
                mock.call('first', '/destination/00000.png'),
                mock.call('second', '/destination/00001.png'),
            ], any_order=True)

    def test_image_writer_propagates_worker_failures(self):
        with load_entry() as (entry, _):
            entry.torchvision.utils.save_image.side_effect = OSError('disk full')
            with self.assertRaisesRegex(OSError, 'disk full'):
                entry.multithread_write(['image'], '/destination')

    def test_empty_camera_split_does_not_encode_an_empty_video(self):
        with load_entry() as (entry, _), mock.patch.object(entry, 'makedirs') as mkdir:
            entry.render_set('/unused', 'test', 7000, [], None, None, None, 'dynerf')
            mkdir.assert_not_called()
            entry.imageio.mimwrite.assert_not_called()
            entry.torch.cuda.synchronize.assert_not_called()

    def test_cli_uses_custom_model_data_config_and_iteration_without_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'custom_model'
            model.mkdir()
            (model / 'cfg_args').write_text(
                "Namespace(sh_degree=3, source_path='/old/data', model_path='/old/model', "
                "images='images', resolution=-1, white_background=True, data_device='cuda', "
                "eval=True, render_process=False, add_points=False, extension='.png', llffhold=8)")
            config = root / 'custom.py'
            config.write_text('ModelHiddenParams = dict(net_width=96)\n')
            for backend, iteration in (('diff_gaussian', 7000), ('gsplat', -1)):
                argv = [str(ENTRY), '-m', str(model), '-s', str(root / 'new_data'),
                        '--configs', str(config), '--iteration', str(iteration),
                        '--raster_backend', backend, '--skip_train', '--skip_test', '--skip_video']
                with self.subTest(backend=backend), load_entry() as (entry, modules), \
                        mock.patch.object(sys, 'argv', argv):
                    namespace = runpy.run_path(str(ENTRY), run_name='__main__')
                    args = namespace['args']
                    self.assertEqual(args.raster_backend, backend)
                    self.assertEqual(args.execution_mode, 'serial')
                    self.assertIsNone(args.tacker_profile)
                    scene_call = modules['scene'].Scene.call_args
                    self.assertEqual(scene_call.args[0].source_path, str(root / 'new_data'))
                    self.assertEqual(scene_call.args[0].model_path, str(model))
                    self.assertEqual(scene_call.kwargs['load_iteration'], iteration)
                    self.assertEqual(entry.GaussianModel.call_args.args[1].net_width, 96)
                    entry.TackerRenderer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
