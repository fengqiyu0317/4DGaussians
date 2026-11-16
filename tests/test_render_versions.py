"""CPU checks for version selection, profile preservation and fail-closed audit."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('render_versions', ROOT / 'scripts/render_versions.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class VersionRunnerTests(unittest.TestCase):
    def options(self, variant='v13', profile=None):
        return runner.parser().parse_args([
            '--variant', variant,
            '--profile', str(profile or ROOT / 'profiles/flame_steak_sm87.qualification.json'),
            '--model', '/tmp/test-version-model', '--source', '/tmp/test-version-data',
            '--output', '/tmp/test-version-output', '--dry-run'])

    def test_versions_select_distinct_kernels_without_changing_profile(self):
        path = ROOT / 'profiles/flame_steak_sm87.qualification.json'
        original = path.read_bytes()
        normal = runner.make_plan(self.options())
        gsplat = runner.make_plan(self.options('gsplat_ca'))
        self.assertEqual(normal['qualification_partition'], 'full_mlp_overlap_graph')
        self.assertEqual(normal['environment']['TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12'], '1')
        self.assertEqual(normal['environment']['TACKER_HEXPLANE_FUSED_REDUCTION'], '1')
        self.assertNotIn('TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12', gsplat['environment'])
        self.assertEqual(gsplat['environment']['TACKER_HEXPLANE_FUSED_REDUCTION'], '0')
        self.assertEqual(gsplat['kernel'], 'tensor_v7_ca')
        self.assertEqual(original, path.read_bytes())
        self.assertEqual(normal['profile_sha256'], gsplat['profile_sha256'])

    def test_eager_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = json.loads(self.options().profile.read_text())
            chosen = next(c for c in profile['candidates'] if c['variant_id'] == profile['selected_variant_id'])
            chosen['partition']['kind'] = 'full_mlp_overlap'
            path = Path(directory) / 'eager.json'
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, 'full_mlp_overlap_graph'):
                runner.make_plan(self.options(profile=path))

    def test_output_cannot_be_inside_inputs_and_one_frame_is_rejected(self):
        for field in ('model', 'source'):
            options = self.options()
            options.output = getattr(options, field) / 'renders'
            with self.assertRaisesRegex(ValueError, 'outside'):
                runner.make_plan(options)
        options = self.options()
        options.frames = 1
        with self.assertRaisesRegex(ValueError, 'two frames'):
            runner.make_plan(options)

    def test_old_experiment_environment_cannot_leak_into_gsplat(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = runner.make_plan(self.options('gsplat_ca'))
            plan['output'] = directory
            with patch.dict(os.environ, {'TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12': '1',
                                         'TACKER_MIXED_V12_SPLIT_TILE': '100'}):
                runner.configure_environment(plan)
                self.assertNotIn('TACKER_EXPERIMENTAL_MIXED_TENSOR_TAIL_V12', os.environ)
                self.assertNotIn('TACKER_MIXED_V12_SPLIT_TILE', os.environ)
                self.assertEqual(os.environ['TACKER_GPU_NAME'], 'Orin')
                self.assertTrue(Path(os.environ['TORCH_EXTENSIONS_DIR']).is_dir())

    def renderer(self, count=2):
        return SimpleNamespace(last_fallback_reason=None, last_used_tacker=True,
            last_qualification_mode=True,
            last_execution_counts={'input_frames': count, 'outputs': count,
                'mixed_launches': count - 1, 'full_deformation': 1,
                'raster_prepasses': count - 1, 'prefix_graph_replays': count - 1},
            slots=[SimpleNamespace(hexplane_graph=None), SimpleNamespace(hexplane_graph=object())])

    def test_two_frame_run_needs_only_the_used_graph_slot(self):
        runner.audit_execution(self.renderer(), 'v13', 2)
        runner.audit_execution(self.renderer(), 'gsplat_ca', 2)

    def test_fallback_or_missing_graph_cannot_be_reported_as_success(self):
        renderer = self.renderer()
        renderer.last_fallback_reason = 'resource mismatch'
        with self.assertRaisesRegex(RuntimeError, 'fell back'):
            runner.audit_execution(renderer, 'v13', 2)
        renderer = self.renderer()
        renderer.slots[1].hexplane_graph = None
        with self.assertRaisesRegex(RuntimeError, 'Graph'):
            runner.audit_execution(renderer, 'v13', 2)
        renderer = self.renderer()
        renderer.last_execution_counts['prefix_graph_replays'] = 0
        with self.assertRaisesRegex(RuntimeError, 'Graph'):
            runner.audit_execution(renderer, 'gsplat_ca', 2)


if __name__ == '__main__':
    unittest.main()
