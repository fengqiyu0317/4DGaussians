"""CPU-only source contracts for render profiling mode integration."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProfileRenderModeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "profile_render.py").read_text(encoding="utf-8")

    def test_default_remains_serial_and_tacker_is_explicit(self):
        self.assertIn(
            'choices=("serial", "split_serial", "two_stream", "tacker")',
            self.source,
        )
        self.assertIn('default="serial"', self.source)
        self.assertIn('parser.add_argument("--tacker-profile"', self.source)
        self.assertIn('parser.add_argument("--qualification-mode"', self.source)
        self.assertIn('parser.add_argument("--qualification-profile"', self.source)
        self.assertIn('parser.add_argument("--workload-name"', self.source)

    def test_renderer_setup_is_outside_profiled_render_loop(self):
        constructor = self.source.index("pipeline_renderer = TackerRenderer(")
        prepare = self.source.index('prepare = getattr(pipeline_renderer, "prepare"')
        profiler_start = self.source.index("cudaProfilerStart()")
        render_range = self.source.index('nvtx_range("profile/render_loop")')
        self.assertLess(constructor, profiler_start)
        self.assertLess(prepare, profiler_start)
        self.assertLess(profiler_start, render_range)

    def test_metadata_records_fallback_qualification_and_p50(self):
        for field in (
            '"actual_execution_mode"',
            '"tacker_fallback_reason"',
            '"two_stream_fallback_reason"',
            '"qualification_mode_executed"',
            '"profile_manifest_sha256"',
            '"p50_frame_ms"',
            '"frame_completion_ms"',
            '"kind": "4dgaussians_tacker_render_profile"',
            '"view_indices": view_indices',
        ):
            self.assertIn(field, self.source)
        self.assertIn("statistics.median(frame_completion_ms)", self.source)

    def test_python_37_grammar(self):
        ast.parse(self.source, "profile_render.py", feature_version=7)

    def test_normal_render_cli_is_also_fail_closed_and_serial_by_default(self):
        render_source = (ROOT / "render.py").read_text(encoding="utf-8")
        self.assertIn('choices=("serial", "two_stream", "tacker")', render_source)
        self.assertIn('default="serial"', render_source)
        self.assertIn("tacker mode requires --tacker-profile", render_source)
        self.assertIn("renderer.render_sequence(views)", render_source)
        ast.parse(render_source, "render.py", feature_version=7)

    def test_normal_render_prepares_pipeline_before_fps_timer(self):
        render_source = (ROOT / "render.py").read_text(encoding="utf-8")
        setup = render_source.index("sequence_renderer = _create_sequence_renderer(")
        prepare = render_source.index('prepare = getattr(sequence_renderer, "prepare"')
        timer = render_source.index("time1 = time()")
        self.assertLess(setup, timer)
        self.assertLess(prepare, timer)


if __name__ == "__main__":
    unittest.main()
