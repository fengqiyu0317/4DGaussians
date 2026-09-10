"""CPU-only source contracts for render profiling mode integration."""

import ast
import os
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock


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
        main_source = self.source[self.source.index("def main(") :]
        constructor = main_source.index("pipeline_renderer = TackerRenderer(")
        prepare = main_source.index('prepare = getattr(pipeline_renderer, "prepare"')
        profiler_start = main_source.index("cudaProfilerStart()")
        measured_loop = main_source.index(
            "for trial_index in range(1, args.trials + 1)"
        )
        self.assertLess(constructor, profiler_start)
        self.assertLess(prepare, profiler_start)
        self.assertLess(profiler_start, measured_loop)

    def test_trials_cli_is_positive_and_defaults_to_one(self):
        self.assertIn(
            'parser.add_argument("--trials", default=1, type=int)', self.source
        )
        self.assertIn("if args.trials <= 0:", self.source)
        self.assertIn('raise ValueError("--trials must be positive")', self.source)

    def test_one_setup_and_warmup_precede_all_measured_trials(self):
        main_source = self.source[self.source.index("def main(") :]
        load = main_source.index("scene = Scene(")
        prepare = main_source.index('prepare = getattr(pipeline_renderer, "prepare"')
        warmup = main_source.index("run_views(\n            warmup_views,")
        trial_loop = main_source.index(
            "for trial_index in range(1, args.trials + 1)"
        )
        self.assertLess(load, prepare)
        self.assertLess(prepare, warmup)
        self.assertLess(warmup, trial_loop)

    def test_each_trial_has_independent_wall_and_cuda_boundaries(self):
        trial_source = self.source[
            self.source.index("def _measure_trial(") : self.source.index(
                "def _aggregate_trials("
            )
        ]
        start_event = trial_source.index("completion_start.record(")
        wall_start = trial_source.index("start_time = perf_counter()")
        submit = trial_source.index("fallback_reason = run_views(")
        end_event = trial_source.index("completion_end.record(")
        synchronize = trial_source.index("torch.cuda.synchronize()")
        wall_end = trial_source.index("elapsed_seconds = perf_counter() - start_time")
        self.assertLess(start_event, wall_start)
        self.assertLess(wall_start, submit)
        self.assertLess(submit, end_event)
        self.assertLess(end_event, synchronize)
        self.assertLess(synchronize, wall_end)

    def test_all_modes_use_the_same_call_site_synchronization_boundary(self):
        run_views_source = self.source[
            self.source.index("def run_views(") : self.source.index(
                "def _measure_trial("
            )
        ]
        self.assertNotIn("pipeline_renderer.synchronize()", run_views_source)
        trial_source = self.source[
            self.source.index("def _measure_trial(") : self.source.index(
                "def _aggregate_trials("
            )
        ]
        self.assertEqual(trial_source.count("torch.cuda.synchronize()"), 1)

    def test_metadata_records_fallback_qualification_and_trial_statistics(self):
        for field in (
            '"actual_execution_mode"',
            '"tacker_fallback_reason"',
            '"two_stream_fallback_reason"',
            '"qualification_mode_executed"',
            '"profile_manifest_sha256"',
            '"profile_selection_sha256"',
            '"selected_variant_id"',
            '"selected_candidate_abi_sha256"',
            '"p50_frame_ms"',
            '"p95_frame_ms"',
            '"max_frame_ms"',
            '"frame_completion_ms"',
            '"cuda_event_total_render_ms"',
            '"total_render_ms"',
            '"trials": trials',
            '"median_throughput_fps"',
            '"primary_metric": "median_throughput_fps"',
            '"kind": "4dgaussians_tacker_render_profile"',
            '"view_indices": view_indices',
            '"source_files": source_files',
            '"profile_render_sha256"',
            '"gaussian_renderer/__init__.py"',
            '"gaussian_renderer/tacker_pipeline.py"',
            '"diff_gaussian_rasterization/__init__.py"',
            '"diff_gaussian_rasterization._C"',
        ):
            self.assertIn(field, self.source)
        self.assertIn("statistics.median(frame_completion_ms)", self.source)

    def test_multi_trial_metadata_binds_one_physical_execution_state(self):
        self.assertIn("execution_signatures = {", self.source)
        self.assertIn(
            '"physical execution mode or fallback changed between trials"',
            self.source,
        )
        self.assertIn(
            '"p50_frame_ms": representative_trial["p50_frame_ms"]',
            self.source,
        )
        self.assertIn('"median_{}".format(name)', self.source)
        self.assertIn("metadata.update(aggregates)", self.source)

    def test_timing_contract_keeps_legacy_fields(self):
        for field in (
            '"timing_method": "perf_counter_with_cuda_synchronize"',
            '"frame_timing_method": "cuda_event_consumer_completion_intervals"',
            '"io_in_timed_region": False',
            '"unit": "whole_sequence"',
            '"setup_policy": "single_load_prepare_warmup_before_all_trials"',
            '"cuda_events": "start_end_and_per_frame_completion"',
        ):
            self.assertIn(field, self.source)

    def test_environment_and_source_provenance_are_best_effort(self):
        for field in (
            '"environment": environment',
            '"repository": repository',
            '"profile_hashes": profile_hashes',
            '"metadata_collection_errors"',
            '"cuda_driver_version"',
            '"clocks.current.graphics"',
            '"temperature.gpu"',
            '"power.management"',
            '"FOURDGS_SOURCE_COMMIT"',
            '"commit_source"',
            '"git_error"',
        ):
            self.assertIn(field, self.source)

    def test_profile_hash_uses_the_same_snapshot_as_runtime_dispatch(self):
        self.assertIn("def _load_profile_snapshot(", self.source)
        self.assertIn("hashlib.sha256(raw).hexdigest()", self.source)
        self.assertIn("tacker_profile_sha256_snapshot=", self.source)
        self.assertIn("qualification_profile_sha256_snapshot=", self.source)
        self.assertIn("profile_override=(", self.source)
        self.assertIn("profile_path=None", self.source)

    def test_provenance_is_snapshotted_before_warmup_and_timed_trials(self):
        provenance_offset = self.source.index(
            "reproducibility = _collect_reproducibility_metadata("
        )
        warmup_offset = self.source.index("run_views(\n            warmup_views")
        trial_offset = self.source.index("trial = _measure_trial(")
        self.assertLess(provenance_offset, warmup_offset)
        self.assertLess(provenance_offset, trial_offset)
        self.assertEqual(
            self.source.count("reproducibility = _collect_reproducibility_metadata("),
            1,
        )

    def test_percentile_and_trial_aggregation_helpers(self):
        tree = ast.parse(self.source, "profile_render.py", feature_version=7)
        wanted = {"_percentile", "_aggregate_trials"}
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        namespace = {"statistics": statistics}
        exec(compile(ast.Module(body=functions), "helpers", "exec"), namespace)

        self.assertAlmostEqual(namespace["_percentile"]([1, 2, 3, 4], 95), 3.85)
        self.assertEqual(namespace["_percentile"]([7], 95), 7.0)
        metric_names = (
            "elapsed_seconds",
            "total_render_ms",
            "throughput_fps",
            "mean_frame_ms",
            "cuda_event_total_render_ms",
            "cuda_event_mean_frame_ms",
            "p50_frame_ms",
            "p95_frame_ms",
            "max_frame_ms",
        )
        first = {name: 10.0 for name in metric_names}
        second = {name: 20.0 for name in metric_names}
        aggregate = namespace["_aggregate_trials"]([first, second])
        for name in metric_names:
            self.assertEqual(aggregate["median_{}".format(name)], 15.0)

    def test_source_commit_environment_fallback_without_git_tree(self):
        tree = ast.parse(self.source, "profile_render.py", feature_version=7)
        wanted = {
            "_environment_submodule_commits",
            "_collect_repository_metadata",
        }
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        namespace = {"os": os}
        exec(compile(ast.Module(body=functions), "helpers", "exec"), namespace)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {
                    "FOURDGS_SOURCE_COMMIT": "a" * 40,
                    "FOURDGS_RASTERIZER_COMMIT": "b" * 40,
                },
                clear=False,
            ):
                errors = []
                metadata = namespace["_collect_repository_metadata"](
                    Path(directory), errors
                )
        self.assertEqual(metadata["commit"], "a" * 40)
        self.assertEqual(metadata["commit_source"], "environment")
        self.assertIn("no .git metadata", metadata["git_error"])
        self.assertTrue(errors)
        submodules = {item["path"]: item for item in metadata["submodules"]}
        raster = submodules[
            "submodules/depth-diff-gaussian-rasterization"
        ]
        self.assertEqual(raster["commit"], "b" * 40)
        self.assertEqual(raster["source"], "environment")
        self.assertIsNone(submodules["submodules/simple-knn"]["commit"])

    def test_clean_submodule_status_keeps_full_gitlink_commit(self):
        tree = ast.parse(self.source, "profile_render.py", feature_version=7)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_parse_submodule_status"
        )
        namespace = {}
        exec(compile(ast.Module(body=[function]), "helper", "exec"), namespace)
        commit = "c" * 40
        parsed = namespace["_parse_submodule_status"](
            " {} submodules/example (heads/main)".format(commit)
        )
        self.assertEqual(parsed[0]["commit"], commit)
        self.assertEqual(parsed[0]["status"], "recorded")

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
