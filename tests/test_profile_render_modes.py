"""CPU-only source contracts for render profiling mode integration."""

import ast
import importlib.util
import os
from pathlib import Path
import statistics
import tempfile
import unittest
import sys
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def _load_profile_render_helpers():
    spec = importlib.util.spec_from_file_location(
        "profile_render_cpu_contract", ROOT / "profile_render.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bootstrap_helpers():
    spec = importlib.util.spec_from_file_location(
        "profile_render_bootstrap_cpu_contract",
        ROOT / "scripts" / "run_profile_render_sealed.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            '"cuda_peak_allocated_bytes"',
            '"cuda_peak_reserved_bytes"',
            '"max_cuda_peak_allocated_bytes_across_trials"',
            '"max_cuda_peak_reserved_bytes_across_trials"',
            '"pre_import": pre_import_snapshot',
            'metadata["post_run"]',
            '"byte_stability": post_run_byte_stability',
            '"profile_resolution_argument"',
            '"profile_resolution_scale"',
            '"original_resolution"',
            '"effective_resolution"',
            '"profile_resolution_contract"',
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
        self.assertIn("tacker_profile_snapshot_error = None", self.source)
        self.assertIn(
            '"sealed pre-import snapshot".format(tacker_record["path"])',
            self.source,
        )
        self.assertIn("profile_snapshot_error=(", self.source)

    def test_production_cli_requires_stable_fd_bootstrap(self):
        profiler = _load_profile_render_helpers()
        with self.assertRaisesRegex(
            profiler.ProvenanceError, "run_profile_render_sealed.py"
        ):
            profiler._cli_main([])

        bootstrap_source = (
            ROOT / "scripts" / "run_profile_render_sealed.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(
            bootstrap_source,
            "run_profile_render_sealed.py",
            feature_version=7,
        )
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(imported, {"hashlib", "os", "stat", "sys"})
        self.assertIn("code = compile(raw, source_path", bootstrap_source)
        self.assertIn('"_PROFILE_RENDER_BOOTSTRAP"', bootstrap_source)

    def test_bootstrap_execution_bytes_are_consumed_and_reverified(self):
        bootstrap = _load_bootstrap_helpers()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "profile_render.py"
            source_path.write_bytes((ROOT / "profile_render.py").read_bytes())
            record, raw = bootstrap._read_stable_source(str(source_path))
            spec = importlib.util.spec_from_file_location(
                "sealed_profile_render_contract", source_path
            )
            profiler = importlib.util.module_from_spec(spec)
            profiler._PROFILE_RENDER_BOOTSTRAP = {
                "protocol": 1,
                "record": record,
                "bytes": raw,
                "execution": {
                    "compiled_sha256": record["sha256"],
                    "compile_mode": "exec",
                    "dont_inherit": True,
                },
            }
            spec.loader.exec_module(profiler)
            consumed, consumed_raw, execution = (
                profiler._consume_profile_render_bootstrap()
            )
            self.assertEqual(consumed_raw, raw)
            self.assertEqual(consumed["sha256"], record["sha256"])
            self.assertEqual(
                execution["compiled_sha256"], record["sha256"]
            )

            source_path.write_bytes(raw + b"\n# swapped\n")
            with self.assertRaisesRegex(
                profiler.ProvenanceError, "changed after bootstrap"
            ):
                profiler._consume_profile_render_bootstrap()

    def test_symlinks_and_atomic_replacement_fail_the_file_seal(self):
        profiler = _load_profile_render_helpers()
        bootstrap = _load_bootstrap_helpers()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_bytes(b"value = 1\n")
            link = root / "link.py"
            link.symlink_to(source)
            with self.assertRaisesRegex(
                profiler.ProvenanceError, "symbolic link"
            ):
                profiler._snapshot_file_bytes(link, "source.test")
            with self.assertRaisesRegex(RuntimeError, "symbolic link"):
                bootstrap._read_stable_source(str(link))

            replacement = root / "replacement.py"
            replacement.write_bytes(b"value = 2\n")
            real_read = os.read
            replaced = []

            def replacing_read(descriptor, size):
                if not replaced:
                    os.replace(str(replacement), str(source))
                    replaced.append(True)
                return real_read(descriptor, size)

            with mock.patch.object(
                profiler.os, "read", side_effect=replacing_read
            ):
                with self.assertRaisesRegex(
                    profiler.ProvenanceError, "changed while being snapshotted"
                ):
                    profiler._snapshot_file_bytes(source, "source.test")

    def test_snapshot_and_internal_are_an_indivisible_pair(self):
        profiler = _load_profile_render_helpers()
        args = SimpleNamespace()
        with self.assertRaisesRegex(
            profiler.ProvenanceError, "must both be set or both be None"
        ):
            profiler.main(
                args,
                None,
                None,
                None,
                pre_import_snapshot={},
                pre_import_internal=None,
            )

    def test_capture_seals_raster_head_and_simple_knn_artifacts(self):
        profiler = _load_profile_render_helpers()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("value = 1\n", encoding="utf-8")
            packages = {}
            for component, package, wrapper_exists in (
                ("rasterizer", "diff_gaussian_rasterization", True),
                ("head", "tacker_4dgs_head", True),
                ("simple_knn", "simple_knn", False),
            ):
                package_root = root / package
                package_root.mkdir()
                wrapper = package_root / "__init__.py"
                if wrapper_exists:
                    wrapper.write_text("value = 1\n", encoding="utf-8")
                binary = package_root / "_C.test.so"
                binary.write_bytes(component.encode("ascii"))
                packages[component] = {
                    "package": package,
                    "wrapper": wrapper,
                    "wrapper_exists": wrapper_exists,
                    "binary": binary,
                    "wrapper_role": "{}.wrapper".format(component),
                    "binary_role": "{}.binary".format(component),
                }
            args = profiler.Namespace(
                configs=None,
                model_path=None,
                qualification_profile=None,
                tacker_profile=None,
            )
            snapshot, internal = profiler._capture_pre_import_snapshot(
                args,
                source_paths={"source.test": source},
                rasterizer_paths=(
                    packages["rasterizer"]["wrapper"],
                    packages["rasterizer"]["binary"],
                ),
                binary_package_paths=packages,
            )
            roles = {record["role"] for record in snapshot["files"]}
            for component in packages:
                self.assertIn("{}.wrapper".format(component), roles)
                self.assertIn("{}.binary".format(component), roles)
                self.assertIn(component, internal["binary_packages"])
            self.assertTrue(
                internal["python_modules"]["simple_knn"]["synthetic"]
            )
            self.assertTrue(
                profiler._verify_pre_import_snapshot(snapshot)["verified"]
            )

    def test_first_party_source_enumeration_covers_transitive_packages(self):
        profiler = _load_profile_render_helpers()
        modules = {}
        for package_name in ("arguments", "gaussian_renderer", "scene", "utils"):
            modules.update(
                profiler._python_modules_under(
                    package_name, ROOT / package_name
                )
            )
        for required in (
            "arguments",
            "gaussian_renderer",
            "gaussian_renderer.tacker_pipeline",
            "scene",
            "scene.gaussian_model",
            "scene.deformation",
            "utils.general_utils",
            "utils.profiling_utils",
            "utils.sh_utils",
        ):
            self.assertIn(required, modules)
        expected_count = sum(
            1
            for package_name in ("arguments", "gaussian_renderer", "scene", "utils")
            for _path in (ROOT / package_name).rglob("*.py")
        )
        self.assertEqual(len(modules), expected_count)

    def test_snapshot_pair_rejects_internal_byte_tampering(self):
        profiler = _load_profile_render_helpers()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.py"
            source.write_bytes(b"value = 1\n")
            record, raw = profiler._snapshot_file_bytes(
                source, "source.test"
            )
            snapshot = {"files": [record]}
            profiler._validate_snapshot_pair(
                snapshot, {"bytes_by_path": {record["path"]: raw}}
            )
            with self.assertRaisesRegex(
                profiler.ProvenanceError, "disagree with record"
            ):
                profiler._validate_snapshot_pair(
                    snapshot,
                    {"bytes_by_path": {record["path"]: raw + b"tamper"}},
                )

    def test_post_run_verifier_includes_loaded_private_binaries(self):
        profiler = _load_profile_render_helpers()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.so"
            path.write_bytes(b"sealed extension")
            record, _raw = profiler._snapshot_file_bytes(
                path, "runtime.head.binary"
            )
            snapshot = {
                "files": [],
                "binary_bindings": [
                    {
                        "component": "head",
                        "module": "tacker_4dgs_head._C",
                        "loaded_path": record["path"],
                        "loaded_stat": record["stat"],
                        "size_bytes": record["size_bytes"],
                        "sha256": record["sha256"],
                    }
                ],
            }
            verified = profiler._verify_pre_import_snapshot(snapshot)
            self.assertEqual(verified["loaded_binary_count"], 1)
            self.assertTrue(verified["loaded_binaries"][0]["unchanged"])
            path.write_bytes(b"replacement")
            with self.assertRaisesRegex(
                profiler.ProvenanceError,
                "loaded private head binary changed",
            ):
                profiler._verify_pre_import_snapshot(snapshot)

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
        exec(
            compile(
                ast.Module(body=functions, type_ignores=[]), "helpers", "exec"
            ),
            namespace,
        )

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
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        )
        first = {name: 10.0 for name in metric_names}
        second = {name: 20.0 for name in metric_names}
        aggregate = namespace["_aggregate_trials"]([first, second])
        for name in metric_names:
            self.assertEqual(aggregate["median_{}".format(name)], 15.0)
        self.assertEqual(
            aggregate["max_cuda_peak_allocated_bytes_across_trials"], 20.0
        )
        self.assertEqual(
            aggregate["max_cuda_peak_reserved_bytes_across_trials"], 20.0
        )

    def test_cli_seals_inputs_before_heavy_import_and_config_execution(self):
        cli_source = self.source[self.source.index("def _cli_main(") :]
        capture = cli_source.index(
            "_capture_pre_import_snapshot(bootstrap_args)"
        )
        activate = cli_source.index(
            "_activate_runtime_imports(pre_import_snapshot, pre_import_internal)"
        )
        config = cli_source.index("_load_config_snapshot(")
        self.assertLess(capture, activate)
        self.assertLess(activate, config)
        import_prefix = self.source[: self.source.index("def _percentile(")]
        self.assertNotIn("\nimport torch\n", import_prefix)
        self.assertNotIn("from gaussian_renderer import", import_prefix)

    def test_recursive_config_executes_snapshot_bytes_and_drift_fails_closed(self):
        profiler = _load_profile_render_helpers()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            wrapper = root / "__init__.py"
            binary = root / "_C.test.so"
            base = root / "base.py"
            config = root / "config.py"
            source.write_text("value = 1\n", encoding="utf-8")
            wrapper.write_text("value = 2\n", encoding="utf-8")
            binary.write_bytes(b"raster")
            base.write_text(
                "ModelParams = {'sh_degree': 3}\n", encoding="utf-8"
            )
            config.write_text(
                "_base_ = 'base.py'\n"
                "ModelParams = {'model_path': 'sealed'}\n",
                encoding="utf-8",
            )
            args = profiler.Namespace(
                configs=str(config),
                model_path=None,
                qualification_profile=None,
                tacker_profile=None,
            )
            pre_import, internal = profiler._capture_pre_import_snapshot(
                args,
                source_paths={"source.test": source},
                rasterizer_paths=(wrapper, binary),
            )
            base.write_text(
                "ModelParams = {'sh_degree': 9}\n", encoding="utf-8"
            )
            loaded = profiler._load_config_snapshot(
                config, pre_import, internal
            )
            self.assertEqual(loaded["ModelParams"]["sh_degree"], 3)
            self.assertEqual(loaded["ModelParams"]["model_path"], "sealed")
            with self.assertRaisesRegex(
                profiler.ProvenanceError, "changed during profiling"
            ):
                profiler._verify_pre_import_snapshot(pre_import)

    def test_missing_profile_absence_is_sealed_and_reverified(self):
        profiler = _load_profile_render_helpers()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            wrapper = root / "__init__.py"
            binary = root / "_C.test.so"
            missing = root / "missing.json"
            source.write_text("value = 1\n", encoding="utf-8")
            wrapper.write_text("value = 2\n", encoding="utf-8")
            binary.write_bytes(b"raster")
            args = profiler.Namespace(
                configs=None,
                model_path=None,
                qualification_profile=None,
                tacker_profile=str(missing),
            )
            pre_import, _internal = profiler._capture_pre_import_snapshot(
                args,
                source_paths={"source.test": source},
                rasterizer_paths=(wrapper, binary),
            )
            profile_record = next(
                record
                for record in pre_import["files"]
                if record["role"] == "profile.tacker"
            )
            self.assertFalse(profile_record["exists"])
            verified = profiler._verify_pre_import_snapshot(pre_import)
            self.assertTrue(verified["verified"])
            missing.write_text("{}\n", encoding="utf-8")
            real_open = os.open

            def reject_missing_open(path, *open_args):
                if os.path.abspath(str(path)) == os.path.abspath(str(missing)):
                    raise AssertionError(
                        "a path whose absence was sealed must not be opened"
                    )
                return real_open(path, *open_args)

            with mock.patch.object(
                profiler.os,
                "open",
                side_effect=reject_missing_open,
            ):
                with self.assertRaisesRegex(
                    profiler.ProvenanceError, "changed during profiling"
                ):
                    profiler._verify_pre_import_snapshot(pre_import)

    def test_sealed_source_loader_retains_original_file_location(self):
        profiler = _load_profile_render_helpers()
        module_name = "profile_render_sealed_source_contract"
        with tempfile.TemporaryDirectory() as directory:
            origin = Path(directory) / "sealed.py"
            raw = b"observed_file = __file__\n"
            finder = profiler._SnapshotImportFinder(
                {module_name: (str(origin), raw, False)},
                Path(directory) / "unused.so",
            )
            sys.meta_path.insert(0, finder)
            try:
                module = importlib.import_module(module_name)
                self.assertEqual(module.observed_file, str(origin))
                self.assertEqual(module.__file__, str(origin))
            finally:
                sys.modules.pop(module_name, None)
                sys.meta_path.remove(finder)

    def test_snapshot_finder_blocks_unsealed_first_party_lazy_imports(self):
        profiler = _load_profile_render_helpers()
        with tempfile.TemporaryDirectory() as directory:
            finder = profiler._SnapshotImportFinder(
                {
                    "sealed_family": (
                        str(Path(directory) / "__init__.py"),
                        b"",
                        True,
                    )
                },
                {},
            )
            with self.assertRaisesRegex(
                ImportError, "not present in pre-import snapshot"
            ):
                finder.find_spec("sealed_family.created_after_snapshot")

    def test_profile_resolution_scale_contract_and_rounding(self):
        profiler = _load_profile_render_helpers()
        self.assertEqual(profiler._profile_resolution_scale(-1), 1)
        for scale in (1, 2, 4, 8):
            self.assertEqual(profiler._profile_resolution_scale(scale), scale)
        self.assertEqual(
            profiler._scaled_profile_resolution(1352, 1014, 4),
            (338, 254),
        )
        with self.assertRaisesRegex(ValueError, "--resolution"):
            profiler._profile_resolution_scale(3)
        with self.assertRaisesRegex(ValueError, "--resolution"):
            profiler._profile_resolution_scale([2])

    def test_resolution_one_is_zero_copy_and_scaled_view_preserves_camera(self):
        profiler = _load_profile_render_helpers()
        geometry = {
            name: object()
            for name in (
                "FoVx",
                "FoVy",
                "world_view_transform",
                "projection_matrix",
                "full_proj_transform",
                "camera_center",
            )
        }

        class FakeImage:
            shape = (3, 1014, 1352)

            def unsqueeze(self, dimension):
                self.unsqueeze_dimension = dimension
                return self

            def squeeze(self, dimension):
                self.squeeze_dimension = dimension
                return self

        image = FakeImage()
        view = SimpleNamespace(
            original_image=image,
            image_width=1352,
            image_height=1014,
            **geometry
        )
        self.assertIs(profiler._scale_profile_view(view, 1), view)

        calls = []

        def interpolate(value, **kwargs):
            calls.append((value, kwargs))
            return value

        profiler.torch = SimpleNamespace(
            nn=SimpleNamespace(
                functional=SimpleNamespace(interpolate=interpolate)
            )
        )
        scaled = profiler._scale_profile_view(view, 4)
        self.assertIsNot(scaled, view)
        self.assertEqual((scaled.image_width, scaled.image_height), (338, 254))
        self.assertEqual(calls[0][1]["size"], (254, 338))
        self.assertEqual(calls[0][1]["mode"], "bilinear")
        for name, value in geometry.items():
            self.assertIs(getattr(scaled, name), value)

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
        exec(
            compile(
                ast.Module(body=functions, type_ignores=[]), "helpers", "exec"
            ),
            namespace,
        )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {
                    "FOURDGS_SOURCE_COMMIT": "a" * 40,
                    "FOURDGS_RASTERIZER_COMMIT": "b" * 40,
                },
                clear=True,
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
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]), "helper", "exec"
            ),
            namespace,
        )
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
