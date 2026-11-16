"""CPU-only contracts for ABI-v5 FullMlpPartition cross-frame dispatch."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import types
import unittest
from unittest import mock

from tests import test_tacker_phase2 as phase2_contracts
from tests.test_tacker_pipeline import (
    FakeEvent,
    FakeStream,
    FakeTensor,
    GaussianRenderState,
    Linear,
    RenderResult,
    Sequential,
    _load_module,
)


def _resources():
    return {
        "block_threads": 384,
        "registers_per_thread": 73,
        "static_shared_memory_bytes": 14592,
        "max_threads_per_block": 1024,
        "active_blocks_per_sm": 1,
        "occupancy": 0.25,
    }


def _full_profile(module, persistent_blocks=0, overlap=False, graph=False):
    profile = module.load_tacker_profile()
    if overlap and graph:
        raise ValueError("overlap and graph test modes are mutually exclusive")
    if graph:
        builder = module.full_mlp_overlap_graph_candidate_contract
    elif overlap:
        builder = module.full_mlp_overlap_candidate_contract
    else:
        builder = module.full_mlp_candidate_contract
    candidate = builder(
        "full_mlp_pb{}".format(persistent_blocks),
        module.HEAD_ORDER,
        worker_groups=1,
        persistent_blocks=persistent_blocks,
        resources=_resources(),
    )
    profile["candidates"].append(candidate)
    profile["selected_variant_id"] = candidate["variant_id"]
    profile["manifest"]["persistent_blocks"] = persistent_blocks
    profile["manifest_sha256"] = module.manifest_sha256(profile["manifest"])
    profile["selection"] = None
    profile["deployment"] = {"enabled": False, "valid": False}
    profile["profile_sha256"] = module.profile_sha256(profile)
    return profile, candidate


class TransposableTensor(FakeTensor):
    def to(self, dtype=None, **_kwargs):
        return TransposableTensor(
            self.value, self.shape, dtype or self.dtype, self.name
        )

    def t(self):
        return FakeTensor(
            self.value, tuple(reversed(self.shape)), self.dtype, self.name
        )


class Grid:
    def __init__(self, log):
        self.log = log
        self.feat_dim = 32

    def __call__(self, points, timestamp):
        self.log.append(("grid", points.shape, timestamp.shape))
        return FakeTensor(9.0, (points.shape[0], 32), "float32", "features")


def _install_tensor_packing(module):
    def stack(values, dim=0):
        if dim != 0:
            raise AssertionError("test packing only supports dim=0")
        return FakeTensor(
            0.0,
            (len(values),) + tuple(values[0].shape),
            values[0].dtype,
            "stacked",
        )

    def cat(values, dim=0):
        if dim != 0:
            raise AssertionError("test packing only supports dim=0")
        return FakeTensor(
            0.0,
            (sum(value.shape[0] for value in values),)
            + tuple(values[0].shape[1:]),
            values[0].dtype,
            "concatenated",
        )

    module._test_torch.stack = stack
    module._test_torch.cat = cat


def _model_and_context(module, log):
    helper = phase2_contracts.GenericTaskTest()
    pc, context, network = helper._model_and_context(
        module, tuple(module.HEAD_ORDER), log
    )
    backbone = Linear(
        32,
        128,
        value_fn=lambda _value: (_ for _ in ()).throw(
            AssertionError("selected backbone ran in Python")
        ),
        log=log,
        label="backbone",
    )
    backbone.weight = TransposableTensor(0.0, (128, 32), "float32")
    network.feature_out = Sequential(backbone)
    network.grid = Grid(log)
    network.no_grid = False
    network.grid_pe = 0
    network.W = 128
    network.D = 0
    pc._deformation.training = False
    pc.get_xyz = FakeTensor(shape=(111525, 3))
    context.means2D = FakeTensor(shape=(2, 3))
    context.colors_precomp = None
    context.cov3D_precomp = None
    context.screenspace_points = FakeTensor(shape=(2, 3))
    return pc, context, network, backbone


class CandidateContractTest(unittest.TestCase):
    def test_contract_registry_and_frozen_manifests(self):
        module = _load_module()
        profile, candidate = _full_profile(module)
        self.assertIs(module.validate_tacker_profile(profile), profile)
        partition = module.resolve_fusion_partition(candidate)
        self.assertIsInstance(partition, module.FullMlpPartition)
        self.assertEqual(partition.variant.abi_version, 5)
        self.assertEqual(partition.variant.worker_groups, 1)
        self.assertEqual(partition.variant.selected_heads, module.HEAD_ORDER)
        self.assertEqual(candidate["physical_cta_threads"], 384)
        self.assertEqual(candidate["persistent_blocks_zero_policy"], "direct_logical_grid")

        root = Path(__file__).resolve().parents[1]
        mixed_path = (
            root
            / "submodules"
            / "depth-diff-gaussian-rasterization"
            / "abi"
            / "tacker_mixed_render_deform_mlp_v5.json"
        )
        deform_path = root / "tacker_ext" / "abi" / "deform_mlp_v1.json"
        self.assertEqual(
            hashlib.sha256(mixed_path.read_bytes()).hexdigest(),
            module.FULL_MLP_MIXED_ABI_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(deform_path.read_bytes()).hexdigest(),
            module.DEFORM_MLP_ABI_SHA256,
        )
        manifest = json.loads(mixed_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["backend_family"], module.FULL_MLP_ABI_FAMILY)
        self.assertEqual(manifest["physical_launch"]["threads"], 384)

    def test_contract_rejects_partial_heads_and_extra_worker_groups(self):
        module = _load_module()
        with self.assertRaisesRegex(ValueError, "all heads"):
            module.full_mlp_candidate_contract("partial", ("pos",))
        with self.assertRaisesRegex(ValueError, "worker_groups == 1"):
            module.full_mlp_candidate_contract(
                "workers", module.HEAD_ORDER, worker_groups=2
            )

    def test_overlap_contract_is_opt_in_and_keeps_physical_abi_v5(self):
        module = _load_module()
        profile, candidate = _full_profile(module, overlap=True)
        self.assertIs(module.validate_tacker_profile(profile), profile)
        partition = module.resolve_fusion_partition(candidate)
        self.assertIsInstance(partition, module.FullMlpOverlapPartition)
        self.assertTrue(partition.overlaps_raster_prepare)
        self.assertEqual(candidate["abi_family"], module.FULL_MLP_ABI_FAMILY)
        self.assertEqual(candidate["cuda_symbol"], "tacker_mix_render_deform_mlp_v5")
        self.assertEqual(candidate["partition"]["kind"], "full_mlp_overlap")
        self.assertTrue(
            candidate["capability_requirements"][
                "prepared_raster_deform_mlp_overlap"
            ]
        )
        _old_profile, old_candidate = _full_profile(module)
        self.assertIsInstance(
            module.resolve_fusion_partition(old_candidate),
            module.FullMlpPartition,
        )
        self.assertNotIsInstance(
            module.resolve_fusion_partition(old_candidate),
            module.FullMlpOverlapPartition,
        )

    def test_graph_overlap_contract_is_a_third_opt_in_schedule(self):
        module = _load_module()
        profile, candidate = _full_profile(module, graph=True)
        self.assertIs(module.validate_tacker_profile(profile), profile)
        partition = module.resolve_fusion_partition(candidate)
        self.assertIsInstance(
            partition, module.FullMlpOverlapGraphPartition
        )
        self.assertTrue(partition.overlaps_raster_prepare)
        self.assertTrue(partition.uses_hexplane_cuda_graph)
        self.assertEqual(
            candidate["partition"]["kind"], "full_mlp_overlap_graph"
        )
        self.assertEqual(
            candidate["prefix_submission"],
            {
                "backend": "torch.cuda.CUDAGraph",
                "capture_version": 1,
                "slot_count": 2,
                "static_input_copies": ["xyz", "timestamp"],
                "output_dtype": "float32",
                "output_width": 32,
            },
        )
        self.assertEqual(candidate["cuda_symbol"], "tacker_mix_render_deform_mlp_v5")
        _overlap_profile, overlap_candidate = _full_profile(
            module, overlap=True
        )
        self.assertIsInstance(
            module.resolve_fusion_partition(overlap_candidate),
            module.FullMlpOverlapPartition,
        )
        self.assertNotIsInstance(
            module.resolve_fusion_partition(overlap_candidate),
            module.FullMlpOverlapGraphPartition,
        )

    def test_v12_default_split_pairs_one_raster_cta_with_two_deform_tiles(self):
        module = _load_module()
        settings = types.SimpleNamespace(image_width=1352, image_height=1014)
        context = types.SimpleNamespace(
            rasterizer=types.SimpleNamespace(raster_settings=settings)
        )
        prepared_raster = types.SimpleNamespace(
            radii=types.SimpleNamespace(numel=lambda: 111525)
        )
        real_get = module.os.environ.get

        def without_split_override(name, default=None):
            if name in {
                "TACKER_MIXED_V12_SPLIT_TILE",
                "TACKER_MIXED_V12_SPLIT_FRACTION",
            }:
                return None
            return real_get(name, default)

        with mock.patch.object(
            module.os.environ,
            "get",
            side_effect=without_split_override,
        ):
            split_tile = module._split_raster_v12_tile(
                context,
                prepared_raster,
                deform_rows=111525,
            )

        raster_blocks = 85 * 64
        deform_tiles = (111525 + 15) // 16
        self.assertEqual(deform_tiles, 6971)
        self.assertEqual(split_tile, 1954)
        self.assertEqual(raster_blocks - split_tile, 3486)
        self.assertEqual(2 * (raster_blocks - split_tile), deform_tiles + 1)

    def test_deform_then_raster_profile_schedule_is_an_opt_in_boundary(self):
        module = _load_module()
        log = []
        renderer = object.__new__(module.TackerRenderer)
        slot = types.SimpleNamespace(
            split_profile_prefix_end=FakeEvent("deform_end", log)
        )
        schedule = {"TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE": "both"}
        with mock.patch.dict(module.os.environ, schedule, clear=False):
            renderer._isolate_profile_deform_side(slot)
        self.assertEqual(log, [])

        schedule["TACKER_PROFILE_MIXED_PIPELINE_SCHEDULE"] = (
            "deform_then_raster"
        )
        with mock.patch.dict(
            module.os.environ, schedule, clear=False
        ), mock.patch.object(
            module,
            "_mixed_pipeline_profile_path",
            return_value="timeline.jsonl",
        ):
            renderer._isolate_profile_deform_side(slot)
        self.assertEqual(log, [("synchronize_event", "deform_end")])


class PartitionExecutionTest(unittest.TestCase):
    def test_cache_feature_prefix_and_finish_never_run_python_mlp(self):
        module = _load_module(available=True, grad_enabled=False)
        _install_tensor_packing(module)
        log = []
        pc, context, network, backbone = _model_and_context(module, log)
        _profile, candidate = _full_profile(module)
        partition = module.resolve_fusion_partition(candidate)
        cached = partition.cache_parameters(pc)
        self.assertEqual(cached[0].shape, (32, 128))
        self.assertEqual(cached[1].shape, (128,))
        self.assertEqual(cached[2].shape, (5, 128, 128))
        self.assertEqual(cached[3].shape, (5, 128))
        self.assertEqual(cached[4].shape, (59, 128))
        self.assertEqual(cached[5].shape, (59,))

        stream = module._test_torch.cuda.Stream(device="cuda:0")
        task = partition.prepare(
            context, pc, stream, cached, FakeEvent("prefix", log)
        )
        self.assertEqual(task.deform_features.shape, (2, 32))
        self.assertEqual(backbone.calls, 0)
        self.assertIn(("event", "prefix", stream.name), log)

        outputs = tuple(
            FakeTensor(20.0 + index, (2, width), "float32")
            for index, width in enumerate(task.output_widths)
        )
        state = partition.finish(task, pc, outputs)
        self.assertEqual(state.means3D.value, 22.0)
        self.assertEqual(state.scales.value, 24.0)
        self.assertEqual(backbone.calls, 0)
        for head_name in module.HEAD_ORDER:
            head = getattr(network, module.HEAD_MODULES[head_name])
            self.assertEqual(head[1].calls, 0)
            self.assertEqual(head[3].calls, 0)
        self.assertEqual(tuple(task.full_mlp_outputs), module.HEAD_ORDER)
        wrong_rows = list(outputs)
        wrong_rows[0] = FakeTensor(0.0, (3, 3), "float32")
        with self.assertRaisesRegex(ValueError, "row count"):
            partition.finish(task, pc, tuple(wrong_rows))

    def test_public_binding_receives_packed_full_mlp_operands(self):
        module = _load_module()
        captured = {}

        class Rasterizer:
            def forward_with_deform_mlp(self, **kwargs):
                captured.update(kwargs)
                outputs = tuple("delta{}".format(index) for index in range(5))
                return "image", FakeTensor(1.0), "depth", outputs

        context = types.SimpleNamespace(
            rasterizer=Rasterizer(),
            means2D="means2d",
            colors_precomp=None,
            cov3D_precomp=None,
            screenspace_points=FakeTensor(),
        )
        state = GaussianRenderState("means", "scales", "rot", "opacity", "shs")
        task = types.SimpleNamespace(
            deform_features="features",
            backbone_weight="backbone_weight",
            backbone_bias="backbone_bias",
            full_first_weights="first_weights",
            full_first_biases="first_biases",
            full_tail_weights="tail_weights",
            full_tail_biases="tail_biases",
        )
        mixed = module._forward_with_deform_mlp(context, state, task, 0)
        result, outputs = module._result_from_mixed_heads(context, mixed, 5)
        self.assertEqual(result.render, "image")
        self.assertEqual(len(outputs), 5)
        self.assertEqual(captured["deform_features"], "features")
        self.assertEqual(captured["tail_weights"], "tail_weights")
        self.assertEqual(captured["persistent_blocks"], 0)


class RuntimeAndSchedulerTest(unittest.TestCase):
    @staticmethod
    def _capabilities(module, candidate):
        result = deepcopy(module._test_capabilities)
        result.update(
            {
                "resource_query_family_aware": True,
                "mixed_render_deform_mlp_abi": 5,
                "mixed_render_deform_mlp": True,
                "mixed_deform_mlp_family": module.FULL_MLP_ABI_FAMILY,
                "mixed_deform_mlp_symbol": candidate["cuda_symbol"],
                "mixed_deform_mlp_manifest": candidate["abi_manifest"],
                "mixed_deform_mlp_manifest_sha256": candidate[
                    "abi_manifest_sha256"
                ],
                "deform_mlp_manifest_sha256": candidate[
                    "deform_mlp_abi_manifest_sha256"
                ],
                "deform_mlp_threads": 128,
                "deform_mlp_named_barrier_id": 2,
                "deform_mlp_direct_grid_default": True,
                "supported_backend_families": [
                    module.FIRST_LINEAR_ABI_FAMILY,
                    module.PACKED_FIRST_LINEAR_ABI_FAMILY,
                    module.WHOLE_HEAD_ABI_FAMILY,
                    module.FULL_MLP_ABI_FAMILY,
                ],
                "supported_mixed_abis": [1, 2, 3, 4, 5],
            }
        )
        return result

    @staticmethod
    def _runtime_resources(module):
        result = module._test_resource_requirements(5, 1)
        result.update(
            {
                "backend_abi_version": 5,
                "backend_family": module.FULL_MLP_ABI_FAMILY,
                "block_threads": 384,
                "registers_per_thread": 73,
                "static_shared_bytes": 14592,
                "static_shared_memory_bytes": 14592,
                "max_threads_per_block": 1024,
                "active_blocks_per_sm": 1,
                "occupancy": 0.25,
            }
        )
        return result

    def test_runtime_capability_resource_and_model_contracts_fail_closed(self):
        module = _load_module(available=True, grad_enabled=False)
        _install_tensor_packing(module)
        pc, _context, network, _backbone = _model_and_context(module, [])
        pipe = types.SimpleNamespace(
            debug=False, compute_cov3D_python=False, convert_SHs_python=False
        )
        profile, candidate = _full_profile(module)
        capabilities = self._capabilities(module, candidate)
        resources = self._runtime_resources(module)
        module.GaussianRasterizer.forward_with_deform_mlp = lambda self: None
        common = dict(
            stage="fine",
            cam_type="dynerf",
            workload_name="flame_steak",
            iteration=14000,
            qualification_mode=True,
        )
        with mock.patch.object(
            module, "_query_rasterizer_capabilities", return_value=capabilities
        ), mock.patch.object(
            module, "_query_rasterizer_variant_resources", return_value=resources
        ):
            self.assertIsNone(
                module.tacker_support_reason(pc, pipe, profile, **common)
            )
            network.grid_pe = 2
            self.assertIn(
                "grid_pe == 0",
                module.tacker_support_reason(pc, pipe, profile, **common),
            )
            network.grid_pe = 0
            bad = deepcopy(capabilities)
            bad["mixed_deform_mlp_manifest_sha256"] = "f" * 64
            with mock.patch.object(
                module, "_query_rasterizer_capabilities", return_value=bad
            ):
                self.assertIn(
                    "mixed_deform_mlp_manifest_sha256",
                    module.tacker_support_reason(pc, pipe, profile, **common),
                )

    def test_two_slot_counts_cover_one_two_and_many_frames(self):
        module = _load_module(available=True, grad_enabled=False)
        _install_tensor_packing(module)
        pc, _context, _network, _backbone = _model_and_context(module, [])
        pipe = types.SimpleNamespace(
            debug=False, compute_cov3D_python=False, convert_SHs_python=False
        )
        profile, _candidate = _full_profile(module)
        with mock.patch.object(
            module, "tacker_support_reason", return_value=None
        ), mock.patch.object(module, "_view_profile_reason", return_value=None):
            for frame_count in (1, 2, 5):
                renderer = module.TackerRenderer(
                    pc,
                    pipe,
                    FakeTensor(),
                    cam_type="dynerf",
                    profile_override=profile,
                    workload_name="flame_steak",
                    iteration=14000,
                    qualification_mode=True,
                )

                def result():
                    value = FakeTensor()
                    return RenderResult(value, value, value, value, value)

                renderer._enqueue_full_deformation = lambda _view, _slot: None
                renderer._enqueue_prefix = lambda _view, _slot: None
                renderer._enqueue_mixed = lambda _current, _next: result()
                renderer._enqueue_suffix = lambda _slot: None
                renderer._enqueue_solo_raster = lambda _slot: result()
                outputs = list(
                    renderer.render_sequence(
                        iter(str(index) for index in range(frame_count))
                    )
                )
                counts = renderer.last_execution_counts
                self.assertEqual(len(outputs), frame_count)
                self.assertEqual(counts["full_deformation"], 1)
                self.assertEqual(counts["prefix"], frame_count - 1)
                self.assertEqual(counts["mixed_launches"], frame_count - 1)
                self.assertEqual(counts["suffix"], frame_count - 1)
                self.assertEqual(counts["solo_raster"], 1)
                self.assertEqual(
                    counts["selected_head_evaluations_per_head"], frame_count
                )

    def test_overlap_waits_for_deform_prefix_only_after_raster_prepare(self):
        module = _load_module(available=True, grad_enabled=False)
        log = module._test_log

        class Task:
            def __init__(self):
                self.mixed_outputs = None

            def record_stream(self, stream):
                log.append(("task_record", stream.name))

        class Partition:
            def prepare_raster(self, _context, _state):
                log.append(("prepare_raster", module._test_torch.cuda.current.name))
                return (FakeTensor(name="prepared"),)

            def split_raster_tile(self, _context, _prepared, _task):
                return None

            def launch_prepared_mixed(
                self, _context, _prepared, _task, raster_start=None
            ):
                assert raster_start is None
                log.append(("launch_mixed", module._test_torch.cuda.current.name))
                return "mixed"

            def result_from_mixed(self, _context, _mixed):
                value = FakeTensor()
                outputs = tuple(FakeTensor() for _ in module.HEAD_ORDER)
                return RenderResult(value, value, value, value, value), outputs

        renderer = object.__new__(module.TackerRenderer)
        renderer.raster_stream = FakeStream("raster", 10, log)
        renderer.raster_prefix_stream = renderer.raster_stream
        renderer._green_contexts = None
        renderer.partition = Partition()
        current = module._TackerSlot(
            ready=FakeEvent("current_ready", log),
            prefix_ready=FakeEvent("unused_prefix", log),
            mixed_done=FakeEvent("unused_mixed", log),
            raster_done=FakeEvent("raster_done", log),
            context=types.SimpleNamespace(),
            state=GaussianRenderState(
                FakeTensor(), FakeTensor(), FakeTensor(), FakeTensor(), FakeTensor()
            ),
        )
        following = module._TackerSlot(
            ready=FakeEvent("unused_ready", log),
            prefix_ready=FakeEvent("next_prefix", log),
            mixed_done=FakeEvent("mixed_done", log),
            raster_done=FakeEvent("unused_raster", log),
            task=Task(),
        )

        renderer._enqueue_raster_prepare(current)
        renderer._enqueue_prepared_mixed(current, following)

        wait_current = log.index(("wait_event", "raster", "current_ready"))
        prepare = log.index(("prepare_raster", "raster"))
        wait_prefix = log.index(("wait_event", "raster", "next_prefix"))
        launch = log.index(("launch_mixed", "raster"))
        self.assertLess(wait_current, prepare)
        self.assertLess(prepare, wait_prefix)
        self.assertLess(wait_prefix, launch)

    def test_graph_prefix_capture_is_owned_and_reused_by_each_slot(self):
        module = _load_module(available=True, grad_enabled=False)
        log = module._test_log

        class Task:
            def record_stream(self, stream):
                log.append(("task_record", stream.name))

        class Partition:
            uses_hexplane_cuda_graph = True

            def create_prefix_graph(self, context, pc, stream):
                del context, pc
                graph = object()
                log.append(("create_graph", stream.name, graph))
                return graph

            def prepare_graph(
                self,
                context,
                pc,
                stream,
                cached,
                graph,
                prefix_ready,
            ):
                del context, pc, cached
                log.append(
                    (
                        "prepare_graph",
                        stream.name,
                        graph,
                        prefix_ready.name,
                    )
                )
                return Task()

        renderer = object.__new__(module.TackerRenderer)
        renderer.deform_stream = FakeStream("deform", 11, log)
        renderer.deform_prefix_stream = renderer.deform_stream
        renderer.partition = Partition()
        renderer.pc = object()
        renderer.cached_head_parameters = object()
        renderer._wait_slot_reuse = lambda _slot, _stream=None: None
        context = types.SimpleNamespace()
        renderer._prepare_context = lambda _view: context
        slot = module._TackerSlot(
            ready=FakeEvent("ready", log),
            prefix_ready=FakeEvent("prefix", log),
            mixed_done=FakeEvent("mixed", log),
            raster_done=FakeEvent("raster", log),
        )

        renderer._enqueue_prefix("view0", slot)
        first_graph = slot.hexplane_graph
        renderer._enqueue_prefix("view1", slot)

        self.assertIs(slot.hexplane_graph, first_graph)
        self.assertEqual(
            len([entry for entry in log if entry[0] == "create_graph"]),
            1,
        )
        prepares = [entry for entry in log if entry[0] == "prepare_graph"]
        self.assertEqual(len(prepares), 2)
        self.assertIs(prepares[0][2], first_graph)
        self.assertIs(prepares[1][2], first_graph)


class RasterHexPlaneExperimentTest(unittest.TestCase):
    def test_mixed_plane_count_defaults_to_all_and_is_bounded(self):
        module = _load_module()
        with mock.patch.dict(module.os.environ, {}, clear=True):
            self.assertEqual(module._raster_hexplane_mixed_planes(), 12)
        for value in ("1", "3", "6", "9", "12"):
            with mock.patch.dict(module.os.environ, {
                "TACKER_RASTER_HEXPLANE_MIXED_PLANES": value,
            }, clear=True):
                self.assertEqual(module._raster_hexplane_mixed_planes(), int(value))
        for value in ("0", "13", "abc", "6.5"):
            with mock.patch.dict(module.os.environ, {
                "TACKER_RASTER_HEXPLANE_MIXED_PLANES": value,
            }, clear=True), self.assertRaises(ValueError):
                module._raster_hexplane_mixed_planes()

    def test_mode_is_opt_in_and_rejects_invalid_ratios(self):
        module = _load_module()
        with mock.patch.dict(module.os.environ, {}, clear=True):
            self.assertEqual(module._raster_hexplane_experiment(), (None, 17))
        for mode, ratio in (("unknown", "17"), ("fused", "0"),
                            ("separate", "2147483648"), ("fused", "abc")):
            with mock.patch.dict(module.os.environ, {
                "TACKER_EXPERIMENTAL_RASTER_HEXPLANE": mode,
                "TACKER_RASTER_HEXPLANE_TASKS_PER_CTA": ratio,
            }, clear=True), self.assertRaises(ValueError):
                module._raster_hexplane_experiment()

    def test_batched_and_block_fused_require_complete_single_task_domains(self):
        module = _load_module()
        for mode in ("batched", "block_fused"):
            with mock.patch.dict(module.os.environ, {
                "TACKER_EXPERIMENTAL_RASTER_HEXPLANE": mode,
            }, clear=True):
                self.assertEqual(module._raster_hexplane_experiment(), (mode, 1))
            for extra in ({"TACKER_RASTER_HEXPLANE_TASKS_PER_CTA": "17"},
                          {"TACKER_RASTER_HEXPLANE_MIXED_PLANES": "6"}):
                with mock.patch.dict(module.os.environ, {
                    "TACKER_EXPERIMENTAL_RASTER_HEXPLANE": mode, **extra,
                }, clear=True), self.assertRaises(ValueError):
                    module._raster_hexplane_experiment()

    def test_deferred_prefix_does_not_publish_unsampled_features(self):
        module = _load_module(available=True, grad_enabled=False)
        _install_tensor_packing(module)
        pc, context, _network, _backbone = _model_and_context(module, [])
        _profile, candidate = _full_profile(module, graph=True)
        partition = module.resolve_fusion_partition(candidate)
        cached = partition.cache_parameters(pc)
        stream = FakeStream("deform", 11, module._test_log)
        prefix_ready = FakeEvent("prefix_ready", module._test_log)
        graph = types.SimpleNamespace(
            deferred_sampling=True,
            enqueue=lambda *_args: FakeTensor(0, (2, 32)),
        )
        task = module.prepare_full_mlp_graph_task(
            context, pc, stream, partition.variant, cached, graph, prefix_ready,
        )
        self.assertIs(task.prefix_ready, prefix_ready)
        self.assertNotIn(("event", "prefix_ready", "deform"), module._test_log)
        graph.deferred_sampling = False
        module.prepare_full_mlp_graph_task(
            context, pc, stream, partition.variant, cached, graph, prefix_ready,
        )
        self.assertIn(("event", "prefix_ready", "deform"), module._test_log)

    def test_fused_tiles_are_excluded_from_remaining_prepass(self):
        module = _load_module(available=True, grad_enabled=False)
        log = module._test_log

        class Graph:
            deferred_sampling = True
            mode = "fused"

            def complete_sampling(self, _context, _prepared, split_tile,
                                  _raster_stream, deform_stream, prefix_ready,
                                  _prefix_end):
                assert split_tile == 1954
                log.append(("fused_prefix", 0, 308))
                prefix_ready.record(deform_stream)
                return 308

        class Partition:
            def split_raster_tile(self, *_args):
                return 1954

            def launch_prepared_mixed(self, *_args, raster_start=None):
                log.append(("mlp_mixed", raster_start))
                return "mixed"

            def result_from_mixed(self, *_args):
                value = FakeTensor()
                return RenderResult(value, value, value, value, value), (value,)

        renderer = object.__new__(module.TackerRenderer)
        renderer.raster_stream = FakeStream("raster", 10, log)
        renderer.deform_stream = FakeStream("deform", 11, log)
        renderer._green_contexts = None
        renderer.partition = Partition()
        renderer._last_execution_counts = {"raster_prepasses": 0}
        rasterizer = types.SimpleNamespace(render_prepared_raster_fast_range=
            lambda _prepared, start, end: log.append(("remaining_prefix", start, end)))
        current = module._TackerSlot(
            FakeEvent("ready", log), FakeEvent("unused", log),
            FakeEvent("unused_mixed", log), FakeEvent("raster_done", log),
            context=types.SimpleNamespace(rasterizer=rasterizer),
            prepared_raster=(FakeTensor(),),
        )
        following = module._TackerSlot(
            FakeEvent("next_ready", log), FakeEvent("prefix_ready", log),
            FakeEvent("mixed_done", log), FakeEvent("next_raster", log),
            hexplane_graph=Graph(), task=types.SimpleNamespace(
                record_stream=lambda _stream: None, mixed_outputs=None),
        )
        renderer._enqueue_prepared_mixed(current, following)
        self.assertIn(("remaining_prefix", 308, 1954), log)
        self.assertIn(("mlp_mixed", 1954), log)
        self.assertLess(log.index(("event", "prefix_ready", "deform")),
                        log.index(("wait_event", "raster", "prefix_ready")))
        self.assertEqual(renderer._last_execution_counts["raster_hexplane_fused_launches"], 1)
        self.assertEqual(renderer._last_execution_counts["raster_hexplane_sampling_graph_replays"], 0)

    def test_unqualified_experiment_is_rejected_before_stream_creation(self):
        module = _load_module()
        renderer = object.__new__(module.TackerRenderer)
        renderer.raster_stream = None
        renderer._closed = False
        renderer._green_config = None
        renderer.qualification_mode = False
        with mock.patch.dict(module.os.environ, {
            "TACKER_EXPERIMENTAL_RASTER_HEXPLANE": "fused",
        }, clear=True), self.assertRaisesRegex(RuntimeError, "qualification mode"):
            renderer._ensure_cuda_resources()


if __name__ == "__main__":
    unittest.main()
