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


def _full_profile(module, persistent_blocks=0):
    profile = module.load_tacker_profile()
    candidate = module.full_mlp_candidate_contract(
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


if __name__ == "__main__":
    unittest.main()
