"""CPU-only contracts for Phase-3.1 C3/C4 production dispatch."""

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
    RenderResult,
    _exact_model,
    _load_module,
)


def _phase31_profile(
    module, builder, heads=("pos", "scales"), worker_groups=2,
    persistent_blocks=80, resources=None,
):
    if resources is None:
        resources = {
            "block_threads": 256 + 128 * worker_groups,
            "registers_per_thread": 32,
            "static_shared_memory_bytes": 0,
            "max_threads_per_block": 1024,
            "active_blocks_per_sm": 1,
            "occupancy": 0.5,
        }
    profile = module.load_tacker_profile()
    candidate = builder(
        "{}_wg{}".format(builder.__name__, worker_groups),
        heads,
        worker_groups=worker_groups,
        persistent_blocks=persistent_blocks,
        resources=resources,
    )
    profile["candidates"].append(candidate)
    profile["selected_variant_id"] = candidate["variant_id"]
    profile["manifest"]["persistent_blocks"] = persistent_blocks
    profile["manifest_sha256"] = module.manifest_sha256(profile["manifest"])
    profile["selection"] = None
    profile["deployment"] = {"enabled": False, "valid": False}
    profile["profile_sha256"] = module.profile_sha256(profile)
    return profile, candidate


class PackedOutputs(FakeTensor):
    def __init__(self, values, rows=2):
        super().__init__(0.0, (len(values), rows, 128), "float32", "packed")
        self.values = tuple(values)
        self.rows = rows

    def __getitem__(self, index):
        if isinstance(index, int):
            return FakeTensor(
                self.values[index], (self.rows, 128), "float32",
                "packed{}".format(index),
            )
        return super().__getitem__(index)


class CandidateContractTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def test_registry_builders_and_selected_variant_dispatch(self):
        cases = (
            (
                self.module.packed_first_linear_candidate_contract,
                self.module.PACKED_FIRST_LINEAR_PARTITION_KIND,
                self.module.PACKED_FIRST_LINEAR_ABI_FAMILY,
                self.module.PACKED_FIRST_LINEAR_BACKEND,
                self.module.PackedFirstLinearPartition,
                3,
            ),
            (
                self.module.whole_head_candidate_contract,
                self.module.WHOLE_HEAD_PARTITION_KIND,
                self.module.WHOLE_HEAD_ABI_FAMILY,
                self.module.WHOLE_HEAD_BACKEND,
                self.module.WholeHeadPartition,
                4,
            ),
        )
        for builder, kind, family, backend, partition_type, abi_version in cases:
            with self.subTest(kind=kind):
                profile, candidate = _phase31_profile(
                    self.module, builder, heads=("pos", "opacity"),
                    worker_groups=2,
                )
                self.assertIs(self.module.validate_tacker_profile(profile), profile)
                self.assertEqual(candidate, builder(
                    candidate["variant_id"],
                    ("pos", "opacity"),
                    worker_groups=2,
                    persistent_blocks=80,
                    resources=candidate["resources"],
                ))
                self.assertEqual(candidate["partition"]["kind"], kind)
                self.assertEqual(candidate["partition"]["backend"], backend)
                self.assertEqual(candidate["abi_family"], family)
                partition = self.module.resolve_fusion_partition(candidate)
                self.assertIsInstance(partition, partition_type)
                self.assertEqual(partition.variant.backend, backend)
                self.assertEqual(partition.variant.family, family)
                self.assertEqual(partition.variant.abi_version, abi_version)

    def test_frozen_phase31_manifests_match_runtime_constants(self):
        root = Path(__file__).resolve().parents[1]
        raster_abi = (
            root / "submodules" / "depth-diff-gaussian-rasterization" / "abi"
        )
        cases = (
            (
                self.module.PACKED_MIXED_ABI_MANIFEST,
                self.module.PACKED_MIXED_ABI_SHA256,
                self.module.PACKED_FIRST_LINEAR_ABI_FAMILY,
                "tacker_mix_render_packed_heads_v3",
            ),
            (
                self.module.WHOLE_HEAD_MIXED_ABI_MANIFEST,
                self.module.WHOLE_HEAD_MIXED_ABI_SHA256,
                self.module.WHOLE_HEAD_ABI_FAMILY,
                "tacker_mix_render_whole_heads_v4",
            ),
        )
        for relative_path, expected_hash, family, symbol in cases:
            with self.subTest(manifest=relative_path):
                path = raster_abi / Path(relative_path).name
                raw = path.read_bytes()
                manifest = json.loads(raw.decode("utf-8"))
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_hash)
                self.assertEqual(manifest["backend_family"], family)
                self.assertEqual(manifest["global_kernel_symbol"], symbol)
                self.assertEqual(
                    manifest["tacker_ext_dependency"]["manifest_sha256"],
                    self.module.HEAD_MULTI_ABI_SHA256,
                )

    def test_contract_tampering_fails_closed(self):
        cases = (
            ("packed", self.module.packed_first_linear_candidate_contract),
            ("whole", self.module.whole_head_candidate_contract),
        )
        mutations = (
            (lambda item: item.__setitem__("abi_family", "wrong"), "abi_family"),
            (
                lambda item: item["partition"].__setitem__("backend", "wrong"),
                "backend",
            ),
            (lambda item: item.__setitem__("cuda_symbol", "wrong"), "cuda_symbol"),
            (
                lambda item: item.__setitem__("abi_manifest_sha256", "f" * 64),
                "abi_manifest_sha256",
            ),
            (
                lambda item: item.__setitem__("stream_lifetimes", []),
                "stream_lifetimes",
            ),
        )
        for family, builder in cases:
            for mutate, message in mutations:
                with self.subTest(family=family, field=message):
                    profile, candidate = _phase31_profile(self.module, builder)
                    mutate(candidate)
                    profile["profile_sha256"] = self.module.profile_sha256(profile)
                    with self.assertRaisesRegex(
                        self.module.TackerProfileError, message
                    ):
                        self.module.validate_tacker_profile(profile)

        profile, candidate = _phase31_profile(
            self.module, self.module.whole_head_candidate_contract
        )
        candidate["tensor_contract"]["output_widths"] = [3, 4]
        profile["profile_sha256"] = self.module.profile_sha256(profile)
        with self.assertRaisesRegex(
            self.module.TackerProfileError, "tensor_contract"
        ):
            self.module.validate_tacker_profile(profile)

    def test_single_and_multi_head_width_and_barrier_contracts(self):
        for count in (1, 2, 5):
            heads = self.module.HEAD_ORDER[:count]
            worker_groups = min(2, count)
            packed = self.module.packed_first_linear_candidate_contract(
                "packed{}".format(count), heads, worker_groups=worker_groups
            )
            whole = self.module.whole_head_candidate_contract(
                "whole{}".format(count), heads, worker_groups=worker_groups
            )
            self.assertEqual(packed["backend_named_barriers"], [])
            self.assertEqual(
                whole["tensor_contract"]["output_widths"],
                [self.module.HEAD_OUTPUT_WIDTHS[name] for name in heads],
            )
            self.assertEqual(
                whole["backend_named_barriers"][-1],
                {
                    "id": 7,
                    "participants": 128 * worker_groups,
                    "purpose": "whole_head_descriptor_broadcast",
                },
            )


class PartitionExecutionTest(unittest.TestCase):
    @staticmethod
    def _fixture(module, selected_heads, log):
        helper = phase2_contracts.GenericTaskTest()
        return helper._model_and_context(module, selected_heads, log)

    @staticmethod
    def _install_stack(module, calls):
        def stack(values, dim=0):
            calls.append((tuple(values), dim))
            return FakeTensor(
                0.0, (len(values),) + tuple(values[0].shape),
                values[0].dtype, "stacked",
            )

        module._test_torch.stack = stack

    def test_c3_uses_one_shared_input_and_stacked_cache(self):
        module = _load_module(available=True, grad_enabled=False)
        selected = ("pos", "scales")
        log = []
        stack_calls = []
        self._install_stack(module, stack_calls)
        pc, context, network = self._fixture(module, selected, log)
        candidate = module.packed_first_linear_candidate_contract(
            "packed", selected, worker_groups=2
        )
        partition = module.resolve_fusion_partition(candidate)
        cached = partition.cache_parameters(pc)
        self.assertEqual(cached[0].shape, (2, 128, 128))
        self.assertEqual(cached[1].shape, (2, 128))
        self.assertEqual(len(stack_calls), 2)

        stream = module._test_torch.cuda.Stream(device="cuda:0")
        event = FakeEvent("prefix", log)
        with mock.patch.object(module, "_poc_fre", side_effect=lambda value, _p: value):
            task = partition.prepare(context, pc, stream, cached, event)
        self.assertIs(task.shared_head_input, task.head_input)
        self.assertEqual(log.count(("pos_prefix_relu", 10.0)), 1)
        self.assertNotIn(("scales_prefix_relu", 10.0), log)
        self.assertEqual(network.pos_deform[1].calls, 0)
        self.assertEqual(network.scales_deform[1].calls, 0)

        state = partition.finish(task, pc, PackedOutputs((70.0, 80.0)))
        self.assertEqual(network.pos_deform[1].calls, 0)
        self.assertEqual(network.scales_deform[1].calls, 0)
        self.assertEqual(network.pos_deform[3].calls, 1)
        self.assertEqual(network.scales_deform[3].calls, 1)
        self.assertEqual(state.means3D.value, 75.0)
        self.assertEqual(state.scales.value, 86.0)

    def test_c4_returns_exact_widths_without_python_tail_recompute(self):
        module = _load_module(available=True, grad_enabled=False)
        selected = tuple(module.HEAD_ORDER)
        log = []
        pc, context, network = self._fixture(module, selected, log)
        candidate = module.whole_head_candidate_contract(
            "whole", selected, worker_groups=2
        )
        partition = module.resolve_fusion_partition(candidate)
        cached = partition.cache_parameters(pc)
        stream = module._test_torch.cuda.Stream(device="cuda:0")
        with mock.patch.object(module, "_poc_fre", side_effect=lambda value, _p: value):
            task = partition.prepare(
                context, pc, stream, cached, FakeEvent("prefix", log)
            )
        self.assertTrue(
            all(value is task.shared_head_input for value in task.whole_head_inputs)
        )
        self.assertEqual(task.output_widths, (3, 3, 4, 1, 48))
        outputs = tuple(
            FakeTensor(20.0 + index, (2, width), "float32")
            for index, width in enumerate(task.output_widths)
        )
        partition.finish(task, pc, outputs)
        for head_name in selected:
            head = getattr(network, module.HEAD_MODULES[head_name])
            self.assertEqual(head[1].calls, 0)
            self.assertEqual(head[3].calls, 0)
        self.assertEqual(
            set(task.whole_head_outputs), set(module.HEAD_ORDER)
        )
        bad = list(outputs)
        bad[-1] = FakeTensor(0.0, (2, 47), "float32")
        with self.assertRaisesRegex(ValueError, r"\[N, 48\]"):
            partition.finish(task, pc, tuple(bad))

    def test_family_specific_public_binding_keywords(self):
        module = _load_module()
        captured = {}

        class Rasterizer:
            def forward_with_packed_heads(self, **kwargs):
                captured["packed"] = kwargs
                return "image", FakeTensor(1), "depth", "packed-output"

            def forward_with_whole_heads(self, **kwargs):
                captured["whole"] = kwargs
                return "image", FakeTensor(1), "depth", ("pos", "opacity")

        context = types.SimpleNamespace(
            rasterizer=Rasterizer(), means2D="means2d", colors_precomp=None,
            cov3D_precomp=None, screenspace_points=FakeTensor(),
        )
        state = GaussianRenderState("means", "scales", "rot", "opacity", "shs")
        packed_variant = module.fusion_variant_from_candidate(
            module.packed_first_linear_candidate_contract(
                "packed", ("pos", "opacity"), worker_groups=2,
                persistent_blocks=17,
            )
        )
        packed_task = types.SimpleNamespace(
            shared_head_input="shared", packed_head_weights="weights",
            packed_head_biases="biases", variant=packed_variant,
        )
        module._forward_with_packed_heads(context, state, packed_task, 17)
        self.assertEqual(captured["packed"]["head_input"], "shared")
        self.assertEqual(captured["packed"]["packed_head_weights"], "weights")

        whole_variant = module.fusion_variant_from_candidate(
            module.whole_head_candidate_contract(
                "whole", ("pos", "opacity"), worker_groups=2,
                persistent_blocks=19,
            )
        )
        whole_task = types.SimpleNamespace(
            whole_head_inputs=("shared", "shared"),
            whole_first_weights=("fw0", "fw1"),
            whole_first_biases=("fb0", "fb1"),
            whole_tail_weights=("tw0", "tw1"),
            whole_tail_biases=("tb0", "tb1"),
            output_widths=(3, 1), variant=whole_variant,
        )
        module._forward_with_whole_heads(context, state, whole_task, 19)
        self.assertEqual(captured["whole"]["head_inputs"], ("shared", "shared"))
        self.assertEqual(captured["whole"]["output_widths"], (3, 1))
        self.assertEqual(captured["whole"]["persistent_blocks"], 19)


class RuntimeFailClosedTest(unittest.TestCase):
    @staticmethod
    def _runtime_resource(module, abi_version, worker_groups, family):
        result = module._test_resource_requirements(abi_version, worker_groups)
        result.update(
            {
                "backend_abi_version": abi_version,
                "backend_family": family,
                "block_threads": 256 + 128 * worker_groups,
                "static_shared_memory_bytes": 0,
                "max_threads_per_block": 1024,
                "active_blocks_per_sm": 1,
            }
        )
        return result

    @staticmethod
    def _capabilities(module, candidate):
        result = deepcopy(module._test_capabilities)
        result.update(
            {
                "supported_backend_families": [
                    module.FIRST_LINEAR_ABI_FAMILY,
                    module.PACKED_FIRST_LINEAR_ABI_FAMILY,
                    module.WHOLE_HEAD_ABI_FAMILY,
                ],
                "supported_mixed_abis": [1, 2, 3, 4],
                "resource_query_family_aware": True,
            }
        )
        if candidate["abi_family"] == module.PACKED_FIRST_LINEAR_ABI_FAMILY:
            result.update(
                {
                    "mixed_render_packed_heads_abi": 3,
                    "mixed_render_packed_heads": True,
                    "mixed_packed_family": candidate["abi_family"],
                    "mixed_packed_symbol": candidate["cuda_symbol"],
                    "mixed_packed_manifest": candidate["abi_manifest"],
                    "mixed_packed_manifest_sha256": candidate["abi_manifest_sha256"],
                    "mixed_packed_head_manifest_sha256": candidate[
                        "head_abi_manifest_sha256"
                    ],
                }
            )
        else:
            result.update(
                {
                    "mixed_render_whole_heads_abi": 4,
                    "mixed_render_whole_heads": True,
                    "mixed_whole_family": candidate["abi_family"],
                    "mixed_whole_symbol": candidate["cuda_symbol"],
                    "mixed_whole_manifest": candidate["abi_manifest"],
                    "mixed_whole_manifest_sha256": candidate["abi_manifest_sha256"],
                    "mixed_whole_head_manifest_sha256": candidate[
                        "head_abi_manifest_sha256"
                    ],
                    "whole_head_tail_features_min": 1,
                    "whole_head_tail_features_max": 128,
                    "whole_head_scratch_bytes_per_worker_group": 512,
                    "whole_head_named_barriers_per_worker_group": 1,
                    "whole_head_barrier_base_id": 2,
                    "whole_head_descriptor_named_barrier_id": 7,
                }
            )
        return result

    def test_family_capability_manifest_and_resources_fail_closed(self):
        module = _load_module(available=True, grad_enabled=False)
        module.GaussianRasterizer.forward_with_packed_heads = lambda self: None
        module.GaussianRasterizer.forward_with_whole_heads = lambda self: None
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False, compute_cov3D_python=False, convert_SHs_python=False
        )
        common = dict(
            stage="fine", cam_type="dynerf", workload_name="flame_steak",
            iteration=14000, qualification_mode=True,
        )
        for builder in (
            module.packed_first_linear_candidate_contract,
            module.whole_head_candidate_contract,
        ):
            profile, candidate = _phase31_profile(module, builder)
            capabilities = self._capabilities(module, candidate)
            resources = self._runtime_resource(
                module,
                3 if builder == module.packed_first_linear_candidate_contract else 4,
                2,
                candidate["abi_family"],
            )
            with self.subTest(family=candidate["abi_family"]), mock.patch.object(
                module, "_query_rasterizer_capabilities", return_value=capabilities
            ), mock.patch.object(
                module, "_query_rasterizer_variant_resources", return_value=resources
            ):
                self.assertIsNone(
                    module.tacker_support_reason(pc, pipe, profile, **common)
                )

            bad_capabilities = deepcopy(capabilities)
            hash_key = (
                "mixed_packed_manifest_sha256"
                if candidate["abi_family"] == module.PACKED_FIRST_LINEAR_ABI_FAMILY
                else "mixed_whole_manifest_sha256"
            )
            bad_capabilities[hash_key] = "f" * 64
            with mock.patch.object(
                module, "_query_rasterizer_capabilities",
                return_value=bad_capabilities,
            ):
                self.assertIn(
                    hash_key,
                    module.tacker_support_reason(pc, pipe, profile, **common),
                )

            bad_resources = deepcopy(resources)
            bad_resources["backend_family"] = "wrong"
            with mock.patch.object(
                module, "_query_rasterizer_capabilities", return_value=capabilities
            ), mock.patch.object(
                module, "_query_rasterizer_variant_resources",
                return_value=bad_resources,
            ):
                self.assertIn(
                    "backend_family",
                    module.tacker_support_reason(pc, pipe, profile, **common),
                )

    def test_c3_c4_first_launch_failure_replays_complete_sequence(self):
        module = _load_module(available=True, grad_enabled=False)
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False, compute_cov3D_python=False, convert_SHs_python=False
        )

        class Context:
            def __init__(self, view):
                self.viewpoint_camera = view
                self.screenspace_points = FakeTensor()

        class Task:
            mixed_outputs = None

            def record_stream(self, _stream):
                pass

        class FailingPartition:
            def __init__(self, variant):
                self.variant = variant

            def prepare(self, _context, _pc, _stream, _cached, prefix_ready):
                prefix_ready.record(module._test_torch.cuda.current)
                return Task()

            def launch_mixed(self, *_args):
                raise RuntimeError("synthetic {} launch failure".format(
                    self.variant.backend
                ))

        tensor = FakeTensor()
        state = GaussianRenderState(tensor, tensor, tensor, tensor, tensor)
        for builder, cache_name, cached in (
            (
                module.packed_first_linear_candidate_contract,
                "_cache_packed_head_parameters",
                (FakeTensor(), FakeTensor()),
            ),
            (
                module.whole_head_candidate_contract,
                "_cache_whole_head_parameters",
                tuple((FakeTensor(), FakeTensor()) for _ in range(4)),
            ),
        ):
            profile, candidate = _phase31_profile(module, builder)
            with self.subTest(family=candidate["abi_family"]), mock.patch.object(
                module, "tacker_support_reason", return_value=None
            ), mock.patch.object(
                module, "_view_profile_reason", return_value=None
            ), mock.patch.object(
                module, cache_name, return_value=cached
            ), mock.patch.object(
                module, "prepare_render_context",
                side_effect=lambda view, *_a, **_k: Context(view),
            ), mock.patch.object(module, "deform_for_render", return_value=state):
                renderer = module.TackerRenderer(
                    pc, pipe, FakeTensor(), cam_type="dynerf",
                    profile_override=profile, workload_name="flame_steak",
                    iteration=14000, qualification_mode=True,
                )
                renderer.partition = FailingPartition(
                    module.fusion_variant_from_candidate(candidate)
                )
                outputs = list(renderer.render_sequence(iter(("0", "1", "2"))))
            self.assertEqual(
                outputs,
                [{"two_stream": "0"}, {"two_stream": "1"}, {"two_stream": "2"}],
            )
            self.assertIn("launch failure", renderer.last_fallback_reason)
            self.assertFalse(renderer.last_used_tacker)

    def test_one_two_and_n_frame_counts_hold_for_new_families(self):
        module = _load_module(available=True, grad_enabled=False)
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False, compute_cov3D_python=False, convert_SHs_python=False
        )
        for builder, cache_name, cached in (
            (
                module.packed_first_linear_candidate_contract,
                "_cache_packed_head_parameters",
                (FakeTensor(), FakeTensor()),
            ),
            (
                module.whole_head_candidate_contract,
                "_cache_whole_head_parameters",
                tuple((FakeTensor(), FakeTensor()) for _ in range(4)),
            ),
        ):
            profile, _candidate = _phase31_profile(module, builder)
            with mock.patch.object(
                module, "tacker_support_reason", return_value=None
            ), mock.patch.object(
                module, "_view_profile_reason", return_value=None
            ), mock.patch.object(module, cache_name, return_value=cached):
                for frame_count in (1, 2, 5):
                    renderer = module.TackerRenderer(
                        pc, pipe, FakeTensor(), cam_type="dynerf",
                        profile_override=profile, workload_name="flame_steak",
                        iteration=14000, qualification_mode=True,
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
                    with self.subTest(
                        family=builder.__name__, frame_count=frame_count
                    ):
                        self.assertEqual(len(outputs), frame_count)
                        self.assertEqual(counts["full_deformation"], 1)
                        self.assertEqual(counts["prefix"], frame_count - 1)
                        self.assertEqual(counts["mixed_launches"], frame_count - 1)
                        self.assertEqual(counts["suffix"], frame_count - 1)
                        self.assertEqual(counts["solo_raster"], 1)
                        self.assertEqual(counts["outputs"], frame_count)
                        self.assertEqual(
                            counts["selected_head_evaluations_per_head"],
                            frame_count,
                        )


if __name__ == "__main__":
    unittest.main()
