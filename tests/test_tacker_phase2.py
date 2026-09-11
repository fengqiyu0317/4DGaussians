"""CPU contracts for Phase-2 candidate partitions and mixed ABI dispatch."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import types
import unittest
from unittest import mock

from tests.test_tacker_pipeline import (
    FakeEvent,
    FakeTensor,
    GaussianRenderState,
    Linear,
    ReLU,
    RenderResult,
    Sequential,
    _exact_model,
    _load_module,
)


def _phase2_profile(module, heads, worker_groups=1, persistent_blocks=80):
    profile = module.load_tacker_profile()
    candidate = module.first_linear_candidate_contract(
        "{}_l1_w{}".format("_".join(heads), worker_groups),
        heads,
        worker_groups=worker_groups,
        persistent_blocks=persistent_blocks,
    )
    profile["candidates"].append(candidate)
    profile["selected_variant_id"] = candidate["variant_id"]
    profile["manifest"]["persistent_blocks"] = persistent_blocks
    profile["manifest_sha256"] = module.manifest_sha256(profile["manifest"])
    profile["selection"] = None
    profile["deployment"] = {"enabled": False, "valid": False}
    profile["profile_sha256"] = module.profile_sha256(profile)
    return profile, candidate


class CandidateContractTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def test_all_c1_heads_and_one_parallel_c2_are_structurally_runnable(self):
        for head_name in self.module.HEAD_ORDER:
            profile, candidate = _phase2_profile(self.module, [head_name])
            self.assertIs(self.module.validate_tacker_profile(profile), profile)
            partition = self.module.resolve_fusion_partition(candidate)
            self.assertEqual(partition.variant.selected_heads, (head_name,))
            self.assertEqual(partition.variant.physical_cta_threads, 384)

        profile, candidate = _phase2_profile(
            self.module, ["pos", "scales"], worker_groups=2
        )
        self.assertIs(self.module.validate_tacker_profile(profile), profile)
        partition = self.module.resolve_fusion_partition(candidate)
        self.assertEqual(partition.variant.selected_heads, ("pos", "scales"))
        self.assertEqual(partition.variant.physical_cta_threads, 512)
        self.assertEqual(len(candidate["backend_subgroups"]), 2)

    def test_frozen_v2_manifests_match_runtime_constants_and_dependency(self):
        root = Path(__file__).resolve().parents[1]
        head_path = root / "tacker_ext" / "abi" / "head_linear_v2.json"
        mixed_path = (
            root
            / "submodules"
            / "depth-diff-gaussian-rasterization"
            / "abi"
            / "tacker_mixed_render_heads_v2.json"
        )
        head_digest = hashlib.sha256(head_path.read_bytes()).hexdigest()
        mixed_digest = hashlib.sha256(mixed_path.read_bytes()).hexdigest()
        self.assertEqual(head_digest, self.module.HEAD_MULTI_ABI_SHA256)
        self.assertEqual(mixed_digest, self.module.MIXED_MULTI_ABI_SHA256)
        mixed_manifest = json.loads(mixed_path.read_text(encoding="utf-8"))
        self.assertEqual(
            mixed_manifest["tacker_ext_dependency"]["manifest_sha256"],
            head_digest,
        )

    def test_partition_graph_thread_ranges_and_abi_hash_are_fail_closed(self):
        for field, replacement, message in (
            ("skipped_python_nodes", [], "skipped_python_nodes"),
            ("stream_lifetimes", [], "stream_lifetimes"),
            ("physical_cta_threads", 384, "physical_cta_threads"),
            ("abi_manifest_sha256", "f" * 64, "abi_manifest_sha256"),
            (
                "head_abi_manifest_sha256",
                "f" * 64,
                "head_abi_manifest_sha256",
            ),
        ):
            profile, candidate = _phase2_profile(
                self.module, ["pos", "scales"], worker_groups=2
            )
            candidate[field] = replacement
            profile["profile_sha256"] = self.module.profile_sha256(profile)
            with self.assertRaisesRegex(self.module.TackerProfileError, message):
                self.module.validate_tacker_profile(profile)

        profile, candidate = _phase2_profile(
            self.module, ["pos", "scales"], worker_groups=2
        )
        candidate["backend_named_barriers"][0]["participants"] = 128
        profile["profile_sha256"] = self.module.profile_sha256(profile)
        with self.assertRaisesRegex(
            self.module.TackerProfileError, "backend_named_barriers"
        ):
            self.module.validate_tacker_profile(profile)

    def test_selected_head_order_and_worker_count_are_canonical(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.module.first_linear_candidate_contract(
                "wrong_order", ["scales", "pos"], worker_groups=1
            )
        with self.assertRaisesRegex(ValueError, "worker_groups"):
            self.module.first_linear_candidate_contract(
                "too_many_workers", ["pos"], worker_groups=2
            )

    def test_valid_candidate_requires_measured_resource_filter_inputs(self):
        profile, candidate = _phase2_profile(self.module, ["opacity"])
        candidate["correctness"] = {
            "valid": True,
            "actual_execution_mode": "tacker",
            "fallback_reason": None,
            "psnr_drop_db": 0.0,
            "ssim_drop": 0.0,
            "lpips_increase": 0.0,
        }
        profile["profile_sha256"] = self.module.profile_sha256(profile)
        with self.assertRaisesRegex(
            self.module.TackerProfileError, "resources must be measured"
        ):
            self.module.validate_tacker_profile(profile)


class GenericTaskTest(unittest.TestCase):
    @staticmethod
    def _head(name, width, selected, log):
        def selected_linear(value):
            if selected:
                raise AssertionError("{} selected Linear ran in Python".format(name))
            return value

        return Sequential(
            ReLU(log, name + "_prefix_relu"),
            Linear(
                128,
                128,
                value_fn=selected_linear,
                log=log,
                label=name + "_l1",
            ),
            ReLU(log, name + "_suffix_relu"),
            Linear(
                128,
                width,
                value_fn=lambda value: value + width,
                log=log,
                label=name + "_tail",
            ),
        )

    def _model_and_context(self, module, selected_heads, log):
        args = types.SimpleNamespace(
            static_mlp=False,
            empty_voxel=False,
            no_dx=False,
            no_ds=False,
            no_dr=False,
            no_do=False,
            no_dshs=False,
            apply_rotation=False,
        )
        widths = {"pos": 3, "scales": 3, "rotations": 4, "opacity": 1, "shs": 48}
        heads = {
            module.HEAD_MODULES[name]: self._head(
                name, widths[name], name in selected_heads, log
            )
            for name in module.HEAD_ORDER
        }
        network = types.SimpleNamespace(
            args=args,
            query_time=lambda *_args: FakeTensor(10.0, (2, 128), "float32"),
            **heads
        )
        deformation = types.SimpleNamespace(
            deformation_net=network,
            pos_poc=FakeTensor(),
            rotation_scaling_poc=FakeTensor(),
        )
        pc = types.SimpleNamespace(
            _deformation=deformation,
            scaling_activation=lambda value: value,
            rotation_activation=lambda value: value,
            opacity_activation=lambda value: value,
        )
        context = types.SimpleNamespace(
            means3D=FakeTensor(2.0, (2, 3)),
            scales=FakeTensor(3.0, (2, 3)),
            rotations=FakeTensor(4.0, (2, 4)),
            opacity=FakeTensor(6.0, (2, 1)),
            shs=FakeTensor(8.0, (2, 16, 3)),
            timestamp=FakeTensor(0.25, (2, 1)),
        )
        return pc, context, network

    def test_two_selected_linears_are_skipped_once_and_outputs_feed_suffixes(self):
        module = _load_module(available=True, grad_enabled=False)
        selected_heads = ("pos", "scales")
        log = []
        pc, context, network = self._model_and_context(
            module, selected_heads, log
        )
        candidate = module.first_linear_candidate_contract(
            "pos_scales", selected_heads, worker_groups=2
        )
        variant = module.fusion_variant_from_candidate(candidate)
        stream = module._test_torch.cuda.Stream(device="cuda:0")
        prefix_ready = FakeEvent("prefix_ready", log)

        with mock.patch.object(module, "_poc_fre", side_effect=lambda value, _poc: value):
            task = module.prepare_fusion_task(
                context, pc, stream, variant, prefix_ready=prefix_ready
            )

        for head_name in selected_heads:
            self.assertEqual(
                getattr(network, module.HEAD_MODULES[head_name])[1].calls, 0
            )
        event_index = log.index(("event", "prefix_ready", stream.name))
        self.assertLess(event_index, log.index(("rotations_prefix_relu", 10.0)))

        state = module.finish_fusion_task(
            task,
            pc,
            {
                "pos": FakeTensor(70.0, (2, 128), "float32"),
                "scales": FakeTensor(80.0, (2, 128), "float32"),
            },
        )
        self.assertEqual(network.pos_deform[1].calls, 0)
        self.assertEqual(network.scales_deform[1].calls, 0)
        self.assertEqual(network.pos_deform[3].calls, 1)
        self.assertEqual(network.scales_deform[3].calls, 1)
        self.assertEqual(state.means3D.value, 75.0)
        self.assertEqual(state.scales.value, 86.0)

    def test_output_count_dtype_shape_and_duplicate_execution_fail_closed(self):
        module = _load_module(available=True, grad_enabled=False)
        pc, context, _network = self._model_and_context(module, ("shs",), [])
        candidate = module.first_linear_candidate_contract("shs", ["shs"])
        stream = module._test_torch.cuda.Stream(device="cuda:0")
        with mock.patch.object(module, "_poc_fre", side_effect=lambda value, _poc: value):
            task = module.prepare_fusion_task(
                context,
                pc,
                stream,
                module.fusion_variant_from_candidate(candidate),
            )
        with self.assertRaisesRegex(TypeError, "FP32"):
            module.finish_fusion_task(
                task, pc, FakeTensor(1.0, (2, 128), "float16")
            )
        with self.assertRaisesRegex(ValueError, r"\[N, 128\]"):
            module.finish_fusion_task(
                task, pc, FakeTensor(1.0, (2, 64), "float32")
            )
        task.executed_python_nodes.append(task.skipped_python_nodes[0])
        with self.assertRaisesRegex(RuntimeError, "executed twice"):
            task.assert_selected_nodes_not_executed_by_python()

    def test_sequence_binding_uses_candidate_worker_group_abi(self):
        module = _load_module()
        captured = {}

        class Rasterizer:
            def forward_with_heads(self, **kwargs):
                captured.update(kwargs)
                return "image", FakeTensor(1), "depth", ("head0", "head1")

        variant = module.FusionVariant(
            "pair", ("pos", "scales"), 2, 17, 2,
            "tacker_mix_render_heads_v2", 512, False
        )
        task = types.SimpleNamespace(
            head_inputs=("input0", "input1"),
            head_weights=("weight0", "weight1"),
            head_biases=("bias0", "bias1"),
            variant=variant,
        )
        context = types.SimpleNamespace(
            rasterizer=Rasterizer(),
            means2D="means2d",
            colors_precomp=None,
            cov3D_precomp=None,
            screenspace_points=FakeTensor(),
        )
        state = GaussianRenderState("means", "scales", "rot", "opacity", "shs")
        mixed = module._forward_with_heads(context, state, task, 17)
        result, outputs = module._result_from_mixed_heads(context, mixed, 2)
        self.assertEqual(outputs, ("head0", "head1"))
        self.assertEqual(result.render, "image")
        self.assertEqual(captured["worker_groups"], 2)
        self.assertEqual(captured["persistent_blocks"], 17)
        self.assertEqual(captured["head_inputs"], ("input0", "input1"))


class RuntimeGateAndOrderingTest(unittest.TestCase):
    @staticmethod
    def _runtime_fixture(module):
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        profile, candidate = _phase2_profile(
            module, ["pos", "scales"], worker_groups=2
        )
        candidate["resources"] = {
            "block_threads": 512,
            "registers_per_thread": 32,
            "static_shared_memory_bytes": 0,
            "max_threads_per_block": 1024,
            "active_blocks_per_sm": 1,
            "occupancy": 0.5,
        }
        profile["profile_sha256"] = module.profile_sha256(profile)
        common = {
            "stage": "fine",
            "cam_type": "dynerf",
            "workload_name": "flame_steak",
            "iteration": 14000,
            "qualification_mode": True,
        }
        return pc, pipe, profile, common

    def test_v2_binary_capability_contract_is_fail_closed(self):
        module = _load_module(available=True, grad_enabled=False)
        pc, pipe, profile, common = self._runtime_fixture(module)
        self.assertIsNone(module.tacker_support_reason(pc, pipe, profile, **common))

        cases = (
            ("mixed_multi_symbol", "wrong_symbol", "mixed_multi_symbol"),
            ("mixed_multi_manifest_sha256", "f" * 64, "mixed_multi_manifest"),
            ("head_multi_manifest_sha256", "f" * 64, "head_multi_manifest"),
            ("worker_group_threads", 64, "worker_group_threads"),
            ("head_descriptor_named_barrier_id", 3, "named_barrier"),
            ("max_head_tasks", 4, "max_head_tasks"),
            ("max_mixed_heads", 4, "max_mixed_heads"),
            ("supported_worker_groups", [1, 3, 4, 5], "worker-group contract"),
            (
                "mixed_threads_by_worker_groups",
                {1: 384, 2: 511, 3: 640, 4: 768, 5: 896},
                "thread-count contract",
            ),
        )
        for field, replacement, message in cases:
            capabilities = deepcopy(module._test_capabilities)
            capabilities[field] = replacement
            with self.subTest(capability=field), mock.patch.object(
                module, "_query_rasterizer_capabilities", return_value=capabilities
            ):
                self.assertIn(
                    message,
                    module.tacker_support_reason(pc, pipe, profile, **common),
                )

    def test_v2_resource_identity_occupancy_and_limits_fail_closed(self):
        module = _load_module(available=True, grad_enabled=False)
        pc, pipe, profile, common = self._runtime_fixture(module)
        cases = []
        wrong_abi = module._test_resource_requirements(2, 2)
        wrong_abi["abi_version"] = 1
        cases.append((wrong_abi, "abi_version"))
        wrong_groups = module._test_resource_requirements(2, 2)
        wrong_groups["worker_groups"] = 1
        cases.append((wrong_groups, "worker_groups"))
        changed = module._test_resource_requirements(2, 2)
        changed["registers_per_thread"] = 33
        cases.append((changed, "registers_per_thread changed"))
        zero = module._test_resource_requirements(2, 2)
        zero["occupancy"] = 0.0
        cases.append((zero, "zero runtime occupancy"))
        too_large = module._test_resource_requirements(2, 2)
        too_large["kernel_max_threads_per_block"] = 511
        cases.append((too_large, "compiled maximum thread count"))
        unsupported = module._test_resource_requirements(2, 2)
        unsupported["launch_supported"] = False
        cases.append((unsupported, "not launch-supported"))
        for resources, message in cases:
            with self.subTest(resource=message), mock.patch.object(
                module,
                "_query_rasterizer_variant_resources",
                return_value=resources,
            ):
                self.assertIn(
                    message,
                    module.tacker_support_reason(pc, pipe, profile, **common),
                )

    def test_multi_head_capability_resource_and_disabled_head_gates(self):
        module = _load_module(available=True, grad_enabled=False)
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        profile, _candidate = _phase2_profile(
            module, ["pos", "scales"], worker_groups=2
        )
        _candidate["resources"] = {
            "block_threads": 512,
            "registers_per_thread": 32,
            "static_shared_memory_bytes": 0,
            "max_threads_per_block": 1024,
            "active_blocks_per_sm": 1,
            "occupancy": 0.5,
        }
        profile["profile_sha256"] = module.profile_sha256(profile)
        common = dict(
            stage="fine",
            cam_type="dynerf",
            workload_name="flame_steak",
            iteration=14000,
            qualification_mode=True,
        )
        self.assertIsNone(module.tacker_support_reason(pc, pipe, profile, **common))

        pc._deformation.deformation_net.args.no_ds = True
        self.assertIn(
            "no_ds", module.tacker_support_reason(pc, pipe, profile, **common)
        )
        pc._deformation.deformation_net.args.no_ds = False
        original = module._query_rasterizer_variant_resources
        zero_occupancy = module._test_resource_requirements(2, 2)
        zero_occupancy["active_blocks_per_multiprocessor"] = 0
        zero_occupancy["occupancy"] = 0.0
        with mock.patch.object(
            module,
            "_query_rasterizer_variant_resources",
            return_value=zero_occupancy,
        ):
            self.assertIn(
                "zero runtime occupancy",
                module.tacker_support_reason(pc, pipe, profile, **common),
            )
        self.assertTrue(callable(original))

    def test_three_frame_slot_reuse_preserves_dependency_order(self):
        module = _load_module(available=True, grad_enabled=False)
        profile, candidate = _phase2_profile(module, ["opacity"])
        operations = []
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )

        class Context:
            def __init__(self, view):
                self.viewpoint_camera = view
                self.screenspace_points = FakeTensor(name="screen" + view)
                self.rasterizer = object()

        class Task:
            def __init__(self, context, variant):
                self.context = context
                self.variant = variant
                self.mixed_outputs = None
                self.head_output = None

            def record_stream(self, _stream):
                pass

        class Partition:
            def __init__(self, variant):
                self.variant = variant

            def prepare(self, context, _pc, _stream, _cached, prefix_ready):
                operations.append("prefix" + context.viewpoint_camera)
                prefix_ready.record(module._test_torch.cuda.current)
                return Task(context, self.variant)

            def launch_mixed(self, context, _state, task):
                operations.append(
                    "R{}+H{}".format(
                        context.viewpoint_camera, task.context.viewpoint_camera
                    )
                )
                return context, task

            def result_from_mixed(self, context, _mixed):
                tensor = FakeTensor(name="mixed" + context.viewpoint_camera)
                return RenderResult(tensor, tensor, FakeTensor(1), FakeTensor(1), tensor), (tensor,)

            def finish(self, task, _pc, outputs):
                operations.append("suffix" + task.context.viewpoint_camera)
                tensor = outputs[0]
                return GaussianRenderState(tensor, tensor, tensor, tensor, tensor)

        def prepare_context(view, *_args, **_kwargs):
            return Context(view)

        def full_deform(context, _pc, stage="fine"):
            del stage
            operations.append("D" + context.viewpoint_camera)
            tensor = FakeTensor(name="state" + context.viewpoint_camera)
            return GaussianRenderState(tensor, tensor, tensor, tensor, tensor)

        def solo(context, _state):
            operations.append("R" + context.viewpoint_camera)
            tensor = FakeTensor(name="solo" + context.viewpoint_camera)
            return RenderResult(tensor, tensor, FakeTensor(1), FakeTensor(1), tensor)

        with mock.patch.object(module, "tacker_support_reason", return_value=None), mock.patch.object(
            module, "_view_profile_reason", return_value=None
        ), mock.patch.object(
            module, "_cache_head_parameters", return_value=((FakeTensor(),), (FakeTensor(),))
        ), mock.patch.object(
            module, "prepare_render_context", side_effect=prepare_context
        ), mock.patch.object(
            module, "deform_for_render", side_effect=full_deform
        ), mock.patch.object(module, "rasterize_state", side_effect=solo):
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
            renderer.partition = Partition(
                module.fusion_variant_from_candidate(candidate)
            )
            outputs = list(renderer.render_sequence(iter(("0", "1", "2"))))

        self.assertEqual(len(outputs), 3)
        self.assertEqual(
            operations,
            ["D0", "prefix1", "R0+H1", "suffix1", "prefix2", "R1+H2", "suffix2", "R2"],
        )
        execution_counts = renderer.last_execution_counts
        self.assertEqual(
            execution_counts,
            {
                "input_frames": 3,
                "full_deformation": 1,
                "prefix": 2,
                "mixed_launches": 2,
                "suffix": 2,
                "solo_raster": 1,
                "outputs": 3,
                "selected_head_evaluations_per_head": 3,
            },
        )
        execution_counts["outputs"] = 0
        self.assertEqual(renderer.last_execution_counts["outputs"], 3)
        reuse_waits = [
            entry for entry in module._test_log
            if entry[0] == "wait_event" and entry[1] == "private1"
        ]
        self.assertTrue(reuse_waits)

    def test_v2_one_and_two_frame_drain_counts_and_order(self):
        module = _load_module(available=True, grad_enabled=False)
        profile, _candidate = _phase2_profile(module, ["opacity"])
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        cases = (
            (
                ("0",),
                ["D0", "R0"],
                {
                    "input_frames": 1,
                    "full_deformation": 1,
                    "prefix": 0,
                    "mixed_launches": 0,
                    "suffix": 0,
                    "solo_raster": 1,
                    "outputs": 1,
                    "selected_head_evaluations_per_head": 1,
                },
            ),
            (
                ("0", "1"),
                ["D0", "prefix1", "R0+H1", "suffix1", "R1"],
                {
                    "input_frames": 2,
                    "full_deformation": 1,
                    "prefix": 1,
                    "mixed_launches": 1,
                    "suffix": 1,
                    "solo_raster": 1,
                    "outputs": 2,
                    "selected_head_evaluations_per_head": 2,
                },
            ),
        )

        with mock.patch.object(
            module, "tacker_support_reason", return_value=None
        ), mock.patch.object(
            module, "_view_profile_reason", return_value=None
        ), mock.patch.object(
            module,
            "_cache_head_parameters",
            return_value=((FakeTensor(),), (FakeTensor(),)),
        ):
            for views, expected_operations, expected_counts in cases:
                operations = []
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

                def state_for(view):
                    tensor = FakeTensor(name="state" + view)
                    return GaussianRenderState(
                        tensor, tensor, tensor, tensor, tensor
                    )

                def result_for(view):
                    tensor = FakeTensor(name="render" + view)
                    return RenderResult(
                        tensor, tensor, FakeTensor(1), FakeTensor(1), tensor
                    )

                def full(view, slot):
                    operations.append("D" + view)
                    slot.context = types.SimpleNamespace(viewpoint_camera=view)
                    slot.state = state_for(view)

                def prefix(view, slot):
                    operations.append("prefix" + view)
                    slot.context = types.SimpleNamespace(viewpoint_camera=view)

                def mixed(current_slot, next_slot):
                    current = current_slot.context.viewpoint_camera
                    following = next_slot.context.viewpoint_camera
                    operations.append("R{}+H{}".format(current, following))
                    return result_for(current)

                def suffix(slot):
                    view = slot.context.viewpoint_camera
                    operations.append("suffix" + view)
                    slot.state = state_for(view)

                def solo(slot):
                    view = slot.context.viewpoint_camera
                    operations.append("R" + view)
                    return result_for(view)

                renderer._enqueue_full_deformation = full
                renderer._enqueue_prefix = prefix
                renderer._enqueue_mixed = mixed
                renderer._enqueue_suffix = suffix
                renderer._enqueue_solo_raster = solo
                outputs = list(renderer.render_sequence(iter(views)))

                with self.subTest(frame_count=len(views)):
                    self.assertEqual(len(outputs), len(views))
                    self.assertEqual(operations, expected_operations)
                    self.assertEqual(
                        renderer.last_execution_counts, expected_counts
                    )

    def test_first_launch_failure_replays_whole_sequence_on_fallback(self):
        module = _load_module(available=True, grad_enabled=False)
        profile, candidate = _phase2_profile(module, ["rotations"])
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )

        class Context:
            def __init__(self, view):
                self.viewpoint_camera = view
                self.screenspace_points = FakeTensor()

        class Task:
            mixed_outputs = None
            head_output = None

            def record_stream(self, _stream):
                pass

        class FailingPartition:
            def __init__(self, variant):
                self.variant = variant

            def prepare(self, _context, _pc, _stream, _cached, prefix_ready):
                prefix_ready.record(module._test_torch.cuda.current)
                return Task()

            def launch_mixed(self, *_args):
                raise RuntimeError("synthetic invalid configuration")

        tensor = FakeTensor()
        state = GaussianRenderState(tensor, tensor, tensor, tensor, tensor)
        with mock.patch.object(module, "tacker_support_reason", return_value=None), mock.patch.object(
            module, "_view_profile_reason", return_value=None
        ), mock.patch.object(
            module, "_cache_head_parameters", return_value=((FakeTensor(),), (FakeTensor(),))
        ), mock.patch.object(
            module, "prepare_render_context", side_effect=lambda view, *_a, **_k: Context(view)
        ), mock.patch.object(module, "deform_for_render", return_value=state):
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
            renderer.partition = FailingPartition(
                module.fusion_variant_from_candidate(candidate)
            )
            outputs = list(renderer.render_sequence(iter(("0", "1", "2"))))

        self.assertEqual(
            outputs,
            [{"two_stream": "0"}, {"two_stream": "1"}, {"two_stream": "2"}],
        )
        self.assertIn("synthetic invalid configuration", renderer.last_fallback_reason)
        self.assertFalse(renderer.last_used_tacker)
        self.assertEqual(renderer.actual_execution_mode, "two_stream")


if __name__ == "__main__":
    unittest.main()
