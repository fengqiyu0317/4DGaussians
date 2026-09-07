"""CPU-only contracts for the physical 4DGS Tacker pipeline."""

from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class FakeTensor:
    def __init__(self, value=0.0, shape=(2, 1), dtype="float32", name=None):
        self.value = value
        self.shape = shape
        self.dtype = dtype
        self.name = name or str(value)
        self.device = "cuda:0"
        self.is_cuda = True
        self.recorded_streams = []

    def detach(self):
        return self

    def to(self, dtype=None, **_kwargs):
        return FakeTensor(self.value, self.shape, dtype or self.dtype, self.name)

    def contiguous(self):
        return self

    def record_stream(self, stream):
        self.recorded_streams.append(stream.name)

    def unsqueeze(self, _dimension):
        return FakeTensor(self.value, self.shape + (1,), self.dtype, self.name)

    def flatten(self, _dimension):
        return self

    def sin(self):
        return self

    def cos(self):
        return self

    def reshape(self, shape):
        return FakeTensor(self.value, tuple(shape), self.dtype, self.name)

    def __getitem__(self, key):
        del key
        return FakeTensor(self.value, self.shape, self.dtype, self.name)

    def __mul__(self, other):
        value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(self.value * value, self.shape, self.dtype)

    def __add__(self, other):
        value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(self.value + value, self.shape, self.dtype)

    def __gt__(self, other):
        value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(float(self.value > value), self.shape, "bool")


class FakeStream:
    def __init__(self, name, handle, log, device="cuda:0"):
        self.name = name
        self.cuda_stream = handle
        self.device = device
        self.log = log

    def wait_stream(self, stream):
        self.log.append(("wait_stream", self.name, stream.name))

    def wait_event(self, event):
        self.log.append(("wait_event", self.name, event.name))

    def synchronize(self):
        self.log.append(("synchronize", self.name))


class FakeEvent:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def record(self, stream):
        self.log.append(("event", self.name, stream.name))

    def synchronize(self):
        self.log.append(("synchronize_event", self.name))


class FakeCuda:
    def __init__(
        self,
        log,
        available=False,
        capability=(8, 6),
        device_name="NVIDIA RTX A6000",
    ):
        self.log = log
        self.available = available
        self.capability = capability
        self.device_name = device_name
        self.current = FakeStream("consumer", 1, log)
        self.stream_count = 0
        self.event_count = 0

    def is_available(self):
        return self.available

    def get_device_capability(self, _device=None):
        return self.capability

    def get_device_name(self, _device=None):
        return self.device_name

    def current_stream(self, _device=None):
        return self.current

    def Stream(self, device=None):
        name = "private{}".format(self.stream_count)
        result = FakeStream(name, 100 + self.stream_count, self.log, device)
        self.stream_count += 1
        self.log.append(("create_stream", name))
        return result

    def Event(self, blocking=False):
        del blocking
        name = "event{}".format(self.event_count)
        self.event_count += 1
        return FakeEvent(name, self.log)

    @contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous


class GaussianRenderState:
    def __init__(self, means3D, scales, rotations, opacities, shs):
        self.means3D = means3D
        self.scales = scales
        self.rotations = rotations
        self.opacities = opacities
        self.shs = shs

    def record_stream(self, stream):
        for value in vars(self).values():
            if hasattr(value, "record_stream"):
                value.record_stream(stream)


class RenderResult:
    def __init__(self, render, viewspace_points, visibility_filter, radii, depth):
        self.render = render
        self.viewspace_points = viewspace_points
        self.visibility_filter = visibility_filter
        self.radii = radii
        self.depth = depth

    def record_stream(self, stream):
        for value in vars(self).values():
            if hasattr(value, "record_stream"):
                value.record_stream(stream)

    def as_dict(self):
        return {
            "render": self.render,
            "viewspace_points": self.viewspace_points,
            "visibility_filter": self.visibility_filter,
            "radii": self.radii,
            "depth": self.depth,
        }


class DummyTwoStreamRenderer:
    def __init__(self, *_args, **_kwargs):
        self.last_fallback_reason = None
        self.synchronize_calls = 0

    def render_sequence(self, views):
        for view in views:
            yield {"two_stream": view}

    def synchronize(self):
        self.synchronize_calls += 1


class GaussianRasterizer:
    def forward_with_head(self, **_kwargs):
        raise AssertionError("the CPU contract must install a physical stub")


def _load_module(available=False, grad_enabled=False, capability=(8, 6)):
    log = []
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = FakeTensor
    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    torch_stub.cuda = FakeCuda(log, available=available, capability=capability)
    torch_stub.is_grad_enabled = lambda: grad_enabled
    torch_stub.ones_like = lambda value: FakeTensor(1.0, value.shape, value.dtype)
    torch_stub.zeros_like = lambda value: FakeTensor(0.0, value.shape, value.dtype)
    torch_stub.cat = lambda values, _dimension: values[0]

    capabilities = {
        "stream_aware": True,
        "mixed_render_head_abi": 1,
        "sm_target": "sm_86",
        "mixed_threads": 384,
        "raster_threads": 256,
        "head_threads": 128,
        "head_thread_base": 256,
        "raster_named_barrier_id": 1,
    }
    diff_stub = types.ModuleType("diff_gaussian_rasterization")
    diff_stub.GaussianRasterizer = GaussianRasterizer
    diff_stub.tacker_capabilities = lambda: dict(capabilities)

    renderer_stub = types.ModuleType("gaussian_renderer")
    renderer_stub.GaussianRasterizer = GaussianRasterizer
    renderer_stub.GaussianRenderState = GaussianRenderState
    renderer_stub.RenderResult = RenderResult
    renderer_stub.TwoStreamRenderer = DummyTwoStreamRenderer
    renderer_stub._record_context_stream = lambda _context, _stream: None
    renderer_stub.deform_for_render = lambda *_args, **_kwargs: None
    renderer_stub.prepare_render_context = lambda *_args, **_kwargs: None
    renderer_stub.rasterize_state = lambda *_args, **_kwargs: None

    name = "tacker_pipeline_cpu_contract"
    path = ROOT / "gaussian_renderer" / "tacker_pipeline.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "torch": torch_stub,
        "diff_gaussian_rasterization": diff_stub,
        "gaussian_renderer": renderer_stub,
        name: module,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    module._test_log = log
    module._test_torch = torch_stub
    module._test_capabilities = capabilities
    return module


def _valid_profile(module):
    with module.DEFAULT_PROFILE_PATH.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    profile["admission"] = {"enabled": True, "valid": True}
    profile["measurements"] = {
        "raster_slowdown_pct": 4.0,
        "mixed_p50_ms": 4.9,
        "solo_raster_p50_ms": 4.0,
        "solo_head_p50_ms": 1.0,
        "tacker_end_to_end_p50_ms": 10.0,
        "two_stream_end_to_end_p50_ms": 10.0,
        "psnr_drop_db": 0.05,
        "ssim_drop": 0.0001,
        "lpips_increase": 0.0001,
    }
    return profile


class ReLU:
    def __init__(self, log=None, label="relu"):
        self.log = log
        self.label = label

    def __call__(self, value):
        if self.log is not None:
            self.log.append((self.label, value.value))
        return value


class Linear:
    def __init__(self, in_features, out_features, value_fn=None, log=None, label="linear"):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = FakeTensor(0, (out_features, in_features), "float32")
        self.bias = FakeTensor(0, (out_features,), "float32")
        self.value_fn = value_fn or (lambda value: value)
        self.log = log
        self.label = label
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        if self.log is not None:
            self.log.append((self.label, value.value))
        return FakeTensor(self.value_fn(value.value), value.shape, "float32")


class Sequential:
    def __init__(self, *modules):
        self.modules = list(modules)

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Sequential(*self.modules[item])
        return self.modules[item]

    def __call__(self, value):
        for module in self.modules:
            value = module(value)
        return value


def _exact_head(output_width):
    return Sequential(
        ReLU(),
        Linear(128, 128),
        ReLU(),
        Linear(128, output_width),
    )


def _exact_model(training=False):
    args = types.SimpleNamespace(
        no_dx=False,
        no_ds=False,
        no_dr=False,
        no_do=False,
        no_dshs=False,
        apply_rotation=False,
        static_mlp=False,
        empty_voxel=False,
    )
    network = types.SimpleNamespace(
        W=128,
        D=0,
        args=args,
        pos_deform=_exact_head(3),
        scales_deform=_exact_head(3),
        rotations_deform=_exact_head(4),
        opacity_deform=_exact_head(1),
        shs_deform=_exact_head(48),
    )
    deformation = types.SimpleNamespace(
        training=training,
        deformation_net=network,
        pos_poc=FakeTensor(),
        rotation_scaling_poc=FakeTensor(),
    )
    return types.SimpleNamespace(
        _deformation=deformation,
        get_xyz=FakeTensor(shape=(111525, 3)),
        scaling_activation=lambda value: value,
        rotation_activation=lambda value: value,
        opacity_activation=lambda value: value,
    )


class ProfileContractTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def test_repository_profile_is_hashed_and_fail_closed(self):
        profile = self.module.load_tacker_profile()
        self.assertEqual(
            self.module.manifest_sha256(profile["manifest"]),
            profile["manifest_sha256"],
        )
        self.assertEqual(
            self.module.tacker_profile_admission_reason(profile),
            "Tacker profile is disabled",
        )

    def test_hash_mismatch_and_weakened_gate_are_rejected(self):
        profile = _valid_profile(self.module)
        profile["manifest"]["persistent_blocks"] = 3
        with self.assertRaisesRegex(self.module.TackerProfileError, "SHA-256"):
            self.module.validate_tacker_profile(profile)

        profile = _valid_profile(self.module)
        profile["thresholds"]["raster_slowdown_pct_max"] = 5.01
        with self.assertRaisesRegex(self.module.TackerProfileError, "weakens"):
            self.module.validate_tacker_profile(profile)

    def test_every_performance_and_quality_threshold_is_enforced(self):
        profile = _valid_profile(self.module)
        self.assertIsNone(self.module.tacker_profile_admission_reason(profile))

        cases = (
            ("raster_slowdown_pct", 5.001, "Raster QoS"),
            ("mixed_p50_ms", 5.0, "strictly faster"),
            ("tacker_end_to_end_p50_ms", 10.001, "two_stream"),
            ("psnr_drop_db", 0.051, "PSNR"),
            ("ssim_drop", 0.00011, "SSIM"),
            ("lpips_increase", 0.00011, "LPIPS"),
        )
        for key, value, reason in cases:
            failing = deepcopy(profile)
            failing["measurements"][key] = value
            self.assertIn(
                reason,
                self.module.tacker_profile_admission_reason(failing),
            )


class SupportGateContractTest(unittest.TestCase):
    def test_exact_contract_admits_and_specific_changes_explain_fallback(self):
        module = _load_module(available=True, grad_enabled=False)
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        profile = _valid_profile(module)

        self.assertIsNone(
            module.tacker_support_reason(
                pc,
                pipe,
                profile,
                stage="fine",
                cam_type="dynerf",
                workload_name="flame_steak",
                iteration=14000,
            )
        )

        self.assertIn(
            "workload name",
            module.tacker_support_reason(
                pc,
                pipe,
                profile,
                stage="fine",
                cam_type="dynerf",
                workload_name=None,
                iteration=14000,
            ),
        )

        pc._deformation.deformation_net.args.no_dx = True
        self.assertIn(
            "no_dx",
            module.tacker_support_reason(
                pc,
                pipe,
                profile,
                stage="fine",
                cam_type="dynerf",
                workload_name="flame_steak",
                iteration=14000,
            ),
        )
        pc._deformation.deformation_net.args.no_dx = False
        module._test_torch.cuda.capability = (8, 0)
        self.assertIn(
            "8.6",
            module.tacker_support_reason(
                pc,
                pipe,
                profile,
                stage="fine",
                cam_type="dynerf",
                workload_name="flame_steak",
                iteration=14000,
            ),
        )
        module._test_torch.cuda.capability = (8, 6)
        module._test_torch.cuda.device_name = "NVIDIA GeForce RTX 3090"
        self.assertIn(
            "RTX A6000",
            module.tacker_support_reason(
                pc,
                pipe,
                profile,
                stage="fine",
                cam_type="dynerf",
                workload_name="flame_steak",
                iteration=14000,
            ),
        )

    def test_camera_resolution_is_bound_to_profile(self):
        module = _load_module()
        profile = _valid_profile(module)
        matching = types.SimpleNamespace(image_width=1352, image_height=1014)
        mismatch = types.SimpleNamespace(image_width=800, image_height=800)
        self.assertIsNone(module._view_profile_reason(matching, profile))
        self.assertIn("resolution", module._view_profile_reason(mismatch, profile))

    def test_qualification_skips_only_measured_admission(self):
        module = _load_module(available=True, grad_enabled=False)
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        candidate = module.load_tacker_profile()
        common = {
            "stage": "fine",
            "cam_type": "dynerf",
            "workload_name": "flame_steak",
            "iteration": 14000,
        }

        self.assertEqual(
            module.tacker_support_reason(pc, pipe, candidate, **common),
            "Tacker profile is disabled",
        )
        self.assertIsNone(
            module.tacker_support_reason(
                pc,
                pipe,
                candidate,
                qualification_mode=True,
                **common
            )
        )

        bad = deepcopy(candidate)
        bad["manifest"]["pair_key"] = "wrong"
        self.assertIn(
            "qualification profile",
            module.tacker_support_reason(
                pc,
                pipe,
                bad,
                qualification_mode=True,
                **common
            ),
        )


class SplitHeadContractTest(unittest.TestCase):
    def test_selected_linear_is_not_repeated_and_physical_output_enters_suffix(self):
        module = _load_module(available=True, grad_enabled=False)
        log = []
        selected = Linear(
            128,
            128,
            value_fn=lambda _value: (_ for _ in ()).throw(
                AssertionError("selected Linear was executed by Python")
            ),
            log=log,
            label="selected",
        )
        suffix = Linear(
            128,
            3,
            value_fn=lambda value: value + 30.0,
            log=log,
            label="pos_suffix",
        )
        pos_head = Sequential(
            ReLU(log, "pos_prefix_relu"),
            selected,
            ReLU(log, "pos_suffix_relu"),
            suffix,
        )

        class OtherHead:
            def __init__(self, label, value):
                self.label = label
                self.value = value

            def __call__(self, _hidden):
                log.append((self.label, self.value))
                return FakeTensor(self.value, (2, 1), "float32")

        class StaticMask:
            def __call__(self, _hidden):
                return FakeTensor(0.5, (2, 1), "float32")

        args = types.SimpleNamespace(
            static_mlp=True,
            empty_voxel=False,
            no_ds=False,
            no_dr=False,
            no_do=False,
            no_dshs=False,
            apply_rotation=False,
        )
        network = types.SimpleNamespace(
            args=args,
            pos_deform=pos_head,
            scales_deform=OtherHead("scale_head", 4.0),
            rotations_deform=OtherHead("rotation_head", 5.0),
            opacity_deform=OtherHead("opacity_head", 7.0),
            shs_deform=OtherHead("shs_head", 8.0),
            static_mlp=StaticMask(),
            query_time=lambda *_args: FakeTensor(10.0, (2, 128), "float32"),
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
        deform_stream = module._test_torch.cuda.Stream(device="cuda:0")
        prefix_ready = FakeEvent("prefix_ready", log)
        weight = FakeTensor(1.0, (128, 128), "float16")
        bias = FakeTensor(0.0, (128,), "float32")

        with mock.patch.object(module, "_poc_fre", side_effect=lambda value, _poc: value):
            task = module.prepare_pos_head_task(
                context,
                pc,
                deform_stream,
                head_weight=weight,
                head_bias=bias,
                prefix_ready=prefix_ready,
            )
        self.assertEqual(selected.calls, 0)
        prefix_index = log.index(("event", "prefix_ready", deform_stream.name))
        self.assertLess(prefix_index, log.index(("scale_head", 4.0)))

        physical_output = FakeTensor(70.0, (2, 128), "float32")
        state = module.finish_pos_head_task(task, pc, physical_output)

        self.assertEqual(selected.calls, 0)
        self.assertIn(("pos_suffix", 70.0), log)
        self.assertEqual(state.means3D.value, 101.0)  # 2*0.5 + (70+30)
        self.assertEqual(state.scales.value, 5.5)  # 3*0.5 + 4
        self.assertEqual(state.rotations.value, 9.0)  # rotation has no mask
        self.assertEqual(state.opacities.value, 10.0)  # 6*0.5 + 7
        self.assertEqual(state.shs.value, 12.0)  # 8*0.5 + 8

    def test_physical_binding_adapter_uses_the_fixed_keyword_abi(self):
        module = _load_module()
        captured = {}

        class Rasterizer:
            def forward_with_head(self, **kwargs):
                captured.update(kwargs)
                return "image", "radii", "depth", "head"

        context = types.SimpleNamespace(
            rasterizer=Rasterizer(),
            means2D="means2d",
            colors_precomp=None,
            cov3D_precomp=None,
        )
        state = GaussianRenderState(
            means3D="means3d",
            scales="scales",
            rotations="rotations",
            opacities="opacity",
            shs="shs",
        )
        task = types.SimpleNamespace(
            head_input="head_input",
            head_weight="head_weight",
            head_bias="head_bias",
        )

        outputs = module._forward_with_head(context, state, task, 17)

        self.assertEqual(outputs, ("image", "radii", "depth", "head"))
        self.assertEqual(
            captured,
            {
                "means3D": "means3d",
                "means2D": "means2d",
                "opacities": "opacity",
                "head_input": "head_input",
                "head_weight": "head_weight",
                "head_bias": "head_bias",
                "shs": "shs",
                "colors_precomp": None,
                "scales": "scales",
                "rotations": "rotations",
                "cov3D_precomp": None,
                "persistent_blocks": 17,
            },
        )


class PipelineOrderingContractTest(unittest.TestCase):
    def test_two_frame_submission_is_d0_prefix1_mixed_suffix1_r1(self):
        module = _load_module(available=True, grad_enabled=False)
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
            def __init__(self, context, prefix_ready):
                self.context = context
                self.prefix_ready = prefix_ready
                self.head_input = FakeTensor(name="head_input" + context.viewpoint_camera)
                self.head_weight = FakeTensor(shape=(128, 128))
                self.head_bias = FakeTensor(shape=(128,))
                self.head_output = None

            def record_stream(self, _stream):
                pass

        def prepare(view, *_args, **_kwargs):
            return Context(view)

        def full_deform(context, _pc, stage="fine"):
            del stage
            operations.append("D" + context.viewpoint_camera)
            tensor = FakeTensor(name="state" + context.viewpoint_camera)
            return GaussianRenderState(tensor, tensor, tensor, tensor, tensor)

        def prefix(context, _pc, _stream, **kwargs):
            operations.append("prefix" + context.viewpoint_camera)
            kwargs["prefix_ready"].record(module._test_torch.cuda.current)
            return Task(context, kwargs["prefix_ready"])

        def mixed(context, _state, task, _persistent):
            operations.append(
                "R{}+H{}".format(
                    context.viewpoint_camera,
                    task.context.viewpoint_camera,
                )
            )
            tensor = FakeTensor(name="mixed" + context.viewpoint_camera)
            return tensor, FakeTensor(1), tensor, FakeTensor(9, (2, 128), "float32")

        def suffix(task, _pc, head_output):
            operations.append("suffix" + task.context.viewpoint_camera)
            self_value = head_output
            return GaussianRenderState(
                self_value, self_value, self_value, self_value, self_value
            )

        def solo(context, _state):
            operations.append("R" + context.viewpoint_camera)
            tensor = FakeTensor(name="solo" + context.viewpoint_camera)
            return RenderResult(tensor, tensor, FakeTensor(1), FakeTensor(1), tensor)

        with mock.patch.object(module, "tacker_support_reason", return_value=None), mock.patch.object(
            module, "_view_profile_reason", return_value=None
        ), mock.patch.object(
            module, "_cache_pos_head_parameters", return_value=(FakeTensor(), FakeTensor())
        ), mock.patch.object(
            module, "prepare_render_context", side_effect=prepare
        ), mock.patch.object(
            module, "deform_for_render", side_effect=full_deform
        ), mock.patch.object(
            module, "prepare_pos_head_task", side_effect=prefix
        ), mock.patch.object(
            module, "_forward_with_head", side_effect=mixed
        ), mock.patch.object(
            module, "finish_pos_head_task", side_effect=suffix
        ), mock.patch.object(
            module, "rasterize_state", side_effect=solo
        ):
            renderer = module.TackerRenderer(
                pc,
                pipe,
                bg_color=FakeTensor(),
                cam_type="dynerf",
                profile_override=module.load_tacker_profile(),
                workload_name="flame_steak",
                iteration=14000,
                qualification_mode=True,
            )
            outputs = list(renderer.render_sequence(iter(("0", "1"))))

        self.assertEqual(len(outputs), 2)
        self.assertEqual(
            operations,
            ["D0", "prefix1", "R0+H1", "suffix1", "R1"],
        )
        self.assertTrue(renderer.last_used_tacker)
        self.assertIsNone(renderer.last_fallback_reason)
        self.assertTrue(renderer.last_qualification_mode)
        self.assertEqual(renderer.actual_execution_mode, "tacker")

        renderer.synchronize()
        synchronized = [
            entry[1]
            for entry in module._test_log
            if entry[0] == "synchronize"
        ]
        self.assertEqual(synchronized, ["private0", "private1", "consumer"])

        with mock.patch.object(
            module,
            "_view_profile_reason",
            return_value="camera resolution does not match the admitted profile",
        ):
            with self.assertRaisesRegex(RuntimeError, "changed mid-sequence"):
                renderer._require_profile_view(object())

    def test_admission_failure_delegates_and_reports_reason(self):
        module = _load_module(available=False, grad_enabled=False)
        delegated = []

        class Fallback:
            def __init__(self, *_args, **_kwargs):
                self.last_fallback_reason = "CUDA is unavailable"
                self.synchronize_calls = 0

            def render_sequence(self, views):
                for view in views:
                    delegated.append(view)
                    yield {"fallback": view}

            def synchronize(self):
                self.synchronize_calls += 1

        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        reason = "measured profile did not pass admission"
        with mock.patch.object(module, "TwoStreamRenderer", Fallback), mock.patch.object(
            module, "tacker_support_reason", return_value=reason
        ), mock.patch.object(
            module, "_cache_pos_head_parameters", return_value=(FakeTensor(), FakeTensor())
        ):
            renderer = module.TackerRenderer(
                pc,
                pipe,
                bg_color=FakeTensor(),
                cam_type="dynerf",
            )
            outputs = list(renderer.render_sequence(iter(("0", "1"))))

        self.assertEqual(outputs, [{"fallback": "0"}, {"fallback": "1"}])
        self.assertEqual(delegated, ["0", "1"])
        self.assertEqual(renderer.last_fallback_reason, reason)
        self.assertFalse(renderer.last_used_tacker)
        self.assertEqual(module._test_torch.cuda.stream_count, 0)
        self.assertEqual(renderer.fallback_backend_reason, "CUDA is unavailable")
        self.assertEqual(renderer.actual_execution_mode, "serial")
        renderer.synchronize()
        self.assertEqual(renderer._fallback_renderer.synchronize_calls, 1)

    def test_qualification_mode_requires_explicit_override(self):
        module = _load_module(available=False, grad_enabled=False)
        pc = _exact_model(training=False)
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        with mock.patch.object(
            module, "_cache_pos_head_parameters", return_value=(FakeTensor(), FakeTensor())
        ):
            renderer = module.TackerRenderer(
                pc,
                pipe,
                bg_color=FakeTensor(),
                cam_type="dynerf",
                workload_name="flame_steak",
                iteration=14000,
                qualification_mode=True,
            )

        self.assertIn("explicit profile_override", renderer.fallback_reason)
        self.assertFalse(renderer.last_qualification_mode)
        self.assertEqual(renderer.actual_execution_mode, "not_run")
        renderer.synchronize()
        self.assertIn(("synchronize_event", "event0"), module._test_log)


if __name__ == "__main__":
    unittest.main()
