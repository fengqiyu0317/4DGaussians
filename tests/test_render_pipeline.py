"""CPU-only contract tests for the split renderer API."""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


def _load_renderer_module(torch_stub=None, diff_stub=None):
    if torch_stub is None:
        torch_stub = types.ModuleType("torch")

        class FakeTensor:
            pass

        torch_stub.Tensor = FakeTensor
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch_stub.is_grad_enabled = lambda: True

    if diff_stub is None:
        diff_stub = types.ModuleType("diff_gaussian_rasterization")
        diff_stub.GaussianRasterizationSettings = object
        diff_stub.GaussianRasterizer = object

    scene = types.ModuleType("scene")
    gaussian_model = types.ModuleType("scene.gaussian_model")
    gaussian_model.GaussianModel = object

    utils = types.ModuleType("utils")
    profiling = types.ModuleType("utils.profiling_utils")
    sh_utils = types.ModuleType("utils.sh_utils")

    @contextmanager
    def nvtx_range(_message):
        yield

    profiling.nvtx_range = nvtx_range
    sh_utils.eval_sh = lambda *_args, **_kwargs: None

    module_name = "gaussian_renderer_contract_test"
    path = Path(__file__).parents[1] / "gaussian_renderer" / "__init__.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "diff_gaussian_rasterization": diff_stub,
        "scene": scene,
        "scene.gaussian_model": gaussian_model,
        "utils": utils,
        "utils.profiling_utils": profiling,
        "utils.sh_utils": sh_utils,
        "torch": torch_stub,
        module_name: module,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def _load_cuda_renderer(grad_enabled=False, stream_aware=True):
    """Load the renderer against a deterministic, CPU-only CUDA API stub."""
    log = []
    state = types.SimpleNamespace(
        grad_enabled=grad_enabled,
        stream_aware=stream_aware,
        private_stream_count=0,
        event_count=0,
        results={},
    )

    class FakeTensor:
        def __init__(self, name):
            self.name = name
            self.is_cuda = True
            self.recorded_streams = []

        def record_stream(self, stream):
            self.recorded_streams.append(stream.name)
            log.append(("record_stream", self.name, stream.name))

    class FakeStream:
        def __init__(self, name, handle, device="cuda:0"):
            self.name = name
            self.cuda_stream = handle
            self.device = device

        def wait_stream(self, stream):
            log.append(("wait_stream", self.name, stream.name))

        def wait_event(self, event):
            log.append(("wait_event", self.name, event.name))

        def synchronize(self):
            log.append(("synchronize", self.name))

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            log.append(("record_event", self.name, stream.name))

    class FakeCuda:
        def __init__(self):
            self.available = True
            self.current = FakeStream("consumer", 1)

        def is_available(self):
            return self.available

        def current_stream(self, _device=None):
            return self.current

        def Stream(self, device=None):
            name = "private{}".format(state.private_stream_count)
            stream = FakeStream(name, 100 + state.private_stream_count, device)
            state.private_stream_count += 1
            log.append(("create_stream", name))
            return stream

        def Event(self, blocking=False):
            del blocking
            name = "event{}".format(state.event_count)
            state.event_count += 1
            log.append(("create_event", name))
            return FakeEvent(name)

        @contextmanager
        def stream(self, stream):
            previous = self.current
            self.current = stream
            try:
                yield
            finally:
                self.current = previous

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = FakeTensor
    torch_stub.cuda = FakeCuda()
    torch_stub.is_grad_enabled = lambda: state.grad_enabled

    diff_stub = types.ModuleType("diff_gaussian_rasterization")
    diff_stub.GaussianRasterizationSettings = object
    diff_stub.GaussianRasterizer = object
    if stream_aware is not None:
        diff_stub.tacker_capabilities = lambda: {
            "stream_aware": state.stream_aware
        }

    module = _load_renderer_module(torch_stub=torch_stub, diff_stub=diff_stub)
    state.log = log
    state.torch = torch_stub
    state.FakeTensor = FakeTensor
    state.FakeStream = FakeStream
    return module, state


def _install_pipeline_stubs(module, harness):
    tensor_type = harness.FakeTensor

    def prepare(view, *_args, **_kwargs):
        harness.log.append(("D", view))
        tensor = lambda suffix: tensor_type("{}:{}".format(view, suffix))
        rasterizer = types.SimpleNamespace(
            raster_settings=(tensor("camera_matrix"), tensor("background"))
        )
        return module.RenderContext(
            viewpoint_camera=view,
            screenspace_points=tensor("screen"),
            rasterizer=rasterizer,
            means3D=tensor("means"),
            means2D=tensor("means2d"),
            scales=tensor("scales"),
            rotations=tensor("rotations"),
            opacity=tensor("opacity"),
            shs=tensor("shs"),
            timestamp=tensor("time"),
            cov3D_precomp=None,
            colors_precomp=None,
            convert_shs_python=False,
        )

    def deform(context, _pc, stage="fine"):
        del stage
        tensor = lambda suffix: tensor_type(
            "{}:{}".format(context.viewpoint_camera, suffix)
        )
        return module.GaussianRenderState(
            means3D=tensor("deformed_means"),
            scales=tensor("deformed_scales"),
            rotations=tensor("deformed_rotations"),
            opacities=tensor("deformed_opacity"),
            shs=tensor("deformed_shs"),
        )

    def rasterize(context, _state):
        view = context.viewpoint_camera
        harness.log.append(("R", view))
        tensors = [tensor_type("{}:out{}".format(view, index)) for index in range(5)]
        result = module.RenderResult(*tensors)
        harness.results[view] = result
        return result

    def serial_render(view, *_args, **_kwargs):
        harness.log.append(("S", view))
        return {"serial": view}

    module.prepare_render_context = prepare
    module.deform_for_render = deform
    module.rasterize_state = rasterize
    module.render = serial_render


def _make_renderer(module):
    xyz = types.SimpleNamespace(device="cuda:0")
    pc = types.SimpleNamespace(get_xyz=xyz)
    pipe = types.SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
    )
    return module.TwoStreamRenderer(pc, pipe, bg_color=object())


renderer = _load_renderer_module()


class SplitRendererContractTest(unittest.TestCase):
    def test_render_result_preserves_legacy_dictionary(self):
        values = [object() for _ in range(5)]
        result = renderer.RenderResult(*values)

        legacy = result.as_dict()

        self.assertEqual(
            list(legacy),
            ["render", "viewspace_points", "visibility_filter", "radii", "depth"],
        )
        self.assertIs(legacy["render"], values[0])
        self.assertIs(legacy["depth"], values[4])

    def test_deformation_and_activation_are_separate_from_rasterization(self):
        context = renderer.RenderContext(
            viewpoint_camera=None,
            screenspace_points=0,
            rasterizer=None,
            means3D=0,
            means2D=0,
            scales=0,
            rotations=0,
            opacity=0,
            shs=0,
            timestamp=0,
            cov3D_precomp=None,
            colors_precomp=None,
            convert_shs_python=False,
        )

        class FakeGaussianModel:
            scaling_activation = staticmethod(lambda value: value + 10)
            rotation_activation = staticmethod(lambda value: value + 20)
            opacity_activation = staticmethod(lambda value: value + 30)

            @staticmethod
            def _deformation(means, scales, rotations, opacity, shs, timestamp):
                del timestamp
                return means + 1, scales + 2, rotations + 3, opacity + 4, shs + 5

        state = renderer.deform_for_render(context, FakeGaussianModel(), stage="fine")

        self.assertEqual(state.means3D, 1)
        self.assertEqual(state.scales, 12)
        self.assertEqual(state.rotations, 23)
        self.assertEqual(state.opacities, 34)
        self.assertEqual(state.shs, 5)

    def test_unsupported_two_stream_configuration_reports_serial_fallback(self):
        pipe = types.SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        with mock.patch.object(renderer.torch.cuda, "is_available", return_value=False):
            reason = renderer.two_stream_support_reason(pipe)

        self.assertEqual(reason, "CUDA is unavailable")

    def test_timestamp_keeps_legacy_tensor_copy_and_detach_path(self):
        class FakeScreen:
            def __add__(self, _value):
                return self

            def retain_grad(self):
                pass

        class FakeTimestamp:
            def __init__(self):
                self.to_device = None
                self.repeat_shape = None

            def to(self, device):
                self.to_device = device
                return self

            def repeat(self, *shape):
                self.repeat_shape = shape
                return self

        xyz = types.SimpleNamespace(
            dtype="float32",
            device="cuda:0",
            shape=(7, 3),
        )
        pc = types.SimpleNamespace(
            get_xyz=xyz,
            active_sh_degree=3,
            _opacity=object(),
            get_features=object(),
            _scaling=object(),
            _rotation=object(),
        )
        cuda_value = types.SimpleNamespace(cuda=lambda: object())
        camera = types.SimpleNamespace(
            FoVx=1.0,
            FoVy=1.0,
            image_height=10,
            image_width=20,
            world_view_transform=cuda_value,
            full_proj_transform=cuda_value,
            camera_center=cuda_value,
            time=object(),
        )
        pipe = types.SimpleNamespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        timestamp = FakeTimestamp()
        tensor_spy = mock.Mock(return_value=timestamp)
        as_tensor_spy = mock.Mock(side_effect=AssertionError("as_tensor must not run"))

        with mock.patch.object(renderer.torch, "zeros_like", return_value=FakeScreen(), create=True), mock.patch.object(
            renderer.torch, "tensor", tensor_spy, create=True
        ), mock.patch.object(
            renderer.torch, "as_tensor", as_tensor_spy, create=True
        ), mock.patch.object(
            renderer, "GaussianRasterizationSettings", lambda **kwargs: kwargs
        ), mock.patch.object(
            renderer, "GaussianRasterizer", lambda raster_settings: raster_settings
        ):
            context = renderer.prepare_render_context(
                camera,
                pc,
                pipe,
                bg_color=object(),
            )

        tensor_spy.assert_called_once_with(camera.time)
        as_tensor_spy.assert_not_called()
        self.assertEqual(timestamp.to_device, "cuda:0")
        self.assertEqual(timestamp.repeat_shape, (7, 1))
        self.assertIs(context.timestamp, timestamp)


class TwoStreamRendererContractTest(unittest.TestCase):
    def test_constructor_is_cuda_lazy_and_call_time_no_grad_is_honored(self):
        module, harness = _load_cuda_renderer(grad_enabled=True, stream_aware=True)
        _install_pipeline_stubs(module, harness)
        pipeline = _make_renderer(module)

        # Construction while grad is enabled neither fixes the fallback result
        # nor creates CUDA resources.
        self.assertEqual(harness.private_stream_count, 0)
        harness.grad_enabled = False
        outputs = list(pipeline.render_sequence(iter(("v0",))))

        self.assertEqual(len(outputs), 1)
        self.assertIsNone(pipeline.last_fallback_reason)
        self.assertEqual(harness.private_stream_count, 2)

        module2, harness2 = _load_cuda_renderer(grad_enabled=False, stream_aware=True)
        _install_pipeline_stubs(module2, harness2)
        fallback = _make_renderer(module2)
        harness2.grad_enabled = True

        self.assertEqual(
            list(fallback.render_sequence(("v0",))),
            [{"serial": "v0"}],
        )
        self.assertEqual(fallback.last_fallback_reason, "autograd is enabled")
        self.assertEqual(harness2.private_stream_count, 0)

    def test_cpu_constructor_and_old_rasterizer_use_serial_fallback(self):
        cpu_module = _load_renderer_module()
        pc = types.SimpleNamespace(get_xyz=types.SimpleNamespace(device="cpu"))
        pipe = types.SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )

        pipeline = cpu_module.TwoStreamRenderer(pc, pipe, bg_color=object())
        cpu_module.render = lambda view, *_args, **_kwargs: {"serial": view}

        self.assertIsNone(pipeline.raster_stream)
        self.assertEqual(
            list(pipeline.render_sequence(iter(("v0", "v1")))),
            [{"serial": "v0"}, {"serial": "v1"}],
        )
        self.assertEqual(pipeline.last_fallback_reason, "CUDA is unavailable")

        old_module, old_harness = _load_cuda_renderer(
            grad_enabled=False,
            stream_aware=None,
        )
        _install_pipeline_stubs(old_module, old_harness)
        old_pipeline = _make_renderer(old_module)
        self.assertEqual(
            list(old_pipeline.render_sequence(("v0",))),
            [{"serial": "v0"}],
        )
        self.assertIn("does not advertise", old_pipeline.last_fallback_reason)
        self.assertEqual(old_harness.private_stream_count, 0)

    def test_submission_order_is_streaming_prefill_steady_drain(self):
        module, harness = _load_cuda_renderer(grad_enabled=False, stream_aware=True)
        _install_pipeline_stubs(module, harness)
        pipeline = _make_renderer(module)

        class LoggedIterator:
            def __init__(self):
                self.values = iter(("v0", "v1", "v2"))

            def __iter__(self):
                return self

            def __next__(self):
                value = next(self.values)
                harness.log.append(("input", value))
                return value

        iterator = pipeline.render_sequence(LoggedIterator())
        first = next(iterator)
        self.assertIn("render", first)
        # The third input has not been pulled just to produce frame zero.
        self.assertEqual(
            [entry for entry in harness.log if entry[0] == "input"],
            [("input", "v0"), ("input", "v1")],
        )
        list(iterator)

        self.assertEqual(
            [entry for entry in harness.log if entry[0] in ("D", "R")],
            [
                ("D", "v0"),
                ("D", "v1"),
                ("R", "v0"),
                ("D", "v2"),
                ("R", "v1"),
                ("R", "v2"),
            ],
        )

    def test_caller_wait_and_result_record_stream_precede_each_yield(self):
        module, harness = _load_cuda_renderer(grad_enabled=False, stream_aware=True)
        _install_pipeline_stubs(module, harness)
        pipeline = _make_renderer(module)

        next(pipeline.render_sequence(("v0",)))
        result = harness.results["v0"]
        for tensor in (
            result.render,
            result.viewspace_points,
            result.visibility_filter,
            result.radii,
            result.depth,
        ):
            self.assertEqual(tensor.recorded_streams, ["private0", "consumer"])

        wait_index = next(
            index
            for index, entry in enumerate(harness.log)
            if entry[0:2] == ("wait_event", "consumer")
        )
        consumer_record_index = next(
            index
            for index, entry in enumerate(harness.log)
            if entry[0] == "record_stream" and entry[2] == "consumer"
        )
        self.assertLess(wait_index, consumer_record_index)
        self.assertIn(("wait_stream", "private0", "consumer"), harness.log)
        self.assertIn(("wait_stream", "private1", "consumer"), harness.log)
        self.assertIn(
            ("record_stream", "v0:camera_matrix", "private0"), harness.log
        )
        self.assertIn(("record_stream", "v0:background", "private0"), harness.log)

    def test_consumer_stream_switch_is_rejected(self):
        module, harness = _load_cuda_renderer(grad_enabled=False, stream_aware=True)
        _install_pipeline_stubs(module, harness)
        pipeline = _make_renderer(module)
        iterator = pipeline.render_sequence(("v0", "v1"))

        next(iterator)
        harness.torch.cuda.current = harness.FakeStream("other", 2)
        with self.assertRaisesRegex(RuntimeError, "same CUDA stream"):
            next(iterator)
        self.assertFalse(pipeline._active)

    def test_interleaved_generators_are_rejected_and_close_releases_guard(self):
        module, harness = _load_cuda_renderer(grad_enabled=False, stream_aware=True)
        _install_pipeline_stubs(module, harness)
        pipeline = _make_renderer(module)
        first = pipeline.render_sequence(("v0", "v1"))

        next(first)
        second = pipeline.render_sequence(("other",))
        with self.assertRaisesRegex(RuntimeError, "interleaved or reentrant"):
            next(second)
        self.assertTrue(pipeline._active)

        first.close()
        self.assertFalse(pipeline._active)


if __name__ == "__main__":
    unittest.main()
