"""CPU contracts for default-off, prefix-only Green Context scheduling."""
import ctypes
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests import test_tacker_pipeline as fixture


def load_api():
    path = fixture.ROOT / "gaussian_renderer/green_contexts.py"
    spec = importlib.util.spec_from_file_location("green_contexts_cpu_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GreenContextsContracts(unittest.TestCase):
    def test_handoff_is_default_off_and_requires_eight_plus_eight(self):
        api = load_api()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(api.handoff_enabled())
        env = {"TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF": "1",
               "TACKER_EXPERIMENTAL_GREEN_CONTEXTS": "1", "TACKER_GREEN_DEFORM_SMS": "8"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(api.handoff_enabled())
        for changes in ({"TACKER_GREEN_DEFORM_SMS": "4"},
                        {"TACKER_EXPERIMENTAL_GREEN_CONTEXTS": "0"},
                        {"TACKER_EXPERIMENTAL_GREEN_PREPARE_HANDOFF": "yes"}):
            with mock.patch.dict(os.environ, dict(env, **changes), clear=True), self.assertRaises(ValueError):
                api.handoff_enabled()

    def test_handoff_passes_both_events_and_records_completion_on_full_stream(self):
        module, renderer = self.make_renderer()
        renderer._ensure_cuda_resources()
        renderer._green_prepare_handoff = True
        renderer.raster_prefix_stream = fixture.FakeStream("green_raster", 202, module._test_log)
        current, following = renderer.slots
        current.green_duplicate_done = module._test_torch.cuda.Event()
        current.green_duplicate_done.cuda_event = 502
        current.green_prepare_done = module._test_torch.cuda.Event()
        following.green_prefix_done = module._test_torch.cuda.Event()
        following.green_prefix_done.cuda_event = 501
        current.context, current.state = object(), mock.Mock()
        renderer.partition.prepare_raster.return_value = fixture.FakeTensor()
        renderer._last_execution_counts = {}
        with mock.patch.object(module, "_record_context_stream"):
            renderer._enqueue_raster_prepare(current, following)
        renderer.partition.prepare_raster.assert_called_once_with(
            current.context, current.state,
            handoff=(renderer.raster_stream.cuda_stream, 501, 502))
        self.assertEqual(renderer._last_execution_counts["raster_prepare_handoffs"], 1)
        self.assertIn(("event", current.green_prepare_done.name, renderer.raster_stream.name), module._test_log)
        renderer.partition.prepare_raster.reset_mock()
        with mock.patch.object(module, "_record_context_stream"):
            renderer._enqueue_raster_prepare(current)  # Drain has no next prefix to join.
        renderer.partition.prepare_raster.assert_called_once_with(current.context, current.state)
        self.assertEqual(renderer._last_execution_counts["raster_prepare_handoffs"], 1)

    def test_default_off_does_not_load_library(self):
        api = load_api()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(api, "_load_native") as native:
            self.assertIsNone(api.experiment_config())
            native.assert_not_called()

    def test_config_validates_opt_in_sm_count_and_library(self):
        api = load_api()
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "green.so"
            library.touch()
            env = {"TACKER_EXPERIMENTAL_GREEN_CONTEXTS": "1", "TACKER_GREEN_CONTEXTS_LIBRARY": str(library)}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(api.experiment_config(), {"deform_sms": 4, "library": str(library)})
            for value in ("4", "8", "12"):
                with mock.patch.dict(os.environ, dict(env, TACKER_GREEN_DEFORM_SMS=value), clear=True):
                    self.assertEqual(api.experiment_config()["deform_sms"], int(value))
            for value in ("0", "6", "16", "abc"):
                with mock.patch.dict(os.environ, dict(env, TACKER_GREEN_DEFORM_SMS=value), clear=True):
                    with self.assertRaises(ValueError):
                        api.experiment_config()
        with mock.patch.dict(os.environ, {"TACKER_EXPERIMENTAL_GREEN_CONTEXTS": "yes"}, clear=True):
            with self.assertRaises(ValueError):
                api.experiment_config()

    def make_renderer(self):
        module = fixture._load_module(available=True, grad_enabled=False)
        pc = fixture._exact_model(training=False)
        with mock.patch.object(module, "_cache_pos_head_parameters", return_value=(fixture.FakeTensor(), fixture.FakeTensor())):
            renderer = module.TackerRenderer(pc, types.SimpleNamespace(), fixture.FakeTensor(),
                profile_override=module.load_tacker_profile(), qualification_mode=True)
        renderer.partition = mock.Mock(spec=module.FullMlpOverlapGraphPartition)
        return module, renderer

    def test_default_keeps_two_streams_and_aliases(self):
        module, renderer = self.make_renderer()
        renderer._ensure_cuda_resources()
        self.assertIs(renderer.raster_prefix_stream, renderer.raster_stream)
        self.assertIs(renderer.deform_prefix_stream, renderer.deform_stream)
        self.assertEqual(len(list(renderer._private_streams())), 2)
        self.assertEqual(renderer.green_contexts_metadata, {"enabled": False})
        self.assertIsNone(renderer.slots[0].green_prepare_done)

    def test_experiment_is_qualification_only_and_rejects_sampling_mixed(self):
        module, renderer = self.make_renderer()
        renderer._green_config = {"deform_sms": 4, "library": "/fake/green.so"}
        renderer.qualification_mode = False
        with self.assertRaisesRegex(RuntimeError, "requires qualification"):
            renderer._ensure_cuda_resources()
        renderer.qualification_mode = True
        with mock.patch.object(module, "_split_raster_v12_enabled", return_value=True), mock.patch.object(module, "_raster_hexplane_experiment", return_value=("fused", 3)):
            with self.assertRaisesRegex(RuntimeError, "original sampling"):
                renderer._ensure_cuda_resources()

    def test_green_streams_are_separate_and_ordinary_streams_survive(self):
        module, renderer = self.make_renderer()
        log = module._test_log
        owner = types.SimpleNamespace(
            deform_stream=fixture.FakeStream("green_deform", 201, log),
            raster_stream=fixture.FakeStream("green_raster", 202, log),
            metadata={"enabled": True, "deform_sms": 4, "raster_sms": 12, "tail_sms": 16},
            close=mock.Mock())
        api = types.SimpleNamespace(get_green_context_pair=mock.Mock(return_value=owner))
        renderer._green_config = {"deform_sms": 4, "library": "/fake/green.so"}
        with mock.patch.dict(sys.modules, {"gaussian_renderer.green_contexts": api}), mock.patch.object(module, "_split_raster_v12_enabled", return_value=True):
            renderer._ensure_cuda_resources()
        self.assertEqual(len(list(renderer._private_streams())), 4)
        self.assertIs(renderer.deform_prefix_stream, owner.deform_stream)
        self.assertIsNot(renderer.deform_prefix_stream, renderer.deform_stream)
        self.assertIsNotNone(renderer.slots[0].green_prefix_done)
        slot = renderer.slots[0]
        slot.has_raster_done = True
        renderer._wait_slot_reuse(slot, renderer.deform_prefix_stream)
        self.assertIn(("wait_event", "green_deform", slot.raster_done.name), log)
        renderer._active = True
        with self.assertRaisesRegex(RuntimeError, "active"):
            renderer.close()
        renderer._active = False
        renderer.close()
        self.assertIsNone(renderer.slots)
        owner.close.assert_not_called()
        renderer.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            renderer._ensure_cuda_resources()

    def test_process_pool_keeps_stream_handles_alive_and_reuses_pair(self):
        api = load_api()
        pair = mock.Mock()
        with mock.patch.object(api, "GreenContextPair", return_value=pair) as constructor:
            first = api.get_green_context_pair("cuda:0", 4, "/fake/green.so")
            second = api.get_green_context_pair("cuda:0", 4, "/fake/green.so")
        self.assertIs(first, second)
        constructor.assert_called_once()
        self.assertTrue(pair._published)
        pair.close.assert_not_called()

    def test_process_pool_does_not_reuse_four_sm_pair_for_eight_sm_request(self):
        api = load_api()
        pairs = [mock.Mock(), mock.Mock()]
        with mock.patch.object(api, "GreenContextPair", side_effect=pairs) as constructor:
            four = api.get_green_context_pair("cuda:0", 4, "/fake/green.so")
            eight = api.get_green_context_pair("cuda:0", 8, "/fake/green.so")
            again = api.get_green_context_pair("cuda:0", 8, "/fake/green.so")
        self.assertIsNot(four, eight)
        self.assertIs(eight, again)
        self.assertEqual(constructor.call_args_list,
                         [mock.call("cuda:0", 4, "/fake/green.so"),
                          mock.call("cuda:0", 8, "/fake/green.so")])

    def test_prepass_and_mixed_tail_use_all_sm_stream_after_prepare_only(self):
        module, renderer = self.make_renderer()
        renderer._ensure_cuda_resources()
        renderer._green_contexts = mock.Mock()
        renderer.raster_prefix_stream = fixture.FakeStream("green_raster", 202, module._test_log)
        current, following = renderer.slots
        current.green_prepare_done = module._test_torch.cuda.Event()
        current.context = object()
        current.state = mock.Mock()
        current.prepared_raster = fixture.FakeTensor(name="prepared")
        following.task = mock.Mock()
        following.hexplane_graph = types.SimpleNamespace(deferred_sampling=False)
        renderer._last_execution_counts = {"raster_prepasses": 0}
        renderer.partition.split_raster_tile.return_value = 8
        renderer.partition.result_from_mixed.return_value = (mock.Mock(), fixture.FakeTensor())
        observed = []
        renderer.partition.launch_prepared_raster_prefix.side_effect = lambda *args: observed.append(module._test_torch.cuda.current.name)
        renderer.partition.launch_prepared_mixed.side_effect = lambda *args, **kwargs: observed.append(module._test_torch.cuda.current.name)
        with mock.patch.object(module, "_record_context_stream"):
            renderer._enqueue_prepared_mixed(current, following)
        self.assertEqual(observed, ["private0", "private0"])
        log = module._test_log
        waited = ("wait_event", "private0", current.green_prepare_done.name)
        self.assertIn(waited, log)
        self.assertIn(("wait_event", "private0", following.prefix_ready.name), log)
        # The Deform join must stay AFTER prepass, not before it.
        events = []
        renderer.partition.launch_prepared_raster_prefix.side_effect = lambda *args: events.append("prepass")
        renderer.partition.launch_prepared_mixed.side_effect = lambda *args, **kwargs: events.append("mixed")
        renderer.raster_stream.wait_event = lambda event: events.append(event.name)
        with mock.patch.object(module, "_record_context_stream"):
            renderer._enqueue_prepared_mixed(current, following)
        self.assertLess(events.index(current.green_prepare_done.name), events.index("prepass"))
        self.assertLess(events.index("prepass"), events.index(following.prefix_ready.name))
        self.assertLess(events.index(following.prefix_ready.name), events.index("mixed"))
        self.assertEqual(renderer._last_execution_counts["raster_prepasses"], 2)
        # Avoid the destructor intentionally touching this test's dummy graph.
        following.hexplane_graph = None
        renderer.close()

    def test_suffix_waits_for_all_prefix_work_and_drain_waits_for_prepare(self):
        module, renderer = self.make_renderer()
        renderer._ensure_cuda_resources()
        slot = renderer.slots[0]
        slot.green_prefix_done = module._test_torch.cuda.Event()
        slot.green_prepare_done = module._test_torch.cuda.Event()
        slot.task = mock.Mock()
        slot.mixed_outputs = fixture.FakeTensor()
        renderer.partition.finish.return_value = mock.Mock()
        renderer._enqueue_suffix(slot)
        self.assertIn(("wait_event", "private1", slot.green_prefix_done.name), module._test_log)
        slot.context = object()
        slot.prepared_raster = fixture.FakeTensor()
        renderer.partition.result_from_prepared_solo.return_value = mock.Mock()
        with mock.patch.object(module, "_record_context_stream"):
            renderer._enqueue_prepared_solo_raster(slot)
        self.assertIn(("wait_event", "private0", slot.green_prepare_done.name), module._test_log)


if __name__ == "__main__":
    unittest.main()
