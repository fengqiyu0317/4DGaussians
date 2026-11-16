"""CPU contracts for the default-off fused reduction API (no CUDA import)."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock


def load_api():
    path = Path(__file__).resolve().parents[1] / "tacker_ext/tacker_4dgs_head/hexplane.py"
    spec = importlib.util.spec_from_file_location("hexplane_reduction_cpu_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FusedReductionContracts(unittest.TestCase):
    def setUp(self):
        self.api = load_api()

    def test_disabled_by_default_and_no_extension_import(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(self.api, "_extension") as extension:
            self.assertIsNone(self.api.fused_reduction_threads())
            extension.assert_not_called()

    def test_thread_options_are_explicit(self):
        for value, expected in ((None, 256), ("128", 128), ("256", 256)):
            env = {"TACKER_HEXPLANE_FUSED_REDUCTION": "1"}
            if value is not None:
                env["TACKER_HEXPLANE_REDUCTION_THREADS"] = value
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(self.api.fused_reduction_threads(), expected)

    def test_invalid_switch_or_threads_rejected(self):
        for enabled, threads in (("yes", "256"), ("1", "64"), ("1", "512"), ("1", "abc")):
            with mock.patch.dict(os.environ, {"TACKER_HEXPLANE_FUSED_REDUCTION": enabled,
                                             "TACKER_HEXPLANE_REDUCTION_THREADS": threads}, clear=True):
                with self.assertRaises(ValueError):
                    self.api.fused_reduction_threads()

    def test_out_api_keeps_caller_buffer(self):
        samples = [object() for _ in range(12)]
        output = object()
        extension = mock.Mock()
        with mock.patch.object(self.api, "_extension", return_value=extension):
            self.api.reduce_features(samples, threads=128)
            extension.reduce_features.assert_called_once_with(tuple(samples), 128)
            self.api.reduce_features(samples, threads=256, output=output)
            extension.reduce_features_out.assert_called_once_with(tuple(samples), output, 256)

    def test_resources_keep_thread_configuration_identity(self):
        extension = mock.Mock()
        extension.reduction_resources.return_value = [[128, 24, 0, 4224, 12, 87],
                                                      [256, 24, 0, 4224, 6, 87]]
        with mock.patch.object(self.api, "_extension", return_value=extension):
            info = self.api.reduction_resources()
        self.assertEqual(info[128]["threads"], 128)
        self.assertEqual(info[256]["static_shared_bytes"], 4224)


if __name__ == "__main__":
    unittest.main()
