"""CPU arithmetic and capture-only contracts for the opt-in event probe."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("overlap_events_cpu", scripts / "profile_hexplane_overlap_events.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class EventProbeTests(unittest.TestCase):
    def test_interval_overlap_clips_disjoint_and_touching_windows(self):
        self.assertEqual(probe.interval_overlap([1, 2], [3, 4]), 0)
        self.assertEqual(probe.interval_overlap([1, 2], [2, 4]), 0)
        self.assertEqual(probe.interval_overlap([1, 4], [2, 3]), 1)

    def test_gaps_between_samples_are_not_sampling_time(self):
        sample = {"sampling_intervals_ms": [[0, 1], [9, 10]],
                  "raster_prepare_interval_ms": [2, 8],
                  "raster_prepass_interval_ms": [11, 15]}
        report = probe.summarize_samples([sample])
        self.assertEqual(report["mean_sampling_bracket_sum_ms"], 2)
        self.assertEqual(report["overlap"]["raster_prepare"]["mean_ms"], 0)
        self.assertEqual(report["sampling_complete_before_prepass_frames"], 1)

    def test_overlap_uses_duration_weighted_ratio_not_mean_of_ratios(self):
        samples = [{"sampling_intervals_ms": [[0, 1]], "raster_prepare_interval_ms": [0, 1],
                    "raster_prepass_interval_ms": [5, 6]},
                   {"sampling_intervals_ms": [[0, 3]], "raster_prepare_interval_ms": [2, 3],
                    "raster_prepass_interval_ms": [2, 4]}]
        report = probe.summarize_samples(samples)
        self.assertEqual(report["overlap"]["raster_prepare"]["fraction_of_sampling_brackets_percent"], 50)
        self.assertEqual(report["overlap"]["raster_prepass"]["fraction_of_sampling_brackets_percent"], 25)

    def fake_cuda(self):
        log = []
        state = {"capturing": False}
        def event(**kwargs):
            self.assertEqual(kwargs, {"enable_timing": True, "external": True})
            return types.SimpleNamespace(record=lambda stream: log.append("record"))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            Event=event, is_current_stream_capturing=lambda: state["capturing"]))
        stream = types.SimpleNamespace(synchronize=lambda: log.append("setup_sync"))
        return torch, stream, log, state

    def test_warmup_sampling_does_not_record_or_increment(self):
        torch, stream, log, state = self.fake_cuda()
        bank = probe.CapturedSamplingEvents(torch, stream, "all")
        log.clear()
        self.assertEqual(bank.sample(lambda: 17), 17)
        self.assertEqual(bank.capture_count, 0)
        self.assertEqual(log, [])

    def test_all_brackets_are_external_and_only_record_during_capture(self):
        torch, stream, log, state = self.fake_cuda()
        bank = probe.CapturedSamplingEvents(torch, stream, "all")
        log.clear()
        state["capturing"] = True
        for _ in range(12):
            bank.sample(lambda: log.append("sample"))
        self.assertEqual(log, ["record", "sample", "record"] * 12)
        self.assertEqual(bank.capture_count, 12)

    def test_last_plane_control_adds_only_two_graph_event_nodes(self):
        torch, stream, log, state = self.fake_cuda()
        bank = probe.CapturedSamplingEvents(torch, stream, "last")
        log.clear()
        state["capturing"] = True
        for _ in range(12):
            bank.sample(lambda: log.append("sample"))
        self.assertEqual(log, ["sample"] * 11 + ["record", "sample", "record"])
        self.assertEqual(bank.plane_ids, (11,))

    def test_unexpected_mode_and_thirteenth_plane_fail(self):
        torch, stream, log, state = self.fake_cuda()
        with self.assertRaises(ValueError):
            probe.CapturedSamplingEvents(torch, stream, "off")
        bank = probe.CapturedSamplingEvents(torch, stream, "all")
        state["capturing"] = True
        for _ in range(12):
            bank.sample(lambda: None)
        with self.assertRaises(RuntimeError):
            bank.sample(lambda: None)


if __name__ == "__main__":
    unittest.main()
