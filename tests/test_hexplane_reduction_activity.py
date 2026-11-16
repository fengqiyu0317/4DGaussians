"""CPU checks that reference and fused timelines measure comparable regions."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("hexplane_reduction_activity_cpu", scripts / "profile_hexplane_reduction_activity.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
import analyze_hexplane_reduction as summary


class ReductionActivityTests(unittest.TestCase):
    def make_csv(self, path, mixed, fused, bad_graph=False, missing_mul=False):
        rows = []
        def add(s, e, name, stream=17, node=1):
            rows.append((s, e, stream, node, 1, 256, 25, name))
        for i in range(12):
            add(1 + i * 2, 2 + i * 2, "grid_sampler_2d_kernel", node=0)
        if mixed:
            add(150, 450, "tacker_mix_raster_hexplane_512_v1", stream=13, node=0)
        else:
            for i in range(12):
                add(100 + i * 30, 108 + i * 30, "grid_sampler_2d_kernel")
        if fused:
            add(460, 550, "hexplane_reduce_features_tiled_v1", node=0 if bad_graph else 1)
        else:
            for i in range(11 if missing_mul else 12):
                add((460 + i * 8) if mixed else (110 + i * 30),
                    (464 + i * 8) if mixed else (120 + i * 30), "MulFunctor")
            add(560, 570, "CatArrayBatchedCopy")
        add(90, 95, "preprocessCUDA", stream=13, node=0)
        add(96, 99, "duplicateWithKeys", stream=13, node=0)
        add(575, 585, "DeviceRadixSortOnesweepKernel", stream=13, node=0)
        add(590, 600, "identifyTileRanges", stream=13, node=0)
        # The mixed synthetic interval need not obey real prepare timing:
        # accounting is deliberately tested independently from submission.
        add(610, 650, "tacker_render_raster_fast_range_v12", stream=13, node=0)
        add(700, 900, "tacker_mix_render_deform_mlp_tensor_tail_split_raster_v12", stream=13, node=0)
        add(1000, 1005, "preprocessCUDA", stream=13, node=0)
        add(1090, 1100, "identifyTileRanges", stream=13, node=0)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("start_ns", "end_ns", "stream_id", "graph_node_id", "grid_x", "block_x", "registers", "name"))
            writer.writerows(rows)

    def test_four_cells_keep_sampling_and_reduction_separate(self):
        for mixed in (False, True):
            for fused in (False, True):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "trace.csv"
                    self.make_csv(path, mixed, fused)
                    rows = probe.analyze(path, 2, {"sampling": "mixed" if mixed else "original", "reduction": fused})
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["merge_kernel_count"], 1 if fused else 13)
                self.assertEqual(rows[0]["post_prepass_to_mlp_start_ms"], 50 / 1e6)
                self.assertEqual(rows[0]["mlp_raster_tail_ms"], 200 / 1e6)

    def test_fused_reduction_must_actually_be_a_graph_node(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.make_csv(path, False, True, bad_graph=True)
            with self.assertRaises(AssertionError):
                probe.analyze(path, 2, {"sampling": "original", "reduction": True})

    def test_reference_requires_all_twelve_multiply_kernels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.make_csv(path, True, False, missing_mul=True)
            with self.assertRaises(AssertionError):
                probe.analyze(path, 2, {"sampling": "mixed", "reduction": False})

    def test_green_prepare_can_switch_to_full_device_prepass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.make_csv(path, False, True)
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields, records = reader.fieldnames + ["context_id"], list(reader)
            for row in records:
                row["context_id"] = "1"
                if row["name"] in ("preprocessCUDA", "identifyTileRanges", "duplicateWithKeys", "DeviceRadixSortOnesweepKernel"):
                    row["stream_id"], row["context_id"] = "146", "3"
                elif row["name"] == "hexplane_reduce_features_tiled_v1" or (
                        row["name"] == "grid_sampler_2d_kernel" and row["graph_node_id"] != "0"):
                    row["stream_id"], row["context_id"] = "143", "2"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
            config = {"sampling": "original", "reduction": True,
                      "green_contexts": True, "green_prepass_sms": 16}
            sample = probe.analyze(path, 2, config)[0]
            streams = sample["stream_identity_context_and_stream"]
            self.assertEqual(streams["raster_prepare"], (3, 146))
            self.assertEqual(streams["raster_prefix"], streams["mixed_tail"])
            with self.assertRaises(AssertionError):
                probe.analyze(path, 2, dict(config, green_prepass_sms=12))

    def test_handoff_counts_both_streams_and_rejects_wrong_sort_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.make_csv(path, False, True)
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields, records = reader.fieldnames + ["context_id"], list(reader)
            for row in records:
                row["context_id"] = "1"
                if row["name"] in ("preprocessCUDA", "duplicateWithKeys"):
                    row["stream_id"], row["context_id"] = "146", "3"
                elif row["name"] == "hexplane_reduce_features_tiled_v1" or (
                        row["name"] == "grid_sampler_2d_kernel" and row["graph_node_id"] != "0"):
                    row["stream_id"], row["context_id"] = "143", "2"
            def write():
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(records)
            write()
            config = {"sampling": "original", "reduction": True, "green_contexts": True,
                      "green_prepass_sms": 16, "green_prepare_handoff": True}
            sample = probe.analyze(path, 2, config)[0]
            self.assertEqual(sample["prepare_kernel_sum_ms"], 28 / 1e6)
            self.assertEqual(sample["sort_through_ranges_span_ms"], 25 / 1e6)
            for row in records:
                if "DeviceRadixSort" in row["name"]:
                    row["stream_id"], row["context_id"] = "146", "3"
            write()
            with self.assertRaises(AssertionError):
                probe.analyze(path, 2, config)


class ReductionRecoveryTests(unittest.TestCase):
    @staticmethod
    def block(key, batch, fps):
        return [{"key": key, "batch": batch, "data": {"throughput_fps": fps}}
                for _ in range(4)]

    def test_first_complete_block_is_selected_not_fastest(self):
        early = [r for key in summary.CONFIGS for r in self.block(key, "early", 1)]
        later = self.block(summary.CONFIGS[0], "later", 999)
        with patch.object(summary, "checked_runs", side_effect=[
                (early, {}, "identity"), (later, {}, "identity")]):
            runs, rejected, duplicates, incomplete = summary.select_trace_blocks(
                [Path("early"), Path("later")], "identity")
        self.assertEqual(len(runs), 24)
        self.assertTrue(all(r["batch"] == "early" for r in runs))
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(rejected, {})
        self.assertEqual(incomplete, ["early", "later"])

    def test_missing_config_or_changed_identity_cannot_complete(self):
        early = self.block(summary.CONFIGS[0], "early", 1)
        for identity in ("identity", "changed"):
            with patch.object(summary, "checked_runs", return_value=(early, {}, identity)):
                with self.assertRaises(AssertionError):
                    summary.select_trace_blocks([Path("early")], "identity")

    def test_partial_abba_never_reuses_individual_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "initial.source_check.txt").write_text("runtime: OK\n")
            for index in range(3):
                (root / (str(index) + ".activity.json")).write_text("{}")
            with self.assertRaises(AssertionError):
                summary.checked_runs(root, 4)
            runs, rejected, identity = summary.checked_runs(root, 4, allow_incomplete=True)
            self.assertEqual(runs, [])
            self.assertIsNone(identity)
            self.assertIn("incomplete", rejected[0]["reason"])
            self.assertFalse((root / "completed.txt").exists())


if __name__ == "__main__":
    unittest.main()
