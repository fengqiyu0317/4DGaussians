"""CPU-only checks of conservative ownership attribution in process audits."""
import importlib.util
from pathlib import Path
import sys
import unittest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("green_analysis_cpu", scripts / "analyze_green_contexts.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class GreenAuditTests(unittest.TestCase):
    tree = "/data/green-test"
    stem = "green-trace/01-green0-on"
    live = ("123 root python3 /usr/bin/python3 /data/green-test/scripts/"
            "profile_hexplane_reduction_activity.py --metadata "
            "/data/green-test/green-trace/01-green0-on.json")

    def analyze(self, *lines):
        return analysis.audit_process_ownership("\n".join(lines), self.tree, self.stem)

    def test_exact_own_run_and_observed_terminal_row_are_not_foreign(self):
        report = self.analyze(self.live, self.live, "123 root python3 [python3]")
        self.assertEqual(report["foreign_processes"], [])
        self.assertEqual(report["proven_own_terminal_rows"], ["123 root python3 [python3]"])

    def test_unknown_terminal_pid_and_different_user_are_foreign(self):
        foreign = ["456 root python3 [python3]", "123 other python3 [python3]"]
        self.assertEqual(self.analyze(self.live, *foreign)["foreign_processes"], sorted(foreign))

    def test_other_run_of_same_driver_and_changed_command_are_foreign(self):
        foreign = [self.live.replace("01-green0-on.json", "02-green0-on.json"),
                   "123 root python3 /usr/bin/python3 other_experiment.py"]
        self.assertEqual(self.analyze(self.live, *foreign)["foreign_processes"], sorted(foreign))

    def test_terminal_before_last_live_row_is_not_waived(self):
        terminal = "123 root python3 [python3]"
        self.assertEqual(self.analyze(terminal, self.live)["foreign_processes"], [terminal])

    def test_missing_or_multiple_own_identities_fail_closed(self):
        for lines in (["123 root python3 [python3]"],
                      [self.live, self.live.replace("123 root", "456 root")]):
            with self.assertRaises(AssertionError):
                self.analyze(*lines)


class GreenPartitionTests(unittest.TestCase):
    def resources(self, deform):
        return {"deform_sms": deform, "raster_sms": 16 - deform,
                "raster_prepare_sms": 16 - deform, "raster_prepass_sms": 16,
                "tail_sms": 16, "scope": "deform_prefix_raster_prepare"}

    def test_dynamic_partitions_keep_prepass_and_tail_full_device(self):
        for deform in (4, 8, 12):
            partition = analysis.resource_partition(self.resources(deform))
            self.assertEqual(partition["deform_prefix_sms"], deform)
            self.assertEqual(partition["raster_prepare_sms"], 16 - deform)
            self.assertEqual(partition["raster_prepass_sms"], 16)
            rows = [{"green": False, "partition": None}, {"green": True, "partition": partition}]
            self.assertEqual(analysis.cohort_partition(rows, deform), partition)

    def test_wrong_counts_or_mixed_cohorts_fail_closed(self):
        for updates in ({"raster_sms": 12}, {"raster_prepare_sms": 12},
                        {"raster_prepass_sms": 12}, {"tail_sms": 8}):
            with self.assertRaises(AssertionError):
                analysis.resource_partition(dict(self.resources(8), **updates))
        rows = [{"green": True, "partition": analysis.resource_partition(self.resources(deform))}
                for deform in (4, 8)]
        with self.assertRaises(AssertionError):
            analysis.cohort_partition(rows)
        with self.assertRaises(AssertionError):
            analysis.cohort_partition(rows[:1], expected_deform_sms=8)

    def test_historical_partitioned_prepass_remains_distinct(self):
        resources = self.resources(4)
        resources.update(scope="prefix_only", raster_prepass_sms=12)
        self.assertEqual(analysis.resource_partition(resources)["raster_prepass_sms"], 12)


class GreenTraceDirectoryTests(unittest.TestCase):
    def test_default_and_resumed_cohort_paths_remain_distinct(self):
        root = Path("/tmp/green-analysis")
        self.assertEqual(analysis.trace_path(root), root / "green-trace")
        resumed = "green-trace-resume-20261010"
        self.assertEqual(analysis.trace_path(root, resumed), root / resumed)

    def test_trace_path_requires_one_child_directory(self):
        for name in ("", ".", "..", "../green-trace", "/tmp/green-trace", "nested/green-trace"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                analysis.trace_path(Path("/tmp/green-analysis"), name)


class GreenHandoffAnalysisTests(unittest.TestCase):
    def test_handoff_arm_requires_consistent_measured_metadata(self):
        import analyze_green_handoff as handoff
        for enabled in (False, True):
            row = {"green_resources": {"prepare_handoff_enabled": enabled,
                   "deform_sms": 8, "raster_prepare_sms": 8, "raster_prepass_sms": 16,
                   "tail_sms": 16, "raster_prepare_suffix_sms": 16 if enabled else 8,
                   "prepare_handoff_boundary": "after_duplicateWithKeys" if enabled else None},
                   "activity": {"config": {"green_prepare_handoff": enabled}}}
            self.assertEqual(handoff.arm(row), "handoff8_to16" if enabled else "fixed8_8")
            row["green_resources"]["raster_prepare_suffix_sms"] = 12
            with self.assertRaises(AssertionError):
                handoff.arm(row)

    def test_handoff_config_mismatch_is_rejected(self):
        import analyze_green_handoff as handoff
        row = {"green_resources": {"prepare_handoff_enabled": True},
               "activity": {"config": {"green_prepare_handoff": False}}}
        with self.assertRaises(AssertionError):
            handoff.arm(row)


if __name__ == "__main__":
    unittest.main()
