import json
from pathlib import Path
import tempfile
import unittest

from scripts import collect_multiscene_tacker_correctness as module


def _report(scene="coffee_martini", mode="two_stream", passed=True):
    modes = {"serial": {"actual_mode": "serial"}}
    modes[mode] = {"actual_mode": mode}
    return {
        "kind": "4dgaussians_tacker_quality_validation",
        "passed": passed,
        "workload": {
            "scene": scene,
            "iteration": 14000,
            "split": "test",
            "frames": 50,
            "view_indices": list(range(50)),
            "resolution": [1352, 1014],
            "gaussian_count": 100,
        },
        "modes": modes,
        "gates": [{"mode": mode, "passed": passed}],
        "deltas": {mode: {"psnr_drop_db": 0.0}},
    }


class CorrectnessCollectorTest(unittest.TestCase):
    def test_collects_valid_and_invalid_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            valid = root / "valid.json"
            invalid = root / "invalid.json"
            baseline.write_text(json.dumps(_report()), encoding="utf-8")
            valid.write_text(json.dumps(_report(mode="tacker")), encoding="utf-8")
            invalid.write_text(
                json.dumps(_report(mode="tacker", passed=False)), encoding="utf-8"
            )

            result = module.collect(
                baseline,
                [("valid", valid), ("invalid", invalid)],
            )

            self.assertTrue(result["candidates"]["serial"]["valid"])
            self.assertTrue(result["candidates"]["valid"]["valid"])
            self.assertFalse(result["candidates"]["invalid"]["valid"])
            self.assertFalse(result["candidates"]["current_tacker"]["valid"])

    def test_rejects_workload_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps(_report()), encoding="utf-8")
            candidate.write_text(
                json.dumps(_report(scene="other", mode="tacker")), encoding="utf-8"
            )

            with self.assertRaisesRegex(module.CorrectnessError, "workload"):
                module.collect(baseline, [("candidate", candidate)])


if __name__ == "__main__":
    unittest.main()
