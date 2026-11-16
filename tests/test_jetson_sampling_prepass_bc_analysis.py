import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('jetson_abc_analysis', scripts / 'analyze_jetson_sampling_prepass_bc.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def test_sampling_gain_and_fusion_regression_have_opposite_signs(self):
        result = analysis.comparisons({'A': 6., 'B': 5.5, 'C': 5.6})
        self.assertAlmostEqual(result['B_vs_A']['saved_ms'], .5)
        self.assertLess(result['C_vs_B']['time_saved_percent'], 0)
        self.assertGreater(result['C_vs_A']['time_saved_percent'], 0)

    def test_reject_missing_completion_foreign_work_and_changed_clocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            phase = Path(temporary)
            with self.assertRaises(ValueError):
                analysis.validate_phase(phase)
            (phase / 'completed.txt').write_text('complete')
            (phase / 'protocol.json').write_text(json.dumps({'hardware': {'GPU_clock_hz': 1300500000}}))
            (phase / 'runtime.sha256.json').write_text('{}')
            job = phase / 'run0'
            job.mkdir()
            (job / 'completed.txt').write_text('complete')
            (job / 'command.json').write_text('[]')
            (job / 'audit_summary.json').write_text('{"passed": true}')
            valid = {'foreign': [], 'clocks': {'cur_freq': 1300500000}}
            for row in (dict(valid, foreign=['foreign GPU job']),
                        dict(valid, clocks={'cur_freq': 1000000000})):
                (job / 'during.audit.jsonl').write_text(json.dumps(row) + '\n')
                with self.assertRaises(ValueError):
                    analysis.validate_phase(phase)
            (job / 'during.audit.jsonl').write_text(json.dumps(valid) + '\n')
            self.assertEqual(analysis.validate_phase(phase)[1], {})


if __name__ == '__main__':
    unittest.main()
