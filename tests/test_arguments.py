"""Regression tests for command-line/cfg_args namespace merging."""

from argparse import ArgumentParser
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arguments import get_combined_args


class CombinedArgumentsTests(unittest.TestCase):
    @staticmethod
    def _parser():
        parser = ArgumentParser()
        parser.add_argument("--model-path")
        parser.add_argument("--tacker-profile")
        return parser

    def test_new_optional_argument_exists_when_older_cfg_omits_it(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "cfg_args").write_text(
                "Namespace(existing_value='from_cfg')", encoding="utf-8"
            )
            argv = ["program", "--model-path", str(model_path)]
            with mock.patch.object(sys, "argv", argv):
                args = get_combined_args(self._parser())

        self.assertTrue(hasattr(args, "tacker_profile"))
        self.assertIsNone(args.tacker_profile)
        self.assertEqual(args.existing_value, "from_cfg")

    def test_none_command_line_default_does_not_replace_cfg_value(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "cfg_args").write_text(
                "Namespace(tacker_profile='cfg-profile.json')", encoding="utf-8"
            )
            argv = ["program", "--model-path", str(model_path)]
            with mock.patch.object(sys, "argv", argv):
                args = get_combined_args(self._parser())

        self.assertEqual(args.tacker_profile, "cfg-profile.json")

    def test_explicit_command_line_value_replaces_cfg_value(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "cfg_args").write_text(
                "Namespace(tacker_profile='cfg-profile.json')", encoding="utf-8"
            )
            argv = [
                "program",
                "--model-path",
                str(model_path),
                "--tacker-profile",
                "cli-profile.json",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = get_combined_args(self._parser())

        self.assertEqual(args.tacker_profile, "cli-profile.json")


if __name__ == "__main__":
    unittest.main()
