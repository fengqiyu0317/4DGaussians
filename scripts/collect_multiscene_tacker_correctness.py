#!/usr/bin/env python3
"""Build benchmark correctness input from per-scene quality reports."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


class CorrectnessError(ValueError):
    pass


def _load(path):
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise CorrectnessError("{} must contain a JSON object".format(resolved))
    return resolved, raw, value


def _candidate(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    return name, path


def _gate(report, mode):
    gates = report.get("gates")
    if not isinstance(gates, list):
        return None
    matches = [item for item in gates if item.get("mode") == mode]
    return matches[0] if len(matches) == 1 else None


def _mode_actual(report, mode):
    modes = report.get("modes")
    if not isinstance(modes, dict) or not isinstance(modes.get(mode), dict):
        return None
    return modes[mode].get("actual_mode")


def _same_workload(left, right):
    keys = ("scene", "iteration", "split", "frames", "view_indices", "resolution", "gaussian_count")
    return all(left.get(key) == right.get(key) for key in keys)


def _evidence(path, raw, report, gate):
    return {
        "quality_report": str(path),
        "quality_report_sha256": hashlib.sha256(raw).hexdigest(),
        "quality_report_passed": report.get("passed") is True,
        "gate": gate,
        "deltas": report.get("deltas"),
    }


def atomic_write_new(path, value):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise CorrectnessError("refusing to overwrite {}".format(target))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target.name), suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, str(target))
        os.unlink(temporary_name)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def collect(baseline_path, candidate_specs):
    baseline_file, baseline_raw, baseline = _load(baseline_path)
    if baseline.get("kind") != "4dgaussians_tacker_quality_validation":
        raise CorrectnessError("baseline has the wrong report kind")
    baseline_workload = baseline.get("workload")
    if not isinstance(baseline_workload, dict):
        raise CorrectnessError("baseline omitted workload")
    two_stream_gate = _gate(baseline, "two_stream")
    baseline_valid = bool(
        baseline.get("passed") is True
        and _mode_actual(baseline, "serial") == "serial"
        and _mode_actual(baseline, "two_stream") == "two_stream"
        and isinstance(two_stream_gate, dict)
        and two_stream_gate.get("passed") is True
    )
    if not baseline_valid:
        raise CorrectnessError("serial/two_stream baseline quality report failed")

    baseline_evidence = _evidence(
        baseline_file, baseline_raw, baseline, two_stream_gate
    )
    result = {
        "candidates": {
            "serial": {"valid": True, "evidence": baseline_evidence},
            "two_stream": {"valid": True, "evidence": baseline_evidence},
            "current_tacker": {
                "valid": False,
                "evidence": "historical incumbent was not requalified for this scene",
            },
        }
    }
    for name, path in candidate_specs:
        candidate_file, candidate_raw, report = _load(path)
        workload = report.get("workload")
        if not isinstance(workload, dict) or not _same_workload(
            baseline_workload, workload
        ):
            raise CorrectnessError(
                "candidate {} workload does not match the baseline".format(name)
            )
        gate = _gate(report, "tacker")
        valid = bool(
            report.get("passed") is True
            and _mode_actual(report, "serial") == "serial"
            and _mode_actual(report, "tacker") == "tacker"
            and isinstance(gate, dict)
            and gate.get("passed") is True
        )
        result["candidates"][name] = {
            "valid": valid,
            "evidence": _evidence(candidate_file, candidate_raw, report, gate),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", type=_candidate, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = collect(args.baseline, args.candidate)
    atomic_write_new(args.output, result)
    print(Path(args.output).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
