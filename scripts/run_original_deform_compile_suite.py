"""Run independent, mirrored child processes for the four deformation arms."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from benchmark_original_deform_compile import MODES, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--phase", choices=("smoke", "quality", "formal"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "summary.json").exists(), "refusing to overwrite a completed suite"
    worker = Path(__file__).with_name("benchmark_original_deform_compile.py")
    schedule = list(MODES)
    if args.phase == "formal":
        schedule = [mode for i in range(args.rounds)
                    for mode in (MODES if i % 2 == 0 else tuple(reversed(MODES)))]
    records = []
    for index, mode in enumerate(schedule):
        stem = f"{index:03d}-{mode}"
        report_path, log_path = output / (stem + ".json"), output / (stem + ".log")
        assert not report_path.exists() and not log_path.exists(), stem
        command = [sys.executable, "-u", str(worker), "--project-root", args.project_root,
                   "--model-path", args.model_path, "--source-path", args.source_path,
                   "--physical-gpu", str(args.physical_gpu), "--mode", mode,
                   "--frames", "3" if args.phase == "smoke" else "50",
                   "--warmup", "3" if args.phase == "smoke" else "20",
                   "--trials", "1", "--output", str(report_path)]
        if args.phase == "quality":
            command.append("--quality")
        print("START", args.phase, stem, time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        with log_path.open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=1800, env=os.environ.copy(), cwd=args.project_root)
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if result.returncode or not report.get("passed"):
            write_json(output / "failure.json", {"index": index, "mode": mode,
                                                  "returncode": result.returncode, "report": report})
            print(log_path.read_text()[-12000:], flush=True)
            raise RuntimeError("child failed: " + stem)
        records.append({"index": index, "mode": mode, "path": str(report_path),
                        "measurements": report["measurements"], "quality": report["quality"],
                        "compile_setup_seconds": report["compile_setup_seconds"],
                        "graph_setup_seconds": report["graph_setup_seconds"]})
        write_json(output / "checkpoint.json", {"phase": args.phase, "records": records,
                                                 "schedule": schedule})
        print("DONE", stem, "deform_ms", report["measurements"]["deform"]["median_wall_ms"],
              "fps", report["measurements"]["end_to_end"]["median_fps"], flush=True)
    summary = {"passed": True, "phase": args.phase, "schedule": schedule, "records": records,
               "scope": "original serial deform; no mixed kernel or qualification profile",
               "compiler": "max-autotune-no-cudagraphs + independent manual CUDA Graph switch",
               "by_mode": {}}
    for mode in MODES:
        children = [r for r in records if r["mode"] == mode]
        summary["by_mode"][mode] = {"independent_processes": len(children)}
        for scope in ("deform", "end_to_end"):
            ms = [r["measurements"][scope]["median_wall_ms"] for r in children]
            fps = [r["measurements"][scope]["median_fps"] for r in children]
            summary["by_mode"][mode][scope] = {"median_wall_ms": statistics.median(ms),
                "median_fps": statistics.median(fps), "min_fps": min(fps), "max_fps": max(fps),
                "cv_percent": statistics.stdev(fps) / statistics.mean(fps) * 100 if len(fps) > 1 else None}
    for mode in MODES:
        for scope in ("deform", "end_to_end"):
            baseline = summary["by_mode"]["eager"][scope]
            target = summary["by_mode"][mode][scope]
            target["speedup_vs_eager"] = target["median_fps"] / baseline["median_fps"]
            target["throughput_gain_percent"] = (target["speedup_vs_eager"] - 1) * 100
            target["time_reduction_percent"] = (1 - target["median_wall_ms"] / baseline["median_wall_ms"]) * 100
    write_json(output / "summary.json", summary)
    print("COMPLETE", str(output / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
