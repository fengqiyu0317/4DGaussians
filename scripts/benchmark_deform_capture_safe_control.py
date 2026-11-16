"""Diagnostic: time the Graph-safe coordinate rewrite without compilation/Graph.

This supplementary control separates the required indexing rewrite from CUDA
Graph replay. It is not part of the four-arm independent-process benchmark.
"""
import argparse
import hashlib
import sys
from pathlib import Path

from benchmark_original_deform_compile import worker, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "model-path", "source-path", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    args.mode, args.frames, args.warmup, args.quality = "eager", 50, 20, False
    sys.path.insert(0, str(Path(args.project_root).resolve()))
    from scene.hexplane import HexPlaneField
    original = HexPlaneField.forward
    HexPlaneField.forward = HexPlaneField.forward_cuda_graph
    try:
        report = worker(args)
    finally:
        HexPlaneField.forward = original
    report["mode"] = "capture_safe_eager_control"
    report["capture_safe_coordinate_indexing"] = True
    report["diagnostic_only"] = True
    report["independent_processes"] = 1
    report["diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(args.output, report)


if __name__ == "__main__":
    main()
