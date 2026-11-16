"""Standalone HexPlane CUDA Graph microbenchmark, not renderer qualification.

Compare the original 12 ATen samplers with the specialized sampler on the
same points/planes. ABBA timing excludes capture, Python dispatch, and NCU.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics

import torch

from tacker_4dgs_head.hexplane import resources


def load_hexplane():
    path = Path(__file__).resolve().parents[1] / "scene" / "hexplane.py"
    spec = importlib.util.spec_from_file_location("hexplane_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(function):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = function()
    return graph, output


def measure(graph, iterations):
    for _ in range(20):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=111525)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    module = load_hexplane()
    torch.manual_seed(20261007)
    axes = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    aabb = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        grids = []
        for level in range(2):
            planes = []
            for plane in range(6):
                suffix = f"grid.grids.{level}.{plane}"
                keys = [key for key in state if key.endswith(suffix)]
                if len(keys) != 1:
                    raise ValueError(f"Expected one {suffix}: {keys}")
                planes.append(state[keys[0]].cuda().contiguous())
            grids.append(planes)
        aabb = next(value for key, value in state.items() if key.endswith("grid.aabb")).cuda()
    else:
        grids = [[torch.rand(1, 16, 150 if ay == 3 else res, res, device="cuda")
                  * 0.8 + 0.1 for _, ay in axes] for res in (64, 128)]
    if args.points:
        import numpy as np
        from plyfile import PlyData
        vertices = PlyData.read(args.points)["vertex"]
        xyz = torch.from_numpy(np.stack([vertices[name] for name in ("x", "y", "z")], 1)).cuda()
        args.rows = xyz.shape[0]
        pts = torch.cat((module.normalize_aabb(xyz, aabb),
                         torch.full((args.rows, 1), 0.5, device="cuda")), dim=1)
    else:
        pts = torch.rand(args.rows, 4, device="cuda") * 2 - 1

    captured = {}
    for label in ("aten", "lowreg"):
        os.environ["TACKER_HEXPLANE_LOWREG"] = "1" if label == "lowreg" else "0"
        captured[label] = capture(lambda: module.interpolate_ms_features(
            pts, grids, grid_dimensions=2, concat_features=True,
            num_levels=None, capture_safe=True))
    for graph, _ in captured.values():
        graph.replay()
    torch.cuda.synchronize()
    expected, actual = captured["aten"][1], captured["lowreg"][1]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    difference = (actual - expected).abs()
    runs = []
    for label in ("aten", "lowreg", "lowreg", "aten") * 3:
        runs.append({"variant": label, "milliseconds": measure(captured[label][0], args.iterations)})
    medians = {label: statistics.median(run["milliseconds"] for run in runs
                                      if run["variant"] == label) for label in captured}
    result = {
        "scope": "standalone HexPlane CUDA Graph microbenchmark; no Raster, no renderer qualification",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "rows": args.rows, "iterations": args.iterations,
        "checkpoint": str(args.checkpoint), "points": str(args.points),
        "resources": resources(),
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "median_milliseconds": medians,
        "speedup": medians["aten"] / medians["lowreg"], "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
