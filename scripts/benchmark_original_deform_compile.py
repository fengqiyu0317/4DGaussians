"""Four-way inference comparison of the original PyTorch deformation network.

The compiler always has its own CUDA Graph support disabled.  The explicit
Graph switch is therefore independent of compilation, including for the
compiled+Graph arm.  No mixed Raster/MLP kernel is used.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback


MODES = ("eager", "eager_graph", "compile", "compile_graph")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def gpu_snapshot(physical_gpu, require_idle=False):
    executable = shutil.which("nvidia-smi")
    result = {"physical_gpu": physical_gpu}
    if executable:
        for key, query in (
            ("gpu", "--query-gpu=index,uuid,name,memory.used,utilization.gpu,clocks.sm,temperature.gpu"),
            ("processes", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"),
        ):
            command = [executable, query, "--format=csv"]
            if key == "gpu":
                command += ["-i", str(physical_gpu)]
            run = subprocess.run(command, capture_output=True, text=True, timeout=15)
            result[key] = run.stdout.strip()
            if run.returncode:
                raise RuntimeError(run.stderr or run.stdout)
        rows = list(csv.reader(result["gpu"].splitlines()))
        gpu = rows[1]
        uuid = gpu[1].strip()
        processes = list(csv.reader(result["processes"].splitlines()))[1:]
        if require_idle:
            assert not any(len(row) > 1 and row[1].strip().isdigit()
                           and row[0].strip() == uuid for row in processes), processes
            util = gpu[4].split()[0]
            if util.isdigit():
                assert int(util) <= 5, gpu
    frequency = Path("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu")
    if frequency.exists():
        result["jetson_frequency_hz"] = {
            name: int((frequency / name).read_text())
            for name in ("min_freq", "max_freq", "cur_freq")
        }
        assert set(result["jetson_frequency_hz"].values()) == {1300500000}, result
        if require_idle:
            run = subprocess.run(["timeout", "2", "tegrastats", "--interval", "500"],
                                 capture_output=True, text=True, timeout=5)
            result["tegrastats"] = run.stdout
            import re
            loads = re.findall(r"GR3D_FREQ (\d+)%", run.stdout)
            assert loads and all(int(load) == 0 for load in loads), run.stdout
    return result


def source_hashes(root, model):
    names = ("scene/deformation.py", "scene/hexplane.py", "gaussian_renderer/__init__.py",
             "scene/gaussian_model.py", "arguments/dynerf/default.py",
             "arguments/dynerf/flame_steak.py")
    paths = [root / name for name in names]
    paths += sorted((model / "point_cloud/iteration_14000").glob("*"))
    paths.append(Path(__file__).resolve())
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def compiler_compatibility_patch(torch):
    """Backport upstream PR 165066 in memory for the affected Torch 2.8 wheel.

    Identity-wrapped constant indices must be expanded with tl.full, not
    tl.broadcast_to on a Python scalar. No installed files are edited.
    https://github.com/pytorch/pytorch/pull/165066/files
    """
    if not torch.__version__.startswith("2.8."):
        return None
    import inspect
    import textwrap
    import sympy
    from torch._inductor.codegen import triton as codegen
    from torch.utils._sympy.functions import Identity

    def identity_sympystr(self, printer):
        return f"({printer.doprint(self.args[0])})"

    Identity._sympystr = identity_sympystr

    def integer_like(expression):
        return isinstance(expression, sympy.Expr) and (
            isinstance(expression, sympy.Integer)
            or (expression.is_integer and len(expression.free_symbols) == 0))

    codegen.__dict__["_experiment_integer_like"] = integer_like
    replacements = {"indexing": "isinstance(index, sympy.Integer)",
                    "load": "isinstance(original_index, sympy.Integer)"}
    for name, old in replacements.items():
        method = getattr(codegen.TritonKernel, name)
        source = textwrap.dedent(inspect.getsource(method))
        assert source.count(old) == 1, (name, "unexpected compiler source")
        variable = "original_index" if name == "load" else "index"
        source = source.replace(old, f"_experiment_integer_like({variable})")
        namespace = {}
        exec(compile(source, "<experiment-upstream-pr-165066>", "exec"), codegen.__dict__, namespace)
        setattr(codegen.TritonKernel, name, namespace[name])
    return {"upstream_pr": "https://github.com/pytorch/pytorch/pull/165066",
            "scope": "process-only TritonKernel.indexing/load and Identity._sympystr",
            "installed_files_modified": False}


def worker(args):
    before = gpu_snapshot(args.physical_gpu, require_idle=True)
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root))
    for key in list(os.environ):
        if key.startswith("TACKER_") or key.startswith("FOURDGS_NVTX"):
            os.environ.pop(key)
    os.environ["FOURDGS_NVTX"] = "0"
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    import torch
    from arguments import ModelParams, ModelHiddenParams, PipelineParams, get_combined_args
    from utils.params_utils import load_config, merge_hparams
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import prepare_render_context, rasterize_state, GaussianRenderState
    import scene.deformation as deformation_module

    # NVTX is disabled for every arm; use its equivalent no-op during tracing.
    deformation_module.nvtx_range = contextlib.nullcontext
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    hidden_params = ModelHiddenParams(parser)
    pipeline_params = PipelineParams(parser)
    saved_argv = sys.argv
    sys.argv = [sys.argv[0], "--model_path", args.model_path, "--source_path", args.source_path]
    try:
        model_args = get_combined_args(parser)
    finally:
        sys.argv = saved_argv
    model_args = merge_hparams(model_args, load_config(root / "arguments/dynerf/flame_steak.py"))
    dataset, hidden, pipeline = (model_params.extract(model_args), hidden_params.extract(model_args),
                                 pipeline_params.extract(model_args))
    assert not pipeline.compute_cov3D_python and not pipeline.convert_SHs_python
    source_before = source_hashes(root, Path(args.model_path))
    start_load = time.perf_counter()
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hidden)
        scene = Scene(dataset, gaussians, load_iteration=14000, shuffle=False)
        net = gaussians._deformation.eval()
        net.requires_grad_(False)
        views = [scene.getTestCameras()[i] for i in range(args.frames)]
        assert len(gaussians.get_xyz) == 111525
        assert (int(views[0].image_width), int(views[0].image_height)) == (1352, 1014)
        assert scene.loaded_iter == 14000 and scene.dataset_type == "dynerf"
        background = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.],
                                  device="cuda")
        first_context = prepare_render_context(views[0], gaussians, pipeline, background,
                                               cam_type=scene.dataset_type)
        fixed = (first_context.means3D, first_context.scales, first_context.rotations,
                 first_context.opacity, first_context.shs)
        timestamps = [torch.full_like(first_context.timestamp, float(view.time)) for view in views]
        original_forward = net.forward
        use_graph = args.mode.endswith("_graph")
        use_compile = args.mode.startswith("compile")
        grid = net.deformation_net.grid
        original_grid_forward = grid.forward
        graph_safe_grid_forward = grid.forward_cuda_graph
        # Graph capture cannot create the CUDA advanced-indexing tensor.  Both
        # compiled arms use the same capture-safe coordinate construction too.
        # Original eager remains the untouched reference; quality checks below
        # compare each optimized arm against that reference.
        def reference(timestamp):
            saved = grid.forward
            grid.forward = original_grid_forward
            try:
                return original_forward(*fixed, timestamp)
            finally:
                grid.forward = saved

        if use_graph or use_compile:
            grid.forward = graph_safe_grid_forward

        callable_ = original_forward
        compatibility_patch = None
        compile_start = time.perf_counter()
        if use_compile:
            import torch._inductor.config as inductor_config
            compatibility_patch = compiler_compatibility_patch(torch)
            inductor_config.triton.cudagraphs = False
            callable_ = torch.compile(original_forward, fullgraph=True, dynamic=False,
                                      mode="max-autotune-no-cudagraphs")
        # Preserve original parameter flags and input strides across warmup,
        # capture, deformation timing and full-frame rendering.
        for _ in range(3):
            callable_(*fixed, timestamps[0])
        torch.cuda.synchronize()
        compile_setup_seconds = time.perf_counter() - compile_start
        graph = None
        graph_output = None
        graph_timestamp = None
        replay_count = 0
        graph_setup_start = time.perf_counter()
        if use_graph:
            graph_timestamp = torch.empty_like(timestamps[0])
            graph_timestamp.copy_(timestamps[0])
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    callable_(*fixed, graph_timestamp)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                graph_output = callable_(*fixed, graph_timestamp)
            stream.synchronize()
        graph_setup_seconds = time.perf_counter() - graph_setup_start

        def run_deform(timestamp, inputs=fixed):
            nonlocal replay_count
            if graph is not None:
                # Model and Gaussian attributes are immutable in this experiment.
                # Only frame time changes; timestamp copy is INCLUDED in timing.
                graph_timestamp.copy_(timestamp, non_blocking=True)
                graph.replay()
                replay_count += 1
                return graph_output
            return callable_(*inputs, timestamp)

        def state_from(outputs):
            xyz, scale, rotation, opacity, shs = outputs
            return GaussianRenderState(xyz, gaussians.scaling_activation(scale),
                                       gaussians.rotation_activation(rotation),
                                       gaussians.opacity_activation(opacity), shs)

        def frame(view):
            context = prepare_render_context(view, gaussians, pipeline, background,
                                             cam_type=scene.dataset_type)
            inputs = (context.means3D, context.scales, context.rotations,
                      context.opacity, context.shs)
            return rasterize_state(context, state_from(run_deform(context.timestamp, inputs)))

        reference_output = reference(timestamps[0])
        candidate_output = run_deform(timestamps[0])
        smoke_errors = []
        for expected, actual in zip(reference_output, candidate_output):
            torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-3)
            assert torch.isfinite(actual).all()
            smoke_errors.append(float((expected - actual).abs().max()))
        del reference_output, candidate_output
        torch.cuda.synchronize()
        print("SETUP", args.mode, "compile_seconds", round(compile_setup_seconds, 3),
              "graph_seconds", round(graph_setup_seconds, 3), flush=True)

        quality = None
        if args.quality:
            from utils.image_utils import psnr
            from utils.loss_utils import ssim
            from lpipsPyTorch.modules.lpips import LPIPS
            lpips_model = LPIPS("vgg").eval().cuda()
            per_view = []
            for i, view in enumerate(views):
                context = prepare_render_context(view, gaussians, pipeline, background,
                                                 cam_type=scene.dataset_type)
                expected_outputs = reference(timestamps[i])
                actual_outputs = run_deform(timestamps[i])
                max_errors = []
                for expected, actual in zip(expected_outputs, actual_outputs):
                    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-3)
                    assert torch.isfinite(actual).all()
                    max_errors.append(float((expected - actual).abs().max()))
                expected_image = rasterize_state(context, state_from(expected_outputs)).render.clamp(0, 1)
                actual_image = rasterize_state(context, state_from(actual_outputs)).render.clamp(0, 1)
                ground_truth = view.original_image[:3].cuda().clamp(0, 1).unsqueeze(0)
                values = {"view": i, "max_abs_deform_errors": max_errors,
                          "image_max_abs_error": float((actual_image - expected_image).abs().max()),
                          "image_rmse": float((actual_image - expected_image).square().mean().sqrt())}
                for name, image in (("reference", expected_image), ("candidate", actual_image)):
                    batch = image.unsqueeze(0)
                    values[name] = {"psnr": float(psnr(batch, ground_truth).mean()),
                                    "ssim": float(ssim(batch, ground_truth)),
                                    "lpips": float(lpips_model(batch, ground_truth).mean())}
                per_view.append(values)
                if (i + 1) % 10 == 0:
                    print("QUALITY", args.mode, i + 1, "/", len(views), flush=True)
            means = {name: {key: statistics.mean(row[name][key] for row in per_view)
                             for key in ("psnr", "ssim", "lpips")}
                     for name in ("reference", "candidate")}
            delta = {key: means["candidate"][key] - means["reference"][key]
                     for key in means["reference"]}
            passed = delta["psnr"] >= -0.05 and delta["ssim"] >= -1e-4 and delta["lpips"] <= 1e-4
            quality = {"passed": passed, "means": means, "candidate_minus_reference": delta,
                       "max_abs_deform_errors": [max(row["max_abs_deform_errors"][j] for row in per_view)
                                                  for j in range(5)],
                       "max_image_rmse": max(row["image_rmse"] for row in per_view),
                       "per_view": per_view}
            assert passed, quality
            del lpips_model, expected_outputs, actual_outputs, expected_image, actual_image, ground_truth
            torch.cuda.synchronize()

        measurements = {}
        for scope in ("deform", "end_to_end"):
            def operation(index):
                return run_deform(timestamps[index]) if scope == "deform" else frame(views[index])
            for i in range(args.warmup):
                result = operation(i % len(views))
                del result
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            records = []
            for trial in range(args.trials):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                replay_before = replay_count
                start.record()
                wall_start = time.perf_counter()
                for i in range(len(views)):
                    result = operation(i)
                    del result
                end.record()
                end.synchronize()
                seconds = time.perf_counter() - wall_start
                records.append({"trial": trial, "wall_ms_per_frame": seconds * 1000 / len(views),
                                "throughput_fps": len(views) / seconds,
                                "cuda_event_ms_per_frame": start.elapsed_time(end) / len(views),
                                "graph_replays": replay_count - replay_before})
            measurements[scope] = {"trials": records,
                "median_wall_ms": statistics.median(row["wall_ms_per_frame"] for row in records),
                "median_fps": statistics.median(row["throughput_fps"] for row in records),
                "median_cuda_event_ms": statistics.median(row["cuda_event_ms_per_frame"] for row in records),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            print("TIMING", args.mode, scope, measurements[scope]["median_wall_ms"], flush=True)
        from torch._dynamo.utils import counters
        compiler_counters = {str(name): dict(value) for name, value in counters.items() if value}
        if use_compile:
            assert compiler_counters.get("stats", {}).get("unique_graphs", 0) >= 1, compiler_counters
            assert not compiler_counters.get("graph_break"), compiler_counters
        if use_graph:
            assert all(row["graph_replays"] == len(views) for measurement in measurements.values()
                       for row in measurement["trials"])
        source_after = source_hashes(root, Path(args.model_path))
        assert source_before == source_after, "runtime source or checkpoint changed during experiment"
        report = {"passed": True, "mode": args.mode, "platform": platform.machine(),
                  "gpu_name": torch.cuda.get_device_name(), "torch": torch.__version__,
                  "cuda_runtime": torch.version.cuda,
                  "triton": __import__("triton").__version__ if importlib.util.find_spec("triton") else None,
                  "precision": {"dtype": "float32", "matmul_precision": "highest", "allow_tf32": False},
                  "scope": "original positional encoding + HexPlane + backbone + five heads + residuals",
                  "renderer_activations": "unchanged eager exp/normalize/sigmoid outside optimization scope",
                  "mixed_kernel": False, "qualification_profile": None,
                  "compiler_mode": "max-autotune-no-cudagraphs" if use_compile else None,
                  "compiler_fullgraph": use_compile, "compiler_internal_cudagraphs": False,
                  "compiler_compatibility_patch": compatibility_patch,
                  "explicit_cuda_graph": use_graph, "graph_replay_count": replay_count,
                  "graph_input_policy": "fixed immutable model/Gaussian attributes; per-frame timestamp D2D copy included",
                  "capture_safe_coordinate_indexing": use_graph or use_compile,
                  "frames": len(views), "warmup_frames": args.warmup,
                  "compile_setup_seconds": compile_setup_seconds, "graph_setup_seconds": graph_setup_seconds,
                  "load_seconds": compile_start - start_load, "quality": quality,
                  "smoke_max_abs_deform_errors": smoke_errors, "measurements": measurements,
                  "compiler_counters": compiler_counters,
                  "gpu_before": before, "gpu_after": gpu_snapshot(args.physical_gpu),
                  "source_sha256": source_before, "source_stable": True}
        write_json(args.output, report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        worker(args)
    except Exception:
        write_json(args.output, {"passed": False, "mode": args.mode, "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
