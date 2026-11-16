#!/usr/bin/env python3
"""Compare inference backends on one checkpoint and an identical view sequence.

Raster-only uses exactly the same cached activated Gaussian states. End-to-end
includes context setup, HexPlane, MLP, activations and rendering; disk I/O,
loading, JIT compilation, warmup and quality metrics are outside both timers.
CUDA event spans include host-submission gaps, not just summed kernel activity.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODES = ("diff_serial", "gsplat_dense", "gsplat_packed", "diff_two_stream",
         "gsplat_two_stream", "tacker_graph", "gsplat_matched_graph", "gsplat_mixed_graph")
THRESHOLDS = {"psnr_drop_db": 0.05, "ssim_drop": 1e-4, "lpips_increase": 1e-4}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def mean(records):
    return {key: statistics.mean(r[key] for r in records) for key in records[0]}


def summarize(records):
    return {key: statistics.median(r[key] for r in records)
            for key in ("wall_ms_per_frame", "cuda_span_ms_per_frame", "fps")}


def main():
    from arguments import ModelParams, ModelHiddenParams, PipelineParams, get_combined_args
    from utils.params_utils import load_config, merge_hparams

    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    hidden = ModelHiddenParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--iteration", type=int, default=14000)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--view-start", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES[:5]))
    parser.add_argument("--mixed-blocks", type=int, default=0)
    parser.add_argument("--qualification-profile", type=Path)
    parser.add_argument("--workload-name", default="flame_steak")
    parser.add_argument("--quality-only", action="store_true")
    args = merge_hparams(get_combined_args(parser), load_config(parser.parse_args().configs))
    if args.frames <= 0 or args.trials <= 0 or args.warmup < 0 or args.view_start < 0:
        parser.error("frames/trials must be positive; warmup/view-start must be nonnegative")
    if "diff_serial" not in args.modes:
        parser.error("diff_serial is required as the reference")
    args.modes = ["diff_serial"] + list(dict.fromkeys(m for m in args.modes if m != "diff_serial"))
    if "tacker_graph" in args.modes and args.qualification_profile is None:
        parser.error("tacker_graph requires --qualification-profile")
    if any(m in args.modes for m in ("gsplat_matched_graph", "gsplat_mixed_graph")):
        if args.qualification_profile is None or "gsplat_dense" not in args.modes:
            parser.error("gsplat Graph modes require --qualification-profile and gsplat_dense reference")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        parser.error("refusing to overwrite an existing report.json")

    import torch
    import gsplat
    from gaussian_renderer import (GaussianModel, TackerRenderer, TwoStreamRenderer,
                                   prepare_render_context, deform_for_render, rasterize_state, render)
    from scene import Scene
    from utils.general_utils import safe_state
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    from lpipsPyTorch.modules.lpips import LPIPS
    from torchvision.utils import save_image

    safe_state(False)
    dataset = model.extract(args)
    base_pipe = pipeline_params.extract(args)
    base_pipe.raster_backend = "diff_gaussian"
    base_pipe.gsplat_packed = False
    torch.set_grad_enabled(False)
    gaussians = GaussianModel(dataset.sh_degree, hidden.extract(args))
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    if scene.loaded_iter != args.iteration:
        raise RuntimeError("wrong checkpoint iteration loaded")
    gaussians._deformation.eval()
    cameras = scene.getTestCameras()
    indices = list(range(args.view_start, args.view_start + args.frames))
    if indices[-1] >= len(cameras):
        raise ValueError("requested view sequence exceeds test cameras")
    views = [cameras[i] for i in indices]
    resolution = [views[0].image_width, views[0].image_height]
    assert all([v.image_width, v.image_height] == resolution for v in views)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    checkpoint = Path(dataset.model_path) / "point_cloud" / ("iteration_%d" % args.iteration)
    fingerprints = {str(p): sha256(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}
    source_paths = [ROOT / "gaussian_renderer" / "__init__.py",
                    ROOT / "gaussian_renderer" / "gsplat_backend.py", Path(__file__),
                    ROOT / "gaussian_renderer" / "tacker_pipeline.py",
                    ROOT / "scene" / "deformation.py", ROOT / "scene" / "hexplane.py",
                    ROOT / "arguments" / "__init__.py", Path(args.configs)]
    if args.qualification_profile:
        source_paths.append(args.qualification_profile)
    if any(m in args.modes for m in ("gsplat_matched_graph", "gsplat_mixed_graph")):
        source_paths.extend([ROOT / "gaussian_renderer" / "gsplat_mixed.py",
                             ROOT / "gsplat_mixed_ext" / "mixed.cu",
                             ROOT / "gsplat_mixed_ext" / "bindings.cpp",
                             ROOT / "gsplat_mixed_ext" / "__init__.py",
                             ROOT / "tacker_ext" / "include" / "deform_mlp_device.cuh"])
    source_before = {str(p.resolve()): sha256(p) for p in source_paths}
    report = {
        "kind": "4dgaussians_gsplat_backend_comparison", "schema_version": 1,
        "workload": {"scene": args.workload_name, "model_path": dataset.model_path,
                     "source_path": dataset.source_path, "iteration": args.iteration,
                     "checkpoint_sha256": fingerprints, "split": "test", "view_indices": indices,
                     "resolution": resolution, "gaussian_count": gaussians.get_xyz.shape[0]},
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "cuda": torch.version.cuda, "gsplat": gsplat.__version__,
                        "gpu": torch.cuda.get_device_name(), "platform": platform.platform(),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "experiment_environment": {k: v for k, v in os.environ.items()
                                                   if k.startswith(("TACKER_", "FOURDGS_"))},
                        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True)},
        "settings": {"radius_clip": 0.0, "rasterize_mode": "classic", "eps2d": 0.3,
                     "near_plane": 0.2, "render_mode": "RGB+D", "lpips_net": "vgg",
                     "warmup_frames": args.warmup, "trials": args.trials,
                     "end_to_end_includes": "context setup + deformation + activation + RGB/depth rendering + final synchronization",
                     "excluded": "checkpoint/view loading, JIT, warmup, image I/O, quality metrics",
                     "raster_only": "shared cached activated Gaussian states; camera contexts prepared before timing",
                     "cuda_span": "CUDA events; includes GPU idle gaps during host submission",
                     "schedule": "all candidates forward, then reverse, alternating per trial"},
        "source_sha256": source_before, "quality_thresholds": THRESHOLDS,
        "raster_only": {}, "end_to_end": {}, "quality": {},
    }
    pipes, renderers = {}, {}
    for mode in args.modes:
        pipe = copy.copy(base_pipe)
        pipe.raster_backend = "gsplat" if mode.startswith("gsplat") else "diff_gaussian"
        pipe.gsplat_packed = mode == "gsplat_packed"
        pipes[mode] = pipe
        if mode.endswith("two_stream"):
            renderers[mode] = TwoStreamRenderer(gaussians, pipe, background, cam_type=scene.dataset_type)
        elif mode == "tacker_graph":
            qualification = json.loads(args.qualification_profile.read_text())
            renderers[mode] = TackerRenderer(
                gaussians, pipe, background, cam_type=scene.dataset_type,
                profile_override=qualification, qualification_mode=True,
                workload_name=args.workload_name, iteration=args.iteration)
            renderers[mode].prepare()
        elif mode in ("gsplat_matched_graph", "gsplat_mixed_graph"):
            from gaussian_renderer.gsplat_mixed import GsplatMixedRenderer
            renderers[mode] = GsplatMixedRenderer(
                gaussians, pipe, background, args.qualification_profile,
                workload_name=args.workload_name, iteration=args.iteration,
                physical_blocks=args.mixed_blocks, fused=mode == "gsplat_mixed_graph",
                cam_type=scene.dataset_type, kernel_variant="v1")
            renderers[mode].prepare()

    def outputs(mode, selected_views):
        renderer = renderers.get(mode)
        if renderer is None:
            for view in selected_views:
                yield render(view, gaussians, pipes[mode], background, cam_type=scene.dataset_type)
        else:
            yield from renderer.render_sequence(selected_views)
            if renderer.last_fallback_reason is not None:
                raise RuntimeError("%s fell back: %s" % (mode, renderer.last_fallback_reason))
            if mode == "tacker_graph":
                if renderer.actual_execution_mode != "tacker" or not renderer.last_qualification_mode:
                    raise RuntimeError("Graph qualification did not execute")
                if renderer.profile.get("selected_variant_id") != "full_mlp_overlap_graph_pb0":
                    raise RuntimeError("unexpected selected Graph variant")
                if renderer.last_execution_counts["mixed_launches"] != max(0, len(selected_views) - 1):
                    raise RuntimeError("Graph mixed launch count does not match the sequence")
            if mode in ("gsplat_matched_graph", "gsplat_mixed_graph"):
                counts = renderer.last_execution_counts
                expected = max(0, len(selected_views) - 1)
                assert counts["prefix_graph_replays"] == expected, counts
                assert counts["mixed_launches"] == (expected if renderer.fused else 0), counts
                assert counts["outputs"] == len(selected_views), counts

    def consume(mode, selected_views):
        for _ in outputs(mode, selected_views):
            pass

    def measure(callback, count):
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        before = perf_counter()
        callback()
        end.record()
        torch.cuda.synchronize()
        elapsed = perf_counter() - before
        return {"wall_ms_per_frame": elapsed * 1000 / count,
                "cuda_span_ms_per_frame": start.elapsed_time(end) / count,
                "fps": count / elapsed}

    warmup_views = [views[i % len(views)] for i in range(args.warmup)]
    for mode in args.modes:
        print("WARMUP", mode, flush=True)
        # Always initialize lazy kernels once outside the timing interval.
        consume(mode, views[:min(3, len(views))])
        if warmup_views:
            consume(mode, warmup_views)
        torch.cuda.synchronize()

    if "gsplat_mixed_graph" in renderers:
        # Real-scene leaf diagnostic: freeze inputs, compare BOTH outputs,
        # and use subgroup clock64 windows to verify overlap within a CTA.
        from gaussian_renderer.gsplat_mixed import prepare_gsplat
        from gsplat.cuda._wrapper import rasterize_to_pixels
        from tacker_4dgs_head import deform_mlp_delta_v1
        mixed_renderer = renderers["gsplat_mixed_graph"]
        context = prepare_render_context(views[0], gaussians, pipes["gsplat_dense"],
                                         background, cam_type=scene.dataset_type)
        state = deform_for_render(context, gaussians)
        prepared = prepare_gsplat(context, state)
        features = gaussians._deformation.deformation_net.grid(context.means3D, context.timestamp).contiguous()
        mlp = [features, *mixed_renderer.cached]
        r = prepared.inputs
        expected_image, expected_alpha = rasterize_to_pixels(*r[:4], prepared.width,
            prepared.height, 16, r[4], r[5], backgrounds=r[6])
        expected_mlp = deform_mlp_delta_v1(*mlp)
        raw = mixed_renderer.extension.forward(r, mlp, prepared.width, prepared.height,
                                               args.mixed_blocks, 2, True)
        leaf_error = {"image_max_abs": float((raw[0] - expected_image).abs().max()),
                      "alpha_max_abs": float((raw[1] - expected_alpha).abs().max()),
                      "mlp_max_abs": [float((a-b).abs().max()) for a,b in zip(raw[3:8], expected_mlp)]}
        torch.testing.assert_close(raw[0], expected_image, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(raw[1], expected_alpha, atol=2e-6, rtol=2e-6)
        for a,b in zip(raw[3:8], expected_mlp):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        windows = raw[8].cpu().tolist()
        fractions = []
        for rs,re,ds,de in windows:
            if rs and ds and min(re-rs,de-ds) > 0:
                fractions.append(max(0,min(re,de)-max(rs,ds))/min(re-rs,de-ds))
        assert fractions and max(fractions) > 0, "no subgroup overlap observed"
        def native_leaf():
            rasterize_to_pixels(*r[:4], prepared.width, prepared.height, 16,
                                r[4],r[5],backgrounds=r[6])
            deform_mlp_delta_v1(*mlp)
        def mixed_leaf():
            mixed_renderer.extension.forward(r,mlp,prepared.width,prepared.height,args.mixed_blocks)
        leaf_trials = {"native_separate": [], "mixed": []}
        for i in range(10):
            arms = [("native_separate",native_leaf),("mixed",mixed_leaf)]
            for label,callback in (arms if i%2 == 0 else list(reversed(arms))):
                leaf_trials[label].append(measure(callback,1))
        report["mixed_leaf"] = {"errors": leaf_error,
            "kernel_symbol": "gsplat_mix_render_deform_v1", "kernel_launches_per_mixed_leaf": 1,
            "paired_ctas": len(fractions), "fraction_ctas_with_overlap": sum(f>0 for f in fractions)/len(fractions),
            "median_overlap_fraction_of_shorter_subgroup": statistics.median(fractions),
            "trials": leaf_trials, "median": {k:summarize(v) for k,v in leaf_trials.items()}}
        del context,state,prepared,features,mlp,r,expected_image,expected_alpha,expected_mlp,raw
        torch.cuda.synchronize()

    if not args.quality_only:
        # Measure normal inference before allocating any cached state tensors.
        for mode in args.modes:
            report["end_to_end"][mode] = {"trials": []}
        for trial in range(args.trials):
            order = args.modes if trial % 2 == 0 else list(reversed(args.modes))
            for mode in order:
                result = measure(lambda: consume(mode, views), len(views))
                report["end_to_end"][mode]["trials"].append(result)
                print("E2E", trial, mode, result, flush=True)
        for mode, value in report["end_to_end"].items():
            value["median"] = summarize(value["trials"])
            if mode in renderers:
                renderer = renderers[mode]
                value["execution_counts"] = getattr(renderer, "last_execution_counts", None)
                value["selected_variant_id"] = renderer.profile.get("selected_variant_id") if mode == "tacker_graph" else None
                value["qualification_executed"] = getattr(renderer, "last_qualification_mode", False)
                value["qualification_profile"] = str(args.qualification_profile) if mode == "tacker_graph" else None
                if mode in ("gsplat_matched_graph", "gsplat_mixed_graph"):
                    value.update(parent_graph_qualification=renderer.qualification_profile,
                                 parent_graph_qualification_sha256=renderer.qualification_sha256,
                                 physical_fusion=renderer.fused,
                                 mixed_resources=renderer.extension.resources(),
                                 physical_blocks=args.mixed_blocks,
                                 hexplane_graph_slot_count=len(renderer.graphs))
        write_json(args.output / "timing_partial.json", report)

        raster_modes = [m for m in args.modes if m in ("diff_serial", "gsplat_dense", "gsplat_packed")]
        contexts = {m: [prepare_render_context(v, gaussians, pipes[m], background,
                                              cam_type=scene.dataset_type) for v in views]
                    for m in raster_modes}
        states = [deform_for_render(c, gaussians) for c in contexts["diff_serial"]]
        torch.cuda.synchronize()

        def raster_sequence(mode):
            for context, state in zip(contexts[mode], states):
                rasterize_state(context, state)

        for mode in raster_modes:
            raster_sequence(mode)
            report["raster_only"][mode] = {"trials": []}
        for trial in range(args.trials):
            order = raster_modes if trial % 2 == 0 else list(reversed(raster_modes))
            for mode in order:
                result = measure(lambda: raster_sequence(mode), len(views))
                report["raster_only"][mode]["trials"].append(result)
                print("RASTER", trial, mode, result, flush=True)
        for value in report["raster_only"].values():
            value["median"] = summarize(value["trials"])
        del contexts, states
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        write_json(args.output / "timing_partial.json", report)

    lpips_model = LPIPS("vgg", "0.1").cuda().eval()
    reference = []
    for mode in args.modes:
        print("QUALITY", mode, flush=True)
        rows = []
        for i, output in enumerate(outputs(mode, views)):
            raw = output["render"]
            depth = output["depth"]
            if not torch.isfinite(raw).all() or not torch.isfinite(depth).all():
                raise RuntimeError("nonfinite image/depth from " + mode)
            image = raw.clamp(0, 1)
            gt = views[i].original_image[:3].cuda()
            row = {"view_index": indices[i],
                   "psnr_db": float(psnr(image[None], gt[None]).mean()),
                   "ssim": float(ssim(image[None], gt[None])),
                   "lpips": float(lpips_model(image[None], gt[None]).mean())}
            if mode == "diff_serial":
                reference.append({"image": image.cpu(), "depth": depth.cpu(),
                                  "radii": output["radii"].cpu()})
            else:
                ref = reference[i]
                delta = image - ref["image"].cuda()
                depth_delta = depth - ref["depth"].cuda()
                row.update({"rgb_mae_vs_reference": float(delta.abs().mean()),
                            "rgb_max_abs_vs_reference": float(delta.abs().max()),
                            "rgb_rmse_vs_reference": float(delta.square().mean().sqrt()),
                            "depth_mae_vs_reference": float(depth_delta.abs().mean()),
                            "depth_max_abs_vs_reference": float(depth_delta.abs().max()),
                            "visibility_disagreement_fraction": float(((output["radii"] > 0) !=
                                                                       (ref["radii"].cuda() > 0)).float().mean())})
                if mode == "gsplat_dense" and i in (0, len(views) // 2, len(views) - 1):
                    # Left to right: GT, legacy, gsplat, absolute difference x10.
                    panel = torch.cat((gt, ref["image"].cuda(), image, delta.abs() * 10), dim=2)
                    save_image(panel, args.output / ("comparison_view_%03d.png" % indices[i]))
            rows.append(row)
        values = mean([{k: v for k, v in r.items() if k != "view_index"} for r in rows])
        report["quality"][mode] = {"means": values, "per_view": rows}
        if mode != "diff_serial":
            baseline = report["quality"]["diff_serial"]["means"]
            drops = {"psnr_drop_db": baseline["psnr_db"] - values["psnr_db"],
                     "ssim_drop": baseline["ssim"] - values["ssim"],
                     "lpips_increase": values["lpips"] - baseline["lpips"]}
            report["quality"][mode].update({"delta_vs_reference": drops,
                                            "quality_gate_passed": all(drops[k] <= v for k, v in THRESHOLDS.items())})
            if mode in ("gsplat_matched_graph", "gsplat_mixed_graph"):
                gs = report["quality"]["gsplat_dense"]["means"]
                drops_gs = {"psnr_drop_db": gs["psnr_db"] - values["psnr_db"],
                            "ssim_drop": gs["ssim"] - values["ssim"],
                            "lpips_increase": values["lpips"] - gs["lpips"]}
                report["quality"][mode].update(delta_vs_gsplat=drops_gs,
                    quality_gate_vs_gsplat_passed=all(drops_gs[k] <= v for k, v in THRESHOLDS.items()))
        write_json(args.output / "quality_partial.json", report)

    # Record binaries actually imported, including JIT gsplat, then verify all
    # source and checkpoint hashes after the run. No runtime inputs are edited.
    binaries = {str(Path(module.__file__).resolve()) for module in list(sys.modules.values())
                if (getattr(module, "__file__", "") or "").endswith(".so")
                and any(name in (getattr(module, "__file__", "") or "")
                        for name in ("gsplat", "diff_gaussian", "tacker_4dgs", "simple_knn"))}
    report["runtime_binary_sha256"] = {p: sha256(p) for p in sorted(binaries)}
    report["inputs_unchanged"] = all(sha256(p) == h for p, h in source_before.items()) and all(
        sha256(p) == h for p, h in fingerprints.items())
    report["completed"] = True
    report["all_quality_gates_passed"] = all(v.get("quality_gate_passed", True)
                                               for v in report["quality"].values())
    if not report["inputs_unchanged"]:
        raise RuntimeError("source/checkpoint changed during comparison")
    write_json(args.output / "report.json", report)
    print("DONE", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()
