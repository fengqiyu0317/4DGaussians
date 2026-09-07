#!/usr/bin/env python3
"""Validate serial/two_stream/tacker render quality on one fixed view batch.

GPU-heavy imports are intentionally delayed until :func:`run_validation`, so
``--help`` and CPU-side source checks do not require PyTorch or CUDA.  The
script never enables the repository's Tacker profile implicitly: callers must
both request the ``tacker`` mode and pass ``--tacker-profile``.
"""

from __future__ import print_function

from argparse import ArgumentParser
import datetime
import json
import math
import os
from pathlib import Path
import sys
import tempfile


SCHEMA_VERSION = 1
EXPECTED_SCENE = "flame_steak"
EXPECTED_ITERATION = 14000
EXPECTED_RESOLUTION = [1352, 1014]
EXPECTED_GAUSSIANS = 111525
QUALITY_THRESHOLDS = {
    "psnr_drop_db_max": 0.05,
    "ssim_drop_max": 0.0001,
    "lpips_increase_max": 0.0001,
}

# Direct execution sets sys.path[0] to ``scripts/``.  Add only this script's
# resolved repository root; no caller-controlled string is evaluated.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_write_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _minimal_failure(error, args=None):
    qualification_profile = getattr(args, "qualification_profile", None)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "4dgaussians_tacker_quality_validation",
        "generated_at_utc": _utc_now(),
        "passed": False,
        "workload": {
            "scene": EXPECTED_SCENE,
            "iteration": EXPECTED_ITERATION,
            "resolution": list(EXPECTED_RESOLUTION),
            "gaussian_count": EXPECTED_GAUSSIANS,
        },
        "thresholds": dict(QUALITY_THRESHOLDS),
        "modes": {},
        "deltas": {},
        "gates": [],
        "errors": [str(error)],
        "qualification": {
            "enabled": bool(getattr(args, "qualification_mode", False)),
            "admission_claimed": False,
            "profile_override": (
                str(Path(qualification_profile).expanduser().resolve())
                if qualification_profile is not None
                else None
            ),
        },
    }
    if args is not None:
        report["requested_modes"] = list(getattr(args, "modes", ()))
    return report


def _select_views(scene, split):
    if split == "train":
        return scene.getTrainCameras()
    if split == "test":
        return scene.getTestCameras()
    return scene.getVideoCameras()


def _selected_indices(total, start, stride, frames):
    if frames <= 0:
        raise ValueError("--frames must be positive")
    if start < 0:
        raise ValueError("--view-start must be non-negative")
    if stride <= 0:
        raise ValueError("--view-stride must be positive")
    indices = [start + offset * stride for offset in range(frames)]
    if not indices or indices[-1] >= total:
        raise ValueError(
            "requested view batch ends at index {} but split has {} views".format(
                indices[-1] if indices else start, total
            )
        )
    return indices


def _ground_truth(view, cam_type, torch):
    if cam_type == "PanopticSports":
        raise ValueError("PanopticSports is outside the admitted Tacker workload")
    ground_truth = view.original_image[:3, :, :]
    if not ground_truth.is_cuda:
        ground_truth = ground_truth.cuda(non_blocking=True)
    return ground_truth.clamp(0.0, 1.0)


def _renderer(
    mode,
    gaussians,
    pipeline,
    background,
    cam_type,
    profile_path,
    qualification_mode,
    qualification_profile,
):
    if mode == "serial":
        return None
    if mode == "two_stream":
        from gaussian_renderer import TwoStreamRenderer

        return TwoStreamRenderer(
            gaussians,
            pipeline,
            background,
            stage="fine",
            cam_type=cam_type,
        )
    if mode == "tacker":
        if qualification_mode:
            if qualification_profile is None:
                raise ValueError(
                    "qualification mode requires --qualification-profile"
                )
            with Path(qualification_profile).expanduser().open(
                "r", encoding="utf-8"
            ) as handle:
                profile_override = json.load(handle)
            if not isinstance(profile_override, dict):
                raise ValueError("qualification profile must be a JSON object")
        else:
            profile_override = None
        if profile_path is None and not qualification_mode:
            raise ValueError(
                "tacker mode requires an explicit --tacker-profile; no profile "
                "is enabled by default"
            )
        from gaussian_renderer.tacker_pipeline import TackerRenderer

        return TackerRenderer(
            gaussians,
            pipeline,
            background,
            stage="fine",
            cam_type=cam_type,
            profile_path=profile_path,
            profile_override=profile_override,
            workload_name=EXPECTED_SCENE,
            iteration=EXPECTED_ITERATION,
            qualification_mode=qualification_mode,
        )
    raise ValueError("unknown render mode: {}".format(mode))


def _mode_outputs(mode, views, gaussians, pipeline, background, cam_type, renderer):
    if mode == "serial":
        from gaussian_renderer import render

        for view in views:
            yield render(
                view,
                gaussians,
                pipeline,
                background,
                stage="fine",
                cam_type=cam_type,
            )
        return

    iterator = renderer.render_sequence(views)
    try:
        for _ in views:
            try:
                yield next(iterator)
            except StopIteration:
                raise RuntimeError(
                    "{} renderer returned fewer frames than requested".format(mode)
                )
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                "{} renderer returned more frames than requested".format(mode)
            )
    finally:
        iterator.close()


def _actual_mode(mode, renderer):
    if mode == "serial":
        return "serial", None
    if mode == "two_stream":
        reason = renderer.last_fallback_reason
        return ("two_stream" if reason is None else "serial_fallback"), reason
    public_mode = getattr(renderer, "actual_execution_mode", None)
    if public_mode in ("tacker", "two_stream", "serial"):
        primary_reason = getattr(renderer, "last_fallback_reason", None)
        backend_reason = getattr(renderer, "fallback_backend_reason", None)
        reasons = [
            reason
            for reason in (primary_reason, backend_reason)
            if isinstance(reason, str) and reason
        ]
        return public_mode, "; ".join(reasons) if reasons else None
    if bool(getattr(renderer, "last_used_tacker", False)):
        return "tacker", None
    reason = getattr(renderer, "last_fallback_reason", None)
    fallback = getattr(renderer, "_fallback_renderer", None)
    nested_reason = getattr(fallback, "last_fallback_reason", None)
    if nested_reason is None:
        return "two_stream_fallback", reason
    return "serial_fallback", "{}; two_stream fallback: {}".format(
        reason, nested_reason
    )


def _evaluate_mode(
    mode,
    views,
    gaussians,
    pipeline,
    background,
    cam_type,
    lpips_model,
    lpips_net,
    profile_path,
    qualification_mode,
    qualification_profile,
    torch,
):
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    renderer = _renderer(
        mode,
        gaussians,
        pipeline,
        background,
        cam_type,
        profile_path,
        qualification_mode,
        qualification_profile,
    )
    per_view = []
    outputs = iter(_mode_outputs(
        mode, views, gaussians, pipeline, background, cam_type, renderer
    ))
    try:
        for local_index, view in enumerate(views):
            try:
                output = next(outputs)
            except StopIteration:
                raise RuntimeError(
                    "{} renderer returned fewer frames than requested".format(mode)
                )
            rendered = output["render"].clamp(0.0, 1.0)
            ground_truth = _ground_truth(view, cam_type, torch)
            rendered_batch = rendered.unsqueeze(0)
            ground_truth_batch = ground_truth.unsqueeze(0)
            values = {
                "batch_index": local_index,
                "psnr_db": float(
                    psnr(rendered_batch, ground_truth_batch).mean().item()
                ),
                "ssim": float(ssim(rendered_batch, ground_truth_batch).item()),
                "lpips": float(
                    lpips_model(rendered_batch, ground_truth_batch).mean().item()
                ),
            }
            if not all(
                math.isfinite(value)
                for key, value in values.items()
                if key != "batch_index"
            ):
                raise RuntimeError(
                    "{} produced a non-finite quality metric".format(mode)
                )
            per_view.append(values)
        try:
            next(outputs)
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                "{} renderer returned more frames than requested".format(mode)
            )
    finally:
        outputs.close()
    if len(per_view) != len(views):
        raise RuntimeError("{} metric count does not match the view batch".format(mode))
    if renderer is not None and callable(getattr(renderer, "synchronize", None)):
        renderer.synchronize()
    torch.cuda.synchronize()
    actual_mode, fallback_reason = _actual_mode(mode, renderer)
    means = {
        key: sum(item[key] for item in per_view) / float(len(per_view))
        for key in ("psnr_db", "ssim", "lpips")
    }
    return {
        "requested_mode": mode,
        "actual_mode": actual_mode,
        "fallback_reason": fallback_reason,
        "qualification_requested": bool(
            mode == "tacker" and qualification_mode
        ),
        "qualification_executed": bool(
            mode == "tacker"
            and getattr(renderer, "last_qualification_mode", False)
        ),
        "lpips_net": lpips_net,
        "means": means,
        "per_view": per_view,
    }


def run_validation(args, dataset, hyperparam, pipeline):
    import torch

    from gaussian_renderer import GaussianModel
    from lpipsPyTorch.modules.lpips import LPIPS
    from scene import Scene

    if args.scene_name.replace("-", "_").lower() != EXPECTED_SCENE:
        raise ValueError("--scene-name must be flame_steak")
    if args.iteration != EXPECTED_ITERATION:
        raise ValueError("--iteration must be 14000")
    modes = list(dict.fromkeys(args.modes))
    if "serial" not in modes:
        raise ValueError("serial must be included as the quality reference")
    if args.qualification_mode and "tacker" not in modes:
        raise ValueError("--qualification-mode requires tacker in --modes")
    if args.qualification_mode and args.qualification_profile is None:
        raise ValueError(
            "--qualification-mode requires an explicit --qualification-profile"
        )
    if not args.qualification_mode and args.qualification_profile is not None:
        raise ValueError(
            "--qualification-profile is accepted only with --qualification-mode"
        )
    if args.tacker_profile is not None and args.qualification_profile is not None:
        raise ValueError(
            "--tacker-profile and --qualification-profile are mutually exclusive"
        )
    if (
        "tacker" in modes
        and args.tacker_profile is None
        and not args.qualification_mode
    ):
        raise ValueError(
            "requesting tacker requires an explicit --tacker-profile"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this validation on 4A6000")
    capability = list(torch.cuda.get_device_capability(args.gpu))
    gpu_name = torch.cuda.get_device_name(args.gpu)
    if capability != [8, 6] or gpu_name != "NVIDIA RTX A6000":
        raise RuntimeError(
            "quality admission requires RTX A6000/sm_86, got {} / {}".format(
                gpu_name, capability
            )
        )

    torch.cuda.set_device(args.gpu)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=args.iteration,
            shuffle=False,
        )
        if scene.loaded_iter != EXPECTED_ITERATION:
            raise RuntimeError("loaded checkpoint is not iteration 14000")
        if scene.dataset_type != "dynerf":
            raise RuntimeError("flame_steak admission requires the dynerf loader")
        gaussians._deformation.eval()
        all_views = _select_views(scene, args.split)
        indices = _selected_indices(
            len(all_views), args.view_start, args.view_stride, args.frames
        )
        views = [all_views[index] for index in indices]
        first_view = views[0]
        resolution = [int(first_view.image_width), int(first_view.image_height)]
        gaussian_count = int(gaussians.get_xyz.shape[0])
        if resolution != EXPECTED_RESOLUTION:
            raise RuntimeError(
                "admission resolution must be {}, got {}".format(
                    EXPECTED_RESOLUTION, resolution
                )
            )
        if gaussian_count != EXPECTED_GAUSSIANS:
            raise RuntimeError(
                "admission Gaussian count must be {}, got {}".format(
                    EXPECTED_GAUSSIANS, gaussian_count
                )
            )

        background = torch.tensor(
            [1, 1, 1] if dataset.white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
        lpips_model = LPIPS(args.lpips_net, "0.1").cuda().eval()
        mode_results = {}
        for mode in modes:
            mode_results[mode] = _evaluate_mode(
                mode,
                views,
                gaussians,
                pipeline,
                background,
                scene.dataset_type,
                lpips_model,
                args.lpips_net,
                args.tacker_profile,
                args.qualification_mode,
                args.qualification_profile,
                torch,
            )

    serial = mode_results["serial"]["means"]
    deltas = {}
    gates = []
    errors = []
    for mode in modes:
        if mode == "serial":
            continue
        means = mode_results[mode]["means"]
        delta = {
            "psnr_drop_db": serial["psnr_db"] - means["psnr_db"],
            "ssim_drop": serial["ssim"] - means["ssim"],
            "lpips_increase": means["lpips"] - serial["lpips"],
        }
        deltas[mode] = delta
        expected_actual = mode
        qualification_ok = bool(
            not (mode == "tacker" and args.qualification_mode)
            or mode_results[mode]["qualification_executed"]
        )
        actual_ok = (
            mode_results[mode]["actual_mode"] == expected_actual
            and qualification_ok
        )
        quality_ok = (
            delta["psnr_drop_db"] <= QUALITY_THRESHOLDS["psnr_drop_db_max"]
            and delta["ssim_drop"] <= QUALITY_THRESHOLDS["ssim_drop_max"]
            and delta["lpips_increase"]
            <= QUALITY_THRESHOLDS["lpips_increase_max"]
        )
        gates.append(
            {
                "mode": mode,
                "actual_mode": mode_results[mode]["actual_mode"],
                "actual_mode_passed": actual_ok,
                "qualification_passed": qualification_ok,
                "quality_passed": quality_ok,
                "passed": actual_ok and quality_ok,
            }
        )
        if not actual_ok:
            errors.append(
                "{} fell back to {}: {}".format(
                    mode,
                    mode_results[mode]["actual_mode"],
                    mode_results[mode]["fallback_reason"],
                )
            )
        if not quality_ok:
            errors.append("{} exceeded at least one quality threshold".format(mode))

    passed = bool(gates) and all(gate["passed"] for gate in gates)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "4dgaussians_tacker_quality_validation",
        "generated_at_utc": _utc_now(),
        "passed": passed,
        "workload": {
            "scene": EXPECTED_SCENE,
            "iteration": scene.loaded_iter,
            "split": args.split,
            "frames": args.frames,
            "view_indices": indices,
            "resolution": resolution,
            "gaussian_count": gaussian_count,
            "model_path": str(Path(dataset.model_path).resolve()),
            "source_path": str(Path(dataset.source_path).resolve()),
        },
        "device": {
            "name": gpu_name,
            "index": args.gpu,
            "compute_capability": capability,
            "cuda_arch": "sm_86",
            "cuda_runtime": torch.version.cuda,
            "pytorch_version": str(torch.__version__),
        },
        "thresholds": dict(QUALITY_THRESHOLDS),
        "modes": mode_results,
        "deltas": deltas,
        "gates": gates,
        "errors": errors,
        "qualification": {
            "enabled": bool(args.qualification_mode),
            "admission_claimed": bool(
                not args.qualification_mode and "tacker" in modes and passed
            ),
            "profile_override": (
                str(Path(args.qualification_profile).expanduser().resolve())
                if args.qualification_profile is not None
                else None
            ),
        },
        "tacker_profile": (
            str(Path(args.tacker_profile).expanduser().resolve())
            if args.tacker_profile is not None
            else None
        ),
    }


def _build_parser():
    from arguments import ModelHiddenParams, ModelParams, PipelineParams

    parser = ArgumentParser(
        description="Validate 4DGaussians serial/two_stream/tacker image quality"
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", type=int, default=EXPECTED_ITERATION)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--scene-name", default=EXPECTED_SCENE)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--view-start", type=int, default=0)
    parser.add_argument("--view-stride", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("serial", "two_stream", "tacker"),
        default=("serial", "two_stream"),
    )
    parser.add_argument("--tacker-profile")
    parser.add_argument(
        "--qualification-mode",
        action="store_true",
        help=(
            "explicitly run qualification-only Tacker bootstrap; this is not "
            "an admission claim and requires --qualification-profile"
        ),
    )
    parser.add_argument("--qualification-profile")
    parser.add_argument("--lpips-net", choices=("alex", "vgg"), default="vgg")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser, model, hyperparam, pipeline


def main():
    parser, model, hyperparam, pipeline = _build_parser()
    from arguments import get_combined_args

    args = get_combined_args(parser)
    try:
        from utils.general_utils import safe_state
        from utils.params_utils import load_config, merge_hparams

        if args.configs:
            args = merge_hparams(args, load_config(args.configs))
        safe_state(args.quiet)
        report = run_validation(
            args,
            model.extract(args),
            hyperparam.extract(args),
            pipeline.extract(args),
        )
    except Exception as error:
        report = _minimal_failure(error, args=args)
    atomic_write_json(args.output, report)
    print(
        "Tacker quality validation {}. Report: {}".format(
            "passed" if report["passed"] else "failed",
            Path(args.output).expanduser().resolve(),
        )
    )
    if report["errors"]:
        for error in report["errors"]:
            print("- {}".format(error))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
