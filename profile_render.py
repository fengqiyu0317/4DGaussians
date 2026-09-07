"""Render-only benchmark entry point for Nsight Systems profiling."""

from argparse import ArgumentParser
import json
from pathlib import Path
import statistics
from time import perf_counter

import torch

from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import (
    GaussianModel,
    TackerRenderer,
    TwoStreamRenderer,
    deform_for_render,
    prepare_render_context,
    rasterize_state,
    render,
)
from scene import Scene
from utils.general_utils import safe_state
from utils.params_utils import load_config, merge_hparams
from utils.profiling_utils import nvtx_range


def select_views(scene, split):
    if split == "train":
        return scene.getTrainCameras()
    if split == "test":
        return scene.getTestCameras()
    return scene.getVideoCameras()


def run_views(
    views,
    execution_mode,
    gaussians,
    pipeline,
    background,
    render_kwargs,
    pipeline_renderer=None,
    completion_events=None,
):
    """Submit frames without copying or encoding any rendered output."""
    fallback_reason = None
    if execution_mode in ("two_stream", "tacker"):
        if pipeline_renderer is None:
            if execution_mode == "two_stream":
                pipeline_renderer = TwoStreamRenderer(
                    gaussians,
                    pipeline,
                    background,
                    stage=render_kwargs["stage"],
                    cam_type=render_kwargs["cam_type"],
                )
            else:
                raise ValueError("tacker mode requires a configured TackerRenderer")
        iterator = pipeline_renderer.render_sequence(views)
        try:
            for frame_index in range(len(views)):
                with nvtx_range("profile/frame_{:04d}".format(frame_index)):
                    next(iterator)
                    if completion_events is not None:
                        completion = torch.cuda.Event(enable_timing=True)
                        completion.record(torch.cuda.current_stream())
                        completion_events.append(completion)
            # Resume once past the final yield so the renderer's non-reentrancy
            # guard is released before this instance is reused after warmup.
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError("render_sequence yielded more frames than requested")
        finally:
            iterator.close()
        pipeline_renderer.synchronize()
        fallback_reason = pipeline_renderer.last_fallback_reason
        return fallback_reason

    for frame_index, view in enumerate(views):
        with nvtx_range("profile/frame_{:04d}".format(frame_index)):
            if execution_mode == "serial":
                render(view, gaussians, pipeline, background, **render_kwargs)
            elif execution_mode == "split_serial":
                context = prepare_render_context(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    cam_type=render_kwargs["cam_type"],
                )
                state = deform_for_render(
                    context,
                    gaussians,
                    stage=render_kwargs["stage"],
                )
                rasterize_state(context, state)
            else:
                raise ValueError("unknown execution mode: {}".format(execution_mode))
            if completion_events is not None:
                completion = torch.cuda.Event(enable_timing=True)
                completion.record(torch.cuda.current_stream())
                completion_events.append(completion)
    return fallback_reason


def main(args, dataset, hyperparam, pipeline):
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.qualification_mode and args.execution_mode != "tacker":
        raise ValueError("--qualification-mode requires --execution-mode tacker")
    if args.qualification_mode and args.qualification_profile is None:
        raise ValueError("--qualification-mode requires --qualification-profile")
    if not args.qualification_mode and args.qualification_profile is not None:
        raise ValueError(
            "--qualification-profile is accepted only with --qualification-mode"
        )
    if args.tacker_profile is not None and args.qualification_profile is not None:
        raise ValueError(
            "--tacker-profile and --qualification-profile are mutually exclusive"
        )
    if args.execution_mode == "tacker":
        if args.workload_name is None:
            raise ValueError("tacker mode requires --workload-name")
        if args.tacker_profile is None and not args.qualification_mode:
            raise ValueError(
                "tacker mode requires --tacker-profile or explicit qualification mode"
            )
    elif args.tacker_profile is not None:
        raise ValueError("--tacker-profile requires --execution-mode tacker")

    qualification_override = None
    if args.qualification_mode:
        try:
            qualification_override = json.loads(
                Path(args.qualification_profile).expanduser().read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError) as error:
            raise ValueError("cannot load qualification profile: {}".format(error))
        if not isinstance(qualification_override, dict):
            raise ValueError("qualification profile must be a JSON object")

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        gaussians._deformation.eval()
        views = select_views(scene, args.split)
        if len(views) == 0:
            raise RuntimeError("the selected split has no cameras")

        view_indices = [index % len(views) for index in range(args.frames)]
        selected = [views[index] for index in view_indices]
        warmup_views = [views[index % len(views)] for index in range(args.warmup)]

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_kwargs = {"stage": "fine", "cam_type": scene.dataset_type}
        pipeline_renderer = None
        if args.execution_mode == "two_stream":
            # Stream/event construction is setup, not per-frame rendering, and
            # therefore remains outside the timed region.
            pipeline_renderer = TwoStreamRenderer(
                gaussians,
                pipeline,
                background,
                stage=render_kwargs["stage"],
                cam_type=render_kwargs["cam_type"],
            )
        elif args.execution_mode == "tacker":
            pipeline_renderer = TackerRenderer(
                gaussians,
                pipeline,
                background,
                stage=render_kwargs["stage"],
                cam_type=render_kwargs["cam_type"],
                profile_path=args.tacker_profile,
                profile_override=qualification_override,
                workload_name=args.workload_name,
                iteration=scene.loaded_iter,
                qualification_mode=args.qualification_mode,
            )

        # Stream/event creation and immutable parameter conversion are setup,
        # even when --warmup=0.  Keep both outside CUDA-event/perf timing.
        if pipeline_renderer is not None:
            prepare = getattr(pipeline_renderer, "prepare", None)
            if callable(prepare):
                prepare()

        run_views(
            warmup_views,
            args.execution_mode,
            gaussians,
            pipeline,
            background,
            render_kwargs,
            pipeline_renderer=pipeline_renderer,
        )
        torch.cuda.synchronize()

        torch.cuda.cudart().cudaProfilerStart()
        completion_start = torch.cuda.Event(enable_timing=True)
        completion_events = []
        completion_start.record(torch.cuda.current_stream())
        start_time = perf_counter()
        with nvtx_range("profile/render_loop"):
            fallback_reason = run_views(
                selected,
                args.execution_mode,
                gaussians,
                pipeline,
                background,
                render_kwargs,
                pipeline_renderer=pipeline_renderer,
                completion_events=completion_events,
            )
        torch.cuda.synchronize()
        elapsed_seconds = perf_counter() - start_time
        torch.cuda.cudart().cudaProfilerStop()

        if len(completion_events) != args.frames:
            raise RuntimeError("frame completion event count does not match --frames")
        cumulative_ms = [
            float(completion_start.elapsed_time(event))
            for event in completion_events
        ]
        frame_completion_ms = []
        previous_ms = 0.0
        for value in cumulative_ms:
            frame_completion_ms.append(value - previous_ms)
            previous_ms = value

        actual_execution_mode = args.execution_mode
        two_stream_fallback_reason = None
        tacker_fallback_reason = None
        qualification_executed = False
        persistent_blocks = None
        profile_manifest_sha256 = None
        if args.execution_mode == "two_stream":
            two_stream_fallback_reason = fallback_reason
            if fallback_reason is not None:
                actual_execution_mode = "serial"
        elif args.execution_mode == "tacker":
            actual_execution_mode = pipeline_renderer.actual_execution_mode
            tacker_fallback_reason = pipeline_renderer.last_fallback_reason
            two_stream_fallback_reason = pipeline_renderer.fallback_backend_reason
            qualification_executed = pipeline_renderer.last_qualification_mode
            persistent_blocks = pipeline_renderer.persistent_blocks
            if pipeline_renderer.profile is not None:
                profile_manifest_sha256 = pipeline_renderer.profile.get(
                    "manifest_sha256"
                )

        first_view = selected[0]
        metadata = {
            "schema_version": 1,
            "kind": "4dgaussians_tacker_render_profile",
            "passed": True,
            "model_path": str(Path(dataset.model_path).resolve()),
            "source_path": str(Path(dataset.source_path).resolve()),
            "iteration": scene.loaded_iter,
            "split": args.split,
            "warmup_frames": args.warmup,
            "profile_frames": args.frames,
            "view_indices": view_indices,
            "execution_mode": args.execution_mode,
            "actual_execution_mode": actual_execution_mode,
            "two_stream_fallback_reason": two_stream_fallback_reason,
            "tacker_fallback_reason": tacker_fallback_reason,
            "qualification_mode_requested": bool(args.qualification_mode),
            "qualification_mode_executed": bool(qualification_executed),
            "workload_name": args.workload_name,
            "tacker_profile": (
                str(Path(args.tacker_profile).expanduser().resolve())
                if args.tacker_profile is not None
                else None
            ),
            "qualification_profile": (
                str(Path(args.qualification_profile).expanduser().resolve())
                if args.qualification_profile is not None
                else None
            ),
            "profile_manifest_sha256": profile_manifest_sha256,
            "persistent_blocks": persistent_blocks,
            "elapsed_seconds": elapsed_seconds,
            "mean_frame_ms": elapsed_seconds * 1000.0 / args.frames,
            "cuda_event_mean_frame_ms": sum(frame_completion_ms) / args.frames,
            "p50_frame_ms": statistics.median(frame_completion_ms),
            "frame_completion_ms": frame_completion_ms,
            "throughput_fps": args.frames / elapsed_seconds,
            "timing_method": "perf_counter_with_cuda_synchronize",
            "frame_timing_method": "cuda_event_consumer_completion_intervals",
            "io_in_timed_region": False,
            "gaussian_count": int(gaussians.get_xyz.shape[0]),
            "image_width": int(first_view.image_width),
            "image_height": int(first_view.image_height),
            "dataset_type": scene.dataset_type,
            "convert_SHs_python": pipeline.convert_SHs_python,
            "compute_cov3D_python": pipeline.compute_cov3D_python,
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "pytorch_version": str(torch.__version__),
            "pipeline_slot_count": (
                2
                if actual_execution_mode in ("two_stream", "tacker")
                else 1
            ),
        }
        if args.metadata:
            Path(args.metadata).write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(
            "Profiled {profile_frames} {split} frames in {actual_execution_mode} mode "
            "at iteration {iteration}: {throughput_fps:.2f} FPS, "
            "{mean_frame_ms:.3f} ms/frame, {gaussian_count} Gaussians "
            "({image_width}x{image_height})".format(**metadata)
        )
        if tacker_fallback_reason is not None:
            print("tacker fell back: {}".format(tacker_fallback_reason))
        if two_stream_fallback_reason is not None:
            print(
                "two_stream backend fell back to serial: {}".format(
                    two_stream_fallback_reason
                )
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Nsight render-only profiling")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--split", choices=("train", "test", "video"), default="test")
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--frames", default=50, type=int)
    parser.add_argument(
        "--execution-mode",
        "--execution_mode",
        dest="execution_mode",
        choices=("serial", "split_serial", "two_stream", "tacker"),
        default="serial",
    )
    parser.add_argument("--tacker-profile", type=str)
    parser.add_argument("--qualification-mode", action="store_true")
    parser.add_argument("--qualification-profile", type=str)
    parser.add_argument("--workload-name", type=str)
    parser.add_argument("--metadata", type=str)
    parser.add_argument("--quiet", action="store_true")

    parsed = get_combined_args(parser)
    if parsed.configs:
        parsed = merge_hparams(parsed, load_config(parsed.configs))

    safe_state(parsed.quiet)
    main(
        parsed,
        model.extract(parsed),
        hyperparam.extract(parsed),
        pipeline.extract(parsed),
    )
