"""Render-only benchmark entry point for Nsight Systems profiling."""

from argparse import ArgumentParser
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
from time import perf_counter

import torch
import diff_gaussian_rasterization as _rasterizer_module

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


def _percentile(values, percentile):
    """Return a linearly interpolated percentile for a non-empty sequence."""

    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def _safe_command(command, cwd=None, timeout=5):
    """Run a metadata-only command without making profiling depend on it."""

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, str(error)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "exit status {}".format(
            completed.returncode
        )
        return None, detail
    # Preserve leading spaces: ``git submodule status`` uses its first byte as
    # a state marker, including a literal space for a clean recorded gitlink.
    return completed.stdout.rstrip("\r\n"), None


def _safe_file_sha256(path, errors, label):
    if path is None:
        return None
    try:
        digest = hashlib.sha256()
        with Path(path).expanduser().open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError) as error:
        errors.append("{}: {}".format(label, error))
        return None


def _atomic_write_json(path, value):
    """Atomically replace a metadata file with finite, fsynced JSON."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target.name), suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(target))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _parse_submodule_status(output):
    submodules = []
    for line in output.splitlines():
        if not line:
            continue
        state = line[0]
        fields = line[1:].strip().split()
        if len(fields) < 2:
            continue
        submodules.append(
            {
                "path": fields[1],
                "commit": fields[0],
                "source": "git",
                "status": {
                    " ": "recorded",
                    "-": "uninitialized",
                    "+": "different_commit",
                    "U": "merge_conflict",
                }.get(state, "unknown"),
            }
        )
    return submodules


def _environment_submodule_commits():
    mappings = (
        (
            "submodules/depth-diff-gaussian-rasterization",
            "FOURDGS_RASTERIZER_COMMIT",
        ),
        ("submodules/simple-knn", "FOURDGS_SIMPLE_KNN_COMMIT"),
    )
    return [
        {
            "path": path,
            "commit": os.environ.get(variable),
            "source": "environment" if os.environ.get(variable) else None,
            "status": "provided" if os.environ.get(variable) else "unavailable",
        }
        for path, variable in mappings
    ]


def _collect_repository_metadata(repo_root, errors):
    metadata = {
        "source_tree": str(repo_root),
        "commit": None,
        "commit_source": None,
        "dirty": None,
        "git_error": None,
        "submodules": [],
    }
    if not (repo_root / ".git").exists():
        git_error = "source tree has no .git metadata"
        metadata["git_error"] = git_error
        errors.append("git commit: {}".format(git_error))
        source_commit = os.environ.get("FOURDGS_SOURCE_COMMIT")
        if source_commit:
            metadata["commit"] = source_commit
            metadata["commit_source"] = "environment"
        metadata["submodules"] = _environment_submodule_commits()
        return metadata

    commit, error = _safe_command(("git", "rev-parse", "HEAD"), cwd=repo_root)
    if error is None:
        metadata["commit"] = commit
        metadata["commit_source"] = "git"
    else:
        metadata["git_error"] = error
        errors.append("git commit: {}".format(error))
        source_commit = os.environ.get("FOURDGS_SOURCE_COMMIT")
        if source_commit:
            metadata["commit"] = source_commit
            metadata["commit_source"] = "environment"

    status, status_error = _safe_command(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=repo_root,
    )
    if status_error is None:
        metadata["dirty"] = bool(status)
    else:
        errors.append("git status: {}".format(status_error))
        if metadata["git_error"] is None:
            metadata["git_error"] = status_error

    submodule_output, submodule_error = _safe_command(
        ("git", "submodule", "status", "--recursive"), cwd=repo_root
    )
    if submodule_error is None:
        metadata["submodules"] = _parse_submodule_status(submodule_output)
    else:
        errors.append("git submodule status: {}".format(submodule_error))
    known_paths = {item["path"] for item in metadata["submodules"]}
    metadata["submodules"].extend(
        item
        for item in _environment_submodule_commits()
        if item["path"] not in known_paths
    )
    return metadata


_NVIDIA_SMI_FIELDS = (
    "index",
    "uuid",
    "name",
    "driver_version",
    "pstate",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "temperature.gpu",
    "power.management",
    "power.draw",
    "power.limit",
)


def _nvidia_smi_metadata(logical_device_index, errors):
    command = (
        "nvidia-smi",
        "--query-gpu={}".format(",".join(_NVIDIA_SMI_FIELDS)),
        "--format=csv,noheader,nounits",
    )
    output, error = _safe_command(command)
    if error is not None:
        errors.append("nvidia-smi: {}".format(error))
        return None

    rows = []
    try:
        for values in csv.reader(io.StringIO(output)):
            if len(values) != len(_NVIDIA_SMI_FIELDS):
                continue
            rows.append(
                {
                    key: value.strip()
                    for key, value in zip(_NVIDIA_SMI_FIELDS, values)
                }
            )
    except (csv.Error, UnicodeError) as parse_error:
        errors.append("nvidia-smi output: {}".format(parse_error))
        return None
    if not rows:
        errors.append("nvidia-smi: query returned no parseable GPU rows")
        return None

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    selector = None
    if visible_devices:
        selectors = [item.strip() for item in visible_devices.split(",")]
        if logical_device_index < len(selectors):
            selector = selectors[logical_device_index]

    selected = None
    if selector is not None:
        for row in rows:
            if row["index"] == selector or row["uuid"] == selector:
                selected = row
                break
    elif logical_device_index < len(rows):
        selected = rows[logical_device_index]
    if selected is None and len(rows) == 1:
        selected = rows[0]
    if selected is None:
        errors.append(
            "nvidia-smi: cannot map logical CUDA device {} to a physical GPU".format(
                logical_device_index
            )
        )
    return selected


def _collect_environment_metadata(errors):
    environment = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "pytorch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn_version": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": None,
        "nvidia_smi": None,
    }
    try:
        environment["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as error:  # Metadata must never invalidate a measured run.
        errors.append("cuDNN version: {}".format(error))

    logical_device_index = 0
    gpu = {}
    try:
        logical_device_index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(logical_device_index)
        gpu = {
            "logical_device_index": logical_device_index,
            "name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
        }
    except Exception as error:  # Metadata must never invalidate a measured run.
        errors.append("CUDA device properties: {}".format(error))
    environment["gpu"] = gpu or None
    environment["nvidia_smi"] = _nvidia_smi_metadata(
        logical_device_index, errors
    )
    if environment["nvidia_smi"] is not None:
        environment["cuda_driver_version"] = environment["nvidia_smi"].get(
            "driver_version"
        )
    else:
        environment["cuda_driver_version"] = None
    return environment


def _collect_reproducibility_metadata(
    args,
    profile_manifest_sha256,
    tacker_profile_sha256_snapshot=None,
    qualification_profile_sha256_snapshot=None,
):
    """Snapshot provenance before warmup/timing; return partial data on failure."""

    errors = []
    repo_root = Path(__file__).resolve().parent
    try:
        repository = _collect_repository_metadata(repo_root, errors)
    except Exception as error:  # Keep optional provenance best-effort.
        errors.append("repository metadata: {}".format(error))
        repository = {
            "source_tree": str(repo_root),
            "commit": os.environ.get("FOURDGS_SOURCE_COMMIT"),
            "commit_source": (
                "environment" if os.environ.get("FOURDGS_SOURCE_COMMIT") else None
            ),
            "dirty": None,
            "git_error": str(error),
            "submodules": _environment_submodule_commits(),
        }
    try:
        environment = _collect_environment_metadata(errors)
    except Exception as error:  # Keep optional provenance best-effort.
        errors.append("environment metadata: {}".format(error))
        environment = {
            "pytorch_version": str(getattr(torch, "__version__", "unknown")),
            "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_driver_version": None,
        }

    profile_render_sha256 = _safe_file_sha256(
        Path(__file__).resolve(), errors, "profile_render.py SHA-256"
    )
    source_files = {
        "profile_render.py": profile_render_sha256,
        "gaussian_renderer/__init__.py": _safe_file_sha256(
            repo_root / "gaussian_renderer" / "__init__.py",
            errors,
            "gaussian_renderer/__init__.py SHA-256",
        ),
        "gaussian_renderer/tacker_pipeline.py": _safe_file_sha256(
            repo_root / "gaussian_renderer" / "tacker_pipeline.py",
            errors,
            "gaussian_renderer/tacker_pipeline.py SHA-256",
        ),
        "diff_gaussian_rasterization/__init__.py": _safe_file_sha256(
            getattr(_rasterizer_module, "__file__", None),
            errors,
            "diff_gaussian_rasterization/__init__.py SHA-256",
        ),
        "diff_gaussian_rasterization._C": _safe_file_sha256(
            getattr(getattr(_rasterizer_module, "_C", None), "__file__", None),
            errors,
            "diff_gaussian_rasterization/_C SHA-256",
        ),
        "configs": _safe_file_sha256(args.configs, errors, "config SHA-256"),
    }
    repository["source_files"] = source_files

    tacker_profile_sha256 = tacker_profile_sha256_snapshot
    if args.tacker_profile is not None and tacker_profile_sha256 is None:
        tacker_profile_sha256 = _safe_file_sha256(
            args.tacker_profile, errors, "tacker profile SHA-256"
        )
    qualification_profile_sha256 = qualification_profile_sha256_snapshot
    if (
        args.qualification_profile is not None
        and qualification_profile_sha256 is None
    ):
        qualification_profile_sha256 = _safe_file_sha256(
            args.qualification_profile, errors, "qualification profile SHA-256"
        )
    active_profile_sha256 = (
        tacker_profile_sha256
        if args.tacker_profile is not None
        else qualification_profile_sha256
    )
    profile_hashes = {
        "active_profile_sha256": active_profile_sha256,
        "tacker_profile_sha256": tacker_profile_sha256,
        "qualification_profile_sha256": qualification_profile_sha256,
        "profile_manifest_sha256": profile_manifest_sha256,
    }
    return {
        "environment": environment,
        "repository": repository,
        "source_files": source_files,
        "profile_hashes": profile_hashes,
        "metadata_collection_errors": errors,
    }


def _load_profile_snapshot(path, label):
    """Load and hash the exact profile bytes used for this process."""

    try:
        raw = Path(path).expanduser().read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("cannot load {} profile: {}".format(label, error))
    if not isinstance(value, dict):
        raise ValueError("{} profile must be a JSON object".format(label))
    return value, hashlib.sha256(raw).hexdigest()


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


def _measure_trial(
    trial_index,
    views,
    execution_mode,
    gaussians,
    pipeline,
    background,
    render_kwargs,
    pipeline_renderer,
):
    """Measure one complete sequence with independent wall/CUDA clocks."""

    completion_start = torch.cuda.Event(enable_timing=True)
    completion_end = torch.cuda.Event(enable_timing=True)
    completion_events = []
    completion_start.record(torch.cuda.current_stream())
    start_time = perf_counter()
    with nvtx_range("profile/trial_{:04d}".format(trial_index)):
        with nvtx_range("profile/render_loop"):
            fallback_reason = run_views(
                views,
                execution_mode,
                gaussians,
                pipeline,
                background,
                render_kwargs,
                pipeline_renderer=pipeline_renderer,
                completion_events=completion_events,
            )
    completion_end.record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    elapsed_seconds = perf_counter() - start_time

    if len(completion_events) != len(views):
        raise RuntimeError("frame completion event count does not match --frames")
    cumulative_ms = [
        float(completion_start.elapsed_time(event)) for event in completion_events
    ]
    frame_completion_ms = []
    previous_ms = 0.0
    for value in cumulative_ms:
        frame_completion_ms.append(value - previous_ms)
        previous_ms = value

    frame_count = len(views)
    total_render_ms = elapsed_seconds * 1000.0
    return {
        "trial_index": trial_index,
        "elapsed_seconds": elapsed_seconds,
        "total_render_ms": total_render_ms,
        "throughput_fps": frame_count / elapsed_seconds,
        "mean_frame_ms": total_render_ms / frame_count,
        "cuda_event_total_render_ms": float(
            completion_start.elapsed_time(completion_end)
        ),
        "cuda_event_mean_frame_ms": sum(frame_completion_ms) / frame_count,
        "p50_frame_ms": statistics.median(frame_completion_ms),
        "p95_frame_ms": _percentile(frame_completion_ms, 95.0),
        "max_frame_ms": max(frame_completion_ms),
        "frame_completion_ms": frame_completion_ms,
        "cumulative_frame_completion_ms": cumulative_ms,
        "fallback_reason": fallback_reason,
    }


def _aggregate_trials(trials):
    metric_names = (
        "elapsed_seconds",
        "total_render_ms",
        "throughput_fps",
        "mean_frame_ms",
        "cuda_event_total_render_ms",
        "cuda_event_mean_frame_ms",
        "p50_frame_ms",
        "p95_frame_ms",
        "max_frame_ms",
    )
    return {
        "median_{}".format(name): statistics.median(
            trial[name] for trial in trials
        )
        for name in metric_names
    }


def _execution_state(execution_mode, pipeline_renderer, fallback_reason):
    """Snapshot the physical backend used by one completed sequence."""

    state = {
        "actual_execution_mode": execution_mode,
        "two_stream_fallback_reason": None,
        "tacker_fallback_reason": None,
        "qualification_mode_executed": False,
    }
    if execution_mode == "two_stream":
        state["two_stream_fallback_reason"] = fallback_reason
        if fallback_reason is not None:
            state["actual_execution_mode"] = "serial"
    elif execution_mode == "tacker":
        state.update(
            {
                "actual_execution_mode": pipeline_renderer.actual_execution_mode,
                "two_stream_fallback_reason": (
                    pipeline_renderer.fallback_backend_reason
                ),
                "tacker_fallback_reason": pipeline_renderer.last_fallback_reason,
                "qualification_mode_executed": bool(
                    pipeline_renderer.last_qualification_mode
                ),
            }
        )
    return state


def main(args, dataset, hyperparam, pipeline):
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
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
    qualification_profile_sha256_snapshot = None
    tacker_override = None
    tacker_profile_sha256_snapshot = None
    if args.qualification_mode:
        (
            qualification_override,
            qualification_profile_sha256_snapshot,
        ) = _load_profile_snapshot(args.qualification_profile, "qualification")
    elif args.execution_mode == "tacker":
        tacker_override, tacker_profile_sha256_snapshot = _load_profile_snapshot(
            args.tacker_profile, "Tacker"
        )

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
                profile_path=None,
                profile_override=(
                    qualification_override
                    if args.qualification_mode
                    else tacker_override
                ),
                workload_name=args.workload_name,
                iteration=scene.loaded_iter,
                qualification_mode=args.qualification_mode,
            )

        # Bind every timed result to the source, config, loaded rasterizer binary,
        # repository, and environment observed before warmup.  Collecting these
        # hashes after all trials would let a concurrent edit relabel measurements
        # with bytes that were never used by this process.
        pre_measurement_manifest_sha256 = None
        if (
            args.execution_mode == "tacker"
            and pipeline_renderer.profile is not None
        ):
            pre_measurement_manifest_sha256 = pipeline_renderer.profile.get(
                "manifest_sha256"
            )
        reproducibility = _collect_reproducibility_metadata(
            args,
            pre_measurement_manifest_sha256,
            tacker_profile_sha256_snapshot=tacker_profile_sha256_snapshot,
            qualification_profile_sha256_snapshot=(
                qualification_profile_sha256_snapshot
            ),
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

        trials = []
        torch.cuda.cudart().cudaProfilerStart()
        try:
            for trial_index in range(1, args.trials + 1):
                trial = _measure_trial(
                    trial_index,
                    selected,
                    args.execution_mode,
                    gaussians,
                    pipeline,
                    background,
                    render_kwargs,
                    pipeline_renderer,
                )
                trial.update(
                    _execution_state(
                        args.execution_mode,
                        pipeline_renderer,
                        trial["fallback_reason"],
                    )
                )
                trials.append(trial)
        finally:
            torch.cuda.cudart().cudaProfilerStop()

        aggregates = _aggregate_trials(trials)
        representative_trial = min(
            trials,
            key=lambda trial: (
                abs(
                    trial["throughput_fps"]
                    - aggregates["median_throughput_fps"]
                ),
                trial["trial_index"],
            ),
        )
        execution_fields = (
            "actual_execution_mode",
            "two_stream_fallback_reason",
            "tacker_fallback_reason",
            "qualification_mode_executed",
        )
        execution_signatures = {
            tuple(trial[field] for field in execution_fields) for trial in trials
        }
        if len(execution_signatures) != 1:
            raise RuntimeError(
                "physical execution mode or fallback changed between trials"
            )
        execution_state = {field: trials[0][field] for field in execution_fields}
        actual_execution_mode = execution_state["actual_execution_mode"]
        two_stream_fallback_reason = execution_state[
            "two_stream_fallback_reason"
        ]
        tacker_fallback_reason = execution_state["tacker_fallback_reason"]
        qualification_executed = execution_state[
            "qualification_mode_executed"
        ]
        persistent_blocks = None
        profile_manifest_sha256 = None
        profile_selection_sha256 = None
        selected_variant_id = None
        selected_candidate_abi_sha256 = None
        if args.execution_mode == "tacker":
            persistent_blocks = pipeline_renderer.persistent_blocks
            if pipeline_renderer.profile is not None:
                active_profile = pipeline_renderer.profile
                profile_manifest_sha256 = active_profile.get("manifest_sha256")
                profile_selection_sha256 = active_profile.get("profile_sha256")
                if active_profile.get("schema_version") == 1:
                    selected_variant_id = "legacy_pos_l1"
                else:
                    selected_variant_id = active_profile.get("selected_variant_id")
                    for candidate in active_profile.get("candidates", []):
                        if candidate.get("variant_id") == selected_variant_id:
                            selected_candidate_abi_sha256 = candidate.get(
                                "abi_manifest_sha256"
                            )
                            break

        environment = reproducibility["environment"]
        repository = reproducibility["repository"]
        source_files = reproducibility["source_files"]
        profile_hashes = reproducibility["profile_hashes"]
        profile_hashes.update(
            {
                "profile_selection_sha256": profile_selection_sha256,
                "selected_candidate_abi_sha256": selected_candidate_abi_sha256,
            }
        )
        gpu_environment = environment.get("gpu") or {}
        nvidia_smi_environment = environment.get("nvidia_smi") or {}

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
            "profile_selection_sha256": profile_selection_sha256,
            "selected_variant_id": selected_variant_id,
            "selected_candidate_abi_sha256": selected_candidate_abi_sha256,
            "persistent_blocks": persistent_blocks,
            "trial_count": args.trials,
            "trials": trials,
            "aggregate_method": "median_across_whole_sequence_trials",
            "primary_metric": "median_throughput_fps",
            "primary_metric_higher_is_better": True,
            "representative_trial_index": representative_trial["trial_index"],
            # Legacy top-level timing keys and their raw samples all describe
            # the same representative trial. Cross-trial summaries live only
            # under the explicit median_* names below.
            "elapsed_seconds": representative_trial["elapsed_seconds"],
            "total_render_ms": representative_trial["total_render_ms"],
            "mean_frame_ms": representative_trial["mean_frame_ms"],
            "cuda_event_total_render_ms": representative_trial[
                "cuda_event_total_render_ms"
            ],
            "cuda_event_mean_frame_ms": representative_trial[
                "cuda_event_mean_frame_ms"
            ],
            "p50_frame_ms": representative_trial["p50_frame_ms"],
            "p95_frame_ms": representative_trial["p95_frame_ms"],
            "max_frame_ms": representative_trial["max_frame_ms"],
            "frame_completion_ms": representative_trial["frame_completion_ms"],
            "cumulative_frame_completion_ms": representative_trial[
                "cumulative_frame_completion_ms"
            ],
            "throughput_fps": representative_trial["throughput_fps"],
            "timing_method": "perf_counter_with_cuda_synchronize",
            "frame_timing_method": "cuda_event_consumer_completion_intervals",
            "io_in_timed_region": False,
            "timing_contract": {
                "unit": "whole_sequence",
                "frames_per_trial": args.frames,
                "trial_count": args.trials,
                "primary_metric": "median_throughput_fps",
                "higher_is_better": True,
                "wall_clock": "perf_counter",
                "wall_clock_completion": "cuda_synchronize_after_each_trial",
                "cuda_events": "start_end_and_per_frame_completion",
                "setup_policy": "single_load_prepare_warmup_before_all_trials",
                "io_in_timed_region": False,
            },
            "gaussian_count": int(gaussians.get_xyz.shape[0]),
            "image_width": int(first_view.image_width),
            "image_height": int(first_view.image_height),
            "dataset_type": scene.dataset_type,
            "convert_SHs_python": pipeline.convert_SHs_python,
            "compute_cov3D_python": pipeline.compute_cov3D_python,
            "gpu_name": gpu_environment.get("name")
            or nvidia_smi_environment.get("name"),
            "cuda_runtime": environment.get("cuda_runtime"),
            "cuda_driver_version": environment.get("cuda_driver_version"),
            "pytorch_version": environment.get("pytorch_version"),
            "source_tree": repository.get("source_tree"),
            "repository_commit": repository.get("commit"),
            "repository_commit_source": repository.get("commit_source"),
            "repository_dirty": repository.get("dirty"),
            "repository_git_error": repository.get("git_error"),
            "profile_render_sha256": source_files.get("profile_render.py"),
            "source_files": source_files,
            "submodule_commits": {
                item["path"]: item["commit"]
                for item in repository.get("submodules", [])
            },
            "active_profile_sha256": profile_hashes.get(
                "active_profile_sha256"
            ),
            "tacker_profile_sha256": profile_hashes.get(
                "tacker_profile_sha256"
            ),
            "qualification_profile_sha256": profile_hashes.get(
                "qualification_profile_sha256"
            ),
            "environment": environment,
            "repository": repository,
            "profile_hashes": profile_hashes,
            "metadata_collection_errors": reproducibility[
                "metadata_collection_errors"
            ],
            "pipeline_slot_count": (
                2
                if actual_execution_mode in ("two_stream", "tacker")
                else 1
            ),
        }
        metadata.update(aggregates)
        if args.metadata:
            _atomic_write_json(args.metadata, metadata)
        print(
            "Profiled {profile_frames} {split} frames in {actual_execution_mode} mode "
            "at iteration {iteration}: {median_throughput_fps:.2f} median FPS over "
            "{trial_count} trial(s), "
            "{median_mean_frame_ms:.3f} median ms/frame, {gaussian_count} Gaussians "
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
    parser.add_argument("--trials", default=1, type=int)
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
