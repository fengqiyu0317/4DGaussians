#!/usr/bin/env python3
"""Rebind disabled schema-v2 Tacker profiles to one exact workload.

The CUDA descriptors and resources are invariant across workloads with the
same model architecture.  The workload manifest is not: scene name, Gaussian
count, iteration and image size are part of the fail-closed runtime contract.
This tool copies an existing disabled qualification profile, replaces only
that workload identity, records the source artifact hash, and recomputes both
schema-v2 semantic digests.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


PROFILE_SEALED_KEYS = (
    "schema_version",
    "workload_key",
    "selection_objective",
    "selected_variant_id",
    "manifest",
    "manifest_sha256",
    "correctness_thresholds",
    "candidates",
    "selection",
    "deployment",
    "provenance",
)


class MaterializationError(ValueError):
    pass


def canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(manifest):
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def profile_sha256(profile):
    payload = {key: profile.get(key) for key in PROFILE_SEALED_KEYS}
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        provenance.pop("generated_at_utc", None)
        payload["provenance"] = provenance
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def normalise_scene_name(value):
    if not isinstance(value, str):
        raise MaterializationError("workload name must be a string")
    result = value.strip().replace("-", "_").lower()
    if not result:
        raise MaterializationError("workload name must be non-empty")
    return result


def normalise_cuda_arch(value):
    if not isinstance(value, str):
        raise MaterializationError("cuda_arch must be a string")
    result = value.strip().lower()
    if (
        len(result) != 5
        or not result.startswith("sm_")
        or not result[3:].isdigit()
    ):
        raise MaterializationError("cuda_arch must use the form sm_XX")
    return result


def normalise_compute_capability(value, cuda_arch):
    if value is None:
        result = [int(cuda_arch[3]), int(cuda_arch[4])]
    elif isinstance(value, str):
        fields = value.strip().split(".")
        if len(fields) != 2 or any(not field.isdigit() for field in fields):
            raise MaterializationError(
                "compute capability must use the form MAJOR.MINOR"
            )
        result = [int(fields[0]), int(fields[1])]
    elif (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(type(field) is int for field in value)
    ):
        result = list(value)
    else:
        raise MaterializationError("compute capability must contain two integers")
    if result != [int(cuda_arch[3]), int(cuda_arch[4])]:
        raise MaterializationError(
            "compute capability does not match cuda_arch {}".format(cuda_arch)
        )
    return result


def workload_key(
    name, iteration, gaussian_count, width, height, cuda_arch="sm_86"
):
    return "{}:{}:{}:{}x{}:{}".format(
        normalise_scene_name(name),
        iteration,
        gaussian_count,
        width,
        height,
        normalise_cuda_arch(cuda_arch),
    )


def materialize_profile(
    source,
    source_path,
    name,
    iteration,
    gaussian_count,
    width,
    height,
    cuda_arch=None,
    compute_capability=None,
    gpu_name=None,
    selected_resources=None,
):
    if not isinstance(source, dict) or source.get("schema_version") != 2:
        raise MaterializationError("source profile must use schema_version 2")
    if source.get("deployment") != {"enabled": False, "valid": False}:
        raise MaterializationError("source profile must remain disabled")
    if source.get("selection") is not None:
        raise MaterializationError("source qualification profile must have null selection")
    manifest = source.get("manifest")
    if not isinstance(manifest, dict):
        raise MaterializationError("source profile omitted manifest")
    if any(value <= 0 for value in (iteration, gaussian_count, width, height)):
        raise MaterializationError("iteration, count and dimensions must be positive")

    retarget_requested = any(
        value is not None for value in (cuda_arch, compute_capability, gpu_name)
    )
    source_cuda_arch = normalise_cuda_arch(
        manifest.get("cuda_arch")
        or str(source.get("workload_key", "")).rsplit(":", 1)[-1]
    )
    target_cuda_arch = normalise_cuda_arch(cuda_arch or source_cuda_arch)
    target_compute_capability = normalise_compute_capability(
        (
            compute_capability
            if compute_capability is not None
            else manifest.get("compute_capability")
            if target_cuda_arch == source_cuda_arch
            else None
        ),
        target_cuda_arch,
    )
    target_gpu_name = (
        str(gpu_name).strip()
        if gpu_name is not None
        else str(manifest.get("gpu_name", "")).strip()
    )
    if retarget_requested and not target_gpu_name:
        raise MaterializationError("gpu_name must be non-empty")
    target_changed = (
        target_cuda_arch != source_cuda_arch
        or target_compute_capability != manifest.get("compute_capability")
        or target_gpu_name != manifest.get("gpu_name")
    )

    scene = normalise_scene_name(name)
    key = workload_key(
        scene,
        iteration,
        gaussian_count,
        width,
        height,
        target_cuda_arch,
    )
    result = json.loads(json.dumps(source))
    result["workload_key"] = key
    result["manifest"]["workload"] = scene
    result["manifest"]["workload_key"] = key
    result["manifest"]["iteration"] = iteration
    result["manifest"]["gaussian_count"] = gaussian_count
    result["manifest"]["resolution"] = [width, height]
    if retarget_requested:
        result["manifest"]["cuda_arch"] = target_cuda_arch
        result["manifest"]["compute_capability"] = target_compute_capability
        result["manifest"]["gpu_name"] = target_gpu_name
        for candidate in result.get("candidates", []):
            if candidate.get("execution_mode") != "tacker":
                continue
            capability = candidate.get("capability_requirements")
            if not isinstance(capability, dict):
                raise MaterializationError(
                    "Tacker candidate omitted capability_requirements"
                )
            capability["cuda_arch"] = target_cuda_arch
            capability["compute_capability"] = target_compute_capability
            if target_changed:
                # Register counts and occupancy are cubin/device facts.  Reusing
                # measurements from another architecture would make runtime
                # admission misleading; qualification queries them again.
                candidate["resources"] = None
    if selected_resources is not None:
        if not isinstance(selected_resources, dict) or not selected_resources:
            raise MaterializationError(
                "selected_resources must be a non-empty object"
            )
        selected = [
            candidate
            for candidate in result.get("candidates", [])
            if candidate.get("variant_id") == result.get("selected_variant_id")
        ]
        if len(selected) != 1 or selected[0].get("execution_mode") != "tacker":
            raise MaterializationError(
                "selected_resources require one selected Tacker candidate"
            )
        selected[0]["resources"] = json.loads(json.dumps(selected_resources))
    result["manifest_sha256"] = manifest_sha256(result["manifest"])
    provenance = dict(result.get("provenance") or {})
    materialization = {
        "generated_by": "scripts/materialize_tacker_workload_profiles.py",
        "source_profile_file_sha256": sha256_file(source_path),
        "workload_key": key,
    }
    if retarget_requested:
        materialization["deployment_target"] = {
            "cuda_arch": target_cuda_arch,
            "compute_capability": target_compute_capability,
            "gpu_name": target_gpu_name,
            "resources_cleared": target_changed,
        }
    if selected_resources is not None:
        materialization["selected_candidate_resources"] = json.loads(
            json.dumps(selected_resources)
        )
    provenance["workload_materialization"] = materialization
    result["provenance"] = provenance
    result["profile_sha256"] = profile_sha256(result)
    return result


def atomic_write_new(path, value):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise MaterializationError("refusing to overwrite {}".format(target))
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
                sort_keys=True,
                allow_nan=False,
            )
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


def parse_profile(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("profile must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-." for char in name.lower()):
        raise argparse.ArgumentTypeError("profile name/path is invalid")
    return name, Path(path).expanduser().resolve()


def parse_resource(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("resource must be NAME=JSON_VALUE")
    name, raw = value.split("=", 1)
    if (
        not name
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in name.lower()
        )
    ):
        raise argparse.ArgumentTypeError("resource name is invalid")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            "resource value must be valid JSON: {}".format(error)
        )
    if (
        not isinstance(parsed, (int, float))
        or isinstance(parsed, bool)
        or parsed < 0
    ):
        raise argparse.ArgumentTypeError(
            "resource value must be a non-negative number"
        )
    return name, parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", type=parse_profile, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workload-name", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--gaussian-count", type=int, required=True)
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--cuda-arch")
    parser.add_argument("--compute-capability")
    parser.add_argument("--gpu-name")
    parser.add_argument(
        "--selected-resource", action="append", type=parse_resource
    )
    args = parser.parse_args(argv)

    selected_resources = None
    if args.selected_resource:
        selected_resources = {}
        for name, value in args.selected_resource:
            if name in selected_resources:
                raise MaterializationError(
                    "duplicate selected resource {}".format(name)
                )
            selected_resources[name] = value

    output_dir = Path(args.output_dir).expanduser().resolve()
    emitted = []
    for label, source_path in args.profile:
        with source_path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        profile = materialize_profile(
            source,
            source_path,
            args.workload_name,
            args.iteration,
            args.gaussian_count,
            args.image_width,
            args.image_height,
            cuda_arch=args.cuda_arch,
            compute_capability=args.compute_capability,
            gpu_name=args.gpu_name,
            selected_resources=selected_resources,
        )
        target = output_dir / "{}.qualification.json".format(label)
        atomic_write_new(target, profile)
        emitted.append(str(target))
    for path in emitted:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
