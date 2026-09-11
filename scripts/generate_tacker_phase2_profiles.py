#!/usr/bin/env python3
"""Emit disabled, hash-sealed qualification profiles for Phase-2 variants.

This is deliberately not an autotuner.  It only resolves compiled resource
facts and creates one replayable profile per C0/C1/C2 physical partition so
the existing quality/FPS drivers can execute each variant explicitly.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LEGACY_HEAD_ABI_SHA256 = (
    "24570aa6e67e8b9b10fa94524fec4dc03a4eb3fdc3bf822af34c2c52ce4937ac"
)


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _normalise_resources(raw):
    result = dict(raw)
    if result.pop("launch_supported", None) is not True:
        raise ValueError("compiled variant is not launch-supported")
    aliases = {
        "block_threads": ("block_threads", "physical_threads"),
        "registers_per_thread": ("registers_per_thread",),
        "static_shared_memory_bytes": (
            "static_shared_memory_bytes",
            "static_shared_bytes",
        ),
        "max_threads_per_block": (
            "max_threads_per_block",
            "kernel_max_threads_per_block",
        ),
        "active_blocks_per_sm": (
            "active_blocks_per_sm",
            "active_blocks_per_multiprocessor",
        ),
    }
    normalized = {}
    for target, sources in aliases.items():
        value = next((result[name] for name in sources if name in result), None)
        if value is None:
            raise ValueError("resource query omitted {}".format(target))
        normalized[target] = value
    if int(normalized["active_blocks_per_sm"]) < 1:
        raise ValueError("compiled variant has zero active blocks per SM")
    # Profile resources are deliberately numeric.  The raw extension also
    # returns the boolean launch_supported gate; it is consumed above rather
    # than copied into a schema that rejects booleans as measurements.
    numeric_result = {
        name: value
        for name, value in result.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    numeric_result.update(normalized)
    return numeric_result


def _qualification_profile(module, template, candidate, persistent_blocks):
    profile = deepcopy(template)
    profile["candidates"] = [
        deepcopy(item)
        for item in template["candidates"]
        if item.get("execution_mode") in ("serial", "two_stream")
    ] + [deepcopy(candidate)]
    profile["selected_variant_id"] = candidate["variant_id"]
    profile["manifest"]["persistent_blocks"] = persistent_blocks
    profile["manifest_sha256"] = module.manifest_sha256(profile["manifest"])
    profile["selection"] = None
    profile["deployment"] = {"enabled": False, "valid": False}
    profile["provenance"] = {
        "template": True,
        "phase": 2,
        "generated_by": "scripts/generate_tacker_phase2_profiles.py",
        "input_sha256": {
            "mixed_abi": candidate["abi_manifest_sha256"],
            "head_abi": (
                module.HEAD_MULTI_ABI_SHA256
                if candidate.get("partition") is not None
                else LEGACY_HEAD_ABI_SHA256
            ),
        },
    }
    profile["note"] = (
        "Disabled Phase-2 qualification profile; never deploy without "
        "correctness and whole-run FPS selection evidence."
    )
    profile["profile_sha256"] = module.profile_sha256(profile)
    module.validate_tacker_profile(profile)
    return profile


def build_profiles(module, template, persistent_blocks, resource_provider):
    """Return deterministic C0, all C1, and one C2 qualification profiles."""

    if type(persistent_blocks) is not int or persistent_blocks < 0:
        raise ValueError("persistent_blocks must be an int >= 0")
    legacy = next(
        item
        for item in template["candidates"]
        if item.get("variant_id") == module.LEGACY_VARIANT_ID
    )
    result = {}

    c0 = deepcopy(legacy)
    c0["variant_id"] = "c0_pos_l1_pb{}".format(persistent_blocks)
    c0["persistent_blocks"] = persistent_blocks
    c0["resources"] = _normalise_resources(resource_provider(1, 1))
    result[c0["variant_id"]] = _qualification_profile(
        module, template, c0, persistent_blocks
    )

    v2_resources = _normalise_resources(resource_provider(2, 1))
    for head_name in module.HEAD_ORDER:
        candidate = module.first_linear_candidate_contract(
            "c1_{}_l1_pb{}".format(head_name, persistent_blocks),
            [head_name],
            worker_groups=1,
            persistent_blocks=persistent_blocks,
            resources=v2_resources,
        )
        result[candidate["variant_id"]] = _qualification_profile(
            module, template, candidate, persistent_blocks
        )

    c2_resources = _normalise_resources(resource_provider(2, 2))
    c2 = module.first_linear_candidate_contract(
        "c2_pos_scales_l1_wg2_pb{}".format(persistent_blocks),
        ["pos", "scales"],
        worker_groups=2,
        persistent_blocks=persistent_blocks,
        resources=c2_resources,
    )
    result[c2["variant_id"]] = _qualification_profile(
        module, template, c2, persistent_blocks
    )
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        default=str(PROJECT_ROOT / "tacker_profiles" / "raster_head_sm86.json"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--persistent-blocks", type=int, default=0)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    from gaussian_renderer import tacker_pipeline as module
    from diff_gaussian_rasterization import tacker_resource_requirements

    template = module.load_tacker_profile(profile_path=args.template)
    profiles = build_profiles(
        module,
        template,
        args.persistent_blocks,
        lambda abi_version, worker_groups: tacker_resource_requirements(
            abi_version=abi_version, worker_groups=worker_groups
        ),
    )
    output_dir = Path(args.output_dir)
    for variant_id, profile in sorted(profiles.items()):
        _atomic_write_json(output_dir / "{}.json".format(variant_id), profile)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "variant_ids": sorted(profiles),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
