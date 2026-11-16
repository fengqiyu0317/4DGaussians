#!/usr/bin/env python3
"""Derive sealed Full-MLP qualification profiles for PB experiments."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


SEALED_KEYS = (
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


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def make_profile(template, persistent_blocks):
    profile = deepcopy(template)
    full_mlp = [
        candidate
        for candidate in profile["candidates"]
        if candidate.get("partition", {}).get("kind") == "full_mlp"
    ]
    if len(full_mlp) != 1:
        raise ValueError("template must contain exactly one full_mlp candidate")

    candidate = full_mlp[0]
    variant_id = "full_mlp_pb{}".format(persistent_blocks)
    candidate["persistent_blocks"] = persistent_blocks
    candidate["variant_id"] = variant_id
    profile["selected_variant_id"] = variant_id
    profile["manifest"]["persistent_blocks"] = persistent_blocks
    profile["manifest_sha256"] = digest(profile["manifest"])

    sealed = {key: profile.get(key) for key in SEALED_KEYS}
    provenance = sealed.get("provenance")
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        provenance.pop("generated_at_utc", None)
        sealed["provenance"] = provenance
    profile["profile_sha256"] = digest(sealed)
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-blocks", nargs="+", type=int, required=True)
    args = parser.parse_args()

    template = json.loads(args.template.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for persistent_blocks in args.persistent_blocks:
        if persistent_blocks < 0:
            raise ValueError("persistent_blocks must be non-negative")
        profile = make_profile(template, persistent_blocks)
        output = args.output_dir / "full_mlp_pb{}.json".format(persistent_blocks)
        output.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("{} {}".format(output, profile["profile_sha256"]))


if __name__ == "__main__":
    main()
