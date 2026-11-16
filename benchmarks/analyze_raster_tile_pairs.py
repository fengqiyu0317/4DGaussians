#!/usr/bin/env python3
"""Summarize the exact two-R mapping induced by candidate persistent blocks."""

import argparse
import json
from pathlib import Path


PERCENTILES = (0, 10, 25, 50, 75, 90, 95, 99, 100)


def percentile_index(size, percentile):
    return int(round((size - 1) * percentile / 100.0))


def summarize_pairs(records, persistent_blocks):
    tile_count = records[0]["tile_count"]
    pair_count_per_frame = max(0, tile_count - persistent_blocks)
    pairs = []
    for record in records:
        counts = record["range_counts"]
        if len(counts) != tile_count:
            raise ValueError("inconsistent tile count in trace")
        for block in range(pair_count_per_frame):
            first = int(counts[block])
            second = int(counts[block + persistent_blocks])
            pairs.append(
                {
                    "frame_index": int(record["frame_index"]),
                    "physical_block": block,
                    "k0": first,
                    "k1": second,
                    "sum": first + second,
                    "absolute_difference": abs(first - second),
                }
            )
    if not pairs:
        return {
            "persistent_blocks": persistent_blocks,
            "pairs_per_frame": 0,
            "pair_samples": 0,
            "sum_percentiles": {},
            "representative_pairs": [],
        }

    ordered_by_sum = sorted(
        pairs,
        key=lambda item: (
            item["sum"],
            item["absolute_difference"],
            item["frame_index"],
            item["physical_block"],
        ),
    )
    representative_pairs = []
    for percentile in PERCENTILES:
        sample = dict(ordered_by_sum[percentile_index(len(pairs), percentile)])
        sample["percentile"] = percentile
        representative_pairs.append(sample)

    # Include the most asymmetric real pair as a separate stress case.
    most_asymmetric = dict(
        max(
            pairs,
            key=lambda item: (
                item["absolute_difference"],
                item["sum"],
            ),
        )
    )
    most_asymmetric["percentile"] = "max_absolute_difference"
    representative_pairs.append(most_asymmetric)

    return {
        "persistent_blocks": persistent_blocks,
        "pairs_per_frame": pair_count_per_frame,
        "pair_samples": len(pairs),
        "sum_mean": sum(item["sum"] for item in pairs) / len(pairs),
        "absolute_difference_mean": (
            sum(item["absolute_difference"] for item in pairs) / len(pairs)
        ),
        "sum_percentiles": {
            str(percentile): ordered_by_sum[
                percentile_index(len(pairs), percentile)
            ]["sum"]
            for percentile in PERCENTILES
        },
        "representative_pairs": representative_pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--persistent-blocks",
        default="3486,4096,4480,4800,5120,5440",
        help="comma-separated PB candidates",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    with args.trace.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError("trace contains no frames")
    tile_count = int(records[0]["tile_count"])
    if any(int(record["tile_count"]) != tile_count for record in records):
        raise ValueError("trace mixes different tile grids")

    candidates = [
        int(value) for value in args.persistent_blocks.split(",") if value
    ]
    if any(value <= 0 or value > tile_count for value in candidates):
        raise ValueError("persistent blocks must be in (0, tile_count]")
    result = {
        "schema_version": 1,
        "source_trace": str(args.trace.resolve()),
        "frames": len(records),
        "tile_count": tile_count,
        "candidates": [
            summarize_pairs(records, persistent_blocks)
            for persistent_blocks in candidates
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
