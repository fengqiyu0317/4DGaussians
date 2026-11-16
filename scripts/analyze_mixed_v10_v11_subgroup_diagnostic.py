#!/usr/bin/env python3
"""Analyze ABI-v10/v11 same-envelope subgroup diagnostic JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


VARIANTS = ("v10", "v11")
MODES = ("both", "raster_only", "deform_only")
METRICS = (
    "raster_duration_cycles",
    "deform_duration_cycles",
    "deform_one_tile_duration_cycles",
    "deform_two_tile_duration_cycles",
    "overlap_window_cycles",
)


def _stats(values):
    if not values:
        return None
    return {
        "samples": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _load(root, variant, mode):
    path = root / f"{variant}_{mode}.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 22:
        raise ValueError(f"{path}: expected 22 records, got {len(records)}")
    expected_abi = f"ABI-{variant}"
    if any(record.get("abi") != expected_abi for record in records):
        raise ValueError(f"{path}: ABI mismatch")
    if any(record.get("mode") != mode for record in records):
        raise ValueError(f"{path}: mode mismatch")
    sequences = Counter(int(record["sequence"]) for record in records)
    if sequences != Counter({index: 2 for index in range(11)}):
        raise ValueError(f"{path}: unexpected sequences {sequences}")
    # Each child contributes two warm-up mixed launches (sequence 0 and 1)
    # and nine measured mixed launches (sequence 2 through 10).
    measured = [record for record in records if int(record["sequence"]) >= 2]
    if len(measured) != 18:
        raise ValueError(f"{path}: expected 18 measured records")
    return records, measured


def _aggregate_metric(records, metric):
    means = [
        float(record[metric]["mean"])
        for record in records
        if record.get(metric, {}).get("count", 0)
    ]
    medians = [
        float(record[metric]["median"])
        for record in records
        if record.get(metric, {}).get("count", 0)
    ]
    return {
        "record_means_cycles": _stats(means),
        "record_cta_medians_cycles": _stats(medians),
    } if means else None


def _fusion(raster_ms, deform_ms, both_ms):
    serial_ms = raster_ms + deform_ms
    ideal_ms = max(raster_ms, deform_ms)
    captured_ms = serial_ms - both_ms
    potential_ms = min(raster_ms, deform_ms)
    return {
        "raster_only_ms": raster_ms,
        "deform_only_ms": deform_ms,
        "both_ms": both_ms,
        "serial_solo_sum_ms": serial_ms,
        "ideal_overlap_lower_bound_ms": ideal_ms,
        "captured_overlap_saving_ms": captured_ms,
        "potential_overlap_saving_ms": potential_ms,
        "fusion_efficiency_percent": captured_ms / potential_ms * 100.0,
        "speedup_vs_serial_solo": serial_ms / both_ms,
        "ideal_speedup_vs_serial_solo": serial_ms / ideal_ms,
        "time_reduction_vs_serial_percent": (1.0 - both_ms / serial_ms) * 100.0,
        "penalty_vs_ideal_overlap_percent": (both_ms / ideal_ms - 1.0) * 100.0,
    }


def analyze(root, sm_clock_hz):
    loaded = {}
    for variant in VARIANTS:
        loaded[variant] = {}
        for mode in MODES:
            raw, measured = _load(root, variant, mode)
            loaded[variant][mode] = (raw, measured)

    variants = {}
    for variant in VARIANTS:
        modes = {}
        for mode in MODES:
            raw, records = loaded[variant][mode]
            elapsed = [float(record["kernel_elapsed_ms"]) for record in records]
            modes[mode] = {
                "raw_records": len(raw),
                "measured_records": len(records),
                "kernel_elapsed_ms": _stats(elapsed),
                "clock64": {
                    metric: _aggregate_metric(records, metric)
                    for metric in METRICS
                },
                "overlap_fraction_of_shorter": _stats([
                    float(record["overlap_fraction_of_shorter"]["mean"])
                    for record in records
                    if record["overlap_fraction_of_shorter"].get("count", 0)
                ]),
            }

        mean_fusion = _fusion(
            modes["raster_only"]["kernel_elapsed_ms"]["mean"],
            modes["deform_only"]["kernel_elapsed_ms"]["mean"],
            modes["both"]["kernel_elapsed_ms"]["mean"],
        )
        median_fusion = _fusion(
            modes["raster_only"]["kernel_elapsed_ms"]["median"],
            modes["deform_only"]["kernel_elapsed_ms"]["median"],
            modes["both"]["kernel_elapsed_ms"]["median"],
        )

        def cycle_mean(mode, metric):
            return modes[mode]["clock64"][metric]["record_means_cycles"]["mean"]

        contention = {
            "raster_slowdown_percent": (
                cycle_mean("both", "raster_duration_cycles")
                / cycle_mean("raster_only", "raster_duration_cycles") - 1.0
            ) * 100.0,
            "deform_all_slowdown_percent": (
                cycle_mean("both", "deform_duration_cycles")
                / cycle_mean("deform_only", "deform_duration_cycles") - 1.0
            ) * 100.0,
            "deform_one_tile_slowdown_percent": (
                cycle_mean("both", "deform_one_tile_duration_cycles")
                / cycle_mean("deform_only", "deform_one_tile_duration_cycles") - 1.0
            ) * 100.0,
            "deform_two_tile_slowdown_percent": (
                cycle_mean("both", "deform_two_tile_duration_cycles")
                / cycle_mean("deform_only", "deform_two_tile_duration_cycles") - 1.0
            ) * 100.0,
        }
        durations_us = {}
        for mode in MODES:
            durations_us[mode] = {}
            for metric in METRICS:
                aggregate = modes[mode]["clock64"][metric]
                durations_us[mode][metric] = (
                    aggregate["record_means_cycles"]["mean"]
                    / sm_clock_hz * 1.0e6
                    if aggregate is not None else None
                )
        first = loaded[variant]["both"][1][0]
        variants[variant] = {
            "logical_blocks": {
                "raster": int(first["raster_logical_blocks"]),
                "deform": int(first["deform_logical_blocks"]),
                "physical": int(first["physical_blocks"]),
                "deform_second_tile_ctas": int(first["deform_second_tile_ctas"]),
            },
            "modes": modes,
            "fusion_from_event_means": mean_fusion,
            "fusion_from_event_medians": median_fusion,
            "clock64_contention": contention,
            "clock64_mean_duration_us": durations_us,
        }

    return {
        "kind": "mixed_v10_v11_balanced_subgroup_diagnostic_summary",
        "input_root": str(root.resolve()),
        "sm_clock_hz": sm_clock_hz,
        "cycle_period_ns": 1.0e9 / sm_clock_hz,
        "warmup_policy": "discard sequence 0 and 1 from each child",
        "variants": variants,
    }


def _markdown(summary):
    v10_fusion = summary["variants"]["v10"]["fusion_from_event_means"]
    v11_fusion = summary["variants"]["v11"]["fusion_from_event_means"]
    raster_reduction = (
        1.0 - v11_fusion["raster_only_ms"] / v10_fusion["raster_only_ms"]
    ) * 100.0
    deform_delta = (
        v11_fusion["deform_only_ms"] / v10_fusion["deform_only_ms"] - 1.0
    ) * 100.0
    both_reduction = (
        1.0 - v11_fusion["both_ms"] / v10_fusion["both_ms"]
    ) * 100.0
    lines = [
        "# Jetson ABI-v10/v11 同包络子组诊断",
        "",
        "计时对象均为 384-thread、40-register、4 CTA/SM 包络；v10 使用 24,960 B static shared memory，v11 shared-path 优化版使用 25,216 B。每个版本/模式有 2 个独立 child，丢弃每个 child 的 2 个 warm-up mixed launch，保留 18 个样本。",
        "",
        "## CUDA Event 整 kernel 时间与融合效率",
        "",
        "| 版本 | Raster-only (ms) | Deform-only (ms) | Both (ms) | R+D (ms) | 融合效率 | 相对串行加速 | 距理想重叠惩罚 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        fusion = summary["variants"][variant]["fusion_from_event_means"]
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f}% | {:.4f}x | {:.2f}% |".format(
                variant,
                fusion["raster_only_ms"],
                fusion["deform_only_ms"],
                fusion["both_ms"],
                fusion["serial_solo_sum_ms"],
                fusion["fusion_efficiency_percent"],
                fusion["speedup_vs_serial_solo"],
                fusion["penalty_vs_ideal_overlap_percent"],
            )
        )
    lines.extend([
        "",
        "融合效率定义为 `(Raster-only + Deform-only - Both) / min(Raster-only, Deform-only)`；100% 表示短分支时间被完全隐藏。表中使用 18 个测量样本的均值。",
        "",
        "v11 相对 v10 将 Raster-only 缩短 {:.2f}%，Deform-only 变化 {:+.3f}%；Both 最终缩短 {:.2f}%。v11 同时包含 fast Raster math 与 Deform shared-path 优化，不应将该行解读为单一改动的 ablation。".format(
            raster_reduction, deform_delta, both_reduction
        ),
        "",
        "## clock64 子组竞争",
        "",
        "| 版本 | Raster 膨胀 | Deform 全部 CTA 膨胀 | Deform 单-tile CTA 膨胀 | Deform 双-tile CTA 膨胀 | Both 中短分支重叠率 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for variant in VARIANTS:
        item = summary["variants"][variant]
        contention = item["clock64_contention"]
        overlap = item["modes"]["both"]["overlap_fraction_of_shorter"]["mean"]
        lines.append(
            "| {} | {:+.2f}% | {:+.2f}% | {:+.2f}% | {:+.2f}% | {:.4f}% |".format(
                variant,
                contention["raster_slowdown_percent"],
                contention["deform_all_slowdown_percent"],
                contention["deform_one_tile_slowdown_percent"],
                contention["deform_two_tile_slowdown_percent"],
                overlap * 100.0,
            )
        )
    lines.extend([
        "",
        "## 说明",
        "",
        "- Raster-only、Deform-only、Both 都启动同一个诊断 kernel 变体，只通过运行时 mode 关闭一组 warp，因此 grid、block、shared memory、launch bounds 和寄存器包络不变。",
        "- `clock64` 位于同一个 CTA/SM 内，可直接衡量两组 warp 同时活跃时各自被拉长多少；CUDA Event 则回答整次 launch 最终节省了多少时间。",
        "- v11 的 Deform 路径使用带行偏移的 first-hidden shared tile，并将 bias 融合进 ReLU/FP16 compaction；v10 保留原 v6 Deform 路径。",
        "- 诊断 kernel 包含 `clock64` 记录和运行时 mode 分支；其绝对 Both 时间不用来替代生产 kernel 时间，三种同包络模式之间的差值才是本实验的核心证据。",
        "",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sm-clock-hz", type=float, default=1300500000.0)
    args = parser.parse_args(argv)
    summary = analyze(args.root, args.sm_clock_hz)
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.root / "analysis.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
