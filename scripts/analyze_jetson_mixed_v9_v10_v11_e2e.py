#!/usr/bin/env python3
"""Summarize the formal Jetson serial/two-stream/v9/v10/v11 E2E run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

from benchmark_tacker_fps import paired_bootstrap_ratio


VARIANTS = ("serial", "two_stream", "v9", "v10", "v11")
NAME = {
    "serial": "Serial",
    "two_stream": "Two-stream",
    "v9": "ABI-v9",
    "v10": "ABI-v10 exact",
    "v11": "ABI-v11 fast math",
}
RUN_PATTERN = re.compile(r"^\d+-round(?P<round>\d+)-(?P<variant>.+)\.metadata\.json$")


def _stats(values):
    mean = statistics.mean(values)
    median = statistics.median(values)
    return {
        "samples": len(values),
        "mean_fps": mean,
        "median_fps": median,
        "median_ms_per_frame": 1000.0 / median,
        "minimum_fps": min(values),
        "maximum_fps": max(values),
        "sample_stdev_fps": statistics.stdev(values),
        "coefficient_of_variation_percent": statistics.stdev(values) / mean * 100.0,
    }


def _load(root):
    by_variant = {variant: {} for variant in VARIANTS}
    integrity = {
        "children": 0,
        "fallbacks": 0,
        "failed": 0,
        "tacker_children": 0,
        "tacker_children_with_49_mixed_launches": 0,
    }
    for path in sorted((root / "fps" / "runs").glob("*.metadata.json")):
        match = RUN_PATTERN.match(path.name)
        if match is None:
            raise ValueError("unexpected run filename: {}".format(path.name))
        variant = match.group("variant")
        round_index = int(match.group("round"))
        if variant not in by_variant:
            raise ValueError("unexpected variant: {}".format(variant))
        doc = json.loads(path.read_text(encoding="utf-8"))
        integrity["children"] += 1
        if doc.get("passed") is not True:
            integrity["failed"] += 1
        expected_mode = "tacker" if variant.startswith("v") else variant
        if doc.get("actual_execution_mode") != expected_mode:
            integrity["fallbacks"] += 1
        fps = float(doc["throughput_fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("invalid FPS in {}".format(path))
        if round_index in by_variant[variant]:
            raise ValueError("duplicate round {} for {}".format(round_index, variant))
        by_variant[variant][round_index] = fps
        if expected_mode == "tacker":
            integrity["tacker_children"] += 1
            if doc.get("pipeline_execution_counts", {}).get("mixed_launches") == 49:
                integrity["tacker_children_with_49_mixed_launches"] += 1

    expected_rounds = set(range(10))
    for variant in VARIANTS:
        if set(by_variant[variant]) != expected_rounds:
            raise ValueError(
                "{} rounds mismatch: {}".format(variant, sorted(by_variant[variant]))
            )
    if integrity != {
        "children": 50,
        "fallbacks": 0,
        "failed": 0,
        "tacker_children": 30,
        "tacker_children_with_49_mixed_launches": 30,
    }:
        raise ValueError("execution integrity failed: {}".format(integrity))

    frequencies = []
    with (root / "run_frequencies.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frequencies.extend(
                (int(row["freq_before_hz"]), int(row["freq_after_hz"]))
            )
    if len(frequencies) != 100 or set(frequencies) != {1300500000}:
        raise ValueError("frequency gate failed: {}".format(sorted(set(frequencies))))
    return by_variant, integrity, frequencies


def _comparison(by_variant, candidate, reference):
    candidate_values = [by_variant[candidate][index] for index in range(10)]
    reference_values = [by_variant[reference][index] for index in range(10)]
    return paired_bootstrap_ratio(
        candidate_values,
        reference_values,
        resamples=10000,
        seed=0,
        label="{}_vs_{}".format(candidate, reference),
    )


def _load_quality(root):
    reports = {}
    for variant in ("v9", "v10", "v11"):
        path = root / "quality" / "{}.json".format(variant)
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("passed") is not True:
            raise ValueError("quality report failed: {}".format(path))
        if doc.get("workload", {}).get("frames") != 50:
            raise ValueError("quality report is not 50-view: {}".format(path))
        if doc.get("modes", {}).get("tacker", {}).get("actual_mode") != "tacker":
            raise ValueError("quality report fell back: {}".format(path))
        reports[variant] = doc

    serial_means = reports["v9"]["modes"]["serial"]["means"]
    for variant in ("v10", "v11"):
        if reports[variant]["modes"]["serial"]["means"] != serial_means:
            raise ValueError("serial quality reference changed for {}".format(variant))

    reference = reports["v9"]["modes"]["tacker"]["per_view"]
    result = {"serial": serial_means}
    for variant in ("v9", "v10", "v11"):
        mode = reports[variant]["modes"]["tacker"]
        candidate = mode["per_view"]
        result[variant] = {
            "passed": True,
            "means": mode["means"],
            "deltas_vs_serial": reports[variant]["deltas"]["tacker"],
            "max_abs_per_view_metric_delta_vs_v9": {
                metric: max(
                    abs(left[metric] - right[metric])
                    for left, right in zip(reference, candidate)
                )
                for metric in ("psnr_db", "ssim", "lpips")
            },
        }
    return result


def _mixed_to_e2e_consistency(root, fps):
    leaf_path = root.parent / "mixed_v9_v10_v11_leaf_timing_v1" / "summary.json"
    leaf = json.loads(leaf_path.read_text(encoding="utf-8"))
    result = {}
    for candidate, reference in (
        ("v10", "v9"),
        ("v11", "v9"),
        ("v11", "v10"),
    ):
        leaf_saving = (
            leaf["variants"][reference]["median_ms"]
            - leaf["variants"][candidate]["median_ms"]
        )
        expected = leaf_saving * 49.0 / 50.0
        observed = (
            fps[reference]["median_ms_per_frame"]
            - fps[candidate]["median_ms_per_frame"]
        )
        result["{}_vs_{}".format(candidate, reference)] = {
            "leaf_saving_ms_per_mixed_launch": leaf_saving,
            "mixed_launches_per_50_frames": 49,
            "expected_average_saving_ms_per_frame": expected,
            "observed_median_saving_ms_per_frame": observed,
            "observed_fraction_of_expected": observed / expected,
        }
    return result


def _markdown(summary):
    lines = [
        "# Jetson ABI-v9/v10/v11 正式端到端测试",
        "",
        "## 结果",
        "",
        "| 版本 | 中位 FPS | 中位 ms/frame | min–max FPS | CV |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = summary["fps"][variant]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f}–{:.3f} | {:.3f}% |".format(
                NAME[variant],
                row["median_fps"],
                row["median_ms_per_frame"],
                row["minimum_fps"],
                row["maximum_fps"],
                row["coefficient_of_variation_percent"],
            )
        )
    lines.extend(
        [
            "",
            "## 配对比较",
            "",
            "| 比较 | 中位 FPS ratio | 提升 | paired bootstrap 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, comparison in summary["paired_comparisons"].items():
        ratio = comparison["median_fps_ratio"]
        ci = comparison["paired_bootstrap_95_ci"]
        lines.append(
            "| {} | {:.6f}× | {:+.3f}% | [{:.6f}, {:.6f}] |".format(
                key.replace("_vs_", " vs "),
                ratio,
                (ratio - 1.0) * 100.0,
                ci["lower"],
                ci["upper"],
            )
        )
    lines.extend(
        [
            "",
            "## Mixed leaf 收益兑现",
            "",
            "| 比较 | mixed 预测节省 (ms/frame) | E2E 实测节省 (ms/frame) | 兑现比例 |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, item in summary["mixed_to_e2e_consistency"].items():
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.2f}% |".format(
                key.replace("_vs_", " vs "),
                item["expected_average_saving_ms_per_frame"],
                item["observed_median_saving_ms_per_frame"],
                item["observed_fraction_of_expected"] * 100.0,
            )
        )
    lines.extend(
        [
            "",
            "## 50-view 画质",
            "",
            "| 版本 | PSNR (dB) | SSIM | LPIPS | gate |",
            "|---|---:|---:|---:|---|",
            "| Serial | {:.6f} | {:.9f} | {:.9f} | reference |".format(
                summary["quality"]["serial"]["psnr_db"],
                summary["quality"]["serial"]["ssim"],
                summary["quality"]["serial"]["lpips"],
            ),
        ]
    )
    for variant in ("v9", "v10", "v11"):
        means = summary["quality"][variant]["means"]
        lines.append(
            "| {} | {:.6f} | {:.9f} | {:.9f} | PASS |".format(
                NAME[variant], means["psnr_db"], means["ssim"], means["lpips"]
            )
        )
    lines.extend(
        [
            "",
            "v10 相对 v9 的最大逐视角指标差为零；v11 相对 v9 的最大逐视角绝对差为 "
            "PSNR `{:.3e} dB`、SSIM `{:.3e}`、LPIPS `{:.3e}`。".format(
                summary["quality"]["v11"]["max_abs_per_view_metric_delta_vs_v9"]["psnr_db"],
                summary["quality"]["v11"]["max_abs_per_view_metric_delta_vs_v9"]["ssim"],
                summary["quality"]["v11"]["max_abs_per_view_metric_delta_vs_v9"]["lpips"],
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## 协议与完整性",
            "",
            "- `flame_steak` iteration 14000，111,525 Gaussians，1352×1014。",
            "- 每个版本 10 个独立 child；每个 child 预热 10 帧后计时 test views 0–49。",
            "- 固定 seed-0 ABBA：每轮五个版本各出现一次，相邻轮反向并轮换起点。",
            "- 50/50 child 成功，30/30 Tacker child 均为 49 次 mixed launch，无 fallback。",
            "- 100/100 个运行前后频率读数均为 1.3005 GHz。",
            "- FPS 使用完整 50 帧 sequence 的 wall-clock throughput；I/O 不在计时区间。",
            "",
            "原始 child metadata 位于 `fps/runs/`，结构化结果见 `summary.json`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    by_variant, integrity, frequencies = _load(root)
    quality = _load_quality(root)
    fps = {
        variant: _stats([by_variant[variant][index] for index in range(10)])
        for variant in VARIANTS
    }
    summary = {
        "schema_version": 1,
        "kind": "jetson_mixed_v9_v10_v11_e2e_with_baselines",
        "device": {
            "gpu_name": "Orin",
            "cuda_arch": "sm_87",
            "locked_gpu_frequency_hz": 1300500000,
            "frequency_checks": len(frequencies),
        },
        "workload": {
            "scene": "flame_steak",
            "iteration": 14000,
            "gaussian_count": 111525,
            "resolution": [1352, 1014],
            "frames_per_child": 50,
            "warmup_frames": 10,
            "children_per_variant": 10,
            "schedule": "abba_seed0",
        },
        "fps": fps,
        "paired_comparisons": {
            "v9_vs_serial": _comparison(by_variant, "v9", "serial"),
            "v9_vs_two_stream": _comparison(by_variant, "v9", "two_stream"),
            "v10_vs_v9": _comparison(by_variant, "v10", "v9"),
            "v10_vs_serial": _comparison(by_variant, "v10", "serial"),
            "v10_vs_two_stream": _comparison(by_variant, "v10", "two_stream"),
            "v11_vs_v10": _comparison(by_variant, "v11", "v10"),
            "v11_vs_v9": _comparison(by_variant, "v11", "v9"),
            "v11_vs_serial": _comparison(by_variant, "v11", "serial"),
            "v11_vs_two_stream": _comparison(by_variant, "v11", "two_stream"),
        },
        "mixed_to_e2e_consistency": _mixed_to_e2e_consistency(root, fps),
        "quality": quality,
        "execution_integrity": integrity,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "analysis.md").write_text(_markdown(summary), encoding="utf-8")
    print(root / "analysis.md")


if __name__ == "__main__":
    main()
