"""Aggregate the CPU Track-consistency audit CSV/JSON outputs."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/track_consistency_cpu_audit")
DATASETS = ("waymo", "nuscenes", "nuplan", "argoverse")


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        key: float(value)
        for key, value in zip(
            ("p10", "p25", "p50", "p75", "p90", "p95"),
            np.quantile(array, (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)),
        )
    }


slot_audits = {}
coverage_rows = []
roi_rows = []
hard_rows = []
coverage_summary = {}
roi_summary = {}

for dataset in DATASETS:
    with (ROOT / f"dataset_slot_audit_{dataset}.json").open() as file:
        slot_audits[dataset] = json.load(file)

    current_coverage = read_csv(ROOT / f"pair_coverage_{dataset}.csv")
    coverage_rows.extend(current_coverage)
    spatial_pairs = [int(row["valid_spatial_pairs"]) for row in current_coverage]
    temporal_pairs = [int(row["valid_temporal_pairs"]) for row in current_coverage]
    spatial_queries = [int(row["valid_spatial_queries"]) for row in current_coverage]
    temporal_queries = [int(row["valid_temporal_queries"]) for row in current_coverage]
    count = len(current_coverage)
    coverage_summary[dataset] = {
        "clips_total": count,
        "clips_with_spatial_positive": sum(value > 0 for value in spatial_pairs),
        "clips_with_temporal_positive": sum(value > 0 for value in temporal_pairs),
        "spatial_positive_clip_ratio": sum(value > 0 for value in spatial_pairs) / count,
        "temporal_positive_clip_ratio": sum(value > 0 for value in temporal_pairs) / count,
        "mean_spatial_valid_pairs_per_clip": statistics.mean(spatial_pairs),
        "median_spatial_valid_pairs_per_clip": statistics.median(spatial_pairs),
        "mean_temporal_valid_pairs_per_clip": statistics.mean(temporal_pairs),
        "median_temporal_valid_pairs_per_clip": statistics.median(temporal_pairs),
        "mean_spatial_valid_queries_per_clip": statistics.mean(spatial_queries),
        "mean_temporal_valid_queries_per_clip": statistics.mean(temporal_queries),
    }

    current_roi = read_csv(ROOT / f"roi_patch_stats_{dataset}.csv")
    roi_rows.extend(current_roi)
    patch_counts = [int(row["patch_count"]) for row in current_roi]
    roi_summary[dataset] = {
        "visible_vehicle_regions": len(patch_counts),
        **quantiles(patch_counts),
        "fraction_lt_2": sum(value < 2 for value in patch_counts) / len(patch_counts),
    }

    hard_rows.extend(read_csv(ROOT / f"hard_negative_size_stats_{dataset}.csv"))

with (ROOT / "dataset_slot_audit.json").open("w") as file:
    json.dump(slot_audits, file, indent=2)

for filename, rows in (
    ("pair_coverage.csv", coverage_rows),
    ("roi_patch_stats.csv", roi_rows),
    ("hard_negative_size_stats.csv", hard_rows),
):
    with (ROOT / filename).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

with (ROOT / "audit_summary.json").open("w") as file:
    json.dump(
        {
            "coverage": coverage_summary,
            "roi_patch_counts": roi_summary,
            "hard_negative_size": {row["dataset"]: row for row in hard_rows},
        },
        file,
        indent=2,
    )

lines = [
    "# Track pair coverage summary",
    "",
    "| Dataset | Clips | Spatial positive clips | Temporal positive clips | "
    "Spatial pairs mean/median | Temporal pairs mean/median | "
    "Spatial queries mean | Temporal queries mean |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for dataset in DATASETS:
    item = coverage_summary[dataset]
    lines.append(
        f"| {dataset} | {item['clips_total']} | "
        f"{item['clips_with_spatial_positive']}/{item['clips_total']} "
        f"({item['spatial_positive_clip_ratio']:.1%}) | "
        f"{item['clips_with_temporal_positive']}/{item['clips_total']} "
        f"({item['temporal_positive_clip_ratio']:.1%}) | "
        f"{item['mean_spatial_valid_pairs_per_clip']:.1f}/"
        f"{item['median_spatial_valid_pairs_per_clip']:.1f} | "
        f"{item['mean_temporal_valid_pairs_per_clip']:.1f}/"
        f"{item['median_temporal_valid_pairs_per_clip']:.1f} | "
        f"{item['mean_spatial_valid_queries_per_clip']:.1f} | "
        f"{item['mean_temporal_valid_queries_per_clip']:.1f} |"
    )
(ROOT / "pair_coverage_summary.md").write_text("\n".join(lines) + "\n")

labels = list(DATASETS)
x = np.arange(len(labels))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].bar(x - 0.18, [coverage_summary[d]["spatial_positive_clip_ratio"] for d in labels], 0.36, label="spatial")
axes[0].bar(x + 0.18, [coverage_summary[d]["temporal_positive_clip_ratio"] for d in labels], 0.36, label="temporal")
axes[0].set_ylim(0, 1.08); axes[0].set_title("Positive clip ratio"); axes[0].legend()
axes[1].bar(x - 0.18, [coverage_summary[d]["mean_spatial_valid_pairs_per_clip"] for d in labels], 0.36, label="spatial")
axes[1].bar(x + 0.18, [coverage_summary[d]["mean_temporal_valid_pairs_per_clip"] for d in labels], 0.36, label="temporal")
axes[1].set_title("Mean valid pairs / clip"); axes[1].legend()
axes[2].bar(x - 0.18, [coverage_summary[d]["mean_spatial_valid_queries_per_clip"] for d in labels], 0.36, label="spatial")
axes[2].bar(x + 0.18, [coverage_summary[d]["mean_temporal_valid_queries_per_clip"] for d in labels], 0.36, label="temporal")
axes[2].set_title("Mean valid queries / clip"); axes[2].legend()
for axis in axes:
    axis.set_xticks(x, labels, rotation=20)
    axis.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(ROOT / "pair_coverage.png", dpi=180); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
percentiles = ("p10", "p25", "p50", "p75", "p90", "p95")
for dataset in labels:
    axes[0].plot(percentiles, [roi_summary[dataset][p] for p in percentiles], marker="o", label=dataset)
axes[0].axhline(2, color="black", linestyle="--", linewidth=1, label="min patches")
axes[0].set_title("ROI patch-count quantiles (18×32)"); axes[0].set_ylabel("patches"); axes[0].legend()
roi_small_percent = [100.0 * roi_summary[d]["fraction_lt_2"] for d in labels]
bars = axes[1].bar(labels, roi_small_percent)
axes[1].bar_label(bars, labels=[f"{value:.3f}%" for value in roi_small_percent])
axes[1].set_title("ROI fraction below 2 patches"); axes[1].set_ylabel("percent")
axes[1].set_ylim(0, max(3.5, max(roi_small_percent) * 1.18))
axes[1].tick_params(axis="x", rotation=20)
for axis in axes: axis.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(ROOT / "roi_patch_distribution.png", dpi=180); plt.close(fig)

hard_by_dataset = {row["dataset"]: row for row in hard_rows}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for dataset in labels:
    axes[0].plot(percentiles, [float(hard_by_dataset[dataset][p]) for p in percentiles], marker="o", label=dataset)
axes[0].axhline(0.2, color="black", linestyle="--", linewidth=1, label="threshold")
axes[0].set_title("Same-class different-ID size distance"); axes[0].legend()
axes[1].bar(labels, [float(hard_by_dataset[d]["fraction_le_0_2"]) for d in labels])
axes[1].set_title("Fraction classified as hard negative"); axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=20)
for axis in axes: axis.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(ROOT / "hard_negative_size_distribution.png", dpi=180); plt.close(fig)

print(json.dumps({"coverage": coverage_summary, "roi": roi_summary, "hard": hard_by_dataset}, indent=2))
