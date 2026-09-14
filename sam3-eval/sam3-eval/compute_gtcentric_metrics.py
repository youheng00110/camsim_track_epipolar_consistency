from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/eval/nuplanhard1000/"
    "sam3_nuplanhard1000_box3d"
)

METHODS = [
    "pairedreal_implicit",
    "plucker",
    "box",
    "implicit",
    "full",
    "petr",
    "pvonly",
    "nocondition",
    "token18000",
    "token24000",
    "tvself",
]


def frame_key(record):
    required = (
        "video_id",
        "time_index",
        "camera_name",
    )

    missing = [
        key
        for key in required
        if key not in record
    ]

    if missing:
        raise KeyError(
            f"Missing frame identity fields: {missing}"
        )

    return (
        str(record["video_id"]),
        int(record["time_index"]),
        str(record["camera_name"]),
    )


def iter_records(method_dir):
    paths = sorted(
        method_dir.glob("records.rank*.jsonl")
    )

    if not paths:
        raise FileNotFoundError(
            f"No records.rank*.jsonl under {method_dir}"
        )

    for path in paths:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)


real_dir = ROOT / "pairedreal_implicit"

print("Loading paired-real detectable GT set...")

real_detectable = {}
real_match_iou = {}

real_gt_total = 0
real_match_total = 0
real_iou_sum = 0.0

for record in iter_records(real_dir):
    key = frame_key(record)

    match_map = {
        str(match["gt_id"]): float(
            match["mask_iou"]
        )
        for match in record["matching"]["matches"]
    }

    real_detectable[key] = set(
        match_map.keys()
    )
    real_match_iou[key] = match_map

    real_gt_total += int(
        record["gt_count"]
    )
    real_match_total += len(
        match_map
    )
    real_iou_sum += sum(
        match_map.values()
    )

print(
    "paired-real views:",
    len(real_detectable),
)
print(
    "paired-real detectable GT:",
    real_match_total,
)
print()

rows = []

for method in METHODS:
    method_dir = ROOT / method

    total_gt = 0
    total_matches = 0
    total_match_iou = 0.0

    calibrated_gt = 0
    calibrated_match = 0
    calibrated_iou_sum = 0.0

    missing_real_views = 0

    for record in iter_records(method_dir):
        key = frame_key(record)

        gen_matches = {
            str(match["gt_id"]): float(
                match["mask_iou"]
            )
            for match in record["matching"]["matches"]
        }

        total_gt += int(
            record["gt_count"]
        )
        total_matches += len(
            gen_matches
        )
        total_match_iou += sum(
            gen_matches.values()
        )

        real_ids = real_detectable.get(
            key
        )

        if real_ids is None:
            missing_real_views += 1
            continue

        calibrated_gt += len(
            real_ids
        )

        for gt_id in real_ids:
            if gt_id not in gen_matches:
                continue

            calibrated_match += 1
            calibrated_iou_sum += (
                gen_matches[gt_id]
            )

    recall = (
        total_matches / total_gt
        if total_gt
        else 0.0
    )

    matched_iou = (
        total_match_iou / total_matches
        if total_matches
        else 0.0
    )

    coverage_iou = (
        total_match_iou / total_gt
        if total_gt
        else 0.0
    )

    calibrated_recall = (
        calibrated_match / calibrated_gt
        if calibrated_gt
        else 0.0
    )

    calibrated_matched_iou = (
        calibrated_iou_sum / calibrated_match
        if calibrated_match
        else 0.0
    )

    calibrated_coverage_iou = (
        calibrated_iou_sum / calibrated_gt
        if calibrated_gt
        else 0.0
    )

    rows.append({
        "method": method,
        "gt_recall": recall,
        "matched_mask_iou": matched_iou,
        "coverage_iou": coverage_iou,
        "real_calibrated_gt": calibrated_gt,
        "real_calibrated_match": calibrated_match,
        "real_calibrated_recall": calibrated_recall,
        "real_calibrated_matched_iou": calibrated_matched_iou,
        "real_calibrated_coverage_iou": calibrated_coverage_iou,
        "missing_real_views": missing_real_views,
    })


print(
    f"{'Method':<22}"
    f"{'Recall':>10}"
    f"{'M-IoU':>10}"
    f"{'Cov-IoU':>11}"
    f"{'RC-Recall':>12}"
    f"{'RC-MIoU':>11}"
    f"{'RC-CovIoU':>12}"
)

print("-" * 88)

for row in rows:
    print(
        f"{row['method']:<22}"
        f"{row['gt_recall']:>10.4f}"
        f"{row['matched_mask_iou']:>10.4f}"
        f"{row['coverage_iou']:>11.4f}"
        f"{row['real_calibrated_recall']:>12.4f}"
        f"{row['real_calibrated_matched_iou']:>11.4f}"
        f"{row['real_calibrated_coverage_iou']:>12.4f}"
    )

csv_path = ROOT / "gtcentric_metrics.csv"

with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            rows[0].keys()
        ),
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print(
    "Saved:",
    csv_path,
)
