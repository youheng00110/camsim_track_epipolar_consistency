from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/eval/nuplanhard1000/"
    "sam3_nuplanhard1000_box3d"
)

# directory_name -> display_name
METHODS = [
    ("pairedreal_implicit", "paired-real"),
    ("plucker", "plucker"),
    ("pluckernew", "pluckerbase"),
    ("box", "box"),
    ("implicit", "implicit"),
    ("full", "full"),
    ("petr", "petr"),
    ("pvonly", "pvonly"),
    ("nocondition", "nocondition"),
    ("token18000", "token18000"),
    ("token24000", "token24000"),
    ("tvself", "tvself"),
    ("urope", "urope"),
]


def iter_records(method_dir: Path):
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


def frame_key(record):
    return (
        str(record["video_id"]),
        int(record["time_index"]),
        str(record["camera_name"]),
    )


# ============================================================
# 1. paired-real 中真正被 SAM 成功识别的 GT
# ============================================================

real_detectable = {}

for record in iter_records(
    ROOT / "pairedreal_implicit"
):
    key = frame_key(record)

    real_detectable[key] = {
        str(match["gt_id"])
        for match in record["matching"]["matches"]
    }


# ============================================================
# 2. 每种方法计算：
#    Coverage-IoU
#    RC-Recall
#    RC-Coverage-IoU
# ============================================================

rows = []

for directory_name, display_name in METHODS:
    total_gt = 0
    total_iou_sum = 0.0

    rc_gt = 0
    rc_match = 0
    rc_iou_sum = 0.0

    missing_real_views = 0

    for record in iter_records(
        ROOT / directory_name
    ):
        key = frame_key(record)

        matches = {
            str(match["gt_id"]): float(
                match["mask_iou"]
            )
            for match in record[
                "matching"
            ]["matches"]
        }

        total_gt += int(
            record["gt_count"]
        )

        total_iou_sum += sum(
            matches.values()
        )

        real_ids = real_detectable.get(key)

        if real_ids is None:
            missing_real_views += 1
            continue

        rc_gt += len(real_ids)

        for gt_id in real_ids:
            if gt_id not in matches:
                continue

            rc_match += 1
            rc_iou_sum += matches[gt_id]

    coverage_iou = (
        total_iou_sum / total_gt
        if total_gt
        else 0.0
    )

    rc_recall = (
        rc_match / rc_gt
        if rc_gt
        else 0.0
    )

    rc_coverage_iou = (
        rc_iou_sum / rc_gt
        if rc_gt
        else 0.0
    )

    if missing_real_views != 0:
        raise RuntimeError(
            f"{display_name}: "
            f"missing_real_views={missing_real_views}"
        )

    rows.append({
        "method": display_name,
        "coverage_iou": coverage_iou,
        "rc_recall": rc_recall,
        "rc_coverage_iou": rc_coverage_iou,
    })


# ============================================================
# 3. DWM 2Hz
#
# 它不能和当前 paired-real 做逐实例时间对齐，
# 因此 RC 两项留空。
# ============================================================

rows.append({
    "method": "DWM (2Hz)",
    "coverage_iou": 0.29856,
    "rc_recall": None,
    "rc_coverage_iou": None,
})


# ============================================================
# 4. Print final paper table
# ============================================================

print()
print(
    f"{'Method':<22}"
    f"{'Coverage-IoU ↑':>18}"
    f"{'RC-Recall ↑':>16}"
    f"{'RC-Coverage-IoU ↑':>22}"
)

print("-" * 78)

for row in rows:
    rc_recall = (
        f"{row['rc_recall']:.4f}"
        if row["rc_recall"] is not None
        else "—"
    )

    rc_cov = (
        f"{row['rc_coverage_iou']:.4f}"
        if row["rc_coverage_iou"] is not None
        else "—"
    )

    print(
        f"{row['method']:<22}"
        f"{row['coverage_iou']:>18.4f}"
        f"{rc_recall:>16}"
        f"{rc_cov:>22}"
    )


# ============================================================
# 5. Save CSV
# ============================================================

output_path = (
    ROOT
    / "box_main_table_gtcentric.csv"
)

with output_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "method",
            "coverage_iou",
            "rc_recall",
            "rc_coverage_iou",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print("Saved:", output_path)
