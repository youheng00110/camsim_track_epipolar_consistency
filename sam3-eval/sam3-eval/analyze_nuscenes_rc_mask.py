#!/usr/bin/env python3

import json
from pathlib import Path
from statistics import mean


BASE = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/eval/nuscenesablation"
)

METHODS = {
    "ori3": {
        "path": BASE / "sam3_nuscenes_ori3_box3d_full",
        "camera_map": {
            "CAM_00": "CAM_FRONT_LEFT",
            "CAM_01": "CAM_FRONT",
            "CAM_02": "CAM_FRONT_RIGHT",
        },
    },
    "ori6_front3": {
        "path": BASE / "sam3_nuscenes_ori6_front3_full",
        "camera_map": {
            "CAM_01": "CAM_FRONT_LEFT",
            "CAM_02": "CAM_FRONT",
            "CAM_03": "CAM_FRONT_RIGHT",
        },
    },
    "pairedreal": {
        "path": BASE / "sam3_nuscenes_ori6_front3_pairedreal_full",
        "camera_map": {
            "CAM_01": "CAM_FRONT_LEFT",
            "CAM_02": "CAM_FRONT",
            "CAM_03": "CAM_FRONT_RIGHT",
        },
    },
}


def canonical_camera(name, camera_map):
    if name in {
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
    }:
        return name

    return camera_map.get(name, name)


def load_method(name, cfg):
    root = cfg["path"]

    files = sorted(root.glob("records.rank*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"{name}: no records.rank*.jsonl under {root}"
        )

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    for path in files:
        print("reading:", path)

    records = []

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                records.append(json.loads(line))

    all_gt = {}
    matched_gt = {}

    frame_keys = set()

    gt_pixels = []
    gt_connected_pixels = []

    sam_pixels = []
    sam_connected_pixels = []
    sam_components = []

    match_ious = []

    raw_gt_count = 0
    gt_count = 0

    raw_detection_count = 0
    detection_count = 0

    matched_count = 0

    filtered_detection_count = 0

    for record in records:
        dataset = str(
            record.get(
                "dataset_name",
                "nuscenes",
            )
        )

        video_id = str(
            record.get(
                "video_id",
                "unknown",
            )
        )

        time_index = int(
            record.get(
                "time_index",
                record.get("frame_index", 0),
            )
        )

        raw_camera = str(
            record.get(
                "camera_name",
                "unknown",
            )
        )

        camera = canonical_camera(
            raw_camera,
            cfg["camera_map"],
        )

        frame_key = (
            dataset,
            video_id,
            time_index,
            camera,
        )

        frame_keys.add(frame_key)

        raw_gt_count += int(
            record.get("raw_gt_count", 0)
        )

        gt_count += int(
            record.get("gt_count", 0)
        )

        raw_detection_count += int(
            record.get(
                "raw_detection_count",
                0,
            )
        )

        detection_count += int(
            record.get(
                "detection_count",
                0,
            )
        )

        matched_count += int(
            record.get(
                "matched_count",
                0,
            )
        )

        filtered_detection_count += int(
            record.get(
                "filtered_small_detection_count",
                0,
            )
        )

        # -----------------------------------------
        # GT mask-related data
        # -----------------------------------------

        projections = record.get(
            "projections",
            [],
        )

        for projection in projections:
            gt_id = str(
                projection.get(
                    "gt_id",
                    projection.get(
                        "instance_token",
                        "",
                    ),
                )
            )

            if not gt_id:
                continue

            key = (
                dataset,
                video_id,
                time_index,
                camera,
                gt_id,
            )

            all_gt[key] = projection

            if (
                projection.get(
                    "visible_pixel_count"
                )
                is not None
            ):
                gt_pixels.append(
                    float(
                        projection[
                            "visible_pixel_count"
                        ]
                    )
                )

            if (
                projection.get(
                    "visible_connected_pixel_count"
                )
                is not None
            ):
                gt_connected_pixels.append(
                    float(
                        projection[
                            "visible_connected_pixel_count"
                        ]
                    )
                )

        # -----------------------------------------
        # SAM mask-related data
        # -----------------------------------------

        detections = record.get(
            "detections",
            [],
        )

        for detection in detections:
            if (
                detection.get(
                    "mask_pixel_count"
                )
                is not None
            ):
                sam_pixels.append(
                    float(
                        detection[
                            "mask_pixel_count"
                        ]
                    )
                )

            if (
                detection.get(
                    "connected_pixel_count"
                )
                is not None
            ):
                sam_connected_pixels.append(
                    float(
                        detection[
                            "connected_pixel_count"
                        ]
                    )
                )

            if (
                detection.get(
                    "mask_component_count"
                )
                is not None
            ):
                sam_components.append(
                    float(
                        detection[
                            "mask_component_count"
                        ]
                    )
                )

        # -----------------------------------------
        # Matched mask IoU
        # -----------------------------------------

        matching = record.get(
            "matching",
            {},
        )

        matches = matching.get(
            "matches",
            [],
        )

        for match in matches:
            gt_id = str(
                match.get(
                    "gt_id",
                    "",
                )
            )

            if not gt_id:
                continue

            key = (
                dataset,
                video_id,
                time_index,
                camera,
                gt_id,
            )

            mask_iou = float(
                match.get(
                    "mask_iou",
                    0.0,
                )
            )

            matched_gt[key] = mask_iou
            match_ious.append(mask_iou)

    raw_recall = (
        matched_count / gt_count
        if gt_count
        else 0.0
    )

    stats = {
        "name": name,
        "records": len(records),
        "frames": frame_keys,
        "all_gt": all_gt,
        "matched_gt": matched_gt,
        "raw_gt_count": raw_gt_count,
        "gt_count": gt_count,
        "raw_detection_count": (
            raw_detection_count
        ),
        "detection_count": detection_count,
        "filtered_detection_count": (
            filtered_detection_count
        ),
        "matched_count": matched_count,
        "raw_recall": raw_recall,
        "mean_mask_iou": (
            mean(match_ious)
            if match_ious
            else 0.0
        ),
        "mean_gt_pixels": (
            mean(gt_pixels)
            if gt_pixels
            else 0.0
        ),
        "mean_gt_connected_pixels": (
            mean(gt_connected_pixels)
            if gt_connected_pixels
            else 0.0
        ),
        "mean_sam_pixels": (
            mean(sam_pixels)
            if sam_pixels
            else 0.0
        ),
        "mean_sam_connected_pixels": (
            mean(sam_connected_pixels)
            if sam_connected_pixels
            else 0.0
        ),
        "mean_sam_components": (
            mean(sam_components)
            if sam_components
            else 0.0
        ),
    }

    return stats


results = {
    name: load_method(name, cfg)
    for name, cfg in METHODS.items()
}


# ============================================================
# Basic mask statistics
# ============================================================

print()
print()
print("=" * 120)
print("RAW MASK STATISTICS")
print("=" * 120)

header = (
    f"{'Method':<18}"
    f"{'Records':>10}"
    f"{'GT':>12}"
    f"{'Det':>12}"
    f"{'Match':>12}"
    f"{'Recall':>10}"
    f"{'MaskIoU':>12}"
    f"{'GTpx':>12}"
    f"{'SAMpx':>12}"
)

print(header)
print("-" * len(header))

for name in METHODS:
    x = results[name]

    print(
        f"{name:<18}"
        f"{x['records']:>10d}"
        f"{x['gt_count']:>12d}"
        f"{x['detection_count']:>12d}"
        f"{x['matched_count']:>12d}"
        f"{x['raw_recall']:>10.4f}"
        f"{x['mean_mask_iou']:>12.4f}"
        f"{x['mean_gt_pixels']:>12.1f}"
        f"{x['mean_sam_pixels']:>12.1f}"
    )


# ============================================================
# Frame / GT overlap check
# ============================================================

ref = results["pairedreal"]

print()
print()
print("=" * 120)
print("OVERLAP WITH PAIRED REAL")
print("=" * 120)

for name in [
    "ori3",
    "ori6_front3",
]:
    gen = results[name]

    common_frames = (
        gen["frames"]
        & ref["frames"]
    )

    ref_gt_on_gen = (
        set(ref["all_gt"])
        & set(gen["all_gt"])
    )

    ref_matched_on_gen = (
        set(ref["matched_gt"])
        & set(gen["all_gt"])
    )

    print()
    print(name)
    print(
        "  generated frames      =",
        len(gen["frames"]),
    )
    print(
        "  pairedreal frames     =",
        len(ref["frames"]),
    )
    print(
        "  common frames         =",
        len(common_frames),
    )
    print(
        "  generated GT          =",
        len(gen["all_gt"]),
    )
    print(
        "  pairedreal GT         =",
        len(ref["all_gt"]),
    )
    print(
        "  exact common GT       =",
        len(ref_gt_on_gen),
    )
    print(
        "  real-detectable GT "
        "available in gen     =",
        len(ref_matched_on_gen),
    )


# ============================================================
# RC metrics
#
# denominator:
# paired-real successfully matched GT that also exists
# in the generated evaluation.
#
# RC-Recall:
# |Matched_gen ∩ Matched_real| / |Matched_real eligible|
#
# RC-MIoU:
# mean generated mask IoU only on common matched GT.
#
# RC-CovIoU:
# sum generated IoU on common matched GT
# / paired-real detectable GT count.
# Missed GT gets IoU=0.
# ============================================================

print()
print()
print("=" * 120)
print("REAL-CALIBRATED METRICS")
print("=" * 120)

rc_results = {}

for name in [
    "ori3",
    "ori6_front3",
]:
    gen = results[name]

    eligible_ref = (
        set(ref["matched_gt"])
        & set(gen["all_gt"])
    )

    common_matched = (
        eligible_ref
        & set(gen["matched_gt"])
    )

    rc_recall = (
        len(common_matched)
        / len(eligible_ref)
        if eligible_ref
        else 0.0
    )

    rc_ious = [
        gen["matched_gt"][key]
        for key in common_matched
    ]

    rc_miou = (
        mean(rc_ious)
        if rc_ious
        else 0.0
    )

    rc_cov_iou = (
        sum(rc_ious)
        / len(eligible_ref)
        if eligible_ref
        else 0.0
    )

    real_ious_same = [
        ref["matched_gt"][key]
        for key in common_matched
    ]

    real_miou_same = (
        mean(real_ious_same)
        if real_ious_same
        else 0.0
    )

    rc_results[name] = {
        "eligible": len(eligible_ref),
        "common": len(common_matched),
        "rc_recall": rc_recall,
        "rc_miou": rc_miou,
        "rc_cov_iou": rc_cov_iou,
        "real_miou_same": (
            real_miou_same
        ),
    }


header = (
    f"{'Method':<18}"
    f"{'RealGT':>12}"
    f"{'Common':>12}"
    f"{'RCRecall':>12}"
    f"{'RC-MIoU':>12}"
    f"{'RC-CovIoU':>14}"
    f"{'RealIoU@same':>16}"
)

print(header)
print("-" * len(header))

for name in [
    "ori3",
    "ori6_front3",
]:
    x = rc_results[name]

    print(
        f"{name:<18}"
        f"{x['eligible']:>12d}"
        f"{x['common']:>12d}"
        f"{x['rc_recall']:>12.4f}"
        f"{x['rc_miou']:>12.4f}"
        f"{x['rc_cov_iou']:>14.4f}"
        f"{x['real_miou_same']:>16.4f}"
    )


# ============================================================
# Common RC-MIoU
#
# Exact same GT subset:
# paired real + ori3 + ori6 all successfully matched.
# ============================================================

common_all = (
    set(ref["matched_gt"])
    & set(results["ori3"]["matched_gt"])
    & set(results["ori6_front3"]["matched_gt"])
)

print()
print()
print("=" * 120)
print("COMMON MATCHED SUBSET")
print("=" * 120)

print(
    "GT matched by pairedreal + ori3 + ori6_front3 =",
    len(common_all),
)

if common_all:
    print()

    for name in [
        "pairedreal",
        "ori3",
        "ori6_front3",
    ]:
        values = [
            results[name]["matched_gt"][key]
            for key in common_all
        ]

        print(
            f"{name:<18}"
            f" Common-RC-MIoU = "
            f"{mean(values):.6f}"
        )


# ============================================================
# Save CSV
# ============================================================

out_csv = BASE / "nuscenes_rc_mask_summary.csv"

with out_csv.open(
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "method,"
        "records,"
        "gt_count,"
        "detection_count,"
        "matched_count,"
        "raw_recall,"
        "raw_mask_iou,"
        "mean_gt_pixels,"
        "mean_sam_pixels,"
        "rc_eligible_gt,"
        "rc_common_matched,"
        "rc_recall,"
        "rc_miou,"
        "rc_cov_iou\n"
    )

    for name in METHODS:
        x = results[name]

        if name in rc_results:
            rc = rc_results[name]
        else:
            rc = {
                "eligible": len(
                    ref["matched_gt"]
                ),
                "common": len(
                    ref["matched_gt"]
                ),
                "rc_recall": 1.0,
                "rc_miou": (
                    mean(
                        ref[
                            "matched_gt"
                        ].values()
                    )
                    if ref["matched_gt"]
                    else 0.0
                ),
                "rc_cov_iou": (
                    mean(
                        ref[
                            "matched_gt"
                        ].values()
                    )
                    if ref["matched_gt"]
                    else 0.0
                ),
            }

        f.write(
            f"{name},"
            f"{x['records']},"
            f"{x['gt_count']},"
            f"{x['detection_count']},"
            f"{x['matched_count']},"
            f"{x['raw_recall']:.8f},"
            f"{x['mean_mask_iou']:.8f},"
            f"{x['mean_gt_pixels']:.4f},"
            f"{x['mean_sam_pixels']:.4f},"
            f"{rc['eligible']},"
            f"{rc['common']},"
            f"{rc['rc_recall']:.8f},"
            f"{rc['rc_miou']:.8f},"
            f"{rc['rc_cov_iou']:.8f}\n"
        )

print()
print("saved:", out_csv)
