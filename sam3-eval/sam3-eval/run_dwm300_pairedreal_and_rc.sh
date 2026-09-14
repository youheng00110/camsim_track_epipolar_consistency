#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Environment
# ============================================================

source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

export PATH=/inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval


# ============================================================
# Paths
# ============================================================

EVAL_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuscenesablation

PREVIEW_ROOT=$EVAL_ROOT/nuplan_merged500

GEN_OUT=$EVAL_ROOT/sam3_nuplan_box300

REAL_OUT=$EVAL_ROOT/sam3_nuplan_box300_pairedreal

CONFIG=$PWD/config_dwm300_pairedreal.yaml


# ============================================================
# Environment check
# ============================================================

echo "============================================================"
echo "ENV CHECK"
echo "============================================================"

which python

python - <<'PY'
import torch
import iopath
import yaml

print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Environment OK")
PY


# ============================================================
# Generate paired-real config
#
# Important:
# Infer the exact shared Box root from the already completed
# generated evaluation, so we use exactly the same GT.
# ============================================================

python - <<'PY'
import json
import os
from pathlib import Path

import yaml


sam_root = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval"
)

preview_root = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/eval/"
    "nuscenesablation/nuplan_merged500"
)

gen_out = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/eval/"
    "nuscenesablation/sam3_nuplan_box300"
)

real_out = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/eval/"
    "nuscenesablation/sam3_nuplan_box300_pairedreal"
)

output_config = sam_root / "config_dwm300_pairedreal.yaml"


# ------------------------------------------------------------
# 1. Verify first300 manifest
# ------------------------------------------------------------

manifest = (
    preview_root
    / "stflow_manifest_first300.jsonl"
)

if not manifest.is_file():
    raise FileNotFoundError(manifest)

video_count = 0

with manifest.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            json.loads(line)
            video_count += 1

print("first300 manifest videos:", video_count)

if video_count != 300:
    raise RuntimeError(
        f"Expected 300 videos, got {video_count}"
    )


# ------------------------------------------------------------
# 2. Infer exact shared Box root from generated SAM records
# ------------------------------------------------------------

record_paths = sorted(
    gen_out.glob("records.rank*.jsonl")
)

if not record_paths:
    raise FileNotFoundError(
        f"No generated SAM records under {gen_out}"
    )

box_manifest_paths = set()

for record_path in record_paths:
    with record_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            value = record.get(
                "box_manifest_path"
            )

            if value:
                box_manifest_paths.add(
                    str(Path(value).resolve())
                )

# All box manifests should live under a common root.
if not box_manifest_paths:
    raise RuntimeError(
        "Cannot infer shared Box root from "
        "existing generated records."
    )

manifest_dirs = [
    str(Path(value).parent)
    for value in box_manifest_paths
]

shared_box_root = Path(
    os.path.commonpath(manifest_dirs)
)

print("Box manifests:")
for value in sorted(box_manifest_paths):
    print(" ", value)

print("Inferred shared_box_root:")
print(" ", shared_box_root)


# ------------------------------------------------------------
# 3. Find an existing SAM config as template
# ------------------------------------------------------------

config_candidates = []

for path in list(sam_root.glob("*.yaml")) + list(
    sam_root.glob("*/*.yaml")
):
    try:
        cfg = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )

        if not isinstance(cfg, dict):
            continue

        score = 0

        output_value = (
            cfg.get("paths", {})
            .get("output_dir")
        )

        preview_value = (
            cfg.get("paths", {})
            .get("preview_root")
        )

        manifest_glob = (
            cfg.get("preview", {})
            .get("manifest_glob", "")
        )

        if output_value:
            try:
                if Path(output_value).resolve() == gen_out.resolve():
                    score += 100
            except Exception:
                pass

        if preview_value:
            try:
                if Path(preview_value).resolve() == preview_root.resolve():
                    score += 30
            except Exception:
                pass

        if "first300" in str(manifest_glob):
            score += 20

        if score > 0:
            config_candidates.append(
                (score, path, cfg)
            )

    except Exception:
        continue


if config_candidates:
    config_candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    score, base_path, config = (
        config_candidates[0]
    )

    print(
        f"Using config template: "
        f"{base_path} (score={score})"
    )

else:
    fallback = (
        sam_root
        / "config_shared_box_nuplan.yaml"
    )

    if not fallback.is_file():
        raise FileNotFoundError(
            "Cannot find suitable SAM config."
        )

    print(
        "Using fallback config:",
        fallback,
    )

    config = yaml.safe_load(
        fallback.read_text(encoding="utf-8")
    )


# ------------------------------------------------------------
# 4. Change ONLY data source/output.
#
# Keep SAM thresholds/matching rules identical.
# ------------------------------------------------------------

config.setdefault("paths", {})

config["paths"]["preview_root"] = str(
    preview_root
)

config["paths"]["shared_box_root"] = str(
    shared_box_root
)

config["paths"]["output_dir"] = str(
    real_out
)


config.setdefault("preview", {})

config["preview"]["manifest_glob"] = (
    "stflow_manifest_first300.jsonl"
)

config["preview"][
    "skip_reference_frames"
] = True

config["preview"]["strict_paths"] = True

# With preview_root pointing directly to nuplan_merged500,
# no method name filtering is needed.
config["preview"]["include_methods"] = []
config["preview"]["exclude_methods"] = []


# Most important change:
# evaluate paired_real instead of generated images.
config["sources"] = {
    "real": {
        "type": "preview_real",
        "group_by_manifest": False,
    }
}


config.setdefault("runtime", {})
config["runtime"]["limit_frames"] = 0
config["runtime"]["overwrite"] = True


# Keep the corrected SAM area rule.
config.setdefault("matching", {})
config["matching"][
    "min_sam_connected_pixels"
] = 0


config.setdefault("visualization", {})
config["visualization"]["enabled"] = True
config["visualization"][
    "max_frames_per_source"
] = 32


output_config.write_text(
    yaml.safe_dump(
        config,
        allow_unicode=True,
        sort_keys=False,
    ),
    encoding="utf-8",
)

print()
print("Generated config:")
print(output_config)

print("preview_root:")
print(config["paths"]["preview_root"])

print("shared_box_root:")
print(config["paths"]["shared_box_root"])

print("output_dir:")
print(config["paths"]["output_dir"])

print("manifest:")
print(config["preview"]["manifest_glob"])

print("sources:")
print(config["sources"])
PY


# ============================================================
# Run paired-real SAM, single GPU
# ============================================================

echo
echo "============================================================"
echo "START paired-real SAM"
echo "GPU 0"
echo "============================================================"

rm -rf "$REAL_OUT"

python -u run_eval.py \
    --config "$CONFIG"


# ============================================================
# Compute GT-centric + RC metrics
# ============================================================

echo
echo "============================================================"
echo "COMPUTE DWM RC METRICS"
echo "============================================================"

python - <<'PY'
import json
from pathlib import Path


ROOT = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/eval/"
    "nuscenesablation"
)

GEN = ROOT / "sam3_nuplan_box300"

REAL = ROOT / "sam3_nuplan_box300_pairedreal"


def iter_records(root):
    paths = sorted(
        root.glob("records.rank*.jsonl")
    )

    if not paths:
        raise FileNotFoundError(
            f"No records under {root}"
        )

    print(
        f"[SCAN] {root.name}: "
        f"{len(paths)} record files"
    )

    for path in paths:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def frame_key(record):
    return (
        str(record["video_id"]),
        int(record["time_index"]),
        str(record["camera_name"]),
    )


# ============================================================
# Load paired-real detectable GT
# ============================================================

real_detectable = {}
real_view_keys = set()

real_gt = 0
real_match = 0
real_iou_sum = 0.0

for record in iter_records(REAL):

    key = frame_key(record)

    if key in real_view_keys:
        raise RuntimeError(
            f"Duplicate real view: {key}"
        )

    real_view_keys.add(key)

    matches = {
        str(match["gt_id"]): float(
            match["mask_iou"]
        )
        for match in record[
            "matching"
        ]["matches"]
    }

    real_detectable[key] = set(
        matches.keys()
    )

    real_gt += int(
        record["gt_count"]
    )

    real_match += len(matches)

    real_iou_sum += sum(
        matches.values()
    )


# ============================================================
# Generated
# ============================================================

gen_view_keys = set()

total_gt = 0
total_match = 0
total_iou = 0.0

rc_gt = 0
rc_match = 0
rc_iou = 0.0

missing_real_views = 0


for record in iter_records(GEN):

    key = frame_key(record)

    if key in gen_view_keys:
        raise RuntimeError(
            f"Duplicate generated view: {key}"
        )

    gen_view_keys.add(key)

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

    total_match += len(matches)

    total_iou += sum(
        matches.values()
    )

    real_ids = real_detectable.get(key)

    if real_ids is None:
        missing_real_views += 1
        continue

    rc_gt += len(real_ids)

    common_ids = (
        real_ids
        .intersection(matches.keys())
    )

    rc_match += len(common_ids)

    rc_iou += sum(
        matches[gt_id]
        for gt_id in common_ids
    )


# ============================================================
# HARD alignment verification
# ============================================================

missing_in_real = (
    gen_view_keys - real_view_keys
)

missing_in_generated = (
    real_view_keys - gen_view_keys
)


print()
print("============================================================")
print("ALIGNMENT CHECK")
print("============================================================")

print(
    "generated views:",
    len(gen_view_keys),
)

print(
    "paired-real views:",
    len(real_view_keys),
)

print(
    "generated missing in real:",
    len(missing_in_real),
)

print(
    "real missing in generated:",
    len(missing_in_generated),
)

print(
    "missing_real_views:",
    missing_real_views,
)


if (
    missing_in_real
    or missing_in_generated
    or missing_real_views
):
    raise RuntimeError(
        "Generated and paired-real frames "
        "are NOT perfectly aligned. "
        "Do not report RC metrics."
    )


# ============================================================
# Metrics
# ============================================================

gt_recall = (
    total_match / total_gt
    if total_gt else 0.0
)

matched_iou = (
    total_iou / total_match
    if total_match else 0.0
)

coverage_iou = (
    total_iou / total_gt
    if total_gt else 0.0
)


rc_recall = (
    rc_match / rc_gt
    if rc_gt else 0.0
)

rc_matched_iou = (
    rc_iou / rc_match
    if rc_match else 0.0
)

rc_coverage_iou = (
    rc_iou / rc_gt
    if rc_gt else 0.0
)


real_recall = (
    real_match / real_gt
    if real_gt else 0.0
)

real_matched_iou = (
    real_iou_sum / real_match
    if real_match else 0.0
)

real_coverage_iou = (
    real_iou_sum / real_gt
    if real_gt else 0.0
)


print()
print("============================================================")
print("PAIRED REAL")
print("============================================================")

print(f"GT                  : {real_gt}")
print(f"Matched             : {real_match}")
print(f"GT Recall           : {real_recall:.6f}")
print(f"Matched MaskIoU     : {real_matched_iou:.6f}")
print(f"Coverage-IoU        : {real_coverage_iou:.6f}")


print()
print("============================================================")
print("DWM 2Hz / first300")
print("============================================================")

print(f"GT                  : {total_gt}")
print(f"Matched             : {total_match}")
print(f"GT Recall           : {gt_recall:.6f}")
print(f"Matched MaskIoU     : {matched_iou:.6f}")
print(f"Coverage-IoU        : {coverage_iou:.6f}")

print()
print(f"RC GT               : {rc_gt}")
print(f"RC Matched          : {rc_match}")
print(f"RC-Recall           : {rc_recall:.6f}")
print(f"RC-Matched-IoU      : {rc_matched_iou:.6f}")
print(
    f"RC-Coverage-IoU     : "
    f"{rc_coverage_iou:.6f}"
)


print()
print("============================================================")
print("MAIN TABLE")
print("============================================================")

print(
    f"{'Method':<22}"
    f"{'Coverage-IoU ↑':>18}"
    f"{'RC-Recall ↑':>16}"
    f"{'RC-Coverage-IoU ↑':>22}"
)

print("-" * 78)

print(
    f"{'DWM (2Hz, 300)':<22}"
    f"{coverage_iou:>18.4f}"
    f"{rc_recall:>16.4f}"
    f"{rc_coverage_iou:>22.4f}"
)
PY


echo
echo "============================================================"
echo "DONE"
echo "============================================================"
