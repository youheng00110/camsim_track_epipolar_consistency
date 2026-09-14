#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 0. Root
# ============================================================

ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim

EVAL=$ROOT/lyh_output/eval/nuscenesablationnew
SAM_ROOT=$ROOT/sam3-eval/sam3-eval

FULL=$EVAL/nuplan6hz_merged300_8cam_full512
CAM3=$EVAL/nuplan6hz_merged300_3cam_512

# 刚才严格抽出来的 300-video Box
BOX_RAW=$EVAL/nuplan6hz_merged300_shared_box

# 自动生成和 512 图匹配的 geometry-only Box
BOX_512=$EVAL/nuplan6hz_merged300_shared_box_512

OUT_ROOT=$EVAL/sam3_nuplan6hz300_box
CFG_ROOT=$OUT_ROOT/configs
LOG_ROOT=$OUT_ROOT/logs
RESULT_ROOT=$OUT_ROOT/results

mkdir -p \
    "$CFG_ROOT" \
    "$LOG_ROOT" \
    "$RESULT_ROOT"


# ============================================================
# 1. Environment
# ============================================================

source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

export PATH=/inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin:$PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

cd "$SAM_ROOT"

echo "============================================================"
echo "ENV"
echo "============================================================"

echo "python = $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python - <<'PY'
import torch
print("torch =", torch.__version__)
print("cuda  =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu   =", torch.cuda.get_device_name(0))
PY


# ============================================================
# 2. Find checkpoint
# ============================================================

CHECKPOINT=""

CANDIDATES=(
    "$ROOT/ckpt/sam3.1/sam3.1_multiplex.pt"
    "$ROOT/sam3-eval/sam3-eval/sam3.1_multiplex.pt"
    "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/ckpt/sam3.1/sam3.1_multiplex.pt"
)

for P in "${CANDIDATES[@]}"; do
    if [[ -f "$P" ]]; then
        CHECKPOINT="$P"
        break
    fi
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: sam3.1_multiplex.pt not found."
    echo "Please copy checkpoint into:"
    echo "$ROOT/ckpt/sam3.1/sam3.1_multiplex.pt"
    exit 1
fi

echo
echo "SAM checkpoint:"
echo "$CHECKPOINT"


# ============================================================
# 3. Inputs
# ============================================================

FULL_MANIFEST=$FULL/stflow_manifest.jsonl
CAM3_MANIFEST=$CAM3/stflow_manifest.jsonl

for P in \
    "$FULL_MANIFEST" \
    "$CAM3_MANIFEST"
do
    if [[ ! -f "$P" ]]; then
        echo "ERROR missing manifest:"
        echo "$P"
        exit 1
    fi
done

if ! find "$BOX_RAW" -name box_manifest.jsonl -type f -print -quit | grep -q .; then
    echo "ERROR: no Box manifest under:"
    echo "$BOX_RAW"
    exit 1
fi


# ============================================================
# 4. Prepare Box geometry for actual target resolution
#
# 自动：
# - 读取 FULL / 3CAM 实际 JPEG 尺寸
# - 要求两组生成图分辨率一致
# - 根据 Box 原 image_size 缩放 lidar_to_image
# - T_lidar_to_camera 不动
# - 删除旧 box image_path，避免加载错误尺寸 Box 图片
# ============================================================

FULL_MANIFEST="$FULL_MANIFEST" \
CAM3_MANIFEST="$CAM3_MANIFEST" \
BOX_RAW="$BOX_RAW" \
BOX_512="$BOX_512" \
python - <<'PY'
import copy
import json
import os
import shutil
from pathlib import Path

from PIL import Image


FULL_MANIFEST = Path(os.environ["FULL_MANIFEST"])
CAM3_MANIFEST = Path(os.environ["CAM3_MANIFEST"])
BOX_RAW = Path(os.environ["BOX_RAW"])
BOX_OUT = Path(os.environ["BOX_512"])


def load_jsonl(path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def resolve(raw, manifest):
    p = Path(str(raw))
    if not p.is_absolute():
        p = manifest.parent / p
    return p.resolve()


def find_sizes(manifest):
    items = load_jsonl(manifest)

    sizes = set()
    cameras = set()
    count = 0

    for video in items:
        for frame in video["frames"]:
            for view in frame["views"]:
                cameras.add(str(view["camera"]))

                p = resolve(
                    view["image_path"],
                    manifest,
                )

                if not p.is_file():
                    raise FileNotFoundError(p)

                # 只抽少量也够判断；
                # 不过全部 metadata 很轻
                with Image.open(p) as im:
                    sizes.add(im.size)

                count += 1

                if len(sizes) > 1:
                    raise RuntimeError(
                        f"Multiple image sizes under {manifest}: "
                        f"{sizes}"
                    )

    return items, sizes, cameras, count


full_items, full_sizes, full_cams, full_count = (
    find_sizes(FULL_MANIFEST)
)

cam3_items, cam3_sizes, cam3_cams, cam3_count = (
    find_sizes(CAM3_MANIFEST)
)


print("=" * 80)
print("TARGET IMAGE SIZE")
print("=" * 80)

print("FULL sizes   =", full_sizes)
print("FULL cameras =", sorted(full_cams))
print("FULL views   =", full_count)

print("3CAM sizes   =", cam3_sizes)
print("3CAM cameras =", sorted(cam3_cams))
print("3CAM views   =", cam3_count)


if len(full_sizes) != 1:
    raise RuntimeError("FULL size is not unique")

if len(cam3_sizes) != 1:
    raise RuntimeError("3CAM size is not unique")

target_size = next(iter(full_sizes))

if next(iter(cam3_sizes)) != target_size:
    raise RuntimeError(
        f"FULL and 3CAM resolution differ: "
        f"{full_sizes} vs {cam3_sizes}"
    )

TARGET_W, TARGET_H = target_size

print()
print("TARGET =", (TARGET_W, TARGET_H))


# ============================================================
# Box
# ============================================================

box_manifests = sorted(
    BOX_RAW.glob("rank_*/box_manifest.jsonl")
)

if not box_manifests:
    raise RuntimeError("No Box manifests")


if BOX_OUT.exists():
    shutil.rmtree(BOX_OUT)


total_videos = 0
total_views = 0
source_sizes = set()


for src_manifest in box_manifests:

    dst_manifest = (
        BOX_OUT
        / src_manifest.relative_to(BOX_RAW)
    )

    dst_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_items = []

    for video in load_jsonl(src_manifest):

        total_videos += 1
        video = copy.deepcopy(video)

        for frame in video.get("frames", []):

            for view in frame.get("views", []):

                total_views += 1

                old_size = view.get("image_size")

                if (
                    isinstance(old_size, (list, tuple))
                    and len(old_size) >= 2
                ):
                    OW = int(round(float(old_size[0])))
                    OH = int(round(float(old_size[1])))

                else:
                    # 如果 manifest 没写 image_size，
                    # 尝试用旧 image_path 的实际尺寸。
                    raw = view.get("image_path")

                    if not raw:
                        raise RuntimeError(
                            "Box view has neither image_size "
                            "nor image_path"
                        )

                    p = resolve(raw, src_manifest)

                    if not p.is_file():
                        raise FileNotFoundError(p)

                    with Image.open(p) as im:
                        OW, OH = im.size

                source_sizes.add((OW, OH))

                sx = TARGET_W / OW
                sy = TARGET_H / OH

                P = view.get("lidar_to_image")

                if P is None:
                    raise RuntimeError(
                        "Box view missing lidar_to_image"
                    )

                P = copy.deepcopy(P)

                # lidar_to_image 可以是 3x4 / 4x4
                if len(P) not in (3, 4):
                    raise RuntimeError(
                        f"Bad projection matrix rows={len(P)}"
                    )

                # 像素 x 行
                for j in range(len(P[0])):
                    P[0][j] = float(P[0][j]) * sx

                # 像素 y 行
                for j in range(len(P[1])):
                    P[1][j] = float(P[1][j]) * sy

                view["lidar_to_image"] = P

                view["image_size"] = [
                    TARGET_W,
                    TARGET_H,
                ]

                # T 不动
                if "T_lidar_to_camera" not in view:
                    raise RuntimeError(
                        "Box view missing T_lidar_to_camera"
                    )

                # 只使用几何投影；
                # 不读取旧尺寸 Box render。
                view.pop("image_path", None)
                view.pop("real_image_path", None)
                view.pop("valid_mask_path", None)

        output_items.append(video)


    with dst_manifest.open(
        "w",
        encoding="utf-8",
    ) as f:

        for video in output_items:
            f.write(
                json.dumps(
                    video,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "write:",
        dst_manifest,
        "videos=",
        len(output_items),
    )


print()
print("=" * 80)
print("BOX PREPARED")
print("=" * 80)

print("videos       =", total_videos)
print("views        =", total_views)
print("source sizes =", sorted(source_sizes))
print("target size  =", (TARGET_W, TARGET_H))
print("output       =", BOX_OUT)

if total_videos != 300:
    raise RuntimeError(
        f"Expected 300 Box videos, got {total_videos}"
    )

print()
print("BOX RESOLUTION PASS")
PY


# ============================================================
# 5. Generate configs
# ============================================================

FULL="$FULL" \
CAM3="$CAM3" \
BOX_512="$BOX_512" \
RESULT_ROOT="$RESULT_ROOT" \
CFG_ROOT="$CFG_ROOT" \
CHECKPOINT="$CHECKPOINT" \
ROOT="$ROOT" \
python - <<'PY'
import copy
import os
from pathlib import Path

import yaml


ROOT = Path(os.environ["ROOT"])
FULL = Path(os.environ["FULL"])
CAM3 = Path(os.environ["CAM3"])
BOX = Path(os.environ["BOX_512"])

RESULT = Path(os.environ["RESULT_ROOT"])
CFG = Path(os.environ["CFG_ROOT"])

CHECKPOINT = os.environ["CHECKPOINT"]


BASE = {
    "paths": {
        "shared_box_root": str(BOX),
        "sam3_repo": str(
            ROOT / "sam3-eval/sam3-eval/sam3"
        ),
        "checkpoint": CHECKPOINT,
    },

    "preview": {
        "manifest_glob": "stflow_manifest.jsonl",
        "skip_reference_frames": True,
        "strict_paths": True,
        "include_methods": [],
        "exclude_methods": [],
    },

    "shared_box": {
        "manifest_glob": "**/box_manifest.jsonl",
        "strict_paths": True,
        "strict_match": True,
    },

    "model": {
        "version": "sam3.1",
        "prompts": [
            "car",
            "truck",
            "bus",
        ],
        "confidence_threshold": 0.25,
        "batch_size": 1,
        "loader_workers": 4,
        "precision": "bfloat16",
        "input_resolution": 1008,
        "save_masks": True,
        "mask_resolution": 256,
        "mask_threshold": 0.5,
        "max_detections_per_prompt": 100,
        "max_detections_per_image": 150,
        "nms_iou_threshold": 0.70,
        "checkpoint_minimum_coverage": 0.95,
        "checkpoint_mmap": True,
    },

    "annotation": {
        "classes": [
            "CAR",
            "TRUCK",
            "BUS",
        ],
        "near_plane": 0.10,
        "min_projected_height_px": 8.0,
        "min_projected_area_px": 64.0,
        "min_in_frame_fraction": 0.10,
    },

    "visibility": {
        "disable_gt_occlusion": True,
        "min_gt_visible_ratio": 0.0,
        "min_gt_visible_connected_pixels": 4,
    },

    "matching": {
        "min_mask_iou": 0.05,
        "max_center_error_norm": 0.80,
        "max_scale_error_log": 1.20,
        "max_cost": 1.10,

        "cost_mask_iou": 0.65,
        "cost_center": 0.20,
        "cost_bottom": 0.05,
        "cost_scale": 0.10,

        "allow_bbox_mask_fallback": False,

        # 不使用 GT 最小面积过滤 SAM
        "min_sam_connected_pixels": 0,
    },

    "visualization": {
        "enabled": True,
        "max_frames_per_source": 16,
        "image_quality": 92,
        "mask_alpha": 0.25,
        "draw_cuboid": True,
        "draw_detection_box": True,
        "draw_filtered_small_sam": True,
    },

    "runtime": {
        "seed": 3407,
        "backend": "sam3.1",
        "limit_frames": 0,
        "overwrite": True,
    },
}


jobs = {
    # --------------------------------------------------------
    # paired-real:
    # 使用 FULL 的真实图，因此 8 camera
    # --------------------------------------------------------
    "pairedreal": {
        "root": FULL,
        "output": RESULT / "pairedreal",
        "sources": {
            "real": {
                "type": "preview_real",
                "group_by_manifest": False,
            }
        },
    },

    "full512": {
        "root": FULL,
        "output": RESULT / "full512",
        "sources": {
            "generated": {
                "type": "preview_generated",
                "group_by_manifest": False,
            }
        },
    },

    "3cam512": {
        "root": CAM3,
        "output": RESULT / "3cam512",
        "sources": {
            "generated": {
                "type": "preview_generated",
                "group_by_manifest": False,
            }
        },
    },
}


for name, job in jobs.items():

    cfg = copy.deepcopy(BASE)

    cfg["paths"]["preview_root"] = str(
        job["root"]
    )

    cfg["paths"]["output_dir"] = str(
        job["output"]
    )

    cfg["sources"] = job["sources"]

    path = CFG / f"{name}.yaml"

    path.write_text(
        yaml.safe_dump(
            cfg,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print()
    print(name)
    print(" preview =", job["root"])
    print(" output  =", job["output"])
PY


# ============================================================
# 6. Scan
#
# FULL:
# 300 × 16 × 8 = 38400
#
# 3CAM:
# 300 × 16 × 3 = 14400
#
# pairedreal follows FULL = 38400
# ============================================================

scan_one() {
    NAME=$1
    EXPECT=$2

    CFG=$CFG_ROOT/$NAME.yaml
    LOG=$LOG_ROOT/${NAME}_scan.log

    echo
    echo "============================================================"
    echo "SCAN $NAME"
    echo "expected = $EXPECT"
    echo "============================================================"

    python -u run_eval.py \
        --config "$CFG" \
        --scan-only \
        2>&1 | tee "$LOG"

    if ! grep -Eq \
        "\"frames\"[[:space:]]*:[[:space:]]*$EXPECT" \
        "$LOG"
    then
        echo "ERROR: $NAME frame count != $EXPECT"
        exit 10
    fi

    if ! grep -Eq \
        "\"shared_box_matched\"[[:space:]]*:[[:space:]]*$EXPECT" \
        "$LOG"
    then
        echo "ERROR: $NAME Box matched != $EXPECT"
        exit 11
    fi

    if ! grep -Eq \
        '"shared_box_missing"[[:space:]]*:[[:space:]]*0' \
        "$LOG"
    then
        echo "ERROR: $NAME has missing Box"
        exit 12
    fi

    echo "[SCAN PASS] $NAME"
}


scan_one pairedreal 38400
scan_one full512    38400
scan_one 3cam512    14400


# ============================================================
# 7. Run SAM sequentially on one GPU
# ============================================================

run_one() {

    NAME=$1

    CFG=$CFG_ROOT/$NAME.yaml
    OUT=$RESULT_ROOT/$NAME
    LOG=$LOG_ROOT/${NAME}.log

    echo
    echo "============================================================"
    echo "RUN $NAME"
    echo "============================================================"

    if [[ -d "$OUT" ]]; then
        BACKUP="${OUT}.bak_$(date +%Y%m%d_%H%M%S)"
        echo "$OUT -> $BACKUP"
        mv "$OUT" "$BACKUP"
    fi

    python -u run_eval.py \
        --config "$CFG" \
        2>&1 | tee "$LOG"

    if [[ ! -f "$OUT/records.rank000.jsonl" ]]; then
        echo "ERROR: records output missing"
        exit 20
    fi

    echo "[RUN PASS] $NAME"
}


run_one pairedreal
run_one full512
run_one 3cam512


# ============================================================
# 8. GT-centric + RC
#
# paired-real 是 8cam。
#
# full512:
#   在全部 8cam 与 real 对齐
#
# 3cam512:
#   自动只在自身 CAM_02/03/04 上
#   使用 paired-real 对应 camera 的 GT 做 RC。
# ============================================================

RESULT_ROOT="$RESULT_ROOT" \
python - <<'PY'
import csv
import json
import os
from pathlib import Path


ROOT = Path(os.environ["RESULT_ROOT"])

REAL = ROOT / "pairedreal"

METHODS = [
    ("full512", ROOT / "full512"),
    ("3cam512", ROOT / "3cam512"),
]


def iter_records(root):

    paths = sorted(
        root.glob("records.rank*.jsonl")
    )

    if not paths:
        raise RuntimeError(
            f"No records under {root}"
        )

    for p in paths:

        with p.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                if line.strip():
                    yield json.loads(line)


def key(r):
    # 使用 shared Box identity，
    # 不依赖 preview local video_id。
    return (
        str(r["box_manifest_path"]),
        str(r["box_video_id"]),
        int(r["time_index"]),
        str(r["camera_name"]),
    )


# ============================================================
# real compact index
# ============================================================

real_index = {}

real_gt = 0
real_match = 0
real_iou = 0.0
real_views = 0


for r in iter_records(REAL):

    k = key(r)

    if k in real_index:
        raise RuntimeError(
            f"Duplicate real view: {k}"
        )

    matches = {
        str(m["gt_id"]): float(
            m["mask_iou"]
        )
        for m in r["matching"]["matches"]
    }

    real_index[k] = matches

    real_views += 1
    real_gt += int(r["gt_count"])
    real_match += len(matches)
    real_iou += sum(matches.values())


if real_views != 38400:
    raise RuntimeError(
        f"Expected 38400 real views, got {real_views}"
    )


real_recall = (
    real_match / real_gt
    if real_gt else 0
)

real_matched_iou = (
    real_iou / real_match
    if real_match else 0
)

real_coverage = (
    real_iou / real_gt
    if real_gt else 0
)


print()
print("=" * 80)
print("PAIRED REAL — 8CAM")
print("=" * 80)

print("views           =", real_views)
print("GT              =", real_gt)
print("matched         =", real_match)
print(f"GT Recall       = {real_recall:.6f}")
print(f"Matched IoU     = {real_matched_iou:.6f}")
print(f"Coverage-IoU    = {real_coverage:.6f}")


rows = []

# 注意：
# paired-real 主行是 8cam reference。
rows.append({
    "method": "paired-real (8cam)",
    "coverage_iou": real_coverage,
    "rc_recall": 1.0,
    "rc_coverage_iou": real_matched_iou,
})


# ============================================================
# methods
# ============================================================

for name, root in METHODS:

    total_gt = 0
    total_match = 0
    total_iou = 0.0

    rc_gt = 0
    rc_match = 0
    rc_iou = 0.0

    views = 0
    missing_real = 0

    method_keys = set()


    for r in iter_records(root):

        k = key(r)

        if k in method_keys:
            raise RuntimeError(
                f"{name}: duplicate view {k}"
            )

        method_keys.add(k)

        views += 1

        matches = {
            str(m["gt_id"]): float(
                m["mask_iou"]
            )
            for m in r["matching"]["matches"]
        }

        total_gt += int(r["gt_count"])
        total_match += len(matches)
        total_iou += sum(matches.values())

        real_matches = real_index.get(k)

        if real_matches is None:
            missing_real += 1
            continue

        real_ids = set(real_matches)

        rc_gt += len(real_ids)

        common = (
            real_ids
            & set(matches)
        )

        rc_match += len(common)

        rc_iou += sum(
            matches[x]
            for x in common
        )


    if missing_real:
        raise RuntimeError(
            f"{name}: missing real views = "
            f"{missing_real}"
        )


    recall = (
        total_match / total_gt
        if total_gt else 0
    )

    matched_iou = (
        total_iou / total_match
        if total_match else 0
    )

    coverage = (
        total_iou / total_gt
        if total_gt else 0
    )

    rc_recall = (
        rc_match / rc_gt
        if rc_gt else 0
    )

    rc_matched_iou = (
        rc_iou / rc_match
        if rc_match else 0
    )

    rc_coverage = (
        rc_iou / rc_gt
        if rc_gt else 0
    )


    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print("views              =", views)
    print("GT                 =", total_gt)
    print("matched            =", total_match)

    print(f"GT Recall          = {recall:.6f}")
    print(f"Matched MaskIoU    = {matched_iou:.6f}")
    print(f"Coverage-IoU       = {coverage:.6f}")

    print("RC GT              =", rc_gt)
    print("RC matched         =", rc_match)

    print(f"RC-Recall          = {rc_recall:.6f}")
    print(f"RC-Matched-IoU     = {rc_matched_iou:.6f}")
    print(f"RC-Coverage-IoU    = {rc_coverage:.6f}")

    rows.append({
        "method": name,
        "coverage_iou": coverage,
        "rc_recall": rc_recall,
        "rc_coverage_iou": rc_coverage,
    })


# ============================================================
# Table
# ============================================================

print()
print("=" * 86)
print("MAIN TABLE")
print("=" * 86)

print(
    f"{'Method':<28}"
    f"{'Coverage-IoU ↑':>18}"
    f"{'RC-Recall ↑':>16}"
    f"{'RC-Coverage-IoU ↑':>22}"
)

print("-" * 84)

for r in rows:

    print(
        f"{r['method']:<28}"
        f"{r['coverage_iou']:>18.4f}"
        f"{r['rc_recall']:>16.4f}"
        f"{r['rc_coverage_iou']:>22.4f}"
    )


out = ROOT / "nuplan6hz300_box_main_table.csv"

with out.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=[
            "method",
            "coverage_iou",
            "rc_recall",
            "rc_coverage_iou",
        ],
    )

    w.writeheader()
    w.writerows(rows)


print()
print("CSV =", out)
PY


echo
echo "============================================================"
echo "ALL DONE"
echo "============================================================"
