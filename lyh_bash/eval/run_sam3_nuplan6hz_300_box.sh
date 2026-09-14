#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 0. ROOT / ENV
# ============================================================

ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim

EVAL=$ROOT/lyh_output/eval/nuscenesablationnew

SAM_ROOT=$ROOT/sam3-eval/sam3-eval

FULL_ROOT=$EVAL/nuplan6hz_merged300_8cam_full512
CAM3_ROOT=$EVAL/nuplan6hz_merged300_3cam_512

# 已经通过 pose signature 严格筛出来的300视频 Box
BOX_RAW=$EVAL/nuplan6hz_merged300_shared_box

# 真正给 SAM 用的 geometry-only / 512 投影 Box
BOX_EVAL=$EVAL/nuplan6hz_merged300_shared_box_eval512

OUT_ROOT=$EVAL/sam3_nuplan6hz_merged300_box512
CFG_ROOT=$OUT_ROOT/configs
LOG_ROOT=$OUT_ROOT/logs
RESULT_ROOT=$OUT_ROOT/results

mkdir -p \
    "$OUT_ROOT" \
    "$CFG_ROOT" \
    "$LOG_ROOT" \
    "$RESULT_ROOT"


# ------------------------------------------------------------
# IMPORTANT: 新区域环境
# ------------------------------------------------------------

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

export PATH=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin:$PATH

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# 外部可覆盖，例如：
# CUDA_VISIBLE_DEVICES=3 bash xxx.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$SAM_ROOT"


echo "============================================================"
echo "ENV CHECK"
echo "============================================================"

echo "python = $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python - <<'PY'
import torch

print("torch =", torch.__version__)
print("CUDA available =", torch.cuda.is_available())
print("GPU count =", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

print("GPU =", torch.cuda.get_device_name(0))
PY


# ============================================================
# 1. Manifest / directory checks
# ============================================================

FULL_MANIFEST=$FULL_ROOT/stflow_manifest.jsonl
CAM3_MANIFEST=$CAM3_ROOT/stflow_manifest.jsonl

for P in \
    "$FULL_MANIFEST" \
    "$CAM3_MANIFEST"
do
    if [[ ! -f "$P" ]]; then
        echo "ERROR: missing manifest:"
        echo "$P"
        exit 1
    fi
done

if [[ ! -d "$BOX_RAW" ]]; then
    echo "ERROR: missing borrowed Box root:"
    echo "$BOX_RAW"
    exit 1
fi

if ! find "$BOX_RAW" \
    -type f \
    -name box_manifest.jsonl \
    -print -quit | grep -q .
then
    echo "ERROR: no box_manifest.jsonl under:"
    echo "$BOX_RAW"
    exit 1
fi

if [[ ! -f "$SAM_ROOT/run_eval.py" ]]; then
    echo "ERROR: SAM evaluator missing"
    exit 1
fi


# ============================================================
# 2. SAM3 checkpoint
# ============================================================

CHECKPOINT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt/sam3.1/sam3.1_multiplex.pt"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: SAM3 checkpoint missing:"
    echo "$CHECKPOINT"
    exit 2
fi

echo
echo "[OK] checkpoint:"
echo "$CHECKPOINT"


# ============================================================
# 3. Prepare exact geometry-only Box @ target resolution
#
# Critical:
#
# - borrowed Box identity already verified = exact 300 videos
# - remove old Box image_path
# - preserve box7_lidar / T_lidar_to_camera
# - if Box projection matrix is not 512 coordinate system:
#       row0 *= target_w / old_w
#       row1 *= target_h / old_h
#
# FULL and 3CAM must agree on camera image sizes.
# ============================================================

FULL_MANIFEST="$FULL_MANIFEST" \
CAM3_MANIFEST="$CAM3_MANIFEST" \
BOX_RAW="$BOX_RAW" \
BOX_EVAL="$BOX_EVAL" \
python - <<'PY'
from __future__ import annotations

import copy
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


FULL_MANIFEST = Path(os.environ["FULL_MANIFEST"])
CAM3_MANIFEST = Path(os.environ["CAM3_MANIFEST"])
BOX_RAW = Path(os.environ["BOX_RAW"])
BOX_EVAL = Path(os.environ["BOX_EVAL"])


def load_jsonl(path):
    result = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result.append(json.loads(line))

    return result


def resolve_path(raw, manifest):
    p = Path(str(raw))

    if not p.is_absolute():
        p = manifest.parent / p

    return p.resolve()


def collect_target_sizes(manifest):
    sizes = defaultdict(set)

    items = load_jsonl(manifest)

    for video in items:
        for frame in video["frames"]:
            for view in frame["views"]:
                cam = str(view["camera"])

                size = view.get("image_size")

                if (
                    isinstance(size, (list, tuple))
                    and len(size) >= 2
                    and int(size[0]) > 0
                    and int(size[1]) > 0
                ):
                    sizes[cam].add(
                        (
                            int(size[0]),
                            int(size[1]),
                        )
                    )
                    continue

                image_path = resolve_path(
                    view["image_path"],
                    manifest,
                )

                with Image.open(image_path) as im:
                    sizes[cam].add(im.size)

    return {
        cam: values
        for cam, values in sizes.items()
    }


full_sizes = collect_target_sizes(
    FULL_MANIFEST
)

cam3_sizes = collect_target_sizes(
    CAM3_MANIFEST
)


print()
print("=" * 90)
print("TARGET IMAGE SIZES")
print("=" * 90)

print("FULL:")
for cam, values in sorted(full_sizes.items()):
    print(" ", cam, values)

print("3CAM:")
for cam, values in sorted(cam3_sizes.items()):
    print(" ", cam, values)


# Every camera should have exactly one resolution.
for name, mapping in [
    ("FULL", full_sizes),
    ("3CAM", cam3_sizes),
]:
    for cam, values in mapping.items():
        if len(values) != 1:
            raise RuntimeError(
                f"{name} {cam} has multiple image sizes: "
                f"{values}"
            )


# Shared cameras must agree.
for cam in set(full_sizes) & set(cam3_sizes):

    if full_sizes[cam] != cam3_sizes[cam]:
        raise RuntimeError(
            f"Image size mismatch for {cam}: "
            f"FULL={full_sizes[cam]} "
            f"3CAM={cam3_sizes[cam]}"
        )


target_size = {
    cam: next(iter(values))
    for cam, values in full_sizes.items()
}


print()
print("Resolved target:")
print(target_size)


# Expected in this experiment.
for cam, size in target_size.items():

    if size[0] != 512:
        raise RuntimeError(
            f"Expected width=512 but "
            f"{cam} is {size}"
        )


# ------------------------------------------------------------
# Copy / adapt Box manifests
# ------------------------------------------------------------

if BOX_EVAL.exists():
    shutil.rmtree(BOX_EVAL)

box_manifests = sorted(
    BOX_RAW.glob(
        "rank_*/box_manifest.jsonl"
    )
)

if not box_manifests:
    raise RuntimeError(
        "No borrowed Box manifests"
    )


total_videos = 0
total_views = 0
scaled_views = 0
already_target_views = 0


for src_manifest in box_manifests:

    dst_manifest = (
        BOX_EVAL
        / src_manifest.parent.name
        / "box_manifest.jsonl"
    )

    dst_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = []


    for video in load_jsonl(src_manifest):

        total_videos += 1

        video = copy.deepcopy(video)


        for frame in video.get(
            "frames",
            [],
        ):

            for view in frame.get(
                "views",
                [],
            ):

                cam = str(
                    view.get("camera")
                )

                # Box has all 8 cameras.
                # target_size should therefore contain it.
                if cam not in target_size:
                    # Ignore extra camera geometry only if
                    # target FULL doesn't evaluate it.
                    continue


                tw, th = target_size[cam]

                # Determine original Box coordinate size.
                old_size = view.get(
                    "image_size"
                )

                ow = None
                oh = None

                if (
                    isinstance(old_size, (list, tuple))
                    and len(old_size) >= 2
                ):
                    ow = int(old_size[0])
                    oh = int(old_size[1])


                # If manifest has no size, inspect old Box image.
                if (
                    (not ow or not oh)
                    and view.get("image_path")
                ):

                    p = resolve_path(
                        view["image_path"],
                        src_manifest,
                    )

                    if p.is_file():
                        with Image.open(p) as im:
                            ow, oh = im.size


                if not ow or not oh:
                    raise RuntimeError(
                        "Cannot determine Box projection "
                        f"resolution for "
                        f"{video.get('video_id')} "
                        f"{cam}"
                    )


                # ------------------------------------------------
                # Scale projection matrix when needed.
                # ------------------------------------------------

                if (ow, oh) != (tw, th):

                    sx = tw / float(ow)
                    sy = th / float(oh)

                    P = view.get(
                        "lidar_to_image"
                    )

                    if P is None:
                        raise RuntimeError(
                            "Box view missing lidar_to_image: "
                            f"{video.get('video_id')} "
                            f"{cam}"
                        )

                    P = copy.deepcopy(P)

                    if (
                        not isinstance(P, list)
                        or len(P) not in (3, 4)
                    ):
                        raise RuntimeError(
                            f"Unexpected projection matrix: {P}"
                        )

                    # pixel x equation
                    P[0] = [
                        float(v) * sx
                        for v in P[0]
                    ]

                    # pixel y equation
                    P[1] = [
                        float(v) * sy
                        for v in P[1]
                    ]

                    view[
                        "lidar_to_image"
                    ] = P


                    # Optional K metadata.
                    if view.get("K") is not None:

                        K = copy.deepcopy(
                            view["K"]
                        )

                        if (
                            isinstance(K, list)
                            and len(K) == 3
                        ):
                            K[0] = [
                                float(v) * sx
                                for v in K[0]
                            ]

                            K[1] = [
                                float(v) * sy
                                for v in K[1]
                            ]

                            view["K"] = K


                    scaled_views += 1

                else:
                    already_target_views += 1


                view["image_size"] = [
                    tw,
                    th,
                ]


                # ------------------------------------------------
                # Geometry-only!
                #
                # This avoids high-resolution Box image being
                # attached and compared against 512 generated image.
                # ------------------------------------------------

                view.pop(
                    "image_path",
                    None,
                )

                view.pop(
                    "real_image_path",
                    None,
                )

                view.pop(
                    "valid_mask_path",
                    None,
                )

                total_views += 1


        output.append(video)


    with dst_manifest.open(
        "w",
        encoding="utf-8",
    ) as f:

        for item in output:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


    print(
        "write:",
        dst_manifest,
        "videos=",
        len(output),
    )


print()
print("=" * 90)
print("BOX PREPARED")
print("=" * 90)

print("videos               =", total_videos)
print("views                =", total_views)
print("scaled views         =", scaled_views)
print(
    "already target views =",
    already_target_views,
)
print("output               =", BOX_EVAL)

if total_videos != 300:
    raise RuntimeError(
        f"Expected 300 Box videos, got "
        f"{total_videos}"
    )

print()
print("BOX PREPARE PASS")
PY


# ============================================================
# 4. Generate configs
# ============================================================

ROOT="$ROOT" \
FULL_ROOT="$FULL_ROOT" \
CAM3_ROOT="$CAM3_ROOT" \
BOX_EVAL="$BOX_EVAL" \
CHECKPOINT="$CHECKPOINT" \
CFG_ROOT="$CFG_ROOT" \
RESULT_ROOT="$RESULT_ROOT" \
python - <<'PY'
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


ROOT = Path(os.environ["ROOT"])
FULL = Path(os.environ["FULL_ROOT"])
CAM3 = Path(os.environ["CAM3_ROOT"])
BOX = Path(os.environ["BOX_EVAL"])
CFG = Path(os.environ["CFG_ROOT"])
RESULT = Path(os.environ["RESULT_ROOT"])

CHECKPOINT = os.environ["CHECKPOINT"]

SAM_REPO = (
    ROOT
    / "sam3-eval"
    / "sam3-eval"
    / "sam3"
)


base = {

    "paths": {
        "shared_box_root":
            str(BOX),

        "sam3_repo":
            str(SAM_REPO),

        "checkpoint":
            CHECKPOINT,
    },


    "preview": {
        "manifest_glob":
            "stflow_manifest.jsonl",

        "skip_reference_frames":
            True,

        "strict_paths":
            True,

        "include_methods":
            [],

        "exclude_methods":
            [],
    },


    "shared_box": {
        "manifest_glob":
            "**/box_manifest.jsonl",

        "strict_paths":
            True,

        "strict_match":
            True,
    },


    "model": {
        "version":
            "sam3.1",

        "prompts": [
            "car",
            "truck",
            "bus",
        ],

        "confidence_threshold":
            0.25,

        "batch_size":
            1,

        "loader_workers":
            4,

        "precision":
            "bfloat16",

        "input_resolution":
            1008,

        "save_masks":
            True,

        "mask_resolution":
            256,

        "mask_threshold":
            0.5,

        "max_detections_per_prompt":
            100,

        "max_detections_per_image":
            150,

        "nms_iou_threshold":
            0.70,

        "checkpoint_minimum_coverage":
            0.95,

        "checkpoint_mmap":
            True,
    },


    "annotation": {
        "classes": [
            "CAR",
            "TRUCK",
            "BUS",
        ],

        "near_plane":
            0.10,

        "min_projected_height_px":
            8.0,

        "min_projected_area_px":
            64.0,

        "min_in_frame_fraction":
            0.10,
    },


    "visibility": {
        "disable_gt_occlusion":
            True,

        "min_gt_visible_ratio":
            0.0,

        "min_gt_visible_connected_pixels":
            4,
    },


    "matching": {
        "min_mask_iou":
            0.05,

        "max_center_error_norm":
            0.80,

        "max_scale_error_log":
            1.20,

        "max_cost":
            1.10,

        "cost_mask_iou":
            0.65,

        "cost_center":
            0.20,

        "cost_bottom":
            0.05,

        "cost_scale":
            0.10,

        "allow_bbox_mask_fallback":
            False,

        # Important:
        # independent of GT size.
        "min_sam_connected_pixels":
            0,
    },


    "visualization": {
        "enabled":
            True,

        "max_frames_per_source":
            16,

        "image_quality":
            92,

        "mask_alpha":
            0.25,

        "draw_cuboid":
            True,

        "draw_detection_box":
            True,

        "draw_filtered_small_sam":
            True,
    },


    "runtime": {
        "seed":
            3407,

        "backend":
            "sam3.1",

        "limit_frames":
            0,

        "overwrite":
            True,
    },
}


jobs = [

    # --------------------------------------------------------
    # paired real:
    # use FULL manifest because it contains all 8 real cameras.
    # --------------------------------------------------------

    (
        "pairedreal",
        FULL,
        RESULT / "pairedreal",
        {
            "real": {
                "type":
                    "preview_real",

                "group_by_manifest":
                    False,
            },
        },
    ),


    (
        "full512",
        FULL,
        RESULT / "full512",
        {
            "generated": {
                "type":
                    "preview_generated",

                "group_by_manifest":
                    False,
            },
        },
    ),


    (
        "3cam512",
        CAM3,
        RESULT / "3cam512",
        {
            "generated": {
                "type":
                    "preview_generated",

                "group_by_manifest":
                    False,
            },
        },
    ),
]


for (
    name,
    preview_root,
    output_dir,
    sources,
) in jobs:

    cfg = copy.deepcopy(base)

    cfg["paths"][
        "preview_root"
    ] = str(
        preview_root
    )

    cfg["paths"][
        "output_dir"
    ] = str(
        output_dir
    )

    cfg["sources"] = sources


    output = (
        CFG / f"{name}.yaml"
    )


    with output.open(
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


    print()
    print(name)
    print(
        " preview =",
        preview_root,
    )
    print(
        " output  =",
        output_dir,
    )
    print(
        " source  =",
        sources,
    )
PY


# ============================================================
# 5. SCAN ONLY
#
# Must pass before SAM inference.
#
# pairedreal = 300 * 16 * 8 = 38400
# full512    = 300 * 16 * 8 = 38400
# 3cam512    = 300 * 16 * 3 = 14400
# ============================================================

scan_one () {

    NAME="$1"
    EXPECT="$2"

    CONFIG="$CFG_ROOT/${NAME}.yaml"
    LOG="$LOG_ROOT/${NAME}_scan.log"


    echo
    echo "============================================================"
    echo "SCAN: $NAME"
    echo "EXPECTED: $EXPECT"
    echo "============================================================"


    python -u run_eval.py \
        --config "$CONFIG" \
        --scan-only \
        2>&1 | tee "$LOG"


    if ! grep -Eq \
        "\"frames\"[[:space:]]*:[[:space:]]*${EXPECT}" \
        "$LOG"
    then
        echo "ERROR: $NAME frames != $EXPECT"
        exit 20
    fi


    if ! grep -Eq \
        "\"shared_box_matched\"[[:space:]]*:[[:space:]]*${EXPECT}" \
        "$LOG"
    then
        echo "ERROR: $NAME shared Box matched != $EXPECT"
        exit 21
    fi


    if ! grep -Eq \
        '"shared_box_missing"[[:space:]]*:[[:space:]]*0' \
        "$LOG"
    then
        echo "ERROR: $NAME has missing Box"
        exit 22
    fi


    echo
    echo "[SCAN PASS] $NAME"
}


scan_one pairedreal 38400
scan_one full512 38400
scan_one 3cam512 14400


# ============================================================
# 6. RUN SAM — single GPU sequentially
# ============================================================

run_one () {

    NAME="$1"

    CONFIG="$CFG_ROOT/${NAME}.yaml"
    OUTPUT="$RESULT_ROOT/${NAME}"
    LOG="$LOG_ROOT/${NAME}.log"


    echo
    echo "============================================================"
    echo "RUN SAM: $NAME"
    echo "============================================================"


    # Preserve any previous result.
    if [[ -d "$OUTPUT" ]]; then

        BACKUP="${OUTPUT}.bak_$(date +%Y%m%d_%H%M%S)"

        echo "backup:"
        echo "$OUTPUT"
        echo " -> $BACKUP"

        mv \
            "$OUTPUT" \
            "$BACKUP"
    fi


    python -u run_eval.py \
        --config "$CONFIG" \
        2>&1 | tee "$LOG"


    if [[ ! -f "$OUTPUT/records.rank000.jsonl" ]]; then
        echo "ERROR: records file missing:"
        echo "$OUTPUT"
        exit 30
    fi


    echo
    echo "[SAM PASS] $NAME"
}


run_one pairedreal
run_one full512
run_one 3cam512


# ============================================================
# 7. GT-centric + RC
#
# pairedreal:
#   all 8 cams once.
#
# full512:
#   compare against all corresponding real 8cam views.
#
# 3cam512:
#   compare ONLY against real CAM_02/03/04 subset.
#
# No second paired-real SAM run needed.
# ============================================================

RESULT_ROOT="$RESULT_ROOT" \
python - <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path


ROOT = Path(
    os.environ["RESULT_ROOT"]
)

REAL = ROOT / "pairedreal"

METHODS = [

    (
        "full512",
        ROOT / "full512",
    ),

    (
        "3cam512",
        ROOT / "3cam512",
    ),
]


def iter_records(root):

    paths = sorted(
        root.glob(
            "records.rank*.jsonl"
        )
    )

    if not paths:
        raise FileNotFoundError(
            f"No records under {root}"
        )

    for path in paths:

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                if line.strip():
                    yield json.loads(
                        line
                    )


def view_key(r):
    """
    Stable because all three runs attach exactly the same
    borrowed shared Box subset.

    manifest path disambiguates local video_id repeated
    across rank_00/rank_01.
    """

    return (
        str(
            r["box_manifest_path"]
        ),

        str(
            r["box_video_id"]
        ),

        int(
            r["time_index"]
        ),

        str(
            r["camera_name"]
        ),
    )


# ============================================================
# Paired-real compact index
# ============================================================

real = {}

real_views = 0
real_gt = 0
real_match = 0
real_iou = 0.0


for r in iter_records(
    REAL
):

    key = view_key(r)

    if key in real:
        raise RuntimeError(
            f"Duplicate real view: "
            f"{key}"
        )


    matches = {
        str(m["gt_id"]):
            float(m["mask_iou"])

        for m in
        r["matching"]["matches"]
    }


    real[key] = {
        "matches":
            matches,

        "gt_count":
            int(r["gt_count"]),

        "camera":
            str(
                r["camera_name"]
            ),
    }


    real_views += 1
    real_gt += int(
        r["gt_count"]
    )
    real_match += len(
        matches
    )
    real_iou += sum(
        matches.values()
    )


if real_views != 38400:
    raise RuntimeError(
        f"Expected 38400 real views, "
        f"got {real_views}"
    )


real_recall = (
    real_match / real_gt
    if real_gt else 0.0
)

real_matched_iou = (
    real_iou / real_match
    if real_match else 0.0
)

real_coverage = (
    real_iou / real_gt
    if real_gt else 0.0
)


# ============================================================
# Real 3-camera subset diagnostics
# ============================================================

CAM3 = {
    "CAM_02",
    "CAM_03",
    "CAM_04",
}

real3_gt = 0
real3_match = 0
real3_iou = 0.0
real3_views = 0


for item in real.values():

    if item["camera"] not in CAM3:
        continue

    real3_views += 1
    real3_gt += item["gt_count"]

    m = item["matches"]

    real3_match += len(m)
    real3_iou += sum(
        m.values()
    )


if real3_views != 14400:
    raise RuntimeError(
        f"Expected 14400 real 3cam views, "
        f"got {real3_views}"
    )


real3_recall = (
    real3_match / real3_gt
    if real3_gt else 0.0
)

real3_matched_iou = (
    real3_iou / real3_match
    if real3_match else 0.0
)

real3_coverage = (
    real3_iou / real3_gt
    if real3_gt else 0.0
)


print()
print("=" * 90)
print("PAIRED REAL — 8 CAM")
print("=" * 90)

print(
    "views             =",
    real_views,
)
print(
    "GT                =",
    real_gt,
)
print(
    "matched           =",
    real_match,
)
print(
    f"GT Recall         = "
    f"{real_recall:.6f}"
)
print(
    f"Matched MaskIoU   = "
    f"{real_matched_iou:.6f}"
)
print(
    f"Coverage-IoU      = "
    f"{real_coverage:.6f}"
)


print()
print("=" * 90)
print("PAIRED REAL — CAM_02/03/04 SUBSET")
print("=" * 90)

print(
    "views             =",
    real3_views,
)
print(
    "GT                =",
    real3_gt,
)
print(
    "matched           =",
    real3_match,
)
print(
    f"GT Recall         = "
    f"{real3_recall:.6f}"
)
print(
    f"Matched MaskIoU   = "
    f"{real3_matched_iou:.6f}"
)
print(
    f"Coverage-IoU      = "
    f"{real3_coverage:.6f}"
)


rows = []


# ============================================================
# Generated methods
# ============================================================

for name, root in METHODS:

    views = 0

    seen = set()

    total_gt = 0
    total_match = 0
    total_iou = 0.0

    rc_gt = 0
    rc_match = 0
    rc_iou = 0.0

    missing_real = 0


    for r in iter_records(
        root
    ):

        key = view_key(r)

        if key in seen:
            raise RuntimeError(
                f"{name}: duplicate view "
                f"{key}"
            )

        seen.add(key)
        views += 1


        matches = {

            str(m["gt_id"]):
                float(m["mask_iou"])

            for m in
            r["matching"]["matches"]
        }


        total_gt += int(
            r["gt_count"]
        )

        total_match += len(
            matches
        )

        total_iou += sum(
            matches.values()
        )


        real_item = real.get(
            key
        )

        if real_item is None:

            missing_real += 1
            continue


        real_ids = set(
            real_item["matches"]
        )

        rc_gt += len(
            real_ids
        )


        common = (
            real_ids
            & set(matches)
        )


        rc_match += len(
            common
        )


        rc_iou += sum(
            matches[gt_id]
            for gt_id in common
        )


    expected = (
        38400
        if name == "full512"
        else 14400
    )


    if views != expected:
        raise RuntimeError(
            f"{name}: expected "
            f"{expected} views, "
            f"got {views}"
        )


    if missing_real:
        raise RuntimeError(
            f"{name}: "
            f"missing_real={missing_real}"
        )


    gt_recall = (
        total_match / total_gt
        if total_gt else 0.0
    )


    matched_iou = (
        total_iou / total_match
        if total_match else 0.0
    )


    coverage = (
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


    rc_coverage = (
        rc_iou / rc_gt
        if rc_gt else 0.0
    )


    print()
    print("=" * 90)
    print(name)
    print("=" * 90)

    print(
        "views             =",
        views,
    )
    print(
        "GT                =",
        total_gt,
    )
    print(
        "matched           =",
        total_match,
    )

    print(
        f"GT Recall         = "
        f"{gt_recall:.6f}"
    )

    print(
        f"Matched MaskIoU   = "
        f"{matched_iou:.6f}"
    )

    print(
        f"Coverage-IoU      = "
        f"{coverage:.6f}"
    )

    print(
        "RC GT             =",
        rc_gt,
    )

    print(
        "RC matched        =",
        rc_match,
    )

    print(
        f"RC-Recall         = "
        f"{rc_recall:.6f}"
    )

    print(
        f"RC-Matched-IoU    = "
        f"{rc_matched_iou:.6f}"
    )

    print(
        f"RC-Coverage-IoU   = "
        f"{rc_coverage:.6f}"
    )


    rows.append(
        {
            "method":
                name,

            "coverage_iou":
                coverage,

            "rc_recall":
                rc_recall,

            "rc_coverage_iou":
                rc_coverage,
        }
    )


# ============================================================
# Main table
# ============================================================

print()
print("=" * 88)
print("MAIN TABLE")
print("=" * 88)

print(
    f"{'Method':<24}"
    f"{'Coverage-IoU ↑':>18}"
    f"{'RC-Recall ↑':>16}"
    f"{'RC-Coverage-IoU ↑':>22}"
)

print("-" * 82)


# paired real 8cam control
print(
    f"{'paired-real (8cam)':<24}"
    f"{real_coverage:>18.4f}"
    f"{1.0:>16.4f}"
    f"{real_matched_iou:>22.4f}"
)


for row in rows:

    print(
        f"{row['method']:<24}"
        f"{row['coverage_iou']:>18.4f}"
        f"{row['rc_recall']:>16.4f}"
        f"{row['rc_coverage_iou']:>22.4f}"
    )


print()
print("3CAM paired-real reference:")
print(
    f"Coverage-IoU={real3_coverage:.4f}, "
    f"RC-Recall=1.0000, "
    f"RC-Coverage-IoU="
    f"{real3_matched_iou:.4f}"
)


# ============================================================
# CSV
# ============================================================

CSV_PATH = (
    ROOT
    / "nuplan6hz_300_box_main_table.csv"
)


with CSV_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "method",
            "coverage_iou",
            "rc_recall",
            "rc_coverage_iou",
        ],
    )

    writer.writeheader()

    writer.writerow(
        {
            "method":
                "paired-real (8cam)",

            "coverage_iou":
                real_coverage,

            "rc_recall":
                1.0,

            "rc_coverage_iou":
                real_matched_iou,
        }
    )

    writer.writerows(
        rows
    )


print()
print("CSV:")
print(CSV_PATH)
PY


echo
echo "============================================================"
echo "ALL DONE"
echo "============================================================"

echo "Result root:"
echo "$RESULT_ROOT"

echo
echo "Main CSV:"
echo "$RESULT_ROOT/nuplan6hz_300_box_main_table.csv"

