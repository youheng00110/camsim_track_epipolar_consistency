#!/bin/bash

set -euo pipefail


# ============================================================
# 0. new area 环境
# ============================================================

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


# ============================================================
# 1. OpenDWM
# ============================================================

export CAMSIM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim
export OPENDWM_ROOT=$CAMSIM_ROOT/OpenDWM

cd "$OPENDWM_ROOT/src" || exit 1

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

[ -d "$OPENDWM_ROOT/externals/TATS/tats/fvd" ] && \
    export PYTHONPATH="$OPENDWM_ROOT/externals/TATS/tats/fvd:$PYTHONPATH"

[ -d "$CAMSIM_ROOT/nuplan-devkit-master" ] && \
    export PYTHONPATH="$CAMSIM_ROOT/nuplan-devkit-master:$PYTHONPATH"

[ -d "$OPENDWM_ROOT/externals/waymo-open-dataset/src" ] && \
    export PYTHONPATH="$OPENDWM_ROOT/externals/waymo-open-dataset/src:$PYTHONPATH"


# ============================================================
# 2. 本地已有权重
# ============================================================

PRETRAIN_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain
CKPT_ROOT=$PRETRAIN_ROOT/ckpt

RAFT_LOCAL=$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth
LOFTR_LOCAL=$CKPT_ROOT/loftr_outdoor.ckpt
I3D_CHECKPOINT=$CKPT_ROOT/i3d_pretrained_400.pt


for P in \
    "$RAFT_LOCAL" \
    "$LOFTR_LOCAL" \
    "$I3D_CHECKPOINT"
do
    if [ ! -f "$P" ]; then
        echo "ERROR: checkpoint missing:"
        echo "$P"
        exit 1
    fi
done


export TORCH_HOME=/root/.cache/torch
CACHE_DIR=$TORCH_HOME/hub/checkpoints

mkdir -p "$CACHE_DIR"

cp -f \
    "$RAFT_LOCAL" \
    "$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

cp -f \
    "$LOFTR_LOCAL" \
    "$CACHE_DIR/loftr_outdoor.ckpt"


echo
echo "============================================================"
echo "LOCAL WEIGHTS"
echo "============================================================"

ls -lh \
    "$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth" \
    "$CACHE_DIR/loftr_outdoor.ckpt" \
    "$I3D_CHECKPOINT"


# ============================================================
# 3. 数据
# ============================================================

BASE=$CAMSIM_ROOT/lyh_output/eval/nuscenesablationnew

SRC=$BASE/nuplan6hz_merged300
DST=$BASE/nuplan6hz_merged300_3cam_512

SRC_MANIFEST=$SRC/stflow_manifest.jsonl
DST_MANIFEST=$DST/stflow_manifest.jsonl

TARGET_W=512
TARGET_H=288

MAX_VIDEOS=300
SEQ_COUNT=19
GATE=16


# ------------------------------------------------------------
# 关键：
#
# 原 generic order:
#
# CAM_00 = CAM_L2
# CAM_01 = CAM_L1
# CAM_02 = CAM_L0
# CAM_03 = CAM_F0
# CAM_04 = CAM_R0
# CAM_05 = CAM_R1
# CAM_06 = CAM_R2
# CAM_07 = CAM_B0
#
# 所以前向三摄：
#
# CAM_02 / CAM_03 / CAM_04
# ------------------------------------------------------------

KEEP_CAMERAS="CAM_02,CAM_03,CAM_04"

CAMERA_PAIRS="CAM_02__CAM_03,CAM_03__CAM_04"


if [ ! -f "$SRC_MANIFEST" ]; then
    echo "ERROR: source manifest missing:"
    echo "$SRC_MANIFEST"
    exit 1
fi

mkdir -p "$DST"


# ============================================================
# 4. 构造 3cam + 512 manifest
#
# fake:
#   1920x1080 -> 512x288
#
# real:
#   1920x1080 -> 512x288
#
# K:
#   x/y pixel coordinate × 4/15
#
# T_cam_to_ego:
#   不动
#
# T_ego_to_world:
#   不动
#
# 最终只留下 CAM_02 / CAM_03 / CAM_04
# ============================================================

SRC="$SRC" \
DST="$DST" \
SRC_MANIFEST="$SRC_MANIFEST" \
DST_MANIFEST="$DST_MANIFEST" \
TARGET_W="$TARGET_W" \
TARGET_H="$TARGET_H" \
KEEP_CAMERAS="$KEEP_CAMERAS" \
python - <<'PY'
import copy
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


SRC = Path(os.environ["SRC"])
DST = Path(os.environ["DST"])

SRC_MANIFEST = Path(os.environ["SRC_MANIFEST"])
DST_MANIFEST = Path(os.environ["DST_MANIFEST"])

TW = int(os.environ["TARGET_W"])
TH = int(os.environ["TARGET_H"])

KEEP = os.environ["KEEP_CAMERAS"].split(",")


try:
    RGB_RESAMPLE = Image.Resampling.BILINEAR
    MASK_RESAMPLE = Image.Resampling.NEAREST
except AttributeError:
    RGB_RESAMPLE = Image.BILINEAR
    MASK_RESAMPLE = Image.NEAREST


def resolve(raw):
    p = Path(raw)

    if not p.is_absolute():
        p = SRC_MANIFEST.parent / p

    return p


# ============================================================
# Load
# ============================================================

items = []

with SRC_MANIFEST.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        if line.strip():
            items.append(
                json.loads(line)
            )


if len(items) != 300:
    raise RuntimeError(
        f"Expected 300 videos, got {len(items)}"
    )


fake_source_sizes = Counter()
real_source_sizes = Counter()

camera_orders_before = Counter()
camera_orders_after = Counter()

fake_count = 0
real_count = 0
mask_count = 0

max_norm_k_error = 0.0
max_cam_T_change = 0.0
max_ego_T_change = 0.0


for vi, item in enumerate(items):

    frames = item["frames"]

    if len(frames) != 19:
        raise RuntimeError(
            f"video={vi}: expected 19 frames, "
            f"got {len(frames)}"
        )


    # 原 camera order
    camera_orders_before[
        tuple(
            v["camera"]
            for v in frames[0]["views"]
        )
    ] += 1


    for fi, frame in enumerate(frames):

        views = frame["views"]

        view_map = {
            v["camera"]: v
            for v in views
        }


        missing = [
            c
            for c in KEEP
            if c not in view_map
        ]

        if missing:
            raise RuntimeError(
                f"video={vi}, frame={fi}: "
                f"missing cameras={missing}, "
                f"available={list(view_map)}"
            )


        ego_before = None

        if "T_ego_to_world" in frame:

            ego_before = np.asarray(
                frame["T_ego_to_world"],
                dtype=np.float64,
            ).copy()


        new_views = []


        # ====================================================
        # 强制顺序：
        #
        # CAM_02 -> CAM_03 -> CAM_04
        #
        # 即：
        #
        # L0 -> F0 -> R0
        # ====================================================

        for camera in KEEP:

            view = copy.deepcopy(
                view_map[camera]
            )


            # ------------------------------------------------
            # camera extrinsic before
            # ------------------------------------------------

            T_cam_before = np.asarray(
                view["T_cam_to_ego"],
                dtype=np.float64,
            ).copy()


            # ====================================================
            # FAKE
            # ====================================================

            fake_src = resolve(
                view["image_path"]
            )

            if not fake_src.is_file():
                raise FileNotFoundError(
                    fake_src
                )


            with Image.open(fake_src) as im:

                im = im.convert("RGB")

                sw, sh = im.size

                fake_source_sizes[
                    (sw, sh)
                ] += 1


                if (sw, sh) != (1920, 1080):

                    raise RuntimeError(
                        f"Unexpected fake size "
                        f"{sw}x{sh} "
                        f"at video={vi}, "
                        f"frame={fi}, "
                        f"camera={camera}"
                    )


                sx = TW / float(sw)
                sy = TH / float(sh)


                if abs(sx - sy) > 1e-12:

                    raise RuntimeError(
                        f"Non-uniform scaling: "
                        f"sx={sx}, sy={sy}"
                    )


                resized = im.resize(
                    (TW, TH),
                    resample=RGB_RESAMPLE,
                )


                fake_rel = Path(
                    "images",
                    f"video_{vi:06d}",
                    f"t{fi:03d}",
                    f"{camera}.jpg",
                )

                fake_dst = (
                    DST
                    / fake_rel
                )

                fake_dst.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                resized.save(
                    fake_dst,
                    format="JPEG",
                    quality=95,
                )


            view["image_path"] = str(
                fake_rel
            )

            fake_count += 1


            # ====================================================
            # REAL
            # ====================================================

            real_raw = view.get(
                "real_image_path"
            )


            if not real_raw:

                raise RuntimeError(
                    f"real_image_path missing: "
                    f"video={vi}, frame={fi}, camera={camera}"
                )


            real_src = resolve(
                real_raw
            )


            if not real_src.is_file():
                raise FileNotFoundError(
                    real_src
                )


            with Image.open(real_src) as im:

                im = im.convert("RGB")

                rw, rh = im.size

                real_source_sizes[
                    (rw, rh)
                ] += 1


                if (rw, rh) != (1920, 1080):

                    raise RuntimeError(
                        f"Unexpected real size "
                        f"{rw}x{rh} "
                        f"at video={vi}, "
                        f"frame={fi}, "
                        f"camera={camera}"
                    )


                resized = im.resize(
                    (TW, TH),
                    resample=RGB_RESAMPLE,
                )


                real_rel = Path(
                    "paired_real",
                    f"video_{vi:06d}",
                    f"t{fi:03d}",
                    f"{camera}.jpg",
                )

                real_dst = (
                    DST
                    / real_rel
                )

                real_dst.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                resized.save(
                    real_dst,
                    format="JPEG",
                    quality=95,
                )


            view["real_image_path"] = str(
                real_rel
            )

            real_count += 1


            # ====================================================
            # K
            #
            # 1920x1080 -> 512x288
            #
            # sx = sy = 4/15 ≈ 0.2666667
            # ====================================================

            K0 = np.asarray(
                view["K"],
                dtype=np.float64,
            )


            if K0.shape != (3, 3):
                raise RuntimeError(
                    f"Bad K shape: {K0.shape}"
                )


            K1 = K0.copy()

            K1[0, :] *= sx
            K1[1, :] *= sy

            K1[2, :] = K0[2, :]


            before = np.asarray([
                K0[0, 0] / sw,
                K0[1, 1] / sh,
                K0[0, 2] / sw,
                K0[1, 2] / sh,
            ])


            after = np.asarray([
                K1[0, 0] / TW,
                K1[1, 1] / TH,
                K1[0, 2] / TW,
                K1[1, 2] / TH,
            ])


            error = float(
                np.max(
                    np.abs(
                        before - after
                    )
                )
            )


            max_norm_k_error = max(
                max_norm_k_error,
                error,
            )


            view["K"] = K1.tolist()

            view["image_size"] = [
                TW,
                TH,
            ]


            # ====================================================
            # MASK
            # ====================================================

            raw_mask = view.get(
                "valid_mask_path"
            )


            if raw_mask:

                mask_src = resolve(
                    raw_mask
                )


                if not mask_src.is_file():

                    raise FileNotFoundError(
                        mask_src
                    )


                with Image.open(mask_src) as mask:

                    mask = mask.convert("L")

                    mask = mask.resize(
                        (TW, TH),
                        resample=MASK_RESAMPLE,
                    )


                    mask_rel = Path(
                        "masks",
                        f"video_{vi:06d}",
                        f"t{fi:03d}",
                        f"{camera}.png",
                    )


                    mask_dst = (
                        DST
                        / mask_rel
                    )

                    mask_dst.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    mask.save(
                        mask_dst,
                        format="PNG",
                    )


                view["valid_mask_path"] = str(
                    mask_rel
                )

                mask_count += 1


            # ====================================================
            # extrinsic 不变
            # ====================================================

            T_cam_after = np.asarray(
                view["T_cam_to_ego"],
                dtype=np.float64,
            )


            max_cam_T_change = max(
                max_cam_T_change,
                float(
                    np.max(
                        np.abs(
                            T_cam_before
                            - T_cam_after
                        )
                    )
                ),
            )


            new_views.append(
                view
            )


        # ====================================================
        # 最终只有三个 camera
        # ====================================================

        frame["views"] = new_views


        if ego_before is not None:

            ego_after = np.asarray(
                frame["T_ego_to_world"],
                dtype=np.float64,
            )


            max_ego_T_change = max(
                max_ego_T_change,
                float(
                    np.max(
                        np.abs(
                            ego_before
                            - ego_after
                        )
                    )
                ),
            )


    camera_orders_after[
        tuple(
            v["camera"]
            for v in frames[0]["views"]
        )
    ] += 1


    if (
        (vi + 1) % 20 == 0
        or vi + 1 == len(items)
    ):

        print(
            f"[prepare] {vi + 1}/{len(items)}",
            flush=True,
        )


# ============================================================
# Save
# ============================================================

with DST_MANIFEST.open(
    "w",
    encoding="utf-8",
) as f:

    for item in items:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


expected = 300 * 19 * 3


print()
print("=" * 100)
print("NUPLAN 3CAM 512 PREPARE RESULT")
print("=" * 100)

print("videos =", len(items))

print()
print("camera order BEFORE:")
for order, n in camera_orders_before.items():
    print(
        n,
        "videos ->",
        list(order),
    )

print()
print("camera order AFTER:")
for order, n in camera_orders_after.items():
    print(
        n,
        "videos ->",
        list(order),
    )

print()
print("semantic order:")
print(
    "CAM_02 = CAM_L0"
)
print(
    "CAM_03 = CAM_F0"
)
print(
    "CAM_04 = CAM_R0"
)

print()
print(
    "fake source sizes =",
    dict(fake_source_sizes),
)

print(
    "real source sizes =",
    dict(real_source_sizes),
)

print()
print(
    "fake count =",
    fake_count,
)

print(
    "real count =",
    real_count,
)

print(
    "mask count =",
    mask_count,
)

print()
print(
    "max normalized K error =",
    max_norm_k_error,
)

print(
    "max T_cam_to_ego change =",
    max_cam_T_change,
)

print(
    "max T_ego_to_world change =",
    max_ego_T_change,
)

print()
print(
    "manifest =",
    DST_MANIFEST,
)


if fake_count != expected:
    raise RuntimeError(
        f"Expected {expected} fake images, "
        f"got {fake_count}"
    )


if real_count != expected:
    raise RuntimeError(
        f"Expected {expected} real images, "
        f"got {real_count}"
    )


if set(camera_orders_after.keys()) != {
    (
        "CAM_02",
        "CAM_03",
        "CAM_04",
    )
}:
    raise RuntimeError(
        f"Wrong final camera order: "
        f"{camera_orders_after}"
    )


if max_norm_k_error > 1e-9:
    raise RuntimeError(
        "Normalized K changed"
    )


if max_cam_T_change != 0:
    raise RuntimeError(
        "T_cam_to_ego changed"
    )


if max_ego_T_change != 0:
    raise RuntimeError(
        "T_ego_to_world changed"
    )


print()
print("PREPARE PASS")
PY


# ============================================================
# 5. ST-FLOW
#
# 三摄顺序：
#
# CAM_02 = L0
# CAM_03 = F0
# CAM_04 = R0
#
# Cross:
#
# CAM_02 <-> CAM_03
# CAM_03 <-> CAM_04
#
# 不闭环 04 <-> 02
# ============================================================

STFLOW_OUTPUT=$DST/stflow_traj_result_gate16_3cam_512.json
STFLOW_LOG=$DST/stflow_gate16_3cam_512.log


echo
echo "============================================================"
echo "RUN NUPLAN 3CAM ST-FLOW @ 512x288"
echo "============================================================"

echo "Temporal/Traj cameras:"
echo "  CAM_02 CAM_03 CAM_04"

echo
echo "Cross pairs:"
echo "  CAM_02 <-> CAM_03"
echo "  CAM_03 <-> CAM_04"


python -m dwm.tools.evaluate_stflow \
    --manifest "$DST_MANIFEST" \
    --output "$STFLOW_OUTPUT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --frame-stride 2 \
    --min-matches 16 \
    --max-matches 256 \
    --loftr-confidence 0.1 \
    --camera-pairs "$CAMERA_PAIRS" \
    --cross-gate-px "$GATE" \
    2>&1 | tee "$STFLOW_LOG"


# ============================================================
# 6. FVD
#
# 同一份 3cam / 512 manifest
# ============================================================

FVD_OUTPUT=$DST/paired_fvd_result_3cam_all19_512.json
FVD_LOG=$DST/fvd_3cam_all19_512.log


echo
echo "============================================================"
echo "RUN NUPLAN 3CAM FVD @ 512x288"
echo "============================================================"

echo "FVD cameras:"
echo "  CAM_02 CAM_03 CAM_04"


if ! python -m dwm.tools.evaluate_fvd_from_paired_manifest \
    --manifest "$DST_MANIFEST" \
    --output "$FVD_OUTPUT" \
    --i3d-checkpoint "$I3D_CHECKPOINT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --sequence-count "$SEQ_COUNT" \
    --camera-names "$KEEP_CAMERAS" \
    --batch-size 2 \
    2>&1 | tee "$FVD_LOG"
then

    echo
    echo "FVD batch-size=2 failed."
    echo "Retry with batch-size=1."

    python -m dwm.tools.evaluate_fvd_from_paired_manifest \
        --manifest "$DST_MANIFEST" \
        --output "$FVD_OUTPUT" \
        --i3d-checkpoint "$I3D_CHECKPOINT" \
        --device cuda \
        --max-videos "$MAX_VIDEOS" \
        --sequence-count "$SEQ_COUNT" \
        --camera-names "$KEEP_CAMERAS" \
        --batch-size 1 \
        2>&1 | tee -a "$FVD_LOG"
fi


# ============================================================
# 7. Results
# ============================================================

echo
echo "============================================================"
echo "DONE"
echo "============================================================"

echo
echo "ST-Flow:"
echo "$STFLOW_OUTPUT"

echo
echo "FVD:"
echo "$FVD_OUTPUT"

echo
echo "FVD result:"
cat "$FVD_OUTPUT"

