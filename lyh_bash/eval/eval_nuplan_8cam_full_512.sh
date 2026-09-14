#!/bin/bash

set -euo pipefail


# ============================================================
# 0. Environment
# ============================================================

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export CAMSIM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim
export OPENDWM_ROOT=$CAMSIM_ROOT/OpenDWM

cd "$OPENDWM_ROOT/src" || exit 1

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [ -d "$OPENDWM_ROOT/externals/TATS/tats/fvd" ]; then
    export PYTHONPATH="$OPENDWM_ROOT/externals/TATS/tats/fvd:$PYTHONPATH"
fi

if [ -d "$CAMSIM_ROOT/nuplan-devkit-master" ]; then
    export PYTHONPATH="$CAMSIM_ROOT/nuplan-devkit-master:$PYTHONPATH"
fi

if [ -d "$OPENDWM_ROOT/externals/waymo-open-dataset/src" ]; then
    export PYTHONPATH="$OPENDWM_ROOT/externals/waymo-open-dataset/src:$PYTHONPATH"
fi


# ============================================================
# 1. Checkpoints
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
echo "================================================================================"
echo "CHECKPOINTS"
echo "================================================================================"

ls -lh \
    "$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth" \
    "$CACHE_DIR/loftr_outdoor.ckpt" \
    "$I3D_CHECKPOINT"


# ============================================================
# 2. Paths
# ============================================================

BASE=$CAMSIM_ROOT/lyh_output/eval/nuscenesablationnew

# 已经处理好的：
#
# generated image = 512x288
# K               = 512x288 coordinate system
#
FAKE_ROOT=$BASE/nuplan6hz_merged300_stflow512
FAKE_MANIFEST=$FAKE_ROOT/stflow_manifest.jsonl


# 原始目录：
#
# paired_real = 1920x1080
#
REAL_ROOT=$BASE/nuplan6hz_merged300
REAL_MANIFEST=$REAL_ROOT/stflow_manifest.jsonl


# 新的统一评测目录
OUT_ROOT=$BASE/nuplan6hz_merged300_8cam_full512
OUT_MANIFEST=$OUT_ROOT/stflow_manifest.jsonl


# Results
STFLOW_OUTPUT=$OUT_ROOT/stflow_traj_result_gate16_8cam_512.json
STFLOW_LOG=$OUT_ROOT/stflow_gate16_8cam_512.log

FVD_OUTPUT=$OUT_ROOT/paired_fvd_result_8cam_all19_512.json
FVD_LOG=$OUT_ROOT/fvd_8cam_all19_512.log


TARGET_W=512
TARGET_H=288

MAX_VIDEOS=300
SEQ_COUNT=19
GATE=16

CAMERAS="CAM_00,CAM_01,CAM_02,CAM_03,CAM_04,CAM_05,CAM_06,CAM_07"


for P in \
    "$FAKE_MANIFEST" \
    "$REAL_MANIFEST"
do
    if [ ! -f "$P" ]; then
        echo "ERROR: manifest missing:"
        echo "$P"
        exit 1
    fi
done


mkdir -p "$OUT_ROOT"


# ============================================================
# 3. Build unified 8-camera 512 manifest
#
# generated:
#
# nuplan6hz_merged300_stflow512
#     already 512x288
#     already has correct 512 K
#
# paired real:
#
# nuplan6hz_merged300/paired_real
#     1920x1080
#          ↓
#     resize 512x288
#
#
# IMPORTANT:
#
# Do NOT modify K again.
# It is already correct for 512x288 in FAKE_MANIFEST.
#
# T_cam_to_ego:
#     unchanged
#
# T_ego_to_world:
#     unchanged
#
# camera order:
#     CAM_00 ... CAM_07
# ============================================================

FAKE_ROOT="$FAKE_ROOT" \
FAKE_MANIFEST="$FAKE_MANIFEST" \
REAL_ROOT="$REAL_ROOT" \
REAL_MANIFEST="$REAL_MANIFEST" \
OUT_ROOT="$OUT_ROOT" \
OUT_MANIFEST="$OUT_MANIFEST" \
TARGET_W="$TARGET_W" \
TARGET_H="$TARGET_H" \
CAMERAS="$CAMERAS" \
python - <<'PY'
import copy
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


FAKE_ROOT = Path(os.environ["FAKE_ROOT"])
FAKE_MANIFEST = Path(os.environ["FAKE_MANIFEST"])

REAL_ROOT = Path(os.environ["REAL_ROOT"])
REAL_MANIFEST = Path(os.environ["REAL_MANIFEST"])

OUT_ROOT = Path(os.environ["OUT_ROOT"])
OUT_MANIFEST = Path(os.environ["OUT_MANIFEST"])

TW = int(os.environ["TARGET_W"])
TH = int(os.environ["TARGET_H"])

CAMERAS = os.environ["CAMERAS"].split(",")


try:
    RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE = Image.BILINEAR


def load_jsonl(path):
    items = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def resolve(raw, manifest):
    p = Path(raw)

    if not p.is_absolute():
        p = manifest.parent / p

    return p


fake_items = load_jsonl(FAKE_MANIFEST)
real_items = load_jsonl(REAL_MANIFEST)


print()
print("=" * 100)
print("INPUT CHECK")
print("=" * 100)

print("fake manifest:")
print(FAKE_MANIFEST)

print()
print("real manifest:")
print(REAL_MANIFEST)

print()
print("fake videos =", len(fake_items))
print("real videos =", len(real_items))


if len(fake_items) != 300:
    raise RuntimeError(
        f"Expected 300 fake videos, got {len(fake_items)}"
    )

if len(real_items) != 300:
    raise RuntimeError(
        f"Expected 300 real videos, got {len(real_items)}"
    )


fake_sizes = Counter()
real_sizes = Counter()
manifest_sizes = Counter()

camera_orders = Counter()

fake_count = 0
real_count = 0

max_ego_diff = 0.0
max_cam_T_diff = 0.0

output_items = []


# ============================================================
# Videos
# ============================================================

for vi, (fake_item, real_item) in enumerate(
    zip(fake_items, real_items)
):

    fake_frames = fake_item["frames"]
    real_frames = real_item["frames"]


    if len(fake_frames) != 19:
        raise RuntimeError(
            f"fake video={vi}: "
            f"expected 19 frames, got {len(fake_frames)}"
        )

    if len(real_frames) != 19:
        raise RuntimeError(
            f"real video={vi}: "
            f"expected 19 frames, got {len(real_frames)}"
        )


    new_item = copy.deepcopy(fake_item)


    for fi in range(19):

        fake_frame = fake_frames[fi]
        real_frame = real_frames[fi]


        # ====================================================
        # A. Check same clip/frame
        # ====================================================

        fake_ego = fake_frame.get(
            "T_ego_to_world"
        )

        real_ego = real_frame.get(
            "T_ego_to_world"
        )


        if fake_ego is None or real_ego is None:
            raise RuntimeError(
                f"T_ego_to_world missing: "
                f"video={vi}, frame={fi}"
            )


        fake_ego = np.asarray(
            fake_ego,
            dtype=np.float64,
        )

        real_ego = np.asarray(
            real_ego,
            dtype=np.float64,
        )


        ego_diff = float(
            np.max(
                np.abs(
                    fake_ego - real_ego
                )
            )
        )


        max_ego_diff = max(
            max_ego_diff,
            ego_diff,
        )


        if ego_diff > 1e-5:
            raise RuntimeError(
                "\nVIDEO/FRAME MISMATCH\n"
                f"video={vi}\n"
                f"frame={fi}\n"
                f"T_ego_to_world diff={ego_diff}\n"
            )


        # ====================================================
        # B. Camera maps
        # ====================================================

        fake_map = {
            v["camera"]: v
            for v in fake_frame["views"]
        }

        real_map = {
            v["camera"]: v
            for v in real_frame["views"]
        }


        missing_fake = [
            c
            for c in CAMERAS
            if c not in fake_map
        ]

        missing_real = [
            c
            for c in CAMERAS
            if c not in real_map
        ]


        if missing_fake:
            raise RuntimeError(
                f"fake missing cameras: "
                f"video={vi}, frame={fi}, "
                f"{missing_fake}; "
                f"available={list(fake_map)}"
            )

        if missing_real:
            raise RuntimeError(
                f"real missing cameras: "
                f"video={vi}, frame={fi}, "
                f"{missing_real}; "
                f"available={list(real_map)}"
            )


        new_views = []


        # ====================================================
        # Enforce exact order:
        #
        # CAM_00
        # CAM_01
        # CAM_02
        # CAM_03
        # CAM_04
        # CAM_05
        # CAM_06
        # CAM_07
        # ====================================================

        for camera in CAMERAS:

            fake_view = copy.deepcopy(
                fake_map[camera]
            )

            real_view = real_map[camera]


            # =================================================
            # C. Generated image
            #
            # Already 512x288.
            # Do NOT resize again.
            # =================================================

            fake_path = resolve(
                fake_view["image_path"],
                FAKE_MANIFEST,
            )


            if not fake_path.is_file():
                raise FileNotFoundError(
                    fake_path
                )


            with Image.open(fake_path) as im:

                fake_sizes[
                    im.size
                ] += 1


                if im.size != (TW, TH):
                    raise RuntimeError(
                        f"\nBAD FAKE SIZE\n"
                        f"video={vi}\n"
                        f"frame={fi}\n"
                        f"camera={camera}\n"
                        f"actual={im.size}\n"
                        f"expected={(TW, TH)}\n"
                    )


            manifest_size = tuple(
                fake_view.get(
                    "image_size",
                    [],
                )
            )

            manifest_sizes[
                manifest_size
            ] += 1


            if manifest_size != (
                TW,
                TH,
            ):

                raise RuntimeError(
                    f"Bad manifest image_size: "
                    f"{manifest_size} "
                    f"at video={vi}, frame={fi}, camera={camera}"
                )


            # -------------------------------------------------
            # K sanity
            #
            # stflow512 的 K 必须已经是 512 coordinate
            # -------------------------------------------------

            K = np.asarray(
                fake_view["K"],
                dtype=np.float64,
            )


            if K.shape != (3, 3):
                raise RuntimeError(
                    f"Bad K shape: {K.shape}"
                )


            # 这里只做合理性检查。
            # fx / width 一般应该是 O(1)，
            # 不应该再出现 1920 coordinate 的数值。
            fx_norm = K[0, 0] / TW
            fy_norm = K[1, 1] / TH


            if not (
                0.1 < fx_norm < 5.0
                and 0.1 < fy_norm < 5.0
            ):
                raise RuntimeError(
                    "\nSuspicious K\n"
                    f"video={vi}, frame={fi}, camera={camera}\n"
                    f"K=\n{K}\n"
                    f"fx/W={fx_norm}\n"
                    f"fy/H={fy_norm}\n"
                )


            # =================================================
            # D. Check extrinsic unchanged relative to source
            # =================================================

            T_fake = np.asarray(
                fake_view["T_cam_to_ego"],
                dtype=np.float64,
            )

            T_real = np.asarray(
                real_view["T_cam_to_ego"],
                dtype=np.float64,
            )


            T_diff = float(
                np.max(
                    np.abs(
                        T_fake - T_real
                    )
                )
            )


            max_cam_T_diff = max(
                max_cam_T_diff,
                T_diff,
            )


            if T_diff > 1e-5:
                raise RuntimeError(
                    "\nCAMERA EXTRINSIC MISMATCH\n"
                    f"video={vi}\n"
                    f"frame={fi}\n"
                    f"camera={camera}\n"
                    f"diff={T_diff}\n"
                )


            # =================================================
            # E. Paired real
            #
            # Original = 1920x1080
            # Resize   = 512x288
            # =================================================

            real_raw = real_view.get(
                "real_image_path"
            )


            if not real_raw:
                raise RuntimeError(
                    f"real_image_path missing: "
                    f"video={vi}, frame={fi}, camera={camera}"
                )


            real_src = resolve(
                real_raw,
                REAL_MANIFEST,
            )


            if not real_src.is_file():
                raise FileNotFoundError(
                    real_src
                )


            with Image.open(real_src) as im:

                im = im.convert("RGB")

                real_sizes[
                    im.size
                ] += 1


                if im.size != (
                    1920,
                    1080,
                ):

                    raise RuntimeError(
                        f"\nUnexpected original real size\n"
                        f"video={vi}\n"
                        f"frame={fi}\n"
                        f"camera={camera}\n"
                        f"size={im.size}\n"
                    )


                im = im.resize(
                    (TW, TH),
                    resample=RESAMPLE,
                )


                real_rel = Path(
                    "paired_real",
                    f"video_{vi:06d}",
                    f"t{fi:03d}",
                    f"{camera}.jpg",
                )


                real_dst = (
                    OUT_ROOT
                    / real_rel
                )


                real_dst.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )


                im.save(
                    real_dst,
                    format="JPEG",
                    quality=95,
                )


            # =================================================
            # F. Build final view
            # =================================================

            # fake 不复制，直接引用 stflow512 中已经存在的图。
            #
            # 用绝对路径可以避免新 manifest 的相对路径解析问题。
            fake_view["image_path"] = str(
                fake_path
            )


            fake_view["real_image_path"] = str(
                real_rel
            )


            # 再明确写一次
            fake_view["image_size"] = [
                TW,
                TH,
            ]


            new_views.append(
                fake_view
            )


            fake_count += 1
            real_count += 1


        new_item["frames"][fi][
            "views"
        ] = new_views


    camera_orders[
        tuple(
            v["camera"]
            for v in new_item["frames"][0]["views"]
        )
    ] += 1


    output_items.append(
        new_item
    )


    if (
        (vi + 1) % 20 == 0
        or vi + 1 == len(fake_items)
    ):

        print(
            f"[prepare] {vi + 1}/{len(fake_items)}",
            flush=True,
        )


# ============================================================
# Save
# ============================================================

with OUT_MANIFEST.open(
    "w",
    encoding="utf-8",
) as f:

    for item in output_items:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


EXPECTED = 300 * 19 * 8


print()
print("=" * 100)
print("FINAL 8CAM 512 MANIFEST")
print("=" * 100)

print(
    "fake sizes =",
    dict(fake_sizes),
)

print(
    "original real sizes =",
    dict(real_sizes),
)

print(
    "manifest sizes =",
    dict(manifest_sizes),
)

print()
print(
    "fake views =",
    fake_count,
)

print(
    "real views =",
    real_count,
)

print(
    "expected =",
    EXPECTED,
)

print()
print(
    "max T_ego_to_world diff =",
    max_ego_diff,
)

print(
    "max T_cam_to_ego diff =",
    max_cam_T_diff,
)

print()
print("camera orders:")

for order, count in camera_orders.items():
    print(
        count,
        "videos ->",
        list(order),
    )

print()
print(
    "output manifest =",
    OUT_MANIFEST,
)


if fake_count != EXPECTED:
    raise RuntimeError(
        f"fake count mismatch: "
        f"{fake_count} != {EXPECTED}"
    )


if real_count != EXPECTED:
    raise RuntimeError(
        f"real count mismatch: "
        f"{real_count} != {EXPECTED}"
    )


EXPECTED_ORDER = tuple(
    f"CAM_{i:02d}"
    for i in range(8)
)


if set(camera_orders.keys()) != {
    EXPECTED_ORDER
}:
    raise RuntimeError(
        f"Wrong camera order: "
        f"{camera_orders}"
    )


if max_ego_diff > 1e-5:
    raise RuntimeError(
        "T_ego_to_world mismatch"
    )


if max_cam_T_diff > 1e-5:
    raise RuntimeError(
        "T_cam_to_ego mismatch"
    )


print()
print("8CAM 512 MANIFEST: PASS")
PY


# ============================================================
# 4. Final sanity check
# ============================================================

OUT_MANIFEST="$OUT_MANIFEST" \
python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image


manifest = Path(
    os.environ["OUT_MANIFEST"]
)


items = []

with manifest.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        if line.strip():

            items.append(
                json.loads(line)
            )


fake_sizes = Counter()
real_sizes = Counter()

fake_missing = 0
real_missing = 0

total = 0


for item in items:

    for frame in item["frames"]:

        cameras = [
            v["camera"]
            for v in frame["views"]
        ]


        if cameras != [
            "CAM_00",
            "CAM_01",
            "CAM_02",
            "CAM_03",
            "CAM_04",
            "CAM_05",
            "CAM_06",
            "CAM_07",
        ]:
            raise RuntimeError(
                f"Bad camera order: {cameras}"
            )


        for view in frame["views"]:

            total += 1


            fp = Path(
                view["image_path"]
            )

            if not fp.is_absolute():
                fp = manifest.parent / fp


            rp = Path(
                view["real_image_path"]
            )

            if not rp.is_absolute():
                rp = manifest.parent / rp


            if not fp.is_file():
                fake_missing += 1
            else:
                with Image.open(fp) as im:
                    fake_sizes[
                        im.size
                    ] += 1


            if not rp.is_file():
                real_missing += 1
            else:
                with Image.open(rp) as im:
                    real_sizes[
                        im.size
                    ] += 1


print()
print("=" * 100)
print("SANITY CHECK")
print("=" * 100)

print("videos       =", len(items))
print("views        =", total)

print()
print(
    "fake sizes  =",
    dict(fake_sizes),
)

print(
    "real sizes  =",
    dict(real_sizes),
)

print()
print(
    "fake missing =",
    fake_missing,
)

print(
    "real missing =",
    real_missing,
)


if len(items) != 300:
    raise RuntimeError(
        "Wrong video count"
    )


if total != 300 * 19 * 8:
    raise RuntimeError(
        "Wrong view count"
    )


if fake_missing != 0:
    raise RuntimeError(
        "Missing fake images"
    )


if real_missing != 0:
    raise RuntimeError(
        "Missing real images"
    )


if set(fake_sizes.keys()) != {
    (512, 288)
}:
    raise RuntimeError(
        f"Bad fake sizes: {fake_sizes}"
    )


if set(real_sizes.keys()) != {
    (512, 288)
}:
    raise RuntimeError(
        f"Bad real sizes: {real_sizes}"
    )


print()
print("SANITY CHECK: PASS")
PY


# ============================================================
# 5. ST-Flow + Traj
#
# Full nuPlan 8-camera ring:
#
# 00 -> 01
# 01 -> 02
# 02 -> 03
# 03 -> 04
# 04 -> 05
# 05 -> 06
# 06 -> 07
# 07 -> 00
#
# Temporal / Traj:
# all 8 cameras
# ============================================================

echo
echo "================================================================================"
echo "RUN NUPLAN FULL 8CAM ST-FLOW @512x288"
echo "================================================================================"

echo
echo "Camera order:"
echo "CAM_00 CAM_01 CAM_02 CAM_03 CAM_04 CAM_05 CAM_06 CAM_07"

echo
echo "Cross-view policy:"
echo "nuPlan full ring"


python -m dwm.tools.evaluate_stflow \
    --manifest "$OUT_MANIFEST" \
    --output "$STFLOW_OUTPUT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --frame-stride 2 \
    --min-matches 16 \
    --max-matches 256 \
    --loftr-confidence 0.1 \
    --pair-policy dataset \
    --cross-gate-px "$GATE" \
    2>&1 | tee "$STFLOW_LOG"


# ============================================================
# 6. FVD
#
# Full 8-camera:
#
# 300 videos × 8 cameras
# = 2400 video samples
#
# each sample = 19 frames
# ============================================================

echo
echo "================================================================================"
echo "RUN NUPLAN FULL 8CAM FVD @512x288"
echo "================================================================================"

echo
echo "Expected FVD samples:"
echo "300 x 8 = 2400"


if ! python -m dwm.tools.evaluate_fvd_from_paired_manifest \
    --manifest "$OUT_MANIFEST" \
    --output "$FVD_OUTPUT" \
    --i3d-checkpoint "$I3D_CHECKPOINT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --sequence-count "$SEQ_COUNT" \
    --camera-names "$CAMERAS" \
    --batch-size 2 \
    2>&1 | tee "$FVD_LOG"
then

    echo
    echo "FVD batch-size=2 failed."
    echo "Retry with batch-size=1."

    python -m dwm.tools.evaluate_fvd_from_paired_manifest \
        --manifest "$OUT_MANIFEST" \
        --output "$FVD_OUTPUT" \
        --i3d-checkpoint "$I3D_CHECKPOINT" \
        --device cuda \
        --max-videos "$MAX_VIDEOS" \
        --sequence-count "$SEQ_COUNT" \
        --camera-names "$CAMERAS" \
        --batch-size 1 \
        2>&1 | tee -a "$FVD_LOG"
fi


# ============================================================
# 7. Summary
# ============================================================

STFLOW_OUTPUT="$STFLOW_OUTPUT" \
FVD_OUTPUT="$FVD_OUTPUT" \
python - <<'PY'
import json
import os


st_path = os.environ["STFLOW_OUTPUT"]
fvd_path = os.environ["FVD_OUTPUT"]


with open(
    st_path,
    "r",
    encoding="utf-8",
) as f:

    st = json.load(f)


with open(
    fvd_path,
    "r",
    encoding="utf-8",
) as f:

    fv = json.load(f)


print()
print("=" * 110)
print("NUPLAN FULL 8CAM @ 512x288")
print("=" * 110)


mean = st.get(
    "mean",
    {}
)


keys = [
    "temporal_l1",
    "cross_raw_epi_px",
    "cross_epi_px",
    "cross_inlier_ratio",
    "cycle_epi_px",
    "traj_epi_px",
    "traj_inlier2",
    "traj_inlier4",
    "edge_coverage",
    "cycle_coverage",
    "stflow_score",
    "stflow_d_score",
    "stflow_c_score",
    "num_temporal_edges",
    "num_cross_raw_edges",
    "num_cross_edges",
    "num_cycle_edges",
    "num_traj_edges",
]


print()
print("ST-FLOW / TRAJ")
print("-" * 110)


for key in keys:

    if key in mean:

        print(
            f"{key:28s} = {mean[key]}"
        )


print()
print("FVD")
print("-" * 110)

print(
    "fvd           =",
    fv.get("fvd"),
)

print(
    "num_videos    =",
    fv.get("num_videos"),
)

print(
    "num_samples   =",
    fv.get("num_samples"),
)

print(
    "camera_names  =",
    fv.get("camera_names"),
)

print(
    "sequence_count=",
    fv.get("sequence_count"),
)


print()
print("ST-Flow result:")
print(st_path)

print()
print("FVD result:")
print(fvd_path)
PY


echo
echo "================================================================================"
echo "ALL DONE"
echo "================================================================================"

echo
echo "Unified manifest:"
echo "$OUT_MANIFEST"

echo
echo "ST-Flow:"
echo "$STFLOW_OUTPUT"

echo
echo "FVD:"
echo "$FVD_OUTPUT"

