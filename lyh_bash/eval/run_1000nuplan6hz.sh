#!/bin/bash

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


# ============================================================
# 0. Checkpoints
# ============================================================

CKPT_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt

RAFT_LOCAL=$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth
LOFTR_LOCAL=$CKPT_ROOT/loftr_outdoor.ckpt
I3D_CHECKPOINT=$CKPT_ROOT/i3d_pretrained_400.pt

for P in \
    "$RAFT_LOCAL" \
    "$LOFTR_LOCAL" \
    "$I3D_CHECKPOINT"
do
    if [ ! -f "$P" ]; then
        echo "ERROR: missing checkpoint:"
        echo "$P"
        exit 1
    fi
done


export TORCH_HOME=/root/.cache/torch
mkdir -p "$TORCH_HOME/hub/checkpoints"

cp -f \
    "$RAFT_LOCAL" \
    "$TORCH_HOME/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

cp -f \
    "$LOFTR_LOCAL" \
    "$TORCH_HOME/hub/checkpoints/loftr_outdoor.ckpt"


# ============================================================
# 1. Paths
# ============================================================

BASE=$CAMSIM_ROOT/lyh_output/eval/nuscenesablationnew

SRC_ROOT=$BASE/nuplan6hz

# 清理 rank01 + resize 后的中间目录
PREP_ROOT=$BASE/1000nuplan6hz_rankclean512

# 最终 merged 1000
OUT_ROOT=$BASE/1000nuplan6hz

FULL_MANIFEST=$OUT_ROOT/stflow_manifest.jsonl
FRONT3_MANIFEST=$OUT_ROOT/stflow_manifest_front3.jsonl


TARGET_W=512
TARGET_H=288

MAX_VIDEOS=1000
SEQ_COUNT=19
GATE=16


echo
echo "================================================================================"
echo "1000 NUPLAN 6HZ"
echo "================================================================================"

echo "source:"
echo "$SRC_ROOT"

echo
echo "prepared ranks:"
echo "$PREP_ROOT"

echo
echo "final output:"
echo "$OUT_ROOT"


for R in 00 01 02 03
do
    P="$SRC_ROOT/rank_$R/stflow_manifest.jsonl"

    if [ ! -f "$P" ]; then
        echo "ERROR: missing:"
        echo "$P"
        exit 1
    fi
done


# ============================================================
# 2. Clean rank00/rank01 + resize everything to 512x288
#
# rank00:
#   keep last len(rank02)
#
# rank01:
#   keep last len(rank03)
#
# rank02:
#   keep all
#
# rank03:
#   keep all
#
#
# Source nuPlan:
#     generated: 1600x900
#
# Target:
#     512x288
#
# 1600x900 and 512x288 are both 16:9,
# therefore normally:
#
#     crop offset = 0
#     sx = 512/1600 = 0.32
#     sy = 288/900  = 0.32
#
# K' = A @ K
#
# The code below also supports other source aspect ratios by
# center cropping first and then resizing.
# ============================================================

rm -rf "$PREP_ROOT"
mkdir -p "$PREP_ROOT"


SRC_ROOT="$SRC_ROOT" \
PREP_ROOT="$PREP_ROOT" \
TARGET_W="$TARGET_W" \
TARGET_H="$TARGET_H" \
python - <<'PY'
import copy
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


SRC_ROOT = Path(os.environ["SRC_ROOT"])
PREP_ROOT = Path(os.environ["PREP_ROOT"])

TW = int(os.environ["TARGET_W"])
TH = int(os.environ["TARGET_H"])

WORKERS = 8


try:
    RGB_RESAMPLE = Image.Resampling.BILINEAR
    MASK_RESAMPLE = Image.Resampling.NEAREST
except AttributeError:
    RGB_RESAMPLE = Image.BILINEAR
    MASK_RESAMPLE = Image.NEAREST


def load_jsonl(path):
    items = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def resolve(path_raw, manifest_path):
    if path_raw is None:
        return None

    p = Path(path_raw)

    if not p.is_absolute():
        p = manifest_path.parent / p

    return p.resolve()


# ============================================================
# Image geometry
# ============================================================

def crop_geometry(w, h):

    # Compare W/H against TW/TH without float error.
    lhs = w * TH
    rhs = h * TW

    if lhs == rhs:
        left = 0
        top = 0
        crop_w = w
        crop_h = h

    elif lhs > rhs:
        # Too wide.
        crop_h = h
        crop_w = int(round(h * TW / TH))

        left = (w - crop_w) // 2
        top = 0

    else:
        # Too tall.
        crop_w = w
        crop_h = int(round(w * TH / TW))

        left = 0
        top = (h - crop_h) // 2

    right = left + crop_w
    bottom = top + crop_h

    sx = TW / crop_w
    sy = TH / crop_h

    return (
        left,
        top,
        right,
        bottom,
        crop_w,
        crop_h,
        sx,
        sy,
    )


def transform_K(K_raw, w, h):

    K = np.asarray(
        K_raw,
        dtype=np.float64,
    )

    if K.shape != (3, 3):
        raise RuntimeError(
            f"Bad K shape: {K.shape}"
        )

    (
        left,
        top,
        right,
        bottom,
        crop_w,
        crop_h,
        sx,
        sy,
    ) = crop_geometry(w, h)

    # Pixel transform:
    #
    # x' = sx * (x - left)
    # y' = sy * (y - top)
    #
    A = np.array(
        [
            [sx, 0.0, -sx * left],
            [0.0, sy, -sy * top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    K_new = A @ K

    return K_new.tolist()


def process_rgb(src, dst):

    if not src.is_file():
        raise FileNotFoundError(src)

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with Image.open(src) as im:

        im = im.convert("RGB")

        w, h = im.size

        (
            left,
            top,
            right,
            bottom,
            crop_w,
            crop_h,
            sx,
            sy,
        ) = crop_geometry(w, h)

        if (
            left != 0
            or top != 0
            or crop_w != w
            or crop_h != h
        ):
            im = im.crop(
                (
                    left,
                    top,
                    right,
                    bottom,
                )
            )

        if im.size != (TW, TH):
            im = im.resize(
                (TW, TH),
                resample=RGB_RESAMPLE,
            )

        im.save(
            dst,
            format="JPEG",
            quality=95,
        )

    return (w, h)


def process_mask(src, dst):

    if not src.is_file():
        raise FileNotFoundError(src)

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with Image.open(src) as im:

        w, h = im.size

        (
            left,
            top,
            right,
            bottom,
            crop_w,
            crop_h,
            sx,
            sy,
        ) = crop_geometry(w, h)

        if (
            left != 0
            or top != 0
            or crop_w != w
            or crop_h != h
        ):
            im = im.crop(
                (
                    left,
                    top,
                    right,
                    bottom,
                )
            )

        if im.size != (TW, TH):
            im = im.resize(
                (TW, TH),
                resample=MASK_RESAMPLE,
            )

        im.save(dst)


# ============================================================
# Load original ranks
# ============================================================

rank_items = {}

for rank in range(4):

    manifest = (
        SRC_ROOT
        / f"rank_{rank:02d}"
        / "stflow_manifest.jsonl"
    )

    rank_items[rank] = load_jsonl(
        manifest
    )


print()
print("=" * 100)
print("ORIGINAL RANK COUNTS")
print("=" * 100)

for rank in range(4):
    print(
        f"rank_{rank:02d}: "
        f"{len(rank_items[rank])}"
    )


n2 = len(rank_items[2])
n3 = len(rank_items[3])


if n2 == 0 or n3 == 0:
    raise RuntimeError(
        "rank02/rank03 must not be empty"
    )


# ============================================================
# Keep only latest preview in rank00 / rank01
# ============================================================

selected = {
    0: rank_items[0][-n2:],
    1: rank_items[1][-n3:],
    2: rank_items[2],
    3: rank_items[3],
}


print()
print("=" * 100)
print("CLEANED RANK COUNTS")
print("=" * 100)

print(
    f"rank_00: keep LAST {n2} "
    f"of {len(rank_items[0])}"
)

print(
    f"rank_01: keep LAST {n3} "
    f"of {len(rank_items[1])}"
)

print(
    f"rank_02: keep ALL {n2}"
)

print(
    f"rank_03: keep ALL {n3}"
)


total_clean = sum(
    len(x)
    for x in selected.values()
)


print()
print(
    "clean total =",
    total_clean,
)


if total_clean < 1000:
    raise RuntimeError(
        f"After cleaning only {total_clean} videos remain; "
        f"need at least 1000."
    )


# ============================================================
# Diagnostics: IDs before / after tail selection
# ============================================================

print()
print("=" * 100)
print("TAIL SELECTION DIAGNOSTICS")
print("=" * 100)

for rank in range(4):

    items = selected[rank]

    print()
    print(
        f"rank_{rank:02d}:"
    )

    print(
        "selected count =",
        len(items),
    )

    print(
        "first old video_id =",
        items[0].get("video_id"),
    )

    print(
        "last old video_id  =",
        items[-1].get("video_id"),
    )


# ============================================================
# Process one video
# ============================================================

def process_one(args):

    rank, local_index, item = args

    source_rank_root = (
        SRC_ROOT
        / f"rank_{rank:02d}"
    )

    source_manifest = (
        source_rank_root
        / "stflow_manifest.jsonl"
    )

    output_rank_root = (
        PREP_ROOT
        / f"rank_{rank:02d}"
    )


    new_item = copy.deepcopy(
        item
    )


    # Give cleaned rank a fresh local ID.
    #
    # This avoids collisions from the two previews that were
    # written into rank00/rank01.
    new_video_id = (
        f"nuplan_video_{local_index:06d}"
    )

    new_item["video_id"] = (
        new_video_id
    )


    frames = new_item.get(
        "frames",
        []
    )


    for frame_pos, frame in enumerate(
        frames
    ):

        views = frame.get(
            "views",
            []
        )


        for view_pos, view in enumerate(
            views
        ):

            camera = view.get(
                "camera",
                f"CAM_{view_pos:02d}",
            )


            # ================================================
            # Generated image
            # ================================================

            old_fake_raw = view.get(
                "image_path"
            )


            if not old_fake_raw:
                raise RuntimeError(
                    f"Missing image_path: "
                    f"rank={rank} "
                    f"video={local_index} "
                    f"frame={frame_pos} "
                    f"camera={camera}"
                )


            old_fake = resolve(
                old_fake_raw,
                source_manifest,
            )


            if not old_fake.is_file():
                raise FileNotFoundError(
                    old_fake
                )


            # Read original generated size before changing K.
            with Image.open(old_fake) as im:
                fake_w, fake_h = im.size


            # image_size must agree with the generated image if present.
            old_manifest_size = view.get(
                "image_size"
            )

            if old_manifest_size:

                manifest_w = int(
                    old_manifest_size[0]
                )

                manifest_h = int(
                    old_manifest_size[1]
                )

                if (
                    manifest_w != fake_w
                    or manifest_h != fake_h
                ):

                    raise RuntimeError(
                        "\nIMAGE_SIZE / FILE SIZE MISMATCH\n"
                        f"rank={rank}\n"
                        f"video={local_index}\n"
                        f"frame={frame_pos}\n"
                        f"camera={camera}\n"
                        f"manifest={old_manifest_size}\n"
                        f"file={(fake_w, fake_h)}\n"
                    )


            # Update camera intrinsic according to:
            #
            # center crop -> resize to 512x288
            #
            if "K" not in view:
                raise RuntimeError(
                    f"K missing at "
                    f"rank={rank}, "
                    f"video={local_index}, "
                    f"frame={frame_pos}, "
                    f"camera={camera}"
                )


            view["K"] = transform_K(
                view["K"],
                fake_w,
                fake_h,
            )


            view["image_size"] = [
                TW,
                TH,
            ]


            fake_rel = Path(
                "images",
                new_video_id,
                f"t{frame_pos:03d}",
                f"{camera}.jpg",
            )


            fake_dst = (
                output_rank_root
                / fake_rel
            )


            process_rgb(
                old_fake,
                fake_dst,
            )


            view["image_path"] = str(
                fake_rel
            )


            # ================================================
            # Paired real
            # ================================================

            old_real_raw = view.get(
                "real_image_path"
            )


            if not old_real_raw:

                raise RuntimeError(
                    "\nPAIRED REAL MISSING\n"
                    f"rank={rank}\n"
                    f"video={local_index}\n"
                    f"frame={frame_pos}\n"
                    f"camera={camera}\n"
                )


            old_real = resolve(
                old_real_raw,
                source_manifest,
            )


            if not old_real.is_file():
                raise FileNotFoundError(
                    old_real
                )


            real_rel = Path(
                "paired_real",
                new_video_id,
                f"t{frame_pos:03d}",
                f"{camera}.jpg",
            )


            real_dst = (
                output_rank_root
                / real_rel
            )


            process_rgb(
                old_real,
                real_dst,
            )


            view["real_image_path"] = str(
                real_rel
            )


            # ================================================
            # Optional valid mask
            # ================================================

            old_mask_raw = view.get(
                "valid_mask_path"
            )


            if old_mask_raw:

                old_mask = resolve(
                    old_mask_raw,
                    source_manifest,
                )


                mask_rel = Path(
                    "valid_masks",
                    new_video_id,
                    f"t{frame_pos:03d}",
                    f"{camera}.png",
                )


                mask_dst = (
                    output_rank_root
                    / mask_rel
                )


                process_mask(
                    old_mask,
                    mask_dst,
                )


                view[
                    "valid_mask_path"
                ] = str(
                    mask_rel
                )


    return local_index, new_item


# ============================================================
# Process each rank
# ============================================================

for rank in range(4):

    output_rank_root = (
        PREP_ROOT
        / f"rank_{rank:02d}"
    )

    output_rank_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    jobs = [
        (
            rank,
            local_index,
            item,
        )
        for local_index, item
        in enumerate(selected[rank])
    ]


    print()
    print("=" * 100)
    print(
        f"PROCESS rank_{rank:02d}: "
        f"{len(jobs)} videos"
    )
    print("=" * 100)


    results = []


    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        for done, result in enumerate(
            executor.map(
                process_one,
                jobs,
            ),
            start=1,
        ):

            results.append(
                result
            )

            if (
                done % 10 == 0
                or done == len(jobs)
            ):
                print(
                    f"[rank_{rank:02d}] "
                    f"{done}/{len(jobs)}",
                    flush=True,
                )


    results.sort(
        key=lambda x: x[0]
    )


    out_manifest = (
        output_rank_root
        / "stflow_manifest.jsonl"
    )


    with out_manifest.open(
        "w",
        encoding="utf-8",
    ) as f:

        for _, item in results:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Final prepared-rank validation
# ============================================================

print()
print("=" * 100)
print("PREPARED RANK VALIDATION")
print("=" * 100)


for rank in range(4):

    path = (
        PREP_ROOT
        / f"rank_{rank:02d}"
        / "stflow_manifest.jsonl"
    )

    items = load_jsonl(path)

    print(
        f"rank_{rank:02d}: "
        f"{len(items)}"
    )


print()
print(
    "PREPARED ROOT:",
    PREP_ROOT,
)

print()
print(
    "RANK CLEAN + 512 RESIZE + K UPDATE: PASS"
)

PY


PREP_EXIT=$?

if [ "$PREP_EXIT" -ne 0 ]; then
    echo "PREPARE FAILED: $PREP_EXIT"
    exit "$PREP_EXIT"
fi


# ============================================================
# 3. Merge first 1000
# ============================================================

echo
echo "================================================================================"
echo "MERGE FIRST 1000"
echo "================================================================================"


rm -rf "$OUT_ROOT"


python -m dwm.tools.merge_rank_preview_manifests_interleave \
    --input-root "$PREP_ROOT" \
    --output-root "$OUT_ROOT" \
    --dataset-name nuplan \
    --max-videos 1000 \
    --overwrite


MERGE_EXIT=$?

if [ "$MERGE_EXIT" -ne 0 ]; then
    echo "MERGE FAILED: $MERGE_EXIT"
    exit "$MERGE_EXIT"
fi


# ============================================================
# 4. Validate merged1000 + build Front3 manifest
#
# nuPlan 8 camera order:
#
# index 0 = CAM_L2
# index 1 = CAM_L1
# index 2 = CAM_L0    = FRONT_LEFT
# index 3 = CAM_F0    = FRONT
# index 4 = CAM_R0    = FRONT_RIGHT
# index 5 = CAM_R1
# index 6 = CAM_R2
# index 7 = CAM_B0
#
# Exported generic IDs normally:
#
# CAM_00 ... CAM_07
#
# Front3 therefore use INDEX [2,3,4].
#
# IMPORTANT:
# preserve original camera ID.
# Do not rename camera names.
# ============================================================

FULL_MANIFEST="$FULL_MANIFEST" \
FRONT3_MANIFEST="$FRONT3_MANIFEST" \
python - <<'PY'
import copy
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


full_manifest = Path(
    os.environ["FULL_MANIFEST"]
)

front_manifest = Path(
    os.environ["FRONT3_MANIFEST"]
)


def load(path):

    items = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                items.append(
                    json.loads(line)
                )

    return items


def resolve(raw, manifest):

    p = Path(raw)

    if not p.is_absolute():
        p = manifest.parent / p

    return p


items = load(
    full_manifest
)


if len(items) != 1000:

    raise RuntimeError(
        f"Expected 1000 merged videos, "
        f"got {len(items)}"
    )


frame_counts = Counter()
camera_orders = Counter()

fake_sizes = Counter()
real_sizes = Counter()

fake_missing = 0
real_missing = 0

total_views = 0


# Sample all images here because after preprocessing
# everything should be 512x288.
for vi, item in enumerate(items):

    frames = item["frames"]

    frame_counts[
        len(frames)
    ] += 1


    if len(frames) != 19:

        raise RuntimeError(
            f"video={vi}: "
            f"expected 19 frames, "
            f"got {len(frames)}"
        )


    for fi, frame in enumerate(frames):

        views = frame["views"]


        if len(views) != 8:

            raise RuntimeError(
                f"video={vi}, frame={fi}: "
                f"expected 8 cameras, "
                f"got {len(views)}"
            )


        camera_orders[
            tuple(
                v["camera"]
                for v in views
            )
        ] += 1


        for view in views:

            total_views += 1


            if tuple(
                view.get(
                    "image_size",
                    []
                )
            ) != (
                512,
                288,
            ):

                raise RuntimeError(
                    f"Bad image_size: "
                    f"{view.get('image_size')}"
                )


            K = np.asarray(
                view["K"],
                dtype=np.float64,
            )


            if K.shape != (
                3,
                3,
            ):

                raise RuntimeError(
                    f"Bad K shape: {K.shape}"
                )


            fp = resolve(
                view["image_path"],
                full_manifest,
            )


            rp = resolve(
                view["real_image_path"],
                full_manifest,
            )


            if not fp.is_file():

                fake_missing += 1

            else:

                with Image.open(fp) as im:

                    fake_sizes[
                        im.size
                    ] += 1


                    if im.size != (
                        512,
                        288,
                    ):

                        raise RuntimeError(
                            f"Bad fake size: "
                            f"{im.size} "
                            f"at {fp}"
                        )


            if not rp.is_file():

                real_missing += 1

            else:

                with Image.open(rp) as im:

                    real_sizes[
                        im.size
                    ] += 1


                    if im.size != (
                        512,
                        288,
                    ):

                        raise RuntimeError(
                            f"Bad real size: "
                            f"{im.size} "
                            f"at {rp}"
                        )


print()
print("=" * 100)
print("MERGED 1000 VALIDATION")
print("=" * 100)

print(
    "videos =",
    len(items),
)

print(
    "frame counts =",
    dict(frame_counts),
)

print(
    "views =",
    total_views,
)

print(
    "fake sizes =",
    dict(fake_sizes),
)

print(
    "real sizes =",
    dict(real_sizes),
)

print(
    "fake missing =",
    fake_missing,
)

print(
    "real missing =",
    real_missing,
)


if fake_missing:
    raise RuntimeError(
        f"{fake_missing} fake images missing"
    )


if real_missing:
    raise RuntimeError(
        f"{real_missing} paired real images missing"
    )


first_order = [
    v["camera"]
    for v in items[0]["frames"][0]["views"]
]


print()
print(
    "8-camera order =",
    first_order,
)


# ============================================================
# Front3 = indices [2,3,4]
# ============================================================

front_indices = [
    2,
    3,
    4,
]


front_names = [
    first_order[i]
    for i in front_indices
]


print()
print(
    "Front3 cameras =",
    front_names,
)


front_items = []


for item in items:

    new_item = copy.deepcopy(
        item
    )


    for frame in new_item["frames"]:

        views = frame["views"]

        frame["views"] = [
            views[i]
            for i in front_indices
        ]


    front_items.append(
        new_item
    )


with front_manifest.open(
    "w",
    encoding="utf-8",
) as f:

    for item in front_items:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


# Save machine-readable shell info.
info_path = (
    full_manifest.parent
    / "camera_eval_info.txt"
)


with info_path.open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "FULL_CAMERAS="
        + ",".join(first_order)
        + "\n"
    )

    f.write(
        "FRONT_CAMERAS="
        + ",".join(front_names)
        + "\n"
    )

    f.write(
        "FRONT_PAIRS="
        + (
            f"{front_names[0]}__{front_names[1]},"
            f"{front_names[1]}__{front_names[2]}"
        )
        + "\n"
    )


print()
print(
    "Front3 manifest:"
)

print(
    front_manifest
)

print()
print(
    "camera info:"
)

print(
    info_path
)

print()
print(
    "MERGED1000 + FRONT3: PASS"
)

PY


CHECK_EXIT=$?

if [ "$CHECK_EXIT" -ne 0 ]; then
    echo "MERGED VALIDATION FAILED: $CHECK_EXIT"
    exit "$CHECK_EXIT"
fi


# ============================================================
# 5. Read camera information
# ============================================================

source "$OUT_ROOT/camera_eval_info.txt"


echo
echo "================================================================================"
echo "EVALUATION CAMERA SETS"
echo "================================================================================"

echo
echo "8cam:"
echo "$FULL_CAMERAS"

echo
echo "Front3:"
echo "$FRONT_CAMERAS"

echo
echo "Front3 cross pairs:"
echo "$FRONT_PAIRS"


# ============================================================
# 6. Results
# ============================================================

STFLOW_8=$OUT_ROOT/stflow_traj_result_gate16_8cam.json
FVD_8=$OUT_ROOT/paired_fvd_result_8cam_all19.json

STFLOW_FRONT=$OUT_ROOT/stflow_traj_result_gate16_front3.json
FVD_FRONT=$OUT_ROOT/paired_fvd_result_front3_all19.json


LOG_8=$OUT_ROOT/stflow_gate16_8cam.log
FVD_LOG_8=$OUT_ROOT/fvd_8cam_all19.log

LOG_FRONT=$OUT_ROOT/stflow_gate16_front3.log
FVD_LOG_FRONT=$OUT_ROOT/fvd_front3_all19.log


# ============================================================
# 7. 8-camera ST-Flow + Traj
#
# Full nuPlan topology:
#
# CAM_00 -> CAM_01
# CAM_01 -> CAM_02
# ...
# CAM_07 -> CAM_00
#
# dataset policy handles nuPlan ring.
# ============================================================

echo
echo "================================================================================"
echo "1/4 RUN 8CAM ST-FLOW + TRAJ"
echo "================================================================================"


python -m dwm.tools.evaluate_stflow \
    --manifest "$FULL_MANIFEST" \
    --output "$STFLOW_8" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --frame-stride 2 \
    --min-matches 16 \
    --max-matches 256 \
    --loftr-confidence 0.1 \
    --pair-policy dataset \
    --cross-gate-px "$GATE" \
    2>&1 | tee "$LOG_8"


ST8_EXIT=${PIPESTATUS[0]}

if [ "$ST8_EXIT" -ne 0 ]; then
    echo "8CAM ST-FLOW FAILED: $ST8_EXIT"
    exit "$ST8_EXIT"
fi


# ============================================================
# 8. 8-camera FVD
#
# 1000 × 8 = 8000 video samples
# 19 frames each
# ============================================================

echo
echo "================================================================================"
echo "2/4 RUN 8CAM FVD"
echo "================================================================================"

echo "Expected samples = 1000 x 8 = 8000"


python -m dwm.tools.evaluate_fvd_from_paired_manifest \
    --manifest "$FULL_MANIFEST" \
    --output "$FVD_8" \
    --i3d-checkpoint "$I3D_CHECKPOINT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --sequence-count "$SEQ_COUNT" \
    --camera-names "$FULL_CAMERAS" \
    --batch-size 2 \
    2>&1 | tee "$FVD_LOG_8"


FVD8_EXIT=${PIPESTATUS[0]}


if [ "$FVD8_EXIT" -ne 0 ]; then

    echo
    echo "8CAM FVD batch-size=2 failed."
    echo "Retry batch-size=1."

    python -m dwm.tools.evaluate_fvd_from_paired_manifest \
        --manifest "$FULL_MANIFEST" \
        --output "$FVD_8" \
        --i3d-checkpoint "$I3D_CHECKPOINT" \
        --device cuda \
        --max-videos "$MAX_VIDEOS" \
        --sequence-count "$SEQ_COUNT" \
        --camera-names "$FULL_CAMERAS" \
        --batch-size 1 \
        2>&1 | tee -a "$FVD_LOG_8"


    FVD8_EXIT=${PIPESTATUS[0]}
fi


if [ "$FVD8_EXIT" -ne 0 ]; then
    echo "8CAM FVD FAILED: $FVD8_EXIT"
    exit "$FVD8_EXIT"
fi


# ============================================================
# 9. Front3 ST-Flow + Traj
#
# Explicit pairs only:
#
# FRONT_LEFT -> FRONT
# FRONT       -> FRONT_RIGHT
#
# No wrap-around.
# ============================================================

echo
echo "================================================================================"
echo "3/4 RUN FRONT3 ST-FLOW + TRAJ"
echo "================================================================================"

echo "cameras:"
echo "$FRONT_CAMERAS"

echo
echo "pairs:"
echo "$FRONT_PAIRS"


python -m dwm.tools.evaluate_stflow \
    --manifest "$FRONT3_MANIFEST" \
    --output "$STFLOW_FRONT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --frame-stride 2 \
    --min-matches 16 \
    --max-matches 256 \
    --loftr-confidence 0.1 \
    --camera-pairs "$FRONT_PAIRS" \
    --pair-policy dataset \
    --cross-gate-px "$GATE" \
    2>&1 | tee "$LOG_FRONT"


STF_EXIT=${PIPESTATUS[0]}

if [ "$STF_EXIT" -ne 0 ]; then
    echo "FRONT3 ST-FLOW FAILED: $STF_EXIT"
    exit "$STF_EXIT"
fi


# ============================================================
# 10. Front3 FVD
#
# 1000 × 3 = 3000 video samples
# 19 frames each
# ============================================================

echo
echo "================================================================================"
echo "4/4 RUN FRONT3 FVD"
echo "================================================================================"

echo "Expected samples = 1000 x 3 = 3000"


python -m dwm.tools.evaluate_fvd_from_paired_manifest \
    --manifest "$FRONT3_MANIFEST" \
    --output "$FVD_FRONT" \
    --i3d-checkpoint "$I3D_CHECKPOINT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --sequence-count "$SEQ_COUNT" \
    --camera-names "$FRONT_CAMERAS" \
    --batch-size 2 \
    2>&1 | tee "$FVD_LOG_FRONT"


FVDF_EXIT=${PIPESTATUS[0]}


if [ "$FVDF_EXIT" -ne 0 ]; then

    echo
    echo "FRONT3 FVD batch-size=2 failed."
    echo "Retry batch-size=1."

    python -m dwm.tools.evaluate_fvd_from_paired_manifest \
        --manifest "$FRONT3_MANIFEST" \
        --output "$FVD_FRONT" \
        --i3d-checkpoint "$I3D_CHECKPOINT" \
        --device cuda \
        --max-videos "$MAX_VIDEOS" \
        --sequence-count "$SEQ_COUNT" \
        --camera-names "$FRONT_CAMERAS" \
        --batch-size 1 \
        2>&1 | tee -a "$FVD_LOG_FRONT"


    FVDF_EXIT=${PIPESTATUS[0]}
fi


if [ "$FVDF_EXIT" -ne 0 ]; then
    echo "FRONT3 FVD FAILED: $FVDF_EXIT"
    exit "$FVDF_EXIT"
fi


# ============================================================
# 11. Final summary
# ============================================================

STFLOW_8="$STFLOW_8" \
FVD_8="$FVD_8" \
STFLOW_FRONT="$STFLOW_FRONT" \
FVD_FRONT="$FVD_FRONT" \
python - <<'PY'
import json
import os
from pathlib import Path


paths = {
    "8CAM STFLOW": Path(
        os.environ["STFLOW_8"]
    ),
    "8CAM FVD": Path(
        os.environ["FVD_8"]
    ),
    "FRONT3 STFLOW": Path(
        os.environ["STFLOW_FRONT"]
    ),
    "FRONT3 FVD": Path(
        os.environ["FVD_FRONT"]
    ),
}


print()
print("=" * 110)
print("1000 NUPLAN 6HZ FINAL RESULTS")
print("=" * 110)


for title in [
    "8CAM STFLOW",
    "FRONT3 STFLOW",
]:

    path = paths[title]

    print()
    print(title)
    print("-" * 110)

    if not path.is_file():
        print("MISSING:", path)
        continue

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)


    mean = data.get(
        "mean",
        {}
    )


    for key in [
        "temporal_l1",
        "cross_raw_epi_px",
        "cross_epi_px",
        "cross_inlier_ratio",
        "traj_epi_px",
        "traj_inlier2",
        "traj_inlier4",
        "edge_coverage",
        "cycle_coverage",
        "stflow_score",
        "stflow_d_score",
        "stflow_c_score",
    ]:

        if key in mean:

            print(
                f"{key:28s} = {mean[key]}"
            )


for title in [
    "8CAM FVD",
    "FRONT3 FVD",
]:

    path = paths[title]

    print()
    print(title)
    print("-" * 110)

    if not path.is_file():
        print("MISSING:", path)
        continue

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)


    print(
        "fvd =",
        data.get("fvd"),
    )

    print(
        "num_videos =",
        data.get("num_videos"),
    )

    print(
        "num_samples =",
        data.get("num_samples"),
    )

    print(
        "camera_names =",
        data.get("camera_names"),
    )


print()
print("=" * 110)

PY


echo
echo "================================================================================"
echo "ALL DONE"
echo "================================================================================"

echo
echo "Final root:"
echo "$OUT_ROOT"

echo
echo "Full 8cam manifest:"
echo "$FULL_MANIFEST"

echo
echo "Front3 manifest:"
echo "$FRONT3_MANIFEST"

echo
echo "8cam ST-Flow:"
echo "$STFLOW_8"

echo
echo "8cam FVD:"
echo "$FVD_8"

echo
echo "Front3 ST-Flow:"
echo "$STFLOW_FRONT"

echo
echo "Front3 FVD:"
echo "$FVD_FRONT"

