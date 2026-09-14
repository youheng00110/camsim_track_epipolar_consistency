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
# 1. Checkpoints
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
# 2. Paths
# ============================================================

BASE=$CAMSIM_ROOT/lyh_output/eval/nuplanhard1000

SRC=$BASE/uropetvtrack
ROOT=$BASE/uropetvtrack_merged1000

MANIFEST=$ROOT/stflow_manifest.jsonl

MAX_VIDEOS=1000
GATE=16

STFLOW_OUTPUT=$ROOT/stflow_traj_result_gate16_8cam.json
STFLOW_LOG=$ROOT/stflow_gate16_8cam.log

FVD_OUTPUT=$ROOT/paired_fvd_result_8cam.json
FVD_LOG=$ROOT/fvd_8cam.log


echo
echo "================================================================================"
echo "UROPETVTRACK NUPLAN 1000 8CAM"
echo "================================================================================"

echo
echo "source:"
echo "$SRC"

echo
echo "output:"
echo "$ROOT"


if [ ! -d "$SRC" ]; then
    echo "ERROR: source directory missing:"
    echo "$SRC"
    exit 1
fi


# ============================================================
# 3. Show rank counts before merge
# ============================================================

echo
echo "================================================================================"
echo "RANK MANIFEST COUNTS"
echo "================================================================================"

python - "$SRC" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])

total = 0

for p in sorted(root.glob("rank_*/stflow_manifest.jsonl")):
    n = sum(
        1
        for line in p.open("r", encoding="utf-8")
        if line.strip()
    )

    print(
        f"{p.parent.name:12s} = {n}"
    )

    total += n

print()
print("raw total =", total)

if total < 1000:
    raise RuntimeError(
        f"Only {total} videos available, need at least 1000"
    )
PY

if [ $? -ne 0 ]; then
    exit 1
fi


# ============================================================
# 4. Merge first 1000
#
# rank_00[0], rank_01[0], ...
# rank_00[1], rank_01[1], ...
#
# until 1000 videos.
# ============================================================

echo
echo "================================================================================"
echo "MERGE FIRST 1000"
echo "================================================================================"

rm -rf "$ROOT"

python -m dwm.tools.merge_rank_preview_manifests_interleave \
    --input-root "$SRC" \
    --output-root "$ROOT" \
    --dataset-name nuplan \
    --max-videos "$MAX_VIDEOS" \
    --overwrite

MERGE_EXIT=$?

if [ "$MERGE_EXIT" -ne 0 ]; then
    echo "MERGE FAILED: $MERGE_EXIT"
    exit "$MERGE_EXIT"
fi


if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: merged manifest missing:"
    echo "$MANIFEST"
    exit 1
fi


# ============================================================
# 5. Strict manifest validation
#
# Check:
# - exactly 1000 videos
# - consistent frame count
# - exactly 8 cameras
# - consistent camera order
# - generated images exist
# - paired real exists
# - K / extrinsic fields present
#
# NO resize
# NO K modification
# ============================================================

MANIFEST="$MANIFEST" python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


manifest = Path(
    os.environ["MANIFEST"]
)


def resolve(raw):
    if raw is None:
        return None

    p = Path(raw)

    if not p.is_absolute():
        p = manifest.parent / p

    return p


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


frame_counts = Counter()
camera_orders = Counter()
manifest_sizes = Counter()

fake_sizes = Counter()
real_sizes = Counter()

fake_missing = []
real_missing = []

total_views = 0

CHECK_IMAGE_LIMIT = 3000
checked_fake = 0
checked_real = 0


for vi, item in enumerate(items):

    frames = item.get(
        "frames",
        []
    )

    frame_counts[
        len(frames)
    ] += 1


    for fi, frame in enumerate(
        frames
    ):

        views = frame.get(
            "views",
            []
        )

        if len(views) != 8:

            raise RuntimeError(
                f"video={vi}, frame={fi}: "
                f"expected 8 cameras, got {len(views)}"
            )


        camera_orders[
            tuple(
                v.get("camera")
                for v in views
            )
        ] += 1


        if frame.get(
            "T_ego_to_world"
        ) is None:

            raise RuntimeError(
                f"T_ego_to_world missing: "
                f"video={vi}, frame={fi}"
            )


        for view in views:

            total_views += 1

            camera = view.get(
                "camera"
            )


            # --------------------------------------------
            # Calibration
            # --------------------------------------------

            if view.get("K") is None:

                raise RuntimeError(
                    f"K missing: "
                    f"video={vi}, frame={fi}, camera={camera}"
                )


            K = np.asarray(
                view["K"],
                dtype=np.float64,
            )

            if K.shape != (3, 3):

                raise RuntimeError(
                    f"Bad K shape: "
                    f"video={vi}, frame={fi}, camera={camera}, "
                    f"shape={K.shape}"
                )


            if view.get(
                "T_cam_to_ego"
            ) is None:

                raise RuntimeError(
                    f"T_cam_to_ego missing: "
                    f"video={vi}, frame={fi}, camera={camera}"
                )


            image_size = tuple(
                view.get(
                    "image_size",
                    []
                )
            )

            manifest_sizes[
                image_size
            ] += 1


            # --------------------------------------------
            # Generated image
            # --------------------------------------------

            fp = resolve(
                view.get(
                    "image_path"
                )
            )


            if (
                fp is None
                or not fp.is_file()
            ):

                fake_missing.append(
                    (
                        vi,
                        fi,
                        camera,
                        str(fp),
                    )
                )

            elif checked_fake < CHECK_IMAGE_LIMIT:

                with Image.open(fp) as im:

                    fake_sizes[
                        im.size
                    ] += 1


                    if (
                        image_size
                        and tuple(im.size)
                        != image_size
                    ):

                        raise RuntimeError(
                            "\nGenerated image_size mismatch\n"
                            f"video={vi}\n"
                            f"frame={fi}\n"
                            f"camera={camera}\n"
                            f"manifest={image_size}\n"
                            f"file={im.size}\n"
                            f"path={fp}\n"
                        )

                checked_fake += 1


            # --------------------------------------------
            # Paired real
            # --------------------------------------------

            rp = resolve(
                view.get(
                    "real_image_path"
                )
            )


            if (
                rp is None
                or not rp.is_file()
            ):

                real_missing.append(
                    (
                        vi,
                        fi,
                        camera,
                        str(rp),
                    )
                )

            elif checked_real < CHECK_IMAGE_LIMIT:

                with Image.open(rp) as im:

                    real_sizes[
                        im.size
                    ] += 1

                checked_real += 1


print()
print("=" * 100)
print("MERGED MANIFEST CHECK")
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
    "total views =",
    total_views,
)

print()
print(
    "camera orders:"
)

for order, n in camera_orders.items():

    print(
        n,
        "frames ->",
        list(order),
    )


print()
print(
    "manifest image_size =",
    dict(manifest_sizes),
)

print(
    "sampled generated sizes =",
    dict(fake_sizes),
)

print(
    "sampled paired-real sizes =",
    dict(real_sizes),
)

print()
print(
    "fake missing =",
    len(fake_missing),
)

print(
    "real missing =",
    len(real_missing),
)


if fake_missing:

    print()
    print(
        "First fake missing:"
    )

    for x in fake_missing[:10]:
        print(x)


if real_missing:

    print()
    print(
        "First real missing:"
    )

    for x in real_missing[:10]:
        print(x)


if len(items) != 1000:

    raise RuntimeError(
        f"Expected 1000 videos, got {len(items)}"
    )


if len(frame_counts) != 1:

    raise RuntimeError(
        f"Inconsistent frame counts: {frame_counts}"
    )


if len(camera_orders) != 1:

    raise RuntimeError(
        f"Inconsistent camera order: {camera_orders}"
    )


if fake_missing:

    raise RuntimeError(
        f"{len(fake_missing)} generated images missing"
    )


if real_missing:

    raise RuntimeError(
        f"{len(real_missing)} paired real images missing"
    )


print()
print(
    "MERGED1000 8CAM MANIFEST: PASS"
)

PY


CHECK_EXIT=$?

if [ "$CHECK_EXIT" -ne 0 ]; then
    echo "MANIFEST CHECK FAILED: $CHECK_EXIT"
    exit "$CHECK_EXIT"
fi


# ============================================================
# 6. Read sequence count + camera names
# ============================================================

readarray -t INFO < <(
MANIFEST="$MANIFEST" python - <<'PY'
import json
import os


manifest = os.environ["MANIFEST"]


with open(
    manifest,
    "r",
    encoding="utf-8",
) as f:

    item = json.loads(
        next(
            line
            for line in f
            if line.strip()
        )
    )


frames = item["frames"]

if not frames:
    raise RuntimeError(
        "Empty frames"
    )


camera_names = [
    v["camera"]
    for v in frames[0]["views"]
]


if len(camera_names) != 8:
    raise RuntimeError(
        f"Expected 8 cameras, got {camera_names}"
    )


print(
    len(frames)
)

print(
    ",".join(camera_names)
)

PY
)


SEQ_COUNT="${INFO[0]}"
CAMERAS="${INFO[1]}"


echo
echo "================================================================================"
echo "EVALUATION INFO"
echo "================================================================================"

echo
echo "videos:"
echo "$MAX_VIDEOS"

echo
echo "sequence count:"
echo "$SEQ_COUNT"

echo
echo "8 cameras:"
echo "$CAMERAS"


# ============================================================
# 7. 8CAM ST-Flow + Traj
#
# nuPlan complete ring selected by dataset policy.
#
# No --start-frame.
# Evaluator handles reference frames automatically.
# ============================================================

echo
echo "================================================================================"
echo "RUN UROPETVTRACK 8CAM ST-FLOW + TRAJ"
echo "================================================================================"


python -m dwm.tools.evaluate_stflow \
    --manifest "$MANIFEST" \
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


ST_EXIT=${PIPESTATUS[0]}


if [ "$ST_EXIT" -ne 0 ]; then

    echo
    echo "ST-FLOW FAILED:"
    echo "$ST_EXIT"

    exit "$ST_EXIT"
fi


# ============================================================
# 8. 8CAM FVD
#
# 1000 videos × 8 cameras
# = 8000 video samples
# ============================================================

echo
echo "================================================================================"
echo "RUN UROPETVTRACK 8CAM FVD"
echo "================================================================================"

echo
echo "Expected:"
echo "1000 videos x 8 cameras = 8000 video samples"
echo "$SEQ_COUNT frames / sample"


python -m dwm.tools.evaluate_fvd_from_paired_manifest \
    --manifest "$MANIFEST" \
    --output "$FVD_OUTPUT" \
    --i3d-checkpoint "$I3D_CHECKPOINT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --sequence-count "$SEQ_COUNT" \
    --camera-names "$CAMERAS" \
    --batch-size 2 \
    2>&1 | tee "$FVD_LOG"


FVD_EXIT=${PIPESTATUS[0]}


if [ "$FVD_EXIT" -ne 0 ]; then

    echo
    echo "FVD batch-size=2 failed."
    echo "Retry batch-size=1."

    python -m dwm.tools.evaluate_fvd_from_paired_manifest \
        --manifest "$MANIFEST" \
        --output "$FVD_OUTPUT" \
        --i3d-checkpoint "$I3D_CHECKPOINT" \
        --device cuda \
        --max-videos "$MAX_VIDEOS" \
        --sequence-count "$SEQ_COUNT" \
        --camera-names "$CAMERAS" \
        --batch-size 1 \
        2>&1 | tee -a "$FVD_LOG"


    FVD_EXIT=${PIPESTATUS[0]}
fi


if [ "$FVD_EXIT" -ne 0 ]; then

    echo
    echo "FVD FAILED:"
    echo "$FVD_EXIT"

    exit "$FVD_EXIT"
fi


# ============================================================
# 9. Summary
# ============================================================

STFLOW_OUTPUT="$STFLOW_OUTPUT" \
FVD_OUTPUT="$FVD_OUTPUT" \
python - <<'PY'
import json
import os


st_path = os.environ[
    "STFLOW_OUTPUT"
]

fvd_path = os.environ[
    "FVD_OUTPUT"
]


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


mean = st.get(
    "mean",
    {}
)


print()
print("=" * 110)
print("UROPETVTRACK NUPLAN 1000 8CAM RESULTS")
print("=" * 110)

print()
print("ST-FLOW / TRAJ")
print("-" * 110)


for key in [
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
]:

    if key in mean:

        print(
            f"{key:28s} = {mean[key]}"
        )


print()
print("FVD")
print("-" * 110)

print(
    "fvd            =",
    fv.get("fvd"),
)

print(
    "num_videos     =",
    fv.get("num_videos"),
)

print(
    "num_samples    =",
    fv.get("num_samples"),
)

print(
    "camera_names   =",
    fv.get("camera_names"),
)

print(
    "sequence_count =",
    fv.get("sequence_count"),
)


print()
print("=" * 110)

print(
    "ST-Flow result:"
)

print(
    st_path
)

print()

print(
    "FVD result:"
)

print(
    fvd_path
)

PY


echo
echo "================================================================================"
echo "ALL DONE"
echo "================================================================================"

echo
echo "Merged manifest:"
echo "$MANIFEST"

echo
echo "ST-Flow:"
echo "$STFLOW_OUTPUT"

echo
echo "FVD:"
echo "$FVD_OUTPUT"

