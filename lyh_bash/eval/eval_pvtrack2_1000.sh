#!/bin/bash

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

cd /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/src || exit 1

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export CAMSIM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim
export OPENDWM_ROOT=$CAMSIM_ROOT/OpenDWM

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [ -d "$OPENDWM_ROOT/externals/TATS/tats/fvd" ]; then
    export PYTHONPATH="$OPENDWM_ROOT/externals/TATS/tats/fvd:$PYTHONPATH"
fi

if [ -d "$CAMSIM_ROOT/nuplan-devkit-master" ]; then
    export PYTHONPATH="$CAMSIM_ROOT/nuplan-devkit-master:$PYTHONPATH"
fi


# ============================================================
# checkpoints
# ============================================================

export TORCH_HOME=/root/.cache/torch

CKPT_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt

RAFT_SOURCE=$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth
LOFTR_SOURCE=$CKPT_ROOT/loftr_outdoor.ckpt
I3D_CHECKPOINT=$CKPT_ROOT/i3d_pretrained_400.pt

CACHE_DIR=$TORCH_HOME/hub/checkpoints

mkdir -p "$CACHE_DIR"

for P in \
    "$RAFT_SOURCE" \
    "$LOFTR_SOURCE" \
    "$I3D_CHECKPOINT"
do
    if [ ! -f "$P" ]; then
        echo "ERROR: checkpoint missing:"
        echo "$P"
        exit 1
    fi
done

cp -f \
    "$RAFT_SOURCE" \
    "$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

cp -f \
    "$LOFTR_SOURCE" \
    "$CACHE_DIR/loftr_outdoor.ckpt"


# ============================================================
# paths
# ============================================================

BASE=$CAMSIM_ROOT/lyh_output/eval/nuplanhard1000

SRC=$BASE/pvtrack2
ROOT=$BASE/pvtrack2_merged1000

MANIFEST=$ROOT/stflow_manifest.jsonl

MAX_VIDEOS=1000
GATE=16


echo
echo "================================================================================"
echo "PVTRACK2 -> MERGED1000 -> ST-FLOW + FVD"
echo "================================================================================"

echo "SRC:"
echo "$SRC"

echo
echo "ROOT:"
echo "$ROOT"


if [ ! -d "$SRC" ]; then
    echo "ERROR: source directory missing:"
    echo "$SRC"
    exit 1
fi


# ============================================================
# 1. merge first 1000
# ============================================================

NEED_MERGE=1

if [ -f "$MANIFEST" ]; then

    COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST" || true)

    echo
    echo "Existing merged manifest videos: $COUNT"

    if [ "$COUNT" -eq 1000 ]; then
        NEED_MERGE=0
        echo "Already merged1000. Skip merge."
    fi
fi


if [ "$NEED_MERGE" -eq 1 ]; then

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

        echo
        echo "Hard-link merge failed."
        echo "Retry with --copy."

        rm -rf "$ROOT"

        python -m dwm.tools.merge_rank_preview_manifests_interleave \
            --input-root "$SRC" \
            --output-root "$ROOT" \
            --dataset-name nuplan \
            --max-videos "$MAX_VIDEOS" \
            --overwrite \
            --copy

        MERGE_EXIT=$?
    fi


    if [ "$MERGE_EXIT" -ne 0 ]; then
        echo "ERROR: merge failed."
        exit "$MERGE_EXIT"
    fi
fi


# ============================================================
# 2. validate merged manifest
# ============================================================

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest missing:"
    echo "$MANIFEST"
    exit 1
fi


MANIFEST="$MANIFEST" python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image


manifest = Path(os.environ["MANIFEST"])

items = []

with manifest.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            items.append(json.loads(line))


print()
print("=" * 100)
print("MERGED MANIFEST CHECK")
print("=" * 100)

print("videos =", len(items))


if len(items) != 1000:
    raise RuntimeError(
        f"Expected 1000 videos, got {len(items)}"
    )


frame_counts = Counter()
camera_orders = Counter()
image_sizes = Counter()

fake_missing = 0
real_none = 0
real_missing = 0

total_views = 0


def resolve(raw):

    if raw is None:
        return None

    p = Path(raw)

    if not p.is_absolute():
        p = manifest.parent / p

    return p


for vi, item in enumerate(items):

    frames = item.get("frames", [])

    frame_counts[len(frames)] += 1

    if not frames:
        raise RuntimeError(
            f"video={vi} has no frames"
        )


    order = tuple(
        view.get("camera")
        for view in frames[0].get("views", [])
    )

    camera_orders[order] += 1


    for frame in frames:

        if tuple(
            view.get("camera")
            for view in frame.get("views", [])
        ) != order:

            raise RuntimeError(
                f"camera order changes in video={vi}"
            )


        for view in frame.get("views", []):

            total_views += 1

            fp = resolve(
                view.get("image_path")
            )

            rp = resolve(
                view.get("real_image_path")
            )


            if fp is None or not fp.is_file():

                fake_missing += 1

            else:

                # 只统计实际生成图尺寸。
                with Image.open(fp) as im:
                    image_sizes[im.size] += 1


            if view.get("real_image_path") is None:

                real_none += 1

            elif rp is None or not rp.is_file():

                real_missing += 1


print()
print("frame count distribution:")
print(dict(frame_counts))

print()
print("camera orders:")

for order, count in camera_orders.items():
    print(
        count,
        "videos ->",
        list(order),
    )

print()
print("generated image sizes:")
print(dict(image_sizes))

print()
print("total views      =", total_views)
print("fake missing     =", fake_missing)
print("real path None   =", real_none)
print("real file missing=", real_missing)


if len(frame_counts) != 1:
    raise RuntimeError(
        f"Inconsistent frame counts: {dict(frame_counts)}"
    )


if len(camera_orders) != 1:
    raise RuntimeError(
        f"Inconsistent camera orders: {dict(camera_orders)}"
    )


if fake_missing != 0:
    raise RuntimeError(
        f"{fake_missing} generated images missing"
    )


if real_none != 0:
    raise RuntimeError(
        f"{real_none} real_image_path entries are None; "
        "FVD cannot run."
    )


if real_missing != 0:
    raise RuntimeError(
        f"{real_missing} paired-real files missing"
    )


print()
print("MERGED MANIFEST: PASS")
PY


CHECK_EXIT=$?

if [ "$CHECK_EXIT" -ne 0 ]; then
    echo "Manifest validation failed."
    exit "$CHECK_EXIT"
fi


# ============================================================
# 3. sequence count / camera list
# ============================================================

INFO=$(MANIFEST="$MANIFEST" python - <<'PY'
import json
import os
from collections import Counter


manifest = os.environ["MANIFEST"]

items = []

with open(manifest, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            items.append(json.loads(line))


frame_counts = Counter(
    len(item["frames"])
    for item in items
)


if len(frame_counts) != 1:
    raise RuntimeError(
        f"Inconsistent frame counts: {dict(frame_counts)}"
    )


seq_count = next(
    iter(frame_counts)
)


cameras = [
    view["camera"]
    for view in items[0]["frames"][0]["views"]
]


print(seq_count)
print(",".join(cameras))
PY
)


SEQ_COUNT=$(echo "$INFO" | sed -n '1p')
CAMERAS=$(echo "$INFO" | sed -n '2p')


echo
echo "================================================================================"
echo "EVALUATION CONFIG"
echo "================================================================================"

echo "videos   = $MAX_VIDEOS"
echo "frames   = $SEQ_COUNT"
echo "cameras  = $CAMERAS"
echo "gate     = $GATE"


# ============================================================
# 4. ST-Flow + Traj
# ============================================================

STFLOW_OUTPUT=$ROOT/stflow_traj_result_gate16.json

mkdir -p "$ROOT/eval_logs"

STFLOW_LOG=$ROOT/eval_logs/stflow_gate16.log


echo
echo "================================================================================"
echo "RUN ST-FLOW + TRAJ"
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


STFLOW_EXIT=${PIPESTATUS[0]}


if [ "$STFLOW_EXIT" -ne 0 ]; then
    echo "ST-Flow failed: $STFLOW_EXIT"
    exit "$STFLOW_EXIT"
fi


# ============================================================
# 5. FVD
# ============================================================

FVD_OUTPUT=$ROOT/paired_fvd_result_all${SEQ_COUNT}.json
FVD_LOG=$ROOT/eval_logs/fvd_all${SEQ_COUNT}.log


echo
echo "================================================================================"
echo "RUN FVD"
echo "================================================================================"

echo "videos  = $MAX_VIDEOS"
echo "cameras = $CAMERAS"
echo "frames  = $SEQ_COUNT"


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
    echo "FVD failed: $FVD_EXIT"
    exit "$FVD_EXIT"
fi


# ============================================================
# 6. summary
# ============================================================

STFLOW_OUTPUT="$STFLOW_OUTPUT" \
FVD_OUTPUT="$FVD_OUTPUT" \
python - <<'PY'
import json
import os
from pathlib import Path


st_path = Path(
    os.environ["STFLOW_OUTPUT"]
)

fvd_path = Path(
    os.environ["FVD_OUTPUT"]
)


print()
print("=" * 110)
print("PVTRACK2 MERGED1000 RESULT")
print("=" * 110)


with st_path.open(
    "r",
    encoding="utf-8",
) as f:

    st = json.load(f)


mean = st.get(
    "mean",
    {}
)


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


with fvd_path.open(
    "r",
    encoding="utf-8",
) as f:

    fv = json.load(f)


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
print("=" * 110)
PY


echo
echo "================================================================================"
echo "ALL DONE"
echo "================================================================================"

echo
echo "Merged root:"
echo "$ROOT"

echo
echo "Manifest:"
echo "$MANIFEST"

echo
echo "ST-Flow:"
echo "$STFLOW_OUTPUT"

echo
echo "FVD:"
echo "$FVD_OUTPUT"

