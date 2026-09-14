#!/bin/bash


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

cd "$OPENDWM_ROOT/src"

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
        echo "ERROR: checkpoint missing:"
        echo "$P"
        exit 1
    fi
done


# RAFT / LoFTR evaluator 会从 torch cache 找权重
export TORCH_HOME=/root/.cache/torch
CACHE_DIR=$TORCH_HOME/hub/checkpoints

mkdir -p "$CACHE_DIR"

cp -f \
    "$RAFT_LOCAL" \
    "$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

cp -f \
    "$LOFTR_LOCAL" \
    "$CACHE_DIR/loftr_outdoor.ckpt"


# ============================================================
# 2. Paths
# ============================================================

BASE=$CAMSIM_ROOT/lyh_output/eval/nuplanhard1000

SRC=$BASE/pvbev
ROOT=$BASE/pvbev_merged1000
MANIFEST=$ROOT/stflow_manifest.jsonl

MAX_VIDEOS=1000
GATE=16


# ============================================================
# 3. Merge first 1000
# ============================================================

echo
echo "================================================================================"
echo "PVBEV / NUPLANHARD1000"
echo "================================================================================"

echo "source:"
echo "$SRC"

echo
echo "merged:"
echo "$ROOT"


NEED_MERGE=1

if [ -f "$MANIFEST" ]; then

    COUNT=$(
        grep -cve '^[[:space:]]*$' "$MANIFEST" || true
    )

    echo
    echo "Existing merged manifest:"
    echo "$MANIFEST"
    echo "videos=$COUNT"

    if [ "$COUNT" -eq 1000 ]; then
        NEED_MERGE=0
        echo "Existing merged1000 valid. Skip merge."
    fi
fi


if [ "$NEED_MERGE" -eq 1 ]; then

    echo
    echo "================================================================================"
    echo "MERGE FIRST 1000"
    echo "================================================================================"

    python -m dwm.tools.merge_rank_preview_manifests_interleave \
        --input-root "$SRC" \
        --output-root "$ROOT" \
        --dataset-name nuplan \
        --max-videos 1000 \
        --overwrite

fi


if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest missing after merge:"
    echo "$MANIFEST"
    exit 1
fi


# ============================================================
# 4. Inspect manifest
#
# IMPORTANT:
# 不 resize
# 不改 K
# 不复制图片
# 这里只检查当前 pvbev 本身是什么分辨率
# ============================================================

echo
echo "================================================================================"
echo "PVBEV ORIGINAL DATA CHECK"
echo "================================================================================"

MANIFEST="$MANIFEST" python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


manifest = Path(os.environ["MANIFEST"])

items = []

with manifest.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            items.append(json.loads(line))


frame_counts = Counter()
camera_counts = Counter()
camera_orders = Counter()

fake_sizes = Counter()
real_sizes = Counter()
manifest_sizes = Counter()

fake_missing = 0
real_missing = 0
real_none = 0

k_stats = {}

# 检查前若干真实文件即可得到尺寸分布；
# 同时遍历所有 manifest metadata。
CHECK_IMAGE_LIMIT = 5000
checked_fake = 0
checked_real = 0


def resolve(raw):
    if raw is None:
        return None

    p = Path(raw)

    if not p.is_absolute():
        p = manifest.parent / p

    return p


for item in items:

    frames = item.get("frames", [])

    frame_counts[len(frames)] += 1

    if frames:
        order = tuple(
            v.get("camera")
            for v in frames[0].get("views", [])
        )

        camera_orders[order] += 1
        camera_counts[len(order)] += 1


    for frame in frames:

        for view in frame.get("views", []):

            cam = view.get("camera")

            image_size = tuple(
                view.get("image_size", [])
            )

            manifest_sizes[image_size] += 1


            # --------------------------------------------
            # K
            # --------------------------------------------

            K_raw = view.get("K")

            if K_raw is not None:

                K = np.asarray(
                    K_raw,
                    dtype=np.float64,
                )

                if K.shape == (3, 3):

                    row = k_stats.setdefault(
                        cam,
                        {
                            "fx": [],
                            "fy": [],
                            "cx": [],
                            "cy": [],
                        },
                    )

                    row["fx"].append(K[0, 0])
                    row["fy"].append(K[1, 1])
                    row["cx"].append(K[0, 2])
                    row["cy"].append(K[1, 2])


            # --------------------------------------------
            # generated
            # --------------------------------------------

            fp = resolve(
                view.get("image_path")
            )

            if fp is None or not fp.is_file():

                fake_missing += 1

            elif checked_fake < CHECK_IMAGE_LIMIT:

                with Image.open(fp) as im:
                    fake_sizes[im.size] += 1

                checked_fake += 1


            # --------------------------------------------
            # paired real
            # --------------------------------------------

            raw_real = view.get(
                "real_image_path"
            )

            if not raw_real:

                real_none += 1

            else:

                rp = resolve(raw_real)

                if not rp.is_file():

                    real_missing += 1

                elif checked_real < CHECK_IMAGE_LIMIT:

                    with Image.open(rp) as im:
                        real_sizes[im.size] += 1

                    checked_real += 1


print("videos:")
print(len(items))

print()
print("frame-count distribution:")
print(dict(frame_counts))

print()
print("camera-count distribution:")
print(dict(camera_counts))

print()
print("camera orders:")

for order, count in camera_orders.items():
    print(
        f"{count} videos -> {list(order)}"
    )


print()
print("manifest image_size:")
print(dict(manifest_sizes))

print()
print("sampled generated image sizes:")
print(dict(fake_sizes))

print()
print("sampled paired-real sizes:")
print(dict(real_sizes))

print()
print("fake missing:")
print(fake_missing)

print()
print("real_image_path None:")
print(real_none)

print()
print("real file missing:")
print(real_missing)


print()
print("K summary:")

for cam, x in k_stats.items():

    print(
        cam,
        "fx=", round(float(np.mean(x["fx"])), 3),
        "fy=", round(float(np.mean(x["fy"])), 3),
        "cx=", round(float(np.mean(x["cx"])), 3),
        "cy=", round(float(np.mean(x["cy"])), 3),
    )


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
        f"{fake_missing} generated images missing"
    )


print()
print("=" * 100)
print("PVBEV INPUT CHECK PASS")
print("=" * 100)

PY


# ============================================================
# 5. Determine sequence count + cameras
# ============================================================

readarray -t INFO < <(
MANIFEST="$MANIFEST" python - <<'PY'
import json
import os

manifest = os.environ["MANIFEST"]

with open(manifest, "r", encoding="utf-8") as f:
    item = json.loads(next(
        line for line in f if line.strip()
    ))

frames = item["frames"]

if not frames:
    raise RuntimeError("Empty frames")

seq_count = len(frames)

cameras = [
    v["camera"]
    for v in frames[0]["views"]
]

print(seq_count)
print(",".join(cameras))
PY
)

SEQ_COUNT="${INFO[0]}"
CAMERAS="${INFO[1]}"

echo
echo "sequence-count = $SEQ_COUNT"
echo "cameras        = $CAMERAS"


# ============================================================
# 6. ST-Flow / Traj
#
# 直接测原始 pvbev：
# - 不 resize generated images
# - 不改 K
# - dataset policy 自动走 nuPlan 相机 topology
# ============================================================

STFLOW_OUTPUT=$ROOT/stflow_traj_result_gate16.json
STFLOW_LOG=$ROOT/stflow_gate16.log


echo
echo "================================================================================"
echo "RUN PVBEV ST-FLOW / TRAJ"
echo "================================================================================"

echo "videos       : $MAX_VIDEOS"
echo "frames       : $SEQ_COUNT"
echo "cameras      : $CAMERAS"
echo "gate         : $GATE"
echo "frame-stride : 2"
echo "pair-policy  : dataset"
echo
echo "NO IMAGE RESIZE"
echo "NO K RESCALE"


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


# ============================================================
# 7. FVD
#
# 只有 paired real 完整时才直接跑。
# 不做任何 resize。
# ============================================================

FVD_OUTPUT=$ROOT/paired_fvd_result_all${SEQ_COUNT}.json
FVD_LOG=$ROOT/fvd_all${SEQ_COUNT}.log


PAIR_STATUS=$(
MANIFEST="$MANIFEST" python - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])

total = 0
missing = 0


for line in manifest.open(
    "r",
    encoding="utf-8",
):

    if not line.strip():
        continue

    item = json.loads(line)

    for frame in item["frames"]:

        for view in frame["views"]:

            total += 1

            raw = view.get(
                "real_image_path"
            )

            if not raw:
                missing += 1
                continue

            p = Path(raw)

            if not p.is_absolute():
                p = manifest.parent / p

            if not p.is_file():
                missing += 1


print(
    "FULL"
    if total > 0 and missing == 0
    else f"MISSING:{missing}/{total}"
)
PY
)


echo
echo "paired-real status: $PAIR_STATUS"


if [ "$PAIR_STATUS" = "FULL" ]; then

    echo
    echo "================================================================================"
    echo "RUN PVBEV FVD"
    echo "================================================================================"

    echo "NO IMAGE RESIZE"


    if ! python -m dwm.tools.evaluate_fvd_from_paired_manifest \
        --manifest "$MANIFEST" \
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

    fi

else

    echo
    echo "================================================================================"
    echo "SKIP FVD"
    echo "================================================================================"

    echo "paired_real is incomplete:"
    echo "$PAIR_STATUS"

    echo
    echo "ST-Flow / Traj has still been evaluated normally."
fi


# ============================================================
# 8. Summary
# ============================================================

echo
echo "================================================================================"
echo "PVBEV EVALUATION DONE"
echo "================================================================================"

echo
echo "manifest:"
echo "$MANIFEST"

echo
echo "ST-Flow:"
echo "$STFLOW_OUTPUT"

if [ -f "$FVD_OUTPUT" ]; then
    echo
    echo "FVD:"
    echo "$FVD_OUTPUT"
fi


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


if st_path.is_file():

    st = json.load(
        st_path.open(
            "r",
            encoding="utf-8",
        )
    )

    mean = st.get(
        "mean",
        {}
    )

    print()
    print("=" * 100)
    print("ST-FLOW / TRAJ")
    print("=" * 100)

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
    ]:

        if key in mean:

            print(
                f"{key:26s} = {mean[key]}"
            )


if fvd_path.is_file():

    fv = json.load(
        fvd_path.open(
            "r",
            encoding="utf-8",
        )
    )

    print()
    print("=" * 100)
    print("FVD")
    print("=" * 100)

    print(
        "fvd         =",
        fv.get("fvd"),
    )

    print(
        "num_videos  =",
        fv.get("num_videos"),
    )

    print(
        "num_samples =",
        fv.get("num_samples"),
    )

    print(
        "cameras     =",
        fv.get("camera_names"),
    )

PY
