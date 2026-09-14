#!/bin/bash

set -uo pipefail


# ============================================================
# 0. 新区域环境
# ============================================================

# 如果新区域仍然使用这个 conda 环境就保留。
# 若你之后换了环境，只需要改这一行。
source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate


export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


# ============================================================
# 1. 新区域代码路径
# ============================================================

export CAMSIM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim
export OPENDWM_ROOT=$CAMSIM_ROOT/OpenDWM

cd "$OPENDWM_ROOT/src" || {
    echo "OpenDWM src not found:"
    echo "$OPENDWM_ROOT/src"
    exit 1
}


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
# 2. 新区域权重
# ============================================================

PRETRAIN_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain
CKPT_ROOT=$PRETRAIN_ROOT/ckpt

RAFT_CHECKPOINT=$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth
LOFTR_CHECKPOINT=$CKPT_ROOT/loftr_outdoor.ckpt


if [ ! -f "$RAFT_CHECKPOINT" ]; then
    echo "RAFT checkpoint missing:"
    echo "$RAFT_CHECKPOINT"
    exit 1
fi

if [ ! -f "$LOFTR_CHECKPOINT" ]; then
    echo "LoFTR checkpoint missing:"
    echo "$LOFTR_CHECKPOINT"
    exit 1
fi


# ------------------------------------------------------------
# 在新区域自动找 I3D
# ------------------------------------------------------------

I3D_CHECKPOINT=""

if [ -f "$CKPT_ROOT/i3d_pretrained_400.pt" ]; then

    I3D_CHECKPOINT=$CKPT_ROOT/i3d_pretrained_400.pt

else

    I3D_CHECKPOINT=$(
        find "$CKPT_ROOT" \
            -maxdepth 2 \
            -type f \
            \( \
                -iname '*i3d*.pt' \
                -o -iname '*i3d*.pth' \
            \) \
            2>/dev/null \
            | head -n 1
    )

fi


if [ -z "$I3D_CHECKPOINT" ] || [ ! -f "$I3D_CHECKPOINT" ]; then

    echo
    echo "============================================================"
    echo "ERROR: I3D checkpoint not found in new area"
    echo "============================================================"

    echo "Searched under:"
    echo "$CKPT_ROOT"

    echo
    echo "Please put i3d_pretrained_400.pt under:"
    echo "$CKPT_ROOT"

    exit 1
fi


echo
echo "============================================================"
echo "Checkpoints"
echo "============================================================"

echo "RAFT:"
echo "$RAFT_CHECKPOINT"

echo
echo "LoFTR:"
echo "$LOFTR_CHECKPOINT"

echo
echo "I3D:"
echo "$I3D_CHECKPOINT"


# ============================================================
# 3. Torch Hub 本地缓存
#
# 当前 ST-Flow 内部：
# torchvision RAFT DEFAULT
# Kornia LoFTR outdoor
#
# 所以把新区域本地权重装进 cache。
# ============================================================

export TORCH_HOME=/root/.cache/torch

CACHE_DIR=$TORCH_HOME/hub/checkpoints

mkdir -p "$CACHE_DIR"

cp -f \
    "$RAFT_CHECKPOINT" \
    "$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

cp -f \
    "$LOFTR_CHECKPOINT" \
    "$CACHE_DIR/loftr_outdoor.ckpt"


echo
echo "============================================================"
echo "Torch cache"
echo "============================================================"

ls -lh \
    "$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

ls -lh \
    "$CACHE_DIR/loftr_outdoor.ckpt"


# ============================================================
# 4. 数据路径
# ============================================================

BASE=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/lyh_output/eval/nuscenesablationnew

SRC=$BASE/nuplan6hz
ROOT=$BASE/nuplan6hz_merged300

MANIFEST=$ROOT/stflow_manifest.jsonl

MAX_VIDEOS=300
GATE=16


echo
echo "============================================================"
echo "Evaluation"
echo "============================================================"

echo "SOURCE:"
echo "$SRC"

echo
echo "OUTPUT:"
echo "$ROOT"

echo
echo "MAX VIDEOS:"
echo "$MAX_VIDEOS"


if [ ! -d "$SRC" ]; then

    echo
    echo "Source directory missing:"
    echo "$SRC"

    exit 1
fi


# ============================================================
# 5. 查看 rank 情况
# ============================================================

echo
echo "============================================================"
echo "Rank manifests"
echo "============================================================"


RANK_COUNT=0
TOTAL_COUNT=0


while IFS= read -r RANK_MANIFEST
do

    RANK_DIR=$(dirname "$RANK_MANIFEST")
    RANK_NAME=$(basename "$RANK_DIR")

    COUNT=$(grep -cve '^[[:space:]]*$' "$RANK_MANIFEST")

    echo "$RANK_NAME : $COUNT videos"

    RANK_COUNT=$((RANK_COUNT + 1))
    TOTAL_COUNT=$((TOTAL_COUNT + COUNT))

done < <(
    find "$SRC" \
        -mindepth 2 \
        -maxdepth 2 \
        -type f \
        -path '*/rank_*/stflow_manifest.jsonl' \
        | sort
)


echo
echo "rank count:  $RANK_COUNT"
echo "total videos: $TOTAL_COUNT"


if [ "$RANK_COUNT" -eq 0 ]; then

    echo "No rank_*/stflow_manifest.jsonl found."
    exit 1

fi


if [ "$TOTAL_COUNT" -lt "$MAX_VIDEOS" ]; then

    echo
    echo "ERROR:"
    echo "Need at least $MAX_VIDEOS videos."
    echo "Only found $TOTAL_COUNT."

    exit 1
fi


# ============================================================
# 6. 检查 merged300
# ============================================================

NEED_MERGE=1


if [ -f "$MANIFEST" ]; then

    EXISTING_COUNT=$(
        grep -cve '^[[:space:]]*$' "$MANIFEST"
    )

    echo
    echo "Existing merged videos:"
    echo "$EXISTING_COUNT"


    if [ "$EXISTING_COUNT" -eq "$MAX_VIDEOS" ]; then

        NEED_MERGE=0

        echo "Existing merged300 is valid."
        echo "Skip merge."

    fi

fi


# ============================================================
# 7. 合并并取前 300
# ============================================================

if [ "$NEED_MERGE" -eq 1 ]; then

    echo
    echo "============================================================"
    echo "Merge rank outputs -> first 300"
    echo "============================================================"


    if ! python -m dwm.tools.merge_rank_preview_manifests_interleave \
        --input-root "$SRC" \
        --output-root "$ROOT" \
        --dataset-name nuplan \
        --max-videos "$MAX_VIDEOS" \
        --overwrite
    then

        echo
        echo "Hard-link merge failed."
        echo "Retry with --copy."


        python -m dwm.tools.merge_rank_preview_manifests_interleave \
            --input-root "$SRC" \
            --output-root "$ROOT" \
            --dataset-name nuplan \
            --max-videos "$MAX_VIDEOS" \
            --overwrite \
            --copy


        MERGE_EXIT=$?


        if [ "$MERGE_EXIT" -ne 0 ]; then

            echo
            echo "Merge failed."

            exit "$MERGE_EXIT"

        fi

    fi

fi


# ============================================================
# 8. 检查合并结果
# ============================================================

if [ ! -f "$MANIFEST" ]; then

    echo "Merged manifest missing:"
    echo "$MANIFEST"

    exit 1
fi


VIDEO_COUNT=$(
    grep -cve '^[[:space:]]*$' "$MANIFEST"
)


echo
echo "============================================================"
echo "Merged result"
echo "============================================================"

echo "videos=$VIDEO_COUNT"


if [ "$VIDEO_COUNT" -ne "$MAX_VIDEOS" ]; then

    echo
    echo "ERROR:"
    echo "Expected $MAX_VIDEOS"
    echo "Actual   $VIDEO_COUNT"

    exit 1
fi


# ============================================================
# 9. 检查帧数 + 获取 sequence count
# ============================================================

SEQ_COUNT=$(
python - "$MANIFEST" <<'PY'
import json
import sys
from collections import Counter

manifest_path = sys.argv[1]

counts = []
camera_counts = []

with open(
    manifest_path,
    "r",
    encoding="utf-8",
) as file:

    for line_number, line in enumerate(
        file,
        start=1,
    ):
        if not line.strip():
            continue

        item = json.loads(line)

        frames = item.get(
            "frames",
            [],
        )

        counts.append(
            len(frames)
        )

        if frames:
            camera_counts.append(
                len(
                    frames[0].get(
                        "views",
                        [],
                    )
                )
            )


frame_distribution = Counter(counts)
camera_distribution = Counter(camera_counts)


print(
    "frame-count distribution:",
    dict(frame_distribution),
    file=sys.stderr,
)

print(
    "camera-count distribution:",
    dict(camera_distribution),
    file=sys.stderr,
)


if len(counts) != 300:

    raise RuntimeError(
        f"Expected 300 videos, got {len(counts)}"
    )


if len(frame_distribution) != 1:

    raise RuntimeError(
        "Inconsistent frame count: "
        f"{dict(frame_distribution)}"
    )


print(
    counts[0]
)
PY
)


SEQ_EXIT=$?


if [ "$SEQ_EXIT" -ne 0 ]; then

    echo "Manifest validation failed."
    exit "$SEQ_EXIT"

fi


echo
echo "sequence_count=$SEQ_COUNT"


# ============================================================
# 10. ST-Flow / Traj
#
# nuPlan:
#   --pair-policy dataset
#
# gate:
#   16
#
# 不设置 --startframe
# ============================================================

mkdir -p "$ROOT/eval_logs"


STFLOW_OUTPUT=$ROOT/stflow_traj_result_gate16.json
STFLOW_LOG=$ROOT/eval_logs/stflow_gate16.log


echo
echo "============================================================"
echo "Run ST-Flow / Traj"
echo "============================================================"

echo "videos:       $MAX_VIDEOS"
echo "gate:         $GATE"
echo "frame stride: 2"
echo "pair policy:  dataset"
echo "startframe:   AUTO"


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

    echo
    echo "ST-Flow failed:"
    echo "$STFLOW_EXIT"

    exit "$STFLOW_EXIT"
fi


# ============================================================
# 11. FVD
# ============================================================

FVD_OUTPUT=$ROOT/paired_fvd_result_all${SEQ_COUNT}.json
FVD_LOG=$ROOT/eval_logs/fvd_all${SEQ_COUNT}.log


echo
echo "============================================================"
echo "Run FVD"
echo "============================================================"

echo "videos:         $MAX_VIDEOS"
echo "sequence-count: $SEQ_COUNT"
echo "I3D:            $I3D_CHECKPOINT"


python -m dwm.tools.evaluate_fvd_from_paired_manifest \
    --manifest "$MANIFEST" \
    --output "$FVD_OUTPUT" \
    --i3d-checkpoint "$I3D_CHECKPOINT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --sequence-count "$SEQ_COUNT" \
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
        --batch-size 1 \
        2>&1 | tee -a "$FVD_LOG"


    FVD_EXIT=${PIPESTATUS[0]}


    if [ "$FVD_EXIT" -ne 0 ]; then

        echo
        echo "FVD failed:"
        echo "$FVD_EXIT"

        exit "$FVD_EXIT"
    fi

fi


# ============================================================
# 12. 打印最终结果
# ============================================================

python - \
    "$STFLOW_OUTPUT" \
    "$FVD_OUTPUT" <<'PY'

import json
import math
import sys


stflow_path = sys.argv[1]
fvd_path = sys.argv[2]


with open(
    stflow_path,
    "r",
    encoding="utf-8",
) as file:
    stflow_data = json.load(file)


with open(
    fvd_path,
    "r",
    encoding="utf-8",
) as file:
    fvd_data = json.load(file)


mean = stflow_data.get(
    "mean",
    {},
)


def fmt(value, digits):

    if not isinstance(
        value,
        (int, float),
    ):
        return "-"

    if not math.isfinite(
        float(value)
    ):
        return "-"

    return (
        f"{float(value):.{digits}f}"
    )


temporal = mean.get(
    "temporal_l1"
)

cross_raw = mean.get(
    "cross_raw_epi_px"
)

cross_inlier = mean.get(
    "cross_inlier_ratio"
)

traj = mean.get(
    "traj_epi_px"
)

traj_inlier2 = mean.get(
    "traj_inlier2"
)

stflow = mean.get(
    "stflow_score"
)

stflow_d = mean.get(
    "stflow_d_score"
)

edge_coverage = mean.get(
    "edge_coverage"
)


stflow_c = None

if (
    isinstance(stflow, (int, float))
    and isinstance(edge_coverage, (int, float))
    and isinstance(cross_inlier, (int, float))
):
    stflow_c = (
        stflow
        * edge_coverage
        * cross_inlier
    )


fvd = fvd_data.get(
    "fvd"
)


print()
print("=" * 130)

print(
    "nuplan6hz | merged300 evaluation"
)

print("=" * 130)


print(
    "| Temporal-L1 ↓ | Cross-Raw-Epi ↓ | "
    "Cross-Inlier ↑ | Traj-Epi ↓ | "
    "Traj-Inlier@2 ↑ | ST-Flow ↑ | "
    "ST-Flow-D ↑ | ST-Flow-C ↑ | FVD ↓ |"
)


print(
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)


print(
    f"| {fmt(temporal, 6)} "
    f"| {fmt(cross_raw, 3)} "
    f"| {fmt(cross_inlier, 4)} "
    f"| {fmt(traj, 3)} "
    f"| {fmt(traj_inlier2, 4)} "
    f"| {fmt(stflow, 3)} "
    f"| {fmt(stflow_d, 3)} "
    f"| {fmt(stflow_c, 3)} "
    f"| {fmt(fvd, 3)} |"
)


print("=" * 130)

PY


echo
echo "============================================================"
echo "DONE"
echo "============================================================"

echo
echo "Merged300:"
echo "$ROOT"

echo
echo "ST-Flow:"
echo "$STFLOW_OUTPUT"

echo
echo "FVD:"
echo "$FVD_OUTPUT"

