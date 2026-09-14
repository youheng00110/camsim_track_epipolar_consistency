#!/bin/bash

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

set -uo pipefail


# ============================================================
# 0. 环境
# ============================================================

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
# 1. 权重
# ============================================================

CKPT_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt

RAFT_CHECKPOINT=$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth
LOFTR_CHECKPOINT=$CKPT_ROOT/loftr_outdoor.ckpt

if [ ! -f "$RAFT_CHECKPOINT" ]; then
    echo "RAFT missing:"
    echo "$RAFT_CHECKPOINT"
    exit 1
fi

if [ ! -f "$LOFTR_CHECKPOINT" ]; then
    echo "LoFTR missing:"
    echo "$LOFTR_CHECKPOINT"
    exit 1
fi


# ------------------------------------------------------------
# 自动寻找 I3D
# ------------------------------------------------------------

I3D_CHECKPOINT=""

if [ -f "$CKPT_ROOT/i3d_pretrained_400.pt" ]; then
    I3D_CHECKPOINT=$CKPT_ROOT/i3d_pretrained_400.pt
else
    I3D_CHECKPOINT=$(
        find "$CKPT_ROOT" \
            -maxdepth 2 \
            -type f \
            \( -iname '*i3d*.pt' -o -iname '*i3d*.pth' \) \
            2>/dev/null \
            | head -n 1
    )
fi

if [ -z "$I3D_CHECKPOINT" ] || [ ! -f "$I3D_CHECKPOINT" ]; then
    echo "I3D checkpoint not found under:"
    echo "$CKPT_ROOT"
    exit 1
fi


# ============================================================
# 2. Torch cache
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
echo "Checkpoints"
echo "============================================================"
echo "RAFT  : $RAFT_CHECKPOINT"
echo "LoFTR : $LOFTR_CHECKPOINT"
echo "I3D   : $I3D_CHECKPOINT"


# ============================================================
# 3. 数据路径
# ============================================================

BASE=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/lyh_output/eval/nuscenesablationnew

SRC=$BASE/nuplan6hz
ROOT=$BASE/nuplan6hz_merged300

MANIFEST=$ROOT/stflow_manifest.jsonl
FRONT_MANIFEST=$ROOT/stflow_manifest_front3.jsonl
FRONT_CONFIG=$ROOT/front3_eval_config.json

MAX_VIDEOS=300
GATE=16


# ============================================================
# 4. 检查 rank
# ============================================================

echo
echo "============================================================"
echo "Rank manifests"
echo "============================================================"

TOTAL_COUNT=0
RANK_COUNT=0

while IFS= read -r RANK_MANIFEST
do
    COUNT=$(grep -cve '^[[:space:]]*$' "$RANK_MANIFEST")

    echo "$(basename "$(dirname "$RANK_MANIFEST")") -> $COUNT videos"

    TOTAL_COUNT=$((TOTAL_COUNT + COUNT))
    RANK_COUNT=$((RANK_COUNT + 1))

done < <(
    find "$SRC" \
        -mindepth 2 \
        -maxdepth 2 \
        -type f \
        -path '*/rank_*/stflow_manifest.jsonl' \
        | sort
)

echo
echo "ranks=$RANK_COUNT"
echo "total=$TOTAL_COUNT"

if [ "$RANK_COUNT" -eq 0 ]; then
    echo "No rank manifests found."
    exit 1
fi

if [ "$TOTAL_COUNT" -lt "$MAX_VIDEOS" ]; then
    echo "Only $TOTAL_COUNT videos, need 300."
    exit 1
fi


# ============================================================
# 5. 合并前 300
# ============================================================

NEED_MERGE=1

if [ -f "$MANIFEST" ]; then

    EXISTING_COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST")

    echo
    echo "Existing merged count: $EXISTING_COUNT"

    if [ "$EXISTING_COUNT" -eq "$MAX_VIDEOS" ]; then
        NEED_MERGE=0
        echo "Existing merged300 valid. Skip merge."
    fi
fi


if [ "$NEED_MERGE" -eq 1 ]; then

    echo
    echo "============================================================"
    echo "Merge -> first 300"
    echo "============================================================"

    if ! python -m dwm.tools.merge_rank_preview_manifests_interleave \
        --input-root "$SRC" \
        --output-root "$ROOT" \
        --dataset-name nuplan \
        --max-videos "$MAX_VIDEOS" \
        --overwrite
    then

        echo "Hard-link failed. Retry --copy."

        python -m dwm.tools.merge_rank_preview_manifests_interleave \
            --input-root "$SRC" \
            --output-root "$ROOT" \
            --dataset-name nuplan \
            --max-videos "$MAX_VIDEOS" \
            --overwrite \
            --copy

        MERGE_EXIT=$?

        if [ "$MERGE_EXIT" -ne 0 ]; then
            exit "$MERGE_EXIT"
        fi
    fi
fi


# ============================================================
# 6. 检查 merged300
# ============================================================

if [ ! -f "$MANIFEST" ]; then
    echo "Merged manifest missing:"
    echo "$MANIFEST"
    exit 1
fi

VIDEO_COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST")

if [ "$VIDEO_COUNT" -ne "$MAX_VIDEOS" ]; then
    echo "Expected 300 videos, got $VIDEO_COUNT"
    exit 1
fi


# ============================================================
# 7. 生成 Front3 manifest
#
# 原始顺序：
#
# 0 CAM_L2
# 1 CAM_L1
# 2 CAM_L0   <- 左前
# 3 CAM_F0   <- 正前
# 4 CAM_R0   <- 右前
# 5 CAM_R1
# 6 CAM_R2
# 7 CAM_B0
#
# 只保留 index 2,3,4
# ============================================================

echo
echo "============================================================"
echo "Create Front3 manifest"
echo "Selected: CAM_L0 / CAM_F0 / CAM_R0"
echo "Indices : 2 / 3 / 4"
echo "============================================================"


python - \
    "$MANIFEST" \
    "$FRONT_MANIFEST" \
    "$FRONT_CONFIG" <<'PY'

import copy
import json
import sys
from collections import Counter


source_path = sys.argv[1]
output_path = sys.argv[2]
config_path = sys.argv[3]


SELECTED_INDICES = [2, 3, 4]

EXPECTED_NAMES = [
    "CAM_L0",
    "CAM_F0",
    "CAM_R0",
]


items = []

with open(
    source_path,
    "r",
    encoding="utf-8",
) as f:

    for line_number, line in enumerate(
        f,
        start=1,
    ):
        if not line.strip():
            continue

        try:
            items.append(json.loads(line))
        except Exception as e:
            raise RuntimeError(
                f"Invalid JSON line {line_number}: {e}"
            )


if len(items) != 300:
    raise RuntimeError(
        f"Expected 300 videos, got {len(items)}"
    )


output_items = []

frame_counts = Counter()
original_camera_counts = Counter()
front_camera_counts = Counter()

first_camera_names = None
selected_source_names = None


for video_index, item in enumerate(items):

    new_item = copy.deepcopy(item)

    frames = new_item.get("frames", [])

    if not frames:
        raise RuntimeError(
            f"video {video_index} has no frames"
        )

    frame_counts[len(frames)] += 1


    for frame_index, frame in enumerate(frames):

        views = frame.get("views", [])

        original_camera_counts[len(views)] += 1


        if len(views) < 5:
            raise RuntimeError(
                f"video={video_index}, "
                f"frame={frame_index}: "
                f"only {len(views)} views"
            )


        names = [
            view.get("camera")
            for view in views
        ]


        if first_camera_names is None:
            first_camera_names = names

            selected_source_names = [
                names[i]
                for i in SELECTED_INDICES
            ]


        # ----------------------------------------------------
        # 强检查：
        # 如果 manifest 已经明确写 CAM_L0/F0/R0，
        # 确认索引 2/3/4 没有错。
        # ----------------------------------------------------

        selected_names = [
            names[i]
            for i in SELECTED_INDICES
        ]

        if all(name is not None for name in selected_names):

            if selected_names != EXPECTED_NAMES:

                print(
                    "WARNING: selected camera names "
                    "do not match expected semantic names.",
                    file=sys.stderr,
                )

                print(
                    "Full camera order:",
                    names,
                    file=sys.stderr,
                )

                print(
                    "Selected:",
                    selected_names,
                    file=sys.stderr,
                )

                print(
                    "Expected:",
                    EXPECTED_NAMES,
                    file=sys.stderr,
                )

                raise RuntimeError(
                    "Camera order mismatch. "
                    "Stop before evaluation."
                )


        selected_views = [
            copy.deepcopy(
                views[index]
            )
            for index in SELECTED_INDICES
        ]


        # 强制规范 camera name，
        # 后面 camera-pairs 可以直接匹配。
        for view, name, source_index in zip(
            selected_views,
            EXPECTED_NAMES,
            SELECTED_INDICES,
        ):
            view["source_view_index"] = source_index
            view["source_camera_name"] = view.get(
                "camera"
            )
            view["camera"] = name


        frame["views"] = selected_views

        front_camera_counts[
            len(selected_views)
        ] += 1


    output_items.append(new_item)


with open(
    output_path,
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


config = {

    "num_videos": len(output_items),

    "selected_view_indices": [
        2,
        3,
        4,
    ],

    "camera_names": [
        "CAM_L0",
        "CAM_F0",
        "CAM_R0",
    ],

    "camera_pairs": [
        "CAM_L0__CAM_F0",
        "CAM_F0__CAM_R0",
    ],

    "camera_pairs_csv": (
        "CAM_L0__CAM_F0,"
        "CAM_F0__CAM_R0"
    ),

    "first_original_camera_names": (
        first_camera_names
    ),

    "selected_source_names": (
        selected_source_names
    ),

    "frame_count_distribution": {
        str(k): v
        for k, v in frame_counts.items()
    },

    "original_camera_count_distribution": {
        str(k): v
        for k, v in original_camera_counts.items()
    },

    "front_camera_count_distribution": {
        str(k): v
        for k, v in front_camera_counts.items()
    },
}


with open(
    config_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print("=" * 80)

print(
    "Original camera order:",
    first_camera_names,
)

print(
    "Selected indices:",
    SELECTED_INDICES,
)

print(
    "Selected cameras:",
    EXPECTED_NAMES,
)

print(
    "Pairs:",
    config["camera_pairs"],
)

print(
    "Videos:",
    len(output_items),
)

print(
    "Frame distribution:",
    dict(frame_counts),
)

print("=" * 80)

PY


FILTER_EXIT=$?

if [ "$FILTER_EXIT" -ne 0 ]; then
    echo "Front3 manifest generation failed."
    exit "$FILTER_EXIT"
fi


# ============================================================
# 8. Front3 检查
# ============================================================

FRONT_COUNT=$(grep -cve '^[[:space:]]*$' "$FRONT_MANIFEST")

echo
echo "Front3 videos: $FRONT_COUNT"

if [ "$FRONT_COUNT" -ne "$MAX_VIDEOS" ]; then
    echo "Front3 manifest count != 300"
    exit 1
fi


echo
echo "============================================================"
echo "Front3 config"
echo "============================================================"

cat "$FRONT_CONFIG"


# ============================================================
# 9. 获取 sequence count
# ============================================================

SEQ_COUNT=$(
python - "$FRONT_MANIFEST" <<'PY'

import json
import sys
from collections import Counter


path = sys.argv[1]

counts = []
view_counts = []
camera_orders = Counter()


with open(
    path,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        if not line.strip():
            continue

        item = json.loads(line)

        frames = item["frames"]

        counts.append(
            len(frames)
        )

        if frames:

            views = frames[0]["views"]

            view_counts.append(
                len(views)
            )

            camera_orders[
                tuple(
                    v["camera"]
                    for v in views
                )
            ] += 1


frame_dist = Counter(counts)
view_dist = Counter(view_counts)


print(
    "frame-count distribution:",
    dict(frame_dist),
    file=sys.stderr,
)

print(
    "view-count distribution:",
    dict(view_dist),
    file=sys.stderr,
)

print(
    "camera orders:",
    dict(camera_orders),
    file=sys.stderr,
)


if len(frame_dist) != 1:
    raise RuntimeError(
        f"Inconsistent frames: {dict(frame_dist)}"
    )

if set(view_counts) != {3}:
    raise RuntimeError(
        f"Expected exactly 3 cameras: {dict(view_dist)}"
    )


expected = (
    "CAM_L0",
    "CAM_F0",
    "CAM_R0",
)

if set(camera_orders) != {expected}:
    raise RuntimeError(
        f"Unexpected camera order: {dict(camera_orders)}"
    )


print(counts[0])

PY
)


SEQ_EXIT=$?

if [ "$SEQ_EXIT" -ne 0 ]; then
    exit "$SEQ_EXIT"
fi


echo
echo "sequence_count=$SEQ_COUNT"


# ============================================================
# 10. ST-Flow / Traj
#
# Temporal:
#   CAM_L0
#   CAM_F0
#   CAM_R0
#
# Traj:
#   CAM_L0
#   CAM_F0
#   CAM_R0
#
# Cross:
#   CAM_L0 -> CAM_F0
#   CAM_F0 -> CAM_R0
#
# 不闭环
#
# 不手动设置 startframe
# ============================================================

mkdir -p "$ROOT/eval_logs"


CAMERA_PAIRS="CAM_L0__CAM_F0,CAM_F0__CAM_R0"

STFLOW_OUTPUT=$ROOT/stflow_traj_result_gate16_front3.json
STFLOW_LOG=$ROOT/eval_logs/stflow_gate16_front3.log


echo
echo "============================================================"
echo "Run Front3 ST-Flow / Traj"
echo "============================================================"

echo "videos      : $MAX_VIDEOS"
echo "cameras     : CAM_L0 CAM_F0 CAM_R0"
echo "indices     : 2 3 4"
echo "pairs       : $CAMERA_PAIRS"
echo "gate        : 16"
echo "stride      : 2"
echo "start frame : AUTO"


python -m dwm.tools.evaluate_stflow \
    --manifest "$FRONT_MANIFEST" \
    --output "$STFLOW_OUTPUT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --frame-stride 2 \
    --min-matches 16 \
    --max-matches 256 \
    --loftr-confidence 0.1 \
    --camera-pairs "$CAMERA_PAIRS" \
    --pair-policy dataset \
    --cross-gate-px "$GATE" \
    2>&1 | tee "$STFLOW_LOG"


STFLOW_EXIT=${PIPESTATUS[0]}

if [ "$STFLOW_EXIT" -ne 0 ]; then
    echo "ST-Flow failed: $STFLOW_EXIT"
    exit "$STFLOW_EXIT"
fi


# ============================================================
# 11. FVD
#
# 用过滤后的 manifest，
# 所以 FVD 也只有三个前向摄像头。
# ============================================================

FVD_OUTPUT=$ROOT/paired_fvd_result_front3_all${SEQ_COUNT}.json
FVD_LOG=$ROOT/eval_logs/fvd_front3_all${SEQ_COUNT}.log


echo
echo "============================================================"
echo "Run Front3 FVD"
echo "============================================================"

echo "videos         : $MAX_VIDEOS"
echo "sequence count : $SEQ_COUNT"


python -m dwm.tools.evaluate_fvd_from_paired_manifest \
    --manifest "$FRONT_MANIFEST" \
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
        --manifest "$FRONT_MANIFEST" \
        --output "$FVD_OUTPUT" \
        --i3d-checkpoint "$I3D_CHECKPOINT" \
        --device cuda \
        --max-videos "$MAX_VIDEOS" \
        --sequence-count "$SEQ_COUNT" \
        --batch-size 1 \
        2>&1 | tee -a "$FVD_LOG"


    FVD_EXIT=${PIPESTATUS[0]}


    if [ "$FVD_EXIT" -ne 0 ]; then
        echo "FVD failed: $FVD_EXIT"
        exit "$FVD_EXIT"
    fi
fi


# ============================================================
# 12. 打印结果
# ============================================================

python - \
    "$STFLOW_OUTPUT" \
    "$FVD_OUTPUT" <<'PY'

import json
import math
import sys


with open(
    sys.argv[1],
    "r",
    encoding="utf-8",
) as f:
    st = json.load(f)


with open(
    sys.argv[2],
    "r",
    encoding="utf-8",
) as f:
    fd = json.load(f)


m = st.get("mean", {})


def fmt(x, n):

    if not isinstance(
        x,
        (int, float),
    ):
        return "-"

    if not math.isfinite(float(x)):
        return "-"

    return f"{float(x):.{n}f}"


temporal = m.get("temporal_l1")
cross_raw = m.get("cross_raw_epi_px")
cross_inlier = m.get("cross_inlier_ratio")

traj = m.get("traj_epi_px")
traj_i2 = m.get("traj_inlier2")

score = m.get("stflow_score")
dscore = m.get("stflow_d_score")

coverage = m.get("edge_coverage")

cscore = None

if (
    isinstance(score, (int, float))
    and isinstance(coverage, (int, float))
    and isinstance(cross_inlier, (int, float))
):
    cscore = (
        score
        * coverage
        * cross_inlier
    )


fvd = fd.get("fvd")


print()
print("=" * 130)

print(
    "nuPlan6Hz | Front3 | 300 videos"
)

print("=" * 130)


print(
    "| Temp-L1 ↓ | Cross-Raw ↓ | "
    "Cross-Inlier ↑ | Traj-Epi ↓ | "
    "Traj-Inlier@2 ↑ | STFlow ↑ | "
    "STFlow-D ↑ | STFlow-C ↑ | FVD ↓ |"
)


print(
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)


print(
    f"| {fmt(temporal,6)} "
    f"| {fmt(cross_raw,3)} "
    f"| {fmt(cross_inlier,4)} "
    f"| {fmt(traj,3)} "
    f"| {fmt(traj_i2,4)} "
    f"| {fmt(score,3)} "
    f"| {fmt(dscore,3)} "
    f"| {fmt(cscore,3)} "
    f"| {fmt(fvd,3)} |"
)


print("=" * 130)

PY


echo
echo "============================================================"
echo "DONE"
echo "============================================================"

echo
echo "Merged:"
echo "$ROOT"

echo
echo "Front3 manifest:"
echo "$FRONT_MANIFEST"

echo
echo "ST-Flow:"
echo "$STFLOW_OUTPUT"

echo
echo "FVD:"
echo "$FVD_OUTPUT"

