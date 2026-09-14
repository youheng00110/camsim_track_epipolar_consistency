#!/bin/bash

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

set -uo pipefail

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
# 权重
# ============================================================

CKPT_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt

RAFT_CHECKPOINT=$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth
LOFTR_CHECKPOINT=$CKPT_ROOT/loftr_outdoor.ckpt
I3D_CHECKPOINT=$CKPT_ROOT/i3d_pretrained_400.pt

for P in \
    "$RAFT_CHECKPOINT" \
    "$LOFTR_CHECKPOINT" \
    "$I3D_CHECKPOINT"
do
    if [ ! -f "$P" ]; then
        echo "Missing checkpoint: $P"
        exit 1
    fi
done


export TORCH_HOME=/root/.cache/torch
CACHE_DIR=$TORCH_HOME/hub/checkpoints

mkdir -p "$CACHE_DIR"

cp -f \
    "$RAFT_CHECKPOINT" \
    "$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

cp -f \
    "$LOFTR_CHECKPOINT" \
    "$CACHE_DIR/loftr_outdoor.ckpt"


# ============================================================
# 路径
# ============================================================

BASE=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/lyh_output/eval/nuscenesablationnew

ROOT=$BASE/nuplan6hz_merged300

MANIFEST=$ROOT/stflow_manifest.jsonl
FRONT_MANIFEST=$ROOT/stflow_manifest_front3.jsonl
FRONT_CONFIG=$ROOT/front3_eval_config.json

MAX_VIDEOS=300
GATE=16


# ============================================================
# 1. 检查 merged300
# ============================================================

if [ ! -f "$MANIFEST" ]; then
    echo "Manifest missing:"
    echo "$MANIFEST"
    exit 1
fi

COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST")

echo "merged videos = $COUNT"

if [ "$COUNT" -ne 300 ]; then
    echo "Expected 300 videos."
    exit 1
fi


# ============================================================
# 2. 生成 Front3 manifest
#
# 原始语义顺序：
#
# 0 CAM_L2
# 1 CAM_L1
# 2 CAM_L0   <- LEFT FRONT
# 3 CAM_F0   <- FRONT
# 4 CAM_R0   <- RIGHT FRONT
# 5 CAM_R1
# 6 CAM_R2
# 7 CAM_B0
#
# merge 后可能变为：
#
# CAM_00 ... CAM_07
#
# 位置顺序没有改变。
#
# 因此仍然选择：
#
# 2,3,4
# ============================================================

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

TARGET_NAMES = [
    "CAM_L0",
    "CAM_F0",
    "CAM_R0",
]

SEMANTIC_FULL_ORDER = [
    "CAM_L2",
    "CAM_L1",
    "CAM_L0",
    "CAM_F0",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
]

GENERIC_FULL_ORDER = [
    "CAM_00",
    "CAM_01",
    "CAM_02",
    "CAM_03",
    "CAM_04",
    "CAM_05",
    "CAM_06",
    "CAM_07",
]


items = []

with open(
    source_path,
    "r",
    encoding="utf-8",
) as f:

    for line_number, line in enumerate(f, 1):

        if not line.strip():
            continue

        try:
            items.append(json.loads(line))
        except Exception as exc:
            raise RuntimeError(
                f"Invalid JSON line {line_number}: {exc}"
            )


if len(items) != 300:
    raise RuntimeError(
        f"Expected 300 videos, got {len(items)}"
    )


output_items = []

frame_counts = Counter()
source_orders = Counter()


for video_index, item in enumerate(items):

    new_item = copy.deepcopy(item)

    frames = new_item.get("frames", [])

    if not frames:
        raise RuntimeError(
            f"video={video_index} has no frames"
        )

    frame_counts[len(frames)] += 1


    for frame_index, frame in enumerate(frames):

        views = frame.get("views", [])

        if len(views) != 8:
            raise RuntimeError(
                f"Expected 8 cameras, got {len(views)} "
                f"at video={video_index}, frame={frame_index}"
            )


        names = [
            view.get("camera")
            for view in views
        ]

        source_orders[tuple(names)] += 1


        # ----------------------------------------------------
        # 两种合法情况：
        #
        # 1. 原始语义名称
        # 2. merge 后的 generic 名称
        # ----------------------------------------------------

        if names == SEMANTIC_FULL_ORDER:

            mapping_type = "semantic"

        elif names == GENERIC_FULL_ORDER:

            mapping_type = "generic"

        else:

            raise RuntimeError(
                "\nUnexpected camera order.\n"
                f"video={video_index}, frame={frame_index}\n"
                f"actual={names}\n"
                f"expected semantic={SEMANTIC_FULL_ORDER}\n"
                f"or generic={GENERIC_FULL_ORDER}"
            )


        selected_views = []

        for source_index, target_name in zip(
            SELECTED_INDICES,
            TARGET_NAMES,
        ):

            view = copy.deepcopy(
                views[source_index]
            )

            view["source_view_index"] = source_index
            view["source_camera_name"] = view.get(
                "camera"
            )

            # 统一恢复成 nuPlan 语义名称
            view["camera"] = target_name

            selected_views.append(view)


        frame["views"] = selected_views


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
    "num_videos": 300,

    "original_semantic_order": (
        SEMANTIC_FULL_ORDER
    ),

    "accepted_generic_order": (
        GENERIC_FULL_ORDER
    ),

    "selected_indices": [
        2,
        3,
        4,
    ],

    "selected_semantic_cameras": [
        "CAM_L0",
        "CAM_F0",
        "CAM_R0",
    ],

    "camera_pairs": [
        "CAM_L0__CAM_F0",
        "CAM_F0__CAM_R0",
    ],

    "frame_count_distribution": {
        str(k): v
        for k, v in frame_counts.items()
    },

    "observed_source_orders": {
        str(k): v
        for k, v in source_orders.items()
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
print("Front3 manifest successfully generated")
print("=" * 80)

print(
    "Selected source indices:",
    SELECTED_INDICES,
)

print(
    "Semantic cameras:",
    TARGET_NAMES,
)

print(
    "Cross pairs:",
    config["camera_pairs"],
)

print(
    "Frame counts:",
    dict(frame_counts),
)

print()
print("Observed source camera orders:")

for order, count in source_orders.items():
    print(
        f"  {list(order)} -> {count} frames"
    )

print()
print("Output:")
print(output_path)

PY


FILTER_EXIT=$?

if [ "$FILTER_EXIT" -ne 0 ]; then
    echo "Front3 conversion failed."
    exit "$FILTER_EXIT"
fi


# ============================================================
# 3. 严格验证过滤结果
# ============================================================

SEQ_COUNT=$(
python - "$FRONT_MANIFEST" <<'PY'

import json
import sys
from collections import Counter


path = sys.argv[1]

video_count = 0
frame_counts = Counter()
view_counts = Counter()
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

        video_count += 1

        frames = item["frames"]

        frame_counts[len(frames)] += 1


        for frame in frames:

            views = frame["views"]

            view_counts[len(views)] += 1

            camera_orders[
                tuple(
                    v["camera"]
                    for v in views
                )
            ] += 1


print(
    "video count:",
    video_count,
    file=sys.stderr,
)

print(
    "frame counts:",
    dict(frame_counts),
    file=sys.stderr,
)

print(
    "view counts:",
    dict(view_counts),
    file=sys.stderr,
)

print(
    "camera orders:",
    dict(camera_orders),
    file=sys.stderr,
)


if video_count != 300:
    raise RuntimeError(
        f"Expected 300 videos, got {video_count}"
    )


if len(frame_counts) != 1:
    raise RuntimeError(
        f"Inconsistent frame counts: {dict(frame_counts)}"
    )


if set(view_counts) != {3}:
    raise RuntimeError(
        f"Expected exactly 3 cameras: {dict(view_counts)}"
    )


expected_order = (
    "CAM_L0",
    "CAM_F0",
    "CAM_R0",
)


if set(camera_orders) != {expected_order}:
    raise RuntimeError(
        f"Wrong Front3 camera order: {dict(camera_orders)}"
    )


print(
    next(iter(frame_counts))
)

PY
)


CHECK_EXIT=$?

if [ "$CHECK_EXIT" -ne 0 ]; then
    echo "Front3 validation failed."
    exit "$CHECK_EXIT"
fi


echo
echo "============================================================"
echo "Front3 ready"
echo "============================================================"

echo "videos      : 300"
echo "frames      : $SEQ_COUNT"
echo "views       : 3"
echo "camera order: CAM_L0 CAM_F0 CAM_R0"
echo "pairs       : CAM_L0__CAM_F0,CAM_F0__CAM_R0"


# ============================================================
# 4. ST-Flow / Traj
#
# 由于 manifest 已经只有三摄：
#
# Temporal:
#   L0 / F0 / R0
#
# Traj:
#   L0 / F0 / R0
#
# Cross:
#   L0 -> F0
#   F0 -> R0
#
# 不测 R0 -> L0
# ============================================================

mkdir -p "$ROOT/eval_logs"


CAMERA_PAIRS="CAM_L0__CAM_F0,CAM_F0__CAM_R0"

STFLOW_OUTPUT=$ROOT/stflow_traj_result_gate16_front3.json
STFLOW_LOG=$ROOT/eval_logs/stflow_gate16_front3.log


echo
echo "============================================================"
echo "Run ST-Flow / Traj"
echo "============================================================"

echo "videos     : 300"
echo "cameras    : CAM_L0 CAM_F0 CAM_R0"
echo "pairs      : $CAMERA_PAIRS"
echo "gate       : 16"
echo "stride     : 2"
echo "startframe : AUTO"


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
    echo "ST-Flow failed."
    exit "$STFLOW_EXIT"
fi


# ============================================================
# 5. FVD
#
# 同一个 Front3 manifest
# 因此 FVD 也是 3 摄
# ============================================================

FVD_OUTPUT=$ROOT/paired_fvd_result_front3_all${SEQ_COUNT}.json
FVD_LOG=$ROOT/eval_logs/fvd_front3_all${SEQ_COUNT}.log


echo
echo "============================================================"
echo "Run FVD"
echo "============================================================"


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
        echo "FVD failed."
        exit "$FVD_EXIT"
    fi
fi


echo
echo "============================================================"
echo "DONE"
echo "============================================================"

echo
echo "Front3 manifest:"
echo "$FRONT_MANIFEST"

echo
echo "ST-Flow:"
echo "$STFLOW_OUTPUT"

echo
echo "FVD:"
echo "$FVD_OUTPUT"

