
# ============================================================
# 0. 新区域环境
# ============================================================

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


# ============================================================
# 1. 代码路径
# ============================================================

export CAMSIM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim
export OPENDWM_ROOT=$CAMSIM_ROOT/OpenDWM

cd "$OPENDWM_ROOT/src" || {
    echo "ERROR: OpenDWM src not found:"
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
# 2. ST-Flow 权重
# ============================================================

PRETRAIN_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain
CKPT_ROOT=$PRETRAIN_ROOT/ckpt

RAFT_CHECKPOINT=$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth
LOFTR_CHECKPOINT=$CKPT_ROOT/loftr_outdoor.ckpt

if [ ! -f "$RAFT_CHECKPOINT" ]; then
    echo "ERROR: RAFT checkpoint missing:"
    echo "$RAFT_CHECKPOINT"
    exit 1
fi

if [ ! -f "$LOFTR_CHECKPOINT" ]; then
    echo "ERROR: LoFTR checkpoint missing:"
    echo "$LOFTR_CHECKPOINT"
    exit 1
fi


# ============================================================
# 3. 安装到 Torch Hub cache
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

ls -lh "$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
ls -lh "$CACHE_DIR/loftr_outdoor.ckpt"


# ============================================================
# 4. Fixed-K DWM 数据
# ============================================================

ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/lyh_output/eval/nuscenesablationnew/nuplan6hz_merged300

MANIFEST=$ROOT/dwmnuplanstflow_manifest_fixedK.jsonl

OUTPUT=$ROOT/stflow_traj_result_gate16_fixedK.json

MAX_VIDEOS=300
GATE=16


echo
echo "============================================================"
echo "DWM fixed-K ST-Flow evaluation"
echo "============================================================"

echo "ROOT:"
echo "$ROOT"

echo
echo "MANIFEST:"
echo "$MANIFEST"

echo
echo "OUTPUT:"
echo "$OUTPUT"

echo
echo "MAX VIDEOS:"
echo "$MAX_VIDEOS"


if [ ! -f "$MANIFEST" ]; then
    echo
    echo "ERROR: fixed-K manifest missing:"
    echo "$MANIFEST"
    exit 1
fi


# ============================================================
# 5. Manifest sanity check
#
# 只检查：
#   - 300 videos
#   - 19 frames
#   - 8 cameras
#   - image_path 可访问
#   - 实际图片分辨率
#
# 不修改任何东西。
# ============================================================

python - <<PY
import json
from collections import Counter
from pathlib import Path
from PIL import Image

manifest = Path("$MANIFEST")

videos = []

with manifest.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            videos.append(json.loads(line))

print()
print("=" * 80)
print("Manifest sanity check")
print("=" * 80)

print("videos =", len(videos))

frame_counts = Counter()
camera_orders = Counter()
sizes = Counter()

total_images = 0
missing_images = 0
missing_examples = []

for video in videos:

    frames = video["frames"]
    frame_counts[len(frames)] += 1

    if frames:
        order = tuple(
            v.get("camera")
            for v in frames[0]["views"]
        )
        camera_orders[order] += 1

    for frame in frames:
        for view in frame["views"]:

            total_images += 1

            p = Path(view["image_path"])

            # Manifest 中 image_path 可以是相对路径。
            # ST-Flow evaluator 本身也是相对于 manifest 所在目录解析。
            if not p.is_absolute():
                p = manifest.parent / p

            if not p.is_file():
                missing_images += 1

                if len(missing_examples) < 5:
                    missing_examples.append(str(p))

                continue

            # 不需要把 45600 张全打开。
            if sum(sizes.values()) < 100:
                with Image.open(p) as im:
                    sizes[im.size] += 1


print("frame_counts =", dict(frame_counts))

print("camera_orders:")
for order, n in camera_orders.items():
    print(" ", n, "videos ->", list(order))

print("total image refs =", total_images)
print("missing images   =", missing_images)

print("sample sizes:")
for size, n in sizes.items():
    print(f"  {size[0]}x{size[1]} -> {n}")

if missing_examples:
    print()
    print("missing examples:")
    for x in missing_examples:
        print(" ", x)


if len(videos) != 300:
    raise RuntimeError(
        f"Expected 300 videos, got {len(videos)}"
    )

if set(frame_counts) != {19}:
    raise RuntimeError(
        f"Unexpected frame counts: {frame_counts}"
    )

if missing_images != 0:
    raise RuntimeError(
        f"{missing_images} image paths are missing"
    )

print()
print("Manifest sanity check: PASS")
PY


# ============================================================
# 6. 重新评测 DWM
#
# 和原来的 ST-Flow 协议保持一致。
#
# 注意：
#   不传 --startframe
#   不做 resize
#   不重新 merge
#   不跑 FVD
#
# 唯一变化就是 manifest 中的 K。
# ============================================================

echo
echo "============================================================"
echo "Run ST-Flow with fixed K"
echo "============================================================"

python -m dwm.tools.evaluate_stflow \
    --manifest "$MANIFEST" \
    --output "$OUTPUT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --frame-stride 2 \
    --min-matches 16 \
    --max-matches 256 \
    --loftr-confidence 0.1 \
    --pair-policy dataset \
    --cross-gate-px "$GATE"


# ============================================================
# 7. 输出结果摘要
# ============================================================

echo
echo "============================================================"
echo "Fixed-K evaluation finished"
echo "============================================================"

echo "Result:"
echo "$OUTPUT"

python - <<PY
import json
from pathlib import Path

p = Path("$OUTPUT")

with p.open("r", encoding="utf-8") as f:
    x = json.load(f)

print()
print("=" * 80)
print("FIXED-K RESULT")
print("=" * 80)

print("num_videos =", x.get("num_videos"))

mean = x.get("mean", {})

keys = [
    "temporal_l1",
    "cross_raw_epi_px",
    "cross_epi_px",
    "cross_inlier_ratio",
    "cycle_epi_px",
    "traj_epi_px",
    "traj_inlier2",
    "traj_inlier4",
    "stflow_error",
    "stflow_score",
    "stflow_d_error",
    "stflow_d_score",
    "stflow_c_score",
    "edge_coverage",
    "cycle_coverage",
]

for k in keys:
    if k in mean:
        print(f"{k:24s} = {mean[k]}")
PY
