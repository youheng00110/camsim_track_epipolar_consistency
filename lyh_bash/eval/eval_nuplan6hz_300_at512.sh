#!/bin/bash

set -euo pipefail


# ============================================================
# 0. 环境
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

cd "$OPENDWM_ROOT/src" || {
    echo "ERROR: OpenDWM src not found"
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
# 2. 使用已有 RAFT / LoFTR 权重
# ============================================================

PRETRAIN_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain
CKPT_ROOT=$PRETRAIN_ROOT/ckpt

RAFT_LOCAL=$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth
LOFTR_LOCAL=$CKPT_ROOT/loftr_outdoor.ckpt

if [ ! -f "$RAFT_LOCAL" ]; then
    echo "ERROR: RAFT checkpoint missing:"
    echo "$RAFT_LOCAL"
    exit 1
fi

if [ ! -f "$LOFTR_LOCAL" ]; then
    echo "ERROR: LoFTR checkpoint missing:"
    echo "$LOFTR_LOCAL"
    exit 1
fi

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
echo "Weights"
echo "============================================================"

ls -lh \
    "$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth" \
    "$CACHE_DIR/loftr_outdoor.ckpt"


# ============================================================
# 3. 输入 / 输出
# ============================================================

BASE=$CAMSIM_ROOT/lyh_output/eval/nuscenesablationnew

SRC=$BASE/nuplan6hz_merged300
DST=$BASE/nuplan6hz_merged300_stflow512

SRC_MANIFEST=$SRC/stflow_manifest.jsonl
DST_MANIFEST=$DST/stflow_manifest.jsonl

OUTPUT=$DST/stflow_traj_result_gate16.json

TARGET_W=512
TARGET_H=288
MAX_VIDEOS=300
GATE=16


if [ ! -f "$SRC_MANIFEST" ]; then
    echo "ERROR: source manifest missing:"
    echo "$SRC_MANIFEST"
    exit 1
fi

mkdir -p "$DST"


echo
echo "============================================================"
echo "Resolution normalization"
echo "============================================================"

echo "SOURCE:"
echo "$SRC"

echo
echo "TARGET:"
echo "$DST"

echo
echo "TARGET RESOLUTION:"
echo "${TARGET_W}x${TARGET_H}"


# ============================================================
# 4. Resize images + masks + update K
#
# 关键原则：
#
# image:
#   1920x1080 -> 512x288
#
# K:
#   first row * 512/1920
#   second row * 288/1080
#
# T_cam_to_ego:
#   不改
#
# T_ego_to_world:
#   不改
#
# camera order:
#   不改
#
# real_image_path:
#   从这个 ST-Flow-only manifest 删除，防止以后误拿去跑 FVD
# ============================================================

python - <<PY
import copy
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


SRC_ROOT = Path("$SRC")
DST_ROOT = Path("$DST")

SRC_MANIFEST = Path("$SRC_MANIFEST")
DST_MANIFEST = Path("$DST_MANIFEST")

TARGET_W = int("$TARGET_W")
TARGET_H = int("$TARGET_H")


def resolve_path(raw, manifest):
    p = Path(raw)

    if not p.is_absolute():
        p = manifest.parent / p

    return p


def load_jsonl(path):
    out = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))

    return out


def write_jsonl(path, items):
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


items = load_jsonl(SRC_MANIFEST)

print("videos =", len(items))

if len(items) != 300:
    raise RuntimeError(
        f"Expected 300 videos, got {len(items)}"
    )


# ------------------------------------------------------------
# Pillow interpolation
# ------------------------------------------------------------

try:
    RGB_RESAMPLE = Image.Resampling.BILINEAR
    MASK_RESAMPLE = Image.Resampling.NEAREST
except AttributeError:
    RGB_RESAMPLE = Image.BILINEAR
    MASK_RESAMPLE = Image.NEAREST


source_sizes = Counter()
target_sizes = Counter()

num_images = 0
num_masks = 0

max_k_check_error = 0.0


for vi, item in enumerate(items):

    frames = item.get("frames", [])

    if len(frames) != 19:
        raise RuntimeError(
            f"video {vi}: expected 19 frames, got {len(frames)}"
        )


    # ========================================================
    # 每个 frame / camera
    # ========================================================

    for fi, frame in enumerate(frames):

        views = frame.get("views", [])

        if len(views) != 8:
            raise RuntimeError(
                f"video={vi}, frame={fi}: "
                f"expected 8 views, got {len(views)}"
            )


        for ci, view in enumerate(views):

            camera = view["camera"]

            # ------------------------------------------------
            # 1. 原图
            # ------------------------------------------------

            src_image = resolve_path(
                view["image_path"],
                SRC_MANIFEST,
            )

            if not src_image.is_file():
                raise FileNotFoundError(src_image)


            with Image.open(src_image) as im:

                im = im.convert("RGB")

                src_w, src_h = im.size

                source_sizes[(src_w, src_h)] += 1

                # 这里原则上应该全是 1920x1080
                if (src_w, src_h) != (1920, 1080):
                    raise RuntimeError(
                        f"Unexpected source resolution "
                        f"at video={vi}, frame={fi}, camera={camera}: "
                        f"{src_w}x{src_h}"
                    )

                sx = TARGET_W / src_w
                sy = TARGET_H / src_h

                resized = im.resize(
                    (TARGET_W, TARGET_H),
                    resample=RGB_RESAMPLE,
                )


                relative_image = Path(
                    "images",
                    f"nuplan_video_{vi:06d}",
                    f"t{fi:03d}",
                    f"{camera}.jpg",
                )

                dst_image = DST_ROOT / relative_image

                dst_image.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                resized.save(
                    dst_image,
                    format="JPEG",
                    quality=95,
                )

                view["image_path"] = str(relative_image)

                num_images += 1
                target_sizes[(TARGET_W, TARGET_H)] += 1


            # ------------------------------------------------
            # 2. K
            #
            # pixel coordinates 跟着 resize 一起缩放
            # ------------------------------------------------

            K_old = np.asarray(
                view["K"],
                dtype=np.float64,
            )

            if K_old.shape != (3, 3):
                raise RuntimeError(
                    f"Bad K shape: {K_old.shape}"
                )

            K_new = K_old.copy()

            K_new[0, :] *= sx
            K_new[1, :] *= sy

            # 第三行不能动
            K_new[2, :] = K_old[2, :]

            view["K"] = K_new.tolist()

            # manifest image_size 同步改成新尺寸
            view["image_size"] = [
                TARGET_W,
                TARGET_H,
            ]


            # ------------------------------------------------
            # normalized K sanity
            #
            # resize 前后：
            #
            # fx/W
            # fy/H
            # cx/W
            # cy/H
            #
            # 应保持完全一致
            # ------------------------------------------------

            before = np.array([
                K_old[0, 0] / src_w,
                K_old[1, 1] / src_h,
                K_old[0, 2] / src_w,
                K_old[1, 2] / src_h,
            ])

            after = np.array([
                K_new[0, 0] / TARGET_W,
                K_new[1, 1] / TARGET_H,
                K_new[0, 2] / TARGET_W,
                K_new[1, 2] / TARGET_H,
            ])

            error = float(
                np.max(
                    np.abs(before - after)
                )
            )

            max_k_check_error = max(
                max_k_check_error,
                error,
            )


            # ------------------------------------------------
            # 3. valid mask
            # ------------------------------------------------

            raw_mask = view.get(
                "valid_mask_path"
            )

            if raw_mask:

                src_mask = resolve_path(
                    raw_mask,
                    SRC_MANIFEST,
                )

                if not src_mask.is_file():
                    raise FileNotFoundError(
                        src_mask
                    )

                with Image.open(src_mask) as mask:

                    mask = mask.convert("L")

                    mask = mask.resize(
                        (TARGET_W, TARGET_H),
                        resample=MASK_RESAMPLE,
                    )

                    relative_mask = Path(
                        "masks",
                        f"nuplan_video_{vi:06d}",
                        f"t{fi:03d}",
                        f"{camera}.png",
                    )

                    dst_mask = (
                        DST_ROOT
                        / relative_mask
                    )

                    dst_mask.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    mask.save(
                        dst_mask,
                        format="PNG",
                    )

                view["valid_mask_path"] = str(
                    relative_mask
                )

                num_masks += 1


            # ------------------------------------------------
            # 4. paired real
            #
            # 新 manifest 只用于 ST-Flow。
            #
            # 不把 1920x1080 paired_real 和 512x288 K 混在一起，
            # 所以直接删除 real_image_path。
            #
            # 原始 manifest 完全没动，FVD 以后仍用原 manifest。
            # ------------------------------------------------

            view.pop(
                "real_image_path",
                None,
            )


    if (
        (vi + 1) % 10 == 0
        or vi + 1 == len(items)
    ):
        print(
            f"[resize] {vi + 1}/{len(items)}"
        )


write_jsonl(
    DST_MANIFEST,
    items,
)


print()
print("=" * 90)
print("RESIZE DONE")
print("=" * 90)

print("source sizes:")
print(dict(source_sizes))

print("target sizes:")
print(dict(target_sizes))

print("images resized =", num_images)
print("masks resized  =", num_masks)

print(
    "max normalized-K preservation error =",
    max_k_check_error,
)

print()
print("manifest:")
print(DST_MANIFEST)


expected_images = 300 * 19 * 8

if num_images != expected_images:
    raise RuntimeError(
        f"Expected {expected_images} images, got {num_images}"
    )

if max_k_check_error > 1e-8:
    raise RuntimeError(
        "Normalized K changed during resize!"
    )

print()
print("RESOLUTION NORMALIZATION: PASS")
PY


# ============================================================
# 5. 再做一次最终 manifest sanity check
# ============================================================

python - <<PY
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


manifest = Path("$DST_MANIFEST")

sizes = Counter()

videos = []

with manifest.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            videos.append(json.loads(line))


first_K = None
first_size = None
first_path = None

missing = 0


for item in videos:

    for frame in item["frames"]:

        for view in frame["views"]:

            p = Path(view["image_path"])

            if not p.is_absolute():
                p = manifest.parent / p

            if not p.is_file():
                missing += 1
                continue

            if sum(sizes.values()) < 100:

                with Image.open(p) as im:
                    sizes[im.size] += 1

            if first_K is None:

                first_K = np.asarray(
                    view["K"],
                    dtype=float,
                )

                first_size = view.get(
                    "image_size"
                )

                first_path = p


print()
print("=" * 90)
print("FINAL MANIFEST CHECK")
print("=" * 90)

print("videos =", len(videos))
print("missing images =", missing)
print("sample sizes =", dict(sizes))

print()
print("first image:")
print(first_path)

print()
print("first manifest image_size:")
print(first_size)

print()
print("first K:")
print(first_K)

print()
print("normalized K:")
print(
    "fx/W =", first_K[0,0] / $TARGET_W,
    "fy/H =", first_K[1,1] / $TARGET_H,
    "cx/W =", first_K[0,2] / $TARGET_W,
    "cy/H =", first_K[1,2] / $TARGET_H,
)


if missing != 0:
    raise RuntimeError(
        f"{missing} images missing"
    )

if set(sizes.keys()) != {($TARGET_W, $TARGET_H)}:
    raise RuntimeError(
        f"Unexpected image resolutions: {sizes}"
    )

print()
print("FINAL MANIFEST CHECK: PASS")
PY


# ============================================================
# 6. ST-Flow
#
# 这里所有 threshold 都保持 GOOD 的原协议：
#
# cross gate = 16 px
# traj       = evaluator 内部 @2 / @4 px
#
# 因为现在两边实际输入都是 512x288，
# 所以终于可以直接公平比较。
# ============================================================

echo
echo "============================================================"
echo "Run DWM ST-Flow @ 512x288"
echo "============================================================"

python -m dwm.tools.evaluate_stflow \
    --manifest "$DST_MANIFEST" \
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
# 7. 打印结果
# ============================================================

python - <<PY
import json
from pathlib import Path

p = Path("$OUTPUT")

with p.open("r", encoding="utf-8") as f:
    x = json.load(f)

print()
print("=" * 100)
print("DWM @ 512x288 RESULT")
print("=" * 100)

print("num_videos =", x.get("num_videos"))

mean = x["mean"]

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
    "num_temporal_edges",
    "num_cross_raw_edges",
    "num_cross_edges",
    "num_cycle_edges",
    "num_traj_edges",
]

for key in keys:
    if key in mean:
        print(
            f"{key:26s} = {mean[key]}"
        )

print()
print("result:")
print(p)
PY

