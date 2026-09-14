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
# 2. 只使用你已有的本地权重
#
# evaluator 本身没有 --raft-checkpoint / --loftr-checkpoint 参数，
# 内部写死：
#   Raft_Large_Weights.DEFAULT
#   LoFTR(pretrained="outdoor")
#
# 所以正确做法是把你已有文件放到 Torch Hub cache。
# 不需要 wget/curl，不重新下载。
# ============================================================

PRETRAIN_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain
CKPT_ROOT=$PRETRAIN_ROOT/ckpt

RAFT_LOCAL=$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth
LOFTR_LOCAL=$CKPT_ROOT/loftr_outdoor.ckpt

if [ ! -f "$RAFT_LOCAL" ]; then
    echo "ERROR: existing RAFT weight missing:"
    echo "$RAFT_LOCAL"
    exit 1
fi

if [ ! -f "$LOFTR_LOCAL" ]; then
    echo "ERROR: existing LoFTR weight missing:"
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
echo "LOCAL WEIGHTS"
echo "============================================================"

echo "RAFT:"
ls -lh "$RAFT_LOCAL"

echo
echo "LoFTR:"
ls -lh "$LOFTR_LOCAL"

echo
echo "Torch Hub cache:"
ls -lh \
    "$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth" \
    "$CACHE_DIR/loftr_outdoor.ckpt"


# ============================================================
# 3. DWM manifests
# ============================================================

ROOT=$CAMSIM_ROOT/lyh_output/eval/nuscenesablationnew/nuplan6hz_merged300

ORIG=$ROOT/stflow_manifest.jsonl
FIXED=$ROOT/dwmnuplanstflow_manifest_fixedK.jsonl

REAL_ORIG=$ROOT/stflow_manifest_REAL_origK.jsonl
REAL_FIXED=$ROOT/stflow_manifest_REAL_fixedK.jsonl

OUT_ORIG=$ROOT/stflow_REAL_origK_20.json
OUT_FIXED=$ROOT/stflow_REAL_fixedK_20.json


for p in "$ORIG" "$FIXED"; do
    if [ ! -f "$p" ]; then
        echo "ERROR: manifest missing:"
        echo "$p"
        exit 1
    fi
done


# ============================================================
# 4. 生成 paired-real A/B manifests
#
# A:
#   real image + 原 DWM K
#
# B:
#   same real image + fixed K
#
# 除 K 外所有东西保持一致。
# ============================================================

python - <<PY
import copy
import json
from pathlib import Path
from PIL import Image
import numpy as np


ROOT = Path("$ROOT")
ORIG = Path("$ORIG")
FIXED = Path("$FIXED")

OUT_ORIG = Path("$REAL_ORIG")
OUT_FIXED = Path("$REAL_FIXED")


def load(path):
    out = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))

    return out


def resolve(path_string, manifest):
    p = Path(path_string)

    if not p.is_absolute():
        p = manifest.parent / p

    return p


def write(path, items):
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


orig = load(ORIG)
fixed = load(FIXED)

assert len(orig) == 300, len(orig)
assert len(fixed) == 300, len(fixed)

real_orig = copy.deepcopy(orig)
real_fixed = copy.deepcopy(fixed)

count = 0


for vi, (a, b) in enumerate(
    zip(real_orig, real_fixed)
):

    assert len(a["frames"]) == len(b["frames"])

    for fi, (af, bf) in enumerate(
        zip(a["frames"], b["frames"])
    ):

        assert len(af["views"]) == len(bf["views"])

        for ci, (av, bv) in enumerate(
            zip(af["views"], bf["views"])
        ):

            assert av["camera"] == bv["camera"], (
                vi, fi, ci,
                av["camera"],
                bv["camera"],
            )

            ra = av.get("real_image_path")
            rb = bv.get("real_image_path")

            if not ra or not rb:
                raise RuntimeError(
                    f"missing real_image_path "
                    f"video={vi} frame={fi} view={ci}"
                )

            # 两个 manifest 必须引用同一个 real frame
            if ra != rb:
                raise RuntimeError(
                    "real_image_path mismatch:\n"
                    f"A={ra}\n"
                    f"B={rb}"
                )

            rp = resolve(ra, ORIG)

            if not rp.is_file():
                raise FileNotFoundError(rp)

            # 只换 image_path。
            # K 各自保持：
            #   A = original K
            #   B = fixed K
            av["image_path"] = ra
            bv["image_path"] = rb

            count += 1


write(OUT_ORIG, real_orig)
write(OUT_FIXED, real_fixed)


print()
print("=" * 90)
print("REAL A/B manifest preparation")
print("=" * 90)

print("videos =", len(real_orig))
print("views  =", count)

print()
print("A: REAL + original K")
print(OUT_ORIG)

print()
print("B: REAL + fixed K")
print(OUT_FIXED)


# ------------------------------------------------------------
# 首帧首相机额外打印
# ------------------------------------------------------------

oa = real_orig[0]["frames"][0]["views"][0]
ob = real_fixed[0]["frames"][0]["views"][0]

real_path = resolve(
    oa["image_path"],
    OUT_ORIG,
)

with Image.open(real_path) as im:
    W, H = im.size


Ka = np.asarray(
    oa["K"],
    dtype=np.float64,
)

Kb = np.asarray(
    ob["K"],
    dtype=np.float64,
)


print()
print("=" * 90)
print("FIRST REAL FRAME CHECK")
print("=" * 90)

print("real image =", real_path)
print("actual size =", (W, H))

print()
print("manifest image_size (orig) =")
print(oa.get("image_size"))

print()
print("manifest image_size (fixed) =")
print(ob.get("image_size"))

print()
print("ORIGINAL K =")
print(Ka)

print()
print("FIXED K =")
print(Kb)

print()
print("ORIGINAL normalized K:")
print([
    Ka[0,0] / W,
    Ka[1,1] / H,
    Ka[0,2] / W,
    Ka[1,2] / H,
])

print()
print("FIXED normalized K:")
print([
    Kb[0,0] / W,
    Kb[1,1] / H,
    Kb[0,2] / W,
    Kb[1,2] / H,
])

print()
print("PASS")
PY


# ============================================================
# 5. REAL + original K
# ============================================================

echo
echo "============================================================"
echo "A: REAL + ORIGINAL DWM K"
echo "============================================================"

python -m dwm.tools.evaluate_stflow \
    --manifest "$REAL_ORIG" \
    --output "$OUT_ORIG" \
    --device cuda \
    --max-videos 20 \
    --frame-stride 2 \
    --min-matches 16 \
    --max-matches 256 \
    --loftr-confidence 0.1 \
    --pair-policy dataset \
    --cross-gate-px 16


# ============================================================
# 6. REAL + fixed K
# ============================================================

echo
echo "============================================================"
echo "B: REAL + FIXED K"
echo "============================================================"

python -m dwm.tools.evaluate_stflow \
    --manifest "$REAL_FIXED" \
    --output "$OUT_FIXED" \
    --device cuda \
    --max-videos 20 \
    --frame-stride 2 \
    --min-matches 16 \
    --max-matches 256 \
    --loftr-confidence 0.1 \
    --pair-policy dataset \
    --cross-gate-px 16


# ============================================================
# 7. 对比
# ============================================================

python - <<PY
import json

paths = {
    "REAL + ORIGINAL K": "$OUT_ORIG",
    "REAL + FIXED K": "$OUT_FIXED",
}

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
    "stflow_d_score",
    "stflow_c_score",
]


print()
print("=" * 100)
print("REAL K A/B RESULT")
print("=" * 100)

for name, path in paths.items():

    with open(path, "r", encoding="utf-8") as f:
        x = json.load(f)

    m = x["mean"]

    print()
    print(name)
    print("-" * 100)

    for key in keys:
        if key in m:
            print(
                f"{key:24s} = {m[key]}"
            )
PY

