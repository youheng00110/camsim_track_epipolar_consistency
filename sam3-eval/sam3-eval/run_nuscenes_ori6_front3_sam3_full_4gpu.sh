#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Environment
# ============================================================

source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

SAM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval

BASE=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuscenesablation

PREVIEW_ROOT=$BASE/nuscenesablationori6_merged500
PREVIEW_MANIFEST=$PREVIEW_ROOT/stflow_manifest_front3.jsonl

BOX_ROOT=$BASE/nuscenesablationori6_box/box_projection

OUT_DIR=$BASE/sam3_nuscenes_ori6_front3_full

CONFIG=$SAM_ROOT/config_nuscenes_ori6_front3_full.yaml

export PATH=/inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin:$PATH
export PYTHONPATH="$SAM_ROOT/sam3${PYTHONPATH:+:$PYTHONPATH}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd "$SAM_ROOT"


# ============================================================
# 1. Input checks
# ============================================================

echo "============================================================"
echo "NuScenes ori6 -> front3 full SAM3 evaluation"
echo "============================================================"
echo "PREVIEW_ROOT     = $PREVIEW_ROOT"
echo "PREVIEW_MANIFEST = $PREVIEW_MANIFEST"
echo "BOX_ROOT         = $BOX_ROOT"
echo "OUTPUT           = $OUT_DIR"
echo

if [[ ! -f "$PREVIEW_MANIFEST" ]]; then
    echo "[ERROR] preview manifest 不存在:"
    echo "$PREVIEW_MANIFEST"
    exit 1
fi

if [[ ! -d "$BOX_ROOT" ]]; then
    echo "[ERROR] box root 不存在:"
    echo "$BOX_ROOT"
    exit 1
fi

echo "===== Preview ====="
ls -lh "$PREVIEW_MANIFEST"

echo
echo "===== Video count ====="
wc -l "$PREVIEW_MANIFEST"

echo
echo "===== Box manifests ====="
find "$BOX_ROOT" \
    -type f \
    -name box_manifest.jsonl \
    -print \
    | sort

BOX_COUNT=$(
    find "$BOX_ROOT" \
        -type f \
        -name box_manifest.jsonl \
        | wc -l
)

echo
echo "BOX_MANIFEST_COUNT=$BOX_COUNT"

if [[ "$BOX_COUNT" -eq 0 ]]; then
    echo "[ERROR] 没有找到 box_manifest.jsonl"
    exit 1
fi


# ============================================================
# 2. Verify front3 manifest
# ============================================================

python - "$PREVIEW_MANIFEST" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path, encoding="utf-8") as f:
    record = json.loads(next(
        line for line in f if line.strip()
    ))

print("\n===== FIRST VIDEO CHECK =====")
print("video_id:", record["video_id"])
print("frames:", len(record["frames"]))

frame = record["frames"][0]

cameras = [
    view["camera"]
    for view in frame["views"]
]

print("views:", len(cameras))
print("cameras:", cameras)

expected = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
]

if cameras != expected:
    raise RuntimeError(
        "front3 camera mismatch: "
        f"got={cameras}, expected={expected}"
    )

print("[OK] front3 cameras correct")
PY


# ============================================================
# 3. Create SAM config
# ============================================================

cat > "$CONFIG" <<EOF
paths:
  preview_root: ${PREVIEW_ROOT}

  shared_box_root: ${BOX_ROOT}

  sam3_repo: /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval/sam3

  checkpoint: /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/ckpt/sam3.1/sam3.1_multiplex.pt

  output_dir: ${OUT_DIR}


preview:
  # 已经是筛好的 front3 manifest。
  # 图片仍然直接指向 merged500/images。
  manifest_glob: 'stflow_manifest_front3.jsonl'

  # 前3帧是 reference，生成质量评测不计算。
  skip_reference_frames: true

  strict_paths: true

  include_methods: []
  exclude_methods: []


shared_box:
  # 直接扫描 ori6 box 的 rank_00~03。
  # 不需要重新合并。
  manifest_glob: '**/box_manifest.jsonl'

  strict_paths: true
  strict_match: true

  # ori6 原始 camera index -> front3 canonical name
  camera_name_map:
    CAM_01: CAM_FRONT_LEFT
    CAM_02: CAM_FRONT
    CAM_03: CAM_FRONT_RIGHT


sources:
  generated:
    type: preview_generated
    group_by_manifest: true


model:
  version: sam3.1

  prompts:
    - car
    - truck
    - bus

  confidence_threshold: 0.25

  batch_size: 1

  # 4 GPU × 2 workers
  loader_workers: 2

  precision: bfloat16
  input_resolution: 1008

  save_masks: true
  mask_resolution: 256
  mask_threshold: 0.5

  max_detections_per_prompt: 100
  max_detections_per_image: 150

  nms_iou_threshold: 0.7

  checkpoint_minimum_coverage: 0.95
  checkpoint_mmap: true


box_projection:
  vehicle_colors_rgb:
    - [0, 0, 255]

  color_tolerance: 100
  min_brightness: 20

  close_kernel: 1

  min_line_pixels: 16
  min_bbox_width: 5
  min_bbox_height: 5
  min_hull_area: 36


annotation:
  classes:
    - CAR
    - TRUCK
    - BUS

  near_plane: 0.1

  min_projected_height_px: 8.0
  min_projected_area_px: 64.0
  min_in_frame_fraction: 0.1


visibility:
  disable_gt_occlusion: true

  min_gt_visible_ratio: 0.0
  min_gt_visible_connected_pixels: 4


matching:
  min_mask_iou: 0.05

  max_center_error_norm: 0.8
  max_scale_error_log: 1.2

  max_cost: 1.1

  cost_mask_iou: 0.65
  cost_center: 0.2
  cost_bottom: 0.05
  cost_scale: 0.1

  allow_bbox_mask_fallback: false

  # 不使用 GT-dependent SAM size threshold
  min_sam_connected_pixels: 0


visualization:
  enabled: true

  # 全量计算，只保存前32张可视化
  max_frames_per_source: 32

  image_quality: 95
  mask_alpha: 0.25

  draw_cuboid: true
  draw_detection_box: true
  draw_filtered_small_sam: true


runtime:
  seed: 3407

  backend: sam3.1

  # 0 = 全量
  limit_frames: 0

  overwrite: true
EOF


# ============================================================
# 4. Code / environment check
# ============================================================

echo
echo "============================================================"
echo "Environment check"
echo "============================================================"

which python
which torchrun

python -m py_compile \
    shared_box_projection.py \
    data_geometry.py \
    matching_visualization.py \
    run_eval.py

python -c "import sam3; print('[OK] sam3 import')"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

if torch.cuda.device_count() < 4:
    raise RuntimeError(
        f"Need 4 GPUs, got {torch.cuda.device_count()}"
    )
PY


# ============================================================
# 5. Full scan-only
#
# 这里不加载 SAM。
# 先确保 RGB ↔ Box 全量匹配。
# ============================================================

echo
echo "============================================================"
echo "Full scan-only"
echo "============================================================"

python run_eval.py \
    --config "$CONFIG" \
    --scan-only


# ============================================================
# 6. Full SAM3 - 4 GPU
# ============================================================

echo
echo "============================================================"
echo "Start SAM3 full evaluation on 4 GPUs"
echo "============================================================"
echo

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --standalone \
    --nproc_per_node=4 \
    run_eval.py \
    --config "$CONFIG"


# ============================================================
# 7. Results
# ============================================================

echo
echo "============================================================"
echo "SAM3 evaluation finished"
echo "============================================================"

echo
echo "OUTPUT:"
echo "$OUT_DIR"

echo
echo "===== rank record count ====="

wc -l "$OUT_DIR"/records.rank*.jsonl || true

echo
echo "===== total evaluated frames ====="

python - "$OUT_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])

paths = sorted(root.glob("records.rank*.jsonl"))

total = 0

for path in paths:
    count = sum(
        1
        for line in path.open(
            encoding="utf-8"
        )
        if line.strip()
    )

    print(
        path.name,
        count,
    )

    total += count

print("--------------------------")
print("TOTAL =", total)
PY


echo
echo "===== summary ====="

if [[ -f "$OUT_DIR/summary.json" ]]; then
    cat "$OUT_DIR/summary.json"
else
    find "$OUT_DIR" \
        -maxdepth 1 \
        -type f \
        -iname '*summary*' \
        -print
fi

echo
echo "===== visualizations ====="

find "$OUT_DIR/visualizations" \
    -type f \
    -name '*.jpg' \
    | head -10 || true

