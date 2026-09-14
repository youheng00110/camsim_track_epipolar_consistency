#!/usr/bin/env bash
set -euo pipefail

source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

SAM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval
BASE=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuscenesablation

PREVIEW_ROOT=$BASE/nuscenesablationori6_merged500
BOX_ROOT=$BASE/nuscenesablationori6_box/box_projection

OUT_DIR=$BASE/sam3_nuscenes_ori6_front3_full
CONFIG=$SAM_ROOT/config_nuscenes_ori6_front3_full_1gpu.yaml

export PATH=/inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin:$PATH
export PYTHONPATH="$SAM_ROOT/sam3${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd "$SAM_ROOT"

echo "========================================"
echo "PREVIEW_ROOT = $PREVIEW_ROOT"
echo "BOX_ROOT     = $BOX_ROOT"
echo "OUT_DIR      = $OUT_DIR"
echo "========================================"

# ------------------------------------------------------------
# Check input
# ------------------------------------------------------------

test -f "$PREVIEW_ROOT/stflow_manifest_front3.jsonl" || {
    echo "[ERROR] missing:"
    echo "$PREVIEW_ROOT/stflow_manifest_front3.jsonl"
    exit 1
}

test -d "$BOX_ROOT" || {
    echo "[ERROR] missing BOX_ROOT:"
    echo "$BOX_ROOT"
    exit 1
}

echo
echo "===== Preview videos ====="
wc -l "$PREVIEW_ROOT/stflow_manifest_front3.jsonl"

echo
echo "===== Box manifests ====="
find "$BOX_ROOT" \
    -name box_manifest.jsonl \
    -type f \
    -print | sort


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

cat > "$CONFIG" <<EOF
paths:
  preview_root: ${PREVIEW_ROOT}
  shared_box_root: ${BOX_ROOT}

  sam3_repo: /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval/sam3

  checkpoint: /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/ckpt/sam3.1/sam3.1_multiplex.pt

  output_dir: ${OUT_DIR}


preview:
  # 已经筛选好的 ori6 中间三个摄像头
  manifest_glob: 'stflow_manifest_front3.jsonl'

  # 不测前3帧 reference
  skip_reference_frames: true

  strict_paths: true
  include_methods: []
  exclude_methods: []


shared_box:
  manifest_glob: '**/box_manifest.jsonl'

  strict_paths: true
  strict_match: true

  # ori6 原始 camera index:
  # CAM_01 -> FRONT_LEFT
  # CAM_02 -> FRONT
  # CAM_03 -> FRONT_RIGHT
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

  # 单卡，保持2个读取worker即可
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

  # 固定为0
  min_sam_connected_pixels: 0


visualization:
  enabled: true

  # 全量算，只保存前32张图
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


# ------------------------------------------------------------
# Compile
# ------------------------------------------------------------

echo
echo "===== Compile check ====="

python -m py_compile \
    shared_box_projection.py \
    data_geometry.py \
    matching_visualization.py \
    run_eval.py

python -c "import sam3; print('[OK] sam3 import')"

python - <<'PY'
import torch
print(
    "torch =", torch.__version__,
    "cuda =", torch.cuda.is_available(),
    "gpu_count =", torch.cuda.device_count(),
)
PY


# ------------------------------------------------------------
# Scan first
# ------------------------------------------------------------

echo
echo "========================================"
echo "SCAN ONLY"
echo "========================================"

python run_eval.py \
    --config "$CONFIG" \
    --scan-only


# ------------------------------------------------------------
# Full evaluation - 1 GPU
# ------------------------------------------------------------

echo
echo "========================================"
echo "START FULL ORI6 FRONT3 SAM3 - 1 GPU"
echo "========================================"

CUDA_VISIBLE_DEVICES=0 \
python run_eval.py \
    --config "$CONFIG"


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

echo
echo "========================================"
echo "DONE"
echo "========================================"

echo "OUTPUT:"
echo "$OUT_DIR"

echo
echo "===== rank records ====="

wc -l "$OUT_DIR"/records.rank*.jsonl || true

echo
echo "===== summary ====="

if [[ -f "$OUT_DIR/summary.json" ]]; then
    cat "$OUT_DIR/summary.json"
else
    find "$OUT_DIR" \
        -maxdepth 1 \
        -type f \
        -print
fi
