#!/usr/bin/env bash
set -euo pipefail

source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

SAM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval
BASE=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuscenesablation

PREVIEW_ROOT=$BASE/nuscenesablationori6_merged500
PREVIEW_MANIFEST=$PREVIEW_ROOT/stflow_manifest_front3.jsonl

BOX_ROOT=$BASE/nuscenesablationori6_box/box_projection

OUT_DIR=$BASE/sam3_nuscenes_ori6_front3_pairedreal_full
CONFIG=$SAM_ROOT/config_nuscenes_ori6_front3_pairedreal_1gpu.yaml

export PATH=/inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin:$PATH
export PYTHONPATH="$SAM_ROOT/sam3${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd "$SAM_ROOT"

echo "============================================================"
echo "ORI6 FRONT3 PAIRED-REAL"
echo "============================================================"
echo "PREVIEW = $PREVIEW_MANIFEST"
echo "BOX     = $BOX_ROOT"
echo "OUTPUT  = $OUT_DIR"
echo

test -f "$PREVIEW_MANIFEST" || {
    echo "[ERROR] missing manifest:"
    echo "$PREVIEW_MANIFEST"
    exit 1
}

test -d "$BOX_ROOT" || {
    echo "[ERROR] missing box root:"
    echo "$BOX_ROOT"
    exit 1
}

echo "===== VIDEO COUNT ====="
wc -l "$PREVIEW_MANIFEST"

echo
echo "===== BOX MANIFESTS ====="
find "$BOX_ROOT" \
    -type f \
    -name box_manifest.jsonl \
    -print | sort


# ============================================================
# Check paired-real paths before loading SAM
# ============================================================

echo
echo "===== CHECK PAIRED REAL ====="

python - "$PREVIEW_MANIFEST" "$PREVIEW_ROOT" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
root = Path(sys.argv[2])

videos = 0
views = 0
real_paths = 0
missing = 0
examples = []

with manifest.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        x = json.loads(line)
        videos += 1

        for frame in x["frames"]:
            for view in frame["views"]:
                views += 1

                value = view.get("real_image_path")
                if not value:
                    missing += 1
                    continue

                p = Path(value)
                if not p.is_absolute():
                    p = root / p

                if p.is_file():
                    real_paths += 1
                    if len(examples) < 5:
                        examples.append(str(p))
                else:
                    missing += 1

print("videos          =", videos)
print("all view-images =", views)
print("paired real OK  =", real_paths)
print("missing         =", missing)

print("\nexamples:")
for p in examples:
    print(" ", p)

if missing:
    raise SystemExit(
        "[ERROR] paired-real path incomplete"
    )
PY


# ============================================================
# Config
# ============================================================

cat > "$CONFIG" <<EOF
paths:
  preview_root: ${PREVIEW_ROOT}

  shared_box_root: ${BOX_ROOT}

  sam3_repo: /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval/sam3

  checkpoint: /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/ckpt/sam3.1/sam3.1_multiplex.pt

  output_dir: ${OUT_DIR}


preview:
  # 只读取已经筛好的 ori6 front3 manifest
  manifest_glob: 'stflow_manifest_front3.jsonl'

  # 和 generated 测评完全一致：
  # 前3帧 reference 不参与
  skip_reference_frames: true

  strict_paths: true

  include_methods: []
  exclude_methods: []


shared_box:
  manifest_glob: '**/box_manifest.jsonl'

  strict_paths: true
  strict_match: true

  # 原 ori6 Box 摄像头编号 -> front3 canonical name
  camera_name_map:
    CAM_01: CAM_FRONT_LEFT
    CAM_02: CAM_FRONT
    CAM_03: CAM_FRONT_RIGHT


sources:
  pairedreal:
    # 关键：使用 manifest 里的 real_image_path
    type: preview_real

    # 所有 paired real 汇总成一个 source
    group_by_manifest: false


model:
  version: sam3.1

  prompts:
    - car
    - truck
    - bus

  confidence_threshold: 0.25

  batch_size: 1
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

  min_sam_connected_pixels: 0


visualization:
  enabled: true

  # 全量计算，只保存前32张检查图
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
# Compile check
# ============================================================

echo
echo "===== COMPILE CHECK ====="

python -m py_compile \
    shared_box_projection.py \
    data_geometry.py \
    matching_visualization.py \
    run_eval.py

python -c "import sam3; print('[OK] sam3 import')"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY


# ============================================================
# Full scan-only first
# ============================================================

echo
echo "============================================================"
echo "FULL SCAN ONLY"
echo "============================================================"

python run_eval.py \
    --config "$CONFIG" \
    --scan-only


# ============================================================
# Full paired-real SAM - 1 GPU
# ============================================================

echo
echo "============================================================"
echo "START PAIRED-REAL FULL SAM3 - 1 GPU"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 \
python run_eval.py \
    --config "$CONFIG"


# ============================================================
# Results
# ============================================================

echo
echo "============================================================"
echo "DONE"
echo "============================================================"

echo "OUTPUT = $OUT_DIR"

echo
echo "===== RECORD COUNT ====="

wc -l "$OUT_DIR"/records.rank000.jsonl || true

echo
echo "===== SUMMARY ====="

if [[ -f "$OUT_DIR/summary.json" ]]; then
    cat "$OUT_DIR/summary.json"
else
    find "$OUT_DIR" \
        -maxdepth 1 \
        -type f \
        -print
fi
