#!/usr/bin/env bash

source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate
# ============================================================
# Environment
# ============================================================

SAM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval

BASE=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuscenesablation

BOX_ROOT=$BASE/nuscenesablationori3_box/box_projection

OUT_DIR=$BASE/sam3_nuscenes_ori3_box3d_full

CONFIG=$SAM_ROOT/config_nuscenes_ori3_box3d_full.yaml

export PATH=/inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin:$PATH
export PYTHONPATH="$SAM_ROOT/sam3${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd "$SAM_ROOT"


# ============================================================
# 1. Find already-merged RGB manifest
#
# 不重新 merge。
# 如果存在多个 ori3_merged*，自动选择 video 数最多的一份。
# ============================================================

BEST_MANIFEST=""
BEST_COUNT=-1

while IFS= read -r manifest; do
    count=$(grep -cve '^[[:space:]]*$' "$manifest")

    echo "[FOUND MERGED] videos=$count  $manifest"

    if (( count > BEST_COUNT )); then
        BEST_COUNT=$count
        BEST_MANIFEST=$manifest
    fi
done < <(
    find "$BASE" \
        -maxdepth 2 \
        -type f \
        -path "$BASE/nuscenesablationori3_merged*/stflow_manifest.jsonl" \
        | sort
)

if [[ -z "$BEST_MANIFEST" ]]; then
    echo "[ERROR] 没找到："
    echo "$BASE/nuscenesablationori3_merged*/stflow_manifest.jsonl"
    exit 1
fi

PREVIEW_ROOT=$(dirname "$BEST_MANIFEST")

echo
echo "============================================================"
echo "Using merged RGB"
echo "============================================================"
echo "PREVIEW_ROOT     = $PREVIEW_ROOT"
echo "PREVIEW_MANIFEST = $BEST_MANIFEST"
echo "VIDEOS           = $BEST_COUNT"
echo


# ============================================================
# 2. Check Box manifests
# ============================================================

if [[ ! -d "$BOX_ROOT" ]]; then
    echo "[ERROR] BOX_ROOT 不存在："
    echo "$BOX_ROOT"
    exit 1
fi

mapfile -t BOX_MANIFESTS < <(
    find "$BOX_ROOT" \
        -type f \
        -name box_manifest.jsonl \
        | sort
)

echo "============================================================"
echo "Box manifests"
echo "============================================================"

printf '%s\n' "${BOX_MANIFESTS[@]}"

echo
echo "BOX_MANIFEST_COUNT=${#BOX_MANIFESTS[@]}"

if (( ${#BOX_MANIFESTS[@]} == 0 )); then
    echo "[ERROR] 没找到 box_manifest.jsonl"
    exit 1
fi


# ============================================================
# 3. Quick structure check
# ============================================================

python - "$BEST_MANIFEST" "$BOX_ROOT" <<'PY'
import json
import sys
from pathlib import Path

preview_path = Path(sys.argv[1])
box_root = Path(sys.argv[2])

with preview_path.open("r", encoding="utf-8") as f:
    preview = json.loads(next(x for x in f if x.strip()))

print("\n===== PREVIEW FIRST VIDEO =====")
print("video_id:", preview["video_id"])
print("frames:", len(preview["frames"]))

f0 = preview["frames"][0]

print("views/frame:", len(f0["views"]))
print(
    "cameras:",
    [v.get("camera") for v in f0["views"]],
)

refs = sum(
    bool(v.get("is_reference_frame", False))
    for frame in preview["frames"]
    for v in frame["views"]
)

print("reference view-images/video:", refs)

box_paths = sorted(
    box_root.rglob("box_manifest.jsonl")
)

with box_paths[0].open("r", encoding="utf-8") as f:
    box = json.loads(next(x for x in f if x.strip()))

print("\n===== BOX FIRST VIDEO =====")
print("manifest:", box_paths[0])
print("video_id:", box["video_id"])
print("frames:", len(box["frames"]))

bf0 = box["frames"][0]

print("boxes frame0:", len(bf0.get("boxes_3d", [])))
print("views frame0:", len(bf0["views"]))

if bf0.get("boxes_3d"):
    b = bf0["boxes_3d"][0]

    print("first class:", b.get("class_name"))
    print(
        "coordinate_frame:",
        b.get("coordinate_frame"),
    )
    print(
        "corners:",
        len(b.get("corners_ref", [])),
    )

print(
    "view has T_reference_ego_to_camera:",
    "T_reference_ego_to_camera"
    in bf0["views"][0],
)
PY


# ============================================================
# 4. Create full SAM config
# ============================================================

cat > "$CONFIG" <<EOF
paths:
  preview_root: ${PREVIEW_ROOT}

  shared_box_root: ${BOX_ROOT}

  sam3_repo: /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval/sam3

  checkpoint: /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/ckpt/sam3.1/sam3.1_multiplex.pt

  output_dir: ${OUT_DIR}


preview:
  # RGB 已经 merged，只读取这一份。
  manifest_glob: 'stflow_manifest.jsonl'

  # 前3帧是 GT reference，不参与生成结果测评。
  skip_reference_frames: true

  strict_paths: true

  include_methods: []
  exclude_methods: []


shared_box:
  # rank_00 ~ rank_03 不需要手工合并。
  manifest_glob: '**/box_manifest.jsonl'

  strict_paths: true
  strict_match: true


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

  # 每个 rank 2 workers，4 GPU 总共 8 workers。
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

  # 必须固定为 0，不使用 GT-dependent SAM mask 过滤。
  min_sam_connected_pixels: 0


visualization:
  enabled: true

  # 全量计算，只保存前32张可视化。
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
# 5. Compile check
# ============================================================

echo
echo "============================================================"
echo "Compile check"
echo "============================================================"

python -m py_compile \
    shared_box_projection.py \
    data_geometry.py \
    matching_visualization.py \
    run_eval.py

python -c "import sam3; print('[OK] sam3 import')"

python - <<'PY'
import torch

print(
    "[CUDA]",
    torch.__version__,
    "available=",
    torch.cuda.is_available(),
    "count=",
    torch.cuda.device_count(),
)
PY


# ============================================================
# 6. Full scan-only
#
# strict_match=true，所以这里如果 RGB / Box 对不上会直接失败。
# 不加载 SAM。
# ============================================================

echo
echo "============================================================"
echo "Full scan-only"
echo "============================================================"

python run_eval.py \
    --config "$CONFIG" \
    --scan-only


# ============================================================
# 7. Full SAM evaluation — 4 GPUs
# ============================================================

echo
echo "============================================================"
echo "Start full SAM3 evaluation"
echo "============================================================"
echo "OUTPUT = $OUT_DIR"
echo

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --standalone \
    --nproc_per_node=4 \
    run_eval.py \
    --config "$CONFIG"


# ============================================================
# 8. Print results
# ============================================================

echo
echo "============================================================"
echo "Finished"
echo "============================================================"

echo "OUTPUT:"
echo "$OUT_DIR"

echo
echo "Files:"
find "$OUT_DIR" \
    -maxdepth 2 \
    -type f \
    | sort

echo
echo "===== SUMMARY ====="

if [[ -f "$OUT_DIR/summary.json" ]]; then
    cat "$OUT_DIR/summary.json"
else
    find "$OUT_DIR" \
        -maxdepth 1 \
        -type f \
        -name '*summary*' \
        -print
fi
