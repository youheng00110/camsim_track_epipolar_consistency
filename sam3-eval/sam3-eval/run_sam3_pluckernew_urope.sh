#!/usr/bin/env bash
source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate
set -uo pipefail

# ============================================================
# 0. Environment
# ============================================================

export PATH=/inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval


# ============================================================
# 1. Paths
# ============================================================

BASE=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuplanhard1000

PLUCKER_SRC=$BASE/pluckernew
UROPE_SRC=$BASE/urope

SHARED_BOX=$BASE/shared_box_preview_paired_200

OUTPUT_ROOT=$BASE/sam3_nuplanhard1000_box3d
CONFIG_ROOT=$PWD/configs_nuplan1000_box3d
LOG_ROOT=$CONFIG_ROOT/logs

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$CONFIG_ROOT"
mkdir -p "$LOG_ROOT"


# ============================================================
# 2. Environment check
# ============================================================

echo
echo "============================================================"
echo "Environment check"
echo "============================================================"

which python

python - <<'PY'
import torch
import iopath

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())
print("iopath: OK")
PY

if [ ! -d "$SHARED_BOX" ]
then
    echo "Shared Box root missing:"
    echo "$SHARED_BOX"
    exit 1
fi


# ============================================================
# 3. Discover the 1000-video manifests and generate configs
# ============================================================

python - <<'PY'
from pathlib import Path
import copy
import json
import os
import shutil
import yaml


base = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/eval/nuplanhard1000"
)

sources = {
    "pluckernew": base / "pluckernew",
    "urope": base / "urope",
}

shared_box = base / "shared_box_preview_paired_200"

config_root = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval/"
    "configs_nuplan1000_box3d"
)

output_root = base / "sam3_nuplanhard1000_box3d"

template_candidates = [
    config_root / "generated_petr.yaml",
    Path(
        "/inspire/qb-ilm/project/advanced-machine-learning/"
        "yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval/"
        "config_shared_box_nuplan.yaml"
    ),
]

template_path = None

for candidate in template_candidates:
    if candidate.is_file():
        template_path = candidate
        break

if template_path is None:
    raise FileNotFoundError(
        "Cannot find generated_petr.yaml or "
        "config_shared_box_nuplan.yaml"
    )

print("Template:", template_path)

base_config = yaml.safe_load(
    template_path.read_text(encoding="utf-8")
)

resolved = {}

for alias, source_root in sources.items():
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"{alias} source directory missing: {source_root}"
        )

    candidates = []

    direct = source_root / "stflow_manifest.jsonl"
    if direct.is_file():
        candidates.append(direct)

    sibling_merged = Path(
        str(source_root) + "_merged1000"
    ) / "stflow_manifest.jsonl"

    if sibling_merged.is_file():
        candidates.append(sibling_merged)

    for path in source_root.rglob(
        "stflow_manifest.jsonl"
    ):
        if "merged1000" in str(path.parent):
            candidates.append(path)

    candidates = list(dict.fromkeys(candidates))

    candidate_info = []

    for manifest in candidates:
        count = 0

        with manifest.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if line.strip():
                    json.loads(line)
                    count += 1

        candidate_info.append(
            (manifest, count)
        )

    exact = [
        item
        for item in candidate_info
        if item[1] == 1000
    ]

    if exact:
        manifest = exact[0][0]
        eval_root = manifest.parent
    else:
        large = [
            item
            for item in candidate_info
            if item[1] > 1000
        ]

        if not large:
            print()
            print(f"{alias} candidate manifests:")

            for manifest, count in candidate_info:
                print(
                    f"  {count:5d}  {manifest}"
                )

            raise RuntimeError(
                f"{alias}: no manifest containing "
                "at least 1000 videos"
            )

        source_manifest = large[0][0]
        source_eval_root = source_manifest.parent

        eval_root = (
            config_root
            / "first1000_staging"
            / alias
        )

        if eval_root.exists():
            shutil.rmtree(eval_root)

        eval_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_manifest = (
            eval_root
            / "stflow_manifest.jsonl"
        )

        written = 0

        with source_manifest.open(
            "r",
            encoding="utf-8",
        ) as source_file:
            with output_manifest.open(
                "w",
                encoding="utf-8",
            ) as output_file:
                for line in source_file:
                    if not line.strip():
                        continue

                    json.loads(line)
                    output_file.write(line)

                    if not line.endswith("\n"):
                        output_file.write("\n")

                    written += 1

                    if written == 1000:
                        break

        for child in source_eval_root.iterdir():
            if child.name == "stflow_manifest.jsonl":
                continue

            target = eval_root / child.name

            if target.exists() or target.is_symlink():
                continue

            os.symlink(
                child,
                target,
                target_is_directory=child.is_dir(),
            )

        print(
            f"{alias}: staged first 1000 "
            f"from {source_manifest}"
        )

    resolved[alias] = eval_root

    config = copy.deepcopy(base_config)

    config["paths"]["preview_root"] = str(
        eval_root
    )

    config["paths"]["shared_box_root"] = str(
        shared_box
    )

    config["paths"]["output_dir"] = str(
        output_root / alias
    )

    config.setdefault("preview", {})
    config["preview"]["manifest_glob"] = (
        "stflow_manifest.jsonl"
    )
    config["preview"]["include_methods"] = []
    config["preview"]["exclude_methods"] = []
    config["preview"]["skip_reference_frames"] = True
    config["preview"]["strict_paths"] = True

    config.setdefault("shared_box", {})
    config["shared_box"]["manifest_glob"] = (
        "**/box_manifest.jsonl"
    )
    config["shared_box"]["strict_match"] = True
    config["shared_box"]["strict_paths"] = True

    config["sources"] = {
        "generated": {
            "type": "preview_generated",
            "group_by_manifest": True,
        }
    }

    config.setdefault("matching", {})

    # Important:
    # Do not use GT-dependent dynamic minimum SAM mask size.
    config["matching"][
        "min_sam_connected_pixels"
    ] = 0

    config.setdefault("visibility", {})
    config["visibility"][
        "disable_gt_occlusion"
    ] = True

    config["visibility"][
        "min_gt_visible_ratio"
    ] = 0.0

    config.setdefault("visualization", {})
    config["visualization"]["enabled"] = True
    config["visualization"][
        "max_frames_per_source"
    ] = 32
    config["visualization"][
        "draw_cuboid"
    ] = True
    config["visualization"][
        "draw_detection_box"
    ] = True

    config.setdefault("runtime", {})
    config["runtime"]["limit_frames"] = 0
    config["runtime"]["overwrite"] = True

    config_path = (
        config_root
        / f"generated_{alias}.yaml"
    )

    config_path.write_text(
        yaml.safe_dump(
            config,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"[{alias}]"
    )
    print(
        "  eval root:",
        eval_root,
    )
    print(
        "  config:",
        config_path,
    )
    print(
        "  output:",
        output_root / alias,
    )

print()
print("Both configs prepared.")
PY


# ============================================================
# 4. Quick config checks
# ============================================================

echo
echo "============================================================"
echo "Config check"
echo "============================================================"

for NAME in pluckernew urope
do
    CONFIG=$CONFIG_ROOT/generated_${NAME}.yaml

    echo
    echo "[$NAME]"

    grep -E \
        'preview_root:|shared_box_root:|output_dir:|min_sam_connected_pixels:' \
        "$CONFIG"
done


# ============================================================
# 5. Run two methods in parallel
#
# GPU 0 -> pluckernew
# GPU 1 -> urope
# ============================================================

PLUCKER_OUT=$OUTPUT_ROOT/pluckernew
UROPE_OUT=$OUTPUT_ROOT/urope

PLUCKER_LOG=$LOG_ROOT/generated_pluckernew.log
UROPE_LOG=$LOG_ROOT/generated_urope.log


PLUCKER_PID=""

if [ -s "$PLUCKER_OUT/summary.json" ]
then
    echo "[SKIP] pluckernew already completed"
else
    rm -rf "$PLUCKER_OUT"

    echo
    echo "============================================================"
    echo "[GPU 0] START pluckernew"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=0 \
    python -u run_eval.py \
        --config "$CONFIG_ROOT/generated_pluckernew.yaml" \
        > "$PLUCKER_LOG" 2>&1 &

    PLUCKER_PID=$!

    echo "pluckernew PID=$PLUCKER_PID"
    echo "log=$PLUCKER_LOG"
fi


UROPE_PID=""

if [ -s "$UROPE_OUT/summary.json" ]
then
    echo "[SKIP] urope already completed"
else
    rm -rf "$UROPE_OUT"

    echo
    echo "============================================================"
    echo "[GPU 1] START urope"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=1 \
    python -u run_eval.py \
        --config "$CONFIG_ROOT/generated_urope.yaml" \
        > "$UROPE_LOG" 2>&1 &

    UROPE_PID=$!

    echo "urope PID=$UROPE_PID"
    echo "log=$UROPE_LOG"
fi


# ============================================================
# 6. Wait
# ============================================================

PLUCKER_STATUS=0
UROPE_STATUS=0

if [ -n "$PLUCKER_PID" ]
then
    wait "$PLUCKER_PID" || PLUCKER_STATUS=$?
fi

if [ -n "$UROPE_PID" ]
then
    wait "$UROPE_PID" || UROPE_STATUS=$?
fi


if [ "$PLUCKER_STATUS" -ne 0 ]
then
    echo
    echo "[FAILED] pluckernew exit=$PLUCKER_STATUS"
    tail -100 "$PLUCKER_LOG"
fi

if [ "$UROPE_STATUS" -ne 0 ]
then
    echo
    echo "[FAILED] urope exit=$UROPE_STATUS"
    tail -100 "$UROPE_LOG"
fi

if [ "$PLUCKER_STATUS" -ne 0 ] || [ "$UROPE_STATUS" -ne 0 ]
then
    exit 1
fi


# ============================================================
# 7. Check results
# ============================================================

for NAME in pluckernew urope
do
    SUMMARY=$OUTPUT_ROOT/$NAME/summary.json

    if [ ! -s "$SUMMARY" ]
    then
        echo "Missing summary: $SUMMARY"
        exit 1
    fi
done


# ============================================================
# 8. Print two new results + rebuild combined table
# ============================================================

python - <<'PY'
from pathlib import Path
import csv
import json


root = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/eval/nuplanhard1000/"
    "sam3_nuplanhard1000_box3d"
)

order = [
    "pairedreal_implicit",
    "plucker",
    "pluckernew",
    "box",
    "implicit",
    "full",
    "petr",
    "pvonly",
    "nocondition",
    "token18000",
    "token24000",
    "tvself",
    "urope",
]

rows = []

for method in order:
    path = root / method / "summary.json"

    if not path.is_file():
        continue

    summary = json.loads(
        path.read_text(encoding="utf-8")
    )

    sources = summary.get("sources", {})

    if not sources:
        continue

    source = next(iter(sources.values()))

    gt = int(source.get("gt_count", 0))
    det = int(
        source.get("detection_count", 0)
    )
    match = int(
        source.get("matched_count", 0)
    )

    rows.append({
        "method": method,
        "frames": int(
            source.get("frames", 0)
        ),
        "gt": gt,
        "det": det,
        "match": match,
        "fn": gt - match,
        "fp": det - match,
        "recall": source.get("recall"),
        "precision": source.get("precision"),
        "f1": source.get("f1"),
        "mask_iou": source.get(
            "mean_mask_iou"
        ),
        "center": source.get(
            "mean_center_error_norm"
        ),
        "bottom": source.get(
            "mean_bottom_error_norm"
        ),
        "scale": source.get(
            "mean_scale_error_log"
        ),
    })


print()
print("=" * 139)

print(
    f"{'Method':<22}"
    f"{'Frames':>9}"
    f"{'GT':>10}"
    f"{'Det':>10}"
    f"{'Match':>10}"
    f"{'FN':>9}"
    f"{'FP':>9}"
    f"{'Recall':>10}"
    f"{'Prec':>10}"
    f"{'F1':>10}"
    f"{'MaskIoU':>11}"
    f"{'Center':>10}"
    f"{'Bottom':>10}"
    f"{'Scale':>10}"
)

print("-" * 139)

for row in rows:
    print(
        f"{row['method']:<22}"
        f"{row['frames']:>9d}"
        f"{row['gt']:>10d}"
        f"{row['det']:>10d}"
        f"{row['match']:>10d}"
        f"{row['fn']:>9d}"
        f"{row['fp']:>9d}"
        f"{row['recall']:>10.4f}"
        f"{row['precision']:>10.4f}"
        f"{row['f1']:>10.4f}"
        f"{row['mask_iou']:>11.4f}"
        f"{row['center']:>10.4f}"
        f"{row['bottom']:>10.4f}"
        f"{row['scale']:>10.4f}"
    )

csv_path = (
    root
    / "sam3_all_results_plus_pluckernew_urope.csv"
)

with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)

print("=" * 139)
print()
print("Combined CSV:", csv_path)

print()
print("New methods only:")

for row in rows:
    if row["method"] not in {
        "pluckernew",
        "urope",
    }:
        continue

    print(
        f"{row['method']}: "
        f"Recall={row['recall']:.4f}, "
        f"Precision={row['precision']:.4f}, "
        f"F1={row['f1']:.4f}, "
        f"MaskIoU={row['mask_iou']:.4f}, "
        f"Center={row['center']:.4f}, "
        f"Bottom={row['bottom']:.4f}, "
        f"Scale={row['scale']:.4f}"
    )
PY

echo
echo "============================================================"
echo "SAM3 evaluation complete"
echo "============================================================"
