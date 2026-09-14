#!/usr/bin/env bash

ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim

TARGET_ROOT=$ROOT/lyh_output/eval/nuscenesablationnew/1000nuplan6hz_rankclean512

BOX_RAW=$ROOT/lyh_output/eval/nuplanhard1000/uropetvtrack/box

SAM_ROOT=$ROOT/sam3-eval/sam3-eval

OUT_ROOT=$TARGET_ROOT/sam3_box_eval
CFG_ROOT=$OUT_ROOT/configs
LOG_ROOT=$OUT_ROOT/logs
RESULT_ROOT=$OUT_ROOT/results

# 精确匹配这1000视频、并调整到目标分辨率后的共享Box
BOX_EVAL=$OUT_ROOT/shared_box_eval

mkdir -p "$CFG_ROOT" "$LOG_ROOT" "$RESULT_ROOT"

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

export PATH=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin:$PATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC="${NPROC:-1}"

cd "$SAM_ROOT"

echo "============================================================"
echo "ENV"
echo "============================================================"
echo "python=$(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC=$NPROC"


# ============================================================
# checkpoint
# ============================================================

CHECKPOINT=$(find \
  /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt \
  -type f \
  -name 'sam3.1_multiplex.pt' \
  -print -quit)

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: sam3.1_multiplex.pt not found"
    exit 1
fi

echo "checkpoint=$CHECKPOINT"


# ============================================================
# 1. 自动识别8cam / 3cam manifest
#    + 对齐Box
#    + 根据目标图像尺寸调整投影
# ============================================================

TARGET_ROOT="$TARGET_ROOT" \
BOX_RAW="$BOX_RAW" \
BOX_EVAL="$BOX_EVAL" \
SAM_ROOT="$SAM_ROOT" \
python - <<'PY'
import copy
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, os.environ["SAM_ROOT"])

from shared_box_projection import video_pose_signature


TARGET = Path(os.environ["TARGET_ROOT"])
BOX_RAW = Path(os.environ["BOX_RAW"])
BOX_EVAL = Path(os.environ["BOX_EVAL"])


def load_jsonl(path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def resolve_path(raw, manifest):
    p = Path(str(raw))
    if not p.is_absolute():
        p = manifest.parent / p
    return p.resolve()


# ------------------------------------------------------------
# 找所有 stflow manifest
# ------------------------------------------------------------

manifests = sorted(
    TARGET.rglob("stflow_manifest.jsonl")
)

if not manifests:
    raise RuntimeError(
        f"No stflow_manifest.jsonl under {TARGET}"
    )


infos = []

print()
print("=" * 90)
print("FOUND TARGET MANIFESTS")
print("=" * 90)


for path in manifests:

    items = load_jsonl(path)

    cams = set()
    frame_counts = set()
    all_views = 0
    ref_views = 0
    real_paths = 0

    for video in items:

        frame_counts.add(
            len(video.get("frames", []))
        )

        for frame in video.get("frames", []):

            for view in frame.get("views", []):

                cams.add(
                    str(view.get("camera"))
                )

                all_views += 1

                if view.get(
                    "is_reference_frame",
                    False,
                ):
                    ref_views += 1

                if view.get(
                    "real_image_path"
                ):
                    real_paths += 1


    info = {
        "path": path,
        "items": items,
        "videos": len(items),
        "cams": cams,
        "frame_counts": frame_counts,
        "all_views": all_views,
        "ref_views": ref_views,
        "eligible": all_views - ref_views,
        "real_paths": real_paths,
    }

    infos.append(info)

    print()
    print(path)
    print(" videos          =", len(items))
    print(" frame counts    =", sorted(frame_counts))
    print(" cameras         =", sorted(cams))
    print(" all views       =", all_views)
    print(" reference views =", ref_views)
    print(" eligible views  =", all_views - ref_views)
    print(" real paths      =", real_paths)


cam8 = [
    x for x in infos
    if x["videos"] == 1000
    and len(x["cams"]) == 8
]

cam3 = [
    x for x in infos
    if x["videos"] == 1000
    and len(x["cams"]) == 3
]


if len(cam8) != 1:
    print()
    print("8cam candidates:")
    for x in cam8:
        print(x["path"])

    raise RuntimeError(
        f"Expected exactly one 1000-video 8cam manifest, got {len(cam8)}"
    )


if len(cam3) != 1:
    print()
    print("3cam candidates:")
    for x in cam3:
        print(x["path"])

    raise RuntimeError(
        f"Expected exactly one 1000-video 3cam manifest, got {len(cam3)}"
    )


cam8 = cam8[0]
cam3 = cam3[0]


print()
print("=" * 90)
print("SELECTED")
print("=" * 90)

print("8CAM =", cam8["path"])
print("3CAM =", cam3["path"])

print(
    "8CAM eligible =",
    cam8["eligible"],
)

print(
    "3CAM eligible =",
    cam3["eligible"],
)


# ------------------------------------------------------------
# 必须是相同1000 videos
# ------------------------------------------------------------

sig8 = {
    video_pose_signature(x)
    for x in cam8["items"]
}

sig3 = {
    video_pose_signature(x)
    for x in cam3["items"]
}

print()
print("8cam signatures =", len(sig8))
print("3cam signatures =", len(sig3))
print("same videos     =", sig8 == sig3)

if len(sig8) != 1000:
    raise RuntimeError(
        f"8cam unique signatures={len(sig8)}, expected 1000"
    )

if len(sig3) != 1000:
    raise RuntimeError(
        f"3cam unique signatures={len(sig3)}, expected 1000"
    )

if sig8 != sig3:
    raise RuntimeError(
        "8cam and 3cam are NOT the same 1000 videos"
    )


# ------------------------------------------------------------
# paired-real 必须存在
# ------------------------------------------------------------

missing_real = 0

for video in cam8["items"]:

    for frame in video["frames"]:

        # reference也检查，不影响最终skip
        for view in frame["views"]:

            raw = view.get(
                "real_image_path"
            )

            if not raw:
                missing_real += 1
                continue

            p = resolve_path(
                raw,
                cam8["path"],
            )

            if not p.is_file():
                missing_real += 1


print()
print("missing real images =", missing_real)

if missing_real:
    raise RuntimeError(
        "8cam manifest cannot provide complete paired-real"
    )


# ------------------------------------------------------------
# 加载所有Box rank
# ------------------------------------------------------------

box_manifests = sorted(
    BOX_RAW.rglob(
        "box_manifest.jsonl"
    )
)

if not box_manifests:
    raise RuntimeError(
        f"No Box manifests under {BOX_RAW}"
    )


box_by_sig = {}
box_source = {}

total_box = 0
duplicate = 0


print()
print("=" * 90)
print("SOURCE BOX")
print("=" * 90)


for path in box_manifests:

    items = load_jsonl(path)

    print(
        path,
        "records=",
        len(items),
    )

    for item in items:

        total_box += 1

        sig = video_pose_signature(
            item
        )

        if sig in box_by_sig:

            duplicate += 1

            # 不允许同pose不同GT
            a = json.dumps(
                box_by_sig[sig],
                sort_keys=True,
                ensure_ascii=False,
            )

            b = json.dumps(
                item,
                sort_keys=True,
                ensure_ascii=False,
            )

            if a != b:
                raise RuntimeError(
                    f"Conflicting Box duplicate: {sig}"
                )

            continue

        box_by_sig[sig] = item
        box_source[sig] = path


box_sigs = set(box_by_sig)

print()
print("Box total records =", total_box)
print("Box unique videos =", len(box_sigs))
print("duplicates        =", duplicate)


missing_box = sig8 - box_sigs

print()
print("=" * 90)
print("BOX COVERAGE")
print("=" * 90)

print("target videos      =", len(sig8))
print("matched Box videos =", len(sig8 & box_sigs))
print("missing Box videos =", len(missing_box))


if missing_box:

    print()
    print("first missing signatures:")

    for sig in list(missing_box)[:20]:
        print(sig)

    raise RuntimeError(
        "Box does not cover all target 1000 videos"
    )


# ------------------------------------------------------------
# 找目标每个camera实际图片尺寸
# ------------------------------------------------------------

sizes = defaultdict(set)

for video in cam8["items"]:

    for frame in video["frames"]:

        for view in frame["views"]:

            cam = str(
                view["camera"]
            )

            size = view.get(
                "image_size"
            )

            if (
                isinstance(size, (list, tuple))
                and len(size) >= 2
            ):

                sizes[cam].add(
                    (
                        int(size[0]),
                        int(size[1]),
                    )
                )

            else:

                p = resolve_path(
                    view["image_path"],
                    cam8["path"],
                )

                with Image.open(p) as im:
                    sizes[cam].add(
                        im.size
                    )


for cam, values in sizes.items():

    if len(values) != 1:
        raise RuntimeError(
            f"{cam} multiple target sizes: {values}"
        )


sizes = {
    cam: next(iter(values))
    for cam, values in sizes.items()
}


print()
print("Target image sizes:")

for cam in sorted(sizes):
    print(cam, sizes[cam])


# ------------------------------------------------------------
# 输出 exact-target Box，保留rank目录
# ------------------------------------------------------------

if BOX_EVAL.exists():
    shutil.rmtree(
        BOX_EVAL
    )

BOX_EVAL.mkdir(
    parents=True,
    exist_ok=True,
)


# 只保留目标1000
grouped = defaultdict(list)

for sig in sig8:

    src = box_source[sig]

    rank_name = (
        src.parent.name
        if src.parent.name.startswith("rank_")
        else "rank_00"
    )

    grouped[rank_name].append(
        copy.deepcopy(
            box_by_sig[sig]
        )
    )


scaled = 0
already = 0
total_views = 0


for rank_name, videos in sorted(
    grouped.items()
):

    dst = (
        BOX_EVAL
        / rank_name
        / "box_manifest.jsonl"
    )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with dst.open(
        "w",
        encoding="utf-8",
    ) as writer:

        for video in videos:

            for frame in video.get(
                "frames",
                [],
            ):

                for view in frame.get(
                    "views",
                    [],
                ):

                    cam = str(
                        view.get("camera")
                    )

                    if cam not in sizes:
                        continue

                    tw, th = sizes[cam]

                    old_size = view.get(
                        "image_size"
                    )

                    if (
                        not isinstance(
                            old_size,
                            (list, tuple),
                        )
                        or len(old_size) < 2
                    ):
                        raise RuntimeError(
                            f"Box missing image_size: "
                            f"{video.get('video_id')} {cam}"
                        )

                    ow = int(
                        old_size[0]
                    )
                    oh = int(
                        old_size[1]
                    )


                    if (ow, oh) != (tw, th):

                        sx = tw / float(ow)
                        sy = th / float(oh)

                        P = copy.deepcopy(
                            view.get(
                                "lidar_to_image"
                            )
                        )

                        if P is None:
                            raise RuntimeError(
                                "missing lidar_to_image"
                            )

                        P[0] = [
                            float(v) * sx
                            for v in P[0]
                        ]

                        P[1] = [
                            float(v) * sy
                            for v in P[1]
                        ]

                        view[
                            "lidar_to_image"
                        ] = P


                        if (
                            view.get("K")
                            is not None
                        ):

                            K = copy.deepcopy(
                                view["K"]
                            )

                            K[0] = [
                                float(v) * sx
                                for v in K[0]
                            ]

                            K[1] = [
                                float(v) * sy
                                for v in K[1]
                            ]

                            view["K"] = K


                        scaled += 1

                    else:
                        already += 1


                    view["image_size"] = [
                        tw,
                        th,
                    ]

                    # evaluator只需要几何
                    view.pop(
                        "image_path",
                        None,
                    )

                    view.pop(
                        "real_image_path",
                        None,
                    )

                    view.pop(
                        "valid_mask_path",
                        None,
                    )

                    total_views += 1


            writer.write(
                json.dumps(
                    video,
                    ensure_ascii=False,
                )
                + "\n"
            )


    print(
        "write",
        dst,
        "videos=",
        len(videos),
    )


# ------------------------------------------------------------
# 保存发现结果，供bash后面使用
# ------------------------------------------------------------

meta = {
    "cam8_manifest":
        str(cam8["path"]),

    "cam3_manifest":
        str(cam3["path"]),

    "cam8_root":
        str(cam8["path"].parent),

    "cam3_root":
        str(cam3["path"].parent),

    "cam8_glob":
        cam8["path"].name,

    "cam3_glob":
        cam3["path"].name,

    "cam8_expected":
        cam8["eligible"],

    "cam3_expected":
        cam3["eligible"],

    "cam8_cameras":
        sorted(cam8["cams"]),

    "cam3_cameras":
        sorted(cam3["cams"]),
}

with (
    BOX_EVAL / "meta.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        meta,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 90)
print("BOX READY")
print("=" * 90)

print("videos        =", sum(len(x) for x in grouped.values()))
print("views         =", total_views)
print("scaled views  =", scaled)
print("already size  =", already)
print("output        =", BOX_EVAL)

print()
print("BOX ALIGNMENT PASS")
PY


# ============================================================
# 2. 生成三个config
# ============================================================

TARGET_ROOT="$TARGET_ROOT" \
BOX_EVAL="$BOX_EVAL" \
CHECKPOINT="$CHECKPOINT" \
ROOT="$ROOT" \
CFG_ROOT="$CFG_ROOT" \
RESULT_ROOT="$RESULT_ROOT" \
python - <<'PY'
import copy
import json
import os
from pathlib import Path

import yaml


BOX = Path(
    os.environ["BOX_EVAL"]
)

ROOT = Path(
    os.environ["ROOT"]
)

CFG_ROOT = Path(
    os.environ["CFG_ROOT"]
)

RESULT_ROOT = Path(
    os.environ["RESULT_ROOT"]
)

CHECKPOINT = os.environ[
    "CHECKPOINT"
]


meta = json.loads(
    (
        BOX / "meta.json"
    ).read_text(
        encoding="utf-8"
    )
)


base = {

    "paths": {
        "shared_box_root":
            str(BOX),

        "sam3_repo":
            str(
                ROOT
                / "sam3-eval"
                / "sam3-eval"
                / "sam3"
            ),

        "checkpoint":
            CHECKPOINT,
    },


    "preview": {
        "skip_reference_frames":
            True,

        "strict_paths":
            True,

        "include_methods":
            [],

        "exclude_methods":
            [],
    },


    "shared_box": {
        "manifest_glob":
            "**/box_manifest.jsonl",

        "strict_paths":
            True,

        "strict_match":
            True,
    },


    "model": {
        "version":
            "sam3.1",

        "prompts": [
            "car",
            "truck",
            "bus",
        ],

        "confidence_threshold":
            0.25,

        "batch_size":
            1,

        "loader_workers":
            4,

        "precision":
            "bfloat16",

        "input_resolution":
            1008,

        "save_masks":
            True,

        "mask_resolution":
            256,

        "mask_threshold":
            0.5,

        "max_detections_per_prompt":
            100,

        "max_detections_per_image":
            150,

        "nms_iou_threshold":
            0.70,

        "checkpoint_minimum_coverage":
            0.95,

        "checkpoint_mmap":
            True,
    },


    "annotation": {
        "classes": [
            "CAR",
            "TRUCK",
            "BUS",
        ],

        "near_plane":
            0.10,

        "min_projected_height_px":
            8.0,

        "min_projected_area_px":
            64.0,

        "min_in_frame_fraction":
            0.10,
    },


    "visibility": {
        "disable_gt_occlusion":
            True,

        "min_gt_visible_ratio":
            0.0,

        "min_gt_visible_connected_pixels":
            4,
    },


    "matching": {
        "min_mask_iou":
            0.05,

        "max_center_error_norm":
            0.80,

        "max_scale_error_log":
            1.20,

        "max_cost":
            1.10,

        "cost_mask_iou":
            0.65,

        "cost_center":
            0.20,

        "cost_bottom":
            0.05,

        "cost_scale":
            0.10,

        "allow_bbox_mask_fallback":
            False,

        "min_sam_connected_pixels":
            0,
    },


    "visualization": {
        "enabled":
            True,

        "max_frames_per_source":
            16,

        "image_quality":
            92,

        "mask_alpha":
            0.25,

        "draw_cuboid":
            True,

        "draw_detection_box":
            True,

        "draw_filtered_small_sam":
            True,
    },


    "runtime": {
        "seed":
            3407,

        "backend":
            "sam3.1",

        "limit_frames":
            0,

        "overwrite":
            True,
    },
}


jobs = {

    "pairedreal": {
        "root":
            meta["cam8_root"],

        "glob":
            meta["cam8_glob"],

        "source": {
            "real": {
                "type":
                    "preview_real",

                "group_by_manifest":
                    False,
            }
        },
    },


    "8cam": {
        "root":
            meta["cam8_root"],

        "glob":
            meta["cam8_glob"],

        "source": {
            "generated": {
                "type":
                    "preview_generated",

                "group_by_manifest":
                    False,
            }
        },
    },


    "3cam": {
        "root":
            meta["cam3_root"],

        "glob":
            meta["cam3_glob"],

        "source": {
            "generated": {
                "type":
                    "preview_generated",

                "group_by_manifest":
                    False,
            }
        },
    },
}


for name, job in jobs.items():

    cfg = copy.deepcopy(
        base
    )

    cfg["paths"][
        "preview_root"
    ] = job["root"]

    cfg["paths"][
        "output_dir"
    ] = str(
        RESULT_ROOT / name
    )

    cfg["preview"][
        "manifest_glob"
    ] = job["glob"]

    cfg["sources"] = (
        job["source"]
    )

    out = (
        CFG_ROOT
        / f"{name}.yaml"
    )

    with out.open(
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(name, "->", out)
PY


# ============================================================
# 3. scan-only：三个都通过再真正跑
# ============================================================

CAM8_EXPECT=$(python - <<PY
import json
print(json.load(open("$BOX_EVAL/meta.json"))["cam8_expected"])
PY
)

CAM3_EXPECT=$(python - <<PY
import json
print(json.load(open("$BOX_EVAL/meta.json"))["cam3_expected"])
PY
)


scan_one () {

    NAME="$1"
    EXPECT="$2"

    echo
    echo "============================================================"
    echo "SCAN $NAME expected=$EXPECT"
    echo "============================================================"

    python -u run_eval.py \
      --config "$CFG_ROOT/$NAME.yaml" \
      --scan-only \
      2>&1 | tee "$LOG_ROOT/${NAME}_scan.log"


    grep -Eq \
      "\"frames\"[[:space:]]*:[[:space:]]*$EXPECT" \
      "$LOG_ROOT/${NAME}_scan.log" \
      || {
        echo "ERROR: $NAME frame count mismatch"
        exit 20
      }


    grep -Eq \
      "\"shared_box_matched\"[[:space:]]*:[[:space:]]*$EXPECT" \
      "$LOG_ROOT/${NAME}_scan.log" \
      || {
        echo "ERROR: $NAME Box alignment failed"
        exit 21
      }


    grep -Eq \
      '"shared_box_missing"[[:space:]]*:[[:space:]]*0' \
      "$LOG_ROOT/${NAME}_scan.log" \
      || {
        echo "ERROR: $NAME missing Box"
        exit 22
      }


    echo "[SCAN PASS] $NAME"
}


scan_one pairedreal "$CAM8_EXPECT"
scan_one 8cam       "$CAM8_EXPECT"
scan_one 3cam       "$CAM3_EXPECT"


# ============================================================
# 4. SAM inference
# ============================================================

run_one () {

    NAME="$1"

    CFG="$CFG_ROOT/$NAME.yaml"
    OUT="$RESULT_ROOT/$NAME"
    LOG="$LOG_ROOT/$NAME.log"

    echo
    echo "============================================================"
    echo "RUN $NAME"
    echo "============================================================"


    # 已经完整完成
    if [[ -f "$OUT/summary.json" ]]; then
        echo "[SKIP] already complete"
        return
    fi


    EXTRA=()

    # 如果之前中断，并且本地run_eval有我们加的resume
    if [[ -f "$OUT/records.rank000.jsonl" ]] \
       && grep -q -- '--resume' "$SAM_ROOT/run_eval.py"
    then
        echo "[RESUME] partial result found"
        EXTRA+=(--resume)

    elif [[ -d "$OUT" ]]; then

        BAK="${OUT}.bak_$(date +%Y%m%d_%H%M%S)"

        echo "backup incomplete result -> $BAK"

        mv "$OUT" "$BAK"
    fi


    if [[ "$NPROC" -gt 1 ]]; then

        python -m torch.distributed.run \
          --standalone \
          --nproc_per_node="$NPROC" \
          run_eval.py \
          --config "$CFG" \
          "${EXTRA[@]}" \
          2>&1 | tee -a "$LOG"

    else

        python -u run_eval.py \
          --config "$CFG" \
          "${EXTRA[@]}" \
          2>&1 | tee -a "$LOG"
    fi


    test -f "$OUT/summary.json" || {
        echo "ERROR: $NAME did not finish"
        exit 30
    }
}


run_one pairedreal
run_one 8cam
run_one 3cam


# ============================================================
# 5. 计算 RC / IoU 并保存
#
# key用 preview_video_signature，而不是rank-local video_id
# 3cam会自然只匹配paired-real对应三个camera。
# ============================================================

RESULT_ROOT="$RESULT_ROOT" \
python - <<'PY'
import csv
import json
import os
from pathlib import Path


ROOT = Path(
    os.environ["RESULT_ROOT"]
)


def records(root):

    files = sorted(
        root.glob(
            "records.rank*.jsonl"
        )
    )

    if not files:
        raise RuntimeError(
            f"No records: {root}"
        )

    for path in files:

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                if line.strip():
                    yield json.loads(
                        line
                    )


def key(r):

    # 最稳：dataset pose signature
    sig = r.get(
        "video_signature",
        r.get(
            "preview_video_signature"
        ),
    )

    if sig is None:
        raise KeyError(
            "record does not contain video_signature"
        )

    time_index = r.get(
        "time_index",
        r.get(
            "preview_time_index"
        ),
    )

    camera = r.get(
        "camera_name",
        r.get(
            "preview_camera_name"
        ),
    )

    return (
        str(sig),
        int(time_index),
        str(camera),
    )


def get_matches(r):

    return {
        str(x["gt_id"]):
            float(x["mask_iou"])

        for x in
        r["matching"]["matches"]
    }


# ------------------------------------------------------------
# paired real index
# ------------------------------------------------------------

real = {}

real_views = 0
real_gt = 0
real_match = 0
real_iou = 0.0


for r in records(
    ROOT / "pairedreal"
):

    k = key(r)

    if k in real:
        raise RuntimeError(
            f"duplicate paired-real key {k}"
        )

    m = get_matches(r)

    real[k] = m

    real_views += 1
    real_gt += int(
        r["gt_count"]
    )

    real_match += len(m)
    real_iou += sum(
        m.values()
    )


real_recall = (
    real_match / real_gt
)

real_matched_iou = (
    real_iou / real_match
)

real_coverage = (
    real_iou / real_gt
)


rows = [{
    "method":
        "pairedreal",

    "views":
        real_views,

    "gt":
        real_gt,

    "matched":
        real_match,

    "gt_recall":
        real_recall,

    "matched_mask_iou":
        real_matched_iou,

    "coverage_iou":
        real_coverage,

    "rc_gt":
        real_match,

    "rc_match":
        real_match,

    "rc_recall":
        1.0,

    "rc_matched_iou":
        real_matched_iou,

    "rc_coverage_iou":
        real_matched_iou,

    "missing_real_views":
        0,
}]


print()
print("=" * 90)
print("PAIRED REAL")
print("=" * 90)

print("views           =", real_views)
print("GT              =", real_gt)
print("matched         =", real_match)
print(f"GT Recall       = {real_recall:.6f}")
print(f"Matched IoU     = {real_matched_iou:.6f}")
print(f"Coverage-IoU    = {real_coverage:.6f}")


# ------------------------------------------------------------
# generated
# ------------------------------------------------------------

for name in [
    "8cam",
    "3cam",
]:

    total_views = 0
    gt = 0
    matched = 0
    iou = 0.0

    rc_gt = 0
    rc_match = 0
    rc_iou = 0.0

    missing = 0


    for r in records(
        ROOT / name
    ):

        total_views += 1

        m = get_matches(r)

        gt += int(
            r["gt_count"]
        )

        matched += len(m)

        iou += sum(
            m.values()
        )


        rm = real.get(
            key(r)
        )

        if rm is None:
            missing += 1
            continue


        real_ids = set(rm)

        rc_gt += len(
            real_ids
        )

        common = (
            real_ids
            & set(m)
        )

        rc_match += len(
            common
        )

        rc_iou += sum(
            m[x]
            for x in common
        )


    if missing:
        raise RuntimeError(
            f"{name}: missing paired-real views={missing}"
        )


    recall = (
        matched / gt
        if gt else 0.0
    )

    matched_iou = (
        iou / matched
        if matched else 0.0
    )

    coverage = (
        iou / gt
        if gt else 0.0
    )

    rc_recall = (
        rc_match / rc_gt
        if rc_gt else 0.0
    )

    rc_matched_iou = (
        rc_iou / rc_match
        if rc_match else 0.0
    )

    rc_coverage = (
        rc_iou / rc_gt
        if rc_gt else 0.0
    )


    rows.append({
        "method":
            name,

        "views":
            total_views,

        "gt":
            gt,

        "matched":
            matched,

        "gt_recall":
            recall,

        "matched_mask_iou":
            matched_iou,

        "coverage_iou":
            coverage,

        "rc_gt":
            rc_gt,

        "rc_match":
            rc_match,

        "rc_recall":
            rc_recall,

        "rc_matched_iou":
            rc_matched_iou,

        "rc_coverage_iou":
            rc_coverage,

        "missing_real_views":
            missing,
    })


    print()
    print("=" * 90)
    print(name)
    print("=" * 90)

    print("views           =", total_views)
    print("GT              =", gt)
    print("matched         =", matched)

    print(
        f"GT Recall       = "
        f"{recall:.6f}"
    )

    print(
        f"Matched IoU     = "
        f"{matched_iou:.6f}"
    )

    print(
        f"Coverage-IoU    = "
        f"{coverage:.6f}"
    )

    print("RC GT           =", rc_gt)
    print("RC matched      =", rc_match)

    print(
        f"RC-Recall       = "
        f"{rc_recall:.6f}"
    )

    print(
        f"RC-Matched-IoU  = "
        f"{rc_matched_iou:.6f}"
    )

    print(
        f"RC-Coverage-IoU = "
        f"{rc_coverage:.6f}"
    )


# ------------------------------------------------------------
# save
# ------------------------------------------------------------

fields = [
    "method",
    "views",
    "gt",
    "matched",
    "gt_recall",
    "matched_mask_iou",
    "coverage_iou",
    "rc_gt",
    "rc_match",
    "rc_recall",
    "rc_matched_iou",
    "rc_coverage_iou",
    "missing_real_views",
]


csv_path = (
    ROOT
    / "box_rc_metrics.csv"
)

json_path = (
    ROOT
    / "box_rc_metrics.json"
)


with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(rows)


with json_path.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        rows,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 72)
print("MAIN TABLE")
print("=" * 72)

print(
    f"{'Method':<18}"
    f"{'Coverage-IoU':>16}"
    f"{'RC-Recall':>14}"
    f"{'RC-Cov-IoU':>16}"
)

print("-" * 64)

for x in rows:

    print(
        f"{x['method']:<18}"
        f"{x['coverage_iou']:>16.4f}"
        f"{x['rc_recall']:>14.4f}"
        f"{x['rc_coverage_iou']:>16.4f}"
    )


print()
print("Saved:")
print(csv_path)
print(json_path)
PY


echo
echo "============================================================"
echo "ALL DONE"
echo "============================================================"

echo "$RESULT_ROOT"
