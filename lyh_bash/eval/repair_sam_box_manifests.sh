#!/usr/bin/env bash
set -eo pipefail

ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim
SAM_ROOT=$ROOT/sam3-eval/sam3-eval

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

ROOT="$ROOT" SAM_ROOT="$SAM_ROOT" python - <<'PY'
from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path

import sys

ROOT = Path(os.environ["ROOT"])
SAM_ROOT = Path(os.environ["SAM_ROOT"])

sys.path.insert(0, str(SAM_ROOT))

from shared_box_projection import video_pose_signature


def load_jsonl(path: Path):
    result = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result.append(json.loads(line))

    return result


def make_paths_absolute(video, manifest_path: Path):
    """
    新 merged manifest 不复制图片。
    将 rank manifest 里的相对路径转成绝对路径，
    这样后面 SAM evaluator 可直接读原 rank 图片。
    """
    video = copy.deepcopy(video)

    for frame in video.get("frames", []):
        for view in frame.get("views", []):

            for field in (
                "image_path",
                "real_image_path",
                "valid_mask_path",
            ):
                raw = view.get(field)

                if not raw:
                    continue

                p = Path(str(raw))

                if not p.is_absolute():
                    p = (
                        manifest_path.parent
                        / p
                    ).resolve()

                view[field] = str(p)

    return video


def interleave_first_1000(
    source_root: Path,
    output_manifest: Path,
):
    """
    rank_00[0], rank_01[0], rank_02[0], rank_03[0],
    rank_00[1], ...
    然后取前1000。

    这和项目 merge_rank_preview_manifests_interleave
    的目标一致。
    """

    rank_manifests = sorted(
        source_root.glob(
            "rank_*/stflow_manifest.jsonl"
        )
    )

    if not rank_manifests:
        raise RuntimeError(
            f"No rank manifests under {source_root}"
        )

    rank_records = []

    print()
    print("=" * 90)
    print("MERGE")
    print(source_root)
    print("=" * 90)

    for path in rank_manifests:

        records = load_jsonl(path)

        rank_records.append(
            (path, records)
        )

        print(
            path.parent.name,
            "records =",
            len(records),
        )


    merged = []

    maximum = max(
        len(records)
        for _, records in rank_records
    )


    for index in range(maximum):

        for manifest_path, records in rank_records:

            if index >= len(records):
                continue

            video = make_paths_absolute(
                records[index],
                manifest_path,
            )

            merged.append(video)

            if len(merged) == 1000:
                break

        if len(merged) == 1000:
            break


    if len(merged) != 1000:
        raise RuntimeError(
            f"Only obtained {len(merged)} videos; expected 1000"
        )


    signatures = [
        video_pose_signature(x)
        for x in merged
    ]

    unique = len(set(signatures))

    print("merged records    =", len(merged))
    print("unique signatures =", unique)


    if unique != 1000:
        from collections import Counter

        c = Counter(signatures)

        duplicates = [
            (sig, count)
            for sig, count in c.items()
            if count > 1
        ]

        print("duplicate signatures:")
        for sig, count in duplicates[:20]:
            print(count, sig)

        raise RuntimeError(
            f"Interleaved first1000 still has duplicates: "
            f"unique={unique}"
        )


    output_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    if output_manifest.exists():

        backup = output_manifest.with_name(
            output_manifest.name
            + ".bak_"
            + time.strftime("%Y%m%d_%H%M%S")
        )

        shutil.copy2(
            output_manifest,
            backup,
        )

        print("backup =", backup)


    with output_manifest.open(
        "w",
        encoding="utf-8",
    ) as f:

        for item in merged:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


    print("write =", output_manifest)

    return merged


# ============================================================
# A. 修复之前 uropetvtrack / pvbev merged1000
# ============================================================

BASE = (
    ROOT
    / "lyh_output"
    / "eval"
    / "nuplanhard1000"
)


urope = interleave_first_1000(
    BASE / "uropetvtrack",
    BASE
    / "uropetvtrack_merged1000"
    / "stflow_manifest.jsonl",
)


pvbev = interleave_first_1000(
    BASE / "pvbev",
    BASE
    / "pvbev_merged1000"
    / "stflow_manifest.jsonl",
)


urope_sigs = {
    video_pose_signature(x)
    for x in urope
}

pvbev_sigs = {
    video_pose_signature(x)
    for x in pvbev
}


print()
print("=" * 90)
print("UROPE vs PVBEV")
print("=" * 90)

print("urope unique =", len(urope_sigs))
print("pvbev unique =", len(pvbev_sigs))

print(
    "same 1000 videos =",
    urope_sigs == pvbev_sigs,
)

print(
    "urope-only =",
    len(urope_sigs - pvbev_sigs),
)

print(
    "pvbev-only =",
    len(pvbev_sigs - urope_sigs),
)


if urope_sigs != pvbev_sigs:
    raise RuntimeError(
        "uropetvtrack and pvbev raw rank results "
        "are not the same 1000 videos"
    )


# ============================================================
# B. 修复 rankclean512
# ============================================================

RANKCLEAN = (
    ROOT
    / "lyh_output"
    / "eval"
    / "nuscenesablationnew"
    / "1000nuplan6hz_rankclean512"
)


CAM8_ROOT = (
    RANKCLEAN
    / "sam_eval_8cam_merged1000"
)

CAM3_ROOT = (
    RANKCLEAN
    / "sam_eval_3cam_merged1000"
)


cam8 = interleave_first_1000(
    RANKCLEAN,
    CAM8_ROOT
    / "stflow_manifest.jsonl",
)


# ============================================================
# 从同一个1000-video 8cam manifest派生3cam
# CAM_02 / CAM_03 / CAM_04
# ============================================================

CAM3 = {
    "CAM_02",
    "CAM_03",
    "CAM_04",
}

cam3 = []


for video in cam8:

    item = copy.deepcopy(video)

    for frame in item.get("frames", []):

        views = [
            v
            for v in frame.get("views", [])
            if str(v.get("camera")) in CAM3
        ]

        cams = {
            str(v.get("camera"))
            for v in views
        }

        if cams != CAM3:
            raise RuntimeError(
                "Missing one of CAM_02/CAM_03/CAM_04: "
                f"{cams}"
            )

        if len(views) != 3:
            raise RuntimeError(
                f"Expected 3 views, got {len(views)}"
            )

        frame["views"] = views

    cam3.append(item)


cam3_manifest = (
    CAM3_ROOT
    / "stflow_manifest.jsonl"
)

cam3_manifest.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with cam3_manifest.open(
    "w",
    encoding="utf-8",
) as f:

    for item in cam3:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


sig8 = {
    video_pose_signature(x)
    for x in cam8
}

sig3 = {
    video_pose_signature(x)
    for x in cam3
}


if (
    len(sig8) != 1000
    or len(sig3) != 1000
    or sig8 != sig3
):
    raise RuntimeError(
        "8cam / 3cam video identity check failed"
    )


# ============================================================
# protocol summary
# ============================================================

def inspect(name, items):

    cameras = set()

    total_views = 0
    refs = 0
    real = 0

    frame_counts = set()

    for video in items:

        frame_counts.add(
            len(video["frames"])
        )

        for frame in video["frames"]:

            for view in frame["views"]:

                total_views += 1

                cameras.add(
                    str(view["camera"])
                )

                if view.get(
                    "is_reference_frame",
                    False,
                ):
                    refs += 1

                if view.get(
                    "real_image_path"
                ):
                    real += 1


    print()
    print(name)
    print(" videos          =", len(items))
    print(" frame counts    =", sorted(frame_counts))
    print(" cameras         =", sorted(cameras))
    print(" all views       =", total_views)
    print(" reference views =", refs)
    print(" eligible views  =", total_views - refs)
    print(" real paths      =", real)


inspect(
    "RANKCLEAN 8CAM",
    cam8,
)

inspect(
    "RANKCLEAN 3CAM",
    cam3,
)


print()
print("=" * 90)
print("REPAIR PASS")
print("=" * 90)

print("8cam:")
print(
    CAM8_ROOT
    / "stflow_manifest.jsonl"
)

print()
print("3cam:")
print(cam3_manifest)
PY
