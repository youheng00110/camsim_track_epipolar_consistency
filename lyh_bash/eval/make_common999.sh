#!/usr/bin/env bash
set -eo pipefail

ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim
SAM_ROOT=$ROOT/sam3-eval/sam3-eval

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

ROOT="$ROOT" SAM_ROOT="$SAM_ROOT" python - <<'PY'
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ["ROOT"])
SAM_ROOT = Path(os.environ["SAM_ROOT"])

sys.path.insert(0, str(SAM_ROOT))

from shared_box_projection import video_pose_signature


def load_jsonl(path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def make_absolute(video, manifest_path):
    video = copy.deepcopy(video)

    for frame in video.get("frames", []):
        for view in frame.get("views", []):
            for key in (
                "image_path",
                "real_image_path",
                "valid_mask_path",
            ):
                raw = view.get(key)

                if not raw:
                    continue

                p = Path(str(raw))

                if not p.is_absolute():
                    p = (
                        manifest_path.parent / p
                    ).resolve()

                view[key] = str(p)

    return video


def interleave(root, limit=1000):
    manifests = sorted(
        root.glob("rank_*/stflow_manifest.jsonl")
    )

    if not manifests:
        raise RuntimeError(
            f"No rank manifests: {root}"
        )

    sources = []

    for p in manifests:
        records = load_jsonl(p)

        print(
            p.parent.name,
            "records =",
            len(records),
        )

        sources.append(
            (p, records)
        )

    result = []

    max_len = max(
        len(x)
        for _, x in sources
    )

    for i in range(max_len):

        for manifest, records in sources:

            if i >= len(records):
                continue

            result.append(
                make_absolute(
                    records[i],
                    manifest,
                )
            )

            if len(result) == limit:
                return result

    return result


def unique_keep_first(items):
    seen = set()
    out = []

    duplicates = []

    for item in items:

        sig = video_pose_signature(item)

        if sig in seen:
            duplicates.append(sig)
            continue

        seen.add(sig)
        out.append(item)

    return out, duplicates


BASE = (
    ROOT
    / "lyh_output"
    / "eval"
    / "nuplanhard1000"
)

UROPE_RAW = BASE / "uropetvtrack"
PVBEV_RAW = BASE / "pvbev"


print("=" * 90)
print("UROPETVTRACK")
print("=" * 90)

urope_raw = interleave(
    UROPE_RAW,
    1000,
)

urope, urope_dups = unique_keep_first(
    urope_raw
)

print(
    "raw =",
    len(urope_raw),
)

print(
    "unique =",
    len(urope),
)

print(
    "duplicates =",
    len(urope_dups),
)


print()
print("=" * 90)
print("PVBEV")
print("=" * 90)

pvbev_raw = interleave(
    PVBEV_RAW,
    1000,
)

pvbev, pvbev_dups = unique_keep_first(
    pvbev_raw
)

print(
    "raw =",
    len(pvbev_raw),
)

print(
    "unique =",
    len(pvbev),
)

print(
    "duplicates =",
    len(pvbev_dups),
)

for sig in pvbev_dups:
    print(
        "duplicate:",
        sig,
    )


urope_by_sig = {
    video_pose_signature(x): x
    for x in urope
}

pvbev_by_sig = {
    video_pose_signature(x): x
    for x in pvbev
}


common = (
    set(urope_by_sig)
    & set(pvbev_by_sig)
)


print()
print("=" * 90)
print("COMMON SET")
print("=" * 90)

print(
    "urope unique =",
    len(urope_by_sig),
)

print(
    "pvbev unique =",
    len(pvbev_by_sig),
)

print(
    "common       =",
    len(common),
)

print(
    "urope only   =",
    len(
        set(urope_by_sig)
        - set(pvbev_by_sig)
    ),
)

print(
    "pvbev only   =",
    len(
        set(pvbev_by_sig)
        - set(urope_by_sig)
    ),
)


# 预期就是999
if len(common) != 999:
    raise RuntimeError(
        f"Expected common=999, got {len(common)}"
    )


# ------------------------------------------------------------
# 保持 urope 的原始顺序
# ------------------------------------------------------------

ordered_signatures = []

for item in urope:

    sig = video_pose_signature(
        item
    )

    if sig in common:
        ordered_signatures.append(
            sig
        )


if len(ordered_signatures) != 999:
    raise RuntimeError(
        "ordered common signature count != 999"
    )


urope_common = [
    urope_by_sig[sig]
    for sig in ordered_signatures
]

pvbev_common = [
    pvbev_by_sig[sig]
    for sig in ordered_signatures
]


# ------------------------------------------------------------
# 再做严格顺序核对
# ------------------------------------------------------------

urope_check = [
    video_pose_signature(x)
    for x in urope_common
]

pvbev_check = [
    video_pose_signature(x)
    for x in pvbev_common
]

if urope_check != pvbev_check:
    raise RuntimeError(
        "Final ordered signature mismatch"
    )


# ------------------------------------------------------------
# write
# ------------------------------------------------------------

outputs = [
    (
        BASE
        / "uropetvtrack_merged999"
        / "stflow_manifest.jsonl",

        urope_common,
    ),

    (
        BASE
        / "pvbev_merged999"
        / "stflow_manifest.jsonl",

        pvbev_common,
    ),
]


for path, items in outputs:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for item in items:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "write:",
        path,
        "records=",
        len(items),
    )


print()
print("=" * 90)
print("COMMON 999 PASS")
print("=" * 90)

print(
    "Both methods contain exactly the same 999 videos."
)
PY
