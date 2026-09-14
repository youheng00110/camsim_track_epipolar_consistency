#!/usr/bin/env bash
ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim

EVAL=$ROOT/lyh_output/eval/nuplanhard1000

# 新方法
PV1000=$EVAL/pvtrack2_merged1000
PV999=$EVAL/pvtrack2_merged999

# 旧999参考集合
REF999=$EVAL/uropetvtrack_merged999/stflow_manifest.jsonl

# 旧Box eval，直接复用 pairedreal
OLD_EVAL=$EVAL/sam3_uropetvtrack_pvbev_merged999_box
CFG_ROOT=$OLD_EVAL/configs
RESULT_ROOT=$OLD_EVAL/results
LOG_ROOT=$OLD_EVAL/logs

SAM_ROOT=$ROOT/sam3-eval/sam3-eval

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p "$PV999" "$CFG_ROOT" "$RESULT_ROOT" "$LOG_ROOT"

# ============================================================
# 1. 从新1000中精确取旧paired-real对应的999
# ============================================================

ROOT="$ROOT" \
REF999="$REF999" \
PV1000="$PV1000/stflow_manifest.jsonl" \
PV999="$PV999/stflow_manifest.jsonl" \
SAM_ROOT="$SAM_ROOT" \
python - <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["SAM_ROOT"])

from shared_box_projection import video_pose_signature

REF = Path(os.environ["REF999"])
PV1000 = Path(os.environ["PV1000"])
PV999 = Path(os.environ["PV999"])


def load(path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


ref = load(REF)
pv = load(PV1000)

print("=" * 90)
print("PVTRACK2 ALIGNMENT")
print("=" * 90)

print("reference records =", len(ref))
print("pvtrack2 records  =", len(pv))

ref_sigs = [
    video_pose_signature(x)
    for x in ref
]

pv_sigs = [
    video_pose_signature(x)
    for x in pv
]

print("reference unique =", len(set(ref_sigs)))
print("pvtrack2 unique  =", len(set(pv_sigs)))


if len(ref) != 999 or len(set(ref_sigs)) != 999:
    raise RuntimeError(
        "Old reference manifest is not exact 999 unique videos"
    )


# pvtrack2 按 signature 去重
pv_by_sig = {}

duplicates = []

for item in pv:
    sig = video_pose_signature(item)

    if sig in pv_by_sig:
        duplicates.append(sig)
        continue

    pv_by_sig[sig] = item


missing = set(ref_sigs) - set(pv_by_sig)
extra = set(pv_by_sig) - set(ref_sigs)

print()
print("missing from pvtrack2 =", len(missing))
print("extra in pvtrack2     =", len(extra))
print("duplicate pvtrack2    =", len(duplicates))

if duplicates:
    print()
    print("duplicate signatures:")
    for x in duplicates[:10]:
        print(" ", x)

if missing:
    print()
    print("MISSING:")
    for x in list(missing)[:20]:
        print(" ", x)

    raise RuntimeError(
        "pvtrack2 does not contain all old 999 paired-real videos"
    )


# 严格按照旧999顺序写
selected = [
    pv_by_sig[sig]
    for sig in ref_sigs
]

out_sigs = [
    video_pose_signature(x)
    for x in selected
]

assert out_sigs == ref_sigs
assert len(selected) == 999


PV999.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with PV999.open(
    "w",
    encoding="utf-8",
) as f:

    for item in selected:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


print()
print("=" * 90)
print("EXCLUDED FROM PVTRACK2")
print("=" * 90)

if extra:
    for x in sorted(extra):
        print(x)
else:
    print("No extra unique video; only duplicate entries may have been removed.")


print()
print("write =", PV999)
print("records = 999")
print("PVTRACK2 COMMON999 PASS")
PY


# ============================================================
# 2. 复制旧uropetvtrack config
#    保证SAM/Box/matching protocol完全一致
# ============================================================

BASE_CFG=$CFG_ROOT/uropetvtrack.yaml
NEW_CFG=$CFG_ROOT/pvtrack2.yaml

if [[ ! -f "$BASE_CFG" ]]; then
    echo "ERROR: old config missing:"
    echo "$BASE_CFG"
    exit 2
fi


BASE_CFG="$BASE_CFG" \
NEW_CFG="$NEW_CFG" \
PV999="$PV999" \
RESULT_ROOT="$RESULT_ROOT" \
python - <<'PY'
import os
from pathlib import Path

import yaml

src = Path(os.environ["BASE_CFG"])
dst = Path(os.environ["NEW_CFG"])

cfg = yaml.safe_load(
    src.read_text(encoding="utf-8")
)

cfg["paths"]["preview_root"] = os.environ["PV999"]

cfg["paths"]["output_dir"] = str(
    Path(os.environ["RESULT_ROOT"])
    / "pvtrack2"
)

cfg["preview"]["manifest_glob"] = "stflow_manifest.jsonl"

cfg["sources"] = {
    "generated": {
        "type": "preview_generated",
        "group_by_manifest": False,
    }
}

dst.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)

print("config =", dst)
print("shared_box_root =", cfg["paths"]["shared_box_root"])
print("output =", cfg["paths"]["output_dir"])
PY


cd "$SAM_ROOT"


# ============================================================
# 3. scan-only
# ============================================================

echo
echo "============================================================"
echo "SCAN PVTRACK2"
echo "============================================================"

python -u run_eval.py \
    --config "$NEW_CFG" \
    --scan-only \
    2>&1 | tee "$LOG_ROOT/pvtrack2_scan.log"


grep -Eq \
'"frames"[[:space:]]*:[[:space:]]*127872' \
"$LOG_ROOT/pvtrack2_scan.log" || {
    echo "ERROR: expected 127872 frames"
    exit 10
}


grep -Eq \
'"shared_box_matched"[[:space:]]*:[[:space:]]*127872' \
"$LOG_ROOT/pvtrack2_scan.log" || {
    echo "ERROR: Box matched != 127872"
    exit 11
}


grep -Eq \
'"shared_box_missing"[[:space:]]*:[[:space:]]*0' \
"$LOG_ROOT/pvtrack2_scan.log" || {
    echo "ERROR: missing Box views"
    exit 12
}


echo "[SCAN PASS] 127872 / 127872"


# ============================================================
# 4. 四卡SAM
# ============================================================

OUT=$RESULT_ROOT/pvtrack2

if [[ -f "$OUT/summary.json" ]]; then

    echo "[SKIP] pvtrack2 already completed"

else

    if [[ -f "$OUT/records.rank000.jsonl" ]] \
       && grep -q -- '--resume' "$SAM_ROOT/run_eval.py"
    then
        EXTRA=(--resume)
        echo "[RESUME] existing partial result"
    else
        EXTRA=()

        if [[ -d "$OUT" ]]; then
            mv "$OUT" \
               "${OUT}.bak_$(date +%Y%m%d_%H%M%S)"
        fi
    fi


    python -m torch.distributed.run \
        --standalone \
        --nproc_per_node=4 \
        run_eval.py \
        --config "$NEW_CFG" \
        "${EXTRA[@]}" \
        2>&1 | tee -a "$LOG_ROOT/pvtrack2.log"
fi


test -f "$OUT/summary.json" || {
    echo "ERROR: pvtrack2 SAM did not finish"
    exit 20
}


# ============================================================
# 5. 流式计算 RC
#    pairedreal完全复用，不重新SAM
# ============================================================

RESULT_ROOT="$RESULT_ROOT" python - <<'PY'
import csv
import json
import os
from itertools import zip_longest
from pathlib import Path

ROOT = Path(os.environ["RESULT_ROOT"])


def path(name, rank):
    p = ROOT / name / f"records.rank{rank:03d}.jsonl"

    if not p.is_file():
        raise FileNotFoundError(p)

    return p


def matches(r):
    return {
        str(x["gt_id"]): float(x["mask_iou"])
        for x in r["matching"]["matches"]
    }


def gt_ids(r):
    return {
        str(x["gt_id"])
        for x in r.get("projections", [])
        if "gt_id" in x
    }


rc_gt = 0
rc_match = 0
rc_iou = 0.0
views = 0


for rank in range(4):

    rp = path("pairedreal", rank)
    gp = path("pvtrack2", rank)

    rank_views = 0

    with (
        rp.open("r", encoding="utf-8") as fr,
        gp.open("r", encoding="utf-8") as fg,
    ):

        real_lines = (
            x for x in fr if x.strip()
        )

        gen_lines = (
            x for x in fg if x.strip()
        )


        for i, (rl, gl) in enumerate(
            zip_longest(
                real_lines,
                gen_lines,
            )
        ):

            if rl is None or gl is None:
                raise RuntimeError(
                    f"rank{rank}: record count mismatch at {i}"
                )

            rr = json.loads(rl)
            gr = json.loads(gl)


            if (
                str(rr["camera_name"])
                !=
                str(gr["camera_name"])
            ):
                raise RuntimeError(
                    f"rank{rank} line{i}: camera mismatch"
                )


            if (
                int(rr["time_index"])
                !=
                int(gr["time_index"])
            ):
                raise RuntimeError(
                    f"rank{rank} line{i}: time mismatch"
                )


            if (
                int(rr["gt_count"])
                !=
                int(gr["gt_count"])
            ):
                raise RuntimeError(
                    f"rank{rank} line{i}: GT count mismatch"
                )


            if gt_ids(rr) != gt_ids(gr):
                raise RuntimeError(
                    f"rank{rank} line{i}: GT identity mismatch"
                )


            rm = matches(rr)
            gm = matches(gr)

            real_ids = set(rm)

            rc_gt += len(real_ids)

            common = real_ids & set(gm)

            rc_match += len(common)

            rc_iou += sum(
                gm[x]
                for x in common
            )

            views += 1
            rank_views += 1


    print(
        f"rank{rank:03d}: "
        f"{rank_views} views aligned"
    )


if views != 127872:
    raise RuntimeError(
        f"expected 127872 views, got {views}"
    )


rc_recall = (
    rc_match / rc_gt
    if rc_gt else 0.0
)

rc_cov = (
    rc_iou / rc_gt
    if rc_gt else 0.0
)


print()
print("=" * 70)
print("PVTRACK2 BOX RC")
print("=" * 70)

print("views           =", views)
print("RC GT           =", rc_gt)
print("RC matched      =", rc_match)
print(f"RC-Recall       = {rc_recall:.6f}")
print(f"RC-Coverage-IoU = {rc_cov:.6f}")


csv_path = (
    ROOT
    / "pvtrack2_box_rc_metrics.csv"
)

with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "method",
            "views",
            "rc_gt",
            "rc_match",
            "rc_recall",
            "rc_coverage_iou",
        ],
    )

    writer.writeheader()

    writer.writerow({
        "method": "pvtrack2",
        "views": views,
        "rc_gt": rc_gt,
        "rc_match": rc_match,
        "rc_recall": rc_recall,
        "rc_coverage_iou": rc_cov,
    })


print()
print("saved:")
print(csv_path)
PY

