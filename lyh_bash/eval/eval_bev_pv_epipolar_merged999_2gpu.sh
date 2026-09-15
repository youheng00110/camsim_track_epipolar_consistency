#!/usr/bin/env bash

set +u

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

CAMSIM_ROOT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim"
OPENDWM_ROOT="${CAMSIM_ROOT}/OpenDWM"
OPENDWM_SRC="${OPENDWM_ROOT}/src"

BASE="${CAMSIM_ROOT}/lyh_output/eval/nuplanhard1000"

TARGET_ROOT="${BASE}/bev_pv_epipolar_merged999"
TARGET_MANIFEST="${TARGET_ROOT}/stflow_manifest.jsonl"

REF_ROOT="${BASE}/pvbev_merged999"
REF_MANIFEST="${REF_ROOT}/stflow_manifest.jsonl"

OLD_SAM_ROOT="${BASE}/sam3_uropetvtrack_pvbev_merged999_box"
OLD_PAIREDREAL="${OLD_SAM_ROOT}/results/pairedreal"
OLD_PVBEV_CONFIG="${OLD_SAM_ROOT}/configs/pvbev.yaml"

NEW_SAM_ROOT="${BASE}/sam3_bev_pv_epipolar_merged999_box"
NEW_SAM_CONFIG="${NEW_SAM_ROOT}/configs/bev_pv_epipolar.yaml"
NEW_SAM_RESULTS="${NEW_SAM_ROOT}/results"

SAM3_EVAL_ROOT="${CAMSIM_ROOT}/sam3-eval/sam3-eval"

CKPT_ROOT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt"

RAFT_CHECKPOINT="${CKPT_ROOT}/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
LOFTR_CHECKPOINT="${CKPT_ROOT}/loftr_outdoor.ckpt"
I3D_CHECKPOINT="${CKPT_ROOT}/i3d_pretrained_400.pt"
SAM3_CHECKPOINT="${CKPT_ROOT}/sam3.1/sam3.1_multiplex.pt"

MAX_VIDEOS=999
GATE=16

LOG_DIR="${TARGET_ROOT}/eval_logs"
mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export CAMSIM_ROOT
export OPENDWM_ROOT

export PYTHONPATH="${OPENDWM_SRC}:${PYTHONPATH:-}"
export PYTHONPATH="${OPENDWM_ROOT}/externals/TATS/tats/fvd:${PYTHONPATH}"
export PYTHONPATH="${CAMSIM_ROOT}/nuplan-devkit-master:${PYTHONPATH}"
export PYTHONPATH="${OPENDWM_ROOT}/externals/waymo-open-dataset/src:${PYTHONPATH}"

export TORCH_HOME=/root/.cache/torch
mkdir -p "${TORCH_HOME}/hub/checkpoints"

cp -f \
    "${RAFT_CHECKPOINT}" \
    "${TORCH_HOME}/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

cp -f \
    "${LOFTR_CHECKPOINT}" \
    "${TORCH_HOME}/hub/checkpoints/loftr_outdoor.ckpt"


echo
echo "===================================================================================================="
echo "BEV PV EPIPOLAR MERGED999 — TWO GPU EVALUATION"
echo "===================================================================================================="
echo "GPU 0 : SAM3 / BOX condition coherence"
echo "GPU 1 : ST-Flow + Traj + FVD"
echo
echo "TARGET:"
echo "${TARGET_MANIFEST}"
echo
echo "REFERENCE:"
echo "${REF_MANIFEST}"
echo


# ==================================================================================================
# 0. Basic checks
# ==================================================================================================

for FILE in \
    "${TARGET_MANIFEST}" \
    "${REF_MANIFEST}" \
    "${OLD_PVBEV_CONFIG}" \
    "${RAFT_CHECKPOINT}" \
    "${LOFTR_CHECKPOINT}" \
    "${I3D_CHECKPOINT}" \
    "${SAM3_CHECKPOINT}"
do
    if [ ! -f "${FILE}" ]; then
        echo "[ERROR] missing file:"
        echo "${FILE}"
        exit 1
    fi
done

if [ ! -d "${OLD_PAIREDREAL}" ]; then
    echo "[ERROR] old pairedreal result does not exist:"
    echo "${OLD_PAIREDREAL}"
    exit 1
fi


# ==================================================================================================
# 1. Verify target and pvbev_merged999 are exactly the same 999 clips
#
# Full 19-frame T_ego_to_world trajectory signature.
# Do not trust video_id alone.
# ==================================================================================================

echo
echo "===================================================================================================="
echo "VERIFY 999/999 SAMPLE ALIGNMENT"
echo "===================================================================================================="

python - "${REF_MANIFEST}" "${TARGET_MANIFEST}" <<'PY'
import sys
import json
import hashlib
import numpy as np

ref_path = sys.argv[1]
target_path = sys.argv[2]


def load(path):
    result = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result.append(json.loads(line))

    return result


def signature(item):
    chunks = []

    frames = item["frames"]

    chunks.append(
        np.asarray(
            [len(frames)],
            dtype=np.int64,
        ).tobytes()
    )

    for frame in frames:
        T = np.asarray(
            frame["T_ego_to_world"],
            dtype=np.float64,
        )

        if T.shape != (4, 4):
            raise RuntimeError(
                "bad T_ego_to_world shape: {}".format(T.shape)
            )

        T = np.round(T, 6)

        chunks.append(T.tobytes())

    return hashlib.sha256(
        b"".join(chunks)
    ).hexdigest()


ref = load(ref_path)
target = load(target_path)

print("reference videos =", len(ref))
print("target videos    =", len(target))

if len(ref) != 999:
    raise RuntimeError(
        "reference manifest is not 999 videos: {}".format(len(ref))
    )

if len(target) != 999:
    raise RuntimeError(
        "target manifest is not 999 videos: {}".format(len(target))
    )

ref_sig = [signature(x) for x in ref]
target_sig = [signature(x) for x in target]

same = sum(
    a == b
    for a, b in zip(ref_sig, target_sig)
)

print("same trajectory order =", "{}/999".format(same))

if same != 999:
    for i, (a, b) in enumerate(zip(ref_sig, target_sig)):
        if a != b:
            print("first mismatch index =", i)
            print("reference video_id   =", ref[i].get("video_id"))
            print("target video_id      =", target[i].get("video_id"))
            break

    raise RuntimeError(
        "target and pvbev_merged999 are not strictly aligned"
    )


# Also check basic frame/camera topology.

for i, (a, b) in enumerate(zip(ref, target)):
    if len(a["frames"]) != len(b["frames"]):
        raise RuntimeError(
            "frame-count mismatch at index {}".format(i)
        )

    for t, (fa, fb) in enumerate(zip(a["frames"], b["frames"])):
        ca = [
            str(v.get("camera"))
            for v in fa["views"]
        ]

        cb = [
            str(v.get("camera"))
            for v in fb["views"]
        ]

        if ca != cb:
            raise RuntimeError(
                "camera-order mismatch at video {} frame {}: {} != {}".format(
                    i,
                    t,
                    ca,
                    cb,
                )
            )

print("[OK] strict sample alignment: 999/999")
PY

ALIGN_STATUS=$?

if [ "${ALIGN_STATUS}" -ne 0 ]; then
    echo
    echo "[ERROR] sample alignment failed."
    echo "Do not reuse pairedreal."
    exit "${ALIGN_STATUS}"
fi


# ==================================================================================================
# 2. Prepare SAM3 config
#
# Reuse:
#   old pairedreal result
#   old pvbev box/projection configuration
#
# Run only generated branch for bev_pv_epipolar_merged999.
# ==================================================================================================

echo
echo "===================================================================================================="
echo "PREPARE SAM3 CONFIG + REUSE PAIREDREAL"
echo "===================================================================================================="

mkdir -p "${NEW_SAM_ROOT}/configs"
mkdir -p "${NEW_SAM_RESULTS}"

if [ -e "${NEW_SAM_RESULTS}/pairedreal" ] || [ -L "${NEW_SAM_RESULTS}/pairedreal" ]; then
    echo "[SAM3] pairedreal link/result already exists:"
    echo "${NEW_SAM_RESULTS}/pairedreal"
else
    ln -s \
        "${OLD_PAIREDREAL}" \
        "${NEW_SAM_RESULTS}/pairedreal"

    echo "[SAM3] reuse pairedreal:"
    echo "${NEW_SAM_RESULTS}/pairedreal"
    echo "  -> ${OLD_PAIREDREAL}"
fi


OLD_PVBEV_CONFIG="${OLD_PVBEV_CONFIG}" \
NEW_SAM_CONFIG="${NEW_SAM_CONFIG}" \
TARGET_ROOT="${TARGET_ROOT}" \
NEW_SAM_RESULTS="${NEW_SAM_RESULTS}" \
python - <<'PY'
import os
from pathlib import Path

import yaml


old_config_path = Path(
    os.environ["OLD_PVBEV_CONFIG"]
)

new_config_path = Path(
    os.environ["NEW_SAM_CONFIG"]
)

target_root = str(
    Path(os.environ["TARGET_ROOT"]).resolve()
)

new_results = str(
    Path(os.environ["NEW_SAM_RESULTS"]).resolve()
)


config = yaml.safe_load(
    old_config_path.read_text(
        encoding="utf-8"
    )
)

if not isinstance(config, dict):
    raise RuntimeError(
        "old pvbev config is not a YAML dictionary"
    )

paths = config.setdefault(
    "paths",
    {},
)

old_shared_box_root = paths.get(
    "shared_box_root"
)

if not old_shared_box_root:
    raise RuntimeError(
        "old pvbev.yaml does not contain paths.shared_box_root; "
        "cannot safely reuse pairedreal/box geometry"
    )

old_shared_box_root = Path(
    str(old_shared_box_root)
).expanduser()

if not old_shared_box_root.exists():
    raise RuntimeError(
        "old shared_box_root does not exist: {}".format(
            old_shared_box_root
        )
    )

paths["preview_root"] = target_root
paths["output_dir"] = new_results


# pvbev.yaml should be the generated branch config.
# Explicitly prevent re-running paired-real if old config contains it.

sources = config.get(
    "sources",
    {}
)

if not isinstance(sources, dict):
    raise RuntimeError(
        "config.sources is not a dictionary"
    )

if "generated" not in sources:
    raise RuntimeError(
        "old pvbev.yaml has no sources.generated"
    )

config["sources"] = {
    "generated": sources["generated"]
}


preview = config.setdefault(
    "preview",
    {},
)

# Target root contains exactly one aligned manifest.
# Avoid an old method-name filter accidentally excluding it.
preview["include_methods"] = []
preview["exclude_methods"] = []


runtime = config.setdefault(
    "runtime",
    {},
)

runtime["overwrite"] = True


new_config_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

new_config_path.write_text(
    yaml.safe_dump(
        config,
        allow_unicode=True,
        sort_keys=False,
    ),
    encoding="utf-8",
)


print("new config       =", new_config_path)
print("preview_root     =", paths["preview_root"])
print("output_dir       =", paths["output_dir"])
print("shared_box_root  =", paths["shared_box_root"])
print("sources          =", list(config["sources"].keys()))
PY

CONFIG_STATUS=$?

if [ "${CONFIG_STATUS}" -ne 0 ]; then
    echo
    echo "[ERROR] failed to prepare SAM3 config."
    exit "${CONFIG_STATUS}"
fi


# ==================================================================================================
# 3. Manifest summary
# ==================================================================================================

MANIFEST_INFO=$(
python - "${TARGET_MANIFEST}" <<'PY'
import json
import sys
from collections import Counter

path = sys.argv[1]

items = []

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            items.append(json.loads(line))

video_count = len(items)

frame_counts = Counter(
    len(x["frames"])
    for x in items
)

if len(frame_counts) != 1:
    raise RuntimeError(
        "inconsistent frame counts: {}".format(
            dict(frame_counts)
        )
    )

seq_count = next(iter(frame_counts))

camera_orders = Counter(
    tuple(
        str(v.get("camera"))
        for v in item["frames"][0]["views"]
    )
    for item in items
)

if len(camera_orders) != 1:
    raise RuntimeError(
        "inconsistent camera orders: {}".format(
            dict(camera_orders)
        )
    )

camera_names = next(iter(camera_orders))

real_missing = 0

for item in items:
    for frame in item["frames"]:
        for view in frame["views"]:
            if view.get("real_image_path") is None:
                real_missing += 1

print(video_count)
print(seq_count)
print(",".join(camera_names))
print(real_missing)
PY
)

INFO_STATUS=$?

if [ "${INFO_STATUS}" -ne 0 ]; then
    echo "[ERROR] manifest validation failed."
    exit "${INFO_STATUS}"
fi

VIDEO_COUNT=$(echo "${MANIFEST_INFO}" | sed -n '1p')
SEQ_COUNT=$(echo "${MANIFEST_INFO}" | sed -n '2p')
CAMERAS=$(echo "${MANIFEST_INFO}" | sed -n '3p')
REAL_MISSING=$(echo "${MANIFEST_INFO}" | sed -n '4p')

echo
echo "===================================================================================================="
echo "MANIFEST INFO"
echo "===================================================================================================="
echo "videos         = ${VIDEO_COUNT}"
echo "frames/video   = ${SEQ_COUNT}"
echo "cameras        = ${CAMERAS}"
echo "missing real   = ${REAL_MISSING}"

if [ "${VIDEO_COUNT}" -ne 999 ]; then
    echo "[ERROR] expected 999 videos."
    exit 1
fi

if [ "${SEQ_COUNT}" -ne 19 ]; then
    echo "[ERROR] expected 19 frames/video."
    exit 1
fi

if [ "${REAL_MISSING}" -ne 0 ]; then
    echo "[ERROR] FVD requires valid real_image_path."
    exit 1
fi


# ==================================================================================================
# 4. GPU 0 — SAM3 generated / box evaluation
# ==================================================================================================

(
    export CUDA_VISIBLE_DEVICES=0

    cd "${SAM3_EVAL_ROOT}" || exit 1

    export PYTHONPATH="${SAM3_EVAL_ROOT}/sam3:${PYTHONPATH:-}"

    SAM3_SCAN_LOG="${NEW_SAM_ROOT}/sam3_scan.log"
    SAM3_RUN_LOG="${NEW_SAM_ROOT}/sam3_generated.log"

    echo
    echo "===================================================================================================="
    echo "[GPU 0] SAM3 SCAN"
    echo "===================================================================================================="
    echo "config = ${NEW_SAM_CONFIG}"
    echo "pairedreal reused from:"
    echo "${OLD_PAIREDREAL}"

    python run_eval.py \
        --config "${NEW_SAM_CONFIG}" \
        --scan-only \
        2>&1 | tee "${SAM3_SCAN_LOG}"

    SCAN_STATUS=${PIPESTATUS[0]}

    if [ "${SCAN_STATUS}" -ne 0 ]; then
        echo "[GPU 0] SAM3 scan failed."
        exit "${SCAN_STATUS}"
    fi

    echo
    echo "===================================================================================================="
    echo "[GPU 0] SAM3 GENERATED / BOX EVAL"
    echo "===================================================================================================="

    python run_eval.py \
        --config "${NEW_SAM_CONFIG}" \
        2>&1 | tee "${SAM3_RUN_LOG}"

    SAM_STATUS=${PIPESTATUS[0]}

    if [ "${SAM_STATUS}" -ne 0 ]; then
        echo "[GPU 0] SAM3 generated evaluation failed."
        exit "${SAM_STATUS}"
    fi

    echo
    echo "[GPU 0] SAM3 DONE"
    echo "results:"
    echo "${NEW_SAM_RESULTS}"

    exit 0
) &

SAM3_PID=$!


# ==================================================================================================
# 5. GPU 1 — ST-Flow + Trajectory, then FVD
# ==================================================================================================

(
    export CUDA_VISIBLE_DEVICES=1

    cd "${OPENDWM_SRC}" || exit 1

    STFLOW_OUTPUT="${TARGET_ROOT}/stflow_traj_result_gate16.json"
    STFLOW_LOG="${LOG_DIR}/stflow_gate16.log"

    FVD_OUTPUT="${TARGET_ROOT}/paired_fvd_result_all${SEQ_COUNT}.json"
    FVD_LOG="${LOG_DIR}/fvd_all${SEQ_COUNT}.log"

    echo
    echo "===================================================================================================="
    echo "[GPU 1] ST-FLOW + TRAJECTORY"
    echo "===================================================================================================="
    echo "manifest = ${TARGET_MANIFEST}"
    echo "videos   = ${VIDEO_COUNT}"
    echo "gate     = ${GATE}"
    echo "cameras  = ${CAMERAS}"

    python -m dwm.tools.evaluate_stflow \
        --manifest "${TARGET_MANIFEST}" \
        --output "${STFLOW_OUTPUT}" \
        --device cuda \
        --max-videos "${VIDEO_COUNT}" \
        --frame-stride 2 \
        --min-matches 16 \
        --max-matches 256 \
        --loftr-confidence 0.1 \
        --pair-policy dataset \
        --cross-gate-px "${GATE}" \
        2>&1 | tee "${STFLOW_LOG}"

    STFLOW_STATUS=${PIPESTATUS[0]}

    if [ "${STFLOW_STATUS}" -ne 0 ]; then
        echo
        echo "[GPU 1] ST-Flow failed."
        echo "[GPU 1] Skip FVD."
        exit "${STFLOW_STATUS}"
    fi


    echo
    echo "===================================================================================================="
    echo "[GPU 1] FVD"
    echo "===================================================================================================="
    echo "videos         = ${VIDEO_COUNT}"
    echo "sequence_count = ${SEQ_COUNT}"
    echo "cameras        = ${CAMERAS}"

    python -m dwm.tools.evaluate_fvd_from_paired_manifest \
        --manifest "${TARGET_MANIFEST}" \
        --output "${FVD_OUTPUT}" \
        --i3d-checkpoint "${I3D_CHECKPOINT}" \
        --device cuda \
        --max-videos "${VIDEO_COUNT}" \
        --sequence-count "${SEQ_COUNT}" \
        --camera-names "${CAMERAS}" \
        --batch-size 2 \
        2>&1 | tee "${FVD_LOG}"

    FVD_STATUS=${PIPESTATUS[0]}

    if [ "${FVD_STATUS}" -ne 0 ]; then
        echo
        echo "[GPU 1] FVD batch-size=2 failed."
        echo "[GPU 1] retry batch-size=1."

        python -m dwm.tools.evaluate_fvd_from_paired_manifest \
            --manifest "${TARGET_MANIFEST}" \
            --output "${FVD_OUTPUT}" \
            --i3d-checkpoint "${I3D_CHECKPOINT}" \
            --device cuda \
            --max-videos "${VIDEO_COUNT}" \
            --sequence-count "${SEQ_COUNT}" \
            --camera-names "${CAMERAS}" \
            --batch-size 1 \
            2>&1 | tee -a "${FVD_LOG}"

        FVD_STATUS=${PIPESTATUS[0]}
    fi

    if [ "${FVD_STATUS}" -ne 0 ]; then
        echo "[GPU 1] FVD failed."
        exit "${FVD_STATUS}"
    fi


    echo
    echo "===================================================================================================="
    echo "[GPU 1] ST-FLOW + FVD DONE"
    echo "===================================================================================================="
    echo "ST-Flow:"
    echo "${STFLOW_OUTPUT}"
    echo
    echo "FVD:"
    echo "${FVD_OUTPUT}"

    exit 0
) &

METRIC_PID=$!


# ==================================================================================================
# 6. Wait for both GPUs
# ==================================================================================================

echo
echo "===================================================================================================="
echo "WORKERS STARTED"
echo "===================================================================================================="
echo "GPU 0 SAM3 PID      = ${SAM3_PID}"
echo "GPU 1 metrics PID   = ${METRIC_PID}"
echo

wait "${SAM3_PID}"
SAM3_STATUS=$?

wait "${METRIC_PID}"
METRIC_STATUS=$?


echo
echo "===================================================================================================="
echo "FINAL STATUS"
echo "===================================================================================================="
echo "SAM3 / box       = ${SAM3_STATUS}"
echo "STFlow + FVD     = ${METRIC_STATUS}"


# ==================================================================================================
# 7. Print result summary
# ==================================================================================================

STFLOW_OUTPUT="${TARGET_ROOT}/stflow_traj_result_gate16.json"
FVD_OUTPUT="${TARGET_ROOT}/paired_fvd_result_all${SEQ_COUNT}.json"

if [ -f "${STFLOW_OUTPUT}" ]; then
    echo
    echo "===================================================================================================="
    echo "ST-FLOW / TRAJ SUMMARY"
    echo "===================================================================================================="

    python - "${STFLOW_OUTPUT}" <<'PY'
import json
import sys

with open(
    sys.argv[1],
    "r",
    encoding="utf-8",
) as f:
    x = json.load(f)

mean = x.get(
    "mean",
    {}
)

for key in [
    "temporal_l1",
    "cross_raw_epi_px",
    "cross_epi_px",
    "cross_inlier_ratio",
    "traj_epi_px",
    "traj_inlier2",
    "traj_inlier4",
    "stflow_score",
    "stflow_d_score",
    "stflow_c_score",
]:
    if key in mean:
        print(
            "{:28s} = {}".format(
                key,
                mean[key],
            )
        )
PY
fi


if [ -f "${FVD_OUTPUT}" ]; then
    echo
    echo "===================================================================================================="
    echo "FVD SUMMARY"
    echo "===================================================================================================="

    python - "${FVD_OUTPUT}" <<'PY'
import json
import sys

with open(
    sys.argv[1],
    "r",
    encoding="utf-8",
) as f:
    x = json.load(f)

for key in [
    "fvd",
    "num_videos",
    "num_samples",
    "camera_names",
    "sequence_count",
]:
    if key in x:
        print(
            "{:20s} = {}".format(
                key,
                x[key],
            )
        )
PY
fi


echo
echo "===================================================================================================="
echo "RESULT PATHS"
echo "===================================================================================================="
echo
echo "ST-Flow:"
echo "${TARGET_ROOT}/stflow_traj_result_gate16.json"
echo
echo "FVD:"
echo "${TARGET_ROOT}/paired_fvd_result_all${SEQ_COUNT}.json"
echo
echo "SAM3 generated:"
echo "${NEW_SAM_RESULTS}"
echo
echo "Reused pairedreal:"
echo "${NEW_SAM_RESULTS}/pairedreal"
echo "  -> ${OLD_PAIREDREAL}"
echo

if [ "${SAM3_STATUS}" -ne 0 ] || [ "${METRIC_STATUS}" -ne 0 ]; then
    echo "[ERROR] at least one worker failed."
    exit 1
fi

echo "ALL DONE."
