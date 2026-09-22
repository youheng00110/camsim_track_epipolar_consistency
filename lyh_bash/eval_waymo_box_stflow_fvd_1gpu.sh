#!/usr/bin/env bash
set +e
set +u

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

CAMSIM_ROOT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim"
OPENDWM_ROOT="${CAMSIM_ROOT}/OpenDWM"
OPENDWM_SRC="${OPENDWM_ROOT}/src"
WAYMO_ROOT="${CAMSIM_ROOT}/lyh_output/eval/waymo"
SRC_ROOT="${WAYMO_ROOT}/box_6hz_18000_1000"
TARGET_ROOT="${WAYMO_ROOT}/box_6hz_18000_1000_merged1000"
MANIFEST="${TARGET_ROOT}/stflow_manifest.jsonl"
LOG_DIR="${TARGET_ROOT}/eval_logs"
CKPT_ROOT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt"
I3D_CHECKPOINT="${CKPT_ROOT}/i3d_pretrained_400.pt"
export DWM_RAFT_WEIGHTS="${CKPT_ROOT}/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
GPU="${GPU:-0}"
MAX_VIDEOS="${MAX_VIDEOS:-1000}"
GATE="${GATE:-16}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CAMSIM_ROOT OPENDWM_ROOT
export PYTHONPATH="${OPENDWM_SRC}:${PYTHONPATH:-}"
export PYTHONPATH="${OPENDWM_ROOT}/externals/TATS/tats/fvd:${PYTHONPATH}"
if [ -d "${CAMSIM_ROOT}/nuplan-devkit-master" ]; then export PYTHONPATH="${CAMSIM_ROOT}/nuplan-devkit-master:${PYTHONPATH}"; fi
if [ -d "${OPENDWM_ROOT}/externals/waymo-open-dataset/src" ]; then export PYTHONPATH="${OPENDWM_ROOT}/externals/waymo-open-dataset/src:${PYTHONPATH}"; fi
mkdir -p "${LOG_DIR}"

if [ ! -f "${I3D_CHECKPOINT}" ]; then echo "[ERROR] missing I3D checkpoint: ${I3D_CHECKPOINT}"; exit 1; fi
if [ ! -f "${DWM_RAFT_WEIGHTS}" ]; then echo "[ERROR] missing local RAFT checkpoint: ${DWM_RAFT_WEIGHTS}"; exit 1; fi
if [ ! -d "${SRC_ROOT}" ]; then echo "[ERROR] missing source root: ${SRC_ROOT}"; exit 1; fi

if [ -f "${MANIFEST}" ]; then
  echo "[MERGE] reuse ${MANIFEST}"
elif [ -e "${TARGET_ROOT}" ]; then
  STALE="${TARGET_ROOT}.stale_$(date +%Y%m%d_%H%M%S)"
  echo "[MERGE] moving incomplete target to ${STALE}"
  mv "${TARGET_ROOT}" "${STALE}" || exit 1
  cd "${OPENDWM_SRC}" || exit 1
  python -m dwm.tools.merge_rank_preview_manifests_interleave --input-root "${SRC_ROOT}" --output-root "${TARGET_ROOT}" --dataset-name waymo --max-videos "${MAX_VIDEOS}" --overwrite
  [ $? -eq 0 ] || exit 1
else
  cd "${OPENDWM_SRC}" || exit 1
  python -m dwm.tools.merge_rank_preview_manifests_interleave --input-root "${SRC_ROOT}" --output-root "${TARGET_ROOT}" --dataset-name waymo --max-videos "${MAX_VIDEOS}" --overwrite
  [ $? -eq 0 ] || exit 1
fi

python - "${MANIFEST}" <<'PY' | tee "${LOG_DIR}/manifest_validate.log"
import json, os, sys
from pathlib import Path
from collections import Counter
p=sys.argv[1]
manifest_dir = Path(p).resolve().parent

def resolve_manifest_path(value):
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path

items=[json.loads(x) for x in open(p) if x.strip()]
frames=Counter(len(x['frames']) for x in items); cams=Counter(tuple(v['camera'] for v in x['frames'][0]['views']) for x in items)
missing=sum(not (path := resolve_manifest_path(v.get('image_path'))) or not path.is_file() for x in items for f in x['frames'] for v in f['views'])
missing_real=sum(not (path := resolve_manifest_path(v.get('real_image_path'))) or not path.is_file() for x in items for f in x['frames'] for v in f['views'])
print(f'videos={len(items)} frames={dict(frames)} cameras={dict(cams)} missing_generated={missing} missing_real={missing_real}')
if not items or len(frames)!=1 or len(cams)!=1 or missing or missing_real: raise SystemExit(1)
PY
[ ${PIPESTATUS[0]} -eq 0 ] || exit 1
SEQ_COUNT=$(python - "${MANIFEST}" <<'PY'
import json,sys
print(len(next(json.loads(x) for x in open(sys.argv[1]) if x.strip())['frames']))
PY
)
cd "${OPENDWM_SRC}" || exit 1
python -m dwm.tools.evaluate_stflow --manifest "${MANIFEST}" --output "${TARGET_ROOT}/stflow_traj_result_gate${GATE}.json" --device cuda --max-videos "${MAX_VIDEOS}" --frame-stride 2 --min-matches 16 --max-matches 256 --loftr-confidence 0.1 --pair-policy waymo --cross-gate-px "${GATE}" 2>&1 | tee "${LOG_DIR}/stflow_gate${GATE}.log"
S=$?; [ "${S}" -eq 0 ] || exit "${S}"
python -m dwm.tools.evaluate_fvd_from_paired_manifest --manifest "${MANIFEST}" --output "${TARGET_ROOT}/paired_fvd_result_all${SEQ_COUNT}.json" --i3d-checkpoint "${I3D_CHECKPOINT}" --device cuda --max-videos "${MAX_VIDEOS}" --sequence-count "${SEQ_COUNT}" --batch-size 2 2>&1 | tee "${LOG_DIR}/fvd_all${SEQ_COUNT}.log"
S=$?
if [ "${S}" -ne 0 ]; then
  echo "[FVD] batch-size=2 failed; retrying batch-size=1"
  python -m dwm.tools.evaluate_fvd_from_paired_manifest --manifest "${MANIFEST}" --output "${TARGET_ROOT}/paired_fvd_result_all${SEQ_COUNT}.json" --i3d-checkpoint "${I3D_CHECKPOINT}" --device cuda --max-videos "${MAX_VIDEOS}" --sequence-count "${SEQ_COUNT}" --batch-size 1 2>&1 | tee -a "${LOG_DIR}/fvd_all${SEQ_COUNT}.log"
  S=$?
fi
echo "[DONE] STFlow=${TARGET_ROOT}/stflow_traj_result_gate${GATE}.json"
echo "[DONE] FVD=${TARGET_ROOT}/paired_fvd_result_all${SEQ_COUNT}.json"
exit "${S}"
