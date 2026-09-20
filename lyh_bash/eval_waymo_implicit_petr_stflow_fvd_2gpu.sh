#!/usr/bin/env bash

set +e
set +u

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

CAMSIM_ROOT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim"
OPENDWM_ROOT="${CAMSIM_ROOT}/OpenDWM"
OPENDWM_SRC="${OPENDWM_ROOT}/src"
WAYMO_ROOT="${CAMSIM_ROOT}/lyh_output/eval/waymo"
CKPT_ROOT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt"
I3D_CHECKPOINT="${CKPT_ROOT}/i3d_pretrained_400.pt"

GPU_IMPLICIT="${GPU_IMPLICIT:-0}"
GPU_PETR="${GPU_PETR:-1}"
MAX_VIDEOS="${MAX_VIDEOS:-1000}"
GATE="${GATE:-16}"

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CAMSIM_ROOT OPENDWM_ROOT
export PYTHONPATH="${OPENDWM_SRC}:${PYTHONPATH:-}"
export PYTHONPATH="${OPENDWM_ROOT}/externals/TATS/tats/fvd:${PYTHONPATH}"
if [ -d "${CAMSIM_ROOT}/nuplan-devkit-master" ]; then
  export PYTHONPATH="${CAMSIM_ROOT}/nuplan-devkit-master:${PYTHONPATH}"
fi
if [ -d "${OPENDWM_ROOT}/externals/waymo-open-dataset/src" ]; then
  export PYTHONPATH="${OPENDWM_ROOT}/externals/waymo-open-dataset/src:${PYTHONPATH}"
fi

mkdir -p "${WAYMO_ROOT}"

merge_one() {
  local name="$1"
  local src="${WAYMO_ROOT}/${name}_6hz_18000_1000"
  local dst="${WAYMO_ROOT}/${name}_6hz_18000_1000_merged1000"
  local manifest="${dst}/stflow_manifest.jsonl"

  if [ ! -d "${src}" ]; then
    echo "[ERROR] missing source: ${src}"
    return 1
  fi
  if [ -f "${manifest}" ]; then
    echo "[MERGE] reuse ${manifest}"
    return 0
  fi
  if [ -e "${dst}" ]; then
    # A previous interrupted merge may leave an incomplete target directory.
    # Preserve it, then rebuild from the untouched rank manifests.
    local stale="${dst}.stale_$(date +%Y%m%d_%H%M%S)"
    echo "[MERGE] target exists without manifest; moving stale target to ${stale}"
    mv "${dst}" "${stale}"
    if [ "$?" -ne 0 ]; then
      echo "[ERROR] cannot move stale target: ${dst}"
      return 1
    fi
  fi

  cd "${OPENDWM_SRC}" || return 1
  echo "[MERGE] ${src} -> ${dst}"
  python -m dwm.tools.merge_rank_preview_manifests_interleave \
    --input-root "${src}" \
    --output-root "${dst}" \
    --dataset-name waymo \
    --max-videos "${MAX_VIDEOS}" \
    --overwrite
  return $?
}

validate_manifest() {
  python - "$1" <<'PY'
import json, os, sys
from collections import Counter
p=sys.argv[1]
items=[]
with open(p, encoding='utf-8') as f:
    for line in f:
        if line.strip(): items.append(json.loads(line))
if not items: raise RuntimeError('empty manifest')
frames=Counter(len(x.get('frames',[])) for x in items)
cams=Counter(tuple(v.get('camera') for v in x['frames'][0].get('views',[])) for x in items)
missing_gen=missing_real=0
for x in items:
    for fr in x.get('frames',[]):
        for v in fr.get('views',[]):
            if not v.get('image_path') or not os.path.exists(v['image_path']): missing_gen += 1
            if not v.get('real_image_path') or not os.path.exists(v['real_image_path']): missing_real += 1
print(f'videos={len(items)} frames={dict(frames)} cameras={dict(cams)} missing_generated={missing_gen} missing_real={missing_real}')
if len(frames)!=1 or len(cams)!=1 or missing_gen or missing_real: raise RuntimeError('manifest validation failed')
print(next(iter(frames)))
PY
}

evaluate_one() {
  local name="$1"
  local gpu="$2"
  local root="${WAYMO_ROOT}/${name}_6hz_18000_1000_merged1000"
  local manifest="${root}/stflow_manifest.jsonl"
  local log_dir="${root}/eval_logs"
  local stflow_out="${root}/stflow_traj_result_gate${GATE}.json"
  local fvd_out="${root}/paired_fvd_result_all19.json"
  mkdir -p "${log_dir}"
  echo "[${name}] GPU=${gpu} root=${root}"
  validate_manifest "${manifest}" 2>&1 | tee "${log_dir}/manifest_validate.log"
  local status=${PIPESTATUS[0]}
  if [ "${status}" -ne 0 ]; then echo "[${name}] validation failed"; return "${status}"; fi
  local seq_count
  seq_count=$(python - "${manifest}" <<'PY'
import json,sys
with open(sys.argv[1],encoding='utf-8') as f:
    x=json.loads(next(line for line in f if line.strip()))
print(len(x['frames']))
PY
)
  export CUDA_VISIBLE_DEVICES="${gpu}"
  cd "${OPENDWM_SRC}" || return 1
  echo "[${name}] STFlow"
  python -m dwm.tools.evaluate_stflow \
    --manifest "${manifest}" \
    --output "${stflow_out}" \
    --device cuda \
    --max-videos "${MAX_VIDEOS}" \
    --frame-stride 2 \
    --min-matches 16 \
    --max-matches 256 \
    --loftr-confidence 0.1 \
    --pair-policy waymo \
    --cross-gate-px "${GATE}" \
    2>&1 | tee "${log_dir}/stflow_gate${GATE}.log"
  status=${PIPESTATUS[0]}
  if [ "${status}" -ne 0 ]; then echo "[${name}] STFlow failed"; return "${status}"; fi
  echo "[${name}] FVD sequence_count=${seq_count}"
  python -m dwm.tools.evaluate_fvd_from_paired_manifest \
    --manifest "${manifest}" \
    --output "${fvd_out}" \
    --i3d-checkpoint "${I3D_CHECKPOINT}" \
    --device cuda \
    --max-videos "${MAX_VIDEOS}" \
    --sequence-count "${seq_count}" \
    --batch-size 2 \
    2>&1 | tee "${log_dir}/fvd_all${seq_count}.log"
  status=${PIPESTATUS[0]}
  if [ "${status}" -ne 0 ]; then
    echo "[${name}] FVD batch-size=2 failed; retry batch-size=1"
    python -m dwm.tools.evaluate_fvd_from_paired_manifest \
      --manifest "${manifest}" --output "${fvd_out}" \
      --i3d-checkpoint "${I3D_CHECKPOINT}" --device cuda \
      --max-videos "${MAX_VIDEOS}" --sequence-count "${seq_count}" --batch-size 1 \
      2>&1 | tee -a "${log_dir}/fvd_all${seq_count}.log"
    status=${PIPESTATUS[0]}
  fi
  echo "[${name}] status=${status} stflow=${stflow_out} fvd=${fvd_out}"
  return "${status}"
}

if [ ! -f "${I3D_CHECKPOINT}" ]; then
  echo "[ERROR] missing I3D checkpoint: ${I3D_CHECKPOINT}"
  exit 1
fi

cd "${OPENDWM_SRC}" || exit 1
merge_one implicit
S1=$?
merge_one petr
S2=$?
if [ "${S1}" -ne 0 ] || [ "${S2}" -ne 0 ]; then
  echo "[ERROR] merge failed implicit=${S1} petr=${S2}"
  exit 1
fi

# Each method owns one GPU; STFlow then FVD run sequentially on that GPU.
evaluate_one implicit "${GPU_IMPLICIT}" >"${WAYMO_ROOT}/implicit_6hz_18000_1000_merged1000/eval_logs/worker.log" 2>&1 &
P1=$!
evaluate_one petr "${GPU_PETR}" >"${WAYMO_ROOT}/petr_6hz_18000_1000_merged1000/eval_logs/worker.log" 2>&1 &
P2=$!
wait "${P1}"; S1=$?
wait "${P2}"; S2=$?
echo "[FINAL] implicit=${S1} petr=${S2}"
[ "${S1}" -eq 0 ] && [ "${S2}" -eq 0 ]
