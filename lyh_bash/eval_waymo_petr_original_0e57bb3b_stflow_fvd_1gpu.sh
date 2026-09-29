#!/usr/bin/env bash
# Merge the first 1000 interleaved samples; run STFlow/Traj then FVD on one GPU.
# CHECK_ONLY=1 performs CPU preflight without merging or evaluating.
set -eo pipefail
source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate
set -u
CAMSIM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim
OPENDWM_ROOT="${CAMSIM_ROOT}/OpenDWM"
EVAL_ROOT="${CAMSIM_ROOT}/lyh_output/eval/waymo"
SRC_ROOT="${EVAL_ROOT}/petr_6hz_18000_1000_original_0e57bb3b"
TARGET_ROOT="${EVAL_ROOT}/petr_6hz_18000_1000_original_0e57bb3b_merged1000"
MANIFEST="${TARGET_ROOT}/stflow_manifest.jsonl"
CKPT_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* && -n "$CUDA_VISIBLE_DEVICES" ]] || { echo 'GPU must select exactly one device'; exit 1; }
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export DWM_RAFT_WEIGHTS="${CKPT_ROOT}/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
export TORCH_HOME="${EVAL_ROOT}/.torch_cache_petr_original_0e57bb3b"
export PYTHONPATH="${OPENDWM_ROOT}/src:${OPENDWM_ROOT}/externals/TATS/tats/fvd:${CAMSIM_ROOT}/nuplan-devkit-master:${PYTHONPATH:-}"
export SRC_ROOT TARGET_ROOT EVAL_ROOT
for weight in "$DWM_RAFT_WEIGHTS" "${CKPT_ROOT}/i3d_pretrained_400.pt" "${CKPT_ROOT}/loftr_outdoor.ckpt"; do
  [[ -s "$weight" ]] || { echo "Missing weight: $weight"; exit 1; }
done
cd "${OPENDWM_ROOT}/src"
exec 9>"${TARGET_ROOT}.eval.lock"
flock -n 9 || { echo 'Another merge/evaluation already holds this lock'; exit 1; }
mkdir -p "${TORCH_HOME}/hub/checkpoints"
cp "$DWM_RAFT_WEIGHTS" "${TORCH_HOME}/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
cp "${CKPT_ROOT}/loftr_outdoor.ckpt" "${TORCH_HOME}/hub/checkpoints/loftr_outdoor.ckpt"
# Exercise the actual weight-loading path while forbidding downloads.
python - <<'PY'
from unittest.mock import patch
from kornia.feature import LoFTR
with patch('torch.hub.download_url_to_file', side_effect=RuntimeError('Offline weight cache missing')):
    model = LoFTR(pretrained='outdoor').eval()
print('[CHECK] LoFTR outdoor loaded from local cache, downloads disabled')
PY

python - <<'PY'
import json, os
from pathlib import Path
source = Path(os.environ['SRC_ROOT'])
ranks = []
for i in range(4):
    with (source / f'rank_{i:02d}' / 'stflow_manifest.jsonl').open() as f:
        ranks.append([json.loads(line) for line in f if line.strip()])
if any(len(rows) < 250 for rows in ranks):
    raise SystemExit('Each rank must contain at least 250 records')
selected = [ranks[i % 4][i // 4] for i in range(1000)]
poses = [json.dumps([f['T_ego_to_world'] for f in row['frames']]) for row in selected]
if len(set(poses)) != 1000:
    raise SystemExit('Duplicate source samples in selected first 1000')
reference = Path(os.environ['EVAL_ROOT']) / 'implicit_6hz_18000_1000_merged1000/stflow_manifest.jsonl'
with reference.open() as f:
    ref = [json.loads(line) for line in f if line.strip()]
ref_poses = [json.dumps([f['T_ego_to_world'] for f in row['frames']]) for row in ref]
if poses != ref_poses:
    raise SystemExit('Sample order differs from implicit_6hz_18000_1000_merged1000')
for row in selected:
    if len(row['frames']) != 19 or any(len(f['views']) != 8 for f in row['frames']):
        raise SystemExit('Expected 19 frames and 8 views')
print('[CHECK] 1000 unique samples, order matches implicit_6hz_18000_1000_merged1000; 19 frames, 8 views')
PY
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  python -m dwm.tools.evaluate_stflow --help >/dev/null
  python -m dwm.tools.evaluate_fvd_from_paired_manifest --help >/dev/null
  echo '[CHECK] Entry imports passed; no merge/evaluation launched'
  exit 0
fi

if [[ ! -f "$MANIFEST" ]]; then
  if [[ -e "$TARGET_ROOT" ]]; then
    STALE_ROOT="${TARGET_ROOT}.preempted_$(date +%Y%m%d_%H%M%S)"
    echo "[RECOVER] preserving incomplete merge at ${STALE_ROOT}"
    mv "$TARGET_ROOT" "$STALE_ROOT"
  fi
  python -m dwm.tools.merge_rank_preview_manifests_interleave \
    --input-root "$SRC_ROOT" --output-root "$TARGET_ROOT" --dataset-name waymo --max-videos 1000
fi
mkdir -p "${TARGET_ROOT}/eval_logs"
python - <<'PY' | tee "${TARGET_ROOT}/eval_logs/manifest_validate.log"
import json, os
from pathlib import Path
p = Path(os.environ['TARGET_ROOT'])
with (p / 'stflow_manifest.jsonl').open() as f:
    rows = [json.loads(line) for line in f if line.strip()]
with (Path(os.environ['EVAL_ROOT']) / 'implicit_6hz_18000_1000_merged1000/stflow_manifest.jsonl').open() as f:
    reference = [json.loads(line) for line in f if line.strip()]
if len(rows) != 1000 or [r['video_id'] for r in rows] != [f'waymo_video_{i:06d}' for i in range(1000)]:
    raise SystemExit('Invalid merged count/IDs')
missing = 0
for row, ref in zip(rows, reference):
    if len(row['frames']) != 19 or [f['T_ego_to_world'] for f in row['frames']] != [f['T_ego_to_world'] for f in ref['frames']]:
        raise SystemExit('Merged sample mismatch')
    for frame in row['frames']:
        if len(frame['views']) != 8:
            raise SystemExit('Invalid view count')
        for view in frame['views']:
            for key in ('image_path', 'real_image_path'):
                missing += not view.get(key) or not (p / view[key]).is_file()
print(f'[CHECK] videos=1000 frames=19 views=8 missing_images={missing}')
if missing:
    raise SystemExit(1)
PY
python -m dwm.tools.evaluate_stflow \
  --manifest "$MANIFEST" --output "${TARGET_ROOT}/stflow_traj_result_gate16.json" \
  --device cuda --max-videos 1000 --frame-stride 2 --min-matches 16 --max-matches 256 \
  --loftr-confidence 0.1 --pair-policy waymo --cross-gate-px 16 \
  2>&1 | tee "${TARGET_ROOT}/eval_logs/stflow_gate16.log"
python -m dwm.tools.evaluate_fvd_from_paired_manifest \
  --manifest "$MANIFEST" --output "${TARGET_ROOT}/paired_fvd_result_all19.json" \
  --i3d-checkpoint "${CKPT_ROOT}/i3d_pretrained_400.pt" --device cuda \
  --max-videos 1000 --sequence-count 19 --batch-size "${FVD_BATCH_SIZE:-2}" \
  2>&1 | tee "${TARGET_ROOT}/eval_logs/fvd_all19.log"
echo "[DONE] Results: ${TARGET_ROOT}"
