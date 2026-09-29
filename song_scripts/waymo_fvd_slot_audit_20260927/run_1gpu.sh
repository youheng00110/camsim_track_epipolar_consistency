#!/usr/bin/env bash
set -eo pipefail
ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur
source "$ROOT/envs/lyhdwm/bin/activate"
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/camsim/OpenDWM/src:$ROOT/camsim/OpenDWM/externals/TATS/tats/fvd:${PYTHONPATH:-}"
OUT="$ROOT/camsim/lyh_output/waymo_fvd_slot_audit_20260927"
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || { echo 'Audit is already running'; exit 1; }
cd "$ROOT/camsim/OpenDWM"
nvidia-smi
python -u "$ROOT/camsim/song_scripts/waymo_fvd_slot_audit_20260927/slot_fvd_audit.py" \
  --root "$ROOT/camsim/lyh_output/eval/waymo" --output "$OUT" \
  --checkpoint "$ROOT/pretrain/ckpt/i3d_pretrained_400.pt" --workers 4 \
  2>&1 | tee -a "$OUT/run.log"
RELAY=/inspire/hdd/global_user/yanjunchi-24040/songbur/waymo_fvd_slot_audit_20260927
mkdir -p "$RELAY"
cp "$OUT/results.json" "$OUT/REPORT.md" "$OUT/preflight.json" "$RELAY/"
