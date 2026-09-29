#!/usr/bin/env bash
set +e
set +u

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate || exit $?
cd /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/src || exit $?

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export ENABLE_DEBUGPY=0

export OPENDWM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM
export CAMSIM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/externals/TATS/tats/fvd:$PYTHONPATH"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/nuplan-devkit-master:$PYTHONPATH"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/externals/waymo-open-dataset/src:$PYTHONPATH"

CONFIG="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/configs/lyh/waymopreview/waymotvpreview_30000_1000.json"
OUTPUT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/lyh_output/waymo_preview/tv_6hz_30000_1000"

echo "Starting Waymo TV 30000 preview (target 1000)"
echo "CONFIG=${CONFIG}"
echo "OUTPUT=${OUTPUT}"

# Verify the original training checkpoint before starting four GPU workers.
CHECKPOINT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt/waymo/tv/30000.pth"
EXPECTED_SHA256="2a875921a7533d4bd08decddddd2a76e6429c850d5c70224b1046c4180b107f0"
printf '%s  %s\n' "${EXPECTED_SHA256}" "${CHECKPOINT}" | sha256sum --check --status || {
  echo "ERROR: TV checkpoint does not match verified original 30000.pth" >&2
  exit 1
}

torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m dwm.preview \
  -c "${CONFIG}" \
  -o "${OUTPUT}"
STATUS=$?
echo "Waymo TV preview status=${STATUS}"
exit "${STATUS}"
