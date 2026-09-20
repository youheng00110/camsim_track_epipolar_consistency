#!/usr/bin/env bash
set +e
set +u

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate
cd /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/src

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

CONFIG="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/configs/lyh/waymopreview/waymoboxpreview_18000_1000.json"
OUTPUT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/lyh_output/waymo_preview/box_6hz_18000_1000"

echo "Starting Waymo box-aware 18000 preview (target 1000)"
echo "CONFIG=${CONFIG}"
echo "OUTPUT=${OUTPUT}"

torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m dwm.preview \
  -c "${CONFIG}" \
  -o "${OUTPUT}"
STATUS=$?
echo "Waymo box-aware preview status=${STATUS}"
exit "${STATUS}"
