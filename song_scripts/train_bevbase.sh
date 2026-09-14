#!/usr/bin/env bash
source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate
cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

unset ENABLE_DEBUGPY
unset DEBUGPY_PORT

export OPENDWM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM
export CAMSIM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONPATH="$OPENDWM_ROOT/externals/TATS/tats/fvd:$PYTHONPATH"
export PYTHONPATH="$CAMSIM_ROOT/nuplan-devkit-master:$PYTHONPATH"
export PYTHONPATH="$OPENDWM_ROOT/externals/waymo-open-dataset/src:$PYTHONPATH"

torchrun \
  --nproc_per_node=8 \
  -m dwm.train \
  -c /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/configs/song_bev/bev.json \
  -o /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output_song/bevbase \
  --log-steps 300 \
  --preview-steps 1000 \
  --checkpointing-steps 6000 \
  --evaluation-steps 200000
