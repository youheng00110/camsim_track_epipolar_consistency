#!/usr/bin/env bash
source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate
cd /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/src

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export ENABLE_DEBUGPY=0

export OPENDWM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM
export CAMSIM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/externals/TATS/tats/fvd:$PYTHONPATH"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/nuplan-devkit-master:$PYTHONPATH"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/externals/waymo-open-dataset/src:$PYTHONPATH"

torchrun \
  --nproc_per_node=8 \
  -m dwm.train \
  -c /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/configs/song/PV_track_train.json \
  -o /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/song_output/pvtrack \
  --log-steps 300 \
  --preview-steps 1000 \
  --checkpointing-steps 2000 \
  --evaluation-steps 200000
