source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate
cd /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/src

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export ENABLE_DEBUGPY=0

export OPENDWM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM
export CAMSIM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/externals/TATS/tats/fvd:$PYTHONPATH"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/nuplan-devkit-master:$PYTHONPATH"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/externals/waymo-open-dataset/src:$PYTHONPATH"

torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m dwm.preview \
  -c "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/configs/camsim/waymo/waymopvonlypreview_quantum_resume.json" \
  -o "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/lyh_output/debug_waymopvonly"
