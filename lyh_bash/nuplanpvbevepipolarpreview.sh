#!/usr/bin/env bash

PROJECT_ROOT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM"

CONFIG_PATH="${PROJECT_ROOT}/configs/lyh/bev_pv_epipolarpreview.json"

OUTPUT_PATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/lyh_output/debug_bevpvepi"


# 启智外层 shell 可能启用了 nounset。
# conda/venv activate 脚本内部会访问尚未定义的 CONDA_PREFIX，
# 所以 source 前必须关闭 nounset。
set +u

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate


cd "${PROJECT_ROOT}/src" || exit 1


export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export ENABLE_DEBUGPY=0

export OPENDWM_ROOT="${PROJECT_ROOT}"

export CAMSIM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim


export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

export PYTHONPATH="${PROJECT_ROOT}/externals/TATS/tats/fvd:${PYTHONPATH}"

export PYTHONPATH="${CAMSIM_ROOT}/nuplan-devkit-master:${PYTHONPATH}"

export PYTHONPATH="${PROJECT_ROOT}/externals/waymo-open-dataset/src:${PYTHONPATH}"


echo "============================================================"
echo "PREVIEW START"
echo "============================================================"
echo "python              = $(which python)"
echo "torchrun            = $(which torchrun)"
echo "project             = ${PROJECT_ROOT}"
echo "config              = ${CONFIG_PATH}"
echo "output              = ${OUTPUT_PATH}"
echo "CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"


exec torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m dwm.preview \
  -c "${CONFIG_PATH}" \
  -o "${OUTPUT_PATH}"