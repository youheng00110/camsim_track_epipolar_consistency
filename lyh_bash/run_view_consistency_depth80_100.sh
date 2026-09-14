#!/usr/bin/env bash
set -euo pipefail

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

cd /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/src

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export OPENDWM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM
export CAMSIM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/externals/TATS/tats/fvd:$PYTHONPATH"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/nuplan-devkit-master:$PYTHONPATH"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/externals/waymo-open-dataset/src:$PYTHONPATH"

CONFIG="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/configs/lyh/bev_pv_epipolar_debug.json"

BASE_OUT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/output/debug_view_consistency"

# Keep these identical across both runs so the sampled time/view pairs,
# query patches, diffusion noise and checkpoint features are directly comparable.
SAMPLES=(100 101 102 103 104)
SEED=0
SIGMA=0.25
MIN_DEPTH=0.1
DEPTH_SAMPLES=1024
MAX_PAIRS=4
TOPK=10

for MAX_DEPTH in 80 100; do
    echo
    echo "================================================================================"
    echo "RUN DEPTH RANGE: ${MIN_DEPTH}-${MAX_DEPTH} m"
    echo "================================================================================"

    python -m dwm.tools.debug_view_consistency_features_depth20_multi \
      -c "${CONFIG}" \
      --sample-indices "${SAMPLES[@]}" \
      --seed "${SEED}" \
      --sigma "${SIGMA}" \
      --min-depth "${MIN_DEPTH}" \
      --max-depth "${MAX_DEPTH}" \
      --depth-samples "${DEPTH_SAMPLES}" \
      --max-pairs "${MAX_PAIRS}" \
      --topk "${TOPK}" \
      --output "${BASE_OUT}/raw_feature_debug_depth_${MAX_DEPTH}m_multi"
done

echo
echo "================================================================================"
echo "DONE"
echo "80m results:"
echo "${BASE_OUT}/raw_feature_debug_depth_80m_multi"
echo
echo "100m results:"
echo "${BASE_OUT}/raw_feature_debug_depth_100m_multi"
echo "================================================================================"
