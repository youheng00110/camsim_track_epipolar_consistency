#!/usr/bin/env bash

set +e
set +u

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

CAMSIM_ROOT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim"

SAM3_ROOT="${CAMSIM_ROOT}/sam3-eval/sam3-eval"

WAYMO_ROOT="${CAMSIM_ROOT}/lyh_output/eval/waymo"

CONFIG="${WAYMO_ROOT}/sam3_waymo_pairedreal_merged1000/configs/pairedreal.yaml"

OUT_ROOT="${WAYMO_ROOT}/sam3_waymo_pairedreal_merged1000"

GPU="${GPU:-0}"

mkdir -p "${OUT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export PYTHONPATH="${SAM3_ROOT}/sam3:${PYTHONPATH:-}"

cd "${SAM3_ROOT}" || true


echo
echo "===================================================================================================="
echo "WAYMO PAIREDREAL-ONLY SAM3"
echo "===================================================================================================="
echo "GPU    = ${GPU}"
echo "CONFIG = ${CONFIG}"
echo
echo "COMMAND:"
echo "python -u run_eval.py --config ${CONFIG}"
echo


# 先 scan，一样不因为失败直接退出
echo "===================================================================================================="
echo "SCAN-ONLY"
echo "===================================================================================================="

set -x

python -u run_eval.py \
    --config "${CONFIG}" \
    --scan-only \
    2>&1 | tee "${OUT_ROOT}/pairedreal_scan.log"

SCAN_STATUS=${PIPESTATUS[0]}

set +x

echo
echo "SCAN_STATUS=${SCAN_STATUS}"


# 无论 scan status 如何，都打印日志并继续，
# 这样有问题时可以直接看到真正的报错。
if [ "${SCAN_STATUS}" -ne 0 ]; then
    echo
    echo "[SCAN FAILED BUT CONTINUE]"
    echo
    tail -n 100 "${OUT_ROOT}/pairedreal_scan.log" 2>/dev/null || true
    echo
fi


echo
echo "===================================================================================================="
echo "FULL PAIREDREAL SAM3"
echo "===================================================================================================="

set -x

python -u run_eval.py \
    --config "${CONFIG}" \
    2>&1 | tee "${OUT_ROOT}/pairedreal_full.log"

SAM_STATUS=${PIPESTATUS[0]}

set +x


echo
echo "===================================================================================================="
echo "FINISHED"
echo "===================================================================================================="
echo "SCAN_STATUS = ${SCAN_STATUS}"
echo "SAM_STATUS  = ${SAM_STATUS}"
echo
echo "results:"
echo "${OUT_ROOT}/results"
echo
echo "full log:"
echo "${OUT_ROOT}/pairedreal_full.log"
echo

true
