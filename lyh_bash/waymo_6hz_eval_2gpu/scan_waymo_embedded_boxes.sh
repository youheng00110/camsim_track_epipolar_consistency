#!/usr/bin/env bash

set +e
set +u

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

CAMSIM_ROOT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim"

WAYMO_ROOT="${CAMSIM_ROOT}/lyh_output/eval/waymo"

SAM3_ROOT="${CAMSIM_ROOT}/sam3-eval/sam3-eval"


export PYTHONUNBUFFERED=1

export PYTHONPATH="${SAM3_ROOT}/sam3:${PYTHONPATH:-}"


export PS4='+ [${BASH_SOURCE##*/}:${LINENO}] '

cd "${SAM3_ROOT}" || {
    echo "[ERROR] cannot cd to ${SAM3_ROOT}"
}


echo
echo "===================================================================================================="
echo "SAM3 EMBEDDED BOX SCAN"
echo "===================================================================================================="


for METHOD in \
    pvonly_6hz \
    nocondition_6hz \
    plucker_6hz
do

    CONFIG="${WAYMO_ROOT}/sam3_${METHOD}_merged1000_box/configs/${METHOD}.yaml"

    LOG="${WAYMO_ROOT}/sam3_${METHOD}_merged1000_box/scan_embedded_boxes.log"


    echo
    echo
    echo "===================================================================================================="
    echo "METHOD=${METHOD}"
    echo "CONFIG=${CONFIG}"
    echo "LOG=${LOG}"
    echo "===================================================================================================="


    if [ ! -f "${CONFIG}" ]; then

        echo "[MISSING CONFIG]"
        echo "${CONFIG}"
        echo "CONTINUE TO NEXT METHOD"

        continue

    fi


    mkdir -p "$(dirname "${LOG}")"


    echo
    echo "COMMAND:"
    echo "python -u run_eval.py --config ${CONFIG} --scan-only"
    echo


    set -x

    python -u run_eval.py \
        --config "${CONFIG}" \
        --scan-only \
        2>&1 | tee "${LOG}"

    STATUS=${PIPESTATUS[0]}

    set +x


    echo
    echo "----------------------------------------------------------------------------------------------------"
    echo "METHOD=${METHOD}"
    echo "STATUS=${STATUS}"
    echo "LOG=${LOG}"
    echo "----------------------------------------------------------------------------------------------------"


    if [ "${STATUS}" -ne 0 ]; then

        echo
        echo "[FAILED BUT CONTINUE]"
        echo
        echo "Last 80 log lines:"
        echo

        tail -n 80 "${LOG}" 2>/dev/null || true

    else

        echo
        echo "[SCAN OK] ${METHOD}"

    fi

done


echo
echo
echo "===================================================================================================="
echo "ALL METHODS ATTEMPTED"
echo "===================================================================================================="
echo "This script intentionally does not fail-fast."
echo

true
