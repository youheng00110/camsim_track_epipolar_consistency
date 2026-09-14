#!/usr/bin/env bash
set -euo pipefail

cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/sam3-eval/sam3-eval

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1

CONFIG_ROOT="$PWD/configs_nuplan1000_box3d"

OUTPUT_ROOT="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuplanhard1000/sam3_nuplanhard1000_box3d"

LOG_ROOT="$CONFIG_ROOT/logs"
mkdir -p "$LOG_ROOT"

METHODS=(
    plucker
    box
    implicit
    full
    petr
    pvonly
    nocondition
    token18000
    token24000
    tvself
)

for METHOD in "${METHODS[@]}"
do
    CONFIG="$CONFIG_ROOT/generated_${METHOD}.yaml"
    OUTPUT="$OUTPUT_ROOT/$METHOD"
    SUMMARY="$OUTPUT/summary.json"
    LOG="$LOG_ROOT/generated_${METHOD}.log"

    if [[ -s "$SUMMARY" ]]
    then
        echo "[SKIP] $METHOD already completed: $SUMMARY"
        continue
    fi

    echo
    echo "=================================================="
    echo "[START] generated method: $METHOD"
    echo "[CONFIG] $CONFIG"
    echo "[OUTPUT] $OUTPUT"
    echo "=================================================="

    bash run.sh "$CONFIG" 4 \
      2>&1 | tee "$LOG"

    if [[ ! -s "$SUMMARY" ]]
    then
        echo "[ERROR] Missing summary after $METHOD"
        exit 1
    fi

    echo "[DONE] $METHOD"
done

REAL_CONFIG="$CONFIG_ROOT/pairedreal_implicit.yaml"
REAL_OUTPUT="$OUTPUT_ROOT/pairedreal_implicit"
REAL_SUMMARY="$REAL_OUTPUT/summary.json"
REAL_LOG="$LOG_ROOT/pairedreal_implicit.log"

if [[ -s "$REAL_SUMMARY" ]]
then
    echo "[SKIP] pairedreal_implicit already completed"
else
    echo
    echo "=================================================="
    echo "[START] paired real control from implicit"
    echo "[CONFIG] $REAL_CONFIG"
    echo "[OUTPUT] $REAL_OUTPUT"
    echo "=================================================="

    bash run.sh "$REAL_CONFIG" 4 \
      2>&1 | tee "$REAL_LOG"

    if [[ ! -s "$REAL_SUMMARY" ]]
    then
        echo "[ERROR] Missing paired-real summary"
        exit 1
    fi

    echo "[DONE] pairedreal_implicit"
fi

echo
echo "All SAM3 nuPlan-1000 evaluations completed."
