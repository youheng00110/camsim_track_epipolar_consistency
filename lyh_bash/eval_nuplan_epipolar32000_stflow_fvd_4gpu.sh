#!/usr/bin/env bash
set -eo pipefail
# Can be launched from conda base; use the existing evaluation environment.
source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate
set -u
ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim
PYTHON=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/python
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}/OpenDWM/src:${ROOT}/OpenDWM/externals/TATS/tats/fvd:${ROOT}/nuplan-devkit-master:${PYTHONPATH:-}"
exec "$PYTHON" "${ROOT}/lyh_bash/nuplan_epipolar32000_eval4.py" "$@"
