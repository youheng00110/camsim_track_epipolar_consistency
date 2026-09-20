#!/usr/bin/env bash

set +e
set +u

METHOD="${1:-pvonly}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "${METHOD}" in
  pvonly)
    TARGET="${SCRIPT_DIR}/waymo_6hz_eval_2gpu/eval_waymo_pvonly_6hz_2gpu.sh"
    ;;
  nocondition)
    TARGET="${SCRIPT_DIR}/waymo_6hz_eval_2gpu/eval_waymo_nocondition_6hz_2gpu.sh"
    ;;
  plucker)
    TARGET="${SCRIPT_DIR}/waymo_6hz_eval_2gpu/eval_waymo_plucker_6hz_2gpu.sh"
    ;;
  *)
    echo "Usage: $0 {pvonly|nocondition|plucker}"
    exit 2
    ;;
esac

echo "COMMAND: bash ${TARGET}"
bash "${TARGET}"
STATUS=$?
echo "STATUS[${METHOD}]=${STATUS}"
exit "${STATUS}"
