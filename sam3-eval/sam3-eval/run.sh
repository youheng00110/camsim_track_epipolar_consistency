#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config.yaml}"
GPU_COUNT="${2:-1}"
PREVIEW_ROOT="${3:-}"

cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/sam3${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA_ARGS=()
if [[ -n "$PREVIEW_ROOT" ]]; then
  EXTRA_ARGS+=(--preview-root "$PREVIEW_ROOT")
fi

if [[ "$GPU_COUNT" -le 1 ]]; then
  python run_eval.py --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
else
  torchrun \
    --standalone \
    --nproc_per_node "$GPU_COUNT" \
    run_eval.py \
    --config "$CONFIG_PATH" \
    "${EXTRA_ARGS[@]}"
fi
