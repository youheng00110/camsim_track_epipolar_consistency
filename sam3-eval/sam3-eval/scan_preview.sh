#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config.yaml}"
PREVIEW_ROOT="${2:-}"

cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/sam3${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "$PREVIEW_ROOT" ]]; then
  python run_eval.py \
    --config "$CONFIG_PATH" \
    --preview-root "$PREVIEW_ROOT" \
    --scan-only
else
  python run_eval.py --config "$CONFIG_PATH" --scan-only
fi
