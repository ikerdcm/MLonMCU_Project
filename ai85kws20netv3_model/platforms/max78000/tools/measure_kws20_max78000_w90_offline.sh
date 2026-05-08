#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "${SCRIPT_DIR}/kws20_measure_metrics_max78000.py" \
  --config "${SCRIPT_DIR}/kws20_measure_max78000_w90_offline_config.json" \
  "$@"
