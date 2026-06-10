#!/usr/bin/env bash
# Requires firmware built with KWS20_CFG_ENABLE_MEASURE=1, KWS20_CFG_MEASURE_LIVE=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
python3 "$ROOT/tools/kws20_measure_metrics_u5.py" \
  --config "$SCRIPT_DIR/kws20_measure_u5_ds_cnn_live_config.json" \
  --mode live "$@"
