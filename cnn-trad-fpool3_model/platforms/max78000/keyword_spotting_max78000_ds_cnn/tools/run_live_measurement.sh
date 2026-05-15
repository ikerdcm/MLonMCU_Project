#!/usr/bin/env bash
# Run live microphone measurement for DS-CNN-L on MAX78000 FTHR.
# Firmware must already be flashed (run build_flash_live.sh first).
# kws20_mode_config.h: ENABLE_MEASURE=1, MEASURE_LIVE=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM="$(cd "$SCRIPT_DIR/../.." && pwd)"
python3 "$PLATFORM/tools/kws20_measure_metrics_max78000_ds_cnn.py" \
  --config "$PLATFORM/tools/kws20_measure_max78000_ds_cnn_live_config.json" \
  --mode live "$@"
