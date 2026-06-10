#!/usr/bin/env bash
# Run offline latency measurement for DS-CNN-L v1 on MAX78000 FTHR.
# Firmware must already be flashed (run build_flash_offline.sh first).
# kws20_mode_config.h: ENABLE_MEASURE=1, MEASURE_LIVE=0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_TOOLS="$(cd "$SCRIPT_DIR/../.." && pwd)/tools"
python3 "$PLATFORM_TOOLS/kws20_measure_metrics_max78000_ds_cnn.py" \
  --config "$SCRIPT_DIR/kws20_measure_max78000_ds_cnn_v1_offline_config.json" \
  --mode offline "$@"
