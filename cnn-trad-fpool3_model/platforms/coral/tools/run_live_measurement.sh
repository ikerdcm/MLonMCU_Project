#!/usr/bin/env bash
# Run live measurement (kws_live, microphone, 120 s window).
# Board must already be flashed with kws_live firmware.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/kws_measure_coral.py" \
  --config "$SCRIPT_DIR/coral_live_config.json" \
  --mode live "$@"
