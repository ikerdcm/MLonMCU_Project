#!/usr/bin/env bash
# Run offline benchmark measurement (kws_bench, 50 inferences).
# Board must already be flashed with kws_bench firmware.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/kws_measure_coral.py" \
  --config "$SCRIPT_DIR/coral_bench_config.json" \
  --mode offline "$@"
