#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
python3 "$ROOT/tools/kws20_measure_metrics_u5.py" \
  --config "$SCRIPT_DIR/kws20_measure_u5_ds_cnn_v1_config.json" \
  --mode offline "$@"
