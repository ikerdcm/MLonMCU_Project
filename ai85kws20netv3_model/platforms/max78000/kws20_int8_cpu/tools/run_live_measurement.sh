#!/usr/bin/env bash
# Run live measurement for kws20_int8_cpu.
# Board must already be flashed with kws20_int8_cpu live firmware.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(dirname "$SCRIPT_DIR")/../tools"
python3 "$TOOLS/kws20_measure_metrics_max78000.py" \
  --config "$SCRIPT_DIR/kws20_int8_cpu_config.json" \
  --mode live "$@"
