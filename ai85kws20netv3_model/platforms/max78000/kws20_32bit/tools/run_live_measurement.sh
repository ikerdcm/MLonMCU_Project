#!/usr/bin/env bash
# Run live measurement for kws20_32bit (float32 CPU inference).
# Board must already be flashed with kws20_32bit firmware.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(dirname "$SCRIPT_DIR")/../tools"
python3 "$TOOLS/kws20_measure_metrics_max78000.py" \
  --config "$SCRIPT_DIR/kws20_32bit_config.json" \
  --mode live "$@"
