#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(dirname "$SCRIPT_DIR")/../tools"
exec python3 "$TOOLS/kws20_measure_metrics_max78000.py" \
  --config "$SCRIPT_DIR/kws20_int8_cpu_offline_config.json" "$@"
