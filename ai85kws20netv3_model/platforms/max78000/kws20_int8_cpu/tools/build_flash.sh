#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(dirname "$SCRIPT_DIR")/../tools"
exec "$TOOLS/build_flash_max78000.sh" \
  --config "$SCRIPT_DIR/kws20_int8_build_flash_config.json" "$@"
