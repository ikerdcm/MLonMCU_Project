#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/../build_flash_max78000.sh" \
  --config "${SCRIPT_DIR}/../kws20_measure_max78000_w90_config.json" \
  "$@"
