#!/usr/bin/env bash
# Build + flash for offline mode (ENABLE_MEASURE=1, MEASURE_LIVE=0).
# kws20_mode_config.h must already have the correct settings before running.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/build_flash_max78000_ds_cnn_v1.sh" "$@"
