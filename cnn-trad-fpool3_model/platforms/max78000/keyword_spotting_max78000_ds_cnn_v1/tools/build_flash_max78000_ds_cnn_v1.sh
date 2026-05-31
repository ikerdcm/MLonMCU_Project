#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/kws20_ds_cnn_max78000_v1_config.json"

CONFIG_PATH="${DEFAULT_CONFIG}"
DO_BUILD=1
DO_FLASH=1

usage() {
  cat <<'EOF'
Usage:
  build_flash_max78000_ds_cnn_v1.sh [--config path] [--build-only] [--flash-only]

Options:
  --config PATH   JSON config file (default: kws20_ds_cnn_max78000_v1_config.json)
  --build-only    Only build, do not flash
  --flash-only    Only flash, do not build
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG_PATH="$2"; shift 2 ;;
    --build-only) DO_FLASH=0; shift ;;
    --flash-only) DO_BUILD=0; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

cfg_get() {
  python3 - <<'PY' "$CONFIG_PATH" "$1"
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
v = cfg.get(sys.argv[2])
print("" if v is None else str(v))
PY
}

[[ ! -f "$CONFIG_PATH" ]] && { echo "Config not found: $CONFIG_PATH" >&2; exit 1; }

PROJECT="$(cfg_get project)"
BOARD="$(cfg_get board)"
MAXIM_PATH="$(cfg_get maxim_path)"
OPENOCD_ROOT="$(cfg_get openocd_root)"
BOARD="${BOARD:-FTHR_RevA}"
ELF="${PROJECT}/build/max78000.elf"

[[ -z "$PROJECT" ]] && { echo "Config value 'project' is required." >&2; exit 1; }

if [[ $DO_BUILD -eq 1 ]]; then
  echo "[BUILD] project=${PROJECT}"
  echo "[BUILD] board=${BOARD}, MAXIM_PATH=${MAXIM_PATH}"

  MAKE_ARGS=("BOARD=${BOARD}" "COMPILER=GCC")
  [[ -n "$MAXIM_PATH" ]] && MAKE_ARGS+=("MAXIM_PATH=${MAXIM_PATH}")

  make -C "$PROJECT" distclean "${MAKE_ARGS[@]}"
  make -C "$PROJECT" -j "${MAKE_ARGS[@]}"
  echo "[BUILD] Done → ${ELF}"
fi

if [[ $DO_FLASH -eq 1 ]]; then
  OPENOCD_BIN=""
  if [[ -n "${OPENOCD_ROOT}" ]]; then
    OPENOCD_BIN="${OPENOCD_ROOT}/bin/Darwin_arm64/openocd"
  fi
  if [[ -z "$OPENOCD_BIN" || ! -x "$OPENOCD_BIN" ]]; then
    OPENOCD_BIN="$(command -v openocd || true)"
  fi

  [[ -z "$OPENOCD_BIN" || ! -x "$OPENOCD_BIN" ]] && { echo "openocd not found (set openocd_root or add openocd to PATH)" >&2; exit 1; }
  [[ ! -f "$ELF" ]]         && { echo "ELF not found: $ELF" >&2; exit 1; }

  OPENOCD_ARGS=()
  if [[ -n "${OPENOCD_ROOT}" ]]; then
    OPENOCD_ARGS+=("-s" "${OPENOCD_ROOT}")
  fi

  echo "[FLASH] elf=${ELF}"
  "$OPENOCD_BIN" \
    "${OPENOCD_ARGS[@]}" \
    -f interface/cmsis-dap.cfg \
    -f max78000.cfg \
    -c "transport select swd; program ${ELF} verify reset exit"
  echo "[FLASH] Done"
fi

echo "DONE"
